"""
bartorch.pipe.cfl_tmp — Subprocess fallback using CFL temp files in /dev/shm.

BART subprocess calls receive inputs and emit outputs as standard CFL file
pairs (``.hdr`` + ``.cfl``) written to ``/dev/shm`` (Linux tmpfs, RAM-backed)
or ``tempfile.gettempdir()`` elsewhere.  No FIFOs, no streaming protocol.

Axis convention
---------------
bartorch users pass data in C-order (last index varies fastest); BART expects
Fortran-order CFL (first index varies fastest).  The zero-copy trick:

  C-order ``(coils, ny, nx)`` bytes ≡ Fortran-order ``(nx, ny, coils)`` bytes.

So we write raw C-order bytes to the ``.cfl`` file and reverse the dims in the
``.hdr`` file.  BART sees a valid Fortran ``(nx, ny, coils)`` array without any
data movement.  On the way back, we read raw bytes and reshape with reversed
header dims to recover the user's C-order shape.

Key reason FIFOs are NOT used
------------------------------
Inspecting ``src/misc/mmio.c`` and ``src/misc/stream.c`` in the BART source
tree reveals that ``.fifo``-suffixed paths trigger the full binary streaming
protocol (``stream_msg`` structs + temp file backing).  Implementing this
protocol in Python would be fragile.  Using plain CFL pairs on ``/dev/shm`` is
simpler, equally fast (RAM-backed), and 100% correct.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch

from bartorch.core.tensor import as_complex64

# ---------------------------------------------------------------------------
# Temp-directory selection
# ---------------------------------------------------------------------------

_TMP_DIR: str | None = None

_BART_DIMS = 16


def _get_tmp_dir() -> str:
    """Return the preferred temp directory for CFL scratch files.

    ``/dev/shm`` is a RAM-backed tmpfs on Linux.  Falls back to the OS
    default (``/tmp``, ``$TMPDIR``, etc.) on other platforms.
    """
    global _TMP_DIR
    if _TMP_DIR is None:
        if Path("/dev/shm").is_dir():
            _TMP_DIR = "/dev/shm"
        else:
            _TMP_DIR = tempfile.gettempdir()
    return _TMP_DIR


def _unique_base(prefix: str = "_bt_") -> str:
    """Return a unique path base (no extension) inside the temp directory."""
    return os.path.join(_get_tmp_dir(), f"{prefix}{uuid.uuid4().hex}")


# ---------------------------------------------------------------------------
# CFL helpers
# ---------------------------------------------------------------------------


def _write_cfl_pair(base: str, tensor: np.ndarray) -> None:
    """Write *tensor* as a CFL pair at *base* + ``.hdr`` / ``.cfl``.

    The tensor is expected in C-order (bartorch convention).  The ``.hdr``
    file records the reversed (Fortran) dims so BART reads the data correctly.
    The ``.cfl`` file contains the raw C-order bytes — identical to the bytes
    BART would read as Fortran-order for the reversed dims.
    """
    array = np.asarray(tensor, dtype=np.complex64)
    # Reversed dims: bartorch (coils, ny, nx) → BART header (nx, ny, coils)
    bart_dims = list(reversed(array.shape))
    padded = bart_dims + [1] * (_BART_DIMS - len(bart_dims))

    with open(base + ".hdr", "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in padded) + "\n")

    # Write raw C-order bytes (same layout as BART's Fortran for reversed dims).
    # np.ascontiguousarray returns the same object when already C-contiguous.
    np.ascontiguousarray(array).tofile(base + ".cfl")


def _read_cfl_pair(base: str) -> np.ndarray:
    """Read a CFL pair written by BART and return a C-order array.

    BART writes data in Fortran order for the dims in the header (e.g.
    ``(nx, ny, coils)``).  We reverse those dims to get the bartorch C-order
    shape ``(coils, ny, nx)`` and reshape the raw bytes accordingly — the
    byte order is already correct, no copy required.
    """
    bart_dims: list[int] = [1]
    with open(base + ".hdr") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            bart_dims = [int(x) for x in line.split()]
            while bart_dims and bart_dims[-1] == 1:
                bart_dims.pop()
            break

    n = int(np.prod(bart_dims)) if bart_dims else 1
    arr = np.fromfile(base + ".cfl", dtype=np.complex64)

    if arr.size != n:
        raise RuntimeError(
            f"CFL size mismatch reading {base!r}: header says {n} elements, file has {arr.size}."
        )

    # Reversed dims: BART header (nx, ny, coils) → bartorch (coils, ny, nx)
    user_dims = list(reversed(bart_dims))
    return arr.reshape(user_dims)  # default C-order reshape


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Context managers for individual input / output CFL scratch files
# ---------------------------------------------------------------------------


@contextmanager
def write_cfl_tmp(
    tensor: torch.Tensor | np.ndarray,
) -> Generator[str, None, None]:
    """Write *tensor* to a scratch CFL pair in ``/dev/shm`` and yield the base path.

    The ``.hdr`` carries reversed dims; the ``.cfl`` holds raw C-order bytes.
    Both files are removed when the context exits.

    Parameters
    ----------
    tensor:
        Input array in C-order (bartorch convention).  Converted to
        ``complex64`` if necessary.

    Yields
    ------
    str
        Base path (no extension) to pass to BART.
    """
    base = _unique_base("_bt_in_")
    if isinstance(tensor, torch.Tensor):
        arr = as_complex64(tensor).detach().cpu().numpy()
    else:
        arr = np.asarray(tensor, dtype=np.complex64)
    _write_cfl_pair(base, arr)
    try:
        yield base
    finally:
        _safe_unlink(base + ".hdr")
        _safe_unlink(base + ".cfl")


@contextmanager
def read_cfl_tmp(
    device: str | torch.device = "cpu",
) -> Generator[tuple[str, list], None, None]:
    """Reserve an output CFL slot in ``/dev/shm`` and yield ``(base, result)``.

    After the context body executes, reads the CFL pair written by BART into a
    plain ``torch.Tensor`` (complex64, C-order) and stores it in
    ``result[0]``.

    Parameters
    ----------
    device:
        Target device for the output tensor.

    Yields
    ------
    tuple[str, list]
        ``(base_path, result_list)`` — read ``result_list[0]`` after the
        context exits.
    """
    base = _unique_base("_bt_out_")
    result: list[torch.Tensor | None] = [None]
    try:
        yield base, result
        arr = _read_cfl_pair(base)
        t = torch.from_numpy(arr.copy()).to(device)
        result[0] = as_complex64(t)
    finally:
        _safe_unlink(base + ".hdr")
        _safe_unlink(base + ".cfl")


# ---------------------------------------------------------------------------
# Main dispatch entry point
# ---------------------------------------------------------------------------


def run_subprocess(
    op_name: str,
    inputs: list[Any],
    output_dims: list[int] | None,
    **kwargs,
) -> torch.Tensor:
    """Run a BART tool as a subprocess, passing data via CFL files in ``/dev/shm``.

    Inputs are written as ``.hdr`` + ``.cfl`` pairs to ``/dev/shm`` (or the OS
    temp directory) before the call; outputs are read back afterwards.  All
    temp files are removed on completion.

    Parameters
    ----------
    op_name:
        BART tool name (e.g. ``"fft"``).
    inputs:
        Array-valued inputs (``torch.Tensor``, ``np.ndarray``, or a string
        path that is passed through verbatim).
    output_dims:
        Ignored — BART writes the correct output dimensions to the ``.hdr``
        file; we read them back.  Kept for API compatibility.
    **kwargs:
        Flag / scalar arguments forwarded to the BART command.

    Returns
    -------
    torch.Tensor
        Output data as a plain complex64 C-order tensor.

    Raises
    ------
    RuntimeError
        If the ``bart`` executable is not on ``$PATH`` or the tool exits
        with a non-zero status.
    """
    bart_bin = shutil.which("bart")
    if bart_bin is None:
        raise RuntimeError(
            "BART executable not found on $PATH and the C++ extension is not "
            "available.  Install BART (https://mrirecon.github.io/bart/) or "
            "build the bartorch C++ extension."
        )

    from contextlib import ExitStack

    argv = [bart_bin, op_name] + _flags_to_argv(kwargs)

    with ExitStack() as stack:
        # --- inputs ---
        for inp in inputs:
            if isinstance(inp, (torch.Tensor, np.ndarray)):
                base = stack.enter_context(write_cfl_tmp(inp))
                argv.append(base)
            else:
                argv.append(str(inp))

        # --- single output ---
        out_base, result_holder = stack.enter_context(read_cfl_tmp())
        argv.append(out_base)

        # --- run BART ---
        ret = subprocess.call(argv)
        if ret != 0:
            raise RuntimeError(
                f"bart {op_name!r} subprocess exited with code {ret}.\nCommand: {' '.join(argv)}"
            )

    out = result_holder[0]
    if out is None:
        raise RuntimeError(f"bart {op_name!r}: no output file was produced at {out_base!r}.")
    return out


# ---------------------------------------------------------------------------
# Argv helpers
# ---------------------------------------------------------------------------


def _flags_to_argv(kwargs: dict) -> list[str]:
    """Convert keyword arguments to BART flag strings.

    Rules:
    - ``key=True``  → ``["-key"]``
    - ``key=False`` or ``key=None`` → skipped
    - ``key="value"`` → ``["-key", "value"]``
    - Keys beginning with ``_`` are silently ignored (internal use).
    """
    argv: list[str] = []
    for key, val in kwargs.items():
        if key.startswith("_") or val is None or val is False:
            continue
        if val is True:
            argv.append(f"-{key}")
        else:
            argv += [f"-{key}", str(val)]
    return argv
