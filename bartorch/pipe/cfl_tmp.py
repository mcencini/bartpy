"""
bartorch.pipe.cfl_tmp — Subprocess fallback using CFL temp files in /dev/shm.

BART subprocess calls receive inputs and emit outputs as standard CFL file
pairs (``.hdr`` + ``.cfl``) written to ``/dev/shm`` (Linux tmpfs, RAM-backed)
or ``tempfile.gettempdir()`` elsewhere.  No FIFOs, no streaming protocol.

Key reason FIFOs are NOT used
------------------------------
Inspecting ``src/misc/mmio.c`` and ``src/misc/stream.c`` in the BART source
tree reveals that ``.fifo``-suffixed paths trigger the full binary streaming
protocol:

  1. ``stream_ensure_fifo(name)`` creates the FIFO via ``mkfifo()``.
  2. The writer opens the FIFO, writes a CFL header (``write_stream_header``),
     then sends ``stream_msg`` structs (24-byte headers + data blocks).
  3. The actual data backing is a memory-mapped temp file whose path is
     embedded in the CFL header.  In "binary" mode the data follows inline in
     the stream, protected by sync messages.

Implementing this protocol in Python would be fragile.  Using plain CFL pairs
on ``/dev/shm`` is simpler, equally fast (RAM-backed), and 100% correct.
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

from bartorch.core.tensor import BartTensor, bart_from_tensor


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
# CFL helpers  (subset of bartorch.utils.cfl, reproduced here to avoid
# circular imports at runtime before the package is fully installed)
# ---------------------------------------------------------------------------

def _write_cfl_pair(base: str, array: np.ndarray) -> None:
    """Write *array* as a CFL pair at *base* + ``.hdr`` / ``.cfl``."""
    array = np.asarray(array, dtype=np.complex64)
    dims = list(array.shape)
    padded = dims + [1] * (_BART_DIMS - len(dims))

    with open(base + ".hdr", "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in padded) + "\n")

    array.ravel(order="F").tofile(base + ".cfl")


def _read_cfl_pair(base: str) -> np.ndarray:
    """Read a CFL pair written by BART and return a Fortran-order complex64 array."""
    with open(base + ".hdr") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            dims = [int(x) for x in line.split()]
            while dims and dims[-1] == 1:
                dims.pop()
            break
    else:
        dims = [1]

    n = int(np.prod(dims)) if dims else 1
    arr = np.fromfile(base + ".cfl", dtype=np.complex64)

    if arr.size != n:
        raise RuntimeError(
            f"CFL size mismatch reading {base!r}: "
            f"header says {n} elements, file has {arr.size}."
        )

    return arr.reshape(dims, order="F")


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

    The files are removed when the context exits.

    Parameters
    ----------
    tensor:
        Input array.  Converted to ``complex64`` if necessary.

    Yields
    ------
    str
        Base path (no extension) to pass to BART.
    """
    base = _unique_base("_bt_in_")
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    _write_cfl_pair(base, arr)
    try:
        yield base
    finally:
        _safe_unlink(base + ".hdr")
        _safe_unlink(base + ".cfl")


@contextmanager
def read_cfl_tmp(
    device: str | torch.device = "cpu",
) -> Generator[tuple[str, BartTensor | None], None, None]:
    """Reserve an output CFL slot in ``/dev/shm`` and yield ``(base, result_holder)``.

    After the context body executes, reads the CFL pair written by BART into a
    ``BartTensor`` and stores it in ``result_holder[0]``.

    Parameters
    ----------
    device:
        Target device for the output ``BartTensor``.

    Yields
    ------
    tuple[str, list]
        ``(base_path, result_list)`` — append the output tensor to
        ``result_list`` after the BART call, or read ``result_list[0]``.
    """
    base = _unique_base("_bt_out_")
    result: list[BartTensor | None] = [None]
    try:
        yield base, result
        # Read output written by BART
        arr = _read_cfl_pair(base)
        t = torch.from_numpy(arr.copy())
        result[0] = bart_from_tensor(t, copy=False)
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
) -> BartTensor:
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
    BartTensor
        Output data read from the CFL file written by BART.

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
                f"bart {op_name!r} subprocess exited with code {ret}.\n"
                f"Command: {' '.join(argv)}"
            )

    out = result_holder[0]
    if out is None:
        raise RuntimeError(
            f"bart {op_name!r}: no output file was produced at {out_base!r}."
        )
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
