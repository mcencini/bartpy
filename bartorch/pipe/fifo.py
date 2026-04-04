"""
bartorch.pipe.fifo — FIFO-based no-disk subprocess wrapper.

BART supports ``*.fifo`` path suffixes (mapped to ``FILE_TYPE_PIPE`` in
``src/misc/io.c``).  We exploit this to pass data to/from BART subprocesses
entirely in memory using POSIX named FIFOs located in ``/dev/shm``.

Protocol (per CFL array)
------------------------
For each *input* array:
  1. ``mkfifo(path + ".fifo")``  — create named FIFO
  2. Write ``path + ".hdr"``     — tiny text file (BART reads header first)
  3. Background thread streams   — complex64 bytes into FIFO write-end
  4. BART subprocess opens FIFO  — reads data in Fortran order

For each *output* array:
  1. ``mkfifo(path + ".fifo")``  — create named FIFO
  2. Write ``path + ".hdr"``     — BART writes the header first (we pre-write
     a placeholder; BART's mmio writes the real header to the CFL path, but
     for a FIFO the header is a separate file that we read after the call)
  3. Background thread reads     — complex64 bytes from FIFO read-end into a
     pre-allocated BartTensor
  4. After BART exits            — tensor is ready

BART's streaming protocol (``src/misc/stream.c``) is richer (sync messages,
binary mode, etc.) and is used for the live-streaming use case.  For simple
tool calls the plain FIFO approach is sufficient and simpler.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch

from bartorch.core.tensor import BartTensor, bart_empty

# ---------------------------------------------------------------------------
# FIFO directory selection
# ---------------------------------------------------------------------------

_FIFO_DIR: str | None = None


def _get_fifo_dir() -> str:
    global _FIFO_DIR
    if _FIFO_DIR is None:
        if Path("/dev/shm").is_dir():
            _FIFO_DIR = "/dev/shm"
        else:
            _FIFO_DIR = tempfile.gettempdir()
    return _FIFO_DIR


def _unique_base(prefix: str = "_bt_") -> str:
    return os.path.join(_get_fifo_dir(), f"{prefix}{uuid.uuid4().hex}")


# ---------------------------------------------------------------------------
# CFL header helpers
# ---------------------------------------------------------------------------

_BART_DIMS = 16


def _write_cfl_header(hdr_path: str, dims: list[int]) -> None:
    """Write a BART CFL header file (text format)."""
    padded = list(dims) + [1] * (_BART_DIMS - len(dims))
    with open(hdr_path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in padded) + "\n")


def _read_cfl_header(hdr_path: str) -> list[int]:
    """Read a BART CFL header and return the dimension list (trailing 1s stripped)."""
    with open(hdr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            dims = [int(x) for x in line.split()]
            while dims and dims[-1] == 1:
                dims.pop()
            return dims
    return [1]


# ---------------------------------------------------------------------------
# Streaming helpers (background threads)
# ---------------------------------------------------------------------------

def _stream_tensor_to_fifo(fifo_path: str, tensor: torch.Tensor) -> None:
    """Write a tensor's bytes into a FIFO (blocking, called in a thread)."""
    arr = tensor.numpy() if tensor.is_cpu else tensor.cpu().numpy()
    arr_f = np.asfortranarray(arr.astype(np.complex64))
    with open(fifo_path, "wb") as f:
        f.write(arr_f.tobytes())


def _stream_fifo_to_tensor(
    fifo_path: str,
    tensor: torch.Tensor,
    nbytes: int,
) -> None:
    """Read *nbytes* from a FIFO into *tensor*'s data buffer (blocking, in a thread)."""
    buf = bytearray(nbytes)
    with open(fifo_path, "rb") as f:
        view = memoryview(buf)
        read = 0
        while read < nbytes:
            chunk = f.readinto(view[read:])
            if chunk == 0:
                break
            read += chunk
    # Copy bytes into tensor via numpy view.
    arr = np.frombuffer(buf, dtype=np.complex64).reshape(
        tensor.shape, order="F"
    )
    tensor.copy_(torch.from_numpy(arr.copy()))


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def cfl_input_fifo(
    tensor: torch.Tensor,
) -> Generator[str, None, None]:
    """Expose *tensor* as a named FIFO pair that BART can read.

    Yields the base path (without extension) to pass to BART.
    The ``.fifo`` file carries the data; the ``.hdr`` file is a regular file.
    """
    base = _unique_base("_bt_in_")
    hdr_path = base + ".hdr"
    fifo_path = base + ".fifo"

    dims = list(tensor.shape)
    _write_cfl_header(hdr_path, dims)
    os.mkfifo(fifo_path)

    t = threading.Thread(
        target=_stream_tensor_to_fifo,
        args=(fifo_path, tensor),
        daemon=True,
    )
    t.start()
    try:
        yield base
    finally:
        t.join()
        _safe_unlink(fifo_path)
        _safe_unlink(hdr_path)


@contextmanager
def cfl_output_fifo(
    dims: list[int],
    device: str | torch.device = "cpu",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Pre-allocate an output BartTensor and expose it as a named FIFO pair.

    Yields ``(base_path, tensor)``; after the context exits the tensor contains
    the data written by BART.
    """
    base = _unique_base("_bt_out_")
    hdr_path = base + ".hdr"
    fifo_path = base + ".fifo"

    out = bart_empty(dims, device=device)
    nbytes = out.numel() * out.element_size()

    # Write a placeholder header — BART will write the real one via its mmio.
    _write_cfl_header(hdr_path, dims)
    os.mkfifo(fifo_path)

    t = threading.Thread(
        target=_stream_fifo_to_tensor,
        args=(fifo_path, out, nbytes),
        daemon=True,
    )
    t.start()
    try:
        yield base, out
    finally:
        t.join()
        _safe_unlink(fifo_path)
        _safe_unlink(hdr_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fifo(
    op_name: str,
    inputs: list[Any],
    output_dims: list[int] | None,
    **kwargs,
) -> BartTensor:
    """Run a BART tool via named FIFOs (no-disk subprocess fallback).

    Parameters
    ----------
    op_name:
        BART tool name (e.g. ``"fft"``).
    inputs:
        List of array-valued inputs.
    output_dims:
        Expected output shape.  If ``None``, a dry-run is attempted first to
        determine output dimensions (not yet implemented — raises if ``None``).
    **kwargs:
        Flag / scalar arguments.

    Returns
    -------
    BartTensor
    """
    import shutil

    bart_bin = shutil.which("bart")
    if bart_bin is None:
        raise RuntimeError(
            "BART executable not found on PATH and the C++ extension is not "
            "available.  Install BART or build the bartorch C++ extension."
        )

    if output_dims is None:
        raise NotImplementedError(
            "Automatic output dimension inference is not yet implemented for "
            "the FIFO fallback path.  Please provide output_dims explicitly."
        )

    # Build argv list
    argv = [bart_bin, op_name]
    argv += _flags_to_argv(kwargs)

    with ExitStack() as stack:
        # Register input FIFOs
        for inp in inputs:
            if isinstance(inp, (torch.Tensor, np.ndarray)):
                base = stack.enter_context(cfl_input_fifo(inp))
                argv.append(base)
            else:
                argv.append(str(inp))

        # Register output FIFO
        out_base, out_tensor = stack.enter_context(
            cfl_output_fifo(output_dims)
        )
        argv.append(out_base)

        # Run BART
        ret = subprocess.call(argv)
        if ret != 0:
            raise RuntimeError(
                f"BART command '{op_name}' exited with code {ret}."
            )

    return out_tensor


# ---------------------------------------------------------------------------
# Argv helpers
# ---------------------------------------------------------------------------

def _flags_to_argv(kwargs: dict) -> list[str]:
    """Convert keyword arguments to BART flag strings."""
    argv = []
    for key, val in kwargs.items():
        if val is None or val is False or key.startswith("_"):
            continue
        if val is True:
            argv.append(f"-{key}")
        else:
            argv += [f"-{key}", str(val)]
    return argv


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# Lazy import to avoid circular dependency at module level
try:
    from contextlib import ExitStack
except ImportError:
    pass  # Python 3.3+ always has ExitStack
