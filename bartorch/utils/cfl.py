"""
bartorch.utils.cfl — NumPy-compatible CFL read/write.

Kept for compatibility with scripts that use the legacy bartpy CFL API.
Returns NumPy arrays (not BartTensors); use
:func:`bartorch.core.tensor.bart_from_tensor` to convert if needed.
"""

from __future__ import annotations

import os
import re

import numpy as np


_BART_DIMS = 16


def readcfl(name: str) -> np.ndarray:
    """Read a CFL file pair (``name.hdr`` / ``name.cfl``) into a NumPy array.

    Returns a Fortran-order ``complex64`` array with trailing singleton
    dimensions stripped.
    """
    # Read header
    with open(name + ".hdr") as f:
        lines = [l.strip() for l in f if not l.startswith("#")]
    dims = [int(x) for x in lines[0].split()]

    # Strip trailing 1s
    while dims and dims[-1] == 1:
        dims.pop()

    n = int(np.prod(dims)) if dims else 1
    arr = np.fromfile(name + ".cfl", dtype=np.complex64)

    if arr.size != n:
        raise ValueError(
            f"CFL data size mismatch: header says {n} elements, "
            f"file has {arr.size}."
        )

    return arr.reshape(dims, order="F")


def writecfl(name: str, array: np.ndarray) -> None:
    """Write a NumPy array as a CFL file pair.

    The array is cast to ``complex64`` and written in Fortran order.
    """
    array = np.asarray(array, dtype=np.complex64)
    dims = list(array.shape)

    # Pad to BART_DIMS
    padded = dims + [1] * (_BART_DIMS - len(dims))

    with open(name + ".hdr", "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in padded) + "\n")

    np.asarray(array, dtype=np.complex64).ravel(order='F').tofile(name + ".cfl")
