"""
bartorch.utils — Utility functions.

CFL read/write for NumPy workflows.

.. note::
    :func:`~bartorch.utils.flags._axes_to_flags` is an internal helper used
    by ``bartorch.tools`` wrappers to convert C-order axis indices to BART
    Fortran-order bitmasks.  It is **not** part of the public API.
"""

from __future__ import annotations

__all__ = ["readcfl", "writecfl"]

from bartorch.utils.cfl import readcfl, writecfl
