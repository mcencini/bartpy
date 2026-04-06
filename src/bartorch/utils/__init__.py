"""
bartorch.utils — Utility functions.

CFL read/write for NumPy workflows.

.. note::
    :func:`~bartorch.utils.flags.axes_to_flags` is an internal helper used by
    the ``bartorch.tools`` wrappers.  It is not part of the public API; import
    it from :mod:`bartorch.utils.flags` if you need it directly.
"""

from __future__ import annotations

__all__ = ["readcfl", "writecfl"]

from bartorch.utils.cfl import readcfl, writecfl
