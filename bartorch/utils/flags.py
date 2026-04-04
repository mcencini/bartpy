"""
bartorch.utils.flags — Axis-index → BART bitmask conversion.

BART internally numbers axes in Fortran (column-major) order.  bartorch
exposes a C-order (row-major) API matching NumPy/PyTorch conventions.
This module provides :func:`axes_to_flags` to convert C-order axis indices
(including negative indices) into the BART bitmask expected by tools such as
``fft``.

Axis mapping for a tensor with *ndim* dimensions
-------------------------------------------------

============  ============================  =================================
Python index  C-order meaning               BART Fortran axis (→ flag bit)
============  ============================  =================================
``0``         slowest-varying (e.g. coils)  ``ndim - 1``
``1``         …                             ``ndim - 2``
``ndim - 1``  fastest-varying (e.g. read)   ``0``
``-1``        same as ``ndim - 1``          ``0``
``-2``        same as ``ndim - 2``          ``1``
============  ============================  =================================

Examples
--------
FFT over the last two axes of a 3-D tensor (coils, ny, nx):

>>> axes_to_flags((1, 2), ndim=3)   # ny and nx → BART axes 1 and 0
3
>>> axes_to_flags((-1, -2), ndim=3)  # same
3

Single axis:

>>> axes_to_flags(0, ndim=2)  # first axis of a 2-D tensor → BART axis 1
2
"""

from __future__ import annotations

__all__ = ["axes_to_flags"]


def axes_to_flags(
    axes: int | tuple[int, ...] | list[int],
    ndim: int,
) -> int:
    """Convert C-order axis indices to a BART bitmask (Fortran-order bits).

    Parameters
    ----------
    axes : int or sequence of int
        C-order axis index or indices.  Negative values are supported
        (e.g. ``-1`` is the last axis).
    ndim : int
        Number of dimensions of the array the axes refer to.  Must be ≥ 1.

    Returns
    -------
    int
        BART bitmask where bit *k* set means "transform along BART Fortran
        axis *k*".

    Raises
    ------
    ValueError
        If *ndim* < 1, if any axis is out of ``[-ndim, ndim-1]``, or if
        the same axis appears more than once.

    Examples
    --------
    >>> axes_to_flags((-1, -2), ndim=3)
    3
    >>> axes_to_flags((0, 1, 2), ndim=3)
    7
    >>> axes_to_flags(2, ndim=3)
    1
    """
    if ndim < 1:
        raise ValueError(f"ndim must be >= 1, got {ndim}")

    if isinstance(axes, int):
        axes = (axes,)
    else:
        axes = tuple(axes)

    normalised: list[int] = []
    for orig in axes:
        a = orig + ndim if orig < 0 else orig
        if a < 0 or a >= ndim:
            raise ValueError(
                f"axis {orig} out of range for ndim={ndim} "
                f"(valid range: [{-ndim}, {ndim - 1}])"
            )
        normalised.append(a)

    if len(normalised) != len(set(normalised)):
        raise ValueError("duplicate axis indices are not allowed")

    flags = 0
    for a in normalised:
        bart_axis = ndim - 1 - a
        flags |= 1 << bart_axis

    return flags
