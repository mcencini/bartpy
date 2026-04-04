"""
bartorch.core.tensor — axis-convention utilities for the BART bridge.

All public bartorch functions accept and return plain ``torch.Tensor`` objects
(``dtype=torch.complex64``).  BART internally uses Fortran (column-major)
order; bartorch users work in standard C (row-major) / PyTorch convention.

The axis reversal is handled transparently at two places:

* **C++ hot path** — ``bartorch/csrc/tensor_bridge.hpp`` reverses the
  dimension array before calling ``register_mem_cfl_non_managed()``.  A
  C-order ``(coils, ny, nx)`` tensor reinterpreted with reversed dims
  ``(nx, ny, coils)`` in Fortran order has the *same* byte layout — zero copy.

* **Subprocess fallback** — ``bartorch.pipe.cfl_tmp`` reverses dims in the
  CFL header and writes raw C-order bytes.  BART reads them as Fortran
  ``(nx, ny, coils)`` — again no data movement.

Public API
----------
as_complex64(t)  — cast to complex64, zero-copy if already correct
reverse_dims(dims) — reverse a dimension list for the C ↔ Fortran bridge
"""

from __future__ import annotations

import torch

__all__ = ["as_complex64", "reverse_dims"]


def as_complex64(t: torch.Tensor) -> torch.Tensor:
    """Return *t* cast to ``torch.complex64``, zero-copy if already correct.

    Parameters
    ----------
    t:
        Input tensor of any dtype.

    Returns
    -------
    torch.Tensor
        Complex64 tensor.  Same storage as *t* when the dtype already matches;
        a cast copy otherwise.
    """
    if t.dtype == torch.complex64:
        return t
    return t.to(torch.complex64)


def reverse_dims(dims: list[int] | tuple[int, ...]) -> list[int]:
    """Reverse a dimension list to map between C-order and Fortran-order.

    bartorch users express shapes in C-order (last index varies fastest in
    memory); BART uses Fortran order (first index varies fastest).  Reversing
    the dim list *without* touching the data switches between the two
    conventions because a C-order ``(a, b, c)`` array and a Fortran-order
    ``(c, b, a)`` array share identical byte layouts.

    Parameters
    ----------
    dims:
        Shape or dimension list.

    Returns
    -------
    list[int]
        Reversed dimension list.

    Examples
    --------
    >>> reverse_dims([8, 256, 256])   # (coils, ny, nx) → BART (nx, ny, coils)
    [256, 256, 8]
    """
    return list(reversed(dims))


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public API)
# ---------------------------------------------------------------------------


def _fortran_strides(dims: list[int]) -> list[int]:
    """Compute column-major (Fortran) strides for the given shape.

    Used internally by the subprocess fallback and tests.
    """
    strides: list[int] = []
    s = 1
    for d in dims:
        strides.append(s)
        s *= d
    return strides
