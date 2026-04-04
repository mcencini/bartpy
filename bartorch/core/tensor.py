"""
BartTensor — a ``torch.Tensor`` subclass that carries metadata required for
zero-copy interoperability with BART's in-memory CFL registry.

A ``BartTensor`` always stores data as ``torch.complex64`` in column-major
(Fortran) order — matching BART's native ``complex float*`` layout — and
records the logical BART dimension tuple (up to 16 dims as per ``DIMS`` in
BART).

Key constraints
---------------
* dtype  : ``torch.complex64`` (``_Complex float`` in C)
* strides: Fortran (column-major) — created via ``torch.empty(...).t()``
  or by allocating with reversed strides from the C++ extension.
* device : CPU or CUDA (GPU BartTensors carry a device pointer that BART's
  CUDA-enabled kernels can consume directly when compiled with ``USE_CUDA``).

The subclass participates in ``__torch_function__`` so that operations that
mix ``BartTensor`` inputs stay on the hot path wherever possible, and fall
back to copies + regular torch ops otherwise.
"""

from __future__ import annotations

import torch


class BartTensor(torch.Tensor):
    """Torch tensor tagged for zero-copy BART hot-path dispatch.

    Do **not** instantiate directly; use :func:`bart_empty`,
    :func:`bart_zeros`, :func:`bart_from_numpy`, or :func:`bart_from_tensor`
    instead.
    """

    # Internal slot: unique name in the BART in-memory CFL registry.
    _bart_name: str | None

    @staticmethod
    def __new__(cls, data: torch.Tensor, bart_name: str | None = None):
        # ``torch.Tensor.__new__`` returns a view; the subclass attribute is
        # attached afterwards in ``__init__``.
        instance = torch.Tensor._make_subclass(cls, data)
        instance._bart_name = bart_name
        return instance

    def __repr__(self):
        return f"BartTensor(shape={tuple(self.shape)}, " \
               f"dtype={self.dtype}, device={self.device}, " \
               f"bart_name={self._bart_name!r})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # All inputs that are BartTensors → attempt hot path via registered
        # dispatch table; unknown ops fall through to the standard path which
        # may return a plain torch.Tensor.
        return super().__torch_function__(func, types, args, kwargs)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def bart_empty(
    dims: list[int],
    device: str | torch.device = "cpu",
) -> BartTensor:
    """Allocate an uninitialised ``BartTensor`` with Fortran-order strides.

    Parameters
    ----------
    dims:
        BART dimension list (up to 16 elements; trailing 1-dims may be
        omitted).
    device:
        ``"cpu"`` or ``"cuda"`` / ``torch.device``.

    Returns
    -------
    BartTensor
        Uninitialised column-major complex64 tensor.
    """
    # Allocate in C-order then permute so the underlying storage is Fortran.
    # torch.empty with memory_format=torch.contiguous_format is row-major;
    # we want column-major (last dim varies slowest in memory).
    t = torch.empty(
        dims[::-1],  # reversed for Fortran order
        dtype=torch.complex64,
        device=device,
    ).contiguous()
    # Restore logical shape via a view with reversed strides.
    t = t.as_strided(
        dims,
        _fortran_strides(dims),
    )
    return BartTensor(t)


def bart_zeros(
    dims: list[int],
    device: str | torch.device = "cpu",
) -> BartTensor:
    """Like :func:`bart_empty` but zero-initialised."""
    t = bart_empty(dims, device=device)
    t.zero_()
    return t


def bart_from_tensor(
    t: torch.Tensor,
    copy: bool = True,
) -> BartTensor:
    """Wrap or copy a ``torch.Tensor`` into a ``BartTensor``.

    Parameters
    ----------
    t:
        Input tensor.  Must be ``complex64``; a cast is performed otherwise.
    copy:
        When ``False``, attempt a zero-copy view (only possible when *t* is
        already column-major ``complex64`` on CPU or CUDA).

    Returns
    -------
    BartTensor
        Fortran-order complex64 tensor sharing (or copying) *t*'s data.
    """
    if t.dtype != torch.complex64:
        t = t.to(torch.complex64)

    expected = _fortran_strides(list(t.shape))
    is_fortran = (list(t.stride()) == expected)

    if copy or not is_fortran:
        out = bart_empty(list(t.shape), device=t.device)
        out.copy_(t)
        return out

    return BartTensor(t)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fortran_strides(dims: list[int]) -> list[int]:
    """Compute column-major (Fortran) strides for the given shape."""
    strides = []
    s = 1
    for d in dims:
        strides.append(s)
        s *= d
    return strides
