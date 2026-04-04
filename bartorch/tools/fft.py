"""FFT tools — bartorch.tools.fft.

Provides multidimensional FFT and inverse-FFT via BART's ``num/fft`` module.
All functions accept plain ``torch.Tensor`` inputs (cast to ``complex64``
automatically) and return a plain ``torch.Tensor``.

Axes are specified as C-order Python indices (including negative indices)
rather than a BART bitmask; the conversion to BART's Fortran-order bitmask is
handled transparently by :func:`~bartorch.utils.flags.axes_to_flags`.
"""

from __future__ import annotations

import torch

from bartorch.core.graph import dispatch
from bartorch.core.tensor import bart_op
from bartorch.utils.flags import axes_to_flags

__all__ = ["fft", "ifft"]


@bart_op
def fft(
    input: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    unitary: bool = False,
    inverse: bool = False,
) -> torch.Tensor:
    """Multidimensional (i)FFT via BART's num/fft module.

    Parameters
    ----------
    input : torch.Tensor
        Input array (any dtype; cast to ``complex64`` automatically).
    axes : int or sequence of int
        C-order axis index or indices to transform.  Negative values are
        supported.  Examples:

        * ``axes=-1``        — transform the last (read) axis only
        * ``axes=(-1, -2)``  — transform the last two axes (typical 2-D FFT)
        * ``axes=(0, 1, 2)`` — transform the first three axes
    unitary : bool, optional
        Apply unitary (1/√N) normalisation.  Default ``False``.
    inverse : bool, optional
        Compute inverse FFT.  Default ``False`` (forward FFT).

    Returns
    -------
    torch.Tensor
        Complex64 tensor with the same shape as *input*.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> ph = bt.phantom([256, 256])
    >>> kspace = bt.fft(ph, axes=(-1, -2))   # 2-D FFT
    >>> kspace.shape
    torch.Size([1, 256, 256])
    """
    flags = axes_to_flags(axes, ndim=input.ndim)
    return dispatch(
        "fft",
        [input],
        None,
        flags=flags,
        u=unitary,
        i=inverse,
    )


@bart_op
def ifft(
    input: torch.Tensor,
    axes: int | tuple[int, ...] | list[int],
    *,
    unitary: bool = False,
) -> torch.Tensor:
    """Inverse multidimensional FFT — convenience alias for ``fft(..., inverse=True)``.

    Parameters
    ----------
    input : torch.Tensor
        K-space data.
    axes : int or sequence of int
        C-order axis index or indices to transform back to image space.
    unitary : bool, optional
        Apply unitary normalisation.  Default ``False``.

    Returns
    -------
    torch.Tensor
        Image-space complex64 tensor.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> ph     = bt.phantom([256, 256])
    >>> kspace = bt.fft(ph, axes=(-1, -2))
    >>> img    = bt.ifft(kspace, axes=(-1, -2))
    """
    flags = axes_to_flags(axes, ndim=input.ndim)
    return dispatch(
        "fft",
        [input],
        None,
        flags=flags,
        u=unitary,
        i=True,
    )
