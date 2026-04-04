"""FFT operations — bartorch.ops.fft.

Provides multidimensional FFT and inverse-FFT via BART's ``num/fft`` module.
All functions accept plain ``torch.Tensor`` inputs (cast to ``complex64``
automatically) and return a plain ``torch.Tensor``.
"""

from __future__ import annotations

import torch

from bartorch.core.graph import dispatch

__all__ = ["fft", "ifft"]


def fft(
    input: torch.Tensor,
    flags: int,
    *,
    unitary: bool = False,
    inverse: bool = False,
    centered: bool = True,
) -> torch.Tensor:
    """Multidimensional (i)FFT via BART's num/fft module.

    Parameters
    ----------
    input : torch.Tensor
        Input array (any dtype; cast to ``complex64`` automatically).
    flags : int
        Bitmask selecting the dimensions to transform.
        Bit 0 → dimension 0 (readout), bit 1 → dimension 1 (phase), etc.
        Example: ``flags=6`` transforms dimensions 1 and 2.
    unitary : bool, optional
        Apply unitary (1/√N) normalisation.  Default ``False``.
    inverse : bool, optional
        Compute inverse FFT.  Default ``False`` (forward FFT).
    centered : bool, optional
        Use the centred (orthogonal) FFT convention (i.e. ``fftshift``
        applied before and after the transform).  Default ``True``.

    Returns
    -------
    torch.Tensor
        Complex64 tensor with the same shape as *input*.

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> ph = ops.phantom([256, 256])
    >>> kspace = ops.fft(ph, flags=3)          # transform dims 0 and 1
    >>> kspace.shape
    torch.Size([256, 256])
    """
    return dispatch(
        "fft",
        [input],
        None,
        flags=flags,
        u=unitary,
        i=inverse,
    )


def ifft(
    input: torch.Tensor,
    flags: int,
    *,
    unitary: bool = False,
    centered: bool = True,
) -> torch.Tensor:
    """Inverse multidimensional FFT — convenience alias for ``fft(..., inverse=True)``.

    Parameters
    ----------
    input : torch.Tensor
        K-space data.
    flags : int
        Bitmask selecting the dimensions to transform back to image space.
    unitary : bool, optional
        Apply unitary normalisation.  Default ``False``.
    centered : bool, optional
        Use centred FFT convention.  Default ``True``.

    Returns
    -------
    torch.Tensor
        Image-space complex64 tensor.

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> ph = ops.phantom([256, 256])
    >>> kspace  = ops.fft(ph, flags=3)
    >>> img_rec = ops.ifft(kspace, flags=3)
    """
    return fft(input, flags, unitary=unitary, inverse=True, centered=centered)
