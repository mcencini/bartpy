"""FFT operations — bartorch.ops.fft."""

from __future__ import annotations

from bartorch.core.graph import dispatch
from bartorch.core.tensor import BartTensor


def fft(
    input: BartTensor,
    flags: int,
    *,
    unitary: bool = False,
    inverse: bool = False,
    centered: bool = True,
) -> BartTensor:
    """Multidimensional (i)FFT via BART's num/fft module.

    Parameters
    ----------
    input:
        Input array (BartTensor or any tensor/ndarray that will be promoted).
    flags:
        Bitmask selecting the dimensions to transform.
    unitary:
        Apply unitary normalization.
    inverse:
        Compute inverse FFT.
    centered:
        Use centered (orthogonal) FFT convention.

    Returns
    -------
    BartTensor
    """
    return dispatch(
        "fft",
        [input],
        None,
        flags=flags,
        u=unitary,
        i=inverse,
        # centered is the default in BART; no flag needed unless False
    )


def ifft(
    input: BartTensor,
    flags: int,
    *,
    unitary: bool = False,
    centered: bool = True,
) -> BartTensor:
    """Convenience wrapper: fft(..., inverse=True)."""
    return fft(input, flags, unitary=unitary, inverse=True, centered=centered)
