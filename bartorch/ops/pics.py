"""Reconstruction ops — bartorch.ops.pics."""

from __future__ import annotations

from bartorch.core.graph import dispatch
from bartorch.core.tensor import BartTensor


def ecalib(
    kspace: BartTensor,
    *,
    calib_size: int | None = None,
    maps: int = 1,
    threshold: float | None = None,
) -> BartTensor:
    """ESPIRiT coil sensitivity estimation.

    Parameters
    ----------
    kspace:
        Fully-sampled calibration k-space (or fully-sampled ACS region).
    calib_size:
        Calibration region size.  ``None`` → auto-detect.
    maps:
        Number of ESPIRiT maps to compute.
    threshold:
        Singular-value threshold.

    Returns
    -------
    BartTensor
        Coil sensitivity maps.
    """
    return dispatch(
        "ecalib",
        [kspace],
        None,
        r=calib_size,
        m=maps,
        t=threshold,
    )


def caldir(
    kspace: BartTensor,
    *,
    calib_size: int,
) -> BartTensor:
    """Direct calibration (caldir)."""
    return dispatch("caldir", [kspace], None, r=calib_size)


def pics(
    kspace: BartTensor,
    sens: BartTensor,
    *,
    lambda_: float = 0.01,
    iter_: int = 30,
    tol: float | None = None,
    wav: bool = False,
    l1: bool = False,
    l2: bool = False,
) -> BartTensor:
    """Parallel Imaging Compressed Sensing (PICS) reconstruction.

    Parameters
    ----------
    kspace:
        Under-sampled k-space data.
    sens:
        Coil sensitivity maps.
    lambda_:
        Regularisation strength.
    iter_:
        Number of iterations.
    tol:
        Convergence tolerance.
    wav:
        Use wavelet regularisation.
    l1:
        Use L1 regularisation (default).
    l2:
        Use L2 (Tikhonov) regularisation.

    Returns
    -------
    BartTensor
        Reconstructed image.
    """
    return dispatch(
        "pics",
        [kspace, sens],
        None,
        r=lambda_,
        i=iter_,
        t=tol,
        RW=wav or None,
        l1=l1 or None,
        l2=l2 or None,
    )
