"""Reconstruction operations — bartorch.ops.pics.

Wraps BART's calibration and parallel-imaging compressed-sensing tools:

* :func:`ecalib`  — ESPIRiT coil sensitivity estimation
* :func:`caldir`  — Direct coil calibration
* :func:`pics`    — Parallel Imaging Compressed Sensing reconstruction
"""

from __future__ import annotations

import torch

from bartorch.core.graph import dispatch

__all__ = ["ecalib", "caldir", "pics"]


def ecalib(
    kspace: torch.Tensor,
    *,
    calib_size: int | None = None,
    maps: int = 1,
    threshold: float | None = None,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using ESPIRiT.

    ESPIRiT computes sensitivity maps directly from the auto-calibration signal
    (ACS) region of the k-space by eigen-decomposition of a calibration matrix.

    Parameters
    ----------
    kspace : torch.Tensor
        Fully-sampled calibration k-space or k-space containing an ACS region.
        Expected shape (C-order): ``(ncoils[, nz], ny, nx)``.
    calib_size : int, optional
        Size of the calibration region in each dimension.
        ``None`` → BART auto-detects from the data.
    maps : int, optional
        Number of ESPIRiT maps to compute.  ``1`` (default) is sufficient for
        most applications; ``2`` handles phase singularities.
    threshold : float, optional
        Singular-value threshold for the calibration matrix.
        ``None`` → use BART default (``0.001``).

    Returns
    -------
    torch.Tensor
        Complex64 sensitivity maps (C-order).

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> kspace = ops.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens = ops.ecalib(kspace, calib_size=24)
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
    kspace: torch.Tensor,
    *,
    calib_size: int,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using direct calibration (CALDIR).

    A simpler, faster alternative to :func:`ecalib` that computes sensitivity
    maps by direct Fourier-space operations on the ACS region.

    Parameters
    ----------
    kspace : torch.Tensor
        Calibration k-space (C-order).
    calib_size : int
        Size of the calibration region.

    Returns
    -------
    torch.Tensor
        Complex64 sensitivity maps.

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> kspace = ops.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens = ops.caldir(kspace, calib_size=24)
    """
    return dispatch("caldir", [kspace], None, r=calib_size)


def pics(
    kspace: torch.Tensor,
    sens: torch.Tensor,
    *,
    lambda_: float = 0.01,
    iter_: int = 30,
    tol: float | None = None,
    wav: bool = False,
    l1: bool = False,
    l2: bool = False,
) -> torch.Tensor:
    """Parallel Imaging Compressed Sensing (PICS) reconstruction.

    Iteratively reconstructs an image from under-sampled k-space data using
    sensitivity encoding and optional compressed-sensing regularisation.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space data (C-order).  Zero-padded entries indicate
        missing samples.
    sens : torch.Tensor
        Coil sensitivity maps, typically from :func:`ecalib` or
        :func:`caldir`.
    lambda_ : float, optional
        Regularisation strength (λ).  Default ``0.01``.
    iter_ : int, optional
        Maximum number of solver iterations.  Default ``30``.
    tol : float, optional
        Convergence tolerance.  ``None`` → iterate for exactly *iter_* steps.
    wav : bool, optional
        Apply wavelet (L1-Wavelet) regularisation.  Default ``False``.
    l1 : bool, optional
        Apply L1 regularisation.  Default ``False``.
    l2 : bool, optional
        Apply L2 (Tikhonov) regularisation.  Default ``False``.

    Returns
    -------
    torch.Tensor
        Reconstructed complex image (C-order).

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> kspace = ops.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens   = ops.ecalib(kspace, calib_size=24)
    >>> reco   = ops.pics(kspace, sens, lambda_=0.005, wav=True)
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
