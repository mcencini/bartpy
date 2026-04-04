"""Reconstruction operations — bartorch.ops.pics.

Wraps BART's calibration and parallel-imaging compressed-sensing tools:

* :func:`ecalib`  — ESPIRiT coil sensitivity estimation
* :func:`caldir`  — Direct coil calibration
* :func:`pics`    — Parallel Imaging Compressed Sensing reconstruction
"""

from __future__ import annotations

from bartorch.core.graph import dispatch
from bartorch.core.tensor import BartTensor

__all__ = ["ecalib", "caldir", "pics"]


def ecalib(
    kspace: BartTensor,
    *,
    calib_size: int | None = None,
    maps: int = 1,
    threshold: float | None = None,
) -> BartTensor:
    """Estimate coil sensitivity maps using ESPIRiT.

    ESPIRiT computes sensitivity maps directly from the auto-calibration signal
    (ACS) region of the k-space by eigen-decomposition of a calibration matrix.

    Parameters
    ----------
    kspace : BartTensor
        Fully-sampled calibration k-space or k-space containing an ACS region.
        Expected shape: ``(nx, ny[, nz], ncoils)``.
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
    BartTensor
        Complex64 sensitivity maps of shape
        ``(nx, ny[, nz], ncoils, maps)``.

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> kspace = ops.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens = ops.ecalib(kspace, calib_size=24)
    >>> sens.shape
    torch.Size([256, 256, 1, 8, 1])
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
    """Estimate coil sensitivity maps using direct calibration (CALDIR).

    A simpler, faster alternative to :func:`ecalib` that computes sensitivity
    maps by direct Fourier-space operations on the ACS region.

    Parameters
    ----------
    kspace : BartTensor
        Calibration k-space.  Expected shape: ``(nx, ny[, nz], ncoils)``.
    calib_size : int
        Size of the calibration region.

    Returns
    -------
    BartTensor
        Complex64 sensitivity maps of shape ``(nx, ny[, nz], ncoils)``.

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> kspace = ops.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens = ops.caldir(kspace, calib_size=24)
    """
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

    Iteratively reconstructs an image from under-sampled k-space data using
    sensitivity encoding and optional compressed-sensing regularisation.

    Parameters
    ----------
    kspace : BartTensor
        Under-sampled k-space data.  Zero-padded entries indicate missing
        samples.  Expected shape: ``(nx, ny[, nz], ncoils)``.
    sens : BartTensor
        Coil sensitivity maps, typically from :func:`ecalib` or
        :func:`caldir`.  Shape must be compatible with *kspace*.
    lambda_ : float, optional
        Regularisation strength (λ).  Default ``0.01``.
    iter_ : int, optional
        Maximum number of solver iterations.  Default ``30``.
    tol : float, optional
        Convergence tolerance.  Iteration stops when the relative residual
        falls below *tol*.  ``None`` → iterate for exactly *iter_* steps.
    wav : bool, optional
        Apply wavelet (L1-Wavelet) regularisation.  Default ``False``.
    l1 : bool, optional
        Apply L1 regularisation (default when neither ``wav`` nor ``l2`` is
        set).  Default ``False``.
    l2 : bool, optional
        Apply L2 (Tikhonov) regularisation.  Default ``False``.

    Returns
    -------
    BartTensor
        Reconstructed complex image of shape ``(nx, ny[, nz])``.

    Examples
    --------
    >>> import bartorch.ops as ops
    >>> kspace = ops.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens   = ops.ecalib(kspace, calib_size=24)
    >>> reco   = ops.pics(kspace, sens, lambda_=0.005, wav=True)
    >>> reco.shape
    torch.Size([256, 256, 1])
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
