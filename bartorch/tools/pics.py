"""Reconstruction tools — bartorch.tools.pics.

Wraps BART's calibration and parallel-imaging compressed-sensing commands:

* :func:`ecalib`  — ESPIRiT coil sensitivity estimation
* :func:`caldir`  — Direct coil calibration
* :func:`pics`    — Parallel Imaging Compressed Sensing reconstruction

All three functions expose the **full BART CLI API** through their keyword
arguments.  Named parameters cover the most commonly used flags; anything
else can be forwarded as ``**extra_flags`` (keyword name = BART flag letter,
value = flag argument or ``True`` for bare flags).

BART flag reference
-------------------
The mapping from Python keyword to BART flag is direct: ``r=0.01`` → ``-r
0.01``, ``R="W:7:0:0.001"`` → ``-R W:7:0:0.001``, ``e=True`` → ``-e``.
"""

from __future__ import annotations

from typing import Any

import torch

from bartorch.core.graph import dispatch
from bartorch.core.tensor import bart_op

__all__ = ["ecalib", "caldir", "pics"]


@bart_op
def ecalib(
    kspace: torch.Tensor,
    *,
    calib_size: int | None = None,
    maps: int = 1,
    threshold: float | None = None,
    crop: float | None = None,
    softcrop: bool = False,
    intensity: bool = False,
    weighting: bool = False,
    **extra_flags: Any,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using ESPIRiT.

    ESPIRiT computes sensitivity maps directly from the auto-calibration signal
    (ACS) region of k-space by eigen-decomposition of a calibration matrix.

    Parameters
    ----------
    kspace : torch.Tensor
        Fully-sampled calibration k-space or k-space containing an ACS region.
        Expected shape (C-order): ``(ncoils[, nz], ny, nx)``.
    calib_size : int, optional
        Size of the calibration region in each dimension (``-r``).
        ``None`` → BART auto-detects.
    maps : int, optional
        Number of ESPIRiT maps to compute (``-m``).  ``1`` (default) is
        sufficient for most applications; ``2`` handles phase singularities.
    threshold : float, optional
        Singular-value threshold for the calibration matrix (``-t``).
        ``None`` → BART default (``0.001``).
    crop : float, optional
        Crop sensitivity maps below this image-domain threshold (``-c``).
    softcrop : bool, optional
        Use soft-crop (``-S``).  Default ``False``.
    intensity : bool, optional
        Intensity-correction (``-I``).  Default ``False``.
    weighting : bool, optional
        Apply k-space weighting (``-W``).  Default ``False``.
    **extra_flags :
        Any additional BART ``ecalib`` flags passed directly (e.g.
        ``v=True`` for verbose output).

    Returns
    -------
    torch.Tensor
        Complex64 sensitivity maps (C-order).

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens = bt.ecalib(kspace, calib_size=24)
    """
    return dispatch(
        "ecalib",
        [kspace],
        None,
        r=calib_size,
        m=maps,
        t=threshold,
        c=crop,
        S=softcrop or None,
        I=intensity or None,
        W=weighting or None,
        **extra_flags,
    )


@bart_op
def caldir(
    kspace: torch.Tensor,
    *,
    calib_size: int,
    **extra_flags: Any,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using direct calibration (CALDIR).

    A simpler, faster alternative to :func:`ecalib` that computes sensitivity
    maps by direct Fourier-space operations on the ACS region.

    Parameters
    ----------
    kspace : torch.Tensor
        Calibration k-space (C-order).
    calib_size : int
        Size of the calibration region (positional BART argument).
    **extra_flags :
        Additional BART ``caldir`` flags.

    Returns
    -------
    torch.Tensor
        Complex64 sensitivity maps.

    Examples
    --------
    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens = bt.caldir(kspace, calib_size=24)
    """
    return dispatch("caldir", [kspace], None, r=calib_size, **extra_flags)


@bart_op
def pics(
    kspace: torch.Tensor,
    sens: torch.Tensor,
    *,
    # Regularisation
    lambda_: float | None = None,
    R: str | None = None,
    l1: bool = False,
    l2: bool = False,
    # Solver
    iter_: int = 30,
    inner_iter: int | None = None,
    tol: float | None = None,
    step: float | None = None,
    admm: bool = False,
    admm_lambda: float | None = None,
    cg_lambda: float | None = None,
    # Output
    real: bool = False,
    gpu: bool = False,
    # Advanced
    subspace_basis: str | None = None,
    init: str | None = None,
    fast_est: bool = False,
    **extra_flags: Any,
) -> torch.Tensor:
    """Parallel Imaging Compressed Sensing (PICS) reconstruction.

    Iteratively reconstructs an image from under-sampled k-space data using
    sensitivity encoding and compressed-sensing regularisation.

    Parameters
    ----------
    kspace : torch.Tensor
        Under-sampled k-space data (C-order).  Zero-padded entries indicate
        missing samples.
    sens : torch.Tensor
        Coil sensitivity maps, typically from :func:`ecalib` or
        :func:`caldir`.

    Regularisation
    --------------
    lambda_ : float, optional
        Regularisation strength λ (``-r``).  Used with ``-l1``/``-l2`` or as
        the default lambda for ``-R`` specs.
    R : str, optional
        Full BART regularisation specification (``-R``), e.g.
        ``"W:7:0:0.001"`` (wavelet), ``"T:7:0:0.001"`` (total variation),
        ``"L:7:0:0.001"`` (locally low-rank).  Multiple regularisers can be
        applied by calling ``pics`` with additional ``R`` flags via
        *extra_flags*.
    l1 : bool, optional
        L1 regularisation flag (``-l1``).  Default ``False``.
    l2 : bool, optional
        L2 (Tikhonov) regularisation flag (``-l2``).  Default ``False``.

    Solver
    ------
    iter_ : int, optional
        Maximum outer solver iterations (``-i``).  Default ``30``.
    inner_iter : int, optional
        Number of inner CG iterations (``-n``).
    tol : float, optional
        Convergence tolerance (``-t``).  ``None`` → iterate for exactly
        *iter_* steps.
    step : float, optional
        Step size (``-s``).
    admm : bool, optional
        Use ADMM algorithm (``-a``).  Default ``False``.
    admm_lambda : float, optional
        ADMM penalty parameter (``-A``).
    cg_lambda : float, optional
        Tikhonov regularisation for the inner CG solve (``-K``).

    Output
    ------
    real : bool, optional
        Cast result to real-valued image (``-c``).  Default ``False``.
    gpu : bool, optional
        Use GPU (``-g``).  Requires BART compiled with CUDA.  Default
        ``False``.

    Advanced
    --------
    subspace_basis : str, optional
        Path to a CFL subspace basis (``-B``).
    init : str, optional
        Warm-start initialisation CFL (``-W``).
    fast_est : bool, optional
        Fast operator-norm estimation (``-e``).  Default ``False``.
    **extra_flags :
        Any additional BART ``pics`` flags not listed above, passed directly.
        For example ``P=0.1`` → ``-P 0.1``, ``N=True`` → ``-N``.

    Returns
    -------
    torch.Tensor
        Reconstructed complex image (C-order).

    Examples
    --------
    Basic PICS with wavelet regularisation (``-R W:7:0:lambda``):

    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens   = bt.ecalib(kspace, calib_size=24)
    >>> reco   = bt.pics(kspace, sens, R="W:7:0:0.005")

    L1-Wavelet shorthand (equivalent to ``R="W:7:0:lambda_"``):

    >>> reco = bt.pics(kspace, sens, lambda_=0.01, l1=True)

    Total variation + L2 Tikhonov, ADMM solver:

    >>> reco = bt.pics(kspace, sens, R="T:7:0:0.01", l2=True, lambda_=1e-4, admm=True)

    Multiple regularisers — use ``**extra_flags`` with any BART flag name.
    BART allows repeated ``-R`` specifications; pass them as differently-named
    extra kwargs and note that the C++ layer receives them as separate flags:

    >>> reco = bt.pics(kspace, sens, R="W:7:0:0.005", **{"R ": "T:7:0:0.002"})

    Pass arbitrary BART flags directly:

    >>> reco = bt.pics(kspace, sens, R="L:7:7:0.01", N=True, u=True)
    """
    return dispatch(
        "pics",
        [kspace, sens],
        None,
        r=lambda_,
        R=R,
        l1=l1 or None,
        l2=l2 or None,
        i=iter_,
        n=inner_iter,
        t=tol,
        s=step,
        a=admm or None,
        A=admm_lambda,
        K=cg_lambda,
        c=real or None,
        g=gpu or None,
        B=subspace_basis,
        W=init,
        e=fast_est or None,
        **extra_flags,
    )
