"""bartorch.tools._commands — Special-case CLI wrappers that override generated ones.

The auto-generated layer in :mod:`bartorch.tools._generated` provides thin
wrappers for every BART command.  This module imports that full suite and then
re-defines a small set of commands that benefit from a richer Python API.
The overrides are purely Pythonic argument-translation layers — each one calls
its auto-generated counterpart in :mod:`bartorch.tools._generated` after
mapping human-readable keyword names to the raw BART flag letters.

* :func:`ecalib` — maps Pythonic keyword names (``calib_size``, ``maps``, …)
  to raw BART flags, then calls :func:`_generated.ecalib`
* :func:`caldir` — maps ``calib_size`` to the BART ``-r`` flag, then calls
  :func:`_generated.caldir`
* :func:`pics`   — ``R`` accepts ``list[str]`` for multiple regularisers;
  translates Pythonic kwargs to BART flags and calls :func:`_generated.pics`

All other commands are re-exported unchanged.
The ``__init__.py`` imports from this module to build the public API.
"""

from __future__ import annotations

from typing import Any

import torch

# Module-level reference used by the override implementations below.
from bartorch.tools import _generated

# Re-export the full generated suite so that ``from _commands import *`` in
# __init__.py exposes every BART command.  The manually-defined functions that
# follow shadow the generated versions of the same name.
from bartorch.tools._generated import *  # noqa: F401,F403
from bartorch.tools._generated import __all__ as _generated_all

__all__ = list(_generated_all)


# ---------------------------------------------------------------------------
# ESPIRiT calibration — Pythonic keyword → BART flag mapping
# ---------------------------------------------------------------------------


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

    .. note::
        **CUDA:** CPU only.  CUDA tensors are automatically moved to CPU before
        dispatch and returned to the original device.

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
    return _generated.ecalib(
        kspace,
        r=calib_size,
        m=maps,
        t=threshold,
        c=crop,
        S=softcrop or None,
        I=intensity or None,
        W=weighting or None,
        **extra_flags,
    )


# ---------------------------------------------------------------------------
# Direct calibration
# ---------------------------------------------------------------------------


def caldir(
    kspace: torch.Tensor,
    *,
    calib_size: int,
    **extra_flags: Any,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using direct calibration (CALDIR).

    A simpler, faster alternative to :func:`ecalib` that computes sensitivity
    maps by direct Fourier-space operations on the ACS region.

    .. note::
        **CUDA:** CPU only.  CUDA tensors are automatically moved to CPU before
        dispatch and returned to the original device.

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
    return _generated.caldir(kspace, r=calib_size, **extra_flags)


# ---------------------------------------------------------------------------
# PICS regularisation guide
# ---------------------------------------------------------------------------

_R_GUIDE = """\
The ``-R`` flag selects a regulariser and is the primary way to configure the
compressed-sensing penalty in PICS.  It uses the syntax::

    "<type>:<transform_flags>:<joint_dims>:<lambda>"

**type** selects the penalty:

* ``W`` — Wavelet (ℓ₁-Wavelet)
* ``T`` — Total Variation (TV)
* ``L`` — Locally Low-Rank (LLR)
* ``B`` — Block-wise Low-Rank
* ``N`` — Nuclear-norm Low-Rank

**transform_flags** is a BART bitmask of the axes to which the transform
is applied.  Use :func:`bartorch.utils.flags.axes_to_flags` to compute it
from C-order Python axis indices.

Common values for a ``(coils, ny, nx)`` k-space (3-D, C-order):

* ``7``  — all three spatial axes (read + phase1 + phase2 in BART)
* ``3``  — last two axes only (phase1 + read)
* ``4``  — first axis only (coils / z)

**joint_dims** selects which additional dimensions are processed jointly
(e.g. for temporal or multi-contrast data); ``0`` = no joint processing.

**lambda** is the regularisation strength (float > 0; larger = more
regularisation).

Multiple regularisers can be stacked by passing a list to ``R``.

Examples
--------
Single wavelet::

    pics(kspace, sens, R="W:7:0:0.005")

TV + wavelet::

    pics(kspace, sens, R=["T:7:0:0.002", "W:7:0:0.005"])

L1-shorthand (equivalent to ``R="W:7:0:lambda_"``)::

    pics(kspace, sens, lambda_=0.01, l1=True)
"""


# ---------------------------------------------------------------------------
# PICS reconstruction
# ---------------------------------------------------------------------------


def pics(
    kspace: torch.Tensor,
    sens: torch.Tensor,
    *,
    # Regularisation
    lambda_: float | None = None,
    R: list[str] | str | None = None,
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

    .. note::
        **CUDA:** GPU-capable when compiled with ``USE_CUDA=ON`` (``gpu=True``).
        When ``gpu=False`` (default), CUDA tensors are automatically moved to
        CPU before dispatch and returned to the original device.

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
    R : str or list of str, optional
        BART regularisation specification(s) (``-R``).

""" + _R_GUIDE + """
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
        Use GPU via BART's internal CUDA support (``-g``).  Requires BART
        compiled with ``USE_CUDA=ON``.  Default ``False``.

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
    Basic PICS with wavelet regularisation:

    >>> import bartorch.tools as bt
    >>> kspace = bt.phantom([256, 256], kspace=True, ncoils=8)
    >>> sens   = bt.ecalib(kspace, calib_size=24)
    >>> reco   = bt.pics(kspace, sens, R="W:7:0:0.005")

    Multiple regularisers (TV + wavelet):

    >>> reco = bt.pics(kspace, sens, R=["T:7:0:0.002", "W:7:0:0.005"])

    L1-Wavelet shorthand:

    >>> reco = bt.pics(kspace, sens, lambda_=0.01, l1=True)

    ADMM solver with TV:

    >>> reco = bt.pics(kspace, sens, R="T:7:0:0.01", admm=True)
    """
    return _generated.pics(
        kspace,
        sens,
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

