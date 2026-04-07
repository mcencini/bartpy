"""encoding_op — factory for persistent BART SENSE encoding operators."""

from __future__ import annotations

import torch

from bartorch.lib.linops import BartLinop


def encoding_op(
    maps: torch.Tensor,
    *,
    ksp_shape: tuple[int, ...] | None = None,
    pattern: torch.Tensor | None = None,
    traj: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
    gpu: bool = False,
) -> BartLinop:
    """Create a persistent BART SENSE encoding operator.

    Wraps BART's ``pics_model()`` library function to construct a Cartesian or
    non-Cartesian SENSE forward model that is persistent (constructed once and
    reused for multiple apply calls).

    Encoding models
    ---------------
    Cartesian (``traj=None``):
        ``E = P ∘ FFT ∘ coil-expansion``
        where ``P`` is the undersampling mask (omitted when ``pattern=None``).

    Non-Cartesian (``traj`` provided):
        ``E = NUFFT ∘ coil-expansion``

    With subspace (``basis`` provided):
        Subspace projection is prepended: ``E = … ∘ Φ``
        where ``Φ`` maps subspace coefficients to the temporal/echo dimension.

    Parameters
    ----------
    maps : torch.Tensor
        Sensitivity maps in bartorch C-order ``(nmaps, nc, [nz,] ny, nx)``,
        complex64.  Typically the output of :func:`bartorch.tools.ecalib`.
    ksp_shape : tuple[int, ...], optional
        K-space output shape in C-order.

        * Cartesian: ``(nc, [nz,] ny, nx)`` — inferred from ``maps`` if omitted.
        * Non-Cartesian: e.g. ``(nc, nspokes, nsamples)`` — must be provided
          when ``traj`` is given.
    pattern : torch.Tensor, optional
        Undersampling mask (Cartesian), same spatial layout as k-space.
        ``None`` (default) means no masking (full k-space).
    traj : torch.Tensor, optional
        Non-Cartesian trajectory in C-order ``(..., 3)`` where the last
        dimension holds (kx, ky, kz) normalised coordinates.
        ``None`` (default) selects the Cartesian model.
    basis : torch.Tensor, optional
        Subspace basis in C-order ``(ncoeff, nt)`` where ``ncoeff`` is the
        number of subspace coefficients and ``nt`` is the number of
        echo/temporal frames.  ``None`` (default) disables subspace projection.
    gpu : bool, optional
        If ``True``, allocate the operator on GPU (requires a CUDA build).
        Default ``False``.

    Returns
    -------
    BartLinop
        Persistent BART encoding operator.  Call :meth:`~BartLinop.forward`,
        :meth:`~BartLinop.adjoint`, :meth:`~BartLinop.normal`, or
        :meth:`~BartLinop.solve` to use it.

    Examples
    --------
    Cartesian SENSE (sensitivity maps from ecalib):

    >>> import bartorch.tools as bt
    >>> import bartorch.lib as bl
    >>> ksp  = bt.phantom([128, 128], kspace=True, ncoils=8)
    >>> sens = bt.ecalib(ksp, calib_size=24, maps=1)
    >>> E    = bl.encoding_op(sens)
    >>> img  = bt.phantom([128, 128])
    >>> ksp2 = E.forward(img)   # A x
    >>> img2 = E.adjoint(ksp2)  # A^H y

    Non-Cartesian SENSE:

    >>> traj = bt.traj(r=True, x=64, y=64)
    >>> E    = bl.encoding_op(sens, ksp_shape=(8, 64, 64), traj=traj)
    """
    from bartorch._bartorch_ext import create_encoding_op  # noqa: PLC0415

    handle = create_encoding_op(
        maps.to(torch.complex64),
        list(ksp_shape) if ksp_shape is not None else [],
        pattern,
        traj,
        basis,
        1 if gpu else 0,
    )
    return BartLinop(handle)
