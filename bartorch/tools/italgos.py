"""Iterative algorithm tools — bartorch.tools.italgos.

Wraps BART's iterative solvers and proximal algorithms from ``src/iter/``.
All functions accept :class:`~bartorch.ops.linops.BartLinop` operators and
plain ``torch.Tensor`` data vectors.  Input tensors are normalised
automatically by the :func:`~bartorch.core.tensor.bart_op` decorator.

These are stubs pending the C++ extension (Phase 5).

Module exports
--------------
conjgrad, ist, fista, irgnm, chambolle_pock
"""

from __future__ import annotations

import torch

from bartorch.core.tensor import bart_op
from bartorch.ops.linops import BartLinop

__all__ = ["conjgrad", "ist", "fista", "irgnm", "chambolle_pock"]


@bart_op
def conjgrad(
    op: BartLinop,
    b: torch.Tensor,
    *,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Conjugate gradient (CG) solver for the normal equations ``A^H A x = A^H b``.

    Wraps BART's ``iter/italgos.c`` CG implementation.

    Parameters
    ----------
    op : BartLinop
        The linear operator *A*.
    b : torch.Tensor
        Right-hand side vector.
    maxiter : int, optional
        Maximum number of CG iterations.  Default ``100``.
    tol : float, optional
        Relative residual tolerance.  Default ``1e-6``.

    Returns
    -------
    torch.Tensor
        Solution *x*.

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 5) is built.
    """
    raise NotImplementedError("italgos.conjgrad() requires the C++ extension.")


@bart_op
def ist(
    op: BartLinop,
    b: torch.Tensor,
    proxg,
    *,
    maxiter: int = 50,
    step: float = 0.95,
) -> torch.Tensor:
    """Iterative Soft-Thresholding (IST) algorithm.

    Solves ``min_x  ½‖A x − b‖² + g(x)`` via the iterative step
    ``x ← prox_{step·g}(x − step · A^H(A x − b))``.

    Parameters
    ----------
    op : BartLinop
        Forward operator *A*.
    b : torch.Tensor
        Observed data.
    proxg : callable
        Proximal operator for the regulariser *g*.  Signature:
        ``proxg(x: torch.Tensor, step: float) -> torch.Tensor``.
    maxiter : int, optional
        Maximum iterations.  Default ``50``.
    step : float, optional
        Step size (must satisfy ``step ≤ 1 / ‖A‖²``).  Default ``0.95``.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 5) is built.
    """
    raise NotImplementedError("italgos.ist() requires the C++ extension.")


@bart_op
def fista(
    op: BartLinop,
    b: torch.Tensor,
    proxg,
    *,
    maxiter: int = 50,
    step: float = 0.95,
) -> torch.Tensor:
    """Fast Iterative Soft-Thresholding Algorithm (FISTA).

    Accelerated variant of :func:`ist` using Nesterov momentum.

    Parameters
    ----------
    op : BartLinop
        Forward operator *A*.
    b : torch.Tensor
        Observed data.
    proxg : callable
        Proximal operator for the regulariser.
    maxiter : int, optional
        Maximum iterations.  Default ``50``.
    step : float, optional
        Step size.  Default ``0.95``.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 5) is built.
    """
    raise NotImplementedError("italgos.fista() requires the C++ extension.")


@bart_op
def irgnm(
    op: BartLinop,
    b: torch.Tensor,
    *,
    maxiter: int = 10,
) -> torch.Tensor:
    """Iteratively Regularised Gauss-Newton Method (IRGNM).

    For non-linear inverse problems of the form ``F(x) = b``.  Each outer
    iteration linearises *F* around the current estimate and solves the
    resulting least-squares subproblem with CG.

    Parameters
    ----------
    op : BartLinop
        Non-linear forward operator *F* (requires ``forward`` and ``jacobian``
        capabilities from the C++ layer).
    b : torch.Tensor
        Observed data.
    maxiter : int, optional
        Number of outer Gauss-Newton iterations.  Default ``10``.

    Returns
    -------
    torch.Tensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 5) is built.
    """
    raise NotImplementedError("italgos.irgnm() requires the C++ extension.")


@bart_op
def chambolle_pock(
    op: BartLinop,
    prox_f,
    prox_g,
    *,
    maxiter: int = 100,
    sigma: float = 1.0,
    tau: float = 1.0,
) -> torch.Tensor:
    """Chambolle-Pock primal-dual splitting algorithm.

    Solves ``min_x  f(A x) + g(x)`` via the primal-dual iterations:

    .. code-block:: text

        y ← prox_{σ f*}(y + σ A x̄)
        x ← prox_{τ g}(x − τ A^H y)
        x̄ ← 2x − x_prev

    Parameters
    ----------
    op : BartLinop
        Linear operator *A*.
    prox_f : callable
        Proximal operator for *f* (in the dual variable).  Signature:
        ``prox_f(y: torch.Tensor, sigma: float) -> torch.Tensor``.
    prox_g : callable
        Proximal operator for *g* (in the primal variable).  Signature:
        ``prox_g(x: torch.Tensor, tau: float) -> torch.Tensor``.
    maxiter : int, optional
        Number of primal-dual iterations.  Default ``100``.
    sigma : float, optional
        Dual step size.  Default ``1.0``.
    tau : float, optional
        Primal step size.  Default ``1.0``.

    Returns
    -------
    torch.Tensor
        Primal solution *x*.

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 5) is built.
    """
    raise NotImplementedError("italgos.chambolle_pock() requires the C++ extension.")
