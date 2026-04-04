"""Iterative algorithms — bartorch.ops.italgos.

Wraps BART's iterative solvers and proximal algorithms from
``src/iter/``.  All functions accept :class:`~bartorch.ops.linops.BartLinop`
operators and :class:`~bartorch.core.tensor.BartTensor` data vectors.

These are currently stubs pending the C++ extension (Phase 2/3).

Module exports
--------------
conjgrad, ist, fista, irgnm, chambolle_pock
"""

from __future__ import annotations

from bartorch.core.tensor import BartTensor
from bartorch.ops.linops import BartLinop

__all__ = ["conjgrad", "ist", "fista", "irgnm", "chambolle_pock"]


def conjgrad(
    op: BartLinop,
    b: BartTensor,
    *,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> BartTensor:
    """Conjugate gradient (CG) solver for the normal equations ``A^H A x = A^H b``.

    Wraps BART's ``iter/italgos.c`` CG implementation.

    Parameters
    ----------
    op : BartLinop
        The linear operator *A*.
    b : BartTensor
        Right-hand side vector.
    maxiter : int, optional
        Maximum number of CG iterations.  Default ``100``.
    tol : float, optional
        Relative residual tolerance.  Default ``1e-6``.

    Returns
    -------
    BartTensor
        Solution *x*.

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("italgos.conjgrad() requires the C++ extension.")


def ist(
    op: BartLinop,
    b: BartTensor,
    proxg,
    *,
    maxiter: int = 50,
    step: float = 0.95,
) -> BartTensor:
    """Iterative Soft-Thresholding (IST) algorithm.

    Solves ``min_x  ½‖A x − b‖² + g(x)`` via the iterative step
    ``x ← prox_{step·g}(x − step · A^H(A x − b))``.

    Parameters
    ----------
    op : BartLinop
        Forward operator *A*.
    b : BartTensor
        Observed data.
    proxg : callable
        Proximal operator for the regulariser *g*.  Signature:
        ``proxg(x: BartTensor, step: float) -> BartTensor``.
    maxiter : int, optional
        Maximum iterations.  Default ``50``.
    step : float, optional
        Step size (must satisfy ``step ≤ 1 / ‖A‖²``).  Default ``0.95``.

    Returns
    -------
    BartTensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 3) is built.
    """
    raise NotImplementedError("italgos.ist() requires the C++ extension.")


def fista(
    op: BartLinop,
    b: BartTensor,
    proxg,
    *,
    maxiter: int = 50,
    step: float = 0.95,
) -> BartTensor:
    """Fast Iterative Soft-Thresholding Algorithm (FISTA).

    Accelerated variant of :func:`ist` using Nesterov momentum.

    Parameters
    ----------
    op : BartLinop
        Forward operator *A*.
    b : BartTensor
        Observed data.
    proxg : callable
        Proximal operator for the regulariser.
    maxiter : int, optional
        Maximum iterations.  Default ``50``.
    step : float, optional
        Step size.  Default ``0.95``.

    Returns
    -------
    BartTensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 3) is built.
    """
    raise NotImplementedError("italgos.fista() requires the C++ extension.")


def irgnm(
    op: BartLinop,
    b: BartTensor,
    *,
    maxiter: int = 10,
) -> BartTensor:
    """Iteratively Regularised Gauss-Newton Method (IRGNM).

    For non-linear inverse problems of the form ``F(x) = b``.  Each outer
    iteration linearises *F* around the current estimate and solves the
    resulting least-squares subproblem with CG.

    Parameters
    ----------
    op : BartLinop
        Non-linear forward operator *F* (requires ``forward`` and ``jacobian``
        capabilities from the C++ layer).
    b : BartTensor
        Observed data.
    maxiter : int, optional
        Number of outer Gauss-Newton iterations.  Default ``10``.

    Returns
    -------
    BartTensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 3) is built.
    """
    raise NotImplementedError("italgos.irgnm() requires the C++ extension.")


def chambolle_pock(
    op: BartLinop,
    prox_f,
    prox_g,
    *,
    maxiter: int = 100,
    sigma: float = 1.0,
    tau: float = 1.0,
) -> BartTensor:
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
        ``prox_f(y: BartTensor, sigma: float) -> BartTensor``.
    prox_g : callable
        Proximal operator for *g* (in the primal variable).  Signature:
        ``prox_g(x: BartTensor, tau: float) -> BartTensor``.
    maxiter : int, optional
        Number of primal-dual iterations.  Default ``100``.
    sigma : float, optional
        Dual step size.  Default ``1.0``.
    tau : float, optional
        Primal step size.  Default ``1.0``.

    Returns
    -------
    BartTensor
        Primal solution *x*.

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 3) is built.
    """
    raise NotImplementedError("italgos.chambolle_pock() requires the C++ extension.")
