"""Iterative algorithms — bartorch.ops.italgos."""

from __future__ import annotations
from bartorch.core.tensor import BartTensor
from bartorch.ops.linops import BartLinop


def conjgrad(
    op: BartLinop,
    b: BartTensor,
    *,
    maxiter: int = 100,
    tol: float = 1e-6,
) -> BartTensor:
    """Conjugate gradient solver for ``op x = b``."""
    raise NotImplementedError("italgos.conjgrad() requires the C++ extension.")


def ist(
    op: BartLinop,
    b: BartTensor,
    proxg,
    *,
    maxiter: int = 50,
    step: float = 0.95,
) -> BartTensor:
    """Iterative soft-thresholding (IST)."""
    raise NotImplementedError


def fista(
    op: BartLinop,
    b: BartTensor,
    proxg,
    *,
    maxiter: int = 50,
    step: float = 0.95,
) -> BartTensor:
    """Fast IST (FISTA / ISTA with momentum)."""
    raise NotImplementedError


def irgnm(
    op: BartLinop,
    b: BartTensor,
    *,
    maxiter: int = 10,
) -> BartTensor:
    """Iteratively regularised Gauss-Newton method."""
    raise NotImplementedError


def chambolle_pock(
    op: BartLinop,
    prox_f,
    prox_g,
    *,
    maxiter: int = 100,
    sigma: float = 1.0,
    tau: float = 1.0,
) -> BartTensor:
    """Chambolle-Pock primal-dual algorithm."""
    raise NotImplementedError
