"""Linear operators — bartorch.ops.linops.

Provides an opaque handle type (:class:`BartLinop`) and factory functions for
constructing and composing BART linear operators (``linop_s*``).

The operators are backed by BART's ``linops/linop.c`` framework and will be
wired to the C++ extension in Phase 2.  All constructor and composition
functions currently raise :exc:`NotImplementedError`; the application
functions (:func:`forward`, :func:`adjoint`, :func:`normal`,
:func:`pseudo_inv`) will call ``bart_command()`` via the hot path once the
extension is built.

Module exports
--------------
BartLinop, identity, diag, fft_linop,
chain, plus, stack,
forward, adjoint, normal, pseudo_inv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bartorch.core.tensor import BartTensor

__all__ = [
    "BartLinop",
    "identity",
    "diag",
    "fft_linop",
    "chain",
    "plus",
    "stack",
    "forward",
    "adjoint",
    "normal",
    "pseudo_inv",
]


@dataclass
class BartLinop:
    """Opaque wrapper around a BART ``linop_s`` pointer.

    Attributes
    ----------
    _ptr : int or None
        Raw C pointer address (Python ``int``) once the C++ extension is
        loaded; ``None`` at stub stage.
    src_dims : list[int] or None
        Shape of the operator's domain (source / input space).
    dst_dims : list[int] or None
        Shape of the operator's codomain (destination / output space).

    Notes
    -----
    ``BartLinop`` instances should be created only through the factory
    functions in this module (:func:`identity`, :func:`diag`,
    :func:`fft_linop`, …).  Do not instantiate directly.
    """

    _ptr: Any = None
    src_dims: list[int] | None = None
    dst_dims: list[int] | None = None

    def __repr__(self) -> str:
        return f"BartLinop(src={self.src_dims}, dst={self.dst_dims}, ptr={self._ptr})"


# ---------------------------------------------------------------------------
# Linop constructors
# ---------------------------------------------------------------------------


def identity(dims: list[int]) -> BartLinop:
    """Create an identity linear operator.

    Parameters
    ----------
    dims : list[int]
        Shape of the domain (and codomain).

    Returns
    -------
    BartLinop
        Identity operator for arrays of shape *dims*.

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.identity() requires the C++ extension.")


def diag(dims: list[int], *, diag_dims: list[int], flags: int) -> BartLinop:
    """Create a diagonal (pointwise multiplication) linear operator.

    Wraps ``linop_cdiag_create()`` from ``linops/someops.h``.

    Parameters
    ----------
    dims : list[int]
        Full shape of the operator's domain.
    diag_dims : list[int]
        Shape of the diagonal array (must be broadcastable to *dims*).
    flags : int
        Bitmask indicating which dimensions the diagonal varies over.

    Returns
    -------
    BartLinop

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.diag() requires the C++ extension.")


def fft_linop(dims: list[int], flags: int) -> BartLinop:
    """Create an FFT linear operator.

    Wraps ``linop_fft_create()`` from ``linops/someops.h``.

    Parameters
    ----------
    dims : list[int]
        Shape of the domain.
    flags : int
        Bitmask selecting the dimensions to transform.

    Returns
    -------
    BartLinop

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.fft_linop() requires the C++ extension.")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def chain(op1: BartLinop, op2: BartLinop) -> BartLinop:
    """Compose two operators: ``op2 ∘ op1`` (apply *op1* first, then *op2*).

    Wraps ``linop_chain()`` from ``linops/linop.h``.

    Parameters
    ----------
    op1 : BartLinop
        First operator to apply.
    op2 : BartLinop
        Second operator to apply.

    Returns
    -------
    BartLinop
        Composed operator whose domain equals *op1*'s domain and codomain
        equals *op2*'s codomain.

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.chain() requires the C++ extension.")


def plus(op1: BartLinop, op2: BartLinop) -> BartLinop:
    """Sum two compatible operators: ``op1 + op2``.

    Wraps ``linop_plus()`` from ``linops/linop.h``.

    Parameters
    ----------
    op1 : BartLinop
    op2 : BartLinop
        Must have the same domain and codomain as *op1*.

    Returns
    -------
    BartLinop

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.plus() requires the C++ extension.")


def stack(op1: BartLinop, op2: BartLinop) -> BartLinop:
    """Stack two operators along a new output dimension.

    Wraps ``linop_stack()`` from ``linops/linop.h``.

    Parameters
    ----------
    op1 : BartLinop
    op2 : BartLinop

    Returns
    -------
    BartLinop

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.stack() requires the C++ extension.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def forward(op: BartLinop, dest_dims: list[int], src: BartTensor) -> BartTensor:
    """Apply the forward linear operator ``A x``.

    Parameters
    ----------
    op : BartLinop
        Operator to apply.
    dest_dims : list[int]
        Expected shape of the output (codomain).
    src : BartTensor
        Input array in the domain of *op*.

    Returns
    -------
    BartTensor
        Result ``A @ src``.

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.forward() requires the C++ extension.")


def adjoint(op: BartLinop, dest_dims: list[int], src: BartTensor) -> BartTensor:
    """Apply the adjoint operator ``A^H x``.

    Parameters
    ----------
    op : BartLinop
        Operator whose adjoint to apply.
    dest_dims : list[int]
        Expected shape of the output (domain of *op*).
    src : BartTensor
        Input array in the codomain of *op*.

    Returns
    -------
    BartTensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.adjoint() requires the C++ extension.")


def normal(op: BartLinop, dest_dims: list[int], src: BartTensor) -> BartTensor:
    """Apply the normal operator ``A^H A x``.

    Parameters
    ----------
    op : BartLinop
    dest_dims : list[int]
    src : BartTensor

    Returns
    -------
    BartTensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.normal() requires the C++ extension.")


def pseudo_inv(
    op: BartLinop,
    lambda_: float,
    dest_dims: list[int],
    src: BartTensor,
) -> BartTensor:
    """Apply the regularised pseudo-inverse ``(A^H A + λ I)^{-1} A^H x``.

    Parameters
    ----------
    op : BartLinop
    lambda_ : float
        Tikhonov regularisation strength λ.
    dest_dims : list[int]
    src : BartTensor

    Returns
    -------
    BartTensor

    Raises
    ------
    NotImplementedError
        Until the C++ extension (Phase 2) is built.
    """
    raise NotImplementedError("linop.pseudo_inv() requires the C++ extension.")
