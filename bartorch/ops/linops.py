"""Linear operators — bartorch.ops.linops.

Opaque BART linop handles are represented as ``BartLinop`` objects.
They are passed between op functions without materialising intermediate
tensors; apply them with forward(), adjoint(), normal(), or pseudo_inv().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bartorch.core.tensor import BartTensor


@dataclass
class BartLinop:
    """Opaque wrapper around a BART linop pointer.

    The underlying C pointer is stored as a Python ``int`` (address) once the
    C++ extension is available.  At stub stage this is ``None``.
    """

    _ptr: Any = None
    src_dims: list[int] | None = None
    dst_dims: list[int] | None = None

    def __repr__(self):
        return f"BartLinop(src={self.src_dims}, dst={self.dst_dims}, ptr={self._ptr})"


# ---------------------------------------------------------------------------
# Linop constructors
# ---------------------------------------------------------------------------


def identity(dims: list[int]) -> BartLinop:
    """Identity operator."""
    raise NotImplementedError("linop.identity() requires the C++ extension.")


def diag(dims: list[int], *, diag_dims: list[int], flags: int) -> BartLinop:
    """Diagonal operator."""
    raise NotImplementedError("linop.diag() requires the C++ extension.")


def fft_linop(dims: list[int], flags: int) -> BartLinop:
    """FFT linear operator."""
    raise NotImplementedError("linop.fft_linop() requires the C++ extension.")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def chain(op1: BartLinop, op2: BartLinop) -> BartLinop:
    """op2 ∘ op1 (apply op1 first, then op2)."""
    raise NotImplementedError


def plus(op1: BartLinop, op2: BartLinop) -> BartLinop:
    """op1 + op2."""
    raise NotImplementedError


def stack(op1: BartLinop, op2: BartLinop) -> BartLinop:
    """Stack two operators along a new dimension."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def forward(op: BartLinop, dest_dims: list[int], src: BartTensor) -> BartTensor:
    """Apply the forward operator."""
    raise NotImplementedError


def adjoint(op: BartLinop, dest_dims: list[int], src: BartTensor) -> BartTensor:
    """Apply the adjoint operator."""
    raise NotImplementedError


def normal(op: BartLinop, dest_dims: list[int], src: BartTensor) -> BartTensor:
    """Apply the normal (A^H A) operator."""
    raise NotImplementedError


def pseudo_inv(
    op: BartLinop,
    lambda_: float,
    dest_dims: list[int],
    src: BartTensor,
) -> BartTensor:
    """Apply the pseudo-inverse (A^H A + λI)^{-1} A^H."""
    raise NotImplementedError
