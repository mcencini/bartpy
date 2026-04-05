"""Tests for bartorch.ops.linops (BartLinop operator algebra)."""

from __future__ import annotations

import pytest
import torch

from bartorch.ops.linops import BartLinop

__all__: list[str] = []


def make_op(ishape, oshape, kind="base"):
    """Create a stub BartLinop with no C++ handle."""
    return BartLinop(ishape=ishape, oshape=oshape, _kind=kind)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


def test_ishape_oshape_stored_as_tuple():
    A = make_op([8, 256, 256], [8, 1, 256, 256])
    assert A.ishape == (8, 256, 256)
    assert A.oshape == (8, 1, 256, 256)


def test_repr_contains_shapes():
    A = make_op([4, 8], [4, 4])
    r = repr(A)
    assert "ishape" in r
    assert "oshape" in r
    assert "4" in r
    assert "8" in r


def test_kind_default_is_base():
    A = make_op([4], [4])
    assert A._kind == "base"


# ---------------------------------------------------------------------------
# Adjoint
# ---------------------------------------------------------------------------


def test_H_swaps_shapes():
    A = make_op([8, 256, 256], [8, 1, 256, 256])
    AH = A.H
    assert AH.ishape == A.oshape
    assert AH.oshape == A.ishape


def test_H_kind():
    A = make_op([4], [8])
    assert A.H._kind == "adjoint"


def test_double_adjoint_shapes():
    A = make_op([8, 256, 256], [8, 1, 256, 256])
    AHH = A.H.H
    assert AHH.ishape == A.ishape
    assert AHH.oshape == A.oshape


def test_H_preserves_original():
    A = make_op([4], [8])
    _ = A.H
    assert A.ishape == (4,)


# ---------------------------------------------------------------------------
# Normal operator
# ---------------------------------------------------------------------------


def test_N_maps_domain_to_itself():
    A = make_op([8, 256, 256], [8, 1, 256, 256])
    AN = A.N
    assert AN.ishape == A.ishape
    assert AN.oshape == A.ishape


def test_N_kind():
    A = make_op([4], [8])
    assert A.N._kind == "normal"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_matmul_linop_shapes():
    A = make_op([4, 4], [8, 4])
    B = make_op([4, 8], [4, 4])
    C = A @ B
    assert isinstance(C, BartLinop)
    assert C.ishape == B.ishape
    assert C.oshape == A.oshape


def test_matmul_kind_is_composition():
    A = make_op([4], [8])
    B = make_op([2], [4])
    C = A @ B
    assert C._kind == "composition"


def test_matmul_args_stored():
    A = make_op([4], [8])
    B = make_op([2], [4])
    C = A @ B
    assert C._args == (A, B)


# ---------------------------------------------------------------------------
# Sum
# ---------------------------------------------------------------------------


def test_add_returns_bartlinop():
    A = make_op([4, 8], [4, 8])
    B = make_op([4, 8], [4, 8])
    assert isinstance(A + B, BartLinop)


def test_add_preserves_shapes():
    A = make_op([4, 8], [4, 8])
    B = make_op([4, 8], [4, 8])
    D = A + B
    assert D.ishape == A.ishape
    assert D.oshape == A.oshape


def test_add_kind():
    A = make_op([4, 8], [4, 8])
    B = make_op([4, 8], [4, 8])
    assert (A + B)._kind == "sum"


def test_add_non_linop_returns_not_implemented():
    A = make_op([4], [4])
    assert A.__add__(42) is NotImplemented


# ---------------------------------------------------------------------------
# Scalar multiplication
# ---------------------------------------------------------------------------


def test_mul_returns_bartlinop():
    A = make_op([4, 8], [4, 8])
    assert isinstance(A * 2.0, BartLinop)


def test_rmul_commutes():
    A = make_op([4, 8], [4, 8])
    E1 = A * 3.0
    E2 = 3.0 * A
    assert E1._kind == E2._kind == "scaled"


def test_mul_preserves_shapes():
    A = make_op([4, 8], [4, 8])
    E = 5 * A
    assert E.ishape == A.ishape
    assert E.oshape == A.oshape


def test_scalar_stored_as_float():
    A = make_op([4], [4])
    E = A * 3
    assert isinstance(E._args[1], float)


def test_mul_non_scalar_returns_not_implemented():
    A = make_op([4], [4])
    assert A.__mul__("bad") is NotImplemented


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def test_call_raises_not_implemented():
    A = make_op([4, 8], [4, 8])
    x = torch.zeros(4, 8, dtype=torch.complex64)
    with pytest.raises(NotImplementedError, match=r"C\+\+ extension"):
        A(x)


def test_matmul_tensor_raises_not_implemented():
    A = make_op([4, 8], [4, 8])
    x = torch.zeros(4, 8, dtype=torch.complex64)
    with pytest.raises(NotImplementedError):
        A @ x


def test_rmatmul_calls_adjoint():
    A = make_op([4, 8], [4, 8])
    x = torch.zeros(4, 8, dtype=torch.complex64)
    with pytest.raises(NotImplementedError):
        x @ A


def test_matmul_unsupported_type_returns_not_implemented():
    A = make_op([4], [4])
    assert A.__matmul__("not_a_tensor") is NotImplemented


# ---------------------------------------------------------------------------
# Chained algebra
# ---------------------------------------------------------------------------


def test_AHA_chain():
    A = make_op([4, 8], [8, 8])
    AHA = A.H @ A
    assert AHA.ishape == A.ishape
    assert AHA.oshape == A.ishape


def test_regularised_normal():
    A = make_op([8, 256], [8, 256])
    I = make_op([8, 256], [8, 256])
    reg = A.N + 0.01 * I
    assert reg.ishape == A.ishape
    assert reg.oshape == A.ishape


def test_three_way_chain():
    A = make_op([4], [8])
    B = make_op([2], [4])
    D = make_op([1], [2])
    C = A @ B @ D
    assert C.ishape == D.ishape
    assert C.oshape == A.oshape
