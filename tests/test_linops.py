"""Tests for bartorch.ops.linops (BartLinop operator algebra)."""

import pytest
import torch

from bartorch.ops.linops import BartLinop


def make_op(ishape, oshape, kind="base"):
    """Create a stub BartLinop with no C++ handle."""
    return BartLinop(ishape=ishape, oshape=oshape, _kind=kind)


class TestBartLinopBasic:
    def test_ishape_oshape_stored_as_tuple(self):
        A = make_op([8, 256, 256], [8, 1, 256, 256])
        assert A.ishape == (8, 256, 256)
        assert A.oshape == (8, 1, 256, 256)

    def test_repr_contains_shapes(self):
        A = make_op([4, 8], [4, 4])
        r = repr(A)
        assert "ishape" in r
        assert "oshape" in r
        assert "4" in r
        assert "8" in r

    def test_kind_default_is_base(self):
        A = make_op([4], [4])
        assert A._kind == "base"


class TestAdjoint:
    def test_H_swaps_shapes(self):
        A = make_op([8, 256, 256], [8, 1, 256, 256])
        AH = A.H
        assert AH.ishape == A.oshape
        assert AH.oshape == A.ishape

    def test_H_kind(self):
        A = make_op([4], [8])
        assert A.H._kind == "adjoint"

    def test_double_adjoint_shapes(self):
        A = make_op([8, 256, 256], [8, 1, 256, 256])
        AHH = A.H.H
        assert AHH.ishape == A.ishape
        assert AHH.oshape == A.oshape

    def test_H_preserves_original(self):
        A = make_op([4], [8])
        _ = A.H
        assert A.ishape == (4,)  # original unchanged


class TestNormalOperator:
    def test_N_maps_domain_to_itself(self):
        A = make_op([8, 256, 256], [8, 1, 256, 256])
        AN = A.N
        assert AN.ishape == A.ishape
        assert AN.oshape == A.ishape

    def test_N_kind(self):
        A = make_op([4], [8])
        assert A.N._kind == "normal"


class TestComposition:
    def test_matmul_linop_shapes(self):
        # C = A @ B  →  ishape = B.ishape, oshape = A.oshape
        A = make_op([4, 4], [8, 4])   # maps (4,4) → (8,4)
        B = make_op([4, 8], [4, 4])   # maps (4,8) → (4,4)
        C = A @ B
        assert isinstance(C, BartLinop)
        assert C.ishape == B.ishape
        assert C.oshape == A.oshape

    def test_matmul_kind_is_composition(self):
        A = make_op([4], [8])
        B = make_op([2], [4])
        C = A @ B
        assert C._kind == "composition"

    def test_matmul_args_stored(self):
        A = make_op([4], [8])
        B = make_op([2], [4])
        C = A @ B
        assert C._args == (A, B)


class TestSum:
    def test_add_returns_bartlinop(self):
        A = make_op([4, 8], [4, 8])
        B = make_op([4, 8], [4, 8])
        D = A + B
        assert isinstance(D, BartLinop)

    def test_add_preserves_shapes(self):
        A = make_op([4, 8], [4, 8])
        B = make_op([4, 8], [4, 8])
        D = A + B
        assert D.ishape == A.ishape
        assert D.oshape == A.oshape

    def test_add_kind(self):
        A = make_op([4, 8], [4, 8])
        B = make_op([4, 8], [4, 8])
        assert (A + B)._kind == "sum"

    def test_add_non_linop_returns_not_implemented(self):
        A = make_op([4], [4])
        result = A.__add__(42)
        assert result is NotImplemented


class TestScalarMultiplication:
    def test_mul_returns_bartlinop(self):
        A = make_op([4, 8], [4, 8])
        E = A * 2.0
        assert isinstance(E, BartLinop)

    def test_rmul_commutes(self):
        A = make_op([4, 8], [4, 8])
        E1 = A * 3.0
        E2 = 3.0 * A
        assert E1._kind == E2._kind == "scaled"

    def test_mul_preserves_shapes(self):
        A = make_op([4, 8], [4, 8])
        E = 5 * A
        assert E.ishape == A.ishape
        assert E.oshape == A.oshape

    def test_scalar_stored_as_float(self):
        A = make_op([4], [4])
        E = A * 3
        assert isinstance(E._args[1], float)

    def test_mul_non_scalar_returns_not_implemented(self):
        A = make_op([4], [4])
        result = A.__mul__("bad")
        assert result is NotImplemented


class TestApplication:
    def test_call_raises_not_implemented(self):
        A = make_op([4, 8], [4, 8])
        x = torch.zeros(4, 8, dtype=torch.complex64)
        with pytest.raises(NotImplementedError, match=r"C\+\+ extension"):
            A(x)

    def test_matmul_tensor_raises_not_implemented(self):
        A = make_op([4, 8], [4, 8])
        x = torch.zeros(4, 8, dtype=torch.complex64)
        with pytest.raises(NotImplementedError):
            A @ x

    def test_rmatmul_calls_adjoint(self):
        A = make_op([4, 8], [4, 8])
        x = torch.zeros(4, 8, dtype=torch.complex64)
        with pytest.raises(NotImplementedError):
            x @ A  # calls A.H(x)

    def test_matmul_unsupported_type_returns_not_implemented(self):
        A = make_op([4], [4])
        result = A.__matmul__("not_a_tensor")
        assert result is NotImplemented


class TestChainedAlgebra:
    """Verify that complex algebra expressions form the correct DAG."""

    def test_AHA_chain(self):
        # (A.H @ A) should have ishape = oshape = A.ishape
        A = make_op([4, 8], [8, 8])
        AHA = A.H @ A
        assert AHA.ishape == A.ishape
        assert AHA.oshape == A.ishape

    def test_regularised_normal(self):
        # A.N + lam * I  (conceptual — shapes must match)
        A = make_op([8, 256], [8, 256])
        I = make_op([8, 256], [8, 256])
        lam = 0.01
        reg = A.N + lam * I
        assert reg.ishape == A.ishape
        assert reg.oshape == A.ishape

    def test_three_way_chain(self):
        # C = A @ B @ D
        A = make_op([4], [8])
        B = make_op([2], [4])
        D = make_op([1], [2])
        C = A @ B @ D
        assert C.ishape == D.ishape
        assert C.oshape == A.oshape
