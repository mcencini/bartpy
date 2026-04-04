"""Tests for bartorch.core.tensor (private utilities and the bart_op decorator)."""

import numpy as np
import pytest
import torch

from bartorch.core.tensor import (
    _as_complex64,
    _fortran_strides,
    _normalise_input,
    _reverse_dims,
    bart_op,
)


# ---------------------------------------------------------------------------
# _reverse_dims
# ---------------------------------------------------------------------------


class TestReverseDims:
    def test_1d(self):
        assert _reverse_dims([5]) == [5]

    def test_2d(self):
        assert _reverse_dims([3, 4]) == [4, 3]

    def test_3d(self):
        assert _reverse_dims([2, 3, 4]) == [4, 3, 2]

    def test_4d(self):
        assert _reverse_dims([8, 1, 256, 256]) == [256, 256, 1, 8]

    def test_accepts_tuple(self):
        assert _reverse_dims((2, 3)) == [3, 2]

    def test_roundtrip(self):
        dims = [8, 4, 256, 256]
        assert _reverse_dims(_reverse_dims(dims)) == dims


# ---------------------------------------------------------------------------
# _as_complex64
# ---------------------------------------------------------------------------


class TestAsComplex64:
    def test_already_complex64_is_same_object(self):
        t = torch.zeros(4, 8, dtype=torch.complex64)
        result = _as_complex64(t)
        assert result is t  # zero-copy

    def test_float32_gets_cast(self):
        t = torch.ones(4, 8, dtype=torch.float32)
        result = _as_complex64(t)
        assert result.dtype == torch.complex64

    def test_complex128_gets_cast(self):
        t = torch.ones(4, 8, dtype=torch.complex128)
        result = _as_complex64(t)
        assert result.dtype == torch.complex64

    def test_shape_preserved(self):
        t = torch.zeros(3, 5, 7, dtype=torch.float32)
        result = _as_complex64(t)
        assert result.shape == t.shape

    def test_values_preserved(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _as_complex64(t)
        assert result[0].real == pytest.approx(1.0)
        assert result[1].real == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _normalise_input
# ---------------------------------------------------------------------------


class TestNormaliseInput:
    def test_tensor_complex64_zero_copy(self):
        t = torch.zeros(4, 8, dtype=torch.complex64)
        assert _normalise_input(t) is t

    def test_tensor_float32_cast(self):
        t = torch.ones(4, 8, dtype=torch.float32)
        assert _normalise_input(t).dtype == torch.complex64

    def test_numpy_array_converted(self):
        arr = np.ones((4, 8), dtype=np.complex64)
        result = _normalise_input(arr)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.complex64

    def test_numpy_float_cast(self):
        arr = np.ones((4, 8), dtype=np.float32)
        result = _normalise_input(arr)
        assert result.dtype == torch.complex64

    def test_int_passthrough(self):
        assert _normalise_input(42) == 42

    def test_string_passthrough(self):
        assert _normalise_input("fft") == "fft"

    def test_bool_passthrough(self):
        assert _normalise_input(True) is True

    def test_none_passthrough(self):
        assert _normalise_input(None) is None


# ---------------------------------------------------------------------------
# _fortran_strides
# ---------------------------------------------------------------------------


class TestFortranStrides:
    def test_1d(self):
        assert _fortran_strides([5]) == [1]

    def test_2d(self):
        assert _fortran_strides([3, 4]) == [1, 3]

    def test_3d(self):
        assert _fortran_strides([2, 3, 4]) == [1, 2, 6]


# ---------------------------------------------------------------------------
# bart_op decorator
# ---------------------------------------------------------------------------


class TestBartOpDecoratorBasic:
    def test_complex64_passthrough_is_zero_copy(self):
        @bart_op
        def my_op(x):
            return x

        t = torch.zeros(4, 8, dtype=torch.complex64)
        assert my_op(t) is t

    def test_float32_cast_to_complex64(self):
        @bart_op
        def my_op(x):
            return x

        t = torch.ones(4, 8, dtype=torch.float32)
        result = my_op(t)
        assert result.dtype == torch.complex64

    def test_numpy_input_converted(self):
        @bart_op
        def my_op(x):
            return x

        arr = np.ones((4, 8), dtype=np.float32)
        result = my_op(arr)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.complex64

    def test_non_tensor_kwargs_pass_through(self):
        @bart_op
        def my_op(x, flags, name="default"):
            return x, flags, name

        t = torch.zeros(4, 8, dtype=torch.complex64)
        _, flags_out, name_out = my_op(t, 3, name="fft")
        assert flags_out == 3
        assert name_out == "fft"

    def test_tensor_kwarg_normalised(self):
        @bart_op
        def my_op(x, y=None):
            return x, y

        t = torch.zeros(4, 8, dtype=torch.complex64)
        t_float = torch.ones(4, 8, dtype=torch.float32)
        _, y_out = my_op(t, y=t_float)
        assert y_out.dtype == torch.complex64

    def test_wraps_preserves_name(self):
        @bart_op
        def my_special_op(x):
            """My docstring."""
            return x

        assert my_special_op.__name__ == "my_special_op"
        assert "My docstring" in my_special_op.__doc__

    def test_multiple_tensor_args_all_normalised(self):
        dtypes_seen = []

        @bart_op
        def my_op(a, b):
            dtypes_seen.append(a.dtype)
            dtypes_seen.append(b.dtype)
            return a

        a = torch.zeros(4, dtype=torch.float32)
        b = torch.zeros(4, dtype=torch.complex128)
        my_op(a, b)
        assert all(d == torch.complex64 for d in dtypes_seen)


class TestBartOpDecoratorRealOutput:
    def test_no_parens_returns_complex(self):
        """@bart_op (no parens) → real_output=False by default."""

        @bart_op
        def my_op(x):
            return torch.ones(4, dtype=torch.complex64)

        t = torch.zeros(4, dtype=torch.complex64)
        result = my_op(t)
        assert result.is_complex()

    def test_real_output_false_default(self):
        @bart_op(real_output=False)
        def my_op(x):
            return torch.ones(4, dtype=torch.complex64)

        t = torch.zeros(4, dtype=torch.complex64)
        assert my_op(t).is_complex()

    def test_real_output_true_returns_real(self):
        @bart_op(real_output=True)
        def my_op(x):
            return torch.ones(4, dtype=torch.complex64)

        t = torch.zeros(4, dtype=torch.complex64)
        result = my_op(t)
        assert not result.is_complex()
        assert result.dtype == torch.float32

    def test_real_output_non_tensor_passthrough(self):
        """When the op returns a non-tensor, real_output=True is a no-op."""

        @bart_op(real_output=True)
        def my_op(x):
            return 42  # not a tensor

        t = torch.zeros(4, dtype=torch.complex64)
        assert my_op(t) == 42


class TestBartOpDecoratorCallStyle:
    def test_no_parens(self):
        """@bart_op works without parentheses."""

        @bart_op
        def f(x):
            return x

        t = torch.zeros(4, dtype=torch.complex64)
        assert f(t) is t

    def test_with_parens_no_args(self):
        """@bart_op() with empty parens also works."""

        @bart_op()
        def f(x):
            return x

        t = torch.zeros(4, dtype=torch.complex64)
        assert f(t) is t

    def test_with_real_output_kwarg(self):
        """@bart_op(real_output=True) works."""

        @bart_op(real_output=True)
        def f(x):
            return torch.ones(4, dtype=torch.complex64)

        t = torch.zeros(4, dtype=torch.complex64)
        assert not f(t).is_complex()


# ---------------------------------------------------------------------------
# Zero-copy byte equivalence (axis convention sanity check)
# ---------------------------------------------------------------------------


class TestAxisConventionIdentity:
    """Verify the core zero-copy property: a C-order array with reversed dims
    has the same raw bytes as the BART Fortran-order view."""

    def test_2d_byte_equivalence(self):
        rows, cols = 3, 4
        data = np.arange(rows * cols, dtype=np.complex64).reshape(rows, cols)
        bart_dims = _reverse_dims([rows, cols])  # [4, 3]
        reinterpreted = data.ravel().reshape(bart_dims, order="F")
        for i in range(rows):
            for j in range(cols):
                assert data[i, j] == reinterpreted[j, i]
