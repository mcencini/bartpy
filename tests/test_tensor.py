"""Tests for bartorch.core.tensor (axis-convention utilities)."""

import numpy as np
import pytest
import torch

from bartorch.core.tensor import _fortran_strides, as_complex64, reverse_dims


class TestReverseDims:
    def test_1d(self):
        assert reverse_dims([5]) == [5]

    def test_2d(self):
        assert reverse_dims([3, 4]) == [4, 3]

    def test_3d(self):
        assert reverse_dims([2, 3, 4]) == [4, 3, 2]

    def test_4d(self):
        assert reverse_dims([8, 1, 256, 256]) == [256, 256, 1, 8]

    def test_accepts_tuple(self):
        assert reverse_dims((2, 3)) == [3, 2]

    def test_roundtrip(self):
        dims = [8, 4, 256, 256]
        assert reverse_dims(reverse_dims(dims)) == dims


class TestAsComplex64:
    def test_already_complex64_is_same_object(self):
        t = torch.zeros(4, 8, dtype=torch.complex64)
        result = as_complex64(t)
        assert result is t  # zero-copy

    def test_float32_gets_cast(self):
        t = torch.ones(4, 8, dtype=torch.float32)
        result = as_complex64(t)
        assert result.dtype == torch.complex64

    def test_complex128_gets_cast(self):
        t = torch.ones(4, 8, dtype=torch.complex128)
        result = as_complex64(t)
        assert result.dtype == torch.complex64

    def test_shape_preserved(self):
        t = torch.zeros(3, 5, 7, dtype=torch.float32)
        result = as_complex64(t)
        assert result.shape == t.shape

    def test_values_preserved(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = as_complex64(t)
        assert result[0].real == pytest.approx(1.0)
        assert result[1].real == pytest.approx(2.0)


class TestFortranStrides:
    """Internal helper — kept for regression."""

    def test_1d(self):
        assert _fortran_strides([5]) == [1]

    def test_2d(self):
        assert _fortran_strides([3, 4]) == [1, 3]

    def test_3d(self):
        assert _fortran_strides([2, 3, 4]) == [1, 2, 6]


class TestAxisConventionIdentity:
    """Verify the core zero-copy property: a C-order array with reversed dims
    has the same raw bytes as the BART Fortran-order view."""

    def test_2d_byte_equivalence(self):
        """C-order (rows, cols) raw bytes == Fortran-order (cols, rows) bytes."""
        rows, cols = 3, 4
        # C-order (rows, cols) array
        data = np.arange(rows * cols, dtype=np.complex64).reshape(rows, cols)
        # BART would see this as Fortran (cols, rows) — same bytes
        bart_dims = list(reversed([rows, cols]))  # [4, 3]
        reinterpreted = data.ravel().reshape(bart_dims, order="F")
        # Element (i,j) in C-order == Element (j,i) in Fortran-order
        for i in range(rows):
            for j in range(cols):
                assert data[i, j] == reinterpreted[j, i]
