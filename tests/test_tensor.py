"""Tests for bartorch.core.tensor (BartTensor, factory helpers)."""

import numpy as np
import pytest
import torch

from bartorch.core.tensor import (
    BartTensor,
    _fortran_strides,
    bart_empty,
    bart_from_tensor,
    bart_zeros,
)


class TestFortranStrides:
    def test_1d(self):
        assert _fortran_strides([5]) == [1]

    def test_2d(self):
        assert _fortran_strides([3, 4]) == [1, 3]

    def test_3d(self):
        assert _fortran_strides([2, 3, 4]) == [1, 2, 6]


class TestBartEmpty:
    def test_shape(self):
        t = bart_empty([4, 8])
        assert list(t.shape) == [4, 8]

    def test_dtype(self):
        t = bart_empty([4, 8])
        assert t.dtype == torch.complex64

    def test_fortran_strides(self):
        t = bart_empty([4, 8])
        assert list(t.stride()) == _fortran_strides([4, 8])

    def test_is_bart_tensor(self):
        t = bart_empty([4, 8])
        assert isinstance(t, BartTensor)


class TestBartZeros:
    def test_zeros(self):
        t = bart_zeros([4, 8])
        assert torch.all(t == 0)


class TestBartFromTensor:
    def test_from_complex64_copies(self):
        src = torch.ones(4, 8, dtype=torch.complex64)
        t = bart_from_tensor(src, copy=True)
        assert isinstance(t, BartTensor)
        assert list(t.shape) == [4, 8]
        assert t.dtype == torch.complex64

    def test_from_float_cast(self):
        src = torch.ones(4, 8, dtype=torch.float32)
        t = bart_from_tensor(src)
        assert t.dtype == torch.complex64

    def test_from_numpy_complex(self):
        arr = np.ones((4, 8), dtype=np.complex64)
        t = bart_from_tensor(torch.from_numpy(arr))
        assert isinstance(t, BartTensor)
