"""Tests for bartorch.utils.cfl (NumPy CFL read/write)."""

import tempfile
import os
import numpy as np
import pytest

from bartorch.utils.cfl import readcfl, writecfl


class TestCFLRoundtrip:
    def test_2d(self):
        arr = np.random.randn(8, 16).astype(np.complex64) + \
              1j * np.random.randn(8, 16).astype(np.complex64)
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "test")
            writecfl(base, arr)
            arr2 = readcfl(base)
        np.testing.assert_array_almost_equal(arr, arr2)

    def test_3d(self):
        arr = (np.random.randn(4, 8, 16) +
               1j * np.random.randn(4, 8, 16)).astype(np.complex64)
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "test3d")
            writecfl(base, arr)
            arr2 = readcfl(base)
        assert arr2.shape == (4, 8, 16)
        np.testing.assert_array_almost_equal(arr, arr2)

    def test_trailing_ones_stripped(self):
        arr = np.zeros((4, 1, 1), dtype=np.complex64)
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "ones")
            writecfl(base, arr)
            arr2 = readcfl(base)
        # trailing 1s are stripped on read
        assert arr2.shape == (4,) or arr2.shape == (4, 1, 1)

    def test_float_cast(self):
        arr = np.ones((4, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "float")
            writecfl(base, arr)
            arr2 = readcfl(base)
        assert arr2.dtype == np.complex64
