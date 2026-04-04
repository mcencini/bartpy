"""Tests for bartorch.utils.flags (axes_to_flags)."""

import pytest

from bartorch.utils.flags import axes_to_flags


class TestAxesToFlagsBasic:
    def test_single_last_axis_ndim2(self):
        # C-order axis 1 (last) → BART axis 0 → bit 0 → 1
        assert axes_to_flags(1, ndim=2) == 1

    def test_single_first_axis_ndim2(self):
        # C-order axis 0 (first) → BART axis 1 → bit 1 → 2
        assert axes_to_flags(0, ndim=2) == 2

    def test_both_axes_ndim2(self):
        # Both axes → BART bits 0 and 1 → 3
        assert axes_to_flags((0, 1), ndim=2) == 3

    def test_last_two_axes_ndim3(self):
        # C-order axes (1, 2) of a 3-D tensor → BART axes (1, 0) → bits 1,0 → 3
        assert axes_to_flags((1, 2), ndim=3) == 3

    def test_first_axis_ndim3(self):
        # C-order axis 0 → BART axis 2 → bit 2 → 4
        assert axes_to_flags(0, ndim=3) == 4

    def test_all_axes_ndim3(self):
        # All three axes → bits 0,1,2 → 7
        assert axes_to_flags((0, 1, 2), ndim=3) == 7


class TestAxesToFlagsNegative:
    def test_minus1_equiv_last(self):
        # -1 == axis ndim-1 == last axis
        for ndim in range(1, 6):
            assert axes_to_flags(-1, ndim=ndim) == axes_to_flags(ndim - 1, ndim=ndim)

    def test_minus2_equiv_second_to_last(self):
        for ndim in range(2, 6):
            assert axes_to_flags(-2, ndim=ndim) == axes_to_flags(ndim - 2, ndim=ndim)

    def test_negative_tuple_equiv_positive(self):
        assert axes_to_flags((-1, -2), ndim=3) == axes_to_flags((1, 2), ndim=3)

    def test_all_negative(self):
        assert axes_to_flags((-1, -2, -3), ndim=3) == 7


class TestAxesToFlagsEdgeCases:
    def test_ndim1_single_axis(self):
        assert axes_to_flags(0, ndim=1) == 1
        assert axes_to_flags(-1, ndim=1) == 1

    def test_list_input(self):
        assert axes_to_flags([0, 1], ndim=2) == 3

    def test_single_int_not_tuple(self):
        # scalar int should be accepted
        assert axes_to_flags(0, ndim=3) == 4


class TestAxesToFlagsErrors:
    def test_ndim_zero_raises(self):
        with pytest.raises(ValueError, match="ndim"):
            axes_to_flags(0, ndim=0)

    def test_axis_too_large_raises(self):
        with pytest.raises(ValueError):
            axes_to_flags(3, ndim=3)

    def test_negative_axis_too_small_raises(self):
        with pytest.raises(ValueError):
            axes_to_flags(-4, ndim=3)

    def test_duplicate_axes_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            axes_to_flags((0, 0), ndim=2)

    def test_duplicate_mixed_sign_raises(self):
        # axis 1 and axis -1 are the same in ndim=2
        with pytest.raises(ValueError, match="duplicate"):
            axes_to_flags((1, -1), ndim=2)


class TestAxesToFlagsConsistency:
    """Cross-check that the resulting bitmask is consistent with the
    intended C-order → BART Fortran-order mapping."""

    def test_typical_2d_fft(self):
        # 2-D image (ny, nx): FFT over both → BART axes (1,0) → flags=3
        assert axes_to_flags((0, 1), ndim=2) == 3

    def test_typical_3d_fft_last_two(self):
        # 3-D k-space (coils, ny, nx): FFT over ny,nx → BART axes (1,0) → flags=3
        assert axes_to_flags((-1, -2), ndim=3) == 3

    def test_roundtrip_all_axes(self):
        # Selecting all axes in any order should give the same mask
        for ndim in range(1, 5):
            expected = (1 << ndim) - 1  # all bits set
            assert axes_to_flags(list(range(ndim)), ndim=ndim) == expected
            assert axes_to_flags(list(range(-ndim, 0)), ndim=ndim) == expected
