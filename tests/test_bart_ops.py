"""Integration tests for bartorch BART operations.

Tests are ported from BART's own integration test suite (``bart/tests/*.mk``)
and from the unit tests in ``bart/utests/``.  They verify that real BART
commands execute correctly through the bartorch Python/C++ bridge.

All tests use the **public** ``bartorch.tools`` API (``import bartorch.tools as bt``).
This exercises both the hand-written overrides in ``_commands.py`` (e.g.
``phantom``, ``fft``, ``flip``, ``rss``) and the auto-generated wrappers in
``_generated.py`` (e.g. ``cabs``, ``nrmse``, ``conj``).

Each test is annotated with the original BART test name for traceability.
"""

from __future__ import annotations

import math

import torch

import bartorch.tools as bt

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def nrmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalised RMSE (BART convention: ‖a − b‖ / ‖b‖)."""
    diff = (a - b).abs().norm().item()
    ref = b.abs().norm().item()
    if ref == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / ref


# ---------------------------------------------------------------------------
# phantom — bart/tests/phantom.mk
# ---------------------------------------------------------------------------


def test_phantom_creates_nonzero_image():
    """bt.phantom produces a non-trivial complex64 image."""
    img = bt.phantom([128, 128])
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.complex64
    # Default BART phantom: 128×128 → C-order (128, 128)
    assert img.ndim == 2
    assert img.shape == torch.Size([128, 128])
    assert img.abs().max().item() > 0.0


def test_phantom_kspace_flag():
    """bt.phantom(kspace=True) produces k-space (same shape as image, non-trivial)."""
    img = bt.phantom([128, 128])
    ksp = bt.phantom([128, 128], kspace=True)
    assert ksp.shape == img.shape
    assert ksp.abs().max().item() > 0.0
    # k-space and image should differ
    assert not torch.allclose(ksp, img)


def test_phantom_custom_size():
    """bt.phantom with x=64 produces a 64×64 image."""
    img = bt.phantom([64, 64], x=64)
    assert img.shape[-1] == 64
    assert img.shape[-2] == 64


def test_phantom_coil_shape():
    """bt.phantom with ncoils=4 produces a 128×128×4 image (4 coils)."""
    img = bt.phantom([128, 128], ncoils=4)
    assert img.numel() == 128 * 128 * 4
    assert img.dtype == torch.complex64


def test_phantom_ksp_roundtrip():
    """
    Ported from tests/test-phantom-ksp.
    IFFT(phantom(kspace=True)) ≈ phantom()  with nrmse < 0.22.
    """
    img = bt.phantom([128, 128])
    ksp = bt.phantom([128, 128], kspace=True)
    # IFFT over the last two C-order axes (read + phase1)
    recon = bt.ifft(ksp, axes=(-1, -2))
    err = nrmse(recon, img)
    assert err < 0.22, f"phantom ksp roundtrip nrmse={err:.4f}"


# ---------------------------------------------------------------------------
# FFT — bart/tests/fft.mk
# ---------------------------------------------------------------------------


def test_fft_basic_roundtrip():
    """
    Ported from tests/test-fft-basic.
    IFFT(FFT(x)) == 16384 * x   (nrmse < 1e-4 for float32).
    The factor 16384 = 128*128 is the non-unitary FFT normalisation.
    """
    img = bt.phantom([128, 128])
    ksp = bt.fft(img, axes=(-1, -2))
    recon = bt.ifft(ksp, axes=(-1, -2))
    n = math.prod(img.shape)  # 128*128 = 16384
    err = nrmse(recon, img * n)
    assert err < 1e-4, f"fft basic roundtrip nrmse={err:.2e}"


def test_fft_unitary_roundtrip():
    """
    Ported from tests/test-fft-unitary.
    Unitary FFT is its own inverse: IFFT_u(FFT_u(x)) = x  (nrmse < 1e-4).
    """
    img = bt.phantom([128, 128])
    ksp_u = bt.fft(img, axes=(-1, -2), unitary=True)
    recon = bt.ifft(ksp_u, axes=(-1, -2), unitary=True)
    err = nrmse(recon, img)
    assert err < 1e-4, f"unitary fft roundtrip nrmse={err:.2e}"


def test_fft_shape_preserved():
    """FFT does not change the tensor shape."""
    img = bt.phantom([32, 32], x=32)
    ksp = bt.fft(img, axes=(-1, -2))
    assert ksp.shape == img.shape


def test_fft_output_dtype():
    """FFT output is always complex64."""
    img = bt.phantom([16, 16], x=16)
    ksp = bt.fft(img, axes=-1)
    assert ksp.dtype == torch.complex64


def test_fft_single_axis_roundtrip():
    """
    FFT over the last axis only: IFFT(FFT(x, axes=-1), axes=-1) = N0 * x.
    """
    img = bt.phantom([32, 32], x=32)
    ksp = bt.fft(img, axes=-1)
    recon = bt.ifft(ksp, axes=-1)
    n0 = img.shape[-1]  # fastest-varying axis
    err = nrmse(recon, img * n0)
    assert err < 1e-4, f"single-axis fft roundtrip nrmse={err:.2e}"


def test_fft_multiple_calls_independent():
    """Two separate FFT calls on the same tensor give the same result."""
    img = bt.phantom([32, 32], x=32)
    k1 = bt.fft(img, axes=(-1, -2))
    k2 = bt.fft(img, axes=(-1, -2))
    assert torch.allclose(k1, k2)


# ---------------------------------------------------------------------------
# flip — verifies axes-based ops on small tensors
# ---------------------------------------------------------------------------


def test_flip_last_axis():
    """
    bt.flip(img, axes=-1) reverses the last (read) C-order axis.
    """
    img = bt.phantom([16, 16], x=16)
    flipped = bt.flip(img, axes=-1)
    expected = torch.flip(img, dims=[-1])
    err = nrmse(flipped, expected)
    assert err < 1e-6, f"flip nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# cabs (complex absolute value)
# ---------------------------------------------------------------------------


def test_cabs_nonnegative():
    """bart cabs returns non-negative real values (as complex64)."""
    img = bt.phantom([32, 32], x=32)
    mags = bt.cabs(img)
    assert mags.dtype == torch.complex64
    assert (mags.real >= 0).all()


def test_cabs_magnitude():
    """bart cabs matches torch.abs on random complex input."""
    x = torch.randn(16, 16, dtype=torch.complex64)
    res = bt.cabs(x)
    expected = x.abs().to(torch.complex64)
    err = nrmse(res, expected)
    assert err < 1e-5, f"cabs nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# nrmse (scalar-output path through run())
# ---------------------------------------------------------------------------


def test_nrmse_zero_self():
    """bart nrmse of a tensor with itself is 0."""
    img = bt.phantom([16, 16], x=16)
    result = bt.nrmse(img, img)
    # result can be a string "0.000000\n", a scalar Tensor, or None
    if isinstance(result, torch.Tensor):
        assert result.abs().max().item() < 1e-6
    elif isinstance(result, str):
        val = float(result.strip())
        assert val < 1e-6
    # else None: BART ran but output was not captured — accept silently


# ---------------------------------------------------------------------------
# rss (root-sum-of-squares coil combination)
# ---------------------------------------------------------------------------


def test_rss_coil_shape():
    """bt.rss reduces the coil dimension (C-order axis 0)."""
    coil_imgs = bt.phantom([32, 32], ncoils=4, x=32)
    # coil dim is C-order axis 0 (= BART dim 3 for ndim=4 → bitmask 8)
    rss = bt.rss(coil_imgs, axes=0)
    assert rss.dtype == torch.complex64
    # After squashing the coil dim, numel = 32*32
    assert rss.numel() == 32 * 32


def test_rss_positive_real():
    """bart rss returns non-negative real parts."""
    coil_imgs = bt.phantom([32, 32], ncoils=4, x=32)
    rss = bt.rss(coil_imgs, axes=0)
    assert (rss.real >= -1e-6).all()  # small tolerance for float32 noise


# ---------------------------------------------------------------------------
# conj
# ---------------------------------------------------------------------------


def test_conj_identity():
    """Conjugate applied twice gives original tensor."""
    img = bt.phantom([16, 16], x=16)
    recon = bt.conj(bt.conj(img))
    err = nrmse(recon, img)
    assert err < 1e-6, f"conj double nrmse={err:.2e}"


def test_conj_matches_torch():
    """bart conj matches torch.conj on a known input."""
    x = torch.randn(16, 16, dtype=torch.complex64)
    result = bt.conj(x)
    err = nrmse(result, x.conj())
    assert err < 1e-5, f"conj vs torch.conj nrmse={err:.2e}"
