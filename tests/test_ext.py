"""tests/test_ext.py — smoke tests for bartorch's compiled C++ extension.

All assertions go through the *public* bartorch.tools API so that these tests
stay valid regardless of internal implementation details.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import bartorch.tools as bt

# ---------------------------------------------------------------------------
# Basic import / callable checks
# ---------------------------------------------------------------------------


def test_basic_tools_callable():
    """Core tools are importable and callable."""
    for name in ("phantom", "fft", "ifft", "rss", "nrmse", "flip", "scale"):
        assert callable(getattr(bt, name)), f"bt.{name} is not callable"


# ---------------------------------------------------------------------------
# Phantom
# ---------------------------------------------------------------------------


def test_phantom_returns_complex64():
    """bt.phantom() returns a non-trivial complex64 tensor."""
    result = bt.phantom(x=64)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.complex64
    assert result.ndim >= 2
    assert result.numel() > 0
    assert result.abs().max().item() > 0.0


def test_phantom_respects_x_flag():
    """bt.phantom(x=32) returns a tensor whose last two dims are 32."""
    result = bt.phantom(x=32)
    assert result.shape[-1] == 32
    assert result.shape[-2] == 32


# ---------------------------------------------------------------------------
# FFT roundtrip
# ---------------------------------------------------------------------------


def test_fft_roundtrip():
    """bt.fft followed by bt.ifft reproduces the input within float32 tolerance."""
    ph = bt.phantom(x=32)
    ksp = bt.fft(ph, axes=(-1, -2))
    img = bt.ifft(ksp, axes=(-1, -2))
    nrmse_val = float(bt.nrmse(img, ph))
    assert nrmse_val < 1e-4, f"FFT roundtrip nrmse={nrmse_val:.2e}"


# ---------------------------------------------------------------------------
# Bad input raises RuntimeError
# ---------------------------------------------------------------------------


def test_bad_input_raises():
    """A BART command that fails raises RuntimeError through the public API."""
    with pytest.raises((RuntimeError, ValueError)):
        # phantom with zero dims is rejected by BART
        bt.phantom([0, 0])


# ---------------------------------------------------------------------------
# torch_prior integration point exists in bt.pics signature
# ---------------------------------------------------------------------------


def test_pics_exposes_torch_prior_param():
    """bt.pics has torch_prior and torch_prior_lambda parameters."""
    sig = inspect.signature(bt.pics)
    params = sig.parameters
    assert "torch_prior" in params, "bt.pics must have a 'torch_prior' param"
    assert "torch_prior_lambda" in params, "bt.pics must have a 'torch_prior_lambda' param"
    assert params["torch_prior"].default is None
    assert params["torch_prior_lambda"].default == pytest.approx(1.0)
