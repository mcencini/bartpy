"""Smoke tests for the compiled C++ extension _bartorch_ext.

These tests verify that:
1. The extension imports cleanly from the bartorch package.
2. The ``run()`` entry point is exposed with the correct signature.
3. ``register_torch_prior`` / ``unregister_torch_prior`` registry
   bindings are exposed and callable (Phase 0.6).
"""

import pytest
import torch

from bartorch import _bartorch_ext as m


def test_ext_imports():
    """Extension module is importable and exposes the expected symbols."""
    assert hasattr(m, "run"), "_bartorch_ext must expose a 'run' callable"
    assert callable(m.run)


def test_ext_run_signature():
    """run() is callable and accepts the expected arguments.

    pybind11 builtins do not always expose ``__text_signature__`` so we
    verify callability and a successful invocation (phantom) instead.
    """
    assert callable(m.run)
    # Smoke-test: run("phantom", ...) succeeds (tested more thoroughly below)
    result = m.run("phantom", [], None, [], {})
    assert result is not None


def test_ext_run_phantom_basic():
    """run() calls bart phantom and returns a non-trivial complex64 tensor."""
    result = m.run("phantom", [], None, [], {})
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.complex64
    # Default phantom is 128×128 (BART default); bartorch returns reversed shape.
    assert result.ndim >= 2
    assert result.numel() > 0
    # Phantom is not all-zero
    assert result.abs().max().item() > 0.0


def test_ext_run_phantom_size():
    """run() with -x 64 flag returns a 64×64 phantom."""
    result = m.run("phantom", [], None, [], {"x": 64})
    assert result.shape[-1] == 64
    assert result.shape[-2] == 64


def test_ext_run_fft_roundtrip():
    """fft followed by inverse fft reproduces the input (within float32 tol)."""
    # Create a 32×32 Shepp-Logan phantom
    ph = m.run("phantom", [], None, [], {"x": 32})
    # Forward FFT over axes 0 and 1 (bitmask=3 in BART = axes 0,1 in Fortran = dims 0,1)
    ksp = m.run("fft", [ph], None, [3], {"u": True})
    # Inverse FFT
    img = m.run("fft", [ksp], None, [3], {"u": True, "i": True})
    # Should match original within float32 tolerance
    nrmse = (img - ph).abs().norm() / ph.abs().norm()
    assert nrmse.item() < 1e-4, f"FFT roundtrip nrmse={nrmse.item():.2e}"


def test_ext_run_bad_command():
    """run() raises RuntimeError for unknown/failed BART commands."""
    with pytest.raises(RuntimeError):
        m.run("_not_a_real_bart_command_xyz", [], None, [], {})


def test_ext_has_torch_prior_registry():
    """register_torch_prior / unregister_torch_prior are exposed."""
    assert hasattr(m, "register_torch_prior")
    assert hasattr(m, "unregister_torch_prior")
    assert callable(m.register_torch_prior)
    assert callable(m.unregister_torch_prior)


def test_ext_register_unregister_torch_prior():
    """register_torch_prior / unregister_torch_prior work without crashing."""
    fn = lambda x: x  # noqa: E731
    dims = [4, 4, 1, 1] + [1] * 12  # 16-element BART img_dims

    m.register_torch_prior("_test_prior", fn, dims)
    m.unregister_torch_prior("_test_prior")


def test_ext_register_torch_prior_accepts_nn_module():
    """A torch.nn.Module is accepted as the denoiser callable."""

    class Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    net = Identity()
    dims = [8, 8, 1, 1] + [1] * 12
    m.register_torch_prior("_test_nn_prior", net, dims)
    m.unregister_torch_prior("_test_nn_prior")
