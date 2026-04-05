"""Smoke tests for the compiled C++ extension _bartorch_ext.

These tests verify that:
1. The extension imports cleanly from the bartorch package.
2. The ``run()`` entry point is exposed with the correct signature.
3. The extension raises a RuntimeError (not ImportError) when called,
   because the stub implementation is expected to raise until Phase 1
   wires up the full bart_command() dispatch.
4. The ``register_torch_prior`` / ``unregister_torch_prior`` registry
   bindings are exposed and callable (Phase 0.6).
"""

import pytest

from bartorch import _bartorch_ext as m


def test_ext_imports():
    """Extension module is importable and exposes the expected symbols."""
    assert hasattr(m, "run"), "_bartorch_ext must expose a 'run' callable"
    assert callable(m.run)


def test_ext_run_raises_runtime_error():
    """run() raises RuntimeError (stub) not a crash or ImportError."""
    with pytest.raises(RuntimeError, match="not yet implemented"):
        m.run("phantom", [], None, {})


def test_ext_has_torch_prior_registry():
    """register_torch_prior / unregister_torch_prior are exposed."""
    assert hasattr(m, "register_torch_prior"), "_bartorch_ext must expose 'register_torch_prior'"
    assert hasattr(m, "unregister_torch_prior"), (
        "_bartorch_ext must expose 'unregister_torch_prior'"
    )
    assert callable(m.register_torch_prior)
    assert callable(m.unregister_torch_prior)


def test_ext_register_unregister_torch_prior():
    """register_torch_prior / unregister_torch_prior work without crashing."""
    import torch

    fn = lambda x: x  # noqa: E731
    dims = [4, 4, 1, 1] + [1] * 12  # 16-element BART img_dims

    m.register_torch_prior("_test_prior", fn, dims)
    m.unregister_torch_prior("_test_prior")


def test_ext_register_torch_prior_accepts_nn_module():
    """A torch.nn.Module is accepted as the denoiser callable."""
    import torch

    class Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    net = Identity()
    dims = [8, 8, 1, 1] + [1] * 12
    m.register_torch_prior("_test_nn_prior", net, dims)
    m.unregister_torch_prior("_test_nn_prior")
