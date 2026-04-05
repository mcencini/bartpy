"""Smoke tests for the compiled C++ extension _bartorch_ext.

These tests verify that:
1. The extension imports cleanly from the bartorch package.
2. The ``run()`` entry point is exposed with the correct signature.
3. The extension raises a RuntimeError (not ImportError) when called,
   because the stub implementation is expected to raise until Phase 1
   wires up the full bart_command() dispatch.
"""

import pytest

# If the extension was not compiled (BARTORCH_SKIP_EXT=1), skip gracefully.
try:
    from bartorch import _bartorch_ext as ext

    HAS_EXT = True
except ImportError:
    HAS_EXT = False


@pytest.mark.skipif(not HAS_EXT, reason="_bartorch_ext not compiled (BARTORCH_SKIP_EXT=1)")
def test_ext_imports():
    """Extension module is importable and exposes the expected symbols."""
    from bartorch import _bartorch_ext as m

    assert hasattr(m, "run"), "_bartorch_ext must expose a 'run' callable"
    assert callable(m.run)


@pytest.mark.skipif(not HAS_EXT, reason="_bartorch_ext not compiled (BARTORCH_SKIP_EXT=1)")
def test_ext_run_raises_runtime_error():
    """run() raises RuntimeError (stub) not a crash or ImportError."""
    import torch

    from bartorch import _bartorch_ext as m

    with pytest.raises(RuntimeError, match="not yet implemented"):
        m.run("phantom", [], None, {})
