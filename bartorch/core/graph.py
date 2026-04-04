"""
bartorch.core.graph — Operation dispatch: hot path vs. fallback.

``dispatch(op_name, inputs, output_dims, **kwargs)`` is the single entry point
used by every op in ``bartorch.ops``.  It decides at call time whether all
inputs qualify for the zero-copy hot path or whether a fallback is needed.

Hot-path criteria (all must hold):
  1. Every array-valued input is a ``BartTensor`` on the same device.
  2. The ``_bartorch_ext`` C extension has been successfully imported.
  3. For CUDA inputs, BART must have been compiled with ``USE_CUDA``.

Fallback hierarchy (first matching wins):
  a. C++ extension hot path (direct ``bart_command`` in-process, zero copy).
  b. FIFO-based subprocess (``bartorch.pipe``), no-disk via named pipes in
     ``/dev/shm``.
  c. Legacy temp-file subprocess (last resort, maintains backward compat).
"""

from __future__ import annotations

from typing import Any

import torch

from bartorch.core.tensor import BartTensor


# Will be replaced by the real extension at runtime once compiled.
_ext = None


def _try_load_ext():
    global _ext
    if _ext is not None:
        return _ext
    try:
        import _bartorch_ext as ext  # noqa: F401  (C++ extension)
        _ext = ext
    except ImportError:
        pass
    return _ext


def _all_bart_tensors(*args) -> bool:
    """Return True if every tensor-like arg is a BartTensor."""
    for a in args:
        if isinstance(a, torch.Tensor) and not isinstance(a, BartTensor):
            return False
    return True


def dispatch(
    op_name: str,
    inputs: list[Any],
    output_dims: list[int] | None,
    **kwargs,
) -> BartTensor | tuple[BartTensor, ...]:
    """Route *op_name* to the appropriate execution path.

    Parameters
    ----------
    op_name:
        BART tool name (e.g. ``"fft"``, ``"pics"``).
    inputs:
        List of positional array inputs.  May be ``BartTensor``,
        ``torch.Tensor``, or ``numpy.ndarray``.
    output_dims:
        Expected output shape.  ``None`` means the shape is inferred at
        runtime by the C++ layer.
    **kwargs:
        Flag / scalar arguments forwarded to the BART command string.

    Returns
    -------
    BartTensor or tuple of BartTensor
        Operation result(s).
    """
    ext = _try_load_ext()

    # ------------------------------------------------------------------ #
    # Path A: C++ hot path                                                #
    # ------------------------------------------------------------------ #
    if ext is not None and _all_bart_tensors(*inputs):
        return _hot_path(ext, op_name, inputs, output_dims, **kwargs)

    # ------------------------------------------------------------------ #
    # Path B: promote inputs then try hot path                            #
    # ------------------------------------------------------------------ #
    if ext is not None:
        promoted = [
            BartTensor(a) if isinstance(a, torch.Tensor)
            else _numpy_to_bart(a) if _is_numpy(a)
            else a
            for a in inputs
        ]
        return _hot_path(ext, op_name, promoted, output_dims, **kwargs)

    # ------------------------------------------------------------------ #
    # Path C: subprocess fallback (CFL temp files in /dev/shm)           #
    # ------------------------------------------------------------------ #
    from bartorch.pipe import run_subprocess  # lazy import
    return run_subprocess(op_name, inputs, output_dims, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hot_path(ext, op_name, inputs, output_dims, **kwargs):
    """Call into the C++ extension for in-process zero-copy execution."""
    return ext.run(op_name, inputs, output_dims, kwargs)


def _is_numpy(a) -> bool:
    try:
        import numpy as np
        return isinstance(a, np.ndarray)
    except ImportError:
        return False


def _numpy_to_bart(a) -> BartTensor:
    import numpy as np
    t = torch.from_numpy(np.asarray(a, dtype=np.complex64))
    from bartorch.core.tensor import bart_from_tensor
    return bart_from_tensor(t, copy=False)
