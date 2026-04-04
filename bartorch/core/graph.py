"""
bartorch.core.graph — Operation dispatch: hot path vs. fallback.

``dispatch(op_name, inputs, output_dims, **kwargs)`` is the single entry point
used by every op in ``bartorch.ops``.  All inputs and outputs are plain
``torch.Tensor`` objects (``dtype=torch.complex64``).

Dispatch hierarchy (first matching path wins):

  a. **Hot path** — C++ extension available: ``_bartorch_ext.run()`` calls
     ``bart_command()`` in-process, registering each tensor's ``data_ptr()``
     directly in BART's in-memory CFL registry.  Zero copies; axis reversal
     is handled transparently inside the bridge.

  b. **Subprocess fallback** — C++ extension absent: ``bartorch.pipe``
     writes CFL file pairs to ``/dev/shm`` and spawns a ``bart`` subprocess.
     CFL headers carry reversed dims; raw bytes are written in C-order (same
     layout as BART's Fortran order for the reversed dims).
"""

from __future__ import annotations

from typing import Any

import torch

from bartorch.core.tensor import as_complex64

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


def dispatch(
    op_name: str,
    inputs: list[Any],
    output_dims: list[int] | None,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Route *op_name* to the appropriate execution path.

    Parameters
    ----------
    op_name:
        BART tool name (e.g. ``"fft"``, ``"pics"``).
    inputs:
        List of positional array inputs.  May be ``torch.Tensor`` (any dtype,
        cast to complex64 automatically) or ``numpy.ndarray``.
    output_dims:
        Expected output shape, or ``None`` when the shape is inferred at
        runtime by the C++ layer or from the CFL header.
    **kwargs:
        Flag / scalar arguments forwarded to the BART command string.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Operation result(s) as plain complex64 tensors in C-order.
    """
    ext = _try_load_ext()

    # ------------------------------------------------------------------ #
    # Normalise inputs to complex64 torch.Tensor                          #
    # ------------------------------------------------------------------ #
    normalised: list[Any] = []
    for a in inputs:
        if isinstance(a, torch.Tensor):
            normalised.append(as_complex64(a))
        elif _is_numpy(a):
            normalised.append(as_complex64(_numpy_to_tensor(a)))
        else:
            normalised.append(a)

    # ------------------------------------------------------------------ #
    # Path A: C++ hot path                                                #
    # ------------------------------------------------------------------ #
    if ext is not None:
        return _hot_path(ext, op_name, normalised, output_dims, **kwargs)

    # ------------------------------------------------------------------ #
    # Path B: subprocess fallback (CFL temp files in /dev/shm)           #
    # ------------------------------------------------------------------ #
    from bartorch.pipe import run_subprocess  # lazy import

    return run_subprocess(op_name, normalised, output_dims, **kwargs)


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


def _numpy_to_tensor(a) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.asarray(a, dtype=np.complex64))
