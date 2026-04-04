"""
bartorch.core.graph — Operation dispatch via the C++ extension.

``dispatch(op_name, inputs, output_dims, **kwargs)`` is the single entry point
used by every op in ``bartorch.ops``.

bartorch requires the compiled C++ extension ``_bartorch_ext``.  The extension
embeds BART and links to the BLAS and FFT libraries bundled with PyTorch — no
external ``bart`` binary is needed, and no data is ever written to ``/dev/shm``
or disk.

If the extension is not available a clear :exc:`ImportError` is raised with
installation instructions.

Normalisation (complex64 cast, numpy → tensor) is performed upstream by the
:func:`~bartorch.core.tensor.bart_op` decorator applied to every op function.
By the time ``dispatch`` is called all array inputs are guaranteed to be
``torch.complex64`` tensors.
"""

from __future__ import annotations

from typing import Any

import torch

_ext = None
_ext_error: ImportError | None = None
_ext_loaded = False


def _get_ext():
    """Return the compiled C++ extension, raising ``ImportError`` if absent."""
    global _ext, _ext_error, _ext_loaded
    if not _ext_loaded:
        _ext_loaded = True
        try:
            import _bartorch_ext as ext  # noqa: F401

            _ext = ext
        except ImportError as exc:
            _ext_error = ImportError(
                "bartorch requires the compiled C++ extension '_bartorch_ext'.\n"
                "  • Build from source:            pip install -e .\n"
                "  • Install a prebuilt wheel:     pip install bartorch\n"
                "The extension embeds BART and does not require an external 'bart' "
                "binary on $PATH."
            )
            _ext_error.__cause__ = exc
    if _ext is None:
        raise _ext_error
    return _ext


def dispatch(
    op_name: str,
    inputs: list[Any],
    output_dims: list[int] | None,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Route *op_name* through the C++ extension.

    Parameters
    ----------
    op_name:
        BART tool name (e.g. ``"fft"``).
    inputs:
        Positional array inputs.  Must be ``torch.complex64`` tensors —
        normalisation is performed upstream by
        :func:`~bartorch.core.tensor.bart_op`.
    output_dims:
        Expected output shape, or ``None`` to infer at runtime.
    **kwargs:
        Flag / scalar arguments forwarded to the BART command string.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Operation result(s) as plain ``complex64`` tensors in C-order.

    Raises
    ------
    ImportError
        If the ``_bartorch_ext`` C++ extension has not been built.
    """
    ext = _get_ext()
    return ext.run(op_name, inputs, output_dims, kwargs)
