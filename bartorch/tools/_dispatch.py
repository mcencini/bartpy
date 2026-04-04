"""
bartorch.tools._dispatch — Generic BART tool dispatch helpers.

:func:`call_bart` is the single low-level entry point used by every function
in :mod:`bartorch.tools`.  It normalises array inputs (via the
:func:`~bartorch.core.tensor.bart_op` decorator), then forwards the call to
:func:`bartorch.core.graph.dispatch`.

:func:`make_tool` returns a minimal wrapper function for a named BART tool
with no additional Python-level argument processing.  It is used by
``build_tools/gen_tools.py`` to auto-generate thin wrappers for every BART
CLI command.
"""

from __future__ import annotations

from typing import Any

import torch

from bartorch.core.graph import dispatch
from bartorch.core.tensor import bart_op

__all__ = ["call_bart", "make_tool"]


@bart_op
def call_bart(
    op_name: str,
    *inputs: torch.Tensor,
    output_dims: list[int] | None = None,
    **flags: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Call any BART tool by name.

    This is the lowest-level Python entry point.  All array *inputs* are
    normalised to ``complex64`` by the :func:`~bartorch.core.tensor.bart_op`
    decorator.

    Parameters
    ----------
    op_name : str
        BART command name, e.g. ``"fft"``, ``"pics"``, ``"nufft"``.
    *inputs : torch.Tensor
        Positional array inputs (after normalisation).
    output_dims : list[int] or None, optional
        Expected output shape.  ``None`` lets the C++ extension infer it
        at runtime.
    **flags :
        BART command-line flags, passed directly to the C++ extension.
        Boolean ``True`` values produce bare flags (e.g. ``i=True`` →
        ``-i``); numeric or string values produce flag-value pairs (e.g.
        ``r=0.01`` → ``-r 0.01``).  ``None`` and ``False`` values are
        ignored.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Result(s) in C-order ``complex64``.

    Raises
    ------
    ImportError
        If the ``_bartorch_ext`` C++ extension has not been built.
    """
    return dispatch(op_name, list(inputs), output_dims, **flags)


def make_tool(name: str):
    """Return a thin wrapper function for BART tool *name*.

    The wrapper normalises inputs via :func:`call_bart` and forwards all
    keyword arguments as BART flags.  It is intended for use by
    ``build_tools/gen_tools.py`` to auto-generate wrappers for every BART
    CLI command.

    Parameters
    ----------
    name : str
        BART command name.

    Returns
    -------
    callable
        ``fn(*inputs, output_dims=None, **flags) -> torch.Tensor``
    """

    def _tool(*inputs, output_dims=None, **flags):
        return call_bart(name, *inputs, output_dims=output_dims, **flags)

    _tool.__name__ = name
    _tool.__qualname__ = name
    _tool.__doc__ = (
        f"Auto-generated wrapper for BART tool ``{name}``.\n\n"
        "Accepts any positional ``torch.Tensor`` / NumPy array inputs followed by\n"
        "keyword arguments that map directly to BART command-line flags.\n\n"
        "Raises\n------\nImportError\n    If the ``_bartorch_ext`` C++ extension is absent.\n"
    )
    return _tool
