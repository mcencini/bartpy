"""
bartorch.tools — Auto-generated high-level CLI wrappers.

This sub-package mirrors the old ``bartpy.tools`` interface but uses
:mod:`bartorch.pipe.fifo` instead of temp-file CFL I/O.  All wrappers accept
and return :class:`~bartorch.core.tensor.BartTensor` (or plain
``torch.Tensor`` / NumPy arrays, which are promoted automatically).

The wrappers are generated at build time by ``build_tools/gen_tools.py``,
which queries each BART tool's ``--interface`` output.
"""

# Auto-generated module is imported lazily.
try:
    from bartorch.tools._generated import *  # noqa: F401,F403
except ImportError:
    pass  # Extension / generated file not yet built.
