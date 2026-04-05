"""bartorch.tools — BART CLI tool wrappers.

Every function in this sub-package routes through
:func:`bartorch.core.graph.dispatch`, which calls the embedded BART C++
extension.  All wrappers accept plain ``torch.Tensor`` / NumPy array inputs
(normalised to ``complex64`` automatically) and return plain ``torch.Tensor``
results.

Axis indices (C-order)
----------------------
Where BART expects a bitmask to select axes, the Python wrappers accept a
scalar axis index or a tuple of indices instead — including negative indices.
The conversion to BART's Fortran-order bitmask is handled transparently by
:func:`bartorch.utils.flags.axes_to_flags`.

Example: ``bt.fft(x, axes=(-1, -2))`` — 2-D FFT over the last two axes.

Named tools
-----------
The following tools have hand-written Pythonic APIs:
``ecalib``, ``caldir``, ``pics``.

All other BART commands (100+) are exposed via auto-generated wrappers in
:mod:`bartorch.tools._generated`.

Generic access
--------------
Any BART command can also be called via :func:`call_bart`::

    import bartorch.tools as bt
    result = bt.call_bart("nufft", traj, kspace, output_dims=[nx, ny])

Auto-generated wrappers (build-time)
-------------------------------------
``build_tools/gen_tools.py`` generates ``bartorch/tools/_generated.py`` at
build time, providing typed wrappers for every BART CLI command (100+).
"""

from __future__ import annotations

from bartorch.tools._dispatch import call_bart, make_tool
from bartorch.tools.pics import caldir, ecalib, pics

__all__ = [
    # Generic entry point
    "call_bart",
    "make_tool",
    # Calibration / reconstruction (hand-written)
    "ecalib",
    "caldir",
    "pics",
]

# Auto-generated wrappers (created at build time by build_tools/gen_tools.py).
# Missing in source trees that have not been built; silently ignored.
try:
    from bartorch.tools._generated import *  # noqa: F401,F403
except ImportError:
    pass
