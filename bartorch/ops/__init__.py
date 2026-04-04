"""
bartorch.ops — Internal linear operator types.

This sub-package provides the abstract operator types used as the internal
building blocks of bartorch's algebraic layer.  It is **not** the entry point
for day-to-day reconstruction work — use :mod:`bartorch.tools` for that.

Available types
---------------
BartLinop
    Opaque handle wrapping a BART ``linop_s*`` linear operator.  Supports
    operator algebra via Python magic methods:

    * ``A(x)``       — forward application
    * ``A @ B``      — composition  (returns :class:`BartLinop`)
    * ``A @ x``      — forward application (alias for ``A(x)``)
    * ``x @ A``      — adjoint application (alias for ``A.H(x)``)
    * ``A + B``      — sum
    * ``scalar * A`` — scalar multiplication
    * ``A.H``        — adjoint operator
    * ``A.N``        — normal operator (``A^H A``)
"""

from bartorch.ops.linops import BartLinop

__all__ = [
    "BartLinop",
]
