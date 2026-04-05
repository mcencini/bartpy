"""Linear operators — bartorch.ops.linops.

:class:`BartLinop` is the public handle type for BART linear operators
(``linop_s*``).  Operators are created by the factory functions in
``bartorch.ops.encoding``, ``bartorch.ops.regularizers``, and directly by the
C++ extension once built.

Operator algebra is expressed via Python magic methods::

    # Composition (apply B first, then A):
    C = A @ B        # → linop_chain(A, B) internally

    # Sum:
    D = A + B        # → linop_plus(A, B) internally

    # Scalar multiplication:
    E = 2.0 * A

    # Adjoint:
    AH = A.H         # → linop with forward/adjoint swapped

    # Normal operator (A^H A):
    AHA = A.N        # → linop_get_normal(A)

    # Application:
    y = A(x)         # → linop_forward(A, x)
    y = A @ x        # same; __matmul__ dispatches on type of rhs

The C++ extension (Phase 3) wires the ``_ptr`` attribute to a real
``linop_s*`` handle.  Until then all constructors return stub instances whose
application methods raise :exc:`NotImplementedError`; the algebra itself
(composition, adjoint, normal, …) works at Python level for design and testing.

Module exports
--------------
BartLinop
"""

from __future__ import annotations

__all__ = ["BartLinop"]

from typing import Any

import torch


class BartLinop:
    """Opaque handle wrapping a BART ``linop_s*`` linear operator.

    Instances are created by the factory functions in
    :mod:`bartorch.ops.encoding` and :mod:`bartorch.ops.regularizers`,
    not instantiated directly by end users.

    Attributes
    ----------
    ishape : tuple[int, ...]
        Domain shape in **C-order** (bartorch convention, e.g.
        ``(coils, ny, nx)``).
    oshape : tuple[int, ...]
        Codomain shape in **C-order**.
    """

    def __init__(
        self,
        ishape: tuple[int, ...] | list[int],
        oshape: tuple[int, ...] | list[int],
        _ptr: Any = None,
        *,
        _kind: str = "base",
        _args: tuple = (),
    ) -> None:
        self.ishape: tuple[int, ...] = tuple(ishape)
        self.oshape: tuple[int, ...] = tuple(oshape)
        self._ptr: Any = _ptr
        self._kind = _kind
        self._args = _args

    # ------------------------------------------------------------------
    # Properties: adjoint and normal operator
    # ------------------------------------------------------------------

    @property
    def H(self) -> "BartLinop":
        """Adjoint operator ``A^H``.

        Returns a :class:`BartLinop` whose forward direction is the adjoint
        of *self* and vice versa.  Domain and codomain are swapped.
        Wraps ``linop_adjoint_create()`` in the C++ extension.
        """
        return BartLinop(
            ishape=self.oshape,
            oshape=self.ishape,
            _kind="adjoint",
            _args=(self,),
        )

    @property
    def N(self) -> "BartLinop":
        """Normal operator ``A^H A``.

        Returns a :class:`BartLinop` that maps the domain to itself.
        Wraps ``linop_get_normal()`` in the C++ extension.
        """
        return BartLinop(
            ishape=self.ishape,
            oshape=self.ishape,
            _kind="normal",
            _args=(self,),
        )

    # ------------------------------------------------------------------
    # Operator algebra
    # ------------------------------------------------------------------

    def __matmul__(
        self, other: "BartLinop | torch.Tensor"
    ) -> "BartLinop | torch.Tensor":
        """Compose with another operator, or apply to a tensor.

        * ``A @ B`` — operator composition: apply *B* first, then *A*.
          Returns a :class:`BartLinop` whose ``ishape`` is *B*'s ``ishape``
          and ``oshape`` is *A*'s ``oshape``.  Wraps ``linop_chain()``
          internally.

        * ``A @ x`` — forward application of *A* to tensor *x*.
          Equivalent to ``A(x)``.

        Parameters
        ----------
        other : BartLinop or torch.Tensor

        Returns
        -------
        BartLinop
            Composed operator (when *other* is a :class:`BartLinop`).
        torch.Tensor
            Result ``A x`` (when *other* is a tensor).
        """
        if isinstance(other, BartLinop):
            return BartLinop(
                ishape=other.ishape,
                oshape=self.oshape,
                _kind="composition",
                _args=(self, other),
            )
        if isinstance(other, torch.Tensor):
            return self(other)
        return NotImplemented

    def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
        """Right-multiply: ``x @ A`` applies the adjoint ``A.H(x)``.

        This implements PyTorch's right-matmul dispatch hook.  When a
        ``torch.Tensor`` ``x`` does ``x @ A``, Python calls
        ``A.__rmatmul__(x)``, which we define as ``A.H(x)`` — consistent
        with the convention that ``x @ A`` reduces a covariant/dual vector
        by ``A^H``.  This is *not* the same as standard matrix-vector
        ``xᵀ A``; it is a deliberate choice for the adjoint application
        shorthand.
        """
        if isinstance(other, torch.Tensor):
            return self.H(other)
        return NotImplemented

    def __add__(self, other: "BartLinop") -> "BartLinop":
        """Sum two operators: ``A + B``.

        Both operators must share the same ``ishape`` and ``oshape``.
        Wraps ``linop_plus()`` internally.
        """
        if not isinstance(other, BartLinop):
            return NotImplemented
        return BartLinop(
            ishape=self.ishape,
            oshape=self.oshape,
            _kind="sum",
            _args=(self, other),
        )

    def __mul__(self, scalar: float) -> "BartLinop":
        """Scale the operator: ``A * scalar``."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return BartLinop(
            ishape=self.ishape,
            oshape=self.oshape,
            _kind="scaled",
            _args=(self, float(scalar)),
        )

    def __rmul__(self, scalar: float) -> "BartLinop":
        """Scale the operator: ``scalar * A``."""
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the forward operator: ``y = A x``.

        Parameters
        ----------
        x : torch.Tensor
            Input in the domain of *self* (complex64, shape ``self.ishape``).

        Returns
        -------
        torch.Tensor
            Output in the codomain (complex64, shape ``self.oshape``).

        Raises
        ------
        NotImplementedError
            Until the C++ extension (Phase 3) is compiled.
        """
        raise NotImplementedError(
            "BartLinop.__call__() requires the bartorch C++ extension (Phase 3). "
            "Operator algebra (A @ B, A + B, A.H, A.N) works at Python level "
            "without the extension."
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BartLinop(ishape={self.ishape}, oshape={self.oshape}, "
            f"kind={self._kind!r})"
        )
