"""
bartorch — PyTorch-native interface to the Berkeley Advanced Reconstruction
Toolbox (BART).

Architecture
------------
* **Hot path** (pure bartorch): when all inputs are ``BartTensor`` objects,
  operations execute entirely inside a single C call with zero copies.
  BART's in-memory CFL registry (``register_mem_cfl_non_managed``) maps
  tensor ``data_ptr()`` buffers directly to BART's named-array namespace.
  ``bart_command()`` then runs the requested tool in-process.

* **Fallback path** (mixed Python): plain ``torch.Tensor``, NumPy arrays, or
  other Python objects trigger an automatic copy into a managed ``BartTensor``
  (or a FIFO-based subprocess call for tools not yet exposed in the C++ layer).

See ``agents.md`` for the full design and roadmap.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bartorch")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Re-export the public surface at package level once sub-modules are built.
# Actual ops/tools are imported lazily to avoid import-time C extension load
# when the package is inspected without a compiled extension present.

__all__ = [
    "__version__",
]
