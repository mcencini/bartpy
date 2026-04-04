"""
bartorch — PyTorch-native interface to the Berkeley Advanced Reconstruction
Toolbox (BART).

Architecture
------------
All public bartorch functions accept and return plain ``torch.Tensor`` objects
(``dtype=torch.complex64``).  There is no user-visible ``BartTensor`` subclass.

* **Hot path** (C++ extension available): registers each tensor's
  ``data_ptr()`` directly in BART's in-memory CFL registry via
  ``register_mem_cfl_non_managed()``.  Axis reversal (C-order ↔ Fortran-order)
  is handled transparently at the C++ boundary — zero copies.
  ``bart_command()`` runs the requested tool in-process.

* **Fallback path** (no C++ extension): writes CFL file pairs to
  ``/dev/shm`` (Linux RAM-backed tmpfs) and invokes BART as a subprocess.
  CFL headers carry reversed dims; raw C-order bytes are written to the
  ``.cfl`` file — no data movement.

Axis convention
---------------
bartorch uses **C-order** (last index varies fastest), matching NumPy and
PyTorch conventions::

    bartorch shape: (coils, phase2, phase1, read)   ← C-order
    BART internal:  (read,  phase1, phase2, coils)  ← Fortran-order

The C++ bridge reverses the dimension array at the boundary; the underlying
bytes are identical in both conventions.

See ``agents.md`` for the full design and implementation roadmap.
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
