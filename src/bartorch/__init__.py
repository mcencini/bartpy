"""
bartorch — PyTorch-native interface to the Berkeley Advanced Reconstruction
Toolbox (BART).

Architecture
------------
All public bartorch functions accept and return plain ``torch.Tensor`` objects
(``dtype=torch.complex64``).  There is no user-visible ``BartTensor`` subclass.

Input normalisation (dtype cast, numpy → tensor conversion) is performed
transparently by the :func:`~bartorch.core.tensor.bart_op` decorator applied
to every op function.  Users never need to call normalisation helpers directly.

**Hot path (C++ extension)**:
    Each tensor's ``data_ptr()`` is registered directly in BART's in-memory
    CFL registry via ``register_mem_cfl_non_managed()``.  Axis reversal
    (C-order ↔ Fortran-order) is handled transparently at the C++ boundary.
    ``bart_command()`` runs the requested tool in-process — zero copies, no
    disk I/O.

**No subprocess fallback**:
    bartorch requires the compiled ``_bartorch_ext`` extension.  The extension
    embeds BART and links to the BLAS and FFT libraries bundled with PyTorch.
    No external ``bart`` binary is needed.  Install from source (``pip install
    -e .``) or via a prebuilt wheel (``pip install bartorch``).

Axis convention
---------------
bartorch uses **C-order** (last index varies fastest), matching NumPy and
PyTorch conventions::

    bartorch shape: (coils, phase2, phase1, read)   ← C-order
    BART internal:  (read,  phase1, phase2, coils)  ← Fortran-order

The C++ bridge reverses the dimension array at the boundary; the underlying
bytes are identical in both conventions — zero copy.

See ``agents.md`` for the full design and implementation roadmap.
"""

from importlib.metadata import PackageNotFoundError, version

# torch must be imported before _bartorch_ext is loaded so that
# THPVariableClass (PyTorch's Python tensor type) is initialised.
# Without this, py::cast<torch::Tensor> in the C++ extension segfaults
# because the Python type object pointer is NULL.
import torch  # noqa: F401  (side-effect import)

try:
    __version__ = version("bartorch")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
]
