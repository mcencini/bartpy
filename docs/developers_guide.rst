Developer's Guide
=================

Architecture Overview
---------------------

All bartorch ops accept and return plain ``torch.Tensor`` (``complex64``).
There is no user-visible ``BartTensor`` subclass.

Input normalisation (complex64 cast, numpy → tensor conversion) is performed
automatically by the :func:`~bartorch.core.tensor.bart_op` decorator applied
to every public op function — no manual conversion is ever needed.

Axis convention
~~~~~~~~~~~~~~~

bartorch uses **C-order** (last index varies fastest), matching NumPy/PyTorch:

.. code-block:: text

    bartorch shape: (coils, phase2, phase1, read)   ← C-order
    BART internal:  (read,  phase1, phase2, coils)  ← Fortran-order

The axis reversal is handled transparently.  A C-order ``(coils, ny, nx)``
array and a Fortran-order ``(nx, ny, coils)`` array have **identical byte
layouts** — so reversing the dims at the boundary is a zero-copy operation.

Execution path
~~~~~~~~~~~~~~

**Hot path (C++ extension)**
   All ops execute via the embedded BART ``bart_command()`` dispatcher.  Each
   tensor's ``data_ptr()`` is registered in BART's in-memory CFL registry via
   ``register_mem_cfl_non_managed()``; BART reads and writes directly into
   tensor memory.  The C++ bridge reverses the dimension array before
   registration — zero copies, no disk I/O.

**CUDA path**
   BART's virtual-pointer (vptr) system detects device memory via
   ``cudaPointerGetAttributes()`` and routes arithmetic to GPU kernels
   automatically.  GPU zero-copy requires BART compiled with ``USE_CUDA=ON``.

.. note::

   There is no subprocess fallback.  bartorch requires the compiled
   ``_bartorch_ext`` C++ extension, which embeds BART and links to the BLAS
   and FFT libraries bundled with PyTorch.  No external ``bart`` binary is
   needed.

Dispatch flow::

   user calls: ops.fft(input, flags=3)
     │
     │  @bart_op decorator (on fft):
     │    • cast all tensor args to complex64  (zero-copy if already correct)
     │    • convert numpy arrays → complex64 torch.Tensor
     │    • pass non-array args unchanged
     │
     ▼
   dispatch("fft", [input], None, flags=3)
     │
     ▼
   _get_ext()  →  raises ImportError if extension absent
     │
     ▼
   _bartorch_ext.run("fft", inputs, output_dims, kwargs)
     • registers data_ptr() with reversed dims in BART CFL registry
     • calls bart_command() in-process
     • returns plain torch.Tensor (complex64, C-order)

Building the C++ Extension
---------------------------

Prerequisites:

- CMake ≥ 3.18
- C++17 compiler (GCC ≥ 9, Clang ≥ 11)
- PyTorch (CPU or CUDA wheel)

The extension links to PyTorch's bundled BLAS and FFT — no separate FFTW3 or
BLAS installation required.

.. code-block:: bash

   git clone --recurse-submodules https://github.com/mcencini/bartpy
   cd bartpy
   pip install -e .

To build with CUDA support:

.. code-block:: bash

   USE_CUDA=1 pip install -e .

Adding New Ops
--------------

1. **C++ side** (``bartorch/csrc/bart_ops.cpp``):
   Add a function that builds the ``argv`` array and delegates to
   ``_bartorch_ext::run_op()``.

2. **Python binding** (``bartorch/csrc/bartorch_ext.cpp``):
   Expose the function via ``pybind11``.

3. **Python op module** (e.g. ``bartorch/ops/mynewop.py``):

   .. code-block:: python

      from bartorch.core.graph import dispatch
      from bartorch.core.tensor import bart_op

      @bart_op
      def my_new_op(input: torch.Tensor, ...) -> torch.Tensor:
          return dispatch("my_tool", [input], None, ...)

   The ``@bart_op`` decorator handles all dtype normalisation automatically.
   Accept and return plain ``torch.Tensor``.

4. **Export** from ``bartorch/ops/__init__.py``.

5. **Tests** in ``tests/test_ops.py``.

Testing
-------

Run the full test suite:

.. code-block:: bash

   pytest tests/ -v

The unit tests (``test_tensor.py``, ``test_linops.py``, ``test_context.py``,
``test_cfl.py``) do not require the C++ extension and run against the Python
stubs.  Integration tests in ``tests/test_ops.py`` require the compiled
extension.

Pre-commit Setup
----------------

Install pre-commit and the hooks defined in ``.pre-commit-config.yaml``:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

The hooks enforce:

- **nbstripout** — strip Jupyter notebook outputs before committing
- **codespell** — spell-check source files (excluding ``attic/``, ``bart/``,
  notebooks, and ``pyproject.toml``)
- **ruff** — Python linting (with ``--fix``) and formatting

Run all hooks manually against all files:

.. code-block:: bash

   pre-commit run --all-files
