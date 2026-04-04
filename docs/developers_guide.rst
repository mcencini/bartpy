Developer's Guide
=================

Architecture Overview
---------------------

All bartorch ops accept and return plain ``torch.Tensor`` (``complex64``).
There is no user-visible ``BartTensor`` subclass.

Axis convention
~~~~~~~~~~~~~~~

bartorch uses **C-order** (last index varies fastest), matching NumPy/PyTorch:

.. code-block:: text

    bartorch shape: (coils, phase2, phase1, read)   ← C-order
    BART internal:  (read,  phase1, phase2, coils)  ← Fortran-order

The axis reversal is handled transparently.  A C-order ``(coils, ny, nx)``
array and a Fortran-order ``(nx, ny, coils)`` array have **identical byte
layouts** — so reversing the dims at the boundary is a zero-copy operation.

bartorch has three execution tiers:

**Hot path (C++ extension)**
   All-bartorch call chains execute entirely in C via the embedded BART
   ``bart_command()`` dispatcher.  Each tensor's ``data_ptr()`` is registered
   in BART's in-memory CFL registry via ``register_mem_cfl_non_managed()``;
   BART reads and writes directly into tensor memory.  The C++ bridge reverses
   the dimension array before registration — zero copies, zero subprocess
   overhead.

**Subprocess fallback (``bartorch.pipe``)**
   When the C++ extension is unavailable, ops write CFL file pairs (``.hdr`` +
   ``.cfl``) to ``/dev/shm`` (Linux RAM-backed tmpfs) and invoke a ``bart``
   subprocess.  The ``.hdr`` carries reversed (Fortran) dims; the ``.cfl``
   contains raw C-order bytes — same byte layout, no copy.  No physical disk
   I/O occurs on Linux.

**CUDA path**
   BART's virtual-pointer (vptr) system detects device memory via
   ``cudaPointerGetAttributes()`` and routes arithmetic to GPU kernels
   automatically.  GPU zero-copy requires BART compiled with ``USE_CUDA=ON``.

Dispatch logic (``bartorch.core.graph``)::

   dispatch(op, inputs, output_dims, **kwargs)
     │
     │  Normalise all inputs to complex64 torch.Tensor
     │
     ├─ C++ ext available?
     │   → Hot path: _bartorch_ext.run()
     │     • registers data_ptr() with reversed dims in BART CFL registry
     │     • calls bart_command() in-process
     │     • returns plain torch.Tensor
     │
     └─ C++ ext not available?
         → Subprocess fallback: bartorch.pipe.run_subprocess()
           • writes CFL pairs to /dev/shm with reversed dims in header
           • spawns bart subprocess
           • reads output CFL, reverses dims back, returns plain torch.Tensor

Building the C++ Extension
---------------------------

Prerequisites:

- CMake ≥ 3.18
- C++17 compiler (GCC ≥ 9, Clang ≥ 11)
- PyTorch (CPU or CUDA wheel)
- FFTW3, BLAS (or use PyTorch's bundled libraries)

.. code-block:: bash

   git clone --recurse-submodules https://github.com/mcencini/bartpy
   cd bartpy
   pip install -e .

The ``setup.py`` CMake driver will build the BART static library and link
``_bartorch_ext.so`` against it and PyTorch's bundled libraries.

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
   Wrap the binding or dispatch call.  Accept ``torch.Tensor``, return
   ``torch.Tensor``.  Do not expose ``BartTensor`` in the public signature.

4. **Export** from ``bartorch/ops/__init__.py``.

5. **Tests** in ``tests/test_ops.py``.

Testing
-------

Run the full test suite:

.. code-block:: bash

   pytest tests/ -v

Run only unit tests that do not require BART or the C++ extension:

.. code-block:: bash

   BARTORCH_SKIP_EXT=1 pytest tests/ -v -k "not integration"

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
