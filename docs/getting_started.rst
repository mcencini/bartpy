Getting Started
===============

Introduction
------------

**bartorch** is a PyTorch-native Python interface to the
`Berkeley Advanced Reconstruction Toolbox (BART) <https://mrirecon.github.io/bart/>`_.

It provides:

- **Zero-copy** tensor ↔ BART data exchange via shared memory (no serialisation)
- **Plain** ``torch.Tensor`` API — no user-visible wrapper subclass
- **DSL hot path** — pure-bartorch call chains execute entirely in C
- **Full PyTorch citizen** — ops integrate with ``torch.autograd``, ``torch.compile``,
  CUDA streams, and DDP

Axis convention
---------------

bartorch uses **C-order** (last index varies fastest), matching NumPy and PyTorch:

.. code-block:: text

    bartorch shape: (coils, phase2, phase1, read)   ← C-order
    BART internal:  (read,  phase1, phase2, coils)  ← Fortran-order

The axis reversal is handled transparently at the C++ boundary — no data copy.

Installation
------------

**From source** (requires a C++ compiler, CMake):

.. code-block:: bash

   git clone --recurse-submodules https://github.com/mcencini/bartpy
   cd bartpy
   pip install -e .

**Prebuilt wheel** (CPU or CUDA):

.. code-block:: bash

   pip install bartorch

The C++ extension embeds BART and links to the BLAS and FFT libraries bundled
with PyTorch — no external ``bart`` binary is required.

Quickstart
----------

.. code-block:: python

   import bartorch.tools as bt
   import torch

   # Generate a 256×256 Shepp-Logan phantom (returns a plain torch.Tensor)
   ph = bt.phantom([256, 256])
   print("Phantom type: ", type(ph))    # torch.Tensor
   print("Phantom dtype:", ph.dtype)    # torch.complex64
   print("Phantom shape:", ph.shape)    # (1, 256, 256) — coils first

   # 2-D FFT using C-order axis indices (no raw bitmask needed)
   kspace = bt.fft(ph, axes=(-1, -2))

   # Linear operator algebra
   # import bartorch.lib as bl
   # E = bl.encoding_op(sens)              # SENSE encoding operator
   # EH = E.adjoint(ksp)                   # adjoint application
   # x = E.solve(ksp, maxiter=30)          # CG reconstruction

How It Works
------------

All ops are decorated with :func:`~bartorch.core.tensor.bart_op`, which
normalises every tensor argument automatically:

- ``torch.Tensor`` of any dtype → cast to ``complex64`` (zero-copy if already correct)
- ``numpy.ndarray`` → converted to ``complex64`` ``torch.Tensor``
- Non-array arguments (ints, strings, …) pass through unchanged

The :func:`~bartorch.core.graph.dispatch` function routes each call through
the embedded C++ extension (``_bartorch_ext``):

1. Each tensor's ``data_ptr()`` is registered in BART's in-memory CFL registry.
2. ``bart_command()`` runs the BART tool in-process — no subprocess, no disk I/O.
3. Output is returned as a plain ``complex64`` ``torch.Tensor`` in C-order.

.. note::

   There is no subprocess fallback.  bartorch requires the compiled C++
   extension.  Install from source (``pip install -e .``) or via a prebuilt
   wheel (``pip install bartorch``).
