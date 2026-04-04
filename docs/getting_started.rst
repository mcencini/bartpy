Getting Started
===============

Introduction
------------

**bartorch** is a PyTorch-native Python interface to the
`Berkeley Advanced Reconstruction Toolbox (BART) <https://mrirecon.github.io/bart/>`_.

It provides:

- **Zero-copy** tensor ↔ BART data exchange via shared memory (no serialisation)
- **DSL hot path** — pure-bartorch call chains execute entirely in C
- **Full PyTorch citizen** — ops integrate with ``torch.autograd``, ``torch.compile``,
  CUDA streams, and DDP

Installation
------------

**Standard install** (requires a C++ compiler, CMake, and BART dependencies):

.. code-block:: bash

   git clone --recurse-submodules https://github.com/mcencini/bartpy
   cd bartpy
   pip install -e .

**Pure-Python / subprocess fallback** (no C++ extension compiled):

.. code-block:: bash

   BARTORCH_SKIP_EXT=1 pip install -e . --no-build-isolation

In this mode bartorch writes CFL file pairs to ``/dev/shm`` (Linux RAM-backed
tmpfs) and invokes BART as a subprocess — no disk I/O, but with a per-call
process-spawn overhead.

Quickstart
----------

.. code-block:: python

   import bartorch as bt
   import bartorch.ops as ops

   # Generate a 256×256 Shepp-Logan phantom (returns a BartTensor)
   ph = ops.phantom([256, 256])
   print("Phantom shape:", ph.shape)   # torch.Size([256, 256])

   # 2-D FFT (flags=3 → dims 0 and 1)
   kspace = ops.fft(ph, flags=3)
   print("k-space shape:", kspace.shape)

   # Compressed-sensing reconstruction with PICS
   # (requires coil sensitivity maps — see docs/examples/ for a full demo)
   # recon = ops.pics(kspace, sens, l=0.001)

Hot path vs subprocess fallback
--------------------------------

When the C++ extension (``_bartorch_ext``) is compiled and available, all ops
execute via the in-memory CFL registry — BART reads and writes directly into
PyTorch tensor memory with zero copies.

When the extension is absent (``BARTORCH_SKIP_EXT=1`` or build failure), the
dispatcher falls back to ``bartorch.pipe``, which writes standard CFL file
pairs to ``/dev/shm`` and runs BART as a subprocess.  Both paths produce
identical numerical results.

.. note::

   The subprocess fallback requires a ``bart`` binary on ``$PATH``.
   The C++ extension embeds BART directly and does **not** require an external
   ``bart`` binary.
