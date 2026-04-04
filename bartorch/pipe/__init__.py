"""
bartorch.pipe — removed.

The subprocess fallback (writing CFL pairs to ``/dev/shm`` and spawning a
``bart`` subprocess) has been removed.  bartorch now requires the compiled C++
extension ``_bartorch_ext`` which embeds BART and links to the BLAS and FFT
libraries bundled with PyTorch.

No external ``bart`` binary is needed.  No data is written to ``/dev/shm`` or
disk during normal operation.

See :mod:`bartorch.core.graph` for the dispatch path.
"""
