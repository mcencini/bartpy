"""
bartorch.pipe — Subprocess fallback using CFL temp files.

When the C++ extension is not available, or when a tool has not yet been
wrapped in the C++ layer, operations fall back to subprocess BART calls.

Why NOT named FIFOs
-------------------
BART's ``*.fifo`` pipe support (``FILE_TYPE_PIPE`` in ``src/misc/io.c``) uses
a rich binary streaming protocol (``src/misc/stream.c``): the FIFO carries a
CFL header followed by ``stream_msg`` structs, while the actual data backing
is a memory-mapped temporary file referenced inside the header.  Implementing
this full protocol from Python would be fragile and complex.

Chosen approach — CFL temp files in ``/dev/shm``
-------------------------------------------------
Instead we write standard CFL file pairs (``.hdr`` + ``.cfl``) to ``/dev/shm``
on Linux (a RAM-backed tmpfs — fully in-memory, no physical disk I/O) or to
``tempfile.gettempdir()`` on other platforms.  BART's subprocess reads and
writes these pairs exactly as it would read/write files.  After the call
returns we read the output CFL pair into a BartTensor and clean up.

This approach is:
  - **Correct**: no custom protocol to implement or break
  - **Fast**: ``/dev/shm`` is RAM-backed on Linux
  - **Portable**: works on macOS and other platforms via ``tempfile``

Platform notes
--------------
- Linux:  ``/dev/shm`` (tmpfs, RAM-backed) — preferred
- macOS:  ``tempfile.gettempdir()`` — writes to ``/tmp`` on local SSD
- Windows: ``BARTORCH_SKIP_EXT=1`` must be set but FIFO-style ops are not
  expected to work; use the C++ extension.
"""

from bartorch.pipe.cfl_tmp import run_subprocess, write_cfl_tmp, read_cfl_tmp

# Keep old names exported for backward compat with any direct imports
run_fifo = run_subprocess

__all__ = ["run_subprocess", "run_fifo", "write_cfl_tmp", "read_cfl_tmp"]
