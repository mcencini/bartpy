"""
bartorch.pipe — FIFO-based subprocess fallback.

When the C++ extension is not available, or when a tool has not yet been
wrapped in the C++ layer, operations fall back to subprocess BART calls using
POSIX named FIFOs (``mkfifo``) so that no temporary files are written to disk.

How BART FIFO support works
---------------------------
BART's ``file_type()`` function (``src/misc/io.c``) maps path suffixes to
internal ``enum file_types_e`` values:

    path ends with ``-``      → FILE_TYPE_PIPE  (stdin/stdout)
    path ends with ``.fifo``  → FILE_TYPE_PIPE  (named FIFO)
    path ends with ``.mem``   → FILE_TYPE_MEM   (in-memory CFL registry)
    path ends with ``.shm``   → FILE_TYPE_SHM   (POSIX shared memory)
    (default)                 → FILE_TYPE_CFL   (regular .hdr/.cfl pair)

For the FIFO fallback we use the ``.fifo`` suffix.  The BART subprocess opens
the FIFO path via its standard mmio routines; a background thread in this
process streams the CFL header + raw complex64 bytes into the write end, and
reads the output FIFO from the read end.

FIFO creation targets ``/dev/shm`` (Linux tmpfs, in-memory) to avoid any
physical disk I/O.  Falls back to ``tempfile.mkdtemp()`` on macOS / other
systems where ``/dev/shm`` is not available.
"""

from bartorch.pipe.fifo import run_fifo, cfl_input_fifo, cfl_output_fifo

__all__ = ["run_fifo", "cfl_input_fifo", "cfl_output_fifo"]
