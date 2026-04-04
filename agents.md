# bartorch — Agent Technical Plan

**Version:** 0.1.0-dev  
**Date:** 2026-04-04  
**Repository:** https://github.com/mcencini/bartpy  
**BART upstream:** https://codeberg.org/mrirecon/bart (GitHub mirror: https://github.com/mrirecon/bart)

---

## 1. Overview

**bartorch** is a PyTorch-native Python interface to the
[Berkeley Advanced Reconstruction Toolbox (BART)](https://mrirecon.github.io/bart/).
It replaces the legacy SWIG-based `bartpy` package with a modern design built
around three key properties:

| Property | Description |
|---|---|
| **Zero-copy** | Tensor ↔ BART data exchange uses shared memory — no serialisation |
| **DSL hot path** | Pure-bartorch call chains stay entirely in C, skipping Python↔C boundaries |
| **Full PyTorch citizen** | Ops integrate with `torch.autograd`, `torch.compile`, CUDA streams, DDP |

---

## 2. Key Insights from BART Source Inspection

### 2.1 In-memory CFL registry (`src/misc/memcfl.c`)

BART maintains a global linked list of named `complex float*` buffers:

```c
void memcfl_register(const char* name, int D, const long dims[D],
                     complex float* data, bool managed);
complex float* memcfl_load(const char* name, int D, long dims[D]);
void memcfl_unlink(const char* name);
```

With `managed = false` BART does **not** own or free the pointer.  
With `-DMEMONLY_CFL` all file I/O is short-circuited to this registry.

**This is the zero-copy hot path:** register a PyTorch tensor's `data_ptr()`
under a `*.mem` name, call `bart_command()`, the output is written directly
into a pre-allocated output tensor.

### 2.2 Embed API (`src/bart_embed_api.h`)

```c
int  bart_command(int len, char* out, int argc, char* argv[]);
void register_mem_cfl_non_managed(const char* name, unsigned int D,
                                  const long dims[], void* ptr);
void deallocate_all_mem_cfl(void);
```

`bart_command()` is a full re-entrant BART dispatcher that accepts the same
argv as the command-line tool, including support for all 100+ BART operations.

### 2.3 FIFO / pipe support (`src/misc/io.c`)

BART's `file_type()` maps path suffixes to `enum file_types_e`:

| Suffix   | Type              | Notes |
|----------|-------------------|-------|
| `-`      | `FILE_TYPE_PIPE`  | stdin/stdout |
| `.fifo`  | `FILE_TYPE_PIPE`  | named FIFO (mkfifo) |
| `.mem`   | `FILE_TYPE_MEM`   | in-memory CFL registry |
| `.shm`   | `FILE_TYPE_SHM`   | POSIX shared memory |
| `.ra`    | `FILE_TYPE_RA`    | RA format |
| (default)| `FILE_TYPE_CFL`   | `.hdr` + `.cfl` pair |

For the **FIFO fallback** (subprocess path), BART tools can be fed data via
named FIFOs in `/dev/shm` (Linux tmpfs, fully in-memory).  The BART subprocess
opens the FIFO path exactly as it would open a CFL file; a background thread
in our process streams the `complex float` bytes in/out.

### 2.4 Streaming protocol (`src/misc/stream.c`, `src/misc/stream_protocol.h`)

For live/incremental data exchange BART has a binary streaming protocol:

```
MSG_HDR_SIZE = 24 bytes per message
Messages: BREAK, FLAGS, BINARY, INDEX, RAW, BLOCK
```

`stream_create_file()` / `stream_load_file()` handle file-backed streaming.
This richer protocol is the target for future pipeline/streaming mode.

### 2.5 CUDA support (`src/num/gpuops.c`, `src/num/vptr.c`)

BART's virtual pointer (vptr) system detects whether a `complex float*` lives
on device via `cudaPointerGetAttributes()` and routes all arithmetic
automatically to GPU kernels (`.cu` files in `src/num/`, `src/noncart/`,
`src/motion/`).

**GPU zero-copy hot path:** register a CUDA `torch.Tensor`'s device pointer
via `register_mem_cfl_non_managed()` → BART's vptr detects device memory →
GPU kernels execute → output written into pre-allocated CUDA `BartTensor`.
Requires BART compiled with `USE_CUDA=ON` and `cufft`/`cublas`/`cudart`.

---

## 3. Repository Layout

```
bartpy/                           ← repository root
├── bart/                         ← git submodule (mrirecon/bart)
│
├── bartorch/                     ← NEW main package
│   ├── __init__.py
│   ├── csrc/                     ← PyTorch C++ extension source
│   │   ├── CMakeLists.txt
│   │   ├── bartorch_ext.cpp      ← pybind11 module entry point
│   │   ├── tensor_bridge.hpp     ← zero-copy Tensor↔CFL bridge (CPU)
│   │   ├── bart_ops.cpp          ← named op implementations (fft, pics, …)
│   │   └── cuda/
│   │       └── cuda_bridge.hpp   ← zero-copy Tensor↔CFL bridge (CUDA)
│   │
│   ├── core/                     ← Python-side core
│   │   ├── __init__.py
│   │   ├── tensor.py             ← BartTensor + factory helpers
│   │   ├── context.py            ← BartContext (in-memory CFL session)
│   │   └── graph.py              ← dispatch: hot path vs FIFO fallback
│   │
│   ├── ops/                      ← Public Python API (mirrors old SWIG layer)
│   │   ├── __init__.py
│   │   ├── fft.py                ← fft(), ifft()
│   │   ├── phantom.py            ← phantom()
│   │   ├── linops.py             ← BartLinop, identity, diag, chain, …
│   │   ├── pics.py               ← ecalib(), caldir(), pics()
│   │   └── italgos.py            ← conjgrad(), ist(), fista(), irgnm(), …
│   │
│   ├── pipe/                     ← FIFO subprocess fallback
│   │   ├── __init__.py
│   │   └── fifo.py               ← cfl_input_fifo, cfl_output_fifo, run_fifo
│   │
│   ├── tools/                    ← Auto-generated CLI wrappers (all 100+ tools)
│   │   ├── __init__.py
│   │   └── _generated.py         ← generated at build time (gitignored)
│   │
│   └── utils/
│       ├── __init__.py
│       └── cfl.py                ← NumPy CFL read/write (compat)
│
├── bartpy/                       ← LEGACY package (unchanged, deprecated shim)
│   └── …
│
├── build_tools/
│   ├── utils.py                  ← original tool-wrapper generator (legacy)
│   └── gen_tools.py              ← NEW: generates bartorch/tools/_generated.py
│
├── tests/
│   ├── test_tensor.py
│   ├── test_context.py
│   ├── test_cfl.py
│   ├── test_ops.py               ← integration tests (require C++ ext or BART)
│   └── test_tools.py
│
├── agents.md                     ← this file
├── pyproject.toml                ← modern build metadata
├── setup.py                      ← CMake build driver
├── .gitmodules                   ← bart submodule
└── .gitignore
```

---

## 4. DSL Hot Path — Design

### 4.1 Concept

When user code only composes `bartorch` objects and functions, the operations
execute entirely inside C without returning to Python between calls:

```
Python call → BartContext → bart_command() → result in pre-allocated tensor
```

When mixed with standard Python packages, we fall back to the FIFO subprocess
path (still no disk I/O) or the simplest possible copy+call.

### 4.2 BartTensor

`BartTensor` is a `torch.Tensor` subclass:
- dtype: `torch.complex64` (= C's `complex float`)
- strides: Fortran (column-major) — matches BART's memory layout
- carries a `_bart_name: str` — the `*.mem` handle in the CFL registry

The subclass participates in `__torch_function__` so that:
- `BartTensor` OP `BartTensor` → stays on hot path
- `BartTensor` OP `torch.Tensor` → promotes `torch.Tensor` to `BartTensor` 
  (copies if not already compatible), then hot path
- Result of any bartorch op is a `BartTensor`

### 4.3 BartContext

A thread-local session that holds the current CFL registry state.  Within a
context, all registered `*.mem` names remain alive so that consecutive ops
share the same backing memory.

```python
with bart_context() as ctx:
    # Both phantom() and fft() share the same in-memory CFL registry session.
    # No re-registration overhead between ops.
    result = bt.fft(bt.phantom([256, 256]), flags=3)
```

Outside a context, each op creates and destroys its own mini-session.

### 4.4 Dispatch logic (`bartorch.core.graph`)

```
dispatch(op_name, inputs, output_dims, **kwargs)
  │
  ├─ All inputs are BartTensor AND C++ ext available?
  │   → Path A: hot path via _bartorch_ext.run()
  │
  ├─ Some inputs are plain torch.Tensor / ndarray AND C++ ext available?
  │   → Promote to BartTensor (copy) → Path A
  │
  └─ C++ ext not available?
      → Path B: FIFO subprocess via bartorch.pipe.run_fifo()
```

### 4.5 Hot-path execution flow (C++ extension)

```
1.  For each input tensor:
      register_mem_cfl_non_managed("_bt_<uuid>.mem", D, dims, tensor.data_ptr())

2.  Allocate output tensor(s):
      torch::Tensor out = make_bart_tensor(output_dims, device)
      register_mem_cfl_non_managed("_bt_<uuid>_out.mem", D, out_dims, out.data_ptr())

3.  Build argv:  ["fft", "-u", "3", "_bt_xxx.mem", "_bt_yyy_out.mem"]

4.  bart_command(0, nullptr, argc, argv)
      → BART reads from _bt_xxx.mem (= tensor.data_ptr(), zero copy)
      → BART writes to _bt_yyy_out.mem (= out.data_ptr(), zero copy)

5.  memcfl_unlink("_bt_xxx.mem"), memcfl_unlink("_bt_yyy_out.mem")

6.  return BartTensor(out)
```

---

## 5. FIFO Fallback — Design

### 5.1 BART FIFO mechanics

BART recognises `*.fifo` suffixes as `FILE_TYPE_PIPE`.  Its `mmio` routines
open a FIFO path with `fopen()` / `read()` / `write()` just like a regular
file.  The CFL header (`.hdr`) is written as a separate plain text file
alongside the FIFO.

### 5.2 Implementation (`bartorch/pipe/fifo.py`)

```
For each input tensor:
  base = "/dev/shm/_bt_in_<uuid>"
  write  base + ".hdr"          (plain text, synchronous)
  mkfifo base + ".fifo"
  thread: write complex64 bytes into base + ".fifo"

For each output:
  base = "/dev/shm/_bt_out_<uuid>"
  write  base + ".hdr"          (placeholder)
  mkfifo base + ".fifo"
  pre-allocate BartTensor
  thread: read complex64 bytes from base + ".fifo" into tensor

subprocess.call(["bart", op_name, *flags, *input_bases, *output_bases])

join all threads, cleanup FIFOs, return BartTensor
```

**Platform notes:**
- Linux: `/dev/shm` (tmpfs, RAM-backed) — preferred
- macOS: `tempfile.gettempdir()` — FIFOs are supported, but not RAM-backed
- Windows: not supported for FIFO path (use C++ extension or WSL)

---

## 6. Build System

### 6.1 Overview

```
pip install -e .
    └── setup.py (CMakeBuild)
            └── CMakeLists.txt (bartorch/csrc/)
                    ├── find_package(Torch)           ← PyTorch's libtorch
                    ├── find_package(pybind11)
                    ├── add_library(bart_static STATIC …)   ← BART static lib
                    │       compiled with -DMEMONLY_CFL
                    │       linked against ${TORCH_LIBRARIES}  ← reuses MKL/FFTW
                    └── pybind11_add_module(_bartorch_ext …)
                            linked against bart_static + libtorch
```

### 6.2 Linking PyTorch's bundled libraries

PyTorch ships MKL (Intel builds) or OpenBLAS (non-Intel) and FFTW3 (or its
own FFT implementation) inside the wheel.  By linking BART against
`${TORCH_LIBRARIES}` and setting the RPATH to PyTorch's lib dir, the BART
static code reuses these libraries:

```cmake
find_package(Torch REQUIRED)
target_link_libraries(bart_static PUBLIC ${TORCH_LIBRARIES})
# RPATH so the .so finds libtorch_cpu.so at runtime
set_target_properties(_bartorch_ext PROPERTIES
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
```

This keeps the wheel thin: no FFTW/BLAS shipped separately.

### 6.3 BART source selection for the wheel

We compile only the BART source files required for the exposed ops, not the
full 100+ tool suite (which links many additional object files).  The current
minimal set (see `bartorch/csrc/CMakeLists.txt`):

| Module | Purpose |
|--------|---------|
| `misc/` | io, mmio, memcfl, stream, debug, opts, utils |
| `num/`  | multind, flpmath, fft, blas, vecops, rand |
| `simu/` | phantom, shape |
| `linops/` | linop, someops, fmac |
| `iter/` | iter, iter2, prox, thresh |
| embed   | bart_embed_api.c, main.c, bart.c |

Additional BART sources will be added as more ops are wrapped.

### 6.4 CUDA build

```bash
CMAKE_ARGS="-DUSE_CUDA=ON -DCUDA_BASE=/usr/local/cuda" pip install -e .
```

Adds:
- BART `.cu` kernel sources (`gpukrnls*.cu`, `gpu_grid.cu`, etc.)
- `CUDAExtension` in CMake
- Links `cufft`, `cudart`, `cublas` from `CUDAToolkit`
- Enables `cuda/cuda_bridge.hpp` in `bartorch_ext.cpp`

---

## 7. Op Coverage — Mapping from Old bartpy SWIG Interface

The following table maps each old SWIG-wrapped module to its new bartorch
equivalent.  All ops route through the same hot-path or FIFO dispatcher.

| Old module | Old class/function | New bartorch path |
|---|---|---|
| `bartpy.num.fft` | `fft()`, `ifft()` | `bartorch.ops.fft` |
| `bartpy.simu.phantom` | `phantom()` | `bartorch.ops.phantom` |
| `bartpy.linops.ops` | `identity()`, `diag()`, … | `bartorch.ops.linops` |
| `bartpy.linops.linop` | `forward()`, `adjoint()`, `normal()`, `pseudo_inv()` | `bartorch.ops.linops` |
| `bartpy.linops.linop` | `plus()`, `chain()`, `stack()` | `bartorch.ops.linops` |
| `bartpy.italgos.italgos` | `conjgrad()`, `ist()`, `fista()` | `bartorch.ops.italgos` |
| `bartpy.italgos.italgos` | `irgnm()`, `irgnm2()`, `chambolle_pock()` | `bartorch.ops.italgos` |
| `bartpy.tools.*` | all 100+ CLI tools | `bartorch.tools.*` (FIFO-based) |

---

## 8. Implementation Roadmap

### Phase 0 — Layout & Infrastructure ✅
- [x] Add `bart` git submodule
- [x] Create `bartorch/` package skeleton
- [x] `BartTensor`, `BartContext`, dispatch graph (Python stubs)
- [x] FIFO fallback (`bartorch/pipe/`)
- [x] `CMakeLists.txt` skeleton
- [x] `pyproject.toml`, updated `setup.py`
- [x] `agents.md` (this file)
- [x] Initial tests (`test_tensor`, `test_context`, `test_cfl`)

### Phase 1 — Build System & C++ Extension Core
- [ ] Resolve BART Makefile → CMake source list for static lib build
- [ ] Implement `_bartorch_ext.run()` in `bartorch_ext.cpp`
  - Register input tensors via `register_mem_cfl_non_managed()`
  - Allocate output tensors with Fortran strides
  - Build argv, call `bart_command()`
  - Unlink CFL names, return `BartTensor`
- [ ] CI: build extension on Ubuntu with PyTorch CPU wheel
- [ ] Tests: `test_ops.py` with `bt.phantom()` and `bt.fft()`

### Phase 2 — FFT, Phantom, ecalib, PICS
- [ ] Implement `bart_ops.cpp` functions: `bart_fft`, `bart_phantom`,
      `bart_ecalib`, `bart_pics`
- [ ] Wire Python bindings in `bartorch/ops/`
- [ ] Integration tests using real MRI demo data

### Phase 3 — Linear Operators & Iterative Algorithms
- [ ] BartLinop: opaque C++ handle wrapping `struct linop_s*`
- [ ] Implement all linop constructors + composition
- [ ] Implement iterative algorithms (conjgrad, IST, FISTA, IRGNM, CP)
- [ ] Tests mirroring `tests/test_linop.py` from old bartpy

### Phase 4 — Auto-generated Tools (`bartorch.tools`)
- [ ] Write `build_tools/gen_tools.py`
  - Query `bart <tool> --interface` for all tools
  - Generate `bartorch/tools/_generated.py` using FIFO-based backend
  - Accept/return `BartTensor` or `torch.Tensor` (auto-promote)
- [ ] CI: regenerate and test on each BART submodule bump

### Phase 5 — CUDA Support
- [ ] Add CUDA sources to `CMakeLists.txt` (`USE_CUDA=ON`)
- [ ] Implement `cuda/cuda_bridge.hpp`
  - `register_cuda_input()` using device pointer
  - `sync_after_bart()` using `cuda_sync_device()`
- [ ] Dispatch update: detect CUDA tensors, verify BART CUDA availability
- [ ] CI: test on GPU runner (if available)

### Phase 6 — DSL Hotpath Enhancements
- [ ] `BartContext.__enter__` pre-allocates a CFL namespace session
- [ ] Chain multiple `bart_command()` calls without re-registering shared tensors
- [ ] `__torch_function__` dispatch for arithmetic on `BartTensor`
- [ ] Benchmarks: compare hot-path vs FIFO vs old bartpy subprocess

### Phase 7 — PyPI Packaging
- [ ] Wheel audit: ensure only `_bartorch_ext.so` + pure Python are packaged
- [ ] CI: build wheels on Linux (x86_64, aarch64) via cibuildwheel
- [ ] CI: macOS (x86_64, arm64) wheels
- [ ] Optional: CUDA wheel (separate package `bartorch-cu118`, `bartorch-cu121`)
- [ ] Publish to PyPI

---

## 9. API Usage Examples

### 9.1 Pure hot path

```python
import bartorch as bt

# Both calls stay entirely in C (no copies, no subprocess)
with bt.bart_context():
    phantom = bt.ops.phantom([256, 256])          # BartTensor, Fortran-order
    kspace  = bt.ops.fft(phantom, flags=3)        # zero-copy in-place style

# phantom and kspace are torch.Tensor-compatible:
print(kspace.shape, kspace.dtype)  # torch.Size([256, 256]), torch.complex64
```

### 9.2 Mixed (plain torch.Tensor input)

```python
import torch, bartorch as bt

data = torch.randn(256, 256, dtype=torch.complex64)  # plain tensor

# Auto-promoted to BartTensor (one copy), then hot path
result = bt.ops.fft(data, flags=3)
```

### 9.3 Full MRI reconstruction

```python
import bartorch as bt

kspace = bt.utils.readcfl("knee_kspace")          # returns numpy, or:
kspace = bt.ops.phantom([256, 256], kspace=True, ncoils=8)

sens   = bt.ops.ecalib(kspace, calib_size=24, maps=1)
reco   = bt.ops.pics(kspace, sens, lambda_=0.01, wav=True)

# Use as a normal torch tensor for downstream processing
import torch
reco_abs = reco.abs()
```

### 9.4 Linear operator pipeline

```python
import bartorch as bt

fft_op = bt.ops.fft_linop([256, 256], flags=3)
eye    = bt.ops.identity([256, 256])
combo  = bt.ops.chain(fft_op, eye)

x = bt.ops.bart_zeros([256, 256])
y = bt.ops.forward(combo, [256, 256], x)
```

### 9.5 High-level CLI wrapper (FIFO path)

```python
import bartorch.tools as bart

# Same as old bartpy.tools, but no temp files on disk
output = bart.pics(kspace, sens, l=1, r=0.01, i=30)
```

---

## 10. Notes on BART Version Compatibility

The `bart` submodule is pinned to `v1.0.00` (commit `731bfd3e`).

The embed API (`bart_embed_api.h`) has been stable since BART 0.7.00.  
The `memcfl_register` / `memcfl_unlink` API used for the hot path was
introduced in the same version.

The streaming protocol in `src/misc/stream.c` is newer and was introduced
around BART 0.9.x; it is used only for future streaming/pipeline features.

---

*End of agents.md*
