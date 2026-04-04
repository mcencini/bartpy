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
| **Zero-copy** | `torch.Tensor` ↔ BART CFL reinterprets `data_ptr()` directly; no serialisation and no `BartTensor` subclass visible to users |
| **C-order Python API** | Axes are reversed at the C++ boundary so users work in NumPy/PyTorch convention; the reversed dims are passed zero-copy to BART's Fortran-order CFL |
| **DSL hot path** | Pure-bartorch call chains stay entirely in C, skipping Python↔C boundaries |
| **Linop algebra** | `BartLinop` supports `@`, `+`, `*`, `.H`, `.N` — composition and sum are built implicitly, not as separate public factories |
| **Full PyTorch citizen** | Ops accept/return plain `torch.Tensor`; integrate with `torch.autograd`, `torch.compile`, CUDA streams, DDP |

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
│   │   ├── tensor_bridge.hpp     ← zero-copy torch.Tensor↔CFL bridge (axis reversal, no copy)
│   │   ├── bart_ops.cpp          ← named op implementations (fft, nufft, pics, …)
│   │   └── cuda/
│   │       └── cuda_bridge.hpp   ← zero-copy CUDA tensor↔CFL bridge
│   │
│   ├── core/                     ← Python-side core
│   │   ├── __init__.py
│   │   ├── context.py            ← BartContext (in-memory CFL session)
│   │   └── graph.py              ← dispatch: hot path vs subprocess fallback
│   │
│   ├── ops/                      ← Public Python API
│   │   ├── __init__.py
│   │   ├── fft.py                ← fft(), ifft(), nufft()
│   │   ├── phantom.py            ← phantom()
│   │   ├── linops.py             ← BartLinop (with __matmul__, __add__, __call__)
│   │   ├── encoding.py           ← sense_op(), nufft_op(), nufft_tseg_op(), wave_op(), subspace_op(), sense_espirit_op()
│   │   ├── regularizers.py       ← wavelet_op(), tv_op(), llr_op(), lr_op()
│   │   ├── pics.py               ← ecalib(), caldir(), pics()
│   │   └── italgos.py            ← conjgrad(), ist(), fista(), irgnm(), chambolle_pock()
│   │
│   ├── pipe/                     ← Subprocess fallback (CFL temp files in /dev/shm)
│   │   ├── __init__.py
│   │   └── cfl_tmp.py            ← write_cfl_tmp, read_cfl_tmp, run_subprocess
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

**The user-facing type is plain `torch.Tensor`.**  `BartTensor` is an
implementation detail of the C++ extension and is **never** part of the public
API.  The Python side works exclusively with standard PyTorch tensors.

When user code calls a bartorch function the C++ extension:
1. Temporarily reinterprets each tensor's `data_ptr()` as a BART CFL buffer
   (zero copy — no intermediate `BartTensor` subclass visible to the user).
2. Executes one or more consecutive `bart_command()` calls in C without
   returning to Python between them (the hot path).
3. Returns the result as a plain `torch.Tensor`.

### 4.2 Axis Convention — C-order Python, reversed Fortran in C

BART internally uses Fortran (column-major) order.  The Python interface uses
**C-order (row-major)** tensors and **reverses the axis names** at the
boundary so that users work in the natural Python/NumPy/PyTorch convention.

| BART C axis order (Fortran)     | bartorch Python axis order (C, reversed) |
|---------------------------------|------------------------------------------|
| `(read, phase1, phase2, coils)` | `(coils, phase2, phase1, read)`          |
| `(samples, views, coils)`       | `(coils, views, samples)`                |
| `(x, y, z)`                     | `(z, y, x)`                             |

The C++ bridge reverses the dimension array before calling
`register_mem_cfl_non_managed()`.  Because the underlying memory layout of a
C-order tensor read in reversed Fortran order is identical to a Fortran-order
tensor in natural order, **no copy is made**.

This convention mirrors PyTorch / NumPy conventions and simplifies downstream
interoperability (e.g. `torch.fft`, `torchkbnufft`).

### 4.3 BartContext

A thread-local session that batches consecutive `bart_command()` calls, keeping
registered `*.mem` names alive across calls so that intermediate tensors are
never re-registered.

```python
with bart_context():
    # phantom() and fft() share the same in-memory CFL session — no re-registration.
    result = bt.fft(bt.phantom([256, 256]), flags=3)
```

Outside a context, each op creates and destroys its own mini-session.

### 4.4 Dispatch logic (`bartorch.core.graph`)

```
dispatch(op_name, inputs, output_shape, **kwargs)
  │
  ├─ C++ ext available?
  │   → Path A: hot path via _bartorch_ext.run()
  │       * accepts any torch.Tensor (C-order, complex64)
  │       * registers data_ptr() zero-copy; reverses dims for BART
  │       * returns plain torch.Tensor (C-order)
  │
  └─ C++ ext not available?
      → Path B: CFL temp files in /dev/shm via bartorch.pipe.run_subprocess()
```

### 4.5 Hot-path execution flow (C++ extension)

```
1.  For each input torch.Tensor (C-order, complex64):
      reversed_dims = reverse(tensor.shape)      # C→Fortran axis reorder
      register_mem_cfl_non_managed("_bt_<uuid>.mem", D, reversed_dims, tensor.data_ptr())
      # BART sees Fortran-order CFL — no data copy

2.  Allocate output tensor (C-order):
      torch::Tensor out = torch::zeros(output_shape, complex64)
      reversed_out_dims = reverse(output_shape)
      register_mem_cfl_non_managed("_bt_<uuid>_out.mem", D, reversed_out_dims, out.data_ptr())

3.  Build argv:  ["fft", "-u", "3", "_bt_xxx.mem", "_bt_yyy_out.mem"]

4.  bart_command(0, nullptr, argc, argv)
      → BART reads from _bt_xxx.mem  (zero copy, correct Fortran layout)
      → BART writes to _bt_yyy_out.mem (zero copy)

5.  memcfl_unlink("_bt_xxx.mem"), memcfl_unlink("_bt_yyy_out.mem")

6.  return out   # plain torch.Tensor, C-order, as seen by user
```

---

## 5. Subprocess Fallback — Design

### 5.1 Why NOT named FIFOs

After inspecting BART source (``src/misc/mmio.c``, ``src/misc/stream.c``),
the ``.fifo``-suffixed pipe mechanism turns out to be a binary streaming
protocol, **not** a simple raw-bytes pipe:

1. ``stream_ensure_fifo(name)`` creates the FIFO via ``mkfifo()``.
2. The writer opens the FIFO and writes a **CFL header** (``write_stream_header``)
   that may contain a ``# Data: <path>`` reference to a memory-mapped temp file.
3. The stream then carries 24-byte ``stream_msg`` structs (BREAK / FLAGS /
   BINARY / INDEX / RAW / BLOCK) to synchronise partial array writes.
4. In "binary" mode (no data file reference) the raw data follows inline,
   framed by sync messages.

Implementing this protocol from Python would be fragile and complex.

### 5.2 Chosen approach — CFL temp files in ``/dev/shm``

Write standard CFL file pairs (``.hdr`` + ``.cfl``) to ``/dev/shm`` on Linux
(a RAM-backed ``tmpfs`` — fully in-memory, no physical disk I/O) or to
``tempfile.gettempdir()`` on other platforms.  BART's subprocess reads and
writes these pairs using its normal CFL path (``FILE_TYPE_CFL``).

```
For each input tensor:
  base = "/dev/shm/_bt_in_<uuid>"
  write  base + ".hdr"          (plain text Dimensions header)
  write  base + ".cfl"          (complex64, Fortran order via ravel('F'))

For each output:
  base = "/dev/shm/_bt_out_<uuid>"

subprocess.call(["bart", op_name, *flags, *input_bases, output_base])

Read output:
  read   output_base + ".hdr"   (parse dimensions)
  read   output_base + ".cfl"   (complex64, reshape with order='F')
  return BartTensor

Cleanup: remove all .hdr / .cfl scratch files
```

**Platform notes:**
- Linux: ``/dev/shm`` (tmpfs, RAM-backed) — preferred
- macOS: ``tempfile.gettempdir()`` → ``/tmp`` on local SSD
- Windows: C++ extension recommended; subprocess path writes to ``%TEMP%``

### 5.3 Implementation (``bartorch/pipe/cfl_tmp.py``)

``run_subprocess(op_name, inputs, output_dims, **kwargs)`` is the single entry
point used by ``bartorch.core.graph.dispatch()`` when the C++ extension is not
available.  ``write_cfl_tmp`` and ``read_cfl_tmp`` are context managers for
individual input/output scratch files; ``_flags_to_argv`` converts keyword
arguments to BART flag strings.

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

## 7. Public API Design

### 7.1 User-facing type: plain `torch.Tensor`

All public bartorch functions accept and return plain `torch.Tensor` objects
(`dtype=torch.complex64`).  The C-order ↔ Fortran axis reversal is handled
transparently inside the C++ extension.  Users never see `BartTensor` or
Fortran-strided tensors.

### 7.2 `BartLinop` — operator class with magic methods

`BartLinop` is the public Python class representing a bounded linear operator
wrapping BART's `struct linop_s*`.  It exposes an interface familiar to
PyTorch / NumPy users:

```python
class BartLinop:
    # Forward application: A(x) or A @ x
    def __call__(self, x: Tensor) -> Tensor: ...
    def __matmul__(self, x: Tensor) -> Tensor: ...   # A @ x

    # Adjoint application: A.H @ x
    @property
    def H(self) -> "BartLinop": ...                  # adjoint operator

    # Composition:  (A @ B)(x) = A(B(x))
    def __matmul__(self, other: "BartLinop") -> "BartLinop": ...

    # Sum: (A + B)(x) = A(x) + B(x)
    def __add__(self, other: "BartLinop") -> "BartLinop": ...

    # Scalar scaling: (2 * A)(x) = 2 * A(x)
    def __mul__(self, scalar: float) -> "BartLinop": ...
    def __rmul__(self, scalar: float) -> "BartLinop": ...

    # Normal operator: A.N = A.H @ A
    @property
    def N(self) -> "BartLinop": ...

    # Shape
    @property
    def ishape(self) -> tuple[int, ...]: ...   # input shape (C-order)
    @property
    def oshape(self) -> tuple[int, ...]: ...   # output shape (C-order)
```

Composition (`A @ B`), sum (`A + B`), and scalar multiplication are built
implicitly from magic methods — they are **not** exposed as separate public
factory functions (`chain()`, `plus()`, etc.).

### 7.3 Public API surface

**Core data ops:**
- `bt.fft(x, ...)` / `bt.ifft(x, ...)` — Cartesian FFT/iFFT
- `bt.nufft(x, traj, ...)` / `bt.nufft_adj(x, traj, ...)` — Non-Cartesian NuFFT

**Phantom / test data:**
- `bt.phantom(shape, ...)` — analytic MRI phantom

**Sensitivity estimation:**
- `bt.ecalib(kspace, ...)` — ESPIRiT coil sensitivity maps
- `bt.caldir(kspace, ...)` — direct calibration

**Full forward encoding operators** (return `BartLinop`):
- `bt.sense_op(sens, mask=None)` — Cartesian SENSE (multi-coil)
- `bt.nufft_op(traj, shape, ...)` — Non-Cartesian SENSE encoding
- `bt.nufft_tseg_op(traj, shape, b0, ti, ...)` — Time-segmented NuFFT with B0 off-resonance correction (`tse()`)
- `bt.wave_op(sens, wave_traj, ...)` — Wave-CAIPI / Wave Shuffling encoding (`linop_wavelet_create()` + k-space PSF)
- `bt.subspace_op(basis, ...)` — Subspace-projected encoding (coefficient → k-space, e.g. T2 shuffling, XD-GRASP)
- `bt.sense_espirit_op(kspace, ...)` — Expanded ESPIRiT multi-map operator (joint estimation via `maps2_create()`)
- `bt.pics(kspace, sens, ...)` — convenience wrapper (ecalib + SENSE + PICS)

**Regularizers** (return `BartLinop` / proximal `BartProxOp`):
- `bt.wavelet_op(shape, wave_type='db4', ...)` — wavelet sparsity transform
- `bt.tv_op(shape, ...)` — gradient / total variation operator
- `bt.llr_op(shape, block_size, ...)` — locally low-rank (Casorati) operator
- `bt.lr_op(shape, ...)` — global low-rank / nuclear norm prox

**Iterative algorithms:**
- `bt.conjgrad(A, b, ...)` — conjugate gradient
- `bt.ist(A, b, prox, ...)` — iterative soft thresholding
- `bt.fista(A, b, prox, ...)` — FISTA
- `bt.irgnm(F, DF, ...)` — iteratively regularised Gauss–Newton
- `bt.chambolle_pock(K, prox_f, prox_g, ...)` — primal-dual

**Nonlinear operators** (return `BartNlinOp`):
- `bt.nlinv_op(...)` — nonlinear inversion for joint image + sensitivity estimation (`struct noir_s*`)
- `bt.moba_op(model, ...)` — model-based reconstruction (T1/T2/T2*/diffusion via `moba`)
- `bt.noir_op(...)` — NOIR nonlinear operator (full NOIR forward + Jacobian)

**Low-level CLI access** (subprocess path):
- `bartorch.tools.*` — auto-generated wrappers for all 100+ BART CLI tools

### 7.4 Op Coverage — Old bartpy → bartorch

| Old module | Old class/function | New bartorch path |
|---|---|---|
| `bartpy.num.fft` | `fft()`, `ifft()` | `bartorch.ops.fft` |
| `bartpy.simu.phantom` | `phantom()` | `bartorch.ops.phantom` |
| `bartpy.linops.linop` | `forward()`, `adjoint()`, `normal()` | `BartLinop.__call__`, `.H`, `.N` |
| `bartpy.linops.linop` | `plus()`, `chain()` | `BartLinop.__add__`, `__matmul__` |
| `bartpy.linops.ops` | `identity()`, `diag()` | `bartorch.ops.linops` (internal helpers) |
| `bartpy.italgos.italgos` | `conjgrad()`, `ist()`, `fista()` | `bartorch.ops.italgos` |
| `bartpy.italgos.italgos` | `irgnm()`, `irgnm2()`, `chambolle_pock()` | `bartorch.ops.italgos` |
| `bartpy.tools.*` | all 100+ CLI tools | `bartorch.tools.*` (subprocess) |

---

## 8. Implementation Roadmap

### Phase 0 — Layout & Infrastructure ✅
- [x] Add `bart` git submodule
- [x] Create `bartorch/` package skeleton
- [x] `BartContext`, dispatch graph (Python stubs)
- [x] Subprocess fallback (`bartorch/pipe/`)
- [x] `CMakeLists.txt` skeleton
- [x] `pyproject.toml`, updated `setup.py`
- [x] `agents.md` (this file)
- [x] Initial tests (`test_context`, `test_cfl`)

### Phase 1 — Build System & C++ Extension Core
- [ ] Resolve BART Makefile → CMake source list for static lib build
- [ ] Implement `tensor_bridge.hpp`: zero-copy `torch.Tensor` ↔ CFL with axis reversal
  - Accept C-order `torch.Tensor` (complex64), reverse dims before `register_mem_cfl_non_managed()`
  - Allocate C-order output tensor, reverse dims for output CFL registration
  - No copy path (reinterpret data_ptr() directly)
- [ ] Implement `_bartorch_ext.run()` in `bartorch_ext.cpp`
  - Build argv, call `bart_command()`
  - Unlink CFL names, return plain `torch.Tensor`
- [ ] CI: build extension on Ubuntu with PyTorch CPU wheel
- [ ] Tests: `test_ops.py` with `bt.phantom()` and `bt.fft()`

### Phase 2 — FFT, NuFFT, Phantom, ecalib, PICS
- [ ] Implement `bart_ops.cpp` functions: `bart_fft`, `bart_nufft`, `bart_phantom`,
      `bart_ecalib`, `bart_pics`
- [ ] Wire Python bindings: `bartorch/ops/fft.py`, `phantom.py`, `pics.py`
- [ ] Integration tests using real MRI demo data

### Phase 3 — `BartLinop` + Encoding Operators
- [ ] `BartLinop`: opaque C++ handle wrapping `struct linop_s*`, exposed with:
  - `__call__`, `__matmul__`, `__add__`, `__mul__` / `__rmul__`, `.H`, `.N`
  - Composition and sum build BART's `linop_chain()` / `linop_plus()` internally
  - `ishape` / `oshape` in C-order (reversed from BART's Fortran dims)
- [ ] Implement `bartorch/ops/encoding.py`:
  - `sense_op()` — Cartesian SENSE via `linop_cdiag_create()` + `linop_fft_create()`
  - `nufft_op()` — Non-Cartesian SENSE via `nufft_create()`
  - `nufft_tseg_op()` — Time-segmented NuFFT via `tse()` + `nufft_create()`
  - `wave_op()` — Wave-CAIPI via `linop_wavelet_create()` + k-space PSF modulation
  - `subspace_op()` — Subspace projection via `linop_cdiag_create()` + basis (`linop_extract_create()`)
  - `sense_espirit_op()` — Expanded ESPIRiT multi-map via `maps2_create()`
- [ ] Integration tests for forward encoding round-trip

### Phase 4 — Regularizers & Proximal Operators
- [ ] `BartProxOp`: C++ handle wrapping `struct operator_p_s*`
- [ ] Implement `bartorch/ops/regularizers.py`:
  - `wavelet_op()` — `linop_wavelet_create()`
  - `tv_op()` — `linop_grad_create()`
  - `llr_op()` — `linop_casorati_create()` + `lrthresh_create()`
  - `lr_op()` — nuclear norm prox via `svthresh()`
- [ ] Tests: verify proximal operator shapes and shrinkage behaviour

### Phase 5 — Iterative Algorithms
- [ ] Implement `bartorch/ops/italgos.py`:
  - `conjgrad()`, `ist()`, `fista()`, `irgnm()`, `chambolle_pock()`
  - Accept `BartLinop` / `BartProxOp` arguments; return `torch.Tensor`
- [ ] Tests mirroring `tests/test_linop.py` from old bartpy

### Phase 6 — Nonlinear Operators
- [ ] `BartNlinOp`: wrapping BART's `struct nlop_s*`
  - `nlinv_op()` — joint image + sensitivity estimation (NLINV / NOIR)
  - `moba_op(model)` — model-based reconstruction for T1/T2/T2*/diffusion
  - `noir_op()` — NOIR full forward + Jacobian
- [ ] Tests for nonlinear forward + Jacobian

### Phase 7 — Auto-generated Tools (`bartorch.tools`)
- [ ] Write `build_tools/gen_tools.py`
  - Query `bart <tool> --interface` for all tools
  - Generate `bartorch/tools/_generated.py` using subprocess backend
  - Accept/return plain `torch.Tensor` (auto-convert via subprocess path)
- [ ] CI: regenerate and test on each BART submodule bump

### Phase 8 — CUDA Support
- [ ] Add CUDA sources to `CMakeLists.txt` (`USE_CUDA=ON`)
- [ ] Implement `cuda/cuda_bridge.hpp`:
  - Register CUDA device pointer via `register_mem_cfl_non_managed()`
  - BART's vptr detects device memory → routes to GPU kernels automatically
  - `cuda_sync_device()` after `bart_command()`
- [ ] Dispatch update: detect CUDA tensors, verify BART CUDA availability
- [ ] CI: test on GPU runner (if available)

### Phase 9 — PyPI Packaging
- [ ] Wheel audit: ensure only `_bartorch_ext.so` + pure Python are packaged
- [ ] CI: build wheels on Linux (x86_64, aarch64) via cibuildwheel
- [ ] CI: macOS (x86_64, arm64) wheels
- [ ] Optional: CUDA wheel (separate package `bartorch-cu118`, `bartorch-cu121`)
- [ ] Publish to PyPI

---

## 9. API Usage Examples

### 9.1 Pure hot path — C-order tensors in, C-order tensors out

```python
import bartorch as bt
import torch

# All bartorch functions accept/return plain torch.Tensor (C-order, complex64).
# The axis reversal and zero-copy bridge happen inside the C++ extension.
with bt.bart_context():
    # shape (z, y) in Python  ↔  BART sees (read=y, phase=z) in Fortran order
    phantom = bt.phantom([256, 256])          # torch.Tensor, shape (256, 256)
    kspace  = bt.fft(phantom, flags=3)        # zero-copy in-process

print(kspace.shape, kspace.dtype)  # torch.Size([256, 256]), torch.complex64
```

### 9.2 Mixed — plain torch.Tensor input, no promotion needed

```python
import torch, bartorch as bt

data = torch.randn(256, 256, dtype=torch.complex64)  # standard PyTorch tensor
result = bt.fft(data, flags=3)                        # accepted directly, no copy
```

### 9.3 Full MRI reconstruction with BartLinop

```python
import bartorch as bt

# shape: (coils, views, samples)  — C-order, reversed from BART's (samples, views, coils)
kspace = bt.phantom([8, 1, 256], kspace=True)
sens   = bt.ecalib(kspace, calib_size=24, maps=1)  # (coils, z, y, x)

# Full forward encoding operator (Cartesian SENSE)
A = bt.sense_op(sens)     # BartLinop: (z, y, x) → (coils, z, y, x)

# Regularizer
W = bt.wavelet_op(A.ishape)   # BartLinop: wavelet sparsity

# Reconstruction: solve  min_x ||A x - y||^2 + lambda ||W x||_1
x0 = torch.zeros(A.ishape, dtype=torch.complex64)
reco = bt.fista(A, kspace, W, lam=0.01, niter=50)  # plain torch.Tensor

import torch
reco_abs = reco.abs()  # interoperable with all PyTorch ops
```

### 9.4 Operator algebra

```python
import bartorch as bt

fft_op = bt.sense_op(sens)          # A: image → k-space
wav_op = bt.wavelet_op(fft_op.ishape)

# Composition via @
combined = fft_op @ wav_op          # BartLinop: coeff → k-space

# Sum via +
reg = wav_op + bt.tv_op(fft_op.ishape)  # combined regularizer

# Adjoint via .H
AT = fft_op.H                       # BartLinop: k-space → image

# Normal equations via .N
ATA = fft_op.N                      # BartLinop: A^H A
```

### 9.5 Non-Cartesian SENSE + iterative recon

```python
import bartorch as bt
import torch

traj = ...  # (2/3, views, samples) torch.Tensor — trajectory
kspace = ... # (coils, views, samples)

# Non-Cartesian SENSE encoding operator
E = bt.nufft_op(traj, img_shape=(256, 256))  # BartLinop

sens  = bt.ecalib(kspace, ...)
A     = bt.sense_op(sens, E)  # compose SENSE + NuFFT

reco  = bt.chambolle_pock(A, kspace, prox_f=bt.wavelet_op(A.ishape), lam=0.005)
```

### 9.6 High-level CLI wrapper (subprocess path)

```python
import bartorch.tools as bart

# Same as old bartpy.tools, but no temp files on disk (/dev/shm used on Linux)
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

## 11. BART Linop & Feature Inventory (Source Inspection, BART v1.0.00)

This section documents which MRI-relevant linear operators and tools are
already implemented in BART and their exposure status in bartorch.

### 11.1 Cartesian SENSE

| Aspect | Detail |
|---|---|
| Source | `src/sense/model.c`, `src/sense/model.h` |
| C API | `sense_init()`, `maps_create()`, `maps2_create()`, `linop_sampling_create()` |
| Tools | `itsense`, `pocsense` |
| Bartorch | Via `pics()` tool; not yet exposed as a Python-side linop |

### 11.2 Non-Cartesian SENSE (NuFFT-based)

| Aspect | Detail |
|---|---|
| Source | `src/noncart/nufft.c`, `src/noncart/nufft_chain.c`, `src/sense/modelnc.c` |
| C API | `nufft_create()`, `nufft_create2()`, `sense_nc_init()` (header: `noncart/nufft.h`, `sense/modelnc.h`) |
| Config | `struct nufft_conf_s` — width, oversamp, pcycle, etc. |
| Tools | `nufft`, `nlinv`, `pics` (non-Cartesian mode) |
| Bartorch | Not yet directly wrapped; accessible via tool wrappers |

### 11.3 Time Segmentation for B0 Off-Resonance Correction

| Aspect | Detail |
|---|---|
| Source | `src/simu/tsegf.c`, `src/simu/tsegf.h` |
| C API | `tse()`, `tse_der()`, `tse_adj()` — analytical phase accumulation for multiple time segments |
| Notes | Composed with `nufft_create()` to produce a full time-segmented encoding operator; each segment applies a phase map and NuFFT, then sums |
| Tools | Used in MOBA, NLINV |
| Bartorch | ❌ Not yet exposed — Phase 3 target: `bt.nufft_tseg_op()` |

### 11.4 Low-Rank Subspace / Casorati Operator

| Aspect | Detail |
|---|---|
| Source | `src/linops/casorati.c/.h`, `src/lowrank/lrthresh.c/.h`, `src/lowrank/svthresh.c/.h`, `src/lowrank/batchsvd.c` |
| C API | `linop_casorati_create()`, `lrthresh_create()`, `svthresh()`, `nuclearnorm()` |
| Notes | `lrthresh_create()` returns `operator_p_s*` (proximal op), not `linop_s*` |
| Tools | `llr`, `pics -L` |
| Bartorch | ❌ Not yet exposed — Phase 3+ target |

### 11.5 Wave-CAIPI / Wave Shuffling

| Aspect | Detail |
|---|---|
| Source | `src/linops/waveop.c/.h` |
| C API | `linop_wave_create()` — PSF modulation along the readout for Wave-CAIPI acceleration |
| Notes | Distinct from `linop_wavelet_create()` (wavelet sparsity); applies oscillating gradient PSF in k-space |
| Tools | `wave` |
| Bartorch | ❌ Not yet exposed — Phase 3 target: `bt.wave_op()` |

### 11.5b Expanded ESPIRiT

| Aspect | Detail |
|---|---|
| Source | `src/calib/calib.c/.h`, `src/sense/model.c/.h` |
| C API | `maps2_create()` — multi-map ESPIRiT operator (more than one set of maps); `calib2()` for calibration |
| Notes | Handles multi-set coil maps for improved conditioning in regions of signal overlap; returns full `linop_s*` |
| Tools | `ecalib -m 2`, `pics` |
| Bartorch | ❌ Not yet exposed — Phase 3 target: `bt.sense_espirit_op()` |

### 11.5c Subspace Projection Encoding

| Aspect | Detail |
|---|---|
| Source | `src/linops/someops.c/.h`, `src/linops/casorati.c/.h` |
| C API | `linop_extract_create()`, `linop_slice_create()`, `linop_cdiag_create()` |
| Notes | Projects coefficient images onto a temporal/spectral subspace basis (e.g. T2 shuffling, XD-GRASP, BART's `pics -B`); composed with `nufft_create()` for full subspace encoding |
| Tools | `pics -B <basis>` |
| Bartorch | ❌ Not yet exposed — Phase 3 target: `bt.subspace_op()` |

### 11.6 Sensitivity Calibration (already exposed)

| Op | Source | C API | Tool | Bartorch |
|---|---|---|---|---|
| ESPIRiT | `src/calib/calib.c/.h` | `calib()`, `calib2()`, `struct ecalib_conf` | `ecalib` | ✅ `bartorch.ops.pics.ecalib()` |
| Direct | `src/calib/direct.c/.h` | `direct_calib()` | `caldir` | ✅ `bartorch.ops.pics.caldir()` |
| PICS | `src/pics.c` | Composite | `pics` | ✅ `bartorch.ops.pics.pics()` |

### 11.7 Linop Framework — Internal vs Public

BART's `linop_chain()`, `linop_plus()`, `linop_stack()`, etc. are used
**internally** by `BartLinop.__matmul__`, `__add__`, and similar magic
methods.  They are **not** exposed as standalone public Python functions.

| C API (`linops/linop.h`) | Used by | Public? |
|---|---|---|
| `linop_chain()`, `linop_chainN()` | `BartLinop.__matmul__` | ❌ internal |
| `linop_plus()`, `linop_plus_FF()` | `BartLinop.__add__` | ❌ internal |
| `linop_stack()`, `linop_stack_cod()` | `BartLinop.stack()` class method | ✅ |
| `linop_pseudo_inv()` | `BartLinop.pseudo_inv()` | ✅ |
| `linop_normal()` | `BartLinop.N` property | ✅ |
| `linop_get_normal()` | `BartLinop.N` property | ❌ internal |

### 11.8 Other Linops Not Yet Exposed (Priority Order)

The following are implemented in BART and should be exposed in Phase 2–3:

| Linop | Source | C API | Notes |
|---|---|---|---|
| **NUFFT** | `src/noncart/nufft.c` | `nufft_create()` | Non-Cartesian k-space encoding |
| **Gradient/TV** | `src/linops/grad.c/.h` | `linop_grad_create()`, `linop_grad_forward_create()` | Finite differences for TV regularization |
| **Wavelet** | `src/linops/waveop.c/.h` | `linop_wavelet_create()` | Wavelet regularization |
| **FFT (linop)** | `src/linops/someops.c/.h` | `linop_fft_create()`, `linop_fftc_create()` | Cartesian FFT |
| **Diagonal** | `src/linops/someops.c/.h` | `linop_cdiag_create()`, `linop_rdiag_create()` | Coil sensitivity multiplication |
| **Reshape** | `src/linops/someops.c/.h` | `linop_reshape_create()`, `linop_permute_create()` | Tensor manipulation |
| **Slice / Extract** | `src/linops/someops.c/.h` | `linop_extract_create()`, `linop_slice_create()` | Subspace/patch extraction |
| **Sum / Repmat** | `src/linops/sum.c/.h` | `linop_sum_create()`, `linop_avg_create()`, `linop_repmat_create()` | Coil combining / replication |
| **Casorati** | `src/linops/casorati.c/.h` | `linop_casorati_create()` | Subspace via matricization |
| **Motion Interp** | `src/motion/interpolate.c/.h` | `linop_interpolate_create()` | Image registration / motion correction |
| **Low-rank thresh** | `src/lowrank/lrthresh.c/.h` | `lrthresh_create()` — returns `operator_p_s*` | Proximal operator, not linop |
| **B0 time seg** | `src/simu/tsegf.c/.h` | `tse()` | Off-resonance correction |

### 11.9 Priority Recommendations

**Phase 2–3 priorities** (most commonly needed for custom algorithms):
1. `nufft_create()` — Non-Cartesian k-space encoding (`bt.nufft_op`)
2. `linop_fft_create()` / `linop_fftc_create()` — Cartesian FFT as linop
3. `linop_cdiag_create()` — Coil sensitivity encoding (`bt.sense_op`)
4. `linop_grad_create()` — Required to build TV regularization (`bt.tv_op`)
5. `tse()` + `nufft_create()` — Time-segmented NuFFT (`bt.nufft_tseg_op`)
6. `linop_wave_create()` — Wave-CAIPI PSF modulation (`bt.wave_op`)
7. `linop_extract_create()` + `linop_cdiag_create()` — Subspace encoding (`bt.subspace_op`)
8. `maps2_create()` + `calib2()` — Expanded ESPIRiT multi-map (`bt.sense_espirit_op`)

**Phase 4 priorities** (regularizers + proximal ops):
9. `linop_wavelet_create()` — Wavelet sparsity (`bt.wavelet_op`)
10. `linop_casorati_create()` + `lrthresh_create()` — Locally low-rank (`bt.llr_op`)
11. `nuclearnorm()` / `svthresh()` — Global low-rank prox (`bt.lr_op`)

**Phase 5–6 (advanced):**
12. NLOP framework (`struct nlop_s*`) — `nlinv_op()`, `moba_op()`, `noir_op()`
13. `tse_der()` / `tse_adj()` — Jacobian of time-segmented encoding (for IRGNM)
14. Wave-CAIPI and motion operators (`linop_interpolate_create()`)

---

*End of agents.md*
