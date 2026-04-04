# bartorch — Agent Technical Plan

**Version:** 0.1.0-dev  
**Date:** 2026-04-04  
**Repository:** https://github.com/mcencini/bartpy  
**BART upstream:** https://codeberg.org/mrirecon/bart (GitHub mirror: https://github.com/mrirecon/bart)

---

## 1. Overview

**bartorch** is a PyTorch-native Python interface to the
[Berkeley Advanced Reconstruction Toolbox (BART)](https://mrirecon.github.io/bart/).

| Property | Description |
|---|---|
| **Zero-copy** | `torch.Tensor` ↔ BART CFL reinterprets `data_ptr()` directly; no serialisation |
| **C-order Python API** | Axes are reversed at the C++ boundary so users work in NumPy/PyTorch convention |
| **DSL hot path** | Pure-bartorch call chains stay entirely in C, skipping Python↔C boundaries |
| **Linop algebra** | `BartLinop` supports `@`, `+`, `*`, `.H`, `.N` — composition and sum built implicitly |
| **Full PyTorch citizen** | Ops accept/return plain `torch.Tensor`; integrate with `torch.autograd`, `torch.compile`, CUDA streams |

**No external `bart` binary needed.** No subprocess spawning, no temp files.
The compiled C++ extension embeds BART and links to the BLAS/FFT libraries
bundled with PyTorch.

---

## 2. Key Insights from BART Source

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

**Zero-copy hot path:** register a PyTorch tensor's `data_ptr()` under a
`*.mem` name, call `bart_command()`, output is written directly into a
pre-allocated output tensor.

### 2.2 Embed API (`src/bart_embed_api.h`)

```c
int  bart_command(int len, char* out, int argc, char* argv[]);
void register_mem_cfl_non_managed(const char* name, unsigned int D,
                                  const long dims[], void* ptr);
void deallocate_all_mem_cfl(void);
```

`bart_command()` is a re-entrant BART dispatcher accepting the same argv as
the command-line tool, supporting all 100+ BART operations.

### 2.3 CUDA support (`src/num/gpuops.c`, `src/num/vptr.c`)

BART's virtual pointer (vptr) system detects device memory via
`cudaPointerGetAttributes()` and routes all arithmetic to GPU kernels.

**GPU zero-copy hot path:** register a CUDA tensor's device pointer →
BART's vptr routes to CUDA kernels → output written into pre-allocated CUDA
tensor.  Requires BART compiled with `USE_CUDA=ON`.

---

## 3. Repository Layout

```
bartpy/                           ← repository root
├── bart/                         ← git submodule (mrirecon/bart, v1.0.00)
│
├── bartorch/                     ← main package
│   ├── __init__.py
│   ├── core/                     ← internal infrastructure
│   │   ├── __init__.py
│   │   ├── context.py            ← BartContext (in-memory CFL session)
│   │   ├── graph.py              ← dispatch() — routes to C++ extension
│   │   └── tensor.py             ← @bart_op decorator, _as_complex64, _reverse_dims
│   │
│   ├── ops/                      ← Internal abstract types only
│   │   ├── __init__.py           ← exports: BartLinop
│   │   └── linops.py             ← BartLinop (@, +, *, .H, .N)
│   │
│   ├── tools/                    ← User-facing CLI tool wrappers
│   │   ├── __init__.py           ← exports all named tools + call_bart
│   │   ├── _dispatch.py          ← call_bart(), make_tool() factory
│   │   ├── fft.py                ← fft(), ifft()  [axes_to_flags]
│   │   ├── phantom.py            ← phantom()
│   │   ├── pics.py               ← ecalib(), caldir(), pics()  [full API]
│   │   ├── italgos.py            ← conjgrad(), ist(), fista(), irgnm(), chambolle_pock()
│   │   └── _generated.py         ← generated at build time (gitignored)
│   │
│   ├── utils/
│   │   ├── __init__.py           ← exports: readcfl, writecfl, axes_to_flags
│   │   ├── cfl.py                ← NumPy CFL read/write
│   │   └── flags.py              ← axes_to_flags(axes, ndim) → int bitmask
│   │
│   └── csrc/                     ← PyTorch C++ extension source
│       ├── CMakeLists.txt
│       ├── bartorch_ext.cpp      ← pybind11 module entry point
│       ├── tensor_bridge.hpp     ← zero-copy torch.Tensor↔CFL (axis reversal)
│       ├── bart_ops.cpp          ← named op implementations
│       └── cuda/
│           └── cuda_bridge.hpp   ← zero-copy CUDA tensor↔CFL
│
├── build_tools/
│   └── gen_tools.py              ← generates bartorch/tools/_generated.py
│
├── tests/
│   ├── test_cfl.py
│   ├── test_context.py
│   ├── test_flags.py             ← axes_to_flags tests
│   ├── test_linops.py
│   └── test_tensor.py
│
├── README.md
├── pyproject.toml
├── setup.py
├── .gitmodules
└── agents.md                     ← this file
```

---

## 4. Design Principles

### 4.1 Package split: `ops` vs `tools`

| Package | Contents | Purpose |
|---|---|---|
| `bartorch.ops` | `BartLinop` | Abstract algebraic types (internal building blocks) |
| `bartorch.tools` | `fft`, `phantom`, `ecalib`, `caldir`, `pics`, `conjgrad`, … | User-facing CLI tool wrappers |

The `ops` package is intentionally minimal.  It only contains `BartLinop` —
the opaque handle for BART `linop_s*` objects.  All concrete operations
(including iterative algorithms that take `BartLinop` arguments) live in
`bartorch.tools`.

### 4.2 Axis convention — C-order Python, reversed Fortran in C

BART uses Fortran (column-major) order internally.  bartorch uses **C-order**
tensors and **reverses the axis indices** at the boundary.

| BART Fortran order | bartorch C-order |
|---|---|
| `(read, phase1, phase2, coils)` | `(coils, phase2, phase1, read)` |
| `(x, y, z)` | `(z, y, x)` |

Because a C-order `(a, b, c)` array and a Fortran-order `(c, b, a)` array
share the same byte layout, **no copy is made**.

### 4.3 Axis indices instead of bitmasks

Where BART CLI tools accept a bitmask (e.g. `fft -u 3`), bartorch tools
accept **C-order axis indices** (including negative indices).  The conversion
is performed by `bartorch.utils.flags.axes_to_flags(axes, ndim)`:

```python
# fft over last two axes of a (coils, ny, nx) tensor
bt.fft(x, axes=(-1, -2))   # → BART flag 3

# fft over all axes of a 3-D tensor
bt.fft(x, axes=(0, 1, 2))  # → BART flag 7
```

Formula: for C-order axis `a` (normalised to non-negative), BART axis =
`ndim - 1 - a`, and the corresponding bit is `1 << (ndim - 1 - a)`.

### 4.4 Full CLI API exposure

Every `bartorch.tools` function exposes the **full BART CLI API** via:

1. Named Python parameters for the most commonly used flags.
2. `**extra_flags` to forward any additional BART flag directly.

The keyword name maps directly to the BART flag letter: `R="W:7:0:0.001"` →
`-R W:7:0:0.001`, `e=True` → `-e`.

Example — `pics`:

```python
# Wavelet + total-variation, ADMM solver, GPU
bt.pics(kspace, sens,
        R="W:7:0:0.005",   # wavelet
        R2="T:7:0:0.002",  # TV (passed via extra_flags)
        admm=True,
        gpu=True)
```

### 4.5 Trivial CLI suite exposure

`build_tools/gen_tools.py` generates `bartorch/tools/_generated.py` at build
time.  Each line creates a thin wrapper via `make_tool()`:

```python
nufft = make_tool("nufft")
walsh = make_tool("walsh")
# ...
```

`make_tool(name)` returns a function with signature
`(*inputs, output_dims=None, **flags)` that calls `call_bart(name, ...)`.

To regenerate after a BART update:

```bash
python build_tools/gen_tools.py
```

The script queries `bart --list` when `bart` is on `$PATH`, falling back to a
built-in list of known BART v1.0.00 tools.

### 4.6 BartContext (thread-local session)

```python
from bartorch.core.context import bart_context

with bart_context():
    # All ops in this block share the same in-memory CFL session.
    # Intermediate tensors are never re-registered — pure C hot path.
    result = bt.fft(bt.phantom([256, 256]), axes=(-1, -2))
```

---

## 5. Hot-Path Execution Flow (C++ extension)

```
1.  For each input torch.Tensor (C-order, complex64):
      reversed_dims = reverse(tensor.shape)      # C→Fortran axis reorder
      register_mem_cfl_non_managed("_bt_<uuid>.mem", D, reversed_dims, tensor.data_ptr())

2.  Allocate output tensor (C-order):
      torch::Tensor out = torch::zeros(output_shape, complex64)
      reversed_out_dims = reverse(output_shape)
      register_mem_cfl_non_managed("_bt_<uuid>_out.mem", D, reversed_out_dims, out.data_ptr())

3.  Build argv:  ["fft", "-u", "3", "_bt_xxx.mem", "_bt_yyy_out.mem"]

4.  bart_command(0, nullptr, argc, argv)
      → BART reads from _bt_xxx.mem  (zero copy, correct Fortran layout)
      → BART writes to _bt_yyy_out.mem (zero copy)

5.  memcfl_unlink("_bt_xxx.mem"), memcfl_unlink("_bt_yyy_out.mem")

6.  return out   # plain torch.Tensor, C-order
```

---

## 6. Build System

- **Backend:** `scikit-build-core >= 0.9`
- **CMake source:** `bartorch/csrc/CMakeLists.txt`
- **PyTorch integration:** self-detects `torch.utils.cmake_prefix_path`
- **BART static lib:** compiled from selected source files (not the full suite)
- **BLAS/FFT:** reuses PyTorch's bundled MKL / OpenBLAS / cuFFT

### Minimal BART source set

| Module | Purpose |
|--------|---------|
| `misc/` | io, mmio, memcfl, stream, debug, opts, utils |
| `num/`  | multind, flpmath, fft, blas, vecops, rand |
| `simu/` | phantom, shape |
| `linops/` | linop, someops, fmac |
| `iter/` | iter, iter2, prox, thresh |
| embed   | bart_embed_api.c, main.c, bart.c |

### CUDA build

```bash
CMAKE_ARGS="-DUSE_CUDA=ON" pip install -e .
```

---

## 7. Public API Reference

### `bartorch.tools`

| Function | BART command | Notes |
|---|---|---|
| `fft(x, axes, ...)` | `fft` | C-order axes → bitmask |
| `ifft(x, axes, ...)` | `fft -i` | Convenience alias |
| `phantom(dims, ...)` | `phantom` | Shepp-Logan, geometric |
| `ecalib(kspace, ...)` | `ecalib` | ESPIRiT sensitivity maps |
| `caldir(kspace, ...)` | `caldir` | Direct calibration |
| `pics(kspace, sens, ...)` | `pics` | Full regularisation API |
| `conjgrad(op, b, ...)` | — | CG solver (Phase 4) |
| `ist(op, b, prox, ...)` | — | IST solver (Phase 4) |
| `fista(op, b, prox, ...)` | — | FISTA solver (Phase 4) |
| `irgnm(op, b, ...)` | — | IRGNM (Phase 4) |
| `chambolle_pock(op, ...)` | — | Primal-dual (Phase 4) |
| `call_bart(name, *inputs, **flags)` | any | Generic entry point |
| `make_tool(name)` | any | Factory for thin wrappers |
| `*` (generated) | all 100+ | Auto-generated at build time |

### `bartorch.ops`

| Type | Description |
|---|---|
| `BartLinop` | Opaque handle for `linop_s*`; supports `@`, `+`, `*`, `.H`, `.N` |

### `bartorch.utils`

| Function | Description |
|---|---|
| `axes_to_flags(axes, ndim)` | C-order axis indices → BART Fortran bitmask |
| `readcfl(name)` | Read `.hdr` / `.cfl` pair into NumPy array |
| `writecfl(name, array)` | Write NumPy array as CFL file pair |

---

## 8. Implementation Roadmap

### Phase 0 — Layout & Infrastructure ✅
- [x] Package skeleton (`core/`, `ops/`, `tools/`, `utils/`, `csrc/`)
- [x] `BartContext` + dispatch graph
- [x] `@bart_op` decorator
- [x] `axes_to_flags()` utility
- [x] `BartLinop` with full operator algebra
- [x] `tools/` with fft, phantom, ecalib, caldir, pics (full API), italgos
- [x] `build_tools/gen_tools.py`
- [x] `pyproject.toml`, `setup.py`
- [x] Tests: context, tensor, cfl, linops, flags

### Phase 1 — C++ Extension Core
- [ ] `tensor_bridge.hpp`: zero-copy `torch.Tensor` ↔ CFL with axis reversal
- [ ] `_bartorch_ext.run()`: build argv, call `bart_command()`, return tensor
- [ ] CI: build extension on Ubuntu with PyTorch CPU wheel
- [ ] Tests: `test_tools.py` with `bt.phantom()` and `bt.fft()`

### Phase 2 — Linop Operators
- [ ] `linop_fft_create()`, `linop_fftc_create()` — FFT as linop
- [ ] `linop_grad_create()` — gradient / TV
- [ ] `linop_cdiag_create()` — diagonal (coil sens.)
- [ ] `nufft_create()` — NUFFT encoding
- [ ] Wire Python `BartLinop` application to C++ handles

### Phase 3 — Full MRI Encoding
- [ ] Cartesian SENSE: `sense_init()`, `maps_create()`, `maps2_create()`
- [ ] Non-Cartesian SENSE: `sense_nc_init()`
- [ ] Wavelet: `linop_wavelet_create()`
- [ ] Low-rank: `linop_casorati_create()`, `lrthresh_create()`

### Phase 4 — Iterative Algorithms
- [ ] Wire `conjgrad`, `ist`, `fista`, `irgnm`, `chambolle_pock` to C++ extension
- [ ] Nonlinear: `nlinv_op()`, `moba_op()`

### Phase 5 — CUDA, Packaging
- [ ] CUDA bridge: register device pointer, route to GPU kernels
- [ ] PyPI wheel (CPU + CUDA variants)

---

## 9. BART Version Compatibility

Pinned to **BART v1.0.00** (git submodule).  All source references in this
document use that version.  The embed API (`bart_embed_api.h`) has been stable
since v0.8.00.

---

## 10. BART Linop Inventory (v1.0.00)

### Priority 1 — Core encoding

| Function | Header | Status |
|---|---|---|
| `linop_fft_create()` | `linops/someops.h` | Phase 2 |
| `linop_cdiag_create()` | `linops/someops.h` | Phase 2 |
| `nufft_create()` | `noncart/nufft.h` | Phase 2 |
| `linop_grad_create()` | `linops/grad.h` | Phase 2 |

### Priority 2 — Advanced encoding

| Function | Header | Status |
|---|---|---|
| `sense_init()` / `maps2_create()` | `sense/model.h` | Phase 3 |
| `sense_nc_init()` | `sense/modelnc.h` | Phase 3 |
| `linop_wavelet_create()` | `linops/waveop.h` | Phase 3 |
| `linop_casorati_create()` | `linops/casorati.h` | Phase 3 |
| `lrthresh_create()` | `lowrank/lrthresh.h` | Phase 3 |

### Priority 3 — Specialised

| Function | Header | Status |
|---|---|---|
| `tse()` | `simu/tsegf.h` | Phase 4 |
| `linop_interpolate_create()` | `motion/interpolate.h` | Phase 4 |
| `linop_reshape_create()` | `linops/someops.h` | Phase 4 |
| `linop_permute_create()` | `linops/someops.h` | Phase 4 |
| `linop_sum_create()` | `linops/sum.h` | Phase 4 |
