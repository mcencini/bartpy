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
├── src/
│   └── bartorch/                 ← main package (src layout)
│       ├── __init__.py
│       ├── core/                     ← internal infrastructure
│       │   ├── __init__.py
│       │   ├── context.py            ← BartContext (in-memory CFL session)
│       │   ├── graph.py              ← dispatch() — routes to C++ extension
│       │   └── tensor.py             ← @bart_op decorator, _as_complex64, _reverse_dims
│       │
│       ├── ops/                      ← Internal abstract types only
│       │   ├── __init__.py           ← exports: BartLinop
│       │   └── linops.py             ← BartLinop (@, +, *, .H, .N)
│       │
│       ├── tools/                    ← User-facing CLI tool wrappers
│       │   ├── __init__.py           ← exports all named tools
│       │   ├── _commands.py          ← Pythonic overrides (fft/ifft, ecalib, caldir, pics, nlinv, moba, nufft)
│       │   └── _generated.py         ← generated from BART source (committed)
│       │
│       ├── utils/
│       │   ├── __init__.py           ← exports: readcfl, writecfl, axes_to_flags
│       │   ├── cfl.py                ← NumPy CFL read/write
│       │   └── flags.py              ← axes_to_flags(axes, ndim) → int bitmask
│       │
│       └── csrc/                     ← PyTorch C++ extension source
│           ├── CMakeLists.txt
│           ├── bartorch_ext.cpp      ← pybind11 module entry point
│           ├── tensor_bridge.hpp     ← zero-copy torch.Tensor↔CFL (axis reversal)
│           ├── bart_ops.cpp          ← named op implementations
│           └── cuda/
│               └── cuda_bridge.hpp   ← zero-copy CUDA tensor↔CFL
│
├── build_tools/
│   └── gen_tools.py              ← generates src/bartorch/tools/_generated.py
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
└── AGENTS.md                     ← this file
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

### 4.5 Auto-generated CLI wrapper suite

`build_tools/gen_tools.py` generates `src/bartorch/tools/_generated.py` at build
time by **parsing the BART C source files** in the `bart/src/` submodule.  It
never calls a system `bart` binary.

For each tool it:

1. Extracts the tool description from `static const char help_str[]`.
2. Reads `struct arg_s args[]` to identify CFL tensor inputs and positional
   plain-value arguments.
3. Reads `const struct opt_s opts[]` to produce named keyword arguments with
   accurate Python type hints and option descriptions.

Each generated function looks like:

```python
@bart_op
def nufft(
    traj: torch.Tensor,
    input_: torch.Tensor,
    *,
    output_dims: list[int] | None = None,
    a: bool = False,         # adjoint
    i: bool = False,         # inverse
    x: tuple[int, int, int] | None = None,   # output dims x:y:z
    l: float | None = None,  # l2 regularisation
    m: int | None = None,    # max CG iterations
    **extra_flags: Any,
) -> torch.Tensor:
    """Perform non-uniform Fast Fourier Transform. ...
    ...
    """
    return dispatch("nufft", [traj, input_], output_dims,
                    a=a or None, i=i or None, x=x, l=l, m=m,
                    **extra_flags)
```

To regenerate after a BART submodule update:

```bash
python build_tools/gen_tools.py
# Optional: specify paths explicitly
python build_tools/gen_tools.py --bart-src bart/src --out src/bartorch/tools/_generated.py
```

If ``bart/src/`` is absent (submodule not initialised), the script aborts
with an error.  Initialise the submodule first::

    git submodule update --init --recursive

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
- **CMake source:** `src/bartorch/csrc/CMakeLists.txt`
- **PyTorch integration:** self-detects `torch.utils.cmake_prefix_path`
- **BART static lib:** compiled from selected source files (not the full suite)
- **BLAS/FFT:** reuses PyTorch's bundled MKL / OpenBLAS / cuFFT
- **FINUFFT:** statically linked from the `finufft/` submodule (see below)

### Minimal BART source set

| Module | Purpose |
|--------|---------|
| `misc/` | io, mmio, memcfl, stream, debug, opts, utils |
| `num/`  | multind, flpmath, fft, blas, vecops, rand, specfun, multiplace, vptr_fun |
| `simu/` | phantom, shape |
| `linops/` | linop, someops, fmac |
| `iter/` | iter, iter2, prox, thresh |
| `noncart/` | grid.c (KB helpers kept; grid2/grid2H replaced by FINUFFT at link time) |
| embed   | bart_embed_api.c, main.c, bart.c |

### FINUFFT silent grid replacement

`src/bartorch/csrc/finufft_grid.cpp` provides `__wrap_grid2` and
`__wrap_grid2H`.  The final `_bartorch_ext.so` is linked with:
```
-Wl,--wrap,grid2  -Wl,--wrap,grid2H   (Linux / GNU ld)
-Wl,-wrap,_grid2  -Wl,-wrap,_grid2H   (macOS / Apple ld)
```
The GNU/Apple linker then redirects every call to `grid2()` / `grid2H()`
(including those in `noncart/nufft.c`) to the FINUFFT-backed
`__wrap_grid2` / `__wrap_grid2H`.  The original BART KB gridder symbols
(`__real_grid2` / `__real_grid2H`) remain in the binary but are never called.

FINUFFT is built as a **static library** from `finufft/` with:
- FFTW backend: MKL's FFTW3 compatibility layer (`libmkl_rt.so`) — no FFTW
  download needed.
- OpenMP spreading parallelism enabled.
- Tests, examples, Python, MATLAB, Fortran, DUCC0 — all disabled.

CMake option: `BARTORCH_USE_FINUFFT` (default `ON`).  Set to `OFF` to fall
back to BART's original KB gridder.

#### Rolloff correction

`finufft_grid.cpp` also wraps BART's three rolloff functions via `--wrap`:

| Symbol wrapped | Purpose |
|---|---|
| `rolloff_correction` | Fill 3-D correction weight array |
| `apply_rolloff_correction2` | Apply with strides + batch dims |
| `apply_rolloff_correction` | Contiguous delegate |

The replacement computes the Fourier transform of FINUFFT's ES kernel:

```
hat_phi(ξ) = 2 · ∫₀^{J/2} exp(β · √(1 − (x/J/2)²)) · cos(2π·ξ·x) dx
```

using a 256-point midpoint quadrature, and applies `1/hat_phi(ξ)` as the
per-pixel deconvolution weight.  This exactly cancels the ES-kernel gain
introduced by `__wrap_grid2` / `__wrap_grid2H`, giving a correct end-to-end
NUFFT pipeline.  The kernel parameters (nspread=7, β=16.1 for tol=1e-6 and
σ=2) are derived from FINUFFT's `setup_spreader` formula and hardcoded for
efficiency; no FINUFFT internal headers are required.

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

### Phase 0.5 — FINUFFT silent grid replacement ✅
- [x] `finufft/` git submodule (FINUFFT v2.6.0-dev)
- [x] `src/bartorch/csrc/finufft_grid.cpp`: `__wrap_grid2` / `__wrap_grid2H`
  using FINUFFT type-1/2 `spreadinterponly=1`
- [x] `src/bartorch/csrc/finufft_grid.cpp`: `__wrap_rolloff_correction` /
  `__wrap_apply_rolloff_correction` / `__wrap_apply_rolloff_correction2` —
  ES-kernel FT rolloff computed via 256-pt midpoint quadrature, matching the
  FINUFFT spreading kernel used in `__wrap_grid2` / `__wrap_grid2H`
- [x] CMakeLists.txt: FINUFFT subdirectory, MKL FFTW3 backend, `--wrap` linker
  options for grid2/grid2H + rolloff, `BARTORCH_USE_FINUFFT` option
- [x] `noncart/grid.c` + deps (`num/specfun.c`, `num/multiplace.c`,
  `num/vptr_fun.c`) added to `BART_SOURCES`
- [x] `tests/test_finufft_grid.py`: import + symbol checks, pure-Python ES
  rolloff weight tests (numerical adjointness tests gated on Phase-1 `run()`)
- [x] AGENTS.md §6 and §8 updated

### Phase 0.6 — Torch prior (plug-and-play) via `--wrap nlop_tf_create` ✅

Hijacks BART's TensorFlow-prior interface (`-R TF:{path}:lambda` in
`grecon/optreg.c`) using the same `--wrap` linker trick as FINUFFT.  No BART
source modifications required.

**Call flow:**

1. Python: `bt.pics(kspace, sens, torch_prior=denoiser, torch_prior_lambda=1.0)`
2. `pics()` in `_commands.py` computes BART Fortran-order `img_dims` from
   the kspace shape, calls `_ext.register_torch_prior(name, fn, img_dims)`, and
   appends `R='TF:{bartorch://<name>}:<lambda>'` to the BART flags.
3. `dispatch("pics", …)` → `bart_command("pics … -R TF:{bartorch://name}:lam")`
4. BART's `optreg.c` TENFL case calls `nlop_tf_create("bartorch://name")`.
5. `__wrap_nlop_tf_create` (in `torch_prior.cpp`) intercepts, looks up the
   callable in `g_prior_registry`, creates `nlop_torch_prior_create(fn, dims)`.
6. BART wraps the nlop in `prox_nlgrad_create(nlop, 1, 1.0, lambda, false)`.
7. Every ADMM/IST iteration calls the proximal operator:
   - `nlop_apply` → GIL → `fn(x)` → `residual = x − D(x)` (cached)
   - `nlop_adjoint` → returns `scale × residual`
   - Update: `z ← z − mu·lambda · (z − D(z))`
   With `mu = 1`, `lambda = 1`: `z ← D(z)` (standard PnP-RED proximal step).
8. Python `finally` block calls `_ext.unregister_torch_prior(name)`.

**nlop semantics (`grad_nlop=false` path):**

| Step | Operation |
|---|---|
| `forward(x) → scalar` | Calls `D(x)`, caches `residual = x − D(x)`, returns `‖residual‖²/2` |
| `adjoint(x, grad=1) → image` | Returns cached `residual = x − D(x)` |
| `deriv(x) → scalar` | Identity approximation (returns 0; not used by `prox_nlgrad`) |

**Denoiser convention:**

```python
fn(x: torch.Tensor) -> torch.Tensor
# x: flat complex64 tensor, length = prod(spatial_dims)
# returns: same shape and dtype
```

No `sigma` argument — the denoiser is called without noise-level information.
Compatible with `torch.nn.Module`, deepinverse `Denoiser`, and plain functions.

**BART img_dims convention:**

bartorch kspace C-order: `(nc, [nz,] ny, nx)` with a singleton z-dim for 2-D.
Fortran dims (reversed): `[nx, ny, nz, nc, 1, …]` — coil at Fortran dim 3.
`img_dims` = Fortran ksp_dims with `dims[3] = 1` (coil zeroed).

- [x] `src/bartorch/csrc/torch_prior.cpp`: global registry, `nlop_torch_prior_create`,
  `__wrap_nlop_tf_create`
- [x] `src/bartorch/csrc/bartorch_ext.cpp`: `register_torch_prior` /
  `unregister_torch_prior` pybind11 bindings
- [x] `src/bartorch/csrc/CMakeLists.txt`: `torch_prior.cpp` added; unconditional
  `--wrap nlop_tf_create` linker flag (Linux + macOS)
- [x] `src/bartorch/tools/_commands.py`: `pics()` extended with `torch_prior` +
  `torch_prior_lambda`; `_bart_img_dims_from_kspace` helper
- [x] `tests/test_tools_pics.py`: Python-side tests for `_bart_img_dims_from_kspace`
  and `pics()` arg merging/validation (no ext required)
- [x] `AGENTS.md` §8 updated

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
