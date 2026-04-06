# bartorch

**bartorch** is a PyTorch-native Python interface to the
[Berkeley Advanced Reconstruction Toolbox (BART)](https://mrirecon.github.io/bart/)
for MRI reconstruction.

## Key features

| Feature | Detail |
|---|---|
| **Zero-copy** | Tensor ↔ BART data exchange via shared `complex float*` — no serialisation overhead |
| **Hot-path DSL** | Pure-bartorch call chains stay entirely in C via `bart_command()`, skipping every Python↔C boundary between ops |
| **CUDA support** | GPU tensor device pointers are registered directly; BART's vptr system routes to CUDA kernels automatically |
| **Full PyTorch citizen** | All ops accept/return plain `torch.Tensor`; participates in autograd, `torch.compile`, and CUDA streams |
| **Thin wheel** | Links against PyTorch's bundled BLAS/FFT (MKL or OpenBLAS + cuFFT) — no duplicate libraries shipped |
| **Full CLI access** | Every BART command available via `bartorch.tools` with auto-generated, properly-typed Python functions parsed from BART source |

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU or CUDA)
- CMake ≥ 3.18
- C/C++ compiler (GCC ≥ 10 or Clang ≥ 13)
- *(optional)* CUDA toolkit ≥ 11.8 for GPU support

## Installation

```bash
# Clone with the BART submodule
git clone --recurse-submodules https://github.com/mcencini/bartpy.git
cd bartpy

# CPU-only
pip install -e .

# With CUDA
CMAKE_ARGS="-DUSE_CUDA=ON" pip install -e .
```

## Quick start

```python
import bartorch.tools as bt

# 2-D Shepp-Logan phantom + FFT — all in C, zero copies
phantom = bt.phantom([256, 256])           # shape (1, 256, 256), complex64
kspace  = bt.fft(phantom, axes=(-1, -2))  # 2-D FFT over last two axes

# Full MRI reconstruction pipeline
kspace = bt.phantom([256, 256], kspace=True, ncoils=8)
sens   = bt.ecalib(kspace, calib_size=24)
reco   = bt.pics(kspace, sens, R="W:7:0:0.005")  # wavelet regularisation

# Standard PyTorch ops work directly on the result
reco_abs = reco.abs()

# Operator algebra (BartLinop)
from bartorch.ops import BartLinop
A = BartLinop(ishape=(8, 256, 256), oshape=(8, 1, 256, 256))
AHA = A.N         # A^H A
AHb = A.H @ reco  # adjoint application (requires C++ extension)
```

## Axis convention

bartorch uses **C-order** (last index varies fastest), matching NumPy and
PyTorch conventions:

```
bartorch shape: (coils, phase2, phase1, read)   ← C-order
BART internal:  (read,  phase1, phase2, coils)  ← Fortran-order
```

The C++ bridge reverses the dimension array at the boundary.  The underlying
bytes are identical — zero copy.

Where BART expects an integer bitmask to select axes, bartorch tools accept
C-order **axis indices** instead (including negative indices).  This applies
to all commands that operate along selected dimensions (`fft`, `ifft`, `flip`,
`rss`, `avg`, `fftshift`, `fftmod`, `cdf97`, `wavelet`, `conv`, `hist`,
`mip`, `std`, `var`, …):

```python
bt.fft(x, axes=(-1, -2))    # last two axes — equivalent to BART flags=3
bt.fft(x, axes=0)            # first axis
bt.flip(x, axes=-1)          # reverse last axis
bt.rss(coil_imgs, axes=0)    # RSS over coil (first) axis
```

## Full CLI access

Every BART command is available in `bartorch.tools` as a properly-typed Python
function. `build_tools/gen_tools.py` parses the **BART C source files**
(`bart/src/`) — no system `bart` binary required — and generates
`bartorch/tools/_generated.py` with one function per tool.

Each generated function has:
- Named keyword arguments with Python type hints, inferred from BART's
  `OPT_*` macros (e.g. `a: bool = False`, `m: int | None = None`)
- A NumPy-style docstring generated from BART's help string and option descriptions
- `**extra_flags` to forward any unlisted flags directly

```python
import bartorch.tools as bt

# Named wrappers with type hints and docstrings
result = bt.pics(kspace, sens, R="T:7:0:0.01")       # total variation PICS
kspace = bt.nufft(traj, kspace_data, adjoint=True)    # adjoint NUFFT
rss    = bt.rss(kspace, axes=(-1, -2))                # root-sum-of-squares over last two axes

# Regenerate wrappers after updating the BART submodule
python build_tools/gen_tools.py
```

## Architecture

```
bartorch/
├── ops/           Internal types: BartLinop (operator algebra)
├── tools/         User-facing BART CLI wrappers
│   ├── _generated.py   Auto-generated wrappers for every BART command (143),
│   │                   parsed from BART source by build_tools/gen_tools.py
│   └── _commands.py    Pythonic overrides (fft/ifft, ecalib, caldir, pics,
│                       nlinv, moba, nufft) — delegate to generated functions
├── core/          Dispatch graph, BartContext, dtype normalisation
├── utils/         CFL read/write, axes_to_flags()
└── csrc/          PyTorch C++ extension (_bartorch_ext)
    ├── bartorch_ext.cpp     pybind11 module entry point
    ├── tensor_bridge.hpp    zero-copy CPU Tensor↔CFL
    ├── bart_ops.cpp         named op implementations
    └── cuda/
        └── cuda_bridge.hpp  zero-copy CUDA Tensor↔CFL
```

## BART submodule

BART is included as a git submodule (canonical upstream:
[codeberg.org/mrirecon/bart](https://codeberg.org/mrirecon/bart),
GitHub mirror: [github.com/mrirecon/bart](https://github.com/mrirecon/bart)).

```bash
git submodule update --init --recursive
```

## Examples

The `examples/` directory at the repository root contains Jupyter notebooks
demonstrating the bartorch API:

| Notebook | Description |
|---|---|
| [`01_high_level_tools.ipynb`](examples/01_high_level_tools.ipynb) | High-level BART tools: phantom, FFT, ESPIRiT, PICS |
| [`02_library_internals.ipynb`](examples/02_library_internals.ipynb) | BartContext hot path, `@bart_op` dtype handling, linop algebra |

The same notebooks are rendered in the
[online documentation](https://mcencini.github.io/bartpy/examples/index.html)
as a gallery via `nbsphinx`.

## License

MIT — see [LICENSE](LICENSE).  
BART is distributed under its own BSD license; see `bart/LICENSE`.

---

## Roadmap

### Phase 1 — C++ extension core (current)

- Implement `tensor_bridge.hpp`: zero-copy `torch.Tensor` ↔ CFL with axis reversal
- Implement `_bartorch_ext.run()`: build argv, call `bart_command()`, return tensor
- CI: build extension on Ubuntu with PyTorch CPU wheel

### Phase 2 — Cartesian encoding & basic linops

| Operator | BART source | C API |
|---|---|---|
| FFT as linop | `linops/someops.h` | `linop_fft_create()`, `linop_fftc_create()` |
| Gradient / TV | `linops/grad.h` | `linop_grad_create()` |
| Diagonal (coil sens.) | `linops/someops.h` | `linop_cdiag_create()`, `linop_rdiag_create()` |
| NUFFT encoding | `noncart/nufft.h` | `nufft_create()`, `nufft_create2()` |

### Phase 3 — Full MRI encoding & advanced regularisation

| Operator | BART source | C API |
|---|---|---|
| Cartesian SENSE | `sense/model.h` | `sense_init()`, `maps_create()`, `maps2_create()` |
| Non-Cartesian SENSE | `sense/modelnc.h` | `sense_nc_init()` |
| Low-rank / Casorati | `linops/casorati.h` | `linop_casorati_create()` |
| Low-rank thresholding | `lowrank/lrthresh.h` | `lrthresh_create()` |
| Wavelet | `linops/waveop.h` | `linop_wavelet_create()` |
| Reshape / Slice / Permute | `linops/someops.h` | various |

### Phase 4 — Iterative algorithms & nonlinear ops

| Algorithm | BART source | Notes |
|---|---|---|
| Conjugate gradient | `iter/italgos.c` | `iter_conjgrad()` |
| IST / FISTA | `iter/iter.c` | `iter2_ist()`, `iter2_fista()` |
| IRGNM | `iter/italgos.c` | `iter2_irgnm()` |
| Chambolle-Pock | `iter/italgos.c` | `iter2_chambolle_pock()` |
| Nonlinear inversion | `noir/model.h` | `struct noir_s*` |
| Model-based recon | `moba/moba.h` | T1/T2/T2*/diffusion |


