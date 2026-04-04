# bartorch

**bartorch** is a PyTorch-native Python interface to the
[Berkeley Advanced Reconstruction Toolbox (BART)](https://mrirecon.github.io/bart/)
for MRI reconstruction.

## Key features

| Feature | Detail |
|---|---|
| **Zero-copy** | Tensor ↔ BART data exchange via shared `complex float*` — no serialisation overhead |
| **Hot-path DSL** | Pure-bartorch call chains stay entirely in C via `bart_command()`, skipping every Python↔C boundary between ops |
| **FIFO fallback** | Tools not yet in the C++ layer run as subprocesses fed via named FIFOs in `/dev/shm` — no temporary files on disk |
| **CUDA support** | GPU `BartTensor` device pointers are registered directly; BART's vptr system routes to CUDA kernels automatically |
| **Full PyTorch citizen** | `BartTensor` subclasses `torch.Tensor`; participates in autograd, `torch.compile`, and CUDA streams |
| **Thin wheel** | Links against PyTorch's bundled BLAS/FFT (MKL or OpenBLAS + cuFFT) — no duplicate libraries shipped |

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

# Pure-Python mode (FIFO fallback only, no C++ extension)
BARTORCH_SKIP_EXT=1 pip install -e .
```

## Quick start

```python
import bartorch as bt

# Hot path: phantom generation + FFT — all in C, zero copies
with bt.core.bart_context():
    phantom = bt.ops.phantom([256, 256])          # BartTensor, complex64
    kspace  = bt.ops.fft(phantom, flags=3)        # zero-copy in-process

# BartTensor is a fully usable torch.Tensor
print(kspace.shape, kspace.dtype)
# → torch.Size([256, 256]) torch.complex64

# Full MRI reconstruction pipeline
kspace = bt.ops.phantom([256, 256], kspace=True, ncoils=8)
sens   = bt.ops.ecalib(kspace, calib_size=24)
reco   = bt.ops.pics(kspace, sens, lambda_=0.01, wav=True)

# Standard PyTorch ops work directly on the result
reco_abs = reco.abs()
```

## Architecture

See [`agents.md`](agents.md) for the full technical design, implementation
roadmap, and BART source-level details.

```
bartorch/
├── core/          BartTensor, BartContext, dispatch graph
├── ops/           Python API: fft, phantom, ecalib, pics, linops, italgos
├── pipe/          FIFO subprocess fallback (no-disk)
├── tools/         Auto-generated wrappers for all 100+ BART CLI tools
├── utils/         NumPy-compatible CFL read/write
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

## License

BSD 3-Clause — see [LICENSE](LICENSE).  
BART is distributed under its own BSD license; see `bart/LICENSE`.
