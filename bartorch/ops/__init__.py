"""
bartorch.ops — Public Python API for BART operations.

Every function in this sub-package routes through
:func:`bartorch.core.graph.dispatch`, which selects the hot path (zero-copy
C++ extension) or the FIFO fallback automatically.

Available ops (mirrors the old bartpy SWIG interface):

  FFT / num
    fft(input, flags, *, unitary, inverse, centered)
    ifft(input, flags, *, unitary, centered)

  Simulation / simu
    phantom(dims, *, kspace, d3, ptype)

  Calibration
    ecalib(kspace, *, calib_size, maps)
    caldir(kspace, *, calib_size)

  Reconstruction
    pics(kspace, sens, *, lambda_, iter, tol)

  Linear operators (returned as opaque handles, not tensors)
    identity(dims)
    diag(dims, *, diag, flags)
    fft_linop(dims, flags)
    chain(op1, op2)
    plus(op1, op2)
    stack(op1, op2)
    forward(op, dest_dims, src)
    adjoint(op, dest_dims, src)
    normal(op, dest_dims, src)
    pseudo_inv(op, lambda_, dest_dims, src)

  Iterative algorithms
    conjgrad(op, b, *, maxiter, tol)
    ist(op, b, proxg, *, maxiter, step)
    fista(op, b, proxg, *, maxiter, step)
    irgnm(op, b, *, maxiter)
    chambolle_pock(op, prox_f, prox_g, *, maxiter)

See agents.md for the full implementation roadmap.
"""

from bartorch.ops.fft import fft, ifft
from bartorch.ops.italgos import chambolle_pock, conjgrad, fista, irgnm, ist
from bartorch.ops.linops import (
    adjoint,
    chain,
    diag,
    fft_linop,
    forward,
    identity,
    normal,
    plus,
    pseudo_inv,
    stack,
)
from bartorch.ops.phantom import phantom
from bartorch.ops.pics import caldir, ecalib, pics

__all__ = [
    "fft",
    "ifft",
    "phantom",
    "identity",
    "diag",
    "fft_linop",
    "chain",
    "plus",
    "stack",
    "forward",
    "adjoint",
    "normal",
    "pseudo_inv",
    "ecalib",
    "caldir",
    "pics",
    "conjgrad",
    "ist",
    "fista",
    "irgnm",
    "chambolle_pock",
]
