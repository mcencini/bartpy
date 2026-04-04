"""
bartorch.ops — Public Python API for BART operations.

Every function in this sub-package routes through
:func:`bartorch.core.graph.dispatch`, which selects the hot path (zero-copy
C++ extension) or the subprocess fallback automatically.  All ops accept and
return plain ``torch.Tensor`` objects (``dtype=torch.complex64``).

Available ops
-------------

FFT / num
    fft(input, flags, *, unitary, inverse, centered)
    ifft(input, flags, *, unitary, centered)

Simulation
    phantom(dims, *, kspace, d3, ptype, ncoils, device)

Calibration
    ecalib(kspace, *, calib_size, maps, threshold)
    caldir(kspace, *, calib_size)

Reconstruction
    pics(kspace, sens, *, lambda_, iter_, tol, wav, l1, l2)

Linear operators
    BartLinop — opaque handle with operator algebra:
        A(x)        — forward application
        A @ B       — composition  (returns BartLinop)
        A @ x       — forward application  (alias for A(x))
        A + B       — sum
        scalar * A  — scalar multiplication
        A.H         — adjoint
        A.N         — normal operator (A^H A)

Iterative algorithms  (Phase 5)
    conjgrad(op, b, *, maxiter, tol)
    ist(op, b, proxg, *, maxiter, step)
    fista(op, b, proxg, *, maxiter, step)
    irgnm(op, b, *, maxiter)
    chambolle_pock(op, prox_f, prox_g, *, maxiter, sigma, tau)

See agents.md for the full implementation roadmap.
"""

from bartorch.ops.fft import fft, ifft
from bartorch.ops.italgos import chambolle_pock, conjgrad, fista, irgnm, ist
from bartorch.ops.linops import BartLinop
from bartorch.ops.phantom import phantom
from bartorch.ops.pics import caldir, ecalib, pics

__all__ = [
    # Data ops
    "fft",
    "ifft",
    "phantom",
    # Calibration / reconstruction
    "ecalib",
    "caldir",
    "pics",
    # Linear operator handle
    "BartLinop",
    # Iterative algorithms
    "conjgrad",
    "ist",
    "fista",
    "irgnm",
    "chambolle_pock",
]
