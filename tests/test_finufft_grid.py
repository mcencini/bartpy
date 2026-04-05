"""tests/test_finufft_grid.py — tests for the FINUFFT-backed grid2/grid2H.

These tests are skipped when the C++ extension has not been built
(BARTORCH_SKIP_EXT=1 or the module is absent), which is the normal state
for the pure-Python CI job.

When the extension IS available (i.e. compiled with BARTORCH_USE_FINUFFT=ON
and noncart/grid.c in BART_SOURCES), the tests exercise the FINUFFT-backed
gridding through the BART embed-API layer.

Test plan
─────────
1. ``test_import`` — sanity-check that the extension is importable.
2. ``test_finufft_grid_module`` — verify the ``__wrap_grid2`` / ``__wrap_grid2H``
   symbols are present in the shared object (confirming the --wrap was applied).
3. Numerical tests (requires fully wired extension + Phase-1 run() impl):
   - ``test_adjointness_2d`` — radial 2-D trajectory, ⟨grid2H(x), y⟩ ≈ ⟨x, grid2(y)⟩
   - ``test_adjointness_3d`` — 3-D radial trajectory
   All are skipped if ``bartorch._bartorch_ext.run`` is not yet implemented.
"""

from __future__ import annotations

import importlib
import math
import os
import pytest


# ---------------------------------------------------------------------------
# Helper: skip if extension unavailable
# ---------------------------------------------------------------------------
def _ext_available() -> bool:
    if os.environ.get("BARTORCH_SKIP_EXT", "0") == "1":
        return False
    try:
        importlib.import_module("bartorch._bartorch_ext")
        return True
    except ImportError:
        return False


skip_no_ext = pytest.mark.skipif(
    not _ext_available(),
    reason="C++ extension not built (BARTORCH_SKIP_EXT=1 or absent)",
)


# ---------------------------------------------------------------------------
# Test 1: import
# ---------------------------------------------------------------------------
@skip_no_ext
def test_import():
    """Extension can be imported without error."""
    import bartorch._bartorch_ext as ext  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2: __wrap symbols present in the shared object
# ---------------------------------------------------------------------------
@skip_no_ext
def test_finufft_wrap_symbols():
    """__wrap_grid2 and __wrap_grid2H are present in _bartorch_ext.so.

    Uses ctypes to inspect the symbol table so we don't need a Python
    binding for the C-level functions.
    """
    import ctypes
    import bartorch._bartorch_ext as ext  # noqa: F401
    import bartorch

    so_path = bartorch.__file__.replace("__init__.py", "_bartorch_ext.so")
    # On some platforms the suffix differs; find the .so by glob.
    import glob, pathlib
    matches = glob.glob(str(pathlib.Path(bartorch.__file__).parent / "_bartorch_ext*.so"))
    if not matches:
        pytest.skip("Cannot locate _bartorch_ext.so to inspect symbols")

    lib = ctypes.CDLL(matches[0])
    assert hasattr(lib, "__wrap_grid2"),  "__wrap_grid2 symbol missing from _bartorch_ext.so"
    assert hasattr(lib, "__wrap_grid2H"), "__wrap_grid2H symbol missing from _bartorch_ext.so"


# ---------------------------------------------------------------------------
# Numerical tests — require Phase-1 run() to be implemented
# ---------------------------------------------------------------------------
def _run_available() -> bool:
    if not _ext_available():
        return False
    try:
        import bartorch._bartorch_ext as ext
        # run() currently raises NotImplementedError until Phase-1 is complete
        import inspect
        src = inspect.getsource(ext.run)
        return "NotImplementedError" not in src and "not yet implemented" not in src
    except Exception:
        return False


skip_no_run = pytest.mark.skipif(
    not _run_available(),
    reason="bartorch.run() not yet implemented (Phase-1 pending)",
)


@skip_no_run
def test_adjointness_2d():
    """Adjointness of grid2 / grid2H for a 2-D radial trajectory.

    Checks:  |⟨grid2H(x), y⟩ − ⟨x, grid2(y)⟩| / (‖x‖·‖y‖) < 1e-4
    """
    import torch
    import bartorch as bt

    torch.manual_seed(0)

    # Image dimensions (non-oversampled)
    Nx, Ny = 64, 64
    # Number of radial spokes and readout points
    n_spokes, n_ro = 128, 64
    M = n_spokes * n_ro  # total k-space samples

    # Random radial trajectory in [-0.5, 0.5] × Nx (BART units)
    angles = torch.linspace(0, math.pi, n_spokes)
    r      = torch.linspace(-n_ro / 2, n_ro / 2, n_ro)
    traj_x = torch.outer(torch.cos(angles), r).reshape(-1)
    traj_y = torch.outer(torch.sin(angles), r).reshape(-1)
    traj_z = torch.zeros(M)
    # BART trajectory: complex float, shape [3, n_ro, n_spokes] → [3, M, 1]
    # (real part = position, imaginary part = 0)
    traj = torch.stack([traj_x, traj_y, traj_z]).reshape(3, n_ro, n_spokes, 1)
    traj = traj.to(torch.complex64)

    # Random grid (oversampled image) and k-space data
    os = 2
    grid = torch.randn(Nx * os, Ny * os, 1, 1, dtype=torch.complex64)
    ksp  = torch.randn(1, n_ro, n_spokes, 1, dtype=torch.complex64)

    # grid2H: grid → kspace
    Ag_ksp = bt.tools.gridH(traj, grid)          # placeholder names; adjust for actual API
    # grid2: kspace → grid
    AH_ksp_g = bt.tools.grid(traj, ksp)

    # Adjointness: ⟨Ag_ksp, ksp⟩ ≈ ⟨grid, AH_ksp_g⟩
    lhs = (Ag_ksp * ksp.conj()).sum().real
    rhs = (AH_ksp_g * grid.conj()).sum().real
    rel_err = abs(float(lhs - rhs)) / (float(Ag_ksp.norm()) * float(ksp.norm()))
    assert rel_err < 1e-4, f"Adjointness error {rel_err:.2e} exceeds 1e-4"


@skip_no_run
def test_adjointness_3d():
    """Adjointness of grid2 / grid2H for a 3-D Koosh-ball trajectory."""
    import torch
    import bartorch as bt

    torch.manual_seed(1)
    # Coarser dims to keep the test fast
    N = 16
    M = 512

    # Random trajectory on the unit sphere, scaled to [-N/2, N/2]
    pts = torch.randn(3, M, dtype=torch.float32)
    pts = pts / pts.norm(dim=0, keepdim=True) * (N / 2) * torch.rand(M)
    traj = torch.zeros(3, M, 1, 1, dtype=torch.complex64)
    traj[:, :, 0, 0] = pts.to(torch.complex64)

    os = 2
    grid = torch.randn(N * os, N * os, N * os, 1, dtype=torch.complex64)
    ksp  = torch.randn(1, M, 1, 1, dtype=torch.complex64)

    Ag_ksp   = bt.tools.gridH(traj, grid)
    AH_ksp_g = bt.tools.grid(traj, ksp)

    lhs = (Ag_ksp * ksp.conj()).sum().real
    rhs = (AH_ksp_g * grid.conj()).sum().real
    rel_err = abs(float(lhs - rhs)) / (float(Ag_ksp.norm()) * float(ksp.norm()))
    assert rel_err < 1e-4, f"3-D adjointness error {rel_err:.2e} exceeds 1e-4"
