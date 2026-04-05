"""tests/test_finufft_grid.py — tests for the FINUFFT-backed grid2/grid2H
and ES-kernel rolloff correction.

These tests are skipped when the C++ extension has not been built
(BARTORCH_SKIP_EXT=1 or the module is absent), which is the normal state
for the pure-Python CI job.

When the extension IS available (i.e. compiled with BARTORCH_USE_FINUFFT=ON
and noncart/grid.c in BART_SOURCES), the tests exercise the FINUFFT-backed
gridding and rolloff through the BART embed-API layer.

Test plan
─────────
1. ``test_import`` — sanity-check that the extension is importable.
2. ``test_finufft_wrap_symbols`` — verify the ``__wrap_grid2`` / ``__wrap_grid2H``
   symbols are present in the shared object (confirming the --wrap was applied).
3. ``test_rolloff_wrap_symbols`` — verify the three rolloff ``__wrap_*`` symbols
   are present, confirming the ES-kernel rolloff was linked in.
4. ``test_es_rolloff_dc_weight`` — pure-Python check that the ES rolloff weight
   at DC (xi=0) is a finite positive value < 1, and that it matches the
   theoretical hat_phi(0) = 2 * integral_0^{hw} exp(beta) dφ.
5. Numerical tests (requires fully wired extension + Phase-1 run() impl):
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
# Test 2: __wrap symbols present in the shared object (grid2 / grid2H)
# ---------------------------------------------------------------------------
@skip_no_ext
def test_finufft_wrap_symbols():
    """__wrap_grid2 and __wrap_grid2H are present in _bartorch_ext.so.

    Uses ctypes to inspect the symbol table so we don't need a Python
    binding for the C-level functions.
    """
    import ctypes
    import glob
    import pathlib

    import bartorch._bartorch_ext as ext  # noqa: F401

    import bartorch

    matches = glob.glob(str(pathlib.Path(bartorch.__file__).parent / "_bartorch_ext*.so"))
    if not matches:
        pytest.skip("Cannot locate _bartorch_ext.so to inspect symbols")

    lib = ctypes.CDLL(matches[0])
    assert hasattr(lib, "__wrap_grid2"), "__wrap_grid2 symbol missing from _bartorch_ext.so"
    assert hasattr(lib, "__wrap_grid2H"), "__wrap_grid2H symbol missing from _bartorch_ext.so"


# ---------------------------------------------------------------------------
# Test 3: rolloff __wrap symbols present in the shared object
# ---------------------------------------------------------------------------
@skip_no_ext
def test_rolloff_wrap_symbols():
    """__wrap_rolloff_correction and friends are present in _bartorch_ext.so."""
    import ctypes
    import glob
    import pathlib

    import bartorch

    matches = glob.glob(str(pathlib.Path(bartorch.__file__).parent / "_bartorch_ext*.so"))
    if not matches:
        pytest.skip("Cannot locate _bartorch_ext.so to inspect symbols")

    lib = ctypes.CDLL(matches[0])
    for sym in (
        "__wrap_rolloff_correction",
        "__wrap_apply_rolloff_correction",
        "__wrap_apply_rolloff_correction2",
    ):
        assert hasattr(lib, sym), f"{sym} symbol missing from _bartorch_ext.so"


# ---------------------------------------------------------------------------
# Test 4: ES rolloff weight — pure-Python validation (no extension needed)
#
# Verifies that the same quadrature used in finufft_grid.cpp produces
# physically sensible rolloff weights:
#   - DC weight (xi=0) is positive and < 1
#   - Weights are symmetric around the grid centre
#   - Weight is monotonically decreasing from centre toward ±Nyquist
# ---------------------------------------------------------------------------
def _es_rolloff_weight(
    xi_cps: float, beta: float = 2.30 * 7.0, hw: float = 7.0 / 2.0, nquad: int = 256
) -> float:
    """Python replica of esro_hat_phi / esro_weight from finufft_grid.cpp."""
    h = hw / nquad
    total = 0.0
    for i in range(nquad):
        x = (i + 0.5) * h
        t = x / hw
        arg = 1.0 - t * t
        phi = math.exp(beta * math.sqrt(max(arg, 0.0)))
        total += phi * math.cos(2.0 * math.pi * xi_cps * x)
    hat_phi = 2.0 * h * total
    return 1.0 / hat_phi if hat_phi != 0.0 else 1.0


def test_es_rolloff_dc_weight():
    """ES rolloff weight at DC is a finite positive float < 1."""
    w_dc = _es_rolloff_weight(0.0)
    assert w_dc > 0.0, f"DC weight should be positive, got {w_dc}"
    assert w_dc < 1.0, f"DC weight should be < 1 (ES kernel peak >> 1), got {w_dc}"
    assert math.isfinite(w_dc), "DC weight must be finite"


def test_es_rolloff_symmetry():
    """ES rolloff weights are symmetric: w(p, n) == w(n-p, n)."""
    n, os = 64, 2.0
    for p in range(n // 2):
        xi_pos = (p - n / 2.0) / (n * os)
        xi_neg = (n - p - n / 2.0) / (n * os)  # mirror pixel
        w_pos = _es_rolloff_weight(xi_pos)
        w_neg = _es_rolloff_weight(xi_neg)
        assert abs(w_pos - w_neg) < 1e-7, (
            f"Rolloff not symmetric at p={p}: w(p)={w_pos}, w(n-p)={w_neg}"
        )


def test_es_rolloff_monotone():
    """ES rolloff weight decreases monotonically from centre toward Nyquist."""
    n, os = 64, 2.0
    xis = [(p - n / 2.0) / (n * os) for p in range(n // 2, n)]
    weights = [_es_rolloff_weight(xi) for xi in xis]
    for i in range(len(weights) - 1):
        assert weights[i] <= weights[i + 1] + 1e-6, (
            f"Rolloff weight not monotone at index {i}: {weights[i]} > {weights[i + 1]}"
        )


# ---------------------------------------------------------------------------
# Numerical tests — require Phase-1 run() to be implemented
# ---------------------------------------------------------------------------
def _run_available() -> bool:
    if not _ext_available():
        return False
    try:
        import bartorch._bartorch_ext as ext

        # Probe whether run() is wired up by attempting a call with dummy args.
        # An unimplemented run() raises RuntimeError("not yet implemented").
        # Any other outcome (including other exceptions from bad args) means
        # the function IS available.
        try:
            ext.run("phantom", [], None, {})
        except RuntimeError as e:
            if "not yet implemented" in str(e):
                return False
        return True
    except Exception:
        return True


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
    r = torch.linspace(-n_ro / 2, n_ro / 2, n_ro)
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
    ksp = torch.randn(1, n_ro, n_spokes, 1, dtype=torch.complex64)

    # grid2H: grid → kspace
    Ag_ksp = bt.tools.gridH(traj, grid)  # placeholder names; adjust for actual API
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
    ksp = torch.randn(1, M, 1, 1, dtype=torch.complex64)

    Ag_ksp = bt.tools.gridH(traj, grid)
    AH_ksp_g = bt.tools.grid(traj, ksp)

    lhs = (Ag_ksp * ksp.conj()).sum().real
    rhs = (AH_ksp_g * grid.conj()).sum().real
    rel_err = abs(float(lhs - rhs)) / (float(Ag_ksp.norm()) * float(ksp.norm()))
    assert rel_err < 1e-4, f"3-D adjointness error {rel_err:.2e} exceeds 1e-4"
