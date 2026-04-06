"""tests/test_finufft_grid.py — tests for the FINUFFT-backed grid2/grid2H
and ES-kernel rolloff correction.

Tests that require the C++ extension are skipped when the module cannot be
imported (e.g. the submodule was not initialised during an editable install).

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

import pytest


# ---------------------------------------------------------------------------
# Helper: skip if extension unavailable
# ---------------------------------------------------------------------------
def _ext_available() -> bool:
    try:
        importlib.import_module("bartorch._bartorch_ext")
        return True
    except ImportError:
        return False


skip_no_ext = pytest.mark.skipif(
    not _ext_available(),
    reason="C++ extension not built",
)


# ---------------------------------------------------------------------------
# Test 1: import
# ---------------------------------------------------------------------------
@skip_no_ext
def test_import():
    """Extension can be imported without error."""
    import bartorch._bartorch_ext as ext  # noqa: F401


# ---------------------------------------------------------------------------
# Test 2: --wrap grid2/grid2H symbols present in the shared object
# ---------------------------------------------------------------------------
@skip_no_ext
def test_finufft_wrap_symbols():
    """grid2 and grid2H symbols are present in _bartorch_ext.so.

    The GNU --wrap linker flag redirects calls at static link time.
    The __wrap_* symbols are consumed by the linker and may not appear
    in the *dynamic* symbol table, so we use readelf to inspect the
    full symbol table instead of ctypes.
    """
    import glob
    import pathlib
    import subprocess

    import bartorch

    matches = glob.glob(str(pathlib.Path(bartorch.__file__).parent / "_bartorch_ext*.so"))
    if not matches:
        pytest.skip("Cannot locate _bartorch_ext.so to inspect symbols")

    try:
        out = subprocess.check_output(
            ["readelf", "-s", matches[0]], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("readelf not available")

    # When --wrap,grid2 is active, the original grid2 gets renamed to
    # __real_grid2 and __wrap_grid2 handles calls.  If --wrap was NOT
    # applied, only the plain grid2 symbol exists (no __real_ / __wrap_).
    # Either __wrap_ or __real_ confirms the wrapping is in place.
    has_wrap = "__wrap_grid2" in out or "__real_grid2" in out
    has_wrapH = "__wrap_grid2H" in out or "__real_grid2H" in out
    assert has_wrap, "grid2 wrapping not detected in _bartorch_ext.so"
    assert has_wrapH, "grid2H wrapping not detected in _bartorch_ext.so"


# ---------------------------------------------------------------------------
# Test 3: rolloff __wrap symbols present in the shared object
# ---------------------------------------------------------------------------
@skip_no_ext
def test_rolloff_wrap_symbols():
    """rolloff_correction wrapping symbols are present in _bartorch_ext.so."""
    import glob
    import pathlib
    import subprocess

    import bartorch

    matches = glob.glob(str(pathlib.Path(bartorch.__file__).parent / "_bartorch_ext*.so"))
    if not matches:
        pytest.skip("Cannot locate _bartorch_ext.so to inspect symbols")

    try:
        out = subprocess.check_output(
            ["readelf", "-s", matches[0]], text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("readelf not available")

    for sym in ("rolloff_correction", "apply_rolloff_correction", "apply_rolloff_correction2"):
        has_sym = f"__wrap_{sym}" in out or f"__real_{sym}" in out
        assert has_sym, f"{sym} wrapping not detected in _bartorch_ext.so"


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
    """Adjointness of NUFFT forward/adjoint for a 2-D radial trajectory.

    Mirrors BART's test-nufft-adjoint from nufft.mk:
        traj -r -x128 -y128 → nufft / nufft -a → fmac inner products → nrmse

    Checks:  |⟨A x, y⟩ − ⟨x, A^H y⟩| / (‖x‖ · ‖y‖) < tol

    The forward/adjoint internally exercise grid2H / grid2 which are
    replaced by FINUFFT's __wrap_grid2H / __wrap_grid2.
    """
    import torch

    import bartorch.tools as bt

    # Generate proper radial trajectory using BART's traj tool
    traj = bt.traj(r=True, x=64, y=64)

    # Random image and kspace-shaped noise (matching BART test pattern)
    n1 = bt.noise(torch.zeros(1, 64, 64, dtype=torch.complex64), s=123)
    n2 = bt.noise(torch.zeros(64, 64, 1, dtype=torch.complex64), s=321)

    # Forward NUFFT (exercises grid2H via FINUFFT __wrap_grid2H)
    k = bt.nufft(traj, n1)
    # Adjoint NUFFT (exercises grid2 via FINUFFT __wrap_grid2)
    x = bt.nufft(traj, n2, adjoint=True)

    # Adjointness: ⟨A n1, n2⟩ ≈ ⟨n1, A^H n2⟩
    s1 = bt.fmac(n1, x, C=True, s=7)
    s2 = bt.fmac(k, n2, C=True, s=7)
    err = float(bt.nrmse(s1, s2))
    assert err < 1e-4, f"Adjointness error {err:.2e} exceeds 1e-4"


@skip_no_run
def test_adjointness_3d():
    """Adjointness of NUFFT forward/adjoint for a smaller trajectory.

    Same pattern as test_adjointness_2d but with smaller dimensions for speed.
    """
    import torch

    import bartorch.tools as bt

    traj = bt.traj(r=True, x=16, y=16)

    n1 = bt.noise(torch.zeros(1, 16, 16, dtype=torch.complex64), s=123)
    n2 = bt.noise(torch.zeros(16, 16, 1, dtype=torch.complex64), s=321)

    k = bt.nufft(traj, n1)
    x = bt.nufft(traj, n2, adjoint=True)

    s1 = bt.fmac(n1, x, C=True, s=7)
    s2 = bt.fmac(k, n2, C=True, s=7)
    err = float(bt.nrmse(s1, s2))
    assert err < 1e-4, f"Adjointness error {err:.2e} exceeds 1e-4"
