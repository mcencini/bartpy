"""Tests for the torch_prior plug-and-play path in bartorch.tools.pics.

These tests cover:

* ``_bart_img_dims_from_kspace`` — converts a kspace tensor shape to the
  BART Fortran-order ``img_dims`` array (length 16).
* ``pics(..., torch_prior=fn)`` argument validation and R-flag merging logic.
  The mocking-based tests exercise the Python layer independently of the
  C++ dispatch stub.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from bartorch.tools._commands import _bart_img_dims_from_kspace, pics

__all__: list[str] = []

_BART_DIMS = 16
_COIL_DIM = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(nc=2, ny=8, nx=8):
    kspace = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    sens = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    return kspace, sens


# ---------------------------------------------------------------------------
# _bart_img_dims_from_kspace
# ---------------------------------------------------------------------------


def test_img_dims_2d_4d_tensor():
    """2-D case: (nc, 1, ny, nx) → img_dims with nx,ny spatial, coil=1."""
    nc, ny, nx = 8, 64, 96
    ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)

    assert len(dims) == _BART_DIMS
    # Fortran reversed from C-order [nc, 1, ny, nx] → [nx, ny, 1, nc, ...]
    assert dims[0] == nx
    assert dims[1] == ny
    assert dims[2] == 1
    # Coil dim (Fortran index 3) must be zeroed to 1 for img_dims
    assert dims[_COIL_DIM] == 1
    # All remaining dims are 1
    assert all(d == 1 for d in dims[4:])


def test_img_dims_3d_tensor():
    """3-D case: (nc, nz, ny, nx) → img_dims with full spatial, coil=1."""
    nc, nz, ny, nx = 4, 16, 32, 48
    ksp = torch.zeros(nc, nz, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)

    assert len(dims) == _BART_DIMS
    assert dims[0] == nx
    assert dims[1] == ny
    assert dims[2] == nz
    assert dims[_COIL_DIM] == 1  # coil zeroed
    assert all(d == 1 for d in dims[4:])


def test_img_dims_returns_list_of_ints():
    ksp = torch.zeros(2, 1, 8, 8, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    assert isinstance(dims, list)
    assert all(isinstance(d, int) for d in dims)


def test_img_dims_length_always_16():
    for shape in [(2, 1, 4, 4), (4, 8, 16, 16), (1, 1, 32, 32)]:
        ksp = torch.zeros(*shape, dtype=torch.complex64)
        assert len(_bart_img_dims_from_kspace(ksp)) == _BART_DIMS


def test_img_dims_coil_dim_always_1():
    """Coil dim (Fortran index 3) must be 1 regardless of nc."""
    for nc in (1, 4, 16, 32):
        ksp = torch.zeros(nc, 1, 8, 8, dtype=torch.complex64)
        dims = _bart_img_dims_from_kspace(ksp)
        assert dims[_COIL_DIM] == 1, f"coil dim not 1 for nc={nc}: {dims}"


def test_img_dims_single_coil():
    nc, ny, nx = 1, 16, 16
    ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    assert dims[0] == nx
    assert dims[1] == ny
    assert dims[_COIL_DIM] == 1


def test_img_dims_non_square_spatial():
    nc, ny, nx = 2, 32, 64
    ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    assert dims[0] == nx
    assert dims[1] == ny


def test_img_dims_3d_coil_zeroed():
    """3-D kspace: original nc must NOT appear at Fortran dim 3 in img_dims."""
    nc, nz, ny, nx = 6, 10, 20, 30
    ksp = torch.zeros(nc, nz, ny, nx, dtype=torch.complex64)
    dims = _bart_img_dims_from_kspace(ksp)
    # Before zeroing, ksp_dims[3] == nc (6). After zeroing it must be 1.
    assert dims[_COIL_DIM] == 1
    # The original nc value must not appear at the coil position.
    assert dims[_COIL_DIM] != nc


# ---------------------------------------------------------------------------
# pics() — torch_prior argument validation
# ---------------------------------------------------------------------------


def test_torch_prior_lambda_default_is_float():
    """torch_prior_lambda has a sane numeric default (1.0)."""
    sig = inspect.signature(pics)
    default = sig.parameters["torch_prior_lambda"].default
    assert isinstance(default, float)
    assert default == 1.0


def test_tf_reg_string_format():
    """Verify the -R TF:{bartorch://...}:lambda string is well-formed."""
    import bartorch.tools._commands as cmds

    captured = {}

    class FakeExt:
        def register_torch_prior(self, name, fn, dims):
            captured["name"] = name
            captured["dims"] = dims

        def unregister_torch_prior(self, name):
            pass

    original_get_ext = None
    try:
        import bartorch.core.graph as graph_mod

        original_get_ext = graph_mod._get_ext
    except Exception:
        pytest.skip("bartorch.core.graph not importable")

    call_args = {}

    def fake_generated_pics(kspace, sens, **kwargs):
        call_args.update(kwargs)
        raise RuntimeError("sentinel")

    original_generated_pics = cmds._generated.pics
    try:
        graph_mod._get_ext = lambda: FakeExt()
        cmds._generated.pics = fake_generated_pics
        kspace, sens = _make_inputs()
        with pytest.raises(RuntimeError, match="sentinel"):
            pics(kspace, sens, torch_prior=lambda x: x, torch_prior_lambda=0.05)
    finally:
        graph_mod._get_ext = original_get_ext
        cmds._generated.pics = original_generated_pics

    # Verify the R flag contains the bartorch:// sentinel and lambda.
    r_val = call_args.get("R", "")
    r_str = r_val if isinstance(r_val, str) else r_val[0]
    assert r_str.startswith("TF:{bartorch://"), repr(r_str)
    assert r_str.endswith("}:0.05"), repr(r_str)
    # Name must be a non-empty string starting with _btprior_
    assert captured["name"].startswith("_btprior_")
    # dims must have length 16 and coil dim 3 == 1
    assert len(captured["dims"]) == _BART_DIMS
    assert captured["dims"][_COIL_DIM] == 1


def test_torch_prior_r_merging_with_existing_r_string():
    """torch_prior appended after an existing R string -> list of two."""
    import bartorch.core.graph as graph_mod
    import bartorch.tools._commands as cmds

    class FakeExt:
        def register_torch_prior(self, *a):
            pass

        def unregister_torch_prior(self, *a):
            pass

    call_args = {}

    def fake_generated_pics(kspace, sens, **kwargs):
        call_args.update(kwargs)
        raise RuntimeError("sentinel")

    original_get_ext = graph_mod._get_ext
    original_pics = cmds._generated.pics
    try:
        graph_mod._get_ext = lambda: FakeExt()
        cmds._generated.pics = fake_generated_pics
        kspace, sens = _make_inputs()
        with pytest.raises(RuntimeError, match="sentinel"):
            pics(
                kspace,
                sens,
                R="W:7:0:0.005",
                torch_prior=lambda x: x,
                torch_prior_lambda=1.0,
            )
    finally:
        graph_mod._get_ext = original_get_ext
        cmds._generated.pics = original_pics

    r_val = call_args.get("R")
    assert isinstance(r_val, list), f"Expected list, got {type(r_val)}"
    assert len(r_val) == 2
    assert r_val[0] == "W:7:0:0.005"
    assert r_val[1].startswith("TF:{bartorch://")


def test_torch_prior_r_merging_with_existing_r_list():
    """torch_prior appended after an existing R list -> list of three."""
    import bartorch.core.graph as graph_mod
    import bartorch.tools._commands as cmds

    class FakeExt:
        def register_torch_prior(self, *a):
            pass

        def unregister_torch_prior(self, *a):
            pass

    call_args = {}

    def fake_generated_pics(kspace, sens, **kwargs):
        call_args.update(kwargs)
        raise RuntimeError("sentinel")

    original_get_ext = graph_mod._get_ext
    original_pics = cmds._generated.pics
    try:
        graph_mod._get_ext = lambda: FakeExt()
        cmds._generated.pics = fake_generated_pics
        kspace, sens = _make_inputs()
        with pytest.raises(RuntimeError, match="sentinel"):
            pics(
                kspace,
                sens,
                R=["T:7:0:0.002", "W:7:0:0.005"],
                torch_prior=lambda x: x,
                torch_prior_lambda=1.0,
            )
    finally:
        graph_mod._get_ext = original_get_ext
        cmds._generated.pics = original_pics

    r_val = call_args.get("R")
    assert isinstance(r_val, list)
    assert len(r_val) == 3
    assert r_val[-1].startswith("TF:{bartorch://")


def test_unique_names_across_calls():
    """Each pics() call registers a different name (UUID-based)."""
    import bartorch.core.graph as graph_mod
    import bartorch.tools._commands as cmds

    names = []

    class FakeExt:
        def register_torch_prior(self, name, *a):
            names.append(name)

        def unregister_torch_prior(self, *a):
            pass

    def fake_generated_pics(kspace, sens, **kwargs):
        raise RuntimeError("sentinel")

    original_get_ext = graph_mod._get_ext
    original_pics = cmds._generated.pics
    try:
        graph_mod._get_ext = lambda: FakeExt()
        cmds._generated.pics = fake_generated_pics
        kspace, sens = _make_inputs()
        for _ in range(3):
            with pytest.raises(RuntimeError, match="sentinel"):
                pics(kspace, sens, torch_prior=lambda x: x, torch_prior_lambda=1.0)
    finally:
        graph_mod._get_ext = original_get_ext
        cmds._generated.pics = original_pics

    assert len(names) == 3
    assert len(set(names)) == 3, "Names must be unique across calls"
