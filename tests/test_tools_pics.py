"""Tests for the torch_prior plug-and-play path in bartorch.tools.pics.

These tests cover:

* ``_bart_img_dims_from_kspace`` — converts a kspace tensor shape to the
  BART Fortran-order ``img_dims`` array (length 16).
* ``pics(..., torch_prior=fn)`` argument validation and R-flag merging logic.
  The mocking-based tests exercise the Python layer independently of the
  C++ dispatch stub.

Tests that actually exercise the C++ ``__wrap_nlop_tf_create`` path are in
``tests/test_ext.py``.
"""

from __future__ import annotations

import re

import pytest
import torch

from bartorch.tools._commands import _bart_img_dims_from_kspace, pics

__all__: list[str] = []

_BART_DIMS = 16
_COIL_DIM = 3


# ---------------------------------------------------------------------------
# _bart_img_dims_from_kspace
# ---------------------------------------------------------------------------


class TestBartImgDimsFromKspace:
    """Verify BART img_dims computation for various kspace tensor shapes."""

    def test_2d_4d_tensor(self):
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

    def test_3d_tensor(self):
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

    def test_returns_list_of_ints(self):
        ksp = torch.zeros(2, 1, 8, 8, dtype=torch.complex64)
        dims = _bart_img_dims_from_kspace(ksp)
        assert isinstance(dims, list)
        assert all(isinstance(d, int) for d in dims)

    def test_length_always_16(self):
        for shape in [(2, 1, 4, 4), (4, 8, 16, 16), (1, 1, 32, 32)]:
            ksp = torch.zeros(*shape, dtype=torch.complex64)
            assert len(_bart_img_dims_from_kspace(ksp)) == _BART_DIMS

    def test_coil_dim_always_1(self):
        """Coil dim (Fortran index 3) must be 1 regardless of nc."""
        for nc in (1, 4, 16, 32):
            ksp = torch.zeros(nc, 1, 8, 8, dtype=torch.complex64)
            dims = _bart_img_dims_from_kspace(ksp)
            assert dims[_COIL_DIM] == 1, f"coil dim not 1 for nc={nc}: {dims}"

    def test_single_coil(self):
        nc, ny, nx = 1, 16, 16
        ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
        dims = _bart_img_dims_from_kspace(ksp)
        assert dims[0] == nx
        assert dims[1] == ny
        assert dims[_COIL_DIM] == 1

    def test_non_square_spatial(self):
        nc, ny, nx = 2, 32, 64
        ksp = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
        dims = _bart_img_dims_from_kspace(ksp)
        assert dims[0] == nx
        assert dims[1] == ny

    def test_3d_coil_zeroed(self):
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


class TestPicsTorchPriorValidation:
    """Argument-validation tests for pics(…, torch_prior=fn).

    Tests inspect function signatures or use monkey-patching so they do not
    call into the C++ extension and do not require real kspace / sensitivity-map
    data.
    """

    @staticmethod
    def _make_inputs(nc=2, ny=8, nx=8):
        kspace = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
        sens = torch.zeros(nc, 1, ny, nx, dtype=torch.complex64)
        return kspace, sens

    def test_torch_prior_lambda_default_is_float(self):
        """torch_prior_lambda has a sane numeric default (1.0)."""
        import inspect

        sig = inspect.signature(pics)
        default = sig.parameters["torch_prior_lambda"].default
        assert isinstance(default, float)
        assert default == 1.0

    def test_tf_reg_string_format(self):
        """Verify the -R TF:{bartorch://…}:lambda string is well-formed."""
        # We monkey-patch the extension to capture the R argument.
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
            kspace, sens = self._make_inputs()
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

    def test_torch_prior_r_merging_with_existing_r_string(self):
        """torch_prior appended after an existing R string → list of two."""
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
            kspace, sens = self._make_inputs()
            with pytest.raises(RuntimeError, match="sentinel"):
                pics(kspace, sens, R="W:7:0:0.005", torch_prior=lambda x: x, torch_prior_lambda=1.0)
        finally:
            graph_mod._get_ext = original_get_ext
            cmds._generated.pics = original_pics

        r_val = call_args.get("R")
        assert isinstance(r_val, list), f"Expected list, got {type(r_val)}"
        assert len(r_val) == 2
        assert r_val[0] == "W:7:0:0.005"
        assert r_val[1].startswith("TF:{bartorch://")

    def test_torch_prior_r_merging_with_existing_r_list(self):
        """torch_prior appended after an existing R list → list of three."""
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
            kspace, sens = self._make_inputs()
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

    def test_unique_names_across_calls(self):
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
            kspace, sens = self._make_inputs()
            for _ in range(3):
                with pytest.raises(RuntimeError, match="sentinel"):
                    pics(kspace, sens, torch_prior=lambda x: x, torch_prior_lambda=1.0)
        finally:
            graph_mod._get_ext = original_get_ext
            cmds._generated.pics = original_pics

        assert len(names) == 3
        assert len(set(names)) == 3, "Names must be unique across calls"


# ---------------------------------------------------------------------------
# pics() — torch_prior integration (real BART call)
# ---------------------------------------------------------------------------
# Ported from the spirit of bart/tests/test-pics-pi but with a plug-and-play
# torch prior instead of L2 regularisation.  The prior is a dummy identity
# denoiser (returns its input unchanged), which is mathematically equivalent
# to zero regularisation, so the reconstruction quality should be comparable
# to (or better than) the L2-only baseline.


def _identity_denoiser(x: torch.Tensor) -> torch.Tensor:
    """Trivial identity: returns input unchanged (zero regularisation)."""
    return x.clone()


class IdentityDenoiserModule(torch.nn.Module):
    """nn.Module wrapping the identity denoiser."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class TestPicsTorchPriorIntegration:
    """End-to-end integration tests for pics(…, torch_prior=fn).

    These tests call BART's pics via the C++ extension with a real phantom
    kspace and ESPIRiT sensitivity maps.  The torch prior is a dummy identity
    function so the result should be at least as good as L2-only reconstruction.
    """

    def test_torch_prior_lambda_callable(self):
        """pics() with identity-lambda prior runs without error."""
        import bartorch.tools as bt

        ksp = bt.phantom([64, 64], kspace=True, ncoils=4)
        sens = bt.ecalib(ksp, maps=1)
        reco = bt.pics(
            ksp,
            sens,
            lambda_=0.01,
            S=True,
            torch_prior=_identity_denoiser,
            torch_prior_lambda=0.0,
        )
        assert isinstance(reco, torch.Tensor)
        assert reco.dtype == torch.complex64
        assert reco.numel() > 0

    def test_torch_prior_nn_module(self):
        """pics() with an nn.Module identity prior runs without error."""
        import bartorch.tools as bt

        ksp = bt.phantom([64, 64], kspace=True, ncoils=4)
        sens = bt.ecalib(ksp, maps=1)
        denoiser = IdentityDenoiserModule()
        reco = bt.pics(
            ksp,
            sens,
            lambda_=0.01,
            S=True,
            torch_prior=denoiser,
            torch_prior_lambda=0.0,
        )
        assert isinstance(reco, torch.Tensor)
        assert reco.dtype == torch.complex64
        assert reco.numel() > 0

    def test_torch_prior_output_is_nonzero(self):
        """pics() with torch prior returns a non-trivially-zero reconstruction."""
        import bartorch.tools as bt

        ksp = bt.phantom([64, 64], kspace=True, ncoils=4)
        sens = bt.ecalib(ksp, maps=1)
        reco = bt.pics(
            ksp,
            sens,
            lambda_=0.01,
            S=True,
            torch_prior=_identity_denoiser,
            torch_prior_lambda=0.0,
        )
        assert reco.abs().max().item() > 0.0
