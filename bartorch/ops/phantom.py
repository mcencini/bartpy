"""Phantom / simulation — bartorch.ops.phantom."""

from __future__ import annotations
from typing import Literal

from bartorch.core.graph import dispatch
from bartorch.core.tensor import BartTensor


def phantom(
    dims: list[int],
    *,
    kspace: bool = False,
    d3: bool = False,
    ptype: Literal["shepp", "geo", "circ", "ring", "star", "bart"] = "shepp",
    ncoils: int = 1,
    device: str = "cpu",
) -> BartTensor:
    """Generate a numerical phantom using BART's simu/phantom module.

    Parameters
    ----------
    dims:
        Spatial dimensions, e.g. ``[256, 256]`` (2-D) or ``[64, 64, 64]`` (3-D).
    kspace:
        Return k-space instead of image-space data.
    d3:
        3-D phantom.
    ptype:
        Phantom type string.
    ncoils:
        Number of coil channels (default 1 = single coil).
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    BartTensor
    """
    return dispatch(
        "phantom",
        [],
        dims,
        k=kspace,
        d3=d3,
        T=ptype,
        s=ncoils if ncoils > 1 else None,
        _device=device,
    )
