"""Phantom / simulation — bartorch.ops.phantom.

Wraps BART's ``simu/phantom`` module for generating numerical MRI phantoms
(Shepp-Logan and geometric shapes) in image space or k-space.
"""

from __future__ import annotations

from typing import Literal

import torch

from bartorch.core.graph import dispatch
from bartorch.core.tensor import bart_op

__all__ = ["phantom"]


@bart_op
def phantom(
    dims: list[int],
    *,
    kspace: bool = False,
    d3: bool = False,
    ptype: Literal["shepp", "geo", "circ", "ring", "star", "bart"] = "shepp",
    ncoils: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a numerical MRI phantom using BART's ``simu/phantom`` module.

    Parameters
    ----------
    dims : list[int]
        Spatial dimensions, e.g. ``[256, 256]`` (2-D) or ``[64, 64, 64]``
        (3-D).  The length determines whether a 2-D or 3-D phantom is created.
    kspace : bool, optional
        Return k-space data instead of image-space data.  Default ``False``.
    d3 : bool, optional
        Force 3-D phantom generation.  Default ``False``.
    ptype : {'shepp', 'geo', 'circ', 'ring', 'star', 'bart'}, optional
        Phantom type.  Default ``'shepp'`` (Shepp-Logan).

        * ``'shepp'`` — classic Shepp-Logan phantom
        * ``'geo'``   — geometric shapes
        * ``'circ'``  — circle phantom
        * ``'ring'``  — ring phantom
        * ``'star'``  — star phantom
        * ``'bart'``  — BART logo phantom
    ncoils : int, optional
        Number of coil sensitivity channels.  ``1`` = single-coil (uniform).
        For ``ncoils > 1`` BART generates Biot-Savart coil profiles.
        Default ``1``.
    device : str, optional
        Target device: ``'cpu'`` or ``'cuda'``.  Default ``'cpu'``.

    Returns
    -------
    torch.Tensor
        Complex64 tensor.  For 2-D single-coil: shape ``(1, ny, nx)``
        (C-order, coils first).

    Examples
    --------
    Single-coil 2-D Shepp-Logan phantom:

    >>> import bartorch.ops as ops
    >>> ph = ops.phantom([256, 256])
    >>> ph.shape
    torch.Size([1, 256, 256])

    8-coil 256×256 k-space:

    >>> kspace = ops.phantom([256, 256], kspace=True, ncoils=8)
    >>> kspace.shape
    torch.Size([8, 1, 256, 256])
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
