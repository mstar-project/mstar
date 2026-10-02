"""Causal depthwise conv1d over a recurrent state pool.

Vendored from vLLM (Apache-2.0); see :mod:`kernels` for provenance and changes.
``causal_conv1d_fn`` is the varlen/chunked path, ``causal_conv1d_update`` the
single-token one. Both take the conv state as a pool plus per-row slot indices,
matching the ``RecurrentStatePool`` layout.
"""
from mstar.utils.causal_conv1d.kernels import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)

__all__ = ["PAD_SLOT_ID", "causal_conv1d_fn", "causal_conv1d_update"]
