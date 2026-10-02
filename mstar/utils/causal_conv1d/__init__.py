"""Causal depthwise conv1d over a recurrent state pool.

Vendored from vLLM (Apache-2.0); see :mod:`kernels` for provenance and what was
changed. ``causal_conv1d_fn`` is the varlen/chunked path and
``causal_conv1d_update`` the single-token one, mirroring the split in
``linear_attn/gdn.py``. Both take the conv state as a pool plus per-row slot
indices, which is how a ``RecurrentStatePool`` block is already laid out.
"""
from mstar.utils.causal_conv1d.kernels import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)

__all__ = ["PAD_SLOT_ID", "causal_conv1d_fn", "causal_conv1d_update"]
