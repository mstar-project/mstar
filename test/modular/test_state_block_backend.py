"""Phi-4-multimodal's vision tower runs FA3, because FlashInfer pads its head_dim of 72 to 128.

A read within a step built its ragged resource on the platform's default, FlashInfer, so a tower that pinned another
kernel lost the pin on the contract, and its outputs moved with the kernel. ``backend`` in ``kernel_settings`` keeps the
pin, and a deployment's ``resources:`` block still overrides it.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources.attn.ragged.config import RaggedAttnBackend
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.state import Array, Attend, Block, KVEntry, Read
from mstar.engine.state.adapters import build_specs


def _ragged(settings: dict):
    tower = Array(
        "tokens", nodes={"tower"}, layers={0, 1}, entry=KVEntry(heads=2, head_dim=72), max_len=0,
        reads={"tower_attn": Read(mask=Block(), op=Attend(q_heads=2))},
    )
    (spec,) = build_specs([tower], kernel_settings={"tower_attn": settings})
    return spec


def test_a_block_reads_backend_setting_reaches_its_ragged_config():
    assert _ragged({}).config.backend is RaggedAttnBackend.AUTO
    assert _ragged({"backend": "dense"}).config.backend is RaggedAttnBackend.DENSE, "the tower would run FlashInfer"


def test_a_deployments_backend_overrides_the_models():
    spec = _ragged({"backend": "dense"})
    apply_yaml_overrides([spec], {"resources": {"tower_attn": {"backend": "flashinfer"}}})
    assert spec.config.backend is RaggedAttnBackend.FLASHINFER, "a deployment couldn't move the tower off FA3"


def test_a_block_reads_unknown_backend_is_refused_at_load():
    with pytest.raises(ValueError, match=r"Array 'tokens': kernel_settings\['tower_attn'\]: "
                                         "'fa9' is not a valid RaggedAttnBackend"):
        _ragged({"backend": "fa9"})
