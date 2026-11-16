"""Released checkpoint headers -> native meta parameters; no weight data or GPU.

The fixture is the exact JSON header fetched at the pinned revision. These tests
validate names, shapes, scale layout and coverage, not numerical dequantization.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mstar.model.ministral3.checkpoint import HF_FILENAME, VISION_PREFIXES, audit_checkpoint
from mstar.model.ministral3.components.language_model import Ministral3DenseLM
from mstar.model.ministral3.config import Ministral3TextConfig

FIXTURES = Path(__file__).parent / "fixtures" / "ministral3"
HEADER_BYTES = (FIXTURES / "model.safetensors.header.json").read_bytes()
HEADER = json.loads(HEADER_BYTES)
QUANTIZATION = json.loads((FIXTURES / "config.json").read_text())["quantization_config"]
Q = "language_model.model.layers.0.self_attn.q_proj"


@pytest.fixture
def model():
    config = Ministral3TextConfig.from_hf(FIXTURES, tokenizer_eos_token_id=2)
    with torch.device("meta"):
        return Ministral3DenseLM(config, attn_key="attn", kv_key="kv_cache", pos_key="rope")


def test_pinned_header_maps_every_text_weight_and_classifies_every_entry(model):
    provenance = json.loads((FIXTURES / "checkpoint_reference.json").read_text())
    record = next(entry for entry in provenance["files"] if entry["filename"] == HF_FILENAME)
    assert hashlib.sha256(HEADER_BYTES).hexdigest() == record["header_sha256"]
    assert len(HEADER_BYTES) == record["header_bytes"]
    audit = audit_checkpoint(HEADER, model, QUANTIZATION)
    assert len(audit.weights) == 236
    assert sum(row.dtype == "F8_E4M3" for row in audit.weights) == 182
    assert len(audit.native_parameters) == 158
    assert set(audit.native_parameters) == set(dict(model.named_parameters()))
    assert len(audit.excluded_vision) == 222
    assert all(name.startswith(VISION_PREFIXES) for name in audit.excluded_vision)
    consumed = {row.source for row in audit.weights}
    consumed.update(row.weight_scale for row in audit.weights if row.weight_scale)
    consumed.update(row.activation_scale for row in audit.weights if row.activation_scale)
    assert consumed.isdisjoint(audit.excluded_vision)
    assert consumed | set(audit.excluded_vision) == HEADER.keys() - {"__metadata__"}
    assert len(consumed) == 600
    assert audit.aliases == {"lm_head.weight": "embed_tokens.weight"}
    assert model.lm_head.weight is model.embed_tokens.weight
    assert all(parameter.is_meta for parameter in model.parameters())


def test_fused_projection_slices_have_the_checkpoint_geometry(model):
    rows = {row.source: row for row in audit_checkpoint(HEADER, model, QUANTIZATION).weights}
    layer = "language_model.model.layers.0."
    for projection, size, shard in (("q", 4096, "q"), ("k", 1024, "k"), ("v", 1024, "v")):
        row = rows[f"{layer}self_attn.{projection}_proj.weight"]
        assert (row.target, row.shard, row.shape) == ("layers.0.self_attn.qkv_proj.weight", shard, (size, 3072))
        assert row.weight_scale == f"{layer}self_attn.{projection}_proj.weight_scale_inv"
        assert row.activation_scale == f"{layer}self_attn.{projection}_proj.activation_scale"
    for projection, shard in (("gate", 0), ("up", 1)):
        row = rows[f"{layer}mlp.{projection}_proj.weight"]
        assert (row.target, row.shard, row.shape) == ("layers.0.mlp.gate_up_proj.weight", shard, (9216, 3072))
    assert rows[f"{layer}self_attn.o_proj.weight"].shape == (3072, 4096)


@pytest.mark.parametrize("name", [
    Q + ".weight", Q + ".weight_scale_inv", Q + ".activation_scale",
    "language_model.model.embed_tokens.weight", "language_model.model.norm.weight",
    "language_model.model.layers.25.mlp.up_proj.weight",
])
def test_missing_weights_or_scales_fail(model, name):
    header = dict(HEADER)
    del header[name]
    with pytest.raises(ValueError, match="Missing checkpoint tensor"):
        audit_checkpoint(header, model, QUANTIZATION)


@pytest.mark.parametrize("name,field,value", [
    (Q + ".weight", "shape", [3072, 3072]),  # wrong hidden_size / heads assumption
    (Q + ".weight", "dtype", "BF16"),
    (Q + ".weight_scale_inv", "shape", [32, 24]),  # block FP8 is a different layout
    (Q + ".weight_scale_inv", "dtype", "F32"),
    (Q + ".activation_scale", "shape", [1]),
])
def test_wrong_shapes_and_dtypes_fail(model, name, field, value):
    header = deepcopy(HEADER)
    header[name][field] = value
    with pytest.raises(ValueError, match="expected"):
        audit_checkpoint(header, model, QUANTIZATION)


@pytest.mark.parametrize("name", [
    "language_model.model.layers.0.self_attn.q_norm.weight",
    "language_model.model.layers.0.self_attn.q_proj.input_scale",
    "language_model.lm_head.weight", "vision_tower_extra.weight", "surprise.weight",
])
def test_unclassified_keys_are_not_silently_dropped(model, name):
    header = {**HEADER, name: {"shape": [1], "dtype": "BF16"}}
    with pytest.raises(ValueError, match="Unclassified checkpoint tensors"):
        audit_checkpoint(header, model, QUANTIZATION)


@pytest.mark.parametrize("changes", [
    {"quant_method": "int8"}, {"activation_scheme": "dynamic"}, {"weight_block_size": [128, 128]},
])
def test_other_quantization_layouts_are_not_assumed_compatible(model, changes):
    with pytest.raises(ValueError, match="static per-tensor FP8"):
        audit_checkpoint(HEADER, model, {**QUANTIZATION, **changes})


def test_native_parameter_shape_drift_is_caught_by_real_loader(model):
    model.norm.weight = torch.nn.Parameter(torch.empty(3073, device="meta"))
    with pytest.raises(AssertionError, match="shape mismatch"):
        audit_checkpoint(HEADER, model, QUANTIZATION)


def test_untied_native_head_is_not_filled_from_a_missing_checkpoint_tensor(model):
    model.config = replace(model.config, tie_word_embeddings=False)
    with pytest.raises(ValueError, match="tied"):
        audit_checkpoint(HEADER, model, QUANTIZATION)


def test_audit_refuses_to_write_into_materialized_parameters(model):
    # One small CPU parameter suffices; no real model allocation is needed.
    model.norm.weight = torch.nn.Parameter(torch.empty(3072))
    with pytest.raises(ValueError, match="meta-only"):
        audit_checkpoint(HEADER, model, QUANTIZATION)


def test_audit_requires_full_native_geometry(model):
    model.embed_tokens.weight = torch.nn.Parameter(torch.empty(65536, 3072, device="meta"))
    with pytest.raises(ValueError, match="unsharded embedding"):
        audit_checkpoint(HEADER, model, QUANTIZATION)
