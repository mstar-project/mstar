"""Tests for the Ling-2.0 weight loader.

Three pure-Python tests verify the rename map + expert fusion converters
in isolation. Two CUDA/snapshot-gated tests load the real released
checkpoint into a 1-layer LingMoeModel and verify a forward pass
produces finite logits — the strongest signal we have that the model
code matches the upstream architecture byte-for-byte.

Snapshot lookup mirrors the other ming tests: ``MING_FLASH_OMNI_DIR``
env var, then the default HF Hub cache layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from mminf.model.ming_omni_flash.components.model import LingMoeModel
from mminf.model.ming_omni_flash.loader import (
    _compile_rename_rules,
    _rename_key,
    build_ling_weight_converters,
    load_thinker_weights,
)
from mminf.model.utils import _apply_operations


def _find_local_snapshot() -> str | None:
    """Locate a Ming-flash-omni-2.0 snapshot on disk, or None."""
    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and (Path(override) / "config.json").exists():
        return override

    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = hub_root / "models--inclusionAI--Ming-flash-omni-2.0" / "snapshots"
    if not repo_dir.exists():
        return None
    for snap in sorted(repo_dir.iterdir()):
        if (snap / "config.json").exists():
            return str(snap)
    return None


# Real-config values for the released ckpt, used by tests that
# instantiate a model matching the real architecture's hidden dims
# (so weight shapes line up).
def _real_thinker_dims(num_hidden_layers: int = 1) -> dict:
    return dict(
        vocab_size=157184,
        hidden_size=4096,
        intermediate_size=9216,
        moe_intermediate_size=1024,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=32,
        num_kv_heads=4,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=2_400_000.0,
        max_position_embeddings=32768,
        partial_rotary_factor=0.5,
        mrope_section=[8, 12, 12],
        num_experts=256,
        num_experts_per_tok=8,
        num_shared_experts=1,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        first_k_dense_replace=1,
    )


# ---------------------------------------------------------------------------
# Rename map + fusion converter unit tests
# ---------------------------------------------------------------------------


def test_rename_rules_resolve_layer0_keys() -> None:
    """Every layer-0 LLM ckpt key (after stripping ``model.``) renames to
    a parameter that exists in a 1-layer dense-only LingMoeModel."""
    compiled = _compile_rename_rules()
    # Build a small but architecturally-shaped 1-layer dense model.
    model = LingMoeModel(**_real_thinker_dims(num_hidden_layers=1))
    target_keys = set(model.state_dict().keys())

    # The layer-0 ckpt keys we expect to map. Outer ``model.`` is the
    # multimodal wrapper (BailingMM2NativeForConditionalGeneration); inner
    # ``model.`` is HF's BailingMoeV2ForCausalLM.model convention — except
    # for ``model.lm_head.weight`` which sits directly under the wrapper.
    layer0_ckpt_keys = [
        "model.lm_head.weight",                  # → stripped: lm_head.weight (direct match)
        "model.model.word_embeddings.weight",
        "model.model.norm.weight",
        "model.model.layers.0.input_layernorm.weight",
        "model.model.layers.0.post_attention_layernorm.weight",
        "model.model.layers.0.attention.query_key_value.weight",
        "model.model.layers.0.attention.dense.weight",
        "model.model.layers.0.attention.q_norm.weight",
        "model.model.layers.0.attention.k_norm.weight",
        "model.model.layers.0.mlp.gate_proj.weight",
        "model.model.layers.0.mlp.up_proj.weight",
        "model.model.layers.0.mlp.down_proj.weight",
    ]
    for k in layer0_ckpt_keys:
        # Loader strips the outer ``model.`` prefix first; if the stripped
        # form is already a target key, no rename runs.
        stripped = k.removeprefix("model.")
        if stripped in target_keys:
            continue
        renamed = _rename_key(stripped, compiled)
        assert renamed is not None, f"No rename rule for {stripped!r}"
        assert renamed in target_keys, (
            f"Renamed {stripped!r} → {renamed!r} not in model state_dict"
        )


def test_rename_rules_resolve_moe_layer_keys() -> None:
    """MoE-layer (layer 1+) keys map to a 2-layer model's state_dict."""
    compiled = _compile_rename_rules()
    model = LingMoeModel(**_real_thinker_dims(num_hidden_layers=2))
    target_keys = set(model.state_dict().keys())

    # Pass the post-outer-strip form to _rename_key (same as the loader does).
    moe_ckpt_keys = [
        "model.model.layers.1.mlp.gate.weight",
        "model.model.layers.1.mlp.gate.expert_bias",
        "model.model.layers.1.mlp.image_gate.weight",
        "model.model.layers.1.mlp.audio_gate.weight",
        "model.model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.model.layers.1.mlp.shared_experts.up_proj.weight",
        "model.model.layers.1.mlp.shared_experts.down_proj.weight",
    ]
    for k in moe_ckpt_keys:
        stripped = k.removeprefix("model.")
        renamed = _rename_key(stripped, compiled)
        assert renamed is not None, f"No rename rule for {stripped!r}"
        assert renamed in target_keys, (
            f"Renamed {stripped!r} → {renamed!r} not in model state_dict"
        )

    # Per-expert keys aren't IN target_keys directly (they fuse into
    # ``experts.gate_up_proj`` etc.), but the rename must still produce
    # a parseable, layer-correct name.
    expert_ckpt_keys = [
        "model.model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.model.layers.1.mlp.experts.255.down_proj.weight",
    ]
    for k in expert_ckpt_keys:
        stripped = k.removeprefix("model.")
        renamed = _rename_key(stripped, compiled)
        assert renamed is not None and renamed.startswith("layers.1.mlp.experts."), \
            f"Expert key {stripped!r} renamed badly: {renamed!r}"


def test_expert_fusion_converter_packs_correctly() -> None:
    """Hand-build per-expert tensors, run them through the WeightConverters,
    verify ``gate_up_proj`` packing is [gate, up] in dim=1 and that
    expert k's weights end up at slice k along dim=0."""
    converters = build_ling_weight_converters()
    moe_inter, hidden = 16, 8
    num_experts = 4

    # Per-expert gate/up/down tensors with distinguishable values.
    expert_kvs = {}
    for j in range(num_experts):
        expert_kvs[f"layers.5.mlp.experts.{j}.gate_proj.weight"] = (
            torch.full((moe_inter, hidden), float(j * 10 + 1))
        )
        expert_kvs[f"layers.5.mlp.experts.{j}.up_proj.weight"] = (
            torch.full((moe_inter, hidden), float(j * 10 + 2))
        )
        expert_kvs[f"layers.5.mlp.experts.{j}.down_proj.weight"] = (
            torch.full((hidden, moe_inter), float(j * 10 + 3))
        )

    # Fuse gate + up.
    gate_up_conv = converters[0]
    gate_up_subset = {
        k: v for k, v in expert_kvs.items()
        if "gate_proj" in k or "up_proj" in k
    }
    gate_up_packed = _apply_operations(gate_up_subset, gate_up_conv)
    assert gate_up_packed.shape == (num_experts, 2 * moe_inter, hidden)
    # Expert 0's gate slice (first half of dim 1) should be all 1.0
    # (= 0 * 10 + 1).
    assert torch.equal(
        gate_up_packed[0, :moe_inter], torch.full((moe_inter, hidden), 1.0)
    )
    # Expert 0's up slice (second half of dim 1) should be all 2.0.
    assert torch.equal(
        gate_up_packed[0, moe_inter:], torch.full((moe_inter, hidden), 2.0)
    )
    # Expert 2's gate slice should be all 21.0.
    assert torch.equal(
        gate_up_packed[2, :moe_inter], torch.full((moe_inter, hidden), 21.0)
    )

    # Fuse down_proj.
    down_conv = converters[1]
    down_subset = {
        k: v for k, v in expert_kvs.items() if "down_proj" in k
    }
    down_packed = _apply_operations(down_subset, down_conv)
    assert down_packed.shape == (num_experts, hidden, moe_inter)
    assert torch.equal(
        down_packed[3], torch.full((hidden, moe_inter), 33.0)
    )


def test_loader_strict_raises_on_missing_params(tmp_path: Path) -> None:
    """A snapshot with only ``lm_head.weight`` (missing every other param)
    must trigger the strict-mode KeyError."""
    # Build a minimal snapshot with one shard + index.json.
    from safetensors.torch import save_file
    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({"model.lm_head.weight": torch.zeros(157184, 4096)}, shard)
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {"model.lm_head.weight": shard.name},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    # Tiny dim variant so the 1-layer model fits easily.
    dims = _real_thinker_dims(num_hidden_layers=1)
    model = LingMoeModel(**dims)
    with pytest.raises(KeyError, match="Missing thinker parameters"):
        load_thinker_weights(model, str(tmp_path), device="cpu", strict=True)


# ---------------------------------------------------------------------------
# Real-checkpoint smoke (CUDA + snapshot required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def snapshot_dir() -> str:
    snap = _find_local_snapshot()
    if snap is None:
        pytest.skip(
            "Ming-flash-omni-2.0 snapshot not found. Set MING_FLASH_OMNI_DIR "
            "or download via `huggingface-cli download`."
        )
    return snap


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="real-ckpt smoke needs CUDA (embed + lm_head + 1 layer ≈ 3 GB)")
def test_load_layer0_real_weights_runs_forward(snapshot_dir: str) -> None:
    """Load embed + dense-layer-0 + norm + lm_head from the real ckpt
    into a 1-layer LingMoeModel; run a forward; verify shape + finite."""
    dims = _real_thinker_dims(num_hidden_layers=1)
    model = LingMoeModel(**dims).to(torch.bfloat16).cuda()
    load_thinker_weights(model, snapshot_dir, device="cuda", strict=True)
    model.eval()

    # Run a forward on a handful of arbitrary in-vocab token ids.
    import torch.nn.functional as F

    class _Cache:
        def set_layer_idx(self, i):
            pass
        def run_attention(self, q, k, v):
            num_heads = q.shape[1]
            num_kv = k.shape[1]
            if num_heads // num_kv > 1:
                k = k.repeat_interleave(num_heads // num_kv, dim=1)
                v = v.repeat_interleave(num_heads // num_kv, dim=1)
            q4 = q.transpose(0, 1).unsqueeze(0)
            k4 = k.transpose(0, 1).unsqueeze(0)
            v4 = v.transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True, scale=q.shape[-1] ** -0.5)
            return out.squeeze(0).transpose(0, 1).contiguous()

    input_ids = torch.tensor([100, 200, 300, 400], device="cuda")
    with torch.no_grad():
        out = model(_Cache(), input_ids=input_ids)

    assert out.shape == (4, dims["vocab_size"])
    assert torch.isfinite(out).all(), \
        f"Non-finite logits after 1-layer forward; max={out.abs().max().item()}"
    assert out.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="real-ckpt smoke needs CUDA")
def test_layer0_attention_weights_match_expected_shapes(snapshot_dir: str) -> None:
    """After load, every layer-0 attention parameter has the expected
    shape (catches rename mistakes that swap two params of different
    shape — e.g. q_norm vs k_norm if they happened to differ)."""
    dims = _real_thinker_dims(num_hidden_layers=1)
    model = LingMoeModel(**dims).to(torch.bfloat16).cuda()
    load_thinker_weights(model, snapshot_dir, device="cuda", strict=True)

    head_dim = dims["head_dim"]
    hidden = dims["hidden_size"]
    n_heads = dims["num_attention_heads"]
    n_kv = dims["num_kv_heads"]

    expected = {
        "layers.0.self_attn.qkv_proj.weight": ((n_heads + 2 * n_kv) * head_dim, hidden),
        "layers.0.self_attn.dense.weight": (hidden, n_heads * head_dim),
        "layers.0.self_attn.q_norm.weight": (head_dim,),
        "layers.0.self_attn.k_norm.weight": (head_dim,),
        "layers.0.input_layernorm.weight": (hidden,),
        "layers.0.post_attention_layernorm.weight": (hidden,),
        "embed_tokens.weight": (dims["vocab_size"], hidden),
        "lm_head.weight": (dims["vocab_size"], hidden),
    }
    state = dict(model.state_dict())
    for name, shape in expected.items():
        assert name in state, f"{name} missing from loaded state_dict"
        assert tuple(state[name].shape) == shape, (
            f"{name}: expected {shape}, got {tuple(state[name].shape)}"
        )
        assert torch.isfinite(state[name]).all(), \
            f"{name} contains non-finite values after load"
