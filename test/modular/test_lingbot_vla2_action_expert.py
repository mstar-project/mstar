"""Deterministic CPU unit, parity, and weight-loading tests for LingBotVLAActionExpert."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Causal,
    KVEntry,
    Positions,
    PrefixSuffixGroups,
    Read,
    SubmoduleStep,
    add_read,
)
from mstar.engine.state.adapters import StateSetup, build_specs, node_handles, state_steps
from mstar.model.components.lingbot_vla2_action_expert import (
    LingBotVLAActionExpert,
    LingBotVLAActionExpertConfig,
    load_action_expert_weights,
)
from mstar.model.components.mrope import compute_3d_cos_sin, compute_inv_freq
from mstar.model.lingbot_vla2.config import ACTION_WALK, LLM_ATTN, LLM_KV, PREFILL_WALK, SUFFIX_ATTN

ROPE_THETA = 500_000.0
MROPE_SECTION = (2, 3, 3)


CHUNK_SIZE = 4
MAX_SEQ_LEN = 256


def _small_config(
    *,
    num_layers: int = 2,
    hidden_size: int = 24,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    routed_intermediate_size: int = 16,
    shared_intermediate_size: int = 24,
    num_experts: int = 4,
    top_k: int = 2,
    max_state_dim: int = 6,
    max_action_dim: int = 8,
    action_fp32: bool = False,
    fp32_attention: bool = True,
) -> LingBotVLAActionExpertConfig:
    return LingBotVLAActionExpertConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        top_k=top_k,
        routed_intermediate_size=routed_intermediate_size,
        shared_intermediate_size=shared_intermediate_size,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        rms_norm_eps=1e-6,
        rope_theta=ROPE_THETA,
        mrope_section=MROPE_SECTION,
        action_fp32=action_fp32,
        fp32_attention=fp32_attention,
    )


def _fill_weights(module: torch.nn.Module, gen: torch.Generator, dtype: torch.dtype) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.ndim >= 2:
                val = torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.1
            elif "norm" in name:
                val = 1.0 + torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.05
            else:
                val = torch.randn(param.shape, generator=gen, dtype=torch.float32) * 0.02
            param.copy_(val.to(dtype))


def _eager_reference_attention(
    prefix_k_layers: list[torch.Tensor],
    prefix_v_layers: list[torch.Tensor],
    prefix_lengths: list[int],
    fp32_attention: bool,
):
    """Exact PrefixSuffixGroups(first_group_len=1) attention reference over batch rows."""

    def _attn(layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bs, suffix_len, q_heads, head_dim = q.shape
        assert bs == len(prefix_lengths)
        kv_heads = k.shape[2]
        rep = q_heads // kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        outputs = []
        pref_k = prefix_k_layers[layer_idx]
        pref_v = prefix_v_layers[layer_idx]
        p_offset = 0
        for b in range(bs):
            p_len = prefix_lengths[b]
            q_b = q[b]
            k_suf = k[b]
            v_suf = v[b]
            k_full = torch.cat([pref_k[p_offset : p_offset + p_len], k_suf], dim=0)
            v_full = torch.cat([pref_v[p_offset : p_offset + p_len], v_suf], dim=0)
            p_offset += p_len

            k_rep = k_full.repeat_interleave(rep, dim=1)
            v_rep = v_full.repeat_interleave(rep, dim=1)
            mask = torch.zeros((suffix_len, p_len + suffix_len), dtype=torch.bool, device=q.device)
            mask[:, : p_len + 1] = True
            if suffix_len > 1:
                mask[1:, p_len + 1 :] = True

            q_h = q_b.transpose(0, 1)
            k_h = k_rep.transpose(0, 1)
            v_h = v_rep.transpose(0, 1)
            if fp32_attention:
                q_h, k_h, v_h = q_h.float(), k_h.float(), v_h.float()
            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(k_h.dtype)
            out_b = torch.matmul(probs, v_h).transpose(0, 1).to(q.dtype)
            outputs.append(out_b)
        return torch.stack(outputs, dim=0)

    return _attn


def _build_kv_state(cfg: LingBotVLAActionExpertConfig, dtype: torch.dtype):
    array = Array(
        LLM_KV,
        nodes={"LLM", "action_expert"},
        layers=frozenset(range(cfg.num_hidden_layers)),
        entry=KVEntry(heads=cfg.num_key_value_heads, head_dim=cfg.head_dim),
        max_len=MAX_SEQ_LEN,
        positions=Positions(),
        reads={
            LLM_ATTN: Read(mask=Causal(), op=Attend(q_heads=cfg.num_attention_heads), walks={PREFILL_WALK}),
            SUFFIX_ATTN: Read(
                mask=PrefixSuffixGroups(first_group_len=1, fp32_attention=cfg.fp32_attention),
                op=Attend(q_heads=cfg.num_attention_heads),
                walks={ACTION_WALK},
            ),
        },
    )
    setup = StateSetup()
    specs = {s.resource_key: s for s in build_specs([array], setup=setup)}
    specs[LLM_KV].config.max_num_pages = 32
    specs[LLM_KV].config.page_size = 16
    specs[LLM_KV].config.dtype = dtype

    transfer_info = TransferEngineInfo(
        my_entity_id="test_worker",
        transfer_engine=LocalTransferEngine("localhost"),
        my_session_id="test_session",
    )
    resources: dict = {}
    for key, spec in specs.items():
        if not spec.depends_on():
            info = EngineResourceInfo(
                device=torch.device("cpu"),
                transfer_engine_info=transfer_info,
                kv_dtype=dtype,
                compute_dtype=dtype,
                nodes=spec.nodes,
                needs_remote_transfer=False,
            )
            resources[key] = build_resource(spec, info)
    for key, spec in specs.items():
        if spec.depends_on():
            deps = {dep: specs[dep] for dep in spec.depends_on()}
            info = EngineResourceInfo(
                device=torch.device("cpu"),
                transfer_engine_info=transfer_info,
                kv_dtype=dtype,
                compute_dtype=dtype,
                nodes=spec.nodes,
                dependencies=deps,
                needs_remote_transfer=False,
            )
            res = build_resource(spec, info)
            res.bind_dependencies(resources)
            resources[key] = res

    from types import SimpleNamespace

    dummy_model = SimpleNamespace(
        declare_state=lambda: [array],
        kernel_settings=lambda: {},
        get_combined_graph_walks=lambda: {},
        get_graph_walk_graphs=lambda: {
            PREFILL_WALK: SimpleNamespace(get_nodes=lambda: {"LLM"}),
            ACTION_WALK: SimpleNamespace(get_nodes=lambda: {"action_expert"}),
        },
        get_node_resources=lambda: [],
    )
    built: dict = {}
    node_handles(dummy_model, "LLM", resources, setup=setup, built=built)
    node_handles(dummy_model, "action_expert", resources, setup=setup, built=built)
    handle = built[LLM_KV]
    runner = StepRunner(
        resources,
        node_resources={
            "LLM": [k for k, s in specs.items() if "LLM" in s.nodes],
            "action_expert": [k for k, s in specs.items() if "action_expert" in s.nodes],
        },
    )
    return resources, handle, runner


def _run_step(runner: StepRunner, handle, declared: SubmoduleStep, ctx: StepContext, fn):
    declared.set_ctx(ctx)
    expanded = state_steps(declared, {LLM_KV: handle}, graph_walk=ctx.graph_walk)
    expanded.set_ctx(ctx)
    outcome = runner.admit(expanded)
    assert outcome.ok, outcome
    plan = runner.plan(expanded)
    ctx_planned = StepContext(
        request_ids=ctx.request_ids,
        graph_walk=ctx.graph_walk,
        slot=ctx.slot,
        capture=ctx.capture,
        plan_results=plan,
    )
    expanded.set_ctx(ctx_planned)
    expanded.begin_calls()
    try:
        out = fn()
        expanded.check_calls()
        runner.commit(expanded)
        expanded.commit_calls()
        return out
    finally:
        expanded.end_calls()


@pytest.mark.parametrize("fp32_attention", [True, False], ids=["fp32_attn", "native_attn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("prefix_lengths", [[5], [4, 7]], ids=["bs1", "bs2"])
def test_action_expert_denoise_matches_eager_reference_via_kv_state(fp32_attention, dtype, prefix_lengths):
    gen = torch.Generator().manual_seed(2026)
    cfg = _small_config(fp32_attention=fp32_attention)
    expert = LingBotVLAActionExpert(cfg).eval()
    _fill_weights(expert, gen, dtype)
    expert = expert.to(dtype)
    expert.reset_non_persistent_buffers()

    bs = len(prefix_lengths)
    total_pref = sum(prefix_lengths)
    prefix_k = [
        torch.randn(total_pref, cfg.num_key_value_heads, cfg.head_dim, generator=gen, dtype=dtype)
        for _ in range(cfg.num_hidden_layers)
    ]
    prefix_v = [
        torch.randn(total_pref, cfg.num_key_value_heads, cfg.head_dim, generator=gen, dtype=dtype)
        for _ in range(cfg.num_hidden_layers)
    ]
    state = torch.randn(bs, cfg.max_state_dim, generator=gen, dtype=torch.float32)
    x_t = torch.randn(bs, CHUNK_SIZE, cfg.max_action_dim, generator=gen, dtype=torch.float32)
    prefix_end = torch.tensor(prefix_lengths, dtype=torch.long)

    ref_fn = _eager_reference_attention(prefix_k, prefix_v, prefix_lengths, cfg.fp32_attention)
    with torch.no_grad():
        expected = expert.denoise_actions(
            state, x_t, prefix_end, num_steps=3, attn_fn=ref_fn
        )

    _, handle, runner = _build_kv_state(cfg, dtype=dtype)
    for mod in expert.modules():
        if hasattr(mod, "bind_state"):
            mod.bind_state({LLM_KV: handle})

    rids = tuple(f"req_{i}" for i in range(bs))
    for rid in rids:
        runner.ingest_request(rid)

    prefill_step = SubmoduleStep(
        segments=tuple(Segment(rid, "main", plen) for rid, plen in zip(rids, prefix_lengths, strict=True)),
        state={LLM_KV: ArrayStep(keep=True)},
    )
    prefill_ctx = StepContext(graph_walk=PREFILL_WALK, request_ids=rids, slot=0, capture=False)

    def _fill_prefix():
        dummy_q = torch.zeros(total_pref, cfg.num_attention_heads, cfg.head_dim, dtype=dtype)
        for layer in range(cfg.num_hidden_layers):
            _ = add_read(handle, layer, dummy_q, k=prefix_k[layer], v=prefix_v[layer])

    _run_step(runner, handle, prefill_step, prefill_ctx, _fill_prefix)

    suffix_span = 1 + CHUNK_SIZE
    action_step = SubmoduleStep(
        segments=tuple(Segment(rid, "main", suffix_span) for rid in rids),
        state={LLM_KV: ArrayStep(keep=False)},
    )
    action_ctx = StepContext(graph_walk=ACTION_WALK, request_ids=rids, slot=0, capture=False)
    with torch.no_grad():
        actual = _run_step(
            runner,
            handle,
            action_step,
            action_ctx,
            lambda: expert.denoise_actions(state, x_t, prefix_end, num_steps=3),
        )

    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def test_action_expert_meta_init_resets_fp32_mrope_buffers():
    cfg = _small_config()
    with torch.device("meta"):
        expert = LingBotVLAActionExpert(cfg)
    expert = expert.to_empty(device="cpu").to(torch.bfloat16)
    expert.reset_non_persistent_buffers(device="cpu")
    expected_inv = compute_inv_freq(cfg.head_dim, cfg.rope_theta, device=torch.device("cpu"))
    assert expert.mrope_inv_freq.dtype == torch.float32
    torch.testing.assert_close(expert.mrope_inv_freq, expected_inv, rtol=0, atol=0)

    # Confirm 3D cos/sin table matches mrope helper directly.
    pos = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.long)
    cos, sin = compute_3d_cos_sin(pos, expert.mrope_inv_freq, list(cfg.mrope_section))
    assert cos.shape == (3, cfg.head_dim)
    assert sin.shape == (3, cfg.head_dim)


def _expert_checkpoint(
    tmp_path: Path,
    cfg: LingBotVLAActionExpertConfig,
    *,
    omit: str | None = None,
    extra: str | None = None,
):
    gen = torch.Generator().manual_seed(99)
    ref = LingBotVLAActionExpert(cfg)
    _fill_weights(ref, gen, torch.float32)

    expert_prefix = "model.qwenvl_with_expert.qwen_expert.model."
    tensors: dict[str, torch.Tensor] = {}
    for name, param in ref.named_parameters():
        val = param.detach().clone()
        if name.startswith("suffix_inputs."):
            key = "model." + name.removeprefix("suffix_inputs.")
            if key != omit:
                tensors[key] = val
        elif ".self_attn.self_attn.qkv_proj." in name or ".self_attn.qkv_proj." in name:
            # Qwen3/MoE wrapper stores inner self_attn.self_attn.qkv_proj
            clean = name.replace(".self_attn.self_attn.qkv_proj.", ".self_attn.qkv_proj.")
            q_dim = cfg.num_attention_heads * cfg.head_dim
            kv_dim = cfg.num_key_value_heads * cfg.head_dim
            q, k, v = val.split([q_dim, kv_dim, kv_dim], dim=0)
            for part, chunk in (("q", q), ("k", k), ("v", v)):
                key = expert_prefix + clean.replace(".qkv_proj.", f".{part}_proj.")
                if key != omit:
                    tensors[key] = chunk
        elif ".self_attn.self_attn." in name:
            key = expert_prefix + name.replace(".self_attn.self_attn.", ".self_attn.")
            if key != omit:
                tensors[key] = val
        elif name.endswith(".mlp.gate.e_score_correction_bias"):
            key = expert_prefix + name.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
            if key != omit:
                tensors[key] = val
        elif name.endswith(".mlp.experts.gate_up_proj"):
            moe_inter = cfg.routed_intermediate_size
            gate, up = val.split([moe_inter, moe_inter], dim=1)
            for part, chunk in (("gate_proj", gate), ("up_proj", up)):
                key = expert_prefix + name.replace(".experts.gate_up_proj", f".experts.{part}")
                if key != omit:
                    tensors[key] = chunk
        elif name.endswith(".mlp.shared_expert.gate_up_proj.weight"):
            shared_inter = cfg.shared_intermediate_size
            gate, up = val.split([shared_inter, shared_inter], dim=0)
            for part, chunk in (("gate_proj.weight", gate), ("up_proj.weight", up)):
                key = expert_prefix + name.replace(".shared_expert.gate_up_proj.weight", f".shared_expert.{part}")
                if key != omit:
                    tensors[key] = chunk
        else:
            key = expert_prefix + name
            if key != omit:
                tensors[key] = val

    if extra is not None:
        tensors[extra] = torch.zeros(1)

    shard = tmp_path / "model-00001-of-00001.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard.name for k in tensors}})
    )
    return ref


def test_load_action_expert_weights_roundtrip_and_strict_validation(tmp_path):
    cfg = _small_config()
    ref = _expert_checkpoint(tmp_path, cfg)
    loaded = LingBotVLAActionExpert(cfg)
    load_action_expert_weights(loaded, tmp_path)
    for name, param in ref.named_parameters():
        torch.testing.assert_close(dict(loaded.named_parameters())[name], param, rtol=0, atol=0)

    bad_missing = tmp_path / "missing"
    bad_missing.mkdir()
    _expert_checkpoint(bad_missing, cfg, omit="model.qwenvl_with_expert.qwen_expert.model.norm.weight")
    with pytest.raises(ValueError, match="unloaded"):
        load_action_expert_weights(LingBotVLAActionExpert(cfg), bad_missing)

    bad_extra = tmp_path / "extra"
    bad_extra.mkdir()
    _expert_checkpoint(bad_extra, cfg, extra="model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.unexpected_param")
    with pytest.raises(ValueError, match="Unexpected MoE parameter"):
        load_action_expert_weights(LingBotVLAActionExpert(cfg), bad_extra)


def test_action_expert_rejects_invalid_inputs():
    gen = torch.Generator().manual_seed(7)
    cfg = _small_config()
    expert = LingBotVLAActionExpert(cfg)
    _fill_weights(expert, gen, torch.float32)

    state = torch.zeros(1, cfg.max_state_dim)
    x_t = torch.zeros(1, CHUNK_SIZE, cfg.max_action_dim)
    pos = torch.tensor([4], dtype=torch.long)

    with pytest.raises((ValueError, RuntimeError)):
        expert.denoise_actions(torch.zeros(1, cfg.max_state_dim + 1), x_t, pos)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        expert.denoise_actions(state, torch.zeros(2, CHUNK_SIZE, cfg.max_action_dim), pos)
    with pytest.raises(ValueError, match="num_steps"):
        expert.denoise_actions(state, x_t, pos, num_steps=0)
    with pytest.raises((ValueError, RuntimeError)):
        expert.denoise_actions(state, torch.zeros(1, CHUNK_SIZE, cfg.max_action_dim + 1), pos)


@pytest.mark.parametrize("action_fp32", [False, True], ids=["action_bf16", "action_fp32"])
def test_action_expert_bfloat16_without_autocast_both_action_fp32_modes(action_fp32):
    gen = torch.Generator().manual_seed(2027)
    cfg = _small_config(action_fp32=action_fp32, fp32_attention=False)
    expert = LingBotVLAActionExpert(cfg).eval()
    _fill_weights(expert, gen, torch.bfloat16)
    expert = expert.to(torch.bfloat16)
    expert.reset_non_persistent_buffers()
    assert expert.suffix_inputs.state_proj.weight.dtype == torch.bfloat16
    assert expert.mrope_inv_freq.dtype == torch.float32

    prefix_lengths = [4, 6]
    bs = len(prefix_lengths)
    total_pref = sum(prefix_lengths)
    prefix_k = [
        torch.randn(total_pref, cfg.num_key_value_heads, cfg.head_dim, generator=gen, dtype=torch.bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]
    prefix_v = [
        torch.randn(total_pref, cfg.num_key_value_heads, cfg.head_dim, generator=gen, dtype=torch.bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]
    state = torch.randn(bs, cfg.max_state_dim, generator=gen, dtype=torch.float32)
    x_t = torch.randn(bs, CHUNK_SIZE, cfg.max_action_dim, generator=gen, dtype=torch.float32)
    prefix_end = torch.tensor(prefix_lengths, dtype=torch.long)

    ref_fn = _eager_reference_attention(prefix_k, prefix_v, prefix_lengths, cfg.fp32_attention)
    with torch.no_grad():
        pos = torch.arange(1 + CHUNK_SIZE, dtype=torch.long).unsqueeze(0).expand(3, -1)
        pos_batch = torch.cat([pos + p for p in prefix_lengths], dim=1)
        step_out = expert.forward_step(
            state, x_t, torch.full((bs,), 0.5, dtype=torch.float32), pos_batch, attn_fn=ref_fn
        )
        assert step_out.dtype == (torch.float32 if action_fp32 else torch.bfloat16)
        assert torch.isfinite(step_out).all()

        denoised = expert.denoise_actions(state, x_t, prefix_end, num_steps=2, attn_fn=ref_fn)
        assert denoised.dtype == torch.float32
        assert torch.isfinite(denoised).all()
