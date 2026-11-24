"""CPU tests for DeepSeek-V4.1 full model, weight loader, chunked prefill, Engine lifecycle, and torch.compile."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import SamplingReqConfig
from mstar.engine.resources.kv import manager as kv_mod
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import StepContext
from mstar.engine.state.adapters import StateSetup
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import declared_resources
from mstar.model.components.deepseek_v41 import DeepSeekV41Config, DeepSeekV41TextBackbone
from mstar.model.deepseek_v41.config import REQUEST_META, SAMPLER, WINDOW_STATE
from mstar.model.deepseek_v41.deepseek_v41_model import DeepSeekV41Model
from mstar.model.deepseek_v41.submodules import DeepSeekV41LLMSubmodule
from mstar.model.deepseek_v41.weight_loader import load_deepseek_v41_weights
from mstar.model.submodule_base import InputMetadata
from mstar.testing_utils.resource import StubResource
from mstar.utils.fp8_block import E8M0_DTYPE, FP8_DTYPE


def _quantize_fp8_pair(weight: torch.Tensor, block: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce E4M3 weight + E8M0 [out//block, in//block] scale pair."""
    out_dim, in_dim = weight.shape
    blocks = weight.float().unflatten(0, (out_dim // block, block)).unflatten(2, (in_dim // block, block))
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    scale = (2.0 ** torch.ceil(torch.log2(amax / 448.0))).to(E8M0_DTYPE)
    wide = scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    q = (weight.float() / wide).clamp(-448.0, 448.0).to(FP8_DTYPE)
    return q, scale


def _pack_fp4_pair(rows: int, cols: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce packed E2M1 [rows, cols//2] int8 + E8M0 [rows, cols//32] scale pair."""
    nib_lo = torch.randint(0, 8, (rows, cols // 2), generator=gen, dtype=torch.int8)
    nib_hi = torch.randint(0, 8, (rows, cols // 2), generator=gen, dtype=torch.int8)
    packed = (nib_lo | (nib_hi << 4)).to(torch.int8)
    scale = torch.full((rows, cols // 32), 0.125, dtype=E8M0_DTYPE)
    return packed, scale


def _synthetic_hf_checkpoint(cfg: DeepSeekV41Config, seed: int = 0) -> list[tuple[str, torch.Tensor]]:
    """Build a full root-level HF DeepSeek-V4.1 checkpoint tensor stream for ``cfg``."""
    gen = torch.Generator().manual_seed(seed)
    items: list[tuple[str, torch.Tensor]] = []
    block = cfg.fp8_block

    def randn(shape, std=0.04, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * std).to(dtype)

    def add_fp8(stem: str, out_dim: int, in_dim: int):
        w, s = _quantize_fp8_pair(randn((out_dim, in_dim), dtype=torch.float32), block)
        items.append((f"{stem}.weight", w))
        items.append((f"{stem}.scale", s))

    items.append(("embed.weight", randn((cfg.vocab_size, cfg.hidden_size))))
    items.append(("norm.weight", torch.ones(cfg.hidden_size, dtype=torch.bfloat16)))
    items.append(("head.weight", randn((cfg.vocab_size, cfg.hidden_size))))

    for i in range(cfg.num_hidden_layers):
        for site in ("attn", "ffn"):
            items.append((f"layers.{i}.hc_{site}_base", randn((2 + cfg.hc_mult) * cfg.hc_mult, 0.1, torch.float32)))
            items.append(
                (
                    f"layers.{i}.hc_{site}_fn",
                    randn(((2 + cfg.hc_mult) * cfg.hc_mult, cfg.hc_mult * cfg.hidden_size), 0.05, torch.float32),
                )
            )
            items.append((f"layers.{i}.hc_{site}_scale", torch.ones(3, dtype=torch.float32)))
        items.append((f"layers.{i}.attn_norm.weight", torch.ones(cfg.hidden_size, dtype=torch.bfloat16)))
        items.append((f"layers.{i}.ffn_norm.weight", torch.ones(cfg.hidden_size, dtype=torch.bfloat16)))

        # Attention
        items.append((f"layers.{i}.attn.attn_sink", randn((cfg.num_attention_heads,), 0.1, torch.float32)))
        items.append((f"layers.{i}.attn.q_norm.weight", torch.ones(cfg.q_lora_rank, dtype=torch.bfloat16)))
        items.append((f"layers.{i}.attn.kv_norm.weight", torch.ones(cfg.head_dim, dtype=torch.bfloat16)))
        add_fp8(f"layers.{i}.attn.wq_a", cfg.q_lora_rank, cfg.hidden_size)
        add_fp8(f"layers.{i}.attn.wq_b", cfg.num_attention_heads * cfg.head_dim, cfg.q_lora_rank)
        add_fp8(f"layers.{i}.attn.wkv", cfg.head_dim, cfg.hidden_size)
        add_fp8(
            f"layers.{i}.attn.wo_a",
            cfg.o_groups * cfg.o_lora_rank,
            cfg.num_attention_heads * cfg.head_dim // cfg.o_groups,
        )
        add_fp8(f"layers.{i}.attn.wo_b", cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank)

        if i in cfg.kv_source_layer_ids:
            items.append((f"layers.{i}.attn.compressor.norm.weight", torch.ones(cfg.head_dim, dtype=torch.bfloat16)))
            items.append((f"layers.{i}.attn.compressor.wkv.weight", randn((cfg.head_dim, cfg.hidden_size))))
            if cfg.compress_ratios[i] > 1:
                items.append((f"layers.{i}.attn.compressor.wgate.weight", randn((cfg.head_dim, cfg.hidden_size))))

        if i in cfg.index_source_layer_ids:
            add_fp8(f"layers.{i}.attn.indexer.wq_b", cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
            items.append(
                (f"layers.{i}.attn.indexer.weights_proj.weight", randn((cfg.index_n_heads, cfg.hidden_size)))
            )
            if i in cfg.kv_source_layer_ids:
                items.append(
                    (f"layers.{i}.attn.indexer.k_norm.weight", torch.ones(cfg.index_head_dim, dtype=torch.bfloat16))
                )
                items.append((f"layers.{i}.attn.indexer.wk.weight", randn((cfg.index_head_dim, cfg.head_dim))))

        # MoE
        items.append((f"layers.{i}.ffn.gate.weight", randn((cfg.n_routed_experts, cfg.hidden_size))))
        items.append((f"layers.{i}.ffn.gate.bias", randn((cfg.n_routed_experts,), 0.05, torch.float32)))
        if cfg.vision_enabled:
            items.append((f"layers.{i}.ffn.gate.bias_vl", randn((cfg.n_routed_experts,), 0.05, torch.float32)))
        for proj, (r, c) in (
            ("w1", (cfg.moe_intermediate_size, cfg.hidden_size)),
            ("w3", (cfg.moe_intermediate_size, cfg.hidden_size)),
            ("w2", (cfg.hidden_size, cfg.moe_intermediate_size)),
        ):
            add_fp8(f"layers.{i}.ffn.shared_experts.{proj}", r, c)
            for e in range(cfg.n_routed_experts):
                pw, ps = _pack_fp4_pair(r, c, gen)
                items.append((f"layers.{i}.ffn.experts.{e}.{proj}.weight", pw))
                items.append((f"layers.{i}.ffn.experts.{e}.{proj}.scale", ps))

    for lid, rows in zip(cfg.engram_layer_ids, cfg.engram_num_embeddings, strict=True):
        # Engram embed stores 1 E8M0 scale per 32 cols on every row (rows is a prime sum, not a multiple of 32).
        raw_emb = randn((rows, cfg.engram_head_dim), dtype=torch.float32)
        emb_blocks = raw_emb.unflatten(1, (cfg.engram_head_dim // block, block))
        s_row = (2.0 ** torch.ceil(torch.log2(emb_blocks.abs().amax(dim=-1).clamp_min(1e-4) / 448.0))).to(E8M0_DTYPE)
        w_emb = (emb_blocks / s_row.float().unsqueeze(-1)).flatten(1).clamp(-448.0, 448.0).to(FP8_DTYPE)
        items.append((f"layers.{lid}.engram.embed.weight", w_emb))
        items.append((f"layers.{lid}.engram.embed.scale", s_row))
        add_fp8(
            f"layers.{lid}.engram.wkv",
            cfg.hidden_size * (cfg.hc_mult + 1),
            cfg.engram_hash_cols * cfg.engram_head_dim,
        )
        items.append((f"layers.{lid}.engram.q_weight", randn((cfg.hc_mult, cfg.hidden_size))))
        items.append((f"layers.{lid}.engram.k_weight", randn((cfg.hc_mult, cfg.hidden_size))))

    # Out-of-scope Phase 2/3/4 tensors that must be cleanly skipped
    items.extend(
        [
            ("vision.blocks.0.attn.wo.weight", torch.zeros(4, 4, dtype=torch.bfloat16)),
            ("aligner.w1.weight", torch.zeros(4, 4, dtype=torch.bfloat16)),
            ("image_start", torch.zeros(cfg.hidden_size, dtype=torch.bfloat16)),
            ("image_end", torch.zeros(cfg.hidden_size, dtype=torch.bfloat16)),
            ("image_newline", torch.zeros(cfg.hidden_size, dtype=torch.bfloat16)),
            ("mtp.0.norm.weight", torch.ones(cfg.hidden_size, dtype=torch.bfloat16)),
        ]
    )
    return items


def test_meta_full_config_state_dict_matches_pinned_architecture() -> None:
    raw = json.loads((Path(__file__).parent / "data/deepseek_v41_pinned_config.json").read_text())
    cfg = DeepSeekV41Config.from_hf(raw)
    with torch.device("meta"):
        backbone = DeepSeekV41TextBackbone(cfg)
    sd = backbone.state_dict()
    assert sd["embed.weight"].shape == (129280, 5120)
    assert sd["head.weight"].shape == (129280, 5120)
    assert sd["norm.weight"].shape == (5120,)
    assert sd["layers.0.attn.attn_sink"].shape == (64,)
    assert sd["layers.0.attn.attn_sink"].dtype == torch.float32
    assert sd["layers.2.attn.compressor.wgate.weight"].shape == (512, 5120)
    assert sd["layers.20.attn.compressor.wgate"] if False else "layers.20.attn.compressor.wgate.weight" not in sd
    assert sd["layers.20.attn.indexer.wk.weight"].shape == (128, 512)
    assert "layers.24.attn.indexer.wk.weight" not in sd
    assert sd["engrams.1.embed.weight"].shape == (384006168, 256)
    assert sd["engrams.1.embed.scale"].shape == (384006168, 8)
    assert sd["engrams.14.embed.weight"].shape == (384016682, 256)


def test_weight_loader_loads_pairs_skips_vision_mtp_and_refuses_faults() -> None:
    cfg = DeepSeekV41Config.reduced()
    tensors = _synthetic_hf_checkpoint(cfg, seed=11)
    with torch.device("meta"):
        m = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True).to(torch.float32)
    m.to_empty(device="cpu")
    loaded = load_deepseek_v41_weights(m, tensors, device="cpu")
    assert len(loaded) == len(dict(m.named_parameters()))
    assert m.layers[1].attn.compressor.wkv.weight.dtype == torch.float32
    assert m.layers[4].attn.compressor.wkv.weight.dtype == torch.bfloat16

    # Missing tensor refused
    with torch.device("meta"):
        m2 = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True)
    m2.to_empty(device="cpu")
    with pytest.raises((KeyError, ValueError), match="missing"):
        load_deepseek_v41_weights(m2, [pair for pair in tensors if pair[0] != "norm.weight"], device="cpu")

    # Unpaired .scale without .weight refused
    with torch.device("meta"):
        m3 = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True)
    m3.to_empty(device="cpu")
    with pytest.raises(ValueError, match="unpaired"):
        without_wqa = [pair for pair in tensors if pair[0] != "layers.0.attn.wq_a.weight"]
        load_deepseek_v41_weights(m3, without_wqa, device="cpu")

    # Duplicate tensor refused (both normal and post-pair-pop quantized weights)
    with torch.device("meta"):
        m4 = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True)
    m4.to_empty(device="cpu")
    with pytest.raises(ValueError, match="duplicate"):
        load_deepseek_v41_weights(m4, [*tensors, tensors[0]], device="cpu")

    with torch.device("meta"):
        m5 = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True)
    m5.to_empty(device="cpu")
    wqa_pair = [pair for pair in tensors if pair[0].startswith("layers.0.attn.wq_a.")]
    with pytest.raises(ValueError, match="duplicate"):
        load_deepseek_v41_weights(m5, [*tensors, *wqa_pair], device="cpu")

    # Unrecognized key refused
    with torch.device("meta"):
        m6 = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True)
    m6.to_empty(device="cpu")
    with pytest.raises(ValueError, match="unrecognized"):
        load_deepseek_v41_weights(m6, [*tensors, ("layers.0.bogus.weight", torch.zeros(4))], device="cpu")

    # Text-only config skips bias_vl cleanly when present in checkpoint
    with torch.device("meta"):
        m7 = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True)
    m7.to_empty(device="cpu")
    extra_bias_vl = [("layers.0.ffn.gate.bias_vl", torch.zeros(cfg.n_routed_experts, dtype=torch.float32))]
    load_deepseek_v41_weights(m7, [*tensors, *extra_bias_vl], device="cpu")


def test_chunked_prefill_matches_one_shot_prefill_on_cpu() -> None:
    cfg = DeepSeekV41Config.reduced()
    tensors = _synthetic_hf_checkpoint(cfg, seed=22)
    with torch.device("meta"):
        m = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True).to(torch.float32)
    m.to_empty(device="cpu")
    load_deepseek_v41_weights(m, tensors, device="cpu")
    m.eval()

    ids = torch.randint(0, cfg.vocab_size, (2, 13))
    with torch.no_grad():
        one_shot = m(ids, 0)
        cache = m.new_cache(2, 32, "cpu", torch.float32)
        # Split mid-group (5, 4, 3, 1) across ratio=2 compressor boundaries
        c1 = m(ids[:, :5], 0, cache=cache)
        c2 = m(ids[:, 5:9], 5, cache=cache)
        c3 = m(ids[:, 9:12], 9, cache=cache)
        c4 = m(ids[:, 12:13], 12, cache=cache)
        chunked = torch.cat([c1, c2, c3, c4], dim=1)
    torch.testing.assert_close(chunked, one_shot, rtol=1e-4, atol=1e-4)
    assert torch.equal(chunked.argmax(-1), one_shot.argmax(-1))


def test_model_contract_registration_loop_caps_and_prompt_validation() -> None:
    model = DeepSeekV41Model("", config_variant="reduced", tokenizer_mode="byte")
    assert state_declaration_problems(model) == []
    assert "messages" in model.request_kwargs()
    for max_tokens in (2, 64):
        assert 1 + model.get_loop_caps(max_output_tokens=max_tokens)["decode_loop"] == max_tokens
    out = model.process_prompt("hello", ["text"], ["text"], max_output_tokens=10)
    assert out.new_input_tensors["text_inputs"][0].tolist() == list(b"hello")
    assert out.metadata["state_layout"] == {}
    assert model.postprocess(torch.tensor([ord("o"), ord("k"), 1]), "text") == b"ok"

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.process_prompt("a" * 200, ["text"], ["text"], max_output_tokens=100)


class _GreedySampler(StubResource):
    def sample(self, request_ids, logits, **kwargs):
        del request_ids, kwargs
        return logits.argmax(-1)


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def remove_request(self, request_id):
        del request_id


def _batch(**kwargs) -> ExecutingBatch:
    kwargs.setdefault(
        "per_request_input_metadata",
        {
            rid: InputMetadata(dynamic_loop_iter_counts=dict(info.dynamic_loop_iter_counts))
            for rid, info in kwargs["per_request_info"].items()
        },
    )
    return ExecutingBatch(**kwargs)


def test_engine_cpu_interleaved_prefill_decode_parity_and_slot_cleanup(monkeypatch) -> None:
    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=lambda x, w, eps=1e-6: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        ).to(x.dtype)
        * w.to(x.dtype)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setattr(kv_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = DeepSeekV41Config.reduced()
    model = DeepSeekV41Model("", config_variant="reduced", tokenizer_mode="byte")
    tensors = _synthetic_hf_checkpoint(cfg, seed=33)
    with torch.device("meta"):
        backbone = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True).to(torch.float32)
    backbone.to_empty(device="cpu")
    load_deepseek_v41_weights(backbone, tensors, device="cpu")
    backbone.eval()
    sub = DeepSeekV41LLMSubmodule(backbone, cfg).eval()

    setup = StateSetup()
    specs = declared_resources(model, {}, setup)
    engine = Engine(None, autocast_dtype=None, state_setup=setup)
    engine.load_model(
        {"LLM": sub},
        specs,
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e",
            my_session_id="s",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
        kv_cache_type=torch.float32,
        model=model,
    )
    greedy = _GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.sampler_handle.resource = greedy
    engine.warmup()

    def info(rid: str, walk: str, max_tokens: int = 64) -> CurrentForwardPassInfo:
        return CurrentForwardPassInfo(
            request_id=f"wire-{rid}",
            rid_handle=rid,
            graph_walk=walk,
            fwd_index=0,
            random_seed=0,
            max_tokens=max_tokens,
            resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)},
        )

    def run(walk: str, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rids = list(inputs)
        batch = _batch(
            node_name="LLM",
            per_request_info={rid: info(rid, walk) for rid in rids},
            per_request_input_tensors={rid: {"text_inputs": [ids]} for rid, ids in inputs.items()},
            step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
        )
        engine.prepare_inputs(batch)
        assert not batch.failed_requests, batch.failed_requests
        outputs = engine.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        engine.finalize_batch(batch)
        return {rid: outputs.per_rid_outputs[rid]["new_token"][0] for rid in rids}

    win_pool = engine._resources[WINDOW_STATE]
    initial_free = win_pool.num_free_slots

    a = torch.tensor([5, 9, 13, 7, 21, 2], dtype=torch.long)
    b = torch.tensor([44, 3, 8], dtype=torch.long)
    for rid in ("a", "b"):
        engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})

    first = run("prefill", {"a": a, "b": b})
    seq = {"a": [first["a"]], "b": [first["b"]]}
    for _ in range(4):
        step = run("decode", {rid: row[-1] for rid, row in seq.items()})
        for rid, row in seq.items():
            row.append(step[rid])

    assert setup.host_fields.fields("a", REQUEST_META, "main", 0)["pos"] == len(a) + 4
    assert setup.host_fields.fields("b", REQUEST_META, "main", 0)["pos"] == len(b) + 4
    assert win_pool.num_free_slots == initial_free - 2

    # Whole-sequence prefill parity for each request independently
    for rid, prompt in (("a", a), ("b", b)):
        full = torch.cat([prompt, *seq[rid][:-1]])
        ref_id = f"{rid}_ref"
        engine.add_request(ref_id, {SAMPLER: SamplingReqConfig(temperature=0.0)})
        ref = run("prefill", {ref_id: full})[ref_id]
        assert ref.item() == seq[rid][-1].item()

    # Stop check on EOS token ID
    batch_stop = _batch(
        node_name="LLM",
        per_request_info={"a": info("a", "decode")},
        step_context=StepContext(request_ids=("a",), graph_walk="decode", slot=0, capture=False),
    )
    eos = torch.tensor([cfg.eos_token_ids[0]], dtype=torch.long)
    non_eos = torch.tensor([cfg.eos_token_ids[0] + 1], dtype=torch.long)
    assert engine.check_stop_for_batch(batch_stop, {"a": {"new_token": [non_eos]}}) == {}
    assert engine.check_stop_for_batch(batch_stop, {"a": {"new_token": [eos]}}) == {"a": {"decode_loop"}}

    # Per-request max_seq_len cap failure isolates only the offending request
    too_long = torch.zeros(cfg.max_seq_len + 1, dtype=torch.long)
    engine.add_request("too_long", {SAMPLER: SamplingReqConfig(temperature=0.0)})
    bad_batch = _batch(
        node_name="LLM",
        per_request_info={"too_long": info("too_long", "prefill")},
        per_request_input_tensors={"too_long": {"text_inputs": [too_long]}},
        step_context=StepContext(request_ids=("too_long",), graph_walk="prefill", slot=0, capture=False),
    )
    engine.prepare_inputs(bad_batch)
    assert set(bad_batch.failed_requests) == {"too_long"}

    for rid in ("a", "b", "a_ref", "b_ref", "too_long"):
        engine.remove_request(rid)
    assert win_pool.num_free_slots == initial_free


def test_backbone_layer_loop_compiles_without_per_layer_recompiles(dynamo_counts) -> None:
    class _StubFFN(torch.nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(hidden, hidden, bias=False)

        def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
            del image_mask
            return self.proj(x)

    def compiles(n: int) -> int:
        cfg = DeepSeekV41Config.reduced(
            num_hidden_layers=n,
            compress_ratios=(0,) * (n + 3),
            kv_source_layer_ids=(),
            index_source_layer_ids=(),
            candidate_source_layer_id=-1,
            engram_layer_ids=(),
            engram_num_embeddings=(),
            dspark_target_layer_ids=(0,),
        )
        m = DeepSeekV41TextBackbone(cfg)
        for layer in m.layers:
            layer.ffn = _StubFFN(cfg.hidden_size)
        cache = m.new_cache(1, 16, "cpu", torch.float32)
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        return dynamo_counts(lambda inp: m.forward_hidden(inp, 0, cache=cache), x).compiles

    assert compiles(4) == compiles(2)


def test_multi_engram_chunked_prefill_and_masks() -> None:
    cfg = DeepSeekV41Config.reduced(engram_layer_ids=(1, 4), engram_num_embeddings=(0, 0), vision_enabled=True)
    tensors = _synthetic_hf_checkpoint(cfg, seed=44)
    with torch.device("meta"):
        m = DeepSeekV41TextBackbone(cfg, engram_wkv_reference=True).to(torch.float32)
    m.to_empty(device="cpu")
    load_deepseek_v41_weights(m, tensors, device="cpu")
    m.eval()

    ids = torch.randint(0, cfg.vocab_size, (2, 11))
    tok_mask = torch.ones_like(ids, dtype=torch.bool)
    tok_mask[:, 4:6] = False
    img_mask = ~tok_mask
    with torch.no_grad():
        full = m(ids, 0, token_mask=tok_mask, image_mask=img_mask)
        cache = m.new_cache(2, 32, "cpu", torch.float32)
        p1 = m(ids[:, :5], 0, cache=cache, token_mask=tok_mask[:, :5], image_mask=img_mask[:, :5])
        p2 = m(ids[:, 5:], 5, cache=cache, token_mask=tok_mask[:, 5:], image_mask=img_mask[:, 5:])
        chunked = torch.cat([p1, p2], dim=1)
    torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-4)
    assert torch.equal(chunked.argmax(-1), full.argmax(-1))


def test_get_submodule_safetensors_bfloat16_and_engine_lifecycle(tmp_path: Path, monkeypatch) -> None:
    from safetensors.torch import save_file

    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(
        rmsnorm=lambda x, w, eps=1e-6: (
            x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        ).to(x.dtype)
        * w.to(x.dtype)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    monkeypatch.setattr(kv_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cfg = DeepSeekV41Config.reduced()
    tensors = dict(_synthetic_hf_checkpoint(cfg, seed=55))
    save_file(tensors, str(tmp_path / "model.safetensors"))

    model = DeepSeekV41Model(str(tmp_path), config_variant="reduced", tokenizer_mode="byte")
    sub = model.get_submodule("LLM", device="cpu", autocast_dtype=torch.bfloat16)
    assert isinstance(sub, DeepSeekV41LLMSubmodule)
    backbone = sub.model
    # Critical fp32/uint8 storage preserved under bfloat16 loading
    assert backbone.layers[0].attn.attn_sink.dtype == torch.float32
    assert backbone.layers[0].attn_hc.fn.dtype == torch.float32
    assert backbone.layers[0].ffn.experts.gate_up_proj_scale_inv.dtype == torch.float32
    assert backbone.layers[0].ffn.gate.e_score_correction_bias.dtype == torch.float32
    assert backbone.layers[1].attn.compressor.wkv.weight.dtype == torch.float32
    assert backbone.layers[4].attn.compressor.wkv.weight.dtype == torch.bfloat16
    assert backbone.engrams["1"].embed.weight.dtype == torch.uint8
    assert backbone.engrams["1"].wkv.weight_scale_inv.dtype == torch.float32

    setup = StateSetup()
    specs = declared_resources(model, {}, setup)
    engine = Engine(None, autocast_dtype=torch.bfloat16, state_setup=setup)
    engine.load_model(
        {"LLM": sub},
        specs,
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=torch.device("cpu"),
        transfer_engine_info=TransferEngineInfo(
            my_entity_id="e",
            my_session_id="s",
            transfer_engine=LocalTransferEngine("localhost"),
        ),
        kv_cache_type=torch.bfloat16,
        model=model,
    )
    greedy = _GreedySampler()
    engine._resources[SAMPLER] = greedy
    engine._submodules["LLM"].resources[SAMPLER] = greedy
    sub.sampler_handle.resource = greedy
    engine.warmup()

    rid = "bf16_req"
    engine.add_request(rid, {SAMPLER: SamplingReqConfig(temperature=0.0)})
    prompt = torch.tensor([12, 34, 56, 78], dtype=torch.long)
    prefill = _batch(
        node_name="LLM",
        per_request_info={
            rid: CurrentForwardPassInfo(
                request_id="wire-bf16",
                rid_handle=rid,
                graph_walk="prefill",
                fwd_index=0,
                random_seed=0,
                max_tokens=16,
                resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)},
            )
        },
        per_request_input_tensors={rid: {"text_inputs": [prompt]}},
        step_context=StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False),
    )
    engine.prepare_inputs(prefill)
    assert not prefill.failed_requests
    out = engine.exec_and_postprocess(prefill)
    engine.finalize_batch(prefill)
    tok = out.per_rid_outputs[rid]["new_token"][0]
    assert 0 <= int(tok.item()) < cfg.vocab_size

    decode = _batch(
        node_name="LLM",
        per_request_info={
            rid: CurrentForwardPassInfo(
                request_id="wire-bf16",
                rid_handle=rid,
                graph_walk="decode",
                fwd_index=1,
                random_seed=0,
                max_tokens=16,
                resource_configs={SAMPLER: SamplingReqConfig(temperature=0.0)},
            )
        },
        per_request_input_tensors={rid: {"text_inputs": [tok]}},
        step_context=StepContext(request_ids=(rid,), graph_walk="decode", slot=0, capture=False),
    )
    engine.prepare_inputs(decode)
    assert not decode.failed_requests
    out2 = engine.exec_and_postprocess(decode)
    engine.finalize_batch(decode)
    assert 0 <= int(out2.per_rid_outputs[rid]["new_token"][0].item()) < cfg.vocab_size
    engine.remove_request(rid)


def test_registry_and_shipped_tp8_config_validate_offline(monkeypatch) -> None:
    import yaml

    from mstar.api_server.openai.adapters import ADAPTER_REGISTRY, TextChatAdapter
    from mstar.cli.main import DEFAULT_CONFIGS
    from mstar.model.registry import HF_MODELS, MODEL_REGISTRY
    from mstar.validation import knobs, validate_deployment

    assert MODEL_REGISTRY["deepseek_v41"] == ("mstar.model.deepseek_v41.deepseek_v41_model", "DeepSeekV41Model")
    assert HF_MODELS["deepseek_v41"] == {"model_path_hf": "deepseek-ai/DeepSeek-V4.1-Flash"}
    assert DEFAULT_CONFIGS["deepseek_v41"] == "deepseek_v41_tp8.yaml"
    assert isinstance(ADAPTER_REGISTRY["deepseek_v41"], TextChatAdapter)

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "deepseek_v41_tp8.yaml"
    raw_yaml = yaml.safe_load(cfg_path.read_text())
    model = DeepSeekV41Model(HF_MODELS["deepseek_v41"]["model_path_hf"], **(raw_yaml.get("model_kwargs") or {}))
    assert state_declaration_problems(model) == []
    # Worktree uses shared .venv mstar_rust built from main repo checkout; validate deployment logic offline.
    monkeypatch.setattr(knobs, "require_rust_extension", lambda *a, **kw: None)
    deployment = validate_deployment(model, raw_yaml, str(cfg_path))
    assert len(deployment.worker_graphs) == 2
    assert deployment.sharding_config.groups[0].tp_size == 8
