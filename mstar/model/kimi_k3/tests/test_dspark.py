"""Kimi K3.1's DSpark drafter (``Inferact/Kimi-K3.1-Pro-DSpark-preview``): the release's config and
tensor names against the port, the drafter's math against an unabsorbed reference written here
from the weights, its context cache, and the target's taps."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.dspark.config import DSparkConfig
from mstar.model.kimi_k3.dspark.model import DSparkContextStore, DSparkDraft
from mstar.model.kimi_k3.dspark.rope import yarn_inv_freq

FIXTURES = Path(__file__).parent / "fixtures"


def _release_config() -> dict:
    return json.loads((FIXTURES / "k31_dspark_config.json").read_text())


def test_release_config_parses_to_the_pinned_geometry():
    cfg = DSparkConfig.from_hf(_release_config())
    assert cfg == DSparkConfig()  # the defaults are the release
    assert (cfg.num_attention_heads, cfg.markov_rank, cfg.block_size, cfg.causal) == (96, 512, 7, True)
    assert cfg.target_layer_ids == (75, 79, 83, 87, 91) and cfg.context_width == 5 * 7168
    target = KimiK3ModelConfig.k31()
    cfg.check_target(target.hidden_size, target.num_hidden_layers, target.vocab_size)
    # vLLM's yarn: attn_factor cancels the tables' mscale(32) and mscale_all_dim 0 leaves 1/sqrt(d)
    draft = DSparkDraft(DSparkConfig.reduced(rope=cfg.rope), nn.Identity(), nn.Identity())
    assert draft.rope.table_scale == pytest.approx(1.0, abs=1e-12)
    assert draft.rope.attn_scale_factor == 1.0


@pytest.mark.parametrize("change", [
    {"rope_interleave": False}, {"markov_head_type": "gated"}, {"use_sliding_window": True},
    {"draft_vocab_size": 32000}, {"architectures": ["Qwen3DSparkModel"]}, {"mla_use_output_gate": True},
    {"dflash_config": {}, "full_attention_causal": None}, {"num_key_value_heads": 8},
])
def test_release_config_refuses_what_the_port_does_not_implement(change):
    raw = {key: value for key, value in {**_release_config(), **change}.items() if value is not None}
    with pytest.raises(ValueError, match="DSpark drafter config"):
        DSparkConfig.from_hf(raw)


def test_every_release_tensor_loads_into_the_drafter():
    """The release's 67 tensors by name and shape (its safetensors header): all but the confidence
    head land in a parameter, and every parameter gets one."""
    tensors = json.loads((FIXTURES / "k31_dspark_tensors.json").read_text())["tensors"]
    assert len(tensors) == 67
    with torch.device("meta"):
        draft = DSparkDraft(DSparkConfig(), nn.Embedding(163840, 7168), nn.Linear(7168, 163840, bias=False))
        stream = [(name, torch.empty(shape, dtype=torch.bfloat16)) for name, (dtype, shape) in tensors.items()]
    assert {dtype for dtype, _ in tensors.values()} == {"BF16"}
    loaded = draft.load_weights(stream, strict=True)
    assert loaded == {name for name, _ in draft.named_parameters()}
    assert sum(p.numel() for p in draft.parameters()) == sum(
        math.prod(shape) for name, (_, shape) in tensors.items() if not name.startswith("confidence_head."))


def _tiny(causal: bool = True, seed: int = 0) -> tuple[DSparkDraft, DSparkConfig]:
    cfg = DSparkConfig.reduced(causal=causal)
    torch.manual_seed(seed)
    draft = DSparkDraft(cfg, nn.Embedding(cfg.vocab_size, cfg.hidden_size),
                        nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False))
    with torch.no_grad():
        for p in list(draft.parameters()) + [draft.embed_tokens.weight, draft.lm_head.weight]:
            p.normal_(0, 0.3 if p.dim() == 1 else 0.08)
    return draft, cfg


def _rope(x: torch.Tensor, positions: torch.Tensor, cfg: DSparkConfig, fp32_angles: bool = False) -> torch.Tensor:
    """DeepSeek's reference rotation (complex pairs of adjacent dims), tables scaled as vLLM's.
    ``fp32_angles``: the angles rounded as fp32 computes them (position x frequency in fp32, as vLLM's
    and HF's tables do), the rest in float64; otherwise exact."""
    p = cfg.rope
    m = lambda s: 1.0 if p.factor <= 1 else 0.1 * s * math.log(p.factor) + 1.0  # noqa: E731
    scale = m(p.mscale) / m(p.mscale_all_dim) * p.attn_factor
    inv = yarn_inv_freq(x.shape[-1], p)
    if fp32_angles:
        angles = (positions.float()[:, None] * inv.float()[None, :]).double()
    else:
        angles = positions.double()[:, None] * inv.double()[None, :]
    cis = torch.polar(torch.full_like(angles, scale), angles)
    if x.dim() == 3:
        cis = cis[:, None, :]
    pairs = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(pairs * cis).flatten(-2).to(x.dtype)


def _reference(draft: DSparkDraft, cfg: DSparkConfig, states: torch.Tensor, anchor: int,
               trace: dict | None = None, fp32_angles: bool = False):
    """Unabsorbed MLA from the weights: keys and values through ``kv_b_proj`` for the context
    states and the block, the block at positions ``Tc ..``, causal or both ways within it.
    ``trace`` collects, per layer ``i``, the context cache rows (``("ctx", i)``), the block after the
    attention (``("attn", i)``) and after the MLP (``("mlp", i)``), then ``"final"`` (normed).
    ``fp32_angles``: rope angles rounded as fp32 computes them (``_rope``)."""
    sd = {name: t.detach().double().cpu() for name, t in draft.state_dict().items()}
    k, tc, h = cfg.block_size, states.shape[0], cfg.num_attention_heads
    nope, rope, vd, rank = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * w

    ids = torch.tensor([anchor] + [cfg.mask_token_id] * (k - 1))
    x = draft.embed_tokens.weight.detach().double().cpu()[ids]
    ctx = states.double()
    pos_c, pos_b = torch.arange(tc), torch.arange(tc, tc + k)
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        hb = rms(x, sd[f"{p}input_layernorm.weight"])
        q = F.linear(rms(F.linear(hb, sd[f"{p}self_attn.q_a_proj.weight"]), sd[f"{p}self_attn.q_a_layernorm.weight"]),
                     sd[f"{p}self_attn.q_b_proj.weight"]).view(k, h, nope + rope)
        q = torch.cat([q[..., :nope], _rope(q[..., nope:], pos_b, cfg, fp32_angles)], -1)
        src = torch.cat([ctx, hb])
        kv = F.linear(src, sd[f"{p}self_attn.kv_a_proj_with_mqa.weight"])
        c = rms(kv[:, :rank], sd[f"{p}self_attn.kv_a_layernorm.weight"])
        k_pe = _rope(kv[:, rank:], torch.cat([pos_c, pos_b]), cfg, fp32_angles)
        if trace is not None:
            trace["ctx", i] = torch.cat([c, k_pe], -1)[:tc]
        kvb = F.linear(c, sd[f"{p}self_attn.kv_b_proj.weight"]).view(-1, h, nope + vd)
        keys = torch.cat([kvb[..., :nope], k_pe[:, None, :].expand(-1, h, rope)], -1)
        scores = torch.einsum("qhd,khd->hqk", q, keys) * (nope + rope) ** -0.5
        if cfg.causal:
            allowed = torch.arange(tc + k)[None, :] <= tc + torch.arange(k)[:, None]
            scores = scores.masked_fill(~allowed, float("-inf"))
        o = torch.einsum("hqk,khd->qhd", scores.softmax(-1), kvb[..., nope:]).reshape(k, h * vd)
        x = x + F.linear(o, sd[f"{p}self_attn.o_proj.weight"])
        hm = rms(x, sd[f"{p}post_attention_layernorm.weight"])
        gate, up = F.linear(hm, sd[f"{p}mlp.gate_up_proj.weight"]).chunk(2, -1)
        if trace is not None:
            trace["attn", i] = x
        x = x + F.linear(F.silu(gate) * up, sd[f"{p}mlp.down_proj.weight"])
        if trace is not None:
            trace["mlp", i] = x
    final = rms(x, sd["final_norm.weight"])
    if trace is not None:
        trace["final"] = final
    logits = F.linear(final, draft.lm_head.weight.detach().double().cpu())
    drafts, prev = [], anchor
    for i in range(k):
        bias = F.linear(sd["markov_head.markov_w1.weight"][prev], sd["markov_head.markov_w2.weight"])
        prev = int((logits[i] + bias).argmax())
        drafts.append(prev)
    return torch.tensor(drafts), logits


@pytest.mark.parametrize("causal", [True, False])
def test_drafter_matches_an_unabsorbed_reference(causal):
    draft, cfg = _tiny(causal)
    states = torch.randn(9, cfg.hidden_size)
    drafts, logits = draft.draft_dense(states, anchor=17)
    ref_drafts, ref_logits = _reference(draft, cfg, states, 17)
    torch.testing.assert_close(logits.double(), ref_logits, atol=2e-5, rtol=0)
    assert torch.equal(drafts, ref_drafts)


def test_causal_block_rows_ignore_the_rows_after_them():
    """Row ``i`` of a causal block sees the context and rows ``<= i``: the last row alone differs
    from attending both ways (its view is the same), every earlier one moves."""
    causal, cfg = _tiny(True)
    both = DSparkDraft(replace(cfg, causal=False), causal.embed_tokens, causal.lm_head)
    both.load_state_dict(causal.state_dict())
    states = torch.randn(6, cfg.hidden_size)
    # with one layer the last row's view is identical; deeper layers mix the earlier rows in
    one = DSparkDraft(replace(cfg, num_hidden_layers=1), causal.embed_tokens, causal.lm_head)
    one_both = DSparkDraft(replace(cfg, num_hidden_layers=1, causal=False), causal.embed_tokens, causal.lm_head)
    sd = {n: t for n, t in causal.state_dict().items() if not n.startswith("layers.1.")}
    one.load_state_dict(sd)
    one_both.load_state_dict(sd)
    a, b = one.draft_dense(states, 3)[1], one_both.draft_dense(states, 3)[1]
    torch.testing.assert_close(a[-1], b[-1], atol=1e-6, rtol=0)
    assert (a[:-1] - b[:-1]).abs().amax(dim=-1).min() > 1e-4
    assert (causal.draft_dense(states, 3)[1] - both.draft_dense(states, 3)[1]).abs().max() > 1e-4


def test_context_store_appends_in_chunks_rewinds_and_batches():
    """Context rows appended chunk by chunk equal them computed at once; an append over a
    rejected tail replaces it; a batch of two requests drafts as each alone."""
    draft, cfg = _tiny()
    states = {"a": torch.randn(11, cfg.hidden_size), "b": torch.randn(5, cfg.hidden_size)}
    store = DSparkContextStore(draft)
    store.append("a", states["a"][:4], 0)
    store.append("a", torch.randn(5, cfg.hidden_size), 4)  # a verify's block, then rejected past 4
    store.append("a", states["a"][4:11], 4)
    store.append("b", states["b"], 0)
    assert store.length("a") == 11
    whole = draft.context_latents(states["a"], torch.arange(11))
    for got, want in zip(store.contexts(["a"])[0], whole, strict=True):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=0)
    anchors = torch.tensor([7, 40])
    drafts, logits = draft.draft(anchors, store.contexts(["a", "b"]))
    for row, rid in enumerate(("a", "b")):
        alone_drafts, alone_logits = draft.draft_dense(states[rid], int(anchors[row]))
        torch.testing.assert_close(logits[row], alone_logits, atol=1e-5, rtol=0)
        assert torch.equal(drafts[row], alone_drafts)
    with pytest.raises(ValueError, match="only 5 are cached"):
        store.append("b", states["b"], 6)


def test_context_accumulator_sums_to_combine():
    draft, cfg = _tiny()
    aux = [torch.randn(7, cfg.target_hidden_size) for _ in cfg.target_layer_ids]
    acc = draft.context_accumulator()
    for j, state in enumerate(aux):
        acc(j, state)
    torch.testing.assert_close(acc.finish(), draft.combine(aux), atol=1e-5, rtol=0)


def test_target_taps_are_the_layers_output_streams():
    """The engine's prefill taps (``aux_layers``) against the reference's per-layer streams, and the
    logits unchanged by tapping."""
    from mstar.model.kimi_k3.tests import engine_helpers as H
    from mstar.model.kimi_k3.tests.reference import reference_logits

    engine, sub, model, sd = H.build_engine(seed=2)
    cfg = model.config
    lm = sub.language_model.model
    plain = lm.forward
    got: dict[int, torch.Tensor] = {}
    taps = (1, 3)
    lm.forward = lambda ids, rows=None: plain(ids, rows=rows, aux_layers=taps,
                                              aux_sink=lambda j, s: got.__setitem__(taps[j], s.detach().clone()))
    prompt = torch.randint(0, cfg.vocab_size, (13,), generator=torch.Generator().manual_seed(1))
    H.add_request(engine, "a")
    H.run_step(engine, "prefill", {"a": prompt})
    want: dict[int, torch.Tensor] = {}
    ref = reference_logits(sd, prompt, cfg, taps=want)
    for layer in taps:
        torch.testing.assert_close(got[layer].float(), want[layer].float(), atol=5e-5, rtol=0)
    torch.testing.assert_close(engine._resources["sampler"].logits[-1][-1], ref[-1], atol=5e-5, rtol=0)
    with pytest.raises(ValueError, match="aux_layers"):
        plain(prompt, aux_layers=(cfg.num_hidden_layers - 1,), aux_sink=lambda j, s: None)


def _tp_rank(rank: int, store: str, out_dir: str) -> None:
    import datetime

    import torch.distributed as dist

    from mstar.distributed.all_reduce import AllReduce
    from mstar.distributed.communication import CommGroup

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120))
    torch.set_num_threads(1)
    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    group.device_group = group.cpu_group = dist.new_group([0, 1])
    group.initialized = True
    group._all_reduce = AllReduce(group.device_group)
    try:
        full, cfg = _tiny()
        draft = DSparkDraft(cfg, full.embed_tokens, full.lm_head, comm_group=group)
        stream = [(n.replace("gate_up_proj", "gate_proj"), t.chunk(2)[0]) if "gate_up_proj" in n else (n, t)
                  for n, t in full.state_dict().items()]
        stream += [(n.replace("gate_up_proj", "up_proj"), t.chunk(2)[1])
                   for n, t in full.state_dict().items() if "gate_up_proj" in n]
        draft.load_weights(stream)
        g = torch.Generator().manual_seed(4)
        states = [torch.randn(n, cfg.hidden_size, generator=g) for n in (6, 10)]
        with torch.inference_mode():
            contexts = [draft.context_latents(s, torch.arange(s.shape[0])) for s in states]
        aux = [torch.randn(5, cfg.target_hidden_size, generator=g) for _ in cfg.target_layer_ids]
        with torch.inference_mode():  # as the engine runs it
            drafts, logits = draft.draft(torch.tensor([3, 9]), contexts)
            combined = draft.combine(aux)
        torch.save({"drafts": drafts, "logits": logits, "combined": combined}, f"{out_dir}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_ranks_draft_as_one(tmp_path):
    """TP2 over gloo: heads, the MLP and ``context_proj`` split, the all-gather and the reductions:
    both ranks draft what one rank does."""
    import torch.multiprocessing as mp

    mp.spawn(_tp_rank, args=(str(tmp_path / "store"), str(tmp_path)), nprocs=2)
    full, cfg = _tiny()
    g = torch.Generator().manual_seed(4)
    states = [torch.randn(n, cfg.hidden_size, generator=g) for n in (6, 10)]
    aux = [torch.randn(5, cfg.target_hidden_size, generator=g) for _ in cfg.target_layer_ids]
    drafts, logits = full.draft(torch.tensor([3, 9]),
                                [full.context_latents(s, torch.arange(s.shape[0])) for s in states])
    for rank in (0, 1):
        got = torch.load(tmp_path / f"r{rank}.pt")
        torch.testing.assert_close(got["logits"], logits, atol=2e-5, rtol=0)
        torch.testing.assert_close(got["combined"], full.combine(aux), atol=2e-5, rtol=0)
        assert torch.equal(got["drafts"], drafts)


def test_rope_tables_stay_fp32_in_a_bf16_drafter():
    """GPU job 1465: ``.to(torch.bfloat16)`` cast the rope's ``inv_freq`` buffer, so at position
    ~8K a bf16 drafter's angles were off by up to ~10 rad (bf16 vs fp32 drafts agreed 2/28). The
    tables of a bf16 drafter equal an fp32 one's at the release's rope, far into the context."""
    cfg = DSparkConfig.reduced(qk_rope_head_dim=64, rope=DSparkConfig().rope)
    d32 = DSparkDraft(cfg, nn.Identity(), nn.Identity())
    d16 = DSparkDraft(cfg, nn.Identity(), nn.Identity()).to(torch.bfloat16)
    positions = torch.tensor([0, 511, 8191, 131071])
    for got, want in zip(d16.rope.cos_sin(positions), d32.rope.cos_sin(positions), strict=True):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
    x = torch.randn(4, 3, 64)
    torch.testing.assert_close(d16.rope.apply(x.bfloat16(), positions).float(),
                               d32.rope.apply(x, positions), atol=3e-2, rtol=0)


def test_an_fp32_drafter_normalizes_in_fp32_whatever_the_kernel(monkeypatch):
    """GPU job 1465: on CUDA ``mstar::rms_norm`` runs an fp32 input in bf16 (FlashInfer takes half
    precision only), so the fp32 drafter, the GPU check's oracle, was 1.07 off the float64 reference
    (logit std 3.2). With the norm kernel stood in by that downcast here, the fp32 drafter still
    matches the reference."""
    from mstar.model.components import norm

    def cuda_like(x, weight, eps=1e-6, rms_norm_dtype=None):
        h = x.bfloat16() if x.dtype == torch.float32 else x
        out = h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + eps).to(h.dtype) * weight.to(h.dtype)
        return out.to(x.dtype)

    monkeypatch.setattr(norm, "run_rms_norm", cuda_like)
    draft, cfg = _tiny()
    states = torch.randn(9, cfg.hidden_size)
    drafts, logits = draft.draft_dense(states, anchor=17)
    ref_drafts, ref_logits = _reference(draft, cfg, states, 17)
    torch.testing.assert_close(logits.double(), ref_logits, atol=2e-5, rtol=0)


def test_at_long_context_the_drafter_is_fp32_rope_angles_off_exact():
    """GPU job 1473: at 8192 context rows the fp32 drafter drifted from the float64 reference by
    ~20x what it does at 512 (context rows 4e-4 -> 7.6e-3, final norm median/std 1.5e-5 -> 6e-4),
    from the first layer's cache rows on. The cause is the rope angle: position x frequency rounds in
    fp32 (2.4e-4 rad at 8192, 3.9e-3 at 131072), as vLLM's and HF's fp32 tables round it, the
    convention the drafter keeps. Against a reference rounding its angles the same way and exact
    otherwise, the drafter at ~8K positions is as close as at short ones; against exact angles it is not."""
    cfg = DSparkConfig.reduced(qk_rope_head_dim=64, rope=DSparkConfig().rope)
    torch.manual_seed(0)
    draft = DSparkDraft(cfg, nn.Embedding(cfg.vocab_size, cfg.hidden_size),
                        nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False))
    with torch.no_grad():
        for p in list(draft.parameters()) + [draft.embed_tokens.weight, draft.lm_head.weight]:
            p.normal_(0, 0.3 if p.dim() == 1 else 0.08)
    states = torch.randn(8190, cfg.hidden_size)
    _, logits = draft.draft_dense(states, anchor=5)
    errs = {}
    for fp32_angles in (True, False):
        trace: dict = {}
        _, ref = _reference(draft, cfg, states, 5, trace=trace, fp32_angles=fp32_angles)
        cache = draft.context_latents(states, torch.arange(states.shape[0]))[0]
        errs[fp32_angles] = ((logits.double() - ref).abs().max().item(),
                             (cache.double() - trace["ctx", 0]).abs().max().item())
    assert errs[True][0] < 1e-5 and errs[True][1] < 1e-5, errs
    assert errs[False][1] > 10 * errs[True][1], errs  # the exact-angle cache rows: the rounding itself
