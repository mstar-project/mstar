"""Qwen3 with a DSpark drafter (speculative decoding), and plain Qwen3, on the
real `Engine` with a tiny random-weight target and drafter on the CPU
(`qwen3_cpu`): greedy parity with plain decoding at several ``k``, an
always-reject control, a prompt-prefix hit on the verifying node, a sampled
sanity run, and every startup refusal of a drafter that doesn't fit.

A random drafter rarely guesses right, so the parity tests wrap the real
``propose`` (its KV writes and block attention still run) and substitute the
target's own continuation, its first spoiled draft chosen per case: every
accepted count 0..k is then exercised, and the output must be plain greedy's
whatever the drafts.
"""
from __future__ import annotations

import collections
import dataclasses

import pytest
import torch
from qwen3_cpu import (
    MASK,
    NODE,
    STOP,
    VOCAB,
    Driver,
    decode,
    draft_block,
    engine,
    forced_drafts,
    make_model,
    patch_cpu,
    reference_greedy,
    write_checkpoints,
)

from mstar.engine.resources.draft.startup import check_deployment, geometry_problems
from mstar.model.components.dspark.config import DSparkConfig
from mstar.model.components.dspark.drafter import DSparkDrafter

PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
MAX = 20


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    from mstar.engine.resources.kv import manager as manager_mod

    patch_cpu(monkeypatch)
    # MSTAR_KV_DEBUG_ASSERTS: verified <= device <= reserved lengths, and positions at KV length
    # plus a fixed offset, checked through every take-back here
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


@pytest.fixture
def ckpt(tmp_path):
    return write_checkpoints(tmp_path)


def _references(target, max_tokens=MAX, extra=8):
    """Each prompt followed by its plain greedy continuation, past the budget."""
    model = make_model(target)
    return {rid: p + reference_greedy(model, p, max_tokens + extra) for rid, p in PROMPTS.items()}


# ── plain Qwen3 ──


def test_without_a_draft_block_it_is_plain_qwen3(ckpt):
    target, _ = ckpt
    model = make_model(target)
    assert model.verify_nodes() == [] and model.drafter_decl is None
    out, _ = decode(model, PROMPTS, MAX)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference_greedy(model, prompt, MAX)


def test_plain_qwen3_decodes_with_no_captures(ckpt):
    """accelerator_graph: false builds no sampler slot masters, so decode reads
    each row's routed token rather than a loop-back slot."""
    target, _ = ckpt
    model = make_model(target)
    eng = engine(model, slot_masters=False)
    assert not eng._resources["sampler"].has_slot_masters
    driver = Driver(eng, model)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, MAX)
    driver.prefill(list(PROMPTS), PROMPTS)
    out = driver.run(list(PROMPTS))
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference_greedy(model, prompt, MAX)


# ── greedy parity ──


@pytest.mark.parametrize("k", [1, 3, 4])
@pytest.mark.parametrize("first_wrong", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("batched", [True, False])
def test_greedy_output_is_plain_greedy_whatever_the_drafts(ckpt, monkeypatch, k, first_wrong, batched):
    """Draft ``first_wrong`` of every block is spoiled (None: none is), so every
    step accepts exactly that many, every count 0..k across the cases; a step
    whose block runs past the reference's stop accepts fewer, at most one a row."""
    if first_wrong is not None and first_wrong >= k:
        pytest.skip("no such draft")
    target, dspark = ckpt
    monkeypatch.setattr(DSparkDrafter, "propose", forced_drafts(_references(target), lambda j: j == first_wrong))
    model = make_model(target, draft_block(dspark, k))
    out, driver = decode(model, PROMPTS, MAX, batched=batched)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference_greedy(model, prompt, MAX), (rid, k, first_wrong)
    want = k if first_wrong is None else first_wrong
    accepted = collections.Counter(driver.accepted)
    assert set(accepted) <= set(range(want + 1)), accepted
    assert accepted[want] >= len(driver.accepted) - len(PROMPTS), accepted


@pytest.mark.parametrize("k", [3, 4])
def test_every_other_draft_spoiled(ckpt, monkeypatch, k):
    target, dspark = ckpt
    monkeypatch.setattr(DSparkDrafter, "propose", forced_drafts(_references(target), lambda j: j % 2 == 1))
    model = make_model(target, draft_block(dspark, k))
    out, driver = decode(model, PROMPTS, MAX)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference_greedy(model, prompt, MAX)
    assert set(driver.accepted) <= {0, 1} and 1 in driver.accepted


def test_the_real_drafter_alone_keeps_parity(ckpt):
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 4))
    out, _ = decode(model, PROMPTS, MAX)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference_greedy(model, prompt, MAX)


def test_an_always_rejected_drafter_still_decodes_plain_greedy(ckpt, monkeypatch):
    target, dspark = ckpt
    monkeypatch.setattr(DSparkDrafter, "propose", forced_drafts(_references(target), lambda j: True))
    model = make_model(target, draft_block(dspark, 3))
    out, driver = decode(model, PROMPTS, MAX)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference_greedy(model, prompt, MAX)
    assert set(driver.accepted) == {0}


def test_a_stop_id_inside_a_block_ends_the_reply_there(tmp_path, monkeypatch):
    # a target whose greedy continuation hits the stop id: find a prompt that does
    target, dspark = write_checkpoints(tmp_path)
    plain = make_model(target)
    prompt = next(
        [a, b] for a in range(VOCAB - 2) for b in range(0, VOCAB - 2, 7)
        if STOP in reference_greedy(plain, [a, b], 12)[1:]
    )
    ref = reference_greedy(plain, prompt, 12)
    monkeypatch.setattr(DSparkDrafter, "propose", forced_drafts({1: prompt + ref + [0] * 8}))
    model = make_model(target, draft_block(dspark, 4))
    out, _ = decode(model, {1: prompt}, 12)
    assert out[1] == ref and out[1][-1] == STOP


# ── what DSpark drafts ──


@pytest.mark.parametrize("spoil", ["every2", "none"])
def test_dspark_drafts_what_a_dense_recomputation_drafts(ckpt, monkeypatch, spoil):
    """Every verify step's real drafts equal DSpark recomputed densely from the
    target's taps over the kept tokens: so the prefill's and observe's context
    writes, the take-back of rejected rows from the drafter's layers, the block's
    hidden last row and the Markov head are each what DSpark computes."""
    from qwen3_cpu import reference_drafts

    target, dspark = ckpt
    k = 3
    rule = {"every2": lambda j: j % 2 == 1, "none": lambda j: False}[spoil]
    record: list = []
    monkeypatch.setattr(DSparkDrafter, "propose", forced_drafts(_references(target), rule, record=record))
    model = make_model(target, draft_block(dspark, k))
    prompts = {1: PROMPTS[1], 3: PROMPTS[3]}
    out, _ = decode(model, prompts, 12)
    checked = 0
    for rids, anchors, drafts in record:
        for row, (rid, at) in enumerate(zip(rids, anchors, strict=True)):
            seq = prompts[rid] + out[rid]
            assert reference_drafts(model, seq, at, k) == drafts[row].tolist(), (rid, at)
            checked += 1
    assert checked >= 6


# ── the drafter's KV in the target's pool ──


def test_the_drafter_adds_its_layers_to_the_targets_pool(ckpt):
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3), max_seq_len=200)
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    kv = specs["kv_cache"]
    assert kv.config.num_layers == 3 + 2
    # the pool holds two committed verify steps past the serving limit, which ingest holds back
    assert kv.draft_slack == 2 * 4 and kv.config.max_seq_len == 200 + 8 and kv.request_limit == 200
    assert "dspark_kv" not in specs
    assert specs["dspark_attn"].config.kv_cache == "kv_cache"
    assert specs["dspark_taps"].layers == (1, 2)
    assert specs["sampler"].verifies and not specs["sampler"].enable_repetition_penalty


def test_a_prompt_prefix_hit_reuses_the_drafters_context(ckpt, monkeypatch):
    """The cached prompt pages carry the drafter's layers: a request that hits
    them drafts exactly as the cold request did, and decodes plain greedy."""
    target, dspark = ckpt
    prompt = list(range(1, 15))
    refs = {rid: prompt + reference_greedy(make_model(target), prompt, MAX + 8) for rid in (1, 2)}
    drafts: list[torch.Tensor] = []
    monkeypatch.setattr(DSparkDrafter, "propose", forced_drafts(refs, lambda j: j == 1, record=drafts))
    model = make_model(target, draft_block(dspark, 3))
    eng = engine(model, prefix_cache=True)
    assert eng._resources["kv_cache"]._keyed_walks["main"] == frozenset({"prefill_text"})
    driver = Driver(eng, model)
    driver.add(1, prompt, MAX, keyed=True)
    driver.prefill([1], {1: prompt})
    driver.run([1])
    cold = list(drafts)
    drafts.clear()
    driver.add(2, prompt, MAX, keyed=True)
    driver.prefill([2], {2: prompt}, prefix=True)
    # whole pages, the last one kept back to compute the prompt's last row
    assert driver.prefix_hit == 8
    driver.run([2])
    reference = reference_greedy(model, prompt, MAX)
    assert driver.out[1] == reference and driver.out[2] == reference
    assert len(drafts) == len(cold)
    assert all(torch.equal(a[2], b[2]) for a, b in zip(drafts, cold, strict=True))


# ── sampling ──


def test_sampled_verify_is_distributed_as_the_target(ckpt, monkeypatch):
    """At temperature 0.3 the 2nd and 3rd tokens (the first a verify step emits,
    and the next, from the bonus row or an accepted draft) follow the target's
    distribution: their marginals over 1,600 seeds are close, in total
    variation, to the exact ones. The drafter proposes the target's own greedy
    continuation, so both branches of speculative sampling run: drafts accepted
    with probability p(d), and the residual draw after a rejection."""
    target, dspark = ckpt
    prompt, temperature, k = [5, 9, 2], 0.3, 3
    plain = make_model(target)
    model = make_model(target, draft_block(dspark, k))
    lm = _dense_lm(plain)
    real = DSparkDrafter.propose
    driver = None

    def greedy(self, inputs):
        real(self, inputs)
        n = len(inputs.request_ids)
        anchors = inputs.resources["rope"].pos_ids("main").view(n, k + 1)[:, 0].tolist()
        rows = []
        for rid, at in zip(inputs.request_ids, anchors, strict=True):
            seq = (prompt + driver.out[rid])[: at + 1]
            row = []
            with torch.no_grad():
                for _ in range(k):
                    row.append(int(lm.logits(lm(lm.embed(torch.tensor(seq)), label="main"))[-1].argmax()))
                    seq = seq + [row[-1]]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.long)

    monkeypatch.setattr(DSparkDrafter, "propose", greedy)
    second, third, accepted = collections.Counter(), collections.Counter(), []
    for batch in range(200):
        driver = Driver(engine(model), model)
        rids = [batch * 8 + i for i in range(8)]
        for rid in rids:
            driver.add(rid, prompt, 3, temperature=temperature, seed=rid)
        driver.prefill(rids, dict.fromkeys(rids, prompt))
        driver.run(rids)
        for rid in rids:
            out = driver.out[rid]
            if len(out) > 1:
                second[out[1]] += 1
            if len(out) > 2:
                third[out[2]] += 1
        accepted += getattr(driver, "accepted", [])
    exact2, exact3 = _exact_marginals(plain, prompt, temperature)
    share = sum(a > 0 for a in accepted) / len(accepted)
    # both branches ran often
    assert 0.15 < share < 0.85, share
    for seen, exact in ((second, exact2), (third, exact3)):
        total = sum(seen.values())
        tv = 0.5 * sum(abs(seen.get(t, 0) / total - float(exact[t])) for t in range(VOCAB))
        # about twice the sampling error of ~1,500 draws over these distributions
        assert total > 1400 and tv < 0.11, (total, tv)


def _exact_marginals(model, prompt, temperature, floor=1e-7):
    """The 2nd token's marginal over replies that didn't stop at the 1st, and the
    3rd's over replies that didn't stop at the 1st or 2nd."""
    first = _next_token_probs(model, prompt, temperature)
    second = torch.zeros(VOCAB, dtype=torch.float64)
    third = torch.zeros(VOCAB, dtype=torch.float64)
    for a in range(VOCAB):
        if a == STOP or first[a] < floor:
            continue
        p2 = _next_token_probs(model, prompt + [a], temperature)
        second += first[a] * p2
        for b in range(VOCAB):
            if b == STOP or first[a] * p2[b] < floor:
                continue
            third += first[a] * p2[b] * _next_token_probs(model, prompt + [a, b], temperature)
    return second / second.sum(), third / third.sum()


def _dense_lm(model):
    from qwen3_cpu import DTYPE, _DenseAttention, _DensePositions, init_random

    lm = model.build_lm("cpu", dtype=DTYPE)
    init_random(lm, 0)
    attn, pos = _DenseAttention(), _DensePositions()
    for layer in lm.layers:
        layer.self_attn.bind_resources({"attn": attn, "kv_cache": attn, "rope": pos})
    return lm.eval()


def _next_token_probs(model, seq, temperature=1.0):
    from qwen3_cpu import DTYPE, _DenseAttention, _DensePositions, init_random

    lm = model.build_lm("cpu", dtype=DTYPE)
    init_random(lm, 0)
    attn, pos = _DenseAttention(), _DensePositions()
    for layer in lm.layers:
        layer.self_attn.bind_resources({"attn": attn, "kv_cache": attn, "rope": pos})
    with torch.no_grad():
        return torch.softmax(lm.logits(lm(lm.embed(torch.tensor(seq)), label="main"))[-1] / temperature, -1)


# ── the checkpoint's config ──


# deepseek-ai/dspark_qwen3_8b_block7's config.json, the keys the drafter reads
_CHECKPOINT = {
    "model_type": "qwen3", "block_size": 7, "markov_rank": 256, "mask_token_id": 151669,
    "target_layer_ids": [1, 9, 17, 25, 33], "hidden_size": 4096, "num_hidden_layers": 5, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 12288, "vocab_size": 151936,
    "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}, "dtype": "bfloat16",
}


def test_the_checkpoints_config_parses():
    config = DSparkConfig.from_hf(_CHECKPOINT)
    assert config.taps == (2, 10, 18, 26, 34) and config.block_size == 7 and config.rope_theta == 1e6


@pytest.mark.parametrize("change, message", [
    ({"speculators_model_type": "dspark"}, "not a DSpark drafter's config"),
    ({"dflash_config": {"block_size": 7}}, "not a DSpark drafter's config"),
    ({"is_causal": True}, "causal"),
])
def test_a_format_or_option_the_drafter_doesnt_implement_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        DSparkConfig.from_hf({**_CHECKPOINT, **change})


# ── startup refusals ──


def _problems(model):
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    (decl,) = model.verify_nodes()
    return geometry_problems(decl, specs)


@pytest.mark.parametrize("change, message", [
    ({"target_hidden_size": 64}, "residual width is 64"),
    ({"vocab_size": 101}, "vocabulary is 101"),
    ({"dtype": "bfloat16"}, "dtype is bfloat16"),
    # it rotates through the target's positions, and its attention plans over the target's pool
    ({"rope_parameters": {"rope_theta": 1e6, "rope_type": "default"}}, "rope_theta is 1000000.0"),
    ({"num_attention_heads": 8, "head_dim": 16}, "has 8 query heads"),
])
def test_a_drafter_that_doesnt_fit_its_target_is_refused(tmp_path, change, message):
    target, dspark = write_checkpoints(tmp_path, dspark=change)
    model = make_model(target, draft_block(dspark, 3))
    problems = _problems(model)
    assert any(message in p for p in problems), problems


def test_a_drafter_that_cant_join_the_pool_is_reported_once(tmp_path, monkeypatch):
    """Its join fails while the model declares its resources: validation lists that, and the
    speculative-decoding checks, which read the declarations, add nothing on top."""
    from mstar.engine.resources.kv.config import PagedKVConfig
    from mstar.validation import DeploymentError, validate_deployment

    def join(self, layer_set):
        raise ValueError(f"layer set {layer_set.name!r} can't join this pool")

    monkeypatch.setattr(PagedKVConfig, "join", join)
    target, dspark = write_checkpoints(tmp_path)
    model = make_model(target, draft_block(dspark, 3))
    config = {"max_seq_len": 64, "model_kwargs": {"draft": draft_block(dspark, 3)},
              "node_groups": [{"node_names": ["LLM"], "ranks": [0]}]}
    import yaml

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(DeploymentError) as raised:
        validate_deployment(model, config, str(path))
    problems = raised.value.problems
    assert len(problems) == 1, problems
    assert "declaring the node resources failed" in problems[0]
    assert "layer set 'dspark' can't join this pool" in problems[0]


def test_a_fitting_drafter_has_no_problems(ckpt):
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3))
    assert _problems(model) == []
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert check_deployment(model.verify_nodes(), specs, {}, (), {}, True) == []


@pytest.mark.parametrize("shards", [2, 4], ids=["split", "replicated"])
def test_the_geometry_check_reads_the_checkpoints_heads_on_a_sharded_pool(tmp_path, shards):
    # a TP/SP rank's pool holds a slice of the heads (one replicated head when there are
    # fewer heads than ranks); the drafter is checked against the target's whole count
    target, dspark = write_checkpoints(tmp_path)
    model = make_model(target, draft_block(dspark, 3))
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    (decl,) = model.verify_nodes()
    config = specs["kv_cache"].config
    config.shard(shards)
    assert config.num_kv_heads < config.total_kv_heads and config.num_qo_heads < config.total_qo_heads
    assert geometry_problems(decl, specs) == []

    # a drafter of other query heads is still refused once sharded, against the target's whole count
    target, dspark = write_checkpoints(tmp_path / "wide", dspark={"num_attention_heads": 8})
    model = make_model(target, draft_block(dspark, 3))
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    (decl,) = model.verify_nodes()
    specs["kv_cache"].config.shard(shards)
    assert any("has 8 query heads; the attention over KV 'kv_cache' plans 4" in p
               for p in geometry_problems(decl, specs))


def test_a_pool_without_the_drafters_layers_is_refused(ckpt):
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3))
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    specs["kv_cache"].config.num_layers = 3
    (decl,) = model.verify_nodes()
    assert any("holds layers 3..4 of KV 'kv_cache', which has 3" in p for p in geometry_problems(decl, specs))
    other = dataclasses.replace(specs["dspark_taps"], layers=(1, 3))
    specs["dspark_taps"] = other
    assert any("TapSpec 'dspark_taps' declares (1, 3)" in p for p in geometry_problems(decl, specs))


def test_the_engine_refuses_a_misfit_drafter_at_load(ckpt):
    from qwen3_cpu import cpu_specs

    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3))
    specs = cpu_specs(model)
    for spec in specs:
        if spec.resource_key == "dspark_taps":
            spec.hidden_size = 64
    with pytest.raises((RuntimeError, ValueError), match="64 wide"):
        engine(model, specs=specs)


@pytest.mark.parametrize("change, message", [
    ({"draft_vocab_size": 50}, "reduced draft vocabulary"),
    ({"use_sliding_window": True}, "sliding-window"),
    ({"rope_parameters": {"rope_theta": 1e4, "rope_type": "yarn"}}, "rope scaling"),
])
def test_drafters_the_model_cant_serve_are_refused_at_construction(tmp_path, change, message):
    target, dspark = write_checkpoints(tmp_path, dspark=change)
    with pytest.raises(ValueError, match=message):
        make_model(target, draft_block(dspark, 3))


def test_more_drafts_than_the_drafter_was_trained_on_are_refused(ckpt):
    target, dspark = ckpt
    with pytest.raises(ValueError, match="num_draft_tokens must be 1..4"):
        make_model(target, draft_block(dspark, 5))


def test_a_method_qwen3_has_no_drafter_for_is_refused(ckpt, monkeypatch):
    """A method the engine knows but Qwen3 doesn't draft with, such as a model's own MTP head."""
    from mstar.engine.resources.draft import config

    monkeypatch.setitem(config.DRAFT_METHODS, "other", config.DraftMethod("other", needs_model=False))
    target, _ = ckpt
    with pytest.raises(ValueError, match="Qwen3 drafts with method"):
        make_model(target, {"method": "other", "num_draft_tokens": 2})


def test_a_default_repetition_penalty_is_refused_when_drafting(ckpt):
    import json

    target, dspark = ckpt
    (target / "generation_config.json").write_text(json.dumps({"repetition_penalty": 1.05}))
    model = make_model(target, draft_block(dspark, 3))
    with pytest.raises(ValueError, match="repetition_penalty"):
        model.verify_nodes()


def test_the_mask_token_is_the_checkpoints(ckpt):
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3))
    assert model.drafter_decl.config.mask_token_id == MASK
    assert model.drafter_decl.verify_steps()["dspark_attn"].hide_last == 1
    assert NODE == "LLM"


@pytest.mark.parametrize("path, tp", [("configs/qwen3_8b_dspark.yaml", 1), ("configs/qwen3_8b_dspark_tp2.yaml", 2)])
def test_the_shipped_configs_name_the_rfc_drafter(path, tp):
    import yaml

    from mstar.engine.resources.draft.config import DraftConfig, DraftMode

    config = yaml.safe_load(open(path))
    draft = DraftConfig.from_model_kwargs(config["model_kwargs"])
    assert (draft.method, draft.model, draft.num_draft_tokens, draft.mode) == (
        "dspark", "deepseek-ai/dspark_qwen3_8b_block7", 7, DraftMode.AUTO,
    )
    # the checkpoint's context bounds a request; the model takes no length of its own
    assert config["model"] == "qwen3_8b" and "max_seq_len" not in config["model_kwargs"]
    (group,) = config["node_groups"]
    assert group["node_names"] == ["LLM"] and int(group.get("tp_size", 1)) == tp


def _hf_names(module, prefix: str = "") -> dict[str, torch.Tensor]:
    """``module``'s weights under HF's unfused names (q/k/v, gate/up)."""
    from mstar.model.components.qwen3_lm import Qwen3DenseLM

    out = {}
    lm = module if isinstance(module, Qwen3DenseLM) else module.backbone
    cfg = lm.config
    gen = torch.Generator().manual_seed(1)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32) * 0.05

    h, inter = cfg.hidden_size, cfg.intermediate_size
    q, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    out["embed_tokens.weight"] = rand(cfg.vocab_size, h)
    out["norm.weight"] = torch.ones(h)
    out["lm_head.weight"] = rand(cfg.vocab_size, h)
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        out.update({
            p + "input_layernorm.weight": torch.ones(h), p + "post_attention_layernorm.weight": torch.ones(h),
            p + "self_attn.q_proj.weight": rand(q, h), p + "self_attn.k_proj.weight": rand(kv, h),
            p + "self_attn.v_proj.weight": rand(kv, h), p + "self_attn.o_proj.weight": rand(h, q),
            p + "self_attn.q_norm.weight": torch.ones(cfg.head_dim),
            p + "self_attn.k_norm.weight": torch.ones(cfg.head_dim),
            p + "mlp.gate_proj.weight": rand(inter, h), p + "mlp.up_proj.weight": rand(inter, h),
            p + "mlp.down_proj.weight": rand(h, inter),
        })
    return {prefix + name: t for name, t in out.items()}


def test_the_checkpoints_load_under_their_own_names(tmp_path):
    """The target's ``model.*`` names and the DSpark checkpoint's (``fc``,
    ``hidden_norm``, ``markov_head``, its own embedding and head, a confidence
    head that is skipped) load through `get_submodule`, every parameter set."""
    from safetensors.torch import save_file

    target, dspark = write_checkpoints(tmp_path)
    model = make_model(target, draft_block(dspark, 3))
    lm = model.build_lm("cpu", dtype=torch.float32)
    weights = _hf_names(lm, "model.")
    weights["lm_head.weight"] = weights.pop("model.lm_head.weight")
    save_file(weights, str(target / "model.safetensors"))
    from mstar.model.components.dspark.drafter import DSparkDrafter

    with torch.device("meta"):
        shape = DSparkDrafter(model.drafter_decl)
    own = _hf_names(shape)
    cfg = model.drafter_decl.config
    own.update({
        "fc.weight": torch.randn(cfg.hidden_size, len(cfg.taps) * cfg.target_hidden_size) * 0.05,
        "hidden_norm.weight": torch.ones(cfg.hidden_size),
        "markov_head.markov_w1.weight": torch.randn(cfg.vocab_size, cfg.markov_rank),
        "markov_head.markov_w2.weight": torch.randn(cfg.vocab_size, cfg.markov_rank),
        "confidence_head.proj.weight": torch.randn(1, cfg.hidden_size + cfg.markov_rank),
    })
    save_file(own, str(dspark / "model.safetensors"))
    sub = model.get_submodule("LLM", device="cpu", autocast_dtype=torch.float32)
    assert sub.drafter is not None
    params = dict(sub.named_parameters())
    torch.testing.assert_close(params["model.embed_tokens.weight"], weights["model.embed_tokens.weight"])
    torch.testing.assert_close(params["drafter.fc.weight"], own["fc.weight"])
    torch.testing.assert_close(params["drafter.markov_head.markov_w2.weight"], own["markov_head.markov_w2.weight"])
    torch.testing.assert_close(params["drafter.backbone.lm_head.weight"], own["lm_head.weight"])
    # its layers sit after the target's in the one pool, placed there by the layer set they name
    # (test_drafter_layer_sets checks where)
    assert all(layer.self_attn._kv_layer_set == "dspark" for layer in sub.drafter.backbone.layers)


def test_a_drafter_without_an_embedding_shares_the_targets(tmp_path):
    from mstar.model.components.dspark.drafter import DSparkDrafter

    target, dspark = write_checkpoints(tmp_path)
    model = make_model(target, draft_block(dspark, 3))
    lm = model.build_lm("cpu", dtype=torch.float32)
    with torch.device("meta"):
        shape = DSparkDrafter(model.drafter_decl)
    own = {k: v for k, v in _hf_names(shape).items() if not k.startswith(("embed_tokens", "lm_head"))}
    cfg = model.drafter_decl.config
    own.update({
        "fc.weight": torch.zeros(cfg.hidden_size, len(cfg.taps) * cfg.target_hidden_size),
        "hidden_norm.weight": torch.ones(cfg.hidden_size),
        "markov_head.markov_w1.weight": torch.zeros(cfg.vocab_size, cfg.markov_rank),
        "markov_head.markov_w2.weight": torch.zeros(cfg.vocab_size, cfg.markov_rank),
    })
    module = DSparkDrafter(model.drafter_decl)
    module.load_weights(own.items(), target=lm)
    assert module.backbone.embed_tokens is lm.embed_tokens and module.backbone.lm_head is lm.lm_head


def test_a_verify_row_that_declares_one_token_fails_before_admit(ckpt, monkeypatch, caplog):
    """A row's first verify step goes through the model's own prepare_inputs: a
    span other than k + 1 fails the request rather than under-reserving its KV."""
    import torch as _torch

    from mstar.model.qwen3.submodules import Qwen3LMSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    real = Qwen3LMSubmodule.prepare_inputs

    def one_token(self, graph_walk, fwd_info, inputs, has_loopback=False, **kw):
        if graph_walk == "decode":
            return ARNodeInputs(input_ids=inputs["text_inputs"][0].reshape(-1)[-1:], input_seq_len=1)
        return real(self, graph_walk, fwd_info, inputs, has_loopback=has_loopback, **kw)

    monkeypatch.setattr(Qwen3LMSubmodule, "prepare_inputs", one_token)
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3))
    eng = engine(model)
    driver = Driver(eng, model)
    driver.add(1, PROMPTS[1], MAX)
    driver.prefill([1], PROMPTS)
    batch = driver._batch("decode", {1: _torch.tensor([driver.out[1][-1]])})
    eng.prepare_inputs(batch)
    # a model bug: the client sees an internal error, the log says what
    assert 1 in batch.failed_requests and batch.request_ids == []
    assert "k + 1 = 4 tokens" in caplog.text and "declared 1" in caplog.text


def test_a_drafter_config_without_its_block_size_is_refused(tmp_path):
    import json

    target, dspark = write_checkpoints(tmp_path)
    config = json.loads((dspark / "config.json").read_text())
    del config["block_size"]
    (dspark / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="block_size"):
        make_model(target, draft_block(dspark, 3))


def test_a_drafter_borrowing_the_embedding_must_share_its_width(tmp_path):
    target, dspark = write_checkpoints(tmp_path, dspark={"hidden_size": 64})
    model = make_model(target, draft_block(dspark, 3))
    lm = model.build_lm("cpu", dtype=torch.float32)
    module = DSparkDrafter(model.drafter_decl)
    with pytest.raises(RuntimeError, match="64"):
        module.load_weights([], target=lm)


def test_the_taps_are_kept_in_the_dtype_the_node_runs_in(ckpt):
    from mstar.engine.resources.base import EngineResourceInfo
    from mstar.engine.resources.taps.config import TapSpec
    from mstar.engine.resources.taps.resource import TapResource

    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3))
    (spec,) = [s for s in model.get_node_resources() if isinstance(s, TapSpec)]
    cpu = torch.device("cpu")
    assert spec.dtype is None
    assert TapResource.build(spec, EngineResourceInfo(device=cpu, compute_dtype=torch.float16)).dtype is torch.float16
    assert TapResource.build(spec, EngineResourceInfo(device=cpu, kv_dtype=torch.float64)).dtype is torch.float64


def test_the_prefix_cache_is_rooted_in_the_drafters_weights(tmp_path):
    """Prompt pages hold the drafter's KV layers: another drafter checkpoint
    (same shapes) must not hit them."""
    roots = []
    for name, markov_rank in (("a", 8), ("b", 8), ("c", 16)):
        target, dspark = write_checkpoints(tmp_path / name, dspark={"markov_rank": markov_rank})
        model = make_model(target, draft_block(dspark, 3))
        assert model.verify_nodes()[0].checkpoint == str(dspark)
        if name == "b":
            # same shapes, other weights: a shard of another size
            (dspark / "model.safetensors").write_bytes(b"x" * 10)
        eng = engine(model, prefix_cache=True)
        roots.append(eng._resources["kv_cache"]._prefix_root)
    assert len(set(roots)) == 3


# min_tokens and repetition_window: the scaffold forwards neither (`SAMPLING_KEYS`)
@pytest.mark.parametrize("knob", [{"penalize_prompt": False}, {"repetition_penalty": 1.1}])
def test_a_knob_verify_cant_honour_is_refused_not_dropped(ckpt, knob):
    """Every sampling knob reaches the sampler's ingest check."""
    from mstar.engine.resources.sampler.config import SamplerSpec

    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, 3))
    (spec,) = [s for s in model.get_node_resources() if isinstance(s, SamplerSpec)]
    config = model.get_request_resource_configs(["text"], ["text"], knob)["sampler"]
    for name, value in knob.items():
        assert getattr(config, name) == value
    with pytest.raises(ValueError):
        config.validate(spec)


@pytest.mark.parametrize("ids, message", [([0, 2], "last layer's"), ([0, 3], "last layer's")])
def test_a_drafter_reading_the_last_layers_output_is_refused(tmp_path, ids, message):
    """HF's hidden_states[L] is after the final norm; the taps aren't."""
    target, dspark = write_checkpoints(tmp_path, dspark={"target_layer_ids": ids})
    with pytest.raises(ValueError, match=message):
        make_model(target, draft_block(dspark, 3))


def _drafter_weights(model, drop=(), extra=None):
    from mstar.model.components.dspark.drafter import DSparkDrafter

    with torch.device("meta"):
        shape = DSparkDrafter(model.drafter_decl)
    weights = {k: v for k, v in _hf_names(shape).items() if not k.startswith(drop)}
    cfg = model.drafter_decl.config
    weights.update({
        "fc.weight": torch.zeros(cfg.hidden_size, len(cfg.taps) * cfg.target_hidden_size),
        "hidden_norm.weight": torch.ones(cfg.hidden_size),
        "markov_head.markov_w1.weight": torch.zeros(cfg.vocab_size, cfg.markov_rank),
        "markov_head.markov_w2.weight": torch.zeros(cfg.vocab_size, cfg.markov_rank),
        **(extra or {}),
    })
    return weights


def test_a_drafter_tying_its_head_uses_its_own_embedding(tmp_path):
    """A tied drafter's head is its embedding, not the target's head."""
    from mstar.model.components.dspark.drafter import DSparkDrafter

    target, dspark = write_checkpoints(tmp_path, dspark={"tie_word_embeddings": True})
    model = make_model(target, draft_block(dspark, 3))
    lm = model.build_lm("cpu", dtype=torch.float32)
    weights = _drafter_weights(model, drop=("lm_head",))
    module = DSparkDrafter(model.drafter_decl)
    module.load_weights(weights.items(), target=lm)
    assert module.backbone.lm_head is not lm.lm_head
    torch.testing.assert_close(module.backbone.lm_head.weight, weights["embed_tokens.weight"])


def test_a_checkpoints_mask_embedding_is_refused_not_ignored(tmp_path):
    """A learned mask row would be dropped silently, losing acceptance."""
    from mstar.model.components.dspark.drafter import DSparkDrafter

    target, dspark = write_checkpoints(tmp_path)
    model = make_model(target, draft_block(dspark, 3))
    weights = _drafter_weights(model, extra={"mask_embedding": torch.zeros(1, 32)})
    with pytest.raises(RuntimeError, match="mask_embedding"):
        DSparkDrafter(model.drafter_decl).load_weights(weights.items(), target=model.build_lm("cpu"))


def test_an_uncaptured_verifying_node_warms_up_and_decodes(ckpt, monkeypatch):
    """accelerator_graph false captures nothing, and warmup builds the
    sampler's slot masters a verify step writes its tokens to, rather than refusing."""
    target, dspark = ckpt
    monkeypatch.setattr(DSparkDrafter, "propose", forced_drafts(_references(target), lambda j: j == 1))
    model = make_model(target, draft_block(dspark, 3), accelerator_graph=False)
    eng = engine(model, slot_masters=False, warmup=True)
    assert eng._resources["sampler"].has_slot_masters
    out, _ = decode(model, PROMPTS, MAX, eng=eng)
    for rid, prompt in PROMPTS.items():
        assert out[rid] == reference_greedy(model, prompt, MAX)


# ── what drafting turns on and off on the decoder scaffold (#1321's gates) ──


def test_the_built_node_drafts_with_the_module_it_was_given(ckpt):
    """The drafter is an nn.Module: it lives in the node's ``_modules``, and nothing at class
    level may shadow it (the engine refuses a verifying node whose ``drafter`` reads None)."""
    from qwen3_cpu import build_submodule

    target, dspark = ckpt
    sub = build_submodule(make_model(target, draft_block(dspark, 3)))
    from qwen3_cpu import bind_share

    assert isinstance(sub.drafter, DSparkDrafter) and sub.drafter is sub._modules["drafter"]
    # k is the engine's to bind; before it, a verifying node can't say its block
    with pytest.raises(RuntimeError, match="bind_draft"):
        _ = sub._block
    assert bind_share(sub, 3)._block == 4
    plain = build_submodule(make_model(target))
    assert plain.drafter is None and plain._block == 1


def _attention_splits(model):
    from mstar.engine.resources.attn.config import AttentionSpec

    (attn,) = [s for s in model.get_node_resources() if isinstance(s, AttentionSpec) and s.resource_key == "attn"]
    return attn.config.splits


def test_drafting_turns_off_combined_walks_and_the_mixed_split_and_the_engine_turns_off_rebuild(ckpt):
    """The model turns off combined walks and the mixed split; it keeps its
    rebuild policy, which the engine leaves out for a verifying node."""
    from mstar.engine.engine import Engine

    target, dspark = ckpt
    plain, drafting = make_model(target), make_model(target, draft_block(dspark, 3))
    assert plain.get_combined_graph_walks() and plain.rebuild_policies() and _attention_splits(plain)
    assert drafting.get_combined_graph_walks() == {}
    assert drafting.rebuild_policies() == plain.rebuild_policies()
    eng = Engine.__new__(Engine)
    eng._submodules, eng._rebuild_off_logged = {}, set()
    assert eng.rebuild_policies(plain) == plain.rebuild_policies() and eng.rebuild_policies(drafting) == {}
    assert _attention_splits(drafting) == ()


@pytest.mark.parametrize("captured", [True, False])
def test_a_drafting_node_captures_k_plus_one_decode_rows_and_no_mixed_step(ckpt, captured):
    from qwen3_cpu import build_submodule

    from mstar.model.keys import MIXED

    target, dspark = ckpt

    from qwen3_cpu import bind_share

    def configs(draft):
        sub = build_submodule(make_model(target, draft, accelerator_graph=captured))
        if draft is not None:
            # the engine binds k before it asks for the captures
            bind_share(sub, draft["num_draft_tokens"])
        return {c.capture_graph_walk: c for c in sub.get_accelerator_graph_configs(torch.device("cpu"))}

    plain, drafting = configs(None), configs(draft_block(dspark, 3))
    if not captured:
        assert plain == {} and drafting == {}
        return
    assert MIXED in plain and MIXED not in drafting
    assert plain["decode"].single_request_inputs.input_seq_len == 1
    assert drafting["decode"].single_request_inputs.input_seq_len == 3 + 1


def test_a_checkpoint_defaulting_a_repetition_penalty_is_refused_when_drafting(ckpt):
    """It would be every request's default, which a verifying sampler refuses: at startup, with
    the reason, rather than a 400 on every request that sets none. Without a draft block the
    penalty is served."""
    import json

    target, dspark = ckpt
    (target / "generation_config.json").write_text(json.dumps({"do_sample": False, "repetition_penalty": 1.1}))
    plain = make_model(target)
    assert plain.verify_nodes() == []
    assert plain.get_request_resource_configs(["text"], ["text"], {})["sampler"].repetition_penalty == 1.1
    with pytest.raises(ValueError, match="repetition_penalty to 1.1"):
        make_model(target, draft_block(dspark, 3)).verify_nodes()


# the scaffold's adopters (module, class), and those whose decode reads the routed token
_ADOPTERS = [
    ("mstar.model.qwen3.submodules", "Qwen3LMSubmodule"),
    ("mstar.model.qwen3_5.submodules", "LLMSubmodule"),
    ("mstar.model.ripple.submodules", "RippleLLMSubmodule"),
    ("mstar.model.muse.submodules", "LLMSubmodule"),
    ("mstar.model.higgs_audio.submodules", "HiggsAudioLLMSubmodule"),
    ("mstar.model.qwen3_asr.submodules", "Qwen3ASRLLMSubmodule"),
    ("mstar.model.cosmos3.submodules", "Cosmos3ReasonerSubmodule"),
    ("mstar.model.orpheus.submodules", "OrpheusLLMSubmodule"),
    ("mstar.model.command_a_plus.submodules", "CommandAPlusLLMSubmodule"),
    ("mstar.model.canary_qwen.submodules", "CanaryQwenLLMSubmodule"),
    ("mstar.model.canary.submodules", "CanaryDecoderSubmodule"),
]
_ROUTED = {"OrpheusLLMSubmodule", "CommandAPlusLLMSubmodule", "CanaryQwenLLMSubmodule", "CanaryDecoderSubmodule"}


def _adopter(module: str, name: str):
    import importlib

    try:
        return getattr(importlib.import_module(module), name)
    except ImportError as exc:
        pytest.skip(f"{module} needs an optional dependency: {exc}")


@pytest.mark.parametrize("module, name", _ADOPTERS, ids=[f"{m.split('.')[2]}.{n}" for m, n in _ADOPTERS])
def test_the_declared_loopback_is_the_nodes_own(module, name):
    """A model's verify node declares ``node_device_loopback()``, read off the class: None on a
    node that reads its routed token, which drafting then refuses rather than switching on."""
    cls = _adopter(module, name)
    loopback = cls.node_device_loopback()
    assert (loopback is None) == (name in _ROUTED), loopback
    if loopback is not None:
        assert (loopback.walk, loopback.signal, loopback.buffer) == (cls.DECODE_WALK, "text_inputs", "new_token")


@pytest.mark.parametrize("module, name", _ADOPTERS, ids=[f"{m.split('.')[2]}.{n}" for m, n in _ADOPTERS])
@pytest.mark.parametrize("text_prompts", [True, False])
def test_the_declared_token_stop_is_the_one_the_node_runs(module, name, text_prompts):
    """``node_token_stop`` gives what the node's `__init__` sets, and no adopter sets its own
    after it, so a verify node declared before any submodule exists stops the node's way."""
    import ast
    import inspect
    from types import SimpleNamespace

    from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
    from mstar.model.multimodal import vision_encoder

    cls = _adopter(module, name)
    tree = ast.parse(inspect.getsource(cls))
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Attribute) and t.attr == "token_stop" for t in n.targets)]
    assert not assigns, f"{name} sets its own token_stop"
    encoders = (vision_encoder(),)
    node = object.__new__(cls)
    SampledTokenDecoder.__init__(
        node, SimpleNamespace(embed_dtype=torch.bfloat16), SimpleNamespace(hidden_size=8, stop_token_ids=(1, 2)),
        encoders, text_prompts=text_prompts,
    )
    assert cls.node_token_stop((1, 2), encoders, text_prompts) == node.token_stop
