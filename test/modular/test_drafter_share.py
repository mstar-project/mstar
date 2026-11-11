"""The engine owns a drafter's share of its node's steps (`drafter_step`, from `engine_step`): on each prompt and
verify step, the drafter's own resources' steps (`VerifyNode.prefill_steps` / `verify_steps`) and the loop-back
state it stores through the node's sampler step (`carry_writes` / `verify_writes`). The target declares none of
it. Checked for every drafter on the CPU harness's real engine load: DSpark, EAGLE-3 and a draft model on Qwen3."""
from __future__ import annotations

import dataclasses
import json

import pytest
import torch
from qwen3_cpu import engine as qwen3_engine
from qwen3_cpu import make_model, patch_cpu, write_checkpoints

from mstar.engine.resources.draft.drafter import DRAFT_CARRY, DRAFT_KEEP_FIRST, carry_writes, verify_writes
from mstar.engine.resources.step import StepContext, SubmoduleStep
from mstar.model.submodule_base import ARNodeInputs, engine_step

DRAFT_MODEL = dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=97,
                   rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=True,
                   torch_dtype="float32", use_sliding_window=False)
# the EAGLE authors' Llama-style format, at the CPU target's sizes
EAGLE3 = {"architectures": ["LlamaForCausalLMEagle3"], "draft_vocab_size": 97, "hidden_size": 32,
          "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 1,
          "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 10_000.0, "vocab_size": 97, "torch_dtype": "float32",
          "eagle_config": {"eagle_aux_hidden_state_layer_ids": [0, 1, 2]}}


def _qwen3_engine(tmp_path, monkeypatch, method: str):
    patch_cpu(monkeypatch)
    target, dspark = write_checkpoints(tmp_path)
    if method == "dspark":
        return qwen3_engine(make_model(target, {"method": "dspark", "model": str(dspark), "num_draft_tokens": 3}))
    draft = tmp_path / method
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(DRAFT_MODEL if method == "draft_model" else EAGLE3))
    return qwen3_engine(make_model(target, {"method": method, "model": str(draft), "num_draft_tokens": 3}))


def _old_step(declared, sampler: str, writes, own, lag, walk: str):
    """The step a target declared itself before the engine owned the drafter's share: its sampler step listing
    the writes, the drafter's steps after its own, the lag copied in by the engine. Built as a `SubmoduleStep`,
    whose None segments mean the step's."""
    steps = {**declared.steps, sampler: dataclasses.replace(declared.steps[sampler], loopback_writes=writes),
             **{key: dataclasses.replace(value) for key, value in own.items()}}
    old = SubmoduleStep(segments=declared.segments, steps=steps)
    if lag is not None and walk == lag.walk:
        for key, value in ((lag.kv_key, {lag.layer_set: lag.lag}), (lag.pos_key, lag.lag), (lag.attn_key, lag.lag)):
            steps[key] = dataclasses.replace(steps[key], lag=value)
        old = dataclasses.replace(old, steps=steps)
    return old


def _walk_kwargs(draft, walk: str) -> dict:
    n = draft.decl.config.block if walk == draft.walk else 3
    return dict(graph_walk=walk, request_ids=[1, 2], per_request_info={},
                inputs=[ARNodeInputs(input_ids=torch.tensor([5] * n), input_seq_len=n)] * 2)


@pytest.mark.parametrize("method", ["dspark", "eagle3", "draft_model"])
def test_each_walks_step_is_the_one_its_target_declared_before_and_the_target_declares_no_share(
    tmp_path, monkeypatch, method,
):
    eng = _qwen3_engine(tmp_path, monkeypatch, method)
    (node, draft), = eng._drafts.items()
    decl, sub, sampler = draft.decl, eng._submodules[node].submodule, draft.sampler_key
    walks = {walk: (decl.prefill_steps, carry_writes(decl.needs)) for walk in draft.token_stop.prefill_walks}
    walks[draft.walk] = (decl.verify_steps, verify_writes(decl.needs))
    for walk, (own, writes) in walks.items():
        kwargs = _walk_kwargs(draft, walk)
        declared, step = sub.declare_step(**kwargs), engine_step(sub, **kwargs)
        assert not declared.steps.keys() & own.keys(), walk
        assert declared.steps[sampler].loopback_writes == frozenset(), walk
        old = _old_step(declared, sampler, writes, own, draft.lag, walk)
        assert list(step.steps) == list(old.steps) and step == old, walk
    if method == "draft_model":
        # a lagged drafter's keep flag, which only a verify step sets
        assert walks[draft.walk][1] == {DRAFT_CARRY, DRAFT_KEEP_FIRST}


def test_a_pre_plans_key_is_the_real_steps_on_a_prompt_walk(tmp_path, monkeypatch):
    """The runner matches a staged pre-plan to the step that runs by its key, every resource's segments
    included: both are declared through `engine_step`, so the sampler step's rows are the same in both."""
    eng = _qwen3_engine(tmp_path, monkeypatch, "dspark")
    (node, draft), = eng._drafts.items()
    sub = eng._submodules[node].submodule
    keys = []
    for preplan in (True, False):
        step = engine_step(sub, **_walk_kwargs(draft, "prefill_text"))
        step.set_ctx(StepContext(request_ids=(1, 2), graph_walk="prefill_text", slot=0, capture=False,
                                 is_preplan=preplan))
        keys.append(eng._runner._step_key(step))
    assert keys[0] == keys[1]
    assert dict(keys[0][-1])[draft.sampler_key] == tuple(step.segments)


def test_a_target_that_declares_its_drafters_step_itself_is_refused(tmp_path, monkeypatch):
    eng = _qwen3_engine(tmp_path, monkeypatch, "dspark")
    (node, draft), = eng._drafts.items()
    sub = eng._submodules[node].submodule
    real = sub.declare_step

    def doubled(*args, **kwargs):
        step = real(*args, **kwargs)
        return type(step)(segments=step.segments, steps={**step.steps, **draft.decl.verify_steps})

    sub.declare_step = doubled
    block = draft.decl.config.block
    with pytest.raises(RuntimeError, match="its drafter's steps, which the engine adds"):
        engine_step(sub, graph_walk=draft.walk, request_ids=[1],
                    inputs=[ARNodeInputs(input_ids=torch.tensor([5] * block), input_seq_len=block)])
