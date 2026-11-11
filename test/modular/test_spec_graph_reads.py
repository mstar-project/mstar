"""The engine binds a verifying node's graph reads on its sampler (`Engine._set_up_draft`, `verify_writes`): a leased
step of the verify walk stages the drafter's loop-back state off the masters, and the forward reads the staged rows.
Built through the real engine load, for a lagged drafter (a Qwen3 draft model on the CPU harness: its carry and its
keep flag) and for a carrying toy with no lag."""
from __future__ import annotations

import json

import pytest
import torch
from qwen3_cpu import engine as qwen3_engine
from qwen3_cpu import make_model, patch_cpu, write_checkpoints

from mstar.engine.resources.draft.drafter import DRAFT_CARRY, DRAFT_KEEP_FIRST
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.model.keys import PREFILL_TEXT as PREFILL_WALK

# the decoder scaffold's decode walk, which Qwen3 verifies on
DECODE_WALK = "decode"

DRAFT = dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=97,
             rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=512, tie_word_embeddings=True,
             torch_dtype="float32", use_sliding_window=False)


@pytest.fixture
def toy_engine(monkeypatch):
    """`spec_toy.engine`, its KV transfer stubbed as `test_spec_engine` does."""
    from spec_toy import engine
    from test_spec_engine import _StubTransfer

    from mstar.engine.resources.kv import manager as manager_mod

    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    return engine


def _leased(rids, walk: str, bs: int = 4, slot: int = 0) -> StepContext:
    """A replayed step's context, a real one so every field the resources read has its default."""
    lease = SlotLease(slot=slot, bucket=BucketKey(graph_walk=walk, bs=bs, num_tokens=bs))
    ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=slot, capture=False, slot_lease=lease)
    # a replay's padding rows carry negative handles
    ctx.set_padded_rids(tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids))))
    return ctx


def _draft_model_engine(tmp_path, monkeypatch):
    patch_cpu(monkeypatch)
    target, _ = write_checkpoints(tmp_path)
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(DRAFT))
    model = make_model(target, {"method": "draft_model", "model": str(draft), "num_draft_tokens": 3})
    return qwen3_engine(model)


def _check_staged(sampler, walk: str, names: frozenset[str]) -> None:
    for rid in (1, 2):
        sampler.ingest_request(rid)
        if sampler.ask(rid, "LLM", walk) is not None:
            assert sampler.reserve(rid, "LLM", walk, None) is None
    for name in names:
        width = sampler._cg_buffers.loopback[name].master.shape[1]
        sampler.store_loopback_state(name, [1, 2], torch.tensor([[1] * width, [2] * width]))
    sampler.plan(SamplerStep(), _leased([2, 1], walk))
    for name in names:
        rows = sampler.loopback_state(name, [2, 1])
        staged = sampler._cg_buffers.loopback[name].read_view(0, 4)
        assert rows.data_ptr() == staged.data_ptr(), f"{name} is not read from its staged rows"
        assert rows[:2, 0].tolist() == [2, 1]


def test_a_draft_models_verify_walk_stages_its_carry_and_keep_flag(tmp_path, monkeypatch):
    eng = _draft_model_engine(tmp_path, monkeypatch)
    sampler = eng._resources["sampler"]
    assert sampler._graph_reads == {DECODE_WALK: frozenset({DRAFT_CARRY, DRAFT_KEEP_FIRST})}
    _check_staged(sampler, DECODE_WALK, frozenset({DRAFT_CARRY, DRAFT_KEEP_FIRST}))
    # a prompt step stages nothing: it stores the first carry, it reads none
    sampler.plan(SamplerStep(), _leased([2, 1], PREFILL_WALK))
    assert sampler._staged == frozenset()


def test_a_carrying_drafter_with_no_lag_stages_only_its_carry(toy_engine):
    from spec_toy import carry_oracle

    eng = toy_engine(3, carry_oracle)
    sampler = eng._resources["sampler"]
    assert sampler._graph_reads == {"decode": frozenset({DRAFT_CARRY})}
    _check_staged(sampler, "decode", frozenset({DRAFT_CARRY}))


def test_another_walks_leased_step_reads_the_masters(toy_engine):
    from spec_toy import carry_oracle

    walk = "prefill"
    sampler = toy_engine(3, carry_oracle)._resources["sampler"]
    sampler.ingest_request(1)
    if sampler.ask(1, "LLM", walk) is not None:
        sampler.reserve(1, "LLM", walk, None)
    sampler.plan(SamplerStep(), _leased([1], walk))
    rows = sampler.loopback_state(DRAFT_CARRY, [1])
    assert rows.data_ptr() != sampler._cg_buffers.loopback[DRAFT_CARRY].read_view(0, 4).data_ptr()
