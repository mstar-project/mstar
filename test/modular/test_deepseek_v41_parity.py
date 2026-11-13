"""DeepSeek-V4.1 through the real ``Engine`` on CPU against the checkpoint's own reference.

A reduced model with random weights, written in the checkpoint's names, dtypes and packing, is
loaded by M* (its loader, resources and submodule) and by the reference ``inference/model.py``
(through its ``convert.py``; its tilelang kernels replaced by torch, ``_testing``). Both run a
prefill and teacher-forced decode steps, and the logits are compared step by step.

The reference is the checkpoint's code, which CI does not have: set ``DSV41_REFERENCE_DIR`` to a
snapshot directory holding ``inference/`` (the model repo's small files) to run these.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.csa import manager as csa_manager
from mstar.engine.resources.csa.config import Fp4Rows
from mstar.model.deepseek_v41 import _testing
from mstar.model.deepseek_v41._testing import add_request, cpu_engine, run_request, run_step
from mstar.model.deepseek_v41._testing import logit_gap as _gap
from mstar.model.deepseek_v41._testing import random_ids as _ids
from mstar.model.deepseek_v41.components import attention
from mstar.model.deepseek_v41.config import MOE_EXPERTS, DeepseekV41Config

REFERENCE_DIR = os.environ.get("DSV41_REFERENCE_DIR")
pytestmark = [
    pytest.mark.skipif(
        not REFERENCE_DIR or not (Path(REFERENCE_DIR) / "inference" / "model.py").is_file(),
        reason="needs the checkpoint's inference/ code: set DSV41_REFERENCE_DIR to a snapshot directory"),
]

PROMPT, STEPS = 45, 14
# fp32 on both sides leaves summation order only: about 1e-6 of the logits' spread, measured
FP32_TOL = 2e-5


@pytest.fixture(autouse=True)
def _cpu_engine(monkeypatch):
    _testing.patch_cpu_engine(monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def reference():
    pytest.importorskip("sympy")  # the reference's prime search
    return _testing.load_reference(REFERENCE_DIR)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    path = tmp_path_factory.mktemp("dsv41")
    config, tokenizer = _testing.write_snapshot(path, DeepseekV41Config.reduced(), seed=0)
    return path, config, tokenizer


def _reference_logits(model, ids: torch.Tensor, prompt: int) -> torch.Tensor:
    out = [model(ids[None, :prompt], 0)[1][0]]
    for i in range(prompt, ids.numel()):
        out.append(model(ids[None, i:i + 1], i)[1][0])
    return torch.stack(out).float()


def _reference_run(reference, snapshot, tmp_path, dtype: torch.dtype, ids: torch.Tensor, prompt: int):
    path, config, tokenizer = snapshot
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = _testing.reference_model(reference, path, config, tokenizer, tmp_path)
        return _reference_logits(model, ids, prompt)
    finally:
        torch.set_default_dtype(default)


def _exact_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """The reference's attention as one fp32 softmax: an equally valid kernel for it."""
    batch = torch.arange(q.shape[0])[:, None, None]
    rows = kv[batch, topk_idxs.clamp_min(0).long()].float()
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), rows) * softmax_scale
    scores = scores.masked_fill(~(topk_idxs != -1)[:, :, None, :], -torch.inf)
    sink = attn_sink.float()[None, None, :, None].expand(*scores.shape[:3], 1)
    prob = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", prob, rows).to(q.dtype)


@pytest.fixture(scope="module")
def fp32_reference(reference, snapshot, tmp_path_factory):
    ids = _ids(snapshot[1], PROMPT + STEPS, seed=1)
    return ids, _reference_run(reference, snapshot, tmp_path_factory.mktemp("ref32"), torch.float32, ids, PROMPT)


class TestParity:
    @pytest.mark.parametrize("moe_experts", MOE_EXPERTS)
    def test_fp32_logits_match_the_reference_step_by_step(self, snapshot, fp32_reference, moe_experts):
        ids, want = fp32_reference
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32, moe_experts=moe_experts)
        got = run_request(engine, sampler, ids, PROMPT)
        assert got.shape == want.shape
        assert torch.equal(got.argmax(-1), want.argmax(-1))
        assert _gap(got, want) < FP32_TOL, _gap(got, want)

    def test_fp32_logits_match_the_reference_with_the_engram_tables_on_the_host(self, snapshot, fp32_reference):
        ids, want = fp32_reference
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32, engram_tables="host")
        got = run_request(engine, sampler, ids, PROMPT)
        assert torch.equal(got.argmax(-1), want.argmax(-1))
        assert _gap(got, want) < FP32_TOL, _gap(got, want)

    def test_bf16_stays_within_the_references_own_spread(self, reference, snapshot, tmp_path, monkeypatch):
        """bf16 as shipped. With random weights at this size nothing has a margin, so a
        rounding moves a top-k and the logits with it: the reference itself moves by 2.6 to
        4.3 sigma when only its attention kernel is swapped for an exact softmax (seeds 0-2).
        M* is held to that spread, which shows it runs in bf16, not that it is right: the
        fp32 case above is the gate, and bf16 parity needs real weights."""
        ids = _ids(snapshot[1], PROMPT + STEPS, seed=1)
        want = _reference_run(reference, snapshot, tmp_path / "a", torch.bfloat16, ids, PROMPT)
        monkeypatch.setattr(reference.model, "sparse_attn", _exact_sparse_attn)
        other = _reference_run(reference, snapshot, tmp_path / "b", torch.bfloat16, ids, PROMPT)
        engine, sampler, _ = cpu_engine(snapshot[0], torch.bfloat16)
        got = run_request(engine, sampler, ids, PROMPT)
        assert torch.isfinite(got).all()
        spread = _gap(other, want)
        assert spread > 0  # the two reference kernels did differ
        assert _gap(got, want) < 2 * spread, (_gap(got, want), spread)

    def test_two_requests_in_one_step_each_match_their_own_reference(
            self, reference, snapshot, fp32_reference, tmp_path):
        ids_a, want_a = fp32_reference
        short = 22
        ids_b = _ids(snapshot[1], short + STEPS, seed=2)
        want_b = _reference_run(reference, snapshot, tmp_path, torch.float32, ids_b, short)
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        for rid in "ab":
            add_request(engine, rid)
        rows = [run_step(engine, sampler, "prefill", {"a": ids_a[:PROMPT], "b": ids_b[:short]})]
        for step in range(STEPS):
            rows.append(run_step(engine, sampler, "decode", {
                "a": ids_a[PROMPT + step].view(1), "b": ids_b[short + step].view(1)}))
        assert _gap(torch.stack([r["a"] for r in rows]), want_a) < FP32_TOL
        assert _gap(torch.stack([r["b"] for r in rows]), want_b) < FP32_TOL

    def test_a_prompt_fed_in_two_steps_matches_one(self, snapshot, fp32_reference):
        ids, want = fp32_reference
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        add_request(engine, "r")
        cut = 19  # inside a window and a pooled group
        run_step(engine, sampler, "prefill", {"r": ids[:cut]})
        got = run_step(engine, sampler, "prefill", {"r": ids[cut:PROMPT]})["r"]
        assert _gap(got[None], want[:1]) < FP32_TOL


@pytest.mark.parametrize("geometry", [
    # the checkpoint's Engram shape: two layers, 2- to 4-grams
    dict(engram_layer_ids=(0, 5), engram_num_embeddings=(0, 0), engram_max_ngram_size=4, engram_n_heads=3),
    dict(hc_mult=2, num_experts_per_tok=3, sliding_window=5),
    dict(candidate_source_layer_id=-1),
    dict(compress_ratios=(0, 4, 4, 4, 1, 1, 1, 0), index_topk=4, candidate_topk_blocks=2, candidate_block_size=4),
], ids=["engram-2-layers-4-grams", "2-streams-window-5", "no-candidates", "ratio-4"])
def test_other_geometries_match_the_reference(reference, geometry, tmp_path):
    config, tokenizer = _testing.write_snapshot(tmp_path / "snap", DeepseekV41Config.reduced(**geometry), seed=3)
    ids = _ids(config, 41 + 12, seed=7)
    want = _reference_run(reference, (tmp_path / "snap", config, tokenizer), tmp_path / "ref", torch.float32, ids, 41)
    engine, sampler, _ = cpu_engine(tmp_path / "snap", torch.float32)
    assert _gap(run_request(engine, sampler, ids, 41), want) < FP32_TOL
    if len(config.engram_layer_ids) > 1:
        # and with both tables fetched from the host in one trip a step
        engine, sampler, _ = cpu_engine(tmp_path / "snap", torch.float32, engram_tables="host")
        assert _gap(run_request(engine, sampler, ids, 41), want) < FP32_TOL


@pytest.mark.skipif(
    not REFERENCE_DIR or not (Path(REFERENCE_DIR or ".") / "tokenizer.json").is_file(),
    reason="needs the checkpoint's tokenizer.json in DSV41_REFERENCE_DIR")
def test_the_checkpoints_tokenizer_compresses_to_the_vocabulary_its_tables_were_hashed_over():
    """Every Engram hash multiplier derives from this count, so it has to be the checkpoint's."""
    import json

    from transformers import AutoTokenizer

    from mstar.model.deepseek_v41.components.engram import compressed_token_map

    lookup, count = compressed_token_map(AutoTokenizer.from_pretrained(REFERENCE_DIR))
    config = DeepseekV41Config.from_hf_config(json.loads((Path(REFERENCE_DIR) / "config.json").read_text()))
    assert (len(lookup), count) == (config.vocab_size, config.engram_compressed_vocab_size)


class TestTheGateCanFail:
    """Each mechanism switched off moves the logits past the fp32 tolerance."""

    def _gap_with(self, snapshot, fp32_reference) -> float:
        ids, want = fp32_reference
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        return _gap(run_request(engine, sampler, ids, PROMPT), want)

    def test_without_the_candidate_blocks(self, monkeypatch, snapshot, fp32_reference):
        monkeypatch.setattr(csa_manager, "candidate_blocks",
                            lambda scores, *a, **k: torch.ones_like(scores, dtype=torch.bool))
        assert self._gap_with(snapshot, fp32_reference) > 10 * FP32_TOL

    def test_with_the_caches_latents_on_another_fp4_grid(self, monkeypatch, snapshot, fp32_reference):
        # power-of-two scales where the checkpoint's are E4M3: the same bytes, other values
        monkeypatch.setattr(DeepseekV41Config, "latent_rows", property(lambda self: Fp4Rows(block=16)))
        assert self._gap_with(snapshot, fp32_reference) > 10 * FP32_TOL

    def test_without_the_fp4_rounding_of_the_index_queries(self, monkeypatch, snapshot, fp32_reference):
        monkeypatch.setattr(attention, "round_fp4", lambda x, *a, **k: x)
        assert self._gap_with(snapshot, fp32_reference) > 10 * FP32_TOL

    def test_without_the_sink(self, monkeypatch, snapshot, fp32_reference):
        attend = csa_manager.CSAManager.attend
        monkeypatch.setattr(csa_manager.CSAManager, "attend", lambda self, q, sink, *a, **k: attend(
            self, q, torch.full_like(sink, -1e30), *a, **k))
        assert self._gap_with(snapshot, fp32_reference) > 10 * FP32_TOL
