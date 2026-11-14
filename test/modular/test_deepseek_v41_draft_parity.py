"""DeepSeek-V4.1's draft head through the real ``Engine`` on CPU against the checkpoint's own
``forward_spec``.

A reduced model with random weights in the checkpoint's names and packing, its draft layers
(``mtp.*``) included, is loaded by M* with ``draft_head`` and by the reference. Both run a
prefill, which only fills the draft layers' windows, and teacher-forced decode steps. After
each decode step both draft a block from the token the step sampled, and the block's logits
(Markov correction included), drafted ids and confidences are compared.

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

from mstar.model.deepseek_v41 import _testing
from mstar.model.deepseek_v41._testing import cpu_engine, record_drafts, run_request
from mstar.model.deepseek_v41._testing import logit_gap as _gap
from mstar.model.deepseek_v41._testing import random_ids as _ids
from mstar.model.deepseek_v41.components import draft
from mstar.model.deepseek_v41.components.layers import DeepseekV41ForCausalLM
from mstar.model.deepseek_v41.config import DeepseekV41Config

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


def _snapshot(path, seed: int = 0, **geometry):
    config, tokenizer = _testing.write_snapshot(path, DeepseekV41Config.reduced(**geometry), seed, draft_head=True)
    return path, config, tokenizer


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return _snapshot(tmp_path_factory.mktemp("dsv41-draft"))


def _reference_run(reference, snapshot, workdir, dtype: torch.dtype, ids: torch.Tensor, prompt: int):
    """The reference's logits of every step, and after each decode step the block
    ``forward_spec`` drafts from the token that step sampled: ``(ids, logits, confidence)``."""
    path, config, tokenizer = snapshot
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = _testing.reference_model(reference, path, config, tokenizer, workdir, draft_head=True)
        token, logits, hidden = model(ids[None, :prompt], 0)
        assert model.forward_spec(token, hidden, 0) is None  # a prefill only fills the windows
        steps, drafts = [logits[0]], []
        for i in range(prompt, ids.numel()):
            token, logits, hidden = model(ids[None, i:i + 1], i)
            drafted, draft_logits, confidence = model.forward_spec(token, hidden, i)
            assert torch.equal(drafted[:, 0], token)
            steps.append(logits[0])
            drafts.append((drafted[0, 1:], draft_logits[0].float(), confidence[0].float()))
        return torch.stack(steps).float(), drafts
    finally:
        torch.set_default_dtype(default)


def _run(path, dtype: torch.dtype, ids: torch.Tensor, prompt: int):
    """M*'s logits of every step, and the block its draft head drafts after each decode step."""
    engine, sampler, model = cpu_engine(path, dtype, draft_head=True)
    drafts = record_drafts(model)
    logits = run_request(engine, sampler, ids, prompt)
    assert len(drafts) == ids.numel() - prompt + 1  # the prefill's, then one a decode step
    return logits, [(d.ids[0], d.logits[0], d.confidence[0]) for d in drafts[1:]]


def _draft_gap(got, want) -> float:
    """The largest gap over the steps' blocks: logits in units of their spread, confidences in
    units of theirs over the run."""
    spread = torch.stack([confidence for _, _, confidence in want]).std()
    return max(max(_gap(g[1], w[1]), ((g[2] - w[2]).abs().max() / spread).item())
               for g, w in zip(got, want, strict=True))


@pytest.fixture(scope="module")
def fp32_reference(reference, snapshot, tmp_path_factory):
    ids = _ids(snapshot[1], PROMPT + STEPS, seed=1)
    return ids, _reference_run(reference, snapshot, tmp_path_factory.mktemp("ref32"), torch.float32, ids, PROMPT)


class TestDraftParity:
    def test_fp32_drafts_match_forward_spec_step_by_step(self, snapshot, fp32_reference):
        """Measured: the blocks' logits and confidences are the reference's bit for bit."""
        ids, (want_logits, want) = fp32_reference
        logits, got = _run(snapshot[0], torch.float32, ids, PROMPT)
        assert _gap(logits, want_logits) < FP32_TOL
        assert len(got) == len(want) == STEPS
        for (got_ids, _, _), (want_ids, _, _) in zip(got, want, strict=True):
            assert torch.equal(got_ids, want_ids)
        assert _draft_gap(got, want) < FP32_TOL, _draft_gap(got, want)

    def test_the_head_leaves_the_backbones_logits_alone(self, snapshot, fp32_reference):
        """A snapshot that ships its draft layers serves the same logits with them unloaded."""
        ids, _ = fp32_reference
        on, _ = _run(snapshot[0], torch.float32, ids, PROMPT)
        engine, sampler, _ = cpu_engine(snapshot[0], torch.float32)
        assert torch.equal(run_request(engine, sampler, ids, PROMPT), on)

    def _bf16(self, reference, tmp_path, seed):
        snapshot = _snapshot(tmp_path / "snap", seed, **_testing.KERNEL_WIDTHS)
        ids = _ids(snapshot[1], PROMPT + STEPS, seed=seed + 1)
        want_logits, want = _reference_run(reference, snapshot, tmp_path / "ref", torch.bfloat16, ids, PROMPT)
        logits, got = _run(snapshot[0], torch.bfloat16, ids, PROMPT)
        assert torch.equal(logits, want_logits)
        return got, want

    @pytest.mark.parametrize("seed", [0, 2])
    def test_bf16_drafts_are_forward_specs_bit_for_bit(self, reference, tmp_path, seed):
        """bf16 as shipped, on the wider reduced model (the backbone's own bit-identity holds
        there): the draft rows' entries are laid out as the reference's ring, so the block-wise
        softmax rounds alike."""
        got, want = self._bf16(reference, tmp_path, seed)
        for step, (g, w) in enumerate(zip(got, want, strict=True)):
            for name, a, b in zip(("ids", "logits", "confidence"), g, w, strict=True):
                assert torch.equal(a, b), f"step {step}: {name}"

    def test_bf16_drafts_leave_forward_spec_only_at_a_rounding_tie(self, reference, tmp_path):
        """Seed 1, decode step 6: one output of a draft layer's ``wo_b`` sums to within 6e-8 of
        the midpoint of two bf16 values. M* sums a row in one fp32 GEMM and rounds up; the
        reference's GEMM sums per 32-wide scale block, lands on the midpoint and rounds to
        even. That block's last row moves by 7e-4 of the logits' spread, and nothing else in
        the 14 steps differs."""
        got, want = self._bf16(reference, tmp_path, seed=1)
        off = [step for step, (g, w) in enumerate(zip(got, want, strict=True)) if not torch.equal(g[1], w[1])]
        assert len(off) <= 1
        assert all(torch.equal(g[0], w[0]) for g, w in zip(got, want, strict=True))
        assert max(_gap(g[1], w[1]) for g, w in zip(got, want, strict=True)) < 2e-3


@pytest.mark.parametrize("geometry", [
    dict(dspark_block_size=6),
    dict(dspark_target_layer_ids=(6,)),
    dict(dspark_target_layer_ids=(3, 5), dspark_block_size=6),
    # one draft layer, which is first and last; top-3 experts; a window the prompt has not filled
    dict(num_nextn_predict_layers=1, compress_ratios=(0, 2, 2, 2, 1, 1, 1, 0), dspark_num_experts_per_tok=3),
], ids=["block-6", "one-target", "two-targets-block-6", "one-draft-layer"])
def test_other_draft_geometries_match_forward_spec(reference, geometry, tmp_path):
    snapshot = _snapshot(tmp_path / "snap", seed=3, **geometry)
    ids = _ids(snapshot[1], 41 + 12, seed=7)
    want_logits, want = _reference_run(reference, snapshot, tmp_path / "ref", torch.float32, ids, 41)
    logits, got = _run(snapshot[0], torch.float32, ids, 41)
    assert _gap(logits, want_logits) < FP32_TOL
    assert got[0][1].shape[0] == snapshot[1].dspark_block_size
    for (got_ids, _, _), (want_ids, _, _) in zip(got, want, strict=True):
        assert torch.equal(got_ids, want_ids)
    assert _draft_gap(got, want) < FP32_TOL, _draft_gap(got, want)


def test_a_window_still_filling_matches_forward_spec(reference, tmp_path):
    """A prompt shorter than the window: the draft rows attend the tokens there are."""
    snapshot = _snapshot(tmp_path / "snap", seed=5, sliding_window=16)
    ids = _ids(snapshot[1], 5 + 14, seed=9)
    _, want = _reference_run(reference, snapshot, tmp_path / "ref", torch.float32, ids, 5)
    _, got = _run(snapshot[0], torch.float32, ids, 5)
    for (got_ids, _, _), (want_ids, _, _) in zip(got, want, strict=True):
        assert torch.equal(got_ids, want_ids)
    assert _draft_gap(got, want) < FP32_TOL, _draft_gap(got, want)


class TestTheDraftGateCanFail:
    """Each mechanism of the head switched off moves its drafts past the fp32 tolerance."""

    def _gap_with(self, snapshot, fp32_reference, edit=None) -> float:
        ids, (_, want) = fp32_reference
        engine, sampler, model = cpu_engine(snapshot[0], torch.float32, draft_head=True)
        drafts = record_drafts(model)
        if edit is not None:
            edit(model.get_submodule("LLM").language_model)
        run_request(engine, sampler, ids, PROMPT)
        return _draft_gap([(d.ids[0], d.logits[0], d.confidence[0]) for d in drafts[1:]], want)

    def test_as_built_it_passes(self, snapshot, fp32_reference):
        assert self._gap_with(snapshot, fp32_reference) < FP32_TOL

    def test_without_the_markov_correction(self, monkeypatch, snapshot, fp32_reference):
        forward = draft.MarkovHead.forward

        def no_bias(self, tokens):
            bias, embedding = forward(self, tokens)
            return torch.zeros_like(bias), embedding

        monkeypatch.setattr(draft.MarkovHead, "forward", no_bias)
        assert self._gap_with(snapshot, fp32_reference) > 10 * FP32_TOL

    def test_with_the_window_not_updated(self, monkeypatch, snapshot, fp32_reference):
        """The draft layers' windows keep what the prefill left: no decode step's hidden state."""
        keep = draft.DraftAttention._keep

        def prompt_only(self, main, rows, rope):
            ring = keep(self, main, rows, rope)
            if int(rows.req_rows.max()) > 1:
                self.prompt_ring = ring.clone()
            return self.prompt_ring

        monkeypatch.setattr(draft.DraftAttention, "_keep", prompt_only)
        assert self._gap_with(snapshot, fp32_reference) > 10 * FP32_TOL

    def test_with_the_taps_taken_after_their_layers(self, monkeypatch, snapshot, fp32_reference):
        """The head reads the streams entering each target layer, not the ones leaving it."""
        def edit(lm):
            after = []
            for layer in lm.config.dspark_target_layer_ids:
                lm.model.layers[layer].register_forward_hook(lambda module, args, out: after.append(out[0].mean(dim=1)))
            head = DeepseekV41ForCausalLM.draft

            def draft_after(self, tokens, taps, *args):
                taken = [after.pop(0) for _ in taps]
                return head(self, tokens, taken, *args)

            monkeypatch.setattr(DeepseekV41ForCausalLM, "draft", draft_after)

        assert self._gap_with(snapshot, fp32_reference, edit) > 10 * FP32_TOL
