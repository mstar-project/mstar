"""DeepSeek-V4.1 greedy decoding by draft and verify, through the real ``Engine`` on CPU.

A decode step carries a request's last emitted token and drafts of the tokens after it; the
drafts that hold are kept and the rest taken back out of the request's context. Held here to
plain greedy decoding of the same reduced random-weight model: the tokens are the same, and
after any number of kept and rejected drafts the request is where a request that decoded the
same tokens one at a time is.

Random weights make a poor draft head (it is right about once in a vocabulary), so the
drafts mostly come from a stand-in that knows the continuation and is wrong at a set rate:
every count of kept drafts is run. What is exact is the mechanics, not a speed-up.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.deepseek_v41 import _testing
from mstar.model.deepseek_v41._testing import (
    add_request,
    cpu_engine,
    draft_verify,
    greedy,
    likely_drafter,
    logit_gap,
    random_ids,
    record_drafts,
    record_logits,
    run_step,
)
from mstar.model.deepseek_v41.config import CSA, DeepseekV41Config, kv_cache_key

NEW = 40
# stand-in drafters, right three times in four, whose first NEW tokens keep every count of drafts
DRAFTER_SEEDS = (16, 26, 11)
# a row of a block of rows and the same row run alone sum in another order: about 1e-6 of the
# logits' spread, measured
BLOCK_TOL = 2e-5


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    _testing.patch_cpu_engine(monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module", params=[0, 1, 2])
def case(request, tmp_path_factory):
    """A snapshot with its draft layers, a prompt, and its plain greedy continuation with the
    logits of every token and of the step after the last."""
    seed = request.param
    path = tmp_path_factory.mktemp(f"dsv41-verify-{seed}")
    config, _ = _testing.write_snapshot(path, DeepseekV41Config.reduced(), seed=seed, draft_head=True)
    # one prompt past the window, one that leaves it filling
    prompt = random_ids(config, (21, 3, 12)[seed], seed=seed + 10)
    with pytest.MonkeyPatch.context() as patch:
        _testing.patch_cpu_engine(patch)
        engine, sampler, model = cpu_engine(path, torch.float32, draft_head=True)
        heads = record_drafts(model)
        tokens, logits = greedy(engine, sampler, prompt, NEW + 8)
    return path, config, prompt, tokens, logits, heads, DRAFTER_SEEDS[seed]


def _state(engine, config, rid="r"):
    """What the engine counts of a request: its tokens and each cache's rows."""
    return (engine._resources[CSA]._tokens[rid],
            *(engine._resources[kv_cache_key(ratio)].stored_len(rid) for ratio in config.source_ratios))


def _next(engine, sampler, model, token):
    """The logits and the draft block of one more plain decode step."""
    heads = record_drafts(model)
    logits = run_step(engine, sampler, "decode", {"r": token.view(1)})["r"]
    return logits, heads[-1]


class TestTakeBack:
    """Tokens run and taken back leave nothing behind."""

    @pytest.mark.parametrize("wrong", [1, 2, 3, 4, 5])
    def test_a_request_that_took_tokens_back_decodes_as_one_that_never_ran_them(self, case, wrong):
        path, config, prompt, tokens, logits, _, _ = case
        engine, sampler, model = cpu_engine(path, torch.float32, draft_head=True)
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": prompt})
        at = 0
        for _ in range(4):
            # two true tokens, then drafts that do not hold, each a decode step of its own
            for token in tokens[at:at + 2]:
                assert torch.equal(run_step(engine, sampler, "decode", {"r": token.view(1)})["r"], logits[at + 1])
                at += 1
            kept = _state(engine, config)
            for token in (tokens[at:at + wrong] + 7) % config.vocab_size:
                run_step(engine, sampler, "decode", {"r": token.view(1)})
            assert _state(engine, config)[0] == kept[0] + wrong
            model.get_submodule("LLM").take_back("r", wrong)
            assert _state(engine, config) == kept

    def test_the_counts_follow_the_pooled_groups(self, case):
        """A cache that pools two tokens a row gives back a row per group the tokens completed."""
        path, config, prompt, tokens, _, _, _ = case
        engine, sampler, model = cpu_engine(path, torch.float32, draft_head=True)
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": prompt})
        assert config.source_ratios == (1, 2)
        n = prompt.numel()
        for back in (5, 0, 1, 4, 2, 3):
            run_step(engine, sampler, "decode", {"r": tokens[:6]})
            assert _state(engine, config) == (n + 6, n + 6, (n + 6) // 2)
            model.get_submodule("LLM").take_back("r", back)
            n += 6 - back
            assert _state(engine, config) == (n, n, n // 2)

    def test_more_than_a_step_may_carry_is_refused(self, case, caplog):
        path, config, prompt, tokens, _, _, _ = case
        engine, sampler, model = cpu_engine(path, torch.float32, draft_head=True)
        sub = model.get_submodule("LLM")
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": prompt})
        with pytest.raises(AssertionError, match="'r'"):  # the engine fails the request
            run_step(engine, sampler, "decode", {"r": tokens[:7]})
        assert "at most 5 drafts" in caplog.text
        with pytest.raises(ValueError, match="keeps room for 5"):
            sub.take_back("r", 6)
        with pytest.raises(ValueError, match="cannot take"):
            engine._resources[CSA].take_back("r", prompt.numel() + 1)

    def test_drafts_of_a_request_that_samples_are_refused(self, case):
        from mstar.conductor.request_info import CurrentForwardPassInfo
        from mstar.engine.resources import SamplingReqConfig
        from mstar.model.deepseek_v41.config import SAMPLER

        path, _, prompt, tokens, _, _, _ = case
        engine, sampler, model = cpu_engine(path, torch.float32, draft_head=True)
        sub = engine._submodules["LLM"].submodule
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": prompt})

        def step(ids, sampling):
            info = CurrentForwardPassInfo(
                request_id="wire-r", rid_handle="r", graph_walk="decode", fwd_index=0, random_seed=0, max_tokens=8,
                resource_configs=None if sampling is None else {SAMPLER: SamplingReqConfig(**sampling)})
            return sub.prepare_inputs("decode", info, {"text_inputs": [ids]})

        # kept by argmax: a sampled next token, or one moved by a penalty, would be another
        for sampling in ({"temperature": 0.7}, {"temperature": 0.0, "repetition_penalty": 1.1}, {}, None):
            with pytest.raises(RuntimeError, match="verified by argmax"):
                step(tokens[:3], sampling)
        assert step(tokens[:3], {"temperature": 0.0}).input_seq_len == 3
        assert step(tokens[:1], {"temperature": 0.7}).input_seq_len == 1  # one token: sampled as asked

    def test_without_the_draft_head_a_decode_step_is_one_token(self, case, caplog):
        path, _, prompt, tokens, _, _, _ = case
        engine, sampler, model = cpu_engine(path, torch.float32)
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": prompt})
        with pytest.raises(AssertionError, match="'r'"):
            run_step(engine, sampler, "decode", {"r": tokens[:2]})
        assert "at most 0 drafts" in caplog.text
        with pytest.raises(ValueError, match="keeps room for 0"):
            model.get_submodule("LLM").take_back("r", 1)


class TestDraftVerify:
    def _run(self, case, one_step, drafter="likely", **kwargs):
        path, config, prompt, tokens, _, _, seed = case
        engine, sampler, model = cpu_engine(path, torch.float32, draft_head=True)
        if drafter == "likely":
            drafter = likely_drafter(tokens, config.vocab_size, 0.75, config.dspark_block_size, seed, **kwargs)
        return engine, sampler, model, draft_verify(engine, sampler, model, prompt, NEW, drafter, one_step)

    def test_a_row_a_step_it_is_plain_greedy_bit_for_bit(self, case):
        """Each row of a round as its own decode step, so its arithmetic is a plain step's:
        every token and its logits are plain greedy's, and after the last round the next
        step's logits and the head's next block are too, bit for bit."""
        _, config, _, tokens, logits, heads, _ = case
        engine, sampler, model, run = self._run(case, one_step=False)
        n = run.tokens.numel()
        assert n >= NEW and torch.equal(run.tokens, tokens[:n])
        assert torch.equal(run.logits, logits[:n])
        assert _state(engine, config)[0] == case[2].numel() + n - 1
        after, block = _next(engine, sampler, model, run.tokens[-1])
        assert torch.equal(after, logits[n])
        assert torch.equal(block.logits, heads[n].logits) and torch.equal(block.confidence, heads[n].confidence)

    def test_one_step_a_round_emits_plain_greedys_tokens(self, case):
        """A round's rows in one decode step: the tokens are plain greedy's, and the logits
        and the request's state after it within the rounding of a block of rows."""
        _, config, _, tokens, logits, heads, _ = case
        engine, sampler, model, run = self._run(case, one_step=True)
        n = run.tokens.numel()
        assert n >= NEW and torch.equal(run.tokens, tokens[:n])
        assert logit_gap(run.logits, logits[:n]) < BLOCK_TOL
        assert _state(engine, config)[0] == case[2].numel() + n - 1
        after, block = _next(engine, sampler, model, run.tokens[-1])
        assert logit_gap(after, logits[n]) < BLOCK_TOL
        assert torch.equal(block.ids, heads[n].ids) and logit_gap(block.logits, heads[n].logits) < BLOCK_TOL

    def test_every_count_of_kept_drafts_is_run(self, case):
        config = case[1]
        for one_step in (False, True):
            run = self._run(case, one_step)[3]
            assert set(run.accepted) == set(range(config.dspark_block_size + 1)), run.accepted
            assert sum(run.accepted) + len(run.accepted) == run.tokens.numel() - 1

    def test_what_a_rejected_draft_held_changes_nothing_after_it(self, case):
        """The same rounds with other tokens in the rejected rows: every kept row's logits,
        the step after and the head's next block are the same bit for bit."""
        runs = [self._run(case, one_step=True, miss=miss) for miss in (1, 2)]
        (_, _, _, a), (_, _, _, b) = runs
        assert a.accepted == b.accepted
        assert not all(torch.equal(x, y) for x, y in zip(a.drafted, b.drafted, strict=True))
        assert torch.equal(a.tokens, b.tokens) and torch.equal(a.logits, b.logits)
        (after_a, block_a), (after_b, block_b) = (
            _next(engine, sampler, model, run.tokens[-1]) for engine, sampler, model, run in runs)
        assert torch.equal(after_a, after_b) and torch.equal(block_a.logits, block_b.logits)

    @pytest.mark.parametrize("one_step", [False, True], ids=["a-row-a-step", "one-step"])
    def test_the_heads_own_drafts_emit_plain_greedys_tokens(self, case, one_step):
        """The real head on random weights: it is almost never right, and the output does not
        depend on it."""
        _, _, _, tokens, logits, _, _ = case
        run = self._run(case, one_step, drafter=None)[3]
        n = run.tokens.numel()
        assert n >= NEW and torch.equal(run.tokens, tokens[:n])
        if not one_step:
            assert torch.equal(run.logits, logits[:n])
        assert all(drafts.numel() == 5 for drafts in run.drafted)


class TestDecodeRows:
    """A draft row attends as the one-token step it stands for. In bf16 the softmax rounds its
    probabilities a block of entries at a time, so where a row's entries sit decides the
    logits: here in blocks of 4, over a window the prompt has not filled."""

    def _logits(self, tmp_path, one_step: bool) -> torch.Tensor:
        config, _ = _testing.write_snapshot(tmp_path, DeepseekV41Config.reduced(), seed=0, draft_head=True)
        ids = random_ids(config, 3 + 6, seed=10)
        engine, sampler, model = cpu_engine(tmp_path, torch.bfloat16, draft_head=True)
        engine._resources[CSA].config.softmax_block = 4
        rows = record_logits(model)
        add_request(engine, "r")
        run_step(engine, sampler, "prefill", {"r": ids[:3]})
        for block in ([ids[3:]] if one_step else ids[3:].split(1)):
            run_step(engine, sampler, "decode", {"r": block})
        return torch.cat(rows[1:])

    def test_a_block_of_rows_gives_the_one_token_steps_logits(self, tmp_path):
        assert logit_gap(self._logits(tmp_path, True), self._logits(tmp_path, False)) < BLOCK_TOL

    def test_laid_out_as_a_prompts_rows_it_does_not(self, tmp_path, monkeypatch):
        from mstar.engine.resources.csa.config import CSAStep
        from mstar.model.deepseek_v41 import submodules

        want = self._logits(tmp_path, False)
        monkeypatch.setattr(submodules, "CSAStep", lambda **kwargs: CSAStep())
        assert logit_gap(self._logits(tmp_path, True), want) > 0.1


def test_two_requests_verify_in_one_step_as_each_does_alone(tmp_path):
    """Requests that keep different numbers of drafts, and one that carries none, share a step."""
    config, _ = _testing.write_snapshot(tmp_path, DeepseekV41Config.reduced(), seed=3, draft_head=True)
    prompts = {rid: random_ids(config, n, seed=n) for rid, n in (("a", 21), ("b", 9), ("c", 14))}
    truth, blocks = {}, {}
    for rid, prompt in prompts.items():
        engine, sampler, _ = cpu_engine(tmp_path, torch.float32, draft_head=True)
        truth[rid] = greedy(engine, sampler, prompt, 7, rid)
    # a keeps 2 of 5 drafts, b all 4 of its own, c carries none
    blocks["a"] = torch.cat([truth["a"][0][:3], (truth["a"][0][3:6] + 1) % config.vocab_size])
    blocks["b"] = truth["b"][0][:5]
    blocks["c"] = truth["c"][0][:1]
    engine, sampler, model = cpu_engine(tmp_path, torch.float32, draft_head=True)
    rows, heads = record_logits(model), record_drafts(model)
    for rid in prompts:
        add_request(engine, rid)
    run_step(engine, sampler, "prefill", prompts)
    got = run_step(engine, sampler, "decode", blocks)
    assert rows[-1].shape[0] == 6 + 5 + 1 and heads[-1].ids.shape == (3, config.dspark_block_size)
    kept = {"a": 2, "b": 4, "c": 0}
    for rid, n in kept.items():
        # the sampler saw each request's last kept row: the logits of the token after it
        assert logit_gap(got[rid], truth[rid][1][n + 1]) < BLOCK_TOL
        assert got[rid].argmax() == truth[rid][0][n + 1]
    model.get_submodule("LLM").take_back("a", 3)
    after = run_step(engine, sampler, "decode", {rid: truth[rid][0][n + 1].view(1) for rid, n in kept.items()})
    for rid, n in kept.items():
        assert logit_gap(after[rid], truth[rid][1][n + 2]) < BLOCK_TOL
