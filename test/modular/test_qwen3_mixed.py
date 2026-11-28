"""A synchronous DSpark-drafting Qwen3 node runs its prompts inside its verify steps (`VerifyNode.mixed_walk`,
the scaffold's `_mixed_verify`): on the CPU harness (`qwen3_cpu`, float64) with the real DSpark drafter, a mixed
step's verify rows draft and verify as in a verify step of their own, and its prompt rows leave with their first
token and the drafter's context, as from a prompt step."""
from __future__ import annotations

import pytest
import torch
from qwen3_cpu import (
    NODE,
    PREFILL,
    Driver,
    draft_block,
    drafter_first_layer,
    engine,
    make_model,
    patch_cpu,
    prompt_kv,
    reference_greedy,
    write_checkpoints,
)

from mstar.engine.chunking import ChunkRange
from mstar.engine.resources.kv import manager as manager_mod
from mstar.model.components.dspark.drafter import DSparkDrafter
from mstar.model.keys import MIXED
from mstar.model.qwen3.submodules import Qwen3LMSubmodule

PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17, 4]}
LATE = {3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4: [40, 41, 42]}
MAX = 16
K = 3


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    patch_cpu(monkeypatch)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


@pytest.fixture
def ckpt(tmp_path):
    return write_checkpoints(tmp_path)


def test_a_drafting_node_declares_its_mixed_walk_in_either_mode(ckpt):
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, K, "synchronous"))
    (decl,) = model.verify_nodes()
    assert decl.mixed_walk == MIXED and decl.needs.mixed
    assert model.get_combined_graph_walks() and model.mixed_splits() == ()
    budget = {walk: policy.max_batch_tokens for (node, walk), policy in model.chunking_policy_of().items()}
    sizes, extra = Qwen3LMSubmodule.MIXED_VERIFY_BATCH_SIZES, Qwen3LMSubmodule.MIXED_VERIFY_PROMPT_TOKENS
    assert budget[MIXED] == sizes[-1] * (K + 1) + extra[-1]
    for mode in ("auto", "asynchronous"):
        model = make_model(target, draft_block(dspark, K, mode))
        assert model.verify_nodes()[0].mixed_walk == MIXED and model.get_combined_graph_walks()


def test_a_drafting_nodes_mixed_step_declares_the_targets_attention_as_a_verify_steps(ckpt):
    """So it plans on FA3 where the verify walk does (`FA3RaggedWrapper`): every attention of the target's layers
    in a mixed step, as the engine runs it (`engine_step`, the drafter's share added), is a verify step's, with
    nothing hidden, and a plain node's mixed step's is not."""
    from mstar.engine.resources.attn.config import AttentionStep
    from mstar.model.submodule_base import ARNodeInputs, engine_step

    target, dspark = ckpt
    rows = [ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=K + 1, graph_walk="decode"),
            ARNodeInputs(input_ids=torch.zeros(5, dtype=torch.long), input_seq_len=5, graph_walk=PREFILL)]
    for draft, verify in ((draft_block(dspark, K, "synchronous"), True), (None, False)):
        sub = engine(make_model(target, draft))._submodules[NODE].submodule
        step = engine_step(sub, graph_walk=MIXED, request_ids=[1, 2], inputs=rows)
        own = set(sub.drafter_share.verify_steps) if sub.drafter_share else set()
        attention = {key: s for key, s in step.steps.items() if isinstance(s, AttentionStep) and key not in own}
        assert attention and all(s.verify is verify and not s.hide_last and s.causal for s in attention.values())


def _record_drafts(monkeypatch):
    seen = []
    propose = DSparkDrafter.propose

    def recorded(self, inputs):
        out = propose(self, inputs)
        seen.append(out.clone())
        return out

    monkeypatch.setattr(DSparkDrafter, "propose", recorded)
    return seen


def _start(model, steps: int = 1):
    driver = Driver(engine(model), model)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, MAX)
    driver.prefill(list(PROMPTS), PROMPTS)
    for _ in range(steps):
        driver.step(list(PROMPTS))
    return driver


def _held(driver, model, rids, prompts):
    layers = range(drafter_first_layer(model), len(driver.eng._resources["kv_cache"].kv_cache.tensor))
    return {rid: prompt_kv(driver.eng, rid, len(prompts[rid]), layers) for rid in rids}


@pytest.mark.parametrize("steps", [0, 1], ids=["first-verify-step", "looped-back"])
def test_a_mixed_steps_rows_draft_decode_and_keep_the_drafters_context_as_steps_of_their_own(
    ckpt, monkeypatch, steps,
):
    """The late prompts in one mixed step with the running rows' verify rows, against their own prompt step:
    the running rows draft the same drafts there, the late prompts leave the drafter the same context K/V,
    and every reply is plain greedy. The running rows' first verify step brings their token routed; a later
    one reads it off its loop-back slot."""
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, K, "synchronous"))
    seen = _record_drafts(monkeypatch)

    alone = _start(model, steps)
    for rid, prompt in LATE.items():
        alone.add(rid, prompt, MAX)
    alone.prefill(list(LATE), LATE)
    held_alone = _held(alone, model, LATE, LATE)
    first = len(seen)
    alone.step(list(PROMPTS) + list(LATE))
    drafts_alone = seen[first][: len(PROMPTS)]
    alone.run(list(PROMPTS) + list(LATE))

    mixed = _start(model, steps)
    for rid, prompt in LATE.items():
        mixed.add(rid, prompt, MAX)
    first = len(seen)
    mixed.mixed_step(list(PROMPTS), LATE)
    drafts_mixed = seen[first][: len(PROMPTS)]
    held_mixed = _held(mixed, model, LATE, LATE)
    mixed.run(list(PROMPTS) + list(LATE))

    assert torch.equal(drafts_mixed, drafts_alone)
    for rid in LATE:
        torch.testing.assert_close(held_mixed[rid], held_alone[rid], rtol=1e-9, atol=1e-12)
    want = {rid: reference_greedy(model, p, MAX) for rid, p in {**PROMPTS, **LATE}.items()}
    assert mixed.out == want == alone.out


def test_a_prompt_chunked_over_mixed_steps_replies_as_whole(ckpt):
    """A prompt cut over two mixed steps (`ChunkRange`): no token on its first chunk, its first on its last."""
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, K, "synchronous"))
    driver = _start(model)
    prompt = LATE[3]
    driver.add(3, prompt, MAX)
    driver.mixed_step(list(PROMPTS), {3: prompt}, {3: ChunkRange(0, 6, len(prompt), 6)})
    assert driver.out[3] == []
    driver.mixed_step(list(PROMPTS), {3: prompt}, {3: ChunkRange(6, len(prompt), len(prompt), len(prompt) - 6)})
    assert len(driver.out[3]) == 1
    driver.run(list(PROMPTS) + [3])
    assert driver.out[3] == reference_greedy(model, prompt, MAX)
    assert driver.out[1] == reference_greedy(model, PROMPTS[1], MAX)


def test_a_mixed_replay_pads_to_its_bucket_with_empty_rows_mixed_rows_reads_as_prompt_rows(ckpt):
    """A replay whose rows and tokens fall between buckets: `pad_inputs` lays the padding rows out with none of
    the step's tokens (the layout once gave them negative lengths), and `MixedRows` reads them as empty prompt
    rows after the real ones, which keep their places."""
    from types import SimpleNamespace

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
    from mstar.engine.resources.draft.drafter import mixed_rows
    from mstar.engine.resources.step import BucketKey
    from mstar.model.submodule_base import ARNodeInputs

    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, K, "synchronous"), accelerator_graph=True)
    sub = engine(model)._submodules[NODE].submodule
    (mixed,) = [c for c in sub.get_accelerator_graph_configs(torch.device("cpu")) if c.capture_graph_walk == MIXED]
    block, bs = K + 1, 8
    assert mixed.token_layout(0, 5) == [0] * 5
    real = [ARNodeInputs(input_seq_len=block, graph_walk="decode") for _ in range(2)]
    real.append(ARNodeInputs(input_ids=torch.zeros(50, dtype=torch.long), input_seq_len=50, graph_walk=PREFILL))
    bucket = BucketKey(MIXED, bs, bs * block + Qwen3LMSubmodule.MIXED_VERIFY_PROMPT_TOKENS[0])
    runner = SimpleNamespace(_buckets={bucket: SimpleNamespace(config=mixed)})
    padded = AcceleratorGraphRunner.pad_inputs(runner, SimpleNamespace(bucket=bucket), real)
    lens = [inp.input_seq_len for inp in padded]
    assert lens == [block, block, 50, 0, 0, 0, 0, 0]
    assert all(inp.graph_walk == PREFILL for inp in padded[3:])
    qo = torch.tensor([0, *lens]).cumsum(0)
    kind = torch.cat([torch.arange(block).repeat(2), torch.full((50,), -1)])
    rows = mixed_rows(qo, kind, block)
    assert rows.prompt.tolist() == [False, False] + [True] * 6
    assert rows.at[:2].tolist() == [list(range(block)), list(range(block, 2 * block))]
    assert rows.at[2].tolist() == [2 * block + 49] * block


def test_the_mixed_capture_runs_verify_rows_then_a_prompt_row_on_a_trimmed_ladder(ckpt):
    target, dspark = ckpt
    model = make_model(target, draft_block(dspark, K, "synchronous"), accelerator_graph=True)
    sub = engine(model)._submodules[NODE].submodule
    (mixed,) = [c for c in sub.get_accelerator_graph_configs(torch.device("cpu")) if c.capture_graph_walk == MIXED]
    # its lean twins are a verify bucket's (`MixedVerifyGraphConfig`)
    assert getattr(mixed, "verify_lean_twins", False)
    block = K + 1
    assert list(mixed.capture_batch_sizes) == Qwen3LMSubmodule.MIXED_VERIFY_BATCH_SIZES
    for bs in mixed.capture_batch_sizes:
        assert list(mixed.capture_token_lengths[bs]) == [
            bs * block + extra for extra in Qwen3LMSubmodule.MIXED_VERIFY_PROMPT_TOKENS]
    rows = mixed.token_layout(4 * block + 128, 4)
    assert rows == [block] * 3 + [block + 128]
    assert [sub_row.graph_walk for sub_row in map(mixed.make_node_input, rows)] == ["decode"] * 3 + [PREFILL]
    # one capture a bucket a slot
    assert sum(len(v) for v in mixed.capture_token_lengths.values()) == len(
        Qwen3LMSubmodule.MIXED_VERIFY_PROMPT_TOKENS) * len(mixed.capture_batch_sizes)
    # as wide as the verify walk's captures, so a mixed step never splits its verify rows
    assert mixed.capture_batch_sizes[-1] == max(Qwen3LMSubmodule.DECODE_CAPTURE_BATCH_SIZES)


# ── EAGLE-3 and the draft model ──


def _eagle3_model(tmp_path):
    from test_eagle3_config import _cpu_eagle3

    target, _ = write_checkpoints(tmp_path)
    return make_model(target, {**_cpu_eagle3(tmp_path), "mode": "synchronous"}, max_seq_len=512)


@pytest.mark.parametrize("steps", [0, 1], ids=["first-verify-step", "looped-back"])
def test_eagle3s_mixed_step_drafts_carries_and_writes_as_steps_of_their_own(tmp_path, monkeypatch, steps):
    """EAGLE-3's chain over a mixed step's tokens drafts the running rows' drafts there, and its one pass after
    the target leaves the late prompts the layer's K/V and the first carry their own prompt step would;
    every reply is plain greedy."""
    from mstar.engine.resources.draft.drafter import DRAFT_CARRY
    from mstar.model.components.eagle3 import Eagle3Drafter

    model = _eagle3_model(tmp_path)
    assert model.verify_nodes()[0].mixed_walk == MIXED
    seen = []
    propose = Eagle3Drafter.propose

    def recorded(self, inputs):
        out = propose(self, inputs)
        seen.append(out.clone())
        return out

    monkeypatch.setattr(Eagle3Drafter, "propose", recorded)

    def carries(driver, rids=tuple(LATE)):
        return driver.eng._resources["sampler"].loopback_state(DRAFT_CARRY, list(rids))[: len(rids)].clone()

    alone = _start(model, steps)
    for rid, prompt in LATE.items():
        alone.add(rid, prompt, MAX)
    alone.prefill(list(LATE), LATE)
    held_alone, carry_alone = _held(alone, model, LATE, LATE), carries(alone)
    first = len(seen)
    alone.step(list(PROMPTS) + list(LATE))
    drafts_alone, running_alone = seen[first][: len(PROMPTS)], carries(alone, PROMPTS)
    alone.run(list(PROMPTS) + list(LATE))

    mixed = _start(model, steps)
    for rid, prompt in LATE.items():
        mixed.add(rid, prompt, MAX)
    first = len(seen)
    mixed.mixed_step(list(PROMPTS), LATE)
    drafts_mixed = seen[first][: len(PROMPTS)]
    held_mixed, carry_mixed = _held(mixed, model, LATE, LATE), carries(mixed)
    running_mixed = carries(mixed, PROMPTS)
    mixed.run(list(PROMPTS) + list(LATE))

    assert torch.equal(drafts_mixed, drafts_alone)
    for rid in LATE:
        torch.testing.assert_close(held_mixed[rid], held_alone[rid], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(carry_mixed, carry_alone, rtol=1e-6, atol=1e-8)
    # the running rows' carry after their verify: the same block, the same kept tokens
    torch.testing.assert_close(running_mixed, running_alone, rtol=1e-6, atol=1e-8)
    want = {rid: reference_greedy(model, p, MAX) for rid, p in {**PROMPTS, **LATE}.items()}
    assert mixed.out == want == alone.out


def test_a_draft_models_node_declares_no_mixed_walk(tmp_path):
    """Its passes lag the target (`DrafterNeeds.lag`), so it runs its prompts in steps of their own."""
    import json

    from qwen3_cpu import TARGET

    target, _ = write_checkpoints(tmp_path)
    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps({**TARGET, "num_hidden_layers": 2}))
    model = make_model(target, {"method": "draft_model", "model": str(draft), "num_draft_tokens": K,
                                "mode": "synchronous"}, max_seq_len=512)
    assert not model.verify_nodes()[0].needs.mixed and model.verify_nodes()[0].mixed_walk is None
    assert model.get_combined_graph_walks() == {}


def test_a_padding_row_never_writes_the_last_prompt_rows_next_token():
    """A replay's padding row reads the last real token (`MixedRows.at`), the last prompt row's last: its
    write of a sampled token goes to the sink, so that place keeps its own row's sampled token."""
    from mstar.engine.resources.draft.drafter import mixed_rows
    from mstar.model.components.chained_draft import mixed_next_ids

    block = 3
    # two verify rows, a 5-token final prompt chunk, one padding row; the kind padded to the bucket's tokens
    qo = torch.tensor([0, 3, 6, 11, 11])
    kind = torch.cat([torch.arange(block).repeat(2), torch.full((5,), -1), torch.full((3,), -1)])[:11]
    rows = mixed_rows(qo, kind, block)
    assert rows.empty.tolist() == [False, False, False, True]
    following = torch.tensor([0, 0, 0, 0, 0, 0, 21, 22, 23, 24, -1])
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6], [777, 0, 0], [999, 0, 0]])
    nxt = mixed_next_ids(rows, following, tokens)
    assert nxt.tolist() == [1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 777]
