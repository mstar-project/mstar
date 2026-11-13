"""A verifying Qwen3 node's prompts prefill in chunks (`Qwen3Model.get_chunking_policies`) under the
EAGLE-3 and draft-model drafters too. EAGLE-3 reads each prompt row's following token from the prompt
step (`ChunkBoundary.FOLLOWING`, `DRAFT_FOLLOWING`), so a chunk's last row reads the next chunk's
first; a draft model's state at a row is its own causal stack's, as the target's is. EAGLE-3 runs on
the CPU harness (`qwen3_cpu`), in float64; the draft model's lagged passes need a GPU's attention, so
its chunked prompts are checked in `test_qwen3_draft_model_device`."""
from __future__ import annotations

import json

import pytest
import torch
from qwen3_cpu import TARGET, Driver, drafter_first_layer, engine, make_model, patch_cpu, prompt_kv, write_checkpoints
from test_eagle3_config import _cpu_eagle3

from mstar.engine.resources.draft.drafter import DRAFT_CARRY, DRAFT_FOLLOWING, ChunkBoundary
from mstar.engine.resources.draft.startup import check_deployment
from mstar.engine.resources.kv import manager as manager_mod
from mstar.model.components.eagle3 import Eagle3Drafter

PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
MAX = 12
# the real method, which a run's recorder wraps
PROPOSE = Eagle3Drafter.propose


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    patch_cpu(monkeypatch)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


def _model(root, method: str):
    target, _ = write_checkpoints(root)
    if method == "eagle3":
        block = _cpu_eagle3(root)
    else:
        draft = root / "draft"
        draft.mkdir(exist_ok=True)
        # the CPU target's family and vocabulary, two layers
        (draft / "config.json").write_text(json.dumps({**TARGET, "num_hidden_layers": 2}))
        block = {"method": "draft_model", "model": str(draft), "num_draft_tokens": 3}
    return make_model(target, block, max_seq_len=512)


@pytest.mark.parametrize("method, rule", [("eagle3", ChunkBoundary.FOLLOWING), ("draft_model", ChunkBoundary.NONE)])
def test_each_drafter_declares_its_chunk_rule_and_startup_accepts_a_chunking_node(tmp_path, method, rule):
    model = _model(tmp_path, method)
    (decl,) = model.verify_nodes()
    assert decl.needs.chunk_boundary == rule
    assert any(policy.chunkable for policy in model.chunking_policy_of().values())
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert check_deployment([decl], specs, {}, (), {}, True, chunked=[decl.node]) == []


def _run(model, monkeypatch, size: int | None):
    """The prompts whole (``size`` None) or in chunks of ``size``, then decoded: the replies, the drafter's
    layers at every prompt position after the prompt steps, its stored carries, and every draft proposed."""
    drafts = []

    def recorded(self, inputs):
        out = PROPOSE(self, inputs)
        drafts.append(out.clone())
        return out

    monkeypatch.setattr(Eagle3Drafter, "propose", recorded)
    eng = engine(model)
    driver = Driver(eng, model)
    for rid, prompt in PROMPTS.items():
        driver.add(rid, prompt, MAX)
    if size is None:
        driver.prefill(list(PROMPTS), PROMPTS)
    else:
        assert driver.prefill_chunks(list(PROMPTS), PROMPTS, size) == -(-max(map(len, PROMPTS.values())) // size)
    layers = range(drafter_first_layer(model), len(eng._resources["kv_cache"].kv_cache.tensor))
    held = {rid: prompt_kv(eng, rid, len(p), layers) for rid, p in PROMPTS.items()}
    carry = eng._resources["sampler"].loopback_state(DRAFT_CARRY, list(PROMPTS))[: len(PROMPTS)].clone()
    driver.run(list(PROMPTS))
    return driver.out, held, carry, drafts


@pytest.mark.parametrize("size", [1, 3, 7])
def test_a_chunked_prompt_leaves_eagle3_what_the_whole_prompt_does(tmp_path, monkeypatch, size):
    """Chunks of one token, ending mid-page, and past a page's end: the drafter's layer holds the same K/V
    at every prompt position (to float rounding: the target's attention sums over the chunks), the node
    stores the same first carry, every draft is the same, and so is the reply."""
    model = _model(tmp_path, "eagle3")
    whole, chunked = _run(model, monkeypatch, None), _run(model, monkeypatch, size)
    for rid in PROMPTS:
        torch.testing.assert_close(chunked[1][rid], whole[1][rid], rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(chunked[2], whole[2], rtol=1e-6, atol=1e-8)
    assert chunked[0] == whole[0]
    assert len(chunked[3]) == len(whole[3]) > 0
    assert all(torch.equal(a, b) for a, b in zip(chunked[3], whole[3], strict=True))


def test_eagle3_reads_each_chunks_following_tokens(tmp_path, monkeypatch):
    """A chunk's rows read the prompt's next tokens, its last row the next chunk's first; only a prompt's
    last row reads the token its step sampled."""
    model = _model(tmp_path, "eagle3")
    seen = []
    head = Eagle3Drafter._head

    def recorded(self, hidden, next_ids, keep=None):
        if keep is not None:
            seen.append(next_ids.tolist())
        return head(self, hidden, next_ids, keep)

    monkeypatch.setattr(Eagle3Drafter, "_head", recorded)
    driver = Driver(engine(model), model)
    prompt = PROMPTS[3]
    driver.add(3, prompt, MAX)
    driver.prefill_chunks([3], {3: prompt}, 4)
    assert seen == [prompt[1:5], prompt[5:9], prompt[9:] + [driver.out[3][0]]]


def test_a_prefix_hit_still_cuts_an_eagle3_prompt(tmp_path, monkeypatch):
    """The prompt step's `DRAFT_FOLLOWING` is cut with its ids (`cuts_tensor_inputs`), so the engine's probe
    still finds a cached prefix: the second request's prompt step runs its uncached tail alone, reads the
    tail's following tokens, and stores the carry the cold request did."""
    target, _ = write_checkpoints(tmp_path)
    model = make_model(target, _cpu_eagle3(tmp_path), max_seq_len=512)
    eng = engine(model, prefix_cache=True)
    assert eng._submodules["LLM"].submodule.cuts_tensor_inputs
    rows, following = [], []
    prefill = Eagle3Drafter.prefill

    def recorded(self, inputs):
        rows.append(inputs.input_ids.shape[0])
        following.append(inputs.tensors[DRAFT_FOLLOWING].tolist())
        return prefill(self, inputs)

    monkeypatch.setattr(Eagle3Drafter, "prefill", recorded)
    prompt = list(range(1, 15))
    driver = Driver(eng, model)
    carries = []
    for rid in (1, 2):
        driver.add(rid, prompt, MAX, keyed=True)
        driver.prefill([rid], {rid: prompt})
        carries.append(eng._resources["sampler"].loopback_state(DRAFT_CARRY, [rid])[:1].clone())
        driver.run([rid])
    # whole pages, the last one kept back to compute the prompt's last row
    assert rows[0] == len(prompt) and rows[1] < len(prompt)
    assert following[1] == prompt[len(prompt) - rows[1] + 1:] + [-1]
    torch.testing.assert_close(carries[1], carries[0], rtol=1e-6, atol=1e-8)
    assert driver.out[1] == driver.out[2]
