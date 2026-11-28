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
from mstar.engine.state.adapters import node_resources
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
    specs = {spec.resource_key: spec for spec in node_resources(model)}
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


# The prefix cache under EAGLE-3. Its K/V at a page's last slot is built from the next page's first token, so
# the index names a page by the next page's key (`PagedKVConfig.names_ahead`), which covers that token.
A = list(range(1, 15))


def _b_after(tmp_path, monkeypatch, firsts: list[list[int]], b: list[int], ahead: int | None = None):
    """``b`` on an engine that ran each of ``firsts`` before it, keyed: the drafter's layer K/V at b's prompt
    positions after its prompt step, its first carry, the drafts it proposes, the reply, and the prompt
    tokens the prefix cache held for it. ``ahead`` overrides the pool's page naming."""
    model = make_model(write_checkpoints(tmp_path)[0], _cpu_eagle3(tmp_path), max_seq_len=512)
    eng = engine(model, prefix_cache=True)
    kv = eng._resources["kv_cache"]
    if ahead is not None:
        kv.config.names_ahead = ahead
    driver = Driver(eng, model)
    for rid, prompt in enumerate(firsts, start=10):
        driver.add(rid, prompt, MAX, keyed=True)
        driver.prefill([rid], {rid: prompt})
        driver.run([rid])
    drafts = []

    def recorded(self, inputs):
        out = PROPOSE(self, inputs)
        drafts.append(out.clone())
        return out

    monkeypatch.setattr(Eagle3Drafter, "propose", recorded)
    held_before = kv.prefix_tokens.get("LLM", (0, 0))[1]
    driver.add(1, b, MAX, keyed=True)
    driver.prefill([1], {1: b})
    held = kv.prefix_tokens.get("LLM", (0, 0))[1] - held_before
    layer = [drafter_first_layer(model)]
    out = (
        prompt_kv(eng, 1, len(b), layer),
        eng._resources["sampler"].loopback_state(DRAFT_CARRY, [1])[:1].clone(),
    )
    driver.run([1])
    return (*out, drafts, driver.out[1], held)


def _same_as_cold(warm, cold):
    torch.testing.assert_close(warm[0], cold[0], rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(warm[1], cold[1], rtol=1e-6, atol=1e-8)
    assert warm[3] == cold[3]
    assert len(warm[2]) == len(cold[2]) and all(torch.equal(x, y) for x, y in zip(warm[2], cold[2], strict=True))


def test_a_reused_prefix_page_holds_the_drafter_k_v_of_its_own_following_token(tmp_path, monkeypatch):
    """A's page 1 ends at position 7, whose EAGLE-3 K/V reads A's token at 8. B shares A's first 8 tokens and
    differs at 8, so it may reuse page 0 but not page 1: it drafts as on a cold engine. Named by its own key,
    page 1 was reused, and B's drafter read A's token there."""
    b = A[:8] + [20, 21, 22, 23]
    warm = _b_after(tmp_path, monkeypatch, [A], b)
    _same_as_cold(warm, _b_after(tmp_path, monkeypatch, [], b))
    assert warm[4] == 4, "page 0 alone"


def test_a_page_another_request_filed_first_is_never_read_past_its_following_token(tmp_path, monkeypatch):
    """The index keeps the first writer of a page. A files page 1 for token 9 after it; D shares A's first 8
    tokens, recomputes page 1 for token 50 and files page 2 under it; B matches D through page 1 and differs
    at 12. Trimming the last matched page would still hand B A's page 1, filed first under the same key."""
    d, b = A[:8] + list(range(50, 56)), A[:8] + [50, 51, 52, 53] + list(range(60, 65))
    warm = _b_after(tmp_path, monkeypatch, [A, d], b)
    _same_as_cold(warm, _b_after(tmp_path, monkeypatch, [], b))
    assert warm[4] == 8, "pages 0 and 1, D's"


def test_a_pool_that_names_each_page_by_its_own_key_reuses_as_before(tmp_path, monkeypatch):
    """The naming is the pool's alone: with names_ahead 0 the three requests above lease what they always
    did, B three pages of A's and D's (so its drafter reads A's token at 8, the bug the naming fixes); an
    exact repeat of A leases two pages either way, its whole pages but the last."""
    d, b = A[:8] + list(range(50, 56)), A[:8] + [50, 51, 52, 53] + list(range(60, 65))
    assert _b_after(tmp_path, monkeypatch, [A, d], b, ahead=0)[4] == 12
    assert _b_after(tmp_path, monkeypatch, [A], A, ahead=0)[4] == 8
    assert _b_after(tmp_path, monkeypatch, [A], A)[4] == 8


def test_a_page_is_filed_and_offloaded_under_the_next_pages_key(tmp_path):
    """Under EAGLE-3 a page enters the index once the next page's key covers a whole page, and an offload
    files the pages it carries to the host under those names, so its reload finds them again."""
    from mstar.engine.resources.kv.keys import fingerprint

    model = make_model(write_checkpoints(tmp_path)[0], _cpu_eagle3(tmp_path), max_seq_len=512)
    eng = engine(model, prefix_cache=True)
    kv = eng._resources["kv_cache"]
    assert kv.config.names_ahead == 1
    driver = Driver(eng, model)
    prompt = A[:12]
    driver.add(1, prompt, MAX, keyed=True)
    driver.prefill([1], {1: prompt})
    stream = kv._streams[1]["main"]
    # three whole pages keyed, the third page's successor not yet: two filed
    assert stream.chain.keyed_pages == 3 and stream.chain.cursor == 2
    names = [fingerprint(kv._prefix_root, key) for key in kv._offload_keys(stream)]
    assert names == [fingerprint(kv._prefix_root, key) for key in stream.chain.keys[1:3]]
    assert kv._filed(names, stream.page_indices) == 2
