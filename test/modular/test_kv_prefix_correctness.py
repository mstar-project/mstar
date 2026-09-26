"""A request served from the cache attends exactly what it would have computed.

The whole design rests on one claim: the pages a hit substitutes hold the same
bytes, at the same positions, that this request would have written there itself.
Everything else — the keys, the root, the eligibility rules — exists to make that
claim true, and none of it fails loudly when it is false. A hit that is wrong
produces a plausible answer to the wrong prompt.

So these tests write a value into every KV slot that depends on nothing but the
position it holds, run one sequence under one batch schedule and the same
sequence under another, and compare what the second one ends up attending over
against a run with no cache at all. Nothing here goes near a kernel, so the
comparison is exact: no tolerance, on either the bytes or the positions.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import KVConfig, PositionConfig, StepContext, StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, SubmoduleStep

PAGE_SIZE = 16
ROOT = b"a root"
KV, ROPE = "kv", "rope"
NODE = "LLM"
WALK = "prefill"
PROMPT = list(range(200))
SUFFIX = list(range(9000, 9037))


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info == self.get_kv_transfer_info()

    def remove_request(self, request_id):
        del request_id

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


class _Node:
    """One node's KV and position resources, with a stand-in for its forward."""

    def __init__(self, cached: bool = True):
        device = torch.device("cpu")
        self.kv = KVManager(
            cfg=KVConfig(
                num_layers=2, num_kv_heads=1, head_dim=4, max_seq_len=4096,
                max_num_pages=128, page_size=PAGE_SIZE,
            ),
            name=KV, joint_comm_group=None, transfer_engine_info=None,
            device=device, dtype=torch.float32,
        )
        if cached:
            self.kv.enable_prefix_cache(ROOT)
        self.rope = RopeManager(
            config=PositionConfig(kv_cache=KV), device=device,
            dtype=torch.float32,
        )
        self.runner = StepRunner(
            {KV: self.kv, ROPE: self.rope}, node_resources={NODE: [KV, ROPE]},
        )

    def ingest(self, rid: str, tokens: list[int]) -> None:
        whole = len(tokens) // PAGE_SIZE
        self.runner.ingest_request(rid, {KV: KVReqConfig(
            prefix_keys={"main": chain([
                tokens[at:at + PAGE_SIZE]
                for at in range(0, len(tokens), PAGE_SIZE)
            ])},
            prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        )})

    def resolve(self, rid: str) -> int:
        """Both halves, in the order the engine runs them: every resource has
        to be told the length before any of them is asked to act on it."""
        matched = self.runner.resolve_cached_prefix(rid, NODE, WALK)
        self.runner.apply_cached_prefix(rid, NODE, WALK, None, matched)
        return matched

    def step(self, rid: str, span: int) -> list[int]:
        """Admit, plan, write the slots this step owns, commit."""
        step = SubmoduleStep(
            steps={KV: KVStep(), ROPE: PositionStep()},
            segments=[Segment(rid, "main", span)],
        )
        ctx = StepContext(
            request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
        )
        step.set_ctx(ctx)
        assert self.runner.admit(step).outcome.ok
        results = self.runner.plan(step)
        positions = results[ROPE]["main"].tolist()
        view = results[KV]["main"].views[0]
        self._write(view, positions)
        self.runner.commit(step)
        return positions

    def _write(self, view, positions: list[int]) -> None:
        """Stand in for the forward: a slot's contents depend on its position.

        The point of the test is that a cached page and a freshly computed one
        are indistinguishable, so what is written has to be a function of the
        position alone — which is exactly what a correct prefix reuse claims.
        """
        resident = view.length - view.to_compute
        for offset, position in enumerate(positions):
            slot = resident + offset
            page = view.page_idxs[slot // PAGE_SIZE]
            self.kv.kv_cache.tensor[:, page, :, slot % PAGE_SIZE] = (
                float(position) + 0.5
            )

    def attended(self, rid: str) -> torch.Tensor:
        """Every slot this request would attend over, in order."""
        stream = self.kv._streams[rid]["main"]
        return torch.stack([
            self.kv.kv_cache.tensor[
                :, stream.page_indices[at // PAGE_SIZE], :, at % PAGE_SIZE
            ].clone()
            for at in range(stream.stored_len)
        ])


def _schedule_a(node: _Node, rid: str, tokens: list[int]) -> None:
    """One request, one step for the whole prompt."""
    node.ingest(rid, tokens)
    node.resolve(rid)
    node.step(rid, len(tokens))


def _schedule_b(node: _Node, rid: str, tokens: list[int]) -> tuple[int, list[int]]:
    """The same request, arriving in uneven chunks after whatever was cached.

    Gives back how much it skipped and the positions it actually computed.
    """
    node.ingest(rid, tokens)
    matched = node.resolve(rid)
    positions: list[int] = []
    at = matched
    for chunk in (37, 61, len(tokens)):
        end = min(at + chunk, len(tokens))
        if end <= at:
            break
        positions += node.step(rid, end - at)
        at = end
    return matched, positions


# ── the claim ───────────────────────────────────────────────────────────


def test_a_run_that_hit_the_cache_attends_what_a_fresh_run_computed():
    warm = _Node()
    _schedule_a(warm, "seed", PROMPT)
    warm.kv.remove_request("seed")

    matched, cached = _schedule_b(warm, "b", PROMPT + SUFFIX)
    fresh_node = _Node(cached=False)
    skipped, fresh = _schedule_b(fresh_node, "b", PROMPT + SUFFIX)

    assert skipped == 0, "the fresh run found a cache it was supposed to be without"
    assert cached == fresh[matched:], (
        "the run that skipped a prefix carried on at the wrong positions"
    )
    torch.testing.assert_close(
        warm.attended("b"), fresh_node.attended("b"), atol=0.0, rtol=0.0,
        msg="a request served from the cache attends different bytes than the "
            "same request computed from nothing",
    )


def test_the_hit_actually_happened():
    warm = _Node()
    _schedule_a(warm, "seed", PROMPT)
    warm.kv.remove_request("seed")

    warm.ingest("b", PROMPT + SUFFIX)
    matched = warm.resolve("b")

    assert matched == 192, (
        f"only {matched} tokens matched, so the test above would have proved "
        "nothing about reuse"
    )


def test_the_cached_pages_are_the_seeds_own_pages():
    warm = _Node()
    _schedule_a(warm, "seed", PROMPT)
    seed_pages = list(warm.kv._streams["seed"]["main"].page_indices)
    warm.kv.remove_request("seed")

    _schedule_b(warm, "b", PROMPT + SUFFIX)

    reused = set(warm.kv._streams["b"]["main"].page_indices) & set(seed_pages)
    assert len(reused) == 12, (
        f"{len(reused)} pages were shared with the request that wrote them, so "
        "the second run recomputed what it claimed to reuse"
    )
    warm.kv.assert_pages_conserved()


def test_two_schedules_over_the_same_tokens_agree_without_any_cache():
    """The control: the comparison above is only worth anything if the two
    schedules agree when nothing is being reused."""
    one = _Node(cached=False)
    one.ingest("x", PROMPT + SUFFIX)
    one.step("x", len(PROMPT + SUFFIX))

    other = _Node(cached=False)
    _schedule_b(other, "x", PROMPT + SUFFIX)

    torch.testing.assert_close(
        one.attended("x"), other.attended("x"), atol=0.0, rtol=0.0,
    )


# ── the two nodes that opted in ─────────────────────────────────────────


def test_orpheus_declares_the_tensor_its_llm_prefills_from():
    from mstar.model.base import PrefixStream
    from mstar.model.orpheus.config import KV_CACHE
    from mstar.model.orpheus.orpheus_model import OrpheusModel

    # the method reads nothing off the model, so a stand-in is the whole of it
    declared = OrpheusModel.prefix_key_streams(object())

    assert declared == {
        KV_CACHE: {"main": PrefixStream("text_inputs", "ids", "prefill", "decode")}
    }, "the LLM node's prompt does not arrive under the name it keys"


def test_bagel_declares_the_tensor_its_text_walk_prefills_from():
    from mstar.model.bagel.bagel_model import BagelModel
    from mstar.model.base import PrefixStream

    declared = BagelModel.prefix_key_streams(object())

    assert declared == {
        "kv": {"main": PrefixStream("text_inputs", "ids", "prefill_text", "decode")}
    }, "the text walk's prompt does not arrive under the name it keys"
