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

from mstar.engine.resources import PagedKVConfig, PositionConfig, StepContext, StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.submodule_base import ARNodeInputs

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

    def __init__(self, cached: bool = True, window: int | None = None):
        device = torch.device("cpu")
        cfg = PagedKVConfig(
            num_layers=3, num_kv_heads=1, head_dim=4, max_seq_len=4096,
            max_num_pages=128, page_size=PAGE_SIZE,
        )
        if window is not None:
            # layers 1 and 2 keep the last ``window`` tokens, in pages of half the full layer's
            cfg.keep_window(window, (1, 2))
        self.kv = KVManager(
            cfg=cfg,
            name=KV, joint_comm_group=None, transfer_engine_info=None,
            device=device, dtype=torch.float32,
        )
        if cached:
            self.kv.enable_prefix_cache(ROOT)
        # what each step's forward reads (`attended`)
        self.reads: list[torch.Tensor] = []
        self.rope = RopeManager(
            config=PositionConfig(kv_cache=KV), device=device,
            dtype=torch.float32,
        )
        self.runner = StepRunner(
            {KV: self.kv, ROPE: self.rope}, node_resources={NODE: [KV, ROPE]},
        )

    def ingest(self, rid: str, tokens: list[int], growth: int = 0) -> None:
        """``growth`` is the cap of a decode loop that writes a token per iteration."""
        whole = len(tokens) // PAGE_SIZE
        self.runner.ingest_request(rid, {KV: KVReqConfig(
            prefix_keys={"main": chain([
                tokens[at:at + PAGE_SIZE]
                for at in range(0, len(tokens), PAGE_SIZE)
            ])},
            prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
            loop_caps={"decode_loop": growth} if growth else {},
            kv_layout={"main": StreamLayout(
                spans=(PrefixSpan(len(tokens), len(tokens), WALK),),
                growth=(("decode_loop", 1),) if growth else (), keyed_spans=1,
            )},
        )})

    def resolve(self, rid: str) -> int:
        """Both halves, in the order the engine runs them: every resource has
        to be told the length before any of them is asked to act on it."""
        inputs = ARNodeInputs(input_seq_len=self.kv._overrides[rid].kv_layout["main"].spans[0].length)
        prefix = self.runner.resolve_cached_prefix(rid, NODE, WALK, inputs)
        self.runner.apply_cached_prefix(rid, NODE, WALK, inputs, prefix)
        return prefix.tokens if prefix is not None else 0

    def step(self, rid: str, span: int, walk: str = WALK) -> list[int]:
        """Admit, plan, write the slots this step owns, commit."""
        step = SubmoduleStep(
            steps={KV: KVStep(), ROPE: PositionStep()},
            segments=[Segment(rid, "main", span)],
        )
        ctx = StepContext(
            request_ids=(rid,), graph_walk=walk, slot=0, capture=False,
        )
        step.set_ctx(ctx)
        assert self.runner.admit(step).outcome.ok
        results = self.runner.plan(step)
        positions = results[ROPE]["main"].tolist()
        self._write(rid, positions)
        self.reads.append(self.attended(rid, span))
        self.runner.commit(step)
        return positions

    def _slots(self, rid: str):
        """Each layer's view and the stream's pages and slot count in that layer's group."""
        stream = self.kv._streams[rid]["main"]
        for part, group in zip(stream.groups, self.kv._groups, strict=True):
            for layer in group.layers:
                yield self.kv.kv_cache.layer_view(layer), part, group.page_size

    def _write(self, rid: str, positions: list[int]) -> None:
        """Stand in for the forward: a slot's contents depend on its position.

        The point of the test is that a cached page and a freshly computed one
        are indistinguishable, so what is written has to be a function of the
        position alone — which is exactly what a correct prefix reuse claims.
        Before commit, so the new slots follow each group's stored ones.
        """
        for view, part, page_size in self._slots(rid):
            for offset, position in enumerate(positions):
                slot = part.stored_len + offset
                view[part.page_indices[slot // page_size], :, slot % page_size] = float(position) + 0.5

    def attended(self, rid: str, written: int = 0) -> torch.Tensor:
        """Every slot this request would attend over, in order, layer by layer, with ``written`` slots a step
        wrote and hasn't committed."""
        return torch.stack([
            view[part.page_indices[at // page_size], :, at % page_size].clone()
            for view, part, page_size in self._slots(rid)
            for at in range(part.stored_len + written)
        ])


def _schedule_a(node: _Node, rid: str, tokens: list[int]) -> None:
    """One request, one step for the whole prompt."""
    node.ingest(rid, tokens)
    node.resolve(rid)
    node.step(rid, len(tokens))


def _schedule_b(node: _Node, rid: str, tokens: list[int], chunks: tuple[int, ...] = (37, 61)) -> tuple[int, list[int]]:
    """The same request, arriving in uneven chunks after whatever was cached.

    Gives back how much it skipped and the positions it actually computed.
    """
    node.ingest(rid, tokens)
    matched = node.resolve(rid)
    positions: list[int] = []
    at = matched
    for chunk in (*chunks, len(tokens)):
        end = min(at + chunk, len(tokens))
        if end <= at:
            break
        positions += node.step(rid, end - at)
        at = end
    return matched, positions


# ── the claim ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("window", [None, 40], ids=["full", "window"])
def test_a_run_that_hit_the_cache_attends_what_a_fresh_run_computed(window):
    warm = _Node(window=window)
    _schedule_a(warm, "seed", PROMPT)
    warm.kv.remove_request("seed")

    matched, cached = _schedule_b(warm, "b", PROMPT + SUFFIX)
    fresh_node = _Node(cached=False, window=window)
    # its second step, like the cached run's first, writes tokens 192-229 while the window still holds the hit's
    # pages, and both end with a step to the prompt's end
    skipped, fresh = _schedule_b(fresh_node, "b", PROMPT + SUFFIX, chunks=(192, 37, 61))

    assert matched == 192, "the run found nothing in the cache"
    assert skipped == 0, "the fresh run found a cache it was supposed to be without"
    assert cached == fresh[matched:], (
        "the run that skipped a prefix carried on at the wrong positions"
    )
    torch.testing.assert_close(
        warm.attended("b"), fresh_node.attended("b"), atol=0.0, rtol=0.0,
        msg="a request served from the cache attends different bytes than the "
            "same request computed from nothing",
    )
    torch.testing.assert_close(
        warm.reads[-2], fresh_node.reads[-2], atol=0.0, rtol=0.0,
        msg="the first step past the hit reads different bytes than the same step of a run with no cache",
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


# ── a prompt dropped between its chunks ─────────────────────────────────


def test_a_prompt_dropped_between_chunks_and_run_again_to_where_it_stood_attends_what_a_fresh_run_computed():
    """A preempted prompt is dropped between two of its chunks, and its rebuild runs those chunks again past
    what the cache holds. Another request wrote the whole prompt meanwhile, so the cache holds more of it than
    the dropped request wrote. The rebuild stops where the prompt stood, since its next chunk starts there."""
    node, tokens = _Node(), PROMPT + SUFFIX
    node.ingest("r", tokens, growth=8)
    node.resolve("r")
    node.step("r", 37)
    node.step("r", 61)
    _schedule_b(node, "other", PROMPT)
    node.kv.drop("r", 98)
    node.rope.drop("r", 98)

    start = node.resolve("r")
    assert start == 96, "the rebuild would start past where the prompt stood, or recompute what the cache holds"
    node.step("r", 98 - start)
    node.kv.land_rebuild("r")
    node.step("r", len(tokens) - 98)

    fresh = _Node(cached=False)
    _schedule_b(fresh, "r", tokens)
    torch.testing.assert_close(
        node.attended("r"), fresh.attended("r"), atol=0.0, rtol=0.0,
        msg="a prompt rebuilt between its chunks attends other bytes than the same prompt run without a drop",
    )


# ── the two nodes that opted in ─────────────────────────────────────────


def test_orpheus_declares_the_tensor_its_llm_prefills_from():
    from mstar.model.base import PrefixKeys
    from mstar.model.orpheus.config import KV_CACHE
    from mstar.model.orpheus.orpheus_model import OrpheusModel

    # the method reads nothing off the model, so a stand-in is the whole of it
    declared = OrpheusModel.kv_labels(object())

    assert declared[KV_CACHE]["main"].keys == PrefixKeys("text_inputs", ("prefill",)), (
        "the LLM node's prompt does not arrive under the name it keys"
    )


def test_bagel_declares_the_tensor_its_text_walk_prefills_from():
    from mstar.model.bagel.bagel_model import BagelModel
    from mstar.model.base import PrefixKeys

    declared = BagelModel.kv_labels(object())

    assert declared["kv"]["main"].keys == PrefixKeys("text_inputs", ("prefill_text", "prefill_vit"), growth=True), (
        "the text walk's prompt does not arrive under the name it keys"
    )
