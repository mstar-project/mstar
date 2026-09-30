"""One lease holds a hit across every walk of a layout, and each walk takes its share.

A prompt laid out text, image, text is three walks into one stream. The first
walk's probe matches the whole layout and holds it; the walks it covers are
served from that lease without running, and the one it covers in part runs
from the lease's end.

A served walk never admits, and an image walk reads the position counter while
it prepares its inputs, so the counter moves at the probe, and by an image
block's one position, not its slots.
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")

import pytest
import torch

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.wire import decode, encode
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import KVConfig, PositionConfig, PositionStep, StepRunner
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan
from mstar.engine.resources.kv.keys import PageItem, chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.model.submodule_base import ARNodeInputs
from mstar.utils.ipc_format import InputSignals, WorkerMessage, WorkerMessageType

PAGE_SIZE = 16
ROOT = b"a root"
RID = "r1"
NODE = "LLM"
TEXT_WALK = "prefill"
IMAGE_WALK = "prefill_image"
DIGEST = bytes(range(32))
TEXT = list(range(1, 21))
IMAGE = 30
TAIL = list(range(100, 125))
# slots 0-19 text, 20-49 image, 50-74 text: a hit of 4 pages covers the first two walks
PARTS = [TEXT, IMAGE, TAIL]


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda info, cache: None)


def _config(parts: list) -> KVReqConfig:
    """What the preprocess worker sends for ``parts``: ids, or an image's slot count."""
    total = sum(len(part) if isinstance(part, list) else part for part in parts)
    pages: list[list[int]] = [[] for _ in range(-(-total // PAGE_SIZE))]
    items: dict[int, list[PageItem]] = {}
    spans = []
    at = 0
    for part in parts:
        if isinstance(part, list):
            for slot, token in enumerate(part, start=at):
                pages[slot // PAGE_SIZE].append(token)
            spans.append(PrefixSpan(len(part), len(part), TEXT_WALK))
            at += len(part)
            continue
        for page in range(at // PAGE_SIZE, -(-(at + part) // PAGE_SIZE)):
            first = max(at, page * PAGE_SIZE)
            items.setdefault(page, []).append(PageItem(first - page * PAGE_SIZE, first - at, part, DIGEST))
        spans.append(PrefixSpan(part, 1, IMAGE_WALK, DIGEST))
        at += part
    return KVReqConfig(
        prefix_keys={"main": chain(pages, items)},
        prefix_tail={"main": pages[-1] if total % PAGE_SIZE else []},
        prefix_layout={"main": spans},
    )


def _walks(parts: list) -> list[tuple[str, int]]:
    return [(TEXT_WALK, len(part)) if isinstance(part, list) else (IMAGE_WALK, part) for part in parts]


class _Node:
    """One request's cache and positions under a real runner, after an earlier
    request ran every walk of ``seeded`` and indexed what it filled."""

    def __init__(self, seeded: list | None = PARTS, parts: list | None = None):
        device = torch.device("cpu")
        self.kv = KVManager(
            cfg=KVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
                max_num_pages=64, page_size=PAGE_SIZE,
            ),
            name="kv", joint_comm_group=None, transfer_engine_info=None,
            device=device, dtype=torch.float32,
        )
        self.kv.enable_prefix_cache(ROOT, {"main": (TEXT_WALK, "decode", IMAGE_WALK)})
        self.rope = RopeManager(config=PositionConfig(kv_cache="kv"), device=device, dtype=torch.float32)
        self.runner = StepRunner({"kv": self.kv, "rope": self.rope}, node_resources={NODE: ["kv", "rope"]})
        if seeded:
            self.runner.ingest_request("seed", {"kv": _config(seeded)})
            for walk, length in _walks(seeded):
                self.step(length, walk, rid="seed")
        self.runner.ingest_request(RID, {"kv": _config(parts or seeded)})

    def step(self, span: int, walk: str, rid: str = RID) -> list[int]:
        """Admit, plan and commit one step, and give back its positions."""
        step = SubmoduleStep(steps={"kv": KVStep(), "rope": PositionStep()}, segments=[Segment(rid, "main", span)])
        step.set_ctx(StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False))
        assert self.runner.admit(step).outcome.ok
        positions = self.runner.plan(step)["rope"]["main"].tolist()
        self.runner.commit(step)
        return positions

    def probe(self, walk: str, length: int, **inputs):
        """The engine's probe: every resource answers, then takes the agreed prefix."""
        walk_inputs = ARNodeInputs(input_seq_len=length, **inputs)
        prefix = self.runner.resolve_cached_prefix(RID, NODE, walk, walk_inputs)
        self.runner.apply_cached_prefix(RID, NODE, walk, walk_inputs, prefix)
        return prefix

    def walk(self, walk: str, length: int):
        """One walk as the engine drives it: served whole, or run from the hit's end."""
        prefix = self.probe(walk, length)
        tokens = prefix.tokens if prefix is not None else 0
        if tokens == length:
            self.runner.complete_cached_walk(RID, NODE, walk)
        else:
            self.step(length - tokens, walk)
        return prefix

    @property
    def stream(self):
        return self.kv._streams[RID]["main"]

    @property
    def counter(self) -> int:
        return self.rope.position(RID, "main")



def test_a_hit_across_three_walks_serves_the_two_it_covers_and_trims_the_third():
    node = _Node()

    answers = [node.walk(walk, length).tokens for walk, length in _walks(PARTS)]

    assert answers == [len(TEXT), IMAGE, 4 * PAGE_SIZE - len(TEXT) - IMAGE], (
        "each walk should take the lease's slots past the walks before it"
    )
    assert node.stream.stored_len == len(TEXT) + IMAGE + len(TAIL), (
        "the trimmed walk did not write from where the served walks end"
    )


_DISAGREEING = {
    # one slot more than its span
    "a probe": lambda node: node.probe(IMAGE_WALK, IMAGE + 1),
    # the engine will not cut it, so it writes whole over the lease
    "a walk the engine will not probe": lambda node: node.runner.apply_cached_prefix(
        RID, NODE, IMAGE_WALK, ARNodeInputs(input_seq_len=IMAGE), None,
    ),
}


@pytest.mark.parametrize("disagree", _DISAGREEING.values(), ids=_DISAGREEING.keys())
def test_a_walk_the_layout_no_longer_describes_keeps_the_pages_the_cache_served(disagree):
    parts = [list(range(1, 33)), IMAGE, TAIL]
    node = _Node(parts)
    node.walk(TEXT_WALK, 32)
    lease = list(node.stream.lease)

    disagree(node)

    assert (node.stream.chain, node.stream.stored_len, node.stream.page_indices) == (None, 32, lease[:2]), (
        "the chain dropped at the image let go of the text the cache served, so "
        "the image would be written from slot 0 with that text missing"
    )


def test_a_mismatch_the_served_walks_cannot_survive_fails_the_request_and_says_why(caplog):
    node = _Node()
    node.walk(TEXT_WALK, len(TEXT))

    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError, match="end inside a page"):
        node.probe(IMAGE_WALK, IMAGE + 1)

    disagreement = "prefill_image writes 31 slots where its layout's span 1 has prefill_image write 30"
    assert any(disagreement in record.getMessage() for record in caplog.records), (
        "the failure did not log which walk disagreed with the layout, and how"
    )




def test_each_walk_the_cache_serves_moves_the_counter_by_its_advance_before_the_next_reads_it():
    node = _Node()
    node.walk(TEXT_WALK, len(TEXT))
    past_text = node.counter
    node.walk(IMAGE_WALK, IMAGE)
    past_image = node.counter
    trimmed = node.probe(TEXT_WALK, len(TAIL)).tokens

    positions = node.step(len(TAIL) - trimmed, TEXT_WALK)

    first = len(TEXT) + 1 + trimmed
    assert (past_text, past_image, positions) == (
        len(TEXT), len(TEXT) + 1, list(range(first, first + len(TAIL) - trimmed)),
    ), (
        "a walk after a served one read the counter at the start of what was "
        "served, or an image moved it by its slots rather than its one position"
    )


def test_a_layout_reaches_the_kv_config_across_the_wire():
    config = KVReqConfig()
    config.apply_conductor_config(prefix_layout={"main": [[20, 20, TEXT_WALK, None], [30, 1, IMAGE_WALK, DIGEST]]})
    info = CurrentForwardPassInfo(
        request_id=RID, fwd_index=0, random_seed=0, max_tokens=1,
        graph_walk=TEXT_WALK, partition_name="p0",
    )
    info.resource_configs = {"kv": config}
    msg = WorkerMessage(
        message_type=WorkerMessageType.INPUT_SIGNALS,
        body=InputSignals(request_id=RID, partition_name="p0", request_info=info, inputs=[]),
    )

    decoded = decode(encode(msg)).body.request_info.resource_configs["kv"]

    assert decoded.prefix_layout == {"main": [
        PrefixSpan(20, 20, TEXT_WALK), PrefixSpan(30, 1, IMAGE_WALK, DIGEST),
    ]}, "the layout did not reach the worker as the spans the conductor was sent"
