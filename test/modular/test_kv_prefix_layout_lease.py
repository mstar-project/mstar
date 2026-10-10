"""One lease holds a hit across every walk of a layout, and each walk takes its share.

A prompt laid out text, image, text is three walks into one stream. The first
walk's probe matches the whole layout and holds it; the walks it covers are
served from that lease without running, and the one it covers in part runs
from the lease's end. A served walk never admits, so the position counter moves
at the probe, and by an image block's one position, not its slots.

An image ending inside a page shares that page with the text after it, so the
cache keeps a copy of the page cut at the image's end for a repeat to be served.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_prefix_correctness import NODE, PAGE_SIZE, ROOT
from test_kv_prefix_correctness import _Node as _TextNode

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.model.base import PrefixStream, Span
from mstar.model.submodule_base import ARNodeInputs

RID = "r1"
TEXT_WALK = "prefill"
IMAGE_WALK = "prefill_image"
DIGEST = bytes(range(32))
TEXT = list(range(1, 21))
IMAGE = 30
TAIL = list(range(100, 125))
# slots 0-19 text, 20-49 image, 50-74 text: a hit of 4 pages covers the first two walks
PARTS = [TEXT, IMAGE, TAIL]
# after TEXT and IMAGE the image ends two slots into page 3, which the question fills
ASK, OTHER_ASK = list(range(200, 205)), list(range(300, 305))


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: None)


def _walks(parts: list) -> list[tuple[str, int]]:
    """Each part's walk and slots: ids, or an image's slot count, alone or with its digest."""
    return [
        (TEXT_WALK, len(part)) if isinstance(part, list) else (IMAGE_WALK, part if isinstance(part, int) else part[0])
        for part in parts
    ]


def _config(parts: list, text_walk: str = TEXT_WALK, image_walk: str = IMAGE_WALK) -> KVReqConfig:
    """What the preprocess worker sends for ``parts``, keyed by its own code."""
    layout = [
        Span("ids", text_walk, slots, slots, "text_inputs") if isinstance(part, list)
        # the digest rides where a file's name would, for the stand-in `_digest` to hand back
        else Span("digest", image_walk, slots, 1, part[1] if isinstance(part, tuple) else DIGEST)
        for part, (_, slots) in zip(parts, _walks(parts), strict=True)
    ]
    worker = SimpleNamespace(
        _prefix_streams={"kv": {"main": PrefixStream("text_inputs", "ids", text_walk, "decode", (image_walk,))}},
        _prefix_page_sizes={"kv": PAGE_SIZE}, model=None,
        _digest=lambda span, file_paths, file_states: span.source,
        _page_layout=lambda *args: PreprocessWorkerThread._page_layout(None, *args),
    )
    tensors = {"text_inputs": [torch.tensor(part) for part in parts if isinstance(part, list)]}
    sent = PreprocessWorkerThread._prefix_keys(worker, tensors, {"kv": {"main": layout}}, None, {})
    config = KVReqConfig()
    config.apply_conductor_config(**{name: by_resource["kv"] for name, by_resource in sent.items()})
    return config


class _Node(_TextNode):
    """One request's cache and positions under a real runner, after an earlier
    request ran every walk of ``seeded`` and indexed what it filled."""

    def __init__(self, seeded: list = PARTS, parts: list | None = None):
        super().__init__()
        self.kv.enable_prefix_cache(ROOT, {"main": (TEXT_WALK, "decode", IMAGE_WALK)})
        self.runner.ingest_request("seed", {"kv": _config(seeded)})
        for walk, length in _walks(seeded):
            self.step("seed", length, walk)
        self.runner.ingest_request(RID, {"kv": _config(parts or seeded)})

    def probe(self, walk: str, length: int):
        """The engine's probe: every resource answers, then takes the agreed prefix."""
        walk_inputs = ARNodeInputs(input_seq_len=length)
        prefix = self.runner.resolve_cached_prefix(RID, NODE, walk, walk_inputs)
        self.runner.apply_cached_prefix(RID, NODE, walk, walk_inputs, prefix)
        return prefix

    def walk(self, walk: str, length: int) -> int:
        """One walk as the engine drives it, served whole or run from the hit's end; the slots it was served."""
        prefix = self.probe(walk, length)
        tokens = prefix.tokens if prefix is not None else 0
        if tokens == length:
            self.runner.complete_cached_walk(RID, NODE, walk)
        else:
            self.step(RID, length - tokens, walk)
        return tokens

    @property
    def stream(self):
        return self.kv._streams[RID]["main"]


def test_a_hit_across_three_walks_serves_the_two_it_covers_and_trims_the_third():
    node = _Node()

    answers = [node.walk(walk, length) for walk, length in _walks(PARTS)]

    assert answers == [len(TEXT), IMAGE, 4 * PAGE_SIZE - len(TEXT) - IMAGE], (
        "each walk should take the lease's slots past the walks before it"
    )
    assert node.stream.stored_len == len(TEXT) + IMAGE + len(TAIL), (
        "the trimmed walk did not write from where the served walks end"
    )


def test_each_walk_the_cache_serves_moves_the_counter_by_its_advance_before_the_next_reads_it():
    node = _Node()
    node.walk(TEXT_WALK, len(TEXT))
    past_text = node.rope.position(RID, "main")
    node.walk(IMAGE_WALK, IMAGE)
    past_image = node.rope.position(RID, "main")
    trimmed = node.probe(TEXT_WALK, len(TAIL)).tokens

    positions = node.step(RID, len(TAIL) - trimmed)

    first = len(TEXT) + 1 + trimmed
    assert (past_text, past_image, positions) == (
        len(TEXT), len(TEXT) + 1, list(range(first, first + len(TAIL) - trimmed)),
    ), (
        "a walk after a served one read the counter at the start of what was "
        "served, or an image moved it by its slots rather than its one position"
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
    node = _Node([list(range(1, 33)), IMAGE, TAIL])
    node.walk(TEXT_WALK, 32)
    lease = list(node.stream.lease)

    disagree(node)

    assert (node.stream.chain, node.stream.stored_len, node.stream.page_indices) == (None, 32, lease[:2]), (
        "the chain dropped at the image let go of the text the cache served, so "
        "the image would be written from slot 0 with that text missing"
    )


def test_a_mismatch_the_served_walks_cannot_survive_fails_the_request():
    node = _Node()
    node.walk(TEXT_WALK, len(TEXT))

    # served text ending inside a page: dropped, the image would be written with that text missing
    with pytest.raises(RuntimeError, match="end inside a page"):
        node.probe(IMAGE_WALK, IMAGE + 1)


@pytest.mark.parametrize("probed", [True, False], ids=["probed", "refused"])
def test_a_keyed_text_walk_placing_its_own_positions_fails_probed_or_refused(probed):
    node = _Node()
    inputs = ARNodeInputs(input_seq_len=len(TEXT), custom_pos_ids=torch.arange(len(TEXT)))

    # keyed by its ids, whose positions are their count: its pages would be served at positions they never had
    with pytest.raises(AssertionError, match="the layout keys it by ids"):
        if probed:
            node.runner.resolve_cached_prefix(RID, NODE, TEXT_WALK, inputs)
        else:
            node.runner.apply_cached_prefix(RID, NODE, TEXT_WALK, inputs, None)


def test_a_repeat_with_another_question_is_served_the_whole_image_and_the_slots_first_written():
    node = _Node([TEXT, IMAGE, ASK], [TEXT, IMAGE, OTHER_ASK])
    kv = node.kv.kv_cache.tensor
    written = kv[:, node.kv._streams["seed"]["main"].page_indices[3], :, :2].clone()

    answers = [node.walk(walk, length) for walk, length in _walks([TEXT, IMAGE, OTHER_ASK])]

    assert answers == [len(TEXT), IMAGE, 0], (
        "the repeat matched only the whole pages before the image's end, so the "
        "image's walk ran again for the slots its page shares with the question"
    )
    assert torch.equal(kv[:, node.stream.page_indices[3], :, :2], written), (
        "the repeat's page for the image's end does not hold the slots the first "
        "request wrote there, so its question attends to other KV"
    )


def test_a_missed_probe_asked_again_hashes_nothing_and_is_served_what_was_cached_since(monkeypatch):
    # seeded with another prompt, so the first probe finds none of this one
    node = _Node([TAIL], [TEXT, IMAGE, OTHER_ASK])
    assert node.probe(TEXT_WALK, len(TEXT)) is None
    node.runner.ingest_request("first", {"kv": _config([TEXT, IMAGE, ASK])})
    for walk, length in _walks([TEXT, IMAGE, ASK]):
        node.step("first", length, walk)
    monkeypatch.setattr(manager_mod, "fingerprint", lambda *fields: pytest.fail("a second probe hashed its keys again"))

    answers = [node.walk(walk, length) for walk, length in _walks([TEXT, IMAGE])]

    assert answers == [len(TEXT), IMAGE], (
        "the keys a missed probe kept did not find the pages and the image end "
        "cached since, so a refused step's request lost its hit"
    )


def test_a_page_a_reply_fills_after_an_image_is_matched_by_the_next_turn():
    reply = list(range(400, 409))
    # the image's last two slots, the question and the reply's nine ids fill page 3
    node = _Node([TEXT, IMAGE, ASK], [TEXT, IMAGE, ASK + reply + OTHER_ASK])
    for token in reply:
        node.kv.extend_prefix_chain("seed", NODE, "decode", {"text_inputs": [torch.tensor([token])]})
        node.step("seed", 1, "decode")

    answers = [node.walk(walk, length) for walk, length in _walks([TEXT, IMAGE, ASK + reply + OTHER_ASK])]

    assert answers == [len(TEXT), IMAGE, 4 * PAGE_SIZE - len(TEXT) - IMAGE], (
        "the engine keyed the page the reply filled by another reading of the image on "
        "it than the next turn's prompt is keyed by, so the turn recomputed its history"
    )


def test_a_different_image_is_never_served_another_images_end():
    # three pages of text, then a small image wholly on page 3: only the image's end key tells the two apart
    text = list(range(1, 49))
    node = _Node([text, 10, ASK], [text, (10, bytes(32)), ASK])
    node.walk(TEXT_WALK, len(text))

    assert node.probe(IMAGE_WALK, 10).tokens == 0, (
        "a request was served the end of another image laid out in the same slots"
    )
