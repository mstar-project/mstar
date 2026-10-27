"""One lease holds a hit across every walk of a layout, and each walk takes its share.

A prompt laid out as text, image, text is written by three walks into one
stream. The first walk's probe matches the whole layout and holds it. The walks
the lease covers are served from it without running, and a walk it covers in
part runs from the lease's end. A served walk never admits, so the position
counter moves at the probe, and an image block moves it by one position rather
than by its slot count.

An image that ends inside a page shares that page with the text after it, so
the cache keeps a copy of the page up to the image's end for a repeat to reuse.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_prefix_correctness import KV, NODE, PAGE_SIZE, ROOT, ROPE
from test_kv_prefix_correctness import _Node as _TextNode

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.engine.resources import StepContext
from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.model.base import KVLabel, LabelLayout, PrefixKeys, Span
from mstar.model.submodule_base import ARNodeInputs

RID = "r1"
TEXT_WALK = "prefill"
IMAGE_WALK = "prefill_image"
DIGEST = bytes(range(32))
TEXT = list(range(1, 21))
IMAGE = 30
TAIL = list(range(100, 125))
# Slots 0-19 are text, 20-49 the image and 50-74 text, so a hit of 4 pages covers the first two walks.
PARTS = [TEXT, IMAGE, TAIL]
# After TEXT and IMAGE the image ends two slots into page 3, where the question starts.
ASK, OTHER_ASK = list(range(200, 205)), list(range(300, 305))


@pytest.fixture(autouse=True)
def _no_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", lambda *args, **kwargs: None)


def _walks(parts: list) -> list[tuple[str, int]]:
    """Each part's walk and slot count. A part is a list of ids, or an image's
    slot count alone or paired with its digest."""
    return [
        (TEXT_WALK, len(part)) if isinstance(part, list) else (IMAGE_WALK, part if isinstance(part, int) else part[0])
        for part in parts
    ]


def _config(parts: list, text_walk: str = TEXT_WALK, image_walk: str = IMAGE_WALK, decode: int = 0) -> KVReqConfig:
    """What the preprocess worker sends for ``parts``, keyed by its own code."""
    layout = LabelLayout(spans=tuple(
        Span("ids", text_walk, slots, slots, "text_inputs") if isinstance(part, list)
        # The span's source holds the digest itself, which the stand-in `_digest` returns.
        else Span("digest", image_walk, slots, 1, part[1] if isinstance(part, tuple) else DIGEST)
        for part, (_, slots) in zip(parts, _walks(parts), strict=True)
    ), growth=(("decode_loop", 1),))
    texts = [torch.tensor(part) for part in parts if isinstance(part, list)]
    return _sent(layout, texts, (text_walk, image_walk), decode)


def _sent(layout: LabelLayout, texts: list[torch.Tensor], walks: tuple[str, ...], decode: int = 0) -> KVReqConfig:
    layouts = {"kv": {"main": layout}}
    worker = SimpleNamespace(
        _keyed_labels={"kv": {"main": KVLabel(keys=PrefixKeys("text_inputs", walks, growth=True))}},
        _prefix_page_sizes={"kv": PAGE_SIZE}, model=None,
        _digest=lambda span, tensors, file_paths, file_states, clip=None: span.source,
        _page_layout=lambda *args: PreprocessWorkerThread._page_layout(None, *args),
    )
    tensors = {"text_inputs": texts}
    rows = {key: {label: layout.row() for label, layout in by_label.items()} for key, by_label in layouts.items()}
    sent = PreprocessWorkerThread._prefix_keys(worker, tensors, layouts, rows, None, {})
    config = KVReqConfig()
    config.apply_conductor_config(
        kv_layout=rows["kv"], loop_caps={"decode_loop": decode},
        **{name: by_resource["kv"] for name, by_resource in sent.items()},
    )
    return config


class _Node(_TextNode):
    """One request's cache and positions under a real runner, after an earlier
    request ran every walk of ``seeded`` and indexed what it filled."""

    def __init__(self, seeded: list = PARTS, parts: list | None = None, decode: int = 0):
        super().__init__()
        self.kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((TEXT_WALK, IMAGE_WALK), ("decode",))})
        self.runner.ingest_request("seed", {"kv": _config(seeded, decode=decode)})
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
        """Drive one walk as the engine does, served whole or run from the hit's end.
        Returns the slots it was served."""
        prefix = self.probe(walk, length)
        tokens = prefix.tokens if prefix is not None else 0
        if tokens == length:
            self.runner.complete_cached_walk(RID, NODE, walk)
        else:
            self.step(RID, length - tokens, walk)
        return tokens

    def mixed_step(self, spans: dict[str, int], walks: dict[str, str]) -> None:
        """One step of a combined walk, each row writing its span under its own walk."""
        step = SubmoduleStep(
            steps={KV: KVStep(), ROPE: PositionStep()},
            segments=[Segment(rid, "main", span) for rid, span in spans.items()],
        )
        step.set_ctx(StepContext(
            request_ids=tuple(spans), graph_walk="mixed", slot=0, capture=False, request_walks=walks,
        ))
        assert self.runner.admit(step).outcome.ok
        self.runner.plan(step)
        self.runner.commit(step)

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

    # The served text ends inside a page, so its KV cannot be kept once the layout is dropped.
    with pytest.raises(RuntimeError, match="end inside a page"):
        node.probe(IMAGE_WALK, IMAGE + 1)


@pytest.mark.parametrize("probed", [True, False], ids=["probed", "refused"])
def test_a_keyed_text_walk_placing_its_own_positions_fails_probed_or_refused(probed):
    node = _Node()
    inputs = ARNodeInputs(input_seq_len=len(TEXT), custom_pos_ids=torch.arange(len(TEXT)))

    # Pages keyed by ids are cached at consecutive positions, which a walk that places its own does not have.
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
    # The cache is seeded with another prompt, so the first probe misses.
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
    node = _Node([TEXT, IMAGE, ASK], [TEXT, IMAGE, ASK + reply + OTHER_ASK], decode=len(reply))
    for token in reply:
        node.kv.extend_prefix_chain("seed", NODE, "decode", {"text_inputs": [torch.tensor([token])]})
        node.step("seed", 1, "decode")

    answers = [node.walk(walk, length) for walk, length in _walks([TEXT, IMAGE, ASK + reply + OTHER_ASK])]

    assert answers == [len(TEXT), IMAGE, 4 * PAGE_SIZE - len(TEXT) - IMAGE], (
        "the engine keyed the page the reply filled by another reading of the image on "
        "it than the next turn's prompt is keyed by, so the turn recomputed its history"
    )


def test_a_different_image_is_never_served_another_images_end():
    # Three pages of text, then a small image wholly on page 3, so only the image's end key tells the two images apart.
    text = list(range(1, 49))
    node = _Node([text, 10, ASK], [text, (10, bytes(32)), ASK])
    node.walk(TEXT_WALK, len(text))

    assert node.probe(IMAGE_WALK, 10).tokens == 0, (
        "a request was served the end of another image laid out in the same slots"
    )


def test_a_prompt_prefilled_beside_a_decode_is_matched_by_its_repeat():
    prompt = list(range(500, 500 + 3 * PAGE_SIZE))
    node = _Node(parts=[prompt], decode=1)
    node.runner.ingest_request("first", {"kv": _config([prompt])})

    node.mixed_step({"first": len(prompt), "seed": 1}, {"first": TEXT_WALK, "seed": "decode"})

    assert node.walk(TEXT_WALK, len(prompt)) == 2 * PAGE_SIZE, (
        "a prompt prefilled in a step of a combined walk filed no pages, so its repeat recomputed it"
    )


# One prefill step can write several keyed parts: the ids before an audio clip, then one item
# per complete encoder window, each a slot a position, then rows nothing keys. A streaming
# re-decode of the same clip, longer, is that step again with more windows.
HEAD = list(range(1, 21))
WINDOW = 26


def _one_step(windows: list[bytes], rows: int, window: int = WINDOW) -> tuple[KVReqConfig, int]:
    spans = (
        Span("ids", TEXT_WALK, len(HEAD), len(HEAD), "text_inputs"),
        *(Span("digest", TEXT_WALK, window, window, digest) for digest in windows),
        Span("rows", TEXT_WALK, rows, rows, "text_inputs"),
    )
    return _sent(LabelLayout(spans=spans), [torch.tensor(HEAD)], (TEXT_WALK,)), len(HEAD) + window * len(windows) + rows


_A, _B, _C, _LOUDER_B = (bytes([n]) * 32 for n in (1, 2, 3, 4))


@pytest.mark.parametrize(("windows", "window", "rows", "held"), [
    # the windows both requests share end at slot 72, inside page 4, whose cached end serves it
    ([_A, _B, _C], WINDOW, 5, 20 + 2 * WINDOW),
    # B differs from page 2 on, so the hit stops at page 2, inside A, and A's rest runs
    ([_A, _LOUDER_B, _C], WINDOW, 5, 2 * PAGE_SIZE),
    # the same windows: every keyed slot is served, since the rows after them still run
    ([_A, _B], WINDOW, 5, 20 + 2 * WINDOW),
    # rows reaching past page 4: its cached end is still served only up to the keyed end
    ([_A, _B], WINDOW, 20, 20 + 2 * WINDOW),
    # B and C end on the same page: C's end was never cached, B's is
    ([_A, _B, _C], 3, 5, 20 + 2 * 3),
], ids=["a longer clip", "a window keyed apart", "the same windows", "rows past the end's page",
        "two ends on one page"])
def test_one_step_of_several_keyed_parts_is_served_what_it_shares_and_placed_past_it(windows, window, rows, held):
    node = _TextNode()
    node.kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((TEXT_WALK,))})
    seed, length = _one_step([_A, _B], 7, window)
    node.runner.ingest_request("seed", {"kv": seed})
    node.step("seed", length)
    queried, hit = node.kv.prefix_tokens.get(NODE, (0, 0))
    config, length = _one_step(windows, rows, window)
    node.runner.ingest_request(RID, {"kv": config})

    inputs = ARNodeInputs(input_seq_len=length)
    prefix = node.runner.resolve_cached_prefix(RID, NODE, TEXT_WALK, inputs)
    node.runner.apply_cached_prefix(RID, NODE, TEXT_WALK, inputs, prefix)
    positions = node.step(RID, length - prefix.tokens)

    assert (prefix.tokens, positions) == (held, list(range(held, length))), (
        "a step writing several keyed parts was refused its hit, served slots past its keyed end, "
        "or the slots after a hit that ends inside an item were placed at the item's start"
    )
    after = node.kv.prefix_tokens[NODE]
    assert after[1] - hit <= after[0] - queried, "the cache counted more hit tokens than it was queried for"
