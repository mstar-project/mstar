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
from dataclasses import replace
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
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan
from mstar.engine.resources.kv.manager import KVManager, SpanCursor
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


def _config(
    parts: list, text_walk: str = TEXT_WALK, image_walk: str = IMAGE_WALK, decode: int = 0, opens: tuple | None = None,
) -> KVReqConfig:
    """What the preprocess worker sends for ``parts``, keyed by its own code. ``opens``
    holds the indices of the parts that open a step, by default every part."""
    layouts = {"kv": {"main": LabelLayout(spans=tuple(
        Span("ids", text_walk, slots, slots, "text_inputs", opens_step=opens is None or i in opens)
        if isinstance(part, list)
        # The span's source holds the digest itself, which the stand-in `_digest` returns.
        else Span(
            "digest", image_walk, slots, 1, part[1] if isinstance(part, tuple) else DIGEST,
            opens_step=opens is None or i in opens,
        )
        for i, (part, (_, slots)) in enumerate(zip(parts, _walks(parts), strict=True))
    ), growth=(("decode_loop", 1),))}}
    walks = (text_walk, image_walk)
    worker = SimpleNamespace(
        _keyed_labels={"kv": {"main": KVLabel(keys=PrefixKeys("text_inputs", walks, growth=True))}},
        _prefix_page_sizes={"kv": PAGE_SIZE}, model=None,
        _digest=lambda span, file_paths, file_states: span.source,
        _page_layout=lambda *args: PreprocessWorkerThread._page_layout(None, *args),
    )
    tensors = {"text_inputs": [torch.tensor(part) for part in parts if isinstance(part, list)]}
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


def _one_row(parts: list = PARTS) -> tuple[_Node, int]:
    """A request whose text and image spans are one row of one step, after an
    earlier request wrote them the same way."""
    node = _Node.__new__(_Node)
    _TextNode.__init__(node)
    node.kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((IMAGE_WALK,), ("decode",))})
    node.runner.ingest_request("seed", {"kv": _config(parts, text_walk=IMAGE_WALK, opens=(0,))})
    length = sum(slots for _, slots in _walks(parts))
    node.step("seed", length, IMAGE_WALK)
    node.runner.ingest_request(RID, {"kv": _config(parts, text_walk=IMAGE_WALK, opens=(0,))})
    return node, length


def test_one_step_writing_several_spans_is_served_past_their_whole_pages():
    node, length = _one_row()

    tokens = node.walk(IMAGE_WALK, length)

    assert tokens == 4 * PAGE_SIZE, "the row should run from the lease's end"
    assert node.stream.stored_len == length
    assert node.stream.span_cursor.done == len(PARTS), "the step did not pass every span it wrote"


def test_a_partly_served_row_leaves_the_counter_where_its_own_positions_start():
    node, length = _one_row()

    prefix = node.probe(IMAGE_WALK, length)

    assert prefix.tokens == 4 * PAGE_SIZE and prefix.position == 0, (
        "a row places its own positions, so prepared again it has to read the counter it was prepared at"
    )


def test_a_whole_row_the_cache_holds_moves_the_counter_past_its_spans():
    # the prompt's last token always runs, so a whole row is one a later walk follows
    spans = (
        PrefixSpan(10, 10, IMAGE_WALK), PrefixSpan(22, 1, IMAGE_WALK, digest=DIGEST, opens_step=False),
        PrefixSpan(5, 5, TEXT_WALK),
    )
    cursor = SpanCursor(spans)
    step = cursor.next_step(IMAGE_WALK)

    served = KVManager._answer("main", cursor, step, 32)

    assert (len(step), served.tokens, served.position) == (2, 32, 11), "an item moves it by its advance, not its slots"
    assert KVManager._answer("main", cursor, step, 16).position == 0


def test_a_row_the_layout_does_not_add_up_to_ends_the_chain():
    node, length = _one_row()

    assert node.probe(IMAGE_WALK, length + 1) is None
    assert node.stream.chain is None


def test_a_row_may_place_its_own_positions_though_it_opens_with_text():
    node, length = _one_row()
    row = ARNodeInputs(input_seq_len=length, custom_pos_ids=torch.zeros(3, length))

    assert node.runner.resolve_cached_prefix(RID, NODE, IMAGE_WALK, row).tokens == 4 * PAGE_SIZE


# Two images of one walk, a step each: text and the first image, then text, the second image and text.
BETWEEN = list(range(60, 65))
OTHER_DIGEST = bytes(range(1, 33))
TWO_IMAGES = [TEXT, (IMAGE, DIGEST), BETWEEN, (IMAGE, OTHER_DIGEST), TAIL]
TWO_STEPS = (len(TEXT) + IMAGE, len(BETWEEN) + IMAGE + len(TAIL))


def _two_steps(seeded: list = TWO_IMAGES, parts: list = TWO_IMAGES) -> _Node:
    """A request of two steps of one walk, after an earlier request ran both steps of ``seeded``."""
    node = _Node.__new__(_Node)
    _TextNode.__init__(node)
    node.kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((IMAGE_WALK,), ("decode",))})
    node.runner.ingest_request("seed", {"kv": _config(seeded, text_walk=IMAGE_WALK, opens=(0, 2))})
    for length in TWO_STEPS:
        node.step("seed", length, IMAGE_WALK)
    node.runner.ingest_request(RID, {"kv": _config(parts, text_walk=IMAGE_WALK, opens=(0, 2))})
    return node


def test_a_repeat_of_two_steps_of_one_walk_is_served_from_the_cache(caplog):
    node = _two_steps()

    first, second = (node.walk(IMAGE_WALK, length) for length in TWO_STEPS)

    held = 6 * PAGE_SIZE  # every whole page but the last, which the prompt's final token has to run in
    assert (first, second) == (TWO_STEPS[0], held - TWO_STEPS[0]), (
        "the first step should be served whole and the second from the lease's end"
    )
    assert node.stream.chain is not None and "its chain ends here" not in caplog.text
    assert node.stream.stored_len == sum(TWO_STEPS)
    assert node.stream.span_cursor.done == len(TWO_IMAGES)


def test_the_counter_stands_past_a_served_step_when_the_next_one_reads_it():
    node = _two_steps()

    node.walk(IMAGE_WALK, TWO_STEPS[0])

    assert node.rope.position(RID, "main") == len(TEXT) + 1, "an image moves it by its advance, not its slots"
    assert node.probe(IMAGE_WALK, TWO_STEPS[1]).position == len(TEXT) + 1, (
        "a partly served row places its own positions from where the step before it ended"
    )


def test_two_steps_are_served_no_further_than_the_image_they_share():
    other_second = [TEXT, (IMAGE, DIGEST), BETWEEN, (IMAGE, bytes(range(2, 34))), TAIL]
    node = _two_steps(parts=other_second)

    first, second = (node.walk(IMAGE_WALK, length) for length in TWO_STEPS)

    assert (first, second) == (TWO_STEPS[0], 0), "another image's KV would have served the second step"


def test_consecutive_spans_of_one_walk_with_no_mark_are_a_step_each():
    # as BAGEL lays out two adjacent images, which run a walk each
    parts = [TEXT, (IMAGE, DIGEST), (IMAGE, OTHER_DIGEST), TAIL]
    node = _Node(parts)

    answers = [node.walk(walk, length) for walk, length in _walks(parts)]

    assert answers == [len(TEXT), IMAGE, IMAGE, 6 * PAGE_SIZE - len(TEXT) - 2 * IMAGE], (
        "two adjacent items would be summed against one walk's row, and the layout dropped"
    )
    assert node.stream.chain is not None


def test_windows_past_the_keyed_spans_are_never_looked_up(caplog):
    # as Cosmos3 chains its windows: keyed text, then rows of one walk, a step each
    node = _Node.__new__(_Node)
    _TextNode.__init__(node)
    node.kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((TEXT_WALK,), ("decode",))})
    config = _config([TEXT])
    layout = config.kv_layout["main"]
    windows = (PrefixSpan(8, 8, IMAGE_WALK), PrefixSpan(8, 8, IMAGE_WALK))
    config.kv_layout["main"] = replace(layout, spans=(*layout.spans, *windows))
    node.runner.ingest_request(RID, {"kv": config})

    served = [node.walk(TEXT_WALK, len(TEXT)), *(node.walk(IMAGE_WALK, span.length) for span in windows)]

    assert served == [0, 0, 0] and "its chain ends here" not in caplog.text
    assert node.stream.stored_len == len(TEXT) + 16


def test_a_row_of_the_older_forms_parses_and_opens_a_step():
    config = KVReqConfig()
    config.apply_conductor_config(
        kv_layout={"main": [[[5, 5, IMAGE_WALK, 0], [7, 1, IMAGE_WALK, 0, DIGEST, b"end"]], [], 0, None, 2]},
    )
    spans = config.kv_layout["main"].spans

    assert [span.opens_step for span in spans] == [True, True]
    assert (spans[1].digest, spans[1].end_key) == (DIGEST, b"end")
    assert SpanCursor(spans).next_step(IMAGE_WALK) == spans[:1]


def test_the_mark_reaches_the_manager_on_a_keyed_span():
    spans = _config(TWO_IMAGES, text_walk=IMAGE_WALK, opens=(0, 2)).kv_layout["main"].spans

    assert [span.opens_step for span in spans] == [True, False, True, False, False]
    assert [span.digest for span in spans] == [None, DIGEST, None, OTHER_DIGEST, None]


def test_a_step_that_changes_walk_fails_where_its_layout_is_built():
    # `next_step` ends a step at a change of walk, so the image would be looked up as a step of its own
    def layout(text_walk: str, opens_step: bool = True) -> LabelLayout:
        return LabelLayout(spans=(
            Span("ids", text_walk, len(TEXT), len(TEXT), "text_inputs", opens_step=opens_step),
            Span("digest", IMAGE_WALK, IMAGE, 1, DIGEST, opens_step=False),
        ))

    with pytest.raises(ValueError, match=f"span 1 of walk '{IMAGE_WALK}'.*span 0 before it is of walk '{TEXT_WALK}'"):
        layout(TEXT_WALK).row()
    with pytest.raises(ValueError, match="span 0 .* no span comes before it"):
        layout(IMAGE_WALK, opens_step=False).row()
    assert [row[-1] for row in layout(IMAGE_WALK).row()[0]] == [True, False]


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
