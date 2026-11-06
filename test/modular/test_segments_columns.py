"""`Segments`: a step's rows as columns.

What the engine relies on: rows read back exactly as declared, a column every
row agrees on is held as one value, equal rows compare and hash equal however
they were declared, and no column can be changed through a reference to it.
"""

from __future__ import annotations

import copy
import pickle
from enum import Enum

import pytest

from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.kv.plan import group_by_plan_label
from mstar.engine.resources.step import NO_SEGMENTS, Segment, Segments, StepContext, SubmoduleStep


def _rows(segments: Segments) -> list[tuple]:
    return [tuple(row) for row in segments]


def test_a_column_every_row_agrees_on_is_one_value():
    segs = Segments(["a", "b"], ["main", "main"], [1, 1])
    assert segs.labels == "main" and segs.spans == 1
    assert _rows(segs) == [("a", "main", 1), ("b", "main", 1)]


def test_a_column_rows_disagree_on_stays_per_row():
    segs = Segments(["a", "b"], ["main", "ctx"], [1, 0])
    assert segs.labels == ("main", "ctx") and segs.spans == (1, 0)
    assert _rows(segs) == [("a", "main", 1), ("b", "ctx", 0)]
    assert segs.label_column() == ("main", "ctx") and segs.span_column() == (1, 0)


def test_columns_read_as_columns_either_way():
    segs = Segments(("a", "b", "c"), "main", 2)
    assert segs.label_column() == ("main",) * 3 and segs.span_column() == (2, 2, 2)
    assert list(segs.rows()) == [("a", "main", 2), ("b", "main", 2), ("c", "main", 2)]
    assert list(segs.pairs()) == [("a", "main"), ("b", "main"), ("c", "main")]


@pytest.mark.parametrize("labels,spans", [(["x"], [1, 2]), (["x", "y"], [1])])
def test_a_column_of_the_wrong_length_is_refused(labels, spans):
    with pytest.raises(ValueError, match="request ids"):
        Segments(["a", "b"], labels, spans)


def test_a_str_subclass_label_is_one_value():
    class Label(str, Enum):
        MAIN = "main"

    assert Segments(["a", "b"], Label.MAIN, 1).labels == "main"


def test_copies_and_pickles_are_equal():
    segs = Segments(("a", "b"), ("x", "y"), (1, 2))
    assert copy.deepcopy(segs) == segs and pickle.loads(pickle.dumps(segs)) == segs


def test_the_usual_declaration_reads_each_inputs_length():
    class Row:
        def __init__(self, n):
            self.input_seq_len = n

    assert Segments.of_inputs(["a", "b"], [Row(3), Row(3)]) == Segments(["a", "b"], "main", 3)
    assert Segments.of_inputs(["a", "b"], [Row(3), Row(1)], "ctx").spans == (3, 1)


def test_rows_compare_and_hash_by_value_however_they_were_declared():
    rows = [Segment("a", "main", 1), Segment("b", "main", 1)]
    columns = Segments(("a", "b"), "main", 1)
    assert Segments.from_rows(rows) == columns
    assert hash(Segments.from_rows(rows)) == hash(columns)
    # a tuple of the rows is the same value, and a list of them equals it too
    assert columns == tuple(rows) and columns == rows
    assert hash(columns) == hash(tuple(rows))
    assert columns != Segments(("a", "b"), "main", 2)
    assert columns != Segments(("b", "a"), "main", 1)
    assert columns != Segments(("a", "b"), ("main", "ctx"), 1)


def test_from_rows_keeps_interleaved_order():
    rows = [
        Segment("a", "video", 4), Segment("a", "text", 2),
        Segment("b", "video", 4), Segment("b", "text", 3),
    ]
    segs = Segments.from_rows(rows)
    assert list(segs) == rows
    assert segs.request_ids == ("a", "a", "b", "b")
    assert segs.labels == ("video", "text", "video", "text")
    assert segs.spans == (4, 2, 4, 3)


def test_an_empty_one_answers_as_empty_rows_do():
    for empty in (Segments(()), Segments.from_rows([]), Segments([], "main", []), NO_SEGMENTS):
        assert not empty and len(empty) == 0 and list(empty) == [] and list(empty.rows()) == []
        assert empty.group_by_label() == {}
        assert empty.span_column() == () and empty.label_column() == ()
        assert empty == NO_SEGMENTS


def test_slices_are_segments_and_recanonicalize():
    segs = Segments(("a", "b", "c"), ("x", "x", "y"), (1, 1, 2))
    head = segs[:2]
    assert type(head) is Segments and head.labels == "x" and head.spans == 1
    assert segs[0] == Segment("a", "x", 1) and segs[-1] == Segment("c", "y", 2)
    assert segs[2:] == Segments(("c",), "y", 2)
    assert not segs[3:]


def test_group_by_label_is_the_rows_own_order():
    segs = Segments(("a", "b", "a", "b"), ("x", "y", "y", "x"), (1, 2, 3, 4))
    groups = segs.group_by_label()
    assert list(groups) == ["x", "y"]
    assert _rows(groups["x"]) == [("a", "x", 1), ("b", "x", 4)]
    assert _rows(groups["y"]) == [("b", "y", 2), ("a", "y", 3)]


def test_group_by_label_of_one_label_is_the_same_object():
    segs = Segments(("a", "b"), "main", 1)
    assert segs.group_by_label()["main"] is segs


def test_combined_labels_concat_label_major():
    segs = Segments(("a", "b", "a", "b"), ("cond", "cond", "uncond", "uncond"), 1)
    grouped = group_by_plan_label(segs, {("cond", "uncond"): "cfg"})
    assert list(grouped) == ["cfg"]
    assert _rows(grouped["cfg"]) == [("a", "cond", 1), ("b", "cond", 1), ("a", "uncond", 1), ("b", "uncond", 1)]
    # a source label with no rows is still the caller's error
    with pytest.raises(KeyError):
        group_by_plan_label(segs, {("cond", "other"): "cfg"})


def test_concat_keeps_order_and_recanonicalizes():
    a, b = Segments(("a",), "x", 1), Segments(("b",), "x", 1)
    joined = Segments.concat((a, b))
    assert joined == Segments(("a", "b"), "x", 1) and joined.labels == "x" and joined.spans == 1
    assert Segments.concat((a, NO_SEGMENTS)) is a
    assert Segments.concat(()) == NO_SEGMENTS
    assert Segments.concat((a, Segments(("b",), "y", 2))) == Segments(("a", "b"), ("x", "y"), (1, 2))


def test_nothing_reaches_a_column_through_a_reference_to_it():
    rids, spans = ["a", "b"], [1, 2]
    segs = Segments(rids, "main", spans)
    rids.append("c")
    spans[0] = 9
    assert segs.request_ids == ("a", "b") and segs.spans == (1, 2)
    with pytest.raises(AttributeError):
        segs.spans = 3
    # the engine pads the context's own list after declaring
    ctx = StepContext(request_ids=["a"], graph_walk="w", slot=0, capture=False)
    step = SubmoduleStep(segments=Segments(ctx.request_ids), steps={})
    ctx.request_ids.append(-1)
    assert step.segments.request_ids == ("a",)


def test_a_step_normalizes_rows_once_and_shares_the_result():
    rows = [Segment("a", "main", 1)]
    own = (Segment("a", "ctx", 0),)
    step = SubmoduleStep(segments=rows, steps={"kv": KVStep(), "cross": KVStep(segments=own)})
    assert type(step.segments) is Segments and step.get("kv").segments is step.segments
    assert type(step.get("cross").segments) is Segments and step.get("cross").segments == own
    # a `Segments` is taken as it is: identity is what the plan cache's commit check reads
    segs = Segments(("a",))
    assert SubmoduleStep(segments=segs, steps={}).segments is segs
    assert KVStep(segments=segs).segments is segs


def test_rows_are_built_once():
    segs = Segments(("a", "b"))
    first = list(segs)
    assert all(type(row) is Segment for row in first)
    assert all(x is y for x, y in zip(first, segs, strict=True))
    # read-backs of declared rows are the declared objects
    rows = [Segment("a", "main", 1)]
    assert next(iter(Segments.from_rows(rows))) is rows[0]
