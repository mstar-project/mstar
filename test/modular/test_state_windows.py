"""Windows a read keeps: a kept window on the bidirectional mask, and a window of frames.

LingBot World keeps a sink and the latest frames of its rollout, at sizes that follow each request's resolution, and
Waypoint keeps a ring of frames per layer. A window planned wrong changes what every later frame sees, with no error of
its own, so each declaration today's resources can't plan is refused when the specs are built, naming the read.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.state import Array, Attend, Bidirectional, Causal, KeptWindow, KVEntry, Layout, Read
from mstar.engine.state.adapters import build_specs, kv_layouts
from mstar.engine.state.analysis import state_declaration_problems
from mstar.graph.base import GraphNode, Loop
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import ChunkingPolicy

HEADS, DIM = 2, 8


def _array(name="kv", *, reads, max_len=64, **fields) -> Array:
    return Array(
        name, nodes={"dit"}, layers={0, 1}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=max_len, reads=reads,
        **fields,
    )


def _read(mask, **fields) -> Read:
    return Read(mask=mask, op=Attend(q_heads=HEADS), **fields)


@pytest.mark.parametrize(("make", "message"), [
    (lambda: Bidirectional(kept_window=1), "Bidirectional.kept_window must be a bool, got 1"),
    (lambda: KeptWindow(-1, window=4), "KeptWindow.sink must be an int >= 0, got -1"),
    (lambda: KeptWindow(2, window=-1), "KeptWindow.window must be an int >= 0, got -1"),
    (lambda: KeptWindow(2.0, window=4), "KeptWindow.sink must be an int >= 0, got 2.0"),
    (lambda: KeptWindow(True, window=4), "KeptWindow.sink must be an int >= 0, got True"),
], ids=["kept_window_int", "sink_negative", "window_negative", "sink_float", "sink_bool"])
def test_a_window_refuses_its_own_bad_fields(make, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        make()


def test_a_kept_window_has_no_layout_until_it_is_bound():
    kept = Bidirectional(kept_window=True)
    for arrays in (
        [_array(reads={"attn": _read(kept)})],
        [_array(reads={"attn": _read(kept, walks={"a"}), "other": _read(Causal(), walks={"b"})})],
        [_array("q", reads={"own": _read(Causal())}), _array(reads={"attn": _read(kept, queries="q")})],
    ):
        with pytest.raises(ValueError, match=r"Array 'kv': no layout yet: reads\['attn'\].mask keeps a window"):
            build_specs(arrays)
    plain = _array(reads={"attn": _read(Causal())})
    with pytest.raises(ValueError, match="no layout yet: label 'main' gives kept_windows"):
        kv_layouts([plain], {"kv": {"main": Layout(kept_windows={"attn": KeptWindow(4, window=8)})}}, {})


class _Model:
    """Node ``dit`` prefills from ids, then decodes in loop ``loop``, under the policies and prefix keys given."""

    def __init__(self, *arrays, chunked=(), kv_labels=None, rebuilt=False):
        self.arrays = list(arrays)
        self.chunked = chunked
        self.labels = kv_labels or {}
        self.rebuilt = rebuilt

    def declare_state(self):
        return self.arrays

    def get_graph_walk_graphs(self):
        return {
            "prefill": GraphNode(name="dit", input_names=["ids"], outputs=[]),
            "decode": Loop(section=GraphNode(name="dit", input_names=[], outputs=[]), max_iters=8, outputs=[],
                           name="loop"),
        }

    def get_combined_graph_walks(self):
        return {}

    def get_chunking_policies(self):
        return {"dit": (ChunkingPolicy(walks=self.chunked, chunkable=True),)} if self.chunked else {}

    def kv_labels(self):
        return self.labels

    def rebuild_policies(self):
        if not self.rebuilt:
            return {}
        return {"dit": RebuildPolicy(walks=frozenset({"prefill"}), ids_input="ids", sample_output="token", loop="loop")}


def test_a_rebuild_refuses_a_kept_window_on_its_decode_walks():
    """A rebuild recomputes the decode steps in prefill rows, where a kept window cuts other entries than each step
    saw. Only the kept window takes the loop's walks."""
    kept = _array(reads={"attn": _read(Bidirectional(kept_window=True), walks={"decode"})})
    assert state_declaration_problems(_Model(kept, rebuilt=True)) == [
        "Array 'kv'.reads['attn']: Bidirectional on walk 'decode', which dit's rebuild policy replaces with its "
        "prefill rows. A Block or Bidirectional read sees each step's adds as a whole, so drop the rebuild policy"
    ]
    plain = _array(reads={"attn": _read(Bidirectional(), walks={"decode"})})
    assert state_declaration_problems(_Model(plain, rebuilt=True)) == [], (
        "a plain Bidirectional read on decode walks was refused"
    )
