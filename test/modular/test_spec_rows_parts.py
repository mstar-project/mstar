"""The Python pieces of the columnar speculation: the lazily built loop-back
inputs, the loop-counter snapshots, the request state's batch lookup of
forward-pass infos, and the engine's wrapped infos. The whole path against the
per-row one, on both runtimes, is in ``test/rust/test_spec_rows.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.engine import ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.submodule_base import EMPTY_INPUT_METADATA, InputMetadata  # noqa: E402
from mstar.worker.node_manager_utils import RequestStateManager  # noqa: E402
from mstar.worker.spec_rows import DeferredRowInputs, loop_counter_snapshots  # noqa: E402


def _info(rid: int, counts: dict, partition: str = "p") -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=f"r{rid}", graph_walk="decode", fwd_index=0, random_seed=0,
        max_tokens=9, rid_handle=rid, partition_name=partition,
        dynamic_loop_iter_counts=counts,
    )


def test_loop_back_inputs_read_like_the_dict_threading_builds():
    rows = DeferredRowInputs([3, 1, 2])
    assert list(rows) == [3, 1, 2] and len(rows) == 3 and 1 in rows and 9 not in rows
    # before threading, a row has no inputs yet, as a prep with no edges gives
    assert rows[1] == {} and rows.get(9) is None
    rows.set_signals(["tok", "tok"])
    # the row read above is filled too, and one dict per row, each its own
    assert rows[1] == {"tok": []} and rows[2] == {"tok": []}
    assert rows[2] is rows[2] and rows[2] is not rows[3]
    rows[3]["tok"].append("x")
    assert rows[1]["tok"] == [] and rows[3]["tok"] == ["x"]
    assert rows.materialize() == {3: {"tok": ["x"]}, 1: {"tok": []}, 2: {"tok": []}}
    assert rows.pop(1) == {"tok": []} and list(rows) == [3, 2]


def test_concrete_rows_keep_their_inputs():
    rows = DeferredRowInputs([1, 2])
    fresh = {"tok": ["t"]}
    rows[2] = fresh
    rows.set_signals(["tok", "other"])
    assert rows[2] is fresh and fresh == {"tok": ["t"]}
    assert rows[1] == {"tok": [], "other": []}
    del rows[2]
    assert dict(rows) == {1: {"tok": [], "other": []}}


def test_snapshots_copy_and_advance_only_the_continuing_rows():
    live = [{"loop": 2, "outer": 1}, {"loop": 5}, {}]
    infos = [_info(i, c) for i, c in enumerate(live)]
    metas = loop_counter_snapshots(infos, "loop", n_next_iter=2)
    assert all(isinstance(m, InputMetadata) for m in metas)
    assert [dict(m.dynamic_loop_iter_counts) for m in metas] == [
        {"loop": 3, "outer": 1}, {"loop": 6}, {},
    ]
    assert all(dict(m.stream_chunks) == {} for m in metas)
    live[0]["loop"] = 40  # the shared counters move when N is routed
    assert metas[0].dynamic_loop_iter_counts["loop"] == 3
    # no loop to advance: plain copies
    metas = loop_counter_snapshots(infos, None, n_next_iter=3)
    assert [dict(m.dynamic_loop_iter_counts) for m in metas] == [
        {"loop": 40, "outer": 1}, {"loop": 5}, {},
    ]
    assert metas[1].dynamic_loop_iter_counts is not live[1]


def test_fwd_infos_follow_replacement_and_removal():
    state = RequestStateManager()
    a, b = _info(1, {}), _info(2, {})
    state.add_request(1, a)
    state.add_request(2, b)
    state.add_request(1, _info(1, {}, partition="q"))
    assert state.fwd_infos([2, 1], "p") == [b, a]
    a2 = _info(1, {})
    state.update_request_info(1, "p", current_fwd_info=a2)
    assert state.fwd_infos([1], "p")[0] is a2 is state.get_fwd_info(1, "p")
    # a partition the request never registered is not created by an update
    state.update_request_info(2, "q", current_fwd_info=_info(2, {}, partition="q"))
    with pytest.raises(KeyError):
        state.fwd_infos([2], "q")
    # handles are recycled: removal purges every partition's entry
    state.remove_request(1)
    for partition in ("p", "q"):
        with pytest.raises(KeyError):
            state.fwd_infos([1], partition)


def test_wrapped_infos_pair_each_info_with_its_metadata():
    infos = {1: _info(1, {"loop": 1}), 2: _info(2, {"loop": 4})}
    meta = {2: InputMetadata(dynamic_loop_iter_counts={"loop": 5})}
    batch = ExecutingBatch(
        node_name="n", per_request_info=infos, per_request_input_metadata=meta,
        step_context=StepContext(request_ids=(1, 2), graph_walk="decode", slot=0, capture=False),
    )
    wrapped = batch.per_request_info_wrapped
    assert list(wrapped) == [1, 2]
    assert wrapped[1].info is infos[1] and wrapped[1].metadata is EMPTY_INPUT_METADATA
    assert wrapped[2].metadata is meta[2] and wrapped[2].dynamic_loop_iter_counts == {"loop": 5}
    assert wrapped[2].max_tokens == 9 and wrapped[2].request_id == "r2"
