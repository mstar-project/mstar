"""The Python pieces of the columnar speculation: the lazily built loop-back
inputs, the loop-counter snapshots, the request state's batch lookup of
forward-pass infos, and the engine's wrapped infos. The whole path against the
per-row one, on both runtimes, is in ``test/rust/test_spec_rows.py``."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.worker.node_manager_utils import RequestStateManager  # noqa: E402
from mstar.worker.spec_rows import DeferredRowInputs  # noqa: E402


def _info(rid: int, counts: dict, partition: str = "p") -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id=f"r{rid}", graph_walk="decode", fwd_index=0, random_seed=0,
        max_tokens=9, rid_handle=rid, partition_name=partition,
        dynamic_loop_iter_counts=counts,
    )


def test_concrete_rows_keep_their_inputs():
    rows = DeferredRowInputs([1, 2])
    fresh = {"tok": ["t"]}
    rows[2] = fresh
    rows.set_signals(["tok", "other"])
    assert rows[2] is fresh and fresh == {"tok": ["t"]}
    assert rows[1] == {"tok": [], "other": []}
    del rows[2]
    assert dict(rows) == {1: {"tok": [], "other": []}}


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
