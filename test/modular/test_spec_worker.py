"""Speculative decoding through the real ``StepExecutor.postprocess``: every
path that reaches postprocess hands the step's verify result to the ledger
before it returns, so the take-back at a later admit finds it recorded.

A ``make_worker`` with the fields postprocess reads (AGENTS.md), over
the toy engine (`spec_toy.py`). The loop paths that decide *when* postprocess
runs relative to the next admits (yield-away, a TP follower's serial and
speculative pops) need the Rust graph runtime and are not built here; the TP
simulator checks the ordering they rely on (`tp_async_sim.explore_verify`).
"""
from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spec_toy import NODE, Driver, _batch, engine, every  # noqa: E402

from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.graph.runtime.base import PhasedPostprocess  # noqa: E402
from mstar.testing_utils.worker import make_worker  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.speculation import Speculator  # noqa: E402
from mstar.worker.step_executor import PendingBatch  # noqa: E402

WALK = "decode"


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


class _ReachedRuntimeError(Exception):
    pass


class _Runtime(SimpleNamespace, PhasedPostprocess):
    """Stubbed phase methods under the reference begin / resume."""


def _worker(eng, stopped=frozenset(), failed=frozenset()):
    def begin(request):
        raise _ReachedRuntimeError

    worker = make_worker(
        device=torch.device("cpu"),
        graph_runtime=_Runtime(
            cleanup_consumed_inputs=lambda node, rids, wgs: ([], []),
            clear_pending_loop_stops=lambda: None,
            pending_loop_stop_rids=lambda walks, loop: set(stopped),
            begin_postprocess=begin,
            supports_inline_emit=False,
        ),
        tensor_manager=SimpleNamespace(cleanup_collectable=lambda *args: None, get_tensor=None),
        engine_manager=SimpleNamespace(get_engine=lambda node: eng),
        request_state=SimpleNamespace(buffer_publish_info=lambda *args: None),
        verifying_nodes=eng.verifying_nodes(),
    )
    worker.lifecycle.failed_rids.update(failed)
    return worker


def _verify_step(eng, driver, rids):
    batch = _batch(WALK, {rid: torch.tensor([driver.out[rid][-1]]) for rid in rids}, driver._max)
    eng.prepare_inputs(batch)
    outputs = eng.exec_and_postprocess(batch)
    assert batch.admit_error is None and batch.verify_seq is not None
    pending = PendingBatch(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph=dict.fromkeys(rids, 0),
            output_signals=["new_token"],
        ),
        node_batch=batch, node_name=NODE, partition="p", graph_walk=WALK, future=Future(),
        speculative_new_iter=True, loop_name="decode_loop",
    )
    return batch, pending, outputs


def _setup():
    eng = engine(3, every(2))
    driver = Driver(eng)
    for rid, prompt in {1: [5, 9, 2], 2: [17]}.items():
        driver.add(rid, prompt, 30)
    return eng, driver


def _recorded(eng, seq) -> bool:
    entry = eng._drafts[NODE].ledger._entries.get(seq)
    return entry is None or entry.accepted is not None


def test_the_normal_path_cuts_and_records_before_the_stop_check():
    eng, driver = _setup()
    batch, pending, outputs = _verify_step(eng, driver, [1, 2])
    with pytest.raises(_ReachedRuntimeError):
        _worker(eng).executor.postprocess(pending, outputs)
    assert _recorded(eng, batch.verify_seq)
    # each row's output is its emitted run, not the whole block
    for rid in (1, 2):
        assert outputs.per_rid_outputs[rid]["new_token"][0].shape[-1] <= 4


def test_a_step_whose_rows_all_stopped_still_records():
    """A speculative step past every row's stop: no output leaves, but the
    result is the take-back's."""
    eng, driver = _setup()
    batch, pending, outputs = _verify_step(eng, driver, [1, 2])
    _worker(eng, stopped={1, 2}).executor.postprocess(pending, outputs)
    assert _recorded(eng, batch.verify_seq)


def test_a_step_whose_rows_all_failed_still_records():
    eng, driver = _setup()
    batch, pending, outputs = _verify_step(eng, driver, [1, 2])
    worker = _worker(eng, failed={1, 2})
    pending.speculative_new_iter = False
    worker.executor.postprocess(pending, outputs)
    assert _recorded(eng, batch.verify_seq)


def test_the_main_loop_error_path_voids_a_verify_step_it_never_recorded():
    """No later admit waits on the result of a step that committed, then failed its iteration."""
    eng, driver = _setup()
    batch, pending, _ = _verify_step(eng, driver, [1])
    pending.future = None
    assert not _recorded(eng, batch.verify_seq)
    worker = _worker(eng)
    worker._graph_runtime.set_in_flight = lambda *args: None
    worker.pipeline._fail_requests = lambda failed: None
    worker.pipeline._handle_error(RuntimeError("the step's iteration failed"), (pending,), None)
    assert _recorded(eng, batch.verify_seq)


def test_a_verify_step_that_raised_after_its_commit_is_voided():
    """A raise after commit reaches postprocess as `_step_result`'s empty outputs: its rows fail with that error,
    postprocess returns instead of raising into the main loop's handler, and no later admit waits on the step."""
    eng, driver = _setup()
    batch, pending, _ = _verify_step(eng, driver, [1, 2])
    assert batch.verify_seq in eng._drafts[NODE].ledger._entries
    pending.future.set_exception(RuntimeError("out of memory staging the step's rows"))
    worker = _worker(eng)
    failed = {}
    worker.pipeline._fail_requests = failed.update
    outputs = worker.pipeline._step_result(pending)
    assert outputs.host_rows is None and outputs.check_stop_buffers is None
    worker.pipeline._fail_step_rows(pending, outputs)
    worker.pipeline._finish_step(pending, outputs)
    assert set(failed) == {1, 2}
    assert batch.verify_seq not in eng._drafts[NODE].ledger._entries


def test_without_the_early_record_the_next_admits_raise():
    """Negative control: a step whose postprocess left early unrecorded
    makes the admit two steps on raise."""
    eng, driver = _setup()
    batch, pending, outputs = _verify_step(eng, driver, [1, 2])
    worker = _worker(eng, stopped={1, 2})
    worker.executor._record_verify = lambda batch_N, outputs: None
    worker.executor.postprocess(pending, outputs)
    assert not _recorded(eng, batch.verify_seq)
    eng._drafts[NODE].ledger.committed(None)
    with pytest.raises(RuntimeError, match="never recorded"):
        driver.step([1, 2])


def _tp_worker(node, leader: bool, synchronous: bool):
    worker = make_worker(
        async_sched=(True, None),
        synchronous_verify_nodes={node} if synchronous else set(),
        parallel_nodes={node},
        parallel_leader_nodes={node} if leader else set(),
        is_tp_follower=not leader,
    )
    worker.speculator._async_schedulable = lambda batch: True
    return worker


@pytest.mark.parametrize("synchronous", [True, False])
def test_a_synchronous_verify_node_is_serial_on_leader_and_follower_alike(synchronous):
    """The leader of a synchronous verify node never speculates, so its follower
    must not wait for a head or a marker the leader never sends."""
    pending = SimpleNamespace(node_name=NODE, batch=SimpleNamespace(node_name=NODE))
    leader = _tp_worker(NODE, leader=True, synchronous=synchronous)
    follower = _tp_worker(NODE, leader=False, synchronous=synchronous)
    assert leader.speculator.can_speculate(pending.batch) is (not synchronous)
    assert follower.speculator.is_tp_follow_pending(pending) is (not synchronous)


@pytest.mark.parametrize("verify", [True, False])
def test_a_verify_steps_uncut_outputs_never_reach_the_speculated_step(verify):
    """The next step of a verify walk reads each row's token off the
    loop-back slot; the uncut block it would otherwise thread is left out. A
    plain step's outputs are threaded as before (negative control)."""
    from mstar.model.submodule_base import BatchedModelOutput

    block = torch.arange(4)
    outputs = BatchedModelOutput(per_rid_outputs={1: {"ids": [block]}}, verify_step=verify)
    inputs = {1: {}}
    speculation = SimpleNamespace(
        node_batch=SimpleNamespace(per_request_input_tensors=inputs, request_ids=[1]),
        continuing_rids={1}, consumed_edges=[("ids", None)],
        scheduled_batch=SimpleNamespace(chunk_ranges={}),
    )
    Speculator.thread_outputs(Speculator.__new__(Speculator), speculation, outputs)
    assert inputs[1]["ids"] == ([] if verify else [block])
    assert speculation.continuing_rids == {1}


def test_a_row_the_verify_step_skipped_is_not_read_off_its_loop_back_slot():
    """A continuing row that step N skipped wrote no bonus
    token; the speculated step drops it rather than read a stale slot."""
    from mstar.model.submodule_base import BatchedModelOutput

    outputs = BatchedModelOutput(per_rid_outputs={1: {"ids": [torch.arange(4)]}}, verify_step=True,
                                 row_request_ids=(1,))
    inputs = {1: {}, 2: {}}
    dropped_from = []
    speculation = SimpleNamespace(
        node_batch=SimpleNamespace(
            per_request_input_tensors=inputs, request_ids=[1, 2], per_request_info={1: None, 2: None},
            per_request_input_metadata={}, final_stream_rids=set(), partition_done_rids=set(),
            drop_rids=lambda rids: None,
        ),
        continuing_rids={1, 2}, consumed_edges=[("ids", None)],
        scheduled_batch=SimpleNamespace(chunk_ranges={}, discard_rid=dropped_from.append),
    )
    Speculator.thread_outputs(Speculator.__new__(Speculator), speculation, outputs)
    assert inputs[1]["ids"] == [] and 2 not in inputs
    assert speculation.continuing_rids == {1} and speculation.dropped == {2} and dropped_from == [2]
