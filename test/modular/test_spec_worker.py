"""Speculative decoding through the real ``StepExecutor.postprocess``: every
path that reaches postprocess hands the step's verify result to the ledger
before it returns, so the take-back at a later admit finds it recorded.

A ``make_worker`` with the fields postprocess reads (AGENTS.md), over
the toy engine (`spec_toy.py`). The loop paths that decide *when* postprocess
runs relative to the next admits (yield-away, a TP follower's serial and
speculative pops) need the Rust graph runtime and are not built here; the TP
simulator checks the ordering they rely on (`tp_async_sim.explore_verify`).

A real `Preemption` over the toy's pool, with test_kv_host_index's host pool
copying on the CPU, runs main's preemption of a verifying row end to end.
"""
from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_kv_host_index as host_index  # noqa: E402
from spec_toy import NODE, STOP, Driver, _batch, engine, every, reference, stub_offload  # noqa: E402

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.resources import Residency  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.manager import Victims  # noqa: E402
from mstar.graph.runtime.base import PhasedPostprocess, PostprocessNeed, PostprocessStep  # noqa: E402
from mstar.testing_utils.worker import make_worker, preempting  # noqa: E402
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
            get_rid_string=str,
        ),
        tensor_manager=SimpleNamespace(cleanup_collectable=lambda *args: None, get_tensor=None),
        engine_manager=SimpleNamespace(get_engine=lambda node: eng, engine=eng),
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


@pytest.mark.parametrize("synchronous", [True, False])
def test_a_synchronous_verify_node_off_lockstep_builds_no_speculation(synchronous):
    """With ``enable_async_scheduling`` on, the runtime calls the walk async-schedulable, so on a node that isn't
    lockstep `can_speculate` alone keeps a synchronous verify step from being planned ahead of the last one's
    postprocess. Negative control: an asynchronous verifying node speculates."""
    from mstar.worker.speculation import SpeculationTarget

    tried = []
    chain_step = SimpleNamespace(node_batch=SimpleNamespace(node_name=NODE), tp_seq=-1, is_yield_away=False)
    worker = make_worker(
        graph_runtime=SimpleNamespace(is_async_schedulable=lambda node, walk: True),
        verifying_nodes={NODE}, synchronous_verify_nodes={NODE} if synchronous else set(),
        scheduler=SimpleNamespace(next_batch=lambda exclude_target: None, chunks=lambda node, walk: False),
        peek=False,
    )
    speculator = worker.speculator
    speculator._choose_spec_target = lambda batch: SpeculationTarget(
        SimpleNamespace(node_name=NODE, graph_walk=WALK), [1],
    )
    speculator._try_speculate_next = lambda pending, target: tried.append(target) or chain_step
    batch = ScheduledBatch(
        node_name=NODE, graph_walk=WALK, request_to_worker_graph={1: 0}, output_signals=["new_token"],
    )
    pending = SimpleNamespace(
        batch=batch, node_name=NODE, graph_walk=WALK, node_batch=SimpleNamespace(admit_error=None), tp_seq=-1,
    )
    assert speculator.can_speculate(batch) is not synchronous
    speculation, _ = speculator.build(pending)
    if synchronous:
        assert speculation is None and not tried
    else:
        assert speculation is chain_step and len(tried) == 1


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


def _chain_worker(eng):
    """`_worker`, with what a pipeline pass reads of the runtime around postprocess: postprocess ends after
    the record, a refused step's rows go back on the runtime's queues, and the ready scan finds nothing else."""
    worker = _worker(eng)
    worker._graph_runtime.begin_postprocess = lambda request: PostprocessStep(PostprocessNeed.DONE)
    worker._graph_runtime.set_in_flight = lambda *args: None
    worker._graph_runtime.update_dynamic_loop_iters = lambda *args: None
    worker._graph_runtime.push_back_node = lambda *args: None
    worker._graph_runtime.get_ready_nodes = lambda *args, **kwargs: []
    worker.tensor_manager.drop_host_copies = lambda: None
    worker.request_state.flush_publish_info = lambda *args: None
    return worker


def _chain_step(worker, eng, driver, rids, postprocess=True):
    """One verify step through the pipeline's own pass (`_finish_chain_step`): postprocessed when it
    admitted, its failure handled when it was refused."""
    from mstar.model.submodule_base import BatchedModelOutput
    from mstar.utils.profiler import PhaseLaps

    batch = _batch(WALK, {rid: torch.tensor([driver.out[rid][-1]]) for rid in rids}, driver._max)
    eng.prepare_inputs(batch)
    outputs = eng.exec_and_postprocess(batch)
    pending = PendingBatch(
        batch=ScheduledBatch(
            node_name=NODE, graph_walk=WALK, request_to_worker_graph=dict.fromkeys(rids, 0),
            output_signals=["new_token"],
        ),
        node_batch=batch, node_name=NODE, partition="p", graph_walk=WALK, future=Future(),
        speculative_new_iter=True, loop_name="decode_loop",
    )
    pending.future.set_result(outputs if batch.admit_error is None else BatchedModelOutput())
    if batch.admit_error is None and not postprocess:
        return batch
    worker.pipeline._set_pending(pending)
    worker.pipeline._finish_chain_step(PhaseLaps())
    if batch.admit_error is None:
        for rid in rids:
            driver.out[rid] += outputs.per_rid_outputs[rid]["new_token"][0].reshape(-1).tolist()
    return batch


@pytest.mark.parametrize("postprocessed", [True, False], ids=["pipeline", "without_postprocess"])
def test_a_refused_verify_step_finds_every_result_recorded_before_preemption(monkeypatch, postprocessed):
    """Rank 0 picks victims in the pass of the step whose admit was refused, after
    the pass that postprocessed the step before it, so no row holds an unrecorded
    verify result there, which it checks first (`_refuse_unrecorded_victims`).
    Negative control: a verify step on row 3 that commits and never reaches
    postprocess trips the check."""
    from mstar.engine.resources.kv.manager import KVManager

    # KV's host copy needs CUDA: an offload that moves nothing, as test_spec_engine's stub
    monkeypatch.setattr(KVManager, "supports_eviction", property(lambda self: True))
    monkeypatch.setattr(KVManager, "offload", lambda self, rid: 1)
    eng = engine(3, every(2), max_num_pages=24)
    driver = Driver(eng)
    for rid, prompt in {1: [5, 9, 2], 2: [17], 3: list(range(1, 40))}.items():
        driver.add(rid, prompt, 200)
    worker = _chain_worker(eng)
    worker.lifecycle.arrivals.update({1: 0, 2: 1, 3: 2})
    seen = []
    check = worker.pipeline._refuse_unrecorded_victims

    def spy():
        seen.append({rid: eng.verify_unrecorded(NODE, rid) for rid in (1, 2, 3)})
        return check()

    worker.pipeline._refuse_unrecorded_victims = spy
    _chain_step(worker, eng, driver, [3], postprocess=postprocessed)
    if not postprocessed:
        with pytest.raises(RuntimeError, match=r"unrecorded verify results: \[3\]"):
            for _ in range(50):
                _chain_step(worker, eng, driver, [1, 2])
        return
    for _ in range(50):
        if _chain_step(worker, eng, driver, [1, 2]).admit_error is not None:
            break
    assert seen == [{1: False, 2: False, 3: False}], "the refusal never reached the check, or found a row unrecorded"
    # the victim is the row outside the refused step
    assert 3 in worker.admission.held_until


def _victims(move: str, rid: int) -> Victims:
    return Victims(**{name: [rid] if name == move else [] for name in ("to_host", "to_drop", "to_ask")})


def _debug_move(monkeypatch, eng, victims: Victims) -> None:
    """`Preemption._move` of ``victims`` for row 1 over the toy engine's paged KV cache, under
    ``MSTAR_KV_DEBUG_ASSERTS``."""
    preemption = preempting(_worker(eng))
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    preemption._move(preemption._caches["kv"], preemption.counts[NODE], victims, 1)


def test_the_debug_victim_check_passes_a_returning_row_asked_again(monkeypatch):
    """A row moved to the host and admitted back holds an allotment and no page until it reloads, so the pool
    can ask it again (`KVManager._choose`). Its verify results stay held for its first admit back."""
    stub_offload(monkeypatch)
    eng, driver = _setup()
    driver.step([1, 2])
    driver.step([1])
    assert eng.offload_request(NODE, 2) > 0
    assert eng.verify_pending(NODE, 2) and eng.verify_returning(NODE, 2)
    _debug_move(monkeypatch, eng, _victims("to_ask", 2))
    assert eng.verify_pending(NODE, 2) and eng.verify_returning(NODE, 2)


@pytest.mark.parametrize("move", ["to_host", "to_drop", "to_ask"])
def test_the_debug_victim_check_refuses_a_resident_row_with_an_unrecorded_step(monkeypatch, move):
    stub_offload(monkeypatch)
    eng, driver = _setup()
    batch, _, _ = _verify_step(eng, driver, [2])
    assert not _recorded(eng, batch.verify_seq) and not eng.verify_returning(NODE, 2)
    with pytest.raises(RuntimeError, match=r"preemption on LLM: .*\[2\]"):
        _debug_move(monkeypatch, eng, _victims(move, 2))


@pytest.mark.parametrize("move", ["to_host", "to_drop", "to_ask"])
def test_the_debug_victim_check_passes_resident_victims_with_nothing_unrecorded(monkeypatch, move):
    """Negative control: moved or dropped, a row whose results are recorded and not yet applied; asked again,
    one that holds none."""
    stub_offload(monkeypatch)
    eng, driver = _setup()
    driver.step([1, 2])
    driver.add(3, [4, 4], 30)
    rid = 3 if move == "to_ask" else 2
    assert not eng.verify_unrecorded(NODE, rid) and eng.verify_pending(NODE, rid) is (rid == 2)
    _debug_move(monkeypatch, eng, _victims(move, rid))


class _HostPool(host_index._HostPool):
    """test_kv_host_index's host pool, with its copies made on the CPU, so a row the toy moved to the host and
    back reads the KV it wrote."""

    def __init__(self, pages: int, device_cache: torch.Tensor):
        super().__init__(pages)
        self.cpu = torch.zeros(device_cache.shape[0], pages, *device_cache.shape[2:], dtype=device_cache.dtype)

    def copy_out(self, gpu_kv_cache, copies):
        super().copy_out(gpu_kv_cache, copies)
        # the toy's pool has the one storage
        self.cpu[:, copies.cpu] = gpu_kv_cache.tensor[:, copies.gpu]

    def copy_in(self, gpu_kv_cache, copies):
        super().copy_in(gpu_kv_cache, copies)
        gpu_kv_cache.tensor[:, copies.gpu] = self.cpu[:, copies.cpu]


def _hosted(max_num_pages: int, host_pages: int = 64):
    """The toy engine on a pool with a host pool, and a pipeline whose `Preemption` moves rows over it."""
    eng = engine(3, every(2), max_num_pages=max_num_pages)
    kv = eng._resources["kv"]
    kv._cpu_pool = _HostPool(host_pages, kv.kv_cache.tensor)
    worker = _chain_worker(eng)
    preempting(worker)
    return eng, Driver(eng), worker


def _back(eng, rid: int) -> None:
    """A readiness scan's admit and reload of ``rid``, back from the host."""
    eng.admit_waiting(lambda other, node: None)
    info = CurrentForwardPassInfo(str(rid), WALK, 0, 0, 30)
    assert eng.check_ready(NODE, rid, info).ready



_PROMPTS = {1: list(range(1, 21)), 2: [5, 9, 2, 7, 1, 3, 3, 8]}


def _run_short(worker, eng, driver, rids) -> None:
    """Verify steps of ``rids`` through the pipeline until one's admit is refused and handled."""
    for _ in range(60):
        if _chain_step(worker, eng, driver, rids).admit_error is not None:
            return
    raise AssertionError(f"{rids} never ran short")


def _decode_back(worker, eng, driver, rid: int, max_tokens: int) -> None:
    """``rid``, back from the host, decodes to its limit. Every result it held is taken back at its first admit."""
    ledger = eng._drafts[NODE].ledger
    held = set(ledger._steps_of.get(rid, ()))
    _back(eng, rid)
    _chain_step(worker, eng, driver, [rid])
    assert not eng.verify_returning(NODE, rid) and held.isdisjoint(ledger._steps_of.get(rid, ()))
    for _ in range(max_tokens):
        if len(driver.out[rid]) >= max_tokens or driver.out[rid][-1] == STOP:
            return
        assert _chain_step(worker, eng, driver, [rid]).admit_error is None
    raise AssertionError(f"{rid} never reached its limit")


def test_a_row_that_runs_short_moves_a_newer_verifying_row_to_the_host():
    """`KVManager.preempt` chooses the newer verifying row, `Preemption._move` offloads it and the engine marks it
    returning, so the results it holds are due at its first admit back, after its reload."""
    eng, driver, worker = _hosted(max_num_pages=12)
    for arrival, (rid, prompt) in enumerate(_PROMPTS.items()):
        driver.add(rid, prompt, 200 if rid == 1 else 30, arrival=arrival)
    _chain_step(worker, eng, driver, [1, 2])
    _run_short(worker, eng, driver, [1])
    assert eng.residency(NODE, 2) is Residency.ON_HOST
    assert eng.verify_returning(NODE, 2) and eng.verify_pending(NODE, 2)
    # the victim came from the pool, not from the fallback eviction
    assert worker.preemption.counts[NODE].victims == 1 and not worker.admission.held_until
    # the row that ran short runs on in the room the move freed
    assert _chain_step(worker, eng, driver, [1]).admit_error is None
    assert driver.out[1] == reference(_PROMPTS[1], 200)[:len(driver.out[1])]
    eng.remove_request(1)
    _decode_back(worker, eng, driver, 2, 30)
    assert driver.out[2] == reference(_PROMPTS[2], 30)


def test_a_row_that_runs_short_with_nothing_newer_to_move_moves_itself_to_the_host():
    """The branch where no victim covers the shortage and the host takes the requester whole
    (``Victims([rid], [], [])``): it is marked returning as any victim is. In synchronous mode it holds no result
    by then, since the refused admit took back everything it held before admitting."""
    eng, driver, worker = _hosted(max_num_pages=12)
    for arrival, (rid, prompt) in enumerate(_PROMPTS.items()):
        driver.add(rid, prompt, 30 if rid == 1 else 20, arrival=arrival)
    _chain_step(worker, eng, driver, [1, 2])
    _run_short(worker, eng, driver, [2])
    assert eng.residency(NODE, 2) is Residency.ON_HOST and eng.residency(NODE, 1) is Residency.RESIDENT
    assert eng.verify_returning(NODE, 2)
    assert worker.preemption.counts[NODE].victims == 1 and not worker.admission.held_until
    eng.remove_request(1)
    _decode_back(worker, eng, driver, 2, 20)
    assert driver.out[2] == reference(_PROMPTS[2], 20)



def test_a_returning_row_the_pool_asks_again_passes_the_debug_victim_check(monkeypatch):
    """Moved to the host, then admitted back with an allotment and no page before it reloads, a verifying row is
    one `KVManager._choose` asks again when an older row runs short. Under ``MSTAR_KV_DEBUG_ASSERTS`` the move
    goes through, and the row's results stay held for its first admit back."""
    from mstar.engine.resources.kv.manager import KVManager

    eng, driver, worker = _hosted(max_num_pages=12)
    chosen = []
    preempt = KVManager.preempt
    monkeypatch.setattr(KVManager, "preempt", lambda self, *args: chosen.append(preempt(self, *args)) or chosen[-1])
    prompts = {1: list(range(1, 21)), 2: [5, 9, 2, 7], 3: _PROMPTS[2]}
    for arrival, (rid, prompt) in enumerate(prompts.items()):
        driver.add(rid, prompt, 20 if rid == 3 else 200, arrival=arrival)
    _chain_step(worker, eng, driver, [1, 2, 3])
    _run_short(worker, eng, driver, [1, 2])
    assert chosen == [Victims([3], [], [])] and eng.verify_returning(NODE, 3)
    eng.remove_request(1)
    eng.admit_waiting(lambda other, node: None)
    assert eng.residency(NODE, 3) is Residency.ON_HOST and eng.verify_pending(NODE, 3)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    _run_short(worker, eng, driver, [2])
    assert chosen[1:] == [Victims([], [], [3])]
    assert eng.verify_returning(NODE, 3) and eng.verify_pending(NODE, 3)
    eng.remove_request(2)
    _decode_back(worker, eng, driver, 3, 20)
    assert driver.out[3] == reference(_PROMPTS[2], 20)
