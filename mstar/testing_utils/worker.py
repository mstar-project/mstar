"""Workers and their components for tests, built with no process groups, sockets
or engines.

``make_worker`` skips ``Worker.__init__`` and starts from what it would leave
behind: the step-to-step state (``Worker._init_step_state``, the method
``__init__`` itself uses), a timer that is off, and a worker that leads nothing.
A test supplies the collaborators the code under test reaches (usually a
``SimpleNamespace`` of what it calls), so one it forgot is an ``AttributeError``
rather than a stub answering for it.
"""
from types import SimpleNamespace

import torch

from mstar.profile.worker import WorkerProfileInfo
from mstar.worker.admission import Admission
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.input_ingest import InputIngest
from mstar.worker.lockstep import Lockstep
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.phase_timer import PhaseTimer
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.worker import Worker

# Built by ``__init__`` from the deployment; no default stands in for them.
_COLLABORATORS = frozenset({
    "communicator", "wakeup_event", "tensor_manager", "_graph_runtime", "engine_manager",
    "request_state", "scheduler", "admission", "lockstep", "lifecycle", "ingest", "layout", "streams",
    "parallel_groups", "partition_topology", "_my_consumer_connections", "_all_wg_nodes",
})


def make_services(**overrides) -> WorkerServices:
    """``WorkerServices`` with only what the test names; the rest are None."""
    fields = dict.fromkeys(WorkerServices.__slots__)
    fields.update(worker_id="w0", device=torch.device("cpu"), tensor_manager=SimpleNamespace(get_tensor=None))
    fields.update(overrides)
    return WorkerServices(**fields)


def make_scheduler(
    engine_manager=None, *, parallel_leader_nodes=None, parallel_nodes=frozenset(),
    combined_walk_of=None, chunk_loops=None, side_keys=frozenset(), request_state=None,
    graph_runtime=None, tensor_manager=None, async_sched=(False, None), model_config=None,
    batch_builder_type=None, speculation_mixes_walks=None, **scheduler_kwargs,
) -> MicroScheduler:
    """A real ``MicroScheduler``, with a real ``Lockstep`` and ``Admission``
    under it (``.lockstep``, ``.admission``). ``parallel_leader_nodes`` defaults
    to leading every node."""
    model_config = dict(model_config or {})
    if batch_builder_type is not None:
        model_config["batch_builder"] = batch_builder_type.value
    if speculation_mixes_walks is not None:
        model_config["speculation_mixes_walks"] = speculation_mixes_walks
    services = make_services(
        engine_manager=engine_manager, request_state=request_state, graph_runtime=graph_runtime,
        tensor_manager=tensor_manager or SimpleNamespace(get_tensor=None),
    )
    if parallel_leader_nodes is None:
        parallel_leader_nodes = _AllNodes()
    parallel_nodes = frozenset(parallel_nodes)
    layout = WorkerLayout(
        parallel_nodes=parallel_nodes, parallel_leader_nodes=parallel_leader_nodes,
        is_tp_follower=bool(parallel_nodes - parallel_leader_nodes),
        combined_walk_of=combined_walk_of or {}, chunk_loops=chunk_loops or ChunkLoops(),
        side_keys=frozenset(side_keys),
    )
    admission = Admission(services=services, layout=layout)
    lockstep = Lockstep(services=services, layout=layout, admission=admission, async_sched=async_sched)
    return MicroScheduler(
        services=services, layout=layout, lifecycle=RequestLifecycle(layout=layout), lockstep=lockstep,
        admission=admission, model_config=model_config, **scheduler_kwargs,
    )


def bound(scheduler: MicroScheduler, request_state=None, *, runtime=None) -> MicroScheduler:
    """``scheduler``, pointed at another fake along with what is under it, for
    a test that builds a fresh fake per call."""
    if runtime is not None:
        scheduler.runtime = runtime
        scheduler.lockstep._runtime = runtime
        scheduler.admission._runtime = runtime
    if request_state is not None:
        scheduler.request_state = request_state
        scheduler.lockstep._request_state = request_state
        scheduler.admission._request_state = request_state
    return scheduler


class _AllNodes(frozenset):
    """A leader set naming every node, for a scheduler test with no deployment."""

    def __contains__(self, node) -> bool:
        return True

    def __rsub__(self, other):
        return frozenset()

    def __rand__(self, other):
        return frozenset(other)


def make_worker(**overrides) -> Worker:
    """A ``Worker`` with no threads, sockets or engines. ``overrides`` replace
    fields and methods by name; a name ``Worker`` does not have raises
    ``TypeError``, so a test cannot go on setting a field that has moved."""
    worker = Worker.__new__(Worker)
    worker._init_step_state()
    worker.worker_id = "w0"
    worker.device = torch.device("cpu")
    worker.enable_nvtx = False
    worker.enable_prof = False
    worker.profile_info = WorkerProfileInfo()
    worker.timer = PhaseTimer(worker_id="w0", period=0)
    worker.chunk_loops = ChunkLoops()
    worker._launch_wait_s = 0.0
    worker._max_consecutive_spec = 1024
    # a worker that leads and follows nothing
    worker.parallel_nodes = set()
    worker.parallel_leader_nodes = set()
    worker.is_tp_follower = False
    # no streams, no loops, no declared stop policies
    worker._streaming_edge_names = set()
    worker._innermost_loop_by_node = {}
    worker._stop_token_signal = {}

    async_sched = (overrides.pop("tp_async_sched", False), overrides.pop("tp_async_nodes", None))
    for name, value in overrides.items():
        if name not in _COLLABORATORS and not hasattr(worker, name):
            raise TypeError(f"Worker has no field {name!r}")
        setattr(worker, name, value)
    if "enable_nvtx" in overrides:
        worker.timer.enable_nvtx = overrides["enable_nvtx"]
    _build_components(worker, async_sched)
    return worker


def deliver(worker: Worker, *messages) -> None:
    """Messages through ``InputIngest``, then what the main loop does at the end
    of its preamble: report recorded failures and run the teardowns now ready."""
    worker.ingest._dispatch(list(messages))
    worker._report_failures()
    worker._teardown(worker._in_flight_rids)


def _build_components(worker: Worker, async_sched) -> None:
    """The components the test did not supply, real, over the worker's collaborators."""
    services = WorkerServices(
        worker_id=worker.worker_id, device=worker.device,
        graph_runtime=getattr(worker, "_graph_runtime", None),
        engine_manager=getattr(worker, "engine_manager", None),
        tensor_manager=getattr(worker, "tensor_manager", None),
        request_state=getattr(worker, "request_state", None),
        communicator=getattr(worker, "communicator", None),
        parallel_groups=getattr(worker, "parallel_groups", None),
        streams=getattr(worker, "streams", None),
    )
    parallel, leaders = frozenset(worker.parallel_nodes), frozenset(worker.parallel_leader_nodes)
    layout = WorkerLayout(
        parallel_nodes=parallel, parallel_leader_nodes=leaders, is_tp_follower=worker.is_tp_follower,
        chunk_loops=worker.chunk_loops, streaming_edge_names=frozenset(worker._streaming_edge_names),
        all_wg_nodes=getattr(worker, "_all_wg_nodes", None) or {},
    )
    if "lifecycle" not in worker.__dict__:
        worker.lifecycle = getattr(getattr(worker, "scheduler", None), "lifecycle", None) or RequestLifecycle(
            layout=layout,
        )
    if "admission" not in worker.__dict__:
        worker.admission = Admission(services=services, layout=layout)
    if "lockstep" not in worker.__dict__:
        worker.lockstep = Lockstep(
            services=services, layout=layout, admission=worker.admission, async_sched=async_sched,
        )
    if "ingest" not in worker.__dict__:
        worker.ingest = InputIngest(
            services=services, layout=layout, lifecycle=worker.lifecycle, lockstep=worker.lockstep,
            wakeup_event=getattr(worker, "wakeup_event", None), metrics=worker._metrics,
            enable_nvtx=worker.enable_nvtx,
        )
