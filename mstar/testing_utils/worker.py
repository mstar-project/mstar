"""Workers and their components for tests, built with no process groups, sockets
or engines.

``make_worker`` skips ``Worker.__init__`` and builds what it would: the real
components, over the collaborators the test names (usually a ``SimpleNamespace``
of what the code under test calls), so one it forgot is an ``AttributeError``
rather than a stub answering for it. Components read their collaborators when
they are built, so pass them to ``make_worker`` rather than setting them after.
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
from mstar.worker.preemption import Preemption
from mstar.worker.rebuild import Rebuilds
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.speculation import Speculator
from mstar.worker.step_executor import StepExecutor
from mstar.worker.step_pipeline import StepPipeline
from mstar.worker.worker import Worker

# What ``Worker.__init__`` builds from the deployment. No default stands in for a
# collaborator, and a component left out is built, real, over the others.
_COLLABORATORS = ("communicator", "tensor_manager", "engine_manager", "request_state", "streams", "parallel_groups")
_COMPONENTS = (
    "lifecycle", "admission", "lockstep", "scheduler", "preemption", "ingest", "executor", "speculator", "pipeline",
)
# an engine with no paged KV cache, so the default `Preemption` moves nothing
_NO_PAGED_KV = SimpleNamespace(engine=SimpleNamespace(paged_kv_caches=list))
_LAYOUT = frozenset(WorkerLayout.__dataclass_fields__)


def make_services(**overrides) -> WorkerServices:
    """``WorkerServices`` with only what the test names; the rest are None."""
    fields = dict.fromkeys(WorkerServices.__slots__)
    fields.update(
        worker_id="w0", device=torch.device("cpu"), tensor_manager=SimpleNamespace(get_tensor=None),
        rebuilds=Rebuilds({}, {}),
    )
    fields.update(overrides)
    return WorkerServices(**fields)


def make_scheduler(
    engine_manager=None, *, parallel_leader_nodes=None, parallel_nodes=frozenset(),
    combined_walk_of=None, chunk_loops=None, side_keys=frozenset(), request_state=None,
    graph_runtime=None, tensor_manager=None, async_sched=(False, None), model_config=None,
    batch_builder_type=None, speculation_mixes_walks=None, rebuilds=None, walk_priorities=None, layout_fields=None,
    **scheduler_kwargs,
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
        rebuilds=rebuilds if rebuilds is not None else Rebuilds({}, {}),
    )
    if parallel_leader_nodes is None:
        parallel_leader_nodes = _AllNodes()
    parallel_nodes = frozenset(parallel_nodes)
    layout = WorkerLayout(
        parallel_nodes=parallel_nodes, parallel_leader_nodes=parallel_leader_nodes,
        is_tp_follower=bool(parallel_nodes - parallel_leader_nodes),
        combined_walk_of=combined_walk_of or {}, chunk_loops=chunk_loops or ChunkLoops(),
        side_keys=frozenset(side_keys), walk_priorities=walk_priorities or {},
        **(layout_fields or {}),
    )
    lifecycle = RequestLifecycle(layout=layout, graph_runtime=graph_runtime)
    admission = Admission(services=services, layout=layout, lifecycle=lifecycle)
    lockstep = Lockstep(services=services, layout=layout, admission=admission, async_sched=async_sched)
    return MicroScheduler(
        services=services, layout=layout, lifecycle=lifecycle, lockstep=lockstep,
        admission=admission, model_config=model_config, **scheduler_kwargs,
    )


def bound(scheduler: MicroScheduler, request_state=None, *, runtime=None) -> MicroScheduler:
    """``scheduler``, pointed at another fake along with what is under it, for
    a test that builds a fresh fake per call."""
    if runtime is not None:
        scheduler.runtime = runtime
        scheduler.lifecycle._runtime = runtime
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


def make_worker(
    *, graph_runtime=None, worker_id="w0", device=torch.device("cpu"), wakeup_event=None, metrics=None,
    profile_info=None, timer=None, launch_wait_s=0.0, max_consecutive_spec=1024, peek=True,
    enable_nvtx=False, enable_prof=False, async_sched=(False, None), rebuilds=None, **named,
) -> Worker:
    """A ``Worker`` with no threads, sockets or engines, leading and following
    nothing. ``named`` holds the collaborators, the components to replace and the
    ``WorkerLayout`` fields; any other name raises ``TypeError``, so a test cannot
    go on setting a field that has moved."""
    unknown = named.keys() - _LAYOUT - set(_COLLABORATORS) - set(_COMPONENTS) - {"partition_topology"}
    if unknown:
        raise TypeError(f"make_worker has no field {sorted(unknown)}")
    layout_fields = {name: named.pop(name) for name in named.keys() & _LAYOUT}
    for name in (
        "parallel_nodes", "parallel_leader_nodes", "streaming_edge_names", "side_keys",
        "verifying_nodes", "synchronous_verify_nodes",
    ):
        if name in layout_fields:
            layout_fields[name] = frozenset(layout_fields[name])
    layout = WorkerLayout(**layout_fields)

    worker = Worker.__new__(Worker)
    worker.worker_id, worker.device, worker.layout = worker_id, device, layout
    worker.enable_nvtx, worker.enable_prof = enable_nvtx, enable_prof
    worker.profile_info = profile_info if profile_info is not None else WorkerProfileInfo()
    worker.timer = timer if timer is not None else PhaseTimer(worker_id=worker_id, enable_nvtx=enable_nvtx, period=0)
    worker.wakeup_event = wakeup_event
    worker._metrics = metrics
    worker._graph_runtime = graph_runtime
    worker.partition_topology = named.pop("partition_topology", None)
    for name in _COLLABORATORS:
        setattr(worker, name, named.get(name))
    if worker.tensor_manager is None:
        worker.tensor_manager = SimpleNamespace(get_tensor=None)
    # no node declares a rebuild policy unless the test passes one
    worker._rebuilds = rebuilds if rebuilds is not None else Rebuilds({}, {})
    # no node writes a paged KV cache here, so no request keeps its ids
    worker._kv_writes = {}
    services = WorkerServices(
        worker_id=worker_id, device=device, graph_runtime=graph_runtime, rebuilds=worker._rebuilds,
        **{name: getattr(worker, name) for name in _COLLABORATORS},
    )

    # a scheduler the test built brings the components under it
    under_scheduler = named.get("scheduler")

    def component(name, build):
        given = named.get(name, getattr(under_scheduler, name, None))
        setattr(worker, name, given if given is not None else build())

    component("lifecycle", lambda: RequestLifecycle(layout=layout, graph_runtime=services.graph_runtime))
    component("admission", lambda: Admission(services=services, layout=layout, lifecycle=worker.lifecycle))
    component("lockstep", lambda: Lockstep(
        services=services, layout=layout, admission=worker.admission, async_sched=async_sched,
    ))
    component("scheduler", lambda: MicroScheduler(
        services=services, layout=layout, lifecycle=worker.lifecycle, lockstep=worker.lockstep,
        admission=worker.admission, model_config={},
    ))
    component("preemption", lambda: Preemption(
        _NO_PAGED_KV, worker._rebuilds, worker.scheduler, worker.request_state, worker.admission.log_throttled,
        worker._rid_str,
    ))
    component("ingest", lambda: InputIngest(
        services=services, layout=layout, lifecycle=worker.lifecycle, lockstep=worker.lockstep,
        wakeup_event=wakeup_event, metrics=metrics, enable_nvtx=enable_nvtx, set_rebuildable=worker._set_rebuildable,
    ))
    component("executor", lambda: StepExecutor(
        services=services, layout=layout, timer=worker.timer, lockstep=worker.lockstep,
        lifecycle=worker.lifecycle, preemption=worker.preemption, wakeup_event=wakeup_event, metrics=metrics,
        profile_info=worker.profile_info, enable_nvtx=enable_nvtx, enable_prof=enable_prof,
    ))
    component("speculator", lambda: Speculator(
        services=services, layout=layout, scheduler=worker.scheduler, lifecycle=worker.lifecycle,
        lockstep=worker.lockstep, admission=worker.admission, timer=worker.timer,
        max_consecutive=max_consecutive_spec, peek=peek, enable_nvtx=enable_nvtx,
    ))
    component("pipeline", lambda: StepPipeline(
        services=services, layout=layout, timer=worker.timer, lifecycle=worker.lifecycle,
        lockstep=worker.lockstep, admission=worker.admission, scheduler=worker.scheduler,
        executor=worker.executor, ingest=worker.ingest, speculator=worker.speculator, preemption=worker.preemption,
        metrics=metrics,
        profile_info=worker.profile_info, launch_wait_s=launch_wait_s, enable_nvtx=enable_nvtx,
    ))
    return worker


def preempting(worker: Worker) -> Preemption:
    """A real ``Preemption`` over the worker's engine manager, rebuilds, scheduler and request state, handed to the
    worker and to the executor and pipeline that call it. For a test that sets those up after ``make_worker``."""
    worker.preemption = worker.executor.preemption = worker.pipeline.preemption = Preemption(
        worker.engine_manager, worker._rebuilds, worker.scheduler, worker.request_state,
        worker.admission.log_throttled, worker._rid_str,
    )
    return worker.preemption


def deliver(worker: Worker, *messages) -> None:
    """Messages through ``InputIngest``, then what the main loop does at the end
    of its preamble: report recorded failures and run the teardowns now ready."""
    worker.ingest._dispatch(list(messages))
    worker.pipeline._report_failures()
    worker.pipeline._teardown(worker.pipeline.in_flight_rids)
