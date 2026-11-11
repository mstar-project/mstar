import gc
import logging
import math
import os
import sys
import threading
from dataclasses import replace
from functools import cached_property, partial

import torch

from mstar.communication.codec import WireCodec
from mstar.communication.communicator import CommProtocol, make_communicator
from mstar.communication.event import EventWakeup
from mstar.communication.tensors import (
    LocalTransferEngine,
    create_tensor_communication_manager,
)
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.engine import Engine
from mstar.engine.resources import ResourceReqConfig
from mstar.engine.resources.kv.config import KVReqConfig
from mstar.engine.resources.kv.transfer import (
    TransferEngineInfo,
    make_deployment_kv_shm_dir,
)
from mstar.graph.runtime.base import (
    GraphRuntime,
)
from mstar.graph.runtime.utils import require_rust_extension
from mstar.model.base import Model, WorkerGraph
from mstar.profile.worker import WorkerProfileInfo
from mstar.utils import cpus, knobs
from mstar.utils.device import DeviceCapability
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    SetupDone,
)
from mstar.utils.profiler import (
    nvtx_enabled,
)
from mstar.worker.admission import Admission
from mstar.worker.engine_manager import EngineManager
from mstar.worker.host_pool import HostPool
from mstar.worker.input_ingest import InputIngest
from mstar.worker.lockstep import Lockstep, tp_async_sched_setting
from mstar.worker.micro_scheduler import (
    MicroScheduler,
)
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.phase_timer import PhaseTimer
from mstar.worker.preemption import Preemption
from mstar.worker.rebuild import (
    DEFAULT_REBUILD_HOST_MB,
    DEFAULT_REBUILD_MEDIA_HOST_MB,
    Rebuilds,
    node_writes,
    policies_run_here,
)
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.speculation import (
    Speculator,
)
from mstar.worker.step_executor import StepExecutor
from mstar.worker.step_pipeline import StepPipeline
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker_metrics import WorkerMetrics

logger = logging.getLogger(__name__)


def _make_graph_runtime(
    communicator,
    tensor_manager,
    **kwargs,
) -> GraphRuntime:
    """The Rust graph runtime.

    It requires the Rust communicator and the Rust bookkeeper: it sends
    WORKER_GRAPHS_DONE and INPUT_SIGNALS itself on an Arc to the same
    transport, and settles refcounts on a share of the same bookkeeper, rather
    than hopping back into Python for either. Anything else has no such object
    to share, so this raises on it.
    """
    require_rust_extension()

    from mstar.communication.rust_communicator import RustZMQCommunicator
    from mstar.graph.runtime.rust import RustGraphRuntime

    if not isinstance(communicator, RustZMQCommunicator):
        raise ValueError(
            "The graph runtime needs the Rust communicator, but this worker "
            f"built a {type(communicator).__name__}."
        )
    if not issubclass(communicator.codec, WireCodec):
        raise ValueError(
            "The graph runtime needs the msgpack codec, but this worker "
            f"built a {communicator.codec.__name__}."
        )
    bookkeeping = tensor_manager.tensor_store.bookkeeping
    if not hasattr(bookkeeping, "_rust"):
        raise ValueError(
            "The graph runtime needs the Rust TensorBookkeeping: it "
            "holds a share of it, so a Python one cannot be handed over. "
            f"Got {type(bookkeeping).__name__}."
        )
    return RustGraphRuntime(
        bookkeeping=bookkeeping,
        communicator=communicator,
        **kwargs,
    )


def _make_rebuilds(
    model: Model | None, model_config: dict | None, my_walks: set[tuple[str, str]], lockstep_nodes: set[str],
    device: torch.device, engine: Engine, followed: frozenset[str] = frozenset(),
) -> Rebuilds:
    """What a dropped request is rebuilt from, on each node whose model declares how and whose rebuild this worker
    runs: a disaggregated decode node can't run the prefill walk a rebuild needs.

    A worker that follows one of those nodes sets aside twice ``rebuild_host_mb``. Beside the ids of every request
    rank 0 keeps ids for, it holds those of requests rank 0 has torn down until it tears them down too."""
    graphs = model.get_graph_walk_graphs() if model is not None else {}
    policies = {} if model is None else policies_run_here(model.rebuild_policies(), graphs, my_walks, lockstep_nodes)
    copies = any(names for node, policy in policies.items() for names in policy.check(node, graphs).values())
    host_mb = _serving_mb(
        model_config, "rebuild_host_mb", DEFAULT_REBUILD_MEDIA_HOST_MB if copies else DEFAULT_REBUILD_HOST_MB,
    )
    if not host_mb:
        policies = {}
    nbytes = host_mb << (21 if followed & policies.keys() else 20)
    return Rebuilds(
        policies, graphs, HostPool(nbytes, device) if policies else None, engine.host_allowance,
        {node: dtype for node in policies if (dtype := engine.node_dtype(node)) is not None}, followed,
    )


def _serving_mb(model_config: dict | None, key: str, default: int) -> int:
    value = (model_config or {}).get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key}: {value!r} is not a non-negative int")
    return value


def _wait_ms(raw: str) -> float:
    """Milliseconds for ``Event.wait``, which overflows past TIMEOUT_MAX."""
    value = float(raw)
    if not math.isfinite(value) or value / 1000.0 > threading.TIMEOUT_MAX:
        raise ValueError(f"{raw!r} is not a usable wait")
    return value


def _env_number(name: str, default, parse, minimum):
    """A numeric tuning knob from the environment.

    A malformed value falls back to *default* and one below *minimum* is
    raised to it, each with a warning, as ``MSTAR_RECOMPILE_LIMIT`` already
    does. The worker reads these when it is built: raising from ``run()``
    would come after weight load, warmup and capture, and after the conductor
    has counted the worker ready.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = parse(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s=%r is below %s; using %s", name, raw, minimum, minimum)
        return minimum
    return value


class Worker:
    """One per GPU: builds the worker's components, and runs ``StepPipeline``
    once warmup is done. What each component owns is in its own module."""

    # Metrics are off unless __init__ gets a status interval
    _metrics: WorkerMetrics | None = None

    @cached_property
    def device_capability(self) -> DeviceCapability:
        return DeviceCapability.of(self.device)

    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        model: Model,
        my_worker_graphs: list[WorkerGraph],
        model_config: dict,
        all_worker_graph_ids_to_graph_walks: dict[int, set[str]],
        all_worker_graph_ids_to_nodes: dict[int, set[str]],
        all_worker_graph_ids_to_dyn_loops: dict[int, set[str]],
        sharding_config: ShardingConfig,
        parallel_groups: WorkerParallelGroups,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        enable_nvtx: bool = False,
        enable_prof: bool = False,
        tcp_transfer_device="",
        dist_init_method=None,
        status_interval_s: float | None = None,
        local_worker_devices: list[str] | None = None,
        all_worker_graph_ids_to_instances: dict[int, list[list[str]]] | None = None,
    ):
        # before anything below builds a resource or a submodule that reads one
        knobs.configure((model_config or {}).get("knobs"))
        self.worker_id = worker_id
        self.device = device
        self.enable_nvtx = nvtx_enabled(enable_nvtx, device)

        # How long the submitter holds off the GIL waiting for the GPU thread
        # to reach the launch. submit_spec often sits on this cap, but raising
        # it to 8ms did not help; tunable for another look.
        launch_wait_s = _env_number(
            "MSTAR_LAUNCH_WAIT_MS", 5.0, _wait_ms, 0.0,
        ) / 1000.0
        max_consecutive_spec = _env_number(
            "MSTAR_MAX_CONSECUTIVE_SPEC_STEPS", 1024, int, 0,
        )
        if max_consecutive_spec == 0:
            logger.warning(
                "Worker %s: MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=0 turns speculation "
                "off: every step yields before the next one is built",
                worker_id,
            )
        self.timer = PhaseTimer(worker_id=worker_id, enable_nvtx=self.enable_nvtx)

        self.enable_prof = enable_prof
        self.profile_info = WorkerProfileInfo()

        if self.device_capability.is_accelerator and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)
        step_cpus, process_cpus = self._plan_step_cpus(worker_ids, local_worker_devices)

        # ``dist_init_method`` is normally provided by the conductor — it
        # picks a free TCP port at startup so multiple ``mstar`` runs on
        # the same host don't collide. The ``tcp://{hostname}:29500``
        # fallback is for standalone Worker construction (e.g. tests);
        # production paths always pass a value.
        if dist_init_method is None:
            dist_init_method = f"tcp://{hostname}:29500"

        self.parallel_groups = parallel_groups
        self.parallel_groups.init_dist(
            init_method=dist_init_method,
            device=self.device,
        )

        layout = WorkerLayout.derive(
            model, my_worker_graphs, all_worker_graph_ids_to_nodes,
            sharding_config, parallel_groups, model_config,
        )
        self.layout = layout
        node_to_partition = dict(layout.node_to_partition)
        node_names = set(layout.node_names)

        self.communicator = make_communicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        self.wakeup_event = EventWakeup()
        self.communicator.register_event_for_poll(self.wakeup_event)

        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id=worker_id,
            hostname=hostname,
            device=self.device,
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=enable_prof
        )
        kv_shm_dir_factory = None
        if (
            isinstance(
                self.tensor_manager.transfer_engine, LocalTransferEngine,
            )
            and self.device.type != "cuda"
        ):
            kv_shm_dir_factory = partial(
                make_deployment_kv_shm_dir,
                self.tensor_manager._shm_namespace,
                socket_path_prefix=socket_path_prefix,
                dist_init_method=dist_init_method,
            )

        topology = model.get_partition_topology() if model else None
        # The graph runtime owns the per-request queues and the graph state.
        self._graph_runtime = _make_graph_runtime(
            my_worker_id=self.worker_id,
            my_worker_graphs=my_worker_graphs,
            all_wg_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
            all_wg_ids_to_dyn_loops=all_worker_graph_ids_to_dyn_loops,
            all_wg_ids_to_nodes=all_worker_graph_ids_to_nodes,
            node_to_partition=node_to_partition,
            sharding_config=sharding_config,
            tensor_manager=self.tensor_manager,
            communicator=self.communicator,
            combined_walk_of=dict(layout.combined_walk_of),
            walk_driven_signals=frozenset(
                conn.edge_name for conn in topology.connections
                if conn.consumer_walk is not None
            ) if topology is not None else frozenset(),
        )

        self.engine_manager = EngineManager.build(
            node_names,
            device=device,
            model_config=model_config,
            parallel_groups=self.parallel_groups,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=self.tensor_manager.my_session_id,
                transfer_engine=self.tensor_manager.transfer_engine,
                shm_dir_factory=kv_shm_dir_factory,
            ),
            graph_runtime=self._graph_runtime,
            model=model,
            enable_nvtx=self.enable_nvtx,
            enable_prof=self.enable_prof,
            device_capability=self.device_capability,
        )
        self._graph_runtime.set_inline_streams(self.engine_manager.inline_streams.keys)
        self._never_publish_unshared(all_worker_graph_ids_to_instances)
        if layout.step_repeats is not None:
            for node, walks in layout.step_repeats.asked_walks().items():
                self.engine_manager.get_engine(node).enable_repeats(node, walks)

        self.request_state = RequestStateManager(
            node_to_partition=node_to_partition,
        )

        self.streams = StreamBufferManager(
            self.request_state, self.tensor_manager, list(layout.consumer_connections),
            model.get_graph_walk_graphs() if model else None,
            graph_runtime=self._graph_runtime, topology=topology,
        )

        layout = replace(
            layout, side_keys=frozenset(self.engine_manager.side_keys),
            tensor_streams=self.engine_manager.inline_streams.tensor_streams,
        )
        self.layout = layout
        self._rebuilds = _make_rebuilds(
            model, model_config, layout.my_walks, layout.parallel_nodes, self.device, self.engine_manager.engine,
            layout.followed_nodes,
        )
        services = WorkerServices(
            worker_id=worker_id,
            device=device,
            graph_runtime=self._graph_runtime,
            engine_manager=self.engine_manager,
            tensor_manager=self.tensor_manager,
            request_state=self.request_state,
            communicator=self.communicator,
            parallel_groups=self.parallel_groups,
            rebuilds=self._rebuilds,
            streams=self.streams,
        )
        self.lifecycle = RequestLifecycle(layout=layout, graph_runtime=services.graph_runtime)
        self.admission = Admission(services=services, layout=layout, lifecycle=self.lifecycle)
        # TP async scheduling: the leader speculates N+1 during forward N and
        # broadcasts it at once; followers rebuild it during their own N.
        self.lockstep = Lockstep(
            services=services, layout=layout, admission=self.admission,
            async_sched=tp_async_sched_setting(
                os.environ.get("MSTAR_TP_ASYNC_SCHED"), model_config.get("tp_async_sched"),
            ),
            wg_instances=all_worker_graph_ids_to_instances,
        )
        # without these the runtime sees no parallel node, and its speculation
        # filter never holds one to the leader
        self._graph_runtime.set_node_metadata(
            parallel_nodes=layout.parallel_nodes,
            parallel_leader_nodes=layout.parallel_leader_nodes,
            tp_async_nodes={n for n in node_names if self.lockstep.async_for(n)},
        )
        self.scheduler = MicroScheduler(
            services=services, layout=layout, lifecycle=self.lifecycle, lockstep=self.lockstep,
            admission=self.admission, model_config=model_config,
        )
        self.preemption = Preemption(
            self.engine_manager, self._rebuilds, self.scheduler, self.request_state, self.admission.log_throttled,
            self._rid_str,
        )
        self._set_kv_writes(model, layout.my_walks)
        if layout.side_keys:
            logger.info(
                "Worker %s: %s run beside the chain on a side stream",
                worker_id, list(layout.side_keys),
            )

        self.tensor_manager.rid_to_str = self._rid_str

        self.partition_topology = topology

        if status_interval_s is not None:
            self._metrics = self._start_metrics(status_interval_s)
        self.ingest = InputIngest(
            services=services, layout=layout, lifecycle=self.lifecycle, lockstep=self.lockstep,
            wakeup_event=self.wakeup_event, metrics=self._metrics, enable_nvtx=self.enable_nvtx,
            set_rebuildable=self._set_rebuildable,
        )
        self.executor = StepExecutor(
            services=services, layout=layout, timer=self.timer, lockstep=self.lockstep,
            lifecycle=self.lifecycle, preemption=self.preemption, wakeup_event=self.wakeup_event,
            metrics=self._metrics, profile_info=self.profile_info, device_capability=self.device_capability,
            enable_nvtx=self.enable_nvtx, enable_prof=enable_prof, step_cpus=step_cpus,
            process_cpus=process_cpus,
        )
        self.speculator = Speculator(
            services=services, layout=layout, scheduler=self.scheduler, lifecycle=self.lifecycle,
            lockstep=self.lockstep, admission=self.admission, timer=self.timer,
            max_consecutive=max_consecutive_spec,
            peek=os.environ.get("MSTAR_SPEC_PEEK_FOR_FAIRNESS", "1") == "1", enable_nvtx=self.enable_nvtx,
        )
        self.pipeline = StepPipeline(
            services=services, layout=layout, timer=self.timer, lifecycle=self.lifecycle,
            lockstep=self.lockstep, admission=self.admission, scheduler=self.scheduler,
            executor=self.executor, ingest=self.ingest, speculator=self.speculator, preemption=self.preemption,
            metrics=self._metrics,
            profile_info=self.profile_info, launch_wait_s=launch_wait_s, enable_nvtx=self.enable_nvtx,
        )

    def _start_metrics(self, interval_s: float) -> WorkerMetrics:
        """Metrics on: send the conductor a WorkerStatus every ``interval_s``,
        and from ``ENABLE_METRICS`` on record the steps of the nodes this rank
        leads and the tensors every rank reads."""
        return WorkerMetrics(
            self.worker_id, interval_s, self.layout.parallel_leader_nodes,
            self.engine_manager.engine, self.scheduler, self._rebuilds,
            send=lambda status: self.communicator.send("conductor", ConductorMessage(
                message_type=ConductorMessageType.WORKER_STATUS, body=status,
            )),
            enable_transfers=self.tensor_manager.enable_transfer_metrics,
            preemptions=self.preemption.counts,
        )

    def _rid_str(self, request_id: int) -> str:
        """The wire identity, for a message about to leave this process."""
        return self._graph_runtime.get_rid_string(request_id)

    def _never_publish_unshared(
        self, wg_instances: dict[int, list[list[str]]] | None,
    ) -> None:
        """Turn off publication for good of each resource whose declaring
        nodes have one lockstep instance between them in this deployment:
        ``_skip_unshared_publish`` would skip it for every request."""
        if wg_instances is None:
            return
        engine = self.engine_manager.engine
        keys = []
        for key, nodes in engine.publisher_nodes.items():
            instances = {
                frozenset(instance)
                for wg_id, wg_nodes in self.layout.all_wg_nodes.items()
                if not nodes.isdisjoint(wg_nodes)
                for instance in wg_instances.get(wg_id, ())
            }
            if len(instances) == 1 and self.worker_id in next(iter(instances)):
                keys.append(key)
        if keys:
            engine.never_publish(keys)

    def _set_kv_writes(self, model: Model | None, my_walks: frozenset[tuple[str, str]]) -> None:
        """Record what each node of each paged KV cache writes here, and give each cache the loops that grow it
        in a walk this worker runs on one of its nodes."""
        labels = model.kv_labels() if model is not None else {}
        # key -> node -> its `NodeWrites`
        self._kv_writes = {}
        for kv in self.engine_manager.engine.paged_kv_caches():
            self._kv_writes[kv.key] = node_writes(labels.get(kv.key, {}), kv.nodes, my_walks)
            kv.cache.local_loops = {loop for writes in self._kv_writes[kv.key].values() for loop in writes.loops}

    def _set_rebuildable(self, request_id: int, configs: dict[str, ResourceReqConfig]) -> frozenset[str]:
        kv_configs = {key: config for key, config in configs.items() if isinstance(config, KVReqConfig)}
        layouts = {key: config.keyed_layouts() for key, config in kv_configs.items()}
        unpausable = any(config.unpausable for config in kv_configs.values())
        ingested = self._rebuilds.ingest_request(request_id, self._kv_writes, layouts, unpausable)
        for key, rebuildable in ingested.rebuildable.items():
            if key in configs:
                configs[key].rebuildable = rebuildable
        return ingested.unheld

    def _plan_step_cpus(
        self, worker_ids: list[str], local_worker_devices: list[str] | None,
    ) -> tuple[cpus.StepCpus | None, frozenset[int]]:
        """Build this process's CPU plan, return this worker's step CPUs (None:
        not pinned) and the mask everything else keeps, and log them. A
        ``step_thread_core`` number the mask can't honour raises.

        ``local_worker_devices`` is the device of every worker the conductor
        started, in ``worker_ids`` order; standalone, this worker is the only
        one.
        """
        devices = local_worker_devices or [str(self.device)]
        slot = worker_ids.index(self.worker_id) if local_worker_devices else 0
        plan = cpus.devices_plan(knobs.step_thread_core(), devices, knobs.idle_hold())
        cpus.install(plan)
        step = plan.workers[slot] if plan.workers else None
        if step is None:
            # a host that doesn't call for pinning is the API server's one line
            if not plan.declined:
                logger.info("Worker %s: step threads not pinned (%s)", self.worker_id, plan.why)
        else:
            logger.info(
                "Worker %s: step threads (main, GPU, plan) on CPUs %s (%s; %s, worker %d of %d, mask %s)",
                self.worker_id, cpus.format_cpu_list(step.cpus), step.why,
                self.device, slot + 1, len(devices), cpus.format_cpu_list(plan.mask),
            )
        return step, plan.mask

    def run(self) -> None:
        switch_interval = os.environ.get("MSTAR_PY_SWITCH_INTERVAL_SEC", "")
        if switch_interval:
            try:
                sys.setswitchinterval(float(switch_interval))
                logger.info(
                    "Worker %s: Python thread switch interval set to %ss",
                    self.worker_id,
                    switch_interval,
                )
            except ValueError:
                logger.warning(
                    "Worker %s: ignoring invalid MSTAR_PY_SWITCH_INTERVAL_SEC=%r",
                    self.worker_id,
                    switch_interval,
                )

        # Bound the load-time asymmetry between workers before any
        # subgroup NCCL collective fires inside the per-bs CUDA-graph
        # capture loop. Without this fence, a worker with a small model
        # (e.g. an 8B Talker) can finish loading, enter warmup, and hit
        # its first subgroup barrier while a worker with a 30B Thinker
        # is still streaming safetensors shards. The subgroup NCCL comm
        # is created lazily on that first collective; its connect-retry
        # budget is ~33 s, which is shorter than the load-time delta on
        # large multi-tower models. Syncing here means every worker
        # reaches warmup at the same wall-clock instant, so subgroup
        # bootstrap completes within the retry budget.
        self.parallel_groups.barrier_all()
        self.lockstep.verify_async_sched_agrees()

        # CUDA graph capture before entering the main loop
        repeats = self.layout.step_repeats
        self.engine_manager.warmup_all(repeats.asked_walks() if repeats is not None else {})
        if repeats is not None:
            repeats.settle(self.executor.repeat_refusal)

        # Sync every worker before the main loop opens. Per-batch-size
        # captures inside CudaGraphRunner are already barriered on the
        # node-local TP group, but that doesn't bound the time between
        # ``warmup_and_capture`` returning and ``run()`` starting to
        # schedule. Without this fence, a TP leader can finish warmup
        # quickly, schedule its first batch, and ZMQ-send
        # ``ScheduleTPNode`` to a follower that's still inside another
        # engine's ``warmup``. The follower can't service the message
        # yet, but the leader will sit on the first NCCL collective.
        self.parallel_groups.barrier_all()
        self.parallel_groups.restore_default_timeout()

        # Everything tracked right now — weights, capture buffers, wrapper
        # state — lives for the process, so gen2 gains nothing by walking it
        # every cycle. Collect first so nothing garbage gets made permanent,
        # then move the rest out of GC's reach. Refcounting still frees these,
        # and objects created after this stay fully collected; only a cycle
        # alive at this instant would now be retained.
        gc.collect()
        gc.freeze()
        logger.info(
            "Worker %s: gc.freeze() after warmup — %d objects moved to the "
            "permanent generation", self.worker_id, gc.get_freeze_count(),
        )
        if gen0 := knobs.gc_gen0_threshold():
            gc.set_threshold(gen0, *gc.get_threshold()[1:])
            logger.info("Worker %s: gc thresholds %s", self.worker_id, gc.get_threshold())
        self.timer.track_gc()

        # Setup (weight load + warmup + CUDA-graph capture) is complete. Tell
        # the conductor this worker is ready. The conductor blocks its main
        # loop until every worker reports in, so the API server only advertises
        # readiness once all workers can actually serve.
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.SETUP_DONE,
                body=SetupDone(worker_id=self.worker_id),
            ),
        )

        self.pipeline.run()
