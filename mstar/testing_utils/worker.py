"""A ``Worker`` for tests, built without its ``__init__``.

``Worker.__init__`` joins the process groups, opens the control mesh, and builds
the graph runtime and the engines. ``make_worker`` skips those and starts from
what they would leave behind: the step-to-step state (``Worker._init_step_state``,
the method ``__init__`` itself uses), a timer that is off, and a worker that
leads nothing. A test supplies the collaborators the code under test reaches
(usually a ``SimpleNamespace`` of what it calls), so one it forgot is an
``AttributeError`` rather than a stub answering for it.
"""
import torch

from mstar.profile.worker import WorkerProfileInfo
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.phase_timer import PhaseTimer
from mstar.worker.worker import Worker

# Built by ``__init__`` from the deployment; no default stands in for them.
_COLLABORATORS = frozenset({
    "communicator", "wakeup_event", "tensor_manager", "_graph_runtime", "engine_manager",
    "request_state", "scheduler", "streams", "parallel_groups", "partition_topology",
    "_my_consumer_connections", "_all_wg_nodes", "_lockstep_consumers",
})


def make_worker(**overrides) -> Worker:
    """A ``Worker`` with no threads, sockets or engines. ``overrides`` replace
    fields and methods by name; a name ``Worker`` does not have raises
    ``TypeError``. Only names passed here are checked: a field a test sets on
    the returned worker is not."""
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
    worker.tp_async_sched = False
    worker.tp_async_nodes = None
    worker._setup_tp_read_check()
    # no streams, no loops, no declared stop policies
    worker._walk_drivers = {}
    worker._walk_driven_edges = {}
    worker._signal_consumer = {}
    worker._streaming_edge_names = set()
    worker._innermost_loop_by_node = {}
    worker._stop_token_signal = {}

    for name, value in overrides.items():
        if name not in _COLLABORATORS and not hasattr(worker, name):
            raise TypeError(f"Worker has no field {name!r}")
        setattr(worker, name, value)
    if "enable_nvtx" in overrides:
        worker.timer.enable_nvtx = overrides["enable_nvtx"]
    return worker
