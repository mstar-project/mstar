"""Acceptance tests: a TP=2 follower's async input read fails after rank 0 has
read its own copy of that input.

Rank 0 schedules a TP step from its own copy of each row's inputs, so a
follower whose read of one of them fails can be left holding a step it cannot
run as rank 0 runs it. Whatever the fix, these must hold:

  A1  no hang, and only the request whose read failed fails; a request
      running beside it completes
  A2  nothing of the failed row leaves either rank: no routed output, no
      tensor registered to be read elsewhere, no output sent on
  A3  with ``prefix_cache_tp`` on, the failed row's pages never become
      leasable, so a later request with the same prefix doesn't reuse them
  A4  a request running beside the failed one gives bit-identical outputs
      to the same run with no failure
  A5  the same for a text-only row, which has no image grid

Each run is two real ranks of one TP=2 instance, each in its own process
with a gloo process group, running the real ``Worker.run`` loop, the real
``MicroScheduler``, the real shared-memory tensor manager, the real
``Engine`` (prepare, admit, forward, commit through the real ``StepRunner``
and ``KVManager``) and Qwen3.5's real ``prepare_inputs``, which reads
``image_grid_thw`` on the host to size the image span. The LLM forward is a
stand-in that all-reduces each rank's partial over the TP group, as a
row-parallel layer does, so a rank that never joins a step, or joins it with
another token count, shows up as a collective timeout or a gloo abort.

Fakes: the graph runtime and request state (inputs land through
``ingest_inputs_batch``; what a rank routes and sends is recorded off
``complete_and_route_batch`` and ``send_outputs``), the conductor (played by
the test: FAIL_REQUESTS starts the drain, both ranks' READS_DONE sends the
REMOVE), and the reads, whose futures the test resolves. The follower's read
of r1 resolves when the follower first hears from rank 0 about a step, or a
couple of seconds in, whichever comes first, so it fails after rank 0 has
scheduled from its own copy under any design that lets it.

The tests check behavior only, never a hook, so a design passes them however
it gets there (holding the row on rank 0, as the read gate in #754 does, or
zero-filling the follower's span and dropping the row at the end of the
step). Per-rank setup that a design adds (``Worker._setup_read_gate``) is
called when it exists; a design with its own setup has to be added to
``_DESIGN_SETUP``.
"""

from __future__ import annotations

import datetime
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.utils.ipc_format import WorkerMessageType

HERE = Path(__file__).resolve().parent

WORKERS = ("worker_0", "worker_1")
NODE = "LLM"
VISION_WALK, TEXT_WALK = "prefill_vision", "prefill"
KV = "kv"
HIDDEN = 16
VOCAB = 512
MERGE = 2
PAGE_SIZE = 16
ROOT = b"tp read failure"
SIGNALS = ("new_token", "hidden")
# A rank waiting this long in a collective for its partner is the hang.
COLLECTIVE_S = 8.0
# The follower's late read resolves at its first SCHEDULE_TP, or after this.
TRIGGER_S = 2.0
# The whole run; a run still going past this hung.
DEADLINE_S = 30.0
# Per-rank setup a fix adds to Worker.__init__ (see the module docstring).
_DESIGN_SETUP = ("_setup_read_gate", "_setup_tp_read_check")

# What each request reads. A grid prompt is Qwen3.5's prefill_vision layout:
# text, image, text; an image of grid (t, h, w) takes t*(h/2)*(w/2) tokens.
_PROMPTS = {
    "grid": {
        "r0": dict(texts=(5, 7), grid=(1, 16, 24)),
        "r1": dict(texts=(6, 4), grid=(1, 32, 32)),
    },
    "text": {
        "r0": dict(texts=(40,)),
        "r1": dict(texts=(53,)),
    },
}
# the input of r1 whose follower read fails
_FAILING = {"grid": "image_grid_thw", "text": "text_inputs"}


def _ids(n: int, seed: int) -> list[int]:
    return [3 + (seed * 7919 + i * 31) % (VOCAB - 3) for i in range(n)]


def _inputs(kind: str, rid: str) -> dict[str, list[torch.Tensor]]:
    """The tensors each rank reads for ``rid``, by signal."""
    spec = _PROMPTS[kind][rid]
    seed = int(rid[1:]) + 1
    texts = [torch.tensor(_ids(n, seed * 10 + i)) for i, n in enumerate(spec["texts"])]
    if kind == "text":
        return {"text_inputs": texts}
    t, h, w = spec["grid"]
    span = t * (h // MERGE) * (w // MERGE)
    gen = torch.Generator().manual_seed(seed)
    return {
        "text_inputs": texts,
        "image_grid_thw": [torch.tensor([[t, h, w]], dtype=torch.int64)],
        "vision_embeds": [torch.randn(span, HIDDEN, generator=gen)],
    }


def _walk(kind: str) -> str:
    return VISION_WALK if kind == "grid" else TEXT_WALK


def _prompt_tokens(kind: str, rid: str) -> list[int]:
    return [int(v) for t in _inputs(kind, rid)["text_inputs"] for v in t]


@dataclass
class Scenario:
    kind: str = "grid"
    # the follower's read of r1's _FAILING input fails
    fail: bool = True
    prefix_cache: bool = False
    # rank 1's main loop stalls this long, once, in the first message poll
    # after its first SCHEDULE_TP, so rank 0 routes that step first
    follower_stall_s: float = 0.0
    shm_root: str = ""
    store: str = ""


# ── the rank (runs in its own process) ──────────────────────────────────


class _Stop(BaseException):
    """Ends ``Worker.run`` from inside the loop; not an Exception, so the
    loop's error handler lets it through."""


class _Communicator:
    """Messages to and from the parent process, which routes them between the
    ranks and plays the conductor. Every message crosses the wire codec."""

    def __init__(self, worker_id, inbox, outbox, on_message, stall_s=0.0):
        self.worker_id = worker_id
        self.inbox, self.outbox = inbox, outbox
        self.on_message = on_message
        self.stopped = False
        self._held = []
        self._stall_s, self._scheduled = stall_s, False

    def send(self, dest, msg=None):
        from mstar.communication.wire import encode
        self.outbox.put(("send", self.worker_id, dest, encode(msg)))

    def _pull(self, timeout):
        try:
            item = self.inbox.get(timeout=timeout) if timeout else self.inbox.get_nowait()
        except queue.Empty:
            return False
        if item == "stop":
            self.stopped = True
        else:
            self._held.append(item)
        return True

    def get_all_new_messages(self):
        from mstar.communication.wire import decode
        if self._scheduled and self._stall_s:
            time.sleep(self._stall_s)
            self._stall_s = 0.0
        while self._pull(0):
            pass
        if self.stopped:
            raise _Stop
        held, self._held = self._held, []
        messages = [decode(b) for b in held]
        self._scheduled |= any(m.message_type == WorkerMessageType.SCHEDULE_TP for m in messages)
        self.on_message(messages)
        return messages

    def wait_for_work(self, timeout_ms=10):
        self.on_message([])
        if self._held or self.stopped:
            return
        self._pull(min(timeout_ms, 20) / 1000)


class _Rope:
    """The position counter Qwen3.5's prepare_inputs reads its start from."""

    def position(self, rid, label):
        return 0


def _submodule_class():
    from mstar.engine.resources.kv.config import KVStep
    from mstar.engine.resources.step import Segment, SubmoduleStep
    from mstar.model.qwen3_5.config import ROPE
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    class _LLM(LLMSubmodule):
        """Qwen3.5's LLM node: everything but the model is the real submodule
        (``prepare_inputs``, which reads the image grid on the host, and the
        capture configs' dummy rows), so a fix that builds a row the way the
        submodule does gets the real code. The forward is a stand-in: each
        rank's partial of every token's embedding is summed over the TP group,
        and a row's output is its mean, which every rank then holds whole. A
        row's output depends on nothing but its own tokens."""

        def __init__(self, group, rank):
            torch.manual_seed(1234)
            model = torch.nn.Module()
            model.model = torch.nn.Module()
            model.model.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
            super().__init__(
                model=model,
                config=SimpleNamespace(
                    vision_start_token_id=1, vision_end_token_id=2, hidden_size=HIDDEN,
                    stop_token_ids=(),
                ),
                vision_config=SimpleNamespace(spatial_merge_size=MERGE),
            )
            self.head = torch.randn(HIDDEN, VOCAB)
            self.requires_grad_(False)
            self.bind_node_resources({ROPE: _Rope()})
            self.group, self.rank = group, rank

        # the real model's step, stop and emit settings, which read state this
        # stand-in doesn't have
        def _device_loopback(self):
            return False

        def uniform_row_inputs(self, graph_walk, has_loopback=False):
            return None

        def device_loopback_signals(self, graph_walk):
            return frozenset()

        def max_batch_size(self, graph_walk):
            return None

        def max_rows_in_combined(self, graph_walk):
            return None

        def postprocess(self, request_id, request_info, outputs, **kwargs):
            return

        def check_stop(self, request_id, request_info, outputs):
            return set()

        def check_stop_batched(self, request_ids, request_infos, host_rows):
            return None

        def step_is_input_free(self, graph_walk):
            return False

        def inline_client_signals(self, graph_walk):
            return {}

        def can_batch(self, batch, model_inputs):
            return True

        def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
            return SubmoduleStep(
                steps={KV: KVStep()},
                segments=[
                    Segment(rid, "main", inp.input_seq_len)
                    for rid, inp in zip(request_ids, inputs, strict=True)
                ],
            )

        def preprocess(self, graph_walk, engine_inputs, inputs):
            return {"embeds": [
                inp.input_embeds if inp.input_embeds is not None
                else self.model.model.embed_tokens(inp.input_ids)
                for inp in inputs
            ]}

        def forward(self, graph_walk, engine_inputs, **kwargs):
            raise AssertionError("every step here runs batched")

        def forward_batched(self, graph_walk, engine_inputs, embeds):
            packed = torch.cat(embeds).to(torch.float32)
            # rank 0 holds a third of each value and rank 1 two thirds
            partial = packed * ((self.rank + 1) / 3.0)
            whole = self.group.all_reduce(partial)
            out, at = {}, 0
            for rid, rows in zip(engine_inputs.request_ids, embeds, strict=True):
                hidden = whole[at:at + rows.shape[0]].mean(dim=0)
                at += rows.shape[0]
                out[rid] = {
                    "new_token": [torch.argmax(hidden @ self.head).reshape(1)],
                    "hidden": [hidden.clone()],
                }
            return out

    return _LLM


def _kv_manager(rank, prefix_cache):
    from mstar.engine.resources.base import KeyedWalks
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import PagedKVConfig
    from mstar.engine.resources.kv.manager import KVManager

    class _NoTransfer:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    manager_mod.KVTransferManager = _NoTransfer
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2,
            max_num_pages=128, page_size=PAGE_SIZE, prefix_cache_tp=prefix_cache,
        ),
        name=KV, joint_comm_group=SimpleNamespace(rank=rank, world_size=2),
        transfer_engine_info=None, device=torch.device("cpu"), dtype=torch.float32,
    )
    if prefix_cache:
        kv.enable_prefix_cache(ROOT, {"main": KeyedWalks((TEXT_WALK,), ("decode",))})
    return kv


def _kv_config(tokens):
    from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
    from mstar.engine.resources.kv.keys import chain

    whole = len(tokens) // PAGE_SIZE
    return KVReqConfig(
        prefix_cache=True,
        prefix_keys={"main": chain([tokens[at:at + PAGE_SIZE] for at in range(0, len(tokens), PAGE_SIZE)])},
        prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        kv_layout={"main": StreamLayout(
            spans=(PrefixSpan(len(tokens), len(tokens), TEXT_WALK),), keyed_spans=1,
        )},
    )


def _engine(rank, group, kv, prefix_cache, graph_runtime):
    from mstar.engine.engine import Engine, SubmoduleManagement
    from mstar.engine.resources import StepRunner

    sub = _submodule_class()(group, rank)
    engine = Engine(graph_runtime=graph_runtime, autocast_dtype=None)
    if rank == 0:
        engine._tp_leader_nodes = {NODE}
    else:
        engine._tp_follower_nodes = {NODE}
    engine._resources = {KV: kv}
    engine._runner = StepRunner(engine._resources, node_resources={NODE: [KV]})
    engine._submodules = {NODE: SubmoduleManagement(
        submodule=sub, forward=sub.forward, forward_batched=sub.forward_batched,
        joint_comm_group=SimpleNamespace(rank=rank, world_size=2), resources={KV: kv},
    )}
    engine._keyed_walks = {NODE: {TEXT_WALK}} if prefix_cache else {}
    engine._prefix_model = "_LLM"
    engine._device = torch.device("cpu")
    engine._loopback_steps = {}
    engine._autocast_dtype_for = lambda submodule: None
    engine.reserve_replay_slot = lambda batch: None
    return engine


class _Graph:
    """The graph runtime and request state of one rank, for one node: a row is
    ready once each of its walk's inputs landed, and what the rank routes and
    sends is reported to the parent."""

    def __init__(self, rank, kind, report):
        self.rank, self.kind, self.report = rank, kind, report
        self.walk = _walk(kind)
        self.required = tuple(_inputs(kind, "r0"))
        self.landed: dict[str, dict[str, list[int]]] = {}
        self.ready: set[str] = set()
        self.in_flight: set[str] = set()
        self.done: set[str] = set()
        self.per_request_info = {}
        self.fwd = {}
        self.tensor_manager = None
        self.combined_walk_of = {}
        # no client copy rides inline: every output is stored and routed
        self.supports_inline_emit = False

    def admit(self, rid, info):
        self.per_request_info[rid] = SimpleNamespace(
            request_id=rid, stream_buffers={}, stream_buffers_by_consumer={}, ended_streams=set(),
        )
        self.fwd[rid] = info
        self.landed[rid] = {}

    # rid handles are the wire ids themselves
    def get_rid_handle(self, wire_rid):
        return wire_rid if wire_rid in self.per_request_info else None

    def get_rid_string(self, rid):
        return rid

    def get_sharding_config(self, rid):
        group = SimpleNamespace(
            nodes={NODE}, tp_size=2, _tp_rank=self.rank, _workers=list(WORKERS),
        )
        return SimpleNamespace(groups=[group], get_sharding_group=lambda node, walk: group)

    # inputs
    def ingest_inputs_batch(self, block, can_buffer):
        at = 0
        for i, rid in enumerate(block.rids):
            n = block.tensors_per_edge[i]
            signal = block.signal_names[block.signal_name_idxs[i]]
            if rid in self.landed:
                self.landed[rid][signal] = list(block.uuids[at:at + n])
            at += n
        for rid, landed in self.landed.items():
            if rid not in self.done and rid not in self.in_flight and all(s in landed for s in self.required):
                self.ready.add(rid)

    # the scheduler's view
    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        from mstar.graph.runtime.base import ReadyNodeSpec
        if exclude_target is not None and exclude_target == (NODE, self.walk):
            return []
        if target is not None and target != (NODE, self.walk):
            return []
        rids = [r for r in sorted(self.ready) if r not in exclude_rids and r in self.per_request_info]
        return [ReadyNodeSpec(NODE, self.walk, rids)] if rids else []

    def has_ready_excluding(self, exclude_rids, exclude_target=None):
        return bool(self.get_ready_nodes(exclude_rids, exclude_target=exclude_target))

    def get_worker_graph_id_for_node(self, node_name, graph_walk):
        return 0

    def pop_rids(self, node_name, graph_walk, request_ids, check_ready=False):
        from mstar.graph.runtime.base import ColumnarEdgeSpecs, PopRidsOutput
        from mstar.utils.containers import ParallelList
        if check_ready and not all(r in self.ready for r in request_ids):
            return None
        block = ColumnarEdgeSpecs.empty()
        rids = []
        for rid in request_ids:
            if rid not in self.ready and check_ready:
                continue
            self.ready.discard(rid)
            rids.append(rid)
            for signal, uuids in self.landed.get(rid, {}).items():
                block.add(rid, signal, uuids, False)
        return PopRidsOutput(
            wg_ids=ParallelList(rids, [0] * len(rids)), input_edges=block,
            output_signals=SIGNALS, rid_walk_idx=[0] * len(rids), walk_output_signals=(SIGNALS,),
        )

    def pop_walk_rids(self, node_name, rows, check_ready=False):
        return self.pop_rids(node_name, rows.walks[0], list(rows.rids), check_ready)

    def get_nodes(self, node_name, rids, wg_ids):
        return [SimpleNamespace(name=node_name) for _ in rids]

    def push_back_node(self, node_name, rids, wg_ids):
        for rid in rids:
            if rid in self.per_request_info and rid not in self.done:
                self.ready.add(rid)

    def set_in_flight(self, node_name, rids, wg_ids, in_flight):
        for rid in rids:
            (self.in_flight.add if in_flight else self.in_flight.discard)(rid)

    def is_async_schedulable(self, node_name, walk):
        return False

    def get_output_signals(self, node_name, walk):
        return list(SIGNALS)

    def get_partition_for_node(self, node_name):
        return "default"

    def get_graph_walk(self, rid, partition):
        return self.walk

    def get_fwd_info(self, rid, partition):
        return self.fwd[rid]

    def ingested_stream_chunks(self, rid, node_name):
        return {}

    # post-processing
    def cleanup_consumed_inputs(self, node_name, rids, wg_ids):
        from mstar.graph.runtime.base import FreedTensors
        return FreedTensors.none()

    def clear_pending_loop_stops(self):
        pass

    def pending_loop_stop_rids(self, walks, loop_name):
        return set()

    def stop_loops_batched(self, **kwargs):
        return []

    def update_dynamic_loop_iters(self, per_request_info, partition):
        pass

    def mark_stream_partition_done(self, rid, partition):
        pass

    def complete_and_route_batch(self, route_input):
        """Every output of a routed row goes to a remote reader (the client), so
        the worker registers each one to be read."""
        from mstar.graph.runtime.base import FreedTensors, RouteOutput
        rids, uuids, reg_rids, at = route_input.wg_ids.keys, list(route_input.tensors), [], 0
        outputs = {}
        for i, rid in enumerate(rids):
            outputs[rid] = {}
            for j, signal in enumerate(route_input.output_signals):
                n = route_input.num_tensors[i * len(route_input.output_signals) + j]
                tensors = [self.tensor_manager.get_tensor(u) for u in uuids[at:at + n]]
                outputs[rid][signal] = [
                    (str(t.dtype), tuple(t.shape), t.contiguous().numpy().tobytes()) for t in tensors
                ]
                reg_rids += [rid] * n
                at += n
            self.done.add(rid)
        self.report(("routed", self.rank, outputs))
        return RouteOutput(
            completion_id=0, register_uuids=uuids, register_rids=reg_rids,
            new_token_output_idxs=[], local_streaming_by_signal={},
            freed_inputs=FreedTensors.none(),
        )

    def send_outputs(self, send_input):
        self.report(("sent", self.rank, sorted(set(send_input.per_request_info.keys))))
        return []

    def buffer_publish_info(self, *args):
        pass

    def flush_publish_info(self, *args):
        pass

    def get_pending_publish_info(self, rid, partition):
        return {}

    def remove_request(self, rid):
        self.per_request_info.pop(rid, None)
        self.landed.pop(rid, None)
        self.ready.discard(rid)
        self.report(("removed", self.rank, rid))

    def commit_speculation(self, *args, **kwargs):
        pass


def _pointer(tensor, uuid):
    from mstar.graph.base import TensorPointerInfo
    return TensorPointerInfo(
        list(tensor.shape), tensor.dtype, tensor.nbytes, 0, list(tensor.stride()),
        uuid, "s", "api_server",
    )


class _Reads:
    """This rank's reads of every request's inputs, through the real SHM
    manager. A read's buffer is registered when it starts, holds the values
    once it lands, and keeps whatever it held if it fails."""

    def __init__(self, tm, kind, rank, scenario):
        from mstar.communication.tensors import FutureAndPointers
        from mstar.graph.base import GraphEdge
        self.tm, self.late, self.fired = tm, [], False
        uuid = 1000 * (rank + 1)
        for rid in ("r0", "r1"):
            for signal, tensors in _inputs(kind, rid).items():
                infos, buffers = [], []
                for t in tensors:
                    uuid += 1
                    info = _pointer(t, uuid)
                    buf = torch.zeros_like(t)
                    tm.tensor_store.put_tensor(rid, uuid, buf, info)
                    infos.append(info)
                    buffers.append((buf, t))
                edge = GraphEdge(next_node=NODE, name=signal, tensor_info=infos)
                future = Future()
                tm.pending.append(FutureAndPointers(future=future, graph_edges=[edge], request_id=rid))
                if rank == 1 and rid == "r1":
                    fails = scenario.fail and signal == _FAILING[kind]
                    self.late.append((future, buffers, fails))
                else:
                    self._land(future, buffers)

    @staticmethod
    def _land(future, buffers):
        for buf, values in buffers:
            buf.copy_(values)
        future.set_result(None)

    def fire(self):
        """The follower's reads of r1 resolve."""
        if self.fired:
            return
        self.fired = True
        for future, buffers, fails in self.late:
            if fails:
                future.set_exception(RuntimeError("Mooncake read failed. Status: -1"))
            else:
                self._land(future, buffers)


def _tp_group(rank):
    import torch.distributed as dist

    from mstar.distributed.all_reduce import AllReduce
    from mstar.distributed.communication import CommGroup

    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    group.device_group = group.cpu_group = dist.group.WORLD
    group.initialized = True
    group._all_reduce = AllReduce(group.device_group)
    return group


def _worker(rank, scenario, comm, graph, engine, tm, group):
    import torch.distributed as dist

    from mstar.distributed.communication import CommGroup, JointGroups
    from mstar.utils.containers import RecentSet
    from mstar.worker.micro_scheduler import MicroScheduler
    from mstar.worker.worker import _REMOVED_RIDS_KEEP, Worker

    w = Worker.__new__(Worker)
    w._walk_drivers = {}
    w._walk_driven_edges = {}
    w.worker_id = WORKERS[rank]
    w.device = torch.device("cpu")
    w.enable_nvtx = w.enable_prof = False
    w._launch_wait_s = 1.0
    w._max_consecutive_spec = 1
    w._device_loopback = {}
    w._host_rows_slot = 0
    w._stream_buffer_rids = set()
    w._phase_period = 0
    w._phase_buf = {}
    w._gpu_spans = []
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.parallel_groups = SimpleNamespace(
        barrier_all=dist.barrier,
        restore_default_timeout=lambda: None,
        get_tp_config_for_node=lambda node: group,
        get_sp_config_for_node=lambda node: SimpleNamespace(world_size=1),
        get_joint_group_for_node=lambda node: JointGroups(tp_group=group, sp_group=CommGroup.trivial()),
        get_instance_rank_for_node=lambda node: rank,
        get_instance_world_size_for_node=lambda node: 2,
    )
    w.communicator = comm
    w.wakeup_event = SimpleNamespace(drain=lambda: None, register_future=lambda f: None)
    w.tensor_manager = tm
    w._graph_runtime = graph
    w.engine_manager = SimpleNamespace(
        get_engine=lambda node: engine, warmup_all=lambda: None,
        remove_request=engine.remove_request, evictable_nodes=lambda: (),
    )
    w.request_state = graph
    w.parallel_nodes = {NODE}
    w.parallel_leader_nodes = {NODE} if rank == 0 else set()
    w.is_tp_follower = rank != 0
    w.tp_async_sched, w.tp_async_nodes = False, None
    w._tp_broadcast_seq = 0
    w._tp_nospec = RecentSet(64)
    w._tp_leader_gap_warned = False
    w.scheduler = MicroScheduler(w.engine_manager, parallel_leader_nodes=w.parallel_leader_nodes)
    w.scheduler.runtime = graph
    w._unprocessed_messages = {}
    w._removed_rids = RecentSet(_REMOVED_RIDS_KEEP)
    w._last_active = {}
    w._hold_logged = {}
    w._in_flight_rids = set()
    w._pending_removes = {}
    w._removes_awaiting_step = {}
    w._pending_drains, w._draining_rids, w._reads_done_sent = set(), set(), set()
    w.streaming_buffers = {}
    w.partition_topology = None
    w._my_consumer_connections = []
    w._streaming_edge_names = set()
    w._consumer_node_cache = {}
    w._stream_partition = {}
    w._innermost_loop_by_node = {}
    w._metrics = None
    for setup in _DESIGN_SETUP:
        if hasattr(w, setup):
            getattr(w, setup)()
    return w


def _probe(engine, rid, tokens):
    """The leases rank 0 would take for a new request with ``tokens`` if it
    sent a step for it now, from this rank's own index."""
    from mstar.engine.resources.step import StepContext
    engine.add_request(rid, {KV: _kv_config(tokens)})
    batch = SimpleNamespace(
        node_name=NODE, request_ids=(rid,),
        step_context=StepContext(request_ids=(rid,), graph_walk=TEXT_WALK, slot=0, capture=False),
    )
    hits = engine.decide_cached_prefix(batch)
    leases = hits.get(rid, {})
    return {label: list(pages) for label, pages in leases.items()}


def _rank_main(rank, scenario, inbox, outbox):
    sys.path[:0] = [str(HERE), str(HERE.parents[1])]
    os.environ.setdefault("MSTAR_PRE_PLAN_SPEC", "0")
    import logging
    logging.basicConfig(level=logging.WARNING, format=f"[rank {rank}] %(levelname)s %(name)s: %(message)s")
    try:
        import conftest  # noqa: F401  (the triton stub)
        import torch.distributed as dist

        import mstar.communication.wire_types  # noqa: F401  (registers the tags)
        from mstar.communication.tensors import SharedMemoryCommunicationManager
        from mstar.conductor.request_info import CurrentForwardPassInfo
        from mstar.model.qwen3_5.qwen3_5_model import IMAGE_PART, TEXT_PART

        dist.init_process_group(
            "gloo", init_method=f"file://{scenario.store}", rank=rank, world_size=2,
            timeout=datetime.timedelta(seconds=COLLECTIVE_S),
        )
        group = _tp_group(rank)
        kind = scenario.kind
        report = outbox.put
        graph = _Graph(rank, kind, report)
        kv = _kv_manager(rank, scenario.prefix_cache)
        engine = _engine(rank, group, kv, scenario.prefix_cache, graph)
        tm = SharedMemoryCommunicationManager(
            WORKERS[rank], "localhost", "cpu", None, shm_dir=os.path.join(scenario.shm_root, str(rank)),
        )
        graph.tensor_manager = tm
        order = [TEXT_PART, IMAGE_PART, TEXT_PART] if kind == "grid" else [TEXT_PART]
        for rid in ("r0", "r1"):
            graph.admit(rid, CurrentForwardPassInfo(
                request_id=rid, graph_walk=_walk(kind), fwd_index=0, random_seed=0, max_tokens=1,
                step_metadata={"prefill_order": order},
            ))
            tokens = _prompt_tokens(kind, rid)
            engine.add_request(rid, {KV: _kv_config(tokens)} if scenario.prefix_cache else None)
        reads = _Reads(tm, kind, rank, scenario)
        started = time.monotonic()

        def on_message(messages):
            from mstar.utils.ipc_format import WorkerMessageType
            if rank != 1 or reads.fired:
                return
            if any(m.message_type == WorkerMessageType.SCHEDULE_TP for m in messages) \
                    or time.monotonic() - started > TRIGGER_S:
                reads.fire()

        comm = _Communicator(
            WORKERS[rank], inbox, outbox, on_message, scenario.follower_stall_s if rank == 1 else 0.0,
        )
        tm.communicator = comm
        w = _worker(rank, scenario, comm, graph, engine, tm, group)
        try:
            w.run()
        except _Stop:
            pass
        final = {}
        if scenario.prefix_cache:
            failed_prefix = _prompt_tokens(kind, "r1")[:3 * PAGE_SIZE] + _ids(9, 77)
            healthy_prefix = _prompt_tokens(kind, "r0")[:2 * PAGE_SIZE] + _ids(9, 78)
            final["probe_failed_prefix"] = _probe(engine, "p1", failed_prefix)
            final["probe_healthy_prefix"] = _probe(engine, "p0", healthy_prefix)
        report(("final", rank, final))
    except BaseException:
        report(("crash", rank, traceback.format_exc()))
        raise


# ── the parent: routes messages and plays the conductor ─────────────────


@dataclass
class Run:
    hung: bool = False
    crashed: dict = field(default_factory=dict)
    exitcodes: dict = field(default_factory=dict)
    # rid -> the rank that first reported it failed, and why
    failed: dict = field(default_factory=dict)
    # rid -> ranks that sent READS_DONE for it
    reads_done: dict = field(default_factory=dict)
    removed: dict = field(default_factory=dict)
    # rank -> rid -> signal -> [(dtype, shape, bytes)]
    routed: dict = field(default_factory=lambda: {0: {}, 1: {}})
    sent: dict = field(default_factory=lambda: {0: set(), 1: set()})
    # messages a rank sent anywhere but the other rank and the conductor
    other_sends: list = field(default_factory=list)
    final: dict = field(default_factory=dict)
    elapsed: float = 0.0

    def completed(self, rid):
        """Every rank routed the request's output; it never failed. The run
        stops once each request completed or was torn down, so a follower
        that routes after rank 0 still gets its outputs recorded."""
        return all(rid in routed for routed in self.routed.values()) and rid not in self.failed

    def torn_down(self, rid):
        return rid in self.failed and self.reads_done.get(rid, set()) >= {0, 1}

    def summary(self):
        return (
            f"hung={self.hung} crashed={sorted(self.crashed)} exit={self.exitcodes} "
            f"failed={ {r: e[:120] for r, e in self.failed.items()} } "
            f"reads_done={self.reads_done} routed={ {k: sorted(v) for k, v in self.routed.items()} } "
            f"after {self.elapsed:.1f}s"
        )


def _run(tmp_path, scenario: Scenario) -> Run:
    import mstar.communication.wire_types  # noqa: F401
    from mstar.communication.wire import decode, encode
    from mstar.utils.ipc_format import (
        ConductorMessageType,
        DrainRequest,
        MessageSource,
        RemoveRequest,
        WorkerMessage,
        WorkerMessageType,
    )

    tmp_path.mkdir(parents=True, exist_ok=True)
    scenario.shm_root = str(tmp_path / "shm")
    scenario.store = str(tmp_path / "store")
    ctx = mp.get_context("spawn")
    inboxes = [ctx.Queue() for _ in WORKERS]
    outbox = ctx.Queue()
    procs = [ctx.Process(target=_rank_main, args=(r, scenario, inboxes[r], outbox), daemon=True) for r in (0, 1)]
    for p in procs:
        p.start()
    run = Run()
    start = time.monotonic()
    stopping_since = None

    def to_both(message):
        for box in inboxes:
            box.put(encode(message))

    while True:
        now = time.monotonic()
        if stopping_since is None:
            settled = all(run.completed(r) or run.torn_down(r) for r in ("r0", "r1"))
            if settled or now - start > DEADLINE_S:
                run.hung = not settled
                stopping_since = now
                for box in inboxes:
                    box.put("stop")
        elif len(run.final) + len(run.crashed) == 2 or now - stopping_since > COLLECTIVE_S + 5:
            break
        if stopping_since is None and not all(p.is_alive() for p in procs):
            # a rank died: what it left is all there is
            run.hung = not all(run.completed(r) or run.torn_down(r) for r in ("r0", "r1"))
            stopping_since = now
            for box in inboxes:
                box.put("stop")
        try:
            item = outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        tag = item[0]
        if tag == "send":
            _, src, dest, data = item
            if dest in WORKERS:
                inboxes[WORKERS.index(dest)].put(data)
                continue
            msg = decode(data)
            if dest != "conductor":
                run.other_sends.append((src, dest, msg))
                continue
            rank = WORKERS.index(src)
            if msg.message_type == ConductorMessageType.FAIL_REQUESTS:
                for rid, error in msg.body.errors.items():
                    if rid in run.failed:
                        continue
                    run.failed[rid] = f"rank {rank}: {error}"
                    to_both(WorkerMessage(
                        message_type=WorkerMessageType.DRAIN_REQUEST,
                        body=DrainRequest(request_id=rid, source=MessageSource.CONDUCTOR),
                    ))
            elif msg.message_type == ConductorMessageType.READS_DONE:
                rid = msg.body.request_id
                run.reads_done.setdefault(rid, set()).add(WORKERS.index(msg.body.entity_id))
                if run.reads_done[rid] >= {0, 1} and rid not in run.removed:
                    run.removed[rid] = True
                    to_both(WorkerMessage(
                        message_type=WorkerMessageType.REMOVE_REQUEST,
                        body=RemoveRequest(request_id=rid, source=MessageSource.CONDUCTOR),
                    ))
        elif tag == "routed":
            for rid, outputs in item[2].items():
                run.routed[item[1]].setdefault(rid, []).append(outputs)
        elif tag == "sent":
            run.sent[item[1]] |= set(item[2])
        elif tag == "final":
            run.final[item[1]] = item[2]
        elif tag == "crash":
            run.crashed[item[1]] = item[2]
    for p in procs:
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
    run.exitcodes = {r: p.exitcode for r, p in enumerate(procs)}
    for r, code in run.exitcodes.items():
        if code not in (0, None) and r not in run.crashed:
            run.crashed[r] = f"exit code {code} (a gloo abort is a collective with another shape)"
    run.elapsed = time.monotonic() - start
    return run


# ── the tests ───────────────────────────────────────────────────────────

KINDS = [
    pytest.param("grid", id="grid"),
    # A5: a text-only row has no grid, so its step has rank 0's shape anyway
    pytest.param("text", id="text_only"),
]


def _healthy_ok(run: Run):
    assert not run.crashed, run.summary() + "\n" + "\n".join(map(str, run.crashed.values()))
    assert not run.hung, "the instance hung: " + run.summary()
    assert run.completed("r0"), "r0, beside the failed request, did not complete: " + run.summary()


@pytest.mark.parametrize("kind", KINDS)
def test_a_followers_failed_read_fails_only_its_request_and_nothing_hangs(tmp_path, kind):
    """A1 (and A5 for text_only). The follower's read of r1's image grid (or
    text) fails after rank 0 read its own copy: the run settles before the
    deadline, r1 fails and is torn down on both ranks, and r0 completes."""
    run = _run(tmp_path, Scenario(kind=kind))
    _healthy_ok(run)
    assert set(run.failed) == {"r1"}, run.summary()
    assert "input tensor read failed" in run.failed["r1"], run.summary()
    assert run.torn_down("r1"), "r1's drain never finished on both ranks: " + run.summary()


@pytest.mark.parametrize("kind", KINDS)
def test_nothing_of_the_failed_row_leaves_either_rank(tmp_path, kind):
    """A2 (and A5). No rank routes r1's outputs (so nothing is persisted,
    registered for another reader or sent to the client), and no rank sends
    anything for r1 past its runtime."""
    run = _run(tmp_path, Scenario(kind=kind))
    _healthy_ok(run)
    for rank in (0, 1):
        assert "r1" not in run.routed[rank], f"rank {rank} routed r1's outputs: " + run.summary()
        assert "r1" not in run.sent[rank], f"rank {rank} sent r1's outputs on: " + run.summary()
    out = [s for s in run.other_sends if getattr(s[2].body, "request_id", None) == "r1"
           and s[2].message_type != WorkerMessageType.TENSOR_RECEIVED]  # an input's ack
    assert not out, out


def test_the_failed_rows_pages_never_become_leasable(tmp_path):
    """A3. With prefix_cache_tp on, a text-only r1 whose follower read failed
    leaves no pages in either rank's index: a new request sharing r1's first
    three pages gets no lease, on rank 0 (whose lease the followers take) or
    on the follower. A request sharing r0's prefix does, so the cache is on
    and r0's commit indexed it."""
    run = _run(tmp_path, Scenario(kind="text", prefix_cache=True))
    _healthy_ok(run)
    assert set(run.final) == {0, 1}, "a rank never reported its index: " + run.summary()
    for rank in (0, 1):
        assert run.final[rank]["probe_healthy_prefix"], f"rank {rank}: r0's prefix isn't cached at all"
        assert not run.final[rank]["probe_failed_prefix"], (
            f"rank {rank} would lease the failed row's pages to a new request: "
            f"{run.final[rank]['probe_failed_prefix']}"
        )


def test_a_run_waits_for_a_follower_that_routes_after_rank_0(tmp_path):
    """The harness itself: rank 1 stalls in its poll while its first step runs,
    so rank 0 has routed both requests before rank 1 has. The run still ends
    with both requests routed on both ranks, so A4 compares real outputs on
    every rank, not a follower's missing ones."""
    run = _run(tmp_path, Scenario(kind="text", fail=False, follower_stall_s=1.5))
    _healthy_ok(run)
    for rank in (0, 1):
        assert set(run.routed[rank]) == {"r0", "r1"}, f"rank {rank}: " + run.summary()


@pytest.mark.parametrize("kind", KINDS)
def test_a_request_beside_the_failed_one_is_bit_identical_to_a_run_without_failure(tmp_path, kind):
    """A4 (and A5). r0's outputs (token and hidden state, by bytes) on each
    rank equal the same run's with every read landing."""
    clean = _run(tmp_path / "clean", Scenario(kind=kind, fail=False))
    _healthy_ok(clean)
    assert clean.completed("r1") and not clean.failed, "the run without failure failed: " + clean.summary()
    run = _run(tmp_path / "failed", Scenario(kind=kind))
    _healthy_ok(run)
    for rank in (0, 1):
        assert run.routed[rank].get("r0") == clean.routed[rank].get("r0"), (
            f"rank {rank}: r0's outputs differ from the run without failure"
        )
