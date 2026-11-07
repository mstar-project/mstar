"""What a worker does for steps that stand for several iterations of a Loop:
the ask, why a walk can't repeat, and what a run leaves behind when it lands
(its columns cut into items, its count settled, its tokens streamed).

``StepRepeats`` (``step_repeats.py``) holds the settings and the rules; these
are the ones that need the worker's runtime, engine and streams, kept out of
``worker.py``. ``Worker`` mixes them in.
"""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from mstar.engine.device_run import RUN_COUNTS, RUN_DESCRIPTORS, RUN_REPLAYS, RUN_SKIPPED, TokenStream, run_item
from mstar.engine.resources.resident import NOT_ASKED
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import EMITTED_COUNT, BatchedModelOutput, HostRows
from mstar.streaming.topology import ProducerWalkCtx
from mstar.utils.profiler import phase_record
from mstar.worker.micro_scheduler import CHAIN_LANE
from mstar.worker.step_repeats import RepeatConfig, edge_refusal, loop_back, run_columns

if TYPE_CHECKING:
    from mstar.graph.runtime.base import PostprocessStep
    from mstar.utils.threads import ThreadPool
    from mstar.worker.lanes import Lane
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.speculation import Speculation
    from mstar.worker.worker import PendingBatch

# How often a wait on a run that streams its tokens looks for new ones.
_LANDED_POLL_S = 1e-4
# How often a run's wait scans for ready rows when nothing landed to say so
_QUIET_SCAN_S = 10 * _LANDED_POLL_S


class RepeatingSteps:
    """A ``Worker``'s repeat logic. Reads the worker's own attributes."""

    def _group_repeat_config(self, unit):
        """Only the graph's complete Loop domain has a connected repeat setting.

        This deliberately does not make its closer eligible to repeat alone.
        Unsupported run controls are rejected during worker construction.
        """
        if unit.loop_name is None or self._step_repeats is None:
            return None
        config = self._step_repeats.config(unit.nodes[-1])
        if config.max <= 1:
            return None
        if config != RepeatConfig(max=config.max):
            raise ValueError("connected fixed-count Loops currently accept only step_repeats.max")
        return config

    def _stamp_group_repeats(self, group) -> None:
        if group.unit.streaming_edges:
            if self._step_repeats is None:
                return
            for scheduled, executing in group.members:
                if not executing.request_ids:
                    continue
                key = scheduled.node_name, scheduled.graph_walk
                loop = self._innermost_loop_by_node.get((scheduled.graph_walk, scheduled.node_name))
                if loop is None:
                    continue
                room = self._graph_runtime.loop_room(*key, loop, list(executing.request_ids), 0, False)
                count, _ = self._step_repeats._most(key, lambda room=room: room)
                scheduled.repeats = executing.step_context.repeats = count
            return
        config = self._group_repeat_config(group.unit)
        if config is None:
            return
        walk = group.members[0][0].graph_walk
        engine = self.engine_manager.get_engine(group.unit.nodes[0])
        for i, node in enumerate(group.unit.nodes):
            policies = engine.get_stop_policies(node, walk)
            if (policies is None or not set(policies) <= {group.unit.loop_name}
                    or any(p.reads_token for p in policies.values())
                    or (policies and i + 1 < len(group.unit.nodes))):
                return
            edges = self._step_repeats.edges[node, walk]
            columns = frozenset(signal for signal, dest in edges.items)
            if engine.keep_run_columns(node, walk, columns):
                return
        rows = list(group.members[0][1].request_ids)
        key = group.unit.nodes[-1], walk
        room = self._graph_runtime.loop_room(*key, group.unit.loop_name, rows, 0, False)
        if room <= 0:
            raise ValueError("a claimed connected Loop has no remaining iteration")
        count, _ = self._step_repeats._most(key, lambda: room)
        group.capacity = config.max
        for scheduled, executing in group.members:
            scheduled.repeats = executing.step_context.repeats = count

    def _stamp_repeats(
        self, speculation: Speculation, pending: PendingBatch, in_chain: int, owes_run: bool = False,
    ) -> None:
        """Ask for as many iterations as the step can have; its resources narrow
        it at admit (`_await_admit_settled` takes the answer). ``owes_run``: the
        step is the run ready rows wait for, which needs no turn in the chain."""
        if self._step_repeats is None or not self._step_repeats.asked:
            return
        batch = speculation.scheduled_batch
        speculation.node_batch.step_context.owed_run = owes_run
        repeats, why = self._step_repeats.ceiling(
            speculation, max(in_chain, int(owes_run)),
            lambda: self._graph_runtime.loop_room(
                batch.node_name, batch.graph_walk,
                self._innermost_loop_by_node[(batch.graph_walk, batch.node_name)],
                list(speculation.continuing_rids), pending.batch.repeats,
                self._step_repeats.config(batch.node_name).past_cap,
            ),
        )
        if owes_run and self._step_repeats.owed_length(batch.node_name) < repeats:
            repeats, why = self._step_repeats.owed_length(batch.node_name), "ready_waits"
        if self._ring_signals.get(batch.node_name):
            # a ring caps a run as a loop's room does: its rows have room for this many
            room = self.engine_manager.get_engine(batch.node_name).ring_room(
                batch.node_name, speculation.continuing_rids,
            )
            if (room or 1) < repeats:
                repeats, why = max(1, room or 1), "its ring's room"
        # Retained snapshots bound execution, including zero-real-item tails.
        # Legacy destructive windows instead bound it by their real item count.
        windows = {
            rid: {edge.name: edge for edge in edges if edge.chunk.reservation is not None}
            for rid, edges in speculation.node_batch.stream_windows.items()
        }
        for rid, edges in speculation.consumed_streaming_edges.items():
            for edge in edges:
                if edge.next_node == batch.node_name and (edge.chunk.num_items or edge.chunk.reservation is not None):
                    windows.setdefault(rid, {})[edge.name] = edge
        windows = {rid: list(edges.values()) for rid, edges in windows.items() if edges}
        if windows:
            items = min(
                self.streams.window_bound(rid, edge) if edge.chunk.reservation is not None else edge.chunk.num_items
                for rid, edges in windows.items() for edge in edges
            )
            if items < repeats:
                repeats, why = items, "its stream window"
        speculation.node_batch.stream_windows = windows
        if repeats > 1 and self._walk_drivers:
            steady = self._steady_consumer_walks(
                speculation.scheduled_batch, speculation.node_batch, speculation.continuing_rids, repeats,
            )
            if steady < repeats:
                repeats, why = steady, "its consumer's walk changes"
        speculation.node_batch.step_context.repeats = repeats
        self._step_repeats.note_ask((batch.node_name, batch.graph_walk), repeats, why)

    def _unchained_repeats(self, batch: ScheduledBatch) -> int:
        """The ask for a step the scheduler picked: more than one iteration only
        for a node whose steps never chain (`StepRepeats.unchained_ceiling`), up
        to its Loop's room with nothing of it in flight. What the chain's ask
        caps besides that room, such a walk doesn't repeat with
        (`_unchained_refusal`)."""
        if self._step_repeats is None or not self._step_repeats.asked:
            return 1
        key = (batch.node_name, batch.graph_walk)
        # the runtime's name for the Loop: a Loop with no name of its own has one per build
        loop = self._innermost_loop_by_node.get((batch.graph_walk, batch.node_name))
        if loop is None:
            repeats, why = 1, "the runtime holds no Loop of it here"
        else:
            repeats, why = self._step_repeats.unchained_ceiling(batch, lambda: self._graph_runtime.loop_room(
                batch.node_name, batch.graph_walk, loop, list(batch.request_to_worker_graph), 0,
                self._step_repeats.config(batch.node_name).past_cap,
            ))
        edges = self._step_repeats.edges.get(key)
        if edges is not None and not edges.chained:  # a chained node repeats on its chain's step
            self._step_repeats.note_ask(key, repeats, why)
        return repeats

    def _steady_consumer_walks(self, batch, node_batch, rids, repeats: int) -> int:
        """How far a run may go while every row's streamed items keep their
        consumer walk: a pass stamps one walk on what it sends, so a run
        stops short of the iteration whose items would run another
        (``Connection.consumer_walk``)."""
        key = (batch.node_name, batch.graph_walk)
        drivers = [
            self._walk_drivers[partition] for partition in {
                self._signal_consumer[signal] for signal in self._graph_runtime.get_output_signals(*key)
                if signal in self._signal_consumer
            } if partition in self._walk_drivers
        ]
        if not drivers:
            return repeats
        ctx = ProducerWalkCtx(producer_walk=batch.graph_walk, node=batch.node_name)
        metadata = node_batch.per_request_input_metadata
        loop = self._innermost_loop_by_node.get((batch.graph_walk, batch.node_name))
        for rid in rids:
            meta = metadata.get(rid)
            iters = dict(meta.dynamic_loop_iter_counts) if meta else {}
            k = iters.get(loop, 0)
            ctx.fwd_info = node_batch.per_request_info.get(rid)
            for conn in drivers:
                first = None
                for i in range(repeats):
                    ctx.loop_iters = {**iters, loop: k + i}
                    walk = conn.consumer_walk(ctx)
                    if first is None:
                        first = walk
                    elif walk != first:
                        repeats = i
                        break
        ctx.fwd_info = None
        return max(1, repeats)

    def _unchained_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why a walk whose steps never chain can't repeat, or None: its ask is
        made on a step the scheduler picked, which has only the Loop's room to
        go on (`_unchained_repeats`), not what `_stamp_repeats` also caps."""
        engine = self.engine_manager.get_engine(node_name)
        if node_name in self.parallel_nodes:
            return "deployment: its steps never chain, and only a chain's step carries a TP group's ask"
        if engine.streamed_iteration_inputs(node_name):
            return "node: its steps never chain, and only a chain's step is cut to its stream window"
        if self._ring_signals.get(node_name):
            return "node: its steps never chain, and only a chain's step is cut to its ring's room"
        signals = self._graph_runtime.get_output_signals(node_name, graph_walk)
        if any(self._signal_consumer.get(signal) in self._walk_drivers for signal in signals):
            return "node: its steps never chain, and only a chain's step keeps its consumer's walk steady"
        return None

    def _repeat_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why no step of this walk can stand for several iterations, or None:
        something about its rows has to reach the host every iteration."""
        engine = self.engine_manager.get_engine(node_name)
        if not self._step_repeats.edges[(node_name, graph_walk)].chained:
            refusal = self._unchained_refusal(node_name, graph_walk)
            if refusal is not None:
                return refusal
        if (node_name, graph_walk) not in self._stop_token_signal:
            # a count stop is the loop's cap, which a run never passes (`loop_room`)
            return "node: the runtime doesn't check its stop: its own check_stop decides"
        edges = self._step_repeats.edges[(node_name, graph_walk)]
        refusal = edge_refusal(edges, engine.streamed_iteration_inputs(node_name), node_name in self.parallel_nodes)
        if refusal is None:
            # the client's copy of a token rides the step's token column
            inline = (
                engine.inline_client_signals(node_name, graph_walk) if self._graph_runtime.supports_inline_emit else {}
            )
            refusal = engine.keep_run_columns(node_name, graph_walk, run_columns(edges, inline))
        return refusal if refusal is not None else engine.repeat_refusal(node_name, graph_walk, loop_back(edges))

    def _rows_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why this walk's loop-back values can't stay in the node's device
        rows between its steps, or None: asked where the node's config asks
        for resident planning or repeats."""
        engine = self.engine_manager.get_engine(node_name)
        if self._step_repeats.limit(node_name) == 1 and not engine.resident_asked(node_name):
            return NOT_ASKED
        return engine.rows_refusal(node_name, graph_walk)

    def _capture_run_requests(self):
        """One startup capture map; standalone settings remain unmodified."""
        if self._step_repeats is None:
            return {}
        requests = dict(self._step_repeats.device_stops())
        for unit in self.scheduler.execution_units.values():
            if unit.streaming_edges:
                for node, walk in zip(unit.nodes, unit.graph_walks, strict=True):
                    requests.setdefault((node, walk), self._step_repeats.config(node))
        return requests

    def _ask_run_columns(self, node_name: str, graph_walk: str) -> None:
        engine = self.engine_manager.get_engine(node_name)
        inline = (
            engine.inline_client_signals(node_name, graph_walk) if self._graph_runtime.supports_inline_emit else {}
        )
        edges = self._step_repeats.edges.get((node_name, graph_walk))
        # A member outside a Loop executes at most once, but its parent still
        # records zero/one physical output independently of the other member.
        columns = (self._graph_runtime.get_output_signals(node_name, graph_walk) if edges is None
                   else run_columns(edges, inline))
        engine.keep_run_columns(node_name, graph_walk, columns)

    def _stream_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why this walk's runs don't send their tokens as they land, or None,
        having set them to."""
        if not self._step_repeats.config(node_name).stream_tokens:
            return NOT_ASKED
        if not self._step_repeats.may_repeat(node_name, graph_walk):
            return "repeats: the walk doesn't repeat"
        if not self._graph_runtime.supports_inline_emit:
            return "graph runtime: it has no inline emit"
        if self._is_tp_follower_node(node_name):
            # its landing sends the client nothing either
            return "TP: the leader sends the tokens"
        engine = self.engine_manager.get_engine(node_name)
        client = engine.inline_client_signals(node_name, graph_walk)
        policy = next(
            (p for p in (engine.get_stop_policies(node_name, graph_walk) or {}).values() if p.reads_token), None,
        )
        edges = self._walk_graphs[graph_walk].get_nodes()[node_name].outputs
        signals = list(dict.fromkeys(edge.name for edge in edges))
        signal = next((s for s in client if s in signals), None)
        # only an inline landing can skip what went ahead of it
        if signal is None or policy is None or not self._all_on_device(engine, node_name, graph_walk, signals):
            return "node: its tokens don't ride inline"
        if (node_name, graph_walk) in self._withholding:
            return "node: it withholds an output, so its landing doesn't ride inline"
        buffer = client[signal]
        edge = next(e for e in edges if e.name == signal and e.next_node == EMIT_TO_CLIENT)
        refusal = engine.stream_tokens(node_name, graph_walk, buffer, self._step_repeats.config(node_name).max)
        if refusal is None:
            self._streamed[(node_name, graph_walk)] = (policy, signal, edge.output_modality)
        return refusal

    def _ask_rings(self) -> None:
        """``step_repeats: stream_queue``: each streamed signal of an asking
        node whose consumer is on this worker keeps its items in a ring of the
        node's rows, which the consumer reads its chunks from."""
        self._ring_refusals: dict[tuple[str, str], str | None] = {}
        self._ring_signals: dict[str, frozenset[str]] = {}
        engine = self.engine_manager.get_engine("")
        for node in {node for node, _ in self._step_repeats.edges}:
            queue = self._step_repeats.config(node).stream_queue
            if queue is None:
                continue
            walks_of: dict[str, set[str]] = {}
            for walk, section in self._walk_graphs.items():
                for edge in getattr(section.get_nodes().get(node), "outputs", ()):
                    if edge.is_streaming:
                        walks_of.setdefault(edge.name, set()).add(walk)
            ringed = set()
            for signal, walks in sorted(walks_of.items()):
                why = self._ring_refusal_for(node, signal, queue["depth"], walks)
                if why is None:
                    rows = engine.ask_ring(node, signal, queue["depth"], walks)
                    if isinstance(rows, str):
                        why = rows
                    else:
                        self.streams.rings[signal] = rows
                        ringed.add(signal)
                for walk in walks:
                    self._ring_refusals.setdefault((node, walk), why)
            self._ring_signals[node] = frozenset(ringed)

    def _ring_refusal_for(self, producer: str, signal: str, depth: int, walks: Collection[str]) -> str | None:
        """Why ``signal``'s items can't stay in a ring ``depth`` deep, or None."""
        conn = next((c for c in self.streams.connections if c.edge_name == signal), None)
        consumer = self.streams.consumer_node.get(signal)
        if conn is None or consumer is None:
            return "consumer: it is on another worker (that is `carrier: p2p`)"
        if conn.join_dim is not None:
            return "consumer: it joins a producer's pieces into one item"
        if any(
            node == consumer and self._step_repeats.limit(node) > 1 and edges.streamed
            and not any(
                (node, walk) in zip(unit.nodes, unit.graph_walks, strict=True)
                and (producer, node, signal) in unit.streaming_edges
                and dict(zip(unit.nodes, unit.graph_walks, strict=True))[producer] in walks
                for unit in self.scheduler.execution_units.values()
            )
            for (node, walk), edges in self._step_repeats.edges.items()
        ):
            return "consumer: it reads the stream an item an iteration"
        policy, take = conn.chunk_policy_factory(), 0
        for _ in range(64):
            if not policy.is_ready(1 << 20):
                break
            take = max(take, policy.window_size())
            policy.register_chunk(policy.next_chunk_size(1 << 20))
        if take > depth:
            return f"its chunk policy takes {take} items at once, more than the ring's depth of {depth}"
        return None

    def _ring_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why this walk's streamed items don't stay in a ring on the device, or None."""
        if self._step_repeats.config(node_name).stream_queue is None:
            return NOT_ASKED
        return self._ring_refusals.get((node_name, graph_walk), "graph: the walk streams nothing")

    def _with_ring_room(self, node_name: str, rids: list[int]) -> list[int]:
        """``rids`` but those whose ring is full: such a row leaves the step,
        and is ready again (``LoopbackRows.admit_retrieve``) once its consumer
        frees room."""
        engine = self.engine_manager.get_engine(node_name)
        return [rid for rid in rids if engine.ring_room(node_name, [rid]) > 0]

    def _queue_ring_items(
        self, batch_N: PendingBatch, outputs: BatchedModelOutput, step: PostprocessStep, withheld,
    ) -> None:
        """STORE, for a ringed signal: each row's kept items are positions in
        its ring, which its consumer's buffer takes, not tensors to store and
        route; the runtime counts them toward the producer's done."""
        signals = self._ring_signals.get(batch_N.node_name) if self._ring_signals else None
        bases_of = batch_N.node_batch.ring_bases
        if not signals or not bases_of:
            return
        width = batch_N.batch.repeats
        reserved = batch_N.node_batch.ring_width
        rows = self.engine_manager.get_engine(batch_N.node_name).ring_rows(batch_N.node_name)
        withhold, withheld = withheld
        for signal in signals:
            bases = bases_of.get(signal)
            if not bases:
                continue
            skipped = withheld if signal in withhold else ()
            consumer = self._walk_driven_edges.get(signal)
            routed = set(step.wg_ids.keys)
            for rid, base in bases.items():
                rid_outputs = outputs.get(rid)
                positions = []
                if rid_outputs and signal in rid_outputs and rid in routed:
                    del rid_outputs[signal]
                    positions = [base + i for i in range(step.kept.get(rid, width)) if (rid, i) not in skipped]
                # what the step reserved and the consumer never gets: a run that
                # ended short, a withheld item, a row that routed none
                rows.ring_skip(signal, rid, sorted(set(range(base, base + reserved)) - set(positions)))
                if not positions:
                    continue
                walk = None
                if consumer:
                    req_info = self.request_state.per_request_info[rid]
                    walk = req_info.per_partition_info[batch_N.partition].current_fwd_info.stream_consumer_walks.get(
                        consumer,
                    )
                self.streams.push_positions(rid, signal, positions, walk)
                if positions:
                    self._graph_runtime.add_stream_items(rid, signal, len(positions))

    def _odds_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why this walk's runs don't shorten on their rows' stop odds, or None."""
        if self._step_repeats.config(node_name).eos_odds is None:
            return NOT_ASKED
        if not self._step_repeats.may_repeat(node_name, graph_walk):
            return "repeats: the walk doesn't repeat"
        return self._odds_refusals.get((node_name, graph_walk))

    def _launch_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why this walk's runs stay on the host's launch loop, or None."""
        if not self._step_repeats.may_repeat(node_name, graph_walk):
            return NOT_ASKED
        return self.engine_manager.get_engine(node_name).launch_refusal(node_name, graph_walk)

    def _device_stop_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why this walk's runs don't skip a replay whose rows have all
        stopped, or None: asked by the node's ``device_stop``."""
        if self._step_repeats.config(node_name).device_stop == "off":
            return NOT_ASKED
        if not self._step_repeats.may_repeat(node_name, graph_walk):
            return "repeats: the walk doesn't repeat"
        return self.engine_manager.get_engine(node_name).device_stop_refusal(node_name, graph_walk)

    def _end_run_for_ready_rows(self, step: PendingBatch) -> None:
        """``step``'s run on the device ends at its next iteration, not at its count."""
        members = (((step.node_name, step.node_batch),) if step.group is None or not step.group.submitted else
                   ((scheduled.node_name, executing) for scheduled, executing in step.group.members))
        for node, executing in members:
            if executing.step_context.repeats > 1 and not self._step_repeats.config(node).ready_waits:
                self.engine_manager.get_engine(node).yield_runs(node)

    def _landed_skipped(self, step: PendingBatch) -> bool:
        """Whether ``step``'s run began covered by a yield and ran nothing, landed
        here: asked before anything is speculated from it, since its outputs are
        no outputs. The step before it has landed, so ``run_start`` is due at once.
        Nothing moved on the device, so what admit counted goes back and the rows
        that can retry wait as a refused step's do: the runtime still holds their
        inputs."""
        node_batch = step.node_batch
        if node_batch.run_rows or (step.group is not None and step.group.submitted):
            return False  # logical member closure also handles proved-zero runs
        started = node_batch.run_started
        if started is None or node_batch.step_context.repeats <= 1:
            return False
        while not int(started[0]):
            # in this order, the word again last: the launch tail sets the event after the
            # fence lifts, and a short skipped graph can end between two looks
            if step.future.done():
                event = node_batch.completion_event
                if (event is None or event.query()) and not int(started[0]):
                    return False  # it raised before it launched
            time.sleep(_LANDED_POLL_S)
        if int(started[0]) != RUN_SKIPPED:
            return False
        outputs = step.future.result()
        phase_record(f"repeats.ran.{step.graph_walk}#", 0)
        engine = self.engine_manager.get_engine(step.node_name)
        engine.settle_run(node_batch, 0)
        self._return_unexecuted(step, outputs)
        return True

    def _return_unexecuted(self, step: PendingBatch, outputs: BatchedModelOutput) -> None:
        """Return a proved-zero, already settled receipt without consuming its inputs.

        Both a skipped physical run and a descriptor never accepted by a closed
        run window own ordinary ready work. Neither completes a graph iteration.
        Publication/drain and resource settlement must precede this transition.
        """
        node_batch = step.node_batch
        engine = self.engine_manager.get_engine(step.node_name)
        for signal, bases in node_batch.ring_bases.items():
            # what it reserved holds no room: only postprocess gives it back otherwise
            for rid, base in bases.items():
                engine.ring_rows(step.node_name).ring_skip(signal, rid, range(base, base + node_batch.ring_width))
        self._keep_rows_to_retry(step)
        if node_batch.stream_windows:
            self._wait_for_step(step)
            keep = set(step.batch.request_to_worker_graph) - node_batch.failed_requests.keys()
            self._settle_stream_windows(node_batch, {}, keep)
        self._push_back_batch(step.batch)
        if node_batch.failed_requests:
            self._fail_step_rows(step, outputs)

    def _is_device_run(self, step: PendingBatch) -> bool:
        if step.group is not None and step.group.submitted and step.group.unit.streaming_edges:
            return step.node_batch.completion_event is not None and any(
                executing.step_context.repeats > 1 and not self._step_repeats.config(scheduled.node_name).ready_waits
                for scheduled, executing in step.group.members
            )
        return (
            (step.node_name, step.graph_walk) in self._device_runs
            and step.node_batch.step_context.repeats > 1 and step.node_batch.completion_event is not None
        )

    @staticmethod
    def _run_receipt_rows(step: PendingBatch, rids: list[int]) -> PendingBatch:
        """An output/retry view after the original resource receipt settled once."""
        original = step.node_batch
        scheduled = step.batch._sub_batch(rids)
        scheduled.repeats = max((original.step_context.run_counts[rid] for rid in rids), default=0)
        view = original.select_rows(rids)
        view.step_context.repeats = scheduled.repeats
        return replace(step, batch=scheduled, node_batch=view, joined=[])

    def _finish_live_run(self, active: PendingBatch, gpu_executor: ThreadPool) -> None:
        """Close one capture, settle logical receipts, then route their own outputs.

        Publication is no longer offered by the caller. The execution event and
        publication-stream drain precede count inspection. Capture and resource
        readers outlive every collection/tail copy, including failed tails.
        """
        engine = self.engine_manager.get_engine(active.node_name)
        membership = engine.run_membership(active.node_batch)
        if membership is None:
            raise ValueError("a live run has no captured membership owner")
        receipts = (active, *active.joined)
        self._wait_for_step(active)
        raw = active.node_batch.outputs
        identities = raw.check_stop_buffers[RUN_DESCRIPTORS].cpu().tolist()
        counts = Counter(handle for ids in identities for handle in ids if handle)
        descriptors = {row.handle: row for receipt in receipts for row in receipt.node_batch.run_rows.values()}
        if counts.keys() - descriptors.keys():
            raise ValueError("a captured output has no attached logical receipt")
        stream = membership.publication_stream
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            membership.drain()
        if stream is not None:
            stream.synchronize()
        membership.acknowledge(membership.control.cpu().tolist())

        try:
            # Settle every owner before invoking any model tail or graph route;
            # a failure in the first tail must not leave later H reservations.
            for receipt in receipts:
                batch = receipt.node_batch
                actual = {rid: counts[row.handle] for rid, row in batch.run_rows.items()}
                engine.settle_run(batch, max(actual.values(), default=0), actual)
            for receipt in receipts:
                batch = receipt.node_batch
                actual = batch.step_context.run_counts
                zero = [rid for rid in batch.request_ids if not actual[rid]]
                live = [rid for rid in batch.request_ids if actual[rid]]
                if zero:
                    self._return_unexecuted(self._run_receipt_rows(receipt, zero), BatchedModelOutput())
                if not live:
                    continue
                part = self._run_receipt_rows(receipt, live)
                self._settle_stream_windows(part.node_batch, actual)

                def collect(part=part):
                    batch = part.node_batch
                    output = engine.collect_run_outputs(batch, [(raw, identities)])
                    batch.outputs = output
                    self._stage_host_rows(output, part.lane)
                    if self.device_capability.is_accelerator:
                        batch.completion_event = torch.Event(device=self.device)
                        batch.completion_event.record(torch.accelerator.current_stream(self.device))
                    batch.resource_publish_info = engine.finalize_batch(
                        batch, publish_request_ids=self._publishable_request_ids(batch),
                    )
                    return output

                output = gpu_executor.submit(collect).result()
                # The pinned-host ring may be reused by the next receipt; route
                # this one first, after its own copy event, on the main thread.
                self._wait_for_step(part)
                self._clear_in_flight_flag(part.batch)
                self._finish_step(part, output)
        finally:
            # A tail can raise after queuing copies. Fence the executor stream
            # before releasing any output reader or resetting capture padding.
            def release():
                if self.device_capability.is_accelerator:
                    torch.accelerator.current_stream(self.device).synchronize()
                engine.finish_run_outputs(active.node_batch, BatchedModelOutput())
                for receipt in receipts:
                    batch = receipt.node_batch
                    for descriptor in batch.run_rows.values():
                        if not membership.release(descriptor):
                            raise ValueError("a closed run still has a live descriptor reader")
                        batch.step_context.resident.unpin((descriptor.bindings[0],))
                    batch.run_rows.clear()
            gpu_executor.submit(release).result()

    def _offer_run_row(self, active: PendingBatch, gpu_executor: ThreadPool) -> bool:
        """Take one ordinary ready step and publish it only after owner admission."""
        engine = self.engine_manager.get_engine(active.node_name)
        membership = engine.run_membership(active.node_batch)
        lease = active.node_batch.step_context.slot_lease
        stream = membership.publication_stream
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            for part in (active, *active.joined):
                for rid, descriptor in part.node_batch.run_rows.items():
                    if self._is_tearing_down(rid):
                        membership.cancel(descriptor)
        available = membership.available_rows(lease.bucket.bs)
        if not available or active.node_batch.completion_event.query():
            return False
        target = active.node_name, active.graph_walk
        chain = self.scheduler.pick_chain_step(self.request_state, target, ())
        if not chain.pick.taken:
            return False
        # The existing scheduler owns ordering, capture groups and backlog.
        # One logical receipt reserves one descriptor; unchosen rows stay there.
        pick = chain.pick._replace(
            taken=chain.pick.taken[:1], held_back=chain.pick.held_back | set(chain.pick.taken[1:]),
        )
        scheduled = self.scheduler.take_chain_step(self.request_state, chain._replace(pick=pick))
        if scheduled is None:
            return False
        horizon = active.node_batch.step_context.repeats
        node_batch = self._build_executing_batch(scheduled)
        if (scheduled.graph_walk != active.graph_walk or scheduled.request_walks or scheduled.chunk_ranges
                or node_batch.step_context.repeats < horizon):
            self._push_back_batch(scheduled)
            return False
        node_batch.step_context.repeats = scheduled.repeats = horizon
        self._set_in_flight_flag(scheduled, True)
        receipt = replace(active, batch=scheduled, node_batch=node_batch, future=None, joined=[])
        active.joined.append(receipt)
        held = {rid for part in (active, *active.joined) for rid in part.batch.request_to_worker_graph}
        self._hold_rows(active.lane, held)

        def prepare():
            stream = membership.publication_stream
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                node_batch.rows_only = self._all_on_device(
                    engine, scheduled.node_name, scheduled.graph_walk, scheduled.output_signals,
                )
                engine.prepare_inputs(node_batch)
                try:
                    descriptor = engine.admit_run_row(node_batch, active.node_batch, available[0])
                except ValueError:
                    if node_batch.committed is not None:
                        raise
                    descriptor = None  # this request needs an ordinary step
                if stream is not None:
                    node_batch.completion_event = torch.cuda.Event()
                    node_batch.completion_event.record(stream)
                return descriptor

        receipt.future = gpu_executor.submit(prepare)
        descriptor = receipt.future.result()
        if descriptor is None:
            active.joined.remove(receipt)
            self._return_unexecuted(receipt, BatchedModelOutput())
            self._hold_rows(active.lane, held - set(scheduled.request_to_worker_graph))
            return False
        # Graph in-flight state, request holds and the committed H-step are all
        # attached before the publication stream can expose this descriptor.
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            membership.publish_row(descriptor)
        self._count_turn(receipt)
        return True

    def _await_run(
        self, batch_N: PendingBatch, spec_pending: PendingBatch | None, lanes: Sequence[Lane], gpu_executor: ThreadPool,
    ) -> bool:
        """Serve messages, input readiness and lanes until the run lands.

        Joining captures offer compatible ordinary receipts at boundaries and
        close them here; return True once their outputs have been routed.
        Other runs retain the existing fairness/yield and ordinary landing.
        """
        live_run = bool(batch_N.node_batch.run_rows)
        if live_run and spec_pending is not None:
            raise ValueError("a joining capture must close before its successor is submitted")
        rows = set(batch_N.batch.request_to_worker_graph)
        if spec_pending is not None:
            # the loop's admit fence, for what follows: N+1's launch needs nothing of N's postprocess
            self._await_admit(spec_pending)
            rows.update(spec_pending.batch.request_to_worker_graph)
        self._hold_rows(CHAIN_LANE, rows)
        engine = self.engine_manager.get_engine(batch_N.node_name)
        has_ready = self.scheduler.has_ready_excluding
        state, chain = self.request_state, (batch_N.node_name, batch_N.graph_walk)
        waiting = has_ready(state, None)
        yielded = False

        def end_for_ready_rows() -> None:
            nonlocal yielded
            # Work ready already follows the chain's peek (``MSTAR_SPEC_PEEK_FOR_FAIRNESS``). With it
            # off, the first row to turn ready on an idle wait still ends the run, of any walk,
            # and a row of this walk waits while other work does.
            if not yielded and (
                (self._spec_peek and has_ready(state, chain)) or (not waiting and has_ready(state, None))
            ):
                yielded = True
                if batch_N.group is None or not batch_N.group.submitted:
                    engine.yield_runs(batch_N.node_name)
                else:
                    self._end_run_for_ready_rows(batch_N)

        scanned = time.monotonic()

        def serve() -> None:
            # Mostly a row turns ready as one of these lands, and a run's wait is mostly none
            # of them. Not only: a KV copy from the host tier or an OOM hold running out show
            # only to a scan, so one runs every _QUIET_SCAN_S whatever landed.
            nonlocal scanned
            arrived = self._process_messages()
            arrived |= self.wakeup_event.drain()
            arrived |= any([lane.finish() for lane in lanes])
            now = time.monotonic()
            if not arrived and now - scanned < _QUIET_SCAN_S:
                return
            scanned = now
            self._check_ready_tensors()
            self._poll_stream_buffers()
            for lane in lanes:
                lane.launch(gpu_executor)
            if live_run:
                self._offer_run_row(batch_N, gpu_executor)
            end_for_ready_rows()

        if live_run:
            self._offer_run_row(batch_N, gpu_executor)
        end_for_ready_rows()
        node_batch = batch_N.node_batch
        if node_batch.token_stream is not None:
            if live_run:
                raise ValueError("live descriptor token streaming is not configured")
            self._stream_tokens(batch_N, serve)
            return False
        while not node_batch.completion_event.query():
            serve()
            time.sleep(_LANDED_POLL_S)
        if live_run:
            self._finish_live_run(batch_N, gpu_executor)
        return live_run

    def _stream_tokens(self, batch_N: PendingBatch, serve: Callable[[], None] | None = None) -> None:
        """While a run is in flight, send the client what it has landed, and
        ``serve`` between looks. The step's rows are those still live: a row
        that stopped, failed or drained after it was built sends nothing."""
        node_batch = batch_N.node_batch
        stream: TokenStream = node_batch.token_stream
        walk = batch_N.graph_walk
        policy, signal, modality = self._streamed[(batch_N.node_name, walk)]
        infos = node_batch.per_request_info_wrapped
        loop = self._innermost_loop_by_node.get((walk, batch_N.node_name))
        live = list(node_batch.request_ids)
        # the runtime's counts: the step before this one has landed, and a
        # speculated step's own counts move by one however far a run went
        counts, caps = self._graph_runtime.loop_progress(
            batch_N.node_name, live, [walk] * len(live), [loop] * len(live),
        )
        rows = {
            rid: (policy.limits(infos[rid]), count, cap)
            for rid, count, cap in zip(live, counts, caps, strict=True) if cap >= 0
        }

        def decide(rid: int, column: int, token: int) -> tuple[bool, bool]:
            if rid not in rows:
                return False, True
            limits, count, cap = rows[rid]
            return True, count + column + 1 >= cap or policy.stops(count + column, token, limits)

        while True:
            finished = node_batch.completion_event.query()
            rids, values = stream.take(decide)
            if rids:
                self._graph_runtime.send_tokens(rids, values, signal, modality)
            if finished:
                return
            if serve is not None:
                serve()
            time.sleep(_LANDED_POLL_S)

    def _settle_run(self, batch_N: PendingBatch, host_rows: "HostRows | None", engine) -> None:
        """A run on the device tells how many iterations it ran: from here on
        the step stood for those. Its columns past them were never written."""
        replays = None if host_rows is None else host_rows.buffers.pop(RUN_REPLAYS, None)
        if replays is None:
            return
        ran = int(replays[0])
        # what a run came to; ``repeats.narrowed.*`` counts only what admit cut
        phase_record(f"repeats.ran.{batch_N.graph_walk}#", ran)
        width = batch_N.batch.repeats
        if ran < width:
            # before the prefix chains read them: nothing past ``ran`` was produced
            for name, buffer in host_rows.buffers.items():
                if buffer.dim() > 1 and buffer.shape[1] == width:
                    host_rows.buffers[name] = buffer[:, :ran]
        executions = host_rows.buffers.pop(RUN_COUNTS)
        counts = dict(zip(host_rows.request_ids, executions[:len(host_rows.request_ids)].tolist(), strict=True))
        engine.settle_run(batch_N.node_batch, ran, counts)
        self._settle_stream_windows(batch_N.node_batch, counts)
        batch_N.batch.repeats = ran

    def _settle_stream_windows(
        self, node_batch, counts: Mapping[int, int], keep=frozenset(), *,
        transitions: Mapping[tuple[int, str], tuple[int, int]] | None = None,
    ) -> None:
        """Retire source inputs only after their reader completed. A skipped
        or refused reader pushed back with its graph inputs keeps its snapshot.
        Graph cleanup independently releases the reader's tensor references.
        """
        for rid, edges in node_batch.stream_windows.items():
            if rid not in keep:
                for edge in edges:
                    reservation = edge.chunk.reservation
                    if reservation is not None:
                        if reservation.decisions:
                            if transitions is None or (rid, edge.name) not in transitions:
                                raise ValueError("a connected stream needs its actual semantic receipt")
                            pops, final = transitions[rid, edge.name]
                            self.streams.settle_window(rid, edge, pops, final_transition=final)
                        else:
                            self.streams.settle_window(rid, edge, counts.get(rid, 0))
        node_batch.stream_windows = {
            rid: edges for rid, edges in node_batch.stream_windows.items() if rid in keep
        }

    def _cut_run_items(
        self, batch_N: PendingBatch, outputs: BatchedModelOutput, step: PostprocessStep,
        host_rows: "HostRows | None", cpu_outputs, engine,
    ) -> None:
        """Each row's per-iteration outputs as the items it routes: a step that
        repeated holds a column per replay of each (``Engine.keep_run_columns``),
        of which a row routes the iterations its stop kept (``step.kept``);
        an iteration whose token is a stop id routes none of the withheld
        signals (``_withheld_items``); and a replay that emitted a count
        (``EMITTED_COUNT``) routes that many of each, along dim 1."""
        width = batch_N.batch.repeats
        # the step's columns, a replay each, as it ran them: a run on the device
        # that ended early settled ``width`` to what it ran, down to 1
        columns = outputs.run_columns
        withhold, withheld = (
            self._withheld_items(batch_N, host_rows, cpu_outputs) if self._withholding else ((), frozenset())
        )
        if not withheld:
            withhold = ()
        emitted = None if host_rows is None else host_rows.buffers.get(EMITTED_COUNT)
        if not columns and not withhold and emitted is None:
            return
        if not columns and emitted is None:
            # one iteration a row: a withheld row just has no item, and the batch's
            # row views still describe the rest
            for rid, _ in withheld:
                for name in withhold:
                    outputs.get(rid, {}).pop(name, None)
            return
        signals = set(withhold) | columns
        if emitted is not None:
            # a step that emits a count splits its outputs, run or not
            signals |= engine.run_columns(batch_N.node_name, batch_N.graph_walk)
        row_of = {} if emitted is None else {rid: i for i, rid in enumerate(host_rows.request_ids)}
        emitted = None if emitted is None else emitted.reshape(emitted.shape[0], -1).tolist()
        # a spanned output's span per replay, [rows, repeats, 2]
        spans = outputs.host_spans or outputs.output_spans or {}
        span_of = {
            name: dict(zip(outputs.span_request_ids, span.tolist(), strict=False))
            for name, span in spans.items() if span.dim() == 3
        }
        copies = None
        for rid in step.wg_ids.keys:
            rid_outputs = outputs.get(rid)
            if not rid_outputs:
                continue
            counts = None if emitted is None else emitted[row_of[rid]]
            kept = step.kept.get(rid, width)
            for name in signals & rid_outputs.keys():
                (column,) = rid_outputs[name]
                items = []
                for i in range(kept):
                    if name in withhold and (rid, i) in withheld:
                        continue
                    if copies is None:
                        copies = self._run_copies(column)
                    replay = run_item(column, i, copies) if name in columns else column
                    if name in span_of:
                        start, end = span_of[name][rid][i]
                        if end <= start:
                            continue
                        replay = replay[..., start:end]
                    if counts is None:
                        items.append(replay)
                    else:
                        items.extend(replay[:, j] for j in range(counts[i]))
                if items:
                    rid_outputs[name] = items
                else:
                    del rid_outputs[name]
        if copies is not None:
            # stored for readers on other streams (``skip_cuda_sync``): done before they are
            copies.synchronize()
        # the entries are no longer row views of the step's batch clones
        outputs.row_views = None

    def _run_copies(self, like: torch.Tensor) -> "torch.cuda.Stream | None":
        """The side stream a run's item copies go on, which the host waits for
        alone: what they copy belongs to a step that has landed, and they are
        handed to readers on other streams (the API server's transport, the
        next step) that nothing orders after the main thread's own stream."""
        if like.device.type != "cuda":
            return None
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=like.device)
        return self._copy_stream
