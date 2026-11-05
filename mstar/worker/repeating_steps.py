"""What a worker does for steps that stand for several iterations of a Loop:
the ask, why a walk can't repeat, and what a run leaves behind when it lands
(its columns cut into items, its count settled, its tokens streamed).

``StepRepeats`` (``step_repeats.py``) holds the settings and the rules; these
are the ones that need the worker's runtime, engine and streams, kept out of
``worker.py``. ``Worker`` mixes them in.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from mstar.engine.device_run import RUN_REPLAYS, TokenStream, run_item
from mstar.engine.resources.resident import NOT_ASKED
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.submodule_base import EMITTED_COUNT, BatchedModelOutput, HostRows
from mstar.streaming.topology import ProducerWalkCtx
from mstar.worker.step_repeats import edge_refusal, loop_back, run_columns

if TYPE_CHECKING:
    from mstar.graph.runtime.base import PostprocessStep
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.speculation import Speculation
    from mstar.worker.worker import PendingBatch

logger = logging.getLogger(__name__)

# How often a wait on a run that streams its tokens looks for new ones.
_LANDED_POLL_S = 1e-4
# a TP step's settled count waits this many later seqs for the other side's
# (``_settle_tp_repeats``): a chain's last step never gets the leader's
_TP_REPEATS_KEPT = 256


class RepeatingSteps:
    """A ``Worker``'s repeat logic. Reads the worker's own attributes."""

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
        repeats = self._step_repeats.ceiling(
            speculation, max(in_chain, int(owes_run)),
            lambda: self._graph_runtime.loop_room(
                batch.node_name, batch.graph_walk, speculation.loop_name,
                list(speculation.continuing_rids), pending.batch.repeats,
                self._step_repeats.config(batch.node_name).past_cap,
            ),
        )
        if owes_run:
            repeats = min(repeats, self._step_repeats.owed_length(batch.node_name))
        if self._ring_signals.get(batch.node_name):
            # a ring caps a run as a loop's room does: its rows have room for this many
            room = self.engine_manager.get_engine(batch.node_name).ring_room(
                batch.node_name, speculation.continuing_rids,
            )
            repeats = max(1, min(repeats, room or 1))
        # a stream window: as many iterations as every row has items for, and
        # what a row took past the step's own count goes back after its admit
        windows = {
            rid: [e for e in edges if e.chunk.num_items and e.next_node == batch.node_name]
            for rid, edges in speculation.consumed_streaming_edges.items()
        }
        windows = {rid: edges for rid, edges in windows.items() if edges}
        if windows:
            repeats = min(repeats, *(e.chunk.num_items for edges in windows.values() for e in edges))
            speculation.node_batch.stream_windows = windows
        if repeats > 1 and self._walk_drivers:
            repeats = self._steady_consumer_walks(speculation, repeats)
        speculation.node_batch.step_context.repeats = repeats

    def _unchained_repeats(self, batch: ScheduledBatch) -> int:
        """The ask for a step the scheduler picked: more than one iteration only
        for a node whose steps never chain (`StepRepeats.unchained_ceiling`), up
        to its Loop's room with nothing of it in flight. What the chain's ask
        caps besides that room, such a walk doesn't repeat with
        (`_unchained_refusal`)."""
        if self._step_repeats is None or not self._step_repeats.asked:
            return 1
        key = (batch.node_name, batch.graph_walk)
        edges = self._step_repeats.edges.get(key)
        if edges is None:
            return 1
        return self._step_repeats.unchained_ceiling(batch, lambda: self._graph_runtime.loop_room(
            batch.node_name, batch.graph_walk, edges.loop, list(batch.request_to_worker_graph), 0,
            self._step_repeats.config(batch.node_name).past_cap,
        ))

    def _steady_consumer_walks(self, speculation: Speculation, repeats: int) -> int:
        """How far a run may go while every row's streamed items keep their
        consumer walk: a pass stamps one walk on what it sends, so a run
        stops short of the iteration whose items would run another
        (``Connection.consumer_walk``)."""
        batch = speculation.scheduled_batch
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
        metadata = speculation.node_batch.per_request_input_metadata
        loop = speculation.loop_name
        for rid in speculation.continuing_rids:
            meta = metadata.get(rid)
            iters = dict(meta.dynamic_loop_iter_counts) if meta else {}
            k = iters.get(loop, 0)
            ctx.fwd_info = speculation.node_batch.per_request_info.get(rid)
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

    def _ask_run_columns(self, node_name: str, graph_walk: str) -> None:
        engine = self.engine_manager.get_engine(node_name)
        inline = (
            engine.inline_client_signals(node_name, graph_walk) if self._graph_runtime.supports_inline_emit else {}
        )
        edges = self._step_repeats.edges[(node_name, graph_walk)]
        engine.keep_run_columns(node_name, graph_walk, run_columns(edges, inline))

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
                why = self._ring_refusal_for(signal, queue["depth"])
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

    def _ring_refusal_for(self, signal: str, depth: int) -> str | None:
        """Why ``signal``'s items can't stay in a ring ``depth`` deep, or None."""
        conn = next((c for c in self.streams.connections if c.edge_name == signal), None)
        consumer = self.streams.consumer_node.get(signal)
        if conn is None or consumer is None:
            return "consumer: it is on another worker (that is `carrier: p2p`)"
        if conn.join_dim is not None:
            return "consumer: it joins a producer's pieces into one item"
        if any(
            node == consumer and self._step_repeats.limit(node) > 1
            and edges.streamed
            for (node, _), edges in self._step_repeats.edges.items()
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

    def _settle_tp_repeats(self, seq: int, *, follower: int = 0, leader: int = 0) -> None:
        """A TP step's settled iterations, as this follower settled them or as
        the leader says it did (on the head after it), whichever comes second
        checks the other. Every rank's resources narrow alike; if one didn't,
        the ranks would run different counts of the same collectives."""
        mine, theirs = self._tp_repeats
        if follower:
            mine[seq] = follower
        if leader:
            theirs[seq] = leader
        if seq in mine and seq in theirs:
            if mine[seq] != theirs[seq]:
                raise SystemExit(
                    f"Worker {self.worker_id}: TP step {seq} settled at {mine.pop(seq)} iterations here and "
                    f"{theirs.pop(seq)} on the leader; the ranks would split"
                )
            del mine[seq], theirs[seq]
        for record in (mine, theirs):
            if len(record) > _TP_REPEATS_KEPT:
                for old in [k for k in record if k < seq - _TP_REPEATS_KEPT]:
                    del record[old]

    def _await_run(self, batch_N: PendingBatch) -> None:
        """Wait out a run on the device, ending it at its next iteration once
        a new request arrives: the yield the step's speculation asks for comes
        before any arrival during the run. What arrived is held for
        ``_process_messages``, in order."""
        event = batch_N.node_batch.completion_event
        while not event.query():
            if self._yields_on_arrival(batch_N):
                return
            time.sleep(_LANDED_POLL_S)

    def _stream_tokens(self, batch_N: PendingBatch) -> None:
        """While a run is in flight, send the client what it has landed. The
        step's rows are those still live: a row that stopped, failed or
        drained after it was built sends nothing."""
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

        # a run on the device still ends early for a new request (``_await_run``)
        watch = bool(self._device_runs) and (batch_N.node_name, walk) in self._device_runs
        while True:
            finished = node_batch.completion_event.query()
            rids, values = stream.take(decide)
            if rids:
                self._graph_runtime.send_tokens(rids, values, signal, modality)
            if finished:
                return
            if watch and self._yields_on_arrival(batch_N):
                watch = False
            time.sleep(_LANDED_POLL_S)

    def _settle_run(self, batch_N: PendingBatch, host_rows: "HostRows | None", engine) -> None:
        """A run on the device tells how many iterations it ran: from here on
        the step stood for those. Its columns past them were never written."""
        replays = None if host_rows is None else host_rows.buffers.pop(RUN_REPLAYS, None)
        if replays is None:
            return
        ran = int(replays[0])
        width = batch_N.batch.repeats
        if ran < width:
            # before the prefix chains read them: nothing past ``ran`` was produced
            for name, buffer in host_rows.buffers.items():
                if buffer.dim() > 1 and buffer.shape[1] == width:
                    host_rows.buffers[name] = buffer[:, :ran]
            engine.end_run_early(batch_N.node_batch, ran)
            batch_N.batch.repeats = ran

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
