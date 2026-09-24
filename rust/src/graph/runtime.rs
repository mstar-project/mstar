//! The batched surface. One `GraphRuntime` per worker owns every request's
//! walk state; Python calls it once per forward pass, not once per request.

use crate::graph::frames;
use crate::graph::compile::{EMIT_TO_CLIENT, EMPTY_DESTINATION, LoopArg, NodeArg, compile_one};
use crate::graph::shard::{FanoutDest, GroupTemplate, ShardingTemplate};
use crate::graph::spec::*;
use crate::graph::request::{LoopStopTime, RequestInfo, WgIndex, WorkerGraphMeta};
use crate::graph::state::{RequestState, RoutedEdge, SpecNode, TensorRef};
use crate::tensors::{SharedBookkeeping, TensorBookkeeping};
use crate::communicator::RawZmqCommunicator;
use crate::PyZmqCommunicator;
use std::sync::Arc;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};


/// Python's `NestedLoopIndices`.
#[pyclass]
#[derive(Clone)]
pub struct NestedLoopIdx {
    #[pyo3(get)] pub loop_name_order: Vec<String>,
    #[pyo3(get)] pub loop_indices: Vec<(String, u32)>,
    #[pyo3(get)] pub wg_fwd_pass_idx: u32,
}


/// rid string <-> worker-local handle. Handles are RECYCLED, so everything
/// keyed by one must be purged when it is freed, or the next request to get
/// that integer inherits the leftovers.
struct InternedRids {
    rid_names: Vec<Option<String>>,
    rid_to_handle: FxHashMap<String, u32>,
    free: Vec<u32>,
}

impl InternedRids {
    fn new() -> Self {
        Self {
            rid_names: vec![],
            rid_to_handle: FxHashMap::default(),
            free: vec![],
        }
    }

    /// True in the second slot when the handle is newly allocated rather than
    /// recycled, i.e. the caller has to grow its per-handle vectors.
    fn intern(&mut self, rid: &str) -> (u32, bool) {
        if let Some(&h) = self.rid_to_handle.get(rid) {
            return (h, false);
        }
        let (h, grew) = match self.free.pop() {
            Some(h) => {
                self.rid_names[h as usize] = Some(rid.to_string());
                (h, false)
            }
            None => {
                self.rid_names.push(Some(rid.to_string()));
                ((self.rid_names.len() - 1) as u32, true)
            }
        };
        self.rid_to_handle.insert(rid.to_string(), h);
        (h, grew)
    }

    fn handle(&self, rid: &str) -> Option<u32> {
        self.rid_to_handle.get(rid).copied()
    }

    fn name(&self, handle: u32) -> Option<&str> {
        self.rid_names.get(handle as usize)?.as_deref()
    }

    fn release(&mut self, handle: u32) -> bool {
        let Some(slot) = self.rid_names.get_mut(handle as usize) else {
            return false;
        };
        let Some(name) = slot.take() else {
            return false; // already removed; removal is idempotent by design
        };
        self.rid_to_handle.remove(&name);
        self.free.push(handle);
        true
    }

    fn len(&self) -> usize {
        self.rid_names.len()
    }
}


struct AsyncEnabledChecker {
    async_scheduling_enabled_nodes: FxHashSet<u32>,
    parallel_nodes: FxHashSet<u32>,
    tp_async_nodes: FxHashSet<u32>,
    parallel_leader_nodes: FxHashSet<u32>,
}

impl AsyncEnabledChecker{
    fn new(
        async_scheduling_enabled_nodes: FxHashSet<u32>
    ) -> Self {
        Self {
            async_scheduling_enabled_nodes,
            parallel_nodes: FxHashSet::default(),
            tp_async_nodes: FxHashSet::default(),
            parallel_leader_nodes: FxHashSet::default(),
        }
    }

    fn update_metadata(
        &mut self,
        parallel_nodes: FxHashSet<u32>,
        tp_async_nodes: FxHashSet<u32>,
        parallel_leader_nodes: FxHashSet<u32>,
    ) {
        self.parallel_nodes = parallel_nodes;
        self.tp_async_nodes = tp_async_nodes;
        self.parallel_leader_nodes = parallel_leader_nodes;
    }

    fn can_speculate(
        &self, curr_node: u32, new_node: u32
    ) -> bool {
        self.async_scheduling_enabled_nodes.contains(&curr_node) 
            & (!self.parallel_nodes.contains(&new_node) 
                || (
                    self.tp_async_nodes.contains(&new_node)
                        && self.parallel_leader_nodes.contains(&new_node)
                        && curr_node == new_node
                )
            )
    }
}

/// Every worker graph on one worker, plus every request's walk state in each.
#[pyclass]
pub struct GraphRuntime {
    interner: StrToId,
    graphs: Vec<GraphRef>,
    wg_ids: Vec<u32>,
    shard: ShardingTemplate,

    /// `[wg][handle]`; None where the request is not registered with that
    /// worker graph (Python only adds a request to the graphs its partition
    /// uses).
    states: Vec<Vec<Option<RequestState>>>,
    rids: InternedRids,
    /// `[handle]`, parallel to `states`' inner dimension.
    requests: Vec<Option<RequestInfo>>,
    async_checker: AsyncEnabledChecker,

    /// Every worker graph in the deployment, local and remote. The remote ones
    /// answer "who owns this node" when an output leaves this worker.
    all_worker_graphs: Vec<WorkerGraphMeta>,
    /// walk -> the LOCAL worker graphs active in it. Static, so a request only
    /// stores the slice its partition selected.
    walk_to_local_wgs: FxHashMap<Sym, Vec<WgIndex>>,

    /// (walk, node) -> local worker graph. Built once; last write wins, as in
    /// the Python inverted index.
    node_owner: FxHashMap<(Sym, Sym), WgIndex>,
    /// Loop stops from this iteration's check_stop; cleared every iteration.
    pending_loop_stops: FxHashSet<(u32, Sym, Sym)>,
    /// Routing parked between complete_and_route_batch and send_outputs.
    completions: FxHashMap<u64, Completion>,
    completion_counter: u64,

    /// A share of the SAME bookkeeper Python handed TensorStore, taken at
    /// construction. Routing reads descriptors and adjusts refcounts through
    /// it, so a copy would diverge from what the store believes.
    bookkeeping: SharedBookkeeping,

    /// A share of the SAME transport the worker's communicator owns, so the
    /// frames this runtime decides on can be sent from here rather than
    /// handed back to Python one at a time. None where the worker built the
    /// pyzmq communicator, which has no shareable object.
    communicator: Option<Arc<RawZmqCommunicator>>,
}

 impl GraphRuntime {
    /// Does THIS worker run `node` for this request?
    ///
    /// Compiling the node is not enough: under data parallelism the conductor
    /// picks a replica per worker-graph group, so a worker can hold the graph
    /// while this particular request's copy of that node lives on a peer.
    /// Python asks the same question by popping its own id out of the fanout.
    fn runs_for_request(&self, rid: u32, node: Sym) -> bool {
        let me = self.shard.me;
        self.info(rid).is_some_and(|i| {
            i.node_to_workers
                .iter()
                .any(|(&(n, _), workers)| n == node && workers.contains(&me))
        })
    }

    /// The request id string behind a handle.
    fn rid_name(&self, rid: u32) -> PyResult<String> {
        self.rids
            .name(rid)
            .map(|s| s.to_string())
            .ok_or_else(|| PyValueError::new_err("unknown rid handle"))
    }

    /// Python's `WorkerGraphIO.get_nested_loop_idxs_for_node`, interned: the
    /// loop context a node runs in, as (order, indices, wg_fwd_pass_idx).
    fn nested_idxs_interned(
        &self, wg: WgIndex, rid: u32, node: NodeId,
    ) -> Option<(Vec<Sym>, Vec<(Sym, u32)>, u32)> {
        let state = self.state(wg, rid)?;
        let g = self.g(wg);
        let fwd = state.num_times_run;
        // A node outside every loop has no context, only the pass index.
        let Some(lid) = g.node(node).loop_id else {
            return Some((vec![], vec![], fwd));
        };
        let order = g.loop_order(lid).into_iter().map(|l| g.lp(l).name).collect();
        // Every loop in the graph, as Python's get_loop_indices returns.
        let indices = g
            .loops
            .iter()
            .enumerate()
            .map(|(i, lp)| (lp.name, state.loop_iter(i as LoopId)))
            .collect();
        Some((order, indices, fwd))
    }

    /// Descriptors for a set of uuids. A uuid whose descriptor is already gone
    /// is skipped -- a tensor can be collected before the frame naming it goes
    /// out.
    fn infos(&self, uuids: &[u64]) -> Vec<crate::tensors::TensorPointerInfo> {
        let bk = self.bookkeeping.lock().unwrap();
        uuids.iter().filter_map(|u| bk.info_raw(*u).cloned()).collect()
    }

    /// The same descriptors, cut down to what THIS destination should see.
    ///
    /// The bookkeeper holds the whole tensor; the fanout decided each
    /// destination's slice of it and recorded it on the ref. Python does this
    /// in `fanout_graph_edges`, which clones the info and overwrites exactly
    /// these three fields -- so an unsharded edge, whose ref still spans the
    /// whole thing, comes out unchanged.
    fn sliced_infos(&self, refs: &[TensorRef]) -> Vec<crate::tensors::TensorPointerInfo> {
        let bk = self.bookkeeping.lock().unwrap();
        Self::sliced_infos_locked(&bk, refs)
    }

    /// The same, for a caller that already holds the guard.
    fn sliced_infos_locked(
        bk: &crate::tensors::Bookkeeping, refs: &[TensorRef],
    ) -> Vec<crate::tensors::TensorPointerInfo> {
        refs.iter()
            .filter_map(|r| {
                let mut i = bk.info_raw(r.uuid).cloned()?;
                if let Some(d) = i.dims.first_mut() {
                    *d = r.dim0;
                }
                i.nbytes = r.nbytes;
                i.offset = r.offset;
                Some(i)
            })
            .collect()
    }

    /// Send, or drop on the floor when this runtime has no transport. Silence
    /// is right for the no-transport case: the frame builders are exercised
    /// without a socket, and the worker's flag gate already refuses to pair
    /// the Rust runtime with a communicator it cannot share.
    fn dispatch(&self, peer: &str, bytes: &[u8]) -> PyResult<()> {
        if let Some(comm) = &self.communicator {
            comm.send(peer, bytes)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// One WORKER_GRAPHS_DONE frame. Drains the request's pending persist
    /// signals, token counts and emitted signal names -- they accumulate
    /// across sends and flush onto the next completion, so a persist signal
    /// cannot race the message announcing it.
    ///
    /// `output_loop_indices` is NOT drained: the conductor keeps it for the
    /// life of the request, and Python re-sends it on every WGD.
    fn worker_graphs_done_frame(
        &mut self,
        rid: u32,
        worker_graph_ids: &[u32],
        is_first_tp_rank: bool,
        partition_name: &str,
        stream_tokens_consumed: &[(String, i64)],
        request_info_encoded: Option<&[u8]>,
        profiling_encoded: Option<&[u8]>,
        speculative: bool,
    ) -> PyResult<Vec<u8>> {
        let request_id = self.rid_name(rid)?;
        // `and not speculative`, as Python has it: a speculatively-scheduled
        // node has not really finished the partition, so reporting it done
        // here would tell the conductor the stream ended a pass early.
        let partition_done = !speculative
            && self
                .interner
                .get(partition_name)
                .and_then(|p| self.info(rid).map(|i| i.stream_done(p)))
                .unwrap_or(false);

        let (persist, new_tokens, output_signals, loop_indices) =
            match self.requests[rid as usize].as_mut() {
                Some(info) => {
                    let (p, n, o) = info.pending.take_for_send();
                    (p, n, o, info.pending.output_loop_indices.clone())
                }
                None => (vec![], vec![], vec![], vec![]),
            };
        let persist_signals: Vec<(Sym, Vec<crate::tensors::TensorPointerInfo>)> =
            persist.into_iter().map(|(sig, uuids)| (sig, self.infos(&uuids))).collect();

        let frame = frames::WorkerGraphsDone {
            request_id: &request_id,
            worker_graph_ids,
            is_first_tp_rank,
            persist_signals,
            new_token_counts: &new_tokens,
            output_signal_names: output_signals,
            partition_name,
            partition_done,
            stream_tokens_consumed,
            output_loop_indices: loop_indices,
            resource_publish_info: frames::spliced_field(
                request_info_encoded, "resource_publish_info",
            ),
            profiling: frames::split_profiling(profiling_encoded),
        };
        let bk = self.bookkeeping.lock().unwrap();
        Ok(frame.encode(&self.interner, bk.strings()))
    }

    /// This request's loop stop observations, interned. What STOP_LOOPS
    /// carries so a peer can tell a newer stop from a duplicate.
    fn loop_stop_times_of(
        &self, rid: u32,
    ) -> Vec<(Sym, (Vec<Sym>, Vec<(Sym, u32)>, u32))> {
        let Some(info) = self.info(rid) else { return vec![] };
        info.loop_stop_times
            .iter()
            .map(|(&name, t)| {
                (
                    name,
                    (
                        t.loop_name_order.clone(),
                        t.loop_indices.iter().map(|(&n, &i)| (n, i)).collect(),
                        t.wg_fwd_pass_idx,
                    ),
                )
            })
            .collect()
    }

    fn g(&self, wg: u32) -> &GraphRef {
        &self.graphs[wg as usize]
    }
    fn nid(&self, wg: u32, name: &str) -> Option<NodeId> {
        self.interner.get(name).and_then(|s| self.g(wg).by_name.get(&s).copied())
    }
    /// Python passes the deployment-wide worker graph id; `states` and
    /// `graphs` are indexed by this worker's local position.
    fn wg_index(&self, wg_id: u32) -> Option<WgIndex> {
        self.wg_ids.iter().position(|&id| id == wg_id).map(|i| i as WgIndex)
    }

    /// Offer one signal to each live worker graph until one claims it.
    ///
    /// A claim loop, not a lookup: a node can refuse a signal it owns when the
    /// name does not match an input or both ready slots are already full.
    fn ingest_one(
        &mut self,
        rid: u32,
        spec: &EdgeSpecArg,
        can_buffer: bool,
        is_streaming: bool,
    ) -> bool {
        let (Some(dest), Some(signal)) = (
            self.interner.get(&spec.next_node),
            self.interner.get(&spec.signal),
        ) else {
            return false; // a name this worker never compiled
        };
        let Some(info) = self.requests.get(rid as usize).and_then(|r| r.as_ref())
        else {
            return false; // never admitted here, or already removed
        };
        // Resolved once: the descriptors are the same whichever worker graph
        // claims the signal.
        let tensors: Vec<TensorRef> = {
            let bk = self.bookkeeping.lock().unwrap();
            spec.uuids.iter().map(|&u| bk.tensor_ref(u)).collect()
        };

        let live: Vec<WgIndex> = info
            .partitions
            .values()
            .flat_map(|p| p.walk_worker_graphs.iter().copied())
            .collect();
        for wg in live {
            let Some(node) = self.graphs[wg as usize].by_name.get(&dest).copied()
            else {
                continue;
            };
            let Some(slot) = self.graphs[wg as usize].node(node).slot_of(signal)
            else {
                continue; // the node does not take this input
            };
            let Some(state) = self.state_mut(wg, rid) else {
                continue;
            };
            // The streaming gate is re-checked per signal on purpose:
            // ingesting one can be what makes the next node eligible.
            if is_streaming && !state.is_ready_for_streaming(node) {
                continue;
            }
            if state.ingest(
                node, slot, tensors.clone(), can_buffer,
                spec.is_final_streaming_chunk,
            ) {
                return true;
            }
        }
        false
    }

    /// Walk every (node, walk, rid) whose graph inputs are satisfied, calling
    /// `f` for each. `f` returns false to stop -- that is what lets the peek
    /// bail on the first match instead of building the whole list.
    fn scan_ready(
        &self,
        exclude_rids: &FxHashSet<u32>,
        target: Option<&(String, String)>,
        exclude_target: Option<&(String, String)>,
        mut f: impl FnMut(Sym, Sym, u32) -> bool,
    ) {
        let sym_of = |s: &String| self.interner.get(s);
        let target = target.map(|(n, w)| (sym_of(n), sym_of(w)));
        let exclude_target = exclude_target.map(|(n, w)| (sym_of(n), sym_of(w)));

        for (rid, slot) in self.requests.iter().enumerate() {
            let rid = rid as u32;
            let Some(info) = slot.as_ref() else { continue };
            if exclude_rids.contains(&rid) {
                continue;
            }
            for part in info.partitions.values() {
                for &wg in &part.walk_worker_graphs {
                    let Some(state) = self.state(wg, rid) else {
                        continue;
                    };
                    let g = &self.graphs[wg as usize];
                    for node in 0..g.nodes.len() as NodeId {
                        if !state.is_ready(node) {
                            continue;
                        }
                        let name = g.node(node).name;
                        let walk = part.graph_walk;
                        if let Some((tn, tw)) = target {
                            if tn != Some(name) || tw != Some(walk) {
                                continue;
                            }
                        }
                        if let Some((xn, xw)) = exclude_target {
                            if xn == Some(name) && xw == Some(walk) {
                                continue;
                            }
                        }
                        if !f(name, walk, rid) {
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Run the source node's outputs through the speculative slots and report
    /// which destinations become ready. Mutates and then clears those slots,
    /// exactly as Python's ingest_for_speculation / clear_speculative_inputs
    /// pair does.
    fn spec_targets(
        &mut self, wg: WgIndex, source: NodeId, rid: u32,
    ) -> Option<Vec<SpecNode>> {
        let g = self.graphs[wg as usize].clone();
        if g.node(source).outputs.is_empty() {
            return None; // nothing to feed a spec target
        }
        // Structure only: readiness turns on WHICH slots fill, not on the
        // tensors, and the real outputs do not exist until the step lands.
        let edges: Vec<RoutedEdge> = g
            .node(source)
            .outputs
            .iter()
            .map(|e| RoutedEdge {
                name: e.name,
                dest: e.dest,
                dest_sym: e.dest_sym,
                persist: e.persist,
                new_token: e.new_token,
                streaming: e.streaming,
                modality: e.modality,
                tensors: vec![],
                persist_for_loop: false,
                declined_local: false,
                worker: None,
            })
            .collect();
        let state = self.state_mut(wg, rid)?;
        let out = state.ingest_for_speculation(source, &edges);
        state.clear_speculative_inputs();
        Some(out)
    }

    /// The target, plus its output edge names.
    ///
    /// The names ride along because a speculated batch never goes through
    /// pop_rids: without them the caller has to come back and ask for them
    /// once per forward pass, on the hottest path there is.
    fn spec_output(
        &self, g: &GraphRef, graph_walk: &str, sn: &SpecNode,
    ) -> (String, String, bool, Option<String>, Vec<String>) {
        // Sorted and deduped, as Python's `sorted({edge.name for ...})`:
        // one node can carry the same signal to two destinations.
        let mut signals: Vec<String> = g
            .node(sn.node)
            .outputs
            .iter()
            .map(|e| self.interner.name(e.name).to_string())
            .collect();
        signals.sort();
        signals.dedup();
        (
            self.interner.name(g.node(sn.node).name).to_string(),
            graph_walk.to_string(),
            sn.is_new_loop_iter,
            sn.loop_id
                .map(|lid| self.interner.name(g.lp(lid).name).to_string()),
            signals,
        )
    }

    fn prep_spec(
        &mut self, input: SpecPrepArg, follower: bool,
    ) -> PyResult<SpecPrepOut> {
        if input.rids.len() != input.streaming_edges_per_rid.len() {
            return Err(PyValueError::new_err(
                "prep_spec_rids: rids and streaming_edges_per_rid must match",
            ));
        }
        let Some(wg) = self.owner_of(&input.spec_node_name, &input.graph_walk)
        else {
            return Ok(SpecPrepOut::default());
        };
        let (Some(spec_node), Some(curr_node)) = (
            self.nid(wg, &input.spec_node_name),
            self.nid(wg, &input.curr_node_name),
        ) else {
            return Ok(SpecPrepOut::default());
        };
        let wg_id = self.wg_ids[wg as usize];
        let same_node = spec_node == curr_node;

        // Loop context from ONE rid, applied to the batch: which loop a node
        // belongs to and whether the target is a loop-back are structural.
        let loop_ctx = if follower {
            None
        } else {
            input.rids.first().and_then(|&r| {
                self.spec_targets(wg, curr_node, r)?
                    .into_iter()
                    .find(|sn| sn.node == spec_node)
            })
        };

        let mut out = SpecPrepOut::default();
        let mut undo: Vec<(u32, Vec<(u8, bool)>)> = Vec::new();
        let mut cursor = 0usize;

        for (i, &rid) in input.rids.iter().enumerate() {
            let count = input.streaming_edges_per_rid[i];
            let slice = cursor..cursor + count;
            cursor += count;

            if !follower {
                if let Some(ctx) = &loop_ctx {
                    if !self.can_continue_loop(wg, rid, ctx, &input.graph_walk) {
                        continue; // the loop has ended; nothing more to run
                    }
                }
                if let Some(room) = input.room_for_continuing {
                    if out.ready_rids.len() as u32 >= room {
                        // Room is spoken for by the backlog. Skipped BEFORE
                        // any ingest, so there is nothing to roll back.
                        continue;
                    }
                }
            }

            match self.prep_one(
                wg, rid, spec_node, curr_node, same_node,
                &input.streaming_edges[slice.clone()], slice.start,
            ) {
                Some((kept, edges, ingested)) => {
                    undo.push((rid, ingested));
                    out.consumed_streaming_edge_idxs.extend(kept);
                    out.ready_rids.push(rid);
                    out.wg_ids.push(wg_id);
                    out.input_edges_per_rid.push(edges.len());
                    out.input_edges.extend(edges);
                }
                None if follower => {
                    // Undo every rid prepped so far, or this rank joins the
                    // collective with a batch the leader never sent.
                    for (done_rid, slots) in undo {
                        self.undo_spec_ingest(wg, done_rid, spec_node, &slots);
                    }
                    return Ok(SpecPrepOut {
                        all_or_nothing_failed: true,
                        ..Default::default()
                    });
                }
                None => {}
            }
        }
        Ok(out)
    }

    /// False once this rid's loop has ended: a stop is already pending, or the
    /// next iteration would be past the last.
    fn can_continue_loop(
        &self, wg: WgIndex, rid: u32, ctx: &SpecNode, graph_walk: &str,
    ) -> bool {
        if !ctx.is_new_loop_iter {
            return true;
        }
        let Some(lid) = ctx.loop_id else { return true };
        let g = &self.graphs[wg as usize];
        let loop_name = g.lp(lid).name;
        if let Some(w) = self.interner.get(graph_walk) {
            if self.pending_loop_stops.contains(&(rid, w, loop_name)) {
                return false;
            }
        }
        let Some(state) = self.state(wg, rid) else {
            return false;
        };
        state.loop_iter(lid) + 1 < g.lp(lid).max_iters && !state.loop_finished(lid)
    }

    /// One rid: ingest its chunks, check readiness, gather inputs. Returns
    /// (consumed indices, input edges, what was ingested) or None after
    /// rolling its own back.
    #[allow(clippy::type_complexity)]
    fn prep_one(
        &mut self,
        wg: WgIndex,
        rid: u32,
        spec_node: NodeId,
        curr_node: NodeId,
        same_node: bool,
        edges: &[EdgeSpecArg],
        base_idx: usize,
    ) -> Option<(Vec<usize>, Vec<(String, String, Vec<u64>, bool)>, Vec<(u8, bool)>)> {
        let g = self.graphs[wg as usize].clone();
        let tensors_of = |spec: &EdgeSpecArg| -> Vec<TensorRef> {
            let bk = self.bookkeeping.lock().unwrap();
            spec.uuids.iter().map(|&u| bk.tensor_ref(u)).collect()
        };
        let resolved: Vec<(usize, u8, Vec<TensorRef>, bool)> = edges
            .iter()
            .enumerate()
            .filter_map(|(k, e)| {
                let signal = self.interner.get(&e.signal)?;
                let slot = g.node(spec_node).slot_of(signal)?;
                Some((base_idx + k, slot, tensors_of(e), e.is_final_streaming_chunk))
            })
            .collect();

        let state = self.state_mut(wg, rid)?;
        // Held True across the ingest so a streaming input cannot re-add the
        // node to the ready queue underneath us.
        state.set_spec_scheduled(spec_node, true);

        // ingest reports success without saying WHICH slot it used, so the
        // slot is inferred by peeking first -- the rollback removes from the
        // two separately.
        let mut kept = Vec::new();
        let mut ingested = Vec::new();
        for (idx, slot, tensors, final_chunk) in resolved {
            let already = state.has_input(spec_node, slot, false);
            if state.ingest(spec_node, slot, tensors, same_node, final_chunk) {
                kept.push(idx);
                ingested.push((slot, already));
            }
        }

        let source_edges: Vec<RoutedEdge> = g
            .node(curr_node)
            .outputs
            .iter()
            .map(|e| RoutedEdge {
                name: e.name, dest: e.dest, 
                dest_sym: e.dest_sym, persist: e.persist,
                new_token: e.new_token, streaming: e.streaming,
                modality: e.modality, tensors: vec![],
                persist_for_loop: false, declined_local: false,
                worker: None,
            })
            .collect();
        state.ingest_for_speculation(curr_node, &source_edges);
        let ready = state.ready_for_speculation(spec_node, same_node, false);
        state.clear_speculative_inputs();
        state.set_spec_scheduled(spec_node, false); // reset if the rid drops

        if !ready {
            for &(slot, from_next) in &ingested {
                state.remove_input(spec_node, slot, from_next);
            }
            return None;
        }

        let edges_out = state
            .input_tensors(spec_node, same_node)
            .into_iter()
            .map(|(name, tensors, final_chunk)| {
                (name, tensors.iter().map(|t| t.uuid).collect(), final_chunk)
            })
            .collect::<Vec<(Sym, Vec<u64>, bool)>>();
        let node_name = self.interner.name(g.node(spec_node).name).to_string();
        let edges_out = edges_out
            .into_iter()
            .map(|(name, uuids, final_chunk)| {
                (
                    self.interner.name(name).to_string(),
                    node_name.clone(),
                    uuids,
                    final_chunk,
                )
            })
            .collect();
        Some((kept, edges_out, ingested))
    }

    fn undo_spec_ingest(
        &mut self, wg: WgIndex, rid: u32, node: NodeId, slots: &[(u8, bool)],
    ) {
        if let Some(state) = self.state_mut(wg, rid) {
            for &(slot, from_next) in slots {
                state.remove_input(node, slot, from_next);
            }
        }
    }

    /// Whether this request's current walk actually contains the loop. A stop
    /// for one it does not is a model bug: logged by the caller and dropped.
    fn check_dyn_loop(&self, rid: u32, partition: &str, loop_name: &str) -> bool {
        let (Some(p), Some(l)) = (
            self.interner.get(partition),
            self.interner.get(loop_name),
        ) else {
            return false;
        };
        let Some(info) = self.info(rid) else { return false };
        let Some(walk) = info.walk(p) else { return false };
        info.dyn_loop_to_workers.contains_key(&(l, walk))
    }

    fn dyn_loop_workers(&self, rid: u32, partition: &str, name: &str) -> Vec<Sym> {
        let (Some(p), Some(l)) = (
            self.interner.get(partition),
            self.interner.get(name),
        ) else {
            return vec![];
        };
        let Some(info) = self.info(rid) else { return vec![] };
        let Some(walk) = info.walk(p) else { return vec![] };
        info.dyn_loop_to_workers
            .get(&(l, walk))
            .cloned()
            .unwrap_or_default()
    }

    /// Register the finish signal on every worker graph carrying a named loop,
    /// and snapshot the stop time from the one owning the last-run node.
    fn stop_loops_for_rid(
        &mut self,
        rid: u32,
        partition: &str,
        loop_names: &[String],
        last_node_run: Option<&str>,
    ) {
        let Some(p) = self.interner.get(partition) else { return };
        let live: Vec<WgIndex> = match self.info(rid).and_then(|i| i.partitions.get(&p)) {
            Some(part) => part.walk_worker_graphs.clone(),
            None => return,
        };
        let syms: Vec<Sym> = loop_names
            .iter()
            .filter_map(|n| self.interner.get(n))
            .collect();

        // In disaggregated mode one loop name can live on several worker
        // graphs, each with its own finish signal, so this still fans out.
        for &wg in &live {
            let g = self.graphs[wg as usize].clone();
            for &sym in &syms {
                let Some(&lid) = g.loop_by_name.get(&sym) else { continue };
                if let Some(state) = self.state_mut(wg, rid) {
                    state.register_loop_finish(lid);
                }
            }
        }

        // A stop time is one observation per loop, so only the worker graph
        // owning the last-run node is asked.
        let Some(node_name) = last_node_run else { return };
        let walk = match self.info(rid).and_then(|i| i.walk(p)) {
            Some(w) => w,
            None => return,
        };
        let walk_name = self.interner.name(walk).to_string();
        let Some(owner) = self.owner_of(node_name, &walk_name) else { return };
        let g = self.graphs[owner as usize].clone();
        for &sym in &syms {
            let Some(&lid) = g.loop_by_name.get(&sym) else { continue };
            let Some(state) = self.state(owner, rid) else {
                continue;
            };
            let order: Vec<Sym> = g
                .loop_order(lid)
                .into_iter()
                .map(|l| g.lp(l).name)
                .collect();
            let indices: FxHashMap<Sym, u32> = g
                .loop_order(lid)
                .into_iter()
                .map(|l| (g.lp(l).name, state.loop_iter(l)))
                .collect();
            let stop = LoopStopTime {
                loop_name_order: order,
                loop_indices: indices,
                wg_fwd_pass_idx: state.num_times_run,
            };
            if let Some(info) = self.requests[rid as usize].as_mut() {
                info.loop_stop_times.insert(sym, stop);
            }
        }
    }

    fn owner_of(&self, node: &str, walk: &str) -> Option<WgIndex> {
        let n = self.interner.get(node)?;
        let w = self.interner.get(walk)?;
        self.node_owner.get(&(w, n)).copied()
    }

    fn live_wgs(&self, walk: Sym) -> Vec<WgIndex> {
        self.walk_to_local_wgs.get(&walk).cloned().unwrap_or_default()
    }

    /// The walk's local worker graphs, restricted to the ones THIS REQUEST is
    /// registered with -- Python's `[wg for wg in info.worker_graph_ids if
    /// walk in ...]`.
    ///
    /// Walk-wide is nearly the same thing, because a graph the request is not
    /// in has no state and gets skipped. Not everywhere, though:
    /// get_dynamic_loop_iters reads the list directly, so with two partitions
    /// of one request sharing a walk on this worker, one partition would
    /// report the other's loop counters.
    fn live_wgs_for(&self, rid: u32, walk: Sym) -> Vec<WgIndex> {
        let all = self.live_wgs(walk);
        match self.info(rid) {
            Some(info) => all
                .into_iter()
                .filter(|w| info.worker_graphs.contains(w))
                .collect(),
            None => all,
        }
    }

    /// Bounds-checked. A handle can legitimately outlive its request -- a
    /// message for a rid this rank already removed is a benign race -- and an
    /// unchecked index would panic ACROSS the FFI boundary.
    fn state(&self, wg: WgIndex, rid: u32) -> Option<&RequestState> {
        self.states.get(wg as usize)?.get(rid as usize)?.as_ref()
    }

    fn state_mut(&mut self, wg: WgIndex, rid: u32) -> Option<&mut RequestState> {
        self.states.get_mut(wg as usize)?.get_mut(rid as usize)?.as_mut()
    }

    fn info(&self, rid: u32) -> Option<&RequestInfo> {
        self.requests.get(rid as usize)?.as_ref()
    }

    /// The destination as Python names it, so edges compare directly.
    fn dest_name(&self, wg: u32, d: Dest) -> String {
        match d {
            Dest::Local(n) => self.interner.name(self.g(wg).node(n).name).to_string(),
            Dest::External(s) => self.interner.name(s).to_string(),
            Dest::EmitToClient => EMIT_TO_CLIENT.to_string(),
            Dest::Empty => EMPTY_DESTINATION.to_string(),
        }
    }
}

/// `PopRidsOutput`. Edges are flat and rid-major, `input_edges_per_rid[i]`
/// belonging to `rids[i]`.
#[pyclass]
#[derive(Default)]
pub struct PopRidsOut {
    #[pyo3(get)] pub rids: Vec<u32>,
    #[pyo3(get)] pub wg_ids: Vec<u32>,
    /// (signal, next_node, uuids, is_final_streaming_chunk)
    #[pyo3(get)] pub input_edges: Vec<(String, String, Vec<u64>, bool)>,
    #[pyo3(get)] pub input_edges_per_rid: Vec<usize>,
    /// The node's output edge names, sorted and deduped. Reported by the POP
    /// so the caller does not cross back once per forward pass just to ask.
    #[pyo3(get)] pub output_signals: Vec<String>,
}

/// One edge, bound for one worker. Post-fanout, so a signal read by several
/// workers is several of these.
pub struct WireEdge {
    pub rid: u32,
    /// Interned, not a String: this is a grouping key and a dispatch target,
    /// and `take_send_plan` builds one of these per edge per destination.
    pub worker: Sym,
    pub signal: String,
    pub next_node: String,
    /// The REFS, not bare uuids: the fanout may have sliced them for this
    /// destination, and the bookkeeper still holds the whole tensor. Looking
    /// the descriptor up by uuid alone would put the unsliced dims back on
    /// the wire.
    pub tensors: Vec<TensorRef>,
    pub streaming: bool,
    pub shard_dim: Option<u32>,
    pub total_fanin: u32,
}

/// What a completion has to send. Grouped by kind rather than by message so
/// the caller builds one frame per (worker, rid) instead of per edge.
#[pyclass]
#[derive(Default)]
pub struct SendPlan {
    #[pyo3(get)] pub partition: String,
    pub to_workers: Vec<WireEdge>,
    /// (rid, signal, uuids). Python's `to_conductor` is built BEFORE the
    /// fanout, so a persist signal reports the whole tensor.
    #[pyo3(get)] pub persist: Vec<(u32, String, Vec<u64>)>,
    /// (rid, signal, modality, tensors) -- sliced, as to_workers is.
    pub emit: Vec<(u32, String, String, Vec<TensorRef>)>,
    /// (rid, finished worker graph ids, is_first_tp_rank)
    #[pyo3(get)] pub completed: Vec<(u32, Vec<u32>, bool)>,
    /// The pre-completion loop context, snapshotted at route time.
    pub nested: FxHashMap<u32, (Vec<Sym>, Vec<(Sym, u32)>, u32)>,
}

/// One `NestedLoopIndices` as Python hands it over.
#[derive(FromPyObject)]
pub struct LoopStopArg {
    #[pyo3(item)] loop_name_order: Vec<String>,
    #[pyo3(item)] loop_indices: Vec<(String, u32)>,
    #[pyo3(item)] wg_fwd_pass_idx: u32,
}

/// `RouteInput`.
#[derive(FromPyObject)]
pub struct RouteArg {
    #[pyo3(item)] partition: String,
    #[pyo3(item)] graph_walk: String,
    #[pyo3(item)] node_name: String,
    #[pyo3(item)] output_signals: Vec<String>,
    #[pyo3(item)] rids: Vec<u32>,
    #[pyo3(item)] tensors: Vec<u64>,
    #[pyo3(item)] num_tensors: Vec<usize>,
}

/// `RouteOutput`.
#[pyclass]
#[derive(Default)]
pub struct RouteOut {
    #[pyo3(get)] pub completion_id: u64,
    #[pyo3(get)] pub register_tensor_idxs: Vec<usize>,
    #[pyo3(get)] pub register_rids: Vec<u32>,
    #[pyo3(get)] pub new_token_output_idxs: Vec<usize>,
    #[pyo3(get)] pub local_streaming_tensor_idxs: Vec<usize>,
    /// Consumed inputs the completion itself dropped to zero, with each one's
    /// `mem_registered` flag -- empty in the normal order, where
    /// `cleanup_consumed_inputs` has already taken them.
    #[pyo3(get)] pub freed_input_uuids: Vec<u64>,
    #[pyo3(get)] pub freed_input_registered: Vec<bool>,
    /// Rids this completion will actually build a frame for. The caller skips
    /// preparing `per_request_info` for everyone else -- that payload is
    /// re-encoded every pass (it is mutated in place, so it cannot be
    /// cached), and inside a loop on a single-worker deployment neither an
    /// INPUT_SIGNALS nor a WORKER_GRAPHS_DONE goes out on most passes.
    #[pyo3(get)] pub rids_needing_request_info: Vec<u32>,
}

/// Routing parked between complete_and_route_batch and send_outputs.
pub struct Completion {
    pub partition: String,
    pub graph_walk: String,
    pub node_name: String,
    pub wg: WgIndex,
    pub routing: FxHashMap<u32, Vec<RoutedEdge>>,
    /// Per rid: the persist signals, taken BEFORE the fanout, as Python's
    /// `to_conductor` is. Persist is orthogonal to routing: every rank reports
    /// its own copy, unsliced, whatever the fanout decided. Kept apart because
    /// a persist edge bound for EMPTY_DESTINATION has no sharding group, so on
    /// a rank other than 0 the replicated fanout drops it -- and with it the
    /// conductor's only record that this rank produced the signal.
    pub persist: FxHashMap<u32, Vec<(Sym, Vec<TensorRef>)>>,
    pub completed_wgs: FxHashMap<u32, Vec<u32>>,
    /// Per rid: is this rank the TP group's rank 0 for the completed node?
    /// Computed at completion, not at send: the conductor counts one report
    /// per request, so every rank claiming rank 0 multiplies it by tp_size.
    pub first_tp_rank: FxHashMap<u32, bool>,
    /// Per rid: was the completed node SPECULATIVELY scheduled? Captured
    /// before `complete`, which clears the flag. A speculative batch has not
    /// really finished the partition, so it must not report it done.
    pub speculative: FxHashMap<u32, bool>,
    /// Per rid: the loop context as it stood BEFORE this completion, which is
    /// what the outgoing frames report. Captured here because `complete` and
    /// `stop_loops` both advance loop state, so it cannot be re-derived at
    /// send time -- and captured in RUST because the caller would otherwise
    /// have to ask for it per rid, convert it to strings on the way out, and
    /// hand it back to be re-interned.
    pub nested: FxHashMap<u32, (Vec<Sym>, Vec<(Sym, u32)>, u32)>,
}

/// `SpeculationPrepInput`.
#[derive(FromPyObject)]
pub struct SpecPrepArg {
    #[pyo3(item)] spec_node_name: String,
    #[pyo3(item)] curr_node_name: String,
    #[pyo3(item)] graph_walk: String,
    #[pyo3(item)] rids: Vec<u32>,
    #[pyo3(item)] room_for_continuing: Option<u32>,
    #[pyo3(item)] streaming_edges: Vec<EdgeSpecArg>,
    #[pyo3(item)] streaming_edges_per_rid: Vec<usize>,
}

/// `SpeculationPrepOutput`.
#[pyclass]
#[derive(Default)]
pub struct SpecPrepOut {
    #[pyo3(get)] pub consumed_streaming_edge_idxs: Vec<usize>,
    #[pyo3(get)] pub ready_rids: Vec<u32>,
    #[pyo3(get)] pub wg_ids: Vec<u32>,
    /// (signal, next_node, uuids, is_final_streaming_chunk)
    #[pyo3(get)] pub input_edges: Vec<(String, String, Vec<u64>, bool)>,
    #[pyo3(get)] pub input_edges_per_rid: Vec<usize>,
    /// Follower path only: the batch could not be built and was rolled back.
    all_or_nothing_failed: bool,
}

/// Python's `EdgeSpec`: what crosses for one arriving signal.
#[derive(FromPyObject)]
pub struct EdgeSpecArg {
    #[pyo3(item)] signal: String,
    #[pyo3(item)] next_node: String,
    #[pyo3(item)] uuids: Vec<u64>,
    #[pyo3(item)] is_final_streaming_chunk: bool,
}

/// One `ShardingGroup` as configured.
#[derive(FromPyObject)]
pub struct ShardingGroupArg {
    #[pyo3(item)] nodes: Vec<String>,
    #[pyo3(item)] tp_size: u32,
    /// None = every graph walk.
    #[pyo3(item)] graph_walks: Option<Vec<String>>,
    /// This worker's rank in the group; the conductor sets it per worker.
    #[pyo3(item)] tp_rank: Option<u32>,
}

/// Everything `ShardingConfig` carries.
#[derive(FromPyObject)]
pub struct ShardingArg {
    #[pyo3(item)] groups: Vec<ShardingGroupArg>,
    /// signal -> shard dim; None means replicated, same as absent.
    #[pyo3(item)] shard_dim: Vec<(String, Option<u32>)>,
    #[pyo3(item)] tp_enabled_nodes: Vec<String>,
    #[pyo3(item)] sp_enabled_nodes: Vec<String>,
}

/// A worker graph owned by another worker: enough to route to it, no spec.
#[derive(FromPyObject)]
pub struct RemoteWorkerGraphArg {
    #[pyo3(item)] wg_id: u32,
    #[pyo3(item)] graph_walks: Vec<String>,
    #[pyo3(item)] nodes: Vec<String>,
    #[pyo3(item)] dyn_loops: Vec<String>,
}

#[derive(FromPyObject)]
pub struct WorkerGraphArg {
    #[pyo3(item)] wg_id: u32,
    /// Which graph walks this worker graph is active in — routing uses it to
    /// tell a destination owned by a SIBLING worker graph on this worker from
    /// one that leaves the worker entirely.
    #[pyo3(item)] graph_walks: Vec<String>,
    #[pyo3(item)] nodes: Vec<NodeArg>,
    #[pyo3(item)] loops: Vec<LoopArg>,
    // Leadership is NOT here: it is per worker, not per worker graph, and
    // arrives via set_node_metadata's parallel_leader_nodes.
}

#[pymethods]
impl GraphRuntime {
    #[new]
    #[pyo3(signature = (
        worker_graphs, remote_worker_graphs, sharding, bookkeeping, me,
        communicator = None,
    ))]
    fn new(
        worker_graphs: Vec<WorkerGraphArg>,
        // Owned by other workers; needed to answer "who runs this node" when
        // an output leaves this worker.
        remote_worker_graphs: Vec<RemoteWorkerGraphArg>,
        sharding: ShardingArg,
        bookkeeping: PyRef<'_, TensorBookkeeping>,
        me: String,
        communicator: Option<PyRef<'_, PyZmqCommunicator>>,
    ) -> PyResult<Self> {
        let mut it = StrToId::default();
        let mut graphs = Vec::with_capacity(worker_graphs.len());
        let mut wg_ids = Vec::with_capacity(worker_graphs.len());
        let mut node_owner: FxHashMap<(Sym, Sym), u32> = FxHashMap::default();

        let mut async_enabled: FxHashSet<u32> = FxHashSet::default();

        // Compile graphs
        for (wg, wga) in worker_graphs.iter().enumerate() {
            let graph = compile_one(&mut it, &wga.nodes, &wga.loops)?;
            for walk in &wga.graph_walks {
                let walk_sym = it.intern(walk);
                for n in &graph.nodes {
                    // Last write wins, matching the Python inverted index in
                    // WorkerGraphsManager.__post_init__.
                    node_owner.insert((walk_sym, n.name), wg as u32);

                    if n.async_enabled {
                        async_enabled.insert(n.name);
                    }
                }
            }
            wg_ids.push(wga.wg_id.clone());
            graphs.push(graph);
        }

        let me_sym = it.intern(&me);

        // Sharding TEMPLATE. The per-request ShardMap comes from
        // `instantiate`, because a data-parallel replica puts the same node on
        // different workers -- the binding is not deployment-wide.
        let shard = ShardingTemplate {
            groups: sharding.groups.iter().map(|g| GroupTemplate {
                nodes: g.nodes.iter().map(|n| it.intern(n)).collect(),
                tp_size: g.tp_size,
                graph_walks: g.graph_walks.as_ref().map(
                    |ws| ws.iter().map(|w| it.intern(w)).collect()
                ),
                tp_rank: g.tp_rank,
            }).collect(),
            // A None dim reads exactly like an absent key, so drop it here
            // rather than carrying an Option nothing distinguishes.
            shard_dim: sharding.shard_dim.iter()
                .filter_map(|(sig, dim)| dim.map(|d| (it.intern(sig), d)))
                .collect(),
            tp_enabled_nodes: sharding.tp_enabled_nodes.iter()
                .map(|n| it.intern(n)).collect(),
            sp_enabled_nodes: sharding.sp_enabled_nodes.iter()
                .map(|n| it.intern(n)).collect(),
            me: me_sym,
        };

        let states: Vec<Vec<Option<RequestState>>> =
            (0..graphs.len()).map(|_| Vec::new()).collect();

        // Local worker graph metadata. Remote ones arrive via
        // `set_remote_worker_graphs`, which the worker calls with the
        // conductor's all_worker_graph_ids_to_* maps.
        let mut all_worker_graphs = Vec::with_capacity(worker_graphs.len());
        let mut walk_to_local_wgs: FxHashMap<Sym, Vec<WgIndex>> = FxHashMap::default();
        for (idx, wga) in worker_graphs.iter().enumerate() {
            let walks: Vec<Sym> = wga.graph_walks.iter().map(|w| it.intern(w)).collect();
            for &walk in &walks {
                walk_to_local_wgs.entry(walk).or_default().push(idx as WgIndex);
            }
            all_worker_graphs.push(WorkerGraphMeta {
                wg_id: wga.wg_id,
                graph_walks: walks,
                nodes: graphs[idx].nodes.iter().map(|n| n.name).collect(),
                dyn_loops: graphs[idx].loops.iter().map(|l| l.name).collect(),
                local: Some(idx as WgIndex),
            });
        }

        let known: FxHashSet<u32> =
            all_worker_graphs.iter().map(|m| m.wg_id).collect();
        for wga in &remote_worker_graphs {
            if known.contains(&wga.wg_id) {
                continue; // ours; already compiled above
            }
            all_worker_graphs.push(WorkerGraphMeta {
                wg_id: wga.wg_id,
                graph_walks: wga.graph_walks.iter().map(|w| it.intern(w)).collect(),
                nodes: wga.nodes.iter().map(|n| it.intern(n)).collect(),
                dyn_loops: wga.dyn_loops.iter().map(|l| it.intern(l)).collect(),
                local: None,
            });
        }

        Ok(Self{
            interner: it,
            graphs,
            wg_ids,
            shard,
            states,
            rids: InternedRids::new(),
            requests: Vec::new(),
            async_checker: AsyncEnabledChecker::new(async_enabled),
            all_worker_graphs,
            walk_to_local_wgs,
            node_owner,
            pending_loop_stops: FxHashSet::default(),
            completions: FxHashMap::default(),
            completion_counter: 0,
            bookkeeping: bookkeeping.share(),
            communicator: communicator.map(|c| c.share()),
        })
    }

    /// The three sets are node NAMES. A name this worker does not host is
    /// dropped: it cannot be a speculation target here.
    fn set_node_metadata(
        &mut self,
        parallel_nodes: Vec<String>,
        parallel_leader_nodes: Vec<String>,
        tp_async_nodes: Vec<String>,
    ) {
        let syms = |it: &StrToId, names: &[String]| -> FxHashSet<Sym> {
            names.iter().filter_map(|n| it.get(n)).collect()
        };
        let parallel = syms(&self.interner, &parallel_nodes);
        let leaders = syms(&self.interner, &parallel_leader_nodes);
        let tp_async = syms(&self.interner, &tp_async_nodes);
        self.async_checker.update_metadata(parallel, tp_async, leaders);
    }

    /// One call per partition, as the conductor sends one NewRequest each.
    ///
    /// `worker_graph_workers` is flattened; `workers_per_worker_graph` gives
    /// each worker graph's slice, parallel to `worker_graph_ids`.
    fn add_request(
        &mut self,
        request_id: String,
        partition: String,
        graph_walk: String,
        worker_graph_ids: Vec<u32>,
        worker_graph_workers: Vec<String>,
        workers_per_worker_graph: Vec<usize>,
    ) -> PyResult<u32> {
        let (handle, grew) = self.rids.intern(&request_id);
        if grew {
            for per_wg in self.states.iter_mut() {
                per_wg.push(None);
            }
            self.requests.push(None);
        }

        let partition_sym = self.interner.intern(&partition);
        let walk_sym = self.interner.intern(&graph_walk);
        let workers: Vec<Sym> = worker_graph_workers
            .iter()
            .map(|w| self.interner.intern(w))
            .collect();

        let is_new = self.requests[handle as usize].is_none();
        if is_new {
            // The conductor sends the same worker_graph -> workers map on
            // every NewRequest for a request, so this only has to run once.
            let mut info = RequestInfo::default();
            let mut cursor = 0usize;
            for (i, &wg_id) in worker_graph_ids.iter().enumerate() {
                let count = workers_per_worker_graph.get(i).copied().unwrap_or(0);
                let slice = &workers[cursor..cursor + count];
                cursor += count;
                let Some(meta) =
                    self.all_worker_graphs.iter().find(|m| m.wg_id == wg_id)
                else {
                    continue; // a worker graph this deployment never declared
                };
                for &walk in &meta.graph_walks {
                    for &node in &meta.nodes {
                        info.node_to_workers.insert((node, walk), slice.to_vec());
                    }
                    for &lp in &meta.dyn_loops {
                        info.dyn_loop_to_workers
                            .entry((lp, walk))
                            .or_default()
                            .extend_from_slice(slice);
                    }
                }
            }
            // clone_empty() + setup(node_to_workers), per Python.
            info.shard = Some(
                self.shard
                    .instantiate(&info.node_to_workers)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
            self.requests[handle as usize] = Some(info);
        }

        // Open EVERY local worker graph of this partition, not just the ones
        // in the admission walk. A partition moves through several walks and
        // each has its own worker graph; a request with no state in the next
        // walk's graph cannot have a signal ingested there, so the node never
        // becomes ready and the request stops after one pass. Python opens
        // all of them -- the conductor sends the partition's full list.
        let mine: Vec<WgIndex> = worker_graph_ids
            .iter()
            .filter_map(|&wg_id| self.wg_index(wg_id))
            .collect();
        for &wg in &mine {
            // In range by construction: the vectors were grown for this
            // handle above.
            if self.states[wg as usize][handle as usize].is_none() {
                self.states[wg as usize][handle as usize] =
                    Some(RequestState::new(self.graphs[wg as usize].clone()));
            }
        }

        // The partition's LIVE list is still only the current walk's graphs:
        // that is what selects where this pass's inputs route. Restricted to
        // this request's own graphs, as Python's is.
        let live: Vec<WgIndex> = self
            .live_wgs(walk_sym)
            .into_iter()
            .filter(|w| mine.contains(w))
            .collect();
        let info = self.requests[handle as usize].as_mut().expect("just set");
        for &wg in &mine {
            if !info.worker_graphs.contains(&wg) {
                info.worker_graphs.push(wg);
            }
        }
        info.set_partition(partition_sym, walk_sym, live);

        Ok(handle)
    }

    /// Frees the handle for reuse, so everything keyed by it must be gone
    /// first -- a leak would silently attach to the next request.
    fn remove_request(&mut self, rid: u32) {
        if !self.rids.release(rid) {
            return;
        }
        // Routing parked by complete_and_route_batch whose send never ran --
        // an exception between the two abandons it. Handles are recycled, so
        // a stale entry would make the next request to get this integer send
        // another request's outputs.
        self.completions.retain(|_, c| {
            c.routing.remove(&rid);
            c.completed_wgs.remove(&rid);
            !c.routing.is_empty()
        });

        if let Some(info) = self.requests[rid as usize].take() {
            for &wg in &info.worker_graphs {
                self.states[wg as usize][rid as usize] = None;
            }
        }

        // Cleared once per postprocess, not per removal, so a request admitted
        // between a stop and that clear would read as already stopped: the
        // worker drops its outputs on a speculative new iteration and prep
        // excludes it from speculation.
        self.pending_loop_stops.retain(|&(r, _, _)| r != rid);
    }

    fn get_rid_handle(&self, rid: &str) -> Option<u32> {
        self.rids.handle(rid)
    }

    fn get_rid_string(&self, handle: u32) -> Option<String> {
        self.rids.name(handle).map(|s| s.to_string())
    }

    fn get_walk(&self, rid: u32, partition: &str) -> Option<String> {
        let p = self.interner.get(partition)?;
        let walk = self.info(rid)?.walk(p)?;
        Some(self.interner.name(walk).to_string())
    }

    /// True when the walk changed. Re-derives the live worker graphs, which is
    /// what the routing and ready scans read.
    fn set_walk(&mut self, rid: u32, partition: &str, walk: &str) -> bool {
        let Some(p) = self.interner.get(partition) else { return false };
        let walk_sym = self.interner.intern(walk);
        let live = self.live_wgs_for(rid, walk_sym);
        let Some(info) = self.requests.get_mut(rid as usize).and_then(|r| r.as_mut())
        else {
            return false;
        };
        info.set_walk(p, walk_sym, || live)
    }

    fn mark_stream_partition_done(&mut self, rid: u32, partition: &str) {
        let Some(p) = self.interner.get(partition) else { return };
        if let Some(info) = self.requests.get_mut(rid as usize).and_then(|r| r.as_mut()) {
            info.mark_stream_done(p);
        }
    }

    /// Structural: which local worker graph runs this node in this walk.
    fn get_worker_graph_id_for_node(
        &self, node: &str, graph_walk: &str,
    ) -> PyResult<u32> {
        self.owner_of(node, graph_walk)
            .map(|wg| self.wg_ids[wg as usize])
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Could not find worker graph for node {node:?}, \
                     graph_walk {graph_walk:?}"
                ))
            })
    }

    /// Structural, so no rid: whether the node opts into async scheduling.
    fn is_async_schedulable(&self, node_name: &str, graph_walk: &str) -> bool {
        match self.owner_of(node_name, graph_walk) {
            Some(wg) => self
                .nid(wg, node_name)
                .is_some_and(|n| self.graphs[wg as usize].node(n).async_enabled),
            None => false,
        }
    }

    /// The node's output signal names. Structural: a request's edge objects
    /// are its own, but their names are not.
    fn get_output_signals(&self, node_name: &str, graph_walk: &str) -> Vec<String> {
        let Some(wg) = self.owner_of(node_name, graph_walk) else {
            return vec![];
        };
        let Some(n) = self.nid(wg, node_name) else {
            return vec![];
        };
        let mut names: Vec<String> = self.graphs[wg as usize]
            .node(n)
            .outputs
            .iter()
            .map(|e| self.interner.name(e.name).to_string())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// The (signal, dest) pairs `source_node` emits into `dest_node`. Used by
    /// a speculative batch to know which of the in-flight outputs it consumes.
    fn get_consumed_edges(
        &self, source_node: &str, dest_node: &str, graph_walk: &str,
    ) -> Vec<(String, String)> {
        let Some(wg) = self.owner_of(source_node, graph_walk) else {
            return vec![];
        };
        let Some(n) = self.nid(wg, source_node) else {
            return vec![];
        };
        self.graphs[wg as usize]
            .node(n)
            .outputs
            .iter()
            .filter(|e| self.dest_name(wg, e.dest) == dest_node)
            .map(|e| {
                (
                    self.interner.name(e.name).to_string(),
                    self.dest_name(wg, e.dest),
                )
            })
            .collect()
    }

    // -- pending loop stops: good for exactly one iteration ----------------

    fn has_pending_loop_stop(
        &self, rid: u32, graph_walk: &str, loop_name: &str,
    ) -> bool {
        let (Some(w), Some(l)) = (
            self.interner.get(graph_walk),
            self.interner.get(loop_name),
        ) else {
            return false;
        };
        self.pending_loop_stops.contains(&(rid, w, l))
    }

    fn pending_loop_stop_rids(
        &self, graph_walk: &str, loop_name: &str,
    ) -> Vec<u32> {
        let (Some(w), Some(l)) = (
            self.interner.get(graph_walk),
            self.interner.get(loop_name),
        ) else {
            return vec![];
        };
        self.pending_loop_stops
            .iter()
            .filter(|(_, gw, ln)| *gw == w && *ln == l)
            .map(|(rid, _, _)| *rid)
            .collect()
    }

    fn clear_pending_loop_stops(&mut self) {
        self.pending_loop_stops.clear();
    }

    fn push_back_node(
        &mut self, node_name: String, rids: Vec<u32>, wg_ids: Vec<u32>,
    ) -> PyResult<()> {
        if rids.len() != wg_ids.len() {
            return Err(PyValueError::new_err(
                "push_back_node: rids and wg_ids must be the same length",
            ));
        }
        for (rid, wg_id) in rids.into_iter().zip(wg_ids) {
            let Some(wg) = self.wg_index(wg_id) else { continue };
            let Some(node_id) = self.nid(wg, &node_name) else { continue };
            if let Some(state) = self.state_mut(wg, rid) {
                state.push_back(node_id);
            }
        }
        Ok(())
    }

    // --------- Inputs ----------

    /// Returns the INDICES of signals no worker graph claimed. The streaming
    /// path hands the original edge back to its buffer, so an index is enough
    /// and the edge never has to survive the round trip.
    #[pyo3(signature = (rids, signals, can_buffer = true, is_streaming = false))]
    fn ingest_inputs_batch(
        &mut self,
        rids: Vec<u32>,
        signals: Vec<EdgeSpecArg>,
        can_buffer: bool,
        is_streaming: bool,
    ) -> PyResult<Vec<usize>> {
        if rids.len() != signals.len() {
            return Err(PyValueError::new_err(
                "ingest_inputs_batch: rids and signals must be the same length",
            ));
        }
        let mut uningested = Vec::new();
        for (i, (rid, spec)) in rids.iter().zip(&signals).enumerate() {
            if !self.ingest_one(*rid, spec, can_buffer, is_streaming) {
                uningested.push(i);
            }
        }
        Ok(uningested)
    }

    /// Per rid, loop name -> current iteration, for the partition's live
    /// worker graphs.
    fn get_dynamic_loop_iters(
        &self, request_ids: Vec<u32>, partition: &str,
    ) -> Vec<Vec<(String, u32)>> {
        let Some(p) = self.interner.get(partition) else {
            return request_ids.iter().map(|_| vec![]).collect();
        };
        request_ids
            .iter()
            .map(|&rid| {
                let mut out = Vec::new();
                let Some(info) = self.info(rid) else { return out };
                let Some(part) = info.partitions.get(&p) else { return out };
                for &wg in &part.walk_worker_graphs {
                    let Some(state) = self.state(wg, rid) else {
                        continue;
                    };
                    let g = &self.graphs[wg as usize];
                    for (lid, iter) in state.loop_indices().into_iter().enumerate() {
                        out.push((
                            self.interner.name(g.lp(lid as LoopId).name).to_string(),
                            iter,
                        ));
                    }
                }
                out
            })
            .collect()
    }

    /// No-op by construction. Python clears `tensor_info` off the node's
    /// output edges because the edges are per-request objects that carry it;
    /// here outputs are passed into `complete_and_route_batch`, so there is
    /// nothing stale to clear.
    /// A no-op by construction: nothing here caches a node's outputs between
    /// passes. `complete` derives them from the tensors it is handed, so there
    /// is no stale set to drop -- where Python clears `GraphNode.outputs`.
    /// Kept so the contract is satisfied; the caller no longer invokes it.
    fn reset_outputs(
        &mut self, node_name: &str, rids: Vec<u32>, wg_ids: Vec<u32>,
    ) {
        let _ = (node_name, rids, wg_ids);
    }

    /// Release the inputs the just-executed node consumed, dereferencing them
    /// in the bookkeeper the store shares.
    ///
    /// Returns the ones that hit zero, with each one's `mem_registered` flag,
    /// because the rest of their teardown -- the shm file, the arena slot,
    /// the memory unregistration -- lives on the tensor manager, which is
    /// Python. Dropping them here and saying nothing is what left a consumed
    /// input's shm file sitting until the request was torn down: the Python
    /// runtime dereferences through the manager and reclaims as it goes.
    fn cleanup_consumed_inputs(
        &mut self, node_name: &str, rids: Vec<u32>, wg_ids: Vec<u32>,
    ) -> PyResult<(Vec<u64>, Vec<bool>)> {
        if rids.len() != wg_ids.len() {
            return Err(PyValueError::new_err(
                "cleanup_consumed_inputs: rids and wg_ids must be the same length",
            ));
        }
        let mut freed: Vec<u64> = Vec::new();
        for (rid, wg_id) in rids.into_iter().zip(wg_ids) {
            let Some(wg) = self.wg_index(wg_id) else { continue };
            let Some(node) = self.nid(wg, node_name) else { continue };
            if let Some(state) = self.state_mut(wg, rid) {
                freed.extend(state.clear_consumed_inputs(node));
            }
        }
        if freed.is_empty() {
            return Ok((vec![], vec![]));
        }
        let mut bk = self.bookkeeping.lock().unwrap();
        Ok(bk.drop_refs(freed.into_iter().map(|u| (u, 1)), true))
    }

    // --------- Scheduling ----------

    /// Every (node, walk, rid) whose GRAPH inputs are satisfied.
    ///
    /// Graph level only. Engine readiness stays with the caller because it can
    /// FAIL a request, which is a scheduling decision rather than a graph one.
    #[pyo3(signature = (exclude_rids, target = None, exclude_target = None))]
    fn get_ready_nodes(
        &self,
        exclude_rids: Vec<u32>,
        target: Option<(String, String)>,
        exclude_target: Option<(String, String)>,
    ) -> Vec<(String, String, Vec<u32>)> {
        let excluded: FxHashSet<u32> = exclude_rids.into_iter().collect();
        let mut grouped: FxHashMap<(Sym, Sym), Vec<u32>> = FxHashMap::default();
        self.scan_ready(&excluded, target.as_ref(), exclude_target.as_ref(),
            |node, walk, rid| {
                grouped.entry((node, walk)).or_default().push(rid);
                true // keep scanning
            });
        grouped
            .into_iter()
            .map(|((n, w), rids)| {
                (
                    self.interner.name(n).to_string(),
                    self.interner.name(w).to_string(),
                    rids,
                )
            })
            .collect()
    }

    /// Stops at the first match rather than building the list. Graph readiness
    /// is necessary but not sufficient, so a False here is final and lets the
    /// caller skip its engine pass entirely.
    #[pyo3(signature = (exclude_rids, exclude_target = None))]
    fn has_ready_excluding(
        &self,
        exclude_rids: Vec<u32>,
        exclude_target: Option<(String, String)>,
    ) -> bool {
        let excluded: FxHashSet<u32> = exclude_rids.into_iter().collect();
        let mut found = false;
        self.scan_ready(&excluded, None, exclude_target.as_ref(), |_, _, _| {
            found = true;
            false // stop
        });
        found
    }

    /// Pop `node_name` for exactly `request_ids`.
    ///
    /// With `check_ready`, all or none: verified for every rid before anything
    /// is popped, so a partially ready set is retried intact. Engine readiness
    /// is assumed to have been checked already.
    #[pyo3(signature = (node_name, graph_walk, request_ids, check_ready = false))]
    fn pop_rids(
        &mut self,
        node_name: &str,
        graph_walk: &str,
        request_ids: Vec<u32>,
        check_ready: bool,
    ) -> Option<PopRidsOut> {
        let wg = self.owner_of(node_name, graph_walk)?;
        let node = self.nid(wg, node_name)?;
        let wg_id = self.wg_ids[wg as usize];

        if check_ready {
            for &rid in &request_ids {
                let ready = self.state(wg, rid)
                    .is_some_and(|s| s.is_ready(node));
                if !ready {
                    return None; // unknown rid counts as not ready
                }
            }
        }

        let mut out = PopRidsOut::default();
        // Structural: the same for every rid in the batch.
        out.output_signals = {
            let g = self.g(wg).clone();
            let mut v: Vec<String> = g
                .node(node)
                .outputs
                .iter()
                .map(|e| self.interner.name(e.name).to_string())
                .collect();
            v.sort();
            v.dedup();
            v
        };
        for rid in request_ids {
            let Some(state) = self.state_mut(wg, rid) else {
                continue;
            };
            state.take_for_schedule(node);
            let inputs = state.input_tensors(node, false);
            out.rids.push(rid);
            out.wg_ids.push(wg_id);
            out.input_edges_per_rid.push(inputs.len());
            for (name, tensors, final_chunk) in inputs {
                out.input_edges.push((
                    self.interner.name(name).to_string(),
                    node_name.to_string(),
                    tensors.iter().map(|t| t.uuid).collect(),
                    final_chunk,
                ));
            }
        }
        Some(out)
    }

    // --------- Speculation ----------

    /// Which nodes could run next, after the current node's outputs land.
    ///
    /// Filters for async eligibility; the per-rid loop-completion filter lives
    /// in prep_spec_rids.
    fn speculate_node(
        &mut self, node_name: &str, graph_walk: &str, sample_rid: u32,
    ) -> Vec<(String, String, bool, Option<String>, Vec<String>)> {
        let Some(wg) = self.owner_of(node_name, graph_walk) else {
            return vec![];
        };
        let Some(source) = self.nid(wg, node_name) else { return vec![] };
        let Some(spec_nodes) = self.spec_targets(wg, source, sample_rid) else {
            return vec![];
        };
        let g = self.graphs[wg as usize].clone();
        spec_nodes
            .into_iter()
            .filter(|sn| {
                // The DESTINATION opts out of async scheduling. Mirrors the
                // source-side check the caller already made; without it a
                // structurally ineligible destination is picked here and then
                // dropped per rid.
                g.node(sn.node).async_enabled
                    && self.async_checker.can_speculate(
                        g.node(source).name, g.node(sn.node).name,
                    )
            })
            .map(|sn| self.spec_output(&g, graph_walk, &sn))
            .collect()
    }

    /// The loop context of a target chosen elsewhere.
    ///
    /// A follower cannot use speculate_node: that filter requires the node be
    /// a parallel LEADER node, and a follower by definition is not. The leader
    /// already decided; this reports what the target is.
    fn get_spec_target(
        &mut self,
        curr_node_name: &str,
        spec_node_name: &str,
        graph_walk: &str,
        sample_rid: u32,
    ) -> Option<(String, String, bool, Option<String>, Vec<String>)> {
        let wg = self.owner_of(curr_node_name, graph_walk)?;
        let source = self.nid(wg, curr_node_name)?;
        let want = self.nid(wg, spec_node_name)?;
        let spec_nodes = self.spec_targets(wg, source, sample_rid)?;
        let g = self.graphs[wg as usize].clone();
        spec_nodes
            .into_iter()
            .find(|sn| sn.node == want)
            .map(|sn| self.spec_output(&g, graph_walk, &sn))
    }

    /// Ingest each rid's stream chunks, check readiness, gather inputs.
    ///
    /// Per-rid best-effort: a rid that fails is rolled back and skipped. The
    /// loop-completion filter (a stop already pending, or the next iteration
    /// being past the last) lives here.
    fn prep_spec_rids(&mut self, input: SpecPrepArg) -> PyResult<SpecPrepOut> {
        self.prep_spec(input, false)
    }

    /// The TP-follower counterpart: ALL or nothing, no room cap, no loop
    /// filter. Rank 0 committed to this exact composition and sits on the
    /// collective until every follower joins, so one rid failing has to roll
    /// the whole batch back.
    fn prep_follow_spec_rids(
        &mut self, input: SpecPrepArg,
    ) -> PyResult<Option<SpecPrepOut>> {
        let out = self.prep_spec(input, true)?;
        Ok(if out.all_or_nothing_failed { None } else { Some(out) })
    }

    // --------- Postprocess ----------

    /// Mark each node complete, route its outputs, and settle the refcounts.
    ///
    /// `tensors` is flat and rid-major over `rids`;
    /// `num_tensors[i * output_signals.len() + s]` of them belong to rid i's
    /// signal s. The routing is parked under the returned completion id for
    /// `send_outputs` to consume.
    fn complete_and_route_batch(
        &mut self, input: RouteArg,
    ) -> PyResult<RouteOut> {
        let n_sig = input.output_signals.len();
        if input.rids.len() * n_sig != input.num_tensors.len() {
            return Err(PyValueError::new_err(
                "complete_and_route_batch: num_tensors must be \
                 rids * output_signals",
            ));
        }
        let Some(wg) = self.owner_of(&input.node_name, &input.graph_walk)
        else {
            return Err(PyValueError::new_err(format!(
                "no worker graph for node {:?} in walk {:?}",
                input.node_name, input.graph_walk
            )));
        };
        let Some(node) = self.nid(wg, &input.node_name) else {
            return Err(PyValueError::new_err("unknown node"));
        };
        let g = self.graphs[wg as usize].clone();
        let signals: Vec<Option<Sym>> = input
            .output_signals
            .iter()
            .map(|s| self.interner.get(s))
            .collect();
        let uuid_to_idx: FxHashMap<u64, usize> = input
            .tensors
            .iter()
            .enumerate()
            .map(|(i, &u)| (u, i))
            .collect();

        // The local worker graphs live in this walk; the done-sweep below
        // intersects each request's registered graphs with these.
        let walk_sym = self.interner.get(&input.graph_walk);
        let walk_wgs: Vec<WgIndex> = walk_sym
            .and_then(|w| self.walk_to_local_wgs.get(&w).cloned())
            .unwrap_or_default();
        let walk = walk_sym.unwrap_or_default();

        let mut out = RouteOut::default();
        let mut routing: FxHashMap<u32, Vec<RoutedEdge>> = FxHashMap::default();
        let mut persist: FxHashMap<u32, Vec<(Sym, Vec<TensorRef>)>> =
            FxHashMap::default();
        let mut completed_wgs: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        let mut first_tp_rank: FxHashMap<u32, bool> = FxHashMap::default();
        let mut speculative: FxHashMap<u32, bool> = FxHashMap::default();
        let mut staged: FxHashSet<u64> = FxHashSet::default();
        let mut sends_a_frame: FxHashSet<u32> = FxHashSet::default();
        let mut nested_snapshot: FxHashMap<u32, (Vec<Sym>, Vec<(Sym, u32)>, u32)> =
            FxHashMap::default();
        let mut cursor = 0usize;

        for (i, &rid) in input.rids.iter().enumerate() {
            // Slice this rid's uuids back out of the flat, rid-major layout.
            let mut by_signal: FxHashMap<Sym, Vec<TensorRef>> =
                FxHashMap::default();
            let mut owned: Vec<u64> = Vec::new();
            {
                let bk = self.bookkeeping.lock().unwrap();
                for (s, sig) in signals.iter().enumerate() {
                    let count = input.num_tensors[i * n_sig + s];
                    let slice = &input.tensors[cursor..cursor + count];
                    cursor += count;
                    if let Some(sig) = sig {
                        by_signal.insert(
                            *sig,
                            slice.iter().map(|&u| bk.tensor_ref(u)).collect(),
                        );
                    }
                    owned.extend_from_slice(slice);
                }
            }

            let out_tensors: Vec<Vec<TensorRef>> = g
                .node(node)
                .outputs
                .iter()
                .map(|e| by_signal.get(&e.name).cloned().unwrap_or_default())
                .collect();

            // Before complete(), which advances the loop counters via
            // advance_loop. stop_loops runs earlier in the worker's pass but
            // only sets `finish_signal` (state.rs::register_loop_finish) --
            // it does NOT move curr_iter -- so capturing here reports the
            // same counters a snapshot taken before it would have.
            if let Some(idx) = self.nested_idxs_interned(wg, rid, node) {
                nested_snapshot.insert(rid, idx);
            }
            let Some(state) = self.state_mut(wg, rid) else {
                continue;
            };
            // Before complete(), which clears the flag.
            let was_speculative = state.is_spec_scheduled(node);
            let completed = state.complete(node, &out_tensors);
            let pre_shard_edges = completed.edges;
            let freed_inputs = completed.freed;
            // A loop that cached this node's outputs holds a reference on
            // each, as `Loop.maybe_cache_output` takes one. Applied before
            // `freed_inputs`, which can release the same tensors (an advance
            // un-caches what this pass cached).
            if !completed.taken.is_empty() {
                let mut bk = self.bookkeeping.lock().unwrap();
                for uuid in completed.taken {
                    bk.increment_ref(uuid, 1)?;
                }
            }
            // Both taken before the fanout, as Python takes `to_conductor`
            // and `new_token_outputs` off the node's own outputs. A rank other
            // than 0 loses these edges in the replicated fanout -- their
            // destination (EMPTY_DESTINATION) has no sharding group -- and
            // reporting them is not the fanout's business anyway.
            let persist_pre: Vec<(Sym, Vec<TensorRef>)> = pre_shard_edges
                .iter()
                .filter(|e| e.persist)
                .map(|e| (e.name, e.tensors.clone()))
                .collect();
            {
                // First edge of a signal name wins, as in Python's
                // _count_new_tokens: one output routed twice is two edges
                // over the SAME tensors.
                let mut seen: FxHashSet<Sym> = FxHashSet::default();
                for e in pre_shard_edges.iter().filter(|e| e.new_token) {
                    if !seen.insert(e.name) {
                        continue;
                    }
                    out.new_token_output_idxs.extend(
                        e.tensors
                            .iter()
                            .filter_map(|t| uuid_to_idx.get(&t.uuid).copied()),
                    );
                }
            }
            let me_sym = self.shard.me;
            let mut edges = if let Some(info) = &self.requests[rid as usize] {
                if let Some(sharding) = &info.shard {
                    let mut sharded_edges: Vec<RoutedEdge> = vec![];
                    for edge in &pre_shard_edges {
                        let dst_walk = if edge.streaming {
                            None
                        } else { Some(walk) };
                        let mut shards: Vec<FanoutDest> = vec![];
                        // The node's interned NAME, which the sharding map is
                        // keyed by -- not its NodeId. Both are u32, so passing
                        // the id compiled and looked up whichever node's name
                        // happened to share that integer.
                        sharding.fanout(
                            edge.name, g.node(node).name,
                            walk, edge.dest_sym,
                            dst_walk, &edge.tensors,
                            &mut shards
                        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
                        for shard in shards {
                            let mut new_edge = edge.clone();
                            new_edge.tensors = shard.tensors;
                            if edge.dest.is_to_worker() {
                                new_edge.worker = Some(shard.worker);
                                // A peer's copy cannot be ingested here, so a
                                // Local dest has to become a wire send.
                                if shard.worker != me_sym {
                                    new_edge.dest = Dest::External(edge.dest_sym);
                                }
                            }
                            sharded_edges.push(new_edge);
                        }
                    }
                    sharded_edges
                } else {
                    pre_shard_edges
                }
            } else {
                pre_shard_edges
            };
            
            speculative.insert(rid, was_speculative);

            // No group means singleton / non-TP, which is rank 0. The
            // conductor counts one report per request, so a rank wrongly
            // claiming rank 0 multiplies that count by tp_size.
            let node_sym = self.interner.get(&input.node_name);
            first_tp_rank.insert(
                rid,
                self.info(rid)
                    .zip(node_sym)
                    .and_then(|(i, n)| i.shard.as_ref()?.group_of(n, walk_sym))
                    .is_none_or(|g| g.tp_rank == Some(0)),
            );

            // Sweep EVERY worker graph this request runs in this walk, not
            // just the one owning the completed node: a wg can become done
            // without ingesting an edge here, when the completed node's
            // outputs all go to EMPTY_DESTINATION / EMIT_TO_CLIENT / a
            // streaming partition.
            //
            // Each one that is done is RESET. A worker graph completes many
            // times over a request, and without the reset `is_done` latches:
            // root_entity_done early-returns on it, node flags and loop
            // counters never clear, and the request reports done exactly once
            // and can never become ready again.
            let sweep: Vec<WgIndex> = match &self.requests[rid as usize] {
                Some(info) => info
                    .worker_graphs
                    .iter()
                    .copied()
                    .filter(|w| walk_wgs.contains(w))
                    .collect(),
                None => vec![wg],
            };
            for w in sweep {
                let Some(st) = self.state_mut(w, rid) else { continue };
                if !st.is_done {
                    continue;
                }
                st.reset();
                completed_wgs
                    .entry(rid)
                    .or_default()
                    .push(self.wg_ids[w as usize]);
            }

            // Local destinations ingest now, so a consumer on this worker sees
            // the routed tensors without a round trip.
            //
            // `Dest::Local` only means "a node in THIS worker graph". A node in
            // a SIBLING worker graph on this same worker compiles to
            // `External`, and one walk routinely spans several graphs
            // (vit_encoder and LLM are separate graphs in BAGEL's prefill_vit).
            // Resolved against node_owner, which is Python's
            // _walk_node_to_wg_id: without this the edge only ever leaves on
            // the wire, and the sibling graph's node never receives it.
            let mut local_counts: FxHashMap<u64, i64> = FxHashMap::default();
            let mut ingested_locally: Vec<bool> = Vec::with_capacity(edges.len());
            for e in &edges {
                // Post-fanout, an edge belongs to exactly ONE worker. A peer's
                // copy still names a node this worker runs, so resolving it by
                // name alone ingests the peer's tensors here too -- filling the
                // next-iteration slot and starving the real loop-back edge.
                // Python pops its own id out of the fanout and wires the rest.
                let target = if e.worker.is_some_and(|w| w != me_sym) {
                    None
                } else { match e.dest {
                    Dest::Local(d) => Some((wg, d)),
                    // A streaming destination is local when THIS WORKER runs
                    // it at all -- Python resolves the fanout with
                    // dest_graph_walk=None and asks whether its own id is in
                    // it. The consumer normally lives in another walk
                    // entirely (Orpheus streams from `decode` into
                    // `snac_chunk`), so a walk-keyed lookup would never
                    // find it.
                    Dest::External(name) if e.streaming => self
                        .graphs
                        .iter()
                        .position(|g| g.by_name.contains_key(&name))
                        .filter(|_| self.runs_for_request(rid, name))
                        .map(|i| (i as WgIndex, self.graphs[i].by_name[&name])),
                    // A non-streaming destination has to be live in THIS walk,
                    // as Python's _walk_node_to_wg_id lookup is.
                    Dest::External(name) => walk_sym
                        .and_then(|w| self.node_owner.get(&(w, name)).copied())
                        .filter(|_| self.runs_for_request(rid, name))
                        .and_then(|owner| {
                            let g2 = self.graphs[owner as usize].clone();
                            g2.by_name.get(&name).copied().map(|d| (owner, d))
                        }),
                    _ => None,
                }};
                let mut took = false;
                if let Some((owner, d)) = target {
                    if e.streaming {
                        // A streaming edge never goes straight into the node:
                        // it lands in the worker's StreamBuffer, which decides
                        // when a chunk is whole. Reported to the caller via
                        // local_streaming_tensor_idxs below.
                        took = true;
                    } else {
                        let slot =
                            self.graphs[owner as usize].node(d).slot_of(e.name);
                        if let Some(slot) = slot {
                            if let Some(st) = self.state_mut(owner, rid) {
                                // A declining graph falls through to the wire,
                                // as Python's `leftover` branch does.
                                took = st.ingest(
                                    d, slot, e.tensors.clone(), true, false,
                                );
                            }
                        }
                    }
                }
                if took {
                    for t in &e.tensors {
                        *local_counts.entry(t.uuid).or_insert(0) += 1;
                    }
                }
                ingested_locally.push(took);
            }

            // A local consumer that refused the edge makes it remote, as
            // Python's `leftover` branch does. Recorded before the frame
            // check so it, the registration, the reference count and
            // take_send_plan all agree.
            for (e, &took) in edges.iter_mut().zip(&ingested_locally) {
                e.declined_local = !took && matches!(e.dest, Dest::Local(_));
            }

            // Which rids will actually produce a frame carrying
            // `per_request_info`: an INPUT_SIGNALS to a peer, or a
            // WORKER_GRAPHS_DONE. EmitToClient does not carry it.
            //
            // `is_local` is the whole point. A node in a SIBLING worker graph
            // on this worker compiles to External (Orpheus streams new_token
            // from the LLM graph into snac_decoder), and counting that as a
            // send makes every decode pass look like it needs the payload --
            // which is exactly nothing skipped. What matters is whether any
            // destination worker is someone OTHER than us, mirroring
            // take_send_plan's own skip.
            let me_now = self.shard.me;
            for (e, &is_local) in edges.iter().zip(&ingested_locally) {
                let dest = match e.dest {
                    Dest::External(d) => d,
                    Dest::Local(d) if e.declined_local => self.g(wg).node(d).name,
                    _ => continue,
                };
                // A refused edge goes on the wire even when we are its owner.
                let goes_out = e.declined_local || match e.worker {
                    // Post-fanout: one edge, one destination worker.
                    Some(w) => w != me_now || !is_local,
                    None => walk_sym
                        .and_then(|w| {
                            let workers =
                                self.info(rid)?.node_to_workers.get(&(dest, w))?;
                            Some(workers.iter().any(|&x| x != me_now))
                        })
                        .unwrap_or(false),
                };
                if goes_out {
                    sends_a_frame.insert(rid);
                    break;
                }
            }

            // Staged before the routed edges, so a persist signal is
            // registered for a remote read whether or not the fanout kept an
            // edge for it -- Python stages `routing.persist` unconditionally.
            for (_name, tensors) in &persist_pre {
                for t in tensors {
                    let Some(&idx) = uuid_to_idx.get(&t.uuid) else { continue };
                    if staged.insert(t.uuid) {
                        out.register_tensor_idxs.push(idx);
                        out.register_rids.push(rid);
                    }
                }
            }

            for (e, &is_local) in edges.iter().zip(&ingested_locally) {
                let idxs: Vec<usize> = e
                    .tensors
                    .iter()
                    .filter_map(|t| uuid_to_idx.get(&t.uuid).copied())
                    .collect();
                // `is_local`, not Dest::Local: the consumer is just as local
                // when it lives in a SIBLING worker graph on this worker, which
                // compiles to External (Orpheus streams new_token from the LLM
                // graph into the snac_decoder graph).
                if e.streaming && is_local {
                    out.local_streaming_tensor_idxs.extend(&idxs);
                }
                // What a remote consumer will read. Deduped by uuid and
                // skipping anything already staged, so a re-emitted edge does
                // not stage twice.
                // A tensor handled locally is not staged for a remote read;
                // Python leaves streaming_local and the locally-ingested edge
                // out of the register set for the same reason.
                let remote = e.persist
                    || e.declined_local
                    || (!is_local
                        && matches!(e.dest, Dest::External(_) | Dest::EmitToClient));
                if remote {
                    for (&idx, t) in idxs.iter().zip(&e.tensors) {
                        if staged.insert(t.uuid) {
                            out.register_tensor_idxs.push(idx);
                            out.register_rids.push(rid);
                        }
                    }
                }
            }

            // Inputs the completion cleared, and tensors a loop reset or
            // un-cached. Dereferenced here so the result
            // does not depend on whether cleanup_consumed_inputs ran first --
            // and reported for the same reason that call reports its own: the
            // teardown of anything that hit zero is the tensor manager's, and
            // this side cannot reach it.
            if !freed_inputs.is_empty() {
                let mut bk = self.bookkeeping.lock().unwrap();
                let (uuids, registered) =
                    bk.drop_refs(freed_inputs.into_iter().map(|u| (u, 1)), true);
                out.freed_input_uuids.extend(uuids);
                out.freed_input_registered.extend(registered);
            }

            // How many references each edge really represents. Python counts
            // the POST-fanout edges -- to_workers is keyed by worker, so a
            // tensor read by N workers is counted N times -- while the edges
            // here are pre-fanout, one per graph edge. Counting them once
            // settles the hold to 1 with N reads outstanding, and the first
            // release frees a tensor the other N-1 are still reading.
            //
            // EMPTY_DESTINATION is 0, not 1: it routes nowhere, and Python
            // drops it before the count.
            let edge_refs: Vec<i64> = edges
                .iter()
                .zip(&ingested_locally)
                .map(|(e, &is_local)| {
                    match e.dest {
                        // Already in local_counts, unless it was refused --
                        // then nothing counted it and it goes on the wire.
                        Dest::Local(_) if e.declined_local => 1,
                        Dest::Local(_) | Dest::Empty => 0,
                        Dest::EmitToClient => 1,
                        // One reference per destination WORKER, minus this
                        // one when the edge was already ingested into a local
                        // graph -- that copy is in local_counts. Python does
                        // the same by popping its own id out of the fanout.
                        Dest::External(d) => walk_sym
                            .and_then(|w| {
                                if let Some(dst) = e.worker {
                                    // Post-fanout: this edge IS one
                                    // destination. Ours only when nothing
                                    // ingested it -- otherwise local_counts
                                    // already has it, and take_send_plan
                                    // skips the self-send.
                                    Some(usize::from(!(dst == me_sym && is_local)))
                                } else {
                                    let workers =
                                        self.info(rid)?.node_to_workers.get(&(d, w))?;
                                    let mine = self.node_owner.contains_key(&(w, d));
                                    Some(
                                        workers
                                            .iter()
                                            .filter(|&&x| !(mine && x == me_sym))
                                            .count(),
                                    )
                                }
                            })
                            .unwrap_or(0) as i64,
                    }
                })
                .collect();

            // Settle from the safety hold of 1 to the real fanout. persist is
            // excluded: those are held by the marker, and counting them would
            // double-count a signal whose destination is EMPTY_DESTINATION.
            {
                let mut bk = self.bookkeeping.lock().unwrap();
                for uuid in owned {
                    let mut count = local_counts.get(&uuid).copied().unwrap_or(0);
                    for (e, &refs) in edges.iter().zip(&edge_refs) {
                        if refs == 0 {
                            continue;
                        }
                        count += refs
                            * e.tensors.iter().filter(|t| t.uuid == uuid).count() as i64;
                    }
                    // Pre-fanout: a rank whose persist edge the fanout
                    // dropped still holds the tensor for the conductor, and
                    // an unmarked one is dereferenced to zero right here.
                    if persist_pre
                        .iter()
                        .any(|(_, ts)| ts.iter().any(|t| t.uuid == uuid))
                    {
                        bk.set_persist(uuid, true);
                    }
                    let delta = count - 1;
                    if delta > 0 {
                        bk.increment_ref(uuid, delta)?;
                    } else if delta < 0 {
                        // Through drop_refs, not a bare decrement: an output
                        // nothing consumes reaches zero here, and only the
                        // tensor manager can tear it down, so it has to be
                        // reported -- otherwise it is held until the request
                        // ends.
                        let (uuids, registered) =
                            bk.drop_refs(std::iter::once((uuid, -delta)), true);
                        out.freed_input_uuids.extend(uuids);
                        out.freed_input_registered.extend(registered);
                    }
                }
            }

            routing.insert(rid, edges);
            persist.insert(rid, persist_pre);
        }

        // A wg can finish without this node routing anything for that rid, so
        // the completed set is its own source of frames.
        for rid in completed_wgs.keys() {
            if !sends_a_frame.contains(rid) {
                sends_a_frame.insert(*rid);
            }
        }
        out.rids_needing_request_info = sends_a_frame.into_iter().collect();

        self.completion_counter += 1;
        out.completion_id = self.completion_counter;
        self.completions.insert(
            self.completion_counter,
            Completion {
                partition: input.partition.clone(),
                graph_walk: input.graph_walk.clone(),
                node_name: input.node_name.clone(),
                wg,
                routing,
                persist,
                completed_wgs,
                first_tp_rank,
                speculative,
                nested: nested_snapshot,
            },
        );
        Ok(out)
    }

    /// Stop loops, record the pending stops, and report who to tell.
    ///
    /// Returns (worker, loop names) per rid for the caller to send. The frames
    /// are still built in Python: a Rust encoder has to reproduce the typed
    /// msgpack wire.py emits byte for byte, which is its own piece of work.
    #[allow(clippy::type_complexity)]
    fn stop_loops_batched(
        &mut self,
        py: Python<'_>,
        partition: &str,
        graph_walk: &str,
        last_node_run: &str,
        rids: Vec<u32>,
        loop_names: Vec<Vec<String>>,
    ) -> PyResult<()> {
        // One release for the whole batch; see send_outputs.
        py.allow_threads(move || -> PyResult<()> {
            if rids.len() != loop_names.len() {
                return Err(PyValueError::new_err(
                    "stop_loops_batched: rids and loop_names must be the same length",
                ));
            }
            let Some(walk) = self.interner.get(graph_walk) else {
                return Ok(());
            };
            for (rid, names) in rids.into_iter().zip(loop_names) {
                let wanted: Vec<String> = names
                    .into_iter()
                    .filter(|n| self.check_dyn_loop(rid, partition, n))
                    .collect();
                if wanted.is_empty() {
                    continue;
                }
                self.stop_loops_for_rid(rid, partition, &wanted, Some(last_node_run));
                for name in &wanted {
                    if let Some(l) = self.interner.get(name) {
                        self.pending_loop_stops.insert((rid, walk, l));
                    }
                }
                let mut per_worker: FxHashMap<Sym, Vec<String>> = FxHashMap::default();
                for name in &wanted {
                    for w in self.dyn_loop_workers(rid, partition, name) {
                        per_worker.entry(w).or_default().push(name.clone());
                    }
                }
                // Never to ourselves: this rank originated the stop and has
                // already applied it. A self-send would land in apply_peer_loop_stops
                // and stop the loops a second time.
                let me = self.shard.me;
                let request_id = self.rid_name(rid)?;
                // Snapshotted BEFORE the sends: stop_loops_for_rid above already
                // recorded this stop, and every peer must see the same context.
                let stop_times = self.loop_stop_times_of(rid);
                for (worker, names) in per_worker {
                    if worker == me {
                        continue;
                    }
                    let bytes = frames::StopLoops {
                        request_id: &request_id,
                        partition_name: partition,
                        loop_names: &names,
                        loop_stop_times: stop_times.clone(),
                    }
                    .encode(&self.interner);
                    self.dispatch(self.interner.name(worker), &bytes)?;
                }
            }
            Ok(())
        })
    }

    /// A peer's STOP_LOOPS landing here.
    ///
    /// Stops only the loops whose incoming observation is NEWER than this
    /// rank's, and does NOT fan out: the rank that originated the stop already
    /// told everyone, so re-sending would loop.
    fn apply_peer_loop_stops(
        &mut self,
        rid: u32,
        partition: &str,
        loop_names: Vec<String>,
        stop_times: Vec<LoopStopArg>,
    ) -> PyResult<()> {
        if loop_names.len() != stop_times.len() {
            return Err(PyValueError::new_err(
                "apply_peer_loop_stops: names and times must be the same length",
            ));
        }
        let Some(p) = self.interner.get(partition) else { return Ok(()) };
        if self.info(rid).is_none_or(|i| !i.partitions.contains_key(&p)) {
            return Ok(());
        }
        let mut newer = Vec::new();
        for (name, arg) in loop_names.iter().zip(stop_times) {
            let sym = self.interner.intern(name);
            let stop = LoopStopTime {
                loop_name_order: arg
                    .loop_name_order
                    .iter()
                    .map(|n| self.interner.intern(n))
                    .collect(),
                loop_indices: arg
                    .loop_indices
                    .iter()
                    .map(|(n, i)| (self.interner.intern(n), *i))
                    .collect(),
                wg_fwd_pass_idx: arg.wg_fwd_pass_idx,
            };
            let info = self.requests[rid as usize].as_mut().expect("checked");
            if stop.later_than(info.loop_stop_times.get(&sym), sym) {
                newer.push(name.clone());
            }
            info.loop_stop_times.insert(sym, stop);
        }
        if !newer.is_empty() {
            // No last_node_run and no fan-out: the originating rank took the
            // snapshot and told everyone already.
            self.stop_loops_for_rid(rid, partition, &newer, None);
        }
        Ok(())
    }

    /// Send every frame the parked plan calls for: INPUT_SIGNALS to each peer
    /// worker, RESULT_TENSORS to the api server, and one WORKER_GRAPHS_DONE
    /// per finished request. The whole of Python's send_outputs.
    ///
    /// `request_infos` and `profiling` are Python-owned payloads, already
    /// encoded. The loop context each frame reports is the one
    /// `complete_and_route_batch` snapshotted BEFORE the completion advanced
    /// it, parked with the plan.
    ///
    /// Everything arrives struct-of-arrays -- a rid list beside a value list
    /// -- because that is the shape `ParallelList` already holds it in.
    /// Zipping them into tuples on the Python side just to unzip here is a
    /// per-rid interpreted loop per argument, every forward pass.
    #[pyo3(signature = (
        completion_id, info_rids, request_infos,
        ntc_rids, new_token_counts,
        consumed_rids, stream_tokens_consumed, prof_rids, profiling,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn send_outputs(
        &mut self,
        py: Python<'_>,
        completion_id: u64,
        info_rids: Vec<u32>,
        request_infos: Vec<Option<Vec<u8>>>,
        ntc_rids: Vec<u32>,
        // Taken as dicts: PyO3 extracts them directly, so Python does not
        // have to flatten each one to a list of pairs.
        new_token_counts: Vec<FxHashMap<String, i64>>,
        consumed_rids: Vec<u32>,
        stream_tokens_consumed: Vec<FxHashMap<String, i64>>,
        prof_rids: Vec<u32>,
        profiling: Vec<Option<Vec<u8>>>,
    ) -> PyResult<()> {
        // Released for the whole frame phase rather than around each send.
        // Nothing below touches Python: the arguments arrive already
        // extracted, the frames are msgpack built from the interner and the
        // bookkeeper, and dispatch goes straight to the zmq socket. Holding
        // the GIL across any of it freezes every Python thread in the worker
        // -- a PUSH send blocks at the peer's high-water mark, so a stalled
        // conductor or peer would take the whole process down with it, and
        // the encoding is real CPU work besides. One release per pass also
        // costs one reacquisition rather than one per frame, which matters
        // because a reacquire can queue behind a runnable Python thread for
        // a whole switch interval.
        //
        // The `&mut self` borrow is held across the release, so no other
        // Python thread may call into this runtime while a send is in
        // flight: it would get `Already mutably borrowed` rather than block.
        // Only the worker's main loop does today.
        py.allow_threads(move || -> PyResult<()> {
            let request_infos: Vec<(u32, Option<Vec<u8>>)> =
                info_rids.into_iter().zip(request_infos).collect();
            let new_token_counts: Vec<(u32, Vec<(String, i64)>)> = ntc_rids
                .into_iter()
                .zip(new_token_counts)
                .map(|(r, m)| (r, m.into_iter().collect()))
                .collect();
            let stream_tokens_consumed: Vec<(u32, Vec<(String, i64)>)> = consumed_rids
                .into_iter()
                .zip(stream_tokens_consumed)
                .map(|(r, m)| (r, m.into_iter().collect()))
                .collect();
            let profiling: Vec<(u32, Option<Vec<u8>>)> =
                prof_rids.into_iter().zip(profiling).collect();
            let spec_flags = self
                .completions
                .get(&completion_id)
                .map(|c| c.speculative.clone())
                .unwrap_or_default();
            let mut plan = self.take_send_plan(completion_id)?;
            let partition = plan.partition.clone();
            let encoded: FxHashMap<u32, Option<Vec<u8>>> =
                request_infos.into_iter().collect();
            let profiling: FxHashMap<u32, Option<Vec<u8>>> =
                profiling.into_iter().collect();
            let consumed: FxHashMap<u32, Vec<(String, i64)>> =
                stream_tokens_consumed.into_iter().collect();
            let nested = std::mem::take(&mut plan.nested);

            // One frame per (request, worker): the plan is per edge, and a worker
            // taking several of a request's signals should see one message.
            //
            // Indexed rather than scanned. A Vec probed with `find` is
            // O(edges x groups) with a String compare per probe, and both grow
            // with the batch; the Vec stays only to keep frame order stable.
            let mut grouped: Vec<((u32, Sym), Vec<frames::OutEdge>)> = Vec::new();
            let mut at: FxHashMap<(u32, Sym), usize> = FxHashMap::default();
            {
                // One acquisition for every descriptor lookup in the batch, not
                // one per edge. Released before the sends -- holding it across
                // zmq would park the GPU and plan threads on our I/O.
                let bk = self.bookkeeping.lock().unwrap();
                for w in plan.to_workers {
                    let infos = Self::sliced_infos_locked(&bk, &w.tensors);
                    let key = (w.rid, w.worker);
                    let edge = frames::OutEdge {
                        name: w.signal, next_node: w.next_node,
                        is_streaming: w.streaming, infos,
                        shard_dim: w.shard_dim, total_fanin: w.total_fanin,
                    };
                    match at.get(&key) {
                        Some(&i) => grouped[i].1.push(edge),
                        None => {
                            at.insert(key, grouped.len());
                            grouped.push((key, vec![edge]));
                        }
                    }
                }
            }
            // Encoding is pure CPU, so it all happens under one more acquisition;
            // the dispatches then run with nothing held.
            let mut frames_out: Vec<(Sym, Vec<u8>)> = Vec::with_capacity(grouped.len());
            {
                let bk = self.bookkeeping.lock().unwrap();
                for ((rid, worker), edges) in &grouped {
                    let request_id = self.rids.name(*rid).ok_or_else(|| {
                        PyValueError::new_err(format!("unknown rid handle {rid}"))
                    })?;
                    frames_out.push((
                        *worker,
                        frames::InputSignals {
                            request_id,
                            partition_name: &partition,
                            edges,
                            request_info_encoded: encoded
                                .get(rid).and_then(|b| b.as_deref()),
                        }
                        .encode(bk.strings()),
                    ));
                }
            }
            for (worker, bytes) in &frames_out {
                let name = self.interner.name(*worker).to_string();
                self.dispatch(&name, bytes)?;
            }

            for (rid, signal, modality, refs) in plan.emit {
                let request_id = self.rid_name(rid)?;
                let sig = self.interner.intern(&signal);
                let infos = self.sliced_infos(&refs);
                let loop_indices = nested.get(&rid).cloned();
                if let Some(info) = self.requests[rid as usize].as_mut() {
                    info.pending.output_signals.push(sig);
                    if let Some(idx) = loop_indices.clone() {
                        info.pending.set_loop_indices(sig, idx);
                    }
                }
                let bytes = {
                    let bk = self.bookkeeping.lock().unwrap();
                    frames::ResultTensors {
                        request_id: &request_id,
                        modality: &modality,
                        signal: &signal,
                        infos,
                        loop_indices,
                    }
                    .encode(&self.interner, bk.strings())
                };
                self.dispatch("api_server", &bytes)?;
            }

            for (rid, signal, uuids) in plan.persist {
                let sig = self.interner.intern(&signal);
                if let Some(info) = self.requests[rid as usize].as_mut() {
                    info.pending.persist.push((sig, uuids));
                }
            }
            for (rid, counts) in &new_token_counts {
                if let Some(info) = self.requests[*rid as usize].as_mut() {
                    info.pending.add_new_tokens(counts);
                }
            }

            for (rid, wg_ids, is_first_tp_rank) in plan.completed {
                let bytes = self.worker_graphs_done_frame(
                    rid, &wg_ids, is_first_tp_rank, &partition,
                    consumed.get(&rid).map(|v| v.as_slice()).unwrap_or(&[]),
                    encoded.get(&rid).and_then(|b| b.as_deref()),
                    profiling.get(&rid).and_then(|b| b.as_deref()),
                    spec_flags.get(&rid).copied().unwrap_or(false),
                )?;
                self.dispatch("conductor", &bytes)?;
            }
            Ok(())
        })
    }

    fn get_loop_stop_times(
        &self, rid: u32,
    ) -> Vec<(String, Vec<String>, Vec<(String, u32)>, u32)> {
        let Some(info) = self.info(rid) else { return vec![] };
        info.loop_stop_times
            .iter()
            .map(|(&name, t)| {
                (
                    self.interner.name(name).to_string(),
                    t.loop_name_order
                        .iter()
                        .map(|&n| self.interner.name(n).to_string())
                        .collect(),
                    t.loop_indices
                        .iter()
                        .map(|(&n, &i)| (self.interner.name(n).to_string(), i))
                        .collect(),
                    t.wg_fwd_pass_idx,
                )
            })
            .collect()
    }

    /// Consume a completion and report what to send.
    ///
    /// Returns per rid: the peer edges (grouped by worker), the persist
    /// signals, the emit-to-client edges, and whether a worker graph finished.
    /// Frames are built by the caller -- see stop_loops_batched.
    fn take_send_plan(&mut self, completion_id: u64) -> PyResult<SendPlan> {
        let c = self.completions.remove(&completion_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown completion {completion_id}"))
        })?;
        let g = self.graphs[c.wg as usize].clone();
        let mut plan = SendPlan {
            partition: c.partition.clone(),
            nested: c.nested.clone(),
            ..Default::default()
        };

        let src_node = self.interner.get(&c.node_name);
        // Every rank reports its own persist signals, unsliced: the conductor
        // merges them per source rank and fans them back out, which is how a
        // rank other than 0 ever gets the next walk's inputs.
        for (&rid, signals) in &c.persist {
            for (name, tensors) in signals {
                plan.persist.push((
                    rid,
                    self.interner.name(*name).to_string(),
                    tensors.iter().map(|t| t.uuid).collect(),
                ));
            }
        }
        for (rid, edges) in c.routing {
            let walk = self.interner.get(&c.graph_walk);
            // What the receiver needs to put a sharded arrival back together.
            // Python stamps both in fanout_graph_edges, on the sending side.
            let sharding = self.requests[rid as usize]
                .as_ref()
                .and_then(|i| i.shard.as_ref());
            let wire_shape = |signal: Sym, dest: Sym, streaming: bool, worker: Sym| {
                match (sharding, src_node, walk) {
                    (Some(sh), Some(src), Some(w)) => (
                        sh.shard_dim_of(signal),
                        // The same walk convention the fanout used: a
                        // streaming consumer is resolved walk-independently.
                        sh.fanin(
                            signal, src, w, dest,
                            if streaming { None } else { Some(w) },
                            worker,
                        ),
                    ),
                    _ => (None, 1),
                }
            };
            for e in edges {
                let name = self.interner.name(e.name).to_string();
                match e.dest {
                    Dest::EmitToClient => plan.emit.push((
                        rid,
                        name.clone(),
                        self.interner.name(e.modality).to_string(),
                        e.tensors.clone(),
                    )),
                    Dest::External(dest) => {
                        // `Sym::MAX` is the fanout's marker for a destination
                        // with no sharding group, i.e. one no worker runs.
                        // Caught here rather than at the name lookup, which
                        // would index past the interner and panic ACROSS the
                        // FFI boundary. Python raises on the same graph, from
                        // route_node_outputs, with this message.
                        if e.worker == Some(Sym::MAX) {
                            return Err(PyValueError::new_err(format!(
                                "Output edge targets unknown node/graph walk: \
                                 {}. Check graph construction.",
                                self.interner.name(dest),
                            )));
                        }
                        // Who runs that node for THIS request.
                        let workers = if let Some(w) = e.worker {
                            vec![w]
                        } else {
                            walk
                                .and_then(|w| {
                                    self.info(rid)?.node_to_workers.get(&(dest, w)).cloned()
                                })
                                .unwrap_or_default()
                        };
                        // Skip ourselves when the node lives in a local worker
                        // graph: complete_and_route_batch already ingested it
                        // there, and a self-send would arrive for a slot that
                        // is already full.
                        // Same rule as the routing side: a streaming consumer
                        // counts as ours whatever walk it lives in.
                        let mine = if e.streaming {
                            self.graphs.iter().any(|g| g.by_name.contains_key(&dest))
                        } else {
                            walk.map(|w| self.node_owner.contains_key(&(w, dest)))
                                .unwrap_or(false)
                        };
                        let me_sym = self.shard.me;
                        for worker in workers {
                            if mine && worker == me_sym {
                                continue;
                            }
                            let (shard_dim, total_fanin) =
                                wire_shape(e.name, dest, e.streaming, worker);
                            plan.to_workers.push(WireEdge {
                                rid,
                                worker,
                                signal: name.clone(),
                                next_node: self.interner.name(dest).to_string(),
                                tensors: e.tensors.clone(),
                                streaming: e.streaming,
                                shard_dim,
                                total_fanin,
                            });
                        }
                    }
                    // Refused locally, so send it to the node's owner like
                    // Python does. `mine` is not consulted: the point is that
                    // our own copy did not take it.
                    Dest::Local(d) if e.declined_local => {
                        let dest = g.node(d).name;
                        let workers = walk
                            .and_then(|w| {
                                self.info(rid)?.node_to_workers.get(&(dest, w)).cloned()
                            })
                            .unwrap_or_default();
                        for worker in workers {
                            let (shard_dim, total_fanin) =
                                wire_shape(e.name, dest, e.streaming, worker);
                            plan.to_workers.push(WireEdge {
                                rid,
                                worker,
                                signal: name.clone(),
                                next_node: self.interner.name(dest).to_string(),
                                tensors: e.tensors.clone(),
                                streaming: e.streaming,
                                shard_dim,
                                total_fanin,
                            });
                        }
                    }
                    Dest::Local(_) | Dest::Empty => {}
                }
            }
            if let Some(wgs) = c.completed_wgs.get(&rid) {
                let _ = &g;
                plan.completed.push((
                    rid,
                    wgs.clone(),
                    c.first_tp_rank.get(&rid).copied().unwrap_or(true),
                ));
            }
        }
        Ok(plan)
    }

    fn stream_partition_done(&self, rid: u32, partition: &str) -> bool {
        let Some(p) = self.interner.get(partition) else { return false };
        self.info(rid).is_some_and(|i| i.stream_done(p))
    }

    /// Parked routing awaiting a send. Non-zero between
    /// complete_and_route_batch and send_outputs; a number that only grows
    /// means sends are being abandoned.
    fn num_parked_completions(&self) -> usize {
        self.completions.len()
    }

    fn num_handles(&self) -> usize {
        self.rids.len()
    }

    fn set_speculatively_scheduled(
        &mut self, node: String,
        wg_id: u32, rids: Vec<u32>,
        speculatively_scheduled: bool,
    ) -> PyResult<()> {
        let wg = self.wg_index(wg_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown worker graph id {wg_id}"))
        })?;
        // NOT interner.get(): that is the STRING id, and set_spec_scheduled
        // wants the node's index within this worker graph. The two id spaces
        // are both u32, so only the Option here made the mix-up visible.
        let node_id = self.nid(wg, &node).ok_or_else(|| {
            PyValueError::new_err(format!(
                "node {node:?} is not in worker graph {wg_id}"
            ))
        })?;
        for rid in rids {
            if let Some(state) = self.state_mut(wg, rid) {
                state.set_spec_scheduled(node_id, speculatively_scheduled);
            }
        }
        Ok(())
    }

    /// Whether this rid's node is currently marked speculatively scheduled.
    ///
    /// Exposed so the invariant that the flag SURVIVES completion is
    /// assertable from the parity tests: `State::complete` used to clear it,
    /// which let `refresh_ready` re-add a node whose rids were still in
    /// flight. Python holds the same state in
    /// `GraphNode._speculatively_scheduled`.
    fn is_speculatively_scheduled(
        &self, node: String, wg_id: u32, rid: u32,
    ) -> PyResult<bool> {
        let wg = self.wg_index(wg_id).ok_or_else(|| {
            PyValueError::new_err(format!("unknown worker graph id {wg_id}"))
        })?;
        let node_id = self.nid(wg, &node).ok_or_else(|| {
            PyValueError::new_err(format!(
                "node {node:?} is not in worker graph {wg_id}"
            ))
        })?;
        Ok(self
            .state(wg, rid)
            .map(|s| s.is_spec_scheduled(node_id))
            .unwrap_or(false))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_are_recycled_and_the_slot_is_clean() {
        // The hazard this guards: a leaked handle-keyed entry does not read as
        // stale, it reads as the NEXT request inheriting another's state.
        let mut rids = InternedRids::new();
        let (a, grew_a) = rids.intern("req-a");
        assert!(grew_a, "first handle has to grow the per-handle vectors");
        assert_eq!(rids.handle("req-a"), Some(a));
        assert_eq!(rids.name(a), Some("req-a"));

        assert!(rids.release(a));
        assert_eq!(rids.handle("req-a"), None, "the string must not resolve");
        assert_eq!(rids.name(a), None, "and the slot must be empty");

        let (b, grew_b) = rids.intern("req-b");
        assert_eq!(b, a, "the freed handle is reused");
        assert!(!grew_b, "reuse must NOT grow the vectors");
        assert_eq!(rids.name(b), Some("req-b"));
        assert_eq!(rids.len(), 1);
    }

    #[test]
    fn interning_the_same_rid_twice_is_stable() {
        // add_request runs once per partition, so this happens on every
        // multi-partition request.
        let mut rids = InternedRids::new();
        let (a, grew_a) = rids.intern("req");
        let (b, grew_b) = rids.intern("req");
        assert_eq!(a, b);
        assert!(grew_a && !grew_b, "only the first call allocates a slot");
    }

    #[test]
    fn releasing_twice_is_idempotent() {
        // A REMOVE can arrive for a rid this rank already tore down; a second
        // release must not hand the same handle to the free list twice, or two
        // future requests would share it.
        let mut rids = InternedRids::new();
        let (a, _) = rids.intern("req");
        assert!(rids.release(a));
        assert!(!rids.release(a));

        let (b, _) = rids.intern("x");
        let (c, _) = rids.intern("y");
        assert_ne!(b, c, "a double free would have aliased these");
    }

    #[test]
    fn releasing_an_unknown_handle_is_ignored() {
        let mut rids = InternedRids::new();
        assert!(!rids.release(42));
    }
}
