//! Per-request walk state — the mutable half of Python's GraphNode / Loop /
//! GraphStateRegistry objects, as one flat allocation per request.
//!
//! Field-for-field with Python except where noted:
//!   Loop + its LoopStateRegistry merge into one `LoopState` (strictly 1:1).
//!   `_managing_registry` pointers become `loop_id` / `parent` indices.
//!   `WorkerGraphStateRegistry.reset_for_iter` and its `ready_next_iter` /
//!   `ready_streaming_next_iter` sets are NOT ported: they are dead
//!   (reset_for_iter is only ever called on a LoopStateRegistry).

use crate::graph::spec::{Dest, EdgeSpec, GraphRef, LoopId, NodeId, Sym};
use serde::Serialize;

/// The only per-edge payload that crosses the boundary. `uuid` is a dense u64
/// handle — today's `str(uuid4())` would be interned at the tensor_manager seam.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct TensorRef {
    pub uuid: u64,
    pub dim0: i64,
    pub nbytes: i64,
    pub offset: i64,
}

/// Python's `ReadySignals`: which inputs have arrived, and their tensors.
/// `ready_names` is the mask; `is_ready` / `is_ready_for_streaming` are derived.
#[derive(Clone)]
struct Slot {
    mask: u64,
    /// Python's `GraphEdge._final_stream_chunk`, per input slot: the
    /// consuming pass reports the partition done, so it has to survive the
    /// ingest.
    final_chunk: u64,
    tensors: Vec<Option<Vec<TensorRef>>>,
}

impl Slot {
    fn new(n: usize) -> Self {
        Slot { mask: 0, final_chunk: 0, tensors: vec![None; n] }
    }
    fn clear(&mut self) {
        self.mask = 0;
        self.final_chunk = 0;
        for t in self.tensors.iter_mut() {
            *t = None;
        }
    }
    fn set(&mut self, slot: u8, t: Vec<TensorRef>, final_chunk: bool) {
        self.mask |= 1 << slot;
        if final_chunk {
            self.final_chunk |= 1 << slot;
        } else {
            self.final_chunk &= !(1 << slot);
        }
        self.tensors[slot as usize] = Some(t);
    }
    fn take(&mut self, slot: u8) -> Option<Vec<TensorRef>> {
        self.mask &= !(1 << slot);
        self.final_chunk &= !(1 << slot);
        self.tensors[slot as usize].take()
    }
    fn is_final_chunk(&self, slot: u8) -> bool {
        self.final_chunk >> slot & 1 == 1
    }
    fn has(&self, slot: u8) -> bool {
        self.mask >> slot & 1 == 1
    }
}

#[derive(Clone)]
struct NodeState {
    cur: Slot,       // ready_signals
    next: Slot,      // ready_next_iter
    spec: Slot,      // speculative_signals
    completed: bool,
    scheduled: bool, // _speculatively_scheduled
}

impl NodeState {
    fn new(n: usize) -> Self {
        NodeState {
            cur: Slot::new(n), next: Slot::new(n), spec: Slot::new(n),
            completed: false, scheduled: false,
        }
    }
    fn clear(&mut self) {
        self.cur.clear();
        self.next.clear();
        self.spec.clear();
        self.completed = false;
        self.scheduled = false;
    }
}

#[derive(Clone, Default)]
struct LoopState {
    curr_iter: u32,          // Loop.curr_iter
    finish_signal: bool,     // Loop._finish_signal
    done: bool,              // Loop.is_done
    completed_entities: u32, // inner registry's _num_completed_entities
    cached: Vec<(Sym, Vec<TensorRef>)>, // _cached_outputs (extends within an iter)
    accum: Vec<(Sym, Vec<TensorRef>)>,  // _accumulated_cache (extends across iters)
    ext_inputs: Vec<(Sym, NodeId, Vec<TensorRef>)>, // _ingested_external_inputs
    ext_names: Vec<Sym>,                            // _ingested_external_input_names
}

impl LoopState {
    /// Python's `Loop._reset_metadata`.
    fn reset_metadata(&mut self) {
        self.ext_inputs.clear();
        self.ext_names.clear();
        self.curr_iter = 0;
        self.finish_signal = false;
        self.done = false;
    }
}

/// An edge the state machine routed outward. Which *worker* it goes to is
/// `shard.rs`'s job; this is just "what came out".
#[derive(Clone)]
pub struct RoutedEdge {
    pub name: Sym,
    pub dest: Dest,
    pub dest_sym: Sym,
    pub persist: bool,
    pub new_token: bool,
    pub streaming: bool,
    pub modality: Sym,
    pub tensors: Vec<TensorRef>,
    /// Python's `GraphEdge._persist_for_loop`: a loop's saved external input
    /// being re-injected; the tensor manager must not dereference it.
    pub persist_for_loop: bool,
    /// A `Dest::Local` consumer refused the edge (its slots were full), so it
    /// goes out on the wire instead -- staged, counted and sent like any
    /// remote edge, as Python does.
    pub declined_local: bool,
    pub worker: Option<Sym>,
}

fn routed(e: &EdgeSpec, tensors: Vec<TensorRef>) -> RoutedEdge {
    RoutedEdge {
        name: e.name, dest: e.dest, dest_sym: e.dest_sym, persist: e.persist,
        new_token: e.new_token, streaming: e.streaming, modality: e.modality,
        tensors, persist_for_loop: false, declined_local: false,
        worker: None,
    }
}

/// Python's `SpeculativeNodeInfo`.
pub struct SpecNode {
    pub node: NodeId,
    pub is_new_loop_iter: bool,
    pub loop_id: Option<LoopId>,
}

pub struct RequestState {
    pub graph: GraphRef,
    nodes: Vec<NodeState>,
    loops: Vec<LoopState>,
    /// WorkerGraphStateRegistry.ready_names / .ready_for_streaming as bitsets.
    pub ready: Vec<u64>,
    pub ready_streaming: Vec<u64>,
    root_completed: u32,
    pub is_done: bool,
    pub num_times_run: u32, // WorkerGraphIO.num_times_run
    spec_dirty: Vec<NodeId>, // _nodes_with_speculative_inputs
}

impl RequestState {
    pub fn new(graph: GraphRef) -> Self {
        let nodes = graph.nodes.iter().map(|n| NodeState::new(n.inputs.len())).collect();
        let loops = vec![LoopState::default(); graph.loops.len()];
        let w = graph.n_ready_words();
        let mut s = Self {
            nodes, loops, ready: vec![0; w], ready_streaming: vec![0; w],
            root_completed: 0, is_done: false, num_times_run: 0,
            spec_dirty: Vec::new(), graph,
        };
        s.seed_streaming_ready();
        s
    }

    /// Nodes whose inputs are all streaming start ready-for-streaming
    /// (Python's `only_streaming_inputs`).
    fn seed_streaming_ready(&mut self) {
        for (i, n) in self.graph.nodes.iter().enumerate() {
            if n.only_streaming {
                self.ready_streaming[i / 64] |= 1 << (i % 64);
            }
        }
    }

    // -- ready bitsets -------------------------------------------------------

    #[inline]
    fn bit(id: NodeId) -> (usize, u32) {
        ((id / 64) as usize, id % 64)
    }

    #[inline]
    /// Python's `ReadySignals.is_ready_for_streaming`: every non-streaming
    /// input has arrived, so a streaming chunk can be taken now.
    pub fn is_ready_for_streaming(&self, id: NodeId) -> bool {
        let (w, b) = Self::bit(id);
        self.ready_streaming[w] >> b & 1 == 1
    }

    pub fn is_ready(&self, id: NodeId) -> bool {
        let (w, b) = Self::bit(id);
        self.ready[w] >> b & 1 == 1
    }

    /// Recompute both ready bits for a node from its current slot. Replaces
    /// `WorkerGraphStateRegistry.register_ingested_input`'s branchy updates.
    fn refresh_ready(&mut self, id: NodeId) {
        let spec = self.graph.node(id);
        let st = &self.nodes[id as usize];
        let live = !st.completed;
        let ready = st.cur.mask == spec.full_mask && live;
        let (w, b) = Self::bit(id);
        // `scheduled` gates the ADD only, never a removal. Python's
        // register_ingested_input skips the queue add for a
        // speculatively-scheduled node -- so it does not get double-queued
        // while its spec batch runs -- but a node already queued stays
        // queued. Folding the flag into the predicate (`live`) instead made
        // marking a node withdraw it from the ready set, and unmarking put it
        // back. The streaming half of this rule lives in
        // note_ingested_for_streaming.
        if ready {
            if !st.scheduled {
                self.ready[w] |= 1 << b
            }
        } else {
            self.ready[w] &= !(1 << b)
        }
    }

    /// Python's `ReadySignals.is_ready_for_streaming`, folded into the
    /// registry set `register_ingested_input` maintains.
    ///
    /// "Streaming-ready" means every input still missing is a streaming one,
    /// which arrives incrementally -- so the node can run on what it has.
    /// Python expressed that with an `issuperset` that was a tautology (both
    /// operands are subsets of input_names), which made any node with any
    /// input at all read as streaming-ready; fixed there, mirrored here.
    fn note_ingested_for_streaming(&mut self, id: NodeId) {
        let spec = self.graph.node(id);
        let st = &self.nodes[id as usize];
        // The mask alone, no liveness: Python's `ReadySignals.is_ready`.
        let full = st.cur.mask == spec.full_mask;
        // input_names ⊆ ready_names ∪ streaming_inputs, as bits: nothing is
        // missing except streaming slots.
        let only_streaming_missing =
            spec.full_mask & !st.cur.mask & !spec.streaming_mask == 0;
        let sched = st.scheduled;
        let (w, b) = Self::bit(id);
        if full {
            self.ready_streaming[w] &= !(1 << b);
        } else if only_streaming_missing && !sched {
            // Only the add is gated on _speculatively_scheduled, as in Python.
            self.ready_streaming[w] |= 1 << b;
        }
    }

    /// Back to seeded membership, for the two paths where Python rebuilds the
    /// whole set: `reset_for_iter` and `clear`.
    fn reseed_streaming_ready(&mut self, id: NodeId) {
        let only = self.graph.node(id).only_streaming;
        let (w, b) = Self::bit(id);
        if only {
            self.ready_streaming[w] |= 1 << b;
        } else {
            self.ready_streaming[w] &= !(1 << b);
        }
    }

    // -- ingest --------------------------------------------------------------

    /// Python's `GraphNode.ingest_input` + the registry chain's
    /// `register_ingested_input`. Returns false when the edge is rejected
    /// (both slots already hold this input), i.e. "try the next destination".
    pub fn ingest(
        &mut self, node: NodeId, slot: u8, tensors: Vec<TensorRef>,
        can_buffer: bool, final_chunk: bool,
    ) -> bool {
        let name = self.graph.node(node).inputs[slot as usize];
        {
            let st = &mut self.nodes[node as usize];
            if !st.cur.has(slot) {
                st.cur.set(slot, tensors.clone(), final_chunk);
            } else if can_buffer && !st.next.has(slot) {
                st.next.set(slot, tensors.clone(), final_chunk);
            } else {
                return false;
            }
        }
        // An ingest bubbles through every enclosing loop, each recording it if
        // it is external at that level (Loop.ingest_external_input recursing
        // via _managing_registry), then reaches the root's ready sets.
        self.record_external(node, name, &tensors);
        self.refresh_ready(node);
        self.note_ingested_for_streaming(node);
        true
    }

    fn record_external(&mut self, node: NodeId, name: Sym, t: &[TensorRef]) {
        let graph = self.graph.clone();
        let mut cur = graph.node(node).loop_id;
        while let Some(lid) = cur {
            let ls = graph.lp(lid);
            if ls.external_inputs.iter().any(|(n, d)| *n == name && *d == node) {
                let st = &mut self.loops[lid as usize];
                if !st.ext_names.contains(&name) {
                    st.ext_names.push(name);
                    st.ext_inputs.push((name, node, t.to_vec()));
                }
            }
            cur = ls.parent;
        }
    }

    /// Python's `ReadySignals.remove` — speculation rollback. No dereference.
    pub fn remove_input(&mut self, node: NodeId, slot: u8, from_next: bool) -> Option<Vec<TensorRef>> {
        let st = &mut self.nodes[node as usize];
        let t = if from_next { st.next.take(slot) } else { st.cur.take(slot) };
        self.refresh_ready(node);
        t
    }

    /// Whether a slot is already filled, for inferring which slot an ingest
    /// will land in before making it.
    pub fn has_input(&self, node: NodeId, slot: u8, next_iter: bool) -> bool {
        let st = &self.nodes[node as usize];
        if next_iter { st.next.has(slot) } else { st.cur.has(slot) }
    }

    pub fn input_tensors(
        &self, node: NodeId, next_iter: bool,
    ) -> Vec<(Sym, Vec<TensorRef>, bool)> {
        let spec = self.graph.node(node);
        let st = &self.nodes[node as usize];
        let slot = if next_iter { &st.next } else { &st.cur };
        spec.inputs.iter().enumerate()
            .filter_map(|(i, &n)| slot.tensors[i].as_ref().map(
                |t| (n, t.clone(), slot.is_final_chunk(i as u8))))
            .collect()
    }

    /// Unconditional, as Python's `pop_ready_nodes` is. Gating on readiness
    /// dropped the rid from a batch the caller had already committed to.
    pub fn take_for_schedule(&mut self, node: NodeId) {
        let (w, b) = Self::bit(node);
        self.ready[w] &= !(1 << b);
    }

    /// Undo `take_for_schedule`: put the node back in the ready set.
    ///
    /// Unconditional, as Python's `ready_node_names.add(node_name)` is. A
    /// recompute is wrong here: cleanup_consumed_inputs normally runs before
    /// the push back, so the input slots are already empty and the node would
    /// silently stay unready -- losing the batch the caller is handing back.
    pub fn push_back(&mut self, node: NodeId) {
        let (w, b) = Self::bit(node);
        self.ready[w] |= 1 << b;
    }

    /// Python's `ReadySignals.clear` on the CURRENT slot: release the inputs
    /// the just-executed node consumed.
    ///
    /// Returns the uuids to dereference. A loop's external inputs are held
    /// for re-injection on the next iteration (Python's `_persist_for_loop`),
    /// so they are excluded -- and that is structural, from the spec's
    /// `external_inputs`, not per-request state.
    pub fn clear_consumed_inputs(&mut self, node: NodeId) -> Vec<u64> {
        let graph = self.graph.clone();
        let spec = graph.node(node);
        let held: Vec<Sym> = {
            let mut names = Vec::new();
            let mut cur = spec.loop_id;
            while let Some(lid) = cur {
                let ls = graph.lp(lid);
                names.extend(
                    ls.external_inputs
                        .iter()
                        .filter(|(_, d)| *d == node)
                        .map(|(n, _)| *n),
                );
                cur = ls.parent;
            }
            names
        };

        let st = &mut self.nodes[node as usize];
        let mut freed = Vec::new();
        for (i, name) in spec.inputs.iter().enumerate() {
            if held.contains(name) {
                continue;
            }
            if let Some(tensors) = st.cur.tensors[i].take() {
                freed.extend(tensors.iter().map(|t| t.uuid));
            }
            st.cur.mask &= !(1 << i);
            st.cur.final_chunk &= !(1 << i);
        }
        self.refresh_ready(node);
        freed
    }

    /// The flag alone: it gates future ingests, and touching readiness here
    /// is what made marking withdraw an already-ready node.
    pub fn set_spec_scheduled(&mut self, node: NodeId, on: bool) {
        self.nodes[node as usize].scheduled = on;
    }

    pub fn is_spec_scheduled(&self, node: NodeId) -> bool {
        self.nodes[node as usize].scheduled
    }

    // -- completion ----------------------------------------------------------

    /// Python's `WorkerGraphIO.mark_node_complete`. Returns the edges to route,
    /// with loop-back edges of any finishing loop already filtered out.
    /// Third return: uuids this cleared and the caller must dereference.
    ///
    /// Python's `mark_entity_complete` clears a top-level entity's ready
    /// signals THROUGH the tensor manager, so the dereference happens whether
    /// or not the caller already cleaned up. Clearing here without reporting
    /// them would leak a reference on any path where completion precedes the
    /// explicit cleanup.
    pub fn complete(
        &mut self, node: NodeId, out_tensors: &[Vec<TensorRef>],
    ) -> (Vec<RoutedEdge>, Vec<(Sym, NodeId)>, Vec<u64>) {
        let graph = self.graph.clone();
        let spec = graph.node(node);
        let prev_done = self.is_done;
        let mut out: Vec<RoutedEdge> = Vec::with_capacity(spec.outputs.len() + 2);

        let mut freed: Vec<u64> = Vec::new();
        {
            let st = &mut self.nodes[node as usize];
            // NOTE: this cannot clear .scheduled, or else refresh_ready re-adds
            // a node while its rids are still in flight as a speculative batch.
            st.completed = true;
            
            // A loop member keeps `cur` until its loop advances, so later
            // loop-back arrivals land in `next` — Python only clears
            // ready_signals for top-level entities (base.py:758-767).
            if spec.loop_id.is_none() {
                for t in st.cur.tensors.iter().flatten() {
                    freed.extend(t.iter().map(|x| x.uuid));
                }
                st.cur.clear();
            }
        }
        self.refresh_ready(node);

        for (i, e) in spec.outputs.iter().enumerate() {
            out.push(routed(e, out_tensors.get(i).cloned().unwrap_or_default()));
        }

        let mut filtered: Vec<(Sym, NodeId)> = Vec::new();
        match spec.loop_id {
            None => self.root_entity_done(),
            Some(lid) => self.entity_completed(lid, 0, &mut out, &mut filtered),
        }

        if self.is_done && !prev_done {
            self.num_times_run += 1;
        }
        (out, filtered, freed)
    }

    fn root_entity_done(&mut self) {
        if self.is_done {
            return;
        }
        self.root_completed += 1;
        self.is_done = self.root_completed == self.graph.n_root_entities();
    }

    /// One entity of loop `lid` finished; `out[edges_from..]` are its outputs.
    /// Mirrors `LoopStateRegistry.mark_entity_complete` + `Loop.complete_iter`.
    fn entity_completed(
        &mut self, lid: LoopId, edges_from: usize,
        out: &mut Vec<RoutedEdge>, filtered: &mut Vec<(Sym, NodeId)>,
    ) {
        let graph = self.graph.clone();
        let lspec = graph.lp(lid);
        self.cache_outputs(lid, edges_from, out);

        {
            let st = &mut self.loops[lid as usize];
            st.completed_entities += 1;
            if st.completed_entities < lspec.n_entities() {
                return;
            }
            st.completed_entities = 0;
        }

        let finishing = {
            let st = &self.loops[lid as usize];
            st.finish_signal || lspec.max_iters == st.curr_iter + 1
        };
        if finishing {
            self.finish_loop(lid, out, filtered);
        } else {
            self.advance_loop(lid, out);
        }
    }

    /// Python's `Loop.maybe_cache_output`: snapshot tensor_info for any edge
    /// matching a declared loop output, deduped by name. Regular outputs
    /// *extend* within an iteration (cleared on advance); accumulated outputs
    /// extend across iterations.
    fn cache_outputs(&mut self, lid: LoopId, from: usize, out: &[RoutedEdge]) {
        let graph = self.graph.clone();
        let lspec = graph.lp(lid);
        let mut seen: Vec<Sym> = Vec::new();
        for e in &out[from..] {
            if seen.contains(&e.name) {
                continue;
            }
            seen.push(e.name);
            let st = &mut self.loops[lid as usize];
            let bucket = if lspec.output_names.contains(&e.name) {
                &mut st.cached
            } else if lspec.accum_names.contains(&e.name) {
                &mut st.accum
            } else {
                continue;
            };
            match bucket.iter_mut().find(|(n, _)| *n == e.name) {
                Some((_, v)) => v.extend_from_slice(&e.tensors),
                None => bucket.push((e.name, e.tensors.clone())),
            }
        }
    }

    fn finish_loop(
        &mut self, lid: LoopId, out: &mut Vec<RoutedEdge>,
        filtered: &mut Vec<(Sym, NodeId)>,
    ) {
        let graph = self.graph.clone();
        let lspec = graph.lp(lid);
        self.loops[lid as usize].done = true;

        // Drop loop-back edges: the loop is over, nothing re-enters it.
        let lb = &lspec.loop_back;
        out.retain(|e| !matches!(e.dest, Dest::Local(d)
            if lb.iter().any(|(n, dn)| *n == e.name && *dn == d)));
        // Python surfaces these as NodeCompletionOutput.filtered_signals.
        filtered.extend_from_slice(lb);

        // Clear the body subtree. The loop's OWN curr_iter survives until a
        // parent advance (Python: inner_registry.clear() doesn't touch it).
        self.clear_subtree(lid);

        let (cached, accum) = {
            let st = &mut self.loops[lid as usize];
            (std::mem::take(&mut st.cached), std::mem::take(&mut st.accum))
        };
        let own_edges_from = out.len();
        for o in &lspec.outputs {
            let t = cached.iter().find(|(n, _)| *n == o.name)
                .map(|(_, t)| t.clone()).unwrap_or_default();
            out.push(routed(o, t));
        }
        for o in &lspec.accumulated {
            let t = accum.iter().find(|(n, _)| *n == o.name)
                .map(|(_, t)| t.clone()).unwrap_or_default();
            out.push(routed(o, t));
        }

        // Cascade. NOTE two deliberate divergences from Python, both of which
        // only bite for NESTED loops (see README 7.5):
        //   * Python cascades BEFORE populating its own outputs, so a parent
        //     caches the child's still-empty tensor_info. We populate first.
        //   * Python discards the cascade's return value, so a parent that
        //     finishes in the same event never routes its outputs. We append
        //     to the same `out`.
        match lspec.parent {
            None => self.root_entity_done(),
            Some(p) => self.entity_completed(p, own_edges_from, out, filtered),
        }
    }

    fn advance_loop(&mut self, lid: LoopId, out: &mut Vec<RoutedEdge>) {
        let graph = self.graph.clone();
        {
            let st = &mut self.loops[lid as usize];
            st.curr_iter += 1;
            st.cached.clear(); // Loop._uncache_outputs
        }
        self.reset_subtree_for_iter(lid);

        // Emit the loop's saved external inputs for the caller to route, like
        // Python's `complete_iter` returning `_ingested_external_inputs`.
        // Ingesting them here instead would double-ingest on the per-event
        // path, where the caller mirrors Python and routes them too.
        let _ = &graph;
        let ext = self.loops[lid as usize].ext_inputs.clone();
        for (name, dest, t) in ext {
            out.push(RoutedEdge {
                name, dest: Dest::Local(dest),
                // The dest node's interned NAME, not its NodeId: `dest_sym`
                // is what the sharding fanout looks a group up by, and the
                // two are different namespaces that happen to share a repr.
                dest_sym: graph.node(dest).name,
                persist: false, new_token: false,
                streaming: false, modality: 0, tensors: t,
                persist_for_loop: true, declined_local: false,
                worker: None,
            });
        }
    }

    /// Python's `GraphStateRegistry.reset_for_iter` on a loop's inner registry:
    /// member nodes promote next-iter slots; child loops reset recursively AND
    /// have their metadata cleared (`Loop.reset_for_outer_iter`).
    fn reset_subtree_for_iter(&mut self, lid: LoopId) {
        let graph = self.graph.clone();
        let lspec = graph.lp(lid);
        for &n in &lspec.member_nodes {
            {
                let st = &mut self.nodes[n as usize];
                st.cur.clear();
                std::mem::swap(&mut st.cur, &mut st.next);
                st.completed = false;
            }
            self.refresh_ready(n);
            self.reseed_streaming_ready(n);
        }
        for &c in &lspec.child_loops {
            self.reset_subtree_for_iter(c);
            let st = &mut self.loops[c as usize];
            st.completed_entities = 0;
            st.reset_metadata();
        }
    }

    /// Python's `GraphStateRegistry.clear()` on a loop's inner registry:
    /// members fully cleared; child loops cleared and metadata reset
    /// (`Loop.clear`, which also drops the accumulated cache).
    fn clear_subtree(&mut self, lid: LoopId) {
        let graph = self.graph.clone();
        let lspec = graph.lp(lid);
        for &n in &lspec.member_nodes {
            self.nodes[n as usize].clear();
            self.refresh_ready(n);
            self.reseed_streaming_ready(n);
        }
        for &c in &lspec.child_loops {
            self.clear_subtree(c);
            let st = &mut self.loops[c as usize];
            st.completed_entities = 0;
            st.cached.clear();
            st.accum.clear();
            st.reset_metadata();
        }
    }

    // -- loops ---------------------------------------------------------------

    /// Python's `register_loop_finish_signal`; caller drops the returned
    /// loop-back (name, dest) pairs from this iteration's routing.
    pub fn register_loop_finish(&mut self, lid: LoopId) -> &[(Sym, NodeId)] {
        self.loops[lid as usize].finish_signal = true;
        &self.graph.lp(lid).loop_back
    }

    pub fn loop_iter(&self, lid: LoopId) -> u32 {
        self.loops[lid as usize].curr_iter
    }

    pub fn loop_finished(&self, lid: LoopId) -> bool {
        self.loops[lid as usize].done
    }

    pub fn loop_indices(&self) -> Vec<u32> {
        self.loops.iter().map(|l| l.curr_iter).collect()
    }

    // -- speculation ---------------------------------------------------------

    /// Python's `WorkerGraphIO.ingest_for_speculation`.
    pub fn ingest_for_speculation(&mut self, source: NodeId, edges: &[RoutedEdge]) -> Vec<SpecNode> {
        let graph = self.graph.clone();
        let src_loop = graph.node(source).loop_id;
        let mut dests: Vec<NodeId> = Vec::new();
        let mut next_iter: Vec<NodeId> = Vec::new();

        for e in edges {
            let Dest::Local(d) = e.dest else { continue };
            let Some(slot) = graph.node(d).slot_of(e.name) else { continue };
            self.nodes[d as usize].spec.set(slot, e.tensors.clone(), false);
            if !self.spec_dirty.contains(&d) {
                self.spec_dirty.push(d);
            }
            if !dests.contains(&d) {
                dests.push(d);
            }
            if let Some(lid) = src_loop {
                if graph.lp(lid).loop_back.iter().any(|(n, dn)| *n == e.name && *dn == d)
                    && !next_iter.contains(&d)
                {
                    next_iter.push(d);
                }
            }
        }

        dests.into_iter()
            .filter(|&d| self.ready_for_speculation(d, d == source, true))
            .map(|d| SpecNode {
                node: d,
                is_new_loop_iter: next_iter.contains(&d),
                loop_id: graph.node(d).loop_id,
            })
            .collect()
    }

    /// Python's `GraphNode.is_ready_for_speculation`.
    pub fn ready_for_speculation(&self, node: NodeId, check_next_iter: bool, allow_streaming: bool) -> bool {
        let spec = self.graph.node(node);
        let mut needed = spec.full_mask;
        if allow_streaming {
            needed &= !spec.streaming_mask;
        }
        let st = &self.nodes[node as usize];
        let have = if check_next_iter { st.next.mask } else { st.cur.mask } | st.spec.mask;
        needed & !have == 0
    }

    pub fn clear_speculative_inputs(&mut self) {
        for n in std::mem::take(&mut self.spec_dirty) {
            self.nodes[n as usize].spec.clear();
        }
    }

    // -- lifecycle -----------------------------------------------------------

    /// Python's `WorkerGraphIO.clear()` — end of a full forward pass.
    pub fn reset(&mut self) {
        for (i, n) in self.nodes.iter_mut().enumerate() {
            let _ = i;
            n.clear();
        }
        for l in self.loops.iter_mut() {
            *l = LoopState::default();
        }
        for w in self.ready.iter_mut() { *w = 0 }
        for w in self.ready_streaming.iter_mut() { *w = 0 }
        self.root_completed = 0;
        self.is_done = false;
        self.spec_dirty.clear();
        self.seed_streaming_ready();
    }
}
