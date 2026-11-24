//! Per-request walk state — the mutable half of Python's GraphNode / Loop /
//! GraphStateRegistry objects, as one pipeline of `PassState`s per request.
//!
//! Field-for-field with Python except where noted:
//!   Loop + its LoopStateRegistry merge into one `LoopState` (strictly 1:1).
//!   `_managing_registry` pointers become `loop_id` / `parent` indices.
//!   Each active pass of a `(wg, rid)` lives in its own `PassState` in a
//!   `VecDeque<PassState>`, enforcing per-node FIFO pass ordering while letting
//!   upstream entities pipeline ahead without losing `next`-pass inputs on pass
//!   retirement.

use std::collections::VecDeque;

use crate::graph::spec::{
    CompiledGraph, Dest, EdgeSpec, GraphRef, LoopId, NodeId, Sym,
};
use rustc_hash::FxHashMap;
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
}

impl NodeState {
    fn new(n: usize) -> Self {
        NodeState {
            cur: Slot::new(n), next: Slot::new(n), spec: Slot::new(n),
            completed: false,
        }
    }
    fn clear(&mut self) {
        self.cur.clear();
        self.next.clear();
        self.spec.clear();
        self.completed = false;
    }
}

#[derive(Clone, Default)]
struct LoopState {
    max_iters: u32,          // Loop.max_iters, this request's copy
    curr_iter: u32,          // Loop.curr_iter
    finish_signal: bool,     // Loop._finish_signal
    done: bool,              // Loop.is_done
    completed_mask: Vec<u64>,
    completed_count: u32,
    cached: Vec<(Sym, Vec<TensorRef>)>, // _cached_outputs (extends within an iter)
    accum: Vec<(Sym, Vec<TensorRef>)>,  // _accumulated_cache (extends across iters)
    ext_inputs: Vec<(Sym, NodeId, Vec<TensorRef>)>, // _ingested_external_inputs
    ext_names: Vec<Sym>,                            // _ingested_external_input_names
}

impl LoopState {
    fn new(max_iters: u32, mask_words: usize) -> Self {
        Self {
            max_iters,
            completed_mask: vec![0; mask_words],
            ..Self::default()
        }
    }

    /// Python's `Loop._reset_metadata`.
    fn reset_metadata(&mut self) {
        self.ext_inputs.clear();
        self.ext_names.clear();
        self.curr_iter = 0;
        self.finish_signal = false;
        self.done = false;
        self.completed_mask.fill(0);
        self.completed_count = 0;
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
    /// A local consumer refused the edge (its slots were full), so it
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

/// What `RequestState::complete` hands back.
pub struct Completed {
    /// The edges to route, loop-back edges of a finishing loop already
    /// filtered out.
    pub edges: Vec<RoutedEdge>,
    /// Uuids to dereference: inputs the completion cleared and tensors a
    /// loop reset or un-cached, as Python dereferences them.
    pub freed: Vec<u64>,
    /// Uuids to increment: tensors a loop cached, as `maybe_cache_output`
    /// takes a reference on each. Apply before `freed`, which can name the
    /// same tensor.
    pub taken: Vec<u64>,
    /// Finished-loop cache holds to exchange for routed readers AFTER fanout.
    pub transferred: Vec<u64>,
    /// The `(walk, walk_pass_id)` of the pass in which this node completed.
    pub pass_walk: Sym,
    pub walk_pass_id: u32,
}

/// Python's `SpeculativeNodeInfo`.
pub struct SpecNode {
    pub node: NodeId,
    pub is_new_loop_iter: bool,
    pub loop_id: Option<LoopId>,
}

/// One active pass of a `(worker_graph, rid)`.
#[derive(Clone)]
struct PassState {
    walk: Sym,
    walk_pass_id: u32,
    wg_pass_id: Option<u32>,
    nodes: Vec<NodeState>,
    loops: Vec<LoopState>,
    ready: Vec<u64>,
    ready_streaming: Vec<u64>,
    root_completed_mask: Vec<u64>,
    root_completed_count: u32,
    is_done: bool,
    spec_dirty: Vec<NodeId>,
}

impl PassState {
    fn new(
        g: &CompiledGraph,
        walk: Sym,
        walk_pass_id: u32,
        loop_max_iters: &[u32],
    ) -> Self {
        let nodes = g.nodes.iter().map(|n| NodeState::new(n.inputs.len())).collect();
        let loops = g
            .loops
            .iter()
            .enumerate()
            .map(|(i, l)| LoopState::new(loop_max_iters[i], l.n_mask_words()))
            .collect();
        let w = g.n_ready_words();
        let mut p = Self {
            walk,
            walk_pass_id,
            wg_pass_id: None,
            nodes,
            loops,
            ready: vec![0; w],
            ready_streaming: vec![0; w],
            root_completed_mask: vec![0; g.n_root_mask_words()],
            root_completed_count: 0,
            is_done: false,
            spec_dirty: Vec::new(),
        };
        p.seed_streaming_ready(g);
        p
    }

    #[inline]
    fn bit(id: NodeId) -> (usize, u32) {
        ((id / 64) as usize, id % 64)
    }

    fn seed_streaming_ready(&mut self, g: &CompiledGraph) {
        for (i, n) in g.nodes.iter().enumerate() {
            if n.only_streaming {
                self.ready_streaming[i / 64] |= 1 << (i % 64);
            }
        }
    }

    fn is_node_loop_done(&self, g: &CompiledGraph, id: NodeId) -> bool {
        let mut cur = g.node(id).loop_id;
        while let Some(lid) = cur {
            if self.loops[lid as usize].done {
                return true;
            }
            cur = g.lp(lid).parent;
        }
        false
    }

    fn shares_active_loop(&self, g: &CompiledGraph, a: NodeId, b: NodeId) -> bool {
        let mut cur_a = g.node(a).loop_id;
        while let Some(lid_a) = cur_a {
            if !self.loops[lid_a as usize].done {
                let mut cur_b = g.node(b).loop_id;
                while let Some(lid_b) = cur_b {
                    if lid_b == lid_a {
                        return true;
                    }
                    cur_b = g.lp(lid_b).parent;
                }
            }
            cur_a = g.lp(lid_a).parent;
        }
        false
    }

    fn is_node_finished(&self, g: &CompiledGraph, id: NodeId) -> bool {
        self.nodes[id as usize].completed || self.is_node_loop_done(g, id)
    }

    fn refresh_ready(&mut self, g: &CompiledGraph, id: NodeId, in_flight: bool) {
        let spec = g.node(id);
        let st = &self.nodes[id as usize];
        let live = !st.completed && !self.is_node_loop_done(g, id);
        let ready = st.cur.mask == spec.full_mask && live;
        let (w, b) = Self::bit(id);
        if ready {
            if !in_flight {
                self.ready[w] |= 1 << b;
            }
        } else {
            self.ready[w] &= !(1 << b);
        }
    }

    fn note_ingested_for_streaming(
        &mut self,
        g: &CompiledGraph,
        id: NodeId,
        in_flight: bool,
    ) {
        let spec = g.node(id);
        let st = &self.nodes[id as usize];
        let (w, b) = Self::bit(id);
        if st.completed || self.is_node_loop_done(g, id) {
            self.ready_streaming[w] &= !(1 << b);
            return;
        }
        let full = st.cur.mask == spec.full_mask;
        let only_streaming_missing =
            spec.full_mask & !st.cur.mask & !spec.streaming_mask == 0;
        if full {
            self.ready_streaming[w] &= !(1 << b);
        } else if only_streaming_missing && !in_flight {
            self.ready_streaming[w] |= 1 << b;
        }
    }

    fn reseed_streaming_ready(&mut self, g: &CompiledGraph, id: NodeId) {
        let only = g.node(id).only_streaming;
        let (w, b) = Self::bit(id);
        if only && !self.nodes[id as usize].completed && !self.is_node_loop_done(g, id) {
            self.ready_streaming[w] |= 1 << b;
        } else {
            self.ready_streaming[w] &= !(1 << b);
        }
    }

    fn record_external(
        &mut self,
        g: &CompiledGraph,
        node: NodeId,
        name: Sym,
        t: &[TensorRef],
    ) {
        let spec = g.node(node);
        if spec.slot_of(name).is_some_and(|i| spec.streaming_mask >> i & 1 == 1) {
            return;
        }
        let mut cur = g.node(node).loop_id;
        while let Some(lid) = cur {
            let ls = g.lp(lid);
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

    fn is_loop_back_input(&self, g: &CompiledGraph, node: NodeId, name: Sym) -> bool {
        let node_name = g.node(node).name;
        let mut cur = g.node(node).loop_id;
        while let Some(lid) = cur {
            let ls = g.lp(lid);
            if !self.loops[lid as usize].done
                && ls.loop_back.iter().any(|&(n, d)| n == name && d == node_name)
            {
                return true;
            }
            cur = ls.parent;
        }
        false
    }

    fn release_slot(
        &self,
        g: &CompiledGraph,
        node: NodeId,
        next: bool,
        refs_released: &mut Vec<u64>,
    ) {
        let spec = g.node(node);
        let held = spec.held_mask;
        let st = &self.nodes[node as usize];
        let slot = if next { &st.next } else { &st.cur };
        for i in 0..spec.inputs.len() {
            if held >> i & 1 == 1 {
                continue;
            }
            if let Some(tensors) = &slot.tensors[i] {
                refs_released.extend(tensors.iter().map(|t| t.uuid));
            }
        }
    }

    fn root_entity_done(&mut self, g: &CompiledGraph, entity_idx: u32) {
        if self.is_done {
            return;
        }
        let w = (entity_idx / 64) as usize;
        let b = entity_idx % 64;
        if self.root_completed_mask[w] >> b & 1 == 0 {
            self.root_completed_mask[w] |= 1 << b;
            self.root_completed_count += 1;
            self.is_done = self.root_completed_count == g.n_root_entities();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn entity_completed(
        &mut self,
        g: &CompiledGraph,
        lid: LoopId,
        child_entity_idx: u32,
        edges_from: usize,
        out: &mut Vec<RoutedEdge>,
        stride: u32,
        in_flight: &[bool],
        refs_taken: &mut Vec<u64>,
        refs_released: &mut Vec<u64>,
        refs_transferred: &mut Vec<u64>,
    ) {
        if self.loops[lid as usize].done {
            return;
        }
        let lspec = g.lp(lid);
        self.cache_outputs(g, lid, edges_from, out, refs_taken);

        {
            let st = &mut self.loops[lid as usize];
            let w = (child_entity_idx / 64) as usize;
            let b = child_entity_idx % 64;
            if st.completed_mask[w] >> b & 1 == 0 {
                st.completed_mask[w] |= 1 << b;
                st.completed_count += 1;
            }
            if st.completed_count < lspec.n_entities() {
                return;
            }
            st.completed_mask.fill(0);
            st.completed_count = 0;
        }

        let finishing = {
            let st = &self.loops[lid as usize];
            st.finish_signal || st.curr_iter.saturating_add(stride) >= st.max_iters
        };
        if finishing {
            self.finish_loop(
                g, lid, out, in_flight, refs_taken, refs_released, refs_transferred,
            );
        } else {
            self.advance_loop(g, lid, out, stride, in_flight, refs_released);
        }
    }

    fn cache_outputs(
        &mut self,
        g: &CompiledGraph,
        lid: LoopId,
        from: usize,
        out: &[RoutedEdge],
        refs_taken: &mut Vec<u64>,
    ) {
        let lspec = g.lp(lid);
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
            refs_taken.extend(e.tensors.iter().map(|t| t.uuid));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_loop(
        &mut self,
        g: &CompiledGraph,
        lid: LoopId,
        out: &mut Vec<RoutedEdge>,
        in_flight: &[bool],
        refs_taken: &mut Vec<u64>,
        refs_released: &mut Vec<u64>,
        refs_transferred: &mut Vec<u64>,
    ) {
        let lspec = g.lp(lid);
        self.loops[lid as usize].done = true;

        let lb = &lspec.loop_back;
        out.retain(|e| !lb.contains(&(e.name, e.dest_sym)));

        self.clear_subtree(g, lid, in_flight, refs_released);

        let (cached, accum) = {
            let st = &mut self.loops[lid as usize];
            (std::mem::take(&mut st.cached), std::mem::take(&mut st.accum))
        };
        for (_, tensors) in cached.iter().chain(accum.iter()) {
            refs_transferred.extend(tensors.iter().map(|t| t.uuid));
        }
        let own_edges_from = out.len();
        for o in &lspec.outputs {
            let t = cached
                .iter()
                .find(|(n, _)| *n == o.name)
                .map(|(_, t)| t.clone())
                .unwrap_or_default();
            out.push(routed(o, t));
        }
        for o in &lspec.accumulated {
            let t = accum
                .iter()
                .find(|(n, _)| *n == o.name)
                .map(|(_, t)| t.clone())
                .unwrap_or_default();
            out.push(routed(o, t));
        }

        match lspec.parent {
            None => self.root_entity_done(g, lspec.entity_idx),
            Some(p) => self.entity_completed(
                g,
                p,
                lspec.entity_idx,
                own_edges_from,
                out,
                1,
                in_flight,
                refs_taken,
                refs_released,
                refs_transferred,
            ),
        }
    }

    fn advance_loop(
        &mut self,
        g: &CompiledGraph,
        lid: LoopId,
        out: &mut Vec<RoutedEdge>,
        stride: u32,
        in_flight: &[bool],
        refs_released: &mut Vec<u64>,
    ) {
        {
            let st = &mut self.loops[lid as usize];
            st.curr_iter = st.curr_iter.saturating_add(stride);
        }
        self.reset_subtree_for_iter(g, lid, in_flight, refs_released);
        let cached = std::mem::take(&mut self.loops[lid as usize].cached);
        for (_, t) in &cached {
            refs_released.extend(t.iter().map(|x| x.uuid));
        }

        let ext = self.loops[lid as usize].ext_inputs.clone();
        for (name, dest, t) in ext {
            out.push(RoutedEdge {
                name,
                dest: Dest::Local(dest),
                dest_sym: g.node(dest).name,
                persist: false,
                new_token: false,
                streaming: false,
                modality: 0,
                tensors: t,
                persist_for_loop: true,
                declined_local: false,
                worker: None,
            });
        }
    }

    fn reset_subtree_for_iter(
        &mut self,
        g: &CompiledGraph,
        lid: LoopId,
        in_flight: &[bool],
        refs_released: &mut Vec<u64>,
    ) {
        let lspec = g.lp(lid);
        for &n in &lspec.member_nodes {
            self.release_slot(g, n, false, refs_released);
            {
                let st = &mut self.nodes[n as usize];
                st.cur.clear();
                std::mem::swap(&mut st.cur, &mut st.next);
                st.completed = false;
            }
            self.refresh_ready(g, n, in_flight[n as usize]);
            self.reseed_streaming_ready(g, n);
        }
        for &c in &lspec.child_loops {
            self.reset_subtree_for_iter(g, c, in_flight, refs_released);
            self.loops[c as usize].reset_metadata();
        }
    }

    /// Clear a finished loop's inner state in this pass. Importantly, DO NOT
    /// reseed `ready_streaming` for a finished loop in this pass — a finished
    /// loop stays finished in this `PassState`, while Pass `p+1` gets a fresh
    /// `PassState` with seeded `ready_streaming`.
    fn clear_subtree(
        &mut self,
        g: &CompiledGraph,
        lid: LoopId,
        in_flight: &[bool],
        refs_released: &mut Vec<u64>,
    ) {
        let lspec = g.lp(lid);
        for &n in &lspec.member_nodes {
            self.release_slot(g, n, false, refs_released);
            self.release_slot(g, n, true, refs_released);
            self.nodes[n as usize].clear();
            self.refresh_ready(g, n, in_flight[n as usize]);
            let (w, b) = Self::bit(n);
            self.ready_streaming[w] &= !(1 << b);
        }
        for &c in &lspec.child_loops {
            self.clear_subtree(g, c, in_flight, refs_released);
            let st = &mut self.loops[c as usize];
            for (_, t) in std::mem::take(&mut st.accum) {
                refs_released.extend(t.iter().map(|x| x.uuid));
            }
            let st = &mut self.loops[c as usize];
            st.cached.clear();
            st.reset_metadata();
        }
    }

    fn is_idle(&self) -> bool {
        self.root_completed_count == 0
            && self.nodes.iter().all(|n| {
                !n.completed && n.cur.mask == 0 && n.next.mask == 0 && n.spec.mask == 0
            })
            && self
                .loops
                .iter()
                .all(|l| l.curr_iter == 0 && l.ext_inputs.is_empty() && !l.done && l.completed_count == 0)
    }
}

pub struct RequestState {
    pub graph: GraphRef,
    passes: VecDeque<PassState>,
    loop_max_iters: Vec<u32>,
    next_pass_by_walk: FxHashMap<Sym, u32>,
    in_flight: Vec<bool>,
    /// Active-pass summary of ready nodes (O(set bits) scan in `GraphRuntime::scan_ready`).
    pub ready: Vec<u64>,
    pub is_done: bool,
    pub num_times_run: u32, // WorkerGraphIO.num_times_run
    refs_taken: Vec<u64>,
    refs_released: Vec<u64>,
    refs_transferred: Vec<u64>,
}

impl RequestState {
    pub fn new(graph: GraphRef) -> Self {
        let loop_max_iters = graph.loops.iter().map(|l| l.max_iters).collect();
        let w = graph.n_ready_words();
        let in_flight = vec![false; graph.nodes.len()];
        Self {
            passes: VecDeque::new(),
            loop_max_iters,
            next_pass_by_walk: FxHashMap::default(),
            in_flight,
            ready: vec![0; w],
            is_done: false,
            num_times_run: 0,
            refs_taken: Vec::new(),
            refs_released: Vec::new(),
            refs_transferred: Vec::new(),
            graph,
        }
    }

    #[inline]
    fn bit(id: NodeId) -> (usize, u32) {
        ((id / 64) as usize, id % 64)
    }

    fn alloc_pass(&mut self, walk: Sym) -> usize {
        let entry = self.next_pass_by_walk.entry(walk).or_insert(0);
        let walk_pass_id = *entry;
        *entry += 1;
        let idx = self.passes.len();
        self.passes.push_back(PassState::new(
            &self.graph,
            walk,
            walk_pass_id,
            &self.loop_max_iters,
        ));
        idx
    }

    /// Index of the oldest `PassState` in which `node` is not yet finished
    /// (FIFO pass ordering invariant per node).
    pub fn active_pass_idx(&self, node: NodeId) -> Option<usize> {
        self.passes
            .iter()
            .position(|p| !p.is_node_finished(&self.graph, node))
    }

    pub fn active_pass_info(&self, node: NodeId) -> Option<(Sym, u32)> {
        self.active_pass_idx(node)
            .map(|idx| (self.passes[idx].walk, self.passes[idx].walk_pass_id))
    }

    /// Pass index of `node`'s active pass for `nested_idxs_interned` / `wg_fwd_pass_idx`.
    pub fn active_wg_pass_idx(&self, node: NodeId) -> u32 {
        match self.active_pass_idx(node) {
            Some(idx) => {
                // Passes ahead of retired ones have sequential wg pass indices.
                let unretired_done = self.passes.iter().take(idx).filter(|p| p.is_done).count() as u32;
                self.num_times_run + idx as u32 - unretired_done
            }
            None => self.num_times_run,
        }
    }

    fn active_loop_pass_idx(&self, lid: LoopId) -> Option<usize> {
        self.passes
            .iter()
            .position(|p| !p.loops[lid as usize].done)
    }

    fn refresh_summary_node(&mut self, node: NodeId) {
        let (w, b) = Self::bit(node);
        let ready = self
            .active_pass_idx(node)
            .is_some_and(|idx| self.passes[idx].ready[w] >> b & 1 == 1);
        if ready {
            self.ready[w] |= 1 << b;
        } else {
            self.ready[w] &= !(1 << b);
        }
    }

    fn refresh_summary_all(&mut self) {
        self.ready.fill(0);
        for node in 0..self.graph.nodes.len() as NodeId {
            self.refresh_summary_node(node);
        }
    }

    // -- ready bitsets -------------------------------------------------------

    /// Python's `ReadySignals.is_ready_for_streaming`: every non-streaming
    /// input has arrived, so a streaming chunk can be taken now.
    pub fn is_ready_for_streaming(&self, id: NodeId) -> bool {
        let spec = self.graph.node(id);
        match self.active_pass_idx(id) {
            Some(idx) => {
                if spec.only_streaming && self.graph.n_root_entities() > 1 && idx >= 2 {
                    return false;
                }
                let (w, b) = Self::bit(id);
                self.passes[idx].ready_streaming[w] >> b & 1 == 1
            }
            None => {
                if !spec.only_streaming {
                    return false;
                }
                if self.graph.n_root_entities() > 1 && self.passes.len() >= 2 {
                    return false;
                }
                true
            }
        }
    }

    pub fn is_ready(&self, id: NodeId) -> bool {
        let (w, b) = Self::bit(id);
        self.ready[w] >> b & 1 == 1
    }

    // -- ingest --------------------------------------------------------------

    fn count_unconsumed_for(&self, g: &CompiledGraph, node: NodeId, slot: u8) -> u32 {
        let spec = g.node(node);
        let mut count = 0u32;
        for p in &self.passes {
            if p.is_node_loop_done(g, node)
                || (spec.loop_id.is_none() && p.nodes[node as usize].completed)
            {
                continue;
            }
            let st = &p.nodes[node as usize];
            if st.cur.has(slot) {
                count += 1;
            }
            if st.next.has(slot) {
                count += 1;
            }
        }
        count
    }

    /// Convenience wrapper for unit tests that operate on a single default walk `0`.
    pub fn ingest(
        &mut self,
        g: &CompiledGraph,
        node: NodeId,
        slot: u8,
        tensors: &[TensorRef],
        can_buffer: bool,
        final_chunk: bool,
    ) -> bool {
        self.ingest_for_walk(g, node, slot, tensors, can_buffer, final_chunk, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ingest_for_walk(
        &mut self,
        g: &CompiledGraph,
        node: NodeId,
        slot: u8,
        tensors: &[TensorRef],
        can_buffer: bool,
        final_chunk: bool,
        walk: Sym,
    ) -> bool {
        let spec = g.node(node);
        let name = spec.inputs[slot as usize];

        let (target_idx, to_next) = if !can_buffer {
            let idx = match self.active_pass_idx(node) {
                Some(idx) => {
                    if spec.only_streaming && g.n_root_entities() > 1 && idx >= 2 {
                        return false;
                    }
                    if self.passes[idx].nodes[node as usize].cur.has(slot) {
                        return false;
                    }
                    idx
                }
                None => {
                    if spec.only_streaming && g.n_root_entities() > 1 && self.passes.len() >= 2 {
                        return false;
                    }
                    self.alloc_pass(walk)
                }
            };
            (idx, false)
        } else {
            // Bounded 2-pass buffering per (node, slot): keeps `test_a_second_signal_buffers_then_refuses`
            // and StreamBuffer/InputIngest backpressure bounded at 2 buffered slots.
            if self.count_unconsumed_for(g, node, slot) >= 2 {
                return false;
            }
            if let Some(lb_idx) = self.passes.iter().position(|p| {
                !p.is_node_loop_done(g, node)
                    && p.is_loop_back_input(g, node, name)
                    && (p.nodes[node as usize].cur.has(slot) || p.nodes[node as usize].completed)
                    && !p.nodes[node as usize].next.has(slot)
            }) {
                (lb_idx, true)
            } else {
                match self.active_pass_idx(node) {
                    Some(idx) => {
                        let p = &self.passes[idx];
                        if !p.nodes[node as usize].cur.has(slot) {
                            (idx, false)
                        } else {
                            let next_free = ((idx + 1)..self.passes.len()).find(|&j| {
                                !self.passes[j].is_node_finished(g, node)
                                    && !self.passes[j].nodes[node as usize].cur.has(slot)
                            });
                            let j = next_free.unwrap_or_else(|| self.alloc_pass(walk));
                            (j, false)
                        }
                    }
                    None => {
                        let idx = self.alloc_pass(walk);
                        (idx, false)
                    }
                }
            }
        };

        let in_flight = self.in_flight[node as usize];
        {
            let pass = &mut self.passes[target_idx];
            let st = &mut pass.nodes[node as usize];
            if to_next {
                st.next.set(slot, tensors.to_vec(), final_chunk);
            } else {
                st.cur.set(slot, tensors.to_vec(), final_chunk);
                pass.record_external(g, node, name, tensors);
            }
            pass.refresh_ready(g, node, in_flight);
            pass.note_ingested_for_streaming(g, node, in_flight);
        }
        self.refresh_summary_node(node);
        true
    }

    /// Route a local edge directly into the pass `(walk, walk_pass_id)` where
    /// the producer ran, keeping intra-pass loop-back in `next` and isolating
    /// cross-pass deliveries into their exact `PassState`.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_in_pass(
        &mut self,
        g: &CompiledGraph,
        walk: Sym,
        walk_pass_id: u32,
        node: NodeId,
        slot: u8,
        tensors: &[TensorRef],
        can_buffer: bool,
        final_chunk: bool,
    ) -> bool {
        let Some(idx) = self
            .passes
            .iter()
            .position(|p| p.walk == walk && p.walk_pass_id == walk_pass_id)
        else {
            return false;
        };
        let spec = g.node(node);
        if self.passes[idx].is_node_loop_done(g, node)
            || (spec.loop_id.is_none() && self.passes[idx].nodes[node as usize].completed)
        {
            return false;
        }
        let name = spec.inputs[slot as usize];
        let in_flight = self.in_flight[node as usize];
        let pass = &mut self.passes[idx];
        let st = &mut pass.nodes[node as usize];
        let to_next = if !st.cur.has(slot) && !st.completed {
            false
        } else if can_buffer && !st.next.has(slot) {
            true
        } else {
            return false;
        };
        if to_next {
            st.next.set(slot, tensors.to_vec(), final_chunk);
        } else {
            st.cur.set(slot, tensors.to_vec(), final_chunk);
            pass.record_external(g, node, name, tensors);
        }
        pass.refresh_ready(g, node, in_flight);
        pass.note_ingested_for_streaming(g, node, in_flight);
        self.refresh_summary_node(node);
        true
    }

    /// Python's `ReadySignals.remove` — speculation rollback. No dereference.
    pub fn remove_input(
        &mut self,
        node: NodeId,
        slot: u8,
        from_next: bool,
    ) -> Option<Vec<TensorRef>> {
        let idx = self.active_pass_idx(node)?;
        let in_flight = self.in_flight[node as usize];
        let t = {
            let pass = &mut self.passes[idx];
            let st = &mut pass.nodes[node as usize];
            let t = if from_next {
                st.next.take(slot)
            } else {
                st.cur.take(slot)
            };
            pass.refresh_ready(&self.graph, node, in_flight);
            t
        };
        self.refresh_summary_node(node);
        t
    }

    /// Whether a slot is already filled, for inferring which slot an ingest
    /// will land in before making it.
    pub fn has_input(&self, node: NodeId, slot: u8, next_iter: bool) -> bool {
        let Some(idx) = self.active_pass_idx(node) else {
            return false;
        };
        if next_iter {
            if self.graph.node(node).loop_id.is_some() {
                self.passes[idx].nodes[node as usize].next.has(slot)
            } else {
                self.passes
                    .get(idx + 1)
                    .is_some_and(|p| p.nodes[node as usize].cur.has(slot))
            }
        } else {
            self.passes[idx].nodes[node as usize].cur.has(slot)
        }
    }

    pub fn input_tensors(
        &self,
        node: NodeId,
        next_iter: bool,
    ) -> Vec<(Sym, Vec<TensorRef>, bool)> {
        let Some(idx) = self.active_pass_idx(node) else {
            return Vec::new();
        };
        self.input_tensors_in_pass(idx, node, next_iter)
    }

    pub fn input_tensors_in_pass(
        &self,
        idx: usize,
        node: NodeId,
        next_iter: bool,
    ) -> Vec<(Sym, Vec<TensorRef>, bool)> {
        if idx >= self.passes.len() {
            return Vec::new();
        }
        let spec = self.graph.node(node);
        if next_iter && spec.loop_id.is_none() {
            let Some(next_pass) = self.passes.get(idx + 1) else {
                return Vec::new();
            };
            let st = &next_pass.nodes[node as usize];
            return spec
                .inputs
                .iter()
                .enumerate()
                .filter_map(|(i, &n)| {
                    st.cur.tensors[i]
                        .as_ref()
                        .map(|t| (n, t.clone(), st.cur.is_final_chunk(i as u8)))
                })
                .collect();
        }
        let st = &self.passes[idx].nodes[node as usize];
        let slot = if next_iter { &st.next } else { &st.cur };
        let carried = if next_iter {
            spec.held_mask & st.cur.mask & !st.next.mask
        } else {
            0
        };
        spec.inputs
            .iter()
            .enumerate()
            .filter_map(|(i, &n)| {
                if carried >> i & 1 == 1 {
                    return st
                        .cur
                        .tensors[i]
                        .as_ref()
                        .map(|t| (n, t.clone(), st.cur.is_final_chunk(i as u8)));
                }
                slot.tensors[i]
                    .as_ref()
                    .map(|t| (n, t.clone(), slot.is_final_chunk(i as u8)))
            })
            .collect()
    }

    pub fn take_for_schedule(&mut self, node: NodeId) {
        let (w, b) = Self::bit(node);
        if let Some(idx) = self.active_pass_idx(node) {
            self.passes[idx].ready[w] &= !(1 << b);
        }
        self.ready[w] &= !(1 << b);
    }

    pub fn push_back(&mut self, node: NodeId) {
        let (w, b) = Self::bit(node);
        if let Some(idx) = self.active_pass_idx(node) {
            self.passes[idx].ready[w] |= 1 << b;
        }
        self.ready[w] |= 1 << b;
    }

    pub fn clear_consumed_inputs(&mut self, g: &CompiledGraph, node: NodeId) -> Vec<u64> {
        let Some(idx) = self.active_pass_idx(node) else {
            return Vec::new();
        };
        let spec = g.node(node);
        let held = spec.held_mask;
        let in_flight = self.in_flight[node as usize];
        let mut freed = Vec::new();
        {
            let pass = &mut self.passes[idx];
            let st = &mut pass.nodes[node as usize];
            for i in 0..spec.inputs.len() {
                if held >> i & 1 == 1 {
                    continue;
                }
                if let Some(tensors) = st.cur.tensors[i].take() {
                    freed.extend(tensors.iter().map(|t| t.uuid));
                }
                st.cur.mask &= !(1 << i);
                st.cur.final_chunk &= !(1 << i);
            }
            pass.refresh_ready(g, node, in_flight);
        }
        self.refresh_summary_node(node);
        freed
    }

    pub fn set_in_flight(&mut self, node: NodeId, on: bool) {
        self.in_flight[node as usize] = on;
    }

    pub fn is_in_flight(&self, node: NodeId) -> bool {
        self.in_flight[node as usize]
    }

    // -- completion ----------------------------------------------------------

    pub fn complete(
        &mut self,
        g: &CompiledGraph,
        node: NodeId,
        out_tensors: &[Vec<TensorRef>],
    ) -> Completed {
        self.complete_strided(g, node, out_tensors, 1)
    }

    pub fn complete_strided(
        &mut self,
        g: &CompiledGraph,
        node: NodeId,
        out_tensors: &[Vec<TensorRef>],
        loop_stride: u32,
    ) -> Completed {
        self.complete_strided_for_walk(g, node, out_tensors, loop_stride, 0)
    }

    pub fn complete_strided_for_walk(
        &mut self,
        g: &CompiledGraph,
        node: NodeId,
        out_tensors: &[Vec<TensorRef>],
        loop_stride: u32,
        walk: Sym,
    ) -> Completed {
        let idx = self
            .active_pass_idx(node)
            .unwrap_or_else(|| self.alloc_pass(walk));
        let spec = g.node(node);
        let mut out: Vec<RoutedEdge> = Vec::with_capacity(spec.outputs.len() + 2);
        let mut freed: Vec<u64> = Vec::new();

        let (pass_walk, walk_pass_id, became_done) = {
            let pass = &mut self.passes[idx];
            let prev_done = pass.is_done;
            let in_flight = self.in_flight[node as usize];
            {
                let st = &mut pass.nodes[node as usize];
                st.completed = true;
                if spec.loop_id.is_none() {
                    for t in st.cur.tensors.iter().flatten() {
                        freed.extend(t.iter().map(|x| x.uuid));
                    }
                    st.cur.clear();
                }
            }
            pass.refresh_ready(g, node, in_flight);
            if spec.loop_id.is_none() {
                pass.reseed_streaming_ready(g, node);
            }
            for (i, e) in spec.outputs.iter().enumerate() {
                out.push(routed(e, out_tensors.get(i).cloned().unwrap_or_default()));
            }
            match spec.loop_id {
                None => pass.root_entity_done(g, spec.entity_idx),
                Some(lid) => pass.entity_completed(
                    g,
                    lid,
                    spec.entity_idx,
                    0,
                    &mut out,
                    loop_stride,
                    &self.in_flight,
                    &mut self.refs_taken,
                    &mut self.refs_released,
                    &mut self.refs_transferred,
                ),
            }
            (pass.walk, pass.walk_pass_id, pass.is_done && !prev_done)
        };

        if became_done {
            self.passes[idx].wg_pass_id = Some(self.num_times_run);
            self.num_times_run += 1;
        }
        self.is_done = self.passes.iter().any(|p| p.is_done);
        self.refresh_summary_all();

        freed.append(&mut self.refs_released);
        Completed {
            edges: out,
            freed,
            taken: std::mem::take(&mut self.refs_taken),
            transferred: std::mem::take(&mut self.refs_transferred),
            pass_walk,
            walk_pass_id,
        }
    }

    /// Retire completed passes from the front of the queue, returning
    /// `(walk, walk_pass_id, wg_pass_id)` for each retired pass while preserving
    /// any pipelined future passes intact.
    pub fn retire_done_passes(&mut self) -> Vec<(Sym, u32, u32)> {
        let mut retired = Vec::new();
        while self.passes.front().is_some_and(|p| p.is_done) {
            let p = self.passes.pop_front().expect("checked");
            let wg_pass_id = p.wg_pass_id.unwrap_or_else(|| {
                let id = self.num_times_run;
                self.num_times_run += 1;
                id
            });
            retired.push((p.walk, p.walk_pass_id, wg_pass_id));
        }
        if !retired.is_empty() {
            self.is_done = self.passes.iter().any(|p| p.is_done);
            self.refresh_summary_all();
        }
        retired
    }

    // -- loops ---------------------------------------------------------------

    pub fn register_loop_finish(&mut self, lid: LoopId) {
        if let Some(idx) = self.active_loop_pass_idx(lid) {
            self.passes[idx].loops[lid as usize].finish_signal = true;
        }
    }

    pub fn loop_iter(&self, lid: LoopId) -> u32 {
        self.active_loop_pass_idx(lid)
            .map_or(0, |idx| self.passes[idx].loops[lid as usize].curr_iter)
    }

    pub fn loop_iter_for_node(&self, node: NodeId, lid: LoopId) -> u32 {
        self.active_pass_idx(node)
            .map_or(0, |idx| self.passes[idx].loops[lid as usize].curr_iter)
    }

    pub fn loop_finished(&self, lid: LoopId) -> bool {
        self.active_loop_pass_idx(lid).is_some_and(|idx| {
            let st = &self.passes[idx].loops[lid as usize];
            st.done || st.finish_signal
        })
    }

    pub fn loop_max_iters(&self, lid: LoopId) -> u32 {
        self.loop_max_iters[lid as usize]
    }

    pub fn cap_loop(&mut self, lid: LoopId, cap: u32) {
        self.loop_max_iters[lid as usize] = self.loop_max_iters[lid as usize].min(cap);
        for p in &mut self.passes {
            p.loops[lid as usize].max_iters = p.loops[lid as usize].max_iters.min(cap);
        }
    }

    pub fn set_loop_cap(&mut self, lid: LoopId, cap: u32) {
        self.loop_max_iters[lid as usize] = cap;
        for p in &mut self.passes {
            p.loops[lid as usize].max_iters = cap;
        }
    }

    pub fn loop_indices(&self) -> Vec<u32> {
        (0..self.graph.loops.len())
            .map(|lid| self.loop_iter(lid as LoopId))
            .collect()
    }

    pub fn loop_indices_for_node(&self, node: NodeId) -> Vec<u32> {
        match self.active_pass_idx(node) {
            Some(idx) => self.passes[idx].loops.iter().map(|l| l.curr_iter).collect(),
            None => vec![0; self.graph.loops.len()],
        }
    }

    pub fn is_idle(&self) -> bool {
        self.passes.iter().all(|p| p.is_idle())
    }

    // -- speculation ---------------------------------------------------------

    pub fn shares_active_loop_in_pass(&self, idx: usize, a: NodeId, b: NodeId) -> bool {
        self.passes
            .get(idx)
            .is_some_and(|p| p.shares_active_loop(&self.graph, a, b))
    }

    pub fn ingest_for_speculation(
        &mut self,
        g: &CompiledGraph,
        source: NodeId,
        edges: &[RoutedEdge],
    ) -> Vec<SpecNode> {
        let src_loop = g.node(source).loop_id;
        let mut dests: Vec<NodeId> = Vec::new();
        let mut next_iter: Vec<NodeId> = Vec::new();

        let Some(src_pass) = self.active_pass_idx(source) else {
            let mut spec_masks: Vec<(NodeId, u64)> = Vec::new();
            for e in edges {
                let Dest::Local(d) = e.dest else { continue };
                let Some(slot) = g.node(d).slot_of(e.name) else { continue };
                if d != source && self.active_pass_idx(d).is_some() {
                    continue;
                }
                match spec_masks.iter_mut().find(|(n, _)| *n == d) {
                    Some((_, m)) => *m |= 1 << slot,
                    None => spec_masks.push((d, 1 << slot)),
                }
                if !dests.contains(&d) {
                    dests.push(d);
                }
                if let Some(lid) = src_loop {
                    if g.lp(lid).loop_back.contains(&(e.name, e.dest_sym))
                        && !next_iter.contains(&d)
                    {
                        next_iter.push(d);
                    }
                }
            }
            return dests
                .into_iter()
                .filter(|&d| {
                    let spec = g.node(d);
                    let needed = spec.full_mask & !spec.streaming_mask;
                    let have = spec_masks
                        .iter()
                        .find(|(n, _)| *n == d)
                        .map_or(0, |&(_, m)| m);
                    needed & !have == 0
                })
                .map(|d| SpecNode {
                    node: d,
                    is_new_loop_iter: next_iter.contains(&d),
                    loop_id: g.node(d).loop_id,
                })
                .collect();
        };

        for e in edges {
            let Dest::Local(d) = e.dest else { continue };
            let Some(slot) = g.node(d).slot_of(e.name) else { continue };
            let same_pass = d == source
                || self.active_pass_idx(d) == Some(src_pass)
                || self.passes[src_pass].shares_active_loop(g, source, d);
            if !same_pass {
                continue;
            }
            let pass = &mut self.passes[src_pass];
            pass.nodes[d as usize].spec.set(slot, e.tensors.clone(), false);
            if !pass.spec_dirty.contains(&d) {
                pass.spec_dirty.push(d);
            }
            if !dests.contains(&d) {
                dests.push(d);
            }
            if let Some(lid) = src_loop {
                if g.lp(lid).loop_back.contains(&(e.name, e.dest_sym))
                    && !next_iter.contains(&d)
                {
                    next_iter.push(d);
                }
            }
        }

        dests
            .into_iter()
            .filter(|&d| self.ready_for_speculation_in_pass(Some(src_pass), d, d == source, true))
            .map(|d| SpecNode {
                node: d,
                is_new_loop_iter: next_iter.contains(&d),
                loop_id: g.node(d).loop_id,
            })
            .collect()
    }

    pub fn ready_for_speculation(
        &self,
        node: NodeId,
        check_next_iter: bool,
        allow_streaming: bool,
    ) -> bool {
        self.ready_for_speculation_in_pass(
            self.active_pass_idx(node),
            node,
            check_next_iter,
            allow_streaming,
        )
    }

    pub fn ready_for_speculation_in_pass(
        &self,
        pass_idx: Option<usize>,
        node: NodeId,
        check_next_iter: bool,
        allow_streaming: bool,
    ) -> bool {
        let spec = self.graph.node(node);
        let mut needed = spec.full_mask;
        if allow_streaming {
            needed &= !spec.streaming_mask;
        }
        let Some(idx) = pass_idx else {
            return needed == 0;
        };
        if check_next_iter && spec.loop_id.is_none() {
            let next_mask = self
                .passes
                .get(idx + 1)
                .map_or(0, |p| p.nodes[node as usize].cur.mask);
            let spec_mask = self.passes[idx].nodes[node as usize].spec.mask;
            return needed & !(next_mask | spec_mask) == 0;
        }
        let st = &self.passes[idx].nodes[node as usize];
        let mut have = if check_next_iter { st.next.mask } else { st.cur.mask } | st.spec.mask;
        if check_next_iter {
            have |= st.cur.mask & spec.held_mask;
        }
        needed & !have == 0
    }

    pub fn clear_speculative_inputs(&mut self) {
        for p in &mut self.passes {
            for n in std::mem::take(&mut p.spec_dirty) {
                p.nodes[n as usize].spec.clear();
            }
        }
    }

    // -- lifecycle -----------------------------------------------------------

    /// Full wipe of all active passes.
    pub fn reset(&mut self) {
        self.passes.clear();
        self.ready.fill(0);
        self.is_done = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compile::{compile_one, EdgeArg, LoopArg, NodeArg};
    use crate::graph::spec::StrToId;

    fn edge(name: &str, dest: &str) -> EdgeArg {
        EdgeArg {
            name: name.into(), dest: dest.into(), persist: false,
            new_token: false, streaming: false, modality: "text".into(),
        }
    }

    fn node(name: &str, inputs: &[&str], streaming: &[&str], outputs: Vec<EdgeArg>) -> NodeArg {
        NodeArg {
            name: name.into(), async_enabled: false,
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            streaming_inputs: streaming.iter().map(|s| s.to_string()).collect(),
            outputs,
        }
    }

    fn t(uuid: u64) -> TensorRef {
        TensorRef { uuid, dim0: 1, nbytes: 4, offset: 0 }
    }

    fn uuids(edge: &RoutedEdge) -> Vec<u64> {
        edge.tensors.iter().map(|t| t.uuid).collect()
    }

    /// enc -> [loop L: dec], three iterations. `h` is the loop's external
    /// input (held and re-injected every iteration), `tok` loops back, the
    /// loop outputs the last `tok` and accumulates every `frame`.
    fn decode_loop() -> (StrToId, GraphRef) {
        let mut it = StrToId::default();
        let nodes = vec![
            node("enc", &["x"], &[], vec![edge("h", "dec"), edge("tok", "dec")]),
            node("dec", &["h", "tok"], &[], vec![edge("tok", "dec"), edge("frame", "")]),
        ];
        let loops = vec![LoopArg {
            name: "L".into(), max_iters: 3, parent: None,
            member_nodes: vec!["dec".into()],
            outputs: vec![edge("tok", "emit_to_client")],
            accumulated: vec![edge("frame", "emit_to_client")],
            loop_back: vec![("tok".into(), "dec".into())],
            external_inputs: vec![("h".into(), "dec".into())],
        }];
        let g = compile_one(&mut it, &nodes, &loops).unwrap();
        (it, g)
    }

    #[test]
    fn top_level_completion_releases_unconsumed_inputs() {
        let (_, g) = decode_loop();
        let enc = 0;
        let mut st = RequestState::new(g.clone());
        assert!(st.ingest(&g, enc, 0, &[t(1)], false, false));
        assert!(st.is_ready(enc));
        // No cleanup ran first, so the completion itself releases `x`.
        let done = st.complete(&g, enc, &[vec![t(2)], vec![t(3)]]);
        assert_eq!(done.freed, vec![1]);
        assert!(done.taken.is_empty());
        assert_eq!(done.edges.len(), 2);
        assert!(!st.is_ready(enc));
        assert!(!st.is_done, "the loop is still a pending root entity");
    }

    #[test]
    fn loop_cache_references_balance_across_iterations() {
        let (it, g) = decode_loop();
        let (enc, dec) = (0, 1);
        let (h, tok) = (it.get("h").unwrap(), it.get("tok").unwrap());
        let mut st = RequestState::new(g.clone());

        st.ingest(&g, enc, 0, &[t(1)], false, false);
        assert_eq!(st.clear_consumed_inputs(&g, enc), vec![1]);
        st.complete(&g, enc, &[vec![t(2)], vec![t(3)]]);
        st.ingest(&g, dec, 0, &[t(2)], true, false);
        st.ingest(&g, dec, 1, &[t(3)], true, false);

        let mut last_tok = 3;
        for i in 0..3u64 {
            assert!(st.is_ready(dec), "iteration {i}");
            // `h` is held for re-injection; only the loop-back input goes.
            assert_eq!(st.clear_consumed_inputs(&g, dec), vec![last_tok]);
            let (new_tok, frame) = (10 * (i + 1), 10 * (i + 1) + 1);
            let done = st.complete(&g, dec, &[vec![t(new_tok)], vec![t(frame)]]);
            // maybe_cache_output takes one reference per cached tensor.
            assert_eq!(done.taken, vec![new_tok, frame], "iteration {i}");
            if i < 2 {
                // Advancing un-caches the regular output only; the
                // accumulated one is held across iterations.
                assert_eq!(done.freed, vec![new_tok], "iteration {i}");
                assert!(done.transferred.is_empty());
                assert_eq!(st.loop_iter(0), i as u32 + 1);
                let names: Vec<Sym> = done.edges.iter().map(|e| e.name).collect();
                assert_eq!(names, vec![tok, it.get("frame").unwrap(), h]);
                let reinjected = &done.edges[2];
                assert!(reinjected.persist_for_loop);
                assert_eq!(reinjected.dest_sym, it.get("dec").unwrap());
                assert_eq!(uuids(reinjected), vec![2]);
                st.ingest_in_pass(&g, done.pass_walk, done.walk_pass_id, dec, 0, &reinjected.tensors, true, false);
                st.ingest_in_pass(&g, done.pass_walk, done.walk_pass_id, dec, 1, &done.edges[0].tensors, true, false);
                last_tok = new_tok;
            } else {
                // Finishing hands the caches to the loop's output edges: no
                // release, and the loop-back edge is dropped.
                assert!(done.freed.is_empty());
                assert_eq!(done.transferred, vec![30, 11, 21, 31]);
                assert!(done.edges.iter().all(|e| !matches!(e.dest, Dest::Local(_))));
                let out: Vec<(Dest, Vec<u64>)> = done.edges.iter()
                    .filter(|e| e.dest == Dest::EmitToClient)
                    .map(|e| (e.dest, uuids(e))).collect();
                assert_eq!(out, vec![
                    (Dest::EmitToClient, vec![30]),
                    (Dest::EmitToClient, vec![11, 21, 31]),
                ]);
                assert!(st.passes[0].loops[0].done);
                assert!(st.is_done);
                assert_eq!(st.num_times_run, 1);
            }
        }
    }

    #[test]
    fn advancing_releases_an_unconsumed_current_slot() {
        let (_, g) = decode_loop();
        let dec = 1;
        let mut st = RequestState::new(g.clone());
        st.ingest(&g, dec, 0, &[t(2)], true, false);
        st.ingest(&g, dec, 1, &[t(3)], true, false);
        // A loop-back arrival while the current slot is full is buffered.
        assert!(st.ingest(&g, dec, 1, &[t(4)], true, false));
        assert!(st.has_input(dec, 1, true));
        assert!(!st.ingest(&g, dec, 1, &[t(5)], true, false), "both slots full");

        // No cleanup: the reset releases `tok` (not the held `h`), then
        // promotes the buffered arrival.
        let done = st.complete(&g, dec, &[vec![t(10)], vec![t(11)]]);
        assert_eq!(done.freed, vec![3, 10]);
        let now: Vec<Vec<u64>> = st.input_tensors(dec, false).into_iter()
            .map(|(_, ts, _)| ts.iter().map(|t| t.uuid).collect()).collect();
        assert_eq!(now, vec![vec![4]]);
        assert!(!st.has_input(dec, 1, true));
    }

    #[test]
    fn finish_signal_ends_the_loop_early() {
        let (_, g) = decode_loop();
        let dec = 1;
        let mut st = RequestState::new(g.clone());
        st.ingest(&g, dec, 0, &[t(2)], true, false);
        st.ingest(&g, dec, 1, &[t(3)], true, false);
        st.register_loop_finish(0);
        st.clear_consumed_inputs(&g, dec);
        let done = st.complete(&g, dec, &[vec![t(10)], vec![t(11)]]);
        assert!(st.passes[0].loops[0].done);
        assert_eq!(st.loop_iter(0), 0);
        assert!(done.freed.is_empty());
        let out: Vec<Vec<u64>> = done.edges.iter()
            .filter(|e| e.dest == Dest::EmitToClient).map(uuids).collect();
        assert_eq!(out, vec![vec![10], vec![11]]);
    }

    #[test]
    fn streaming_ready_only_when_nothing_but_streaming_inputs_is_missing() {
        let mut it = StrToId::default();
        let nodes = vec![
            node("talker", &["text", "audio"], &["audio"], vec![]),
            node("vocoder", &["codes"], &["codes"], vec![]),
        ];
        let g = compile_one(&mut it, &nodes, &[]).unwrap();
        let (talker, vocoder) = (0, 1);
        let mut st = RequestState::new(g.clone());
        assert!(st.is_ready_for_streaming(vocoder));
        assert!(!st.is_ready_for_streaming(talker));

        st.ingest(&g, talker, 1, &[t(1)], false, false);
        assert!(!st.is_ready_for_streaming(talker));
        st.remove_input(talker, 1, false);

        st.ingest(&g, talker, 0, &[t(2)], false, false);
        assert!(st.is_ready_for_streaming(talker));
        assert!(!st.is_ready(talker));
        st.ingest(&g, talker, 1, &[t(3)], false, false);
        assert!(st.is_ready(talker));
        assert!(!st.is_ready_for_streaming(talker), "full, so plain ready");
    }

    #[test]
    fn an_only_streaming_node_is_streaming_ready_again_after_completing() {
        let mut it = StrToId::default();
        let nodes = vec![node("vocoder", &["codes"], &["codes"], vec![])];
        let g = compile_one(&mut it, &nodes, &[]).unwrap();
        let snac = 0;
        let mut st = RequestState::new(g.clone());

        assert!(st.is_ready_for_streaming(snac), "seeded: all inputs streamed");
        assert!(st.ingest(&g, snac, 0, &[t(1)], false, false));
        assert!(st.is_ready(snac));
        assert!(!st.is_ready_for_streaming(snac), "full, so plainly ready");

        st.clear_consumed_inputs(&g, snac);
        st.complete(&g, snac, &[vec![t(100)]]);

        assert!(!st.is_ready(snac));
        assert!(
            st.is_ready_for_streaming(snac),
            "an only-streaming node must accept its next chunk after completing",
        );
        assert!(st.ingest(&g, snac, 0, &[t(2)], false, false));
    }

    #[test]
    fn pipelined_consumer_and_successor_preserve_next_chunk_across_pass_retirement() {
        let mut it = StrToId::default();
        let nodes = vec![
            node("vocoder", &["codes"], &["codes"], vec![edge("wave", "post")]),
            node("post", &["wave"], &[], vec![edge("audio", "emit_to_client")]),
        ];
        let g = compile_one(&mut it, &nodes, &[]).unwrap();
        let (vocoder, post) = (0, 1);
        let mut st = RequestState::new(g.clone());

        // Chunk 1 -> vocoder (Pass 0).
        assert!(st.is_ready_for_streaming(vocoder));
        assert!(st.ingest(&g, vocoder, 0, &[t(1)], false, false));
        st.clear_consumed_inputs(&g, vocoder);
        let first = st.complete(&g, vocoder, &[vec![t(2)]]);
        assert!(!st.is_done);
        assert!(st.ingest_in_pass(
            &g, first.pass_walk, first.walk_pass_id, post, 0, &first.edges[0].tensors, true, false
        ));

        // Chunk 2 pipelines into vocoder (Pass 1) BEFORE post finishes Pass 0.
        assert!(st.is_ready_for_streaming(vocoder), "chunk 2 pipelines into Pass 1");
        assert!(st.ingest(&g, vocoder, 0, &[t(4)], false, true));
        st.clear_consumed_inputs(&g, vocoder);
        let second = st.complete(&g, vocoder, &[vec![t(5)]]);
        assert!(!st.is_done, "consumer running twice must not finish the WG");
        assert!(st.ingest_in_pass(
            &g, second.pass_walk, second.walk_pass_id, post, 0, &second.edges[0].tensors, true, false
        ));

        // Bounded 2-deep pipeline: chunk 3 waits in StreamBuffer until Pass 0 retires.
        assert!(!st.is_ready_for_streaming(vocoder), "ahead >= 2 gates chunk 3");
        assert!(!st.ingest(&g, vocoder, 0, &[t(7)], false, false));

        // Post completes Pass 0 (wave_1 = t(2)); Pass 0 retires, preserving Pass 1's wave_2 = t(5)!
        assert_eq!(st.input_tensors(post, false)[0].1[0].uuid, 2);
        st.clear_consumed_inputs(&g, post);
        st.complete(&g, post, &[vec![t(3)]]);
        assert!(st.is_done);
        assert_eq!(st.num_times_run, 1);
        assert_eq!(st.retire_done_passes(), vec![(0, 0, 0)]);

        // Pass 1's wave_2 survived retirement intact, and vocoder is unblocked for chunk 3.
        assert!(!st.is_done);
        assert!(st.is_ready(post));
        assert_eq!(st.input_tensors(post, false)[0].1[0].uuid, 5);
        assert!(st.is_ready_for_streaming(vocoder));

        st.clear_consumed_inputs(&g, post);
        st.complete(&g, post, &[vec![t(6)]]);
        assert!(st.is_done);
        assert_eq!(st.num_times_run, 2);
        assert_eq!(st.retire_done_passes(), vec![(0, 1, 1)]);
        assert!(st.is_idle());
    }

    #[test]
    fn distinct_entity_counting_prevents_double_completion_at_root_and_loop() {
        let mut it = StrToId::default();
        let nodes = vec![
            node("a", &["x"], &[], vec![]),
            node("b", &["y"], &[], vec![]),
        ];
        let g = compile_one(&mut it, &nodes, &[]).unwrap();
        let mut st = RequestState::new(g.clone());
        // Completing `a` twice in two passes keeps Pass 0 waiting on `b` (FIFO per node).
        st.ingest(&g, 0, 0, &[t(1)], true, false);
        st.complete(&g, 0, &[]);
        assert!(!st.is_done);
        st.ingest(&g, 0, 0, &[t(2)], true, false);
        st.complete(&g, 0, &[]);
        assert!(!st.is_done, "b still has not completed Pass 0");

        // Completing `b` for Pass 0 finishes Pass 0, leaving Pass 1 active on `a`.
        st.ingest(&g, 1, 0, &[t(3)], true, false);
        st.complete(&g, 1, &[]);
        assert!(st.is_done);
        assert_eq!(st.retire_done_passes(), vec![(0, 0, 0)]);
        assert!(!st.is_done);
        st.ingest(&g, 1, 0, &[t(4)], true, false);
        st.complete(&g, 1, &[]);
        assert!(st.is_done);
        assert_eq!(st.retire_done_passes(), vec![(0, 1, 1)]);

        // Also test distinct entities inside a 2-node Loop.
        let mut it2 = StrToId::default();
        let nodes2 = vec![
            node("m1", &["x"], &[], vec![]),
            node("m2", &["y"], &[], vec![]),
        ];
        let loops2 = vec![LoopArg {
            name: "L".into(), max_iters: 1, parent: None,
            member_nodes: vec!["m1".into(), "m2".into()],
            outputs: vec![], accumulated: vec![],
            loop_back: vec![], external_inputs: vec![],
        }];
        let g2 = compile_one(&mut it2, &nodes2, &loops2).unwrap();
        let mut st2 = RequestState::new(g2.clone());
        st2.ingest(&g2, 0, 0, &[t(1)], true, false);
        st2.complete(&g2, 0, &[]);
        assert!(!st2.loop_finished(0));
        // Double-completing entity 0 inside the same pass cannot finish the 2-entity loop.
        let mut dummy_edges = Vec::new();
        let (mut tk, mut rel, mut tr) = (Vec::new(), Vec::new(), Vec::new());
        st2.passes[0].entity_completed(
            &g2, 0, 0, 0, &mut dummy_edges, 1, &[false, false], &mut tk, &mut rel, &mut tr,
        );
        assert!(!st2.loop_finished(0));
        assert!(!st2.is_done);
        // Completing m1 again in Pass 1 also cannot finish Pass 0 while m2 is pending.
        st2.ingest(&g2, 0, 0, &[t(9)], true, false);
        st2.complete(&g2, 0, &[]);
        assert!(!st2.is_done);
        st2.ingest(&g2, 1, 0, &[t(2)], true, false);
        st2.complete(&g2, 1, &[]);
        assert!(st2.passes[0].loops[0].done);
        assert!(st2.is_done);
    }

    #[test]
    fn consumer_loop_pipelined_with_successor_in_one_wg() {
        let mut it = StrToId::default();
        let nodes = vec![
            node("vocoder", &["codes"], &["codes"], vec![edge("wave", "")]),
            node("post", &["wave"], &[], vec![edge("audio", "emit_to_client")]),
        ];
        let loops = vec![LoopArg {
            name: "chunk_loop".into(), max_iters: 2, parent: None,
            member_nodes: vec!["vocoder".into()],
            outputs: vec![edge("wave", "post")],
            accumulated: vec![],
            loop_back: vec![], external_inputs: vec![],
        }];
        let g = compile_one(&mut it, &nodes, &loops).unwrap();
        let (vocoder, post) = (0, 1);
        let mut st = RequestState::new(g.clone());

        // Pass 0: Loop runs 2 chunks -> finishes -> emits wave_0 to post, does NOT reseed inside Pass 0!
        assert!(st.ingest(&g, vocoder, 0, &[t(1)], false, false));
        st.clear_consumed_inputs(&g, vocoder);
        st.complete(&g, vocoder, &[vec![t(10)]]);
        assert!(st.is_ready_for_streaming(vocoder), "iter 1 in Pass 0");
        assert!(st.ingest(&g, vocoder, 0, &[t(2)], false, false));
        st.clear_consumed_inputs(&g, vocoder);
        let loop_done = st.complete(&g, vocoder, &[vec![t(11)]]);
        assert!(!st.is_done, "post hasn't run Pass 0");
        assert!(st.passes[0].loops[0].done);
        assert!(st.ingest_in_pass(
            &g, loop_done.pass_walk, loop_done.walk_pass_id, post, 0,
            &loop_done.edges[0].tensors, true, false
        ));

        // Pass 1: chunk 3 starts a fresh Loop pass BEFORE post runs Pass 0!
        assert!(st.is_ready_for_streaming(vocoder), "Pass 1 starts cleanly");
        assert!(st.ingest(&g, vocoder, 0, &[t(3)], false, false));
        assert_eq!(st.passes.len(), 2);
        assert!(st.passes[0].loops[0].done);
        assert!(!st.passes[1].loops[0].done);

        // Post completes Pass 0 -> Pass 0 retires without disturbing Pass 1's Loop!
        st.clear_consumed_inputs(&g, post);
        st.complete(&g, post, &[vec![t(100)]]);
        assert_eq!(st.retire_done_passes(), vec![(0, 0, 0)]);
        assert_eq!(st.passes.len(), 1);
        assert!(!st.passes[0].loops[0].done);
        assert!(st.is_ready(vocoder), "Pass 1 vocoder input preserved");
    }

    #[test]
    fn pipelined_producer_into_successor_loop_isolates_external_inputs() {
        let mut it = StrToId::default();
        let nodes = vec![
            node("vocoder", &["codes"], &["codes"], vec![edge("wave", "post")]),
            node("post", &["wave"], &[], vec![edge("audio", "emit_to_client")]),
        ];
        let loops = vec![LoopArg {
            name: "post_loop".into(), max_iters: 2, parent: None,
            member_nodes: vec!["post".into()],
            outputs: vec![], accumulated: vec![],
            loop_back: vec![], external_inputs: vec![("wave".into(), "post".into())],
        }];
        let g = compile_one(&mut it, &nodes, &loops).unwrap();
        let (vocoder, post) = (0, 1);
        let mut st = RequestState::new(g.clone());

        // Pass 0 vocoder feeds wave_0 into post_loop Pass 0.
        st.ingest(&g, vocoder, 0, &[t(1)], false, false);
        let c0 = st.complete(&g, vocoder, &[vec![t(10)]]);
        assert!(st.ingest_in_pass(&g, c0.pass_walk, c0.walk_pass_id, post, 0, &c0.edges[0].tensors, true, false));

        // Pass 1 vocoder feeds wave_1 into post_loop Pass 1 while Pass 0 is at iter 0!
        st.ingest(&g, vocoder, 0, &[t(2)], false, false);
        let c1 = st.complete(&g, vocoder, &[vec![t(20)]]);
        assert!(st.ingest_in_pass(&g, c1.pass_walk, c1.walk_pass_id, post, 0, &c1.edges[0].tensors, true, false));

        // Pass 0 post_loop advances iter 0 -> iter 1: Pass 1's wave_1 stays untouched!
        st.clear_consumed_inputs(&g, post);
        let p0_i0 = st.complete(&g, post, &[vec![t(100)]]);
        assert_eq!(st.loop_iter_for_node(post, 0), 1);
        let reinjected = p0_i0.edges.iter().find(|e| e.persist_for_loop).unwrap();
        assert!(st.ingest_in_pass(&g, p0_i0.pass_walk, p0_i0.walk_pass_id, post, 0, &reinjected.tensors, true, false));
        assert_eq!(st.input_tensors(post, false)[0].1[0].uuid, 10);

        // Pass 0 post_loop finishes iter 1 and retires -> Pass 1 post_loop runs with wave_1 = 20!
        st.clear_consumed_inputs(&g, post);
        st.complete(&g, post, &[vec![t(101)]]);
        assert_eq!(st.retire_done_passes(), vec![(0, 0, 0)]);
        assert_eq!(st.loop_iter_for_node(post, 0), 0);
        assert_eq!(st.input_tensors(post, false)[0].1[0].uuid, 20);
    }

    #[test]
    fn a_node_with_a_regular_input_is_not_streaming_ready_after_completing() {
        let mut it = StrToId::default();
        let nodes = vec![node("talker", &["text", "audio"], &["audio"], vec![])];
        let g = compile_one(&mut it, &nodes, &[]).unwrap();
        let talker = 0;
        let mut st = RequestState::new(g.clone());
        st.ingest(&g, talker, 0, &[t(1)], false, false);
        st.ingest(&g, talker, 1, &[t(2)], false, false);
        assert!(st.is_ready(talker));
        st.complete(&g, talker, &[]);
        assert!(
            !st.is_ready_for_streaming(talker),
            "`text` is still missing, so a chunk must not be accepted"
        );
    }

    #[test]
    fn speculative_schedule_gates_the_add_not_the_removal() {
        let (_, g) = decode_loop();
        let enc = 0;
        let mut st = RequestState::new(g.clone());
        st.set_in_flight(enc, true);
        st.ingest(&g, enc, 0, &[t(1)], false, false);
        assert!(!st.is_ready(enc), "a scheduled node is not re-queued");

        let mut st = RequestState::new(g.clone());
        st.ingest(&g, enc, 0, &[t(1)], false, false);
        st.set_in_flight(enc, true);
        assert!(st.is_ready(enc), "marking does not withdraw a queued node");
    }

    #[test]
    fn reset_restores_a_fresh_request() {
        let (_, g) = decode_loop();
        let (enc, dec) = (0, 1);
        let mut st = RequestState::new(g.clone());
        st.ingest(&g, enc, 0, &[t(1)], false, false);
        st.complete(&g, enc, &[vec![t(2)], vec![t(3)]]);
        st.ingest(&g, dec, 0, &[t(2)], true, false);
        st.reset();
        assert!(!st.is_ready(enc) && !st.is_ready(dec));
        assert!(!st.has_input(dec, 0, false));
        assert_eq!(st.loop_indices(), vec![0]);
        assert!(!st.is_done);
    }

    #[test]
    fn a_held_external_input_satisfies_next_iter_speculation() {
        let (it, g) = decode_loop();
        let dec = 1;
        let (h, tok) = (it.get("h").unwrap(), it.get("tok").unwrap());
        let mut st = RequestState::new(g.clone());
        st.ingest(&g, dec, 0, &[t(2)], true, false);
        st.ingest(&g, dec, 1, &[t(3)], true, false);

        let spec_edges: Vec<RoutedEdge> = g.node(dec).outputs.iter()
            .map(|e| routed(e, vec![t(30)])).collect();
        let ready = st.ingest_for_speculation(&g, dec, &spec_edges);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].node, dec);
        assert!(ready[0].is_new_loop_iter);
        st.clear_speculative_inputs();

        // The loop-back `tok` goes into the next-iter slot; `h` stays put.
        assert!(st.ingest(&g, dec, 1, &[t(30)], true, false));
        assert!(st.ready_for_speculation(dec, true, false));
        let mut inputs: Vec<(Sym, Vec<u64>)> = st.input_tensors(dec, true)
            .into_iter().map(|(n, ts, _)| (n, ts.iter().map(|x| x.uuid).collect()))
            .collect();
        inputs.sort();
        let mut expected = vec![(h, vec![2]), (tok, vec![30])];
        expected.sort();
        assert_eq!(inputs, expected);
        // The current slot is untouched: `h` was read, not moved.
        assert!(st.has_input(dec, 0, false));
        assert!(!st.has_input(dec, 0, true));
    }

    #[test]
    fn two_node_loop_accepts_loop_back_and_speculates_while_first_member_completed() {
        let mut it = StrToId::default();
        let nodes = vec![
            node("a", &["tok"], &[], vec![edge("tok", "a"), edge("tick", "b")]),
            node("b", &["tick"], &[], vec![]),
        ];
        let loops = vec![LoopArg {
            name: "L".into(), max_iters: 2, parent: None,
            member_nodes: vec!["a".into(), "b".into()],
            outputs: vec![edge("tok", "emit_to_client")],
            accumulated: vec![],
            loop_back: vec![("tok".into(), "a".into())],
            external_inputs: vec![],
        }];
        let g = compile_one(&mut it, &nodes, &loops).unwrap();
        let (a, b) = (0, 1);
        let mut st = RequestState::new(g.clone());

        // Fresh request: speculation on `a` sees loop-back edge with `is_new_loop_iter = true`.
        let fresh_edges = vec![routed(&g.node(a).outputs[0], vec![])];
        let spec_fresh = st.ingest_for_speculation(&g, a, &fresh_edges);
        assert_eq!(spec_fresh.len(), 1);
        assert_eq!(spec_fresh[0].node, a);
        assert!(spec_fresh[0].is_new_loop_iter);

        // Iter 0: `a` completes -> `a.completed = true`, loop still active waiting on `b`.
        st.ingest(&g, a, 0, &[t(1)], true, false);
        st.clear_consumed_inputs(&g, a);
        let done_a = st.complete(&g, a, &[vec![t(10)], vec![t(20)]]);
        assert!(st.ingest_in_pass(&g, done_a.pass_walk, done_a.walk_pass_id, a, 0, &[t(10)], true, false));
        assert!(st.ingest_in_pass(&g, done_a.pass_walk, done_a.walk_pass_id, b, 0, &[t(20)], true, false));

        // While `a` is completed in iter 0, `b` completing iter 0 advances the loop cleanly.
        assert!(!st.passes[0].loops[0].done);
        st.clear_consumed_inputs(&g, b);
        st.complete(&g, b, &[]);
        assert_eq!(st.loop_iter(0), 1);
        assert!(st.is_ready(a), "loop-back token promoted to a.cur");
        assert_eq!(st.input_tensors(a, false)[0].1[0].uuid, 10);
    }

    #[test]
    fn a_top_level_external_input_is_not_carried_into_next_iter() {
        let mut it = StrToId::default();
        let nodes = vec![
            node("top", &["a", "b"], &[], vec![edge("b", "top")]),
        ];
        let g = compile_one(&mut it, &nodes, &[]).unwrap();
        let top = 0;
        let mut st = RequestState::new(g.clone());
        st.ingest(&g, top, 0, &[t(1)], true, false);
        st.ingest(&g, top, 1, &[t(2)], true, false);
        assert!(st.ingest(&g, top, 1, &[t(3)], true, false));
        assert!(!st.ready_for_speculation(top, true, false));
        assert_eq!(st.input_tensors(top, true).len(), 1);
    }
}
