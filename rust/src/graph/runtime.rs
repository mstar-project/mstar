//! The batched surface. One `GraphRuntime` per worker owns every request's
//! walk state; Python calls it once per forward pass, not once per request.

use crate::graph::shard::{Group, ShardMap};
use crate::graph::spec::*;
use crate::graph::state::{RequestState, RoutedEdge, TensorRef};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::FxHashMap;
use serde::Serialize;
use std::sync::Arc;

// -- spec handed over from Python (dicts; the real port's compile seam is
//    mstar/graph/rust_core.py doing the same from GraphSection) --------------

#[derive(FromPyObject)]
struct EdgeArg {
    #[pyo3(item)] name: String,
    #[pyo3(item)] dest: String,
    #[pyo3(item)] persist: bool,
    #[pyo3(item)] new_token: bool,
    #[pyo3(item)] streaming: bool,
    #[pyo3(item)] modality: String,
}

#[derive(FromPyObject)]
struct NodeArg {
    #[pyo3(item)] name: String,
    #[pyo3(item)] inputs: Vec<String>,
    #[pyo3(item)] streaming_inputs: Vec<String>,
    #[pyo3(item)] outputs: Vec<EdgeArg>,
}

#[derive(FromPyObject)]
struct LoopArg {
    #[pyo3(item)] name: String,
    #[pyo3(item)] max_iters: u32,
    #[pyo3(item)] parent: Option<String>,
    /// Directly-owned nodes only — a node inside a child loop belongs there.
    #[pyo3(item)] member_nodes: Vec<String>,
    #[pyo3(item)] outputs: Vec<EdgeArg>,
    #[pyo3(item)] accumulated: Vec<EdgeArg>,
    /// Verbatim `Loop._loop_back_inputs` / `_external_inputs`.
    #[pyo3(item)] loop_back: Vec<(String, String)>,
    #[pyo3(item)] external_inputs: Vec<(String, String)>,
}

#[derive(FromPyObject)]
struct WorkerGraphArg {
    #[pyo3(item)] wg_id: String,
    /// Which graph walks this worker graph is active in — routing uses it to
    /// tell a destination owned by a SIBLING worker graph on this worker from
    /// one that leaves the worker entirely.
    #[pyo3(item)] graph_walks: Vec<String>,
    #[pyo3(item)] nodes: Vec<NodeArg>,
    #[pyo3(item)] loops: Vec<LoopArg>,
    #[pyo3(item)] leader_nodes: Vec<String>,
}

/// Must match `mstar/graph/special_destinations.py`.
const EMIT_TO_CLIENT: &str = "emit_to_client";
const EMPTY_DESTINATION: &str = "";

/// `(uuid, dim0, nbytes, offset)` — the tuple Python hands tensors over as.
#[inline]
fn mk_tensor(t: (u64, i64, i64, i64)) -> TensorRef {
    TensorRef { uuid: t.0, dim0: t.1, nbytes: t.2, offset: t.3 }
}

/// One edge as it goes on the wire to a peer worker. Never becomes a Python
/// object: peer edges come back as one msgpack blob per destination.
#[derive(Serialize)]
struct WireEdge<'a> {
    rid: &'a str,
    name: &'a str,
    next_node: &'a str,
    tensors: Vec<TensorRef>,
    persist: bool,
    streaming: bool,
}

#[pyclass]
pub struct BatchRouting {
    #[pyo3(get)] pub to_workers: Vec<(String, Py<PyBytes>)>,
    /// Flat (rid_index, uuid) for `tensor_manager.set_persist`.
    #[pyo3(get)] pub persist: Vec<(u32, u64)>,
    /// (rid_index, signal, modality, uuid) for EMIT_TO_CLIENT.
    #[pyo3(get)] pub emit: Vec<(u32, String, String, u64)>,
    #[pyo3(get)] pub new_tokens: Vec<(u32, String, u64)>,
    /// rid indices whose worker graph finished this pass.
    #[pyo3(get)] pub completed: Vec<u32>,
    #[pyo3(get)] pub n_local: u32,
}

/// Python's `NestedLoopIndices`.
#[pyclass]
#[derive(Clone)]
pub struct NestedLoopIdx {
    #[pyo3(get)] pub loop_name_order: Vec<String>,
    #[pyo3(get)] pub loop_indices: Vec<(String, u32)>,
    #[pyo3(get)] pub wg_fwd_pass_idx: u32,
}

/// Every worker graph on one worker, plus every request's walk state in each.
///
/// Scoped to the WORKER rather than to a single worker graph so that routing
/// can see which local graph owns a destination node. Python drives it through
/// per-worker-graph views (`mstar.graph.runtime.RustGraphRuntime`), each of
/// which passes its own `wg` index — so the harness interface stays
/// per-worker-graph, matching `WorkerGraphIO`.
#[pyclass]
pub struct GraphRuntime {
    interner: StrToId,
    graphs: Vec<GraphRef>,
    wg_ids: Vec<String>,
    shard: ShardMap,
    /// `[wg][handle]`; None where the request is not registered with that
    /// worker graph (Python only adds a request to the graphs its partition
    /// uses).
    states: Vec<Vec<Option<RequestState>>>,
    rid_names: Vec<String>,
    rid_to_handle: FxHashMap<String, u32>,
    free: Vec<u32>,
    /// (graph walk, node) -> owning worker-graph index.
    node_owner: FxHashMap<(Sym, Sym), u32>,
}

/// Borrow one request's state in one worker graph. A macro, not a method, so
/// the borrow ends at the statement and `self.shard` stays reachable.
macro_rules! st {
    ($s:expr, $wg:expr, $h:expr) => {
        $s.states[$wg as usize][$h as usize]
            .as_ref()
            .expect("request is not registered with this worker graph")
    };
}
macro_rules! st_mut {
    ($s:expr, $wg:expr, $h:expr) => {
        $s.states[$wg as usize][$h as usize]
            .as_mut()
            .expect("request is not registered with this worker graph")
    };
}

impl GraphRuntime {
    fn g(&self, wg: u32) -> &GraphRef {
        &self.graphs[wg as usize]
    }
    fn nid(&self, wg: u32, name: &str) -> Option<NodeId> {
        self.interner.get(name).and_then(|s| self.g(wg).by_name.get(&s).copied())
    }
    fn lid(&self, wg: u32, name: &str) -> Option<LoopId> {
        self.interner.get(name).and_then(|s| self.g(wg).loop_by_name.get(&s).copied())
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

/// Compile one worker graph's spec into a shared `CompiledGraph`.
fn compile_one(
    it: &mut StrToId,
    nodes: &[NodeArg],
    loops: &[LoopArg],
    leader_nodes: &[String],
) -> PyResult<GraphRef> {
    let local: FxHashMap<&str, NodeId> = nodes.iter().enumerate()
        .map(|(i, n)| (n.name.as_str(), i as NodeId)).collect();

    let mk_edge = |it: &mut StrToId, e: &EdgeArg, local: &FxHashMap<&str, NodeId>| {
        let dest = if e.dest == EMIT_TO_CLIENT {
            Dest::EmitToClient
        } else if e.dest == EMPTY_DESTINATION {
            Dest::Empty
        } else if let Some(&id) = local.get(e.dest.as_str()) {
            Dest::Local(id)
        } else {
            Dest::External(it.intern(&e.dest))
        };
        let dest_slot = match dest {
            Dest::Local(id) => nodes[id as usize].inputs.iter()
                .position(|s| *s == e.name).unwrap_or(0) as u8,
            _ => 0,
        };
        EdgeSpec {
            name: it.intern(&e.name), dest, persist: e.persist,
            new_token: e.new_token, streaming: e.streaming,
            modality: it.intern(&e.modality), dest_slot,
        }
    };

    let mut node_specs = Vec::with_capacity(nodes.len());
    for n in nodes {
        if n.inputs.len() > MAX_INPUTS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "node {} has {} inputs; readiness mask holds {}",
                n.name, n.inputs.len(), MAX_INPUTS)));
        }
        let name = it.intern(&n.name);
        let inputs: Vec<Sym> = n.inputs.iter().map(|s| it.intern(s)).collect();
        let full = if inputs.len() == 64 { u64::MAX } else { (1u64 << inputs.len()) - 1 };
        let mut streaming_mask = 0u64;
        for s in &n.streaming_inputs {
            let sym = it.intern(s);
            if let Some(i) = inputs.iter().position(|&x| x == sym) {
                streaming_mask |= 1 << i;
            }
        }
        let outputs = n.outputs.iter().map(|e| mk_edge(it, e, &local)).collect();
        node_specs.push(NodeSpec {
            name, inputs, full_mask: full, streaming_mask,
            only_streaming: full != 0 && streaming_mask == full,
            outputs, loop_id: None,
            is_leader: leader_nodes.iter().any(|l| *l == n.name),
        });
    }

    // Loops: resolve parents by name (order-independent), then derive
    // child_loops and each node's innermost loop.
    let loop_idx: FxHashMap<&str, LoopId> = loops.iter().enumerate()
        .map(|(i, l)| (l.name.as_str(), i as LoopId)).collect();
    let mut loop_specs = Vec::with_capacity(loops.len());
    for l in loops {
        let parent = match &l.parent {
            Some(p) => Some(*loop_idx.get(p.as_str()).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("unknown parent loop {p}"))
            })?),
            None => None,
        };
        let member_nodes: Vec<NodeId> = l.member_nodes.iter()
            .map(|m| local[m.as_str()]).collect();
        let outputs: Vec<EdgeSpec> = l.outputs.iter().map(|e| mk_edge(it, e, &local)).collect();
        let accumulated: Vec<EdgeSpec> = l.accumulated.iter().map(|e| mk_edge(it, e, &local)).collect();
        let resolve = |it: &mut StrToId, v: &Vec<(String, String)>| -> Vec<(Sym, NodeId)> {
            v.iter().filter_map(|(n, d)| local.get(d.as_str()).map(|&id| (it.intern(n), id))).collect()
        };
        let loop_back = resolve(it, &l.loop_back);
        let external_inputs = resolve(it, &l.external_inputs);
        loop_specs.push(LoopSpec {
            name: it.intern(&l.name), max_iters: l.max_iters, parent,
            member_nodes, child_loops: Vec::new(),
            output_names: outputs.iter().map(|e| e.name).collect(),
            accum_names: accumulated.iter().map(|e| e.name).collect(),
            outputs, accumulated, loop_back, external_inputs,
        });
    }
    for i in 0..loop_specs.len() {
        if let Some(p) = loop_specs[i].parent {
            loop_specs[p as usize].child_loops.push(i as LoopId);
        }
    }
    for (i, l) in loop_specs.iter().enumerate() {
        for &m in &l.member_nodes {
            node_specs[m as usize].loop_id = Some(i as LoopId);
        }
    }

    let owned: Vec<NodeId> = loop_specs.iter().flat_map(|l| l.member_nodes.clone()).collect();
    let root_nodes: Vec<NodeId> = (0..nodes.len() as NodeId)
        .filter(|n| !owned.contains(n)).collect();
    let root_loops: Vec<LoopId> = loop_specs.iter().enumerate()
        .filter(|(_, l)| l.parent.is_none()).map(|(i, _)| i as LoopId).collect();

    let by_name = node_specs.iter().enumerate().map(|(i, n)| (n.name, i as NodeId)).collect();
    let loop_by_name = loop_specs.iter().enumerate().map(|(i, l)| (l.name, i as LoopId)).collect();
    Ok(Arc::new(CompiledGraph {
        nodes: node_specs, loops: loop_specs, by_name, loop_by_name, root_nodes, root_loops,
    }))
}

#[pymethods]
impl GraphRuntime {
    #[new]
    #[pyo3(signature = (worker_graphs, workers, me))]
    fn new(
        worker_graphs: Vec<WorkerGraphArg>,
        workers: Vec<String>,
        me: String,
    ) -> PyResult<Self> {
        let mut it = StrToId::default();
        let mut graphs = Vec::with_capacity(worker_graphs.len());
        let mut wg_ids = Vec::with_capacity(worker_graphs.len());
        let mut node_owner: FxHashMap<(Sym, Sym), u32> = FxHashMap::default();

        for (wg, wga) in worker_graphs.iter().enumerate() {
            let graph = compile_one(&mut it, &wga.nodes, &wga.loops, &wga.leader_nodes)?;
            for walk in &wga.graph_walks {
                let walk_sym = it.intern(walk);
                for n in &graph.nodes {
                    // Last write wins, matching the Python inverted index in
                    // WorkerGraphsManager.__post_init__.
                    node_owner.insert((walk_sym, n.name), wg as u32);
                }
            }
            wg_ids.push(wga.wg_id.clone());
            graphs.push(graph);
        }

        let wsyms: Vec<Sym> = workers.iter().map(|w| it.intern(w)).collect();
        let me_sym = it.intern(&me);
        let tp_rank = wsyms.iter().position(|&w| w == me_sym).unwrap_or(0) as u32;
        let group = Group { workers: wsyms, tp_size: workers.len() as u32, tp_rank };
        let node_group = graphs.iter()
            .flat_map(|g| g.nodes.iter().map(|n| (n.name, 0u32)))
            .collect();
        let shard = ShardMap {
            node_group, groups: vec![group], shard_dim: FxHashMap::default(), me: me_sym,
        };

        let states: Vec<Vec<Option<RequestState>>> =
            (0..graphs.len()).map(|_| Vec::new()).collect();
        Ok(Self {
            interner: it, graphs, wg_ids, shard, states,
            rid_names: vec![], rid_to_handle: FxHashMap::default(),
            free: vec![], node_owner,
        })
    }

    /// Index of a worker graph by id — what Python's per-graph view passes
    /// back as `wg`.
    fn wg_index(&self, wg_id: &str) -> PyResult<u32> {
        self.wg_ids.iter().position(|w| w == wg_id)
            .map(|i| i as u32)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(
                format!("unknown worker graph {wg_id}")))
    }

    /// Which local worker graph owns `node` in `graph_walk`, if any.
    fn owner_of(&self, graph_walk: &str, node: &str) -> Option<u32> {
        let walk = self.interner.get(graph_walk)?;
        let n = self.interner.get(node)?;
        self.node_owner.get(&(walk, n)).copied()
    }

    /// Register requests with worker graph `wg`.
    ///
    /// Handles are worker-wide and minted once per rid, so the same handle
    /// addresses that request in every worker graph it belongs to.
    fn add_requests(&mut self, wg: u32, rids: Vec<String>) -> Vec<u32> {
        let mut out = Vec::with_capacity(rids.len());
        for rid in rids {
            let handle = match self.rid_to_handle.get(&rid) {
                Some(&h) => h,
                None => {
                    let h = match self.free.pop() {
                        Some(h) => {
                            self.rid_names[h as usize] = rid.clone();
                            h
                        }
                        None => {
                            self.rid_names.push(rid.clone());
                            for per_wg in self.states.iter_mut() {
                                per_wg.push(None);
                            }
                            (self.rid_names.len() - 1) as u32
                        }
                    };
                    self.rid_to_handle.insert(rid.clone(), h);
                    h
                }
            };
            let graph = self.graphs[wg as usize].clone();
            self.states[wg as usize][handle as usize] = Some(RequestState::new(graph));
            out.push(handle);
        }
        out
    }

    /// Drop these requests from worker graph `wg`. The handle is recycled
    /// only once no worker graph still holds state for it.
    fn remove_requests(&mut self, wg: u32, handles: Vec<u32>) {
        for h in handles {
            self.states[wg as usize][h as usize] = None;
            let still_used = self.states.iter()
                .any(|per_wg| per_wg[h as usize].is_some());
            if !still_used {
                self.rid_to_handle.remove(&self.rid_names[h as usize]);
                self.free.push(h);
            }
        }
    }

    /// Ingest one edge into one request. The per-event mirror the shadow
    /// harness drives; `complete_and_route_batch` is the batched path.
    /// Returns false when the edge is rejected (both slots full), matching
    /// `GraphNode.ingest_input`.
    #[pyo3(signature = (wg, handle, node, signal, tensors, can_buffer=true, final_chunk=false))]
    fn ingest(
        &mut self, wg: u32, handle: u32, node: &str, signal: &str,
        tensors: Vec<(u64, i64, i64, i64)>, can_buffer: bool, final_chunk: bool,
    ) -> PyResult<bool> {
        let Some(nid) = self.nid(wg, node) else { return Ok(false) };
        let Some(ssym) = self.interner.get(signal) else { return Ok(false) };
        let Some(slot) = self.g(wg).node(nid).slot_of(ssym) else { return Ok(false) };
        let t = tensors.into_iter().map(mk_tensor).collect();
        Ok(st_mut!(self, wg, handle).ingest(nid, slot, t, can_buffer, final_chunk))
    }

    /// The inputs a node has received: (signal, [uuid], final_stream_chunk).
    fn input_slots(
        &self, wg: u32, handle: u32, node: &str, next_iter: bool,
    ) -> PyResult<Vec<(String, Vec<u64>, bool)>> {
        let nid = self.nid(wg, node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        Ok(st!(self, wg, handle).input_tensors(nid, next_iter).into_iter()
            .map(|(name, tensors, final_chunk)| (
                self.interner.name(name).to_string(),
                tensors.iter().map(|t| t.uuid).collect(),
                final_chunk,
            )).collect())
    }

    /// Seed one signal into many requests at once; one tensor each.
    fn ingest_batch(
        &mut self, wg: u32, handles: Vec<u32>, node: &str, signal: &str,
        tensors: Vec<(u64, i64, i64, i64)>,
    ) -> PyResult<()> {
        let nid = self.nid(wg, node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        let ssym = self.interner.get(signal).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown signal {signal}"))
        })?;
        let Some(slot) = self.g(wg).node(nid).slot_of(ssym) else {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                format!("{node} takes no input {signal}")));
        };
        for (i, h) in handles.iter().enumerate() {
            let t = vec![mk_tensor(tensors[i])];
            st_mut!(self, wg, *h).ingest(nid, slot, t, true, false);
        }
        Ok(())
    }

    /// THE hot path. One call replaces worker.py's `for rid, wg_id in ...`
    /// loop: mark-complete + loop bookkeeping + fanout + local re-ingest for
    /// the whole batch, returning wire-ready blobs.
    ///
    /// Complete a node for ONE request and hand back the edges to route,
    /// WITHOUT routing them. The per-event mirror: Python's
    /// `mark_node_complete` also leaves routing to the caller, so a shadow
    /// run can compare state after every single event rather than only at
    /// settle points. `complete_and_route_batch` is the fast path.
    ///
    /// Each edge is `(name, next_node, persist, new_token, streaming,
    /// modality, persist_for_loop, n_tensors)`; tensor identity stays on the
    /// Python side, which owns it.
    #[allow(clippy::type_complexity)]
    fn complete_only(
        &mut self,
        wg: u32,
        handle: u32,
        node: &str,
        tensors: Vec<(u64, i64, i64, i64)>,
        tlens: Vec<u32>,
    ) -> PyResult<(
        Vec<(String, String, bool, bool, bool, String, bool, usize)>,
        Vec<(String, String)>,
    )> {
        let nid = self.nid(wg, node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        let n_out = self.g(wg).node(nid).outputs.len();
        let mut cursor = 0usize;
        let mut out_tensors: Vec<Vec<TensorRef>> = Vec::with_capacity(n_out);
        for e in 0..n_out {
            let n = tlens.get(e).copied().unwrap_or(0) as usize;
            out_tensors.push((0..n).map(|k| mk_tensor(tensors[cursor + k])).collect());
            cursor += n;
        }
        let (routed, filtered) = st_mut!(self, wg, handle).complete(nid, &out_tensors);
        let edges = routed.into_iter().map(|e| (
            self.interner.name(e.name).to_string(),
            self.dest_name(wg, e.dest),
            e.persist, e.new_token, e.streaming,
            self.interner.name(e.modality).to_string(),
            e.persist_for_loop,
            e.tensors.len(),
        )).collect();
        let filtered = filtered.into_iter().map(|(s, d)| (
            self.interner.name(s).to_string(),
            self.interner.name(self.g(wg).node(d).name).to_string(),
        )).collect();
        Ok((edges, filtered))
    }

    /// Flat payload: `tlens[r * n_out + e]` tensors for edge `e` of request
    /// `r`, drawn in order from `tensors` as `(uuid, dim0, nbytes, offset)`.
    /// `dim0` / `nbytes` are what the sharded fanout slices on.
    fn complete_and_route_batch(
        &mut self,
        py: Python<'_>,
        wg: u32,
        node: &str,
        handles: Vec<u32>,
        tensors: Vec<(u64, i64, i64, i64)>,
        tlens: Vec<u32>,
    ) -> PyResult<BatchRouting> {
        let nid = self.nid(wg, node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        let nsym = self.g(wg).node(nid).name;
        let n_out = self.g(wg).node(nid).outputs.len();

        let mut per_worker: FxHashMap<Sym, Vec<WireEdge>> = FxHashMap::default();
        let (mut persist, mut emit, mut new_tokens, mut completed) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut n_local = 0u32;
        let mut fan = Vec::with_capacity(4);
        let mut cursor = 0usize;

        for (r, &h) in handles.iter().enumerate() {
            let mut out_tensors: Vec<Vec<TensorRef>> = Vec::with_capacity(n_out);
            for e in 0..n_out {
                let n = tlens.get(r * n_out + e).copied().unwrap_or(0) as usize;
                out_tensors.push((0..n).map(|k| mk_tensor(tensors[cursor + k])).collect());
                cursor += n;
            }

            let (routed, _filtered): (Vec<RoutedEdge>, _) =
                st_mut!(self, wg, h).complete(nid, &out_tensors);
            for e in routed {
                if e.persist {
                    for t in &e.tensors { persist.push((r as u32, t.uuid)) }
                }
                if e.new_token {
                    for t in &e.tensors {
                        new_tokens.push((r as u32, self.interner.name(e.name).to_string(), t.uuid));
                    }
                }
                match e.dest {
                    Dest::Empty => {}
                    Dest::EmitToClient => {
                        for t in &e.tensors {
                            emit.push((
                                r as u32,
                                self.interner.name(e.name).to_string(),
                                self.interner.name(e.modality).to_string(),
                                t.uuid,
                            ));
                        }
                    }
                    Dest::Local(d) => {
                        let dsym = self.g(wg).node(d).name;
                        self.shard.fanout(e.name, nsym, dsym, &e.tensors, &mut fan);
                        for f in fan.iter() {
                            if f.worker == self.shard.me {
                                if let Some(slot) = self.g(wg).node(d).slot_of(e.name) {
                                    st_mut!(self, wg, h).ingest(d, slot, e.tensors.clone(), true, false);
                                    n_local += 1;
                                }
                            } else if f.worker != Sym::MAX {
                                per_worker.entry(f.worker).or_default().push(WireEdge {
                                    rid: &self.rid_names[h as usize],
                                    name: self.interner.name(e.name),
                                    next_node: self.interner.name(dsym),
                                    tensors: e.tensors.clone(),
                                    persist: e.persist, streaming: e.streaming,
                                });
                            }
                        }
                    }
                    Dest::External(dsym) => {
                        self.shard.fanout(e.name, nsym, dsym, &e.tensors, &mut fan);
                        for f in fan.iter() {
                            if f.worker == Sym::MAX { continue }
                            per_worker.entry(f.worker).or_default().push(WireEdge {
                                rid: &self.rid_names[h as usize],
                                name: self.interner.name(e.name),
                                next_node: self.interner.name(dsym),
                                tensors: e.tensors.clone(),
                                persist: e.persist, streaming: e.streaming,
                            });
                        }
                    }
                }
            }
            if st!(self, wg, h).is_done {
                completed.push(r as u32);
                st_mut!(self, wg, h).reset();
            }
        }

        let to_workers = per_worker.into_iter().map(|(w, edges)| {
            let buf = rmp_serde::to_vec_named(&edges).unwrap();
            (self.interner.name(w).to_string(), PyBytes::new(py, &buf).unbind())
        }).collect();

        Ok(BatchRouting { to_workers, persist, emit, new_tokens, completed, n_local })
    }

    /// Replaces MicroScheduler's O(batch x nodes) Python scan: the (node,
    /// [handle]) group with the least-recent round-robin stamp.
    fn ready_scan(&self, wg: u32, last_batch_num: Vec<u32>) -> Option<(String, Vec<u32>)> {
        let mut per_node: Vec<Vec<u32>> = vec![Vec::new(); self.g(wg).nodes.len()];
        for (h, slot) in self.states[wg as usize].iter().enumerate() {
            let Some(st) = slot else { continue };
            for (wi, &word) in st.ready.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let b = w.trailing_zeros();
                    w &= w - 1;
                    let nid = wi as u32 * 64 + b;
                    if self.g(wg).node(nid).is_leader {
                        per_node[nid as usize].push(h as u32);
                    }
                }
            }
        }
        let best = per_node.iter().enumerate().filter(|(_, v)| !v.is_empty())
            .min_by_key(|(i, _)| last_batch_num.get(*i).copied().unwrap_or(0))?;
        Some((self.interner.name(self.g(wg).node(best.0 as u32).name).to_string(), best.1.clone()))
    }

    /// Python's `WorkerGraphQueues.pop_ready_nodes`: take the node off the
    /// ready set for these requests. All-or-none is the scheduler's job.
    fn pop_ready(&mut self, wg: u32, node: &str, handles: Vec<u32>) -> PyResult<Vec<u32>> {
        let nid = self.nid(wg, node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        Ok(handles.into_iter()
            .filter(|h| st_mut!(self, wg, *h).take_for_schedule(nid))
            .collect())
    }

    /// Undo a pop (Python's `push_back_node`, the OOM-hold path).
    fn push_back(&mut self, wg: u32, node: &str, handles: Vec<u32>) -> PyResult<()> {
        let nid = self.nid(wg, node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        for h in handles {
            st_mut!(self, wg, h).push_back(nid);
        }
        Ok(())
    }

    /// Python's `WorkerGraphIO.clear()` — end of a forward pass. Keeps the
    /// handle registered; `remove_requests` is the one that frees it.
    fn reset_request(&mut self, wg: u32, handle: u32) {
        st_mut!(self, wg, handle).reset();
    }

    fn ready_nodes(&self, wg: u32, handle: u32) -> Vec<String> {
        let st = st!(self, wg, handle);
        (0..self.g(wg).nodes.len() as u32).filter(|&n| st.is_ready(n))
            .map(|n| self.interner.name(self.g(wg).node(n).name).to_string()).collect()
    }

    fn ready_for_streaming(&self, wg: u32, handle: u32) -> Vec<String> {
        let st = st!(self, wg, handle);
        (0..self.g(wg).nodes.len() as u32)
            .filter(|&n| st.ready_streaming[(n / 64) as usize] >> (n % 64) & 1 == 1)
            .map(|n| self.interner.name(self.g(wg).node(n).name).to_string()).collect()
    }

    fn is_done(&self, wg: u32, handle: u32) -> bool {
        st!(self, wg, handle).is_done
    }

    /// Batched loop stop (EOS). Returns the loop-back (signal, dest) pairs the
    /// caller must drop from this iteration's routing.
    fn stop_loops_batch(&mut self, wg: u32, handles: Vec<u32>, loop_name: &str) -> Vec<(String, String)> {
        let Some(lid) = self.lid(wg, loop_name) else { return vec![] };
        let mut pairs = Vec::new();
        for h in handles {
            pairs = st_mut!(self, wg, h).register_loop_finish(lid).to_vec();
        }
        pairs.into_iter().map(|(s, d)| (
            self.interner.name(s).to_string(),
            self.interner.name(self.g(wg).node(d).name).to_string(),
        )).collect()
    }

    fn loop_iters(&self, wg: u32, handles: Vec<u32>, loop_name: &str) -> Vec<u32> {
        let Some(lid) = self.lid(wg, loop_name) else { return vec![] };
        handles.iter().map(|h| st!(self, wg, *h).loop_iter(lid)).collect()
    }

    /// Python's `get_loop_indices`: every loop's curr_iter.
    fn loop_indices(&self, wg: u32, handle: u32) -> Vec<(String, u32)> {
        st!(self, wg, handle).loop_indices().into_iter().enumerate()
            .map(|(i, v)| (self.interner.name(self.g(wg).lp(i as LoopId).name).to_string(), v))
            .collect()
    }

    /// Python's `get_nested_loop_idxs`, keyed by loop rather than by node.
    fn nested_loop_idxs_for_loop(&self, wg: u32, handle: u32, loop_name: &str) -> PyResult<NestedLoopIdx> {
        let lid = self.lid(wg, loop_name).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown loop {loop_name}"))
        })?;
        Ok(NestedLoopIdx {
            loop_name_order: self.g(wg).loop_order(lid).into_iter()
                .map(|l| self.interner.name(self.g(wg).lp(l).name).to_string()).collect(),
            loop_indices: self.loop_indices(wg, handle),
            wg_fwd_pass_idx: st!(self, wg, handle).num_times_run,
        })
    }

    /// Python's `get_nested_loop_idxs_for_node`.
    fn nested_loop_idxs_for_node(&self, wg: u32, handle: u32, node: &str) -> PyResult<NestedLoopIdx> {
        let nid = self.nid(wg, node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        let st = st!(self, wg, handle);
        let Some(lid) = self.g(wg).node(nid).loop_id else {
            return Ok(NestedLoopIdx {
                loop_name_order: vec![], loop_indices: vec![],
                wg_fwd_pass_idx: st.num_times_run,
            });
        };
        Ok(NestedLoopIdx {
            loop_name_order: self.g(wg).loop_order(lid).into_iter()
                .map(|l| self.interner.name(self.g(wg).lp(l).name).to_string()).collect(),
            loop_indices: self.loop_indices(wg, handle),
            wg_fwd_pass_idx: st.num_times_run,
        })
    }

    /// Python's `ingest_for_speculation`: returns (node, is_new_loop_iter,
    /// loop_name) for each destination that would be ready.
    fn ingest_for_speculation(&mut self, wg: u32, handle: u32, source: &str) -> PyResult<Vec<(String, bool, Option<String>)>> {
        let src = self.nid(wg, source).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {source}"))
        })?;
        // Anticipated edges are the source's declared outputs, tensors empty:
        // speculation only needs readiness, not values.
        let edges: Vec<RoutedEdge> = self.g(wg).node(src).outputs.iter().map(|e| RoutedEdge {
            name: e.name, dest: e.dest, persist: e.persist, new_token: e.new_token,
            streaming: e.streaming, modality: e.modality, tensors: vec![],
            persist_for_loop: false,
        }).collect();
        let found = st_mut!(self, wg, handle).ingest_for_speculation(src, &edges);
        Ok(found.into_iter().map(|s| (
            self.interner.name(self.g(wg).node(s.node).name).to_string(),
            s.is_new_loop_iter,
            s.loop_id.map(|l| self.interner.name(self.g(wg).lp(l).name).to_string()),
        )).collect())
    }

    fn clear_speculative_inputs(&mut self, wg: u32, handle: u32) {
        st_mut!(self, wg, handle).clear_speculative_inputs();
    }

    fn num_times_run(&self, wg: u32, handle: u32) -> u32 {
        st!(self, wg, handle).num_times_run
    }
}
