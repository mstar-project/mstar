//! The batched surface. One `GraphRuntime` per worker owns every request's
//! walk state; Python calls it once per forward pass, not once per request.

use crate::shard::{Group, ShardMap};
use crate::spec::*;
use crate::state::{RequestState, RoutedEdge, TensorRef};
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

const EMIT_TO_CLIENT: &str = "emit_to_client";
const EMPTY_DESTINATION: &str = "";

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

#[pyclass]
pub struct GraphRuntime {
    interner: StrToId,
    graph: GraphRef,
    shard: ShardMap,
    states: Vec<RequestState>, // dense: request handle == index
    rid_names: Vec<String>,
    free: Vec<u32>,
}

impl GraphRuntime {
    fn nid(&self, name: &str) -> Option<NodeId> {
        self.interner.get(name).and_then(|s| self.graph.by_name.get(&s).copied())
    }
    fn lid(&self, name: &str) -> Option<LoopId> {
        self.interner.get(name).and_then(|s| self.graph.loop_by_name.get(&s).copied())
    }
}

#[pymethods]
impl GraphRuntime {
    #[new]
    #[pyo3(signature = (nodes, loops, workers, me, leader_nodes))]
    fn new(
        nodes: Vec<NodeArg>,
        loops: Vec<LoopArg>,
        workers: Vec<String>,
        me: String,
        leader_nodes: Vec<String>,
    ) -> PyResult<Self> {
        let mut it = StrToId::default();
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
        for n in &nodes {
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
            let outputs = n.outputs.iter().map(|e| mk_edge(&mut it, e, &local)).collect();
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
        for l in &loops {
            let parent = match &l.parent {
                Some(p) => Some(*loop_idx.get(p.as_str()).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!("unknown parent loop {p}"))
                })?),
                None => None,
            };
            let member_nodes: Vec<NodeId> = l.member_nodes.iter()
                .map(|m| local[m.as_str()]).collect();
            let outputs: Vec<EdgeSpec> = l.outputs.iter().map(|e| mk_edge(&mut it, e, &local)).collect();
            let accumulated: Vec<EdgeSpec> = l.accumulated.iter().map(|e| mk_edge(&mut it, e, &local)).collect();
            let resolve = |it: &mut StrToId, v: &Vec<(String, String)>| -> Vec<(Sym, NodeId)> {
                v.iter().filter_map(|(n, d)| local.get(d.as_str()).map(|&id| (it.intern(n), id))).collect()
            };
            let loop_back = resolve(&mut it, &l.loop_back);
            let external_inputs = resolve(&mut it, &l.external_inputs);
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
        let graph = Arc::new(CompiledGraph {
            nodes: node_specs, loops: loop_specs, by_name, loop_by_name, root_nodes, root_loops,
        });

        let wsyms: Vec<Sym> = workers.iter().map(|w| it.intern(w)).collect();
        let me_sym = it.intern(&me);
        let tp_rank = wsyms.iter().position(|&w| w == me_sym).unwrap_or(0) as u32;
        let group = Group { workers: wsyms, tp_size: workers.len() as u32, tp_rank };
        let node_group = graph.nodes.iter().map(|n| (n.name, 0u32)).collect();
        let shard = ShardMap { node_group, groups: vec![group], shard_dim: FxHashMap::default(), me: me_sym };

        Ok(Self { interner: it, graph, shard, states: vec![], rid_names: vec![], free: vec![] })
    }

    /// Batched request admission. Returns dense handles Python keys by.
    fn add_requests(&mut self, rids: Vec<String>) -> Vec<u32> {
        rids.into_iter().map(|rid| match self.free.pop() {
            Some(h) => {
                self.states[h as usize].reset();
                self.rid_names[h as usize] = rid;
                h
            }
            None => {
                self.states.push(RequestState::new(self.graph.clone()));
                self.rid_names.push(rid);
                (self.states.len() - 1) as u32
            }
        }).collect()
    }

    fn remove_requests(&mut self, handles: Vec<u32>) {
        for h in handles {
            self.states[h as usize].reset();
            self.free.push(h);
        }
    }

    /// Seed one signal into many requests at once.
    fn ingest_batch(&mut self, handles: Vec<u32>, node: &str, signal: &str, uuids: Vec<u64>) -> PyResult<()> {
        let nid = self.nid(node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        let ssym = self.interner.get(signal).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown signal {signal}"))
        })?;
        let Some(slot) = self.graph.node(nid).slot_of(ssym) else {
            return Err(pyo3::exceptions::PyKeyError::new_err(
                format!("{node} takes no input {signal}")));
        };
        for (i, h) in handles.iter().enumerate() {
            let t = vec![TensorRef { uuid: uuids[i], dim0: 8, nbytes: 2048, offset: 0 }];
            self.states[*h as usize].ingest(nid, slot, t, true);
        }
        Ok(())
    }

    /// THE hot path. One call replaces worker.py's `for rid, wg_id in ...`
    /// loop: mark-complete + loop bookkeeping + fanout + local re-ingest for
    /// the whole batch, returning wire-ready blobs.
    ///
    /// `uuids` / `tlens` are flat: `tlens[r * n_out + e]` tensors for edge `e`
    /// of request `r`, drawn in order from `uuids`.
    fn complete_and_route_batch(
        &mut self,
        py: Python<'_>,
        node: &str,
        handles: Vec<u32>,
        uuids: Vec<u64>,
        tlens: Vec<u32>,
    ) -> PyResult<BatchRouting> {
        let nid = self.nid(node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        let nsym = self.graph.node(nid).name;
        let n_out = self.graph.node(nid).outputs.len();

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
                out_tensors.push((0..n).map(|k| TensorRef {
                    uuid: uuids[cursor + k], dim0: 8, nbytes: 2048, offset: 0,
                }).collect());
                cursor += n;
            }

            let routed: Vec<RoutedEdge> = self.states[h as usize].complete(nid, &out_tensors);
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
                        // A re-injected loop input is already in the slot.
                        if e.persist_for_loop {
                            n_local += 1;
                            continue;
                        }
                        let dsym = self.graph.node(d).name;
                        self.shard.fanout(e.name, nsym, dsym, &e.tensors, &mut fan);
                        for f in fan.iter() {
                            if f.worker == self.shard.me {
                                if let Some(slot) = self.graph.node(d).slot_of(e.name) {
                                    self.states[h as usize].ingest(d, slot, e.tensors.clone(), true);
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
            if self.states[h as usize].is_done {
                completed.push(r as u32);
                self.states[h as usize].reset();
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
    fn ready_scan(&self, last_batch_num: Vec<u32>) -> Option<(String, Vec<u32>)> {
        let mut per_node: Vec<Vec<u32>> = vec![Vec::new(); self.graph.nodes.len()];
        for (h, st) in self.states.iter().enumerate() {
            for (wi, &word) in st.ready.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let b = w.trailing_zeros();
                    w &= w - 1;
                    let nid = wi as u32 * 64 + b;
                    if self.graph.node(nid).is_leader {
                        per_node[nid as usize].push(h as u32);
                    }
                }
            }
        }
        let best = per_node.iter().enumerate().filter(|(_, v)| !v.is_empty())
            .min_by_key(|(i, _)| last_batch_num.get(*i).copied().unwrap_or(0))?;
        Some((self.interner.name(self.graph.node(best.0 as u32).name).to_string(), best.1.clone()))
    }

    /// Python's `WorkerGraphQueues.pop_ready_nodes`: take the node off the
    /// ready set for these requests. All-or-none is the scheduler's job.
    fn pop_ready(&mut self, node: &str, handles: Vec<u32>) -> PyResult<Vec<u32>> {
        let nid = self.nid(node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        Ok(handles.into_iter()
            .filter(|h| self.states[*h as usize].take_for_schedule(nid))
            .collect())
    }

    fn ready_nodes(&self, handle: u32) -> Vec<String> {
        let st = &self.states[handle as usize];
        (0..self.graph.nodes.len() as u32).filter(|&n| st.is_ready(n))
            .map(|n| self.interner.name(self.graph.node(n).name).to_string()).collect()
    }

    fn ready_for_streaming(&self, handle: u32) -> Vec<String> {
        let st = &self.states[handle as usize];
        (0..self.graph.nodes.len() as u32)
            .filter(|&n| st.ready_streaming[(n / 64) as usize] >> (n % 64) & 1 == 1)
            .map(|n| self.interner.name(self.graph.node(n).name).to_string()).collect()
    }

    fn is_done(&self, handle: u32) -> bool {
        self.states[handle as usize].is_done
    }

    /// Batched loop stop (EOS). Returns the loop-back (signal, dest) pairs the
    /// caller must drop from this iteration's routing.
    fn stop_loops_batch(&mut self, handles: Vec<u32>, loop_name: &str) -> Vec<(String, String)> {
        let Some(lid) = self.lid(loop_name) else { return vec![] };
        let mut pairs = Vec::new();
        for h in handles {
            pairs = self.states[h as usize].register_loop_finish(lid).to_vec();
        }
        pairs.into_iter().map(|(s, d)| (
            self.interner.name(s).to_string(),
            self.interner.name(self.graph.node(d).name).to_string(),
        )).collect()
    }

    fn loop_iters(&self, handles: Vec<u32>, loop_name: &str) -> Vec<u32> {
        let Some(lid) = self.lid(loop_name) else { return vec![] };
        handles.iter().map(|h| self.states[*h as usize].loop_iter(lid)).collect()
    }

    /// Python's `get_loop_indices`: every loop's curr_iter.
    fn loop_indices(&self, handle: u32) -> Vec<(String, u32)> {
        let st = &self.states[handle as usize];
        st.loop_indices().into_iter().enumerate()
            .map(|(i, v)| (self.interner.name(self.graph.lp(i as LoopId).name).to_string(), v))
            .collect()
    }

    /// Python's `get_nested_loop_idxs_for_node`.
    fn nested_loop_idxs_for_node(&self, handle: u32, node: &str) -> PyResult<NestedLoopIdx> {
        let nid = self.nid(node).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {node}"))
        })?;
        let st = &self.states[handle as usize];
        let Some(lid) = self.graph.node(nid).loop_id else {
            return Ok(NestedLoopIdx {
                loop_name_order: vec![], loop_indices: vec![],
                wg_fwd_pass_idx: st.num_times_run,
            });
        };
        Ok(NestedLoopIdx {
            loop_name_order: self.graph.loop_order(lid).into_iter()
                .map(|l| self.interner.name(self.graph.lp(l).name).to_string()).collect(),
            loop_indices: self.loop_indices(handle),
            wg_fwd_pass_idx: st.num_times_run,
        })
    }

    /// Python's `ingest_for_speculation`: returns (node, is_new_loop_iter,
    /// loop_name) for each destination that would be ready.
    fn ingest_for_speculation(&mut self, handle: u32, source: &str) -> PyResult<Vec<(String, bool, Option<String>)>> {
        let src = self.nid(source).ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("unknown node {source}"))
        })?;
        // Anticipated edges are the source's declared outputs, tensors empty:
        // speculation only needs readiness, not values.
        let edges: Vec<RoutedEdge> = self.graph.node(src).outputs.iter().map(|e| RoutedEdge {
            name: e.name, dest: e.dest, persist: e.persist, new_token: e.new_token,
            streaming: e.streaming, modality: e.modality, tensors: vec![],
            persist_for_loop: false,
        }).collect();
        let found = self.states[handle as usize].ingest_for_speculation(src, &edges);
        Ok(found.into_iter().map(|s| (
            self.interner.name(self.graph.node(s.node).name).to_string(),
            s.is_new_loop_iter,
            s.loop_id.map(|l| self.interner.name(self.graph.lp(l).name).to_string()),
        )).collect())
    }

    fn clear_speculative_inputs(&mut self, handle: u32) {
        self.states[handle as usize].clear_speculative_inputs();
    }

    fn num_times_run(&self, handle: u32) -> u32 {
        self.states[handle as usize].num_times_run
    }
}
