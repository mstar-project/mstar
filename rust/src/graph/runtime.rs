//! The batched surface. One `GraphRuntime` per worker owns every request's
//! walk state; Python calls it once per forward pass, not once per request.

use crate::graph::compile::{EMIT_TO_CLIENT, EMPTY_DESTINATION, LoopArg, NodeArg, compile_one};
use crate::graph::shard::{GroupTemplate, ShardingTemplate};
use crate::graph::spec::*;
use crate::graph::request::{RequestInfo, WgIndex, WorkerGraphMeta};
use crate::graph::state::{RequestState, RoutedEdge, SpecNode, TensorRef};
use crate::tensors::{SharedBookkeeping, TensorBookkeeping};
use pyo3::exceptions::PyValueError;
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

    /// A share of the SAME bookkeeper Python handed TensorStore, taken at
    /// construction. Routing reads descriptors and adjusts refcounts through
    /// it, so a copy would diverge from what the store believes.
    bookkeeping: SharedBookkeeping,
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
            let Some(state) = &mut self.states[wg as usize][rid as usize] else {
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
                    let Some(state) = &self.states[wg as usize][rid as usize]
                    else {
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
                persist: e.persist,
                new_token: e.new_token,
                streaming: e.streaming,
                modality: e.modality,
                tensors: vec![],
                persist_for_loop: false,
            })
            .collect();
        let state = self.states[wg as usize].get_mut(rid as usize)?.as_mut()?;
        let out = state.ingest_for_speculation(source, &edges);
        state.clear_speculative_inputs();
        Some(out)
    }

    fn spec_output(
        &self, g: &GraphRef, graph_walk: &str, sn: &SpecNode,
    ) -> (String, String, bool, Option<String>) {
        (
            self.interner.name(g.node(sn.node).name).to_string(),
            graph_walk.to_string(),
            sn.is_new_loop_iter,
            sn.loop_id
                .map(|lid| self.interner.name(g.lp(lid).name).to_string()),
        )
    }

    fn owner_of(&self, node: &str, walk: &str) -> Option<WgIndex> {
        let n = self.interner.get(node)?;
        let w = self.interner.get(walk)?;
        self.node_owner.get(&(w, n)).copied()
    }

    fn live_wgs(&self, walk: Sym) -> Vec<WgIndex> {
        self.walk_to_local_wgs.get(&walk).cloned().unwrap_or_default()
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
    #[pyo3(signature = (worker_graphs, remote_worker_graphs, sharding, bookkeeping, me))]
    fn new(
        worker_graphs: Vec<WorkerGraphArg>,
        // Owned by other workers; needed to answer "who runs this node" when
        // an output leaves this worker.
        remote_worker_graphs: Vec<RemoteWorkerGraphArg>,
        sharding: ShardingArg,
        bookkeeping: PyRef<'_, TensorBookkeeping>,
        me: String,
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
            bookkeeping: bookkeeping.share(),
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

        // Open this partition's worker graphs. A worker graph can serve more
        // than one partition, so the walk state is created once and the
        // per-partition lists just point at it.
        let live = self.live_wgs(walk_sym);
        for &wg in &live {
            if self.states[wg as usize][handle as usize].is_none() {
                self.states[wg as usize][handle as usize] =
                    Some(RequestState::new(self.graphs[wg as usize].clone()));
            }
        }

        let info = self.requests[handle as usize].as_mut().expect("just set");
        for &wg in &live {
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
        if let Some(info) = self.requests[rid as usize].take() {
            for &wg in &info.worker_graphs {
                self.states[wg as usize][rid as usize] = None;
            }
        }
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
        let live = self.live_wgs(walk_sym);
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
            if let Some(state) = &mut self.states[wg as usize][rid as usize] {
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
                    let Some(state) = &self.states[wg as usize][rid as usize]
                    else {
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
    fn reset_outputs(
        &mut self, node_name: &str, rids: Vec<u32>, wg_ids: Vec<u32>,
    ) {
        let _ = (node_name, rids, wg_ids);
    }

    /// Release the inputs the just-executed node consumed, dereferencing them
    /// in the bookkeeper the store shares.
    fn cleanup_consumed_inputs(
        &mut self, node_name: &str, rids: Vec<u32>, wg_ids: Vec<u32>,
    ) -> PyResult<()> {
        if rids.len() != wg_ids.len() {
            return Err(PyValueError::new_err(
                "cleanup_consumed_inputs: rids and wg_ids must be the same length",
            ));
        }
        let mut freed: Vec<u64> = Vec::new();
        for (rid, wg_id) in rids.into_iter().zip(wg_ids) {
            let Some(wg) = self.wg_index(wg_id) else { continue };
            let Some(node) = self.nid(wg, node_name) else { continue };
            if let Some(state) = &mut self.states[wg as usize][rid as usize] {
                freed.extend(state.clear_consumed_inputs(node));
            }
        }
        if !freed.is_empty() {
            let mut bk = self.bookkeeping.lock().unwrap();
            for uuid in freed {
                bk.dereference(uuid, 1);
            }
        }
        Ok(())
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
                let ready = self.states[wg as usize]
                    .get(rid as usize)
                    .and_then(|s| s.as_ref())
                    .is_some_and(|s| s.is_ready(node));
                if !ready {
                    return None; // unknown rid counts as not ready
                }
            }
        }

        let mut out = PopRidsOut::default();
        for rid in request_ids {
            let Some(state) = self.states[wg as usize]
                .get_mut(rid as usize)
                .and_then(|s| s.as_mut())
            else {
                continue;
            };
            if !state.take_for_schedule(node) {
                continue;
            }
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
    ) -> Vec<(String, String, bool, Option<String>)> {
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
                self.async_checker
                    .can_speculate(g.node(source).name, g.node(sn.node).name)
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
    ) -> Option<(String, String, bool, Option<String>)> {
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
        let states = &mut self.states[wg as usize];
        for rid in rids {
            if let Some(state) = &mut states[rid as usize] {
                state.set_spec_scheduled(node_id, speculatively_scheduled);
            }
        }
        Ok(())
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
