//! The batched surface. One `GraphRuntime` per worker owns every request's
//! walk state; Python calls it once per forward pass, not once per request.

use crate::graph::compile::{EMIT_TO_CLIENT, EMPTY_DESTINATION, LoopArg, NodeArg, compile_one};
use crate::graph::shard::{GroupTemplate, ShardMap, ShardingTemplate};
use crate::graph::spec::*;
use crate::graph::request::{RequestInfo, WgIndex, WorkerGraphMeta};
use crate::graph::state::RequestState;
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
    #[pyo3(item)] leader_nodes: Vec<String>,
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

        let mut async_enabled: FxHashSet<u32> = FxHashSet::default();

        // Compile graphs
        for (wg, wga) in worker_graphs.iter().enumerate() {
            let graph = compile_one(&mut it, &wga.nodes, &wga.loops, &wga.leader_nodes)?;
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

        // Intern worker names
        let worker_syms: Vec<Sym> = workers.iter().map(|w| it.intern(w)).collect();
        let me_sym = it.intern(&me);

        // Sharding TEMPLATE. The per-request ShardMap comes from
        // `instantiate`, because a data-parallel replica puts the same node on
        // different workers -- the binding is not deployment-wide.
        let tp_rank = worker_syms.iter().position(|&w| w == me_sym).unwrap_or(0) as u32;
        let shard = ShardingTemplate {
            groups: vec![GroupTemplate {
                nodes: graphs.iter().flat_map(|g| g.nodes.iter().map(|n| n.name)).collect(),
                tp_size: workers.len() as u32,
                graph_walks: None,
                tp_rank: Some(tp_rank),
            }],
            shard_dim: FxHashMap::default(),
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
        })
    }

    /// Worker graphs owned by OTHER workers, from the conductor's global maps.
    /// Needed only to route an output off this worker.
    fn set_remote_worker_graphs(
        &mut self,
        worker_graphs: Vec<RemoteWorkerGraphArg>,
    ) {
        let known: FxHashSet<u32> =
            self.all_worker_graphs.iter().map(|m| m.wg_id).collect();
        for wga in worker_graphs {
            if known.contains(&wga.wg_id) {
                continue; // ours; already compiled
            }
            self.all_worker_graphs.push(WorkerGraphMeta {
                wg_id: wga.wg_id,
                graph_walks: wga.graph_walks.iter().map(|w| self.interner.intern(w)).collect(),
                nodes: wga.nodes.iter().map(|n| self.interner.intern(n)).collect(),
                dyn_loops: wga.dyn_loops.iter().map(|l| self.interner.intern(l)).collect(),
                local: None,
            });
        }
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

    fn num_handles(&self) -> usize {
        self.rids.len()
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
