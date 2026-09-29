//! Port of `ShardingConfig`: the immutable template the conductor hands over,
//! and the per-request instance `clone_empty()` + `setup()` produce.
//!
//! Per request because a data-parallel replica puts the same node on different
//! workers, so the node -> workers binding is not deployment-wide.

use crate::graph::spec::Sym;
use crate::graph::state::TensorRef;
use rustc_hash::{FxHashMap, FxHashSet};

/// `(node, graph_walk)`. `None` walk is Python's streaming lookup: a streaming
/// consumer's walk is not known when the edge is routed.
pub type NodeWalk = (Sym, Option<Sym>);

/// One group as configured, before it knows which workers a request uses.
#[derive(Clone)]
pub struct GroupTemplate {
    pub nodes: Vec<Sym>,
    pub tp_size: u32,
    /// `None` means every graph walk.
    pub graph_walks: Option<Vec<Sym>>,
    /// This worker's rank in the group; the conductor sets it per worker and
    /// `clone_empty` carries it across.
    pub tp_rank: Option<u32>,
}

/// The deployment-wide config. Cloned per request by `instantiate`.
pub struct ShardingTemplate {
    pub groups: Vec<GroupTemplate>,
    /// signal -> shard dim. Python's value type is `int | None`, but an entry
    /// whose value is None reads exactly like an absent one (`.get(sig) is
    /// None` == replicated), so the Nones are dropped on the way in.
    pub shard_dim: FxHashMap<Sym, u32>,
    /// Gate config validation only -- they add no shard_dim entries and do not
    /// affect routing. Carried so this type is the whole ShardingConfig.
    pub tp_enabled_nodes: FxHashSet<Sym>,
    pub sp_enabled_nodes: FxHashSet<Sym>,
    pub me: Sym,
}

/// A group bound to one request's workers.
pub struct Group {
    pub workers: Vec<Sym>,
    pub workers_set: FxHashSet<Sym>,
    pub tp_size: u32,
    /// `None` until the conductor sets it. Defaulting to 0 would make every
    /// rank the broadcaster.
    pub tp_rank: Option<u32>,
}

impl Group {
    /// Python's `register_workers`. The length check is the group's invariant:
    /// a mismatch means the conductor's assignment disagrees with the config.
    fn bind(
        workers: Vec<Sym>,
        tp_size: u32,
        tp_rank: Option<u32>,
    ) -> Result<Self, ShardError> {
        if workers.len() as u32 != tp_size {
            return Err(ShardError::WorkerCount {
                got: workers.len(),
                tp_size,
            });
        }
        Ok(Group {
            workers_set: workers.iter().copied().collect(),
            workers,
            tp_size,
            tp_rank,
        })
    }
}

#[derive(Debug)]
pub enum ShardError {
    WorkerCount { got: usize, tp_size: u32 },
    /// Two groups claim the same node with `graph_walks=None`. Streaming
    /// consumers must resolve to exactly one group, or the `None` lookup is
    /// ambiguous.
    DuplicateStreamingGroup { node: Sym },
    /// A source group the conductor never ranked. Python asserts too.
    MissingTpRank { node: Sym, walk: Sym },
    /// Destination ranks must divide the shard dim evenly.
    NotDivisible { total: i64, dest_tp: u32 },
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::WorkerCount { got, tp_size } => {
                write!(f, "register_workers got {got} workers but tp_size={tp_size}")
            }
            ShardError::DuplicateStreamingGroup { node } => write!(
                f,
                "two groups both claim node {node} with graph_walks=None -- \
                 streaming consumers must have exactly one sharding group"
            ),
            ShardError::MissingTpRank { node, walk } => write!(
                f,
                "source group for node {node} / walk {walk} has no tp_rank; \
                 the conductor sets one per worker"
            ),
            ShardError::NotDivisible { total, dest_tp } => write!(
                f,
                "total shard dim size {total} not divisible by dest tp_size \
                 {dest_tp}"
            ),
        }
    }
}

/// One request's resolved sharding.
pub struct ShardMap {
    pub groups: Vec<Group>,
    /// `(node, walk)` -> index into `groups`, including the `(node, None)`
    /// entries used for streaming.
    pub group_mapping: FxHashMap<NodeWalk, u32>,
    pub shard_dim: FxHashMap<Sym, u32>,
    pub me: Sym,
}

impl ShardingTemplate {
    /// `clone_empty()` followed by `setup(node_to_workers)`, in one step.
    ///
    /// Configured groups claim their nodes first; whatever is left becomes a
    /// singleton group per (node, worker set), exactly as Python does.
    pub fn instantiate(
        &self,
        node_to_workers: &FxHashMap<(Sym, Sym), Vec<Sym>>,
    ) -> Result<ShardMap, ShardError> {
        let all_walks: FxHashSet<Sym> = node_to_workers.keys().map(|&(_, w)| w).collect();

        let mut groups: Vec<Group> = Vec::new();
        let mut mapping: FxHashMap<NodeWalk, u32> = FxHashMap::default();

        for template in &self.groups {
            // A group with graph_walks=None spans every walk, and also gets the
            // (node, None) entry that streaming looks up.
            let spans_all = template.graph_walks.is_none();
            let walks: Vec<Sym> = match &template.graph_walks {
                Some(ws) => ws.clone(),
                None => all_walks.iter().copied().collect(),
            };

            let mut bound: Option<u32> = None;
            for &node in &template.nodes {
                for &walk in &walks {
                    if let Some(workers) = node_to_workers.get(&(node, walk)) {
                        // Python checks every key the group claims, not just
                        // whichever bound it first.
                        if workers.len() as u32 != template.tp_size {
                            return Err(ShardError::WorkerCount {
                                got: workers.len(),
                                tp_size: template.tp_size,
                            });
                        }
                        let idx = match bound {
                            Some(i) => i,
                            None => {
                                groups.push(Group::bind(
                                    workers.clone(),
                                    template.tp_size,
                                    template.tp_rank,
                                )?);
                                let i = (groups.len() - 1) as u32;
                                bound = Some(i);
                                i
                            }
                        };
                        mapping.insert((node, Some(walk)), idx);
                    }
                    if spans_all {
                        let key = (node, None);
                        if let (Some(&existing), Some(idx)) = (mapping.get(&key), bound) {
                            if existing != idx {
                                return Err(ShardError::DuplicateStreamingGroup { node });
                            }
                        }
                        if let Some(idx) = bound {
                            mapping.insert(key, idx);
                        }
                    }
                }
            }
        }

        // Anything no configured group claimed becomes a singleton, grouped by
        // (node, worker set) so one group serves every walk that shares it.
        let mut by_node_workers: FxHashMap<(Sym, Vec<Sym>), Vec<Sym>> = FxHashMap::default();
        let mut walk_combos: FxHashMap<Sym, u32> = FxHashMap::default();
        for (&(node, walk), workers) in node_to_workers {
            if mapping.contains_key(&(node, Some(walk))) {
                continue;
            }
            let key = (node, workers.clone());
            match by_node_workers.get_mut(&key) {
                Some(walks) => walks.push(walk),
                None => {
                    *walk_combos.entry(node).or_insert(0) += 1;
                    by_node_workers.insert(key, vec![walk]);
                }
            }
        }

        for ((node, workers), walks) in by_node_workers {
            // Singletons are tp_size 1 at rank 0: an unconfigured node is not
            // tensor-parallel.
            groups.push(Group::bind(workers, 1, Some(0))?);
            let idx = (groups.len() - 1) as u32;
            for walk in walks {
                mapping.insert((node, Some(walk)), idx);
            }
            if walk_combos.get(&node).copied().unwrap_or(0) == 1 {
                // Unambiguous across walks, so streaming can resolve it too.
                mapping.insert((node, None), idx);
            }
        }

        Ok(ShardMap {
            groups,
            group_mapping: mapping,
            shard_dim: self.shard_dim.clone(),
            me: self.me,
        })
    }
}

pub struct FanoutDest {
    /// `Sym::MAX` for a special destination (the client / the conductor),
    /// which has no sharding group and therefore no worker of its own.
    pub worker: Sym,
    /// The tensors as THIS destination should see them: the source refs
    /// unchanged for a replicated signal, sliced on the shard dim otherwise.
    ///
    /// Sliced here rather than reported as (start, end) for the caller to
    /// apply: the slice is expressible entirely in `TensorRef`'s own fields,
    /// so nothing downstream has to know the shard dim existed.
    pub tensors: Vec<TensorRef>,
}

/// Python's per-destination slice (`ShardingConfig.fanout_graph_edges`):
/// keep rows `start..end` of the leading (canonical) shard dim.
fn slice_rows(t: &TensorRef, start: i64, end: i64) -> TensorRef {
    // nbytes per row of the shard dim; a zero-length dim has no rows to take.
    let row = if t.dim0 == 0 { 0 } else { t.nbytes / t.dim0 };
    TensorRef {
        uuid: t.uuid,
        dim0: end - start,
        nbytes: (end - start) * row,
        // Assigned, not accumulated -- Python sets `new_info.offset = offset`.
        offset: start * row,
    }
}

impl ShardMap {
    pub fn group_of(&self, node: Sym, walk: Option<Sym>) -> Option<&Group> {
        self.group_mapping
            .get(&(node, walk))
            .map(|&i| &self.groups[i as usize])
    }

    /// Destinations for one edge. `out` is reused across calls.
    ///
    /// Errors where Python asserts: an unranked source group, an uneven split.
    pub fn fanout(
        &self,
        signal: Sym,
        src: Sym,
        src_walk: Sym,
        dst: Sym,
        dst_walk: Option<Sym>,
        t: &[TensorRef],
        out: &mut Vec<FanoutDest>,
    ) -> Result<(), ShardError> {
        out.clear();
        let sg = self.group_of(src, Some(src_walk));
        let dg = self.group_of(dst, dst_walk);
        let (src_worker, src_rank, src_tp) = match sg {
            Some(g) => {
                // Not unwrap_or(0): rank 0 is the broadcaster.
                let rank = g.tp_rank.ok_or(ShardError::MissingTpRank {
                    node: src,
                    walk: src_walk,
                })?;
                (Some(g.workers[rank as usize]), rank, g.tp_size)
            }
            None => (None, 0, 1),
        };
        let shard_dim = self.shard_dim.get(&signal).copied();

        // A special destination (client / conductor) has no group. Python does
        // NOT short-circuit here: it gives the dest a single pseudo-worker with
        // tp_size 1 and runs the same branches. That matters both ways -- a
        // replicated signal then leaves from rank 0 ONLY (every rank emitting
        // is how the api server ends up seeing one result twice), and a sharded
        // one is still sliced per source rank rather than sent whole.
        let (dest_workers, dest_tp): (Vec<Sym>, u32) = match dg {
            Some(g) => (g.workers.clone(), g.tp_size),
            None => (vec![Sym::MAX], 1),
        };

        if shard_dim.is_none() {
            // Replicated: the source's own worker if it is in the dest group,
            // plus (rank 0 only) every dest worker not already holding it.
            if let Some(sw) = src_worker {
                if dest_workers.contains(&sw) {
                    out.push(FanoutDest { worker: sw, tensors: t.to_vec() });
                }
            }
            if src_rank == 0 {
                for &w in &dest_workers {
                    if Some(w) != src_worker
                        && !sg.is_some_and(|g| g.workers_set.contains(&w))
                    {
                        out.push(FanoutDest { worker: w, tensors: t.to_vec() });
                    }
                }
            }
            return Ok(());
        }

        // Every tensor, before routing any: truncating would silently drop
        // the remainder rows.
        for x in t {
            let total = x.dim0 * src_tp as i64;
            if total % dest_tp as i64 != 0 {
                return Err(ShardError::NotDivisible { total, dest_tp });
            }
        }

        // WHICH destinations overlap: the first tensor decides, as in Python.
        let first = t.first().map_or(0, |x| x.dim0);
        let per_dest0 = (first * src_tp as i64) / dest_tp as i64;
        let s0_start = src_rank as i64 * first;
        let s0_end = s0_start + first;
        for r in 0..dest_tp {
            let d0_start = r as i64 * per_dest0;
            let d0_end = d0_start + per_dest0;
            if s0_end <= d0_start { break; }
            if s0_start >= d0_end { continue; }
            out.push(FanoutDest {
                worker: dest_workers[r as usize],
                // WHERE each is cut is per tensor: leading dims may differ,
                // and the first one's rows are the wrong bytes for the rest.
                tensors: t
                    .iter()
                    .map(|x| {
                        let per_dest = (x.dim0 * src_tp as i64) / dest_tp as i64;
                        let s_start = src_rank as i64 * x.dim0;
                        let s_end = s_start + x.dim0;
                        let d_start = r as i64 * per_dest;
                        let d_end = d_start + per_dest;
                        // The two spans' overlap, on this tensor's own dim.
                        let (lo, hi) = (s_start.max(d_start), s_end.min(d_end));
                        // Relative to this source's own slab.
                        slice_rows(x, lo - s_start, hi - s_start)
                    })
                    .collect(),
            });
        }
        Ok(())
    }

    /// Python's `compute_fanin`: how many SOURCE ranks contribute to the copy
    /// this destination rank ends up holding.
    ///
    /// Rides the wire as `_total_fanin`, which is what tells the receiver to
    /// buffer the arrival until the other contributors land. Left at 1, a
    /// gather (source tp > dest tp) takes whichever half arrives first and
    /// drops the rest, silently.
    ///
    /// The receiver could derive this -- every descriptor carries
    /// `source_tp_size` -- but Python computes it here, and Python is the
    /// oracle.
    pub fn fanin(
        &self,
        signal: Sym,
        src: Sym,
        src_walk: Sym,
        dst: Sym,
        dst_walk: Option<Sym>,
        dst_worker: Sym,
    ) -> u32 {
        // Replicated: whoever sends it sends the whole thing.
        if !self.shard_dim.contains_key(&signal) {
            return 1;
        }
        let src_tp = self.group_of(src, Some(src_walk)).map_or(1, |g| g.tp_size);
        let dg = self.group_of(dst, dst_walk);
        // A special destination (the client) has no group: tp 1 at rank 0.
        let dest_tp = dg.map_or(1, |g| g.tp_size);
        let dest_rank = dg
            .and_then(|g| g.workers.iter().position(|&w| w == dst_worker))
            .unwrap_or(0) as u32;
        // Scaled integer coords, so the total span is src_tp * dest_tp and
        // both sides divide it evenly.
        let lo = dest_rank * src_tp;
        let hi = lo + src_tp;
        (0..src_tp)
            .filter(|r| r * dest_tp < hi && (r + 1) * dest_tp > lo)
            .count() as u32
    }

    /// The signal's shard dim, if it has one. `_shard_dim` on the wire.
    pub fn shard_dim_of(&self, signal: Sym) -> Option<u32> {
        self.shard_dim.get(&signal).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Syms: nodes 0..=2, walks 10/11, workers 100..=103.
    const NODE_A: Sym = 0;
    const NODE_B: Sym = 1;
    const WALK_X: Sym = 10;
    const WALK_Y: Sym = 11;

    fn n2w(pairs: &[((Sym, Sym), &[Sym])]) -> FxHashMap<(Sym, Sym), Vec<Sym>> {
        pairs.iter().map(|(k, v)| (*k, v.to_vec())).collect()
    }

    fn template(groups: Vec<GroupTemplate>) -> ShardingTemplate {
        ShardingTemplate {
            groups,
            shard_dim: FxHashMap::default(),
            tp_enabled_nodes: FxHashSet::default(),
            sp_enabled_nodes: FxHashSet::default(),
            me: 100,
        }
    }

    #[test]
    fn a_configured_group_claims_its_nodes() {
        let t = template(vec![GroupTemplate {
            nodes: vec![NODE_A],
            tp_size: 2,
            graph_walks: Some(vec![WALK_X]),
            tp_rank: Some(1),
        }]);
        let m = t
            .instantiate(&n2w(&[((NODE_A, WALK_X), &[100, 101])]))
            .unwrap();

        let g = m.group_of(NODE_A, Some(WALK_X)).unwrap();
        assert_eq!(g.workers, vec![100, 101]);
        assert_eq!(g.tp_size, 2);
        assert_eq!(g.tp_rank, Some(1), "clone_empty carries the conductor's rank");
        // graph_walks was explicit, so no streaming entry.
        assert!(m.group_of(NODE_A, None).is_none());
    }

    #[test]
    fn a_group_spanning_all_walks_also_answers_the_streaming_lookup() {
        let t = template(vec![GroupTemplate {
            nodes: vec![NODE_A],
            tp_size: 1,
            graph_walks: None,
            tp_rank: Some(0),
        }]);
        let m = t.instantiate(&n2w(&[((NODE_A, WALK_X), &[100])])).unwrap();
        assert!(m.group_of(NODE_A, Some(WALK_X)).is_some());
        assert!(
            m.group_of(NODE_A, None).is_some(),
            "a streaming consumer's walk is unknown when the edge is routed"
        );
    }

    #[test]
    fn an_unconfigured_node_gets_a_singleton_group() {
        let t = template(vec![]);
        let m = t.instantiate(&n2w(&[((NODE_B, WALK_X), &[102])])).unwrap();
        let g = m.group_of(NODE_B, Some(WALK_X)).unwrap();
        assert_eq!(g.workers, vec![102]);
        assert_eq!(g.tp_size, 1, "an unconfigured node is not tensor-parallel");
        // One walk combination, so the streaming lookup is unambiguous.
        assert!(m.group_of(NODE_B, None).is_some());
    }

    #[test]
    fn walks_sharing_a_worker_set_share_one_singleton() {
        let t = template(vec![]);
        let m = t
            .instantiate(&n2w(&[
                ((NODE_B, WALK_X), &[102]),
                ((NODE_B, WALK_Y), &[102]),
            ]))
            .unwrap();
        let x = m.group_mapping[&(NODE_B, Some(WALK_X))];
        let y = m.group_mapping[&(NODE_B, Some(WALK_Y))];
        assert_eq!(x, y, "same workers => one group, not one per walk");
        assert!(m.group_of(NODE_B, None).is_some());
    }

    #[test]
    fn differing_worker_sets_make_the_streaming_lookup_ambiguous() {
        // Two walk combinations for one node, so there is no single group the
        // (node, None) key could name -- Python leaves it out.
        let t = template(vec![]);
        let m = t
            .instantiate(&n2w(&[
                ((NODE_B, WALK_X), &[102]),
                ((NODE_B, WALK_Y), &[103]),
            ]))
            .unwrap();
        assert_ne!(
            m.group_mapping[&(NODE_B, Some(WALK_X))],
            m.group_mapping[&(NODE_B, Some(WALK_Y))],
        );
        assert!(m.group_of(NODE_B, None).is_none());
    }

    #[test]
    fn a_worker_count_that_disagrees_with_tp_size_is_rejected() {
        // The conductor's assignment contradicting the config is a deployment
        // error, not something to route around.
        let t = template(vec![GroupTemplate {
            nodes: vec![NODE_A],
            tp_size: 4,
            graph_walks: Some(vec![WALK_X]),
            tp_rank: Some(0),
        }]);
        let result = t.instantiate(&n2w(&[((NODE_A, WALK_X), &[100, 101])]));
        assert!(matches!(
            result.err(),
            Some(ShardError::WorkerCount { got: 2, tp_size: 4 })
        ));
    }

    #[test]
    fn shard_dim_reaches_the_instance_and_selects_the_sharded_fanout() {
        // A dropped shard_dim does not fail: it silently takes the replicated
        // branch, sending whole tensors where slices were meant.
        const SIGNAL: Sym = 50;
        let mut t = template(vec![GroupTemplate {
            nodes: vec![NODE_A, NODE_B],
            tp_size: 2,
            graph_walks: Some(vec![WALK_X]),
            tp_rank: Some(0),
        }]);
        t.shard_dim.insert(SIGNAL, 0);

        let m = t
            .instantiate(&n2w(&[
                ((NODE_A, WALK_X), &[100, 101]),
                ((NODE_B, WALK_X), &[100, 101]),
            ]))
            .unwrap();
        assert_eq!(m.shard_dim.get(&SIGNAL), Some(&0));

        // 4 rows, 8 bytes each. Equal tp sizes, so rank 0's slab maps whole
        // onto dest rank 0 -- the slice is the identity, and the point here is
        // that it goes to ONE destination rather than both.
        let tensors = [TensorRef { uuid: 1, dim0: 4, nbytes: 32, offset: 0 }];
        let mut out = Vec::new();
        m.fanout(SIGNAL, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out)
            .unwrap();
        assert_eq!(out.len(), 1, "rank 0's rows belong to dest rank 0 alone");
        assert_eq!(out[0].worker, 100);
        assert_eq!(out[0].tensors, tensors.to_vec());

        // The same edge without a shard dim is replicated: untouched refs.
        let mut out2 = Vec::new();
        m.fanout(999, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out2)
            .unwrap();
        assert!(out2.iter().all(|d| d.tensors == tensors.to_vec()));
    }

    #[test]
    fn an_unsharded_source_is_cut_across_the_destination_ranks() {
        const SIGNAL: Sym = 50;
        // src tp 1 -> dest tp 2: the one slab really is split, which is what
        // exercises the dims/nbytes/offset the slice carries.
        let mut t = template(vec![
            GroupTemplate {
                nodes: vec![NODE_A], tp_size: 1,
                graph_walks: Some(vec![WALK_X]), tp_rank: Some(0),
            },
            GroupTemplate {
                nodes: vec![NODE_B], tp_size: 2,
                graph_walks: Some(vec![WALK_X]), tp_rank: Some(0),
            },
        ]);
        t.shard_dim.insert(SIGNAL, 0);
        let m = t
            .instantiate(&n2w(&[
                ((NODE_A, WALK_X), &[100]),
                ((NODE_B, WALK_X), &[100, 101]),
            ]))
            .unwrap();

        // 4 rows of 8 bytes.
        let tensors = [TensorRef { uuid: 1, dim0: 4, nbytes: 32, offset: 0 }];
        let mut out = Vec::new();
        m.fanout(SIGNAL, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out)
            .unwrap();
        assert_eq!(out.len(), 2);
        // Halves: two rows each, the second starting two rows in.
        assert_eq!(
            out[0].tensors,
            vec![TensorRef { uuid: 1, dim0: 2, nbytes: 16, offset: 0 }],
        );
        assert_eq!(
            out[1].tensors,
            vec![TensorRef { uuid: 1, dim0: 2, nbytes: 16, offset: 16 }],
        );
    }

    /// NODE_A at `src_tp`, NODE_B at `dst_tp`, one sharded signal.
    fn sharded(src_tp: u32, dst_tp: u32, rank: Option<u32>) -> ShardMap {
        const SIGNAL: Sym = 50;
        let mut t = template(vec![
            GroupTemplate {
                nodes: vec![NODE_A], tp_size: src_tp,
                graph_walks: Some(vec![WALK_X]), tp_rank: rank,
            },
            GroupTemplate {
                nodes: vec![NODE_B], tp_size: dst_tp,
                graph_walks: Some(vec![WALK_X]), tp_rank: Some(0),
            },
        ]);
        t.shard_dim.insert(SIGNAL, 0);
        let src: Vec<Sym> = (0..src_tp).map(|i| 100 + i).collect();
        let dst: Vec<Sym> = (0..dst_tp).map(|i| 200 + i).collect();
        t.instantiate(&n2w(&[
            ((NODE_A, WALK_X), &src),
            ((NODE_B, WALK_X), &dst),
        ]))
        .unwrap()
    }

    #[test]
    fn tensors_of_different_leading_extent_are_each_cut_on_their_own_dim() {
        // Cutting the 8-row tensor on the 4-row one's rows would send half
        // of it and call it whole.
        const SIGNAL: Sym = 50;
        let m = sharded(1, 2, Some(0));
        let tensors = [
            TensorRef { uuid: 1, dim0: 4, nbytes: 16, offset: 0 },
            TensorRef { uuid: 2, dim0: 8, nbytes: 64, offset: 0 },
        ];
        let mut out = Vec::new();
        m.fanout(SIGNAL, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out)
            .unwrap();

        assert_eq!(out.len(), 2);
        // Halves of each: 2 rows of the first, 4 of the second.
        assert_eq!(out[0].tensors, vec![
            TensorRef { uuid: 1, dim0: 2, nbytes: 8, offset: 0 },
            TensorRef { uuid: 2, dim0: 4, nbytes: 32, offset: 0 },
        ]);
        assert_eq!(out[1].tensors, vec![
            TensorRef { uuid: 1, dim0: 2, nbytes: 8, offset: 8 },
            TensorRef { uuid: 2, dim0: 4, nbytes: 32, offset: 32 },
        ]);
    }

    #[test]
    fn a_shard_dim_the_destination_ranks_do_not_divide_is_rejected() {
        // 3 rows across 2 ranks: truncating would drop the third silently.
        const SIGNAL: Sym = 50;
        let m = sharded(1, 2, Some(0));
        let tensors = [TensorRef { uuid: 1, dim0: 3, nbytes: 12, offset: 0 }];
        let mut out = Vec::new();
        let r =
            m.fanout(SIGNAL, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out);
        assert!(matches!(
            r.err(),
            Some(ShardError::NotDivisible { total: 3, dest_tp: 2 })
        ));
    }

    #[test]
    fn every_tensor_is_checked_for_divisibility_not_just_the_first() {
        const SIGNAL: Sym = 50;
        let m = sharded(1, 2, Some(0));
        let tensors = [
            TensorRef { uuid: 1, dim0: 4, nbytes: 16, offset: 0 },
            TensorRef { uuid: 2, dim0: 5, nbytes: 20, offset: 0 },
        ];
        let mut out = Vec::new();
        let r =
            m.fanout(SIGNAL, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out);
        assert!(matches!(
            r.err(),
            Some(ShardError::NotDivisible { total: 5, dest_tp: 2 })
        ));
    }

    #[test]
    fn a_source_group_with_no_tp_rank_is_rejected_rather_than_read_as_rank_0() {
        const SIGNAL: Sym = 50;
        let m = sharded(2, 2, None);
        assert_eq!(m.group_of(NODE_A, Some(WALK_X)).unwrap().tp_rank, None);
        let tensors = [TensorRef { uuid: 1, dim0: 4, nbytes: 16, offset: 0 }];
        let mut out = Vec::new();
        let r =
            m.fanout(SIGNAL, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out);
        assert!(matches!(r.err(), Some(ShardError::MissingTpRank { .. })));
    }

    #[test]
    fn a_missing_tp_rank_is_caught_on_the_replicated_path_too() {
        // Replicated routing reads the rank too, so the check precedes the
        // branch; where Python's assert is.
        let m = sharded(2, 2, None);
        let tensors = [TensorRef { uuid: 1, dim0: 4, nbytes: 16, offset: 0 }];
        let mut out = Vec::new();
        let r =
            m.fanout(999, NODE_A, WALK_X, NODE_B, Some(WALK_X), &tensors, &mut out);
        assert!(matches!(r.err(), Some(ShardError::MissingTpRank { .. })));
    }

    #[test]
    fn a_worker_count_mismatch_is_caught_on_a_later_node_too() {
        // The group binds on NODE_A, then claims a narrower NODE_B. Checking
        // only the binding key would route NODE_B on NODE_A's workers.
        let t = template(vec![GroupTemplate {
            nodes: vec![NODE_A, NODE_B],
            tp_size: 2,
            graph_walks: Some(vec![WALK_X]),
            tp_rank: Some(0),
        }]);
        let r = t.instantiate(&n2w(&[
            ((NODE_A, WALK_X), &[100, 101]),
            ((NODE_B, WALK_X), &[102]),
        ]));
        assert!(matches!(
            r.err(),
            Some(ShardError::WorkerCount { got: 1, tp_size: 2 })
        ));
    }

    #[test]
    fn two_requests_can_bind_the_same_node_to_different_workers() {
        // The reason this is per request at all: data-parallel replicas.
        let t = template(vec![]);
        let a = t.instantiate(&n2w(&[((NODE_A, WALK_X), &[100])])).unwrap();
        let b = t.instantiate(&n2w(&[((NODE_A, WALK_X), &[101])])).unwrap();
        assert_eq!(a.group_of(NODE_A, Some(WALK_X)).unwrap().workers, vec![100]);
        assert_eq!(b.group_of(NODE_A, Some(WALK_X)).unwrap().workers, vec![101]);
    }
}
