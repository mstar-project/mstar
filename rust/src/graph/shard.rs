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
    pub shard_dim: FxHashMap<Sym, u32>,
    pub me: Sym,
}

/// A group bound to one request's workers.
pub struct Group {
    pub workers: Vec<Sym>,
    pub workers_set: FxHashSet<Sym>,
    pub tp_size: u32,
    pub tp_rank: u32,
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
            tp_rank: tp_rank.unwrap_or(0),
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
    pub worker: Sym,
    pub full: bool,
    pub start: i64,
    pub end: i64,
}

impl ShardMap {
    pub fn group_of(&self, node: Sym, walk: Option<Sym>) -> Option<&Group> {
        self.group_mapping
            .get(&(node, walk))
            .map(|&i| &self.groups[i as usize])
    }

    /// Destinations for one edge. `out` is reused across calls.
    pub fn fanout(
        &self,
        signal: Sym,
        src: Sym,
        src_walk: Sym,
        dst: Sym,
        dst_walk: Option<Sym>,
        t: &[TensorRef],
        out: &mut Vec<FanoutDest>,
    ) {
        out.clear();
        let sg = self.group_of(src, Some(src_walk));
        let dg = self.group_of(dst, dst_walk);
        let (src_worker, src_rank, src_tp) = match sg {
            Some(g) => (Some(g.workers[g.tp_rank as usize]), g.tp_rank, g.tp_size),
            None => (None, 0, 1),
        };
        let shard_dim = self.shard_dim.get(&signal).copied();

        let Some(dg) = dg else {
            // special destination (client / conductor): single logical dest
            out.push(FanoutDest { worker: Sym::MAX, full: true, start: 0, end: 0 });
            return;
        };

        if shard_dim.is_none() {
            // replicated: the source's own worker if it is in the dest group,
            // plus (rank 0 only) every dest worker not already holding it.
            if let Some(sw) = src_worker {
                if dg.workers.contains(&sw) {
                    out.push(FanoutDest { worker: sw, full: true, start: 0, end: 0 });
                }
            }
            if src_rank == 0 {
                for &w in &dg.workers {
                    if Some(w) != src_worker
                        && !sg.is_some_and(|g| g.workers_set.contains(&w))
                    {
                        out.push(FanoutDest { worker: w, full: true, start: 0, end: 0 });
                    }
                }
            }
            return;
        }

        let sizes: Vec<i64> = t.iter().map(|x| x.dim0).collect();
        let total = sizes.first().copied().unwrap_or(0) * src_tp as i64;
        let per_dest = total / dg.tp_size as i64;
        let s_start = src_rank as i64 * sizes.first().copied().unwrap_or(0);
        let s_end = s_start + sizes.first().copied().unwrap_or(0);
        for r in 0..dg.tp_size {
            let d_start = r as i64 * per_dest;
            let d_end = d_start + per_dest;
            if s_end <= d_start { break; }
            if s_start >= d_end { continue; }
            out.push(FanoutDest {
                worker: dg.workers[r as usize],
                full: false,
                start: s_start.max(d_start),
                end: s_end.min(d_end),
            });
        }
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
        ShardingTemplate { groups, shard_dim: FxHashMap::default(), me: 100 }
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
        assert_eq!(g.tp_rank, 1, "clone_empty carries the conductor's rank");
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
    fn two_requests_can_bind_the_same_node_to_different_workers() {
        // The reason this is per request at all: data-parallel replicas.
        let t = template(vec![]);
        let a = t.instantiate(&n2w(&[((NODE_A, WALK_X), &[100])])).unwrap();
        let b = t.instantiate(&n2w(&[((NODE_A, WALK_X), &[101])])).unwrap();
        assert_eq!(a.group_of(NODE_A, Some(WALK_X)).unwrap().workers, vec![100]);
        assert_eq!(b.group_of(NODE_A, Some(WALK_X)).unwrap().workers, vec![101]);
    }
}
