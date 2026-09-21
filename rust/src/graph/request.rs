//! Per-request bookkeeping that is not walk state: which partitions a request
//! is running, and which workers own which nodes for it.
//!
//! Deliberately NOT stored inside the worker graph queues. A partition's walk
//! and its stream-done flag are per (request, partition); a partition can span
//! several worker graphs, so putting them there would duplicate the value and
//! turn `set_walk` into a scatter-write. The worker graph -> partition
//! direction is static, so it lives on `WorkerGraphMeta` and costs nothing per
//! request.

use crate::graph::shard::ShardMap;
use crate::graph::spec::Sym;
use rustc_hash::FxHashMap;

pub type PartitionId = Sym;
pub type WgIndex = u32; // index into GraphRuntime.graphs, not the wire wg id

/// Static facts about one worker graph, local or remote. The remote ones are
/// needed to answer "who owns this node" when routing an output off-worker.
pub struct WorkerGraphMeta {
    pub wg_id: u32,
    pub graph_walks: Vec<Sym>,
    pub nodes: Vec<Sym>,
    pub dyn_loops: Vec<Sym>,
    /// None for a worker graph owned by another worker.
    pub local: Option<WgIndex>,
}

/// Per (request, partition).
pub struct RequestPartitionInfo {
    pub graph_walk: Sym,
    /// Local worker graphs live in the CURRENT walk. Derived from the walk, so
    /// it is a cache: `set_walk` recomputes it, and nothing else may write it.
    pub walk_worker_graphs: Vec<WgIndex>,
    /// Set when the consuming pass sees the final streaming chunk.
    pub stream_partition_done: bool,
}

/// Python's `NestedLoopIndices`, as the runtime stores it.
#[derive(Clone, Default)]
pub struct LoopStopTime {
    pub loop_name_order: Vec<Sym>,
    pub loop_indices: FxHashMap<Sym, u32>,
    pub wg_fwd_pass_idx: u32,
}

impl LoopStopTime {
    /// Python's `label_context_gt`: is `self` strictly later than `other` in
    /// the loops ENCLOSING `target`? A tie on the enclosing context is not
    /// newer, so a duplicate stop does not re-stop the loop.
    pub fn later_than(&self, other: Option<&LoopStopTime>, target: Sym) -> bool {
        let Some(other) = other else { return true };
        if self.wg_fwd_pass_idx != other.wg_fwd_pass_idx {
            return self.wg_fwd_pass_idx > other.wg_fwd_pass_idx;
        }
        for &name in &self.loop_name_order {
            if name == target {
                break;
            }
            let ours = self.loop_indices.get(&name).copied().unwrap_or(0);
            let theirs = other.loop_indices.get(&name).copied().unwrap_or(0);
            if ours != theirs {
                return ours > theirs;
            }
        }
        false
    }
}

/// Everything the runtime keeps for one request besides its walk state.
#[derive(Default)]
pub struct RequestInfo {
    pub partitions: FxHashMap<PartitionId, RequestPartitionInfo>,
    /// Local worker graphs across every partition of this request.
    pub worker_graphs: Vec<WgIndex>,
    /// (node, walk) -> the workers running it for THIS request. Per request
    /// because data-parallel replicas put the same node on different workers.
    pub node_to_workers: FxHashMap<(Sym, Sym), Vec<Sym>>,
    pub dyn_loop_to_workers: FxHashMap<(Sym, Sym), Vec<Sym>>,
    /// Loop name -> the stop observation this rank has. Worker-only, so it
    /// does not ride CurrentForwardPassInfo across the wire.
    pub loop_stop_times: FxHashMap<Sym, LoopStopTime>,
    /// This request's resolved sharding, from ShardingTemplate::instantiate.
    /// None until add_request has seen the worker assignment.
    pub shard: Option<ShardMap>,

    /// Buffered between send_outputs calls and flushed onto the next
    /// WORKER_GRAPHS_DONE, so a persist signal cannot race the message that
    /// announces it.
    ///
    /// On RequestInfo rather than in a side map keyed by handle: handles are
    /// recycled, and a side map outliving its request hands one request's
    /// persist signals and token counts to whichever request draws that
    /// integer next. Here, remove_request frees them with everything else.
    pub pending: PendingOutputs,
}

/// What a request has accumulated since its last WORKER_GRAPHS_DONE.
#[derive(Default)]
pub struct PendingOutputs {
    /// signal -> uuids. Uuids, not descriptors: the bookkeeper is shared, so
    /// resolving at send time cannot go stale against it.
    pub persist: Vec<(Sym, Vec<u64>)>,
    pub new_tokens: Vec<(String, i64)>,
    /// Emitted signal names, in emission order.
    pub output_signals: Vec<Sym>,
    /// signal -> the loop context it was emitted at.
    pub output_loop_indices: Vec<(Sym, (Vec<Sym>, Vec<(Sym, u32)>, u32))>,
}

impl PendingOutputs {
    /// Everything but `output_loop_indices`, which the conductor keeps for the
    /// life of the request -- Python re-sends it on every WGD.
    pub fn take_for_send(&mut self) -> (Vec<(Sym, Vec<u64>)>, Vec<(String, i64)>, Vec<Sym>) {
        (
            std::mem::take(&mut self.persist),
            std::mem::take(&mut self.new_tokens),
            std::mem::take(&mut self.output_signals),
        )
    }

    /// Accumulate, matching Python: a repeated signal's count adds up, and a
    /// repeated loop index overwrites.
    pub fn add_new_tokens(&mut self, counts: &[(String, i64)]) {
        for (name, n) in counts {
            match self.new_tokens.iter_mut().find(|(k, _)| k == name) {
                Some((_, v)) => *v += n,
                None => self.new_tokens.push((name.clone(), *n)),
            }
        }
    }

    pub fn set_loop_indices(
        &mut self, signal: Sym, idx: (Vec<Sym>, Vec<(Sym, u32)>, u32),
    ) {
        match self.output_loop_indices.iter_mut().find(|(s, _)| *s == signal) {
            Some(slot) => slot.1 = idx,
            None => self.output_loop_indices.push((signal, idx)),
        }
    }
}

impl RequestInfo {
    pub fn walk(&self, partition: PartitionId) -> Option<Sym> {
        self.partitions.get(&partition).map(|p| p.graph_walk)
    }

    /// Record a partition, or replace an existing entry.
    ///
    /// `live` is the worker graphs active in `walk`; the caller derives it
    /// because only it holds the walk -> worker graph index.
    pub fn set_partition(&mut self, partition: PartitionId, walk: Sym, live: Vec<WgIndex>) {
        self.partitions.insert(
            partition,
            RequestPartitionInfo {
                graph_walk: walk,
                walk_worker_graphs: live,
                stream_partition_done: false,
            },
        );
    }

    /// True when the walk actually changed, so the caller can skip whatever
    /// only matters on a transition.
    pub fn set_walk(
        &mut self,
        partition: PartitionId,
        walk: Sym,
        live: impl FnOnce() -> Vec<WgIndex>,
    ) -> bool {
        let Some(info) = self.partitions.get_mut(&partition) else {
            return false;
        };
        if info.graph_walk == walk {
            return false;
        }
        info.graph_walk = walk;
        // The walk selects which worker graphs are live, so a stale list would
        // route this pass's inputs into the previous walk's graphs.
        info.walk_worker_graphs = live();
        true
    }

    pub fn mark_stream_done(&mut self, partition: PartitionId) {
        if let Some(info) = self.partitions.get_mut(&partition) {
            info.stream_partition_done = true;
        }
    }

    pub fn stream_done(&self, partition: PartitionId) -> bool {
        self.partitions
            .get(&partition)
            .is_some_and(|p| p.stream_partition_done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stop(fwd: u32, outer: u32) -> LoopStopTime {
        let mut idx = FxHashMap::default();
        idx.insert(0, outer); // sym 0 == "outer"
        LoopStopTime {
            loop_name_order: vec![0, 1], // outer, then the target
            loop_indices: idx,
            wg_fwd_pass_idx: fwd,
        }
    }

    #[test]
    fn no_previous_observation_is_always_newer() {
        assert!(stop(0, 0).later_than(None, 1));
    }

    #[test]
    fn a_later_forward_pass_wins_outright() {
        assert!(stop(2, 0).later_than(Some(&stop(1, 9)), 1));
        assert!(!stop(1, 9).later_than(Some(&stop(2, 0)), 1));
    }

    #[test]
    fn within_a_pass_the_enclosing_loop_index_decides() {
        // Same forward pass, so the comparison has to fall through to the
        // loops ENCLOSING the target rather than short-circuiting above.
        assert!(stop(3, 2).later_than(Some(&stop(3, 1)), 1));
        assert!(!stop(3, 1).later_than(Some(&stop(3, 2)), 1));
    }

    #[test]
    fn an_identical_observation_is_not_newer() {
        // Or a duplicate stop would re-stop a loop that has since restarted.
        assert!(!stop(3, 2).later_than(Some(&stop(3, 2)), 1));
    }

    #[test]
    fn indices_at_or_inside_the_target_are_ignored() {
        // The target's OWN iteration advancing is not a newer stop context:
        // the scan stops when it reaches the target in loop_name_order.
        let mut a = stop(3, 1);
        let mut b = stop(3, 1);
        a.loop_indices.insert(1, 9); // the target itself
        b.loop_indices.insert(1, 0);
        assert!(!a.later_than(Some(&b), 1));
    }

    #[test]
    fn set_walk_reports_only_real_transitions_and_refreshes_the_cache() {
        let mut info = RequestInfo::default();
        info.set_partition(7, 100, vec![0]);

        // Same walk: no transition, and the cache is left alone.
        assert!(!info.set_walk(7, 100, || vec![9, 9]));
        assert_eq!(info.partitions[&7].walk_worker_graphs, vec![0]);

        // A real transition re-derives the live worker graphs. A stale list
        // here would route the pass into the previous walk's graphs.
        assert!(info.set_walk(7, 200, || vec![1, 2]));
        assert_eq!(info.partitions[&7].walk_worker_graphs, vec![1, 2]);
        assert_eq!(info.walk(7), Some(200));
    }

    #[test]
    fn set_walk_on_an_unknown_partition_is_a_no_op() {
        let mut info = RequestInfo::default();
        assert!(!info.set_walk(7, 100, || unreachable!("must not derive")));
    }

    #[test]
    fn stream_done_is_per_partition() {
        let mut info = RequestInfo::default();
        info.set_partition(1, 10, vec![]);
        info.set_partition(2, 20, vec![]);
        info.mark_stream_done(1);
        assert!(info.stream_done(1));
        assert!(!info.stream_done(2));
    }
}
