//! Port of `ShardingConfig.compute_fanout` / `fanout_graph_edges`.
//! Replicated + equal-TP is the overwhelmingly common case; it is a branch,
//! not a dict walk.

use crate::spec::Sym;
use crate::state::TensorRef;

pub struct Group {
    pub workers: Vec<Sym>,
    pub tp_size: u32,
    pub tp_rank: u32,
}

pub struct ShardMap {
    /// (node sym) -> group index, for the current walk.
    pub node_group: rustc_hash::FxHashMap<Sym, u32>,
    pub groups: Vec<Group>,
    pub shard_dim: rustc_hash::FxHashMap<Sym, u32>,
    pub me: Sym,
}

pub struct FanoutDest {
    pub worker: Sym,
    pub full: bool,
    pub start: i64,
    pub end: i64,
}

impl ShardMap {
    pub fn group_of(&self, node: Sym) -> Option<&Group> {
        self.node_group.get(&node).map(|&i| &self.groups[i as usize])
    }

    /// Returns destinations for one edge. `out` is reused across calls.
    pub fn fanout(&self, signal: Sym, src: Sym, dst: Sym, t: &[TensorRef], out: &mut Vec<FanoutDest>) {
        out.clear();
        let sg = self.group_of(src);
        let dg = self.group_of(dst);
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
                    if Some(w) != src_worker && !sg.map_or(false, |g| g.workers.contains(&w)) {
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
