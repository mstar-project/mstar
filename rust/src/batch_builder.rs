//! Batch composition: which of one scheduling key's popped rows run this
//! step, which stay parked in the backlog and which go back to their ready
//! queues. The Rust backends of `mstar/worker/batch_builder.py`'s builders,
//! which the scheduler core (`graph/runtime/sched.rs`) picks by name when it
//! is configured. A builder sees rows and caps only: popping, parking and
//! in-flight flags are the core's.

use crate::graph::spec::Sym;
use rustc_hash::FxHashSet;

/// One scheduling key's caps, asked of the engine once by the Python side.
pub struct KeyCaps {
    /// The real walks under a combined key; empty for a plain one.
    pub walks: Vec<Sym>,
    /// walk -> cap (None: uncapped), in `SchedKeys.walk_caps` order: each
    /// real walk, then the key's own.
    pub caps: Vec<(Sym, Option<u32>)>,
    /// `SchedKeys.walk_row_caps`: per real walk, rows a mixed step may hold.
    pub row_caps: Vec<(Sym, u32)>,
}

impl KeyCaps {
    /// `caps.get(walk)`: None for an uncapped walk and for one not listed.
    pub fn cap(&self, walk: Sym) -> Option<u32> {
        self.caps.iter().find(|c| c.0 == walk).and_then(|c| c.1)
    }

    pub fn row_cap(&self, walk: Sym) -> Option<u32> {
        self.row_caps.iter().find(|c| c.0 == walk).map(|c| c.1)
    }
}

pub fn min_cap(caps: impl IntoIterator<Item = Option<u32>>) -> Option<u32> {
    caps.into_iter().flatten().min()
}

#[derive(Clone, Copy)]
pub struct Row {
    pub rid: u32,
    pub walk: Sym,
}

/// `BatchBuildRequest`, by rows.
pub struct BuildRequest<'a> {
    /// the key's walk label
    pub label: Sym,
    /// the backlogged rows, then the fresh ones
    pub rows: &'a [Row],
    pub n_backlog: usize,
    /// whether rows carry their own real walks (a combined key's batch)
    pub combined: bool,
    /// rows that cannot run this step; they stay parked
    pub blocked: &'a FxHashSet<u32>,
    pub caps: &'a KeyCaps,
    /// the caller's own rows and their walk
    pub pre: u32,
    pub pre_walk: Option<Sym>,
}

/// Every row of the request, by index, in row order within each part.
#[derive(Default)]
pub struct BuildResult {
    pub scheduled: Vec<usize>,
    pub parked: Vec<usize>,
    pub returned: Vec<usize>,
}

pub trait BatchBuilder: Send + Sync {
    fn build(&self, req: &BuildRequest) -> BuildResult;

    /// `BaseBatchBuilder.room_for_continuing`: how many of a speculation
    /// chain's own rows fit in a step capped at `cap` beside a backlog of
    /// `backlog_rows`.
    fn room_for_continuing(&self, cap: Option<u32>, backlog_rows: usize) -> Option<u32> {
        cap.map(|c| c.saturating_sub(backlog_rows as u32))
    }
}

/// The Rust backend of the builder named `name` (`BatchBuilderBackends.rust`).
pub fn by_name(name: &str) -> Option<Box<dyn BatchBuilder>> {
    match name {
        "fifo" => Some(Box::new(Fifo)),
        _ => None,
    }
}

/// `FIFOBatchBuilder`; see its docstring for the rules and why.
pub struct Fifo;

impl BatchBuilder for Fifo {
    fn build(&self, req: &BuildRequest) -> BuildResult {
        let mut exclude = req.blocked.clone();
        let max_bs: Option<i64> = match single_walk(req) {
            None => {
                let (max_bs, left_out) = compose_walks(req, &exclude);
                exclude.extend(left_out);
                max_bs
            }
            Some(w) => req.caps.cap(w).map(|c| i64::from(c) - i64::from(req.pre)),
        };
        if max_bs.is_some_and(|m| m <= 0) {
            // the caller's rows fill it
            return BuildResult { parked: (0..req.rows.len()).collect(), ..Default::default() };
        }
        let (keep, out): (Vec<usize>, Vec<usize>) =
            (0..req.rows.len()).partition(|&i| !exclude.contains(&req.rows[i].rid));
        let bs = max_bs.map_or(keep.len(), |m| (m as usize).min(keep.len()));
        let mut result = BuildResult { scheduled: keep[..bs].to_vec(), ..Default::default() };
        // A fresh row the cap pushed out goes back to its ready queue only on
        // a step that served a backlog; otherwise it parks.
        for &i in &keep[bs..] {
            if req.n_backlog > 0 && i >= req.n_backlog {
                result.returned.push(i);
            } else {
                result.parked.push(i);
            }
        }
        result.parked.extend(out);
        result
    }
}

/// `_single_walk`: the one walk of every row, the caller's included.
fn single_walk(req: &BuildRequest) -> Option<Sym> {
    if !req.combined {
        return Some(req.label);
    }
    let mut walks: Vec<Sym> = Vec::new();
    for w in req.rows.iter().map(|r| r.walk).chain(req.pre_walk) {
        if !walks.contains(&w) {
            walks.push(w);
        }
    }
    if walks.len() == 1 { Some(walks[0]) } else { None }
}

/// `_compose_walks`: rows left for the step after the caller's, and the rows
/// left out.
fn compose_walks(req: &BuildRequest, exclude: &FxHashSet<u32>) -> (Option<i64>, Vec<u32>) {
    let caps = req.caps;
    let mut n: i64 = i64::from(req.pre);
    let mut walks: Vec<Sym> = req.pre_walk.into_iter().collect();
    let mut counts: Vec<(Sym, i64)> = req.pre_walk.map(|w| (w, n)).into_iter().collect();
    let count = |counts: &Vec<(Sym, i64)>, w: Sym| counts.iter().find(|c| c.0 == w).map_or(0, |c| c.1);
    let mut cap = min_cap(walks.iter().map(|&w| caps.cap(w)));
    let mut left_out = Vec::new();
    for row in req.rows {
        if exclude.contains(&row.rid) {
            continue;
        }
        let walk = row.walk;
        let mut new_cap = cap;
        if !walks.contains(&walk) {
            // a mixed step: the combined walk's own cap where it has one,
            // else the smallest of the walks present
            let mixed_cap = if walks.is_empty() { None } else { caps.cap(req.label) };
            new_cap = if mixed_cap.is_some() { mixed_cap } else { min_cap([cap, caps.cap(walk)]) };
        }
        if new_cap.is_some_and(|c| i64::from(c) <= n) {
            left_out.push(row.rid);
            continue;
        }
        if !caps.row_caps.is_empty() && !walks.is_empty() {
            let over = if !walks.contains(&walk) {
                // the row makes the step mix walks: every walk's rows count
                walks.iter().copied().chain(std::iter::once(walk)).any(|w| {
                    caps.row_cap(w).is_some_and(|rc| {
                        count(&counts, w) + i64::from(w == walk) > i64::from(rc)
                    })
                })
            } else {
                walks.len() > 1
                    && caps.row_cap(walk).is_some_and(|rc| count(&counts, walk) + 1 > i64::from(rc))
            };
            if over {
                left_out.push(row.rid);
                continue;
            }
        }
        if !walks.contains(&walk) {
            walks.push(walk);
        }
        match counts.iter_mut().find(|c| c.0 == walk) {
            Some(c) => c.1 += 1,
            None => counts.push((walk, 1)),
        }
        cap = new_cap;
        n += 1;
    }
    (cap.map(|c| i64::from(c) - i64::from(req.pre)), left_out)
}
