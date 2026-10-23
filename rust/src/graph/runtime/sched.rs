//! The micro-scheduler's pass, next to the graph state it works on.
//!
//! `MicroScheduler.get_next_batch` after the TP-follow FIFO, split at the
//! engine's readiness check: the check is per-rank resource code with side
//! effects (a KV reload), so it stays Python. A pass runs here until it needs
//! verdicts, hands the caller the rows to check (`CheckOut`), and continues
//! from where it stopped when they come back -- plan, check, commit, with one
//! crossing per check request. The backlog and the round-robin table live
//! here, beside the ready masks, pops, in-flight flags and push-backs they
//! drive.
//!
//! What it covers is what `MicroScheduler._rust_core` hands it: no node that
//! splits batches by capture key (so no capture groups) and no speculation
//! chain that mixes walks. Everything else is the Python core's
//! (`micro_scheduler.py`), which is the reference: the logic below follows it
//! step for step, and `test/rust/test_sched_core_parity.py` drives both.

use super::*;

/// A scheduling key: (node, walk), the walk being the combined walk where
/// the real one has one.
pub(super) type Key = (Sym, Sym);

/// One verdict per requested row.
const NOT_READY: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;

/// The engine's caps for one key, asked once by the Python side.
struct KeyCfg {
    /// The real walks under a combined key; empty for a plain one.
    walks: Vec<Sym>,
    /// walk -> cap (None: uncapped), in `MicroScheduler._walk_caps` order:
    /// each real walk, then the key's own.
    caps: Vec<(Sym, Option<u32>)>,
    /// `_walk_row_caps`: per real walk, rows a mixed step may hold.
    row_caps: Vec<(Sym, u32)>,
}

impl KeyCfg {
    /// `caps.get(walk)`: None for an uncapped walk and for one not listed.
    fn cap(&self, walk: Sym) -> Option<u32> {
        self.caps.iter().find(|c| c.0 == walk).and_then(|c| c.1)
    }

    fn row_cap(&self, walk: Sym) -> Option<u32> {
        self.row_caps.iter().find(|c| c.0 == walk).map(|c| c.1)
    }
}

#[derive(Clone)]
struct SRow {
    rid: u32,
    wg: u32,
    walk: Sym,
    /// the inputs the pop reported: (signal, uuids, final chunk)
    edges: Vec<(Sym, Vec<u64>, bool)>,
}

/// `ScheduledBatch`, by rows.
#[derive(Clone)]
struct SBatch {
    node: Sym,
    label: Sym,
    rows: Vec<SRow>,
    /// `request_walks` is set: every row names its real walk
    combined: bool,
    out_sigs: Vec<String>,
    walk_out: Vec<(Sym, Vec<String>)>,
}

impl SBatch {
    fn with_rows(&self, rows: Vec<SRow>) -> SBatch {
        SBatch {
            node: self.node, label: self.label, rows, combined: self.combined,
            out_sigs: self.out_sigs.clone(), walk_out: self.walk_out.clone(),
        }
    }

    /// `ScheduledBatch.merge`: `other`'s rows behind ours.
    fn merge(&mut self, other: SBatch) -> PyResult<()> {
        let mine: FxHashSet<u32> = self.rows.iter().map(|r| r.rid).collect();
        if other.rows.iter().any(|r| mine.contains(&r.rid)) {
            return Err(PyRuntimeError::new_err(
                "a rid is in both halves of a merge; its node was popped twice",
            ));
        }
        self.rows.extend(other.rows);
        self.combined |= other.combined;
        for (walk, sigs) in other.walk_out {
            match self.walk_out.iter_mut().find(|w| w.0 == walk) {
                Some(w) => w.1 = sigs,
                None => self.walk_out.push((walk, sigs)),
            }
        }
        Ok(())
    }

    /// `ScheduledBatch.split_off_first`: up to `bs` rows not in `exclude`,
    /// and the rest (the excluded ones last).
    fn split_off_first(
        self, bs: Option<usize>, exclude: &FxHashSet<u32>,
    ) -> (Option<SBatch>, Option<SBatch>) {
        let bs = bs.unwrap_or(self.rows.len());
        let (keep, out): (Vec<&SRow>, Vec<&SRow>) =
            self.rows.iter().partition(|r| !exclude.contains(&r.rid));
        if keep.len() <= bs && out.is_empty() {
            return (Some(self), None);
        }
        if keep.is_empty() {
            return (None, Some(self));
        }
        let taken: Vec<SRow> = keep[..bs.min(keep.len())].iter().map(|r| (*r).clone()).collect();
        let left: Vec<SRow> = keep[bs.min(keep.len())..]
            .iter()
            .chain(out.iter())
            .map(|r| (*r).clone())
            .collect();
        (Some(self.with_rows(taken)), Some(self.with_rows(left)))
    }

    /// `ScheduledBatch.relabel_if_uniform`.
    fn relabel_if_uniform(&mut self) {
        if !self.combined {
            return;
        }
        let first = self.rows[0].walk;
        if self.rows.iter().any(|r| r.walk != first) {
            return;
        }
        self.label = first;
        if let Some(sigs) = self.walk_out.iter().find(|w| w.0 == first) {
            self.out_sigs = sigs.1.clone();
        }
        self.combined = false;
        self.walk_out.clear();
    }
}

#[derive(Clone, Copy)]
struct Entry {
    rid: u32,
    walk: Sym,
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    KeyLoop,
    KeyScan,
    KeyFresh,
    FinalScan,
    FinalPick,
}

/// Where a `_schedule_key` result goes.
#[derive(Clone, Copy)]
enum After {
    /// the backlog-only attempt of `keys[ki]`
    BacklogOnly,
    /// `keys[ki]` with its fresh rows
    KeyFresh,
    /// the round-robin pick
    Final,
}

/// A `_schedule_key` paused at its backlog's check.
struct KeyCtx {
    key: Key,
    entries: Vec<Entry>,
    batch: SBatch,
    waiting_rows: FxHashSet<u32>,
    cands: Vec<u32>,
    after: After,
}

enum Await {
    /// the fresh scan: (rid, key, real walk) per requested row
    Scan { rows: Vec<(u32, Key, Sym)>, then: Stage },
    Backlog(KeyCtx),
}

pub(super) struct Pass {
    target: Option<Key>,
    pre: u32,
    pre_walk: Option<Sym>,
    skip_walk: Option<Sym>,
    scan_target: Option<(String, String)>,
    scan_exclude: Option<(String, String)>,
    waiting: FxHashSet<u32>,
    failed: FxHashSet<u32>,
    scan_excl: FxHashSet<u32>,
    keys: Vec<Key>,
    ki: usize,
    ready: Option<Vec<(Key, Vec<Entry>)>>,
    stage: Stage,
    awaiting: Option<Await>,
}

#[derive(Default)]
pub(super) struct SchedState {
    keys: FxHashMap<Key, KeyCfg>,
    leaders: FxHashSet<Sym>,
    /// In `MicroScheduler.backlog`'s order: a key taken for a step and parked
    /// again moves to the end, as a dict pop and re-insert does.
    backlog: Vec<(Key, SBatch)>,
    last_batch: FxHashMap<Key, u64>,
    batch_number: u64,
    pass: Option<Pass>,
}

impl SchedState {
    /// `_drop_backlogged_rid` for several: the rows go, then any empty chunk.
    pub(super) fn drop_rids(&mut self, rids: &FxHashSet<u32>) {
        if rids.is_empty() {
            return;
        }
        for (_, batch) in self.backlog.iter_mut() {
            batch.rows.retain(|r| !rids.contains(&r.rid));
        }
        self.backlog.retain(|(_, b)| !b.rows.is_empty());
    }
}

/// `(rids, node_idx, nodes, limit)`: rows to check, in order.
pub(super) type CheckOut = (Vec<u32>, Option<Vec<u32>>, Vec<String>, i64);
/// A `ScheduledBatch` as columns: node, walk label, rids, worker graph ids,
/// the walk names and each row's index into them (None unless the batch
/// mixes walks), output signals, per-walk output signals, and the input
/// edges as `ColumnarEdgeSpecs` columns.
pub(super) type BatchOut = (
    String, String, Vec<u32>, Vec<u32>, Vec<String>, Option<Vec<u32>>,
    Vec<String>, Vec<(String, Vec<String>)>,
    (Vec<String>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<bool>, Vec<u64>),
);

enum KeyStep {
    Done(Option<SBatch>),
    Check(KeyCtx, CheckOut),
}

fn min_cap(caps: impl IntoIterator<Item = Option<u32>>) -> Option<u32> {
    caps.into_iter().flatten().min()
}

impl GraphRuntime {
    fn sym_or_none(&self, s: &str) -> Sym {
        // a name no graph has matches nothing
        self.interner.get(s).unwrap_or(Sym::MAX)
    }

    fn sched_key_of(&self, node: Sym, walk: Sym) -> Key {
        (node, self.combined_walk_of.get(&(node, walk)).copied().unwrap_or(walk))
    }

    fn sched_cfg(&self, key: Key) -> PyResult<&KeyCfg> {
        self.sched.keys.get(&key).ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "scheduling key ({}, {}) was not configured",
                self.interner.try_name(key.0).unwrap_or("?"),
                self.interner.try_name(key.1).unwrap_or("?"),
            ))
        })
    }

    pub(super) fn sched_configure_impl(
        &mut self,
        keys: Vec<(String, String, Vec<String>, Vec<(String, Option<u32>)>, Vec<(String, u32)>)>,
        leaders: Vec<String>,
    ) {
        let mut cfgs = FxHashMap::default();
        for (node, label, walks, caps, row_caps) in keys {
            let key = (self.interner.intern(&node), self.interner.intern(&label));
            let cfg = KeyCfg {
                walks: walks.iter().map(|w| self.interner.intern(w)).collect(),
                caps: caps.iter().map(|(w, c)| (self.interner.intern(w), *c)).collect(),
                row_caps: row_caps.iter().map(|(w, c)| (self.interner.intern(w), *c)).collect(),
            };
            cfgs.insert(key, cfg);
        }
        self.sched.keys = cfgs;
        self.sched.leaders = leaders.iter().map(|n| self.interner.intern(n)).collect();
    }

    /// `_room_left`: what is left of the caller's own walk's cap.
    fn sched_room_left(&self, key: Key, pre_walk: Option<Sym>, pre: u32) -> PyResult<Option<u32>> {
        if pre == 0 {
            return Ok(None);
        }
        let cap = self.sched_cfg(key)?.cap(pre_walk.unwrap_or(key.1));
        Ok(cap.map(|c| c.saturating_sub(pre)))
    }

    /// `_scan_limit`: an upper bound on the rows a step can take.
    fn sched_scan_limit(&self, key: Key, pre: u32) -> PyResult<Option<u32>> {
        let caps = &self.sched_cfg(key)?.caps;
        if caps.is_empty() || caps.iter().any(|c| c.1.is_none()) {
            return Ok(None);
        }
        let max = caps.iter().filter_map(|c| c.1).max().unwrap_or(0);
        Ok(Some(max.saturating_sub(pre)))
    }

    /// `_backlog_fills`.
    fn sched_backlog_fills(&self, p: &Pass, key: Key) -> PyResult<bool> {
        let mut room = self.sched_room_left(key, p.pre_walk, p.pre)?;
        if room.is_none() && p.pre == 0 {
            let caps = &self.sched_cfg(key)?.caps;
            room = if caps.iter().any(|c| c.1.is_none()) {
                None
            } else {
                caps.iter().filter_map(|c| c.1).max()
            };
        }
        let Some(room) = room else { return Ok(false) };
        let live = self.sched.backlog.iter().find(|(k, _)| *k == key).map_or(0, |(_, b)| {
            b.rows.iter()
                .filter(|r| !p.failed.contains(&r.rid) && !p.waiting.contains(&r.rid))
                .count()
        });
        Ok(live as u32 >= room)
    }

    /// The graph-ready rows `_scan_ready` checks, in its order: `(rid, key,
    /// walk, node)` per row, leader nodes only.
    fn sched_scan_rows(&self, p: &Pass) -> Vec<(u32, Key, Sym, Sym)> {
        let mut out = Vec::new();
        // the rows failed earlier in this pass are out too
        let excl: FxHashSet<u32> = p.scan_excl.union(&p.failed).copied().collect();
        for (node, walk, rids) in self.ready_groups(&excl, p.scan_target.as_ref(), p.scan_exclude.as_ref()) {
            if !self.sched.leaders.contains(&node) {
                continue; // only rank 0 can initiate scheduling
            }
            if !self.node_owner.contains_key(&(walk, node)) {
                continue;
            }
            let key = self.sched_key_of(node, walk);
            out.extend(rids.into_iter().map(|rid| (rid, key, walk, node)));
        }
        out
    }

    fn check_out(&self, rids: Vec<u32>, nodes: &[Sym], limit: i64) -> CheckOut {
        let mut names: Vec<Sym> = Vec::new();
        let idx: Vec<u32> = nodes.iter().map(|n| {
            match names.iter().position(|m| m == n) {
                Some(i) => i as u32,
                None => {
                    names.push(*n);
                    (names.len() - 1) as u32
                }
            }
        }).collect();
        let node_idx = if names.len() > 1 { Some(idx) } else { None };
        let names = names.iter().map(|n| self.interner.name(*n).to_string()).collect();
        (rids, node_idx, names, limit)
    }

    /// `_schedule_key` up to its backlog's check; the rest is `sched_key_finish_ctx`.
    fn sched_key_start(
        &mut self, p: &Pass, key: Key, entries: Vec<Entry>, after: After,
    ) -> PyResult<KeyStep> {
        if self.sched_room_left(key, p.pre_walk, p.pre)? == Some(0) {
            // the caller already holds a full batch
            return Ok(KeyStep::Done(None));
        }
        let pos = self.sched.backlog.iter().position(|(k, _)| *k == key);
        let Some(pos) = pos else {
            return Ok(KeyStep::Done(self.sched_key_build(p, key, entries, None, FxHashSet::default())?));
        };
        let (_, batch) = self.sched.backlog.remove(pos);
        let limit = match self.sched_scan_limit(key, p.pre)? {
            None => batch.rows.len() as i64,
            Some(l) => i64::from(l),
        };
        let own_walk = p.pre_walk;
        let by_walk = batch.combined && (own_walk.is_some() || p.skip_walk.is_some());
        let waiting_rows: FxHashSet<u32> = batch.rows.iter()
            .filter(|r| {
                p.waiting.contains(&r.rid)
                    || by_walk && (own_walk.is_some_and(|w| r.walk != w) || Some(r.walk) == p.skip_walk)
            })
            .map(|r| r.rid)
            .collect();
        let cands: Vec<u32> = batch.rows.iter()
            .map(|r| r.rid)
            .filter(|rid| !waiting_rows.contains(rid))
            .collect();
        let ctx = KeyCtx { key, entries, batch, waiting_rows, cands, after };
        if ctx.cands.is_empty() || limit <= 0 {
            return Ok(KeyStep::Done(self.sched_key_finish_ctx(p, ctx, &[])?));
        }
        let req = self.check_out(ctx.cands.clone(), &[key.0], limit);
        Ok(KeyStep::Check(ctx, req))
    }

    /// The rest of `_schedule_key` for a key with a backlog: `_unready_backlog_rids`
    /// on the verdicts, then the build.
    fn sched_key_finish_ctx(
        &mut self, p: &Pass, ctx: KeyCtx, verdicts: &[u8],
    ) -> PyResult<Option<SBatch>> {
        let KeyCtx { key, entries, mut batch, waiting_rows, cands, .. } = ctx;
        let mut not_ready = waiting_rows;
        for (i, rid) in cands.iter().enumerate() {
            if matches!(verdicts.get(i), Some(&NOT_READY) | Some(&FAILED)) {
                not_ready.insert(*rid);
            }
        }
        // a failed row is dropped rather than parked
        batch.rows.retain(|r| !(not_ready.contains(&r.rid) && p.failed.contains(&r.rid)));
        let blocked: FxHashSet<u32> = not_ready.into_iter().filter(|r| !p.failed.contains(r)).collect();
        let backlogged = if batch.rows.is_empty() { None } else { Some(batch) };
        self.sched_key_build(p, key, entries, backlogged, blocked)
    }

    fn sched_key_build(
        &mut self, p: &Pass, key: Key, entries: Vec<Entry>,
        backlogged: Option<SBatch>, blocked: FxHashSet<u32>,
    ) -> PyResult<Option<SBatch>> {
        let fresh = if entries.is_empty() { None } else { self.sched_assemble(key, &entries)? };
        if backlogged.is_none() && fresh.is_none() {
            return Ok(None);
        }
        self.sched_build_and_schedule(p, key, backlogged, fresh, blocked)
    }

    /// `_assemble_batch`: pop the entries' node off their ready queues.
    fn sched_assemble(&mut self, key: Key, entries: &[Entry]) -> PyResult<Option<SBatch>> {
        let combined = !self.sched_cfg(key)?.walks.is_empty();
        let mut walks: Vec<Sym> = Vec::new();
        let mut walk_idx: Vec<u32> = Vec::with_capacity(entries.len());
        for e in entries {
            let w = if combined { e.walk } else { key.1 };
            let i = match walks.iter().position(|x| *x == w) {
                Some(i) => i,
                None => {
                    walks.push(w);
                    walks.len() - 1
                }
            };
            walk_idx.push(i as u32);
        }
        let rids: Vec<u32> = entries.iter().map(|e| e.rid).collect();
        let Some((rows, walk_sigs)) = self.sched_pop(key.0, &walks, &rids, &walk_idx)? else {
            return Ok(None);
        };
        if rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(SBatch {
            node: key.0,
            label: key.1,
            rows,
            combined,
            out_sigs: walk_sigs.first().cloned().unwrap_or_default(),
            walk_out: if combined { walks.into_iter().zip(walk_sigs).collect() } else { Vec::new() },
        }))
    }

    /// `pop_walk_rids` without the check, as rows.
    #[allow(clippy::type_complexity)]
    fn sched_pop(
        &mut self, node: Sym, walks: &[Sym], rids: &[u32], walk_idx: &[u32],
    ) -> PyResult<Option<(Vec<SRow>, Vec<Vec<String>>)>> {
        let node_name = self.interner.name(node).to_string();
        let mut targets = Vec::with_capacity(walks.len());
        for &w in walks {
            let walk_name = self.interner.name(w).to_string();
            let Some(&wg) = self.node_owner.get(&(w, node)) else {
                if self.runs_anywhere(&node_name, &walk_name) {
                    return Ok(None);
                }
                return Err(PyRuntimeError::new_err(format!(
                    "Could not find worker graph for node '{node_name}', graph_walk '{walk_name}'",
                )));
            };
            let Some(nid) = self.nid(wg, &node_name) else {
                return Err(PyRuntimeError::new_err(format!(
                    "worker graph {wg} owns node '{node_name}' in walk '{walk_name}' but has no such node",
                )));
            };
            targets.push((wg, nid));
        }
        let walk_sigs: Vec<Vec<String>> = targets.iter().map(|&(wg, nid)| {
            let g = self.g(wg).clone();
            let mut v: Vec<String> = g.node(nid).outputs.iter()
                .map(|e| self.interner.name(e.name).to_string())
                .collect();
            v.sort();
            v.dedup();
            v
        }).collect();
        let mut rows = Vec::with_capacity(rids.len());
        for (&rid, &w) in rids.iter().zip(walk_idx) {
            let (wg, nid) = targets[w as usize];
            let wg_id = self.wg_ids[wg as usize];
            let Some(state) = self.state_mut(wg, rid) else { continue };
            state.take_for_schedule(nid);
            let edges = state.input_tensors(nid, false).into_iter()
                .map(|(name, tensors, fin)| (name, tensors.iter().map(|t| t.uuid).collect(), fin))
                .collect();
            rows.push(SRow { rid, wg: wg_id, walk: walks[w as usize], edges });
        }
        Ok(Some((rows, walk_sigs)))
    }

    /// `_build_and_schedule` with `FIFOBatchBuilder.build_batch`.
    fn sched_build_and_schedule(
        &mut self, p: &Pass, key: Key,
        backlog: Option<SBatch>, fresh: Option<SBatch>, blocked: FxHashSet<u32>,
    ) -> PyResult<Option<SBatch>> {
        let mut expected: FxHashSet<u32> = FxHashSet::default();
        for b in backlog.iter().chain(fresh.iter()) {
            expected.extend(b.rows.iter().map(|r| r.rid));
        }
        let had_backlog = backlog.is_some();
        let fresh_rids: FxHashSet<u32> = fresh.iter().flat_map(|b| b.rows.iter().map(|r| r.rid)).collect();
        let batch = match (backlog, fresh) {
            (None, None) => return Ok(None),
            (Some(b), None) => b,
            (None, Some(f)) => f,
            (Some(mut b), Some(f)) => {
                b.merge(f)?;
                b
            }
        };

        let mut exclude = blocked;
        let walk = self.sched_single_walk(&batch, key.1, p.pre_walk);
        let max_bs: Option<i64> = match walk {
            None => {
                let (max_bs, left_out) = self.sched_compose_walks(&batch, &exclude, key, p)?;
                exclude.extend(left_out);
                max_bs
            }
            Some(w) => self.sched_cfg(key)?.cap(w).map(|c| i64::from(c) - i64::from(p.pre)),
        };

        let (scheduled, parked, returned) = if max_bs.is_some_and(|m| m <= 0) {
            (None, Some(batch), None) // the caller's rows fill it
        } else {
            let (scheduled, remainder) = batch.split_off_first(max_bs.map(|m| m as usize), &exclude);
            match remainder {
                None => (scheduled, None, None),
                Some(remainder) => {
                    let give_back: FxHashSet<u32> = if had_backlog {
                        remainder.rows.iter()
                            .map(|r| r.rid)
                            .filter(|r| fresh_rids.contains(r) && !exclude.contains(r))
                            .collect()
                    } else {
                        FxHashSet::default()
                    };
                    if give_back.is_empty() {
                        (scheduled, Some(remainder), None)
                    } else {
                        let (parked, returned) = remainder.split_off_first(None, &give_back);
                        (scheduled, parked, returned)
                    }
                }
            }
        };

        // `_refuse_lost_rows`: every row handed in comes back out
        let mut got: FxHashSet<u32> = FxHashSet::default();
        for b in scheduled.iter().chain(parked.iter()).chain(returned.iter()) {
            got.extend(b.rows.iter().map(|r| r.rid));
        }
        if got != expected {
            return Err(PyRuntimeError::new_err(format!(
                "batch builder dropped rows for ({}, {})",
                self.interner.name(key.0), self.interner.name(key.1),
            )));
        }
        let node_name = self.interner.name(key.0).to_string();
        if let Some(parked) = parked {
            let rids: Vec<u32> = parked.rows.iter().map(|r| r.rid).collect();
            let wgs: Vec<u32> = parked.rows.iter().map(|r| r.wg).collect();
            match self.sched.backlog.iter_mut().find(|(k, _)| *k == key) {
                Some((_, existing)) => existing.merge(parked)?,
                None => self.sched.backlog.push((key, parked)),
            }
            // claimed like a running batch, so an input arriving while it
            // waits does not put its node back on the ready queue
            self.set_in_flight(node_name.clone(), rids, wgs, true)?;
        }
        if let Some(returned) = returned {
            let rids: Vec<u32> = returned.rows.iter().map(|r| r.rid).collect();
            let wgs: Vec<u32> = returned.rows.iter().map(|r| r.wg).collect();
            self.set_in_flight(node_name.clone(), rids.clone(), wgs.clone(), false)?;
            self.push_back_node(node_name, rids, wgs)?;
        }
        let Some(mut scheduled) = scheduled else { return Ok(None) };
        self.sched.batch_number += 1;
        self.sched.last_batch.insert(key, self.sched.batch_number);
        scheduled.relabel_if_uniform();
        Ok(Some(scheduled))
    }

    /// `_single_walk`: the one walk of every row, the caller's included.
    fn sched_single_walk(&self, batch: &SBatch, label: Sym, pre_walk: Option<Sym>) -> Option<Sym> {
        if !batch.combined {
            return Some(label);
        }
        let mut walks: Vec<Sym> = Vec::new();
        for w in batch.rows.iter().map(|r| r.walk).chain(pre_walk) {
            if !walks.contains(&w) {
                walks.push(w);
            }
        }
        if walks.len() == 1 { Some(walks[0]) } else { None }
    }

    /// `_compose_walks`: rows left for the step after the caller's, and the
    /// rows left out.
    fn sched_compose_walks(
        &self, batch: &SBatch, exclude: &FxHashSet<u32>, key: Key, p: &Pass,
    ) -> PyResult<(Option<i64>, FxHashSet<u32>)> {
        let cfg = self.sched_cfg(key)?;
        let mut n: i64 = i64::from(p.pre);
        let mut walks: Vec<Sym> = p.pre_walk.into_iter().collect();
        let mut counts: Vec<(Sym, i64)> = p.pre_walk.map(|w| (w, n)).into_iter().collect();
        let count = |counts: &Vec<(Sym, i64)>, w: Sym| counts.iter().find(|c| c.0 == w).map_or(0, |c| c.1);
        let mut cap = min_cap(walks.iter().map(|&w| cfg.cap(w)));
        let mut left_out = FxHashSet::default();
        for row in &batch.rows {
            if exclude.contains(&row.rid) {
                continue;
            }
            let walk = row.walk;
            let mut new_cap = cap;
            if !walks.contains(&walk) {
                let mixed_cap = if walks.is_empty() { None } else { cfg.cap(key.1) };
                new_cap = if mixed_cap.is_some() { mixed_cap } else { min_cap([cap, cfg.cap(walk)]) };
            }
            if new_cap.is_some_and(|c| i64::from(c) <= n) {
                left_out.insert(row.rid);
                continue;
            }
            if !cfg.row_caps.is_empty() && !walks.is_empty() {
                let over = if !walks.contains(&walk) {
                    walks.iter().copied().chain(std::iter::once(walk)).any(|w| {
                        cfg.row_cap(w).is_some_and(|rc| {
                            count(&counts, w) + i64::from(w == walk) > i64::from(rc)
                        })
                    })
                } else {
                    walks.len() > 1
                        && cfg.row_cap(walk).is_some_and(|rc| count(&counts, walk) + 1 > i64::from(rc))
                };
                if over {
                    left_out.insert(row.rid);
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
        Ok((cap.map(|c| i64::from(c) - i64::from(p.pre)), left_out))
    }

    /// Begin a pass: `get_next_batch` after the TP-follow FIFO.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sched_begin_impl(
        &mut self,
        target: Option<(String, String)>,
        exclude_target: Option<(String, String)>,
        pre: u32,
        waiting: Vec<u32>,
        failed: Vec<u32>,
        scan_exclude: Vec<u32>,
    ) -> PyResult<(bool, Option<CheckOut>, Option<BatchOut>)> {
        let sym_pair = |rt: &Self, t: &(String, String)| (rt.sym_or_none(&t.0), rt.sym_or_none(&t.1));
        let target_key = target.as_ref().map(|t| {
            let (n, w) = sym_pair(self, t);
            self.sched_key_of(n, w)
        });
        let pre_walk = if pre > 0 { target.as_ref().map(|t| self.sym_or_none(&t.1)) } else { None };
        let skip_walk = exclude_target.as_ref().map(|t| self.sym_or_none(&t.1));
        let key_strings = |rt: &Self, k: Key| (
            rt.interner.try_name(k.0).unwrap_or("").to_string(),
            rt.interner.try_name(k.1).unwrap_or("").to_string(),
        );
        // the runtime matches a target against real and combined walks alike
        let scan_target = if pre_walk.is_some() {
            target.clone()
        } else {
            target_key.map(|k| key_strings(self, k)).or(target.clone())
        };
        let p = Pass {
            target: target_key,
            pre,
            pre_walk,
            skip_walk,
            scan_target,
            scan_exclude: exclude_target,
            waiting: waiting.into_iter().collect(),
            failed: failed.into_iter().collect(),
            scan_excl: scan_exclude.into_iter().collect(),
            keys: Vec::new(),
            ki: 0,
            ready: None,
            stage: Stage::KeyLoop,
            awaiting: None,
        };
        if let Some(t) = p.target {
            if !self.sched.keys.contains_key(&t) {
                return Err(PyRuntimeError::new_err("target is not a configured scheduling key"));
            }
            if self.sched_room_left(t, p.pre_walk, p.pre)? == Some(0) {
                // another walk's rows wait on their queues for the yield
                return Ok((true, None, None));
            }
        }
        let mut p = p;
        p.keys = self.sched.backlog.iter()
            .map(|(k, _)| *k)
            .filter(|k| p.target.is_none_or(|t| t == *k))
            .collect();
        self.sched_run(p, None)
    }

    pub(super) fn sched_resume_impl(
        &mut self, verdicts: &[u8],
    ) -> PyResult<(bool, Option<CheckOut>, Option<BatchOut>)> {
        let p = self.sched.pass.take().ok_or_else(|| {
            PyRuntimeError::new_err("sched_resume: no pass is waiting for verdicts")
        })?;
        self.sched_run(p, Some(verdicts))
    }

    /// Take a FAILED verdict's rid out of the pass and the backlog.
    fn sched_note_failed(&mut self, p: &mut Pass, rids: &[u32], verdicts: &[u8]) -> PyResult<()> {
        if rids.len() != verdicts.len() {
            return Err(PyValueError::new_err("sched_resume: one verdict per requested row"));
        }
        let failed: FxHashSet<u32> = rids.iter().zip(verdicts)
            .filter(|(_, v)| **v == FAILED)
            .map(|(r, _)| *r)
            .collect();
        self.sched.drop_rids(&failed);
        p.failed.extend(failed);
        Ok(())
    }

    fn sched_run(
        &mut self, mut p: Pass, verdicts: Option<&[u8]>,
    ) -> PyResult<(bool, Option<CheckOut>, Option<BatchOut>)> {
        // a _schedule_key result still to act on, with where it goes
        let mut result: Option<(Option<SBatch>, After)> = None;
        if let Some(aw) = p.awaiting.take() {
            let v = verdicts.ok_or_else(|| PyRuntimeError::new_err("sched_resume: verdicts missing"))?;
            match aw {
                Await::Scan { rows, then } => {
                    let rids: Vec<u32> = rows.iter().map(|r| r.0).collect();
                    self.sched_note_failed(&mut p, &rids, v)?;
                    p.ready = Some(Self::sched_ready_from(&rows, v));
                    p.stage = then;
                }
                Await::Backlog(ctx) => {
                    self.sched_note_failed(&mut p, &ctx.cands.clone(), v)?;
                    let after = ctx.after;
                    result = Some((self.sched_key_finish_ctx(&p, ctx, v)?, after));
                }
            }
        }
        loop {
            if let Some((scheduled, after)) = result.take() {
                match after {
                    After::BacklogOnly => {
                        if scheduled.is_some() {
                            return Ok(self.sched_done(scheduled));
                        }
                        p.stage = Stage::KeyScan;
                    }
                    After::KeyFresh => {
                        if scheduled.is_some() {
                            return Ok(self.sched_done(scheduled));
                        }
                        p.ki += 1;
                        p.stage = Stage::KeyLoop;
                    }
                    After::Final => return Ok(self.sched_done(scheduled)),
                }
            }
            match p.stage {
                Stage::KeyLoop => {
                    if p.ki >= p.keys.len() {
                        p.stage = Stage::FinalScan;
                        continue;
                    }
                    let key = p.keys[p.ki];
                    // no fresh row could join a step its backlog fills,
                    // unless every backlogged row is blocked
                    if self.sched_backlog_fills(&p, key)? {
                        match self.sched_key_start(&p, key, Vec::new(), After::BacklogOnly)? {
                            KeyStep::Done(b) => result = Some((b, After::BacklogOnly)),
                            KeyStep::Check(ctx, req) => return Ok(self.sched_suspend(p, Await::Backlog(ctx), req)),
                        }
                    } else {
                        p.stage = Stage::KeyScan;
                    }
                }
                Stage::KeyScan | Stage::FinalScan => {
                    let then = if p.stage == Stage::KeyScan { Stage::KeyFresh } else { Stage::FinalPick };
                    if p.ready.is_some() {
                        p.stage = then;
                        continue;
                    }
                    let rows = self.sched_scan_rows(&p);
                    if rows.is_empty() {
                        p.ready = Some(Vec::new());
                        p.stage = then;
                        continue;
                    }
                    let rids: Vec<u32> = rows.iter().map(|r| r.0).collect();
                    let nodes: Vec<Sym> = rows.iter().map(|r| r.3).collect();
                    let req = self.check_out(rids, &nodes, -1);
                    let rows = rows.into_iter().map(|(rid, key, walk, _)| (rid, key, walk)).collect();
                    return Ok(self.sched_suspend(p, Await::Scan { rows, then }, req));
                }
                Stage::KeyFresh => {
                    let key = p.keys[p.ki];
                    let entries = Self::ready_for(&p, key);
                    match self.sched_key_start(&p, key, entries, After::KeyFresh)? {
                        KeyStep::Done(b) => result = Some((b, After::KeyFresh)),
                        KeyStep::Check(ctx, req) => return Ok(self.sched_suspend(p, Await::Backlog(ctx), req)),
                    }
                }
                Stage::FinalPick => {
                    // A walk with no room (a cap of 0) takes nothing, so it
                    // stays the least recent; picked anyway, it would starve
                    // the walk that frees the room.
                    let mut best: Option<(Key, u64)> = None;
                    for (key, _) in p.ready.as_deref().unwrap_or(&[]) {
                        if self.sched_cfg(*key)?.cap(key.1) == Some(0) {
                            continue;
                        }
                        let last = self.sched.last_batch.get(key).copied().unwrap_or(0);
                        if best.is_none_or(|(_, b)| last < b) {
                            best = Some((*key, last));
                        }
                    }
                    let Some((rr, _)) = best else { return Ok(self.sched_done(None)) };
                    if p.keys.contains(&rr) {
                        return Ok(self.sched_done(None));
                    }
                    let entries = Self::ready_for(&p, rr);
                    match self.sched_key_start(&p, rr, entries, After::Final)? {
                        KeyStep::Done(b) => result = Some((b, After::Final)),
                        KeyStep::Check(ctx, req) => return Ok(self.sched_suspend(p, Await::Backlog(ctx), req)),
                    }
                }
            }
        }
    }

    fn ready_for(p: &Pass, key: Key) -> Vec<Entry> {
        p.ready.as_deref().unwrap_or(&[]).iter()
            .find(|(k, _)| *k == key)
            .map(|(_, e)| e.clone())
            .unwrap_or_default()
    }

    fn sched_ready_from(rows: &[(u32, Key, Sym)], verdicts: &[u8]) -> Vec<(Key, Vec<Entry>)> {
        let mut ready: Vec<(Key, Vec<Entry>)> = Vec::new();
        for (&(rid, key, walk), &v) in rows.iter().zip(verdicts) {
            if v != READY {
                continue;
            }
            let entry = Entry { rid, walk };
            match ready.iter_mut().find(|(k, _)| *k == key) {
                Some((_, e)) => e.push(entry),
                None => ready.push((key, vec![entry])),
            }
        }
        ready
    }

    fn sched_suspend(
        &mut self, mut p: Pass, aw: Await, req: CheckOut,
    ) -> (bool, Option<CheckOut>, Option<BatchOut>) {
        p.awaiting = Some(aw);
        self.sched.pass = Some(p);
        (false, Some(req), None)
    }

    fn sched_done(&self, batch: Option<SBatch>) -> (bool, Option<CheckOut>, Option<BatchOut>) {
        (true, None, batch.map(|b| self.batch_out(b)))
    }

    fn batch_out(&self, b: SBatch) -> BatchOut {
        let name = |s: Sym| self.interner.name(s).to_string();
        let mut walk_names: Vec<Sym> = Vec::new();
        let mut walk_idx: Vec<u32> = Vec::with_capacity(b.rows.len());
        let mut cols = EdgeColumns::default();
        for r in &b.rows {
            let i = match walk_names.iter().position(|w| *w == r.walk) {
                Some(i) => i,
                None => {
                    walk_names.push(r.walk);
                    walk_names.len() - 1
                }
            };
            walk_idx.push(i as u32);
            for (sig, uuids, fin) in &r.edges {
                cols.push(&self.interner, r.rid, *sig, uuids.iter().copied(), *fin);
            }
        }
        (
            name(b.node),
            name(b.label),
            b.rows.iter().map(|r| r.rid).collect(),
            b.rows.iter().map(|r| r.wg).collect(),
            walk_names.iter().map(|w| name(*w)).collect(),
            if b.combined { Some(walk_idx) } else { None },
            b.out_sigs,
            b.walk_out.into_iter().map(|(w, s)| (name(w), s)).collect(),
            (cols.signal_names, cols.signal_name_idxs, cols.rids, cols.tensors_per_edge,
             cols.is_final_streaming_chunk, cols.uuids),
        )
    }

    /// `room_for_continuing` (no mixing chains): how many of a speculation
    /// chain's own rows fit beside `(node, walk)`'s backlog.
    pub(super) fn sched_room_for_continuing_impl(&self, node: &str, walk: &str) -> PyResult<Option<u32>> {
        let (n, w) = (self.sym_or_none(node), self.sym_or_none(walk));
        let key = self.sched_key_of(n, w);
        let cap = self.sched_cfg(key)?.cap(w);
        let backlog = self.sched.backlog.iter().find(|(k, _)| *k == key).map(|(_, b)| b);
        Ok(match backlog {
            Some(b) if b.combined => {
                let Some(cap) = cap else { return Ok(None) };
                let own = b.rows.iter().filter(|r| r.walk == w).count() as u32;
                Some(cap.saturating_sub(own))
            }
            _ => cap.map(|c| c.saturating_sub(backlog.map_or(0, |b| b.rows.len() as u32))),
        })
    }

    /// `_backlog_has_schedulable`; `exclude_key` is already a scheduling key.
    pub(super) fn sched_backlog_has_schedulable_impl(
        &self, exclude_key: Option<(String, String)>, exclude_walk: Option<String>,
        waiting: Vec<u32>, failed: Vec<u32>,
    ) -> bool {
        let exclude = exclude_key.map(|(n, w)| (self.sym_or_none(&n), self.sym_or_none(&w)));
        let exclude_walk = exclude_walk.map(|w| self.sym_or_none(&w));
        let waiting: FxHashSet<u32> = waiting.into_iter().collect();
        let failed: FxHashSet<u32> = failed.into_iter().collect();
        for (key, batch) in &self.sched.backlog {
            let excluded = Some(*key) == exclude;
            let only_other = excluded && exclude_walk.is_some();
            if excluded && !only_other {
                continue;
            }
            for r in &batch.rows {
                // a plain batch's rows all run its label
                let walk = if batch.combined { r.walk } else { batch.label };
                if only_other && Some(walk) == exclude_walk {
                    continue;
                }
                if failed.contains(&r.rid) || waiting.contains(&r.rid) {
                    continue;
                }
                return true;
            }
        }
        false
    }

    /// Per backlogged chunk, its node and how many of its rows are not in
    /// `exclude` (zero counts included, as `pending_rows` adds them).
    pub(super) fn sched_backlog_rows_impl(&self, exclude: Vec<u32>) -> Vec<(String, u32)> {
        let exclude: FxHashSet<u32> = exclude.into_iter().collect();
        self.sched.backlog.iter()
            .map(|((n, _), b)| (
                self.interner.name(*n).to_string(),
                b.rows.iter().filter(|r| !exclude.contains(&r.rid)).count() as u32,
            ))
            .collect()
    }

    /// For tests: the backlog in order as `(node, label, rids, walks)` (walks
    /// empty for a plain batch), the round-robin table and the counter.
    #[allow(clippy::type_complexity)]
    pub(super) fn sched_state_impl(
        &self,
    ) -> (Vec<(String, String, Vec<u32>, Vec<String>)>, Vec<(String, String, u64)>, u64) {
        let name = |s: Sym| self.interner.name(s).to_string();
        let backlog = self.sched.backlog.iter()
            .map(|((n, l), b)| (
                name(*n), name(*l),
                b.rows.iter().map(|r| r.rid).collect(),
                if b.combined { b.rows.iter().map(|r| name(r.walk)).collect() } else { Vec::new() },
            ))
            .collect();
        let mut rr: Vec<(String, String, u64)> = self.sched.last_batch.iter()
            .map(|((n, l), v)| (name(*n), name(*l), *v))
            .collect();
        rr.sort();
        (backlog, rr, self.sched.batch_number)
    }

    /// Every (node, walk) a local worker graph runs, sorted.
    pub(super) fn local_node_walks_impl(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self.node_owner.keys()
            .map(|(w, n)| (self.interner.name(*n).to_string(), self.interner.name(*w).to_string()))
            .collect();
        out.sort();
        out
    }
}
