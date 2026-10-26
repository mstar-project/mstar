//! The micro-scheduler's pass, next to the graph state it works on.
//!
//! `MicroScheduler.get_next_batch` after the TP-follow FIFO, split at the
//! engine's readiness check: the check is per-rank resource code with side
//! effects (a KV reload), so it stays Python. A pass runs here until it needs
//! verdicts, hands the caller the rows to check (`CheckOut`), and continues
//! from where it stopped when they come back -- plan, check, commit, with one
//! crossing per check request. The backlog and the round-robin table live
//! here, beside the ready masks, pops, in-flight flags and push-backs they
//! drive. Which popped rows a step takes is the configured batch builder's
//! (`crate::batch_builder`).
//!
//! The Python core (`mstar/worker/sched_core/python.py`) is the reference:
//! the logic below follows it step for step, and
//! `test/rust/test_sched_core_parity.py` drives both.

use super::*;
use crate::batch_builder::{self, BatchBuilder, BuildRequest, KeyCaps, Row, min_cap};

/// A scheduling key: (node, walk), the walk being the combined walk where
/// the real one has one.
pub(super) type Key = (Sym, Sym);

/// One verdict per requested row (`sched_core.base.Verdict`).
const READY: u8 = 1;
const FAILED: u8 = 2;
const UNCHECKED: u8 = 3;
/// The group of a row whose node does not split batches by capture key.
const NO_GROUP: i64 = -1;

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
    /// capture group id, `NO_GROUP` for a node that does not split by one
    group: i64,
}

/// `_Stage`: the pass's current attempt. Each backlogged key is tried on
/// its backlog alone (when that fills a step), then with the scan's fresh
/// rows; after them, the round-robin pick over the scan.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    BacklogOnly,
    WithFresh,
    Pick,
    /// a mixing chain at its cap: park its combined walk's other ready rows
    Park,
}

/// `_BacklogCheck`: a key's attempt paused at its backlog's check.
struct KeyCtx {
    key: Key,
    entries: Vec<Entry>,
    /// the group the step must share, as known before the backlog's check
    anchor_group: Option<i64>,
    batch: SBatch,
    waiting_rows: FxHashSet<u32>,
    cands: Vec<u32>,
}

/// The check the pass waits on.
enum Await {
    /// the scan: (rid, key, real walk) per requested row, and whether it
    /// named an anchor
    Scan { rows: Vec<(u32, Key, Sym)>, anchored: bool },
    Backlog(KeyCtx),
}

/// `sched_begin`'s arguments: `PassRequest`.
pub(super) struct Begin {
    pub target: Option<(String, String)>,
    pub exclude_target: Option<(String, String)>,
    pub pre: u32,
    /// `capture_group_of`
    pub anchor: Option<u32>,
    pub waiting: Vec<u32>,
    pub failed: Vec<u32>,
    pub scan_exclude: Vec<u32>,
}

pub(super) struct Pass {
    target: Option<Key>,
    pre: u32,
    pre_walk: Option<Sym>,
    /// a merge takes only this walk's backlogged rows: `pre_walk` unless
    /// the chain mixes walks
    own_walk: Option<Sym>,
    /// a yield-away leaves this walk's backlogged rows parked
    skip_walk: Option<Sym>,
    /// the rid whose capture group the step must share
    anchor: Option<u32>,
    /// the scan's anchor group, when it named one
    scan_anchor: Option<i64>,
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
    keys: FxHashMap<Key, KeyCaps>,
    /// the configured batch builder's Rust backend
    builder: Option<Box<dyn BatchBuilder>>,
    leaders: FxHashSet<Sym>,
    /// (node, real walk) whose speculation chain mixes walks
    mixing: FxHashSet<(Sym, Sym)>,
    /// In `PythonSchedCore.backlog`'s order: a key taken for a step and parked
    /// again moves to the end, as a dict pop and re-insert does.
    backlog: Vec<(Key, SBatch)>,
    last_batch: FxHashMap<Key, u64>,
    batch_number: u64,
    pass: Option<Pass>,
}

impl SchedState {
    /// `SchedCore.drop_rids`: the rows go, then any empty chunk.
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

/// `CheckRequest`: `(rids, node_idx, nodes, limit, walks, anchor, by_anchor)`.
pub(super) type CheckOut = (
    Vec<u32>, Option<Vec<u32>>, Vec<String>, i64, Option<Vec<String>>, Option<u32>, bool,
);
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

impl GraphRuntime {
    fn sym_or_none(&self, s: &str) -> Sym {
        // a name no graph has matches nothing
        self.interner.get(s).unwrap_or(Sym::MAX)
    }

    fn sched_key_of(&self, node: Sym, walk: Sym) -> Key {
        (node, self.combined_walk_of.get(&(node, walk)).copied().unwrap_or(walk))
    }

    fn sched_cfg(&self, key: Key) -> PyResult<&KeyCaps> {
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
        mixing: Vec<(String, String)>,
        builder: &str,
    ) -> PyResult<()> {
        self.sched.builder = Some(batch_builder::by_name(builder).ok_or_else(|| {
            PyValueError::new_err(format!("no Rust backend for batch builder {builder:?}"))
        })?);
        let mut cfgs = FxHashMap::default();
        for (node, label, walks, caps, row_caps) in keys {
            let key = (self.interner.intern(&node), self.interner.intern(&label));
            let cfg = KeyCaps {
                walks: walks.iter().map(|w| self.interner.intern(w)).collect(),
                caps: caps.iter().map(|(w, c)| (self.interner.intern(w), *c)).collect(),
                row_caps: row_caps.iter().map(|(w, c)| (self.interner.intern(w), *c)).collect(),
            };
            cfgs.insert(key, cfg);
        }
        self.sched.keys = cfgs;
        self.sched.leaders = leaders.iter().map(|n| self.interner.intern(n)).collect();
        self.sched.mixing = mixing.iter()
            .map(|(n, w)| (self.interner.intern(n), self.interner.intern(w)))
            .collect();
        Ok(())
    }

    fn sched_builder(&self) -> PyResult<&dyn BatchBuilder> {
        self.sched.builder.as_deref()
            .ok_or_else(|| PyRuntimeError::new_err("the scheduler core was not configured"))
    }

    /// `speculation_mixes`.
    fn sched_mixes(&self, node: Sym, walk: Sym) -> bool {
        self.sched.mixing.contains(&(node, walk))
    }

    /// `walk_set_cap`: a mixed step's cap, the combined walk's own where it
    /// has one, else the smallest of the walks present.
    fn sched_walk_set_cap(&self, key: Key, walks: &[Sym]) -> PyResult<Option<u32>> {
        let cfg = self.sched_cfg(key)?;
        Ok(cfg.cap(key.1).or_else(|| min_cap(walks.iter().map(|&w| cfg.cap(w)))))
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

    /// The graph-ready rows `_scan_request` checks, in its order: `(rid, key,
    /// walk)` per row, leader nodes only.
    fn sched_scan_rows(&self, p: &Pass) -> Vec<(u32, Key, Sym)> {
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
            out.extend(rids.into_iter().map(|rid| (rid, key, walk)));
        }
        out
    }

    /// A check request for `rids`, row i of scheduling key `keys[i]`.
    fn check_out(
        &self, rids: Vec<u32>, keys: &[Key], limit: i64, anchor: Option<u32>, by_anchor: bool,
    ) -> CheckOut {
        let mut distinct: Vec<Key> = Vec::new();
        let idx: Vec<u32> = keys.iter().map(|k| {
            match distinct.iter().position(|d| d == k) {
                Some(i) => i as u32,
                None => {
                    distinct.push(*k);
                    (distinct.len() - 1) as u32
                }
            }
        }).collect();
        let node_idx = if distinct.len() > 1 { Some(idx) } else { None };
        let name = |s: Sym| self.interner.name(s).to_string();
        let nodes = distinct.iter().map(|k| name(k.0)).collect();
        let walks = distinct.iter().map(|k| name(k.1)).collect();
        (rids, node_idx, nodes, limit, Some(walks), anchor, by_anchor)
    }

    /// `_start_key`: a key's attempt up to its backlog's check; the rest is
    /// `sched_key_finish_ctx`.
    fn sched_key_start(
        &mut self, p: &Pass, key: Key, entries: Vec<Entry>, anchor_group: Option<i64>,
    ) -> PyResult<KeyStep> {
        if self.sched_room_left(key, p.pre_walk, p.pre)? == Some(0) {
            // the caller already holds a full batch
            return Ok(KeyStep::Done(None));
        }
        let pos = self.sched.backlog.iter().position(|(k, _)| *k == key);
        let Some(pos) = pos else {
            let built = self.sched_key_build(p, key, entries, None, FxHashSet::default(), anchor_group)?;
            return Ok(KeyStep::Done(built));
        };
        let (_, batch) = self.sched.backlog.remove(pos);
        let limit = match self.sched_scan_limit(key, p.pre)? {
            None => batch.rows.len() as i64,
            Some(l) => i64::from(l),
        };
        let own_walk = p.own_walk;
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
        let ctx = KeyCtx { key, entries, anchor_group, batch, waiting_rows, cands };
        if ctx.cands.is_empty() || limit <= 0 {
            return Ok(KeyStep::Done(self.sched_key_finish_ctx(p, ctx, &[], None)?));
        }
        // only rows of the step's capture group count toward the limit
        let req = self.check_out(ctx.cands.clone(), &[key], limit, p.anchor, true);
        Ok(KeyStep::Check(ctx, req))
    }

    /// `_finish_key`: the rest of a key's attempt: `_backlog_blocked`
    /// on the verdicts, then the build.
    fn sched_key_finish_ctx(
        &mut self, p: &Pass, ctx: KeyCtx, verdicts: &[u8], checked_anchor: Option<i64>,
    ) -> PyResult<Option<SBatch>> {
        let KeyCtx { key, entries, anchor_group, mut batch, waiting_rows, cands, .. } = ctx;
        // waiting, not ready, failed or off the step's capture group
        let mut blocked = waiting_rows;
        for (i, rid) in cands.iter().enumerate() {
            if verdicts.get(i).is_some_and(|v| *v != READY && *v != UNCHECKED) {
                blocked.insert(*rid);
            }
        }
        // a failed row is dropped rather than parked
        batch.rows.retain(|r| !(blocked.contains(&r.rid) && p.failed.contains(&r.rid)));
        blocked.retain(|r| !p.failed.contains(r));
        let backlogged = if batch.rows.is_empty() { None } else { Some(batch) };
        let anchor_group = checked_anchor.or(anchor_group);
        self.sched_key_build(p, key, entries, backlogged, blocked, anchor_group)
    }

    fn sched_key_build(
        &mut self, p: &Pass, key: Key, entries: Vec<Entry>,
        backlogged: Option<SBatch>, mut blocked: FxHashSet<u32>, anchor_group: Option<i64>,
    ) -> PyResult<Option<SBatch>> {
        let fresh = if entries.is_empty() { None } else { self.sched_assemble(key, &entries)? };
        if backlogged.is_none() && fresh.is_none() {
            return Ok(None);
        }
        if let Some(fresh) = &fresh {
            blocked.extend(Self::off_group(fresh, &entries, anchor_group));
        }
        self.sched_build_and_schedule(p, key, backlogged, fresh, blocked)
    }

    /// `_off_group`: the fresh rows outside the step's capture group -- the
    /// anchor's, or with none the first fresh row's.
    fn off_group(fresh: &SBatch, entries: &[Entry], anchor_group: Option<i64>) -> Vec<u32> {
        if entries.iter().all(|e| e.group == NO_GROUP) {
            return Vec::new();
        }
        let group_of: FxHashMap<u32, i64> = entries.iter().map(|e| (e.rid, e.group)).collect();
        let group = |rid: u32| group_of.get(&rid).copied().unwrap_or(NO_GROUP);
        if fresh.rows.iter().all(|r| group(r.rid) == NO_GROUP) {
            return Vec::new();
        }
        let wanted = anchor_group.unwrap_or_else(|| group(fresh.rows[0].rid));
        fresh.rows.iter().map(|r| r.rid).filter(|rid| group(*rid) != wanted).collect()
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

    /// `_build_and_schedule`: the configured builder composes one step out of
    /// the backlog and the fresh rows, and the core carries out its verdict.
    fn sched_build_and_schedule(
        &mut self, p: &Pass, key: Key,
        backlog: Option<SBatch>, fresh: Option<SBatch>, blocked: FxHashSet<u32>,
    ) -> PyResult<Option<SBatch>> {
        let mut expected: FxHashSet<u32> = FxHashSet::default();
        for b in backlog.iter().chain(fresh.iter()) {
            expected.extend(b.rows.iter().map(|r| r.rid));
        }
        let n_backlog = backlog.as_ref().map_or(0, |b| b.rows.len());
        let batch = match (backlog, fresh) {
            (None, None) => return Ok(None),
            (Some(b), None) => b,
            (None, Some(f)) => f,
            (Some(mut b), Some(f)) => {
                b.merge(f)?;
                b
            }
        };
        let rows: Vec<Row> = batch.rows.iter().map(|r| Row { rid: r.rid, walk: r.walk }).collect();
        let built = self.sched_builder()?.build(&BuildRequest {
            label: key.1, rows: &rows, n_backlog, combined: batch.combined, blocked: &blocked,
            caps: self.sched_cfg(key)?, pre: p.pre, pre_walk: p.pre_walk,
        });
        let part = |idx: &[usize]| {
            (!idx.is_empty()).then(|| batch.with_rows(idx.iter().map(|&i| batch.rows[i].clone()).collect()))
        };
        let (scheduled, parked, returned) = (part(&built.scheduled), part(&built.parked), part(&built.returned));

        // `_refuse_lost_rows`: every row handed in comes back out, once
        let mut got: FxHashSet<u32> = FxHashSet::default();
        for b in scheduled.iter().chain(parked.iter()).chain(returned.iter()) {
            got.extend(b.rows.iter().map(|r| r.rid));
        }
        let n_out = built.scheduled.len() + built.parked.len() + built.returned.len();
        if got != expected || n_out != rows.len() {
            return Err(PyRuntimeError::new_err(format!(
                "batch builder dropped rows for ({}, {})",
                self.interner.name(key.0), self.interner.name(key.1),
            )));
        }
        let node_name = self.interner.name(key.0).to_string();
        if let Some(parked) = parked {
            self.sched_park(key, parked)?;
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

    /// `_backlog`: park `batch` behind whatever already waits under `key`.
    fn sched_park(&mut self, key: Key, batch: SBatch) -> PyResult<()> {
        let rids: Vec<u32> = batch.rows.iter().map(|r| r.rid).collect();
        let wgs: Vec<u32> = batch.rows.iter().map(|r| r.wg).collect();
        match self.sched.backlog.iter_mut().find(|(k, _)| *k == key) {
            Some((_, existing)) => existing.merge(batch)?,
            None => self.sched.backlog.push((key, batch)),
        }
        // claimed like a running batch, so an input arriving while it waits
        // does not put its node back on the ready queue
        self.set_in_flight(self.interner.name(key.0).to_string(), rids, wgs, true)
    }

    /// Begin a pass: `get_next_batch` after the TP-follow FIFO.
    pub(super) fn sched_begin_impl(
        &mut self, b: Begin,
    ) -> PyResult<(bool, Option<CheckOut>, Option<BatchOut>)> {
        let Begin { target, exclude_target, pre, anchor, waiting, failed, scan_exclude } = b;
        let sym_pair = |rt: &Self, t: &(String, String)| (rt.sym_or_none(&t.0), rt.sym_or_none(&t.1));
        let key_strings = |rt: &Self, k: Key| (
            rt.interner.try_name(k.0).unwrap_or("").to_string(),
            rt.interner.try_name(k.1).unwrap_or("").to_string(),
        );
        let target_syms = target.as_ref().map(|t| sym_pair(self, t));
        let target_key = target_syms.map(|(n, w)| self.sched_key_of(n, w));
        let pre_walk = if pre > 0 { target_syms.map(|t| t.1) } else { None };
        // A chain merges and yields away within its own real walk, or its
        // whole combined walk when it mixes walks.
        let mixes = pre_walk.is_some() && target_syms.is_some_and(|(n, w)| self.sched_mixes(n, w));
        let own_walk = if mixes { None } else { pre_walk };
        let skip_walk = exclude_target.as_ref()
            .map(|t| sym_pair(self, t))
            .filter(|&(n, w)| !self.sched_mixes(n, w))
            .map(|t| t.1);
        // the runtime matches a target against real and combined walks alike
        let as_key = |rt: &Self, t: &(String, String)| {
            let (n, w) = sym_pair(rt, t);
            key_strings(rt, rt.sched_key_of(n, w))
        };
        let scan_target = match &target {
            Some(t) if own_walk.is_some() => Some(t.clone()),
            Some(t) => Some(as_key(self, t)),
            None => None,
        };
        let scan_exclude_target = match &exclude_target {
            Some(t) if skip_walk.is_some() => Some(t.clone()),
            Some(t) => Some(as_key(self, t)),
            None => None,
        };
        let mut p = Pass {
            target: target_key,
            pre,
            pre_walk,
            own_walk,
            skip_walk,
            anchor,
            scan_anchor: None,
            scan_target,
            scan_exclude: scan_exclude_target,
            waiting: waiting.into_iter().collect(),
            failed: failed.into_iter().collect(),
            scan_excl: scan_exclude.into_iter().collect(),
            keys: Vec::new(),
            ki: 0,
            ready: None,
            stage: Stage::BacklogOnly,
            awaiting: None,
        };
        if let Some(t) = p.target {
            if !self.sched.keys.contains_key(&t) {
                return Err(PyRuntimeError::new_err("target is not a configured scheduling key"));
            }
            if self.sched_room_left(t, p.pre_walk, p.pre)? == Some(0) {
                if mixes && !self.sched_cfg(t)?.walks.is_empty() {
                    // the chain never yields for its combined walk's other
                    // rows, so park them: `room_for_continuing` gives them
                    // first claim. Excluded in the scan, so the chain's own
                    // rows pay no check.
                    p.scan_target = Some(key_strings(self, t));
                    p.scan_exclude = Some(key_strings(self, (t.0, pre_walk.unwrap_or(t.1))));
                    p.stage = Stage::Park;
                    return self.sched_run(p, None, None, None);
                }
                // otherwise another walk's rows wait on their queues for the yield
                return Ok((true, None, None));
            }
        }
        p.keys = self.sched.backlog.iter()
            .map(|(k, _)| *k)
            .filter(|k| p.target.is_none_or(|t| t == *k))
            .collect();
        self.sched_run(p, None, None, None)
    }

    pub(super) fn sched_resume_impl(
        &mut self, verdicts: &[u8], groups: Option<Vec<i64>>, anchor_group: Option<i64>,
    ) -> PyResult<(bool, Option<CheckOut>, Option<BatchOut>)> {
        let p = self.sched.pass.take().ok_or_else(|| {
            PyRuntimeError::new_err("sched_resume: no pass is waiting for verdicts")
        })?;
        if groups.as_ref().is_some_and(|g| g.len() != verdicts.len()) {
            return Err(PyValueError::new_err("sched_resume: one group per requested row"));
        }
        self.sched_run(p, Some(verdicts), groups.as_deref(), anchor_group)
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

    /// `_run`: advance the pass until it waits on a check or is done.
    fn sched_run(
        &mut self, mut p: Pass, verdicts: Option<&[u8]>, groups: Option<&[i64]>,
        anchor_group: Option<i64>,
    ) -> PyResult<(bool, Option<CheckOut>, Option<BatchOut>)> {
        if let Some(aw) = p.awaiting.take() {
            let v = verdicts.ok_or_else(|| PyRuntimeError::new_err("sched_resume: verdicts missing"))?;
            match aw {
                Await::Scan { rows, anchored } => {
                    let rids: Vec<u32> = rows.iter().map(|r| r.0).collect();
                    self.sched_note_failed(&mut p, &rids, v)?;
                    p.ready = Some(Self::sched_ready_from(&rows, v, groups));
                    // the anchor's group only when the scan named one;
                    // otherwise the step anchors on its own first row
                    p.scan_anchor = if anchored { anchor_group } else { None };
                }
                Await::Backlog(ctx) => {
                    self.sched_note_failed(&mut p, &ctx.cands.clone(), v)?;
                    let scheduled = self.sched_key_finish_ctx(&p, ctx, v, anchor_group)?;
                    if let Some(done) = Self::sched_attempted(&mut p, scheduled) {
                        return Ok(self.sched_done(done));
                    }
                }
            }
        }
        loop {
            let (key, entries, anchor_group) = match p.stage {
                Stage::BacklogOnly => {
                    if p.ki == p.keys.len() {
                        p.stage = Stage::Pick;
                        continue;
                    }
                    // no fresh row could join a step its backlog fills,
                    // unless every backlogged row is blocked
                    if !self.sched_backlog_fills(&p, p.keys[p.ki])? {
                        p.stage = Stage::WithFresh;
                        continue;
                    }
                    (p.keys[p.ki], Vec::new(), None)
                }
                _ => {
                    // scanned once, when a step could first take fresh rows
                    if p.ready.is_none() {
                        if let Some((aw, req)) = self.sched_scan(&mut p) {
                            return Ok(self.sched_suspend(p, aw, req));
                        }
                    }
                    let key = match p.stage {
                        Stage::Park => {
                            self.sched_park_other_walks(&p)?;
                            return Ok(self.sched_done(None));
                        }
                        Stage::WithFresh => p.keys[p.ki],
                        _ => match self.sched_rr_pick(&p)? {
                            Some(key) => key,
                            None => return Ok(self.sched_done(None)),
                        },
                    };
                    (key, Self::ready_for(&p, key), p.scan_anchor)
                }
            };
            match self.sched_key_start(&p, key, entries, anchor_group)? {
                KeyStep::Check(ctx, req) => return Ok(self.sched_suspend(p, Await::Backlog(ctx), req)),
                KeyStep::Done(scheduled) => {
                    if let Some(done) = Self::sched_attempted(&mut p, scheduled) {
                        return Ok(self.sched_done(done));
                    }
                }
            }
        }
    }

    /// `_attempted`: where an attempt leaves the pass. `Some(step)` when it
    /// is done (the round-robin pick is the last attempt); None when it goes
    /// on to the key's fresh attempt or the next key.
    fn sched_attempted(p: &mut Pass, scheduled: Option<SBatch>) -> Option<Option<SBatch>> {
        if scheduled.is_some() || p.stage == Stage::Pick {
            return Some(scheduled);
        }
        if p.stage == Stage::BacklogOnly {
            p.stage = Stage::WithFresh;
        } else {
            p.ki += 1;
            p.stage = Stage::BacklogOnly;
        }
        None
    }

    /// `_scan_request`: the scan's check, or None (and an empty scan) when
    /// no row is graph-ready.
    fn sched_scan(&self, p: &mut Pass) -> Option<(Await, CheckOut)> {
        let rows = self.sched_scan_rows(p);
        if rows.is_empty() {
            p.ready = Some(Vec::new());
            return None;
        }
        let rids: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let keys: Vec<Key> = rows.iter().map(|r| r.1).collect();
        // the anchor's group, for a step the scan was narrowed to
        let anchor = if p.scan_target.is_some() { p.anchor } else { None };
        let req = self.check_out(rids, &keys, -1, anchor, false);
        Some((Await::Scan { rows, anchored: anchor.is_some() }, req))
    }

    /// `_rr_pick`: the least recently scheduled key with fresh rows, unless
    /// the pass already tried it as a backlogged key. A walk with no room (a
    /// cap of 0) takes nothing, so it stays the least recent; picked anyway,
    /// it would starve the walk that frees the room.
    fn sched_rr_pick(&self, p: &Pass) -> PyResult<Option<Key>> {
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
        Ok(best.map(|(key, _)| key).filter(|key| !p.keys.contains(key)))
    }

    /// `_park_other_walks`: backlog the scan's rows of the target's combined
    /// walk (the scan left out the chain's own walk).
    fn sched_park_other_walks(&mut self, p: &Pass) -> PyResult<()> {
        let key = p.target.expect("a park pass has a target");
        let entries = Self::ready_for(p, key);
        if !entries.is_empty() {
            if let Some(parked) = self.sched_assemble(key, &entries)? {
                self.sched_park(key, parked)?;
            }
        }
        Ok(())
    }

    fn ready_for(p: &Pass, key: Key) -> Vec<Entry> {
        p.ready.as_deref().unwrap_or(&[]).iter()
            .find(|(k, _)| *k == key)
            .map(|(_, e)| e.clone())
            .unwrap_or_default()
    }

    fn sched_ready_from(
        rows: &[(u32, Key, Sym)], verdicts: &[u8], groups: Option<&[i64]>,
    ) -> Vec<(Key, Vec<Entry>)> {
        let mut ready: Vec<(Key, Vec<Entry>)> = Vec::new();
        for (i, (&(rid, key, walk), &v)) in rows.iter().zip(verdicts).enumerate() {
            if v != READY {
                continue;
            }
            let group = groups.map_or(NO_GROUP, |g| g[i]);
            let entry = Entry { rid, walk, group };
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

    /// `room_for_continuing`: how many of a speculation chain's own rows fit
    /// beside `(node, walk)`'s backlog.
    pub(super) fn sched_room_for_continuing_impl(&self, node: &str, walk: &str) -> PyResult<Option<u32>> {
        let (n, w) = (self.sym_or_none(node), self.sym_or_none(walk));
        let key = self.sched_key_of(n, w);
        let mut cap = self.sched_cfg(key)?.cap(w);
        let backlog = self.sched.backlog.iter().find(|(k, _)| *k == key).map(|(_, b)| b);
        Ok(match backlog {
            Some(b) if b.combined && self.sched_mixes(n, w) => {
                // the chain's merge takes every backlogged row
                let mut walks: Vec<Sym> = vec![w];
                for r in &b.rows {
                    if !walks.contains(&r.walk) {
                        walks.push(r.walk);
                    }
                }
                if walks.len() > 1 {
                    cap = self.sched_walk_set_cap(key, &walks)?;
                }
                self.sched_builder()?.room_for_continuing(cap, b.rows.len())
            }
            Some(b) if b.combined => {
                // the chain's merge takes only its own walk's backlogged rows
                let Some(cap) = cap else { return Ok(None) };
                let own = b.rows.iter().filter(|r| r.walk == w).count() as u32;
                Some(cap.saturating_sub(own))
            }
            _ => self.sched_builder()?.room_for_continuing(cap, backlog.map_or(0, |b| b.rows.len())),
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
