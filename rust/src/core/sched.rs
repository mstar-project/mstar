//! The micro-scheduler  the Rust port of
//! `mstar/worker/micro_scheduler.py`'s decision logic.
//!
//! The seam is decide-vs-mutate: the caller hands a SNAPSHOT of ready work
//! (one [`ReadyEntry`] per ready (node, request) pair, with the engine-level
//! readiness and priority already evaluated — those live with the engines on
//! the Python side), plus the current monotonic time; the scheduler returns
//! which (node, walk, requests) to batch. Popping the ready queues and
//! executing stay with the worker. Time is always passed in, never read —
//! decisions are deterministic and testable.
//!
//! Ported semantics, asserted by `test/rust/test_sched_parity.py`:
//! round-robin by least-recent (node, walk) batch number; priority mode
//! (lowest engine priority, then the walk with the most requests); OOM
//! hold-with-backoff; deferred removes; leader-node filtering; target /
//! exclude filters; max-batch truncation; TP-follower batches first, with
//! the consecutive-batch cap yielding to other ready work (fairness).
//!
//! On exact ties (equal round-robin recency, equal walk counts) the Python
//! implementation's choice follows set/dict iteration order; here the first
//! entry in snapshot order wins. Callers that need reproducibility across
//! the two must not depend on tie order (mstar's does not).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// One ready (node, request) pair, with engine state pre-evaluated.
#[derive(Debug, Clone)]
pub struct ReadyEntry {
    pub node: String,
    pub walk: String,
    pub request_id: String,
    pub worker_graph_id: String,
    /// `engine.check_ready(node, rid, fwd_info)` — e.g. KV cache read in.
    pub engine_ready: bool,
    /// Engine priority (lower schedules first in priority mode; mstar:
    /// KV_CACHE = 0, STATELESS = 2, unknown = 99).
    pub priority: u32,
    /// Whether this node may INITIATE a batch on this rank
    /// (mstar's `parallel_leader_nodes`; follower ranks replay instead).
    pub leader: bool,
}

/// The scheduling decision. The caller pops `request_ids` (in order) for
/// `node` from their worker-graph queues and executes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledBatch {
    pub node: String,
    pub walk: String,
    pub request_ids: Vec<String>,
    pub worker_graph_ids: Vec<String>,
    /// True when this batch replays a TP leader's decision.
    pub tp_follow: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingType {
    Priority,
    RoundRobin,
}

/// Optional constraints for one `get_next_batch` call.
#[derive(Debug, Default, Clone)]
pub struct BatchFilter {
    pub max_batch_size: Option<usize>,
    pub target_node: Option<String>,
    pub target_walk: Option<String>,
    /// Skip this (node, walk) pair entirely.
    pub exclude_target: Option<(String, String)>,
}

#[derive(Debug, Clone)]
struct TpFollow {
    node: String,
    walk: String,
    request_ids: Vec<String>,
}

#[derive(Debug)]
pub struct MicroScheduler {
    sched_type: SchedulingType,
    batch_number: u64,
    /// (node, walk) -> batch number of its last scheduled batch (round-robin).
    last_batch_num: BTreeMap<(String, String), u64>,
    /// request -> monotonic ms until which it is held (OOM backoff).
    held_until: BTreeMap<String, u64>,
    /// Requests with a deferred remove: stop initiating new work.
    pending_removes: BTreeSet<String>,
    /// Leader decisions to replay, in arrival order.
    tp_pending: VecDeque<TpFollow>,
    consec_tp_follower_batches: u32,
    max_consec_tp_follower_batches: u32,
    /// mstar's `HOLD_BACKOFF_SECONDS` (50 ms), in ms.
    hold_backoff_ms: u64,
}

impl MicroScheduler {
    pub fn new(sched_type: SchedulingType, max_consec_tp_follower_batches: u32) -> Self {
        Self {
            sched_type,
            batch_number: 0,
            last_batch_num: BTreeMap::new(),
            held_until: BTreeMap::new(),
            pending_removes: BTreeSet::new(),
            tp_pending: VecDeque::new(),
            consec_tp_follower_batches: 0,
            max_consec_tp_follower_batches,
            hold_backoff_ms: 50,
        }
    }

    /// OOM backoff: hold these requests for `hold_backoff_ms` from `now_ms`.
    pub fn hold_requests(&mut self, request_ids: &[String], now_ms: u64) {
        for rid in request_ids {
            self.held_until
                .insert(rid.clone(), now_ms + self.hold_backoff_ms);
        }
    }

    pub fn add_pending_remove(&mut self, request_id: &str) {
        self.pending_removes.insert(request_id.to_string());
    }

    pub fn clear_pending_remove(&mut self, request_id: &str) {
        self.pending_removes.remove(request_id);
    }

    /// A TP leader's batch decision to replay on this follower rank.
    pub fn register_tp_follow(&mut self, node: String, walk: String, request_ids: Vec<String>) {
        self.tp_pending.push_back(TpFollow {
            node,
            walk,
            request_ids,
        });
    }

    /// Admissible = not held, not pending-remove, engine-ready.
    fn admissible(&self, e: &ReadyEntry, now_ms: u64) -> bool {
        if !e.engine_ready || self.pending_removes.contains(&e.request_id) {
            return false;
        }
        match self.held_until.get(&e.request_id) {
            Some(&t) => t <= now_ms,
            None => true,
        }
    }

    /// Any admissible ready work other than `exclude`? (mstar's
    /// `has_ready_excluding` — the spec-chain fairness peek. Note: like
    /// mstar's, this does NOT apply the leader filter.)
    pub fn has_ready_excluding(
        &self,
        ready: &[ReadyEntry],
        exclude: Option<(&str, &str)>,
        now_ms: u64,
    ) -> bool {
        ready.iter().any(|e| {
            self.admissible(e, now_ms)
                && exclude != Some((e.node.as_str(), e.walk.as_str()))
        })
    }

    fn try_schedule_tp_follow(
        &mut self,
        ready: &[ReadyEntry],
        now_ms: u64,
    ) -> Option<ScheduledBatch> {
        let head = self.tp_pending.front()?.clone();
        // Fairness: after N consecutive follower batches, yield if anything
        // else is ready (identical to the leader's spec-chain cap).
        if self.consec_tp_follower_batches >= self.max_consec_tp_follower_batches
            && self.has_ready_excluding(
                ready,
                Some((head.node.as_str(), head.walk.as_str())),
                now_ms,
            )
        {
            return None;
        }
        // Every rid of the leader's decision must be ready here (graph-level
        // AND engine-level); otherwise wait — replay order is fixed.
        let mut worker_graph_ids = Vec::with_capacity(head.request_ids.len());
        for rid in &head.request_ids {
            let entry = ready.iter().find(|e| {
                e.node == head.node && e.request_id == *rid
            })?;
            if !entry.engine_ready {
                return None;
            }
            worker_graph_ids.push(entry.worker_graph_id.clone());
        }
        self.batch_number += 1;
        self.last_batch_num
            .insert((head.node.clone(), head.walk.clone()), self.batch_number);
        self.tp_pending.pop_front();
        Some(ScheduledBatch {
            node: head.node,
            walk: head.walk,
            request_ids: head.request_ids,
            worker_graph_ids,
            tp_follow: true,
        })
    }

    /// The scheduling decision (mstar's `get_next_batch`).
    pub fn get_next_batch(
        &mut self,
        ready: &[ReadyEntry],
        filter: &BatchFilter,
        now_ms: u64,
    ) -> Option<ScheduledBatch> {
        // Expire stale holds (mstar does this each call).
        self.held_until.retain(|_, &mut t| t > now_ms);

        match self.try_schedule_tp_follow(ready, now_ms) {
            Some(batch) => {
                self.consec_tp_follower_batches += 1;
                return Some(batch);
            }
            None => self.consec_tp_follower_batches = 0,
        }

        // Group admissible leader entries by node, preserving snapshot order.
        let mut by_node: Vec<(String, Vec<&ReadyEntry>)> = Vec::new();
        for e in ready {
            if !e.leader || !self.admissible(e, now_ms) {
                continue;
            }
            if let Some(t) = &filter.target_node {
                if e.node != *t {
                    continue;
                }
            }
            if let Some(t) = &filter.target_walk {
                if e.walk != *t {
                    continue;
                }
            }
            if let Some((xn, xw)) = &filter.exclude_target {
                if e.node == *xn && e.walk == *xw {
                    continue;
                }
            }
            match by_node.iter_mut().find(|(n, _)| *n == e.node) {
                Some((_, v)) => v.push(e),
                None => by_node.push((e.node.clone(), vec![e])),
            }
        }
        if by_node.is_empty() {
            return None;
        }

        let (node, walk) = match self.sched_type {
            SchedulingType::Priority => Self::select_priority(&by_node)?,
            SchedulingType::RoundRobin => self.select_round_robin(&by_node)?,
        };

        let mut request_ids = Vec::new();
        let mut worker_graph_ids = Vec::new();
        for (n, entries) in &by_node {
            if *n != node {
                continue;
            }
            for e in entries {
                if e.walk == walk {
                    request_ids.push(e.request_id.clone());
                    worker_graph_ids.push(e.worker_graph_id.clone());
                }
            }
        }
        if let Some(cap) = filter.max_batch_size {
            request_ids.truncate(cap);
            worker_graph_ids.truncate(cap);
        }
        if request_ids.is_empty() {
            return None;
        }

        self.batch_number += 1;
        self.last_batch_num
            .insert((node.clone(), walk.clone()), self.batch_number);
        Some(ScheduledBatch {
            node,
            walk,
            request_ids,
            worker_graph_ids,
            tp_follow: false,
        })
    }

    /// Lowest engine priority wins; within it, the walk with the most
    /// requests (mstar maximizes batch size; the rest wait a cycle).
    fn select_priority(by_node: &[(String, Vec<&ReadyEntry>)]) -> Option<(String, String)> {
        let (node, entries) = by_node
            .iter()
            .min_by_key(|(_, entries)| entries.first().map(|e| e.priority).unwrap_or(99))?;
        let mut walk_counts: Vec<(String, usize)> = Vec::new();
        for e in entries {
            match walk_counts.iter_mut().find(|(w, _)| *w == e.walk) {
                Some((_, c)) => *c += 1,
                None => walk_counts.push((e.walk.clone(), 1)),
            }
        }
        let (walk, _) = walk_counts.into_iter().max_by_key(|&(_, c)| c)?;
        Some((node.clone(), walk))
    }

    /// Least-recently-batched (node, walk) wins (mstar's round-robin).
    fn select_round_robin(
        &self,
        by_node: &[(String, Vec<&ReadyEntry>)],
    ) -> Option<(String, String)> {
        let mut best: Option<(u64, String, String)> = None;
        for (node, entries) in by_node {
            for e in entries {
                let step = self
                    .last_batch_num
                    .get(&(node.clone(), e.walk.clone()))
                    .copied()
                    .unwrap_or(0);
                if best.as_ref().map(|(s, _, _)| step < *s).unwrap_or(true) {
                    best = Some((step, node.clone(), e.walk.clone()));
                }
            }
        }
        best.map(|(_, n, w)| (n, w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(node: &str, walk: &str, rid: &str) -> ReadyEntry {
        ReadyEntry {
            node: node.into(),
            walk: walk.into(),
            request_id: rid.into(),
            worker_graph_id: "wg0".into(),
            engine_ready: true,
            priority: 0,
            leader: true,
        }
    }

    #[test]
    fn round_robin_rotates_across_node_walks() {
        let mut s = MicroScheduler::new(SchedulingType::RoundRobin, 1);
        let ready = vec![entry("A", "w", "r1"), entry("B", "w", "r2")];
        let f = BatchFilter::default();
        let b1 = s.get_next_batch(&ready, &f, 0).unwrap();
        let b2 = s.get_next_batch(&ready, &f, 0).unwrap();
        assert_ne!(b1.node, b2.node);
    }

    #[test]
    fn priority_prefers_low_then_biggest_walk() {
        let mut s = MicroScheduler::new(SchedulingType::Priority, 1);
        let mut kv1 = entry("KV", "decode", "r1");
        kv1.priority = 0;
        let mut kv2 = entry("KV", "prefill", "r2");
        kv2.priority = 0;
        let mut kv3 = entry("KV", "prefill", "r3");
        kv3.priority = 0;
        let mut st = entry("VOC", "decode", "r4");
        st.priority = 2;
        let b = s
            .get_next_batch(&[kv1, kv2, kv3, st], &BatchFilter::default(), 0)
            .unwrap();
        assert_eq!((b.node.as_str(), b.walk.as_str()), ("KV", "prefill"));
        assert_eq!(b.request_ids, vec!["r2", "r3"]);
    }

    #[test]
    fn holds_expire_after_backoff() {
        let mut s = MicroScheduler::new(SchedulingType::RoundRobin, 1);
        s.hold_requests(&["r1".to_string()], 1000);
        let ready = vec![entry("A", "w", "r1")];
        assert!(s.get_next_batch(&ready, &BatchFilter::default(), 1010).is_none());
        assert!(s.get_next_batch(&ready, &BatchFilter::default(), 1051).is_some());
    }

    #[test]
    fn tp_follow_first_with_fairness_cap() {
        let mut s = MicroScheduler::new(SchedulingType::RoundRobin, 1);
        s.register_tp_follow("A".into(), "w".into(), vec!["r1".into()]);
        s.register_tp_follow("A".into(), "w".into(), vec!["r1".into()]);
        let ready = vec![entry("A", "w", "r1"), entry("B", "w", "r2")];
        let f = BatchFilter::default();
        let b1 = s.get_next_batch(&ready, &f, 0).unwrap();
        assert!(b1.tp_follow);
        // consec cap = 1 and B is ready: the second follow yields to B.
        let b2 = s.get_next_batch(&ready, &f, 0).unwrap();
        assert!(!b2.tp_follow);
        assert_eq!(b2.node, "B");
        // then the queued follow goes through.
        let b3 = s.get_next_batch(&ready, &f, 0).unwrap();
        assert!(b3.tp_follow);
    }

    #[test]
    fn tp_follow_waits_for_all_rids_ready() {
        let mut s = MicroScheduler::new(SchedulingType::RoundRobin, 1);
        s.register_tp_follow("A".into(), "w".into(), vec!["r1".into(), "r2".into()]);
        // On a follower rank the followed node is not leader-initiable
        // (mstar's parallel_leader_nodes filter) — model that here.
        let follower = |rid: &str| {
            let mut e = entry("A", "w", rid);
            e.leader = false;
            e
        };
        let only_r1 = vec![follower("r1")];
        assert!(s.get_next_batch(&only_r1, &BatchFilter::default(), 0).is_none());
        let both = vec![follower("r1"), follower("r2")];
        let b = s.get_next_batch(&both, &BatchFilter::default(), 0).unwrap();
        assert_eq!(b.request_ids, vec!["r1", "r2"]);
    }

    #[test]
    fn filters_and_truncation() {
        let mut s = MicroScheduler::new(SchedulingType::RoundRobin, 1);
        let ready = vec![
            entry("A", "w", "r1"),
            entry("A", "w", "r2"),
            entry("A", "w", "r3"),
        ];
        let f = BatchFilter {
            max_batch_size: Some(2),
            ..Default::default()
        };
        let b = s.get_next_batch(&ready, &f, 0).unwrap();
        assert_eq!(b.request_ids.len(), 2);

        let f = BatchFilter {
            exclude_target: Some(("A".into(), "w".into())),
            ..Default::default()
        };
        assert!(s.get_next_batch(&ready, &f, 0).is_none());
    }
}
