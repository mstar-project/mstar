//! A step's postprocess -- its stops, its route, its send -- run here,
//! stopping for the worker where a phase needs Python.
//!
//! The GIL stays held where the separate calls held it (the stop check, a
//! stored step's route and send), with peer frames sent try-first
//! (`send_frames`). Only an inline step's route and send release it, as the
//! one call they replaced did: nothing in them touches Python.
//!
//! Python's `PhasedPostprocess` (`mstar/graph/runtime/base.py`) is the
//! reference: the phases below follow it, and
//! `test/rust/test_postprocess.py` drives both over the same steps.

use super::*;

/// What a step waits on (Python's `PostprocessNeed`).
pub(super) const NEED_DONE: u8 = 0;
pub(super) const NEED_STOPS: u8 = 1;
pub(super) const NEED_STORE: u8 = 2;
pub(super) const NEED_SEND: u8 = 3;

/// A step's token column, read in place: `rows` integers of `itemsize`
/// bytes, `stride` elements apart, row i belonging to the caller's
/// `row_rids[i]`. The caller holds the tensor for the step.
pub(super) struct TokenColumn {
    ptr: usize,
    rows: usize,
    stride: usize,
    itemsize: usize,
}

impl TokenColumn {
    /// None for a null pointer or a dtype other than int32 / int64.
    pub(super) fn new(ptr: usize, rows: usize, stride: usize, dtype: &str) -> Option<Self> {
        let itemsize = match dtype {
            "int64" => 8,
            "int32" => 4,
            _ => return None,
        };
        (ptr != 0).then_some(TokenColumn { ptr, rows, stride, itemsize })
    }

    /// Row `row`'s value; the callers check `row < self.rows`.
    fn read(&self, row: usize) -> i64 {
        debug_assert!(row < self.rows);
        // SAFETY: row < rows, which the callers check against the column the
        // caller holds for the step.
        unsafe {
            let at = (self.ptr as *const u8).add(row * self.stride * self.itemsize);
            if self.itemsize == 8 {
                std::ptr::read_unaligned(at as *const i64)
            } else {
                i64::from(std::ptr::read_unaligned(at as *const i32))
            }
        }
    }

    /// The column in `rids` order; None when a rid has no row.
    fn values_for(&self, rids: &[u32], row_rids: &[u32]) -> Option<Vec<i64>> {
        if row_rids.len() > self.rows {
            return None;
        }
        if row_rids.len() >= rids.len() && row_rids[..rids.len()] == rids[..] {
            return Some((0..rids.len()).map(|i| self.read(i)).collect());
        }
        let row_of: FxHashMap<u32, usize> =
            row_rids.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        rids.iter().map(|r| row_of.get(r).map(|&i| self.read(i))).collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Stops,
    Route,
}

/// The stored outputs a STORE break resumes with (`RouteTensors`).
pub(super) struct Stored {
    pub tensors: Vec<u64>,
    pub num_tensors: Vec<usize>,
    pub inline_signal: Option<String>,
    pub inline_values: Vec<i64>,
    pub logprobs: Vec<Option<f64>>,
}

/// A step's postprocess (`PostprocessRequest`), and how far it got.
pub(super) struct Postprocess {
    pub partition: String,
    pub node_name: String,
    pub graph_walk: String,
    pub output_signals: Vec<String>,
    pub rids: Vec<u32>,
    pub walks: Option<Vec<String>>,
    pub rid_walk_idx: Option<Vec<u32>>,
    /// stops known before the check: stream-terminated loops
    pub known_stops: Vec<(u32, Vec<String>)>,
    pub runtime_stops: bool,
    pub tokens: Option<TokenColumn>,
    pub row_rids: Vec<u32>,
    pub inline: Option<(String, TokenColumn)>,
    pub stream_done: Vec<u32>,
    pub python_send: bool,
    phase: Phase,
    stops: Option<Vec<(u32, Vec<String>)>>,
    stopped: Vec<u32>,
    stored: Option<Stored>,
    completion_id: u64,
}

impl Postprocess {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        partition: String, node_name: String, graph_walk: String, output_signals: Vec<String>,
        rids: Vec<u32>, walks: Option<Vec<String>>, rid_walk_idx: Option<Vec<u32>>,
        known_stops: Vec<(u32, Vec<String>)>, runtime_stops: bool,
        tokens: Option<TokenColumn>, row_rids: Vec<u32>, inline: Option<(String, TokenColumn)>,
        stream_done: Vec<u32>, python_send: bool,
    ) -> Self {
        Postprocess {
            partition, node_name, graph_walk, output_signals, rids, walks, rid_walk_idx,
            known_stops, runtime_stops, tokens, row_rids, inline, stream_done, python_send,
            phase: Phase::Stops, stops: None, stopped: Vec::new(), stored: None, completion_id: 0,
        }
    }

    /// `_keep_rows`: the rows the step routes, less `dropped`.
    fn keep_rows(&mut self, dropped: &FxHashSet<u32>) {
        if dropped.is_empty() {
            return;
        }
        let keep: Vec<usize> = (0..self.rids.len()).filter(|&i| !dropped.contains(&self.rids[i])).collect();
        self.rids = keep.iter().map(|&i| self.rids[i]).collect();
        if let Some(idx) = &self.rid_walk_idx {
            self.rid_walk_idx = Some(keep.iter().map(|&i| idx[i]).collect());
        }
        if let Some(stops) = &mut self.stops {
            stops.retain(|(rid, _)| !dropped.contains(rid));
        }
        // a dropped row's streams do not end here
        self.stream_done.retain(|rid| !dropped.contains(rid));
    }

    /// Whether the send needs what only the worker has (`_route_needs_python`).
    fn send_needs_python(&self, route: &RouteOut) -> bool {
        self.python_send
            || !self.stopped.is_empty()
            || !route.register_uuids.is_empty()
            || !route.new_token_output_idxs.is_empty()
            || !route.local_streaming_signals.is_empty()
            || !route.rids_needing_request_info.is_empty()
            || !route.freed_input_uuids.is_empty()
    }

    fn route_arg(&self, s: Stored) -> RouteArg {
        RouteArg {
            partition: self.partition.clone(),
            graph_walk: self.graph_walk.clone(),
            node_name: self.node_name.clone(),
            output_signals: self.output_signals.clone(),
            rids: self.rids.clone(),
            tensors: s.tensors,
            num_tensors: s.num_tensors,
            inline_signal: s.inline_signal,
            inline_values: s.inline_values,
            logprobs: s.logprobs,
            walks: if self.rid_walk_idx.is_some() { self.walks.clone() } else { None },
            rid_walk_idx: self.rid_walk_idx.clone(),
        }
    }

    fn done(&self, completed: Vec<u32>) -> PostprocessOut {
        PostprocessOut { completed_rids: completed, ..self.out(NEED_DONE) }
    }

    fn out(&self, need: u8) -> PostprocessOut {
        PostprocessOut {
            need,
            rids: if need == NEED_DONE { Vec::new() } else { self.rids.clone() },
            rid_walk_idx: if need == NEED_DONE { None } else { self.rid_walk_idx.clone() },
            stopped_rids: self.stopped.clone(),
            ..Default::default()
        }
    }
}

/// `_merged_stops`: the loops each rid stops, in first-seen order.
fn merged_stops(parts: [&[(u32, Vec<String>)]; 2]) -> Vec<(u32, Vec<String>)> {
    let mut out: Vec<(u32, Vec<String>)> = Vec::new();
    let mut at: FxHashMap<u32, usize> = FxHashMap::default();
    for part in parts {
        for (rid, loops) in part {
            let i = *at.entry(*rid).or_insert_with(|| {
                out.push((*rid, Vec::new()));
                out.len() - 1
            });
            for l in loops {
                if !out[i].1.contains(l) {
                    out[i].1.push(l.clone());
                }
            }
        }
    }
    out
}

/// `PostprocessStep`, as columns.
#[pyclass]
#[derive(Default)]
pub struct PostprocessOut {
    #[pyo3(get)] pub need: u8,
    /// The rows the step routes, less the dropped ones (STORE, SEND).
    #[pyo3(get)] pub rids: Vec<u32>,
    #[pyo3(get)] pub rid_walk_idx: Option<Vec<u32>>,
    #[pyo3(get)] pub stopped_rids: Vec<u32>,
    #[pyo3(get)] pub completed_rids: Vec<u32>,
    /// Wall time this crossing ran with the GIL released (an inline route
    /// and send).
    #[pyo3(get)] pub nogil_ns: u64,
    pub route: Option<RouteOut>,
}

#[pymethods]
impl PostprocessOut {
    /// The route a SEND break's payloads finish; taken, not cloned.
    fn take_route(&mut self) -> Option<RouteOut> {
        self.route.take()
    }
}

impl GraphRuntime {
    pub(super) fn sym_or_none(&self, s: &str) -> Sym {
        // a name no graph has matches nothing
        self.interner.get(s).unwrap_or(Sym::MAX)
    }

    /// `check_stops_inline`'s check: `(stop_rids, stop_rules, missing)`.
    #[allow(clippy::type_complexity)]
    pub(super) fn stops_over(
        &self, partition: &str, node_name: &str, graph_walk: &str, rids: &[u32],
        tokens: Option<&TokenColumn>, row_rids: &[u32],
    ) -> PyResult<(Vec<u32>, Vec<u32>, Vec<u32>)> {
        let (Some(node), Some(walk)) = (self.interner.get(node_name), self.interner.get(graph_walk))
        else {
            return Err(PyKeyError::new_err("check_stops: no rules for this (node, walk)"));
        };
        let Some(rules) = self.stop_rules.get(&(node, walk)) else {
            return Err(PyKeyError::new_err("check_stops: no rules for this (node, walk)"));
        };
        let (mut stop_rids, mut stop_rules, mut missing) = (Vec::new(), Vec::new(), Vec::new());
        if rules.is_empty() {
            return Ok((stop_rids, stop_rules, missing));
        }
        let tokens = if rules.iter().any(|r| r.ar) {
            match tokens {
                Some(t) if row_rids.len() <= t.rows => Some(t),
                _ => return Err(PyValueError::new_err(
                    "check_stops: no int32 / int64 token column covering the rows",
                )),
            }
        } else {
            None
        };
        let in_order = row_rids.len() >= rids.len() && row_rids[..rids.len()] == rids[..];
        let row_of: FxHashMap<u32, usize> = if in_order || tokens.is_none() {
            FxHashMap::default()
        } else {
            row_rids.iter().enumerate().map(|(i, &r)| (r, i)).collect()
        };
        let p = self.interner.get(partition);
        for (i, &rid) in rids.iter().enumerate() {
            let token = match tokens {
                None => 0,
                Some(t) if in_order => t.read(i),
                Some(t) => match row_of.get(&rid) {
                    Some(&row) => t.read(row),
                    None => continue,
                },
            };
            let Some(info) = self.info(rid) else { continue };
            let Some((_, limits)) = info.stop_limits.iter().find(|(k, _)| *k == (node, walk)) else {
                missing.push(rid);
                continue;
            };
            for (k, (rule, &(max_iters, ignore_eos))) in rules.iter().zip(limits).enumerate() {
                let count = p.map_or(0, |p| self.loop_count(rid, p, rule.lp));
                if count + 1 >= max_iters || (rule.ar && !ignore_eos && rule.eos.contains(&token)) {
                    stop_rids.push(rid);
                    stop_rules.push(k as u32);
                }
            }
        }
        Ok((stop_rids, stop_rules, missing))
    }

    /// `stop_loops_batched` without the send: the rids whose loops stopped,
    /// and the STOP_LOOPS frames for their peers.
    #[allow(clippy::type_complexity)]
    pub(super) fn stop_loops_frames(
        &mut self, partition: &str, graph_walk: &str, last_node_run: &str,
        rids: Vec<u32>, loop_names: Vec<Vec<String>>,
    ) -> PyResult<(Vec<u32>, Vec<(String, Vec<u8>)>)> {
        if rids.len() != loop_names.len() {
            return Err(PyValueError::new_err(
                "stop_loops_batched: rids and loop_names must be the same length",
            ));
        }
        let mut frames: Vec<(String, Vec<u8>)> = Vec::new();
        let Some(walk) = self.interner.get(graph_walk) else {
            return Ok((Vec::new(), frames));
        };
        let mut stopped_rids = Vec::new();
        for (rid, names) in rids.into_iter().zip(loop_names) {
            let wanted: Vec<String> = names
                .into_iter()
                .filter(|n| self.check_dyn_loop(rid, partition, n))
                .collect();
            if wanted.is_empty() {
                continue;
            }
            self.stop_loops_for_rid(rid, partition, &wanted, Some(last_node_run));
            for name in &wanted {
                if let Some(l) = self.interner.get(name) {
                    self.pending_loop_stops.insert((rid, walk, l));
                }
            }
            let mut per_worker: FxHashMap<Sym, Vec<String>> = FxHashMap::default();
            for name in &wanted {
                for w in self.dyn_loop_workers(rid, partition, name) {
                    per_worker.entry(w).or_default().push(name.clone());
                }
            }
            // Never to ourselves: this rank originated the stop and has
            // already applied it. A self-send would land in apply_peer_loop_stops
            // and stop the loops a second time.
            let me = self.shard.me;
            let request_id = self.rid_name(rid)?;
            // Snapshotted BEFORE the sends: stop_loops_for_rid above already
            // recorded this stop, and every peer must see the same context.
            let stop_times = self.loop_stop_times_of(rid);
            for (worker, names) in per_worker {
                if worker == me {
                    continue;
                }
                let bytes = frames::StopLoops {
                    request_id: &request_id,
                    partition_name: partition,
                    loop_names: &names,
                    loop_stop_times: stop_times.clone(),
                }
                .encode(&self.interner);
                frames.push((self.interner.name(worker).to_string(), bytes));
            }
            stopped_rids.push(rid);
        }
        Ok((stopped_rids, frames))
    }

    /// The loop name of each of (node, walk)'s stop rules, by index.
    fn stop_rule_loops(&self, node_name: &str, graph_walk: &str) -> Vec<String> {
        let key = (self.sym_or_none(node_name), self.sym_or_none(graph_walk));
        self.stop_rules.get(&key).map_or_else(Vec::new, |rules| {
            rules.iter().map(|r| self.interner.name(r.lp).to_string()).collect()
        })
    }

    /// `_stop_step_loops`: stop the step's loops, per walk, and send the
    /// peers' STOP_LOOPS; the rids whose loops stopped.
    fn pp_stop_loops(&mut self, py: Python<'_>, pp: &Postprocess) -> PyResult<Vec<u32>> {
        let stops = pp.stops.as_deref().unwrap_or(&[]);
        if stops.is_empty() {
            return Ok(Vec::new());
        }
        let walk_of: FxHashMap<u32, &str> = match (&pp.rid_walk_idx, &pp.walks) {
            (Some(idx), Some(walks)) => {
                pp.rids.iter().zip(idx).map(|(&r, &w)| (r, walks[w as usize].as_str())).collect()
            }
            _ => pp.rids.iter().map(|&r| (r, pp.graph_walk.as_str())).collect(),
        };
        let mut by_walk: Vec<(&str, Vec<u32>, Vec<Vec<String>>)> = Vec::new();
        for (rid, loops) in stops {
            let Some(&walk) = walk_of.get(rid) else { continue };
            match by_walk.iter_mut().find(|(w, _, _)| *w == walk) {
                Some((_, rids, names)) => {
                    rids.push(*rid);
                    names.push(loops.clone());
                }
                None => by_walk.push((walk, vec![*rid], vec![loops.clone()])),
            }
        }
        let mut stopped = Vec::new();
        for (walk, rids, names) in by_walk {
            let walk = walk.to_string();
            let (rids, frames) = self.stop_loops_frames(&pp.partition, &walk, &pp.node_name, rids, names)?;
            self.send_frames(py, frames)?;
            stopped.extend(rids);
        }
        Ok(stopped)
    }

    /// `_run_postprocess`, from wherever `pp` stands: DONE, or a break with
    /// the step parked in `self.postprocess`.
    pub(super) fn pp_run(&mut self, py: Python<'_>, mut pp: Postprocess) -> PyResult<PostprocessOut> {
        if pp.phase == Phase::Stops {
            if pp.stops.is_none() && pp.runtime_stops {
                // a check that fails, or misses a row's limits, leaves the
                // decision to check_stop
                let checked = self.stops_over(
                    &pp.partition, &pp.node_name, &pp.graph_walk, &pp.rids,
                    pp.tokens.as_ref(), &pp.row_rids,
                );
                if let Ok((stop_rids, stop_rules, missing)) = checked {
                    if missing.is_empty() {
                        let loops = self.stop_rule_loops(&pp.node_name, &pp.graph_walk);
                        let found: Vec<(u32, Vec<String>)> = stop_rids.iter().zip(&stop_rules)
                            .map(|(&r, &k)| (r, vec![loops[k as usize].clone()]))
                            .collect();
                        pp.stops = Some(merged_stops([&pp.known_stops, &found]));
                    }
                }
            }
            if pp.stops.is_none() {
                return Ok(self.pp_park(pp, NEED_STOPS));
            }
            pp.stopped = self.pp_stop_loops(py, &pp)?;
            pp.phase = Phase::Route;
        }
        if pp.rids.is_empty() {
            return Ok(pp.out(NEED_DONE)); // every row was dropped
        }
        if let Some(s) = pp.stored.take() {
            let route = self.complete_and_route_batch(pp.route_arg(s))?;
            pp.completion_id = route.completion_id;
            if pp.send_needs_python(&route) {
                return Ok(self.pp_park_route(pp, route));
            }
            let (completed, frames) = self.pp_send_frames(&pp, SendArgs::default())?;
            self.send_frames(py, frames)?;
            return Ok(pp.done(completed));
        }
        let values = pp.inline.as_ref().and_then(|(_, col)| col.values_for(&pp.rids, &pp.row_rids));
        let Some(values) = values else {
            return Ok(self.pp_park(pp, NEED_STORE));
        };
        // Inline: nothing stored, so nothing here touches Python.
        let n = pp.rids.len() * pp.output_signals.len();
        let signal = pp.inline.as_ref().map(|(s, _)| s.clone());
        let t = std::time::Instant::now();
        let sent = py.detach(|| -> PyResult<Result<Vec<u32>, RouteOut>> {
            let route = self.complete_and_route_batch(pp.route_arg(Stored {
                tensors: Vec::new(), num_tensors: vec![0; n], inline_signal: signal,
                inline_values: values, logprobs: Vec::new(),
            }))?;
            pp.completion_id = route.completion_id;
            if pp.send_needs_python(&route) {
                return Ok(Err(route));
            }
            // Nothing in the frames comes from Python: no rid carries
            // per_request_info, so none completes a worker graph, and the
            // count, consumption, profiling and publication payloads are all
            // read only for those.
            let (completed, frames) = self.pp_send_frames(&pp, SendArgs::default())?;
            self.dispatch_frames_nogil(frames)?;
            Ok(Ok(completed))
        })?;
        let nogil_ns = t.elapsed().as_nanos() as u64;
        let mut out = match sent {
            Ok(completed) => pp.done(completed),
            Err(route) => self.pp_park_route(pp, route),
        };
        out.nogil_ns = nogil_ns;
        Ok(out)
    }

    fn pp_park(&mut self, pp: Postprocess, need: u8) -> PostprocessOut {
        let out = pp.out(need);
        self.postprocess = Some(pp);
        out
    }

    /// Park a routed step for its send's Python payloads (SEND).
    fn pp_park_route(&mut self, pp: Postprocess, route: RouteOut) -> PostprocessOut {
        let mut out = self.pp_park(pp, NEED_SEND);
        out.route = Some(route);
        out
    }

    /// The send: the stream partitions this pass reports done, then its
    /// frames; the rids a WORKER_GRAPHS_DONE goes out for.
    #[allow(clippy::type_complexity)]
    fn pp_send_frames(
        &mut self, pp: &Postprocess, args: SendArgs,
    ) -> PyResult<(Vec<u32>, Vec<(String, Vec<u8>)>)> {
        for &rid in &pp.stream_done {
            self.mark_stream_partition_done(rid, &pp.partition);
        }
        self.build_send_frames(
            pp.completion_id, args.info_rids, args.request_infos, args.ntc_rids,
            args.new_token_counts, args.consumed_rids, args.stream_tokens_consumed,
            args.prof_rids, args.profiling, args.publish_rids, args.publications, args.skip_rids,
        )
    }

    /// The parked step, for a resume.
    pub(super) fn pp_take(&mut self) -> PyResult<Postprocess> {
        self.postprocess.take().ok_or_else(|| {
            PyRuntimeError::new_err("resume_postprocess: no step is waiting")
        })
    }

    pub(super) fn pp_resume_stops(
        &mut self, py: Python<'_>, rids: Vec<u32>, loops: Vec<Vec<String>>, dropped: Vec<u32>,
    ) -> PyResult<PostprocessOut> {
        let mut pp = self.pp_take()?;
        let given: Vec<(u32, Vec<String>)> = rids.into_iter().zip(loops).collect();
        pp.stops = Some(merged_stops([&pp.known_stops, &given]));
        pp.keep_rows(&dropped.into_iter().collect());
        self.pp_run(py, pp)
    }

    pub(super) fn pp_resume_store(&mut self, py: Python<'_>, stored: Stored) -> PyResult<PostprocessOut> {
        let mut pp = self.pp_take()?;
        pp.stored = Some(stored);
        self.pp_run(py, pp)
    }

    pub(super) fn pp_resume_send(&mut self, py: Python<'_>, args: SendArgs) -> PyResult<PostprocessOut> {
        let pp = self.pp_take()?;
        let (completed, frames) = self.pp_send_frames(&pp, args)?;
        self.send_frames(py, frames)?;
        Ok(pp.done(completed))
    }
}

/// `send_outputs`' Python payloads, as columns.
#[derive(Default)]
pub(super) struct SendArgs {
    pub info_rids: Vec<u32>,
    pub request_infos: Vec<Option<Vec<u8>>>,
    pub ntc_rids: Vec<u32>,
    pub new_token_counts: Vec<FxHashMap<String, i64>>,
    pub consumed_rids: Vec<u32>,
    pub stream_tokens_consumed: Vec<FxHashMap<String, i64>>,
    pub prof_rids: Vec<u32>,
    pub profiling: Vec<Option<Vec<u8>>>,
    pub publish_rids: Vec<u32>,
    pub publications: Vec<Option<Vec<u8>>>,
    /// rids that failed after routing: none of their frames go out
    pub skip_rids: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_rows_drops_a_rows_stops_and_stream_done() {
        let mut pp = Postprocess::new(
            "p".into(), "n".into(), "w".into(), vec!["s".into()], vec![1, 2, 3], None, None,
            vec![(2, vec!["l".into()])], false, None, Vec::new(), None, vec![1, 2], false,
        );
        pp.stops = Some(vec![(2, vec!["l".into()]), (3, vec!["l".into()])]);
        pp.keep_rows(&[2].into_iter().collect());
        assert_eq!(pp.rids, vec![1, 3]);
        assert_eq!(pp.stops, Some(vec![(3, vec!["l".to_string()])]));
        assert_eq!(pp.stream_done, vec![1]);
    }
}
