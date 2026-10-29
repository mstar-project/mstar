//! Which request holds which resident slot of a persistent batch
//! (`mstar/engine/rows/batch.py`'s `PersistentBatch`, whose Python version
//! this is slot for slot).
//!
//! A joining row takes the lowest free slot, and when none is free, the slot
//! of the row that ran least recently (ties: the lower slot) and isn't in
//! the step, which leaves. Assignment is a pure function of the committed
//! state and the step's requests, so ranks fed the same steps agree.

use std::collections::BTreeSet;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rustc_hash::{FxHashMap, FxHashSet};

/// One step's change, as `RowDiff`'s columns.
#[derive(Debug, Default, PartialEq)]
pub struct Diff {
    pub same_step: bool,
    pub idx: Vec<i64>,
    pub join_rids: Vec<i64>,
    pub join_slots: Vec<i64>,
    pub leave_rids: Vec<i64>,
    pub leave_slots: Vec<i64>,
    pub dropped: Vec<i64>,
}

#[derive(Debug)]
pub enum Refused {
    /// a bad step: too many rows, or a request named twice
    Step(String),
    /// a commit against the wrong state
    State(String),
}

pub struct RowSlots {
    pub max_rows: usize,
    pub capacity: usize,
    slot_of: FxHashMap<i64, i64>,
    free: BTreeSet<i64>,
    /// the version each slot last ran at, for choosing what to evict
    last_run: Vec<i64>,
    order: Vec<i64>,
    version: i64,
    /// removed requests whose row has not left yet
    forgotten: FxHashSet<i64>,
}

impl RowSlots {
    pub fn new(max_rows: usize, capacity: usize) -> Result<Self, String> {
        if max_rows == 0 {
            return Err(format!("max_rows must be positive, got {max_rows}"));
        }
        if capacity < max_rows {
            return Err(format!("capacity {capacity} is below the {max_rows} rows a step may run"));
        }
        Ok(Self {
            max_rows,
            capacity,
            slot_of: FxHashMap::default(),
            free: (0..capacity as i64).collect(),
            last_run: vec![0; capacity],
            order: Vec::new(),
            version: 0,
            forgotten: FxHashSet::default(),
        })
    }

    fn sorted_by_slot(&self, rids: impl Iterator<Item = i64>) -> Vec<i64> {
        let mut v: Vec<i64> = rids.collect();
        v.sort_by_key(|r| self.slot_of[r]);
        v
    }

    pub fn propose(&self, rids: &[i64]) -> Result<Diff, Refused> {
        let n = rids.len();
        if n > self.max_rows {
            return Err(Refused::Step(format!("{n} rows exceed the persistent batch's {}", self.max_rows)));
        }
        if self.forgotten.is_empty() && rids == self.order.as_slice() {
            return Ok(Diff { same_step: true, ..Default::default() });
        }
        let wanted: FxHashSet<i64> = rids.iter().copied().collect();
        if wanted.len() != n {
            return Err(Refused::Step("a step names a request twice".into()));
        }
        let joins: Vec<i64> = rids
            .iter()
            .copied()
            .filter(|r| self.forgotten.contains(r) || !self.slot_of.contains_key(r))
            .collect();
        let mut leave_rids = self.sorted_by_slot(self.forgotten.iter().copied());
        let short = joins.len() as i64 - self.free.len() as i64 - leave_rids.len() as i64;
        if short > 0 {
            let mut idle: Vec<(i64, i64, i64)> = self
                .slot_of
                .iter()
                .filter(|(r, _)| !wanted.contains(r) && !self.forgotten.contains(r))
                .map(|(&r, &s)| (self.last_run[s as usize], s, r))
                .collect();
            idle.sort_unstable();
            leave_rids.extend(idle.iter().take(short as usize).map(|&(_, _, r)| r));
        }
        let leave_slots: Vec<i64> = leave_rids.iter().map(|r| self.slot_of[r]).collect();
        let take = joins.len().min(self.free.len());
        let mut join_slots: Vec<i64> = self.free.iter().take(take).copied().collect();
        join_slots.extend(leave_slots.iter().take(joins.len() - take));
        if join_slots.len() != joins.len() {
            return Err(Refused::Step("no slots left for the step's joins".into()));
        }
        let joined: FxHashMap<i64, i64> = joins.iter().copied().zip(join_slots.iter().copied()).collect();
        let idx = rids.iter().map(|r| joined.get(r).copied().unwrap_or_else(|| self.slot_of[r])).collect();
        let dropped = leave_rids.iter().copied().filter(|r| self.forgotten.contains(r)).collect();
        Ok(Diff { same_step: false, idx, join_rids: joins, join_slots, leave_rids, leave_slots, dropped })
    }

    /// The diff checking `rids` in without a step; the committed step keeps
    /// its other rows, in order. Returns it with that order.
    pub fn without(&self, rids: &[i64]) -> (Vec<i64>, Diff) {
        let mut gone: FxHashSet<i64> = rids.iter().copied().filter(|r| self.slot_of.contains_key(r)).collect();
        gone.extend(self.forgotten.iter().copied());
        let leave_rids = self.sorted_by_slot(gone.iter().copied());
        let order: Vec<i64> = self.order.iter().copied().filter(|r| !gone.contains(r)).collect();
        let diff = Diff {
            same_step: false,
            idx: order.iter().map(|r| self.slot_of[r]).collect(),
            leave_slots: leave_rids.iter().map(|r| self.slot_of[r]).collect(),
            dropped: leave_rids.iter().copied().filter(|r| self.forgotten.contains(r)).collect(),
            leave_rids,
            ..Default::default()
        };
        (order, diff)
    }

    /// Make a diff taken at `base_version` the committed state. Returns the
    /// slots that requests removed while holding a row left.
    #[allow(clippy::too_many_arguments)]
    pub fn commit(
        &mut self, base_version: i64, same_step: bool, order: &[i64], idx: &[i64], join_rids: &[i64],
        join_slots: &[i64], leave_rids: &[i64], leave_slots: &[i64],
    ) -> Result<Vec<i64>, Refused> {
        if base_version != self.version {
            return Err(Refused::State(format!(
                "a diff taken at version {base_version} committed at {}", self.version
            )));
        }
        if same_step {
            return Ok(Vec::new());
        }
        // checked before anything changes: the worker never removes a request
        // a step it has planned still runs
        if !self.forgotten.is_empty() {
            let leaving: FxHashSet<i64> = leave_rids.iter().copied().collect();
            let joined: FxHashSet<i64> = join_rids.iter().copied().collect();
            let stale: Vec<i64> = order
                .iter()
                .copied()
                .filter(|r| self.forgotten.contains(r) && !leaving.contains(r) && !joined.contains(r))
                .collect();
            if !stale.is_empty() {
                return Err(Refused::State(format!("a step keeps rows of removed requests {stale:?}")));
            }
        }
        let mut released = Vec::new();
        for (r, &s) in leave_rids.iter().zip(leave_slots) {
            self.slot_of.remove(r);
            if self.forgotten.remove(r) {
                released.push(s);
            }
        }
        let reused: FxHashSet<i64> = join_slots.iter().copied().collect();
        for (&r, &s) in join_rids.iter().zip(join_slots) {
            self.slot_of.insert(r, s);
            self.free.remove(&s);
        }
        for s in leave_slots {
            if !reused.contains(s) {
                self.free.insert(*s);
            }
        }
        self.order = order.to_vec();
        self.version += 1;
        for &s in idx {
            self.last_run[s as usize] = self.version;
        }
        Ok(released)
    }

    pub fn forget(&mut self, rid: i64) -> bool {
        if !self.slot_of.contains_key(&rid) {
            return false;
        }
        self.forgotten.insert(rid);
        true
    }
}

fn refused(e: Refused) -> PyErr {
    match e {
        Refused::Step(m) => PyValueError::new_err(m),
        Refused::State(m) => PyRuntimeError::new_err(m),
    }
}

fn int64s<'py>(py: Python<'py>, v: &[i64]) -> Bound<'py, PyBytes> {
    PyBytes::new(py, &v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>())
}

fn view(buf: &PyBuffer<i64>) -> PyResult<&[i64]> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("diff columns must be C-contiguous int64"));
    }
    // SAFETY: a C-contiguous int64 buffer held by the caller for the call, under the GIL
    Ok(unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const i64, buf.item_count()) })
}

type PyDiff<'py> = (
    bool, Bound<'py, PyBytes>, Vec<i64>, Bound<'py, PyBytes>, Vec<i64>, Bound<'py, PyBytes>, Vec<i64>,
);

fn to_py<'py>(py: Python<'py>, d: &Diff) -> PyDiff<'py> {
    (
        d.same_step, int64s(py, &d.idx), d.join_rids.clone(), int64s(py, &d.join_slots),
        d.leave_rids.clone(), int64s(py, &d.leave_slots), d.dropped.clone(),
    )
}

/// `RowSlots` for Python: a diff comes back as `(same_step, idx, join_rids,
/// join_slots, leave_rids, leave_slots, dropped)`, slot columns as int64 bytes.
#[pyclass(name = "RowSlots")]
pub struct PyRowSlots {
    inner: RowSlots,
}

#[pymethods]
impl PyRowSlots {
    #[new]
    fn new(max_rows: usize, capacity: usize) -> PyResult<Self> {
        Ok(Self { inner: RowSlots::new(max_rows, capacity).map_err(PyValueError::new_err)? })
    }

    fn propose<'py>(&self, py: Python<'py>, rids: Vec<i64>) -> PyResult<PyDiff<'py>> {
        Ok(to_py(py, &self.inner.propose(&rids).map_err(refused)?))
    }

    fn without<'py>(&self, py: Python<'py>, rids: Vec<i64>) -> (Vec<i64>, PyDiff<'py>) {
        let (order, d) = self.inner.without(&rids);
        (order, to_py(py, &d))
    }

    #[allow(clippy::too_many_arguments)]
    fn commit(
        &mut self, base_version: i64, same_step: bool, order: Vec<i64>, idx: PyBuffer<i64>,
        join_rids: Vec<i64>, join_slots: PyBuffer<i64>, leave_rids: Vec<i64>, leave_slots: PyBuffer<i64>,
    ) -> PyResult<Vec<i64>> {
        self.inner
            .commit(
                base_version, same_step, &order, view(&idx)?, &join_rids, view(&join_slots)?, &leave_rids,
                view(&leave_slots)?,
            )
            .map_err(refused)
    }

    fn forget(&mut self, rid: i64) -> bool {
        self.inner.forget(rid)
    }

    fn slot_of(&self, rid: i64) -> Option<i64> {
        self.inner.slot_of.get(&rid).copied()
    }

    fn is_checked_out(&self, rid: i64) -> bool {
        self.inner.slot_of.contains_key(&rid) && !self.inner.forgotten.contains(&rid)
    }

    fn held(&self) -> FxHashMap<i64, i64> {
        self.inner.slot_of.clone()
    }

    fn last_run(&self, slot: usize) -> i64 {
        self.inner.last_run[slot]
    }

    #[getter]
    fn n_held(&self) -> usize {
        self.inner.slot_of.len()
    }

    #[getter]
    fn version(&self) -> i64 {
        self.inner.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(s: &mut RowSlots, rids: &[i64]) -> Diff {
        let d = s.propose(rids).unwrap();
        let v = s.version;
        s.commit(v, d.same_step, rids, &d.idx, &d.join_rids, &d.join_slots, &d.leave_rids, &d.leave_slots)
            .unwrap();
        d
    }

    #[test]
    fn joins_take_the_lowest_free_slots_and_keep_them() {
        let mut s = RowSlots::new(4, 8).unwrap();
        let d = commit(&mut s, &[7, 3]);
        assert_eq!(d.join_slots, vec![0, 1]);
        let d = commit(&mut s, &[3, 9]);
        assert_eq!((d.idx.clone(), d.join_slots.clone()), (vec![1, 2], vec![2]));
        assert!(d.leave_rids.is_empty());
        assert!(s.propose(&[3, 9]).unwrap().same_step);
    }

    #[test]
    fn without_room_the_least_recently_run_row_leaves() {
        let mut s = RowSlots::new(2, 3).unwrap();
        commit(&mut s, &[1, 2]);
        commit(&mut s, &[3]);
        let d = s.propose(&[4, 3]).unwrap();
        assert_eq!((d.leave_rids, d.join_rids, d.join_slots), (vec![1], vec![4], vec![0]));
    }

    #[test]
    fn a_step_keeping_a_removed_row_is_refused_before_anything_changes() {
        let mut s = RowSlots::new(4, 4).unwrap();
        commit(&mut s, &[1, 2]);
        let d = s.propose(&[1, 2, 3]).unwrap();
        s.forget(2);
        let v = s.version;
        let r = s.commit(v, false, &[1, 2, 3], &d.idx, &d.join_rids, &d.join_slots, &d.leave_rids, &d.leave_slots);
        assert!(matches!(r, Err(Refused::State(_))));
        assert_eq!(s.slot_of.len(), 2);
        assert_eq!(s.version, v);
    }
}
