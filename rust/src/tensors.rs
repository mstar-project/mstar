//! Port of `TensorBookkeeping`: per-uuid reference state and descriptors.
//!
//! Never sees a tensor. `TensorStore` stays Python because freeing a record
//! inside a `py.allow_threads` section would drop a `Py<PyAny>` without the
//! GIL; everything here is integers and interned strings, so it can be held
//! across one.
//!
//! The descriptors matter as much as the refcounts: ingest and routing carry
//! uuids, and anything that has to put a descriptor back on the wire (a
//! disaggregated loop re-emitting its external inputs) looks it up here.

use crate::graph::spec::{StrToId, Sym};
use crate::graph::state::TensorRef;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct ReferenceInfo {
    ref_count: i64,
    persist: bool,
    mem_registered: bool,
}

/// `TensorPointerInfo` with its strings interned. `source_entity`,
/// `source_session_id`, the dtype name and `shm_segment` are drawn from a
/// handful of values but repeat on every tensor, so they are ids here and
/// strings only at the boundary.
#[derive(Clone)]
pub struct TensorPointerInfo {
    pub dims: Vec<i64>,
    /// Short torch name ("float16"), matching the wire codec's `_dtype_name`.
    pub dtype: Sym,
    pub nbytes: i64,
    pub address: u64,
    pub stride: Vec<i64>,
    pub uuid: u64,
    pub source_session_id: Sym,
    pub source_entity: Sym,
    pub offset: i64,
    pub source_tp_size: u32,
    pub source_tp_rank: u32,
    /// SHM-arena transport only; None for per-uuid files and Mooncake.
    pub shm_segment: Option<Sym>,
    pub shm_offset: i64,
    pub source_node_name: Option<Sym>,
    pub source_graph_walk: Option<Sym>,
}

/// One descriptor as Python hands it over.
#[derive(FromPyObject)]
pub struct TensorInfoArg {
    #[pyo3(item)] pub dims: Vec<i64>,
    #[pyo3(item)] pub dtype: String,
    #[pyo3(item)] pub nbytes: i64,
    #[pyo3(item)] pub address: u64,
    #[pyo3(item)] pub stride: Vec<i64>,
    #[pyo3(item)] pub uuid: u64,
    #[pyo3(item)] pub source_session_id: String,
    #[pyo3(item)] pub source_entity: String,
    #[pyo3(item)] pub offset: i64,
    #[pyo3(item)] pub source_tp_size: u32,
    #[pyo3(item)] pub source_tp_rank: u32,
    #[pyo3(item)] pub shm_segment: Option<String>,
    #[pyo3(item)] pub shm_offset: i64,
    #[pyo3(item)] pub source_node_name: Option<String>,
    #[pyo3(item)] pub source_graph_walk: Option<String>,
}

/// What `get_info` hands back: field-for-field with `TensorPointerInfo`, with
/// the interned ids resolved. Python rebuilds the dataclass from it (`dtype`
/// via the wire codec's `_dtype_from_name`).
#[pyclass]
#[derive(Clone)]
pub struct TensorInfoOut {
    #[pyo3(get)] pub dims: Vec<i64>,
    #[pyo3(get)] pub dtype: String,
    #[pyo3(get)] pub nbytes: i64,
    #[pyo3(get)] pub address: u64,
    #[pyo3(get)] pub stride: Vec<i64>,
    #[pyo3(get)] pub uuid: u64,
    #[pyo3(get)] pub source_session_id: String,
    #[pyo3(get)] pub source_entity: String,
    #[pyo3(get)] pub offset: i64,
    #[pyo3(get)] pub source_tp_size: u32,
    #[pyo3(get)] pub source_tp_rank: u32,
    #[pyo3(get)] pub shm_segment: Option<String>,
    #[pyo3(get)] pub shm_offset: i64,
    #[pyo3(get)] pub source_node_name: Option<String>,
    #[pyo3(get)] pub source_graph_walk: Option<String>,
}

/// The state itself, held apart from the `#[pyclass]` wrapper so Rust-side
/// holders can share it. `GraphRuntime` needs the SAME bookkeeper Python
/// handed to `TensorStore` -- a copy would diverge the moment either side
/// adjusted a refcount.
#[derive(Default)]
pub struct Bookkeeping {
    ref_info: FxHashMap<u64, ReferenceInfo>,
    tensor_info: FxHashMap<u64, TensorPointerInfo>,
    strings: StrToId,
}

/// Uncontended in practice -- everything runs under the GIL today -- but a
/// Mutex is what makes the pyclass Send + Sync, which pyo3 requires. Hold it
/// for the shortest span possible: a lock held across a call back into Python
/// would deadlock against a Python-side method on the same object.
pub type SharedBookkeeping = Arc<Mutex<Bookkeeping>>;

impl Bookkeeping {
    fn intern_info(&mut self, arg: &TensorInfoArg) -> TensorPointerInfo {
        TensorPointerInfo {
            dims: arg.dims.clone(),
            dtype: self.strings.intern(&arg.dtype),
            nbytes: arg.nbytes,
            address: arg.address,
            stride: arg.stride.clone(),
            uuid: arg.uuid,
            source_session_id: self.strings.intern(&arg.source_session_id),
            source_entity: self.strings.intern(&arg.source_entity),
            offset: arg.offset,
            source_tp_size: arg.source_tp_size,
            source_tp_rank: arg.source_tp_rank,
            shm_segment: arg.shm_segment.as_deref().map(|s| self.strings.intern(s)),
            shm_offset: arg.shm_offset,
            source_node_name: arg
                .source_node_name
                .as_deref()
                .map(|s| self.strings.intern(s)),
            source_graph_walk: arg
                .source_graph_walk
                .as_deref()
                .map(|s| self.strings.intern(s)),
        }
    }

    fn export(&self, info: &TensorPointerInfo) -> TensorInfoOut {
        let name = |s: Sym| self.strings.name(s).to_string();
        TensorInfoOut {
            dims: info.dims.clone(),
            dtype: name(info.dtype),
            nbytes: info.nbytes,
            address: info.address,
            stride: info.stride.clone(),
            uuid: info.uuid,
            source_session_id: name(info.source_session_id),
            source_entity: name(info.source_entity),
            offset: info.offset,
            source_tp_size: info.source_tp_size,
            source_tp_rank: info.source_tp_rank,
            shm_segment: info.shm_segment.map(name),
            shm_offset: info.shm_offset,
            source_node_name: info.source_node_name.map(name),
            source_graph_walk: info.source_graph_walk.map(name),
        }
    }

    /// Reference state, or None when the uuid is not tracked. Every mutator
    /// below is a no-op on an untracked uuid, matching Python: a late ack for
    /// a tensor already collected is benign, not an error.
    fn entry(&mut self, uuid: u64) -> Option<&mut ReferenceInfo> {
        self.ref_info.get_mut(&uuid)
    }

    // -- the operations, callable from either side ------------------------

    /// Start tracking a uuid. Reference state resets: `put_tensor` on a live
    /// uuid means a NEW tensor, not an update (that is `update_info`).
    fn put_tensor(&mut self, uuid: u64, info: TensorInfoArg) {
        let interned = self.intern_info(&info);
        self.ref_info.insert(uuid, ReferenceInfo::default());
        self.tensor_info.insert(uuid, interned);
    }

    fn put_tensor_batch(
        &mut self,
        uuids: Vec<u64>,
        infos: Vec<TensorInfoArg>,
    ) -> PyResult<()> {
        if uuids.len() != infos.len() {
            return Err(PyValueError::new_err(
                "put_tensor_batch: uuids and infos must be the same length",
            ));
        }
        for (uuid, info) in uuids.into_iter().zip(infos) {
            self.put_tensor(uuid, info);
        }
        Ok(())
    }

    /// Rebind a descriptor without touching its refcount. Needed where the
    /// tensor lands before its final descriptor exists: a slice re-points an
    /// arriving info at a freshly minted uuid, and a fan-in consolidation
    /// mints one for a tensor it has just concatenated.
    fn update_info(&mut self, uuid: u64, info: TensorInfoArg) {
        let interned = self.intern_info(&info);
        self.tensor_info.insert(uuid, interned);
    }

    fn update_info_batch(
        &mut self,
        uuids: Vec<u64>,
        infos: Vec<TensorInfoArg>,
    ) -> PyResult<()> {
        if uuids.len() != infos.len() {
            return Err(PyValueError::new_err(
                "update_info_batch: uuids and infos must be the same length",
            ));
        }
        for (uuid, info) in uuids.into_iter().zip(infos) {
            self.update_info(uuid, info);
        }
        Ok(())
    }

    fn get_info(&self, uuid: u64) -> Option<TensorInfoOut> {
        self.tensor_info.get(&uuid).map(|i| self.export(i))
    }

    fn get_info_batch(&self, uuids: Vec<u64>) -> Vec<Option<TensorInfoOut>> {
        uuids.into_iter().map(|u| self.get_info(u)).collect()
    }

    /// Drop the record. The caller frees the tensor itself.
    fn forget_tensor(&mut self, uuid: u64) {
        self.ref_info.remove(&uuid);
        self.tensor_info.remove(&uuid);
    }

    fn is_tracked(&self, uuid: u64) -> bool {
        self.ref_info.contains_key(&uuid)
    }

    fn increment_ref(&mut self, uuid: u64, n: i64) -> PyResult<()> {
        if n < 0 {
            return Err(PyValueError::new_err(format!(
                "Tried to increment tensor {uuid} reference by {n}"
            )));
        }
        if let Some(e) = self.entry(uuid) {
            e.ref_count += n;
        }
        Ok(())
    }

    fn increment_ref_batch(
        &mut self,
        uuids: Vec<u64>,
        counts: Vec<i64>,
    ) -> PyResult<()> {
        if uuids.len() != counts.len() {
            return Err(PyValueError::new_err(
                "increment_ref_batch: uuids and counts must be the same length",
            ));
        }
        for (uuid, n) in uuids.into_iter().zip(counts) {
            self.increment_ref(uuid, n)?;
        }
        Ok(())
    }

    /// A negative `n` is legal here: `set_output_ref_counts` corrects downward
    /// from the safety hold by dereferencing a negative delta.
    fn dereference(&mut self, uuid: u64, n: i64) {
        if let Some(e) = self.entry(uuid) {
            e.ref_count -= n;
        }
    }

    fn dereference_batch(
        &mut self,
        uuids: Vec<u64>,
        counts: Vec<i64>,
    ) -> PyResult<()> {
        if uuids.len() != counts.len() {
            return Err(PyValueError::new_err(
                "dereference_batch: uuids and counts must be the same length",
            ));
        }
        for (uuid, n) in uuids.into_iter().zip(counts) {
            self.dereference(uuid, n);
        }
        Ok(())
    }

    fn set_persist(&mut self, uuid: u64, persist: bool) {
        if let Some(e) = self.entry(uuid) {
            e.persist = persist;
        }
    }

    fn set_persist_batch(&mut self, uuids: Vec<u64>, persist: bool) {
        for uuid in uuids {
            self.set_persist(uuid, persist);
        }
    }

    fn set_mem_registered(&mut self, uuid: u64, mem_registered: bool) {
        if let Some(e) = self.entry(uuid) {
            e.mem_registered = mem_registered;
        }
    }

    fn is_registered(&self, uuid: u64) -> bool {
        self.ref_info.get(&uuid).is_some_and(|e| e.mem_registered)
    }

    /// No references left and not being persisted for the conductor.
    fn can_gc(&self, uuid: u64) -> bool {
        self.ref_info
            .get(&uuid)
            .is_some_and(|e| e.ref_count <= 0 && !e.persist)
    }

    /// Which of `uuids` are now free. One call instead of one per tensor after
    /// a batch of refcount changes.
    fn collectable(&self, uuids: Vec<u64>) -> Vec<u64> {
        uuids.into_iter().filter(|&u| self.can_gc(u)).collect()
    }

    fn len(&self) -> usize {
        self.ref_info.len()
    }

    /// The shape facts routing needs, without rebuilding the whole descriptor.
    ///
    /// A uuid with no descriptor keeps its identity and zeroes the rest: the
    /// uuid is what downstream routes on, and Python tolerates the same case
    /// (`get_info` returning None inside a tensor_info list). It means the
    /// store lost the descriptor, which is a bug upstream of here.
    pub fn tensor_ref(&self, uuid: u64) -> TensorRef {
        match self.tensor_info.get(&uuid) {
            Some(i) => TensorRef {
                uuid,
                dim0: i.dims.first().copied().unwrap_or(0),
                nbytes: i.nbytes,
                offset: i.offset,
            },
            None => TensorRef { uuid, dim0: 0, nbytes: 0, offset: 0 },
        }
    }
}

/// The Python handle. Holds only a share of the state, so
/// `GraphRuntime::new` can take the same one.
#[pyclass]
pub struct TensorBookkeeping {
    inner: SharedBookkeeping,
}

impl TensorBookkeeping {
    /// A second handle on the same state, for a Rust-side holder.
    pub fn share(&self) -> SharedBookkeeping {
        Arc::clone(&self.inner)
    }
}

#[pymethods]
impl TensorBookkeeping {
    #[new]
    fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(Bookkeeping::default())) }
    }

    // Each of these locks, delegates and drops. Nothing below calls back into
    // Python, so the lock is never held across a re-entry.

    fn put_tensor(&self, uuid: u64, info: TensorInfoArg) {
        self.inner.lock().unwrap().put_tensor(uuid, info)
    }

    fn put_tensor_batch(
        &self, uuids: Vec<u64>, infos: Vec<TensorInfoArg>,
    ) -> PyResult<()> {
        self.inner.lock().unwrap().put_tensor_batch(uuids, infos)
    }

    fn update_info(&self, uuid: u64, info: TensorInfoArg) {
        self.inner.lock().unwrap().update_info(uuid, info)
    }

    fn update_info_batch(
        &self, uuids: Vec<u64>, infos: Vec<TensorInfoArg>,
    ) -> PyResult<()> {
        self.inner.lock().unwrap().update_info_batch(uuids, infos)
    }

    fn get_info(&self, uuid: u64) -> Option<TensorInfoOut> {
        self.inner.lock().unwrap().get_info(uuid)
    }

    fn get_info_batch(&self, uuids: Vec<u64>) -> Vec<Option<TensorInfoOut>> {
        self.inner.lock().unwrap().get_info_batch(uuids)
    }

    fn forget_tensor(&self, uuid: u64) {
        self.inner.lock().unwrap().forget_tensor(uuid)
    }

    fn is_tracked(&self, uuid: u64) -> bool {
        self.inner.lock().unwrap().is_tracked(uuid)
    }

    #[pyo3(signature = (uuid, n = 1))]
    fn increment_ref(&self, uuid: u64, n: i64) -> PyResult<()> {
        self.inner.lock().unwrap().increment_ref(uuid, n)
    }

    fn increment_ref_batch(
        &self, uuids: Vec<u64>, counts: Vec<i64>,
    ) -> PyResult<()> {
        self.inner.lock().unwrap().increment_ref_batch(uuids, counts)
    }

    #[pyo3(signature = (uuid, n = 1))]
    fn dereference(&self, uuid: u64, n: i64) {
        self.inner.lock().unwrap().dereference(uuid, n)
    }

    fn dereference_batch(
        &self, uuids: Vec<u64>, counts: Vec<i64>,
    ) -> PyResult<()> {
        self.inner.lock().unwrap().dereference_batch(uuids, counts)
    }

    fn set_persist(&self, uuid: u64, persist: bool) {
        self.inner.lock().unwrap().set_persist(uuid, persist)
    }

    fn set_persist_batch(&self, uuids: Vec<u64>, persist: bool) {
        self.inner.lock().unwrap().set_persist_batch(uuids, persist)
    }

    fn set_mem_registered(&self, uuid: u64, mem_registered: bool) {
        self.inner.lock().unwrap().set_mem_registered(uuid, mem_registered)
    }

    fn is_registered(&self, uuid: u64) -> bool {
        self.inner.lock().unwrap().is_registered(uuid)
    }

    fn can_gc(&self, uuid: u64) -> bool {
        self.inner.lock().unwrap().can_gc(uuid)
    }

    fn collectable(&self, uuids: Vec<u64>) -> Vec<u64> {
        self.inner.lock().unwrap().collectable(uuids)
    }

    fn __len__(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_share_sees_the_same_state() {
        // The point of the split: GraphRuntime holds a share of the very
        // bookkeeper Python gave TensorStore, so a refcount either side
        // adjusts is visible to the other.
        let py_handle = TensorBookkeeping::new();
        let shared = py_handle.share();
        shared.lock().unwrap().ref_info.insert(1, ReferenceInfo::default());

        assert!(py_handle.is_tracked(1), "the Python handle sees it");
        py_handle.increment_ref(1, 1).unwrap();
        assert!(
            !shared.lock().unwrap().can_gc(1),
            "and the Rust share sees the Python handle's change"
        );
    }

    fn bk() -> Bookkeeping {
        Bookkeeping::default()
    }

    fn track(b: &mut Bookkeeping, uuid: u64) {
        b.ref_info.insert(uuid, ReferenceInfo::default());
    }

    #[test]
    fn a_fresh_tensor_is_immediately_collectable() {
        // ref 0 and not persisted: nothing is holding it yet.
        let mut b = bk();
        track(&mut b, 1);
        assert!(b.can_gc(1));
    }

    #[test]
    fn an_untracked_uuid_is_never_collectable() {
        // Distinct from "tracked with no refs": a tensor already collected
        // must not report collectable a second time.
        let b = bk();
        assert!(!b.can_gc(99));
        assert!(!b.is_tracked(99));
        assert!(!b.is_registered(99));
    }

    #[test]
    fn mutating_an_untracked_uuid_is_a_no_op_not_an_error() {
        // A late TENSOR_RECEIVED can arrive for a tensor already collected.
        let mut b = bk();
        b.increment_ref(42, 3).unwrap();
        b.dereference(42, 1);
        b.set_persist(42, true);
        b.set_mem_registered(42, true);
        assert!(!b.is_tracked(42));
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn persist_holds_a_tensor_with_no_references() {
        let mut b = bk();
        track(&mut b, 1);
        b.set_persist(1, true);
        assert!(!b.can_gc(1), "the conductor still needs it");
        b.set_persist(1, false);
        assert!(b.can_gc(1));
    }

    #[test]
    fn references_gate_collection_until_they_drain() {
        let mut b = bk();
        track(&mut b, 1);
        b.increment_ref(1, 2).unwrap();
        assert!(!b.can_gc(1));
        b.dereference(1, 1);
        assert!(!b.can_gc(1));
        b.dereference(1, 1);
        assert!(b.can_gc(1));
    }

    #[test]
    fn a_negative_increment_is_rejected() {
        // Python asserts here: decrementing through increment_ref would mean
        // a caller lost track of which direction it was adjusting.
        let mut b = bk();
        track(&mut b, 1);
        assert!(b.increment_ref(1, -1).is_err());
    }

    #[test]
    fn dereference_accepts_a_negative_delta() {
        // set_output_ref_counts corrects downward from the safety hold by
        // dereferencing a negative amount.
        let mut b = bk();
        track(&mut b, 1);
        b.dereference(1, -2);
        assert!(!b.can_gc(1), "a negative dereference is an increment");
    }

    #[test]
    fn put_tensor_resets_reference_state() {
        // put_tensor on a live uuid means a NEW tensor, so stale refs must not
        // carry over and pin it forever.
        let mut b = bk();
        track(&mut b, 1);
        b.increment_ref(1, 5).unwrap();
        b.set_persist(1, true);
        b.ref_info.insert(1, ReferenceInfo::default());
        assert!(b.can_gc(1));
    }

    #[test]
    fn forget_drops_the_descriptor_with_the_refcount() {
        // Descriptors are keyed by uuid and uuids are never reused, so a leak
        // here grows without bound over a long-lived worker.
        let mut b = bk();
        track(&mut b, 1);
        b.tensor_info.insert(
            1,
            TensorPointerInfo {
                dims: vec![4],
                dtype: 0,
                nbytes: 8,
                address: 0,
                stride: vec![1],
                uuid: 1,
                source_session_id: 0,
                source_entity: 0,
                offset: 0,
                source_tp_size: 1,
                source_tp_rank: 0,
                shm_segment: None,
                shm_offset: 0,
                source_node_name: None,
                source_graph_walk: None,
            },
        );
        b.forget_tensor(1);
        assert!(!b.is_tracked(1));
        assert!(b.tensor_info.is_empty());
    }

    #[test]
    fn collectable_filters_to_the_free_ones() {
        let mut b = bk();
        for u in 1..=3 {
            track(&mut b, u);
        }
        b.increment_ref(2, 1).unwrap();
        b.set_persist(3, true);
        assert_eq!(b.collectable(vec![1, 2, 3, 99]), vec![1]);
    }

    #[test]
    fn mem_registered_is_independent_of_collectability() {
        let mut b = bk();
        track(&mut b, 1);
        b.set_mem_registered(1, true);
        assert!(b.is_registered(1));
        assert!(b.can_gc(1), "registration does not hold a reference");
    }
}
