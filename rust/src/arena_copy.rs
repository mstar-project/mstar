//! The per-tensor hot paths of `ArenaShmCommunicationManager`
//! (`mstar/communication/arena.py`) with no Python call per tensor: staging a
//! batch into the arena and reading a batch out of peers' segments, plus the
//! Python surface of [`Capacity`], which decides where the bytes go.
//!
//! Each runs in three passes: read the Python objects and claim the work with
//! the GIL, copy without it, then publish with it. Whatever needs torch's own
//! calls comes back to the manager as a row index: a non-contiguous source, a
//! tensor on another device.
//!
//! Device copies use cudart's `cudaMemcpyAsync` from function pointers the
//! manager resolved with `_cudart()`, and only where it says the platform is
//! CUDA: this file links no vendor library, and on ROCm or XPU every device row
//! is deferred. They go on the manager's copy streams, in program order, so its
//! stream event and trailing sync cover them as they do torch's own copies.

use std::os::raw::{c_char, c_void};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::arena_capacity::{
    monotonic_now, slot_counts, Capacity, CapacityPolicy, Reserved, Slot, StageError, SpillFiles,
    IN_FILE, PENDING,
};
use crate::bindings::{PySegmentedShmArena, PyShmArena};
use crate::shm::ShmArena;
use crate::tensors::{RidKey, TensorBookkeeping, TensorPointerInfo};
use crate::torch_abi::{
    abi, AbiError, OwnedTensor, TensorFacts, TensorSpec, TorchAbi, DEVICE_CPU, DEVICE_CUDA,
};

/// Below this many host bytes a batch keeps the GIL through its memcpys: they
/// take less time than handing the GIL over and waiting to get it back.
const KEEP_GIL_MAX_BYTES: usize = 16 << 10;
/// A gather group is found by a linear scan, so a batch of all-different shapes
/// stops looking here and reads the rest one by one.
const MAX_GATHER_GROUPS: usize = 32;
/// `c10::ScalarType::Byte`, kept stable by torch's serialization.
const SCALAR_UINT8: i32 = 0;

const CUDA_HOST_TO_DEVICE: i32 = 1;
const CUDA_DEVICE_TO_HOST: i32 = 2;

type MemcpyAsyncFn = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, usize) -> i32;
type GetLastErrorFn = unsafe extern "C" fn() -> i32;
type ErrorStringFn = unsafe extern "C" fn(i32) -> *const c_char;
type StreamSyncFn = unsafe extern "C" fn(usize) -> i32;

/// libcudart's entry points, as loaded by the manager's `_cudart()`.
struct Cudart {
    memcpy_async: MemcpyAsyncFn,
    get_last_error: GetLastErrorFn,
    error_string: ErrorStringFn,
    stream_synchronize: StreamSyncFn,
}

impl Cudart {
    /// Queue one copy on `stream`. On failure the error is cleared: left set, the
    /// thread's next torch kernel launch would raise it instead.
    fn copy(
        &self,
        dst: *mut u8,
        src: *const u8,
        n: usize,
        kind: i32,
        stream: usize,
    ) -> Result<(), String> {
        let rc = unsafe { (self.memcpy_async)(dst.cast(), src.cast(), n, kind, stream) };
        if rc == 0 {
            return Ok(());
        }
        unsafe { (self.get_last_error)() };
        let why = unsafe { std::ffi::CStr::from_ptr((self.error_string)(rc)) };
        Err(format!(
            "cudaMemcpyAsync of {n} bytes failed: {} (rc={rc})",
            why.to_string_lossy()
        ))
    }

    /// Wait for everything queued on `stream`.
    fn synchronize(&self, stream: usize) -> Result<(), String> {
        let rc = unsafe { (self.stream_synchronize)(stream) };
        if rc == 0 {
            return Ok(());
        }
        unsafe { (self.get_last_error)() };
        let why = unsafe { std::ffi::CStr::from_ptr((self.error_string)(rc)) };
        Err(format!(
            "cudaStreamSynchronize failed: {} (rc={rc})",
            why.to_string_lossy()
        ))
    }
}

/// One tensor being staged.
struct StageRow {
    index: u32,
    uuid: u64,
    src: *const u8,
    nbytes: usize,
    on_device: bool,
    /// `(segment, offset)`, or `(IN_FILE, 0)` for a row spilled to its file.
    placed: Option<(u32, u64)>,
}

// `src` is kept valid by the batch's `keep` list, across the pass that uses it.
unsafe impl Send for StageRow {}

/// One tensor being read, once its Python descriptor has been read.
struct ReadRow {
    /// Null for a row read from a file, or with no bytes.
    src: *const u8,
    /// The row's spill file, when the producer had no room in its arena.
    file: Option<String>,
    /// Where in the file: a TP slice starts past its tensor's first byte.
    file_offset: u64,
    nbytes: usize,
    dtype: i32,
    dims_at: u32,
    ndim: u32,
    segment: u32,
}

// `src` points into a mapping held by the batch's `segments` list, and only
// passes it to a copy.
unsafe impl Send for ReadRow {}
unsafe impl Sync for ReadRow {}

/// Rows of one shape, dtype and segment, read as one copy.
struct GatherGroup {
    rows: Vec<u32>,
}

/// What a descriptor contributes to the store, past what the read itself uses.
struct ReadInfo<'py> {
    uuid: u64,
    dims: Vec<i64>,
    stride: Vec<i64>,
    dtype: Bound<'py, PyAny>,
    scalar: i32,
    nbytes: i64,
    address: u64,
    session: Bound<'py, PyString>,
    entity: Bound<'py, PyString>,
    offset: i64,
    source_tp_size: u32,
    source_tp_rank: u32,
    segment: Option<Bound<'py, PyString>>,
    shm_offset: i64,
    node: Option<Bound<'py, PyString>>,
    walk: Option<Bound<'py, PyString>>,
}

fn int_list(ob: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    // torch.Size is a tuple; a descriptor off the wire may hold a list.
    if let Ok(tuple) = ob.downcast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter_borrowed() {
            out.push(item.extract::<i64>()?);
        }
        return Ok(out);
    }
    ob.extract()
}

fn opt_str<'py>(ob: Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyString>>> {
    if ob.is_none() {
        return Ok(None);
    }
    Ok(Some(ob.downcast_into()?))
}

fn read_info<'py>(info: &Bound<'py, PyAny>, abi: &dyn TorchAbi) -> PyResult<ReadInfo<'py>> {
    let py = info.py();
    let dtype = info.getattr(intern!(py, "dtype"))?;
    let scalar = unsafe { abi.dtype_scalar(dtype.as_ptr()) }
        .ok_or_else(|| PyValueError::new_err("a descriptor's dtype is not a torch.dtype"))?;
    Ok(ReadInfo {
        uuid: info.getattr(intern!(py, "uuid"))?.extract()?,
        dims: int_list(&info.getattr(intern!(py, "dims"))?)?,
        stride: int_list(&info.getattr(intern!(py, "stride"))?)?,
        dtype,
        scalar,
        nbytes: info.getattr(intern!(py, "nbytes"))?.extract()?,
        address: info.getattr(intern!(py, "address"))?.extract()?,
        session: info.getattr(intern!(py, "source_session_id"))?.downcast_into()?,
        entity: info.getattr(intern!(py, "source_entity"))?.downcast_into()?,
        offset: info.getattr(intern!(py, "offset"))?.extract()?,
        source_tp_size: info.getattr(intern!(py, "source_tp_size"))?.extract()?,
        source_tp_rank: info.getattr(intern!(py, "source_tp_rank"))?.extract()?,
        segment: opt_str(info.getattr(intern!(py, "shm_segment"))?)?,
        shm_offset: info.getattr(intern!(py, "shm_offset"))?.extract()?,
        node: opt_str(info.getattr(intern!(py, "_source_node_name"))?)?,
        walk: opt_str(info.getattr(intern!(py, "_source_graph_walk"))?)?,
    })
}

/// What a staging pass did. `deferred` indexes the call's uuids: those are for
/// the manager's Python path. `error`, when set, is the copy that failed; the
/// rows before it are staged and published, the rest released. Reported rather
/// than raised, so the caller still sees `queued` and can sync.
#[pyclass]
pub struct StageOut {
    #[pyo3(get)]
    queued: bool,
    #[pyo3(get)]
    deferred: Vec<u32>,
    #[pyo3(get)]
    error: Option<String>,
    /// The errno of a failed spill write, which the caller raises as `OSError`.
    #[pyo3(get)]
    errno: Option<i32>,
    /// Columns parallel to the call's uuids, `-1` where nothing was staged and
    /// `-2` in `segments` for a row spilled to a file; present only when asked
    /// for.
    #[pyo3(get)]
    segments: Option<Vec<i64>>,
    #[pyo3(get)]
    offsets: Option<Vec<i64>>,
    #[pyo3(get)]
    nbytes: Option<Vec<i64>>,
    /// The source tensors, which the queued copies read until the caller's
    /// trailing sync, whatever a concurrent teardown does to the store: hold
    /// this object until then.
    #[allow(dead_code)]
    sources: Vec<Py<PyAny>>,
}

/// What a read did: the rows are in the store but for `deferred`. `keep` must
/// outlive the batch's device copies.
#[pyclass(get_all)]
pub struct ReadOut {
    did_work: bool,
    deferred: Vec<u32>,
    keep: Vec<Py<PyAny>>,
}

/// What `read_edges` did. `other_*` are parallel columns of `(request, edge,
/// info)` positions that it left for the caller; `keep` as in [`ReadOut`].
#[pyclass(get_all)]
pub struct ReadEdgesOut {
    did_work: bool,
    keep: Vec<Py<PyAny>>,
    other_requests: Vec<u32>,
    other_edges: Vec<u32>,
    other_infos: Vec<u32>,
}

/// One manager's transport. Every method takes the GIL only for its first and
/// last pass.
#[pyclass(name = "ArenaTransport", frozen)]
pub struct PyArenaTransport {
    capacity: Capacity,
    cuda: Option<Cudart>,
    /// The manager's device index, or -1 with no device. Device rows on another
    /// device go to the Python path, which picks that device's stream.
    device_index: i32,
    d2h_stream: Option<usize>,
    h2d_stream: Option<usize>,
    gather_item_bytes: usize,
    dtype_names: Mutex<FxHashMap<i32, String>>,
}

#[pymethods]
impl PyArenaTransport {
    #[new]
    #[pyo3(signature = (
        arena, spill_dir, spill_prefix, entity, spill, spill_after_s, full_timeout_s, slot_ttl_s,
        shm_total=None, device_index=-1, gather_item_bytes=0, cuda=None, d2h_stream=None,
        h2d_stream=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        arena: Option<PyRef<'_, PySegmentedShmArena>>,
        spill_dir: String,
        spill_prefix: String,
        entity: String,
        spill: bool,
        spill_after_s: f64,
        full_timeout_s: f64,
        slot_ttl_s: f64,
        shm_total: Option<usize>,
        device_index: i32,
        gather_item_bytes: usize,
        cuda: Option<(usize, usize, usize, usize)>,
        d2h_stream: Option<usize>,
        h2d_stream: Option<usize>,
    ) -> Self {
        // The addresses are ctypes function pointers of libcudart's own symbols.
        let cuda = cuda.map(|(memcpy, last_error, error_string, sync)| unsafe {
            use std::mem::transmute as t;
            Cudart {
                memcpy_async: t::<usize, MemcpyAsyncFn>(memcpy),
                get_last_error: t::<usize, GetLastErrorFn>(last_error),
                error_string: t::<usize, ErrorStringFn>(error_string),
                stream_synchronize: t::<usize, StreamSyncFn>(sync),
            }
        });
        let policy = CapacityPolicy {
            spill,
            spill_after_s,
            full_timeout_s,
            slot_ttl_s,
        };
        Self {
            capacity: Capacity::new(
                arena.map(|a| a.share()),
                SpillFiles::new(spill_dir, spill_prefix, entity),
                policy,
                shm_total,
            ),
            cuda,
            device_index,
            d2h_stream,
            h2d_stream,
            gather_item_bytes,
            dtype_names: Mutex::new(FxHashMap::default()),
        }
    }

    /// Turn page-locking on, from the runtime's `(register, unregister,
    /// get_last_error)` addresses, and pin the segments already there.
    fn set_pins(&self, py: Python<'_>, fns: (usize, usize, usize), budget: usize) {
        py.detach(|| self.capacity.set_pins(fns.0, fns.1, fns.2, budget));
        self.flush_notes(py);
    }

    /// Unregister everything pinned; what is mapped later stays unpinned. Must
    /// run before the mappings go.
    fn release_pins(&self, py: Python<'_>) {
        py.detach(|| self.capacity.release_pins());
        self.flush_notes(py);
    }

    #[getter]
    fn pinned_bytes(&self, py: Python<'_>) -> usize {
        py.detach(|| self.capacity.pinned_bytes())
    }

    /// Base addresses of the own segments that are registered.
    #[getter]
    fn pinned_own(&self, py: Python<'_>) -> Vec<usize> {
        py.detach(|| self.capacity.pinned_own())
    }

    /// `(uuids with an arena slot, spill files)`.
    fn live(&self) -> (usize, usize) {
        slot_counts(&self.capacity.slots.lock())
    }

    /// Whether a uuid has a slot or file, or is being staged.
    fn has_slot(&self, uuid: u64) -> bool {
        self.capacity.slots.lock().contains_key(&uuid)
    }

    /// `(segment, offset)` of a uuid's slot. `KeyError` while it has none.
    fn slot(&self, uuid: u64) -> PyResult<(u32, u64)> {
        match self.capacity.slots.lock().get(&uuid) {
            Some(s) if s.seg != PENDING && s.seg != IN_FILE => Ok((s.seg, s.offset)),
            _ => Err(PyKeyError::new_err(uuid)),
        }
    }

    /// The path of a uuid this entity spilled, or None if it did not.
    fn spilled_path(&self, entity: &str, uuid: u64) -> Option<String> {
        let spilled = self.capacity.slots.lock().get(&uuid).is_some_and(|s| s.seg == IN_FILE);
        spilled.then(|| self.capacity.spill.path(entity, uuid))
    }

    fn spill_path(&self, entity: &str, uuid: u64) -> String {
        self.capacity.spill.path(entity, uuid)
    }

    /// A place for `nbytes`, by every layer but the file: None when it has to
    /// spill. `RuntimeError` past a strict wait. GIL released: it can grow, pin
    /// and wait for ACKs.
    fn reserve(&self, py: Python<'_>, nbytes: usize) -> PyResult<Option<(usize, usize)>> {
        let got = py.detach(|| match self.capacity.reserve(nbytes) {
            Reserved::Slot { seg, offset, .. } => Ok(Some((seg, offset))),
            Reserved::Spill => Ok(None),
            Reserved::Failed(why) => Err(why),
        });
        self.flush_notes(py);
        got.map_err(PyRuntimeError::new_err)
    }

    /// Record a slot `reserve` gave. False if `uuid` already has one or is being
    /// staged: the caller gives its slot back with `return_slot`.
    fn claim(&self, uuid: u64, segment: u32, offset: u64) -> bool {
        let mut slots = self.capacity.slots.lock();
        if slots.contains_key(&uuid) {
            return false;
        }
        let slot = Slot {
            seg: segment,
            offset,
            staged_at: monotonic_now(),
        };
        slots.insert(uuid, slot);
        true
    }

    fn return_slot(&self, py: Python<'_>, segment: u32, offset: u64) {
        py.detach(|| self.capacity.return_slot(segment, offset));
    }

    /// Spill a tensor's bytes to its own file, recording it for `free`. False if
    /// `uuid` already has a slot or file or is being staged, which writes nothing.
    fn spill_write(&self, py: Python<'_>, uuid: u64, data: &[u8]) -> PyResult<bool> {
        let now = monotonic_now();
        {
            let mut slots = self.capacity.slots.lock();
            if slots.contains_key(&uuid) {
                return Ok(false);
            }
            let slot = Slot {
                seg: PENDING,
                offset: 0,
                staged_at: now,
            };
            slots.insert(uuid, slot);
        }
        let written = py.detach(|| self.capacity.spill.write(uuid, data));
        let mut slots = self.capacity.slots.lock();
        if let Err(e) = written {
            slots.remove(&uuid);
            return Err(spill_pyerr(py, e));
        }
        match slots.get_mut(&uuid) {
            Some(slot) => slot.seg = IN_FILE,
            None => {
                // Torn down while it was written: its reclaim found only the claim.
                drop(slots);
                self.capacity.spill_unlink(uuid);
            }
        }
        Ok(true)
    }

    /// Give back each uuid's slot or file. GIL released: a growth holds the
    /// segments lock across an mmap.
    fn free(&self, py: Python<'_>, uuids: Vec<u64>) {
        py.detach(|| self.capacity.free(uuids));
    }

    /// The TTL backstop; how many slots and files it freed.
    fn reclaim_expired(&self, py: Python<'_>) -> usize {
        let n = py.detach(|| self.capacity.reclaim_expired());
        self.flush_notes(py);
        n
    }

    /// The spill files, then this arena's own segments, for a process that is
    /// stopping. A second call finds nothing left.
    fn unlink_owned(&self, py: Python<'_>) {
        py.detach(|| self.capacity.unlink_owned());
    }

    fn unlink_segments(&self, py: Python<'_>) {
        py.detach(|| self.capacity.unlink_segments());
    }

    /// A peer's segment as a buffer, mapped and pinned the first time.
    fn peer(&self, py: Python<'_>, name: &str) -> PyResult<PyShmArena> {
        let mapped = py.detach(|| self.capacity.map_peer(name));
        self.flush_notes(py);
        Ok(PyShmArena::of(mapped.map_err(PyRuntimeError::new_err)?))
    }

    fn has_peer(&self, name: &str) -> bool {
        self.capacity.peer(name).is_some()
    }

    /// Drop the peer mappings whose file is gone. The caller makes sure no copy
    /// still reads one.
    fn evict_dead_peers(&self, py: Python<'_>) {
        py.detach(|| self.capacity.evict_dead_peers());
        self.flush_notes(py);
    }

    /// Stage `uuids` from `store`: reserve, copy, publish the placement. Run it
    /// in the D2H stream's context, and hold the result through the trailing
    /// sync that lands the copies before this batch's descriptors can ship.
    fn stage(
        &self,
        py: Python<'_>,
        store: PyRef<'_, TensorBookkeeping>,
        uuids: Vec<u64>,
        columns: bool,
    ) -> PyResult<StageOut> {
        let abi = abi().ok_or_else(|| PyRuntimeError::new_err("torch_abi_load has not run"))?;
        let store = store.share();
        let now = monotonic_now();
        let mut rows = Vec::with_capacity(uuids.len());
        let mut keep = Vec::with_capacity(uuids.len());
        let mut deferred = Vec::new();
        let (mut any_device, mut host_bytes) = (false, 0usize);
        {
            let bk = store.lock().expect("bookkeeping lock");
            let mut slots = self.capacity.slots.lock();
            let mut facts = TensorFacts::default();
            for (i, &uuid) in uuids.iter().enumerate() {
                // Already staged, or being staged by another thread: claiming
                // here under the lock is what stops two reserves for one uuid.
                if bk.is_registered(uuid) || slots.contains_key(&uuid) {
                    continue;
                }
                let Some(source) = bk.send_source(uuid) else {
                    deferred.push(i as u32);
                    continue;
                };
                let readable = unsafe { abi.facts(source.as_ptr(), &mut facts) }
                    && facts.plain != 0
                    && facts.contiguous != 0;
                if !readable || !self.can_copy_from(&facts) {
                    deferred.push(i as u32);
                    continue;
                }
                let on_device = facts.device_type == DEVICE_CUDA;
                let nbytes = facts.nbytes as usize;
                any_device |= on_device;
                host_bytes += if on_device { 0 } else { nbytes };
                let slot = Slot {
                    seg: PENDING,
                    offset: 0,
                    staged_at: now,
                };
                slots.insert(uuid, slot);
                keep.push(source.clone_ref(py));
                rows.push(StageRow {
                    index: i as u32,
                    uuid,
                    src: facts.data,
                    nbytes,
                    on_device,
                    placed: None,
                });
            }
        }

        // With the GIL only for a small host batch that fits as it is: growing,
        // waiting and writing files all happen without it.
        let mut pass = StagePass::default();
        if !any_device && host_bytes <= KEEP_GIL_MAX_BYTES {
            self.copy_stages(&mut rows, &mut pass, false);
        }
        if pass.next < rows.len() && pass.error.is_none() {
            py.detach(|| self.copy_stages(&mut rows, &mut pass, true));
        }
        self.flush_notes(py);

        let mut loose = Vec::new();
        {
            let mut bk = store.lock().expect("bookkeeping lock");
            let mut slots = self.capacity.slots.lock();
            for row in &rows {
                let Some((seg, offset)) = row.placed else {
                    slots.remove(&row.uuid);
                    continue;
                };
                // Torn down while we copied: its reclaim found only the claim, or a
                // store that frees in place never reached the table at all.
                let held = bk.has_tensor(row.uuid);
                let Some(slot) = slots.get_mut(&row.uuid).filter(|_| held) else {
                    slots.remove(&row.uuid);
                    loose.push((row.uuid, seg, offset));
                    continue;
                };
                slot.seg = seg;
                slot.offset = offset;
                if seg == IN_FILE {
                    bk.set_mem_registered(row.uuid, true);
                    continue;
                }
                if let Some(arena) = &self.capacity.arena {
                    bk.publish_staged(row.uuid, seg as usize, offset as i64, || {
                        arena.segment_name(seg as usize)
                    });
                }
            }
        }
        if !loose.is_empty() {
            py.detach(|| {
                // A queued copy could land in a slot given to someone else.
                if let (true, Some(cuda), Some(stream)) = (pass.queued, &self.cuda, self.d2h_stream) {
                    let _ = cuda.synchronize(stream);
                }
                for (uuid, seg, offset) in loose {
                    if seg == IN_FILE {
                        self.capacity.spill_unlink(uuid);
                    } else {
                        self.capacity.return_slot(seg, offset);
                    }
                }
            });
        }

        let (error, errno) = match pass.error {
            Some(e) => (Some(e.message), e.errno),
            None => (None, None),
        };
        let mut out = StageOut {
            queued: pass.queued,
            deferred,
            error,
            errno,
            segments: None,
            offsets: None,
            nbytes: None,
            sources: keep,
        };
        if columns {
            let mut segments = vec![-1i64; uuids.len()];
            let mut offsets = vec![-1i64; uuids.len()];
            let mut nbytes = vec![-1i64; uuids.len()];
            for row in &rows {
                if let Some((seg, offset)) = row.placed {
                    segments[row.index as usize] = if seg == IN_FILE { -2 } else { i64::from(seg) };
                    offsets[row.index as usize] = offset as i64;
                    nbytes[row.index as usize] = row.nbytes as i64;
                }
            }
            (out.segments, out.offsets, out.nbytes) = (Some(segments), Some(offsets), Some(nbytes));
        }
        Ok(out)
    }

    /// Read `infos` from peers' segments into new tensors on the manager's
    /// device and put them in `store`, each with `refs` references. Run it in
    /// the H2D stream's context, which tags the new device blocks. `host_max`:
    /// a segment row at least this big on a host device is left to the caller's
    /// HostCopier. `origins`: local uuid -> the producer's, for a slice re-pointed
    /// to a uuid of this process, whose file is under the producer's.
    #[pyo3(signature = (store, rids, infos, refs, device_type, origins, host_max=None))]
    #[allow(clippy::too_many_arguments)]
    fn read<'py>(
        &self,
        py: Python<'py>,
        store: PyRef<'py, TensorBookkeeping>,
        rids: Vec<RidKey>,
        infos: &Bound<'py, PyList>,
        refs: i64,
        device_type: i32,
        origins: &Bound<'py, PyDict>,
        host_max: Option<usize>,
    ) -> PyResult<ReadOut> {
        if rids.len() != infos.len() {
            return Err(PyValueError::new_err("rids and infos must be parallel"));
        }
        let infos: Vec<_> = infos.iter().collect();
        self.read_rows(py, &store, &rids, &infos, refs, device_type, origins, host_max)
    }

    /// `read` for the descriptors of `requests`, a list of `(rid, edges)`: every
    /// one this call can read goes straight through, and what comes back is a
    /// column of positions for the caller to settle one by one. Those are a
    /// tensor of this entity's own, one the store already holds, one met earlier
    /// in the batch, and one left to the HostCopier.
    #[pyo3(signature = (store, requests, refs, device_type, origins, host_max=None))]
    #[allow(clippy::too_many_arguments)]
    fn read_edges<'py>(
        &self,
        py: Python<'py>,
        store: PyRef<'py, TensorBookkeeping>,
        requests: &Bound<'py, PyList>,
        refs: i64,
        device_type: i32,
        origins: &Bound<'py, PyDict>,
        host_max: Option<usize>,
    ) -> PyResult<ReadEdgesOut> {
        let mut infos = Vec::new();
        let mut uuids = Vec::new();
        let mut rids = Vec::new();
        let mut places: Vec<(u32, u32, u32)> = Vec::new();
        let mut others: Vec<(u32, u32, u32)> = Vec::new();
        let own = self.capacity.spill.entity();
        for (r, request) in requests.iter().enumerate() {
            let request = request.downcast_into::<PyTuple>()?;
            let rid: RidKey = request.get_borrowed_item(0)?.extract()?;
            for (e, edge) in request.get_borrowed_item(1)?.try_iter()?.enumerate() {
                let tensor_info = edge?.getattr(intern!(py, "tensor_info"))?;
                for (i, info) in tensor_info.try_iter()?.enumerate() {
                    let info = info?;
                    let place = (r as u32, e as u32, i as u32);
                    let entity = info.getattr(intern!(py, "source_entity"))?;
                    if entity.downcast::<PyString>()?.to_str()? == own {
                        others.push(place);
                        continue;
                    }
                    uuids.push(info.getattr(intern!(py, "uuid"))?.extract::<u64>()?);
                    infos.push(info);
                    rids.push(rid.clone());
                    places.push(place);
                }
            }
        }
        {
            // Apart from the attribute reads above, which can allocate: a
            // collection could run a finalizer, which must not find this lock.
            let bk = store.share_lock();
            let mut seen = FxHashSet::with_capacity_and_hasher(uuids.len(), Default::default());
            let kept: Vec<bool> =
                uuids.iter().map(|&u| !bk.has_tensor(u) && seen.insert(u)).collect();
            drop(bk);
            if kept.contains(&false) {
                let mut k = kept.iter();
                others.extend(places.iter().zip(&kept).filter(|(_, &keep)| !keep).map(|(p, _)| *p));
                infos.retain(|_| *k.next().expect("parallel"));
                let mut k = kept.iter();
                rids.retain(|_| *k.next().expect("parallel"));
                let mut k = kept.iter();
                places.retain(|_| *k.next().expect("parallel"));
            }
        }
        let out = self.read_rows(py, &store, &rids, &infos, refs, device_type, origins, host_max)?;
        others.extend(out.deferred.iter().map(|&i| places[i as usize]));
        others.sort_unstable();
        Ok(ReadEdgesOut {
            did_work: out.did_work,
            keep: out.keep,
            other_requests: others.iter().map(|p| p.0).collect(),
            other_edges: others.iter().map(|p| p.1).collect(),
            other_infos: others.iter().map(|p| p.2).collect(),
        })
    }
}

impl PyArenaTransport {
    #[allow(clippy::too_many_arguments)]
    fn read_rows<'py>(
        &self,
        py: Python<'py>,
        store: &PyRef<'py, TensorBookkeeping>,
        rids: &[RidKey],
        infos: &[Bound<'py, PyAny>],
        refs: i64,
        device_type: i32,
        origins: &Bound<'py, PyDict>,
        host_max: Option<usize>,
    ) -> PyResult<ReadOut> {
        let abi = abi().ok_or_else(|| PyRuntimeError::new_err("torch_abi_load has not run"))?;
        let on_device = device_type != DEVICE_CPU;
        let stream = match (on_device, &self.cuda, self.h2d_stream) {
            (false, _, _) => None,
            (true, Some(cuda), Some(stream)) => Some((cuda, stream)),
            _ => return Err(PyRuntimeError::new_err("no CUDA copy stream to read onto")),
        };

        // First pass: the descriptors, and which segment each one reads. A peer
        // segment met for the first time is mapped without the GIL (pinning takes
        // tens of ms), then the pass runs again.
        let parsed = loop {
            let pass = self.parse_reads(rids, infos, on_device, origins, host_max, abi)?;
            if pass.missing.is_empty() {
                break pass;
            }
            let missing = pass.missing;
            let mapped = py.detach(|| {
                missing.iter().try_for_each(|name| self.capacity.map_peer(name).map(|_| ()))
            });
            self.flush_notes(py);
            mapped.map_err(PyRuntimeError::new_err)?;
        };
        let ParsedReads {
            infos: parsed,
            rows,
            dims_flat,
            owners,
            deferred,
            did_work,
            host_bytes,
            any_file,
            segments,
            ..
        } = parsed;

        // Second pass: allocate and copy.
        let groups = if on_device {
            gather_groups(&rows, &dims_flat, self.gather_item_bytes)
        } else {
            Vec::new()
        };
        let device_index = if on_device { self.device_index } else { -1 };
        let job = ReadJob {
            rows: &rows,
            dims_flat: &dims_flat,
            groups: &groups,
            device_type,
            device_index,
            stream,
        };
        let copied = if on_device || any_file || host_bytes > KEEP_GIL_MAX_BYTES {
            py.detach(|| job.run(abi))
        } else {
            job.run(abi)
        }
        .map_err(|e| e.into_pyerr(py))?;

        // Third pass: Python objects first, since allocating one can run a
        // collection and so arbitrary finalizers, which must not find the lock.
        let mut tensors = Vec::with_capacity(rows.len());
        for t in copied.tensors {
            tensors.push(abi.wrap(py, t)?);
        }
        // The copies read these until the event. The mappings too: a dead peer's
        // eviction must not unmap one under a copy, whatever the manager's gates see.
        let mut keep = Vec::with_capacity(copied.hosts.len());
        for host in copied.hosts {
            keep.push(abi.wrap(py, host)?);
        }
        if on_device {
            for arena in segments {
                keep.push(Py::new(py, PyShmArena::of(arena))?.into_any());
            }
        }
        let mut last_scalar = None;
        for info in &parsed {
            if last_scalar != Some(info.scalar) {
                self.remember_dtype(info)?;
                last_scalar = Some(info.scalar);
            }
        }
        let mut displaced = Vec::new();
        {
            let names = self.dtype_names.lock().expect("dtype names lock");
            let mut bk = store.share_lock();
            for ((info, rid), tensor) in parsed.iter().zip(owners).zip(tensors) {
                let mut sym = |s: &Bound<'py, PyString>| -> PyResult<_> {
                    Ok(bk.strings_mut().intern(s.to_str()?))
                };
                let record = TensorPointerInfo {
                    dims: info.dims.clone(),
                    dtype: 0,
                    nbytes: info.nbytes,
                    address: info.address,
                    stride: info.stride.clone(),
                    uuid: info.uuid,
                    source_session_id: sym(&info.session)?,
                    source_entity: sym(&info.entity)?,
                    offset: info.offset,
                    source_tp_size: info.source_tp_size,
                    source_tp_rank: info.source_tp_rank,
                    shm_segment: info.segment.as_ref().map(&mut sym).transpose()?,
                    shm_offset: info.shm_offset,
                    source_node_name: info.node.as_ref().map(&mut sym).transpose()?,
                    source_graph_walk: info.walk.as_ref().map(&mut sym).transpose()?,
                };
                let dtype = bk.strings_mut().intern(&names[&info.scalar]);
                displaced.extend(bk.put_read(
                    TensorPointerInfo { dtype, ..record },
                    rid,
                    tensor,
                    refs,
                ));
            }
        }
        drop(displaced);
        Ok(ReadOut {
            did_work,
            deferred,
            keep,
        })
    }
}

/// The first pass of a read.
struct ParsedReads<'py> {
    infos: Vec<ReadInfo<'py>>,
    rows: Vec<ReadRow>,
    dims_flat: Vec<i64>,
    owners: Vec<RidKey>,
    deferred: Vec<u32>,
    /// Peer segments to map before the rows mean anything.
    missing: Vec<String>,
    /// Keeps each row's source mapped across the copies.
    segments: Vec<Arc<ShmArena>>,
    did_work: bool,
    host_bytes: usize,
    any_file: bool,
}

impl PyArenaTransport {
    fn flush_notes(&self, py: Python<'_>) {
        let notes = self.capacity.notes.take();
        if notes.is_empty() {
            return;
        }
        let logger = py
            .import("logging")
            .and_then(|m| m.getattr("getLogger"))
            .and_then(|f| f.call1(("mstar.communication.arena",)));
        if let Ok(logger) = logger {
            for (level, message) in notes {
                let _ = logger.call_method1("log", (level, message));
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_reads<'py>(
        &self,
        rids: &[RidKey],
        infos: &[Bound<'py, PyAny>],
        on_device: bool,
        origins: &Bound<'py, PyDict>,
        host_max: Option<usize>,
        abi: &dyn TorchAbi,
    ) -> PyResult<ParsedReads<'py>> {
        let mut out = ParsedReads {
            infos: Vec::with_capacity(infos.len()),
            rows: Vec::with_capacity(infos.len()),
            dims_flat: Vec::with_capacity(infos.len()),
            owners: Vec::with_capacity(infos.len()),
            deferred: Vec::new(),
            missing: Vec::new(),
            segments: Vec::new(),
            did_work: false,
            host_bytes: 0,
            any_file: false,
        };
        let mut last: Option<(*mut pyo3::ffi::PyObject, u32)> = None;
        for (i, (info, rid)) in infos.iter().zip(rids).enumerate() {
            let info = read_info(info, abi)?;
            let nbytes = usize::try_from(info.nbytes)
                .map_err(|_| PyValueError::new_err("a descriptor's nbytes is negative"))?;
            let (src, file, segment) = match &info.segment {
                None => {
                    let uuid = match origins.get_item(info.uuid)? {
                        Some(producer) => producer.extract()?,
                        None => info.uuid,
                    };
                    out.any_file = true;
                    let path = self.capacity.spill.path(info.entity.to_str()?, uuid);
                    (std::ptr::null(), Some(path), u32::MAX)
                }
                Some(name) => {
                    if !on_device && host_max.is_some_and(|max| nbytes >= max) {
                        out.deferred.push(i as u32);
                        continue;
                    }
                    let segment = match last {
                        Some((ptr, at)) if ptr == name.as_ptr() => at,
                        _ => {
                            let name_str = name.to_str()?;
                            let Some(arena) = self.capacity.peer(name_str) else {
                                if !out.missing.iter().any(|m| m == name_str) {
                                    out.missing.push(name_str.to_string());
                                }
                                continue;
                            };
                            let at = out
                                .segments
                                .iter()
                                .position(|s| Arc::ptr_eq(s, &arena))
                                .unwrap_or_else(|| {
                                    out.segments.push(arena);
                                    out.segments.len() - 1
                                }) as u32;
                            last = Some((name.as_ptr(), at));
                            at
                        }
                    };
                    let arena = &out.segments[segment as usize];
                    let start = info.shm_offset + info.offset;
                    let end = start.checked_add(info.nbytes);
                    if nbytes > 0
                        && (start < 0 || end.is_none_or(|end| end as usize > arena.size()))
                    {
                        return Err(PyValueError::new_err(format!(
                            "descriptor for {}: {} bytes at {start} run past its {} byte segment",
                            info.uuid,
                            info.nbytes,
                            arena.size(),
                        )));
                    }
                    let src = if nbytes > 0 {
                        unsafe { arena.as_mut_ptr().add(start as usize).cast_const() }
                    } else {
                        std::ptr::null()
                    };
                    (src, None, segment)
                }
            };
            out.did_work |= nbytes > 0;
            out.host_bytes += nbytes;
            out.rows.push(ReadRow {
                src,
                file,
                file_offset: info.offset as u64,
                nbytes,
                dtype: info.scalar,
                dims_at: out.dims_flat.len() as u32,
                ndim: info.dims.len() as u32,
                segment,
            });
            out.dims_flat.extend_from_slice(&info.dims);
            out.owners.push(rid.clone());
            out.infos.push(info);
        }
        Ok(out)
    }

    fn can_copy_from(&self, facts: &TensorFacts) -> bool {
        match facts.device_type {
            DEVICE_CPU => true,
            DEVICE_CUDA => {
                self.cuda.is_some()
                    && self.d2h_stream.is_some()
                    && facts.device_index == self.device_index
            }
            _ => false,
        }
    }

    /// The names the store keys dtypes by, one Python `str()` per dtype ever.
    fn remember_dtype(&self, info: &ReadInfo<'_>) -> PyResult<()> {
        if self.dtype_names.lock().expect("dtype names lock").contains_key(&info.scalar) {
            return Ok(());
        }
        let name = info.dtype.str()?;
        let name = name.to_str()?;
        let name = name.strip_prefix("torch.").unwrap_or(name).to_string();
        self.dtype_names.lock().expect("dtype names lock").insert(info.scalar, name);
        Ok(())
    }

    /// Place and copy the rows from `pass.next`, stopping at a failure, which
    /// leaves the rest unplaced. `may_block` false stops at the first row that
    /// needs more than an existing segment's free space, for the caller to run
    /// again without the GIL.
    fn copy_stages(&self, rows: &mut [StageRow], pass: &mut StagePass, may_block: bool) {
        while pass.next < rows.len() {
            let row = &mut rows[pass.next];
            let place = match self.capacity.reserve_now(row.nbytes) {
                Some((seg, offset, base)) => Reserved::Slot { seg, offset, base },
                None if !may_block => return,
                None => self.capacity.reserve(row.nbytes),
            };
            pass.next += 1;
            let done = match place {
                Reserved::Slot { seg, offset, base } => self
                    .copy_to_slot(row, unsafe { base.add(offset) })
                    .map(|queued| {
                        pass.queued |= queued;
                        row.placed = Some((seg as u32, offset as u64));
                    })
                    .inspect_err(|_| self.capacity.return_slot(seg as u32, offset as u64)),
                Reserved::Spill => self.spill_row(row).map(|()| row.placed = Some((IN_FILE, 0))),
                Reserved::Failed(message) => Err(StageError {
                    errno: None,
                    message,
                }),
            };
            if let Err(e) = done {
                pass.error = Some(e);
                return;
            }
        }
    }

    fn copy_to_slot(&self, row: &StageRow, dst: *mut u8) -> Result<bool, StageError> {
        if row.nbytes == 0 {
            return Ok(false);
        }
        if row.on_device {
            let cuda = self.cuda.as_ref().expect("checked by can_copy_from");
            let stream = self.d2h_stream.expect("checked by can_copy_from");
            cuda.copy(dst, row.src, row.nbytes, CUDA_DEVICE_TO_HOST, stream)
                .map_err(|message| StageError {
                    errno: None,
                    message,
                })?;
            return Ok(true);
        }
        unsafe { std::ptr::copy_nonoverlapping(row.src, dst, row.nbytes) };
        Ok(false)
    }

    /// Slower, never fails for lack of an arena: a device source comes down first
    /// and waits for its copy, then the bytes go to the row's own file.
    fn spill_row(&self, row: &StageRow) -> Result<(), StageError> {
        if !row.on_device {
            let data = if row.nbytes == 0 {
                &[][..]
            } else {
                unsafe { std::slice::from_raw_parts(row.src, row.nbytes) }
            };
            return self.capacity.spill.write(row.uuid, data);
        }
        let cuda = self.cuda.as_ref().expect("checked by can_copy_from");
        let stream = self.d2h_stream.expect("checked by can_copy_from");
        let mut data = vec![0u8; row.nbytes];
        if row.nbytes > 0 {
            let copied = cuda
                .copy(data.as_mut_ptr(), row.src, row.nbytes, CUDA_DEVICE_TO_HOST, stream)
                .and_then(|()| cuda.synchronize(stream));
            copied.map_err(|message| StageError {
                errno: None,
                message,
            })?;
        }
        self.capacity.spill.write(row.uuid, &data)
    }
}

fn spill_pyerr(py: Python<'_>, e: StageError) -> PyErr {
    match e.errno {
        Some(errno) => {
            let class = py.get_type::<pyo3::exceptions::PyOSError>();
            match class.call1((errno, e.message.clone())) {
                Ok(err) => PyErr::from_value(err),
                Err(_) => PyRuntimeError::new_err(e.message),
            }
        }
        None => PyRuntimeError::new_err(e.message),
    }
}

/// How far a stage has got, across its pass with the GIL and the one without.
#[derive(Default)]
struct StagePass {
    next: usize,
    queued: bool,
    error: Option<StageError>,
}

/// Which rows read as one copy: at least two of one shape, dtype and segment,
/// each small enough that gathering beats a copy per item.
fn gather_groups(rows: &[ReadRow], dims_flat: &[i64], item_bytes: usize) -> Vec<GatherGroup> {
    let dims = |r: &ReadRow| &dims_flat[r.dims_at as usize..(r.dims_at + r.ndim) as usize];
    let mut groups: Vec<GatherGroup> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.nbytes == 0 || row.nbytes > item_bytes || row.file.is_some() {
            continue;
        }
        let found = groups.iter().position(|g| {
            let first = &rows[g.rows[0] as usize];
            first.nbytes == row.nbytes
                && first.segment == row.segment
                && first.dtype == row.dtype
                && dims(first) == dims(row)
        });
        match found {
            Some(g) => groups[g].rows.push(i as u32),
            None if groups.len() < MAX_GATHER_GROUPS => groups.push(GatherGroup {
                rows: vec![i as u32],
            }),
            None => {}
        }
    }
    groups.retain(|g| g.rows.len() >= 2);
    groups
}

/// Why a read's copies could not run.
enum ReadError {
    Abi(AbiError),
    Copy(String),
    File(std::io::Error),
}

impl From<AbiError> for ReadError {
    fn from(e: AbiError) -> Self {
        ReadError::Abi(e)
    }
}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        ReadError::File(e)
    }
}

impl ReadError {
    fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            ReadError::Abi(e) => e.into_pyerr(py),
            ReadError::Copy(message) => PyRuntimeError::new_err(message),
            ReadError::File(e) => e.into(),
        }
    }
}

struct ReadCopied {
    /// Parallel to the job's rows.
    tensors: Vec<OwnedTensor>,
    /// The pinned gather buffers, which the copies read until the stream's event.
    hosts: Vec<OwnedTensor>,
}

struct ReadJob<'a> {
    rows: &'a [ReadRow],
    dims_flat: &'a [i64],
    groups: &'a [GatherGroup],
    device_type: i32,
    device_index: i32,
    stream: Option<(&'a Cudart, usize)>,
}

impl ReadJob<'_> {
    fn dims(&self, row: &ReadRow) -> &[i64] {
        &self.dims_flat[row.dims_at as usize..(row.dims_at + row.ndim) as usize]
    }

    fn check_size(row: &ReadRow, got: i64) -> Result<(), ReadError> {
        if got == row.nbytes as i64 {
            return Ok(());
        }
        Err(ReadError::Copy(format!(
            "a descriptor says {} bytes, but its dims and dtype make {got}",
            row.nbytes
        )))
    }

    /// Every allocation happens before the first copy, so a failed allocation
    /// leaves nothing in flight into a tensor about to be dropped.
    fn run(&self, abi: &dyn TorchAbi) -> Result<ReadCopied, ReadError> {
        let mut out: Vec<Option<OwnedTensor>> = Vec::new();
        out.resize_with(self.rows.len(), || None);
        let mut in_group = vec![false; self.rows.len()];
        let mut gathered = Vec::with_capacity(self.groups.len());
        for group in self.groups {
            let first = &self.rows[group.rows[0] as usize];
            let k = group.rows.len();
            let mut dims = Vec::with_capacity(first.ndim as usize + 1);
            dims.push(k as i64);
            dims.extend_from_slice(self.dims(first));
            let device = abi.empty(TensorSpec {
                dims: &dims,
                dtype: first.dtype,
                device_type: self.device_type,
                device_index: self.device_index,
                pinned: false,
            })?;
            Self::check_size(first, device.nbytes / k as i64)?;
            let host = abi.empty(TensorSpec {
                dims: &[device.nbytes],
                dtype: SCALAR_UINT8,
                device_type: DEVICE_CPU,
                device_index: -1,
                pinned: true,
            })?;
            for &r in &group.rows {
                in_group[r as usize] = true;
            }
            gathered.push((group, device, host));
        }
        for (row, (slot, grouped)) in self.rows.iter().zip(out.iter_mut().zip(&in_group)) {
            if *grouped {
                continue;
            }
            let t = abi.empty(TensorSpec {
                dims: self.dims(row),
                dtype: row.dtype,
                device_type: self.device_type,
                device_index: self.device_index,
                pinned: false,
            })?;
            Self::check_size(row, t.nbytes)?;
            *slot = Some(t);
        }
        // Views allocate no device memory, so they can precede the copies.
        let mut hosts = Vec::with_capacity(gathered.len());
        let mut copies = Vec::with_capacity(gathered.len());
        for (group, device, host) in gathered {
            let views = abi.unbind(&device, group.rows.len())?;
            for (view, &r) in views.into_iter().zip(&group.rows) {
                out[r as usize] = Some(view);
            }
            let item = self.rows[group.rows[0] as usize].nbytes;
            for (k, &r) in group.rows.iter().enumerate() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.rows[r as usize].src,
                        host.data.add(k * item),
                        item,
                    );
                }
            }
            copies.push((device, host.data.cast_const()));
            hosts.push(host);
        }
        // A file row on a device comes to a host buffer first, which its queued
        // copy then reads. On a host device it goes straight to the tensor.
        let mut from_file = vec![std::ptr::null(); self.rows.len()];
        for (i, row) in self.rows.iter().enumerate() {
            let Some(path) = row.file.as_deref().filter(|_| row.nbytes > 0) else {
                continue;
            };
            if self.stream.is_some() {
                let host = abi.empty(TensorSpec {
                    dims: &[row.nbytes as i64],
                    dtype: SCALAR_UINT8,
                    device_type: DEVICE_CPU,
                    device_index: -1,
                    pinned: false,
                })?;
                let dst = unsafe { std::slice::from_raw_parts_mut(host.data, row.nbytes) };
                SpillFiles::read(path, row.file_offset, dst)?;
                from_file[i] = host.data.cast_const();
                hosts.push(host);
            } else {
                let data = out[i].as_ref().expect("allocated above").data;
                let dst = unsafe { std::slice::from_raw_parts_mut(data, row.nbytes) };
                SpillFiles::read(path, row.file_offset, dst)?;
            }
        }
        if let Err(message) = self.queue_copies(&copies, &out, &in_group, &from_file) {
            // Some copies may be in flight into tensors the error is about to free.
            if let Some((cuda, stream)) = self.stream {
                let _ = cuda.synchronize(stream);
            }
            return Err(ReadError::Copy(message));
        }
        Ok(ReadCopied {
            tensors: out.into_iter().map(|t| t.expect("every row filled")).collect(),
            hosts,
        })
    }

    fn queue_copies(
        &self,
        gathers: &[(OwnedTensor, *const u8)],
        out: &[Option<OwnedTensor>],
        in_group: &[bool],
        from_file: &[*const u8],
    ) -> Result<(), String> {
        if let Some((cuda, stream)) = self.stream {
            for (device, src) in gathers {
                let n = device.nbytes as usize;
                cuda.copy(device.data, *src, n, CUDA_HOST_TO_DEVICE, stream)?;
            }
        }
        for (i, ((row, t), grouped)) in self.rows.iter().zip(out).zip(in_group).enumerate() {
            if *grouped || row.nbytes == 0 {
                continue;
            }
            let dst = t.as_ref().expect("allocated above").data;
            let src = if row.file.is_some() { from_file[i] } else { row.src };
            match self.stream {
                Some((cuda, stream)) => cuda.copy(dst, src, row.nbytes, CUDA_HOST_TO_DEVICE, stream)?,
                None if row.file.is_some() => {} // read straight into `dst`
                None => unsafe { std::ptr::copy_nonoverlapping(src, dst, row.nbytes) },
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(nbytes: usize, dtype: i32, segment: u32, dims_at: u32) -> ReadRow {
        ReadRow {
            src: std::ptr::null(),
            file: None,
            file_offset: 0,
            nbytes,
            dtype,
            dims_at,
            ndim: 1,
            segment,
        }
    }

    #[test]
    fn a_group_needs_two_rows_of_one_shape_dtype_segment_and_a_small_size() {
        let dims = [4, 2];
        let rows = [
            row(16, 1, 0, 0), // with the next
            row(16, 1, 0, 0),
            row(16, 1, 0, 1), // another shape
            row(16, 2, 0, 0), // another dtype
            row(16, 1, 1, 0), // another segment
            row(0, 1, 0, 0),  // no bytes
            row(32, 1, 0, 0), // past the item limit
        ];
        let groups = gather_groups(&rows, &dims, 16);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].rows, vec![0, 1]);
        assert!(gather_groups(&rows, &dims, 0).is_empty(), "a limit of 0 turns it off");
    }

    #[test]
    fn a_batch_of_all_different_shapes_stops_looking_for_groups() {
        let dims: Vec<i64> = (0..2 * MAX_GATHER_GROUPS as i64).collect();
        let rows: Vec<_> = (0..dims.len() as u32).map(|i| row(8, 1, 0, i)).collect();
        assert!(gather_groups(&rows, &dims, 16).is_empty());
    }
}
