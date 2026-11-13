//! The per-tensor hot paths of `ArenaShmCommunicationManager`
//! (`mstar/communication/arena.py`) with no Python call per tensor: staging a
//! batch into the arena and reading a batch out of peers' segments.
//!
//! Each runs in three passes: read the Python objects and claim the work with
//! the GIL, copy without it, then publish with it. Whatever this cannot do comes
//! back to the manager as a row index for its Python path: a segment growth, a
//! non-contiguous source, a tensor on another device. So this module never pins,
//! grows, spills or backpressures, and those stay in one place.
//!
//! Device copies use cudart's `cudaMemcpyAsync` from function pointers the
//! manager resolved with `_cudart()`, and only where it says the platform is
//! CUDA: this file links no vendor library, and on ROCm or XPU every device row
//! is deferred. They go on the manager's copy streams, in program order, so its
//! stream event and trailing sync cover them as they do torch's own copies.

use std::os::raw::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString, PyTuple};
use rustc_hash::FxHashMap;

use crate::bindings::{PySegmentedShmArena, PyShmArena};
use crate::shm::{SegmentedShmArena, ShmArena};
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
}

/// Where a staged uuid's bytes are. `seg == PENDING` is a claim: some thread is
/// staging it now and has yet to publish.
#[derive(Clone, Copy)]
struct Slot {
    seg: u32,
    offset: u64,
    staged_at: f64,
}

const PENDING: u32 = u32::MAX;

#[derive(Default)]
struct SlotTable(Mutex<FxHashMap<u64, Slot>>);

impl SlotTable {
    fn lock(&self) -> std::sync::MutexGuard<'_, FxHashMap<u64, Slot>> {
        self.0.lock().expect("slot table lock")
    }
}

/// The manager's `_arena_locs`: uuid -> `(segment_idx, offset)` for sender-side
/// reclaim. Here so a staged batch records its slots with no Python object per
/// tensor. A uuid still being staged counts as present, and freeing one only
/// drops the claim: its stager finds that out and frees the slot itself.
#[pyclass(name = "ArenaSlots", frozen)]
pub struct PyArenaSlots {
    /// None for an entity with no arena of its own: it stages nothing.
    arena: Option<Arc<SegmentedShmArena>>,
    table: Arc<SlotTable>,
}

#[pymethods]
impl PyArenaSlots {
    #[new]
    #[pyo3(signature = (arena))]
    fn new(arena: Option<PyRef<'_, PySegmentedShmArena>>) -> Self {
        Self {
            arena: arena.map(|a| a.share()),
            table: Arc::new(SlotTable::default()),
        }
    }

    fn __len__(&self) -> usize {
        self.table.lock().len()
    }

    fn __contains__(&self, uuid: u64) -> bool {
        self.table.lock().contains_key(&uuid)
    }

    fn __getitem__(&self, uuid: u64) -> PyResult<(u32, u64)> {
        match self.table.lock().get(&uuid) {
            Some(s) if s.seg != PENDING => Ok((s.seg, s.offset)),
            _ => Err(PyKeyError::new_err(uuid)),
        }
    }

    /// Record a slot the Python path reserved. False if `uuid` already has one
    /// or is being staged, and the caller gives its slot back.
    fn claim(&self, uuid: u64, segment: u32, offset: u64, staged_at: f64) -> bool {
        let mut table = self.table.lock();
        if table.contains_key(&uuid) {
            return false;
        }
        let slot = Slot {
            seg: segment,
            offset,
            staged_at,
        };
        table.insert(uuid, slot);
        true
    }

    /// Back to the arena, for each of `uuids` that has a slot. GIL released: a
    /// growth holds the segments lock across an mmap.
    fn free(&self, py: Python<'_>, uuids: Vec<u64>) {
        py.detach(|| {
            let mut slots = Vec::with_capacity(uuids.len());
            {
                let mut table = self.table.lock();
                for uuid in &uuids {
                    if let Some(s) = table.remove(uuid) {
                        if s.seg != PENDING {
                            slots.push(s);
                        }
                    }
                }
            }
            if let Some(arena) = &self.arena {
                for s in slots {
                    arena.free(s.seg as usize, s.offset as usize);
                }
            }
        });
    }

    /// The slots staged before `cutoff`, for the TTL backstop.
    fn older_than(&self, cutoff: f64) -> Vec<u64> {
        let table = self.table.lock();
        table
            .iter()
            .filter(|(_, s)| s.seg != PENDING && s.staged_at < cutoff)
            .map(|(u, _)| *u)
            .collect()
    }
}

/// One tensor being staged.
struct StageRow {
    index: u32,
    uuid: u64,
    src: *const u8,
    nbytes: usize,
    on_device: bool,
    placed: Option<(u32, u64)>,
}

// `src` is kept valid by the batch's `keep` list, across the pass that uses it.
unsafe impl Send for StageRow {}

/// One tensor being read, once its Python descriptor has been read.
struct ReadRow {
    src: *const u8,
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
    segment: Bound<'py, PyString>,
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
        segment: info.getattr(intern!(py, "shm_segment"))?.downcast_into()?,
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
    /// Columns parallel to the call's uuids, `-1` where nothing was staged;
    /// present only when asked for.
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

/// What a read did. Either `missing` names peer segments to map and the call
/// did nothing, or the rows are in the store but for `deferred`. `keep` must
/// outlive the batch's device copies.
#[pyclass(get_all)]
pub struct ReadOut {
    missing: Vec<String>,
    did_work: bool,
    deferred: Vec<u32>,
    keep: Vec<Py<PyAny>>,
}

/// The copies themselves, with their streams and the peer mappings. One per
/// manager; every method takes the GIL only for its first and last pass.
#[pyclass(name = "ArenaCopier", frozen)]
pub struct PyArenaCopier {
    /// None for an entity with no arena of its own: it reads, and defers every
    /// stage to the manager's spill.
    arena: Option<Arc<SegmentedShmArena>>,
    slots: Arc<SlotTable>,
    /// Segments the manager has pinned and named: one a concurrent growth has
    /// only just added is not used until then.
    usable_segments: AtomicUsize,
    peers: RwLock<FxHashMap<String, Arc<ShmArena>>>,
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
impl PyArenaCopier {
    #[new]
    #[pyo3(signature = (slots, usable_segments, device_index, gather_item_bytes, cuda=None, d2h_stream=None, h2d_stream=None))]
    fn new(
        slots: PyRef<'_, PyArenaSlots>,
        usable_segments: usize,
        device_index: i32,
        gather_item_bytes: usize,
        cuda: Option<(usize, usize, usize, usize)>,
        d2h_stream: Option<usize>,
        h2d_stream: Option<usize>,
    ) -> PyResult<Self> {
        if abi().is_none() {
            return Err(PyRuntimeError::new_err("torch_abi_load has not run"));
        }
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
        Ok(Self {
            arena: slots.arena.clone(),
            slots: slots.table.clone(),
            usable_segments: AtomicUsize::new(usable_segments),
            peers: RwLock::new(FxHashMap::default()),
            cuda,
            device_index,
            d2h_stream,
            h2d_stream,
            gather_item_bytes,
            dtype_names: Mutex::new(FxHashMap::default()),
        })
    }

    fn set_usable_segments(&self, n: usize) {
        self.usable_segments.store(n, Ordering::Release);
    }

    fn add_peer(&self, name: String, arena: PyRef<'_, PyShmArena>) {
        self.peers.write().expect("peers lock").insert(name, arena.share());
    }

    fn remove_peer(&self, name: &str) {
        self.peers.write().expect("peers lock").remove(name);
    }

    /// Stage `uuids` from `store`: reserve, copy, publish the placement. Run it
    /// in the D2H stream's context, and hold the result through the trailing
    /// sync that lands the copies before this batch's descriptors can ship.
    fn stage(
        &self,
        py: Python<'_>,
        store: PyRef<'_, TensorBookkeeping>,
        uuids: Vec<u64>,
        now: f64,
        columns: bool,
    ) -> PyResult<StageOut> {
        let abi = abi().ok_or_else(|| PyRuntimeError::new_err("torch_abi_load has not run"))?;
        let store = store.share();
        let mut rows = Vec::with_capacity(uuids.len());
        let mut keep = Vec::with_capacity(uuids.len());
        let mut deferred = Vec::new();
        let (mut any_device, mut host_bytes) = (false, 0usize);
        {
            let bk = store.lock().expect("bookkeeping lock");
            let mut slots = self.slots.lock();
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

        let (queued, error) = if any_device || host_bytes > KEEP_GIL_MAX_BYTES {
            py.detach(|| self.copy_stages(&mut rows))
        } else {
            self.copy_stages(&mut rows)
        };

        let mut loose = Vec::new();
        let mut publish_error = None;
        {
            let mut bk = store.lock().expect("bookkeeping lock");
            let mut slots = self.slots.lock();
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
                    loose.push((seg, offset));
                    continue;
                };
                slot.seg = seg;
                slot.offset = offset;
                if let Err(e) = bk.publish_staged(row.uuid, seg as usize, offset as i64) {
                    publish_error = Some(e);
                }
            }
        }
        if let (false, Some(arena)) = (loose.is_empty(), &self.arena) {
            py.detach(|| {
                // A queued copy could land in a slot given to someone else.
                if let (true, Some(cuda), Some(stream)) = (queued, &self.cuda, self.d2h_stream) {
                    unsafe { (cuda.stream_synchronize)(stream) };
                    unsafe { (cuda.get_last_error)() };
                }
                for (seg, offset) in loose {
                    arena.free(seg as usize, offset as usize);
                }
            });
        }

        let mut out = StageOut {
            queued,
            deferred,
            error: error.or(publish_error),
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
                    segments[row.index as usize] = i64::from(seg);
                    offsets[row.index as usize] = offset as i64;
                    nbytes[row.index as usize] = row.nbytes as i64;
                }
            }
            (out.segments, out.offsets, out.nbytes) = (Some(segments), Some(offsets), Some(nbytes));
        }
        for row in &rows {
            if row.placed.is_none() && out.error.is_none() {
                out.deferred.push(row.index);
            }
        }
        Ok(out)
    }

    /// Read `infos` from peers' segments into new tensors on the manager's
    /// device and put them in `store`, each with `refs` references. Run it in
    /// the H2D stream's context, which tags the new device blocks. `host_max`:
    /// a host-device row at least this big is left to the caller's HostCopier.
    #[pyo3(signature = (store, rids, infos, refs, device_type, host_max=None))]
    #[allow(clippy::too_many_arguments)]
    fn read<'py>(
        &self,
        py: Python<'py>,
        store: PyRef<'py, TensorBookkeeping>,
        rids: Vec<RidKey>,
        infos: &Bound<'py, PyList>,
        refs: i64,
        device_type: i32,
        host_max: Option<usize>,
    ) -> PyResult<ReadOut> {
        let abi = abi().ok_or_else(|| PyRuntimeError::new_err("torch_abi_load has not run"))?;
        let on_device = device_type != DEVICE_CPU;
        let stream = match (on_device, &self.cuda, self.h2d_stream) {
            (false, _, _) => None,
            (true, Some(cuda), Some(stream)) => Some((cuda, stream)),
            _ => return Err(PyRuntimeError::new_err("no CUDA copy stream to read onto")),
        };

        // First pass: the descriptors, and which segment each one reads.
        let mut parsed = Vec::with_capacity(infos.len());
        let mut rows = Vec::with_capacity(infos.len());
        let mut dims_flat = Vec::with_capacity(infos.len());
        let mut owners = Vec::with_capacity(infos.len());
        let mut deferred = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        let mut segments: Vec<Arc<ShmArena>> = Vec::new();
        let (mut did_work, mut host_bytes) = (false, 0usize);
        {
            let mut last: Option<(*mut pyo3::ffi::PyObject, u32)> = None;
            let mut rids = rids.into_iter();
            for (i, info) in infos.iter().enumerate() {
                let rid = rids
                    .next()
                    .ok_or_else(|| PyValueError::new_err("rids and infos must be parallel"))?;
                let info = read_info(&info, abi)?;
                let nbytes = usize::try_from(info.nbytes)
                    .map_err(|_| PyValueError::new_err("a descriptor's nbytes is negative"))?;
                if !on_device && host_max.is_some_and(|max| nbytes >= max) {
                    deferred.push(i as u32);
                    continue;
                }
                let segment = match last {
                    Some((ptr, at)) if ptr == info.segment.as_ptr() => at,
                    _ => {
                        let name = info.segment.to_str()?;
                        let found = self.peers.read().expect("peers lock").get(name).cloned();
                        let Some(arena) = found else {
                            if !missing.iter().any(|m| m == name) {
                                missing.push(name.to_string());
                            }
                            continue;
                        };
                        let at = match segments.iter().position(|s| Arc::ptr_eq(s, &arena)) {
                            Some(at) => at,
                            None => {
                                segments.push(arena);
                                segments.len() - 1
                            }
                        } as u32;
                        last = Some((info.segment.as_ptr(), at));
                        at
                    }
                };
                let arena = &segments[segment as usize];
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
                did_work |= nbytes > 0;
                host_bytes += nbytes;
                rows.push(ReadRow {
                    src: if nbytes > 0 {
                        unsafe { arena.as_mut_ptr().add(start as usize).cast_const() }
                    } else {
                        std::ptr::null()
                    },
                    nbytes,
                    dtype: info.scalar,
                    dims_at: dims_flat.len() as u32,
                    ndim: info.dims.len() as u32,
                    segment,
                });
                dims_flat.extend_from_slice(&info.dims);
                owners.push(rid);
                parsed.push(info);
            }
        }
        if !missing.is_empty() {
            return Ok(ReadOut {
                missing,
                did_work: false,
                deferred: Vec::new(),
                keep: Vec::new(),
            });
        }

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
        let copied = if on_device || host_bytes > KEEP_GIL_MAX_BYTES {
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
                    shm_segment: Some(sym(&info.segment)?),
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
            missing: Vec::new(),
            did_work,
            deferred,
            keep,
        })
    }
}

impl PyArenaCopier {
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

    /// Reserve and copy each row. Stops at a failed copy; a row with nothing
    /// reserved stays unplaced, for the manager's growth and spill path.
    fn copy_stages(&self, rows: &mut [StageRow]) -> (bool, Option<String>) {
        let Some(arena) = &self.arena else {
            return (false, None);
        };
        let usable = self.usable_segments.load(Ordering::Acquire);
        let mut queued = false;
        for row in rows.iter_mut() {
            let Some((seg, offset, base)) = arena.reserve_existing(row.nbytes, usable) else {
                continue;
            };
            if row.nbytes > 0 {
                let dst = unsafe { base.add(offset) };
                if row.on_device {
                    let cuda = self.cuda.as_ref().expect("checked by can_copy_from");
                    let stream = self.d2h_stream.expect("checked by can_copy_from");
                    if let Err(e) = cuda.copy(dst, row.src, row.nbytes, CUDA_DEVICE_TO_HOST, stream)
                    {
                        arena.free(seg, offset);
                        return (queued, Some(e));
                    }
                    queued = true;
                } else {
                    unsafe { std::ptr::copy_nonoverlapping(row.src, dst, row.nbytes) };
                }
            }
            row.placed = Some((seg as u32, offset as u64));
        }
        (queued, None)
    }
}

/// Which rows read as one copy: at least two of one shape, dtype and segment,
/// each small enough that gathering beats a copy per item.
fn gather_groups(rows: &[ReadRow], dims_flat: &[i64], item_bytes: usize) -> Vec<GatherGroup> {
    let dims = |r: &ReadRow| &dims_flat[r.dims_at as usize..(r.dims_at + r.ndim) as usize];
    let mut groups: Vec<GatherGroup> = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if row.nbytes == 0 || row.nbytes > item_bytes {
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

    fn check_size(row: &ReadRow, got: i64) -> Result<(), AbiError> {
        if got == row.nbytes as i64 {
            return Ok(());
        }
        Err(AbiError {
            out_of_memory: false,
            message: format!(
                "a descriptor says {} bytes, but its dims and dtype make {got}",
                row.nbytes
            ),
        })
    }

    /// Every allocation happens before the first copy, so a failed allocation
    /// leaves nothing in flight into a tensor about to be dropped.
    fn run(&self, abi: &dyn TorchAbi) -> Result<ReadCopied, AbiError> {
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
        if let Err(message) = self.queue_copies(&copies, &out, &in_group) {
            // Some copies may be in flight into tensors the error is about to free.
            if let Some((cuda, stream)) = self.stream {
                unsafe { (cuda.stream_synchronize)(stream) };
                unsafe { (cuda.get_last_error)() };
            }
            return Err(AbiError {
                out_of_memory: false,
                message,
            });
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
    ) -> Result<(), String> {
        if let Some((cuda, stream)) = self.stream {
            for (device, src) in gathers {
                let n = device.nbytes as usize;
                cuda.copy(device.data, *src, n, CUDA_HOST_TO_DEVICE, stream)?;
            }
        }
        for ((row, t), grouped) in self.rows.iter().zip(out).zip(in_group) {
            if *grouped || row.nbytes == 0 {
                continue;
            }
            let dst = t.as_ref().expect("allocated above").data;
            match self.stream {
                Some((cuda, stream)) => {
                    cuda.copy(dst, row.src, row.nbytes, CUDA_HOST_TO_DEVICE, stream)?
                }
                None => unsafe { std::ptr::copy_nonoverlapping(row.src, dst, row.nbytes) },
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
