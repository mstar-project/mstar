//! mstar's Rust components, one crate: each module is an independent
//! capability behind its own opt-in flag, and this file is the PyO3
//! surface the Python wrappers drive. Currently: `communicator.rs` (the
//! ZMQ PUSH/PULL control mesh, as a transport + codec split) and `shm.rs`
//! (the shared-memory tensor arena). The crate is not limited to these —
//! new components land as new modules with their own bindings. Also
//! usable as a plain Rust library (rlib) by Rust-side consumers.
//! Build: `maturin develop --release` in rust/.

pub mod communicator;
pub mod core;
pub mod shm;

use std::os::raw::{c_int, c_void};
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use communicator::{RawZmqCommunicator, RecvEvent};
use core::graph::CompiledWalk;
use core::sched::{BatchFilter, MicroScheduler, ReadyEntry, SchedulingType};
use core::tensor::TensorRef;
use core::walk::{IncomingInput, RouteEvent, WalkState};
use core::WalkSet;
use shm::{SegmentedShmArena, ShmArena};

/// Opaque byte frames over ZMQ PUSH/PULL: ipc or tcp endpoints, lazily-cached
/// peers, and wakeup-fd polling (an eventfd wakes `recv_or_wake` instantly).
/// Encoding is the caller's (pickle today; msgpack by swapping the codec).
#[pyclass(name = "ZmqCommunicator")]
struct PyZmqCommunicator {
    inner: RawZmqCommunicator,
}

#[pymethods]
impl PyZmqCommunicator {
    /// Bind the PULL inbox at `ipc://{dir}/{my_id}.ipc`.
    #[new]
    fn new(my_id: &str, dir: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RawZmqCommunicator::bind(my_id, dir)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    /// Bind at an explicit zmq endpoint (e.g. `tcp://0.0.0.0:5701`).
    #[staticmethod]
    fn bind_endpoint(my_id: &str, endpoint: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RawZmqCommunicator::bind_endpoint(my_id, endpoint)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    fn last_endpoint(&self) -> PyResult<String> {
        self.inner
            .last_endpoint()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn register_peer(&self, peer_id: &str, endpoint: &str) {
        self.inner.register_peer(peer_id, endpoint);
    }

    /// Poll `fd` (an eventfd) alongside the inbox; when readable,
    /// `recv_or_wake` returns ("wake", None) immediately. The registrant
    /// reads/clears the fd (level-triggered).
    fn register_wakeup_fd(&self, fd: i32) {
        self.inner.register_wakeup_fd(fd);
    }

    /// Released-GIL send: a PUSH send blocks when the peer is at its
    /// high-water mark, and blocking with the GIL held would freeze every
    /// Python thread in the process (pyzmq releases it here too).
    fn send(&self, py: Python<'_>, peer_id: &str, data: &[u8]) -> PyResult<()> {
        py.allow_threads(|| self.inner.send(peer_id, data))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn try_recv<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner.try_recv().map(|b| PyBytes::new(py, &b))
    }

    /// Drain all currently-queued frames in one call — one GIL round-trip
    /// per batch instead of one `try_recv` per message.
    fn drain<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        let frames = py.allow_threads(|| self.inner.drain());
        frames.iter().map(|b| PyBytes::new(py, b)).collect()
    }

    fn recv_timeout<'py>(&self, py: Python<'py>, timeout_ms: u64) -> Option<Bound<'py, PyBytes>> {
        py.allow_threads(|| self.inner.recv_timeout(Duration::from_millis(timeout_ms)))
            .map(|b| PyBytes::new(py, &b))
    }

    /// ("msg", bytes) | ("wake", None) | ("timeout", None). Raises if a
    /// registered wakeup fd is closed/invalid (the registrant's bug — loud,
    /// instead of degrading blocking waits into an instant-timeout spin).
    fn recv_or_wake<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: u64,
    ) -> PyResult<(&'static str, Option<Bound<'py, PyBytes>>)> {
        let ev = py.allow_threads(|| self.inner.recv_or_wake(Duration::from_millis(timeout_ms)));
        Ok(match ev {
            RecvEvent::Message(b) => ("msg", Some(PyBytes::new(py, &b))),
            RecvEvent::Wake => ("wake", None),
            RecvEvent::WakeFdError => {
                return Err(PyRuntimeError::new_err(
                    "a registered wakeup fd is closed/invalid (POLLERR/\
                     POLLNVAL): fix the EventWakeup lifetime"));
            }
            RecvEvent::Timeout => ("timeout", None),
        })
    }
}

/// Shared-memory tensor arena for cross-process transport. Producer:
/// `create(name, size)` -> `reserve(nbytes)` -> `torch.frombuffer(
/// memoryview(arena)[off:off+n], dtype=..).copy_(cpu_tensor)`; send the
/// offset descriptor; `free(off)` on reclaim. Consumer: `open(name)` and
/// `torch.frombuffer(memoryview(arena)[off:off+n], ..)` (then H2D).
#[pyclass(name = "ShmArena")]
struct PyShmArena {
    arena: ShmArena,
}

#[pymethods]
impl PyShmArena {
    #[staticmethod]
    fn create(name: &str, size: usize) -> PyResult<Self> {
        Ok(Self {
            arena: ShmArena::create(name, size)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[staticmethod]
    fn open(name: &str) -> PyResult<Self> {
        Ok(Self {
            arena: ShmArena::open(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    fn reserve(&self, nbytes: usize) -> PyResult<usize> {
        self.arena
            .reserve(nbytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn free(&self, offset: usize) -> bool {
        self.arena.free(offset)
    }

    #[getter]
    fn size(&self) -> usize {
        self.arena.size()
    }

    #[getter]
    fn bytes_free(&self) -> usize {
        self.arena.bytes_free()
    }

    /// `(base_ptr, len)` for the pinning hook — `cudaHostRegister` the
    /// mapping once (e.g. `torch.cuda.cudart().cudaHostRegister(ptr, len, 0)`).
    fn ptr_len(&self) -> (usize, usize) {
        (self.arena.as_mut_ptr() as usize, self.arena.size())
    }

    fn close(&mut self) {
        self.arena.close();
    }

    /// Whole arena as a writable memoryview -> zero-copy `torch.frombuffer`.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyValueError::new_err("null buffer view"));
        }
        let borrow = slf.borrow();
        let ptr = borrow.arena.as_mut_ptr() as *mut c_void;
        let len = borrow.arena.size() as ffi::Py_ssize_t;
        let ret = ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr, len, 0, flags);
        if ret != 0 {
            Err(PyErr::fetch(slf.py()))
        } else {
            Ok(())
        }
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

/// One segment of a `SegmentedShmArena`, exposed with the buffer protocol so
/// staging stays zero-copy per segment. The mapping never moves, so a
/// memoryview (and a CUDA host-registration of the segment) stays valid for
/// the segment's lifetime.
#[pyclass(name = "ShmSegment")]
struct PyShmSegment {
    seg: std::sync::Arc<ShmArena>,
}

#[pymethods]
impl PyShmSegment {
    #[getter]
    fn size(&self) -> usize {
        self.seg.size()
    }

    /// `(base_ptr, len)` for the pinning hook (see `ShmArena::ptr_len`).
    fn ptr_len(&self) -> (usize, usize) {
        (self.seg.as_mut_ptr() as usize, self.seg.size())
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyValueError::new_err("null buffer view"));
        }
        let borrow = slf.borrow();
        let ptr = borrow.seg.as_mut_ptr() as *mut c_void;
        let len = borrow.seg.size() as ffi::Py_ssize_t;
        let ret = ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr, len, 0, flags);
        if ret != 0 {
            Err(PyErr::fetch(slf.py()))
        } else {
            Ok(())
        }
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

/// Grow-by-segments producer arena with uuid-grouped reclaim.
/// `reserve(n) -> (segment_idx, offset)`; descriptors carry
/// `segment_name(idx)` so consumers keep opening plain `ShmArena`s by name.
/// Segments are created once and never move — registration-friendly.
#[pyclass(name = "SegmentedShmArena")]
struct PySegmentedShmArena {
    arena: SegmentedShmArena,
}

#[pymethods]
impl PySegmentedShmArena {
    #[staticmethod]
    fn create(base: &str, segment_size: usize, max_segments: usize) -> PyResult<Self> {
        Ok(Self {
            arena: SegmentedShmArena::create(base, segment_size, max_segments)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    /// -> (segment_idx, offset); grows by one segment when full (dedicated
    /// segment for oversized allocations), errors at the max_segments cap.
    /// GIL released: the growth path creates + maps a new shm segment
    /// (file create, ftruncate, mmap — milliseconds), which must not stall
    /// other Python threads (the serve loop, stream relays).
    fn reserve(&self, py: Python<'_>, nbytes: usize) -> PyResult<(usize, usize)> {
        py.allow_threads(|| self.arena.reserve(nbytes))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// GIL released: growth (in `reserve`) holds the segments mutex
    /// across an mmap for milliseconds; blocking on that mutex with the
    /// GIL held would freeze every Python thread for the duration.
    fn free(&self, py: Python<'_>, segment: usize, offset: usize) -> bool {
        py.allow_threads(|| self.arena.free(segment, offset))
    }

    #[getter]
    fn num_segments(&self, py: Python<'_>) -> usize {
        py.allow_threads(|| self.arena.num_segments())
    }

    /// `(total_bytes, free_bytes, largest_free_block)` across all segments.
    /// `largest_free_block` collapsing while `free_bytes` stays high is the
    /// fragmentation signature (allocations fail / segments grow despite
    /// healthy total free space).
    fn stats(&self, py: Python<'_>) -> (usize, usize, usize) {
        py.allow_threads(|| self.arena.stats())
    }

    fn segment_name(&self, i: usize) -> String {
        self.arena.segment_name(i)
    }

    /// Shared buffer-protocol view of segment `i`.
    fn segment(&self, i: usize) -> PyResult<PyShmSegment> {
        self.arena
            .segment(i)
            .map(|seg| PyShmSegment { seg })
            .ok_or_else(|| PyValueError::new_err(format!("no segment {i}")))
    }
}

/// A model's compiled walk graphs, built from the JSON spec the Python
/// translator (`mstar/graph/rust_core.py`) produces from `GraphSection`s.
#[pyclass(name = "WalkSet")]
struct PyWalkSet {
    inner: WalkSet,
}

#[pymethods]
impl PyWalkSet {
    #[staticmethod]
    fn from_json(spec: &str) -> PyResult<Self> {
        Ok(Self {
            inner: WalkSet::from_json(spec).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[getter]
    fn walk_names(&self) -> Vec<String> {
        self.inner.walks.keys().cloned().collect()
    }

    /// Fresh per-request walk state over the named walk.
    fn state(&self, walk: &str) -> PyResult<PyWalkState> {
        let graph = self
            .inner
            .get(walk)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyWalkState {
            inner: WalkState::new(graph.clone()),
            graph,
            next_uuid: 1,
        })
    }
}

/// Per-request walk state machine — the Rust `WorkerGraphIO`. Tensor payloads
/// stay on the Python data plane; here every value is an opaque uuid.
#[pyclass(name = "WalkState")]
struct PyWalkState {
    inner: WalkState,
    graph: std::sync::Arc<CompiledWalk>,
    next_uuid: u64,
}

impl PyWalkState {
    fn fresh_ref(&mut self) -> TensorRef {
        let u = self.next_uuid;
        self.next_uuid += 1;
        TensorRef::new(u, vec![], "opaque")
    }
}

#[pymethods]
impl PyWalkState {
    /// Inject external inputs: [(node, input_name), ...].
    fn seed(&mut self, inputs: Vec<(String, String)>) -> PyResult<()> {
        let seeded = inputs
            .into_iter()
            .map(|(node, name)| {
                let t = self.fresh_ref();
                IncomingInput {
                    node,
                    name,
                    tensors: vec![t],
                }
            })
            .collect();
        self.inner
            .seed(seeded)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn ready_nodes(&self) -> Vec<String> {
        self.inner.ready_nodes()
    }

    /// Claim a ready node for execution (mstar's pop from the ready queue).
    fn schedule(&mut self, node: &str) -> PyResult<()> {
        self.inner
            .take_node_inputs(node)
            .map(|_| ())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Complete a scheduled node, producing a value for each of its declared
    /// output edge names. Returns (route_events, walk_done); route_events are
    /// (kind, name, target) with kind in {"emission", "persist", "stream"} —
    /// internal edges route inside the state machine.
    fn complete(&mut self, node: &str) -> PyResult<(Vec<(String, String, String)>, bool)> {
        let names: Vec<String> = self
            .graph
            .nodes
            .get(node)
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown node {node:?}")))?
            .outputs
            .iter()
            .map(|e| e.name.clone())
            .collect();
        let mut outputs = std::collections::BTreeMap::new();
        for name in names {
            let t = self.fresh_ref();
            outputs.insert(name, vec![t]);
        }
        let result = self
            .inner
            .complete_node(node, outputs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let events = result
            .events
            .into_iter()
            .map(|ev| match ev {
                RouteEvent::Emission { name, modality, .. } => (
                    "emission".to_string(),
                    name,
                    modality.unwrap_or_default(),
                ),
                RouteEvent::Persist { name, .. } => ("persist".to_string(), name, String::new()),
                RouteEvent::Stream {
                    name,
                    target_partition,
                    ..
                } => ("stream".to_string(), name, target_partition),
            })
            .collect();
        Ok((events, result.walk_done))
    }

    /// Pure-mode seed: caller-supplied uuids so the Python side can key its
    /// own uuid -> tensor-descriptor store (the mstar value-map pattern).
    fn seed_with(&mut self, inputs: Vec<(String, String, u64)>) -> PyResult<()> {
        let seeded = inputs
            .into_iter()
            .map(|(node, name, uuid)| IncomingInput {
                node,
                name,
                tensors: vec![TensorRef::new(uuid, vec![], "opaque")],
            })
            .collect();
        self.inner
            .seed(seeded)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Pure-mode complete: caller-supplied uuids per output name; route
    /// events return (kind, name, target, uuids) so loop outputs can be
    /// reconstructed with real tensor descriptors at termination.
    fn complete_with(
        &mut self,
        node: &str,
        outputs: Vec<(String, Vec<u64>)>,
    ) -> PyResult<(Vec<(String, String, String, Vec<u64>)>, bool)> {
        let mut map = std::collections::BTreeMap::new();
        for (name, uuids) in outputs {
            let refs = uuids
                .into_iter()
                .map(|u| TensorRef::new(u, vec![], "opaque"))
                .collect();
            map.insert(name, refs);
        }
        let result = self
            .inner
            .complete_node(node, map)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let uu = |ts: Vec<TensorRef>| ts.into_iter().map(|t| t.uuid).collect::<Vec<u64>>();
        let events = result
            .events
            .into_iter()
            .map(|ev| match ev {
                RouteEvent::Emission { name, modality, tensors } => (
                    "emission".to_string(), name,
                    modality.unwrap_or_default(), uu(tensors)),
                RouteEvent::Persist { name, tensors } => (
                    "persist".to_string(), name, String::new(), uu(tensors)),
                RouteEvent::Stream { name, target_partition, tensors } => (
                    "stream".to_string(), name, target_partition, uu(tensors)),
            })
            .collect();
        Ok((events, result.walk_done))
    }

    /// External loop-termination signal (mstar's stop_loops / EOS).
    fn signal_loop_finish(&mut self, loop_name: &str) -> PyResult<()> {
        self.inner
            .signal_loop_finish(loop_name)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn loop_iters(&self) -> Vec<(String, u32)> {
        self.inner.loop_iters()
    }

    fn is_done(&self) -> bool {
        self.inner.is_done()
    }
}

/// The worker's batch-decision engine (`mstar/worker/micro_scheduler.py` in
/// Rust). Decide-vs-mutate seam: takes a snapshot of ready work as tuples
/// `(node, walk, rid, worker_graph_id, engine_ready, priority, leader)` plus
/// an explicit monotonic time; returns the batch to run. Queue pops and
/// execution stay with the caller.
#[pyclass(name = "MicroScheduler")]
struct PyMicroScheduler {
    inner: MicroScheduler,
}

type PyReady = (String, String, String, String, bool, u32, bool);

fn to_entries(ready: Vec<PyReady>) -> Vec<ReadyEntry> {
    ready
        .into_iter()
        .map(|(node, walk, request_id, worker_graph_id, engine_ready, priority, leader)| {
            ReadyEntry {
                node,
                walk,
                request_id,
                worker_graph_id,
                engine_ready,
                priority,
                leader,
            }
        })
        .collect()
}

#[pymethods]
impl PyMicroScheduler {
    #[new]
    #[pyo3(signature = (sched_type = "round_robin", max_consec_tp_follower_batches = 1))]
    fn new(sched_type: &str, max_consec_tp_follower_batches: u32) -> PyResult<Self> {
        let st = match sched_type {
            "priority" => SchedulingType::Priority,
            "round_robin" => SchedulingType::RoundRobin,
            other => {
                return Err(PyValueError::new_err(format!(
                    "sched_type must be 'priority' or 'round_robin', got {other:?}"
                )))
            }
        };
        Ok(Self {
            inner: MicroScheduler::new(st, max_consec_tp_follower_batches),
        })
    }

    fn hold_requests(&mut self, request_ids: Vec<String>, now_ms: u64) {
        self.inner.hold_requests(&request_ids, now_ms);
    }

    fn add_pending_remove(&mut self, request_id: &str) {
        self.inner.add_pending_remove(request_id);
    }

    fn clear_pending_remove(&mut self, request_id: &str) {
        self.inner.clear_pending_remove(request_id);
    }

    fn register_tp_follow(&mut self, node: String, walk: String, request_ids: Vec<String>) {
        self.inner.register_tp_follow(node, walk, request_ids);
    }

    #[pyo3(signature = (ready, exclude, now_ms))]
    fn has_ready_excluding(
        &self,
        ready: Vec<PyReady>,
        exclude: Option<(String, String)>,
        now_ms: u64,
    ) -> bool {
        let entries = to_entries(ready);
        self.inner.has_ready_excluding(
            &entries,
            exclude.as_ref().map(|(n, w)| (n.as_str(), w.as_str())),
            now_ms,
        )
    }

    /// -> None or (node, walk, request_ids, worker_graph_ids, tp_follow).
    #[pyo3(signature = (ready, now_ms, max_batch_size = None, target_node = None,
                        target_walk = None, exclude_target = None))]
    #[allow(clippy::too_many_arguments)]
    fn get_next_batch(
        &mut self,
        ready: Vec<PyReady>,
        now_ms: u64,
        max_batch_size: Option<usize>,
        target_node: Option<String>,
        target_walk: Option<String>,
        exclude_target: Option<(String, String)>,
    ) -> Option<(String, String, Vec<String>, Vec<String>, bool)> {
        let entries = to_entries(ready);
        let filter = BatchFilter {
            max_batch_size,
            target_node,
            target_walk,
            exclude_target,
        };
        self.inner
            .get_next_batch(&entries, &filter, now_ms)
            .map(|b| (b.node, b.walk, b.request_ids, b.worker_graph_ids, b.tp_follow))
    }
}

#[pymodule]
fn mstar_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyZmqCommunicator>()?;
    m.add_class::<PyShmArena>()?;
    m.add_class::<PyShmSegment>()?;
    m.add_class::<PySegmentedShmArena>()?;
    m.add_class::<PyWalkSet>()?;
    m.add_class::<PyWalkState>()?;
    m.add_class::<PyMicroScheduler>()?;
    Ok(())
}
