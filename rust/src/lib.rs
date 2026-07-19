//! mstar's Rust components, one crate: each module is an independent
//! capability behind its own opt-in flag, and this file is the PyO3
//! surface the Python wrappers drive. Currently: `communicator.rs` (the
//! ZMQ PUSH/PULL control mesh, as a transport + codec split) and `shm.rs`
//! (the shared-memory tensor arena). The crate is not limited to these —
//! new components land as new modules with their own bindings. Also
//! usable as a plain Rust library (rlib) by Rust-side consumers.
//! Build: `maturin develop --release` in rust/.

pub mod communicator;
pub mod shm;

use std::os::raw::{c_int, c_void};
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use communicator::{RawZmqCommunicator, RecvEvent};
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

#[pymodule]
fn mstar_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyZmqCommunicator>()?;
    m.add_class::<PyShmArena>()?;
    m.add_class::<PyShmSegment>()?;
    m.add_class::<PySegmentedShmArena>()?;
    Ok(())
}
