//! mstar's Rust transport (RFC #130 Step 1): the ZMQ PUSH/PULL mesh, vendored
//! from mstar-rs. `communicator.rs` is the transport + codec split; this file
//! is the PyO3 surface (`mstar_rust.ZmqCommunicator`) the Python
//! `RustZMQCommunicator` wrapper drives. Build: `maturin develop` in rust/.

pub mod communicator;

use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use communicator::{RawZmqCommunicator, RecvEvent};

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

    fn send(&self, peer_id: &str, data: &[u8]) -> PyResult<()> {
        self.inner
            .send(peer_id, data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn try_recv<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner.try_recv().map(|b| PyBytes::new(py, &b))
    }

    fn recv_timeout<'py>(&self, py: Python<'py>, timeout_ms: u64) -> Option<Bound<'py, PyBytes>> {
        py.allow_threads(|| self.inner.recv_timeout(Duration::from_millis(timeout_ms)))
            .map(|b| PyBytes::new(py, &b))
    }

    /// ("msg", bytes) | ("wake", None) | ("timeout", None).
    fn recv_or_wake<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: u64,
    ) -> (&'static str, Option<Bound<'py, PyBytes>>) {
        let ev = py.allow_threads(|| self.inner.recv_or_wake(Duration::from_millis(timeout_ms)));
        match ev {
            RecvEvent::Message(b) => ("msg", Some(PyBytes::new(py, &b))),
            RecvEvent::Wake => ("wake", None),
            RecvEvent::Timeout => ("timeout", None),
        }
    }
}

#[pymodule]
fn mstar_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZmqCommunicator>()?;
    Ok(())
}
