//! mstar's Rust components, one crate: each module is an independent
//! capability behind its own opt-in flag, and `bindings.rs` is the PyO3
//! surface the Python wrappers drive. Currently: `communicator.rs` (the
//! ZMQ PUSH/PULL control mesh, as a transport + codec split) and `shm.rs`
//! (the shared-memory tensor arena). The crate is not limited to these —
//! new components land as new modules with their own bindings. Also
//! usable as a plain Rust library (rlib) by Rust-side consumers; without the
//! `python` feature (rust/server) it builds only the transport and arena and
//! links no Python, since the bindings, `graph` and `tensors` need pyo3.
//! Build: `maturin develop --release` in rust/.

#[cfg(feature = "python")]
pub mod batch_builder;
#[cfg(feature = "python")]
mod bindings;
pub mod communicator;
#[cfg(feature = "python")]
pub mod graph;
#[cfg(feature = "python")]
pub mod kv_plan;
pub mod shm;
#[cfg(feature = "python")]
pub mod tensors;

#[cfg(feature = "python")]
pub use bindings::PyZmqCommunicator;
