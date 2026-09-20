//! Scoping prototype for the Rust graph port — see ../README.md.
//! Not production code: one worker graph, one walk, no streaming buffers.

mod runtime;
mod shard;
mod spec;
mod state;

use pyo3::prelude::*;

#[pymodule]
fn mstar_graph_proto(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<runtime::GraphRuntime>()?;
    m.add_class::<runtime::BatchRouting>()?;
    Ok(())
}
