//! The per-worker graph runtime: compiled worker-graph specs plus every
//! request's walk state, with a **batched** PyO3 surface so the worker calls
//! it once per forward pass instead of once per request.
//!
//! Port of `mstar/graph/base.py` + `graph_io.py`. The split that matters:
//! Python keeps spec and state in one object per (request, entity), so
//! admitting a request costs a `deepcopy` of the whole section. Here the spec
//! is compiled once and shared behind an `Arc`; state is a flat array.
//!
//! Opt in with `MSTAR_RUST_GRAPH=shadow|1`; see `mstar/graph/runtime/`.

pub mod runtime;
pub mod shard;
pub mod spec;
pub mod state;
