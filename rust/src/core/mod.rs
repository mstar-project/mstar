//! The graph/walk core  compiled walk graphs and the
//! per-request walk state machine — the Rust port of the runtime behavior of
//! `mstar/graph/base.py`'s `GraphNode`/`Loop` registries and
//! `WorkerGraphIO`. Parity is asserted by `test/rust/test_walk_parity.py`,
//! which drives both implementations with identical event sequences.

pub mod error;
pub mod graph;
pub mod sched;
pub mod tensor;
pub mod walk;

pub use error::{CoreError, Result};
pub use graph::{CompiledWalk, Section, WalkSet, EMIT_TO_CLIENT, EMPTY_DESTINATION};
pub use tensor::{TensorRef, Uuid};
pub use sched::{BatchFilter, MicroScheduler, ReadyEntry, ScheduledBatch, SchedulingType};
pub use walk::{CompletionResult, IncomingInput, RouteEvent, WalkState};
