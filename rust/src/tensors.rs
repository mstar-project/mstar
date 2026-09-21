use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};


struct ReferenceInfo {
    ref_count: i64,
    persist: bool,
    mem_registered: bool,
}

impl ReferenceInfo {
    fn new() -> Self {
        Self {
            ref_count: 0,
            persist: false,
            mem_registered: false
        }
    }
}

struct TensorPointerInfo {
    // TODO!
}

#[pyclass]
pub struct TensorBookkeeping {
    ref_info: FxHashMap<u64, ReferenceInfo>,

}