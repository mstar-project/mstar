use serde::{Deserialize, Serialize};

/// Process-wide unique tensor identity. Assigned by the runtime; the Python
/// side keys its `uuid -> torch.Tensor` object store on this.
pub type Uuid = u64;

/// Descriptor for a tensor living on the data plane. The control plane never
/// sees tensor bytes — only this. Mirrors mstar's `TensorPointerInfo`, minus
/// the transport-specific fields (address/offset/session) which belong to the
/// multi-process transport tier (T4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRef {
    pub uuid: Uuid,
    pub dims: Vec<i64>,
    pub dtype: String,
}

impl TensorRef {
    pub fn new(uuid: Uuid, dims: Vec<i64>, dtype: impl Into<String>) -> Self {
        Self {
            uuid,
            dims,
            dtype: dtype.into(),
        }
    }
}
