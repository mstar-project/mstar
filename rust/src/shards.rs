//! Cutting and joining TP-sharded tensors: a tensor's canonical layout has its
//! shard dim leading and contiguous, so a slice is a range of rows.
//!
//! `native` says the torch wrapper reads this manager's tensors (see
//! `ArenaShmCommunicationManager`). Otherwise, on ROCm and XPU devices and with
//! Mooncake, the same computation goes through torch's Python calls.

use std::collections::BTreeMap;

use pyo3::exceptions::{PyAssertionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::torch_abi::{abi, TensorFacts, TorchAbi};

fn native_abi(native: bool) -> PyResult<Option<&'static dyn TorchAbi>> {
    if !native {
        return Ok(None);
    }
    abi()
        .map(Some)
        .ok_or_else(|| PyRuntimeError::new_err("torch_abi_load has not run"))
}

fn permute_contiguous(tensor: &Bound<'_, PyAny>, order: Vec<i64>) -> PyResult<Py<PyAny>> {
    let permuted = tensor.call_method1("permute", PyTuple::new(tensor.py(), order)?)?;
    Ok(permuted.call_method0("contiguous")?.unbind())
}

/// Move `dim` to dim 0, the others keeping their order: the canonical layout.
#[pyfunction]
pub fn shard_lead(tensor: &Bound<'_, PyAny>, dim: i64, native: bool) -> PyResult<Py<PyAny>> {
    if let Some(abi) = native_abi(native)? {
        return abi.lead_contiguous(tensor, dim);
    }
    let ndim: i64 = tensor.getattr("ndim")?.extract()?;
    let order = std::iter::once(dim)
        .chain((0..ndim).filter(|&i| i != dim))
        .collect();
    permute_contiguous(tensor, order)
}

/// The inverse of [`shard_lead`]: dim 0 back to position `dim`.
#[pyfunction]
pub fn shard_undo_lead(tensor: &Bound<'_, PyAny>, dim: i64, native: bool) -> PyResult<Py<PyAny>> {
    if let Some(abi) = native_abi(native)? {
        return abi.unlead_contiguous(tensor, dim);
    }
    let ndim: i64 = tensor.getattr("ndim")?.extract()?;
    let order = (1..=dim)
        .chain(std::iter::once(0))
        .chain(dim + 1..ndim)
        .collect();
    permute_contiguous(tensor, order)
}

/// A canonical tensor's bytes per row: 0 for rows with no bytes (a `[4, 0]`
/// tensor).
fn row_bytes(tensor: &Bound<'_, PyAny>, abi: Option<&dyn TorchAbi>) -> PyResult<i64> {
    if let Some(abi) = abi {
        let mut facts = TensorFacts::default();
        if !unsafe { abi.facts(tensor.as_ptr(), &mut facts) } {
            return Err(PyValueError::new_err("not a tensor"));
        }
        // A tensor that is not `plain` reports no size or dims (see the shim).
        if facts.plain != 0 {
            let numel: i64 = facts.dims[..facts.ndim as usize].iter().product();
            if numel == 0 || facts.ndim == 0 {
                return Ok(0);
            }
            return Ok(facts.nbytes / facts.dims[0]);
        }
    }
    let shape: Vec<i64> = tensor.getattr("shape")?.extract()?;
    let element: i64 = tensor.call_method0("element_size")?.extract()?;
    Ok(element * shape.iter().skip(1).product::<i64>())
}

/// The rows of a canonical tensor that `nbytes` bytes at `offset` of it name, as
/// a view. Rows of zero bytes carry no offset to recover the start from and hold
/// nothing to read, so only the descriptor's row count (`rows`) matters there.
#[pyfunction]
pub fn shard_slice_rows(
    tensor: &Bound<'_, PyAny>,
    offset: i64,
    nbytes: i64,
    rows: i64,
    native: bool,
) -> PyResult<Py<PyAny>> {
    let abi = native_abi(native)?;
    let size = row_bytes(tensor, abi)?;
    let (start, length) = if size == 0 {
        (0, rows)
    } else {
        if offset % size != 0 || nbytes % size != 0 {
            return Err(PyAssertionError::new_err(format!(
                "slice offset {offset} / nbytes {nbytes} not aligned to canonical row size {size}"
            )));
        }
        (offset / size, nbytes / size)
    };
    match abi {
        Some(abi) => abi.narrow(tensor, start, length),
        None => Ok(tensor.call_method1("narrow", (0, start, length))?.unbind()),
    }
}

/// A fan-in edge's slices as they arrive from each source rank.
#[pyclass(name = "BufferedShards")]
pub struct PyBufferedShards {
    total_fanin: usize,
    shard_dim: i64,
    shards: BTreeMap<i64, Vec<Py<PyAny>>>,
}

#[pymethods]
impl PyBufferedShards {
    #[new]
    fn new(total_fanin: usize, shard_dim: i64) -> Self {
        Self {
            total_fanin,
            shard_dim,
            shards: BTreeMap::new(),
        }
    }

    fn add(&mut self, source_rank: i64, tensors: Vec<Py<PyAny>>) {
        self.shards.entry(source_rank).or_default().extend(tensors);
    }

    fn is_done(&self) -> bool {
        self.shards.len() >= self.total_fanin
    }

    /// The whole tensors, joined along the real shard dim in rank order, so they
    /// are back in their original layout.
    fn consolidate(&self, py: Python<'_>, native: bool) -> PyResult<Vec<Py<PyAny>>> {
        if !self.is_done() {
            return Err(PyAssertionError::new_err(
                "Can't consolidate shards until all fanin contributions are received",
            ));
        }
        let counts: Vec<usize> = self.shards.values().map(Vec::len).collect();
        if counts.windows(2).any(|w| w[0] != w[1]) {
            return Err(PyAssertionError::new_err(format!(
                "BufferedShards: source ranks contributed unequal tensor counts: {:?}",
                self.shards
                    .iter()
                    .map(|(rank, l)| (rank, l.len()))
                    .collect::<Vec<_>>(),
            )));
        }
        let abi = native_abi(native)?;
        let mut out = Vec::with_capacity(counts.first().copied().unwrap_or(0));
        for i in 0..counts.first().copied().unwrap_or(0) {
            let parts: Vec<Bound<'_, PyAny>> = self
                .shards
                .values()
                .map(|l| l[i].bind(py).clone())
                .collect();
            out.push(match abi {
                Some(abi) => abi.cat(&parts, self.shard_dim)?,
                None => {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("dim", self.shard_dim)?;
                    py.import("torch")?
                        .call_method("cat", (PyList::new(py, parts)?,), Some(&kwargs))?
                        .unbind()
                }
            });
        }
        Ok(out)
    }
}
