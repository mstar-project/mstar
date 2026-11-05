//! A run of a captured graph, launched from one call with the GIL released:
//! N graph launches and the device-to-device copies between them, in order.
//! Every CUDA entry point arrives as an address, so the crate links no CUDA.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::ffi::c_void;

type GraphLaunch = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
type Memcpy2D = unsafe extern "C" fn(
    *mut c_void, usize, *const c_void, usize, usize, usize, i32, *mut c_void,
) -> i32;

const OP_LAUNCH: u64 = 0;
const OP_COPY: u64 = 1;
const COPY_WORDS: usize = 6;
// cudaMemcpyDeviceToDevice
const DEVICE_TO_DEVICE: i32 = 3;

/// `ops`, flat: a launch is `[0]`, a copy
/// `[1, dst, dst_pitch, src, src_pitch, width, height]`, in bytes.
fn run(launch: GraphLaunch, copy: Memcpy2D, exec: usize, stream: usize, ops: &[u64]) -> Result<(), String> {
    let stream = stream as *mut c_void;
    let mut at = 0;
    while at < ops.len() {
        let op = ops[at];
        at += 1;
        let status = match op {
            // SAFETY: the caller passes a live graph exec and stream, and CUDA's own entry points
            OP_LAUNCH => unsafe { launch(exec as *mut c_void, stream) },
            OP_COPY => {
                let args = ops.get(at..at + COPY_WORDS).ok_or("a copy op is cut short")?;
                at += COPY_WORDS;
                // SAFETY: the addresses are of device tensors the caller keeps alive
                unsafe {
                    copy(
                        args[0] as *mut c_void, args[1] as usize, args[2] as *const c_void, args[3] as usize,
                        args[4] as usize, args[5] as usize, DEVICE_TO_DEVICE, stream,
                    )
                }
            }
            other => return Err(format!("unknown launch op {other}")),
        };
        if status != 0 {
            return Err(format!("launch op {op} failed with CUDA error {status}"));
        }
    }
    Ok(())
}

/// Run `ops` on `stream`; see `run`. `graph_launch` is `cudaGraphLaunch`'s
/// address and `memcpy_2d` `cudaMemcpy2DAsync`'s.
#[pyfunction]
pub fn launch_run(
    py: Python<'_>, graph_launch: usize, memcpy_2d: usize, exec: usize, stream: usize, ops: Vec<u64>,
) -> PyResult<()> {
    if graph_launch == 0 || memcpy_2d == 0 {
        return Err(PyRuntimeError::new_err("launch_run needs both CUDA entry points"));
    }
    // SAFETY: both are the addresses of functions of these signatures
    let (launch, copy) = unsafe {
        (
            std::mem::transmute::<usize, GraphLaunch>(graph_launch),
            std::mem::transmute::<usize, Memcpy2D>(memcpy_2d),
        )
    };
    py.detach(|| run(launch, copy, exec, stream, &ops)).map_err(PyRuntimeError::new_err)
}
