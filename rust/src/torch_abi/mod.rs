//! Tensors from Rust, working from a `PyObject*` with no Python call per tensor.
//!
//! [`TorchAbi`] is the small surface the transport needs, and [`TorchV2`] its
//! implementor for torch major version 2, over the C++ shim in `shim.cpp`. The
//! shim touches torch's inline layout (`THPVariable_Unpack`, `TensorImpl`), so
//! which versions it runs on is a tested fact, not an assumption:
//! `test/modular/test_torch_abi.py` and the startup check in
//! `mstar/communication/torch_abi.py`. An unlisted torch version is refused at
//! startup.
//!
//! The shim is embedded in the extension and loaded after torch is imported,
//! never linked: see `rust/build.rs`. `rust/server` builds neither, since this
//! module needs the `python` feature.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::sync::OnceLock;

use pyo3::exceptions::{PyImportError, PyRuntimeError, PyTypeError};
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

pub const MAX_DIMS: usize = 16;
/// `c10::DeviceType`, which torch keeps stable: it is serialized.
pub const DEVICE_CPU: i32 = 0;
pub const DEVICE_CUDA: i32 = 1;

/// Twin of `MstarFacts` in shim.cpp.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TensorFacts {
    pub data: *mut u8,
    pub nbytes: i64,
    pub device_type: i32,
    pub device_index: i32,
    pub dtype: i32,
    pub ndim: i32,
    /// See the shim: false means take the torch path, `data` is not usable.
    pub plain: u8,
    pub contiguous: u8,
    pad: [u8; 6],
    pub dims: [i64; MAX_DIMS],
}

impl Default for TensorFacts {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            nbytes: 0,
            device_type: 0,
            device_index: 0,
            dtype: 0,
            ndim: 0,
            plain: 0,
            contiguous: 0,
            pad: [0; 6],
            dims: [0; MAX_DIMS],
        }
    }
}

const ERROR_MESSAGE_BYTES: usize = 508;
const CODE_OUT_OF_MEMORY: i32 = 2;

/// Twin of `MstarError` in shim.cpp.
#[repr(C)]
pub struct RawError {
    code: i32,
    message: [c_char; ERROR_MESSAGE_BYTES],
}

/// A torch error from a call that may have run without the GIL.
#[derive(Debug)]
pub struct AbiError {
    pub out_of_memory: bool,
    pub message: String,
}

impl AbiError {
    fn take(raw: &RawError) -> Self {
        let message = unsafe { CStr::from_ptr(raw.message.as_ptr()) };
        Self {
            out_of_memory: raw.code == CODE_OUT_OF_MEMORY,
            message: message.to_string_lossy().into_owned(),
        }
    }

    /// Back to the exception torch's own factories raise for the same failure.
    pub fn into_pyerr(self, py: Python<'_>) -> PyErr {
        if self.out_of_memory {
            let class = py
                .import("torch.cuda")
                .and_then(|m| m.getattr("OutOfMemoryError"));
            if let Ok(class) = class {
                return PyErr::from_type(class.downcast_into().expect("a class"), self.message);
            }
        }
        PyRuntimeError::new_err(self.message)
    }
}

/// An owned tensor that is not a Python object yet, so it can be made and
/// dropped without the GIL. `data` and `nbytes` are of the tensor itself.
pub struct OwnedTensor {
    raw: *mut c_void,
    pub data: *mut u8,
    pub nbytes: i64,
}

// A refcounted TensorImpl*; nothing about it is tied to the creating thread.
unsafe impl Send for OwnedTensor {}

impl Drop for OwnedTensor {
    fn drop(&mut self) {
        abi().expect("an OwnedTensor exists").drop_raw(self.raw);
    }
}

impl OwnedTensor {
    fn into_raw(self) -> *mut c_void {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
}

/// What a new tensor is. `device_index` is `-1` for a CPU tensor.
#[derive(Clone, Copy)]
pub struct TensorSpec<'a> {
    pub dims: &'a [i64],
    pub dtype: i32,
    pub device_type: i32,
    pub device_index: i32,
    pub pinned: bool,
}

/// The torch operations the arena and sharding paths use, one implementor per
/// supported torch major.
pub trait TorchAbi: Send + Sync + 'static {
    /// The implementor, and the torch headers its shim was built against.
    fn describe(&self) -> String;

    /// False when `tensor` is not a tensor. GIL required, as for `dtype_scalar`.
    ///
    /// # Safety
    /// `tensor` is a live Python object.
    unsafe fn facts(&self, tensor: *mut PyObject, out: &mut TensorFacts) -> bool;

    /// The `c10::ScalarType` of a `torch.dtype`.
    ///
    /// # Safety
    /// `dtype` is a live Python object.
    unsafe fn dtype_scalar(&self, dtype: *mut PyObject) -> Option<i32>;

    /// May run without the GIL. A device tensor is allocated under the calling
    /// thread's current stream.
    fn empty(&self, spec: TensorSpec<'_>) -> Result<OwnedTensor, AbiError>;

    /// `tensor`'s `n` rows along dim 0, as views. May run without the GIL.
    fn unbind(&self, tensor: &OwnedTensor, n: usize) -> Result<Vec<OwnedTensor>, AbiError>;

    fn drop_raw(&self, raw: *mut c_void);

    fn wrap(&self, py: Python<'_>, tensor: OwnedTensor) -> PyResult<Py<PyAny>>;

    /// `tensor.narrow(0, start, length)`.
    fn narrow(&self, tensor: &Bound<'_, PyAny>, start: i64, length: i64) -> PyResult<Py<PyAny>>;

    /// `tensor.permute(dim, <the rest in order>).contiguous()`.
    fn lead_contiguous(&self, tensor: &Bound<'_, PyAny>, dim: i64) -> PyResult<Py<PyAny>>;

    /// The inverse of [`Self::lead_contiguous`].
    fn unlead_contiguous(&self, tensor: &Bound<'_, PyAny>, dim: i64) -> PyResult<Py<PyAny>>;

    fn cat(&self, tensors: &[Bound<'_, PyAny>], dim: i64) -> PyResult<Py<PyAny>>;
}

type FactsFn = unsafe extern "C" fn(*mut PyObject, *mut TensorFacts) -> i32;
type DtypeFn = unsafe extern "C" fn(*mut PyObject) -> i32;
type EmptyFn = unsafe extern "C" fn(
    i32,
    *const i64,
    i32,
    i32,
    i32,
    i32,
    *mut *mut u8,
    *mut i64,
    *mut RawError,
) -> *mut c_void;
type DropFn = unsafe extern "C" fn(*mut c_void);
type UnbindFn = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, i64, *mut RawError) -> i32;
type WrapFn = unsafe extern "C" fn(*mut c_void) -> *mut PyObject;
type NarrowFn = unsafe extern "C" fn(*mut PyObject, i64, i64) -> *mut PyObject;
type LeadFn = unsafe extern "C" fn(*mut PyObject, i64) -> *mut PyObject;
type CatFn = unsafe extern "C" fn(*const *mut PyObject, i64, i64) -> *mut PyObject;

/// Torch 2, over the shim. [`Self::MINORS`] mirrors `torch>=2.10,<2.13` in
/// pyproject.toml, and a test holds the two together.
pub struct TorchV2 {
    built_against: String,
    facts: FactsFn,
    dtype_scalar: DtypeFn,
    empty: EmptyFn,
    drop: DropFn,
    unbind: UnbindFn,
    wrap: WrapFn,
    narrow: NarrowFn,
    lead_contiguous: LeadFn,
    unlead_contiguous: LeadFn,
    cat: CatFn,
}

const ABI_REVISION: i32 =
    1000 + (std::mem::size_of::<TensorFacts>() + std::mem::size_of::<RawError>()) as i32;

impl TorchV2 {
    pub const MAJOR: u32 = 2;
    /// Half-open, the supported minors of [`Self::MAJOR`].
    pub const MINORS: (u32, u32) = (10, 13);

    fn supports(major: u32, minor: u32) -> bool {
        major == Self::MAJOR && (Self::MINORS.0..Self::MINORS.1).contains(&minor)
    }

    /// # Safety
    /// `lib` is a handle from `dlopen`, and stays open for the process.
    unsafe fn resolve(lib: *mut c_void) -> Result<Self, String> {
        let sym = |name: &str| -> Result<*mut c_void, String> {
            let c = CString::new(name).expect("a symbol name");
            let p = libc::dlsym(lib, c.as_ptr());
            if p.is_null() {
                Err(format!("the torch ABI shim has no symbol {name}"))
            } else {
                Ok(p)
            }
        };
        use std::mem::transmute as t;
        let revision: unsafe extern "C" fn() -> i32 = t(sym("mstar_torch_abi_revision")?);
        if revision() != ABI_REVISION {
            return Err(format!(
                "the torch ABI shim is revision {}, this extension wants {ABI_REVISION}",
                revision()
            ));
        }
        let built: unsafe extern "C" fn() -> *const c_char = t(sym("mstar_torch_built_against")?);
        Ok(Self {
            built_against: CStr::from_ptr(built()).to_string_lossy().into_owned(),
            facts: t(sym("mstar_facts")?),
            dtype_scalar: t(sym("mstar_dtype_scalar")?),
            empty: t(sym("mstar_empty")?),
            drop: t(sym("mstar_drop")?),
            unbind: t(sym("mstar_unbind")?),
            wrap: t(sym("mstar_wrap")?),
            narrow: t(sym("mstar_narrow")?),
            lead_contiguous: t(sym("mstar_lead_contiguous")?),
            unlead_contiguous: t(sym("mstar_unlead_contiguous")?),
            cat: t(sym("mstar_cat")?),
        })
    }

    fn new_ref(py: Python<'_>, p: *mut PyObject) -> PyResult<Py<PyAny>> {
        if p.is_null() {
            return Err(PyErr::fetch(py));
        }
        Ok(unsafe { Py::from_owned_ptr(py, p) })
    }
}

impl TorchAbi for TorchV2 {
    fn describe(&self) -> String {
        format!("TorchV2, shim built against torch {}", self.built_against)
    }

    unsafe fn facts(&self, tensor: *mut PyObject, out: &mut TensorFacts) -> bool {
        (self.facts)(tensor, out) == 0
    }

    unsafe fn dtype_scalar(&self, dtype: *mut PyObject) -> Option<i32> {
        match (self.dtype_scalar)(dtype) {
            -1 => None,
            s => Some(s),
        }
    }

    fn empty(&self, spec: TensorSpec<'_>) -> Result<OwnedTensor, AbiError> {
        let mut err = RawError {
            code: 0,
            message: [0; ERROR_MESSAGE_BYTES],
        };
        let (mut data, mut nbytes) = (std::ptr::null_mut(), 0i64);
        let raw = unsafe {
            (self.empty)(
                spec.dims.len() as i32,
                spec.dims.as_ptr(),
                spec.dtype,
                spec.device_type,
                spec.device_index,
                spec.pinned as i32,
                &mut data,
                &mut nbytes,
                &mut err,
            )
        };
        if err.code != 0 {
            return Err(AbiError::take(&err));
        }
        Ok(OwnedTensor { raw, data, nbytes })
    }

    fn unbind(&self, tensor: &OwnedTensor, n: usize) -> Result<Vec<OwnedTensor>, AbiError> {
        let mut err = RawError {
            code: 0,
            message: [0; ERROR_MESSAGE_BYTES],
        };
        let mut raw = vec![std::ptr::null_mut(); n];
        unsafe { (self.unbind)(tensor.raw, raw.as_mut_ptr(), n as i64, &mut err) };
        if err.code != 0 {
            return Err(AbiError::take(&err));
        }
        // A row's own data and size are not needed: it only becomes a Python
        // object from here.
        Ok(raw
            .into_iter()
            .map(|raw| OwnedTensor {
                raw,
                data: std::ptr::null_mut(),
                nbytes: 0,
            })
            .collect())
    }

    fn drop_raw(&self, raw: *mut c_void) {
        unsafe { (self.drop)(raw) };
    }

    fn wrap(&self, py: Python<'_>, tensor: OwnedTensor) -> PyResult<Py<PyAny>> {
        Self::new_ref(py, unsafe { (self.wrap)(tensor.into_raw()) })
    }

    fn narrow(&self, tensor: &Bound<'_, PyAny>, start: i64, length: i64) -> PyResult<Py<PyAny>> {
        check_tensor(self, tensor)?;
        Self::new_ref(tensor.py(), unsafe {
            (self.narrow)(tensor.as_ptr(), start, length)
        })
    }

    fn lead_contiguous(&self, tensor: &Bound<'_, PyAny>, dim: i64) -> PyResult<Py<PyAny>> {
        check_tensor(self, tensor)?;
        Self::new_ref(tensor.py(), unsafe {
            (self.lead_contiguous)(tensor.as_ptr(), dim)
        })
    }

    fn unlead_contiguous(&self, tensor: &Bound<'_, PyAny>, dim: i64) -> PyResult<Py<PyAny>> {
        check_tensor(self, tensor)?;
        Self::new_ref(tensor.py(), unsafe {
            (self.unlead_contiguous)(tensor.as_ptr(), dim)
        })
    }

    fn cat(&self, tensors: &[Bound<'_, PyAny>], dim: i64) -> PyResult<Py<PyAny>> {
        let mut raw = Vec::with_capacity(tensors.len());
        for t in tensors {
            check_tensor(self, t)?;
            raw.push(t.as_ptr());
        }
        let Some(first) = tensors.first() else {
            return Err(PyRuntimeError::new_err("cat of no tensors"));
        };
        Self::new_ref(first.py(), unsafe {
            (self.cat)(raw.as_ptr(), raw.len() as i64, dim)
        })
    }
}

fn check_tensor(abi: &dyn TorchAbi, t: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut f = TensorFacts::default();
    if unsafe { abi.facts(t.as_ptr(), &mut f) } {
        Ok(())
    } else {
        Err(PyTypeError::new_err(format!(
            "expected a torch.Tensor, got {}",
            t.get_type()
        )))
    }
}

/// `2.11.0+cu128` -> `(2, 11)`.
fn parse_version(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// Whether some implementor covers this torch. The refusal names what to do.
fn check_version(version: &str) -> Result<(), String> {
    let supported = parse_version(version).is_some_and(|(a, b)| TorchV2::supports(a, b));
    if supported {
        return Ok(());
    }
    let (major, lo, hi) = (TorchV2::MAJOR, TorchV2::MINORS.0, TorchV2::MINORS.1);
    Err(format!(
        "torch {version} has no torch ABI implementor in mstar_rust: TorchV2 covers \
         torch>={major}.{lo},<{major}.{hi}. For another 2.x, widen TorchV2::MINORS in \
         rust/src/torch_abi/mod.rs and the torch pin in pyproject.toml once \
         test/modular/test_torch_abi.py passes on it, built against each supported \
         version; for another major, add an implementor of TorchAbi beside TorchV2."
    ))
}

// Empty when build.rs found no torch to build it against (see rust/build.rs).
#[cfg(mstar_torch_shim)]
const EMBEDDED_SHIM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libmstar_torch_v2.so"));
#[cfg(not(mstar_torch_shim))]
const EMBEDDED_SHIM: &[u8] = &[];

/// Maps the embedded shim from an anonymous file, so the wheel holds one
/// library and no extractable path can go stale or be written to.
#[cfg(target_os = "linux")]
fn dlopen_embedded() -> Result<*mut c_void, String> {
    use std::io::Write;
    use std::os::fd::FromRawFd;

    let name = CString::new("mstar_torch_v2").expect("a name");
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(format!(
            "memfd_create failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(EMBEDDED_SHIM)
        .map_err(|e| format!("writing the shim to its memfd failed: {e}"))?;
    dlopen_path(&format!("/proc/self/fd/{fd}"))
}

#[cfg(not(target_os = "linux"))]
fn dlopen_embedded() -> Result<*mut c_void, String> {
    Err("the embedded torch ABI shim is Linux-only".to_string())
}

fn dlopen_path(path: &str) -> Result<*mut c_void, String> {
    let c = CString::new(path).map_err(|e| e.to_string())?;
    let lib = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if lib.is_null() {
        let why = unsafe { CStr::from_ptr(libc::dlerror()) };
        return Err(format!(
            "dlopen of the torch ABI shim failed: {}",
            why.to_string_lossy()
        ));
    }
    Ok(lib)
}

static ABI: OnceLock<Box<dyn TorchAbi>> = OnceLock::new();

/// The loaded implementor, once `torch_abi_load` has succeeded.
pub fn abi() -> Option<&'static dyn TorchAbi> {
    ABI.get().map(|a| a.as_ref())
}

/// Whether this build carries a shim at all.
#[pyfunction]
fn torch_abi_embedded() -> bool {
    !EMBEDDED_SHIM.is_empty()
}

/// Refuses a torch no implementor covers.
#[pyfunction]
fn torch_abi_check_version(version: &str) -> PyResult<()> {
    check_version(version).map_err(PyRuntimeError::new_err)
}

/// Check `torch_version` and load the shim; idempotent. Call it with torch
/// imported, or the shim's libtorch dependencies will not resolve. `shim_path`
/// loads a shim from a file instead of the embedded one, for the version matrix.
#[pyfunction]
#[pyo3(signature = (torch_version, shim_path=None))]
fn torch_abi_load(torch_version: &str, shim_path: Option<&str>) -> PyResult<String> {
    check_version(torch_version).map_err(PyRuntimeError::new_err)?;
    if let Some(loaded) = abi() {
        return Ok(loaded.describe());
    }
    let lib = match shim_path {
        Some(path) => dlopen_path(path),
        None if EMBEDDED_SHIM.is_empty() => {
            return Err(PyImportError::new_err(
                "this mstar_rust was built with no torch ABI shim, because its build found no \
                 torch to compile against. Rebuild it with torch installed: `touch rust/build.rs \
                 && maturin develop --release -m rust/Cargo.toml`, or set MSTAR_TORCH_DIR to a \
                 directory holding torch's include/ and lib/ (see docs/installation.rst).",
            ));
        }
        None => dlopen_embedded(),
    }
    .map_err(PyImportError::new_err)?;
    let v2 = unsafe { TorchV2::resolve(lib) }.map_err(PyImportError::new_err)?;
    let described = v2.describe();
    let _ = ABI.set(Box::new(v2));
    Ok(described)
}

fn loaded() -> PyResult<&'static dyn TorchAbi> {
    abi().ok_or_else(|| PyRuntimeError::new_err("torch_abi_load has not run"))
}

/// A tensor's facts as Python sees them, for the startup self-test.
#[pyfunction]
fn torch_abi_facts<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let abi = loaded()?;
    let mut f = TensorFacts::default();
    if !unsafe { abi.facts(tensor.as_ptr(), &mut f) } {
        return Err(PyTypeError::new_err(format!(
            "expected a torch.Tensor of at most {MAX_DIMS} dims, got {}",
            tensor.get_type()
        )));
    }
    let out = pyo3::types::PyDict::new(tensor.py());
    out.set_item("data", f.data as usize)?;
    out.set_item("nbytes", f.nbytes)?;
    out.set_item("device_type", f.device_type)?;
    out.set_item("device_index", f.device_index)?;
    out.set_item("dtype", f.dtype)?;
    out.set_item("plain", f.plain != 0)?;
    out.set_item("contiguous", f.contiguous != 0)?;
    let ndim = f.ndim as usize;
    out.set_item("dims", f.dims[..if f.plain != 0 { ndim } else { 0 }].to_vec())?;
    out.set_item("ndim", ndim)?;
    Ok(out)
}

/// The `c10::ScalarType` of a `torch.dtype`.
#[pyfunction]
fn torch_abi_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<i32> {
    unsafe { loaded()?.dtype_scalar(dtype.as_ptr()) }
        .ok_or_else(|| PyTypeError::new_err("expected a torch.dtype"))
}

#[pyfunction]
#[pyo3(signature = (dims, dtype, device_type=DEVICE_CPU, device_index=-1, pinned=false))]
fn torch_abi_empty(
    py: Python<'_>,
    dims: Vec<i64>,
    dtype: &Bound<'_, PyAny>,
    device_type: i32,
    device_index: i32,
    pinned: bool,
) -> PyResult<Py<PyAny>> {
    let abi = loaded()?;
    let dtype = unsafe { abi.dtype_scalar(dtype.as_ptr()) }
        .ok_or_else(|| PyTypeError::new_err("expected a torch.dtype"))?;
    let t = abi
        .empty(TensorSpec {
            dims: &dims,
            dtype,
            device_type,
            device_index,
            pinned,
        })
        .map_err(|e| e.into_pyerr(py))?;
    abi.wrap(py, t)
}

#[pyfunction]
fn torch_abi_narrow(tensor: &Bound<'_, PyAny>, start: i64, length: i64) -> PyResult<Py<PyAny>> {
    loaded()?.narrow(tensor, start, length)
}

#[pyfunction]
fn torch_abi_lead_contiguous(tensor: &Bound<'_, PyAny>, dim: i64) -> PyResult<Py<PyAny>> {
    loaded()?.lead_contiguous(tensor, dim)
}

#[pyfunction]
fn torch_abi_cat(tensors: Vec<Bound<'_, PyAny>>, dim: i64) -> PyResult<Py<PyAny>> {
    loaded()?.cat(&tensors, dim)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(torch_abi_embedded, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_check_version, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_load, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_facts, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_empty, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_narrow, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_lead_contiguous, m)?)?;
    m.add_function(wrap_pyfunction!(torch_abi_cat, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_outside_the_implementors_are_refused_by_name() {
        for ok in ["2.10.0", "2.11.0+cu128", "2.12.1", "2.12.0.dev20260101"] {
            assert!(check_version(ok).is_ok(), "{ok}");
        }
        for no in ["2.9.1", "2.13.0", "3.0.0", "nightly", ""] {
            let why = check_version(no).unwrap_err();
            assert!(why.contains("TorchV2") && why.contains(no), "{why}");
        }
    }
}
