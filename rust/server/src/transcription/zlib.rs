//! `compression_ratio` (openai-whisper's): UTF-8 bytes over their zlib size,
//! by the system's libz, opened at run time as `sndfile.rs` opens libsndfile.
//! The ratio is reported in `verbose_json` and decides the temperature
//! fallback, so it is zlib's own deflate (`zlib.compress`'s defaults), not a
//! reimplementation whose output size differs.

use std::ffi::{c_int, c_ulong, CString};
use std::sync::OnceLock;

type CompressBound = unsafe extern "C" fn(c_ulong) -> c_ulong;
type Compress2 = unsafe extern "C" fn(*mut u8, *mut c_ulong, *const u8, c_ulong, c_int) -> c_int;

struct Lib {
    bound: CompressBound,
    compress2: Compress2,
}

/// `Z_DEFAULT_COMPRESSION`, `zlib.compress`'s level.
const Z_DEFAULT_COMPRESSION: c_int = -1;

fn lib() -> Option<&'static Lib> {
    static LIB: OnceLock<Option<Lib>> = OnceLock::new();
    LIB.get_or_init(|| unsafe {
        let name = CString::new("libz.so.1").ok()?;
        let handle = libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if handle.is_null() {
            eprintln!("WARNING: libz.so.1 does not load; transcription compression ratios read as 0");
            return None;
        }
        let bound = libc::dlsym(handle, c"compressBound".as_ptr());
        let compress2 = libc::dlsym(handle, c"compress2".as_ptr());
        if bound.is_null() || compress2.is_null() {
            return None;
        }
        Some(Lib {
            bound: std::mem::transmute::<*mut libc::c_void, CompressBound>(bound),
            compress2: std::mem::transmute::<*mut libc::c_void, Compress2>(compress2),
        })
    })
    .as_ref()
}

/// `zlib.compress(data)`'s length.
fn compressed_len(data: &[u8]) -> Option<usize> {
    let lib = lib()?;
    unsafe {
        let mut len = (lib.bound)(data.len() as c_ulong);
        let mut out = vec![0u8; len as usize];
        let rc = (lib.compress2)(out.as_mut_ptr(), &mut len, data.as_ptr(), data.len() as c_ulong, Z_DEFAULT_COMPRESSION);
        (rc == 0).then_some(len as usize)
    }
}

/// `compression_ratio`: a repetition loop scores far above prose.
pub fn compression_ratio(text: &str) -> f64 {
    let data = text.as_bytes();
    match compressed_len(data) {
        Some(n) => data.len() as f64 / n.max(1) as f64,
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_repetition_loop_compresses_far_better_than_prose() {
        // zlib.compress(b"") is 8 bytes; a loop's ratio is far over Whisper's 2.4
        assert_eq!(super::compression_ratio(""), 0.0);
        assert!(super::compression_ratio(&"the same words again ".repeat(40)) > 2.4);
        assert!(super::compression_ratio("A short sentence of ordinary prose.") < 2.4);
    }
}
