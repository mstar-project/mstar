//! Compressed speech output (`mp3`, `flac`, `opus`) through the libsndfile the
//! Python `soundfile` package ships, loaded at run time.
//!
//! The Python frontend encodes these with `soundfile.write`, and soundfile's
//! wheels carry one self-contained libsndfile with LAME, libFLAC and libopus
//! inside. Loading that same library, and calling it as `soundfile.write`
//! does, gives the Python frontend's bytes. It is opened, never linked: LAME is
//! LGPL, and this binary is Apache-2.0. The launcher names it in
//! `MSTAR_LIBSNDFILE`; without it only `wav` and `pcm` are produced, as the
//! Python frontend without soundfile.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::OnceLock;

type SfCount = i64;

#[repr(C)]
#[derive(Default)]
struct SfInfo {
    frames: SfCount,
    samplerate: c_int,
    channels: c_int,
    format: c_int,
    sections: c_int,
    seekable: c_int,
}

#[repr(C)]
struct SfVirtualIo {
    get_filelen: extern "C" fn(*mut c_void) -> SfCount,
    seek: extern "C" fn(SfCount, c_int, *mut c_void) -> SfCount,
    read: extern "C" fn(*mut c_void, SfCount, *mut c_void) -> SfCount,
    write: extern "C" fn(*const c_void, SfCount, *mut c_void) -> SfCount,
    tell: extern "C" fn(*mut c_void) -> SfCount,
}

type Sndfile = c_void;
type OpenVirtual = unsafe extern "C" fn(*mut SfVirtualIo, c_int, *mut SfInfo, *mut c_void) -> *mut Sndfile;
type WritefShort = unsafe extern "C" fn(*mut Sndfile, *const i16, SfCount) -> SfCount;
type Seek = unsafe extern "C" fn(*mut Sndfile, SfCount, c_int) -> SfCount;
type Command = unsafe extern "C" fn(*mut Sndfile, c_int, *mut c_void, c_int) -> c_int;
type WriteSync = unsafe extern "C" fn(*mut Sndfile);
type Close = unsafe extern "C" fn(*mut Sndfile) -> c_int;
type Error = unsafe extern "C" fn(*mut Sndfile) -> c_int;
type Strerror = unsafe extern "C" fn(*mut Sndfile) -> *const c_char;

/// The libsndfile entry points `soundfile.write` reaches.
struct Lib {
    open_virtual: OpenVirtual,
    writef_short: WritefShort,
    seek: Seek,
    command: Command,
    write_sync: WriteSync,
    close: Close,
    error: Error,
    strerror: Strerror,
}

// The handle lives for the process, and libsndfile's calls on distinct
// SNDFILEs don't share state.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: OnceLock<Option<Lib>> = OnceLock::new();

const SFM_WRITE: c_int = 0x20;
const SFC_SET_CLIPPING: c_int = 0x10c0;
const SF_TRUE: c_int = 1;
const SEEK_SET: c_int = 0;
const SEEK_CUR: c_int = 1;
const SEEK_END: c_int = 2;

/// libsndfile's opus encoder takes only these rates (`media_io._OPUS_RATES`).
pub const OPUS_RATES: [u32; 5] = [8000, 12000, 16000, 24000, 48000];

/// `media_io._SOUNDFILE_AUDIO`: format | subtype, at the file's own endianness.
fn sf_format(fmt: &str) -> Option<c_int> {
    match fmt {
        "mp3" => Some(0x230000 | 0x0082), // MP3, MPEG_LAYER_III
        "flac" => Some(0x170000 | 0x0002), // FLAC, PCM_16
        "opus" => Some(0x200000 | 0x0064), // OGG, OPUS
        _ => None,
    }
}

/// `media_io.AUDIO_FORMAT_MIME` for the compressed formats.
pub fn mime(fmt: &str) -> &'static str {
    match fmt {
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        _ => "audio/ogg",
    }
}

/// Open `MSTAR_LIBSNDFILE` once; a path that doesn't load is logged and leaves
/// the compressed formats off.
pub fn init() {
    LIB.get_or_init(|| {
        let path = std::env::var("MSTAR_LIBSNDFILE").ok().filter(|p| !p.is_empty())?;
        match unsafe { load(&path) } {
            Ok(lib) => {
                eprintln!("compressed speech output (mp3, flac, opus) through {path}");
                Some(lib)
            }
            Err(e) => {
                eprintln!("WARNING: MSTAR_LIBSNDFILE {path}: {e}; only wav and pcm are produced");
                None
            }
        }
    });
}

/// Whether `mp3`, `flac` and `opus` can be produced.
pub fn available() -> bool {
    LIB.get().is_some_and(Option::is_some)
}

unsafe fn load(path: &str) -> Result<Lib, String> {
    let cpath = CString::new(path).map_err(|e| e.to_string())?;
    let handle = libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
    if handle.is_null() {
        let why = libc::dlerror();
        return Err(if why.is_null() {
            "dlopen failed".to_string()
        } else {
            CStr::from_ptr(why).to_string_lossy().into_owned()
        });
    }
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let p = libc::dlsym(handle, concat!($name, "\0").as_ptr().cast());
            if p.is_null() {
                return Err(format!("no symbol {}", $name));
            }
            std::mem::transmute::<*mut c_void, $ty>(p)
        }};
    }
    Ok(Lib {
        open_virtual: sym!("sf_open_virtual", OpenVirtual),
        writef_short: sym!("sf_writef_short", WritefShort),
        seek: sym!("sf_seek", Seek),
        command: sym!("sf_command", Command),
        write_sync: sym!("sf_write_sync", WriteSync),
        close: sym!("sf_close", Close),
        error: sym!("sf_error", Error),
        strerror: sym!("sf_strerror", Strerror),
    })
}

/// An in-memory file with `io.BytesIO`'s semantics, which is what
/// `soundfile.write` hands libsndfile: a seek past the end is allowed, and a
/// write there pads the gap with zeros.
struct Buffer {
    data: Vec<u8>,
    pos: usize,
}

fn buffer<'a>(user: *mut c_void) -> &'a mut Buffer {
    unsafe { &mut *user.cast::<Buffer>() }
}

extern "C" fn vio_len(user: *mut c_void) -> SfCount {
    buffer(user).data.len() as SfCount
}

extern "C" fn vio_seek(offset: SfCount, whence: c_int, user: *mut c_void) -> SfCount {
    let b = buffer(user);
    let base = match whence {
        SEEK_CUR => b.pos as SfCount,
        SEEK_END => b.data.len() as SfCount,
        _ => 0,
    };
    b.pos = (base + offset).max(0) as usize;
    b.pos as SfCount
}

extern "C" fn vio_read(ptr: *mut c_void, count: SfCount, user: *mut c_void) -> SfCount {
    let b = buffer(user);
    let n = (count.max(0) as usize).min(b.data.len().saturating_sub(b.pos));
    unsafe { std::ptr::copy_nonoverlapping(b.data.as_ptr().add(b.pos), ptr.cast::<u8>(), n) };
    b.pos += n;
    n as SfCount
}

extern "C" fn vio_write(ptr: *const c_void, count: SfCount, user: *mut c_void) -> SfCount {
    let b = buffer(user);
    let n = count.max(0) as usize;
    let src = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), n) };
    let end = b.pos + n;
    if b.data.len() < end {
        b.data.resize(end, 0);
    }
    b.data[b.pos..end].copy_from_slice(src);
    b.pos = end;
    n as SfCount
}

extern "C" fn vio_tell(user: *mut c_void) -> SfCount {
    buffer(user).pos as SfCount
}

/// `soundfile.write(BytesIO(), int16 samples, rate, format=..., subtype=...)`
/// for mono 16-bit PCM, call for call, so the bytes are Python's.
pub fn encode(pcm: &[u8], sample_rate: u32, fmt: &str) -> Result<Vec<u8>, String> {
    let lib = LIB.get().and_then(Option::as_ref).ok_or("no libsndfile")?;
    let format = sf_format(fmt).ok_or_else(|| format!("no libsndfile format for {fmt}"))?;
    let samples: Vec<i16> = pcm.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect();
    let mut out = Buffer { data: Vec::new(), pos: 0 };
    let mut vio = SfVirtualIo { get_filelen: vio_len, seek: vio_seek, read: vio_read, write: vio_write, tell: vio_tell };
    let mut info = SfInfo { samplerate: sample_rate as c_int, channels: 1, format, ..Default::default() };
    unsafe {
        let file = (lib.open_virtual)(&mut vio, SFM_WRITE, &mut info, (&mut out as *mut Buffer).cast());
        if file.is_null() {
            let why = CStr::from_ptr((lib.strerror)(std::ptr::null_mut())).to_string_lossy().into_owned();
            return Err(format!("libsndfile could not open a {fmt} stream: {why}"));
        }
        let failed = |file| CStr::from_ptr((lib.strerror)(file)).to_string_lossy().into_owned();
        (lib.command)(file, SFC_SET_CLIPPING, std::ptr::null_mut(), SF_TRUE);
        let seekable = info.seekable == SF_TRUE;
        let frames = samples.len() as SfCount;
        // SoundFile._cdata_io, then _update_frames
        let curr = if seekable { (lib.seek)(file, 0, SEEK_CUR) } else { 0 };
        let written = (lib.writef_short)(file, samples.as_ptr(), frames);
        if (lib.error)(file) != 0 || written != frames {
            let why = failed(file);
            (lib.close)(file);
            return Err(format!("libsndfile could not write {fmt}: {why}"));
        }
        if seekable {
            (lib.seek)(file, curr + written, SEEK_SET);
            let curr = (lib.seek)(file, 0, SEEK_CUR);
            (lib.seek)(file, 0, SEEK_END);
            (lib.seek)(file, curr, SEEK_SET);
        }
        (lib.write_sync)(file);
        if (lib.close)(file) != 0 {
            return Err(format!("libsndfile could not finish the {fmt} stream"));
        }
    }
    Ok(out.data)
}
