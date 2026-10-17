//! Shared-memory tensor arena — the data plane for cross-process transport.
//!
//! One persistent `/dev/shm` mmap per producer entity plus an in-process
//! first-fit free-list allocator, replacing the per-tensor file open/write/
//! read/unlink dance in `SharedMemoryCommunicationManager` (the Python
//! buffer-protocol view lives in `lib.rs`).
//!
//! - Producer: [`ShmArena::create`], [`ShmArena::reserve`] -> offset, copy
//!   D2H bytes into `arena[off..off+n]`, send the offset as a descriptor,
//!   [`ShmArena::free`] the offset once every consumer has ACKed the
//!   transfer (the embedder tracks which offsets belong to which tensors).
//! - Consumer: [`ShmArena::open`] the same name, read `arena[off..off+n]`
//!   (H2D). Zero syscalls per tensor, one memcpy each way.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Mutex;

use memmap2::{MmapMut, MmapOptions};
use thiserror::Error;

/// Cache-line-friendly alignment; keeps neighbouring tensors off shared lines.
pub const ALIGN: usize = 256;

/// `None` when rounding up would overflow (an absurd size from the caller).
#[inline]
fn align_up(n: usize) -> Option<usize> {
    n.checked_next_multiple_of(ALIGN)
}

#[derive(Debug, Error)]
pub enum ShmError {
    #[error("size must be > 0")]
    ZeroSize,
    #[error("arena full: need {need} B, {free} B free of {total} B")]
    Full {
        need: usize,
        free: usize,
        total: usize,
    },
    /// The filesystem behind the arena cannot hold a new segment. Callers
    /// treat it like `Full` (the Python transport spills to files).
    #[error("arena full: no space to back {size} B at {path}: {source}")]
    NoSpace {
        path: String,
        size: usize,
        source: std::io::Error,
    },
    #[error("io on {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

/// First-fit free-list allocator over the arena. Coalesces on free.
struct Allocator {
    free: Vec<(usize, usize)>,    // (offset, len), sorted by offset, disjoint
    live: HashMap<usize, usize>,  // offset -> aligned len (free() needs only offset)
}

impl Allocator {
    fn new(size: usize) -> Self {
        Self {
            free: vec![(0, size)],
            live: HashMap::new(),
        }
    }

    fn alloc(&mut self, n: usize) -> Option<usize> {
        let n = align_up(n.max(1))?;
        for i in 0..self.free.len() {
            let (off, len) = self.free[i];
            if len >= n {
                if len == n {
                    self.free.remove(i);
                } else {
                    self.free[i] = (off + n, len - n);
                }
                self.live.insert(off, n);
                return Some(off);
            }
        }
        None
    }

    fn dealloc(&mut self, off: usize) -> bool {
        let Some(n) = self.live.remove(&off) else {
            return false; // double-free / unknown -> no-op
        };
        self.free.push((off, n));
        self.free.sort_by_key(|b| b.0);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(self.free.len());
        for &(o, l) in &self.free {
            if let Some(last) = merged.last_mut() {
                if last.0 + last.1 == o {
                    last.1 += l;
                    continue;
                }
            }
            merged.push((o, l));
        }
        self.free = merged;
        true
    }

    fn bytes_free(&self) -> usize {
        self.free.iter().map(|b| b.1).sum()
    }

    /// The fragmentation gauge: the biggest single allocation that can
    /// currently succeed. Collapsing toward zero while `bytes_free` stays
    /// high is the telltale of fragmentation.
    fn largest_free_block(&self) -> usize {
        self.free.iter().map(|&(_, l)| l).max().unwrap_or(0)
    }
}

/// A named shared-memory arena. The producer owns it (unlinks on drop); a
/// consumer opens the same name read/write without ownership.
pub struct ShmArena {
    mmap: MmapMut,
    len: usize,
    path: String,
    owner: bool,
    alloc: Mutex<Allocator>,
}

// The mmap intentionally aliases across processes; in-process access is
// serialised by the allocator Mutex (and, from Python, the GIL).
unsafe impl Send for ShmArena {}
unsafe impl Sync for ShmArena {}

fn shm_path(name: &str) -> String {
    if Path::new("/dev/shm").is_dir() {
        format!("/dev/shm/{name}")
    } else {
        format!("/tmp/{name}")
    }
}

/// Size the segment file AND commit its blocks. `set_len` alone leaves a
/// sparse tmpfs file: a full /dev/shm then surfaces as SIGBUS on the first
/// write to an unbacked page, instead of as an error here.
///
/// The free-space check comes first because tmpfs's `fallocate` only
/// compares the size with the filesystem's total: short of free space, it
/// commits and zeroes every free page before failing and undoing them
/// (about 90 ms with 200 MiB free), and meanwhile other writers to
/// /dev/shm see it full.
#[cfg(target_os = "linux")]
fn preallocate(file: &std::fs::File, size: usize) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    let len =
        libc::off_t::try_from(size).map_err(|_| std::io::Error::from_raw_os_error(libc::EFBIG))?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatvfs(file.as_raw_fd(), &mut st) } == 0 && st.f_blocks != 0 {
        let free = (st.f_bavail as u128) * (st.f_frsize as u128);
        if free < size as u128 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                format!("only {free} B free"),
            ));
        }
    }
    loop {
        // Returns the error number rather than setting errno.
        match unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len) } {
            0 => return Ok(()),
            libc::EINTR => continue,
            e => return Err(std::io::Error::from_raw_os_error(e)),
        }
    }
}

/// Elsewhere the file stays sparse, as before.
#[cfg(not(target_os = "linux"))]
fn preallocate(file: &std::fs::File, size: usize) -> std::io::Result<()> {
    file.set_len(size as u64)
}

fn is_no_space(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::StorageFull
            | std::io::ErrorKind::QuotaExceeded
            | std::io::ErrorKind::FileTooLarge
    )
}

impl ShmArena {
    /// Producer: create (or replace) the arena and own it (unlink on drop).
    /// Its pages are committed here; `NoSpace` (file removed) when the
    /// filesystem cannot hold them.
    pub fn create(name: &str, size: usize) -> Result<Self, ShmError> {
        if size == 0 {
            return Err(ShmError::ZeroSize);
        }
        let path = shm_path(name);
        let io = |source| ShmError::Io {
            path: path.clone(),
            source,
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(io)?;
        let mapped = preallocate(&file, size)
            .map_err(|source| {
                if is_no_space(&source) {
                    ShmError::NoSpace {
                        path: path.clone(),
                        size,
                        source,
                    }
                } else {
                    io(source)
                }
            })
            .and_then(|()| unsafe { MmapOptions::new().len(size).map_mut(&file) }.map_err(io));
        let mmap = match mapped {
            Ok(mmap) => mmap,
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                return Err(e);
            }
        };
        Ok(Self {
            mmap,
            len: size,
            path,
            owner: true,
            alloc: Mutex::new(Allocator::new(size)),
        })
    }

    /// Consumer: open an existing arena read/write; never unlinks.
    pub fn open(name: &str) -> Result<Self, ShmError> {
        let path = shm_path(name);
        let io = |source| ShmError::Io {
            path: path.clone(),
            source,
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(io)?;
        let size = file.metadata().map_err(io)?.len() as usize;
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file) }.map_err(io)?;
        Ok(Self {
            mmap,
            len: size,
            path,
            owner: false,
            alloc: Mutex::new(Allocator::new(size)), // unused on a consumer
        })
    }

    /// Reserve `nbytes`; returns the byte offset into the arena. Producer only.
    pub fn reserve(&self, nbytes: usize) -> Result<usize, ShmError> {
        let mut a = self.alloc.lock().expect("alloc lock");
        a.alloc(nbytes).ok_or_else(|| ShmError::Full {
            need: nbytes,
            free: a.bytes_free(),
            total: self.len,
        })
    }

    /// Release a reserved offset. Producer only. Idempotent (false = unknown).
    pub fn free(&self, offset: usize) -> bool {
        self.alloc.lock().expect("alloc lock").dealloc(offset)
    }

    pub fn size(&self) -> usize {
        self.len
    }

    pub fn bytes_free(&self) -> usize {
        self.alloc.lock().expect("alloc lock").bytes_free()
    }

    /// Largest single contiguous free block (see `Allocator::largest_free_block`).
    pub fn largest_free_block(&self) -> usize {
        self.alloc.lock().expect("alloc lock").largest_free_block()
    }

    /// Base pointer of the arena — used by the `mstar-py` buffer-protocol view
    /// so torch can `frombuffer(arena[off:off+n])` with no intermediate copy.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.mmap.as_ptr() as *mut u8
    }

    /// Byte slice into the arena (bounds-checked; for tests / Rust-side copies).
    pub fn bytes(&self, offset: usize, len: usize) -> &[u8] {
        &self.mmap[offset..offset + len]
    }

    /// Copy `src` into the arena at `offset` (producer D2H staging).
    pub fn write_at(&mut self, offset: usize, src: &[u8]) {
        self.mmap[offset..offset + src.len()].copy_from_slice(src);
    }

    pub fn close(&mut self) {
        if self.owner && !self.path.is_empty() {
            let _ = std::fs::remove_file(&self.path);
            self.path.clear();
        }
    }
}

impl Drop for ShmArena {
    fn drop(&mut self) {
        self.close();
    }
}

// ---------------------------------------------------------------------------
// Segmented arena: grow-by-segments + uuid-grouped reclaim
// ---------------------------------------------------------------------------

/// A producer arena that grows by adding fixed-size segments instead of
/// erroring (or resizing) when full.
///
/// Why segments, not `mremap`: each segment's mapping is created once and
/// never moves, so a CUDA host-registration (`cudaHostRegister`) of a segment
/// stays valid for its lifetime — the property that keeps D2H/H2D copies on
/// side streams truly asynchronous. A dynamically-resized arena would
/// invalidate the registration on every growth. Registration itself is the
/// embedder's job (torch/cuda): watch `num_segments()` and register each new
/// segment's `(ptr, len)` once.
///
/// Descriptors are `(segment_name, offset)`: every segment is an ordinary
/// named [`ShmArena`] (`{base}.seg{i}`), so consumers keep opening arenas
/// lazily by name — a consumer never needs to know segmentation exists.
///
/// Reclaim is offset-based: the embedder tracks each allocation's
/// `(segment, offset)` and calls [`Self::free`] when the consumer has
/// ACKed the transfer.
pub struct SegmentedShmArena {
    base: String,
    segment_size: usize,
    max_segments: usize,
    // Behind a Mutex so every method takes `&self` (interior mutability):
    // the Python binding releases the GIL across reserve's growth path, and
    // a `&mut` PyO3 borrow held there would make any concurrent `&self`
    // call (stats, free) raise "Already mutably borrowed".
    // Growth serializes on this lock instead of the PyCell borrow.
    segments: Mutex<Vec<std::sync::Arc<ShmArena>>>,
    // The last growth that failed for lack of space; only touched with
    // `segments` held.
    failed_growth: Mutex<Option<FailedGrowth>>,
}

/// How long a growth that failed for lack of space is not tried again for
/// the same size or larger. Sends that miss the existing segments meanwhile
/// fail at once (spill, or strict mode's backpressure loop retrying every
/// 2 ms) instead of each repeating the failure, which the free-space check
/// cannot always predict (tmpfs quotas, a race with another writer).
const GROWTH_RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(1);

struct FailedGrowth {
    size: usize,
    at: std::time::Instant,
    reason: String,
}

impl SegmentedShmArena {
    /// Create with one initial segment. `max_segments` caps total growth
    /// (the arena-full backpressure boundary): `segment_size * max_segments`
    /// bytes, after which `reserve` reports `Full`.
    pub fn create(
        base: &str,
        segment_size: usize,
        max_segments: usize,
    ) -> Result<Self, ShmError> {
        if segment_size == 0 || max_segments == 0 {
            return Err(ShmError::ZeroSize);
        }
        let first = ShmArena::create(&format!("{base}.seg0"), segment_size)?;
        Ok(Self {
            base: base.to_string(),
            segment_size,
            max_segments,
            segments: Mutex::new(vec![std::sync::Arc::new(first)]),
            failed_growth: Mutex::new(None),
        })
    }

    /// Reserve `nbytes`; returns `(segment_index, offset)`. Tries existing
    /// segments first-fit, then grows by one segment (a dedicated one when
    /// `nbytes` exceeds the segment size), up to `max_segments`.
    pub fn reserve(&self, nbytes: usize) -> Result<(usize, usize), ShmError> {
        let mut segments = self.segments.lock().expect("segments lock");
        for (i, seg) in segments.iter().enumerate() {
            if let Ok(off) = seg.reserve(nbytes) {
                return Ok((i, off));
            }
        }
        let grow = align_up(nbytes.max(1)).filter(|_| segments.len() < self.max_segments);
        let Some(aligned) = grow else {
            return Err(ShmError::Full {
                need: nbytes,
                free: segments.iter().map(|s| s.bytes_free()).sum(),
                total: segments.iter().map(|s| s.size()).sum(),
            });
        };
        let idx = segments.len();
        let size = self.segment_size.max(aligned);
        let name = format!("{}.seg{idx}", self.base);
        let mut failed = self.failed_growth.lock().expect("failed_growth lock");
        if let Some(f) = failed
            .as_ref()
            .filter(|f| size >= f.size && f.at.elapsed() < GROWTH_RETRY_AFTER)
        {
            return Err(ShmError::NoSpace {
                path: shm_path(&name),
                size,
                source: std::io::Error::new(
                    std::io::ErrorKind::StorageFull,
                    format!(
                        "{} (a {} B segment failed {} ms ago; not retried for {} ms)",
                        f.reason,
                        f.size,
                        f.at.elapsed().as_millis(),
                        GROWTH_RETRY_AFTER.as_millis()
                    ),
                ),
            });
        }
        let seg = match ShmArena::create(&name, size) {
            Ok(seg) => seg,
            Err(e) => {
                if let ShmError::NoSpace { source, .. } = &e {
                    *failed = Some(FailedGrowth {
                        size,
                        at: std::time::Instant::now(),
                        reason: source.to_string(),
                    });
                }
                return Err(e);
            }
        };
        if failed.as_ref().is_some_and(|f| size >= f.size) {
            *failed = None;
        }
        let off = seg.reserve(nbytes)?; // fresh segment sized to fit: infallible
        segments.push(std::sync::Arc::new(seg));
        Ok((idx, off))
    }

    /// Release one offset (embedder-tracked descriptors). Idempotent.
    pub fn free(&self, segment: usize, offset: usize) -> bool {
        self.segments
            .lock()
            .expect("segments lock")
            .get(segment)
            .is_some_and(|s| s.free(offset))
    }

    pub fn num_segments(&self) -> usize {
        self.segments.lock().expect("segments lock").len()
    }

    /// Occupancy + fragmentation snapshot across all segments:
    /// `(total_bytes, free_bytes, largest_free_block)`. The fragmentation
    /// signature is `largest_free_block` collapsing while `free_bytes` stays
    /// high — allocations then fail (or force segment growth) even though
    /// total free space looks healthy.
    pub fn stats(&self) -> (usize, usize, usize) {
        let segments = self.segments.lock().expect("segments lock");
        let total = segments.iter().map(|s| s.size()).sum();
        let free = segments.iter().map(|s| s.bytes_free()).sum();
        let largest = segments
            .iter()
            .map(|s| s.largest_free_block())
            .max()
            .unwrap_or(0);
        (total, free, largest)
    }

    /// The `{base}.seg{i}` arena name — what descriptors carry, and what a
    /// consumer passes to [`ShmArena::open`].
    pub fn segment_name(&self, i: usize) -> String {
        format!("{}.seg{i}", self.base)
    }

    /// A shared handle to segment `i` (buffer views, registration hooks).
    pub fn segment(&self, i: usize) -> Option<std::sync::Arc<ShmArena>> {
        self.segments.lock().expect("segments lock").get(i).cloned()
    }

    /// `(base_ptr, len)` of segment `i` — the registration hook: the embedder
    /// `cudaHostRegister`s each new segment ONCE; the mapping never moves.
    pub fn segment_ptr_len(&self, i: usize) -> Option<(*mut u8, usize)> {
        self.segments
            .lock()
            .expect("segments lock")
            .get(i)
            .map(|s| (s.as_mut_ptr(), s.size()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(tag: &str) -> String {
        format!("mstar_shm_test_{tag}_{:?}", std::thread::current().id())
    }

    #[test]
    fn alloc_free_coalesce() {
        let mut a = Allocator::new(1024);
        let x = a.alloc(200).unwrap(); // -> 256 aligned
        let y = a.alloc(200).unwrap();
        assert_eq!(x, 0);
        assert_eq!(y, 256);
        assert_eq!(a.bytes_free(), 1024 - 512);
        a.dealloc(x);
        a.dealloc(y);
        // Fully coalesced back to one free block.
        assert_eq!(a.free, vec![(0, 1024)]);
        assert!(!a.dealloc(9999)); // unknown offset
    }

    #[test]
    fn producer_consumer_roundtrip_same_process() {
        let n = name("rt");
        let mut prod = ShmArena::create(&n, 4096).unwrap();
        let off = prod.reserve(1500).unwrap();
        let payload: Vec<u8> = (0..1500).map(|i| (i % 251) as u8).collect();
        prod.write_at(off, &payload);

        // Consumer opens the same name and reads at the descriptor offset.
        let cons = ShmArena::open(&n).unwrap();
        assert_eq!(cons.bytes(off, 1500), &payload[..]);

        prod.free(off);
        assert_eq!(prod.bytes_free(), 4096);
    }

    #[test]
    fn reserve_reports_full() {
        let n = name("full");
        let arena = ShmArena::create(&n, 512).unwrap();
        arena.reserve(300).unwrap(); // -> 512 aligned, arena full
        assert!(matches!(arena.reserve(1), Err(ShmError::Full { .. })));
    }

    #[test]
    fn zero_size_rejected() {
        assert!(matches!(
            ShmArena::create(&name("z"), 0),
            Err(ShmError::ZeroSize)
        ));
    }

    // ---- segmented arena (grow-by-segments + uuid reclaim) ----------------

    #[test]
    fn segments_grow_under_pressure_and_names_are_openable() {
        let base = name("seg");
        let mut a = SegmentedShmArena::create(&base, 1024, 4).unwrap();
        assert_eq!(a.num_segments(), 1);

        // Fill segment 0, forcing growth into segment 1.
        let (s0, _) = a.reserve(900).unwrap();
        let (s1, o1) = a.reserve(900).unwrap();
        assert_eq!((s0, s1), (0, 1));
        assert_eq!(a.num_segments(), 2);

        // A late segment is an ordinary named arena a consumer can open —
        // exactly what a (segment_name, offset) descriptor promises.
        let cons = ShmArena::open(&a.segment_name(1)).unwrap();
        assert_eq!(cons.size(), 1024);
        let _ = (cons.bytes(o1, 4), ());

        // The registration hook surface: stable (ptr, len) per segment.
        let (p0, l0) = a.segment_ptr_len(0).unwrap();
        assert!(!p0.is_null());
        assert_eq!(l0, 1024);
    }

    #[test]
    fn oversized_allocation_gets_dedicated_segment() {
        let base = name("big");
        let mut a = SegmentedShmArena::create(&base, 1024, 4).unwrap();
        let (seg, off) = a.reserve(5000).unwrap(); // > segment_size
        assert_eq!((seg, off), (1, 0));
        assert!(a.segment(1).unwrap().size() >= 5000);
    }

    #[test]
    fn growth_cap_reports_full() {
        let base = name("cap");
        let mut a = SegmentedShmArena::create(&base, 512, 2).unwrap();
        a.reserve(500).unwrap(); // fills seg 0
        a.reserve(500).unwrap(); // grows + fills seg 1
        assert!(matches!(a.reserve(500), Err(ShmError::Full { .. })));
        // Freeing makes room again (backpressure boundary, not a dead end).
        assert!(a.free(0, 0));
        assert!(matches!(a.reserve(500), Ok((0, 0))));
    }

    #[test]
    fn overflowing_reserve_is_full_and_changes_nothing() {
        let arena = ShmArena::create(&name("ovf"), 4096).unwrap();
        for n in [usize::MAX, usize::MAX - ALIGN + 2] {
            assert!(matches!(arena.reserve(n), Err(ShmError::Full { .. })));
        }
        assert_eq!(arena.bytes_free(), 4096);
        assert_eq!(arena.reserve(256).unwrap(), 0);
        assert_eq!(arena.reserve(256).unwrap(), 256);

        let a = SegmentedShmArena::create(&name("ovfseg"), 1024, 4).unwrap();
        assert!(matches!(a.reserve(usize::MAX), Err(ShmError::Full { .. })));
        assert_eq!(a.num_segments(), 1);
        assert_eq!(a.stats(), (1024, 1024, 1024));
    }

    /// A size larger than the whole filesystem behind the arena: creating
    /// a segment that big fails before touching a page, so tests exercise
    /// the no-space paths without filling the real /dev/shm. `None` on an
    /// unlimited tmpfs, which has no size to exceed.
    #[cfg(target_os = "linux")]
    fn bigger_than_the_filesystem() -> Option<usize> {
        let dir = shm_path("x");
        let dir = std::ffi::CString::new(dir.strip_suffix("/x").unwrap()).unwrap();
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { libc::statvfs(dir.as_ptr(), &mut st) }, 0);
        (st.f_blocks != 0).then(|| (st.f_blocks as usize) * (st.f_frsize as usize) + (1 << 20))
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn segment_larger_than_the_filesystem_is_no_space_and_leaves_no_file() {
        let Some(too_big) = bigger_than_the_filesystem() else {
            return;
        };

        let n = name("nospace");
        assert!(matches!(
            ShmArena::create(&n, too_big),
            Err(ShmError::NoSpace { .. })
        ));
        assert!(!Path::new(&shm_path(&n)).exists());

        let a = SegmentedShmArena::create(&name("nospaceseg"), 4096, 2).unwrap();
        assert!(matches!(a.reserve(too_big), Err(ShmError::NoSpace { .. })));
        assert_eq!(a.num_segments(), 1);
        assert!(!Path::new(&shm_path(&a.segment_name(1))).exists());
        assert!(matches!(a.reserve(100), Ok((0, 0))));
    }

    /// Sends that keep missing the existing segments must not repeat a
    /// growth that just failed for lack of space; a smaller growth is still
    /// tried, and the failed size is tried again once the window passes.
    #[cfg(target_os = "linux")]
    #[test]
    fn failed_growth_is_not_retried_within_the_window() {
        let Some(too_big) = bigger_than_the_filesystem() else {
            return;
        };
        let a = SegmentedShmArena::create(&name("regrow"), 4096, 3).unwrap();
        let failed_at = std::time::Instant::now();
        let first = a.reserve(too_big).unwrap_err().to_string();
        assert!(first.starts_with("arena full: no space"), "{first}");
        assert!(!first.contains("not retried"), "{first}");
        // A stall past the window legitimately allows the retry.
        let remembered =
            |e: &str| e.contains("not retried") || failed_at.elapsed() >= GROWTH_RETRY_AFTER;
        let again = a.reserve(too_big).unwrap_err();
        assert!(matches!(again, ShmError::NoSpace { .. }));
        assert!(remembered(&again.to_string()), "{again}");

        assert_eq!(a.reserve(4096).unwrap(), (0, 0));
        assert_eq!(a.reserve(100).unwrap(), (1, 0));
        let still = a.reserve(too_big).unwrap_err().to_string();
        assert!(remembered(&still), "{still}");

        std::thread::sleep(GROWTH_RETRY_AFTER + std::time::Duration::from_millis(50));
        let later = a.reserve(too_big).unwrap_err().to_string();
        assert!(!later.contains("not retried"), "{later}");
        assert_eq!(a.num_segments(), 2);
    }
}
