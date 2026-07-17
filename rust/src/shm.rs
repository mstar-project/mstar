//! Shared-memory tensor arena — the data plane for cross-process transport.
//!
//! One persistent `/dev/shm` mmap per producer entity plus an in-process
//! first-fit free-list allocator, replacing the per-tensor file open/write/
//! read/unlink dance in `SharedMemoryCommunicationManager` (the Python
//! buffer-protocol view lives in `lib.rs`).
//!
//! - Producer: [`ShmArena::create`], [`ShmArena::reserve`] -> offset, copy
//!   D2H bytes into `arena[off..off+n]`, send the offset as a descriptor,
//!   [`ShmArena::free`] the offset once the tensor is reclaimed. The conductor
//!   drives reclaim per-tensor: the runtime reports a tensor unreachable
//!   (`Event::Free`) and the conductor frees the owning arena's offset (its
//!   own directly, a worker's via a `free` message), with a per-request sweep
//!   as the backstop.
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

#[inline]
fn align_up(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
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
        let n = align_up(n.max(1));
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

impl ShmArena {
    /// Producer: create (or replace) the arena and own it (unlink on drop).
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
        file.set_len(size as u64).map_err(io)?;
        let mmap = unsafe { MmapOptions::new().len(size).map_mut(&file) }.map_err(io)?;
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
/// Reclaim is uuid-based (mstar's cleanup contract): [`Self::reserve_for`]
/// records the allocation under the caller's uuid and [`Self::free_uuid`]
/// releases everything held by that uuid, idempotently. Offset-based
/// [`Self::free`] remains for embedders that track descriptors themselves.
///
/// Allocations larger than the segment size get a dedicated segment of
/// exactly that size (freed segments are reused like any other).
pub struct SegmentedShmArena {
    base: String,
    segment_size: usize,
    max_segments: usize,
    segments: Vec<std::sync::Arc<ShmArena>>,
    // uuid -> allocations it holds. Multi-tensor uuids supported.
    ledger: Mutex<HashMap<u64, Vec<(usize, usize)>>>, // (segment idx, offset)
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
            segments: vec![std::sync::Arc::new(first)],
            ledger: Mutex::new(HashMap::new()),
        })
    }

    /// Reserve `nbytes`; returns `(segment_index, offset)`. Tries existing
    /// segments first-fit, then grows by one segment (a dedicated one when
    /// `nbytes` exceeds the segment size), up to `max_segments`.
    pub fn reserve(&mut self, nbytes: usize) -> Result<(usize, usize), ShmError> {
        for (i, seg) in self.segments.iter().enumerate() {
            if let Ok(off) = seg.reserve(nbytes) {
                return Ok((i, off));
            }
        }
        if self.segments.len() >= self.max_segments {
            return Err(ShmError::Full {
                need: nbytes,
                free: self.segments.iter().map(|s| s.bytes_free()).sum(),
                total: self.segments.iter().map(|s| s.size()).sum(),
            });
        }
        let idx = self.segments.len();
        let size = self.segment_size.max(align_up(nbytes.max(1)));
        let seg = ShmArena::create(&format!("{}.seg{idx}", self.base), size)?;
        let off = seg.reserve(nbytes)?; // fresh segment sized to fit: infallible
        self.segments.push(std::sync::Arc::new(seg));
        Ok((idx, off))
    }

    /// Reserve `nbytes` and record the allocation under `uuid` (mstar's
    /// uuid-based reclaim). One uuid may hold many allocations.
    pub fn reserve_for(&mut self, uuid: u64, nbytes: usize) -> Result<(usize, usize), ShmError> {
        let (seg, off) = self.reserve(nbytes)?;
        self.ledger
            .lock()
            .expect("ledger lock")
            .entry(uuid)
            .or_default()
            .push((seg, off));
        Ok((seg, off))
    }

    /// Free everything held by `uuid`. Idempotent: an unknown uuid frees
    /// nothing. Returns the number of allocations released.
    pub fn free_uuid(&self, uuid: u64) -> usize {
        let allocs = self
            .ledger
            .lock()
            .expect("ledger lock")
            .remove(&uuid)
            .unwrap_or_default();
        let mut n = 0;
        for (seg, off) in allocs {
            if self.segments.get(seg).is_some_and(|s| s.free(off)) {
                n += 1;
            }
        }
        n
    }

    /// Release one offset (embedder-tracked descriptors). Idempotent.
    pub fn free(&self, segment: usize, offset: usize) -> bool {
        self.segments
            .get(segment)
            .is_some_and(|s| s.free(offset))
    }

    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Occupancy + fragmentation snapshot across all segments:
    /// `(total_bytes, free_bytes, largest_free_block)`. The fragmentation
    /// signature is `largest_free_block` collapsing while `free_bytes` stays
    /// high — allocations then fail (or force segment growth) even though
    /// total free space looks healthy.
    pub fn stats(&self) -> (usize, usize, usize) {
        let total = self.segments.iter().map(|s| s.size()).sum();
        let free = self.segments.iter().map(|s| s.bytes_free()).sum();
        let largest = self
            .segments
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
        self.segments.get(i).cloned()
    }

    /// `(base_ptr, len)` of segment `i` — the registration hook: the embedder
    /// `cudaHostRegister`s each new segment ONCE; the mapping never moves.
    pub fn segment_ptr_len(&self, i: usize) -> Option<(*mut u8, usize)> {
        self.segments.get(i).map(|s| (s.as_mut_ptr(), s.size()))
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
    fn uuid_reclaim_is_grouped_and_idempotent() {
        let base = name("uuid");
        let mut a = SegmentedShmArena::create(&base, 4096, 4).unwrap();
        a.reserve_for(7, 100).unwrap();
        a.reserve_for(7, 100).unwrap(); // same uuid holds two allocations
        a.reserve_for(8, 100).unwrap();
        assert_eq!(a.free_uuid(7), 2);
        assert_eq!(a.free_uuid(7), 0); // idempotent
        assert_eq!(a.free_uuid(999), 0); // unknown uuid: no-op
        assert_eq!(a.free_uuid(8), 1);
        assert_eq!(a.segment(0).unwrap().bytes_free(), 4096);
    }
}
