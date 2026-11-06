//! Where a staged tensor's bytes go, and everything that keeps that place
//! usable: growth, page-locking, backpressure, spill files, reclaim, and the
//! mappings of peers' segments. `arena_copy.rs` drives the copies on top of it,
//! and `mstar/communication/arena.py` takes the same calls on the paths that
//! read tensors through torch.
//!
//! Capacity degrades in layers: an existing segment, a new one up to the cap, a
//! wait for consumers' ACKs, then a per-uuid file. Runs without the GIL except
//! where a method says so, and never calls Python: what is worth logging waits
//! in [`Notes`] for the caller to flush holding the GIL and no lock.

use std::collections::HashMap;
use std::os::raw::c_void;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;

use crate::shm::{create_exclusive, our_uid, shm_path, SegmentedShmArena, ShmArena};

/// `cudaErrorHostMemoryAlreadyRegistered`; HIP's has the same number.
const ALREADY_REGISTERED: i32 = 712;
const BACKPRESSURE_POLL: Duration = Duration::from_millis(2);

pub const LOG_INFO: i32 = 20;
pub const LOG_WARNING: i32 = 30;

/// Log lines waiting for a caller holding the GIL and no lock of this module.
#[derive(Default)]
pub struct Notes(Mutex<Vec<(i32, String)>>);

impl Notes {
    fn add(&self, level: i32, message: String) {
        self.0.lock().expect("notes lock").push((level, message));
    }

    pub fn take(&self) -> Vec<(i32, String)> {
        std::mem::take(&mut *self.0.lock().expect("notes lock"))
    }
}

/// `time.monotonic()`, so a stamp means the same on both sides of the boundary.
pub fn monotonic_now() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// Where a staged uuid's bytes are. `seg` is a segment index, or one of the two
/// markers below.
#[derive(Clone, Copy)]
pub struct Slot {
    pub seg: u32,
    pub offset: u64,
    pub staged_at: f64,
}

/// A claim: some thread is staging the uuid now and has yet to publish.
pub const PENDING: u32 = u32::MAX;
/// Spilled to its per-uuid file.
pub const IN_FILE: u32 = u32::MAX - 1;

#[derive(Default)]
pub struct SlotTable(Mutex<FxHashMap<u64, Slot>>);

impl SlotTable {
    pub fn lock(&self) -> std::sync::MutexGuard<'_, FxHashMap<u64, Slot>> {
        self.0.lock().expect("slot table lock")
    }
}

/// The per-uuid files a tensor goes to when no arena can hold it.
pub struct SpillFiles {
    dir: String,
    /// `mstar_<namespace>_`, or `mstar_` with no deployment namespace.
    prefix: String,
    entity: String,
}

/// Why a stage failed, with the errno of a failed file write, which a Python
/// caller raises as `OSError`.
#[derive(Debug)]
pub struct StageError {
    pub errno: Option<i32>,
    pub message: String,
}

impl SpillFiles {
    pub fn new(dir: String, prefix: String, entity: String) -> Self {
        Self {
            dir,
            prefix,
            entity,
        }
    }

    pub fn path(&self, entity: &str, uuid: u64) -> String {
        format!("{}/{}{entity}_{uuid}", self.dir, self.prefix)
    }

    fn own_path(&self, uuid: u64) -> String {
        self.path(&self.entity, uuid)
    }

    /// Write one tensor's bytes to its own file. A failed write leaves no
    /// partial file: nothing would ever reclaim its space.
    pub fn write(&self, uuid: u64, data: &[u8]) -> Result<(), StageError> {
        use std::io::Write;
        let path = self.own_path(uuid);
        let mut file = Self::create(&path, our_uid())?;
        let written = file.write_all(data);
        // Closed before the failure is reported: its free-space count should not
        // include what this partial file holds.
        drop(file);
        written.map_err(|e| {
            let _ = std::fs::remove_file(&path);
            self.write_error(&path, data.len(), e)
        })
    }

    /// O_EXCL because the name is fully predictable in a world-writable
    /// directory: a dead run's residue we own is replaced, another uid's is
    /// refused.
    fn create(path: &str, uid: u32) -> Result<std::fs::File, StageError> {
        create_exclusive(path, uid).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                StageError {
                    errno: None,
                    message: format!(
                        "refusing to write tensor file {path}: {e}. Another user or an unrelated \
                         deployment holds this path; give this deployment its own ZMQ endpoint so \
                         it derives a different namespace."
                    ),
                }
            } else {
                StageError {
                    errno: e.raw_os_error(),
                    message: format!("tensor file {path}: {e}"),
                }
            }
        })
    }

    fn write_error(&self, path: &str, nbytes: usize, e: std::io::Error) -> StageError {
        if e.raw_os_error() != Some(libc::ENOSPC) {
            return StageError {
                errno: e.raw_os_error(),
                message: format!("tensor file {path}: {e}"),
            };
        }
        let free = free_bytes(&self.dir);
        let reason = if free < nbytes as u128 {
            format!(
                "{} has {free} bytes free, too few for a {nbytes}-byte tensor",
                self.dir
            )
        } else {
            format!(
                "{} reported no space for a {nbytes}-byte tensor, with {free} bytes free after \
                 the failed write: it may be out of inodes, or other writers held the space \
                 meanwhile",
                self.dir
            )
        };
        StageError {
            errno: Some(libc::ENOSPC),
            message: format!(
                "{reason}. Give it more room (docker --shm-size), or pass \
                 --tensor-comm-protocol TCP, which is slower"
            ),
        }
    }

    /// `dst.len()` bytes of a spilled tensor from `offset` (a TP slice's start).
    pub fn read(path: &str, offset: u64, dst: &mut [u8]) -> std::io::Result<()> {
        std::fs::File::open(path)?.read_exact_at(dst, offset)
    }

    fn unlink(&self, uuid: u64) {
        let _ = std::fs::remove_file(self.own_path(uuid));
    }
}

fn free_bytes(dir: &str) -> u128 {
    let Ok(dir) = std::ffi::CString::new(dir) else {
        return 0;
    };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(dir.as_ptr(), &mut st) } != 0 {
        return 0;
    }
    st.f_bavail as u128 * st.f_frsize as u128
}

/// The runtime's host-registration calls, from a library the manager loaded:
/// this crate links no vendor library.
#[derive(Clone, Copy)]
struct PinFns {
    register: unsafe extern "C" fn(*mut c_void, usize, u32) -> i32,
    unregister: unsafe extern "C" fn(*mut c_void) -> i32,
    get_last_error: unsafe extern "C" fn() -> i32,
}

/// Pinned memory comes out of the OS's pageable pool, so it has its own budget.
/// The lock is held across a registration: `release` has to find a segment
/// either not pinned yet or on its list, and registering releases nothing else.
#[derive(Default)]
struct PinState {
    fns: Option<PinFns>,
    budget: usize,
    bytes: usize,
    segments: usize,
    own: Vec<usize>,
    budget_warned: bool,
}

impl PinState {
    /// The bytes it pinned: 0 past the budget or on failure, and copies through
    /// that memory still work, without the async overlap.
    fn pin(&mut self, ptr: *mut u8, nbytes: usize, notes: &Notes) -> usize {
        let Some(fns) = self.fns else {
            return 0;
        };
        if self.bytes + nbytes > self.budget {
            if !self.budget_warned {
                self.budget_warned = true;
                notes.add(
                    LOG_WARNING,
                    format!(
                        "ARENA: pinned-memory budget reached ({} MiB, MSTAR_SHM_ARENA_PIN_MAX_MB); \
                         further segments stay unpinned — copies work but lose async overlap",
                        self.budget >> 20
                    ),
                );
            }
            return 0;
        }
        let rc = unsafe { (fns.register)(ptr.cast(), nbytes, 0) };
        if rc != 0 {
            // Set by an already-registered range too; left set, this thread's
            // next torch kernel launch raises it.
            unsafe { (fns.get_last_error)() };
        }
        if rc != 0 && rc != ALREADY_REGISTERED {
            notes.add(
                LOG_WARNING,
                format!(
                    "host registration of {nbytes} bytes at {ptr:p} failed rc={rc}; copies fall \
                     back to pageable bandwidth"
                ),
            );
            return 0;
        }
        self.bytes += nbytes;
        self.segments += 1;
        nbytes
    }

    fn unpin(&self, ptr: usize) -> bool {
        let Some(fns) = self.fns else {
            return false;
        };
        let rc = unsafe { (fns.unregister)(ptr as *mut c_void) };
        if rc != 0 {
            unsafe { (fns.get_last_error)() };
        }
        rc == 0
    }
}

/// A peer's segment, mapped once. Stays until its file is gone.
pub struct Peer {
    pub arena: Arc<ShmArena>,
    pinned: usize,
}

pub struct CapacityPolicy {
    /// Past the wait, spill rather than fail.
    pub spill: bool,
    pub spill_after_s: f64,
    pub full_timeout_s: f64,
    /// Slots older than this are freed by [`Capacity::reclaim_expired`]; 0 is off.
    pub slot_ttl_s: f64,
}

/// What [`Capacity::reserve`] found.
pub enum Reserved {
    Slot {
        seg: usize,
        offset: usize,
        base: *mut u8,
    },
    /// The arena cannot hold it: the caller writes a file.
    Spill,
    Failed(String),
}

pub struct Capacity {
    pub arena: Option<Arc<SegmentedShmArena>>,
    /// Segments a reserve may land in: only ones already pinned.
    usable: AtomicUsize,
    /// Serializes growth, so a new segment is usable only once it is pinned.
    growth: Mutex<()>,
    pub slots: SlotTable,
    pub spill: SpillFiles,
    policy: CapacityPolicy,
    pins: Mutex<PinState>,
    peers: RwLock<HashMap<String, Peer>>,
    pub notes: Notes,
    shm_total: Option<usize>,
    frag_warned: AtomicBool,
    spill_warned: AtomicBool,
    ttl_reclaimed: AtomicUsize,
}

impl Capacity {
    pub fn new(
        arena: Option<Arc<SegmentedShmArena>>,
        spill: SpillFiles,
        policy: CapacityPolicy,
        shm_total: Option<usize>,
    ) -> Self {
        let usable = arena.as_ref().map_or(0, |a| a.num_segments());
        Self {
            arena,
            usable: AtomicUsize::new(usable),
            growth: Mutex::new(()),
            slots: SlotTable::default(),
            spill,
            policy,
            pins: Mutex::new(PinState::default()),
            peers: RwLock::new(HashMap::new()),
            notes: Notes::default(),
            shm_total,
            frag_warned: AtomicBool::new(false),
            spill_warned: AtomicBool::new(false),
            ttl_reclaimed: AtomicUsize::new(0),
        }
    }

    /// Turn page-locking on and pin the segments already there.
    pub fn set_pins(
        &self,
        register: usize,
        unregister: usize,
        get_last_error: usize,
        budget: usize,
    ) {
        use std::mem::transmute as t;
        // The addresses are ctypes function pointers of the runtime's own symbols.
        let fns = unsafe {
            PinFns {
                register: t::<usize, unsafe extern "C" fn(*mut c_void, usize, u32) -> i32>(
                    register,
                ),
                unregister: t::<usize, unsafe extern "C" fn(*mut c_void) -> i32>(unregister),
                get_last_error: t::<usize, unsafe extern "C" fn() -> i32>(get_last_error),
            }
        };
        {
            let mut pins = self.pins.lock().expect("pins lock");
            pins.fns = Some(fns);
            pins.budget = budget;
        }
        let _growth = self.growth.lock().expect("growth lock");
        self.pin_own(0, self.usable.load(Ordering::Acquire));
    }

    fn pin_own(&self, from: usize, to: usize) {
        let Some(arena) = &self.arena else {
            return;
        };
        let mut pins = self.pins.lock().expect("pins lock");
        for i in from..to {
            if let Some((ptr, len)) = arena.segment_ptr_len(i) {
                if pins.pin(ptr, len, &self.notes) > 0 {
                    pins.own.push(ptr as usize);
                }
            }
        }
    }

    /// Without waiting for anything: `None` sends the caller to [`Self::reserve`],
    /// which it should enter without the GIL. A growth holds the segments lock
    /// across an mmap, so this does not wait on that lock either.
    pub fn reserve_now(&self, nbytes: usize) -> Option<(usize, usize, *mut u8)> {
        self.arena
            .as_ref()?
            .reserve_existing(nbytes.max(1), self.usable.load(Ordering::Acquire))
    }

    /// A place for `nbytes`: every layer but the file, which is the caller's.
    pub fn reserve(&self, nbytes: usize) -> Reserved {
        let Some(arena) = &self.arena else {
            return Reserved::Spill;
        };
        let nbytes = nbytes.max(1);
        let grace = if self.policy.spill {
            self.policy.spill_after_s
        } else {
            self.policy.full_timeout_s
        };
        // From a knob: an infinite wait has no deadline, and a negative one has passed.
        let deadline = Duration::try_from_secs_f64(grace.max(0.0))
            .ok()
            .and_then(|wait| Instant::now().checked_add(wait));
        let mut warned = false;
        loop {
            let error = {
                let _growth = self.growth.lock().expect("growth lock");
                match arena.reserve(nbytes) {
                    Ok((seg, offset)) => {
                        self.adopt_new_segments(arena);
                        let (base, _) = arena
                            .segment_ptr_len(seg)
                            .expect("the segment just reserved in");
                        return Reserved::Slot { seg, offset, base };
                    }
                    // Both the segment cap and a failed growth (no room in
                    // /dev/shm for a new segment) land here as "full".
                    Err(e) => e.to_string(),
                }
            };
            if self.reclaim_expired() > 0 {
                continue;
            }
            self.warn_if_fragmented(arena, nbytes);
            if deadline.is_some_and(|at| Instant::now() > at) {
                if self.policy.spill {
                    if !self.spill_warned.swap(true, Ordering::Relaxed) {
                        self.notes.add(
                            LOG_WARNING,
                            format!("ARENA: spilling sends to the file transport ({error})"),
                        );
                    }
                    return Reserved::Spill;
                }
                return Reserved::Failed(format!(
                    "SHM arena full for >{}s ({} segments: {error}); raise \
                     MSTAR_SHM_ARENA_MAX_SEGMENTS / _SEGMENT_MB, free /dev/shm space, or check \
                     for consumers not ACKing",
                    self.policy.full_timeout_s,
                    arena.num_segments(),
                ));
            }
            if !warned {
                warned = true;
                self.notes.add(
                    LOG_WARNING,
                    format!(
                        "SHM arena at capacity; backpressuring sends until consumers ACK{}",
                        if self.policy.spill {
                            " (then spilling to file)"
                        } else {
                            ""
                        }
                    ),
                );
            }
            std::thread::sleep(BACKPRESSURE_POLL);
        }
    }

    /// Runs with `growth` held.
    fn adopt_new_segments(&self, arena: &SegmentedShmArena) {
        let usable = self.usable.load(Ordering::Acquire);
        let now = arena.num_segments();
        if now == usable {
            return;
        }
        self.pin_own(usable, now);
        self.usable.store(now, Ordering::Release);
        let (total, free, largest) = arena.stats();
        if let Some(shm_total) = self.shm_total {
            if total as f64 > shm_total as f64 * 0.8 {
                self.notes.add(
                    LOG_WARNING,
                    format!(
                        "ARENA: this entity's segments now total {} MiB — over 80% of /dev/shm \
                         ({} MiB). Oversized dedicated segments grow past the static ceiling; \
                         other entities multiply this further.",
                        total >> 20,
                        shm_total >> 20
                    ),
                );
            }
        }
        self.notes.add(
            LOG_INFO,
            format!(
                "ARENA: grew to {now} segments ({} MiB total, {} MiB free, largest free block {} \
                 MiB, {} MiB pinned)",
                total >> 20,
                free >> 20,
                largest >> 20,
                self.pinned_bytes() >> 20
            ),
        );
    }

    fn warn_if_fragmented(&self, arena: &SegmentedShmArena, nbytes: usize) {
        if self.frag_warned.load(Ordering::Relaxed) {
            return;
        }
        let (total, free, largest) = arena.stats();
        if free >= nbytes {
            // Enough TOTAL free space, but no contiguous block large enough.
            self.frag_warned.store(true, Ordering::Relaxed);
            self.notes.add(
                LOG_WARNING,
                format!(
                    "ARENA: fragmentation — need {nbytes} bytes with {free} free but largest free \
                     block is {largest} (of {total} total). Consider larger \
                     MSTAR_SHM_ARENA_SEGMENT_MB."
                ),
            );
        }
    }

    pub fn spill_unlink(&self, uuid: u64) {
        self.spill.unlink(uuid);
    }

    pub fn return_slot(&self, seg: u32, offset: u64) {
        if let Some(arena) = &self.arena {
            arena.free(seg as usize, offset as usize);
        }
    }

    /// Give back what each uuid holds, a slot or a file. A uuid still being
    /// staged only loses its claim: its stager finds that out and gives back
    /// what it reserved.
    pub fn free(&self, uuids: impl IntoIterator<Item = u64>) {
        let mut held = Vec::new();
        {
            let mut slots = self.slots.lock();
            for uuid in uuids {
                if let Some(slot) = slots.remove(&uuid) {
                    if slot.seg != PENDING {
                        held.push((uuid, slot));
                    }
                }
            }
        }
        for (uuid, slot) in held {
            if slot.seg == IN_FILE {
                self.spill.unlink(uuid);
            } else {
                self.return_slot(slot.seg, slot.offset);
            }
        }
    }

    /// The TTL backstop: a request aborted after staging but before every
    /// consumer ACKs defers reclaim forever, and a slot older than the request
    /// timeout cannot have a legitimate reader. Returns how many it freed.
    pub fn reclaim_expired(&self) -> usize {
        if self.policy.slot_ttl_s <= 0.0 {
            return 0;
        }
        let cutoff = monotonic_now() - self.policy.slot_ttl_s;
        let expired: Vec<u64> = self
            .slots
            .lock()
            .iter()
            .filter(|(_, s)| s.seg != PENDING && s.staged_at < cutoff)
            .map(|(uuid, _)| *uuid)
            .collect();
        let n = expired.len();
        if n > 0 {
            self.free(expired);
            let total = self.ttl_reclaimed.fetch_add(n, Ordering::Relaxed) + n;
            self.notes.add(
                LOG_WARNING,
                format!(
                    "ARENA: TTL-reclaimed {n} slot(s)/file(s) older than {:.0}s ({total} total) — \
                     requests aborted without consumer ACKs; the ACK path is leaking",
                    self.policy.slot_ttl_s
                ),
            );
        }
        n
    }

    /// Every spill file this entity still owns, then its segments, for a process
    /// that is stopping. Each file is taken by one caller, so the orphan
    /// watchdog's thread can run this at the same time. Peers keep the mappings
    /// they have.
    pub fn unlink_owned(&self) {
        let files: Vec<u64> = {
            let mut slots = self.slots.lock();
            let files: Vec<u64> = slots
                .iter()
                .filter(|(_, s)| s.seg == IN_FILE)
                .map(|(uuid, _)| *uuid)
                .collect();
            for uuid in &files {
                slots.remove(uuid);
            }
            files
        };
        for uuid in files {
            self.spill.unlink(uuid);
        }
        self.unlink_segments();
    }

    /// A worker exits with its manager alive, so the Rust `Drop` that unlinks
    /// segments never runs without this.
    pub fn unlink_segments(&self) {
        if let Some(arena) = &self.arena {
            for i in 0..arena.num_segments() {
                let _ = std::fs::remove_file(shm_path(&arena.segment_name(i)));
            }
        }
    }

    pub fn peer(&self, name: &str) -> Option<Arc<ShmArena>> {
        self.peers
            .read()
            .expect("peers lock")
            .get(name)
            .map(|p| p.arena.clone())
    }

    /// Map (and pin, within budget) a peer's segment, once.
    pub fn map_peer(&self, name: &str) -> Result<Arc<ShmArena>, String> {
        if let Some(found) = self.peer(name) {
            return Ok(found);
        }
        let mut pins = self.pins.lock().expect("pins lock");
        // Lost the race: another thread mapped it while this one waited.
        if let Some(found) = self.peer(name) {
            return Ok(found);
        }
        let arena = Arc::new(ShmArena::open(name).map_err(|e| e.to_string())?);
        let pinned = pins.pin(arena.as_mut_ptr(), arena.size(), &self.notes);
        self.peers.write().expect("peers lock").insert(
            name.to_string(),
            Peer {
                arena: arena.clone(),
                pinned,
            },
        );
        Ok(arena)
    }

    /// Drop the mappings whose backing file is gone (a producer finished or
    /// restarted, which mints new names): an mmap never evicted would pile up on
    /// a long-lived consumer. The caller makes sure no copy still reads one.
    pub fn evict_dead_peers(&self) {
        let dead: Vec<String> = self
            .peers
            .read()
            .expect("peers lock")
            .keys()
            .filter(|name| !std::path::Path::new(&shm_path(name)).exists())
            .cloned()
            .collect();
        for name in dead {
            let mut pins = self.pins.lock().expect("pins lock");
            let mut peers = self.peers.write().expect("peers lock");
            let Some(peer) = peers.get(&name) else {
                continue;
            };
            if peer.pinned > 0 {
                // Unregister BEFORE unmapping: tearing down a mapping the runtime
                // still holds registered leaves a dangling pinned range. On
                // failure keep the entry, so the accounting stays truthful, and
                // retry next sweep.
                if !pins.unpin(peer.arena.as_mut_ptr() as usize) {
                    self.notes.add(
                        LOG_WARNING,
                        format!(
                            "ARENA: host unregister failed for dead peer {name}; keeping mapping \
                             until it succeeds"
                        ),
                    );
                    continue;
                }
                pins.bytes -= peer.pinned;
            }
            peers.remove(&name);
        }
    }

    /// Unregister everything pinned. Runs before the mappings go: one torn down
    /// still registered leaves the registration behind, and a later mapping at
    /// the same address is then refused. Segments after this stay unpinned.
    pub fn release_pins(&self) {
        let mut pins = self.pins.lock().expect("pins lock");
        for ptr in std::mem::take(&mut pins.own) {
            if !pins.unpin(ptr) {
                self.notes.add(
                    LOG_WARNING,
                    format!("ARENA: host unregister failed for own segment {ptr:#x}"),
                );
            }
        }
        let mut peers = self.peers.write().expect("peers lock");
        for (name, peer) in peers.iter_mut() {
            if peer.pinned > 0 {
                if !pins.unpin(peer.arena.as_mut_ptr() as usize) {
                    self.notes.add(
                        LOG_WARNING,
                        format!("ARENA: host unregister failed for peer {name}"),
                    );
                }
                peer.pinned = 0;
            }
        }
        pins.fns = None;
        pins.bytes = 0;
        pins.segments = 0;
    }

    pub fn pinned_bytes(&self) -> usize {
        self.pins.lock().expect("pins lock").bytes
    }

    pub fn pinned_own(&self) -> Vec<usize> {
        self.pins.lock().expect("pins lock").own.clone()
    }
}

/// `(arena slots, spill files)`.
pub fn slot_counts(slots: &FxHashMap<u64, Slot>) -> (usize, usize) {
    let files = slots.values().filter(|s| s.seg == IN_FILE).count();
    (slots.len() - files, files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capacity(
        tag: &str,
        dir: &std::path::Path,
        max_segments: usize,
        policy: CapacityPolicy,
    ) -> Capacity {
        let base = format!("mstar_cap_test_{tag}_{}", std::process::id());
        let arena = SegmentedShmArena::create(&base, 1 << 20, max_segments).unwrap();
        let spill = SpillFiles {
            dir: dir.to_str().unwrap().to_string(),
            prefix: "mstar_".to_string(),
            entity: "w0".to_string(),
        };
        Capacity::new(Some(Arc::new(arena)), spill, policy, None)
    }

    fn spilling() -> CapacityPolicy {
        CapacityPolicy {
            spill: true,
            spill_after_s: 0.0,
            full_timeout_s: 30.0,
            slot_ttl_s: 0.0,
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mstar_cap_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_new_segment_is_usable_only_after_growth_followed_up() {
        let dir = temp_dir("grow");
        let c = capacity("grow", &dir, 2, spilling());
        assert!(c.reserve_now(1 << 20).is_some());
        assert!(c.reserve_now(1).is_none(), "the one usable segment is full");
        assert!(matches!(c.reserve(1), Reserved::Slot { seg: 1, .. }));
        assert!(
            c.reserve_now(1).is_some(),
            "the grown segment is usable now"
        );
    }

    #[test]
    fn a_full_arena_spills_and_a_strict_one_fails_clearly() {
        let dir = temp_dir("full");
        let c = capacity("full", &dir, 1, spilling());
        assert!(matches!(c.reserve(1 << 20), Reserved::Slot { .. }));
        assert!(matches!(c.reserve(1), Reserved::Spill));
        let notes = c.notes.take();
        assert!(
            notes.iter().any(|(_, m)| m.contains("spilling sends")),
            "{notes:?}"
        );

        let strict = CapacityPolicy {
            spill: false,
            spill_after_s: 0.0,
            full_timeout_s: 0.01,
            slot_ttl_s: 0.0,
        };
        let c = capacity("strict", &dir, 1, strict);
        assert!(matches!(c.reserve(1 << 20), Reserved::Slot { .. }));
        match c.reserve(1) {
            Reserved::Failed(why) => assert!(why.contains("arena full"), "{why}"),
            _ => panic!("expected a failure"),
        }
    }

    #[test]
    fn a_wait_from_any_knob_value_does_not_panic() {
        let dir = temp_dir("knob");
        let mut negative = spilling();
        negative.spill_after_s = -1.0;
        let c = capacity("knob", &dir, 1, negative);
        assert!(matches!(c.reserve(1 << 20), Reserved::Slot { .. }));
        assert!(matches!(c.reserve(1), Reserved::Spill));
        let mut infinite = spilling();
        infinite.spill_after_s = f64::INFINITY;
        let c = capacity("knob2", &dir, 1, infinite);
        assert!(matches!(c.reserve(1), Reserved::Slot { .. }), "room left: no wait at all");
    }

    #[test]
    fn a_spill_file_round_trips_and_is_unlinked_by_free() {
        let dir = temp_dir("spill");
        let c = capacity("spill", &dir, 1, spilling());
        c.spill.write(7, &[1, 2, 3, 4]).unwrap();
        let path = c.spill.path("w0", 7);
        let mut read = [0u8; 2];
        SpillFiles::read(&path, 2, &mut read).unwrap();
        assert_eq!(read, [3, 4]);
        c.slots.lock().insert(
            7,
            Slot {
                seg: IN_FILE,
                offset: 0,
                staged_at: 0.0,
            },
        );
        c.free([7]);
        assert!(!std::path::Path::new(&path).exists());
    }

    #[test]
    fn a_stale_file_of_ours_is_replaced() {
        let dir = temp_dir("stale");
        let c = capacity("stale", &dir, 1, spilling());
        std::fs::write(c.spill.path("w0", 9), b"old bytes").unwrap();
        c.spill.write(9, b"new").unwrap();
        assert_eq!(std::fs::read(c.spill.path("w0", 9)).unwrap(), b"new");
    }

    #[test]
    fn a_file_of_another_uid_is_refused() {
        let dir = temp_dir("foreign");
        let c = capacity("foreign", &dir, 1, spilling());
        let path = c.spill.path("w0", 11);
        std::fs::write(&path, b"squatted").unwrap();
        let e = SpillFiles::create(&path, our_uid() + 1)
            .err()
            .expect("refused");
        assert!(e.message.contains("refusing to write"), "{}", e.message);
        assert_eq!(std::fs::read(&path).unwrap(), b"squatted");
    }

    #[test]
    fn a_full_dir_says_what_ran_out() {
        let dir = temp_dir("enospc");
        let c = capacity("enospc", &dir, 1, spilling());
        let enospc = || std::io::Error::from_raw_os_error(libc::ENOSPC);
        let too_small = c.spill.write_error("p", usize::MAX, enospc());
        assert_eq!(too_small.errno, Some(libc::ENOSPC));
        assert!(
            too_small.message.contains("too few for a"),
            "{}",
            too_small.message
        );
        assert!(
            too_small.message.contains("--shm-size"),
            "{}",
            too_small.message
        );
        let with_room = c.spill.write_error("p", 1, enospc());
        assert!(
            with_room.message.contains("out of inodes"),
            "{}",
            with_room.message
        );
    }

    #[test]
    fn the_ttl_backstop_frees_slots_and_files() {
        let dir = temp_dir("ttl");
        let mut policy = spilling();
        policy.slot_ttl_s = 1e-9;
        let c = capacity("ttl", &dir, 1, policy);
        let Reserved::Slot { seg, offset, .. } = c.reserve(64) else {
            panic!("expected a slot")
        };
        c.spill.write(2, b"x").unwrap();
        let staged_at = monotonic_now() - 1.0;
        {
            let mut slots = c.slots.lock();
            let slot = Slot {
                seg: seg as u32,
                offset: offset as u64,
                staged_at,
            };
            slots.insert(1, slot);
            slots.insert(
                2,
                Slot {
                    seg: IN_FILE,
                    offset: 0,
                    staged_at,
                },
            );
        }
        assert_eq!(c.reclaim_expired(), 2);
        assert!(c.slots.lock().is_empty());
        assert!(!std::path::Path::new(&c.spill.path("w0", 2)).exists());
    }

    #[test]
    fn unlink_owned_takes_each_file_once_and_leaves_slots_alone() {
        let dir = temp_dir("unlink");
        let c = capacity("unlink", &dir, 1, spilling());
        c.spill.write(3, b"x").unwrap();
        let mut slots = c.slots.lock();
        let file = Slot {
            seg: IN_FILE,
            offset: 0,
            staged_at: 0.0,
        };
        slots.insert(3, file);
        drop(slots);
        c.unlink_owned();
        c.unlink_owned();
        assert!(c.slots.lock().is_empty());
        assert!(!std::path::Path::new(&c.spill.path("w0", 3)).exists());
    }
}
