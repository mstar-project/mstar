//! The KV page pool's bookkeeping: the free list, each page's owner count and
//! seal, and every cache stream's page list (`KVManager`'s `PageArena` and
//! `StreamPages`; `mstar/engine/resources/kv/arena.py` picks this or the
//! Python version). The pages' storage and copies stay with `KVCache`.
//!
//! A page goes back to the free list when its last owner releases it. A
//! sealed page is never written again, so a second owner may read it, and
//! freeing clears the seal. Pages are handed out first in, first out, from
//! page 0, as the Python allocator does, so both give out the same pages.
//!
//! Each stream's pages are a `Vec` of their own: a stream that grows moves
//! only its own list, and memory follows the pages streams hold. A plan reads
//! the lists of a step's rows in one call (`plan_columns`).

use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum ArenaError {
    OutOfRange(i64),
    NoOwner(i32),
    NoStream(usize),
}

impl std::fmt::Display for ArenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArenaError::OutOfRange(page) => write!(f, "page {page} is outside the pool"),
            ArenaError::NoOwner(page) => write!(f, "page {page} released with no owner"),
            ArenaError::NoStream(sid) => write!(f, "stream {sid} is not in the arena"),
        }
    }
}

/// What a release did to a page.
#[derive(Debug, PartialEq)]
pub enum Released {
    /// its last owner let go: it goes back to the free list
    Freed,
    /// one owner is left
    Unshared,
    Held,
}

pub struct ArenaCore {
    free: VecDeque<i32>,
    owners: Vec<u32>,
    sealed: Vec<bool>,
    /// (acquires, a hash of every page they handed out), kept when set: ranks
    /// that allocate the same pages in the same order agree on it
    pub trail: Option<(u64, u64)>,
    streams: Vec<Vec<i32>>,
    free_sids: Vec<usize>,
}

impl ArenaCore {
    pub fn new(max_num_pages: usize) -> Self {
        Self {
            free: (0..max_num_pages as i32).collect(),
            owners: vec![0; max_num_pages],
            sealed: vec![false; max_num_pages],
            trail: None,
            streams: Vec::new(),
            free_sids: Vec::new(),
        }
    }

    pub fn max_num_pages(&self) -> usize {
        self.owners.len()
    }

    pub fn num_free(&self) -> usize {
        self.free.len()
    }

    /// The free pages, in the order they will be handed out.
    pub fn free_pages(&self) -> Vec<i32> {
        self.free.iter().copied().collect()
    }

    fn slot(&self, page: i64) -> Result<usize, ArenaError> {
        if page < 0 || page as usize >= self.owners.len() {
            return Err(ArenaError::OutOfRange(page));
        }
        Ok(page as usize)
    }

    /// `n` pages, each with one owner, or None when fewer are free.
    pub fn acquire(&mut self, n: usize) -> Option<Vec<i32>> {
        if self.free.len() < n {
            return None;
        }
        let pages: Vec<i32> = self.free.drain(..n).collect();
        for &page in &pages {
            self.owners[page as usize] = 1;
        }
        if let Some((count, hash)) = self.trail {
            self.trail = Some((count + 1, trail_hash(hash, count, &pages)));
        }
        Some(pages)
    }

    /// One more owner for `page`; true when that makes it shared (two owners).
    pub fn retain(&mut self, page: i64) -> Result<bool, ArenaError> {
        let i = self.slot(page)?;
        self.owners[i] += 1;
        Ok(self.owners[i] == 2)
    }

    /// One owner fewer for `page`. A freed page is unsealed here but reaches
    /// the free list only through `free`, after the whole release.
    pub fn release(&mut self, page: i64) -> Result<Released, ArenaError> {
        let i = self.slot(page)?;
        if self.owners[i] == 0 {
            return Err(ArenaError::NoOwner(page as i32));
        }
        self.owners[i] -= 1;
        Ok(match self.owners[i] {
            0 => {
                self.sealed[i] = false;
                Released::Freed
            }
            1 => Released::Unshared,
            _ => Released::Held,
        })
    }

    pub fn free(&mut self, pages: &[i32]) {
        self.free.extend(pages);
    }

    pub fn seal(&mut self, page: i64) -> Result<(), ArenaError> {
        let i = self.slot(page)?;
        self.sealed[i] = true;
        Ok(())
    }

    pub fn is_sealed(&self, page: i64) -> Result<bool, ArenaError> {
        Ok(self.sealed[self.slot(page)?])
    }

    pub fn owners(&self, page: i64) -> Result<u32, ArenaError> {
        Ok(self.owners[self.slot(page)?])
    }

    /// A new, empty stream; its id until `drop_stream`.
    pub fn new_stream(&mut self) -> usize {
        match self.free_sids.pop() {
            Some(sid) => sid,
            None => {
                self.streams.push(Vec::new());
                self.streams.len() - 1
            }
        }
    }

    pub fn drop_stream(&mut self, sid: usize) {
        if let Some(pages) = self.streams.get_mut(sid) {
            *pages = Vec::new();
            self.free_sids.push(sid);
        }
    }

    pub fn pages(&self, sid: usize) -> Result<&Vec<i32>, ArenaError> {
        self.streams.get(sid).ok_or(ArenaError::NoStream(sid))
    }

    pub fn pages_mut(&mut self, sid: usize) -> Result<&mut Vec<i32>, ArenaError> {
        self.streams.get_mut(sid).ok_or(ArenaError::NoStream(sid))
    }

    /// A plan label's index arrays (`kv_plan::plan_columns`) over the page
    /// lists of the streams `sids`, one per row.
    pub fn plan_columns(
        &self,
        sids: &[usize],
        lens: &[i64],
        spans: &[i64],
        page_size: i64,
    ) -> Result<crate::kv_plan::PlanColumns, String> {
        if sids.len() != lens.len() {
            return Err("plan columns of different lengths".into());
        }
        crate::kv_plan::plan_columns_with(
            |i| self.pages(sids[i]).map(|pages| pages.as_slice()).map_err(|e| format!("row {i}: {e}")),
            lens,
            spans,
            page_size,
        )
    }
}

/// FNV-1a over the previous hash, the acquire's number and its pages.
fn trail_hash(hash: u64, count: u64, pages: &[i32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |v: u64| {
        for byte in v.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(hash);
    feed(count);
    for &page in pages {
        feed(page as u64);
    }
    h
}

mod py {
    use std::sync::{Mutex, MutexGuard};

    use pyo3::exceptions::{PyAssertionError, PyIndexError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::pyclass::CompareOp;
    use pyo3::types::{PyByteArray, PyList, PySlice, PyTuple};

    use super::{ArenaCore, ArenaError, Released};

    impl From<ArenaError> for PyErr {
        fn from(e: ArenaError) -> PyErr {
            match e {
                ArenaError::NoOwner(_) => PyAssertionError::new_err(e.to_string()),
                _ => PyIndexError::new_err(e.to_string()),
            }
        }
    }

    /// The page pool's bookkeeping (see the module docs). Every caller holds
    /// the KV manager's lock; the mutex here only makes the class shareable.
    #[pyclass(name = "PageArena", module = "mstar_rust", frozen)]
    pub struct PyPageArena {
        core: Mutex<ArenaCore>,
        /// the prefix index to tell when a page gains or loses its second
        /// owner (`PrefixIndex.set_shared`), or None
        index: Mutex<Option<Py<PyAny>>>,
        /// queues each page a release leaves to the index alone, for a copy
        /// to the host (`HostWrites.queue`), or None
        writes: Mutex<Option<Py<PyAny>>>,
        /// streams dropped while the arena was locked, freed by `new_stream`
        dropped: Mutex<Vec<usize>>,
    }

    impl PyPageArena {
        fn core(&self) -> MutexGuard<'_, ArenaCore> {
            self.core.lock().unwrap_or_else(|e| e.into_inner())
        }

        fn index(&self, py: Python<'_>) -> Option<Py<PyAny>> {
            self.index.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|ix| ix.clone_ref(py))
        }

        fn writes(&self, py: Python<'_>) -> Option<Py<PyAny>> {
            self.writes.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|w| w.clone_ref(py))
        }
    }

    /// The pages ``pages`` names: a stream's list (any arena's), or a
    /// sequence of page numbers.
    fn pages_of(pages: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
        if let Ok(stream) = pages.downcast::<PyStreamPages>() {
            let stream = stream.get();
            let core = stream.arena.get().core();
            return Ok(core.pages(stream.sid)?.iter().map(|&p| p as i64).collect());
        }
        pages.extract()
    }

    fn pages_arg(pages: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
        if let Ok(stream) = pages.downcast::<PyStreamPages>() {
            let stream = stream.get();
            return Ok(stream.arena.get().core().pages(stream.sid)?.clone());
        }
        pages.extract()
    }

    #[pymethods]
    impl PyPageArena {
        #[new]
        fn new(max_num_pages: usize) -> Self {
            Self {
                core: Mutex::new(ArenaCore::new(max_num_pages)),
                index: Mutex::new(None),
                writes: Mutex::new(None),
                dropped: Mutex::new(Vec::new()),
            }
        }

        #[getter]
        fn max_num_pages(&self) -> usize {
            self.core().max_num_pages()
        }

        #[getter]
        fn num_free(&self) -> usize {
            self.core().num_free()
        }

        fn free_pages(&self) -> Vec<i32> {
            self.core().free_pages()
        }

        #[getter]
        fn get_index(&self, py: Python<'_>) -> Option<Py<PyAny>> {
            self.index(py)
        }

        #[setter]
        fn set_index(&self, index: Option<Py<PyAny>>) {
            *self.index.lock().unwrap_or_else(|e| e.into_inner()) = index;
        }

        #[getter]
        fn get_writes(&self, py: Python<'_>) -> Option<Py<PyAny>> {
            self.writes(py)
        }

        #[setter]
        fn set_writes(&self, writes: Option<Py<PyAny>>) {
            *self.writes.lock().unwrap_or_else(|e| e.into_inner()) = writes;
        }

        #[getter]
        fn get_trail(&self) -> Option<(u64, u64)> {
            self.core().trail
        }

        #[setter]
        fn set_trail(&self, trail: Option<(u64, u64)>) {
            self.core().trail = trail;
        }

        fn acquire(&self, n: usize) -> Option<Vec<i32>> {
            self.core().acquire(n)
        }

        /// One more owner for each page. The index hears of each page that
        /// becomes shared as it happens, with the arena unlocked, so it reads
        /// the counts of the pages before it in ``pages`` already updated.
        fn retain(&self, py: Python<'_>, pages: &Bound<'_, PyAny>) -> PyResult<()> {
            let pages = pages_of(pages)?;
            match self.index(py) {
                None => {
                    let mut core = self.core();
                    for page in pages {
                        core.retain(page)?;
                    }
                }
                Some(index) => {
                    for page in pages {
                        let shared = self.core().retain(page)?;
                        if shared {
                            index.call_method1(py, "set_shared", (page, true))?;
                        }
                    }
                }
            }
            Ok(())
        }

        /// One owner fewer for each page; those left with none return to the
        /// free list together, after the index has heard of the rest.
        fn release(&self, py: Python<'_>, pages: &Bound<'_, PyAny>) -> PyResult<()> {
            let pages = pages_of(pages)?;
            let index = self.index(py);
            let writes = if index.is_some() { self.writes(py) } else { None };
            let mut freed = Vec::new();
            for page in pages {
                let released = self.core().release(page)?;
                match released {
                    Released::Freed => freed.push(page as i32),
                    Released::Unshared => {
                        if let Some(index) = &index {
                            index.call_method1(py, "set_shared", (page, false))?;
                            if let Some(writes) = &writes {
                                let key = index.call_method1(py, "key_of", (page,))?;
                                writes.call_method1(py, "queue", (key,))?;
                            }
                        }
                    }
                    Released::Held => {}
                }
            }
            self.core().free(&freed);
            Ok(())
        }

        fn seal(&self, pages: &Bound<'_, PyAny>) -> PyResult<()> {
            let pages = pages_of(pages)?;
            let mut core = self.core();
            for page in pages {
                core.seal(page)?;
            }
            Ok(())
        }

        fn any_sealed(&self, pages: &Bound<'_, PyAny>) -> PyResult<bool> {
            let pages = pages_of(pages)?;
            let core = self.core();
            for page in pages {
                if core.is_sealed(page)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }

        fn is_sealed(&self, page: i64) -> PyResult<bool> {
            Ok(self.core().is_sealed(page)?)
        }

        fn owners(&self, page: i64) -> PyResult<u32> {
            Ok(self.core().owners(page)?)
        }

        /// Each page's owner count, in order.
        fn owners_of(&self, pages: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
            let pages = pages_of(pages)?;
            let core = self.core();
            pages.into_iter().map(|page| core.owners(page).map_err(PyErr::from)).collect()
        }

        fn new_stream(slf: &Bound<'_, Self>) -> PyStreamPages {
            let arena = slf.get();
            let dropped = std::mem::take(&mut *arena.dropped.lock().unwrap_or_else(|e| e.into_inner()));
            let mut core = arena.core();
            for sid in dropped {
                core.drop_stream(sid);
            }
            let sid = core.new_stream();
            drop(core);
            PyStreamPages { arena: slf.clone().unbind(), sid }
        }

        /// ``kv_plan_columns`` over the page lists of the streams ``sids``,
        /// one per row: ``(bytearray, total, all_have_pages)``.
        fn plan_columns<'py>(
            &self,
            py: Python<'py>,
            sids: Vec<usize>,
            lens: Vec<i64>,
            spans: Vec<i64>,
            page_size: i64,
        ) -> PyResult<(Bound<'py, PyByteArray>, usize, bool)> {
            let cols = self.core().plan_columns(&sids, &lens, &spans, page_size).map_err(PyValueError::new_err)?;
            // SAFETY: plain i32 data viewed as its native-endian bytes
            let bytes = unsafe { std::slice::from_raw_parts(cols.out.as_ptr() as *const u8, cols.out.len() * 4) };
            Ok((PyByteArray::new(py, bytes), cols.total, cols.all_have_pages))
        }
    }

    /// One stream's page list, read and written like the list it replaces;
    /// slices and iteration hand out Python lists. Its id returns to the
    /// arena when the object goes away.
    #[pyclass(name = "StreamPages", module = "mstar_rust", frozen, sequence)]
    pub struct PyStreamPages {
        arena: Py<PyPageArena>,
        sid: usize,
    }

    impl Drop for PyStreamPages {
        fn drop(&mut self) {
            // never waits on the lock: the thread holding it may be this one
            let arena = self.arena.get();
            match arena.core.try_lock() {
                Ok(mut core) => core.drop_stream(self.sid),
                Err(_) => arena.dropped.lock().unwrap_or_else(|e| e.into_inner()).push(self.sid),
            }
        }
    }

    impl PyStreamPages {
        fn core(&self) -> MutexGuard<'_, ArenaCore> {
            self.arena.get().core()
        }

        fn list(&self) -> PyResult<Vec<i32>> {
            Ok(self.core().pages(self.sid)?.clone())
        }
    }

    #[pymethods]
    impl PyStreamPages {
        #[getter]
        fn sid(&self) -> usize {
            self.sid
        }

        #[getter]
        fn n(&self) -> PyResult<usize> {
            Ok(self.core().pages(self.sid)?.len())
        }

        fn __len__(&self) -> PyResult<usize> {
            self.n()
        }

        fn tolist(&self) -> PyResult<Vec<i32>> {
            self.list()
        }

        fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            Ok(PyList::new(py, self.list()?)?.into_any().try_iter()?.into_any())
        }

        fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let mut pages = self.list()?;
            pages.reverse();
            Ok(PyList::new(py, pages)?.into_any().try_iter()?.into_any())
        }

        fn __getitem__<'py>(&self, py: Python<'py>, i: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
            // the arena stays locked only while nothing allocates a Python
            // object (a collection could drop a stream, which takes the lock)
            if let Ok(slice) = i.downcast::<PySlice>() {
                let ix = slice.indices(self.n()? as isize)?;
                let out: Vec<i32> = {
                    let core = self.core();
                    let pages = core.pages(self.sid)?;
                    if ix.step == 1 {
                        pages[ix.start.clamp(0, pages.len() as isize) as usize..ix.stop.max(ix.start).clamp(0, pages.len() as isize) as usize].to_vec()
                    } else {
                        (0..ix.slicelength as isize).map(|k| pages[(ix.start + k * ix.step) as usize]).collect()
                    }
                };
                return Ok(PyList::new(py, out)?.into_any().unbind());
            }
            let k: isize = i.extract()?;
            let page = {
                let core = self.core();
                let pages = core.pages(self.sid)?;
                let k = if k < 0 { k + pages.len() as isize } else { k };
                if k < 0 || k as usize >= pages.len() {
                    return Err(PyIndexError::new_err("page index out of range"));
                }
                pages[k as usize]
            };
            Ok(page.into_pyobject(py)?.into_any().unbind())
        }

        /// Equal to a stream, list or tuple holding the same pages.
        fn __richcmp__(&self, py: Python<'_>, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
            let comparable = other.downcast::<PyStreamPages>().is_ok()
                || other.downcast::<PyList>().is_ok()
                || other.downcast::<PyTuple>().is_ok();
            if !comparable || !matches!(op, CompareOp::Eq | CompareOp::Ne) {
                return Ok(py.NotImplemented());
            }
            // a sequence of something other than page numbers holds no pages
            let same = pages_arg(other).is_ok_and(|pages| self.list().is_ok_and(|mine| mine == pages));
            Ok((same == matches!(op, CompareOp::Eq)).into_pyobject(py)?.to_owned().into_any().unbind())
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(format!("StreamPages({:?})", self.list()?))
        }

        fn extend(&self, pages: &Bound<'_, PyAny>) -> PyResult<()> {
            let pages = pages_arg(pages)?;
            self.core().pages_mut(self.sid)?.extend(pages);
            Ok(())
        }

        /// Make ``pages`` the whole list.
        fn assign(&self, pages: &Bound<'_, PyAny>) -> PyResult<()> {
            let pages = pages_arg(pages)?;
            let mut core = self.core();
            let list = core.pages_mut(self.sid)?;
            list.clear();
            list.extend(pages);
            Ok(())
        }

        fn clear(&self) -> PyResult<()> {
            self.core().pages_mut(self.sid)?.clear();
            Ok(())
        }

        fn __delitem__(&self, i: &Bound<'_, PyAny>) -> PyResult<()> {
            let mut core = self.core();
            let pages = core.pages_mut(self.sid)?;
            let n = pages.len() as isize;
            let (start, stop) = if let Ok(slice) = i.downcast::<PySlice>() {
                let ix = slice.indices(n)?;
                if ix.step != 1 {
                    return Err(PyValueError::new_err("StreamPages deletes contiguous ranges only"));
                }
                (ix.start, ix.stop)
            } else {
                let mut k: isize = i.extract()?;
                if k < 0 {
                    k += n;
                }
                (k, k + 1)
            };
            if start >= 0 && stop > start && stop <= n {
                pages.drain(start as usize..stop as usize);
            }
            Ok(())
        }

        /// Empty the list and give back its memory.
        fn free(&self) -> PyResult<()> {
            *self.core().pages_mut(self.sid)? = Vec::new();
            Ok(())
        }
    }
}

pub use py::{PyPageArena, PyStreamPages};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_come_out_fifo_from_zero_and_return_to_the_back() {
        let mut a = ArenaCore::new(4);
        assert_eq!(a.acquire(2), Some(vec![0, 1]));
        assert_eq!(a.acquire(3), None);
        assert_eq!(a.release(0), Ok(Released::Freed));
        a.free(&[0]);
        assert_eq!(a.free_pages(), vec![2, 3, 0]);
        assert_eq!(a.acquire(3), Some(vec![2, 3, 0]));
        assert_eq!(a.num_free(), 0);
    }

    #[test]
    fn a_page_is_freed_by_its_last_owner_and_loses_its_seal() {
        let mut a = ArenaCore::new(2);
        let page = a.acquire(1).unwrap()[0] as i64;
        assert_eq!(a.retain(page), Ok(true));
        a.seal(page).unwrap();
        assert_eq!(a.retain(page), Ok(false));
        assert_eq!(a.release(page), Ok(Released::Held));
        assert_eq!(a.release(page), Ok(Released::Unshared));
        assert_eq!(a.is_sealed(page), Ok(true));
        assert_eq!(a.release(page), Ok(Released::Freed));
        assert_eq!(a.is_sealed(page), Ok(false));
        assert_eq!(a.release(page), Err(ArenaError::NoOwner(page as i32)));
        assert_eq!(a.retain(5), Err(ArenaError::OutOfRange(5)));
    }

    #[test]
    fn the_trail_moves_with_every_acquire_and_only_when_kept() {
        let mut a = ArenaCore::new(4);
        a.acquire(1);
        assert_eq!(a.trail, None);
        a.trail = Some((0, 0));
        a.acquire(1);
        a.acquire(0);
        let (count, hash) = a.trail.unwrap();
        assert_eq!(count, 2);
        let mut b = ArenaCore::new(4);
        b.acquire(1);
        b.trail = Some((0, 0));
        b.acquire(1);
        b.acquire(0);
        assert_eq!(b.trail, Some((count, hash)));
    }

    #[test]
    fn stream_ids_are_reused_once_dropped() {
        let mut a = ArenaCore::new(4);
        let (s0, s1) = (a.new_stream(), a.new_stream());
        a.pages_mut(s0).unwrap().extend([3, 1]);
        a.drop_stream(s0);
        assert_eq!(a.new_stream(), s0);
        assert!(a.pages(s0).unwrap().is_empty());
        assert_eq!(a.pages(9), Err(ArenaError::NoStream(9)));
        assert_ne!(s0, s1);
    }

    #[test]
    fn a_plan_reads_each_rows_stream() {
        let mut a = ArenaCore::new(32);
        let (s0, s1, sink) = (a.new_stream(), a.new_stream(), a.new_stream());
        a.pages_mut(s0).unwrap().extend([10, 11, 12, 13]);
        a.pages_mut(s1).unwrap().extend([20]);
        a.pages_mut(sink).unwrap().extend([0, 0]);
        // the rows of `kv_plan`'s pool test: 9 tokens -> 3 of 4 pages; wants 2, holds 1; a sink row
        let c = a.plan_columns(&[s0, s1, sink], &[9, 5, 1], &[1, 1, 1], 4).unwrap();
        let n = 3;
        assert_eq!(&c.out[..n + 1], &[0, 1, 2, 3]);
        assert_eq!(&c.out[n + 1..2 * (n + 1)], &[0, 3, 4, 5]);
        assert_eq!(&c.out[2 * (n + 1) + 4 * n..], &[10, 11, 12, 20, 0]);
        assert!(a.plan_columns(&[7], &[1], &[1], 4).is_err());
    }
}
