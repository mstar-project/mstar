//! A KV plan label's index arrays, built from per-row columns in one call
//! (`KVManager._plan_label` / `PageTable.plan_columns` on the Python side).
//!
//! Python collects, per row of the step, the row's extent in the page pool,
//! the pages it holds, its length after the step and its span; this computes
//! FlashInfer's four index arrays, the host lengths and each decode row's
//! write location, and copies every row's pages out of the pool. The numpy
//! fallback in `page_table.py` is the same arithmetic, op for op; a step of a
//! few dozen rows is where the ~30 numpy calls it takes cost more than the
//! per-row Python they replace, and one call here costs less than either.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;

/// The arrays of one plan label, back to back in one `Vec<i32>`:
/// `qo_indptr (n + 1) | kv_indptr (n + 1) | last_page_len (n) | kv_lens (n) |
/// decode_pages (n) | decode_offsets (n) | indices (total)`.
pub struct PlanColumns {
    pub out: Vec<i32>,
    pub total: usize,
    /// every row has a page to write into (a decode step's write locations
    /// are only meaningful then)
    pub all_have_pages: bool,
}

pub fn plan_columns(
    pool: &[i32],
    offs: &[i64],
    held: &[i64],
    lens: &[i64],
    spans: &[i64],
    page_size: i64,
) -> Result<PlanColumns, String> {
    let n = offs.len();
    if held.len() != n || lens.len() != n || spans.len() != n {
        return Err("plan columns of different lengths".into());
    }
    if page_size <= 0 {
        return Err("page_size must be positive".into());
    }
    // a stream can hold more pages than its length needs (a high-water mark),
    // and never more is read than it holds
    let mut npages = Vec::with_capacity(n);
    let mut total: i64 = 0;
    for i in 0..n {
        let p = (lens[i] + page_size - 1).div_euclid(page_size).min(held[i]);
        if p < 0 {
            return Err(format!("row {i}: negative page count"));
        }
        let off = offs[i];
        if off < 0 || (off + p) as usize > pool.len() {
            return Err(format!("row {i}: extent {off}+{p} is outside the page pool"));
        }
        npages.push(p);
        total += p;
    }
    let total = total as usize;
    let mut out = vec![0i32; 2 * (n + 1) + 4 * n + total];
    let (qo, rest) = out.split_at_mut(n + 1);
    let (kv_indptr, rest) = rest.split_at_mut(n + 1);
    let (last, rest) = rest.split_at_mut(n);
    let (kv_lens, rest) = rest.split_at_mut(n);
    let (decode_pages, rest) = rest.split_at_mut(n);
    let (decode_offsets, indices) = rest.split_at_mut(n);
    let (mut q, mut k) = (0i64, 0usize);
    let mut all_have_pages = true;
    for i in 0..n {
        let p = npages[i] as usize;
        let off = offs[i] as usize;
        indices[k..k + p].copy_from_slice(&pool[off..off + p]);
        // `SequenceView.last_page_len(page_size) or page_size`
        let l = (lens[i] - 1).rem_euclid(page_size) + 1;
        last[i] = l as i32;
        // FlashInfer's `get_seq_lens`
        kv_lens[i] = ((npages[i].max(1) - 1) * page_size + l) as i32;
        if p > 0 {
            decode_pages[i] = indices[k + p - 1];
        } else {
            all_have_pages = false;
        }
        decode_offsets[i] = (l - 1) as i32;
        k += p;
        q += spans[i];
        qo[i + 1] = q as i32;
        kv_indptr[i + 1] = k as i32;
    }
    Ok(PlanColumns { out, total, all_have_pages })
}

/// ``plan_columns`` for Python: ``(bytearray, total, all_have_pages)``, the
/// bytearray holding the arrays as native int32 in the order above.
#[pyfunction]
pub fn kv_plan_columns<'py>(
    py: Python<'py>,
    pool: PyBuffer<i32>,
    offs: Vec<i64>,
    held: Vec<i64>,
    lens: Vec<i64>,
    spans: Vec<i64>,
    page_size: i64,
) -> PyResult<(Bound<'py, PyByteArray>, usize, bool)> {
    if !pool.is_c_contiguous() {
        return Err(PyValueError::new_err("the page pool must be C-contiguous"));
    }
    // SAFETY: a C-contiguous int32 buffer, held by the caller for this call
    // under the GIL (nothing can resize it while it is borrowed here)
    let pool = unsafe { std::slice::from_raw_parts(pool.buf_ptr() as *const i32, pool.item_count()) };
    let cols = plan_columns(pool, &offs, &held, &lens, &spans, page_size).map_err(PyValueError::new_err)?;
    // SAFETY: plain i32 data viewed as its native-endian bytes
    let bytes = unsafe { std::slice::from_raw_parts(cols.out.as_ptr() as *const u8, cols.out.len() * 4) };
    Ok((PyByteArray::new(py, bytes), cols.total, cols.all_have_pages))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(c: &PlanColumns, n: usize) -> Vec<Vec<i32>> {
        let mut at = 0;
        let mut parts = Vec::new();
        for len in [n + 1, n + 1, n, n, n, n, c.total] {
            parts.push(c.out[at..at + len].to_vec());
            at += len;
        }
        parts
    }

    #[test]
    fn rows_read_their_extent_prefix_and_clip_to_held_pages() {
        // pages of two streams and a sink run in one pool
        let pool = [10, 11, 12, 13, 20, 21, -1, -1];
        // row 0: 9 tokens at page size 4 -> 3 pages; row 1: wants 2, holds 1;
        // row 2: a sink row of span 1
        let c = plan_columns(&pool, &[0, 4, 6], &[4, 1, 2], &[9, 5, 1], &[1, 1, 1], 4).unwrap();
        let p = split(&c, 3);
        assert_eq!(p[0], vec![0, 1, 2, 3]); // qo
        assert_eq!(p[1], vec![0, 3, 4, 5]); // kv_indptr
        assert_eq!(p[2], vec![1, 1, 1]); // last page len
        assert_eq!(p[3], vec![9, 1, 1]); // kv_lens: clipped row counts its held page
        assert_eq!(p[4], vec![12, 20, -1]); // decode write pages
        assert_eq!(p[5], vec![0, 0, 0]);
        assert_eq!(p[6], vec![10, 11, 12, 20, -1]);
        assert!(c.all_have_pages);
    }

    #[test]
    fn a_full_last_page_and_an_empty_row() {
        let pool = [7, 8];
        let c = plan_columns(&pool, &[0, 0], &[2, 2], &[8, 0], &[3, 0], 4).unwrap();
        let p = split(&c, 2);
        assert_eq!(p[0], vec![0, 3, 3]);
        assert_eq!(p[1], vec![0, 2, 2]);
        assert_eq!(p[2], vec![4, 4]); // a full page, and `0 % 4 or 4`
        assert_eq!(p[3], vec![8, 4]);
        assert!(!c.all_have_pages);
    }

    #[test]
    fn an_extent_past_the_pool_is_refused() {
        assert!(plan_columns(&[1, 2], &[1], &[2], &[8], &[1], 4).is_err());
    }
}
