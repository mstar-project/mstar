//! A persistent-rows step's upload, packed in one call (`mstar/engine/rows/`
//! on the Python side).
//!
//! A step's rows change by many small byte copies: leaving rows read out to
//! their stores, rows moved between slots, joining rows copied in, host
//! values written into rows, and the mirrored slab and the step header
//! uploaded whole. `RowStage` collects them as the resources register them
//! and packs the segment tables and the payload into the step's pinned
//! block, which one H2D and one copy kernel per phase then apply.
//! `rows_step_mirrors` adds each stepped mirror's step to the rows a forward
//! ran. `stage.py`'s `PyRowStage` is the Python version, segment for segment
//! and byte for byte.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// What the payload and tables are aligned to (`stage._ALIGN`).
const ALIGN: usize = 16;
/// Bytes of one segment: `src, dst, nbytes` as int64.
const SEGMENT: usize = 24;

fn round(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

/// Phases of segments, each run after the one before, so a phase may read
/// what an earlier one wrote or write where an earlier one read. Host
/// writes land in the last phase.
pub struct RowStage {
    flat: Vec<Vec<i64>>,
    /// last-phase segments (by index) whose source is an offset into the payload
    from_payload: Vec<usize>,
    payload: Vec<u8>,
}

/// One phase's table in the packed block: its byte offset, its segments and
/// its longest segment, which sizes the copy kernel's grid.
pub type Placed = (usize, usize, i64);

impl RowStage {
    pub fn new(phases: usize) -> Result<Self, String> {
        if phases == 0 {
            return Err("a stage has at least one phase".into());
        }
        Ok(RowStage { flat: vec![Vec::new(); phases], from_payload: Vec::new(), payload: Vec::new() })
    }

    /// `dst_base + dst_idx[i] * dst_stride <- src_base + src_idx[i] * src_stride`,
    /// `nbytes` each, in `phase`.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows(
        &mut self, phase: usize, src_base: i64, src_stride: i64, src_idx: &[i64],
        dst_base: i64, dst_stride: i64, dst_idx: &[i64], nbytes: i64,
    ) -> Result<(), String> {
        let phases = self.flat.len();
        let Some(flat) = self.flat.get_mut(phase) else {
            return Err(format!("phase {phase} of a stage of {phases}"));
        };
        if src_idx.len() != dst_idx.len() {
            return Err(format!("{} sources for {} destinations", src_idx.len(), dst_idx.len()));
        }
        flat.reserve(3 * src_idx.len());
        for (s, d) in src_idx.iter().zip(dst_idx) {
            flat.extend_from_slice(&[src_base + s * src_stride, dst_base + d * dst_stride, nbytes]);
        }
        Ok(())
    }

    /// Host bytes into rows (last phase): `values` holds one piece of
    /// `elem_bytes` per slot. Without `offsets` a piece is a whole row of
    /// `row_bytes`; with them, it lands that many bytes into its row.
    pub fn write(
        &mut self, dst_base: i64, row_bytes: i64, slots: &[i64], values: &[u8], elem_bytes: i64,
        offsets: Option<&[i64]>,
    ) -> Result<(), String> {
        if values.len() as i64 != elem_bytes * slots.len() as i64 {
            return Err(format!("{} bytes for {} pieces of {elem_bytes}", values.len(), slots.len()));
        }
        if offsets.is_none() && elem_bytes != row_bytes {
            return Err("a whole-row write's pieces must be rows".into());
        }
        if let Some(offsets) = offsets {
            if offsets.len() != slots.len() {
                return Err(format!("{} offsets for {} slots", offsets.len(), slots.len()));
            }
        }
        let at = self.put_bytes(values) as i64;
        for (k, &slot) in slots.iter().enumerate() {
            let off = offsets.map_or(0, |o| o[k]);
            self.add_from_payload(at + k as i64 * elem_bytes, dst_base + slot * row_bytes + off, elem_bytes);
        }
        Ok(())
    }

    /// `data` uploaded whole to `dst` (last phase).
    pub fn put(&mut self, data: &[u8], dst: i64) {
        let at = self.put_bytes(data) as i64;
        self.add_from_payload(at, dst, data.len() as i64);
    }

    /// The step header (last phase): its row count, then each of `max_rows`
    /// dense rows' slot to gather from (`pad` past the step's rows) and to
    /// scatter to (`trash` past them), as int32 (`RowSpace._header_for`).
    pub fn header(&mut self, idx: &[i64], max_rows: usize, pad: i32, trash: i32, dst: i64) -> Result<(), String> {
        let n = idx.len();
        if n > max_rows {
            return Err(format!("{n} rows in a header of {max_rows}"));
        }
        let mut words = Vec::with_capacity(1 + 2 * max_rows);
        words.push(n as i32);
        for fill in [pad, trash] {
            words.extend(idx.iter().map(|&s| s as i32));
            words.extend(std::iter::repeat(fill).take(max_rows - n));
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        self.put(&bytes, dst);
        Ok(())
    }

    fn put_bytes(&mut self, data: &[u8]) -> usize {
        let at = self.payload.len();
        self.payload.extend_from_slice(data);
        self.payload.resize(round(self.payload.len()), 0);
        at
    }

    fn add_from_payload(&mut self, at: i64, dst: i64, nbytes: i64) {
        let flat = self.flat.last_mut().expect("a stage has a phase");
        self.from_payload.push(flat.len() / 3);
        flat.extend_from_slice(&[at, dst, nbytes]);
    }

    pub fn is_empty(&self) -> bool {
        self.flat.iter().all(Vec::is_empty)
    }

    fn tables_bytes(&self) -> usize {
        round(self.flat.iter().map(Vec::len).sum::<usize>() / 3 * SEGMENT)
    }

    /// The packed block's size: the tables, then the payload.
    pub fn size(&self) -> usize {
        self.tables_bytes() + self.payload.len()
    }

    /// The tables and payload into `out` (the host side of a block whose
    /// device side starts at `device_base`), payload sources resolved to
    /// device addresses. Each non-empty phase's placement, in order.
    pub fn pack(&self, out: &mut [u8], device_base: i64) -> Result<Vec<Placed>, String> {
        let payload_at = self.tables_bytes();
        if out.len() < payload_at + self.payload.len() {
            return Err(format!("{} bytes for a block of {}", out.len(), self.size()));
        }
        let base = device_base + payload_at as i64;
        let last = self.flat.len() - 1;
        let mut placed = Vec::with_capacity(self.flat.len());
        let mut at = 0;
        for (phase, flat) in self.flat.iter().enumerate() {
            let count = flat.len() / 3;
            if count == 0 {
                continue;
            }
            let mut table = flat.clone();
            if phase == last {
                for &seg in &self.from_payload {
                    table[3 * seg] += base;
                }
            }
            let longest = table.chunks_exact(3).map(|s| s[2]).max().unwrap_or(0);
            for (k, word) in table.iter().enumerate() {
                out[at + 8 * k..at + 8 * k + 8].copy_from_slice(&word.to_ne_bytes());
            }
            placed.push((at, count, longest));
            at += count * SEGMENT;
        }
        out[payload_at..payload_at + self.payload.len()].copy_from_slice(&self.payload);
        Ok(placed)
    }
}

/// Add each stepped mirror's step to the rows of `slots`, in the host slab.
/// `desc` is four int64 per buffer: its byte offset in the slab, its element
/// size (1, 2, 4 or 8), elements a row, and its step; every buffer has
/// `n_slots` rows. Integer addition wraps, as numpy's does.
pub fn step_mirrors(slab: &mut [u8], desc: &[i64], n_slots: i64, slots: &[i64]) -> Result<(), String> {
    if desc.len() % 4 != 0 {
        return Err("a stepped mirror is described by four numbers".into());
    }
    if let Some(&bad) = slots.iter().find(|&&s| s < 0 || s >= n_slots) {
        return Err(format!("slot {bad} is outside the {n_slots} slots"));
    }
    for d in desc.chunks_exact(4) {
        let (offset, size, elems, step) = (d[0], d[1], d[2], d[3]);
        if offset < 0 || elems < 0 || offset + n_slots * elems * size > slab.len() as i64 {
            return Err(format!("a stepped mirror at {offset} runs past the slab"));
        }
        for &slot in slots {
            let row = (offset + slot * elems * size) as usize;
            for e in 0..elems as usize {
                let at = row + e * size as usize;
                macro_rules! add {
                    ($t:ty) => {{
                        const N: usize = std::mem::size_of::<$t>();
                        let cell: &mut [u8; N] = (&mut slab[at..at + N]).try_into().unwrap();
                        *cell = <$t>::from_ne_bytes(*cell).wrapping_add(step as $t).to_ne_bytes();
                    }};
                }
                match size {
                    1 => add!(i8),
                    2 => add!(i16),
                    4 => add!(i32),
                    8 => add!(i64),
                    _ => return Err(format!("element size {size}: only integer rows step")),
                }
            }
        }
    }
    Ok(())
}

// SAFETY (for the views below): C-contiguous buffers of the declared item
// type, held by the caller for the call under the GIL.
fn view<'a, T: pyo3::buffer::Element>(buf: &'a PyBuffer<T>, what: &str) -> PyResult<&'a [T]> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(format!("{what} must be C-contiguous")));
    }
    Ok(unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const T, buf.item_count()) })
}

fn view_mut<'a, T: pyo3::buffer::Element>(buf: &'a PyBuffer<T>, what: &str) -> PyResult<&'a mut [T]> {
    if !buf.is_c_contiguous() || buf.readonly() {
        return Err(PyValueError::new_err(format!("{what} must be writable and C-contiguous")));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(buf.buf_ptr() as *mut T, buf.item_count()) })
}

/// `RowStage` for Python: one per staged step.
#[pyclass(name = "RowStage")]
pub struct PyRowStage {
    inner: RowStage,
}

#[pymethods]
impl PyRowStage {
    #[new]
    fn new(phases: usize) -> PyResult<Self> {
        Ok(Self { inner: RowStage::new(phases).map_err(PyValueError::new_err)? })
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_rows(
        &mut self, phase: usize, src_base: i64, src_stride: i64, src_idx: PyBuffer<i64>,
        dst_base: i64, dst_stride: i64, dst_idx: PyBuffer<i64>, nbytes: i64,
    ) -> PyResult<()> {
        let (s, d) = (view(&src_idx, "src_idx")?, view(&dst_idx, "dst_idx")?);
        self.inner
            .copy_rows(phase, src_base, src_stride, s, dst_base, dst_stride, d, nbytes)
            .map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (dst_base, row_bytes, slots, values, elem_bytes, offsets=None))]
    fn write(
        &mut self, dst_base: i64, row_bytes: i64, slots: PyBuffer<i64>, values: PyBuffer<u8>,
        elem_bytes: i64, offsets: Option<PyBuffer<i64>>,
    ) -> PyResult<()> {
        let offsets = match &offsets {
            Some(o) => Some(view(o, "offsets")?),
            None => None,
        };
        self.inner
            .write(dst_base, row_bytes, view(&slots, "slots")?, view(&values, "values")?, elem_bytes, offsets)
            .map_err(PyValueError::new_err)
    }

    fn put(&mut self, data: PyBuffer<u8>, dst: i64) -> PyResult<()> {
        self.inner.put(view(&data, "data")?, dst);
        Ok(())
    }

    fn header(&mut self, idx: PyBuffer<i64>, max_rows: usize, pad: i32, trash: i32, dst: i64) -> PyResult<()> {
        self.inner
            .header(view(&idx, "idx")?, max_rows, pad, trash, dst)
            .map_err(PyValueError::new_err)
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn pack(&self, out: PyBuffer<u8>, device_base: i64) -> PyResult<Vec<Placed>> {
        self.inner.pack(view_mut(&out, "out")?, device_base).map_err(PyValueError::new_err)
    }
}

/// `step_mirrors` for Python, on the slab in place.
#[pyfunction]
pub fn rows_step_mirrors(slab: PyBuffer<u8>, desc: PyBuffer<i64>, n_slots: i64, slots: PyBuffer<i64>) -> PyResult<()> {
    step_mirrors(view_mut(&slab, "slab")?, view(&desc, "desc")?, n_slots, view(&slots, "slots")?)
        .map_err(PyValueError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(out: &[u8]) -> Vec<i64> {
        out.chunks_exact(8).map(|c| i64::from_ne_bytes(c.try_into().unwrap())).collect()
    }

    #[test]
    fn copies_run_in_their_phase_and_payload_sources_resolve_to_the_block() {
        let mut st = RowStage::new(3).unwrap();
        // a leaving row (slot 3 of a 16-byte buffer at 1000) out to store row 7 at 5000
        st.copy_rows(0, 1000, 16, &[3], 5000, 16, &[7], 16).unwrap();
        // two 8-byte rows written into slots 1 and 2 of a buffer at 2000
        let values: Vec<u8> = (0..16).collect();
        st.write(2000, 8, &[1, 2], &values, 8, None).unwrap();
        let mut out = vec![0u8; st.size()];
        let placed = st.pack(&mut out, 100_000).unwrap();
        // phase 1 is empty and takes no table
        assert_eq!(placed, vec![(0, 1, 16), (24, 2, 8)]);
        let payload_at = round(3 * SEGMENT); // 80
        let w = words(&out[..72]);
        assert_eq!(w[..3], [1000 + 48, 5000 + 112, 16]);
        let base = 100_000 + payload_at as i64;
        assert_eq!(w[3..], [base, 2008, 8, base + 8, 2016, 8]);
        assert_eq!(&out[payload_at..payload_at + 16], &values[..]);
    }

    #[test]
    fn a_middle_phase_sits_between_the_others() {
        let mut st = RowStage::new(3).unwrap();
        st.put(&[7; 4], 9000);
        st.copy_rows(1, 0, 8, &[5], 0, 8, &[2], 8).unwrap();
        st.copy_rows(0, 100, 8, &[1], 200, 8, &[1], 8).unwrap();
        let mut out = vec![0u8; st.size()];
        let placed = st.pack(&mut out, 0).unwrap();
        assert_eq!(placed, vec![(0, 1, 8), (24, 1, 8), (48, 1, 4)]);
        assert_eq!(words(&out[24..48]), vec![40, 16, 8]);
        assert!(st.copy_rows(3, 0, 8, &[], 0, 8, &[], 8).is_err());
        assert!(RowStage::new(0).is_err());
    }

    #[test]
    fn element_writes_land_at_their_offsets() {
        let mut st = RowStage::new(1).unwrap();
        st.write(0, 32, &[2, 5], &[1, 0, 0, 0, 2, 0, 0, 0], 4, Some(&[8, 12])).unwrap();
        let mut out = vec![0u8; st.size()];
        st.pack(&mut out, 0).unwrap();
        let w = words(&out[..48]);
        assert_eq!((w[1], w[2], w[4], w[5]), (2 * 32 + 8, 4, 5 * 32 + 12, 4));
    }

    #[test]
    fn the_header_pads_with_the_pad_and_trash_slots() {
        let mut st = RowStage::new(2).unwrap();
        st.header(&[4, 1], 3, 9, 10, 0).unwrap();
        let ints: Vec<i32> = st.payload[..28].chunks_exact(4).map(|c| i32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(ints, vec![2, 4, 1, 9, 4, 1, 10]);
        assert!(st.header(&[1, 2, 3, 4], 3, 9, 10, 0).is_err());
    }

    #[test]
    fn a_short_block_is_refused() {
        let mut st = RowStage::new(1).unwrap();
        st.put(&[1; 40], 0);
        let mut out = vec![0u8; st.size() - 1];
        assert!(st.pack(&mut out, 0).is_err());
    }

    #[test]
    fn stepping_adds_to_the_rows_of_each_stepped_mirror_only() {
        // an int32 mirror of 4 slots x 2 elements at 0, an int64 one of 4 slots at 32
        let mut slab = vec![0u8; 64];
        step_mirrors(&mut slab, &[0, 4, 2, 1, 32, 8, 1, 3], 4, &[2, 0]).unwrap();
        let a: Vec<i32> = slab[..32].chunks_exact(4).map(|c| i32::from_ne_bytes(c.try_into().unwrap())).collect();
        assert_eq!(a, vec![1, 1, 0, 0, 1, 1, 0, 0]);
        assert_eq!(words(&slab[32..]), vec![3, 0, 3, 0]);
        assert!(step_mirrors(&mut slab, &[0, 4, 2, 1], 4, &[4]).is_err());
        assert!(step_mirrors(&mut slab, &[0, 4, 5, 1], 4, &[0]).is_err());
    }
}
