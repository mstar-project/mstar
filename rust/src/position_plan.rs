//! A step's positions read and moved on the position resource's stream counters
//! (`RopeManager._counters`, request -> label -> next position), in one call per
//! step (`RopeManager._fill_pos_ids`, `RopeManager.commit`). A stream with no
//! counter is at 0. Python's own loop is the fallback for a build without these.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// A row's positions: `start`, `start + 1`, ... or, under a block scheme,
/// `start` throughout. Its one position when `out` has one slot.
pub fn write_row(start: i64, block: bool, out: &mut [i64]) {
    if block {
        out.fill(start);
        return;
    }
    let mut position = start;
    for slot in out {
        *slot = position;
        position += 1;
    }
}

/// A column of a `Segments`: one value for every row, or one per row.
enum Column<'a, 'py> {
    One(&'a Bound<'py, PyAny>),
    PerRow(&'a Bound<'py, PyTuple>),
}

impl<'a, 'py> Column<'a, 'py> {
    fn of(values: &'a Bound<'py, PyAny>) -> Self {
        match values.downcast::<PyTuple>() {
            Ok(column) => Column::PerRow(column),
            Err(_) => Column::One(values),
        }
    }

    fn get(&self, i: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Column::One(value) => (*value).clone(),
            Column::PerRow(column) => column.get_item(i)?,
        })
    }
}

/// An int-or-tuple column of ints, one per row.
fn int_column(values: &Bound<'_, PyAny>, rows: usize) -> PyResult<Vec<i64>> {
    match values.downcast::<PyTuple>() {
        Ok(column) => {
            if column.len() != rows {
                return Err(PyValueError::new_err(format!("{} values for {rows} rows", column.len())));
            }
            column.iter().map(|value| value.extract()).collect()
        }
        Err(_) => Ok(vec![values.extract::<i64>()?; rows]),
    }
}

/// A `Segments`' positions from the counters, in row order, into ``out``
/// (int64): how many there are. With no room for them all nothing is written.
#[pyfunction]
pub fn position_ids_into(
    counters: &Bound<'_, PyDict>,
    segments: &Bound<'_, PyAny>,
    block: bool,
    out: PyBuffer<i64>,
) -> PyResult<usize> {
    if !out.is_c_contiguous() || out.readonly() {
        return Err(PyValueError::new_err("out must be a writable C-contiguous int64 buffer"));
    }
    let py = segments.py();
    let rids = segments.getattr(intern!(py, "request_ids"))?;
    let rids = rids.downcast::<PyTuple>()?;
    let labels = segments.getattr(intern!(py, "labels"))?;
    let labels = Column::of(&labels);
    let spans = int_column(&segments.getattr(intern!(py, "spans"))?, rids.len())?;
    if spans.iter().any(|&span| span < 0) {
        return Err(PyValueError::new_err("a negative span"));
    }
    let total: i64 = spans.iter().sum();
    let room = out.item_count();
    if total as usize > room {
        return Ok(total as usize);
    }
    // SAFETY: a C-contiguous, writable buffer of int64 the caller holds for this
    // call, under the GIL
    let out = unsafe { std::slice::from_raw_parts_mut(out.buf_ptr() as *mut i64, room) };
    let mut at = 0usize;
    for (i, &span) in spans.iter().enumerate() {
        let span = span as usize;
        if span == 0 {
            continue;
        }
        let start = match counters.get_item(rids.get_borrowed_item(i)?)? {
            None => 0,
            Some(streams) => match streams.downcast::<PyDict>()?.get_item(labels.get(i)?)? {
                None => 0,
                Some(position) => position.extract::<i64>()?,
            },
        };
        write_row(start, block, &mut out[at..at + span]);
        at += span;
    }
    Ok(at)
}

/// Move each row's counter by its advance (``advances``: one value for every
/// row, or one per row). A row with no counters yet gets them.
#[pyfunction]
pub fn advance_positions(
    counters: &Bound<'_, PyDict>,
    segments: &Bound<'_, PyAny>,
    advances: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = segments.py();
    let rids = segments.getattr(intern!(py, "request_ids"))?;
    let rids = rids.downcast::<PyTuple>()?;
    let labels = segments.getattr(intern!(py, "labels"))?;
    let labels = Column::of(&labels);
    let advances = int_column(advances, rids.len())?;
    for (i, &advance) in advances.iter().enumerate() {
        if advance == 0 {
            continue;
        }
        let rid = rids.get_borrowed_item(i)?;
        let streams = match counters.get_item(&rid)? {
            Some(streams) => streams.downcast_into::<PyDict>()?,
            None => {
                let streams = PyDict::new(py);
                counters.set_item(&rid, &streams)?;
                streams
            }
        };
        let label = labels.get(i)?;
        let position = match streams.get_item(&label)? {
            Some(position) => position.extract::<i64>()?,
            None => 0,
        };
        streams.set_item(label, position + advance)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_counts_up_from_its_start() {
        let mut out = [0i64; 3];
        write_row(7, false, &mut out);
        assert_eq!(out, [7, 8, 9]);
    }

    #[test]
    fn a_block_row_holds_its_start() {
        let mut out = [0i64; 3];
        write_row(7, true, &mut out);
        assert_eq!(out, [7, 7, 7]);
    }

    #[test]
    fn a_decode_row_is_one_position() {
        let mut out = [0i64; 1];
        write_row(4, false, &mut out);
        assert_eq!(out, [4]);
    }
}
