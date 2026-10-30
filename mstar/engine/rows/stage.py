"""A step's upload, collected as the resources register it and packed in one
call: the segment tables of its copy phases and the payload they read from
(``rust/src/rows.rs``). ``PyRowStage`` is the Python version, segment for
segment and byte for byte, for builds without the extension.

A segment is ``(src, dst, nbytes)`` with raw addresses. Each phase runs after
the one before, so a phase may read what an earlier one wrote or write where
an earlier one read; host writes land in the last phase.
"""
from __future__ import annotations

import numpy as np

try:
    from mstar_rust import RowStage as _RustRowStage
    from mstar_rust import rows_step_mirrors as _rust_step_mirrors
except ImportError:
    _RustRowStage = None
    _rust_step_mirrors = None

_ALIGN = 16
_SEGMENT = 24  # bytes: src, dst, nbytes as int64


def _round(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


def _contiguous(a: np.ndarray, what: str) -> np.ndarray:
    if not a.flags.c_contiguous:
        raise ValueError(f"{what} must be C-contiguous")  # as the Rust stage requires
    return a


class PyRowStage:
    """See ``rust/src/rows.rs``'s ``RowStage``."""

    __slots__ = ("_flat", "_from_payload", "_payload", "_size")

    def __init__(self, phases: int):
        if phases < 1:
            raise ValueError("a stage has at least one phase")
        self._flat: list[list[int]] = [[] for _ in range(phases)]
        self._from_payload: list[int] = []  # last-phase segments whose source is a payload offset
        self._payload: list[tuple[int, np.ndarray]] = []
        self._size = 0

    def copy_rows(
        self, phase: int, src_base: int, src_stride: int, src_idx: np.ndarray,
        dst_base: int, dst_stride: int, dst_idx: np.ndarray, nbytes: int,
    ) -> None:
        """``dst_base + dst_idx[i] * dst_stride <- src_base + src_idx[i] * src_stride``, in ``phase``."""
        if not 0 <= phase < len(self._flat):
            raise ValueError(f"phase {phase} of a stage of {len(self._flat)}")
        _contiguous(src_idx, "src_idx")
        _contiguous(dst_idx, "dst_idx")
        if len(src_idx) != len(dst_idx):
            raise ValueError(f"{len(src_idx)} source rows for {len(dst_idx)} destinations")
        flat = self._flat[phase]
        for s, d in zip(src_idx.tolist(), dst_idx.tolist(), strict=True):
            flat += (src_base + s * src_stride, dst_base + d * dst_stride, nbytes)

    def write(
        self, dst_base: int, row_bytes: int, slots: np.ndarray, values: np.ndarray, elem_bytes: int,
        offsets: np.ndarray | None = None,
    ) -> None:
        """Host bytes into rows, one piece of ``elem_bytes`` a slot: whole rows,
        or with ``offsets``, that many bytes into each row."""
        _contiguous(slots, "slots")
        _contiguous(values, "values")
        if values.nbytes != elem_bytes * len(slots):
            raise ValueError(f"{values.nbytes} bytes for {len(slots)} pieces of {elem_bytes}")
        if offsets is None and elem_bytes != row_bytes:
            raise ValueError("a whole-row write's pieces must be rows")
        if offsets is not None and len(_contiguous(offsets, "offsets")) != len(slots):
            raise ValueError(f"{len(offsets)} offsets for {len(slots)} slots")
        at = self._put_bytes(values)
        offs = [0] * len(slots) if offsets is None else offsets.tolist()
        for k, (slot, off) in enumerate(zip(slots.tolist(), offs, strict=True)):
            self._add_from_payload(at + k * elem_bytes, dst_base + slot * row_bytes + off, elem_bytes)

    def put(self, data: np.ndarray, dst: int) -> None:
        """``data`` uploaded whole to ``dst``."""
        self._add_from_payload(self._put_bytes(data), dst, data.nbytes)

    def header(self, idx: np.ndarray, max_rows: int, pad: int, trash: int, dst: int) -> None:
        """The step header: its row count, then each dense row's slot to
        gather from (``pad`` past the step's rows) and to scatter to
        (``trash`` past them), as int32."""
        n = len(idx)
        if n > max_rows:
            raise ValueError(f"{n} rows in a header of {max_rows}")
        header = np.empty(1 + 2 * max_rows, dtype=np.int32)
        header[0] = n
        header[1:1 + max_rows] = pad
        header[1 + max_rows:] = trash
        header[1:1 + n] = idx
        header[1 + max_rows:1 + max_rows + n] = idx
        self.put(header, dst)

    def _put_bytes(self, data: np.ndarray) -> int:
        at = self._size
        # copied now, as the Rust stage does: a caller may reuse its buffer
        self._payload.append((at, data.copy()))
        self._size = _round(at + data.nbytes)
        return at

    def _add_from_payload(self, at: int, dst: int, nbytes: int) -> None:
        flat = self._flat[-1]
        self._from_payload.append(len(flat) // 3)
        flat += (at, dst, nbytes)

    def is_empty(self) -> bool:
        return not any(self._flat)

    def _tables_bytes(self) -> int:
        return _round(sum(len(f) for f in self._flat) // 3 * _SEGMENT)

    def size(self) -> int:
        return self._tables_bytes() + self._size

    def pack(self, out: np.ndarray, device_base: int) -> list[tuple[int, int, int]]:
        """Tables, then payload, into ``out`` (``uint8``, the host side of a
        block whose device side starts at ``device_base``); each non-empty
        phase's ``(byte offset, segments, longest segment)``."""
        payload_at = self._tables_bytes()
        if len(out) < payload_at + self._size:
            raise ValueError(f"{len(out)} bytes for a block of {self.size()}")
        last = len(self._flat) - 1
        placed, at = [], 0
        for phase, flat in enumerate(self._flat):
            if not flat:
                continue
            rows = np.array(flat, dtype=np.int64).reshape(-1, 3)
            if phase == last and self._from_payload:
                rows[self._from_payload, 0] += device_base + payload_at
            out[at:at + rows.nbytes] = rows.view(np.uint8).reshape(-1)
            placed.append((at, len(rows), int(rows[:, 2].max())))
            at += rows.nbytes
        for off, data in self._payload:
            start = payload_at + off
            out[start:start + data.nbytes] = data.reshape(-1).view(np.uint8)
        return placed


def new_stage(phases: int):
    """A step's stage: the Rust one when the extension has it."""
    return PyRowStage(phases) if _RustRowStage is None else _RustRowStage(phases)


_INT = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}


def py_step_mirrors(slab: np.ndarray, desc: np.ndarray, n_slots: int, slots: np.ndarray) -> None:
    """``rows_step_mirrors``: add each stepped mirror's step to the rows of
    ``slots`` in the host slab. ``desc`` is ``[buffers, 4]`` int64: byte
    offset, element size, elements a row, step."""
    for offset, size, elems, step in desc.tolist():
        rows = slab[offset:offset + n_slots * elems * size].view(_INT[size]).reshape(n_slots, elems)
        rows[slots] += step


step_mirrors = py_step_mirrors if _rust_step_mirrors is None else _rust_step_mirrors
