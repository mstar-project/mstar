"""Graph-safe per-request sampler state for Zonos2's multi-codebook sampler.

Phase 2 of the CUDA-graph work. Replaces the dict-of-growing-tensors state in
:class:`Zonos2LLMSubmodule` (``_history`` grown by ``torch.cat`` every step,
read back as a variable-length window) with fixed-shape, slot-indexed static
buffers — the prerequisite for running ``_sample`` inside a captured
``forward_batched``.

Mirrors the three-tier storage of :class:`mstar.utils.sampling.SamplerBuffers`
(single-codebook), extended to the multi-codebook, windowed-repetition-penalty
case:

* ``master`` — ``[capacity, ...]`` slot-indexed canonical state, one row per
  live request; grown by doubling.
* ``buf`` — ``[max_bs, ...]`` per-step tensor with a stable address (read/written
  inside the graph); populated each step by gathering the active requests' slots.
* pinned ``_slot_idx`` staging for the single H2D slot-index copy.

Two pieces of per-request state:

* **repetition ring** ``ring[cap, C, W]`` (int32) — the last ``W`` frames' codes
  per codebook, written in place with a wrapping ``cursor``. A ``-1`` sentinel
  marks not-yet-written positions (a real code is always ``>= 0``), so the ring
  is a drop-in for ``_rep_ids_batched``'s ``[B, C, W]`` output without a separate
  fill count. The repetition penalty only tests token *presence*
  (``counts > 0`` in :func:`apply_repetition_penalty`), so the ring's set of
  windowed ids reproduces the dict window's penalty bit-for-bit.
* **offset** ``offset[cap]`` (int64) — per-request frame count = the RNG ``step``
  index (replaces ``_step_for``'s ``hist.shape[0]``). Read *before* the write,
  incremented in place after, so it stays independent of batch position and keeps
  the sampler's stateless RNG reproducible.

All per-step mutation is in place (``scatter_``/``add_``/``remainder_``) so the
buffer addresses stay stable across graph replays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Zonos2SamplerBuffers:
    max_batch_size: int
    n_codebooks: int
    window: int
    # Repetition penalty applies to codebooks ``0..repetition_codebooks-1``; a
    # negative value applies it to all codebooks. Codebooks at/after the cutoff
    # are masked to ``-1`` on read (ignored by the penalty), matching ``_rep_ids``.
    repetition_codebooks: int

    # Repetition ring (int32, sentinel -1 = empty).
    ring_master: torch.Tensor    # [capacity, C, W]
    ring_buf: torch.Tensor       # [max_bs, C, W]
    cursor_master: torch.Tensor  # [capacity] int32, next write column mod W
    cursor_buf: torch.Tensor     # [max_bs] int32
    # Per-request frame count / RNG step (int64).
    offset_master: torch.Tensor  # [capacity]
    offset_buf: torch.Tensor     # [max_bs]

    # Static penalty-input staging (rc-masked copy of ring_buf) + rc mask.
    pen_buf: torch.Tensor        # [max_bs, C, W] int32
    _rc_exclude: torch.Tensor | None  # [1, C, 1] bool, True where codebook is excluded

    # Slot-index staging for the per-step gather.
    _slot_idx_cpu: torch.Tensor
    _slot_idx_gpu: torch.Tensor
    _pinned: bool

    # Slot bookkeeping (CPU-only).
    _master_capacity: int
    _rid_to_slot: dict[str, int] = field(default_factory=dict, repr=False)
    _free_slots: list[int] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    @classmethod
    def allocate(
        cls,
        max_batch_size: int,
        n_codebooks: int,
        window: int,
        repetition_codebooks: int,
        device: torch.device | str,
        capacity: int | None = None,
    ) -> "Zonos2SamplerBuffers":
        device = torch.device(device)
        window = max(int(window), 1)
        cap = capacity if capacity is not None else max_batch_size
        pinned = torch.cuda.is_available() and device.type == "cuda"

        def ring(n):
            return torch.full((n, n_codebooks, window), -1, dtype=torch.int32, device=device)

        rc = repetition_codebooks
        rc_exclude = None
        if 0 <= rc < n_codebooks:
            excl = torch.arange(n_codebooks, device=device) >= rc  # [C] bool
            rc_exclude = excl.view(1, n_codebooks, 1)

        return cls(
            max_batch_size=max_batch_size,
            n_codebooks=n_codebooks,
            window=window,
            repetition_codebooks=rc,
            ring_master=ring(cap),
            ring_buf=ring(max_batch_size),
            cursor_master=torch.zeros(cap, dtype=torch.int32, device=device),
            cursor_buf=torch.zeros(max_batch_size, dtype=torch.int32, device=device),
            offset_master=torch.zeros(cap, dtype=torch.int64, device=device),
            offset_buf=torch.zeros(max_batch_size, dtype=torch.int64, device=device),
            pen_buf=ring(max_batch_size),
            _rc_exclude=rc_exclude,
            _slot_idx_cpu=torch.zeros(max_batch_size, dtype=torch.int64, pin_memory=pinned),
            _slot_idx_gpu=torch.zeros(max_batch_size, dtype=torch.int64, device=device),
            _pinned=pinned,
            _master_capacity=cap,
            _free_slots=list(range(cap)),
        )

    # -- slot lifecycle -------------------------------------------------
    def register_request(self, rid: str) -> None:
        """Assign a slot to ``rid`` and reset its master state (outside graph)."""
        if rid in self._rid_to_slot:
            return
        if not self._free_slots:
            self._grow_master(self._master_capacity * 2)
        slot = self._free_slots.pop()
        self._rid_to_slot[rid] = slot
        self.ring_master[slot].fill_(-1)
        self.cursor_master[slot] = 0
        self.offset_master[slot] = 0

    def unregister_request(self, rid: str) -> None:
        """Release ``rid``'s slot (no GPU writes; state is reset on reuse)."""
        slot = self._rid_to_slot.pop(rid, None)
        if slot is not None:
            self._free_slots.append(slot)

    def _grow_master(self, new_capacity: int) -> None:
        """Double-and-copy the master buffers when live requests exceed capacity."""
        old = self._master_capacity
        C, W = self.n_codebooks, self.window
        dev = self.ring_master.device

        new_ring = torch.full((new_capacity, C, W), -1, dtype=torch.int32, device=dev)
        new_ring[:old].copy_(self.ring_master)
        self.ring_master = new_ring

        new_cursor = torch.zeros(new_capacity, dtype=torch.int32, device=dev)
        new_cursor[:old].copy_(self.cursor_master)
        self.cursor_master = new_cursor

        new_offset = torch.zeros(new_capacity, dtype=torch.int64, device=dev)
        new_offset[:old].copy_(self.offset_master)
        self.offset_master = new_offset

        self._free_slots.extend(range(old, new_capacity))
        self._master_capacity = new_capacity

    def ensure_batch_capacity(self, padded_bs: int) -> None:
        """Grow the per-step (``buf``) tensors to hold ``padded_bs`` rows.

        For the eager path, where batch size varies step-to-step before any
        capture. ``buf`` contents are transient (re-gathered every step), so
        this just reallocates them larger; ``master`` (canonical per-slot state)
        is untouched. MUST NOT be called inside a capture epoch — buffer
        addresses must stay stable there; Phase 3 pre-sizes to the capture max.
        """
        if padded_bs <= self.max_batch_size:
            return
        dev = self.ring_buf.device
        C, W = self.n_codebooks, self.window
        self.ring_buf = torch.full((padded_bs, C, W), -1, dtype=torch.int32, device=dev)
        self.pen_buf = torch.full((padded_bs, C, W), -1, dtype=torch.int32, device=dev)
        self.cursor_buf = torch.zeros(padded_bs, dtype=torch.int32, device=dev)
        self.offset_buf = torch.zeros(padded_bs, dtype=torch.int64, device=dev)
        self._slot_idx_cpu = torch.zeros(padded_bs, dtype=torch.int64, pin_memory=self._pinned)
        self._slot_idx_gpu = torch.zeros(padded_bs, dtype=torch.int64, device=dev)
        self.max_batch_size = padded_bs

    # -- per-step gather (outside graph) --------------------------------
    def gather_for_request_ids(self, request_ids: list[str], padded_bs: int) -> None:
        """Populate the per-step buffers for ``request_ids`` from their slots.

        Padding rows (``i >= len(request_ids)``) reuse slot 0 — their sampled
        outputs are discarded by the runner's dummy-rid remap, so their contents
        only need to be well-formed.
        """
        assert padded_bs <= self.max_batch_size, (
            f"padded_bs={padded_bs} exceeds max_batch_size={self.max_batch_size}"
        )
        n = len(request_ids)
        for i, rid in enumerate(request_ids):
            self._slot_idx_cpu[i] = self._rid_to_slot.get(rid, 0)
        for i in range(n, padded_bs):
            self._slot_idx_cpu[i] = 0
        idx = self._slot_idx_gpu[:padded_bs]
        idx.copy_(self._slot_idx_cpu[:padded_bs], non_blocking=self._pinned)

        torch.index_select(self.ring_master, 0, idx, out=self.ring_buf[:padded_bs])
        torch.index_select(self.cursor_master, 0, idx, out=self.cursor_buf[:padded_bs])
        torch.index_select(self.offset_master, 0, idx, out=self.offset_buf[:padded_bs])

    # -- reads (graph-safe) ---------------------------------------------
    def steps(self, padded_bs: int) -> torch.Tensor:
        """Per-request RNG step index (frame count) for this step — pre-write."""
        return self.offset_buf[:padded_bs]

    def repetition_ids(self, padded_bs: int) -> torch.Tensor:
        """``[padded_bs, C, W]`` recent ids for :func:`apply_repetition_penalty`.

        rc-excluded codebooks are set to ``-1`` (ignored). Recomputed into a
        static buffer each step (fixed shape, in-place) so it is capture-safe.
        """
        pb = padded_bs
        self.pen_buf[:pb].copy_(self.ring_buf[:pb])
        if self._rc_exclude is not None:
            self.pen_buf[:pb].masked_fill_(self._rc_exclude, -1)
        return self.pen_buf[:pb]

    # -- write (graph-safe) ---------------------------------------------
    def write_frame(self, codes: torch.Tensor, padded_bs: int) -> None:
        """Write this step's sampled codes into the ring and advance state.

        ``codes``: ``[padded_bs, >=C]`` (the sampled frame; only the first ``C``
        audio-codebook columns are stored). All ops are in place so buffer
        addresses stay stable inside a captured graph.
        """
        pb = padded_bs
        C = self.n_codebooks
        col = self.cursor_buf[:pb].to(torch.int64).view(pb, 1, 1).expand(pb, C, 1)
        src = codes[:, :C].to(self.ring_buf.dtype).view(pb, C, 1)
        self.ring_buf[:pb].scatter_(2, col, src)
        self.cursor_buf[:pb].add_(1)
        self.cursor_buf[:pb].remainder_(self.window)
        self.offset_buf[:pb].add_(1)

    # -- sync back to master (outside graph, post-replay) ---------------
    def sync_after_step(self, request_ids: list[str]) -> None:
        """Copy the real requests' per-step rows back to their master slots."""
        n = len(request_ids)
        if n == 0:
            return
        idx = self._slot_idx_gpu[:n]  # first n slots set by the matching gather
        self.ring_master.index_copy_(0, idx, self.ring_buf[:n])
        self.cursor_master.index_copy_(0, idx, self.cursor_buf[:n])
        self.offset_master.index_copy_(0, idx, self.offset_buf[:n])
