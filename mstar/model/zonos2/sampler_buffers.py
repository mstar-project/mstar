"""Graph-safe per-request sampler state for the Zonos2 multi-codebook sampler.

The buffers here are fixed-shape and slot-indexed, so
:func:`~mstar.model.zonos2.tts_sampling.sample_frame` can run inside a captured
``forward_batched`` graph.

The storage has three tiers, like
:class:`mstar.utils.sampling.SamplerBuffers` for a single codebook. This module
extends that storage to the multi-codebook case with a windowed repetition
penalty:

* ``master`` — the slot-indexed canonical state ``[capacity, ...]``, with one
  row for each live request. It grows by doubling.
* ``buf`` — the per-step tensors ``[max_bs, ...]`` at a stable address. The
  graph reads and writes them. Each step gathers the slots of the active
  requests into them.
* ``_slot_idx`` — pinned staging for the single H2D copy of the slot indices.

Each request has two pieces of state:

* The repetition ring ``ring[cap, C, W]`` (int32). It holds the codes of the
  last ``W`` frames for each codebook. A wrapping ``cursor`` writes it in
  place, and a ``-1`` sentinel marks a position that holds no code yet. A real
  code is always ``>= 0``. The repetition penalty tests only whether a token is
  present (``counts > 0`` in :func:`apply_repetition_penalty`), so the ring
  needs no separate fill count and gives the same penalty as a plain window.
* The offset ``offset[cap]`` (int64). This is the frame count of the request,
  which is also the RNG ``step`` index. The code reads it before the write and
  increments it in place afterwards. It therefore does not depend on the batch
  position, and the stateless RNG of the sampler stays reproducible.

All per-step mutation is in place (``scatter_``, ``add_``, ``remainder_``), so
the buffer addresses stay stable across graph replays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Zonos2SamplerBuffers:
    max_batch_size: int
    n_codebooks: int
    window: int
    # The repetition penalty applies to codebooks ``0`` to
    # ``repetition_codebooks - 1``. A negative value applies it to all
    # codebooks. On read, the code masks the codebooks at or after the cutoff
    # to ``-1``, and the penalty then ignores them.
    repetition_codebooks: int

    # The repetition ring (int32). The sentinel -1 marks an empty position.
    ring_master: torch.Tensor    # [capacity, C, W]
    ring_buf: torch.Tensor       # [max_bs, C, W]
    cursor_master: torch.Tensor  # [capacity] int32, next write column mod W
    cursor_buf: torch.Tensor     # [max_bs] int32
    # The frame count and RNG step of each request (int64).
    offset_master: torch.Tensor  # [capacity]
    offset_buf: torch.Tensor     # [max_bs]

    # Static staging for the penalty input, with the exclusion mask. ``pen_buf``
    # is a masked copy of ``ring_buf``.
    pen_buf: torch.Tensor        # [max_bs, C, W] int32
    _rc_exclude: torch.Tensor | None  # [1, C, 1] bool, True where excluded

    # Slot-index staging for the gather of each step.
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
        """Assign a slot to ``rid`` and reset its master state.

        This method runs outside the graph.
        """
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
        """Release the slot of ``rid``.

        The method does no GPU write. The next request to use the slot resets
        the state.
        """
        slot = self._rid_to_slot.pop(rid, None)
        if slot is not None:
            self._free_slots.append(slot)

    def _grow_master(self, new_capacity: int) -> None:
        """Double and copy the master buffers.

        Call this method when the live requests exceed the capacity.
        """
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

        This method serves the eager path, where the batch size changes from
        step to step. The contents of ``buf`` are transient, because the code
        gathers them again every step. The method therefore only reallocates
        them larger, and it does not touch ``master``, the canonical per-slot
        state. Do not call this method inside a capture epoch, where the buffer
        addresses must stay stable.
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

    # -- gather for each step (outside the graph) -----------------------
    def gather_for_request_ids(self, request_ids: list[str], padded_bs: int) -> None:
        """Fill the per-step buffers for ``request_ids`` from their slots.

        The padding rows (``i >= len(request_ids)``) use slot 0. The dummy-rid
        remap of the runner discards their sampled output, so their contents
        only need to be well formed.
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
        """Return the RNG step index of each request, before the write.

        The step index is the frame count of the request.
        """
        return self.offset_buf[:padded_bs]

    def repetition_ids(self, padded_bs: int) -> torch.Tensor:
        """Return the recent ids ``[padded_bs, C, W]`` for the penalty.

        See :func:`apply_repetition_penalty`. The method sets the excluded
        codebooks to ``-1``, and the penalty then ignores them. It writes into a
        static buffer of fixed shape, in place, so it is safe for capture.
        """
        pb = padded_bs
        self.pen_buf[:pb].copy_(self.ring_buf[:pb])
        if self._rc_exclude is not None:
            self.pen_buf[:pb].masked_fill_(self._rc_exclude, -1)
        return self.pen_buf[:pb]

    # -- write (graph-safe) ---------------------------------------------
    def write_frame(self, codes: torch.Tensor, padded_bs: int) -> None:
        """Write the sampled codes into the ring and advance the state.

        ``codes`` is the sampled frame ``[padded_bs, >=C]``. The method stores
        only the first ``C`` audio-codebook columns. Every operation is in
        place, so the buffer addresses stay stable inside a captured graph.
        """
        pb = padded_bs
        C = self.n_codebooks
        col = self.cursor_buf[:pb].to(torch.int64).view(pb, 1, 1).expand(pb, C, 1)
        src = codes[:, :C].to(self.ring_buf.dtype).view(pb, C, 1)
        self.ring_buf[:pb].scatter_(2, col, src)
        self.cursor_buf[:pb].add_(1)
        self.cursor_buf[:pb].remainder_(self.window)
        self.offset_buf[:pb].add_(1)

    # -- sync back to master (outside the graph, after the replay) ------
    def sync_after_step(self, request_ids: list[str]) -> None:
        """Copy the per-step rows of the real requests back to their slots."""
        n = len(request_ids)
        if n == 0:
            return
        idx = self._slot_idx_gpu[:n]  # the matching gather set the first n slots
        self.ring_master.index_copy_(0, idx, self.ring_buf[:n])
        self.cursor_master.index_copy_(0, idx, self.cursor_buf[:n])
        self.offset_master.index_copy_(0, idx, self.offset_buf[:n])
