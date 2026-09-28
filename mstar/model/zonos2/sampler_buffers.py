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
  requests into them. They are never reallocated: the captured graphs and the
  deferred sync both hold their addresses.
* ``_slot_idx`` — pinned staging for the single H2D copy of the slot indices.

Each request has two pieces of state:

* The repetition ring ``ring[cap, C, W]`` (int32). It holds the codes of the
  last ``W`` frames for each codebook. A wrapping ``cursor`` writes it in
  place, and a ``-1`` sentinel marks a position that holds no code yet. A real
  code is always ``>= 0``. The repetition penalty tests only whether a token is
  present (``counts > 0`` in :func:`apply_repetition_penalty`), so the ring
  needs no separate fill count and gives the same penalty as a plain window.
* The per-request constants in ``const``: the conductor's ``random_seed`` and
  the request's sampling knobs (temperature, top-k, top-p, min-p, penalty,
  repetition window and codebook cutoff). They are set at register time and
  never change, so the sync does not write them back. The ring is as wide as
  the largest window a request may ask for; a smaller window masks the older
  columns by age on read.
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

from mstar.model.zonos2.tts_sampling import SamplingRows, TTSSamplingParams

# Per-request constants: name -> dtype. The first five feed SamplingRows.
_CONSTS = {
    "temperature": torch.float32,
    "topk": torch.int64,
    "top_p": torch.float32,
    "min_p": torch.float32,
    "repetition_penalty": torch.float32,
    "seed": torch.int64,
    "window": torch.int64,       # repetition window, <= the ring width
    "codebooks": torch.int64,    # the penalty covers codebooks [0, codebooks)
}


@dataclass
class Zonos2SamplerBuffers:
    max_batch_size: int
    n_codebooks: int
    # The ring width: the largest repetition window a request may use.
    window: int
    # Used by a request registered without its own params.
    defaults: TTSSamplingParams

    # The repetition ring (int32). The sentinel -1 marks an empty position.
    ring_master: torch.Tensor    # [capacity, C, W]
    ring_buf: torch.Tensor       # [max_bs, C, W]
    cursor_master: torch.Tensor  # [capacity] int32, next write column mod W
    cursor_buf: torch.Tensor     # [max_bs] int32
    # The frame count and RNG step of each request (int64).
    offset_master: torch.Tensor  # [capacity]
    offset_buf: torch.Tensor     # [max_bs]
    # The per-request constants, keyed as ``_CONSTS``.
    const_master: dict[str, torch.Tensor]  # each [capacity]
    const_buf: dict[str, torch.Tensor]     # each [max_bs]

    # Static staging for the penalty input: a masked copy of ``ring_buf``.
    pen_buf: torch.Tensor        # [max_bs, C, W] int32
    _cols: torch.Tensor          # [1, 1, W] int64, ring column index
    _codebook_idx: torch.Tensor  # [1, C, 1] int64

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
        defaults: TTSSamplingParams | None = None,
    ) -> "Zonos2SamplerBuffers":
        """``window`` is the ring width; ``defaults`` falls back to the shipped
        knobs with this window and codebook cutoff."""
        device = torch.device(device)
        window = max(int(window), 1)
        cap = capacity if capacity is not None else max_batch_size
        pinned = torch.cuda.is_available() and device.type == "cuda"
        if defaults is None:
            defaults = TTSSamplingParams(
                repetition_window=window, repetition_codebooks=repetition_codebooks,
            )

        def ring(n):
            return torch.full((n, n_codebooks, window), -1, dtype=torch.int32, device=device)

        def consts(n):
            return {k: torch.zeros(n, dtype=dt, device=device) for k, dt in _CONSTS.items()}

        return cls(
            max_batch_size=max_batch_size,
            n_codebooks=n_codebooks,
            window=window,
            defaults=defaults,
            ring_master=ring(cap),
            ring_buf=ring(max_batch_size),
            cursor_master=torch.zeros(cap, dtype=torch.int32, device=device),
            cursor_buf=torch.zeros(max_batch_size, dtype=torch.int32, device=device),
            offset_master=torch.zeros(cap, dtype=torch.int64, device=device),
            offset_buf=torch.zeros(max_batch_size, dtype=torch.int64, device=device),
            const_master=consts(cap),
            const_buf=consts(max_batch_size),
            pen_buf=ring(max_batch_size),
            _cols=torch.arange(window, device=device).view(1, 1, window),
            _codebook_idx=torch.arange(n_codebooks, device=device).view(1, n_codebooks, 1),
            _slot_idx_cpu=torch.zeros(max_batch_size, dtype=torch.int64, pin_memory=pinned),
            _slot_idx_gpu=torch.zeros(max_batch_size, dtype=torch.int64, device=device),
            _pinned=pinned,
            _master_capacity=cap,
            _free_slots=list(range(cap)),
        )

    # -- slot lifecycle -------------------------------------------------
    def register_request(
        self, rid: str, seed: int = 0, params: TTSSamplingParams | None = None,
    ) -> None:
        """Assign a slot to ``rid``, reset its state, and store its constants.

        ``params`` are the request's own knobs, or ``defaults`` when None. This
        method runs outside the graph.
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
        p = params if params is not None else self.defaults
        if p.repetition_window > self.window:
            raise ValueError(
                f"repetition_window {p.repetition_window} exceeds the ring width {self.window}"
            )
        C = self.n_codebooks
        values = {
            "temperature": p.temperature, "topk": p.topk, "top_p": p.top_p,
            "min_p": p.min_p, "repetition_penalty": p.repetition_penalty,
            "seed": seed, "window": p.repetition_window,
            "codebooks": C if p.repetition_codebooks < 0 else min(p.repetition_codebooks, C),
        }
        for k, v in values.items():
            self.const_master[k][slot] = v

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

        for k, t in self.const_master.items():
            grown = torch.zeros(new_capacity, dtype=t.dtype, device=dev)
            grown[:old].copy_(t)
            self.const_master[k] = grown

        self._free_slots.extend(range(old, new_capacity))
        self._master_capacity = new_capacity

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
        for k, t in self.const_master.items():
            torch.index_select(t, 0, idx, out=self.const_buf[k][:padded_bs])

    # -- reads (graph-safe) ---------------------------------------------
    def steps(self, padded_bs: int) -> torch.Tensor:
        """Return the RNG step index of each request, before the write.

        The step index is the frame count of the request.
        """
        return self.offset_buf[:padded_bs]

    def seeds(self, padded_bs: int) -> torch.Tensor:
        """Return the RNG seed of each request."""
        return self.const_buf["seed"][:padded_bs]

    def rows(self, padded_bs: int) -> SamplingRows:
        """Return each request's sampling knobs, for :func:`sample_frame`."""
        return SamplingRows(**{
            k: self.const_buf[k][:padded_bs]
            for k in ("temperature", "topk", "top_p", "min_p", "repetition_penalty")
        })

    def repetition_ids(self, padded_bs: int) -> torch.Tensor:
        """Return the recent ids ``[padded_bs, C, W]`` for the penalty.

        See :func:`apply_repetition_penalty`. Per request, the method sets to
        ``-1`` the columns older than its window and the codebooks past its
        cutoff, and the penalty then ignores them. It writes into a static
        buffer of fixed shape, in place, so it is safe for capture.
        """
        pb = padded_bs
        cursor = self.cursor_buf[:pb].to(torch.int64).view(pb, 1, 1)
        age = torch.remainder(cursor - 1 - self._cols, self.window)  # 0 is newest
        window = self.const_buf["window"][:pb].view(pb, 1, 1)
        codebooks = self.const_buf["codebooks"][:pb].view(pb, 1, 1)
        drop = (age >= window) | (self._codebook_idx >= codebooks)  # [pb, C, W]
        self.pen_buf[:pb].copy_(self.ring_buf[:pb])
        self.pen_buf[:pb].masked_fill_(drop, -1)
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
