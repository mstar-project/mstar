"""CUTLASS (CuTe DSL) Hopper GEMM with CTA-pipelining hooks.

``D = act(A @ B^T + bias)`` on sm_90a: TMA loads, warp-specialised
(1 DMA warp group + 2 MMA warp groups) persistent mainloop with WGMMA, TMA
store epilogue. This is the persistent warp-specialised kernel family the
paper (arXiv:2607.07862) built on, ported from
``examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py``
in CUTLASS 4.7, with three changes:

* **Row-block-major tile order** (all N tiles of row block 0, then row block
  1, ...) so a row block of the producer's output completes as early as
  possible and the consumer can start on it.
* **Producer role**: after a tile's TMA store has fully landed
  (``cp.async.bulk.wait_group 0``), ``fence.proxy.async`` +
  ``fence.acq_rel.sys`` + one ``red.relaxed.sys.global.add`` (together a
  release) on the consumer-resident counter of that row block. The output tensor and the
  counters live in the *other* GPU's memory (peer mapped over NVLink).
* **Consumer role**: the TMA-issuing warp spins on the row block's counter
  with ``ld.acquire.sys`` (one lane, ``nanosleep`` between polls) until
  ``n_ready`` producer tiles have signalled, then ``fence.proxy.async`` and
  proceeds to issue that tile's loads.

``ROLE_PLAIN`` has neither hook and is the single-GPU baseline for the same
kernel. ``ROLE_REDUCE`` is the fc2 shard of a TP2 layer with the reduction fused
in (``cute_tp2.py``): the peer-owned row blocks first, TMA-stored as bf16 partials
straight into the peer's output with the producer release; then the own row blocks, each
TMA reduce-added (``cp.reduce.async.bulk.tensor`` ``.add``, done in L2) with the bias onto
the peer's partial in the local output once that row block's partials have landed
(``replicated``: range A also stores the partial into the local output, range B also
reduce-adds into the peer's). Data types: A/B/D bf16 or fp16, fp32 accumulate. Constraints: K and N
multiples of the tile (64 / tile_n); M arbitrary (TMA handles the residue);
row-major A ``[M, K]``, B ``[N, K]`` (``nn.Linear`` weight layout), D ``[M, N]``.

Set ``CTA_PIPE_IKET=1`` to compile IKET ranges into the kernels (see
``cute.experimental.iket``); then profile with ``run-iket`` and
``CUTE_DSL_COMPILER_OPT=iket``.
"""

from __future__ import annotations

import math
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import cute, pipeline, utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

ROLE_PLAIN, ROLE_PRODUCER, ROLE_CONSUMER = 0, 1, 2
ROLE_REDUCE = 3  # TP2 fc2 shard with the partial-sum exchange in the epilogue (batch 5)
ACT_NONE, ACT_GELU_TANH, ACT_SILU = 0, 1, 2
ACTIVATIONS = {"none": ACT_NONE, "gelu_tanh": ACT_GELU_TANH, "silu": ACT_SILU}

_TORCH_TO_CUTE = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
SPIN_SLEEP_NS = 256  # consumer backoff between counter polls


def _gelu_tanh(v):
    """tanh-approximate GELU on an fp32 accumulator fragment (Wan2.2 FFN)."""
    k0 = cute.full_like(v, 0.7978845608028654)  # sqrt(2/pi)
    k1 = cute.full_like(v, 0.044715)
    half = cute.full_like(v, 0.5)
    one = cute.full_like(v, 1.0)
    inner = k0 * (v + k1 * v * v * v)
    return half * v * (one + cute.math.tanh(inner))


def _silu(v):
    one = cute.full_like(v, 1.0)
    return v / (one + cute.math.exp(-v))


class CTAPipeGemm:
    """Persistent warp-specialised Hopper GEMM, optionally acting as the
    producer or consumer half of a CTA-pipelined GEMM pair.

    One instance = one compiled variant (role, activation, bias, dtype). Use
    :class:`CuteGemmOp` for the torch-facing wrapper that caches compiles.
    """

    def __init__(
        self,
        role: int,
        activation: int = ACT_NONE,
        has_bias: bool = False,
        *,
        tile_shape_mn: tuple[int, int] = (128, 256),
        acc_dtype=cutlass.Float32,
        fence: bool = True,
        iket: bool = False,
        raster_group: int = 1,
        cluster_shape_mn: tuple[int, int] = (1, 1),
        replicated: bool = False,
    ):
        if role not in (ROLE_PLAIN, ROLE_PRODUCER, ROLE_CONSUMER, ROLE_REDUCE):
            raise ValueError(f"bad role {role}")
        if replicated and role != ROLE_REDUCE:
            raise ValueError("replicated= is a ROLE_REDUCE option")
        if raster_group < 1:
            raise ValueError(f"raster_group must be >= 1, got {raster_group}")
        if tuple(cluster_shape_mn) not in ((1, 1), (1, 2)):
            raise ValueError(f"cluster_shape_mn must be (1, 1) or (1, 2), got {cluster_shape_mn}")
        self.role = role
        self.raster_group = raster_group  # compile-time: part of the compiled variant
        self.activation = activation
        self.has_bias = has_bias
        self.fence = fence
        self.iket = iket
        self.acc_dtype = acc_dtype
        # ROLE_REDUCE: full outputs on both GPUs; range A also stores each partial locally, range B
        # also reduce-adds into the peer's output (in-kernel all-reduce). Compile-time.
        self.replicated = replicated

        # (1, 2): the two CTAs of a cluster compute n-tiles (n, n+1) of the same row block and
        # TMA-multicast A (each loads half of the A tile into both CTAs' smem). Compile-time.
        self.cluster_shape_mn = tuple(cluster_shape_mn)
        self.tile_shape_mnk = (*tile_shape_mn, 1)  # K set in _setup_attributes
        # Two MMA warp groups (cooperative) for 128x256; one otherwise.
        self.atom_layout_mnk = (2, 1, 1) if tile_shape_mn[0] > 64 and tile_shape_mn[1] > 128 else (1, 1, 1)
        self.tiled_mma = None
        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.num_dma_warp_groups + self.num_mma_warp_groups) * 128
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * self.num_warps_per_warp_group
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.num_mma_threads = self.num_mma_warp_groups * 128
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_mma_threads)

    # ------------------------------------------------------------------ setup
    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in (64, 128):
            raise ValueError("CTA tile M must be 64/128")
        if self.tile_shape_mnk[1] not in (64, 128, 256):
            raise ValueError("CTA tile N must be 64/128/256")
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.tile_shape_mnk = (self.tile_shape_mnk[0], self.tile_shape_mnk[1], mma_inst_shape_k * 4)
        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        if is_cooperative:
            self.epi_tile = (min(128, self.tile_shape_mnk[0]), min(32, self.tile_shape_mnk[1]))
        else:
            self.epi_tile = (min(64, self.tile_shape_mnk[0]), min(32, self.tile_shape_mnk[1]))
        self.ab_stage, self.epi_stage = self._compute_stages()
        self._make_smem_layouts()

    def _compute_stages(self):
        a_shape = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.tile_shape_mnk, (0, None, None))
        ab_bytes = cute.size(a_shape) * self.a_dtype.width // 8 + cute.size(b_shape) * self.b_dtype.width // 8
        epi_stage = 4
        epi_bytes = cute.size(self.epi_tile) * self.c_dtype.width // 8 * epi_stage
        ab_stage = (self.smem_capacity // self.occupancy - (1024 + epi_bytes)) // ab_bytes
        return ab_stage, epi_stage

    def _make_smem_layouts(self):
        t = self.tile_shape_mnk
        a_k_major = self.a_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        b_k_major = self.b_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.a_layout, self.a_dtype, t[2] if a_k_major else t[0]), self.a_dtype
        )
        self.a_smem_layout_staged = cute.tile_to_shape(
            a_atom, cute.append(cute.slice_(t, (None, 0, None)), self.ab_stage),
            order=(0, 1, 2) if a_k_major else (1, 0, 2),
        )
        b_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.b_layout, self.b_dtype, t[2] if b_k_major else t[1]), self.b_dtype
        )
        self.b_smem_layout_staged = cute.tile_to_shape(
            b_atom, cute.append(cute.slice_(t, (0, None, None)), self.ab_stage),
            order=(0, 1, 2) if b_k_major else (1, 0, 2),
        )
        c_major_size = self.epi_tile[1] if self.c_layout.is_n_major_c() else self.epi_tile[0]
        c_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.c_layout, self.c_dtype, c_major_size), self.c_dtype
        )
        self.epi_smem_layout_staged = cute.tile_to_shape(
            c_atom, cute.append(self.epi_tile, self.epi_stage),
            order=(1, 0, 2) if self.c_layout.is_m_major_c() else (0, 1, 2),
        )

    @staticmethod
    def _make_tma_load(tensor, smem_layout_staged, smem_tile, num_multicast=1):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        op = (cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp() if num_multicast == 1
              else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp())
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile, num_multicast=num_multicast)

    @staticmethod
    def _make_tma_store(tensor, smem_layout_staged, epi_tile, reduce_add=False):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        op = (cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD) if reduce_add
              else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp())
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, epi_tile)

    @cute.jit
    def _tile_coord(self, tile: cutlass.Int32, num_n_tiles: cutlass.Int32, num_tiles: cutlass.Int32, n_rank):
        """Linear tile id -> (m, n). ``raster_group`` G = 1: row-block-major. G > 1:
        groups of G row blocks, walked n-major (m fastest) inside a group, so one wave
        covers G row blocks x grid/G n-tiles and reuses each B tile G times from L2.
        Cluster (1, 2): ``tile`` indexes pairs of adjacent n-tiles (``num_n_tiles`` even) and
        the CTA of cluster rank ``n_rank`` takes n = 2 * n_pair + n_rank."""
        cn = self.cluster_shape_mn[1]
        if cutlass.const_expr(cn > 1):
            num_n_tiles = num_n_tiles // cn
            num_tiles = num_tiles // cn
        if cutlass.const_expr(self.raster_group == 1):
            m_idx = tile // num_n_tiles
            n_idx = tile - m_idx * num_n_tiles
        else:
            G = self.raster_group  # noqa: N806
            group_tiles = G * num_n_tiles
            g = tile // group_tiles
            r = tile - g * group_tiles
            gm = cutlass.min(G, num_tiles // num_n_tiles - g * G)  # the last group may be short
            n_idx = r // gm
            m_idx = g * G + (r - n_idx * gm)
        if cutlass.const_expr(cn > 1):
            n_idx = n_idx * cn + n_rank
        return m_idx, n_idx

    @cute.jit
    def _reduce_tile_coord(self, tile, a_units, a_rows, a_m0, b_m0, num_n_tiles, num_tiles, n_rank):
        """ROLE_REDUCE: units ``[0, a_units)`` are range A (the peer-owned row blocks
        ``a_m0 + [0, a_rows)``), the rest range B (the own row blocks from ``b_m0``);
        ``_tile_coord`` applies within each range."""
        is_own = tile >= a_units
        m_idx = cutlass.Int32(0)
        n_idx = cutlass.Int32(0)
        if is_own:
            m_idx, n_idx = self._tile_coord(tile - a_units, num_n_tiles, num_tiles - a_rows * num_n_tiles, n_rank)
            m_idx = m_idx + b_m0
        else:
            m_idx, n_idx = self._tile_coord(tile, num_n_tiles, a_rows * num_n_tiles, n_rank)
            m_idx = m_idx + a_m0
        return m_idx, n_idx, is_own

    @cute.jit
    def _wait_row(self, mCnt: cute.Tensor, m_idx, n_ready):
        """CTA-pipelining acquire: wait until ``mCnt[m_idx] >= n_ready`` (every tile of this row
        block has landed in our memory)."""
        if cutlass.const_expr(self.iket):
            cute.experimental.iket.range_push("wait_row", m_idx)
        cnt_addr = (mCnt.iterator + m_idx).toint()
        # One lane polls with a short sleep between polls; the warp
        # barrier orders the other lanes after its acquire.
        if cute.arch.lane_idx() == 0:
            ready = cute.arch.inline_ptx(
                "ld.acquire.sys.global.s32 {$w0}, [{$r0}];",
                write_only_types=[cutlass.Int32], read_only_args=[cnt_addr],
            )
            while ready < n_ready:
                cute.arch.inline_ptx("nanosleep.u32 {$r0};", read_only_args=[SPIN_SLEEP_NS])
                ready = cute.arch.inline_ptx(
                    "ld.acquire.sys.global.s32 {$w0}, [{$r0}];",
                    write_only_types=[cutlass.Int32], read_only_args=[cnt_addr],
                )
        cute.arch.sync_warp()
        if cutlass.const_expr(self.fence):
            # Generic-proxy acquire -> async-proxy (TMA) reads.
            cute.arch.fence_proxy("async.global")
        if cutlass.const_expr(self.iket):
            cute.experimental.iket.range_pop()

    @cute.jit
    def _signal(self, mCnt: cute.Tensor, m_idx, warp_idx):
        """CTA-pipelining release: once this tile's bulk store group has fully completed (it is
        in the peer's memory), bump the peer-resident counter of its row block."""
        if warp_idx == self.epi_store_warp_id:
            if cutlass.const_expr(self.iket):
                cute.experimental.iket.range_push("signal")
            cute.arch.cp_async_bulk_wait_group(0, read=False)
            if cutlass.const_expr(self.fence):
                cute.arch.fence_proxy("async.global")
            cute.arch.sync_warp()
            if cutlass.const_expr(self.fence):
                cute.arch.fence_acq_rel_sys()
            # fence.acq_rel.sys + relaxed red is the PTX release pattern;
            # red.release.sys here would add a second system-scope fence
            # (~2 us per tile, measured).
            with cute.arch.elect_one():
                cnt_addr = (mCnt.iterator + m_idx).toint()
                cute.arch.inline_ptx(
                    "red.relaxed.sys.global.add.s32 [{$r0}], {$r1};",
                    read_only_args=[cnt_addr, cutlass.Int32(1)],
                )
            if cutlass.const_expr(self.iket):
                cute.experimental.iket.range_pop()

    # ------------------------------------------------------------------- host
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        counters: cute.Tensor,
        n_ready: cutlass.Int32,
        num_n_tiles: cutlass.Int32,
        num_tiles: cutlass.Int32,
        c2: cute.Tensor,
        counters2: cute.Tensor,
        a_m0: cutlass.Int32,
        a_rows: cutlass.Int32,
        b_m0: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """ROLE_REDUCE: ``c`` = the peer's output (range A: raw partials, plain stores), ``c2`` = the
        local output (range B: reduce-add), ``counters`` = own (wait), ``counters2`` = the peer's
        (signal). Sharded, an output holds only its owner's row blocks (tile row ``m - a_m0`` in
        ``c``, ``m - b_m0`` in ``c2``); ``replicated``, all rows, and range A also stores into
        ``c2``, range B also reduce-adds into ``c``. Other roles ignore these (aliases of ``c`` /
        ``counters``, zeros)."""
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_load(
            a, self.a_smem_layout_staged, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]), self.cluster_shape_mn[1]
        )
        tma_atom_b, tma_tensor_b = self._make_tma_load(
            b, self.b_smem_layout_staged, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), self.cluster_shape_mn[0]
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store(c, self.epi_smem_layout_staged, self.epi_tile)
        # ROLE_REDUCE (else unused aliases): c2 = reduce-add into the local output (range B);
        # replicated: c3 = plain store into the local output (range A), c4 = reduce-add into the peer's (range B).
        tma_atom_c2, tma_tensor_c2 = tma_atom_c, tma_tensor_c
        tma_atom_c3, tma_tensor_c3 = tma_atom_c, tma_tensor_c
        tma_atom_c4, tma_tensor_c4 = tma_atom_c, tma_tensor_c
        if cutlass.const_expr(self.role == ROLE_REDUCE):
            epi_layout = self.epi_smem_layout_staged
            tma_atom_c2, tma_tensor_c2 = self._make_tma_store(c2, epi_layout, self.epi_tile, reduce_add=True)
            if cutlass.const_expr(self.replicated):
                tma_atom_c3, tma_tensor_c3 = self._make_tma_store(c2, epi_layout, self.epi_tile)
                tma_atom_c4, tma_tensor_c4 = self._make_tma_store(c, epi_layout, self.epi_tile, reduce_add=True)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sA: cute.struct.Align[  # noqa: N815
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes
            ]
            sB: cute.struct.Align[  # noqa: N815
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes
            ]
            sC: cute.struct.Align[  # noqa: N815
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)], self.buffer_align_bytes
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_c, tma_tensor_c,
            bias, counters, n_ready, num_n_tiles, num_tiles,
            tma_atom_c2, tma_tensor_c2, tma_atom_c3, tma_tensor_c3, tma_atom_c4, tma_tensor_c4,
            counters2, a_m0, a_rows, b_m0,
            self.tiled_mma, self.cta_layout_mnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged, self.epi_smem_layout_staged,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(self.cluster_shape_mn[0] * self.cluster_shape_mn[1], 1, 1),  # consecutive block ids
            min_blocks_per_mp=1,
            stream=stream,
        )

    # ----------------------------------------------------------------- device
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
        mBias: cute.Tensor,
        mCnt: cute.Tensor,
        n_ready: cutlass.Int32,
        num_n_tiles: cutlass.Int32,
        num_tiles: cutlass.Int32,
        tma_atom_c2: cute.CopyAtom,
        mC2_mn: cute.Tensor,
        tma_atom_c3: cute.CopyAtom,
        mC3_mn: cute.Tensor,
        tma_atom_c4: cute.CopyAtom,
        mC4_mn: cute.Tensor,
        mCnt2: cute.Tensor,
        a_m0: cutlass.Int32,
        a_rows: cutlass.Int32,
        b_m0: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.role == ROLE_REDUCE):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c2)
                if cutlass.const_expr(self.replicated):
                    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c3)
                    cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c4)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = 0
        b_mcast_mask = 0
        if cutlass.const_expr(self.cluster_shape_mn[1] > 1):
            a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        # A cluster walks the tile list as one unit (static persistent order over clusters).
        cs = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        tile_start, tile_stride, num_units = bidx, gdim, num_tiles
        if cutlass.const_expr(cs > 1):
            tile_start, tile_stride, num_units = bidx // cs, gdim // cs, num_tiles // cs
        a_units = a_rows * num_n_tiles // cs  # ROLE_REDUCE: range A units (peer-owned row blocks)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Every MMA warp releases a stage to each CTA that multicasts into it.
        mcast_size = self.cluster_shape_mn[0] + self.cluster_shape_mn[1] - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)

        # (bM, bK, RestM, RestK) / (bN, bK, RestN, RestK) / (bM, bN, RestM, RestN)
        gA_mk = cute.local_tile(mA_mk, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None))
        gB_nk = cute.local_tile(mB_nk, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None))
        gC_mn = cute.local_tile(mC_mn, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None))
        gC2_mn, gC3_mn, gC4_mn = gC_mn, gC_mn, gC_mn
        if cutlass.const_expr(self.role == ROLE_REDUCE):
            gC2_mn = cute.local_tile(mC2_mn, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None))
            if cutlass.const_expr(self.replicated):
                gC3_mn = cute.local_tile(mC3_mn, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None))
                gC4_mn = cute.local_tile(mC4_mn, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None))

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, cluster_coord_mnk[1], a_cta_layout, cute.group_modes(sA, 0, 2), cute.group_modes(gA_mk, 0, 2)
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, cluster_coord_mnk[0], b_cta_layout, cute.group_modes(sB, 0, 2), cute.group_modes(gB_nk, 0, 2)
        )

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_wg_thread_layout = cute.make_layout(self.num_mma_warp_groups, stride=self.num_threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(mma_wg_thread_layout(warp_group_idx - self.num_dma_warp_groups))

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgC = thr_mma.partition_C(gC_mn)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)
        k_tile_cnt = cute.size(gA_mk, mode=[3])

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ---------------------------------------------------------- DMA warp
        if warp_idx == self.load_warp_id:
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            tile = cutlass.Int32(tile_start)
            while tile < num_units:
                # Static schedule (row-block-major, or grouped by ``raster_group``).
                if cutlass.const_expr(self.role == ROLE_REDUCE):
                    # The mainloop needs nothing from the peer (the epilogue store waits for its partials).
                    m_idx, n_idx, _ = self._reduce_tile_coord(
                        tile, a_units, a_rows, a_m0, b_m0, num_n_tiles, num_tiles, cluster_coord_mnk[1]
                    )
                else:
                    m_idx, n_idx = self._tile_coord(tile, num_n_tiles, num_tiles, cluster_coord_mnk[1])

                if cutlass.const_expr(self.role == ROLE_CONSUMER):
                    # Wait until every producer tile of this row block has landed in our memory.
                    self._wait_row(mCnt, m_idx, n_ready)

                if cutlass.const_expr(self.iket):
                    cute.experimental.iket.range_push("tma_tile")
                tAgA_k = tAgA[(None, m_idx, None)]
                tBgB_k = tBgB[(None, n_idx, None)]
                producer_state.reset_count()
                for _k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom_a, tAgA_k[(None, producer_state.count)], tAsA[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state), mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b, tBgB_k[(None, producer_state.count)], tBsB[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state), mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                if cutlass.const_expr(self.iket):
                    cute.experimental.iket.range_pop()
                tile = tile + tile_stride
            mainloop_pipeline.producer_tail(producer_state)

        # ---------------------------------------------------------- MMA warps
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            read_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            release_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
            num_k_blocks = cute.size(tCrA, mode=[2])

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4), self.c_dtype
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group)
            tRS_sD = thr_copy_r2s.partition_D(sC)  # (R2S, R2S_M, R2S_N, PIPE_D)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)  # (R2S, R2S_M, R2S_N)
            # Tile-local (m, n) coordinate of every accumulator element, in
            # accumulator fragment order: used to fetch each element's bias.
            # ``thr_mma`` above is sliced at the warp group's first thread (only
            # its shape / smem descriptors are used); coordinates need this
            # thread's own slice of the tiled MMA.
            cC = cute.make_identity_tensor((self.tile_shape_mnk[0], self.tile_shape_mnk[1]))
            mma_tidx = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            tCcC = tiled_mma.get_slice(mma_tidx).partition_C(cC)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            tma_store_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage, producer_group=tma_store_group
            )

            tile = cutlass.Int32(tile_start)
            tiles_done = cutlass.Int32(0)
            while tile < num_units:
                is_own = cutlass.Boolean(False)
                if cutlass.const_expr(self.role == ROLE_REDUCE):
                    m_idx, n_idx, is_own = self._reduce_tile_coord(
                        tile, a_units, a_rows, a_m0, b_m0, num_n_tiles, num_tiles, cluster_coord_mnk[1]
                    )
                else:
                    m_idx, n_idx = self._tile_coord(tile, num_n_tiles, num_tiles, cluster_coord_mnk[1])
                c_m = m_idx
                if cutlass.const_expr(self.role == ROLE_REDUCE and not self.replicated):
                    c_m = cutlass.max(m_idx - a_m0, 0)  # range A: the peer's output starts at its first row block
                gC_tile = gC_mn[(None, None, c_m, n_idx)]

                # ---------------------------------------------------- mainloop
                if cutlass.const_expr(self.iket):
                    cute.experimental.iket.range_push("mma_tile")
                read_state.reset_count()
                release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()
                for _k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[k_block_coord], tCrB[k_block_coord], accumulators)
                    cute.nvgpu.warpgroup.commit_group()
                    read_state.advance()
                for _k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    mainloop_pipeline.consumer_wait(read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, read_state.index)
                        cute.gemm(tiled_mma, accumulators, tCrA[k_block_coord], tCrB[k_block_coord], accumulators)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()
                    read_state.advance()
                cute.nvgpu.warpgroup.wait_group(0)
                for _k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(release_state)
                    release_state.advance()
                if cutlass.const_expr(self.iket):
                    cute.experimental.iket.range_pop()

                # ---------------------------------------------------- epilogue
                if cutlass.const_expr(self.iket):
                    cute.experimental.iket.range_push("epi_tile")
                tCgC_for_tma = cute.zipped_divide(gC_tile, self.epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), tCgC_for_tma
                )
                bSG_gD2, bSG_gD3, bSG_gD4 = bSG_gD, bSG_gD, bSG_gD
                if cutlass.const_expr(self.role == ROLE_REDUCE):
                    # Own tiles go to the local output (sharded: tile row m - b_m0; clamped for range A, unused).
                    b_m = m_idx
                    if cutlass.const_expr(not self.replicated):
                        b_m = cutlass.max(m_idx - b_m0, 0)
                    _, bSG_gD2 = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c2, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                        cute.zipped_divide(gC2_mn[(None, None, b_m, n_idx)], self.epi_tile),
                    )
                    if cutlass.const_expr(self.replicated):
                        _, bSG_gD3 = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_c3, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                            cute.zipped_divide(gC3_mn[(None, None, m_idx, n_idx)], self.epi_tile),
                        )
                        _, bSG_gD4 = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_c4, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                            cute.zipped_divide(gC4_mn[(None, None, m_idx, n_idx)], self.epi_tile),
                        )
                epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
                epi_tile_shape = tCgC_for_tma.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
                num_prev_epi_tiles = tiles_done * epi_tile_num
                if cutlass.const_expr(self.role == ROLE_REDUCE):
                    if is_own:  # final rows: + bias here, + the peer's partial by the reduce-add stores
                        if warp_idx == self.epi_store_warp_id:
                            # The storing warp acquires the row block's counter itself (every peer
                            # partial tile has landed) + async-proxy fence before its reduce-adds.
                            self._wait_row(mCnt, m_idx, n_ready)
                        if cutlass.const_expr(self.has_bias):
                            n_base = n_idx * self.tile_shape_mnk[1]
                            for i in cutlass.range(cute.size(accumulators), unroll_full=True):
                                accumulators[i] = accumulators[i] + mBias[n_base + tCcC[i][1]].to(self.acc_dtype)
                elif cutlass.const_expr(self.has_bias):
                    # Bias in accumulator order (before the activation below).
                    n_base = n_idx * self.tile_shape_mnk[1]
                    for i in cutlass.range(cute.size(accumulators), unroll_full=True):
                        accumulators[i] = accumulators[i] + mBias[n_base + tCcC[i][1]].to(self.acc_dtype)

                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                    acc_vec = tRS_rD.load()
                    if cutlass.const_expr(self.activation == ACT_GELU_TANH):
                        acc_vec = _gelu_tanh(acc_vec)
                    elif cutlass.const_expr(self.activation == ACT_SILU):
                        acc_vec = _silu(acc_vec)
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(tRS_sD, mode=[3])
                    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    if warp_idx == self.epi_store_warp_id:
                        if cutlass.const_expr(self.role == ROLE_REDUCE):
                            if is_own:  # reduce-add onto the peer's partial (replicated: in both outputs)
                                cute.copy(tma_atom_c2, bSG_sD[(None, epi_buffer)], bSG_gD2[(None, gmem_coord)])
                                if cutlass.const_expr(self.replicated):
                                    cute.copy(tma_atom_c4, bSG_sD[(None, epi_buffer)], bSG_gD4[(None, gmem_coord)])
                            else:  # the raw partial into the peer's output (replicated: and the local one)
                                cute.copy(tma_atom_c, bSG_sD[(None, epi_buffer)], bSG_gD[(None, gmem_coord)])
                                if cutlass.const_expr(self.replicated):
                                    cute.copy(tma_atom_c3, bSG_sD[(None, epi_buffer)], bSG_gD3[(None, gmem_coord)])
                        else:
                            cute.copy(tma_atom_c, bSG_sD[(None, epi_buffer)], bSG_gD[(None, gmem_coord)])
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()
                if cutlass.const_expr(self.iket):
                    cute.experimental.iket.range_pop()

                if cutlass.const_expr(self.role == ROLE_PRODUCER):
                    # The tile is in the consumer's memory once the bulk store group has completed.
                    self._signal(mCnt, m_idx, warp_idx)
                if cutlass.const_expr(self.role == ROLE_REDUCE):
                    if tile < a_units:  # range A: the partial is in the peer's output
                        self._signal(mCnt2, m_idx, warp_idx)

                tile = tile + tile_stride
                tiles_done = tiles_done + 1
            tma_store_pipeline.producer_tail()


# ---------------------------------------------------------------------- torch
def _cute_2d(t: torch.Tensor) -> cute.Tensor:
    # torch only exports a DLPack capsule while the tensor's device is current;
    # the kernel may still be launched from another device (peer memory).
    with torch.cuda.device(t.device):
        return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def _cute_1d(t: torch.Tensor) -> cute.Tensor:
    with torch.cuda.device(t.device):
        return from_dlpack(t, assumed_align=16).mark_compact_shape_dynamic(mode=0)


_SM_COUNT: dict[torch.device, int] = {}


def _sm_count(device: torch.device) -> int:
    if device not in _SM_COUNT:
        _SM_COUNT[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNT[device]


_MAX_CLUSTERS: dict[tuple[torch.device, int], int] = {}


def _max_clusters(device: torch.device, cluster_size: int) -> int:
    """Co-resident clusters of ``cluster_size`` CTAs at one CTA per SM (a persistent grid must not
    exceed it: clusters need their CTAs in one GPC)."""
    key = (device, cluster_size)
    if key not in _MAX_CLUSTERS:
        with torch.cuda.device(device):
            _MAX_CLUSTERS[key] = utils.HardwareInfo(device.index).get_max_active_clusters(cluster_size)
    return _MAX_CLUSTERS[key]


class CTAPipePlainGemm(CTAPipeGemm):
    """Single-GPU role (no counters). Separate class only so the DSL mangles a
    distinct kernel name; IKET keys its instrumentation table by kernel name."""


class CTAPipeProducerGemm(CTAPipeGemm):
    """GEMM1 role: writes its output tile into peer memory and bumps the row counter."""


class CTAPipeConsumerGemm(CTAPipeGemm):
    """GEMM2 role: spins on the row counter before loading a row block of ``A``."""


class CTAPipeReduceGemm(CTAPipeGemm):
    """TP2 fc2 shard: exchanges bf16 partials with the peer inside the epilogue (``cute_tp2.py``)."""


_ROLE_CLASS = {
    ROLE_PLAIN: CTAPipePlainGemm,
    ROLE_PRODUCER: CTAPipeProducerGemm,
    ROLE_CONSUMER: CTAPipeConsumerGemm,
    ROLE_REDUCE: CTAPipeReduceGemm,
}


class CuteGemmOp:
    """Torch-facing launcher for :class:`CTAPipeGemm`; compiles once per
    (role, activation, bias, dtype, tile) and launches on a torch stream.

    ``launch(a, w, bias, out, counters, *, n_ready, device, stream)`` computes
    ``out = act(a @ w^T + bias)`` where ``a`` is ``[M, K]``, ``w`` is ``[N, K]``
    and ``out`` is ``[M, N]`` (row stride may exceed ``N``: a column slice of a wider
    buffer). ``out`` and ``counters`` may live on a peer device (producer role).
    ``counters`` must have at least ``ceil(M / tile_m)`` int32 entries.

    Host cost: each torch -> ``cute.Tensor`` (DLPack) conversion costs ~10 us. ``static``
    names the arguments (of ``a``, ``w``, ``bias``, ``out``, ``counters``) that are
    persistent buffers owned by the caller; their conversions are cached by
    ``(data_ptr, shape, stride, dtype)``. A cached conversion keeps its buffer alive, so
    call :meth:`clear_cache` when such a buffer is replaced.
    """

    def __init__(
        self,
        role: int,
        activation: str = "none",
        has_bias: bool = False,
        *,
        tile_shape_mn: tuple[int, int] = (128, 256),
        fence: bool = True,
        iket: bool | None = None,
        raster_group: int = 1,
        cluster_shape_mn: tuple[int, int] = (1, 1),
        replicated: bool = False,
    ):
        if iket is None:
            iket = os.environ.get("CTA_PIPE_IKET", "0") == "1"
        self.role = role
        self.tile_m, self.tile_n = tile_shape_mn
        self.gemm = _ROLE_CLASS[role](
            role, ACTIVATIONS[activation], has_bias, tile_shape_mn=tile_shape_mn, fence=fence, iket=iket,
            raster_group=raster_group, cluster_shape_mn=cluster_shape_mn, replicated=replicated,
        )
        self.cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        self.has_bias = has_bias
        self._compiled = None
        self._compiled_key = None
        self._dummy: dict[torch.device, torch.Tensor] = {}
        self._cute_cache: dict[tuple, cute.Tensor] = {}

    def clear_cache(self) -> None:
        self._cute_cache.clear()

    def _cute(self, t: torch.Tensor, cached: bool) -> cute.Tensor:
        conv = _cute_2d if t.dim() == 2 else _cute_1d
        if not cached:
            return conv(t)
        key = (t.data_ptr(), t.shape, t.stride(), t.dtype)
        ct = self._cute_cache.get(key)
        if ct is None:
            ct = self._cute_cache[key] = conv(t)
        return ct

    def num_m_tiles(self, M: int) -> int:
        return -(-M // self.tile_m)

    def _dummy_for(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (dtype, device)
        if key not in self._dummy:
            self._dummy[key] = torch.zeros(16, dtype=dtype, device=device)
        return self._dummy[key]

    def launch(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None,
        out: torch.Tensor,
        counters: torch.Tensor | None,
        *,
        n_ready: int = 0,
        device: torch.device,
        stream: torch.cuda.Stream,
        static: tuple[str, ...] = (),
        reduce: tuple | None = None,
    ) -> None:
        """``reduce`` (ROLE_REDUCE only) = ``(out2, counters2, a_m0, a_rows, b_m0)``: ``out`` is the
        peer's output (range A: the raw partials of row blocks ``a_m0 + [0, a_rows)``), ``out2`` the
        local one (range B: the other ``num_m - a_rows`` row blocks from ``b_m0``, reduce-added onto
        the peer's partials, so it must not be touched in between). Sharded (not ``replicated``), an
        output holds only its owner's row blocks, from row 0; replicated, all ``M`` rows, and each
        kernel writes both. ``counters`` own (wait for ``n_ready``), ``counters2`` the peer's
        (signal); ``static`` may name ``out2``, ``counters2``."""
        M, K = a.shape
        N = w.shape[0]
        if w.shape[1] != K or out.shape[1] != N or (self.role != ROLE_REDUCE and out.shape[0] != M):
            raise ValueError(f"shape mismatch: a {tuple(a.shape)} w {tuple(w.shape)} out {tuple(out.shape)}")
        if K % 64 or N % (self.tile_n * self.cluster_size):
            raise ValueError(f"K ({K}) must be a multiple of 64 and N ({N}) of tile_n x cluster ({self.tile_n} x "
                             f"{self.cluster_size})")
        if a.dtype not in _TORCH_TO_CUTE or a.dtype != w.dtype or a.dtype != out.dtype:
            raise ValueError("A, W and out must all be bf16 or all fp16")
        if not (a.is_contiguous() and w.is_contiguous()):
            raise ValueError("A and W must be contiguous row-major")
        # ``out`` may be a column slice of a wider row-major buffer (the TMA descriptor takes the
        # runtime row stride); TMA needs 16-B aligned rows.
        if out.stride(1) != 1 or out.stride(0) % 8 or out.stride(0) < N or out.data_ptr() % 16:
            raise ValueError("out must be row-major with a 16-B aligned base and row stride (multiple of 8, >= N)")
        if (bias is None) == self.has_bias:
            raise ValueError("bias presence does not match the compiled variant")
        num_m, num_n = self.num_m_tiles(M), N // self.tile_n
        num_tiles = num_m * num_n
        static = set(static)
        if counters is None:
            counters = self._dummy_for(torch.int32, device)
            static.add("counters")
        elif counters.numel() < num_m or counters.dtype != torch.int32:
            raise ValueError("counters must be int32 with >= ceil(M / tile_m) entries")
        if bias is None:
            bias = self._dummy_for(a.dtype, device)
            static.add("bias")
        cs = self.cluster_size
        max_ctas = _sm_count(device) if cs == 1 else cs * _max_clusters(device, cs)
        num_ctas = min(num_tiles, max_ctas)
        if self.role in (ROLE_CONSUMER, ROLE_REDUCE):
            # Same number of tiles per CTA (waves), spread evenly over fewer CTAs:
            # 768 tiles -> 128 CTAs x 6 instead of 132 CTAs x 5-6 (measured faster).
            waves = -(-num_tiles // num_ctas)
            num_ctas = -(-num_tiles // waves)
        num_ctas = -(-num_ctas // cs) * cs  # whole clusters (num_tiles is a multiple of cs)

        cu_stream = cuda.CUstream(stream.cuda_stream)
        args = (self._cute(a, "a" in static), self._cute(w, "w" in static), self._cute(out, "out" in static),
                self._cute(bias, "bias" in static), self._cute(counters, "counters" in static),
                n_ready, num_n, num_tiles)
        if (reduce is not None) != (self.role == ROLE_REDUCE):
            raise ValueError("reduce= is required by (only) ROLE_REDUCE")
        if reduce is not None:
            out2, counters2, a_m0, a_rows, b_m0 = reduce
            if (out2.dtype != a.dtype or out2.dim() != 2 or out2.shape[1] != N or out2.stride(1) != 1
                    or out2.stride(0) % 8 or out2.data_ptr() % 16):
                raise ValueError(f"out2 must be a row-major, 16-B aligned [rows, {N}] tensor of A's dtype")
            if self.gemm.replicated and (out.shape[0] != M or out2.shape[0] != M):
                raise ValueError(f"replicated: out and out2 must have M = {M} rows")
            if counters2 is None or counters2.dtype != torch.int32 or counters2.numel() < num_m:
                raise ValueError("counters2 must be int32 with >= ceil(M / tile_m) entries")
            if not (0 <= a_rows <= num_m and 0 <= a_m0 and a_m0 + a_rows <= num_m and 0 <= b_m0
                    and b_m0 + num_m - a_rows <= num_m):
                raise ValueError(f"bad row ranges a=[{a_m0}, +{a_rows}) b=[{b_m0}, ...) for {num_m} row blocks")
            args += (self._cute(out2, "out2" in static), self._cute(counters2, "counters2" in static),
                     a_m0, a_rows, b_m0)
        else:  # unused: alias ``out`` / ``counters``
            args += (args[2], args[4], 0, 0, 0)
        args += (num_ctas, cu_stream)
        key = (a.dtype,)
        with torch.cuda.device(device):
            if self._compiled is None or self._compiled_key != key:
                self._compiled = cute.compile(self.gemm, *args)
                self._compiled_key = key
            self._compiled(*args)
