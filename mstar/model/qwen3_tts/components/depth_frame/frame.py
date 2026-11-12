"""The host side of `kernel.frame_kernel`: weights, scratch, task lists and the launch."""
from __future__ import annotations

import torch
import triton
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from mstar.engine.state import FusedDrawArgs, draw_row_layout
from mstar.model.qwen3_tts.components.depth_frame import attend, kernel, linear, schedule
from mstar.model.qwen3_tts.components.depth_frame.schedule import NPAGES, TB

XSTAGES = 5  # 16 KB slots x streams through
MAX_ROWS, GROUPS = 64, 16


class DepthFrame:
    """Stacked copies of a `Qwen3TTSCodePredictor`'s weights, the kernel's scratch and task lists, and its launch.

    The kernel is compiled for one CTA and launched in clusters of two, with each CTA's 4 KB combine buffer and its
    mbarrier placed after Triton's shared memory (see `cluster`).
    """

    max_rows = MAX_ROWS

    def __init__(self, code_predictor, talker_table: torch.Tensor):
        cp, fused = code_predictor, code_predictor.fused
        attn = cp.model.layers[0].self_attn
        layers, hidden, inter = len(fused.qkv), fused.o[0].shape[0], fused.down[0].shape[1]
        self.head_dim = attn.head_dim
        self.eps, self.eps_q, self.eps_k = fused.eps, attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon
        self._stack_weights(cp, talker_table, layers, hidden, inter)
        self._allocate_scratch(fused, attn, layers, hidden, inter)
        self._schedule(torch.cuda.get_device_properties(self.X.device).multi_processor_count // 2 * 2, layers)
        self.m16, self.bp, self.bo16, self.b1 = linear.layouts(16)
        self.m32, _, self.bo32, _ = linear.layouts(32)
        self.m64 = linear.layouts(64)[0]
        self.attn_layouts = attend.layouts()
        self._compiled: dict[bool, tuple] = {}
        self._compile_sampled(talker_table.shape[1], attn, layers)

    def _stack_weights(self, cp, talker_table: torch.Tensor, layers: int, hidden: int, inter: int) -> None:
        """Each projection's weights for all layers in one matrix, and the TMA descriptors its tiles load through."""
        fused = cp.fused
        # each 32-row GATEUP tile's gate rows then its up rows, so one task reads one 64-row tile
        gate_up = [torch.stack([g.view(inter // 32, 32, hidden), u.view(inter // 32, 32, hidden)], dim=1)
                   for g, u in zip(fused.gate, fused.up, strict=True)]
        self.w = dict(
            qkv=torch.cat(fused.qkv), o=torch.cat(fused.o), down=torch.cat(fused.down),
            gu=torch.cat(gate_up).reshape(-1, hidden), head=fused.heads.reshape(-1, hidden),
            win=cp.small_to_mtp_projection.weight,
        )
        self.bias_in = cp.small_to_mtp_projection.bias
        self.qn = torch.stack([cp.model.layers[i].self_attn.q_norm.weight for i in range(layers)])
        self.kn = torch.stack([cp.model.layers[i].self_attn.k_norm.weight for i in range(layers)])
        self.tab0 = talker_table
        # group 0's input is a row of the projected Talker table, as groups 1-14's are of `input_tables`
        with torch.no_grad():
            self.tp0 = cp.small_to_mtp_projection(talker_table).contiguous()
        self.cemb = torch.stack([e.weight for e in cp.model.codec_embedding])
        self.tproj = torch.stack(cp.input_tables)

        sl4 = linear.shared_layout(4)

        def tiles(w, rows, chunks):
            # [chunk, 64-column half, row, 64 columns]: one copy writes `chunks` chunks of `rows` rows (`_issue_tile`)
            n, k = w.shape
            return TensorDescriptor(w, [k // 128, 2, n, 64], [128, 64, k, 1], [chunks, 2, rows, 64], sl4)

        self.wd = [tiles(self.w["qkv"], 32, 8), tiles(self.w["o"], 16, 8), tiles(self.w["gu"], 64, 4),
                   tiles(self.w["down"], 16, 4), tiles(self.w["win"], 16, 8), tiles(self.w["head"], 16, 8)]

    def _allocate_scratch(self, fused, attn, layers: int, hidden: int, inter: int) -> None:
        """The activations tasks hand from phase to phase, and the KV cache, positions and partial sums of a frame."""
        dev, bf = self.w["qkv"].device, self.w["qkv"].dtype
        self.X = torch.zeros(MAX_ROWS, hidden, device=dev, dtype=bf)
        self.QKV = torch.zeros(MAX_ROWS, self.w["qkv"].shape[0] // layers, device=dev, dtype=bf)
        self.ATT = torch.zeros(MAX_ROWS, fused.o[0].shape[1], device=dev, dtype=bf)
        self.ACT = torch.zeros(MAX_ROWS, inter, device=dev, dtype=bf)
        self.LOGITS = torch.zeros(MAX_ROWS, fused.heads.shape[1], device=dev, dtype=bf)
        sl = linear.shared_layout()
        self.xd = [TensorDescriptor.from_tensor(t, [MAX_ROWS, 128], sl) for t in (self.X, self.ATT, self.ACT)]
        # frame-local scratch the submodule hands back to `run`: every slot a frame reads, it wrote first
        self.kv_cache = torch.zeros(layers, MAX_ROWS, 2, GROUPS, attn.num_kv_heads, self.head_dim, device=dev, dtype=bf)
        self.depth_positions = torch.arange(GROUPS, dtype=torch.int32, device=dev)[:, None, None].expand(
            GROUPS, MAX_ROWS, 1).contiguous()
        self.partials = torch.zeros(MAX_ROWS, MAX_ROWS, device=dev, dtype=torch.float32)
        self.rope = attend.rope_table(GROUPS, float(attn.rope_theta), self.head_dim, dev)

    def _compile_sampled(self, talker_width: int, attn, layers: int) -> None:
        """Compile the sampled kernel before any graph capture, on tensors laid out as the submodule's, and shrink the
        grid if fewer clusters fit at once than the GPU has SM pairs."""
        dev, bf = self.X.device, self.X.dtype
        z = lambda *shape, dtype=bf: torch.zeros(*shape, device=dev, dtype=dtype)  # noqa: E731
        codes, f32 = z(MAX_ROWS, dtype=torch.long), torch.float32
        draw = FusedDrawArgs(z(MAX_ROWS, dtype=f32), z(MAX_ROWS, dtype=torch.int32), z(MAX_ROWS, dtype=f32),
                             codes, codes)
        tensors = (z(MAX_ROWS, talker_width), codes,
                   z(layers, MAX_ROWS, 2, GROUPS, attn.num_kv_heads, self.head_dim),
                   z(GROUPS, MAX_ROWS, 1, dtype=torch.int32), z(MAX_ROWS, MAX_ROWS, dtype=f32),
                   z(MAX_ROWS, GROUPS, dtype=torch.long), z(MAX_ROWS, talker_width), draw)
        self._compiled[True] = self._compile(self._args(*tensors, MAX_ROWS, True))
        fit = self._clusters(self._compiled[True][0])
        if 2 * fit < self.grid:
            # Some parts hold fewer clusters at once than they have SM pairs. Every CTA spins on the others, so the
            # grid shrinks to what fits.
            self._schedule(2 * fit, layers)
            self._compiled[True] = self._compile(self._args(*tensors, MAX_ROWS, True))

    def _schedule(self, grid: int, layers: int) -> None:
        self.grid = grid
        tasks, ntasks, nctr = schedule.build(schedule.frame_phases(layers, GROUPS), grid, NPAGES, TB)
        self.maxt = tasks.shape[1]
        self.tasks = torch.from_numpy(schedule.pack(tasks, ntasks, TB)).to(self.X.device)
        self.ntasks = torch.from_numpy(ntasks).to(self.X.device)
        self.ctrs = torch.zeros(nctr, device=self.X.device, dtype=torch.int32)

    @torch.compiler.disable
    def run(
        self, last_hidden: torch.Tensor, codes_in: torch.Tensor, kv_cache: torch.Tensor, positions: torch.Tensor,
        partials: torch.Tensor, all_codes: torch.Tensor, codec_embed_sum: torch.Tensor, draw: FusedDrawArgs | None,
    ) -> None:
        """Fill ``all_codes`` and ``codec_embed_sum`` and write ``kv_cache``, as `_depth_loop` does.

        With ``draw``, ``codes_in`` holds each row's group-0 code and groups 1-15 are drawn as the sampler's
        `fused_sample` would, at offsets ``draw.offset`` .. ``draw.offset`` + 14; the caller advances the offsets.
        Without it, ``codes_in`` is ``[rows, 16]`` and every group's code is given.
        """
        rows = last_hidden.shape[0]
        assert rows <= MAX_ROWS, f"the frame kernel runs at most {MAX_ROWS} rows, got {rows}"
        sampled = draw is not None
        if not sampled:
            # typed placeholders: the draw's loads are compiled out
            z = torch.zeros(1, device=last_hidden.device)
            draw = FusedDrawArgs(z, z.int(), z, z.long(), z.long())
        args = self._args(last_hidden, codes_in, kv_cache, positions, partials, all_codes, codec_embed_sum, draw,
                          rows, sampled)
        if sampled not in self._compiled:
            self._compiled[sampled] = self._compile(args)
        compiled, off = self._compiled[sampled]
        args["OFF"] = off
        self.ctrs.zero_()
        compiled[(self.grid // 2, 1, 1)](*[args[n] for n in kernel.frame_kernel.arg_names])

    def _args(self, last_hidden, codes_in, kv_cache, positions, partials, all_codes, codec_embed_sum, draw, rows,
              sampled):
        tensors = (last_hidden, codes_in, kv_cache, positions, partials, all_codes, codec_embed_sum, *draw)
        # The kernel was specialized on the first call's pointers being 16-byte aligned.
        assert all(t.data_ptr() % 16 == 0 for t in tensors), "the frame kernel's tensors must be 16-byte aligned"
        return dict(
            TASKS=self.tasks, NTASKS=self.ntasks, CTRS=self.ctrs,
            x_desc=self.xd[0], att_desc=self.xd[1], act_desc=self.xd[2],
            hid_desc=TensorDescriptor.from_tensor(last_hidden, [MAX_ROWS, 128], linear.shared_layout()),
            wqkv_desc=self.wd[0], wo_desc=self.wd[1], wgu_desc=self.wd[2], wd_desc=self.wd[3], win_desc=self.wd[4],
            whead_desc=self.wd[5],
            X=self.X, QKV=self.QKV, ATT=self.ATT, ACT=self.ACT, LOGITS=self.LOGITS, P=partials, BIAS_IN=self.bias_in,
            QN=self.qn, KN=self.kn, POS=positions, s_pos=positions.stride(0), ROPE=self.rope,
            KV=kv_cache, s_kv_l=kv_cache.stride(0), s_kv_b=kv_cache.stride(1), s_kv_c=kv_cache.stride(2),
            s_kv_n=kv_cache.stride(3), s_kv_h=kv_cache.stride(4),
            CODES_IN=codes_in, s_codes_in=codes_in.stride(0), CODES=all_codes, s_codes=all_codes.stride(0),
            SUM=codec_embed_sum, s_sum=codec_embed_sum.stride(0), TAB0=self.tab0, s_tab0=self.tab0.stride(0),
            CEMB=self.cemb, s_cemb_g=self.cemb.stride(0), s_cemb=self.cemb.stride(1),
            TP0=self.tp0, TPROJ=self.tproj, s_tproj_g=self.tproj.stride(0), s_tp=self.tp0.stride(0),
            TEMP=draw.temperature, TOPK=draw.top_k, TOPP=draw.top_p, SEED=draw.seed, OFFSET=draw.offset,
            M=rows, eps=self.eps, eps_q=self.eps_q, eps_k=self.eps_k, sm_scale=self.head_dim ** -0.5, OFF=1024,
            MAXT=self.maxt, NPAGES=NPAGES, TB=TB, XSTAGES=XSTAGES,
            MMA16=self.m16, MMA32=self.m32, MMA64=self.m64, BLK_P=self.bp, BLK_O16=self.bo16, BLK_O32=self.bo32,
            BLK_1D=self.b1, L1=self.attn_layouts[0], L3=self.attn_layouts[1], K3=self.attn_layouts[2],
            V3=self.attn_layouts[3], E1=gl.BlockedLayout([8], [32], [4], [0]),
            E2=gl.BlockedLayout([1, 4], [1, 32], [2, 2], [1, 0]), SAMPLE=sampled,
            SLAY=draw_row_layout(),
        )

    def _compile(self, args: dict):
        """Compile for one CTA, to launch as clusters of two with the combine buffer after Triton's memory."""
        # A row count Triton would specialize on (1, or a multiple of 16) would be baked into every later launch.
        compiled = kernel.frame_kernel.warmup(**{**args, "M": MAX_ROWS - 1}, num_warps=4, grid=(self.grid,))
        compiled._init_handles()
        off = (compiled.metadata.shared + 15) // 16 * 16
        # what the launcher reads: warps, CTAs per cluster, dynamic shared memory
        assert len(compiled.packed_metadata) == 3, "Triton's launch metadata changed shape"
        compiled.packed_metadata = (4, 2, off + 4096 + 16)
        return compiled, off

    def _clusters(self, compiled) -> int:
        """How many clusters of two can be resident at once."""
        smem = compiled.packed_metadata[2]
        fit = triton.runtime.driver.active.utils.cuOccupancyMaxActiveClusters(compiled.function, smem, 2)
        assert fit > 0, f"no cluster of two fits in {smem} B of shared memory"
        return fit
