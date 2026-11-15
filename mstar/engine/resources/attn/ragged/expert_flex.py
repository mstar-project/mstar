"""Planned flex masks for the Praxis expert's rectangular cross and chunk self reads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask

from mstar.engine.resources.attn.flex import flex_attention_masked
from mstar.engine.resources.attn.ragged.base import RaggedExpertAttnManager, _RaggedManagerBase
from mstar.engine.resources.attn.ragged.config import ExpertAttentionStep, RaggedAttentionConfig, cross_label
from mstar.engine.resources.attn.ragged.expert_mask import noised_frame_offset
from mstar.engine.resources.step import StepContext

_PAD = 0x7FFF


def _packed(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = adjacency.sum(axis=1).astype(np.int32)
    indices = np.argsort(~adjacency, axis=1, kind="stable").astype(np.int32)
    return counts[None, None], indices[None, None]


def _runs(fields: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    length = fields[0].size
    if not length:
        return np.empty(0, np.int64), np.empty(0, np.int64), tuple(f[:0] for f in fields)
    changes = np.zeros(length - 1, dtype=bool)
    for field in fields:
        changes |= field[1:] != field[:-1]
    starts = np.r_[0, np.flatnonzero(changes) + 1]
    ends = np.r_[starts[1:], length]
    return starts, ends, tuple(field[starts] for field in fields)


def _visibility(q: tuple, k: tuple, *, mode: str, window: int, include_noised: bool):
    qs, qc, qstart = q
    ks, kind, frame, kc = k
    same = qs == ks
    if mode == "self":
        rule = qc == kc
    else:
        clean = (kind == 1) & (frame < qstart) & (
            frame >= qstart - window
        )
        rule = (kind == 0) | clean
        if include_noised:
            rule = rule | ((kind == 2) & (kc == qc))
    # Captured padding rows attend a real or padded key, then are selected to
    # zero after the kernel. No real query ever sees a padded key.
    return (same & rule & (qs != _PAD)) | (qs == _PAD)


class FlexExpertWrapper:
    """One planned Q/KV pair; graph buffers keep stable addresses across replays."""

    def __init__(
        self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
        mode: str, chunk_len: int, future_per_chunk: int, tokens_per_frame: int,
        window_frames: int, include_noised: bool, block_size: int, full_blocks: bool,
        max_num_segments: int | None = None, max_total_tokens: int | None = None,
        device: torch.device = torch.device("cuda"), accelerator_graph: bool = False,
        sm_scale: float | None = None, q_data_type: torch.dtype = torch.bfloat16,
    ) -> None:
        if mode not in ("cross", "self"):
            raise ValueError(f"unknown expert mask mode {mode!r}")
        self.mode, self.chunk_len, self.future_per_chunk = mode, chunk_len, future_per_chunk
        self.tokens_per_frame, self.window_frames = tokens_per_frame, window_frames
        self.include_noised = include_noised
        self.block_size, self.full_blocks = block_size, full_blocks
        self.device, self.accelerator_graph = device, accelerator_graph
        self.num_qo_heads, self.num_kv_heads, self.head_dim = num_qo_heads, num_kv_heads, head_dim
        self.q_data_type = q_data_type
        self.sm_scale = sm_scale if sm_scale is not None else head_dim**-0.5
        self.max_num_segments, self.max_total_tokens = max_num_segments, max_total_tokens
        self._planned = None
        self._num_segments = 0
        self._q_len = self._kv_len = 0
        self._q: tuple[torch.Tensor, ...] | None = None
        self._k: tuple[torch.Tensor, ...] | None = None
        self.block_mask: BlockMask | None = None
        if accelerator_graph:
            if max_num_segments is None or max_num_segments <= 0 or max_total_tokens is None or max_total_tokens <= 0:
                raise ValueError("expert graph mode needs positive segment and token ceilings")
            n = -(-max_total_tokens // block_size)
            i32 = dict(dtype=torch.int32, device=device)
            self._q = tuple(torch.full((max_total_tokens,), _PAD if i == 0 else -1, **i32) for i in range(3))
            self._k = tuple(torch.full((max_total_tokens,), _PAD if i == 0 else -1, **i32) for i in range(4))
            self.block_mask = self._mask_over(
                max_total_tokens, max_total_tokens,
                torch.zeros((1, 1, n), **i32), torch.zeros((1, 1, n, n), **i32),
                torch.zeros((1, 1, n), **i32) if full_blocks else None,
                torch.zeros((1, 1, n, n), **i32) if full_blocks else None,
            )

    @property
    def num_segments(self) -> int:
        return self._num_segments

    def forget_plan(self) -> None:
        self._planned = None

    def _mask_over(self, q_len, kv_len, kv_num, kv_idx, full_num, full_idx) -> BlockMask:
        def mask_mod(b, h, qi, ki):
            return _visibility(
                tuple(f[qi] for f in self._q), tuple(f[ki] for f in self._k),
                mode=self.mode, window=self.window_frames, include_noised=self.include_noised,
            )

        return BlockMask.from_kv_blocks(
            kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=self.block_size,
            mask_mod=mask_mod, seq_lengths=(q_len, kv_len), compute_q_blocks=False,
        )

    def _host(self, array: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(array, dtype=np.int32))
        return t.pin_memory() if self.device.type == "cuda" else t

    def _device(self, array: np.ndarray) -> torch.Tensor:
        return self._host(array).to(self.device, non_blocking=True)

    def _fields(
        self, layouts: Sequence[tuple[int, int, int, int]], q_cap: int, k_cap: int,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], int, int]:
        q_parts: list[list[np.ndarray]] = [[], [], []]
        k_parts: list[list[np.ndarray]] = [[], [], [], []]

        def append(parts, fields):
            for dest, field in zip(parts, fields, strict=True):
                dest.append(np.asarray(field, dtype=np.int32))

        c, f, t = self.chunk_len, self.future_per_chunk, self.tokens_per_frame
        for seg, (chunks, glob, clean, noised) in enumerate(layouts):
            if chunks == 0:
                if min(glob, clean) < 0 or noised != 0:
                    raise ValueError("invalid expert capture padding layout")
                append(q_parts, (
                    np.full(glob, _PAD), np.full(glob, -1), np.full(glob, -1),
                ))
                append(k_parts, (
                    np.full(clean, _PAD), np.full(clean, -1),
                    np.full(clean, -1), np.full(clean, -1),
                ))
                continue
            if chunks <= 0 or min(glob, clean, noised) < 0 or noised != chunks * c:
                raise ValueError(f"expert segment {seg} has invalid frame layout")
            offset = noised_frame_offset(noised, clean, c)
            ids = np.arange(chunks, dtype=np.int32)
            qchunk = np.concatenate((ids, np.repeat(ids, f), np.repeat(ids, c)))
            append(q_parts, (
                np.full(qchunk.size, seg), qchunk, qchunk * c + offset,
            ))
            if self.mode == "self":
                append(k_parts, (
                    np.full(qchunk.size, seg), np.full(qchunk.size, 3),
                    np.full(qchunk.size, -1), qchunk,
                ))
            else:
                frame = np.repeat(np.arange(clean + noised, dtype=np.int32), t)
                kind = np.r_[np.zeros(glob, np.int32), np.where(frame < clean, 1, 2)]
                append(k_parts, (
                    np.full(kind.size, seg), kind,
                    np.r_[np.full(glob, -1), frame],
                    np.r_[np.full(glob, -1), np.maximum(frame - clean, 0) // c],
                ))
        q = tuple(np.concatenate(parts) for parts in q_parts)
        k = tuple(np.concatenate(parts) for parts in k_parts)
        q_total, k_total = len(q[0]), len(k[0])
        if q_total > q_cap or k_total > k_cap:
            raise ValueError(f"expert mask spans {q_total} Q and {k_total} KV, ceilings {q_cap}/{k_cap}")
        if q_total < q_cap:
            q = tuple(
                np.pad(a, (0, q_cap - q_total), constant_values=_PAD if i == 0 else -1)
                for i, a in enumerate(q)
            )
        if k_total < k_cap:
            k = tuple(
                np.pad(a, (0, k_cap - k_total), constant_values=_PAD if i == 0 else -1)
                for i, a in enumerate(k)
            )
        return q, k, q_total, k_total

    @torch.compiler.disable
    def plan(self, layouts: Sequence[tuple[int, int, int, int]]) -> None:
        layouts = tuple(layouts)
        if self._planned == layouts:
            return
        if self.accelerator_graph and len(layouts) > self.max_num_segments:
            raise ValueError("expert mask exceeds captured segment ceiling")
        c, f, t = self.chunk_len, self.future_per_chunk, self.tokens_per_frame
        q_total = sum(g if k == 0 else k * (1 + f + c) for k, g, _, _ in layouts)
        k_total = sum(
            clean if k == 0 else (
                k * (1 + f + c) if self.mode == "self" else g + (clean + n) * t
            )
            for k, g, clean, n in layouts
        )
        # A captured bucket uses one token ceiling for both sides. Rectangular
        # layouts may have KV > Q, so _fields must reject either side above
        # this ceiling before touching captured buffers.
        q_cap = self.max_total_tokens if self.accelerator_graph else q_total
        k_cap = self.max_total_tokens if self.accelerator_graph else k_total
        q, k, q_total, k_total = self._fields(layouts, q_cap, k_cap)
        qr_s, qr_e, qr = _runs(q)
        kr_s, kr_e, kr = _runs(k)
        visible = _visibility(
            tuple(a[:, None] for a in qr), tuple(a[None, :] for a in kr),
            mode=self.mode, window=self.window_frames, include_noised=self.include_noised,
        )
        # A real all-masked query has undefined attention. Captured padding is
        # handled separately and never makes a real query valid.
        if visible.size and not visible[qr[0] != _PAD].any(axis=1).all():
            raise ValueError("expert mask leaves a real query with no visible key")
        nq, nk = -(-q_cap // self.block_size), -(-k_cap // self.block_size)
        qlo = np.arange(nq) * self.block_size
        klo = np.arange(nk) * self.block_size
        qhi, khi = np.minimum(qlo + self.block_size, q_cap), np.minimum(klo + self.block_size, k_cap)
        qinc = ((qr_s[None] < qhi[:, None]) & (qr_e[None] > qlo[:, None])).astype(np.float64)
        kinc = ((kr_s[None] < khi[:, None]) & (kr_e[None] > klo[:, None])).astype(np.float64)
        any_ = (qinc @ visible.astype(np.float64) @ kinc.T) > 0
        if self.full_blocks:
            hidden = qinc @ (~visible).astype(np.float64) @ kinc.T
            full = any_ & (hidden == 0) & ((qhi - qlo) == self.block_size)[:, None] & (
                (khi - klo) == self.block_size
            )[None, :]
            partial = any_ & ~full
        else:
            partial, full = any_, None
        kv_num, kv_idx = _packed(partial)
        full_num, full_idx = _packed(full) if full is not None else (None, None)
        if not self.accelerator_graph:
            self._q = tuple(self._device(a) for a in q)
            self._k = tuple(self._device(a) for a in k)
            self.block_mask = self._mask_over(
                q_cap, k_cap,
                *(self._device(a) if a is not None else None for a in (kv_num, kv_idx, full_num, full_idx)),
            )
        else:
            self._planned = None
            for dest, source in zip(self._q, q, strict=True):
                dest.copy_(self._host(source), non_blocking=True)
            for dest, source in zip(self._k, k, strict=True):
                dest.copy_(self._host(source), non_blocking=True)
            mask = self.block_mask
            for dest, source in zip(
                (mask.kv_num_blocks, mask.kv_indices, mask.full_kv_num_blocks, mask.full_kv_indices),
                (kv_num, kv_idx, full_num, full_idx), strict=True,
            ):
                if dest is not None:
                    dest.copy_(self._host(source), non_blocking=True)
        self._q_len, self._kv_len = q_cap, k_cap
        self._num_segments = len(layouts)
        self._planned = layouts

    @torch.compiler.disable
    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, return_lse: bool = False) -> torch.Tensor:
        if return_lse:
            raise ValueError("expert masked attention does not return log-sum-exp")
        if q.shape[0] != self._q_len or k.shape[0] != self._kv_len or v.shape != k.shape:
            raise ValueError("expert Q/KV tensors do not match the planned lengths")
        q, k, v = (x.to(self.q_data_type).transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        out = flex_attention_masked(
            q, k, v, block_mask=self.block_mask, scale=self.sm_scale,
            enable_gqa=self.num_qo_heads != self.num_kv_heads,
        )[0].transpose(0, 1)
        if self.accelerator_graph:
            zero = torch.zeros((), dtype=out.dtype, device=out.device)
            out = torch.where(self._q[0][:, None, None] != _PAD, out, zero)
        return out.contiguous()


class FlexExpertManager(_RaggedManagerBase, RaggedExpertAttnManager):
    """Plan one resource wrapper per paired cross label or self label."""

    wrapper_class = FlexExpertWrapper

    def __init__(
        self, device: torch.device, dtype: torch.dtype, config: RaggedAttentionConfig, key: str,
        mode: str, chunk_len: int, future_per_chunk: int, tokens_per_frame: int,
        window_frames: int, include_noised: bool, block_size: int, full_blocks: bool,
    ):
        super().__init__(device=device, dtype=dtype, config=config, key=key)
        self._kwargs.update(
            mode=mode, chunk_len=chunk_len, future_per_chunk=future_per_chunk,
            tokens_per_frame=tokens_per_frame, window_frames=window_frames,
            include_noised=include_noised, block_size=block_size, full_blocks=full_blocks,
        )

    def _plan_layouts(self, step: ExpertAttentionStep, ctx: StepContext, plan_states, shard_plan) -> None:
        if not isinstance(step, ExpertAttentionStep) or shard_plan is not None:
            raise ValueError("expert attention needs an unsplit ExpertAttentionStep")
        by_label = self._group_segments_by_label(step.segments or ())
        mode, c, f, t = (
            self._kwargs[name] for name in ("mode", "chunk_len", "future_per_chunk", "tokens_per_frame")
        )
        if mode == "cross":
            if len(step.pairs) != 1:
                raise ValueError("expert cross step needs exactly one Q/KV label pair")
            q_label, kv_label = step.pairs[0]
            if q_label == kv_label or q_label not in by_label or kv_label not in by_label:
                raise ValueError("expert cross step needs distinct planned Q/KV labels")
            qs, ks = by_label[q_label], by_label[kv_label]
            label = cross_label(q_label, kv_label)
        else:
            if step.pairs or len(by_label) != 1:
                raise ValueError("expert self step needs one label and no cross pair")
            label, qs = next(iter(by_label.items()))
            ks = qs
        if len(qs) != len(ks) or len(qs) != len(step.layouts):
            raise ValueError("expert attention needs one geometry per paired request segment")
        for q, k, (chunks, glob, clean, noised) in zip(qs, ks, step.layouts, strict=True):
            if q.request_id != k.request_id:
                raise ValueError("expert cross Q/KV segments belong to different requests")
            q_span = glob if chunks == 0 else chunks * (1 + f + c)
            kv_span = clean if chunks == 0 else (
                q_span if mode == "self" else glob + (clean + noised) * t
            )
            if q.span != q_span or k.span != kv_span:
                raise ValueError(
                    f"expert request {q.request_id!r} spans Q={q.span}, KV={k.span}; "
                    f"layout requires Q={q_span}, KV={kv_span}"
                )
        wrapper = self._wrapper_for(ctx.slot_lease, ctx.slot, label)
        wrapper.plan(step.layouts)
        plan_states[label] = wrapper

    def _exchange(self, label: str):
        return None
