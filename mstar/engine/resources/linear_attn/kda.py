"""Kimi Delta Attention planned against a recurrent state pool.

The pool says which slot each row addresses (``RecurrentAddressing``, through
``ctx.plan_results``); this resource adds the token layout the KDA kernels want, ``cu_seqlens``
over the step's packed rows, and picks the walk: all one-token rows take fla's recurrent decode
kernel, anything else the chunked prefill (FlashKDA or fla). The kernels come from
``kda_kernels.py`` and take the pool's per-layer blocks as plain tensors, the way ``GDNManager``
hands FlashInfer its state; a model may install its own (the torch reference off-GPU) with
``set_kernels``.

Every per-step value the kernels read lives in a device buffer that is static under CUDA-graph
capture (per (bucket, slot, label), like the pool's addressing), so a captured decode step
replays with new slots and lengths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from mstar.engine.resources.base import CGSlotKey, CGSlotSpec
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.recurrent.pool import RecurrentAddressing
from mstar.engine.resources.step import Segment, StepContext

logger = logging.getLogger(__name__)


@dataclass
class KDAPlan:
    """One label's rows this step, as the kernels read them."""

    slot_ids: torch.Tensor  # int32 [rows] on device: the pool's addressing (sink for padding rows)
    has_state: torch.Tensor  # bool [rows]: False where the slot reads as zeros
    cu_seqlens: torch.Tensor  # int32 [>= rows + 1] on device: token boundaries of the rows
    cu_seqlens_cpu: list[int]
    num_rows: int  # real (unpadded) rows
    num_tokens: int
    is_decode: bool  # every row appends exactly one token
    # every row is a speculative verify block of k + 1 tokens (the checkpoint recurrence)
    is_verify: bool = False
    _cpu: dict = field(default_factory=dict, repr=False)

    # host copies for reference kernels that loop over rows (a sync on CUDA; the fla kernels never ask)
    @property
    def slot_ids_cpu(self) -> list[int]:
        if "slots" not in self._cpu:
            self._cpu["slots"] = self.slot_ids[: self.num_rows].tolist()
        return self._cpu["slots"]

    @property
    def has_state_cpu(self) -> list[bool]:
        if "has" not in self._cpu:
            self._cpu["has"] = self.has_state[: self.num_rows].tolist()
        return self._cpu["has"]


class KDAManager(LinearAttnManager):
    def __init__(
        self,
        config: LinearAttnConfig,
        geometry: DeltaNetGeometry,
        num_layers: int,
        device: torch.device,
        kernels=None,
        speculative_tokens: int = 0,
    ):
        from mstar.engine.resources.linear_attn.kda_kernels import default_kernels

        self.config = config
        self.geometry = geometry
        self.num_layers = num_layers
        self._device = device
        self._pool_key = config.recurrent_state
        # k > 0: rows of exactly k + 1 tokens are verify blocks (the pool carries the spec blocks)
        self.speculative_tokens = int(speculative_tokens)
        # FlashKDA is specialised for head_dim 128 with a bounded gate; other shapes use fla
        fits = geometry.head_k_dim == 128 and geometry.head_v_dim == 128 and config.gate_lower_bound is not None
        self.kernels = kernels if kernels is not None else default_kernels(device, config.backend.value, fits)

        self._cg_max_bs = 0
        self._cg_cu: dict[CGSlotKey, torch.Tensor] = {}
        self._eager_cu: dict[str, torch.Tensor] = {}
        self._current: dict[str, KDAPlan] = {}
        # pre-planning: nothing here mutates live state, so staging is caching the result
        self._preplanned = False
        self._cached_plan_output: dict[str, KDAPlan] | None = None

    def set_kernels(self, kernels) -> None:
        """Install the kernel bundle (``run_paged(qkv, g_raw, beta_raw, plan, conv, state, params)``)."""
        self.kernels = kernels

    @property
    def cuda_graph_safe(self) -> bool:
        """Whether the decode walk may be captured: every kernel address comes from tensors."""
        return bool(getattr(self.kernels, "cuda_graph_safe", False))

    @property
    def default_layer_idx(self) -> int | None:
        return self._default_layer_idx

    def depends_on(self) -> set[str]:
        # so the pool plans first and its addressing reaches us through `ctx.plan_results`
        return {self._pool_key}

    # Step lifecycle

    @property
    def supports_preplan(self):
        return True

    def plan(self, step: LinearAttnStep, ctx: StepContext):
        assert not (self._preplanned and ctx.is_preplan), (
            "kda preplan is already pending; clear_preplan before planning a different step ahead"
        )
        self.reset_default_cursors()
        if self._preplanned:
            self._current = self._cached_plan_output
            self.clear_preplan()
            return self._current
        addressing: dict[str, RecurrentAddressing] = ctx.plan_results[self._pool_key]
        self._current = {}
        for label, segments in self._group_by_label(step.segments or ()).items():
            self._current[label] = self._build_plan(label, segments, addressing[label], ctx)
        if ctx.is_preplan:
            self._preplanned = True
            self._cached_plan_output = self._current
        return self._current

    def clear_preplan(self):
        self._preplanned = False
        self._cached_plan_output = None

    @staticmethod
    def _group_by_label(segments) -> dict[str, list[Segment]]:
        out: dict[str, list[Segment]] = {}
        for seg in segments:
            out.setdefault(seg.label, []).append(seg)
        return out

    def _cu_buffer(self, label: str, ctx: StepContext, rows: int) -> torch.Tensor:
        """The ``cu_seqlens`` buffer this step fills: static per (bucket, slot, label) under capture,
        sized to the widest batch any runner announced; grown on demand when eager."""
        lease = ctx.slot_lease
        if lease is None:
            buf = self._eager_cu.get(label)
            if buf is None or buf.numel() < rows + 1:
                buf = self._eager_cu[label] = torch.zeros(rows + 1, dtype=torch.int32, device=self._device)
            return buf
        key = CGSlotKey(bucket=lease.bucket, slot=lease.slot, label=label)
        buf = self._cg_cu.get(key)
        if buf is None:
            size = max(self._cg_max_bs, lease.bucket.bs, rows) + 1
            buf = self._cg_cu[key] = torch.zeros(size, dtype=torch.int32, device=self._device)
        assert buf.numel() >= rows + 1, (
            f"kda cu_seqlens for {key} holds {buf.numel()} entries, step needs {rows + 1}")
        return buf

    def _build_plan(
        self, label: str, segments: list[Segment], addressing: RecurrentAddressing, ctx: StepContext,
    ) -> KDAPlan:
        spans = [max(int(seg.span), 0) for seg in segments]
        rows = len(spans)
        cu = [0]
        for s in spans:
            cu.append(cu[-1] + s)
        buf = self._cu_buffer(label, ctx, rows)
        # built on the host and copied in: the values come from Python bookkeeping
        host = torch.tensor(cu, dtype=torch.int32, pin_memory=torch.cuda.is_available())
        buf[: rows + 1].copy_(host, non_blocking=True)
        k1 = self.speculative_tokens + 1
        return KDAPlan(
            slot_ids=addressing.slot_indices[:rows], has_state=addressing.has_state[:rows], cu_seqlens=buf,
            cu_seqlens_cpu=cu, num_rows=rows, num_tokens=cu[-1], is_decode=rows > 0 and all(s == 1 for s in spans),
            is_verify=self.speculative_tokens > 0 and rows > 0 and all(s == k1 for s in spans),
        )

    def current_plan(self, label: str | None = None) -> KDAPlan:
        if label is None:
            label = self._default_label
        found = self._current.get(label)
        if found is None:
            raise KeyError(
                f"kda has no plan for label {label!r}; this step planned {sorted(self._current)}. Every label a "
                "forward runs must carry a segment in the step declaration."
            )
        return found

    # Submodule-level functionality

    @torch.compiler.disable
    def run(
        self, qkv: torch.Tensor, g_raw: torch.Tensor, beta_raw: torch.Tensor,
        conv_layer: torch.Tensor, state_layer: torch.Tensor, params, label: str | None = None, spec=None,
    ) -> torch.Tensor:
        """One layer's KDA over this step's packed tokens: ``qkv [T, 3P]`` (pre-conv), ``g_raw [T, H, D]``,
        ``beta_raw [T, H]``; ``conv_layer`` / ``state_layer`` are the pool's ``block(name, layer)`` views,
        updated in place. Returns ``o [T, H, D]``."""
        assert self.kernels is not None, "no KDA kernels: on CUDA fla/FlashKDA are required, off-GPU call set_kernels"
        plan = self.current_plan(label)
        if plan.is_verify:
            assert spec is not None, "a verify step needs the layer's speculative blocks (SpecBlocks)"
            return self.kernels.run_verify(qkv, g_raw, beta_raw, plan, conv_layer, state_layer, spec, params)
        return self.kernels.run_paged(qkv, g_raw, beta_raw, plan, conv_layer, state_layer, params)

    @torch.compiler.disable
    def set_prefix_len(self, spec_len: torch.Tensor, accepted: torch.Tensor, label: str | None = None) -> None:
        """After the verification: the rows' blocks become their pending prefixes, ``accepted + 1``
        tokens long (the bonus token is always kept). ``spec_len`` is the shared ``[slots, 1]`` int32
        block; padding rows write the sink. Tensor ops only."""
        plan = self.current_plan(label)
        rows = plan.num_rows
        values = (accepted[:rows].to(torch.int32) + 1).unsqueeze(1)
        spec_len.index_copy_(0, plan.slot_ids[:rows].to(torch.long), values)

    # Engine lifecycle

    def build_cuda_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int) -> None:
        del slots, max_seq_len
        self._cg_max_bs = max(self._cg_max_bs, max_bs)

    def cleanup(self):
        self._cg_cu.clear()
        self._eager_cu.clear()
        self._current.clear()


__all__ = ["KDAManager", "KDAPlan"]
