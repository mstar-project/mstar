"""DeepSeek sparse attention (DSA) over a node's paged MLA cache: the resource.

The index keys live on the MLA cache's pages (``paged``: its aux planes ``INDEX_PLANES``). Each
step the submodule stages its rows (``step_inputs``: preprocess outputs, which a CUDA graph
interns and every replay restages) and opens the forward's context from them (``context``,
then ``use``), which the MLA layers read as ``ctx``: they write their index keys and latents
at the step's slots, and select and attend through ``paged``. Under CUDA graphs the resource
keeps each capture slot's sparse-attention plans.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.dsa import sparse_mla
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec
from mstar.engine.resources.dsa.paged import DSAContext, attn_lens
from mstar.engine.resources.step import StepContext
from mstar.utils.h2d import PinnedStager


class DSAManager(Resource):
    """DSA for one node's MLA layers over their cache (``config.kv_cache``)."""

    @classmethod
    def build(cls, spec: DSASpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KV config, sharded once
        kv_config = info.dependency(spec.config.kv_cache).config
        if INDEX_PLANES not in kv_config.aux_planes:
            raise ValueError(
                f"DSA keeps its index keys on {spec.config.kv_cache}'s pages: declare its "
                f"aux_planes[{INDEX_PLANES!r}]")
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        return cls(spec.config, kv_config, info.device)

    def __init__(self, config: DSAConfig, kv_config, device: torch.device):
        self.config = config
        self.page_size = kv_config.page_size
        self.kv_rows = kv_config.max_seq_len  # cache rows a request may hold
        self.num_heads = kv_config.num_qo_heads  # this rank's
        self.kv_lora_rank = kv_config.kv_lora_rank
        self.kpe = kv_config.qk_rope_head_dim
        self._device = device
        # the forward's context, open while its layers run (``use``)
        self.ctx: DSAContext | None = None
        self._int_stager = PinnedStager(torch.int32)
        # under CUDA graphs one sparse-attention plan per (walk, capture slot, rows), and per
        # (walk, capture slot) the workspace and index buffer its plans share
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        self._sparse_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    def step_inputs(self, step_ctx: StepContext, captured: bool,
                    label: str = "main") -> dict[str, Any]:
        """The step's DSA state from the KV plan, as preprocess outputs. The tensors are top
        level, so a CUDA graph interns them and each replay stages the step's values: per row
        its position and request, per request its first entry in the flat page list. Under a
        graph every row scores on its own, the score width spans the serving window (a
        prefill bucket's own rows) and the sparse plan is the slot's own, re-planned here every
        step; the host half rides in ``dsa_meta``."""
        graph = bool(captured)
        walk = step_ctx.graph_walk
        kv_out = step_ctx.plan_results[self.config.kv_cache][label]
        topk, kpool = self.config.topk, self.config.kpool
        pos: list[int] = []
        row_req: list[int] = []
        spans: list[tuple[int, int, int]] = []
        for i, view in enumerate(kv_out.views):
            spans.append((len(pos), view.to_compute, i))
            pos.extend(range(view.length - view.to_compute, view.length))
            row_req.extend([i] * view.to_compute)
        n_real = len(pos)
        rows = n_real
        if graph and step_ctx.slot_lease is not None:
            # a packed prefill bucket's padding rows: position 0 of the first request. Staged, so
            # they don't keep what a larger bucket left in the shared static buffers
            rows = step_ctx.slot_lease.bucket.num_tokens
            pos += [0] * (rows - n_real)
            row_req += [0] * (rows - n_real)
        indptr = kv_out.cpu_indptrs.paged_kv_indptr
        parts = [torch.tensor(pos + row_req, dtype=torch.int32), indptr[:-1].to(torch.int32),
                 kv_out.cpu_indptrs.paged_kv_indices.to(torch.int32)]
        if step_ctx.capture:
            # the static buffer's size: every row's request at the serving window (a replay
            # stages just its own pages)
            pad = len(kv_out.views) * -(-self.kv_rows // self.page_size) - parts[-1].numel()
            parts.append(torch.zeros(pad, dtype=torch.int32))
        n_pos, n_req = len(pos), indptr.numel() - 1
        staged = self._int_stager.to_device(torch.cat(parts), self._device)
        lens = attn_lens(pos, topk, kpool)  # a padding row's is 1
        # score width: the step's most pools; a decode graph's the serving window's, and a
        # captured prefill holds whole prompts no longer than its bucket
        max_pools = (max(pos) + 1) // kpool if pos else 1
        needs_selection = bool(pos) and max(pos) + 1 > topk
        sparse_plan = None
        if graph:
            if walk == "prefill":
                if pos and max(pos) >= rows:
                    raise RuntimeError(f"a captured prefill row sits at {max(pos)}, past its "
                                       f"{rows}-token bucket")
                max_pools, needs_selection = rows // kpool, rows > topk
            else:
                max_pools, needs_selection = self.kv_rows // kpool, True
        sparse = needs_selection or not self.config.dense_below_topk
        if graph and sparse and self._device.type == "cuda":
            width = topk + kpool - 1
            key = (walk, step_ctx.slot, rows)
            sparse_plan = self._sparse_plans.get(key)
            if sparse_plan is None:
                buffers = self._sparse_buffers.get(key[:2])
                if buffers is None or buffers[1].numel() < rows * width + 1:
                    # captures run largest first, so the first is the slot's size; +1: the
                    # plan's spare index entry
                    buffers = self._sparse_buffers[key[:2]] = (
                        torch.empty(sparse_mla.WORKSPACE_BYTES, dtype=torch.uint8,
                                    device=self._device),
                        torch.zeros(rows * width + 1, dtype=torch.int32, device=self._device))
                sparse_plan = self._sparse_plans[key] = sparse_mla.SparseGraphPlan(
                    rows, width, buffers[0], buffers[1])
            sparse_plan.plan(lens, self.num_heads, self.kv_lora_rank, self.kpe,
                             self.config.sm_scale)
        return {
            "dsa_pos": staged[:n_pos],
            "dsa_row_req": staged[n_pos:2 * n_pos],
            "dsa_page_start": staged[2 * n_pos:2 * n_pos + n_req],
            "dsa_pages": staged[2 * n_pos + n_req:],
            "dsa_meta": dict(
                host_pos=None if graph else pos[:n_real],
                spans=None if graph else spans,
                host_page_start=None if graph else indptr[:-1].tolist(),
                max_pools=max(max_pools, 1), page_size=self.page_size,
                needs_selection=needs_selection, decode_rows=graph or walk == "decode",
                sparse_plan=sparse_plan,
            ),
        }

    def context(self, inputs: dict[str, Any]) -> DSAContext | None:
        """The forward's context from ``step_inputs``' outputs, or None without them."""
        meta = inputs.get("dsa_meta")
        if meta is None:
            return None
        return DSAContext(
            pos=inputs["dsa_pos"], row_req=inputs["dsa_row_req"], pages=inputs["dsa_pages"],
            page_start=inputs["dsa_page_start"], host_pos=meta["host_pos"],
            spans=meta["spans"], host_page_start=meta["host_page_start"],
            max_pools=meta["max_pools"], page_size=meta["page_size"],
            topk=self.config.topk, kpool=self.config.kpool,
            needs_selection=meta["needs_selection"], decode_rows=meta["decode_rows"],
            sparse_plan=meta["sparse_plan"])

    @contextmanager
    def use(self, ctx: DSAContext | None) -> Iterator[None]:
        """``ctx`` (a step's, or rows of it) is the one the layers read while the block runs."""
        before, self.ctx = self.ctx, ctx
        try:
            yield
        finally:
            self.ctx = before
