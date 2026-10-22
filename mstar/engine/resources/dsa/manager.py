"""DeepSeek sparse attention (DSA) with a key per token (GLM-5.2's indexer): the resource.

The index keys live in a second KV resource (``config.index_cache``, one ``index_head_dim`` latent
per token per FULL layer) beside the MLA cache (``config.kv_cache``). Each step the submodule
stages its rows (``step_inputs``: preprocess outputs, which a CUDA graph interns and every replay
restages) and opens the forward's context from them (``context``, then ``use``), which the MLA
layers read as ``ctx``: they write their index keys and latents at the step's slots, and select
and attend through ``paged``. Under CUDA graphs the resource keeps each capture slot's
sparse-attention plans.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.dsa import sparse_mla
from mstar.engine.resources.dsa.config import DSAConfig, DSASpec
from mstar.engine.resources.dsa.paged import PagedContext
from mstar.engine.resources.step import StepContext
from mstar.utils.h2d import PinnedStager

_MAIN = "main"


class DSAManager(Resource):
    """DSA for one node's layers over its MLA cache and its index-key store."""

    @classmethod
    def build(cls, spec: DSASpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KV config, sharded once
        if spec.config.index_cache is None:
            raise ValueError(
                "DSA with a key per token stores its keys in a KV resource of their own: "
                "set DSAConfig.index_cache")
        kv_config = info.dependency(spec.config.kv_cache).config
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        index_config = info.dependency(spec.config.index_cache).config
        return cls(spec.config, kv_config, index_config, info.device)

    def __init__(self, config: DSAConfig, kv_config, index_config, device: torch.device):
        self.config = config
        self.page_size = kv_config.page_size
        self.index_page_size = index_config.page_size
        self.max_seq_len = kv_config.max_seq_len  # the serving window
        self.num_heads = kv_config.num_qo_heads  # this rank's
        self.kv_lora_rank = kv_config.kv_lora_rank
        self.qk_rope_head_dim = kv_config.qk_rope_head_dim
        self._device = device
        # the forward's context, open while its layers run (``use``)
        self.ctx: PagedContext | None = None
        self._int_stager = PinnedStager(torch.int32)
        # under CUDA graphs one sparse-attention plan per (walk, slot, rows), and per slot
        # the workspace and index buffer its plans share
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        self._sparse_workspaces: dict[int, torch.Tensor] = {}
        self._sparse_indices: dict[int, torch.Tensor] = {}

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache, self.config.index_cache}

    def step_inputs(self, step_ctx: StepContext, positions: list[int], seq_lens: list[int],
                    needs_selection: bool) -> dict[str, Any]:
        """The step's DSA state as preprocess outputs. Its tensors are top level, so a CUDA
        graph interns them and each replay stages the step's values; the host half rides in
        ``dsa_meta``. Under a graph the page tables span the serving window, the sparse plan is
        the slot's own and re-planned here every step, and every row scores on its own (a row
        within topk selects all its keys): a decode over the window, a prefill bucket over its
        own rows."""
        graph_walk = step_ctx.graph_walk
        graph = step_ctx.capture or step_ctx.slot_lease is not None
        plans = step_ctx.plan_results
        topk = self.config.topk
        lens = [p + 1 for p in positions]
        row_req = [i for i, sl in enumerate(seq_lens) for _ in range(sl)]
        width = max(lens)
        if graph and graph_walk == "prefill":
            # A bucket holds whole prompts (cg_key_info runs a cached prefix's step eager), so
            # its rows see at most its own rows of keys, and one no wider than topk selects
            # nothing. Its padding rows, position 0 of the first request, are staged: the
            # static buffers keep what a larger bucket left there.
            rows = step_ctx.slot_lease.bucket.num_tokens if step_ctx.slot_lease else len(lens)
            if width > rows:
                raise RuntimeError(
                    f"a captured prefill row sees {width} keys, past its {rows}-row bucket")
            lens += [1] * (rows - len(lens))
            row_req += [0] * (rows - len(row_req))
            width, needs_selection = rows, rows > topk
        elif graph:
            width, needs_selection = -(-self.max_seq_len // self.page_size) * self.page_size, True

        def table(key, page):
            rows = [view.page_idxs for view in plans[key][_MAIN].views]
            width = -(-self.max_seq_len // page) if graph else max(len(r) for r in rows)
            # zero-padded in C: the host work is the real pages, not rows x the window
            host = torch.zeros(len(rows), width, dtype=torch.int32)
            for i, r in enumerate(rows):
                host[i, : len(r)] = torch.as_tensor(r, dtype=torch.int32)
            return self._int_stager.to_device(host, self._device)

        starts = [sum(seq_lens[:i]) for i in range(len(seq_lens))]
        sparse_plan = None
        if graph and needs_selection:
            slot, rows = step_ctx.slot, len(lens)
            key = (graph_walk, slot, rows)
            sparse_plan = self._sparse_plans.get(key)
            if sparse_plan is None:
                if slot not in self._sparse_workspaces:
                    self._sparse_workspaces[slot] = torch.empty(
                        sparse_mla.WORKSPACE_BYTES, dtype=torch.uint8, device=self._device)
                indices = self._sparse_indices.get(slot)
                # +1: the plan's spare index entry
                if indices is None or indices.numel() < rows * topk + 1:
                    indices = self._sparse_indices[slot] = torch.zeros(
                        rows * topk + 1, dtype=torch.int32, device=self._device)
                sparse_plan = self._sparse_plans[key] = sparse_mla.SparseGraphPlan(
                    rows, topk, self._sparse_workspaces[slot], indices)
            sparse_plan.plan([min(n, topk) for n in lens], self.num_heads,
                             self.kv_lora_rank, self.qk_rope_head_dim, self.config.sm_scale)
        return {
            "dsa_row_req": self._int_stager.to_device(row_req, self._device),
            "dsa_lens": self._int_stager.to_device(lens, self._device),
            "dsa_kv_table": table(self.config.kv_cache, self.page_size),
            "dsa_index_table": table(self.config.index_cache, self.index_page_size),
            "dsa_meta": dict(
                host_lens=lens,
                spans=[(r0, sl, i) for i, (r0, sl) in enumerate(zip(starts, seq_lens,
                                                                     strict=True))],
                width=width, page_size=self.page_size, topk=topk,
                needs_selection=needs_selection,
                decode_rows=graph or graph_walk == "decode", sparse_plan=sparse_plan),
        }

    @staticmethod
    def context(inputs: dict[str, Any]) -> PagedContext | None:
        """The forward's DSA context from ``step_inputs``' outputs, or None without them."""
        meta = inputs.get("dsa_meta")
        if meta is None:
            return None
        return PagedContext(
            row_req=inputs["dsa_row_req"], lens=inputs["dsa_lens"], host_lens=meta["host_lens"],
            kv_table=inputs["dsa_kv_table"], index_table=inputs["dsa_index_table"],
            spans=meta["spans"], width=meta["width"], page_size=meta["page_size"],
            topk=meta["topk"], needs_selection=meta["needs_selection"],
            decode_rows=meta["decode_rows"], sparse_plan=meta["sparse_plan"])

    @contextmanager
    def use(self, ctx: PagedContext | None) -> Iterator[None]:
        """``ctx`` (a step's, or rows of it) is the one the layers read while the block runs."""
        before, self.ctx = self.ctx, ctx
        try:
            yield
        finally:
            self.ctx = before
