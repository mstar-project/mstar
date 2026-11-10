"""DeepSeek sparse attention (DSA) over a node's paged MLA cache: the resource.

The index keys live on the MLA cache's pages (aux planes ``INDEX_PLANES``), so they share its
page table. A step declares ``DSAStep()``; ``plan`` builds each label's rows from the KV plan
(a ``paged.DSAContext``), into buffers of the (bucket, slot) under a capture lease, which the
graph reads and every plan refills. The MLA layers write keys, select and attend through the
resource (``write_index``, ``select``, ``attend``), which reads the current plan.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch

from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo, Resource
from mstar.engine.resources.dsa import paged, sparse_mla
from mstar.engine.resources.dsa.config import INDEX_PLANES, DSAConfig, DSASpec, DSAStep
from mstar.engine.resources.dsa.paged import DSAContext, attn_lens, attn_lens_device
from mstar.engine.resources.step import BucketKey, StepContext
from mstar.utils.h2d import PinnedStager

_MAIN = "main"


class DSAManager(Resource):
    """DSA for one node's MLA layers over their cache (``config.kv_cache``)."""

    @classmethod
    def build(cls, spec: DSASpec, info: EngineResourceInfo):
        # `dependency` hands back the SPEC; its `.config` is the KV config, sharded once
        kv_config = info.dependency(spec.config.kv_cache).config
        planes = kv_config.aux_planes.get(INDEX_PLANES)
        if planes is None:
            raise ValueError(
                f"DSA keeps its index keys on {spec.config.kv_cache}'s pages: declare its "
                f"aux_planes[{INDEX_PLANES!r}]")
        d = spec.config.index_head_dim
        width = None if d is None else (d if spec.config.kpool == 1 else 3 * d)
        if planes.width != width:
            # a row is the key, or with kpool > 1 [key | gate | key of the pool ending there]
            raise ValueError(f"DSA's {INDEX_PLANES} rows are {width} wide (index_head_dim "
                             f"{d}, kpool {spec.config.kpool}); {spec.config.kv_cache} "
                             f"declares {planes.width}")
        if info.joint_comm_group is not None:
            kv_config.shard(info.joint_comm_group.world_size)
        return cls(spec.config, kv_config, info.device)

    def __init__(self, config: DSAConfig, kv_config, device: torch.device):
        self.config = config
        self.page_size = kv_config.tokens_per_page
        self.kv_rows = kv_config.max_seq_len  # cache rows a request may hold
        self.num_heads = kv_config.num_qo_heads  # this rank's
        self.kv_lora_rank = kv_config.kv_lora_rank
        self.kpe = kv_config.qk_rope_head_dim
        self._device = device
        self._kv = None  # the KV resource whose pages it reads (`bind_dependencies`)
        # label -> this step's plan, which the layers read (None: they attend densely)
        self._current: dict[str, DSAContext | None] = {}
        # pre-planning: a plan made a step ahead, which the next `plan` promotes
        self._preplanned: dict[str, DSAContext] | None = None
        self._int_stager = PinnedStager(torch.int32)
        # (bucket, slot) -> the int32 rows a captured graph reads: pos, row_req, page_start, pages
        self._cg_rows: dict[tuple[BucketKey, int], torch.Tensor] = {}
        # under CUDA graphs one sparse-attention plan per (walk, capture slot, rows), and per
        # (walk, capture slot) the workspace and index buffer its plans share
        self._sparse_plans: dict[tuple[str, int, int], sparse_mla.SparseGraphPlan] = {}
        self._sparse_buffers: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        # an eager step's sparse plans, reused (one workspace)
        self._eager = sparse_mla.EagerSparsePlans(device) if device.type == "cuda" else None

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    def bind_dependencies(self, resources: Mapping[str, Resource]) -> None:
        self._kv = resources[self.config.kv_cache]

    @property
    def force_double_buffer(self) -> bool:
        # a captured slot's sparse plan is re-planned in `plan` through FlashInfer's pinned
        # staging, which the replay before it may still be reading
        return True

    @property
    def supports_preplan(self) -> bool:
        return True

    def clear_preplan(self) -> None:
        self._preplanned = None

    # -- capture buffers ---------------------------------------------------------------------

    def _rows_layout(self, bucket: BucketKey) -> tuple[int, int, int]:
        """A captured bucket's rows, requests and page entries: every request at the window."""
        return bucket.num_tokens, bucket.bs, bucket.bs * -(-self.kv_rows // self.page_size)

    def _cg_buffer(self, bucket: BucketKey, slot: int) -> torch.Tensor:
        key = (bucket, slot)
        buf = self._cg_rows.get(key)
        if buf is None:
            rows, reqs, pages = self._rows_layout(bucket)
            buf = self._cg_rows[key] = torch.zeros(
                2 * rows + reqs + pages, dtype=torch.int32, device=self._device)
        return buf

    def build_accelerator_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int,
                                        max_seq_len: int) -> None:
        # a captured step plans its bucket's rows as one sparse plan, which FlashInfer takes up
        # to sparse_mla.MAX_ROWS; refused here whichever ladder, the model's or the serving
        # config's, the buckets came from
        past = sorted({spec.num_tokens for spec in slots if spec.num_tokens > sparse_mla.MAX_ROWS})
        if past:
            raise ValueError(f"DSA: captured buckets of {past} tokens pass one sparse plan's "
                             f"{sparse_mla.MAX_ROWS} rows; cap the node's prefill ladder there")
        for spec in slots:
            self._cg_buffer(spec.bucket, spec.slot)

    # -- the step ----------------------------------------------------------------------------

    def plan(self, step: DSAStep, ctx: StepContext) -> dict[str, DSAContext]:
        """Each label's rows from the KV plan (``ctx.plan_results``), the plan the layers read
        until the next. A pre-plan is kept and the next call promotes it."""
        if self._preplanned is not None and not ctx.is_preplan:
            self._current, self._preplanned = dict(self._preplanned), None
            return self._current
        plans = {label: self._plan_label(ctx, kv_out, step.mtp_trims)
                 for label, kv_out in ctx.plan_results[self.config.kv_cache].items()}
        if ctx.is_preplan:
            self._preplanned = plans
        else:
            self._current = dict(plans)
        return plans

    def _plan_label(self, step_ctx: StepContext, kv_out, mtp_trims: bool = False) -> DSAContext:
        """One label's rows: per row its position and request, per request its first entry in
        the flat page list. Under a lease every row scores on its own, the score width spans
        the serving window (a prefill bucket's own rows) and the slot's sparse plan is
        re-planned here; ``mtp_trims`` (``DSAStep``) plans it for each row's longest."""
        lease = step_ctx.slot_lease
        graph = lease is not None and lease.bucket is not None
        walk = step_ctx.graph_walk
        topk, kpool = self.config.topk, self.config.kpool
        pos: list[int] = []
        row_req: list[int] = []
        spans: list[tuple[int, int, int]] = []
        for i, view in enumerate(kv_out.views):
            spans.append((len(pos), view.to_compute, i))
            pos.extend(range(view.length - view.to_compute, view.length))
            row_req.extend([i] * view.to_compute)
        n_real = rows = len(pos)
        indptr = kv_out.cpu_indptrs.paged_kv_indptr
        n_req = reqs = indptr.numel() - 1
        page_list = kv_out.cpu_indptrs.paged_kv_indices.to(torch.int32)
        if graph:
            rows, reqs, page_cap = self._rows_layout(lease.bucket)
            if n_real > rows or n_req > reqs or page_list.numel() > page_cap:
                raise RuntimeError(f"dsa: {n_real} rows of {n_req} requests ({page_list.numel()} "
                                   f"pages) overflow the {lease.bucket} capture's buffers")
            # a packed bucket's padding rows: position 0 of the first request
            pos += [0] * (rows - n_real)
            row_req += [0] * (rows - n_real)
        host = torch.cat([torch.tensor(pos + row_req, dtype=torch.int32),
                          indptr[:-1].to(torch.int32),
                          torch.zeros(reqs - n_req, dtype=torch.int32), page_list]).numpy()
        if graph:
            staged = self._cg_buffer(lease.bucket, lease.slot)
        else:
            staged = torch.empty(host.size, dtype=torch.int32, device=self._device)
        self._int_stager.copy_(staged, host)
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
            if (needs_selection or not self.config.dense_below_topk) and self._device.type == "cuda":
                lens = attn_lens(pos, topk, kpool)  # a padding row's is 1
                if mtp_trims:
                    # a row sits at or past its true position: plan it for the longest it can
                    # be, its spare entries masked (SparseGraphPlan.attend's valid)
                    lens = [min(p + 1, topk + kpool - 1) for p in pos]
                sparse_plan = self._graph_plan(walk, step_ctx.slot, rows, lens)
        return DSAContext(
            pos=staged[:rows], row_req=staged[rows:2 * rows],
            page_start=staged[2 * rows:2 * rows + n_req], pages=staged[2 * rows + reqs:],
            host_pos=None if graph else pos[:n_real], spans=None if graph else spans,
            host_page_start=None if graph else indptr[:-1].tolist(),
            max_pools=max(max_pools, 1), page_size=self.page_size, topk=topk, kpool=kpool,
            needs_selection=needs_selection, decode_rows=graph or walk == "decode",
            sparse_plan=sparse_plan, eager=self._eager, captured=graph)

    def _graph_plan(self, walk: str, slot: int, rows: int,
                    lens: list[int]) -> sparse_mla.SparseGraphPlan:
        """The capture slot's sparse plan for ``rows`` rows, re-planned with ``lens``."""
        width = self.config.topk + self.config.kpool - 1
        key = (walk, slot, rows)
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
        sparse_plan.plan(lens, self.num_heads, self.kv_lora_rank, self.kpe, self.config.sm_scale)
        return sparse_plan

    # -- what the layers and the submodule call ----------------------------------------------

    def current(self, label: str | None = None) -> DSAContext | None:
        """This step's plan for ``label``, or None inside ``dense``."""
        label = label or _MAIN
        if label not in self._current:
            raise KeyError(f"dsa: no plan for label {label!r}; the step declares DSAStep()")
        return self._current[label]

    @contextmanager
    def _narrowed(self, ctx: DSAContext | None, label: str | None) -> Iterator[None]:
        label = label or _MAIN
        full = self.current(label)
        self._current[label] = ctx
        try:
            yield
        finally:
            self._current[label] = full

    def rows(self, c0: int, c1: int, label: str | None = None):
        """The step's rows ``[c0, c1)`` alone (``DSAContext.rows``), for a pass over them after
        the rows before them have written the cache (``KVManager.token_window``)."""
        return self._narrowed(self.current(label).rows(c0, c1), label)

    def pick(self, rows: list[int], label: str | None = None):
        """The step's given rows alone (``DSAContext.pick``), under ``KVManager.token_rows``."""
        return self._narrowed(self.current(label).pick(rows), label)

    def dense(self, label: str | None = None):
        """No plan: the layers attend densely (a pass DSA doesn't cover)."""
        return self._narrowed(None, label)

    def _need_kv(self):
        if self._kv is None:
            raise RuntimeError(f"dsa: bind_dependencies never handed it {self.config.kv_cache}")
        return self._kv

    def move_rows(self, pos: torch.Tensor, pad_kpe: float, label: str | None = None) -> None:
        """Move this step's rows to the device positions ``pos``, each at or before the one the
        host planned (an MTP verify block placed after the last step's accepted rows, not its
        whole block): the cache slots its latents and index keys go to, and the positions it
        selects and attends from. A captured plan stays as made (``DSAStep(mtp_trims=True)``):
        each row's entries past its own length attend the cache's spare row, whose rope slot
        gets ``pad_kpe`` for the caller's queries to mask. An eager plan takes exact rows, from
        one host read of ``pos``."""
        ctx, kv = self.current(label), self._need_kv()
        size = ctx.page_size
        pages, offsets = kv.write_slots(label)
        page = ctx.pages[(ctx.page_start[ctx.row_req.long()] + pos // size).long()]
        pages.copy_(page.to(pages.dtype))
        offsets.copy_((pos % size).to(offsets.dtype))
        ctx.pos = pos.to(ctx.pos.dtype)
        if not ctx.captured:
            ctx.host_pos = pos.tolist()
            return
        if not self.kpe:
            raise ValueError("a captured step's spare entries are masked in the rope slot, "
                             "and this cache has none")
        slot, row = kv.spare_row()
        row[:, self.kv_lora_rank:].fill_(pad_kpe)
        ctx.valid = attn_lens_device(pos, ctx.topk, ctx.kpool).to(torch.int32)
        ctx.pad_slot = slot

    def write_index(self, plane: int, k: torch.Tensor, gate: torch.Tensor | None = None,
                    ape: torch.Tensor | None = None, label: str | None = None) -> None:
        """Write the step's index keys (and with kpool > 1 their gates and the keys of the pools
        they complete) to ``plane`` at the step's slots."""
        paged.write_index(self._need_kv(), plane, label, k, self.current(label), gate=gate,
                          ape=ape)

    def select(self, plane: int, q: torch.Tensor, w: torch.Tensor, group=None,
               label: str | None = None) -> torch.Tensor:
        """``[rows, width]`` cache slots each row attends, scored over ``plane``'s keys."""
        return paged.select(q, w, self._need_kv().aux_view(INDEX_PLANES, plane),
                            self.current(label), group)

    def attend(self, plane: int, q_nope: torch.Tensor, q_pe: torch.Tensor, slots: torch.Tensor,
               sm_scale: float, label: str | None = None) -> torch.Tensor:
        """Absorbed MLA of each row over its ``slots`` of ``plane``'s latents."""
        return paged.sparse_attend(q_nope, q_pe, self._need_kv().layer_view(plane), slots,
                                   self.current(label), sm_scale)
