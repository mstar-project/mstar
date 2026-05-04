"""CUDA Graph capture and replay for AR decode and EncDec engines.

Option A keying: separate CUDA graph captures per (graph_walk, requires_cfg, batch_size).
- decode + no_cfg: 1 LLM forward pass (main label only)
- decode + cfg: 2 LLM forward passes (main + cfg_img)

Key requirements for CUDA graph compatibility:
- FlashInfer wrappers must be PERSISTENT (same Python object during capture and replay)
- Static buffers updated via .copy_(), not reassignment
- No dynamic memory allocation inside captured region
- No Python control flow that changes between replays
- advance_seq_lens() is Python-only — called AFTER graph.replay(), not inside

Also provides EncDecCudaGraphWrapper for stateless encoder/decoder submodules
and PiecewiseCudaGraphRunner for capturing transformer block loops (VJepa2 predictors).
"""

import bisect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
from mminf.engine.cuda_graph_config import (
    BasicBatchedCudaGraphConfig,
    CudaGraphConfig,
    CudaGraphConfigType,
    FlashInferPackedCudaGraphConfig,
)
from mminf.engine.kv_store import KVCacheConfig, PagedAllocationManager
from mminf.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine, NodeSubmodule
from mminf.utils.profiler import mark, range_pop, range_push
from mminf.utils.sampling import SamplerBuffers, Sampler, SamplingConfig, make_sampler_from_buffers

logger = logging.getLogger(__name__)


DEFAULT_AR_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class DummyCaptureInput:
    tensors: dict[str, list[torch.Tensor]]  # {tensor_name: [tensor(s)]}
    seq_len: int | None = field(default=None)


@dataclass
class CudaGraphSlot:
    """One captured graph + its private FlashInfer wrappers.

    Phase 3 double-buffer: each (graph_walk, requires_cfg, bs, num_tokens)
    key holds two of these in ``CudaGraphData.slots``. Replay alternates
    between slots so plan(N+1) on the inactive slot's wrapper can run
    concurrently with replay(N) on the active slot.
    """
    graph: torch.cuda.CUDAGraph
    static_inputs: dict
    static_outputs: dict
    static_cache_manager: BatchedCacheManager
    # Cached at capture time: True iff any dummy_rid in static_outputs has a
    # key besides "logits". When False, sample_and_remap can skip the
    # per-rid collection loop entirely.
    has_non_logit_outputs: bool = False

@dataclass
class PiecewiseGraphData:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor                          # [bs, seq_len, embed_dim]
    static_out: torch.Tensor                        # same shape — graph output
    static_pos_bufs: dict[str, torch.Tensor]        # updated via .copy_() before each replay
    static_cache_manager: BatchedCacheManager | None
    dummy_rids: list[str]
    bs: int

@dataclass
class CudaGraphData:
    config: CudaGraphConfig
    bs: int
    # One CudaGraphSlot per double-buffer slot. Always length NUM_SLOTS.
    # Each slot has its own captured graph + persistent FlashInfer wrappers
    # so plan(N+1) on slot[(s+1)%2] runs concurrent with replay(N) on slot[s].
    slots: list[CudaGraphSlot] = field(default_factory=list)
    # Index of the slot the NEXT submission (replay or pre-plan) will use.
    # Main thread reads-and-flips via ``CudaGraphRunner.reserve_slot`` so the
    # GPU thread (replay) and plan_executor thread (pre-plan) always agree on
    # which slot a given iter targets.
    next_slot: int = 0


@dataclass(frozen=True)
class CudaGraphKey:
    graph_walk: str
    requires_cfg: bool
    bs: int
    num_tokens: int


class CudaGraphRunner:
    """Captures and replays CUDA graphs for AR decode batches.

    Option A: separate graphs per (graph_walk, requires_cfg, batch_size).

    Warmup flow:
    1. For each config (decode+no_cfg, decode+cfg):
       a. For each batch_size (largest first for memory reuse):
          - Create persistent FlashInfer wrappers per label
          - Create static BatchedCacheManager with cuda_graph_plan_states
          - Create static input buffers
          - Warmup: 2 forward passes
          - Capture the graph

    Runtime flow:
    1. Look up graph by (graph_walk, requires_cfg, padded_batch_size)
    2. Re-plan persistent wrappers with real page tables (outside graph)
    3. Copy real input embeddings to static buffers
    4. graph.replay()
    5. advance_seq_lens on real request states (Python-only, post-replay)
    6. Clone and remap outputs from dummy to real request IDs
    """

    CAPTURE_BATCH_SIZES = DEFAULT_AR_CAPTURE_BATCH_SIZES
    # Phase 3 double-buffer: capture two graphs + two FlashInfer wrapper sets
    # per (graph_walk, requires_cfg, bs, num_tokens) key. Replay alternates so
    # plan(N+1) on the inactive slot can run concurrent with replay(N) on the
    # active slot. Override via MMINF_NUM_SLOTS=1 to disable double-buffer
    # (e.g., for models where the second capture in the same memory pool
    # races with torch.compile's autotune state — verify capture succeeds
    # before claiming the perf win).
    NUM_SLOTS = int(os.environ.get("MMINF_NUM_SLOTS", "2"))

    def __init__(
        self,
        submodule_name: str,
        submodule: ARNodeSubmodule,
        kv_cache_config: KVCacheConfig,
        alloc_manager: PagedAllocationManager,
        sampler: Sampler,
        buffer_manager: WorkspaceBufferManager,
        device: torch.device,
        autocast_dtype: torch.dtype
    ):
        self.submodule_name = submodule_name
        self.submodule = submodule
        self.capture_configs: list[CudaGraphConfig] = submodule.get_cuda_graph_configs(device)
        self.kv_cache_config = kv_cache_config
        self.alloc_manager = alloc_manager
        self.sampler = sampler
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.buffer_manager = buffer_manager
        self.enable_nvtx = False  # set by AREngine after construction

        self.graphs: dict[CudaGraphKey, CudaGraphData] = {}

        self.memory_pool = None

        # (config_idx, tensor_key) → max-bucket static buffer. Lazily populated
        # by _intern_static_buffer on the first capture to touch each key, which
        # — given warmup_and_capture's largest-first iteration — IS the max
        # bucket. Smaller-bucket captures slice the leading dim of the same
        # buffer, so all captures for a (config, key) pair share one allocation
        # instead of cloning per (bs, num_tokens). See vox-serve's
        # _initialize_prefill_cuda_graphs (cuda_graph_worker.py:228-373) for
        # the canonical pattern; the cuda_graph memory_pool + largest-first
        # capture order keep the slice views' addresses stable across replays.
        self.shared_static_buffers: dict[tuple[int, str], torch.Tensor] = {}
        # Sum of bytes that WOULD have been allocated by per-capture clones
        # (one full tensor per call) — incremented on every _intern_static_buffer
        # call. Compared against the actual shared-buffer footprint at the end
        # of warmup_and_capture to surface the Step 5 savings deterministically
        # (the actual torch.cuda.memory_allocated delta also covers KV cache /
        # model weights / FlashInfer workspaces, so it's noisier).
        self._capture_clone_bytes_naive = 0
        # Phase 3 plan-overlap stream. Lazily created the first time pre_plan
        # is called from Worker.plan_executor.
        self._plan_stream: "torch.cuda.Stream | None" = None

        self.max_bs = max(
            [max(config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES)
            for config in self.capture_configs] or [1]
        )
        self.sampler_buffer: SamplerBuffers = SamplerBuffers.allocate(
            max_batch_size=self.max_bs, device=device
        )

    def warmup_and_capture(self) -> None:
        """Capture graphs for all configs and batch sizes."""
        if self.device is None or not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph capture for %s",
                           self.submodule_name)
            return

        if not hasattr(self.submodule, 'forward_batched'):
            logger.info("Submodule %s does not support batched forward, "
                        "skipping CUDA graph capture", self.submodule_name)
            return

        self.memory_pool = torch.cuda.graphs.graph_pool_handle()
        mem_before = torch.cuda.memory_allocated(self.device)

        for config in self.capture_configs:
            sizes = config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES
            for bs in reversed(sizes):
                for num_tokens in reversed(sorted(config.get_total_tokens(bs))):
                    key = CudaGraphKey(
                        graph_walk=config.capture_graph_walk,
                        requires_cfg=config.requires_cfg,
                        bs=bs, num_tokens=num_tokens
                    )
                    try:
                        cfg_type = config.get_config_type()
                        if cfg_type == CudaGraphConfigType.BASIC_BATCHED:
                            self._capture_one_basic_batched(
                                key, config, self.submodule
                            )
                        elif cfg_type == CudaGraphConfigType.FLASH_INFER_PACKED:
                            self._capture_one_flashinfer_packed(
                                key, config, self.submodule,
                            )
                        logger.info("Captured CUDA graph for %s: %s bs=%d",
                                    self.submodule_name, key, bs)
                    except Exception:
                        logger.warning(
                            "Failed to capture CUDA graph for %s: %s bs=%d",
                            self.submodule_name, key, bs, exc_info=True)

        mem_after = torch.cuda.memory_allocated(self.device)
        shared_bytes = sum(
            t.numel() * t.element_size() for t in self.shared_static_buffers.values()
        )
        # Report both: the deterministic synthetic counter (clean before/after
        # for the buffer-reuse change in isolation) and the actual GPU delta
        # (covers FlashInfer wrappers + dummy KV state too, but is noisier).
        logger.info(
            "CudaGraphRunner[%s]: warmup_and_capture done. "
            "shared_static_buffers: %d entries, %.2f MB resident "
            "(would have been %.2f MB with per-capture clones — saved %.2f MB). "
            "Total cuda alloc delta during warmup: %.2f MB.",
            self.submodule_name,
            len(self.shared_static_buffers),
            shared_bytes / (1024 ** 2),
            self._capture_clone_bytes_naive / (1024 ** 2),
            (self._capture_clone_bytes_naive - shared_bytes) / (1024 ** 2),
            (mem_after - mem_before) / (1024 ** 2),
        )

    def _create_persistent_wrappers(
        self, bs: int, config: CudaGraphConfig,
        total_tokens: int, slot_idx: int = 0,
    ) -> dict:
        """Create persistent FlashInfer wrappers for CUDA graph capture.

        Returns dict of label -> _PlanState with persistent wrappers.

        ``slot_idx`` distinguishes the two double-buffer slots — each slot
        gets its own FlashInfer workspace + index buffers so plan(slot 1)
        can run on plan_stream concurrently with replay(slot 0) on
        default_stream without racing on the wrapper's persistent state.
        """
        from mminf.engine.cache_manager import _PlanState
        from mminf.utils.flashinfer_utils import (
            FlashInferDecodeWrapper,
            FlashInferPrefillWrapper,
        )

        is_decode = (total_tokens == bs)

        cfg = self.kv_cache_config

        # Allocate workspace buffer for CUDA graph wrappers.
        # Each (label, slot) gets its own workspace — slots must NOT share
        # workspace because plan() writes scheduling state there and the
        # captured replay reads it; concurrent plan(slot B) + replay(slot A)
        # would race on shared workspace addresses.
        plan_states = {}
        for label in config.labels:
            ws_label = f"{label}_cugraph_slot{slot_idx}"
            if is_decode:
                wrapper = FlashInferDecodeWrapper(
                    workspace_buffer=self.buffer_manager.get(ws_label),
                    num_qo_heads=cfg.num_qo_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                    page_size=cfg.page_size,
                    batch_size=bs,
                    max_num_pages=cfg.max_num_pages,
                    device=self.device,
                    use_cuda_graph=True,
                    enable_nvtx=self.enable_nvtx,
                )
            else:
                wrapper = FlashInferPrefillWrapper(
                    workspace_buffer=self.buffer_manager.get(ws_label),
                    num_qo_heads=cfg.num_qo_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                    page_size=cfg.page_size,
                    batch_size=bs,
                    max_total_tokens=total_tokens,
                    max_num_pages=cfg.max_num_pages,
                    device=self.device,
                    use_cuda_graph=True,
                    enable_nvtx=self.enable_nvtx,
                )

            # Static pos_ids buffer for RoPE — also per-slot (captured into
            # the slot's graph; both slots' graphs would otherwise read from
            # the same buffer and the second slot's preprocess would clobber
            # the first slot's view if any future change ran replays
            # concurrently).
            static_pos_ids = torch.zeros(
                total_tokens, dtype=torch.long, device=self.device
            )

            plan_states[label] = _PlanState(
                wrapper=wrapper,
                pos_ids=static_pos_ids,
            )

        return plan_states

    def _make_dummy_rids(
        self, config: CudaGraphConfig, bs: int, slot_idx: int = 0,
    ):
        dummy_rids = [
            f"__cg_{config.capture_graph_walk}_{config.requires_cfg}_slot{slot_idx}_{i}__"
            for i in range(bs)
        ]

        # Add dummy requests with all needed labels
        for rid in dummy_rids:
            self.alloc_manager.add_request(rid, labels=config.labels)
        return dummy_rids

    def _free_dummy_rids(self, config: CudaGraphConfig, dummy_rids: list[str]):
        for rid in dummy_rids:
            for label in config.labels:
                self.alloc_manager.reset_label(rid, label, free=True)

    def _intern_static_buffer(
        self, config_idx: int, key: str, value: torch.Tensor,
    ) -> torch.Tensor:
        """Return a leading-dim slice view into the shared buffer for (config_idx, key).

        Allocates the shared buffer at ``value``'s shape on first encounter
        — relies on ``warmup_and_capture``'s largest-first iteration
        (``reversed(sizes)`` × ``reversed(sorted(get_total_tokens(bs)))``) so
        the first capture for each (config_idx, key) is the max bucket.
        Subsequent captures for the same (config_idx, key) re-slice along the
        leading dim, sharing the underlying storage so all (bs, num_tokens)
        buckets cost one max-shape allocation per tensor instead of one full
        clone per bucket.

        Trailing dims (everything past dim 0) must match between the shared
        buffer and ``value`` — the bucket varies the leading dim only. A
        mismatch is a design-level surprise (a tensor whose shape depends on
        bs in a non-leading way), so we hard-fail with a precise message.
        """
        buf_key = (config_idx, key)
        shared = self.shared_static_buffers.get(buf_key)
        if shared is None:
            shared = torch.empty(value.shape, dtype=value.dtype, device=value.device)
            self.shared_static_buffers[buf_key] = shared
        self._capture_clone_bytes_naive += value.numel() * value.element_size()
        leading = value.shape[0]
        if leading > shared.shape[0] or value.shape[1:] != shared.shape[1:]:
            raise RuntimeError(
                f"_intern_static_buffer: capture for key={key!r} (config_idx={config_idx}) "
                f"requires shape {tuple(value.shape)} but shared buffer is "
                f"{tuple(shared.shape)} — captures should be ordered largest-first "
                "by leading dim with matching trailing dims"
            )
        sliced = shared[:leading]
        sliced.copy_(value)
        return sliced

    def _create_cache_mgr_and_dummy_engine_inputs(
        self, dummy_rids, plan_states,
        config: CudaGraphConfig
    ):
        # Create BatchedCacheManager with CUDA graph plan states
        cache_manager = BatchedCacheManager(
            request_ids=dummy_rids,
            active_labels_per_request={rid: "main" for rid in dummy_rids},
            kv_cache=self.alloc_manager.kv_cache,
            alloc_manager=self.alloc_manager,
            buffer_manager=self.buffer_manager,
            kv_cache_config=self.kv_cache_config,
            device=self.device,
            cuda_graph_plan_states=plan_states,
            auto_write_store=False,
            enable_nvtx=self.enable_nvtx,
        )
        # Build per-request metadata
        dummy_metadata = {
            rid: CurrentForwardPassInfo(
                request_id=rid,
                graph_walk=config.capture_graph_walk,
                requires_cfg=config.requires_cfg,
                fwd_index=0,
                random_seed=0,
                max_tokens=1,
                sampling_config={}
            ) for rid in dummy_rids
        }

        return ModelInputsFromEngine(
            request_ids=dummy_rids,
            per_request_info=dummy_metadata,
            cache_manager=cache_manager,
            sampler=make_sampler_from_buffers(
                bufs=self.sampler_buffer,
                request_ids=[], sampling_configs={},
                padded_bs=len(dummy_rids)
            ),
        )

    def _make_dummy_seq_lens(
        self, bs: int,
        total_tokens: int, # total tokens per batch
    )-> list[int]:
        # must ensure that the seq lens array sums to total tokens
        seq_lens = [total_tokens // bs] * bs
        seq_lens[0] += total_tokens % bs
        return seq_lens

    def _build_slot_from_capture(
        self, output, graph, static_inputs, cache_manager,
    ) -> CudaGraphSlot:
        """Wrap one slot's capture artifacts into a CudaGraphSlot."""
        has_non_logit = False
        if isinstance(output, dict):
            for k, v in output.items():
                if k == "__batched_logits__":
                    continue
                if isinstance(v, dict) and any(
                    out_key != "logits" for out_key in v.keys()
                ):
                    has_non_logit = True
                    break
        return CudaGraphSlot(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=output,
            static_cache_manager=cache_manager,
            has_non_logit_outputs=has_non_logit,
        )

    def _register_graph_data(
        self,
        key: CudaGraphKey,
        config: CudaGraphConfig,
        bs: int,
        slots: list[CudaGraphSlot],
    ) -> None:
        """Register a populated CudaGraphData under all replay graph walks.

        Phase 3 double-buffer: ``slots`` is the list of all NUM_SLOTS slots,
        each with its own captured graph + persistent wrappers. Replay picks
        the active slot via ``CudaGraphData.next_slot``; pre-plan targets
        the slot the next replay will use.
        """
        logger.info(
            "CudaGraphRunner: captured graph %s slots=%d has_non_logit_outputs=%s",
            key, len(slots), [s.has_non_logit_outputs for s in slots],
        )
        for graph_walk in config.replay_graph_walks:
            lookup_key = CudaGraphKey(
                graph_walk=graph_walk,
                requires_cfg=config.requires_cfg,
                bs=bs,
                num_tokens=key.num_tokens,
            )
            self.graphs[lookup_key] = CudaGraphData(
                config=config,
                bs=bs,
                slots=slots,
                next_slot=0,
            )

    def _capture_one_flashinfer_packed(
        self, key: CudaGraphKey,
        config: FlashInferPackedCudaGraphConfig,
        submodule: ARNodeSubmodule,
    ):
        """Capture NUM_SLOTS prefill graphs for (bs, num_tokens) bucket.

        Phase 3 double-buffer mirrors the basic_batched path: each slot has
        its own dummy_rids, FlashInfer wrappers, BatchedCacheManager, and
        captured graph. Static input templates are shared across slots via
        the per-config interned buffer pool.
        """
        bs = key.bs
        template_dict = config.num_token_to_inputs[key.num_tokens]
        config_idx = self.capture_configs.index(config)
        captured_slots: list[CudaGraphSlot] = []
        dummy_rids_to_free: list[list[str]] = []

        try:
            for slot_idx in range(self.NUM_SLOTS):
                dummy_rids = self._make_dummy_rids(config, bs, slot_idx)
                dummy_rids_to_free.append(dummy_rids)

                templates = {
                    k: (self._intern_static_buffer(config_idx, k, v)
                        if isinstance(v, torch.Tensor) else v)
                    for k, v in template_dict.items()
                }

                plan_states = self._create_persistent_wrappers(
                    bs, config, total_tokens=key.num_tokens, slot_idx=slot_idx,
                )
                seq_lens = self._make_dummy_seq_lens(bs, key.num_tokens)

                engine_inputs = self._create_cache_mgr_and_dummy_engine_inputs(
                    dummy_rids=dummy_rids, plan_states=plan_states, config=config,
                )
                cache_manager = engine_inputs.cache_manager

                def plan_attention(_cache_manager=cache_manager, _seq_lens=seq_lens):
                    for label in config.labels:
                        _cache_manager.plan_attention(
                            seq_lens=_seq_lens,
                            is_causal=config.causal_attention,
                            label=label, write_store=False,
                        )
                        _cache_manager.plan_rope(seq_lens=_seq_lens, label=label)

                plan_attention()

                static_input_keys = [
                    k for k, v in templates.items()
                    if isinstance(v, torch.Tensor)
                ]

                forward = submodule.forward_batched
                if config.compile:
                    forward = torch.compile(
                        forward,
                        mode="max-autotune-no-cudagraphs",
                        fullgraph=False,
                        dynamic=False,
                    )

                def run_forward(_forward=forward, _engine_inputs=engine_inputs, _templates=templates):
                    return _forward(
                        graph_walk=config.capture_graph_walk,
                        engine_inputs=_engine_inputs,
                        **_templates,
                    )

                torch.cuda.set_device(self.device)
                torch.cuda.synchronize()
                for _ in range(2):
                    with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                        output = run_forward()
                    for rid in dummy_rids:
                        for label in config.labels:
                            state = self.alloc_manager.get_state(rid, label)
                            state.seq_len = 0
                            state.position_id_start = 0
                    plan_attention()
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                    with torch.cuda.graph(graph, pool=self.memory_pool):
                        output = run_forward()
                torch.cuda.synchronize()

                slot = self._build_slot_from_capture(
                    output=output,
                    graph=graph,
                    static_inputs={
                        "preprocessed": templates,
                        "static_input_keys": static_input_keys,
                        "dummy_rids": dummy_rids,
                        "dummy_metadata": engine_inputs.per_request_info,
                    },
                    cache_manager=cache_manager,
                )
                captured_slots.append(slot)

            self._register_graph_data(
                key=key, config=config, bs=bs, slots=captured_slots,
            )
        finally:
            for rids in dummy_rids_to_free:
                self._free_dummy_rids(config, rids)


    def _capture_one_basic_batched(
        self, key: CudaGraphKey,
        config: BasicBatchedCudaGraphConfig,
        submodule: ARNodeSubmodule,
    ) -> None:
        """Capture NUM_SLOTS decode graphs for (bs, single_request_inputs.input_seq_len * bs) bucket.

        Phase 3 double-buffer: each slot gets its own dummy_rids, persistent
        FlashInfer wrappers, BatchedCacheManager, and captured graph. Static
        input buffers are SHARED across slots (preprocess writes them just
        before replay; no race because replays serialize on the GPU thread).
        """
        bs = key.bs
        template = config.single_request_inputs
        if template is None:
            logger.warning(
                "%s.get_cuda_graph_configs returned a BasicBatchedCudaGraphConfig "
                "with single_request_inputs=None for walk=%s — skipping capture",
                self.submodule_name, config.capture_graph_walk,
            )
            return
        config_idx = self.capture_configs.index(config)
        captured_slots: list[CudaGraphSlot] = []
        dummy_rids_to_free: list[list[str]] = []

        try:
            for slot_idx in range(self.NUM_SLOTS):
                # Each slot gets disjoint dummy_rids so its alloc_manager
                # bookkeeping doesn't bleed into the other slot's capture
                # state (page indices, seq_lens). Wrappers and
                # cache_manager are also per-slot.
                dummy_rids = self._make_dummy_rids(config, bs, slot_idx)
                dummy_rids_to_free.append(dummy_rids)
                dummy_inputs = [template.clone() for _ in dummy_rids]

                plan_states = self._create_persistent_wrappers(
                    bs, config,
                    total_tokens=bs * template.input_seq_len,
                    slot_idx=slot_idx,
                )

                engine_inputs = self._create_cache_mgr_and_dummy_engine_inputs(
                    dummy_rids=dummy_rids, plan_states=plan_states, config=config,
                )
                cache_manager = engine_inputs.cache_manager

                # Preprocess (plans attention+rope outside graph).
                preprocessed = submodule.preprocess(
                    graph_walk=config.capture_graph_walk,
                    engine_inputs=engine_inputs,
                    inputs=dummy_inputs,
                )

                # Both slots share the same static input buffers (per
                # config_idx, key) — the captured forward reads them after
                # preprocess writes, and replays don't overlap. Slot 0's
                # capture allocates; slot 1's capture re-interns the same
                # buffer (re-copy is cheap, addresses stay stable).
                for k in list(preprocessed.keys()):
                    v = preprocessed[k]
                    if isinstance(v, torch.Tensor):
                        preprocessed[k] = self._intern_static_buffer(config_idx, k, v)

                static_input_keys = [
                    k for k, v in preprocessed.items()
                    if isinstance(v, torch.Tensor)
                ]

                forward = submodule.forward_batched
                if config.compile:
                    forward = torch.compile(
                        forward,
                        mode="max-autotune-no-cudagraphs",
                        fullgraph=False,
                        dynamic=False,
                    )

                def run_forward(_forward=forward, _engine_inputs=engine_inputs, _preprocessed=preprocessed):
                    return _forward(
                        graph_walk=config.capture_graph_walk,
                        engine_inputs=_engine_inputs,
                        **_preprocessed,
                    )

                torch.cuda.set_device(self.device)
                torch.cuda.synchronize()
                for _ in range(2):
                    with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                        output = run_forward()
                    # Reset seq_lens so capture starts from a clean state.
                    for rid in dummy_rids:
                        for label in config.labels:
                            state = self.alloc_manager.get_state(rid, label)
                            state.seq_len = 0
                            state.position_id_start = 0
                    submodule.preprocess(
                        graph_walk=config.capture_graph_walk,
                        engine_inputs=engine_inputs,
                        inputs=dummy_inputs,
                    )
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                    with torch.cuda.graph(graph, pool=self.memory_pool):
                        output = run_forward()
                torch.cuda.synchronize()

                slot = self._build_slot_from_capture(
                    output=output,
                    graph=graph,
                    static_inputs={
                        "preprocessed": preprocessed,
                        "capture_template": template,
                        "static_input_keys": static_input_keys,
                        "dummy_rids": dummy_rids,
                        "dummy_metadata": engine_inputs.per_request_info,
                    },
                    cache_manager=cache_manager,
                )
                captured_slots.append(slot)

            self._register_graph_data(
                key=key, config=config, bs=bs, slots=captured_slots,
            )
        finally:
            for rids in dummy_rids_to_free:
                self._free_dummy_rids(config, rids)

    def can_run(
        self,
        batch_size: int,
        num_tokens: int,
        graph_walk: str = "decode",
        requires_cfg: bool = False,
    ) -> bool:
        """Check if a captured graph exists for this configuration."""
        return self._get_key_for(
            batch_size, num_tokens,
            graph_walk, requires_cfg,
        ) is not None

    def _get_key_for(
        self,
        batch_size: int,
        num_tokens: int,
        graph_walk: str = "decode",
        requires_cfg: bool = False,
    ) -> CudaGraphKey | None:
        if not self.graphs:
            return None
        config = self._config_for(graph_walk, requires_cfg)
        if config is None:
            return None
        padded_bs = self._get_padded_batch_size(batch_size, config)
        if padded_bs is None:
            return None
        padded_num_tokens = self._get_padded_num_tokens(num_tokens, padded_bs, config)
        if padded_num_tokens is None:
            return None

        key = CudaGraphKey(
            graph_walk=graph_walk,
            requires_cfg=requires_cfg,
            bs=padded_bs,
            num_tokens=padded_num_tokens,
        )
        return key if key in self.graphs else None

    def _config_for(self, graph_walk: str, requires_cfg: bool) -> CudaGraphConfig | None:
        for cfg in self.capture_configs:
            if graph_walk in cfg.replay_graph_walks and cfg.requires_cfg == requires_cfg:
                return cfg
        return None

    def _get_padded_batch_size(
        self,
        batch_size: int,
        config: CudaGraphConfig,
    ) -> int | None:
        """Find smallest captured batch size >= batch_size for this config.

        Mirrors warmup_and_capture's fallback: when a config doesn't override
        capture_batch_sizes (the common case — Bagel et al. just defer to the
        runner's default), capture iterates self.CAPTURE_BATCH_SIZES, so lookup
        has to consult the same list to find a match.
        """
        sizes = sorted(config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES)
        idx = bisect.bisect_left(sizes, batch_size)
        if idx >= len(sizes):
            return None
        return sizes[idx]

    def _get_padded_num_tokens(
        self,
        num_tokens: int,
        padded_bs: int,
        config: CudaGraphConfig,
    ) -> int | None:
        """Find smallest captured token-count >= num_tokens for this config and bs."""
        sizes = sorted(config.get_total_tokens(padded_bs))
        idx = bisect.bisect_left(sizes, num_tokens)
        if idx >= len(sizes):
            return None
        return sizes[idx]

    def _get_basic_batched_key_for(
        self,
        graph_walk: str,
        requires_cfg: bool,
        batch_size: int,
    ) -> CudaGraphKey | None:
        """Look up a captured key by (graph_walk, requires_cfg, bs) alone.

        For ``BASIC_BATCHED`` configs, the captured ``num_tokens`` is uniquely
        determined by the per-input ``input_seq_len`` and the (padded) batch
        size: ``get_total_tokens(bs) == [input_seq_len * bs]``. Callers on
        the decode/spec path can therefore find their captured graph without
        independently knowing ``num_tokens`` — used by ``pre_plan_for_batch``
        and ``reset_pre_plan_state_for_slot``, both of which are
        BASIC_BATCHED-only.
        """
        config = self._config_for(graph_walk, requires_cfg)
        if config is None or config.get_config_type() != CudaGraphConfigType.BASIC_BATCHED:
            return None
        padded_bs = self._get_padded_batch_size(batch_size, config)
        if padded_bs is None:
            return None
        total_tokens = config.get_total_tokens(padded_bs)
        if not total_tokens:
            return None
        key = CudaGraphKey(
            graph_walk=graph_walk,
            requires_cfg=requires_cfg,
            bs=padded_bs,
            num_tokens=total_tokens[0],
        )
        return key if key in self.graphs else None

    def _get_sampler(
        self, per_request_info: dict[str, CurrentForwardPassInfo],
        request_ids: list[str], padded_bs: int
    ):
        # Per-request sampling configs are pre-staged on master GPU buffers
        # (see ``register_request`` / ``update_request_config`` from AREngine);
        # the per-step path is just a slot-index gather + index_select. The
        # ``per_request_info`` argument is unused here but kept on the
        # signature for symmetry with the older callers.
        del per_request_info
        return self.sampler_buffer.gather_for_request_ids(
            request_ids=request_ids, padded_bs=padded_bs,
        )

    def register_request(
        self, request_id: str, sampling_config: SamplingConfig | None = None,
    ) -> None:
        """Allocate a SamplerBuffers slot for ``request_id`` if not present."""
        self.sampler_buffer.register_request(request_id, sampling_config)

    def unregister_request(self, request_id: str) -> None:
        """Release the SamplerBuffers slot for ``request_id``."""
        self.sampler_buffer.unregister_request(request_id)

    def update_request_config(
        self, request_id: str, sampling_config: SamplingConfig,
    ) -> None:
        """Change-detect update for ``request_id``'s master sampling row."""
        self.sampler_buffer.update_request_config(request_id, sampling_config)

    def reserve_slot(
        self,
        graph_walk: str,
        requires_cfg: bool,
        batch_size: int,
        num_tokens: int | None = None,
    ) -> int | None:
        """Allocate the next double-buffer slot for an upcoming submission.

        Called by the Worker's main thread RIGHT BEFORE submitting a replay
        (and, for speculation, the matching pre-plan) so both submissions
        see the same slot. Increments the per-key ``next_slot`` counter so
        the following submission picks the OTHER slot.

        ``num_tokens`` is optional: when ``None`` the runner derives it from
        the captured BASIC_BATCHED config (the only kind that participates
        in pre-reserved replay today). Pass it explicitly for non-BASIC
        configs or if a future caller needs to disambiguate among multiple
        token-bucket captures for the same (walk, cfg, bs).

        Returns the slot index, or ``None`` if no captured graph matches —
        in which case the engine's eager fallback runs (no slot needed).
        """
        if num_tokens is None:
            key = self._get_basic_batched_key_for(
                graph_walk=graph_walk,
                requires_cfg=requires_cfg,
                batch_size=batch_size,
            )
        else:
            key = self._get_key_for(
                batch_size=batch_size, num_tokens=num_tokens,
                graph_walk=graph_walk, requires_cfg=requires_cfg,
            )
        if key is None:
            return None
        data = self.graphs[key]
        if not data.slots:
            return None
        slot = data.next_slot
        data.next_slot = (data.next_slot + 1) % len(data.slots)
        return slot

    def _get_or_make_plan_stream(self) -> "torch.cuda.Stream | None":
        """Lazily allocate a dedicated CUDA stream for pre-planning.

        Pre-plan must NOT submit its kernels to the default stream because
        the order of submissions between the GPU thread (recording the
        prev-batch completion_event) and plan_executor (submitting plan's
        memcpys / CUB scan) is timing-dependent on Python thread scheduling.
        If plan_executor sneaks its submissions in BEFORE the GPU thread
        records the event, the event ends up firing AFTER plan's kernels
        drain — which delays the slow_post side-stream D→H wait that's
        gated on that event. The fix: pre-plan runs on its own stream,
        gated by the prev-batch completion_event so wrapper buffer writes
        never race with the in-flight replay's reads. The captured graph's
        next replay then waits for plan_done_event on default stream.
        """
        if not torch.cuda.is_available():
            return None
        if self._plan_stream is None:
            self._plan_stream = torch.cuda.Stream(device=self.device)
        return self._plan_stream

    def pre_plan_for_batch(
        self,
        graph_walk: str,
        requires_cfg: bool,
        request_ids: list[str],
        per_request_info: dict[str, CurrentForwardPassInfo] | None = None,
        prev_completion_event: "torch.cuda.Event | None" = None,
        slot: int | None = None,
    ) -> bool:
        """Pre-plan FlashInfer attention into the inactive double-buffer slot.

        Runs cache_manager.plan_attention against the slot's persistent
        wrapper(s) on a dedicated plan_stream, once per label in the
        captured config (so multi-label captures like BAGEL CFG decode get
        all labels overlapped), then populates the slot's
        ``_pre_planned_labels`` set and stashes a plan_done_event so the
        captured graph's replay can wait on it before reading the wrappers.

        Phase 3 double-buffer: the slot here is the OTHER slot from the one
        currently being replayed — main thread reserves it via
        ``reserve_slot`` before dispatching pre-plan, so plan(N+1) writes
        slot[(s+1)%2]'s wrapper buffers while replay(N) on slot[s] continues
        on the default stream. No prev_completion_event gating needed for
        wrapper-buffer correctness because the slots are disjoint.
        ``prev_completion_event`` is still accepted for API symmetry but
        ignored — kept so callers can be slot-agnostic.

        Returns True if pre-planning was applied; False otherwise (caller's
        GPU thread plans inline).
        """
        from mminf.utils.profiler import range_pop, range_push

        real_bs = len(request_ids)
        # num_tokens is derivable from the captured BASIC_BATCHED config —
        # don't hardcode it from real_bs (works today only because AR decode
        # has input_seq_len=1; would silently mismatch for any future
        # multi-token-per-request decode/spec capture).
        key = self._get_basic_batched_key_for(
            graph_walk=graph_walk,
            requires_cfg=requires_cfg,
            batch_size=real_bs,
        )
        if key is None:
            return False

        graph_data = self.graphs[key]
        config = graph_data.config
        if not graph_data.slots:
            return False
        if slot is None:
            slot = 0
        slot %= len(graph_data.slots)
        slot_data = graph_data.slots[slot]
        static_cm = slot_data.static_cache_manager

        if self.enable_nvtx:
            range_push("plan_worker.pre_plan", synchronize=False)
        # plan_stream lets plan(slot S+1)'s small kernels submit independently
        # of the GPU thread's default-stream traffic. Disjoint wrapper buffers
        # mean we DON'T gate plan_stream on prev_completion_event — that's
        # the whole point of double-buffering: plan(N+1) overlaps with
        # replay(N) on the GPU. plan_done_event still gates the eventual
        # replay(N+1) (default stream) so it doesn't read buffers before
        # plan(N+1)'s writes are visible.
        plan_stream = self._get_or_make_plan_stream()
        plan_done_event: torch.cuda.Event | None = None

        # Temporarily alias real rids onto this slot's cache_manager so
        # plan_attention reads real request states. The slot's static_cm
        # will be aliased to the same real rids again at replay time
        # (Step 1 swap_states in _run_basic_batched), so the wrapper
        # buffers we write here stay valid for the matching replay.
        saved_request_ids = static_cm.request_ids
        saved_active_labels = static_cm.active_labels
        # Pre-plan every label this captured graph's preprocess will ask for.
        # Multi-label captures (e.g. BAGEL CFG decode, labels=["main",
        # "cfg_img"]) inline-plan once per label; we cover all of them so
        # none falls back to inline plan_attention. Each label's wrapper has
        # its own static buffers — sequential calls on the same plan_stream
        # respect natural FIFO ordering, so a single plan_done_event after
        # the last call covers every label's writes.
        config_labels = config.labels
        # Per-request token count comes from the captured config — the same
        # ``input_seq_len`` that ``_capture_one_basic_batched`` used to size
        # ``total_tokens = bs * input_seq_len``. Don't hardcode 1; today AR
        # decode happens to be 1, but multi-token-per-request decode/spec
        # captures (e.g. tree-spec) would silently mis-plan with [1]*bs.
        per_req_seq_len = config.single_request_inputs.input_seq_len
        try:
            static_cm.request_ids = list(request_ids) + saved_request_ids[len(request_ids):]
            seq_lens = [per_req_seq_len] * len(saved_request_ids)
            if plan_stream is not None:
                with torch.cuda.stream(plan_stream):
                    for label_name in config_labels:
                        static_cm.active_labels = {rid: label_name for rid in request_ids}
                        static_cm.plan_attention(
                            seq_lens=seq_lens,
                            dtype=self.autocast_dtype,
                            label=label_name,
                        )
                plan_done_event = torch.cuda.Event()
                plan_done_event.record(plan_stream)
            else:
                for label_name in config_labels:
                    static_cm.active_labels = {rid: label_name for rid in request_ids}
                    static_cm.plan_attention(
                        seq_lens=seq_lens,
                        dtype=self.autocast_dtype,
                        label=label_name,
                    )
            static_cm._pre_planned_labels = set(config_labels)
            static_cm._plan_done_event = plan_done_event
        finally:
            static_cm.request_ids = saved_request_ids
            static_cm.active_labels = saved_active_labels
            if self.enable_nvtx:
                range_pop(synchronize=False)
        return True

    def reset_pre_plan_state_for_slot(
        self,
        graph_walk: str,
        requires_cfg: bool,
        batch_size: int,
        slot: int | None = None,
    ) -> None:
        """Clear the slot-local ``_pre_planned_labels`` and
        ``_plan_done_event`` set by ``pre_plan_for_batch`` for the same
        (key, slot). Used by the worker to recover from speculation
        drops/failures: drop only the targeted slot's pre-plan state
        rather than wiping every captured graph's slots across every
        engine — the latter would stomp any concurrent in-flight
        pre-plan whose flags have not yet been consumed by its matching
        replay (currently impossible because plan_executor.max_workers=1,
        but a latent footgun if concurrency is raised).

        BASIC_BATCHED-only (paired with ``pre_plan_for_batch``); the key
        is derived from ``(graph_walk, requires_cfg, batch_size)`` via the
        captured config so callers don't need to know ``num_tokens``.
        """
        key = self._get_basic_batched_key_for(
            graph_walk=graph_walk,
            requires_cfg=requires_cfg,
            batch_size=batch_size,
        )
        if key is None:
            return
        graph_data = self.graphs.get(key)
        if graph_data is None or not graph_data.slots:
            return
        if slot is None:
            slot = 0
        slot %= len(graph_data.slots)
        cm = graph_data.slots[slot].static_cache_manager
        cm._pre_planned_labels.clear()
        cm._plan_done_event = None

    def run(
        self,
        graph_walk: str,
        requires_cfg: bool,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: ARNodeSubmodule,
        slot: int | None = None,
        advance_event: "object | None" = None,
        launch_started_event: "object | None" = None,
    ) -> dict:
        """Look up the matching captured graph and dispatch on config type.

        ``slot`` selects one of NUM_SLOTS captured graphs for this key.
        When ``None``, the runner advances the per-key counter itself —
        used by callers that didn't pre-reserve (e.g., non-speculative
        replays where pre-plan isn't dispatched). Worker's speculation
        path passes the slot it reserved before dispatching pre-plan so
        replay(N) and pre-plan(N+1) target compatible slots.

        BasicBatched (decode-style): submodule.preprocess re-plans attention/rope
        and produces packed tensors written into static buffers — same call that
        was captured. FlashInferPacked (vox-serve-style prefill): submodule.preprocess
        packs real per-request inputs (with synthetic zero-length padding for empty
        slots) into the static buffers; trailing static-buffer slots beyond
        real_num_tokens keep capture-time contents (FlashInfer's qo_indptr-based
        attention skips them).
        """
        real_bs = len(request_ids)
        real_num_tokens = sum(inp.input_seq_len for inp in inputs)

        key = self._get_key_for(
            batch_size=real_bs,
            num_tokens=real_num_tokens,
            graph_walk=graph_walk,
            requires_cfg=requires_cfg,
        )
        if key is None:
            raise RuntimeError(
                f"No captured graph for walk={graph_walk!r}, requires_cfg={requires_cfg}, "
                f"bs={real_bs}, num_tokens={real_num_tokens} — _can_use_cuda_graph "
                "should have rejected this batch upstream."
            )

        graph_data: CudaGraphData = self.graphs[key]
        if not graph_data.slots:
            raise RuntimeError(
                f"CudaGraphData for {key} has no slots — capture failed silently?"
            )
        if slot is None:
            # Caller didn't reserve. Advance the counter ourselves so the
            # next reservation lands on the OTHER slot.
            slot = graph_data.next_slot
            graph_data.next_slot = (graph_data.next_slot + 1) % len(graph_data.slots)
        slot %= len(graph_data.slots)
        slot_data = graph_data.slots[slot]

        cfg_type = graph_data.config.get_config_type()
        if cfg_type == CudaGraphConfigType.BASIC_BATCHED:
            return self._run_basic_batched(
                key, graph_data, slot_data,
                request_ids, inputs, per_request_info, submodule,
                advance_event=advance_event,
                launch_started_event=launch_started_event,
            )
        if cfg_type == CudaGraphConfigType.FLASH_INFER_PACKED:
            return self._run_flashinfer_packed(
                key, graph_data, slot_data,
                request_ids, inputs, per_request_info, submodule,
                advance_event=advance_event,
                launch_started_event=launch_started_event,
            )
        raise ValueError(f"Unknown CudaGraphConfigType: {cfg_type}")

    def _run_basic_batched(
        self,
        key: CudaGraphKey,
        graph_data: CudaGraphData,
        slot_data: CudaGraphSlot,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: ARNodeSubmodule,
        advance_event: "object | None" = None,
        launch_started_event: "object | None" = None,
    ) -> dict:
        """Decode-style replay. Pads real inputs to padded_bs by cloning the capture
        template, then routes through submodule.preprocess (which re-plans attention
        and RoPE on the static cache manager) and copies the resulting packed tensors
        into the static buffers before replay.

        Phase 3: ``slot_data`` is the chosen double-buffer slot (graph + persistent
        wrappers + cache_manager). Same logic as before — we just look up the
        slot's graph/cm instead of reading flat fields off ``graph_data``.
        """
        real_bs = len(request_ids)
        padded_bs = key.bs

        graph = slot_data.graph
        static = slot_data.static_inputs
        static_cm = slot_data.static_cache_manager
        static_output = slot_data.static_outputs

        preprocessed = static["preprocessed"]
        dummy_rids = static["dummy_rids"]
        static_input_keys = static["static_input_keys"]
        capture_template = static["capture_template"]
        config_labels = graph_data.config.labels

        # --- Step 1: Swap real request states onto dummy slots ---
        if self.enable_nvtx:
            mark("gpu_thread.preprocess_start")
            range_push("gpu_thread.preprocess", synchronize=False)
        if self.enable_nvtx:
            range_push("cg.swap_states", synchronize=False)
        for i, rid in enumerate(request_ids):
            dummy_rid = dummy_rids[i]
            for label in config_labels:
                real_state = self.alloc_manager.get_state(rid, label)
                # makes state if it doesn't exist
                self.alloc_manager.get_state(dummy_rid, label)
                self.alloc_manager.request_states[dummy_rid][label] = real_state

        # For padding slots (i >= real_bs), ensure dummy states exist
        for i in range(real_bs, padded_bs):
            dummy_rid = dummy_rids[i]
            for label in config_labels:
                self.alloc_manager.get_state(dummy_rid, label)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # --- Step 2: Pad inputs to padded_bs and re-plan via preprocess ---
        if self.enable_nvtx:
            range_push("cg.preprocess_replan", synchronize=False)
        if self.enable_nvtx:
            range_push("cg.preprocess_replan.pad_inputs", synchronize=False)
        real_inputs = list(inputs)
        # Padding slots reuse the capture_template so submodule.preprocess sees the
        # same input shape it saw at capture time and doesn't crash on missing keys.
        for _i in range(real_bs, padded_bs):
            real_inputs.append(capture_template.clone())
        if self.enable_nvtx:
            range_pop(synchronize=False)

        if self.enable_nvtx:
            range_push("cg.preprocess_replan.metadata", synchronize=False)
        real_metadata = self._build_replay_metadata(
            dummy_rids, request_ids, real_bs,
            per_request_info, static["dummy_metadata"],
        )
        engine_inputs = ModelInputsFromEngine(
            request_ids=dummy_rids,
            per_request_info=real_metadata,
            cache_manager=static_cm,
            sampler=self._get_sampler(
                per_request_info=per_request_info,
                request_ids=request_ids,
                padded_bs=padded_bs
            )
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("cg.preprocess_replan.submodule_preprocess", synchronize=False)
        real_inputs = submodule.preprocess(
            graph_walk=key.graph_walk,
            engine_inputs=engine_inputs,
            inputs=real_inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # --- Step 3: Copy real packed tensors into static buffers ---
        if self.enable_nvtx:
            range_push("cg.copy_inputs", synchronize=False)
        for k in static_input_keys:
            real_val = real_inputs.get(k)
            if real_val is None or not isinstance(real_val, torch.Tensor):
                continue
            static_buf = preprocessed[k]
            static_buf[:real_val.shape[0]].copy_(real_val)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_pop(synchronize=False)
            mark("gpu_thread.preprocess_end")

        # --- Step 4: Replay ---
        # Phase 3: if pre-plan was applied for this iter, the wrapper's
        # static buffers were written on a side stream. Make the default
        # stream wait on plan_done_event before replay reads them.
        plan_done_event = getattr(static_cm, "_plan_done_event", None)
        if plan_done_event is not None:
            torch.cuda.default_stream(self.device).wait_event(plan_done_event)
            static_cm._plan_done_event = None
        if self.enable_nvtx:
            mark("gpu_thread.cuda_graph_start")
            range_push("gpu_thread.cuda_graph", synchronize=False)
            range_push("cg.replay", synchronize=False)
        # Release the main thread now that all CPU-side prep is done and
        # we're about to enter the CUDA driver. graph.replay() drops the
        # GIL inside C++, so main-thread postprocess can overlap.
        if launch_started_event is not None:
            launch_started_event.set()
        graph.replay()
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_pop(synchronize=False)
            mark("gpu_thread.cuda_graph_end")

        # --- Step 5: Advance seq_lens on REAL request states (Python-only) ---
        # advance_seq_lens is not captured in the graph; we call it manually so
        # the real states (aliased onto dummy slots) move forward.
        if self.enable_nvtx:
            mark("gpu_thread.postprocess_start")
            range_push("gpu_thread.postprocess", synchronize=False)
        if self.enable_nvtx:
            range_push("cg.advance_seq_lens", synchronize=False)
        for label in config_labels:
            static_cm.set_active_label(label)
            static_cm.advance_seq_lens()
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # Phase 3: signal that alloc_manager state for batch_N is now
        # post-advance. plan_executor's pre_plan(batch_(N+1)) waits on this
        # event instead of prev_future, so plan() starts ~tens of µs into
        # replay(N) (overlapping with replay(N)'s remaining GPU work) rather
        # than after replay(N) fully completes. The event is also signaled
        # in the GPU thread's outer try/finally so a failure path still
        # wakes plan_executor.
        if advance_event is not None:
            advance_event.set()

        # --- Step 6: Sample logits and remap dummy → real outputs ---
        if self.enable_nvtx:
            range_push("cg.sample_and_remap", synchronize=False)
        outputs = self._sample_and_remap(
            request_ids=request_ids,
            dummy_rids=dummy_rids,
            static_output=static_output,
            per_request_info=per_request_info,
            slot_data=slot_data,
            submodule=submodule,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # --- Step 7: Restore dummy states ---
        self._restore_dummy_states(
            dummy_rids=dummy_rids,
            request_ids=request_ids,
            real_bs=real_bs,
            config_labels=config_labels,
            static_cm=static_cm,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
            mark("gpu_thread.postprocess_end")

        return outputs

    def _run_flashinfer_packed(
        self,
        key: CudaGraphKey,
        graph_data: CudaGraphData,
        slot_data: CudaGraphSlot,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: ARNodeSubmodule,
        advance_event: "object | None" = None,
        launch_started_event: "object | None" = None,
    ) -> dict:
        """Prefill-style replay (vox-serve pattern).

        Padding slots are zero-length ARNodeInputs — so qo_indptr (re-planned via
        cache_manager.plan_attention inside preprocess) sums to real_num_tokens,
        which FlashInfer's attention path actually walks. Trailing static-buffer
        slots [real_num_tokens : padded_num_tokens] keep their capture-time
        contents; non-attention compute over them is wasted work, not a correctness
        issue. State swap / advance_seq_lens / output remap mirror _run_basic_matched.

        Phase 3: ``slot_data`` selects one of the captured double-buffer slots.
        Prefill paths don't speculate or pre-plan today, so slot alternation
        for these is just round-robin — perf-neutral but consistent with the
        decode path's bookkeeping.
        """
        real_bs = len(request_ids)
        padded_bs = key.bs

        graph = slot_data.graph
        static = slot_data.static_inputs
        static_cm = slot_data.static_cache_manager
        static_output = slot_data.static_outputs

        templates = static["preprocessed"]
        dummy_rids = static["dummy_rids"]
        static_input_keys = static["static_input_keys"]
        config_labels = graph_data.config.labels

        # --- Step 1: Swap real request states onto dummy slots ---
        if self.enable_nvtx:
            mark("gpu_thread.preprocess_start")
            range_push("gpu_thread.preprocess", synchronize=False)
        if self.enable_nvtx:
            range_push("cg.swap_states", synchronize=False)
        for i, rid in enumerate(request_ids):
            dummy_rid = dummy_rids[i]
            for label in config_labels:
                real_state = self.alloc_manager.get_state(rid, label)
                self.alloc_manager.get_state(dummy_rid, label)
                self.alloc_manager.request_states[dummy_rid][label] = real_state

        for i in range(real_bs, padded_bs):
            dummy_rid = dummy_rids[i]
            for label in config_labels:
                self.alloc_manager.get_state(dummy_rid, label)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # --- Step 2: Build padded per-request inputs and re-plan via preprocess ---
        if self.enable_nvtx:
            range_push("cg.preprocess_replan", synchronize=False)
        if self.enable_nvtx:
            range_push("cg.preprocess_replan.pad_inputs", synchronize=False)
        # Unlike basic_matched, prefill captures don't expose a capture_template
        # ARNodeInputs (the config provides post-preprocess packed dicts instead).
        # Synthesize zero-length ARNodeInputs from the first real input's shape so
        # all required tensor fields exist as empty slices for the padding slots.
        padded_inputs = list(inputs)
        for _i in range(real_bs, padded_bs):
            zero_padding_inp = graph_data.config.zero_padding_input
            if zero_padding_inp is None:
                zero_padding_inp = self._zero_padding_input(inputs[0])
            else:
                zero_padding_inp = zero_padding_inp.clone()
            padded_inputs.append(zero_padding_inp)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        if self.enable_nvtx:
            range_push("cg.preprocess_replan.metadata", synchronize=False)
        real_metadata = self._build_replay_metadata(
            dummy_rids, request_ids, real_bs,
            per_request_info, static["dummy_metadata"],
        )
        engine_inputs = ModelInputsFromEngine(
            request_ids=dummy_rids,
            per_request_info=real_metadata,
            cache_manager=static_cm,
            sampler=self._get_sampler(
                per_request_info=per_request_info,
                request_ids=request_ids,
                padded_bs=padded_bs
            )
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("cg.preprocess_replan.submodule_preprocess", synchronize=False)
        real_packed = submodule.preprocess(
            graph_walk=key.graph_walk,
            engine_inputs=engine_inputs,
            inputs=padded_inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # --- Step 3: Copy real packed tensors into static buffers ---
        if self.enable_nvtx:
            range_push("cg.copy_inputs", synchronize=False)
        for k in static_input_keys:
            real_val = real_packed.get(k)
            if real_val is None or not isinstance(real_val, torch.Tensor):
                continue
            static_buf = templates[k]
            static_buf[:real_val.shape[0]].copy_(real_val)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_pop(synchronize=False)
            mark("gpu_thread.preprocess_end")

        # --- Step 4: Replay ---
        if self.enable_nvtx:
            mark("gpu_thread.cuda_graph_start")
            range_push("gpu_thread.cuda_graph", synchronize=False)
            range_push("cg.replay", synchronize=False)
        # Release the main thread now that all CPU-side prep is done and
        # we're about to enter the CUDA driver. graph.replay() drops the
        # GIL inside C++, so main-thread postprocess can overlap.
        if launch_started_event is not None:
            launch_started_event.set()
        graph.replay()
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_pop(synchronize=False)
            mark("gpu_thread.cuda_graph_end")

        # --- Step 5: Advance seq_lens on REAL request states (Python-only) ---
        if self.enable_nvtx:
            mark("gpu_thread.postprocess_start")
            range_push("gpu_thread.postprocess", synchronize=False)
        if self.enable_nvtx:
            range_push("cg.advance_seq_lens", synchronize=False)
        for label in config_labels:
            static_cm.set_active_label(label)
            static_cm.advance_seq_lens()
        if self.enable_nvtx:
            range_pop(synchronize=False)

        if advance_event is not None:
            advance_event.set()

        # --- Step 6: Sample logits and remap dummy → real outputs ---
        if self.enable_nvtx:
            range_push("cg.sample_and_remap", synchronize=False)
        outputs = self._sample_and_remap(
            request_ids=request_ids,
            dummy_rids=dummy_rids,
            static_output=static_output,
            per_request_info=per_request_info,
            slot_data=slot_data,
            submodule=submodule,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # --- Step 7: Restore dummy states ---
        self._restore_dummy_states(
            dummy_rids=dummy_rids,
            request_ids=request_ids,
            real_bs=real_bs,
            config_labels=config_labels,
            static_cm=static_cm,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
            mark("gpu_thread.postprocess_end")

        return outputs

    def _build_replay_metadata(
        self,
        dummy_rids: list[str],
        request_ids: list[str],
        real_bs: int,
        per_request_info: dict[str, CurrentForwardPassInfo],
        dummy_metadata: dict[str, CurrentForwardPassInfo],
    ) -> dict[str, CurrentForwardPassInfo]:
        """Map dummy_rid → real per_request_info for [:real_bs], dummy_metadata
        from capture for [real_bs:]. Used by both replay paths."""
        out = {}
        for i, dummy_rid in enumerate(dummy_rids):
            if i < real_bs:
                out[dummy_rid] = per_request_info[request_ids[i]]
            else:
                out[dummy_rid] = dummy_metadata[dummy_rid]
        return out

    def _zero_padding_input(self, template: ARNodeInputs) -> ARNodeInputs:
        """Synthetic zero-length ARNodeInputs for prefill padding slots.

        Cloned from the first real input's structure so any tensor fields the
        submodule's preprocess expects are present as length-0 slices — preprocess
        can then concatenate them without shape errors. input_seq_len=0 means
        these slots contribute nothing to qo_indptr, so FlashInfer's attention
        skips them at replay.

        ``custom_pos_ids`` and ``tensor_inputs`` may carry the seq dim in
        positions other than 0 (Thinker prefill_text uses ``(3, seq_len)``
        pos_ids and ``(2, seq_len)`` talker masks); ``_zero_seq_dim`` finds
        the matching dim by matching against the template's ``input_seq_len``.
        """
        pad = template.clone()
        seq_len = pad.input_seq_len
        pad.input_seq_len = 0
        if pad.input_ids is not None:
            pad.input_ids = pad.input_ids[:0]
        if pad.input_embeds is not None:
            pad.input_embeds = pad.input_embeds[:0]
        if isinstance(pad.custom_pos_ids, torch.Tensor):
            pad.custom_pos_ids = self._zero_seq_dim(pad.custom_pos_ids, seq_len)
        elif isinstance(pad.custom_pos_ids, dict):
            pad.custom_pos_ids = {
                k: self._zero_seq_dim(v, seq_len) for k, v in pad.custom_pos_ids.items()
            }
        pad.tensor_inputs = {
            k: (self._zero_seq_dim(v, seq_len) if isinstance(v, torch.Tensor) else v)
            for k, v in pad.tensor_inputs.items()
        }
        return pad

    @staticmethod
    def _zero_seq_dim(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Slice the dim whose size matches ``seq_len`` to length 0.

        Used to build padding inputs for prefill: the seq-dim position varies
        across submodules (Thinker pos_ids are ``(3, seq_len)``, talker masks
        are ``(2, seq_len)``, decode tokens are ``(seq_len,)``). Falls back to
        ``tensor[:0]`` if no dim matches — that path is only hit when the
        submodule has tensor fields whose shape is independent of seq_len, in
        which case the resulting empty leading slice is a no-op for the
        submodule's preprocess concat.
        """
        for dim, size in enumerate(tensor.shape):
            if size == seq_len:
                slicer = [slice(None)] * tensor.ndim
                slicer[dim] = slice(0, 0)
                return tensor[tuple(slicer)]
        return tensor[:0]

    def _restore_dummy_states(
        self,
        dummy_rids: list[str],
        request_ids: list[str],
        real_bs: int,
        config_labels: list[str],
        static_cm: BatchedCacheManager,
    ) -> None:
        """Reset every dummy slot's per-label state and flush real-request KV
        writes to the store for any label whose plan_state had write_store enabled."""
        if self.enable_nvtx:
            range_push("cg.restore_states", synchronize=False)
        for i, rid in enumerate(dummy_rids):
            for label in config_labels:
                self.alloc_manager.reset_label(
                    rid, label, free=i >= real_bs,
                )
        for rid in request_ids:
            for label in config_labels:
                ps = static_cm._plan_states.get(label)
                if ps is not None and ps.write_store:
                    self.alloc_manager.flush_to_store(rid, label)
        if self.enable_nvtx:
            range_pop(synchronize=False)

    def _sample_and_remap(
        self,
        request_ids: list[str],
        dummy_rids: list[str],
        static_output: dict,
        per_request_info: dict[str, CurrentForwardPassInfo],
        slot_data: CudaGraphSlot,
        submodule: ARNodeSubmodule,
        inputs: list[ARNodeInputs] | None = None,
    ) -> dict:
        """Sample logits + copy non-logit per-rid outputs, remapping dummy → real rids.

        Fast path: a __batched_logits__ sentinel holding [padded_bs, V] lets us
        sample once via Sampler.sample without per-rid concat. Fallback path
        collects per-rid logits and concatenates. Either way, dummy → real rid
        remap happens here.

        After the per-rid output construction, ``submodule.unpack_packed_outputs``
        is invoked so prefill-style submodules can slice packed sentinels (e.g.
        ``__batched_thinker_states__``) at real per-request seq_len boundaries —
        the captured forward can't do this slicing itself because the slice ends
        depend on real seq_lens, which only land via plan_attention at replay.
        """
        outputs: dict = {}

        # Fast path: submodule exposed the stacked [padded_bs, V] logits tensor
        # under a sentinel key, so we can sample directly without per-rid
        # iteration or torch.cat.
        batched_logits = static_output.get("__batched_logits__")
        if batched_logits is not None:
            stacked_logits = batched_logits[:len(request_ids)]
            # FlashInfer's top-p / top-k sampling reuses an internal output
            # buffer across calls, so iter-N's ``sampled`` tensor address
            # equals iter-(N+k)'s for some small k. With speculation,
            # iter-N's sampled view is held in the routing path (read by
            # slow_post for emit_to_client + check_stop) past the time
            # iter-(N+k) overwrites the buffer — slow_post then reads
            # iter-(N+k)'s token as if it were iter-N's, emitting the same
            # token twice and producing the mid-sequence "X X Y Y Z Z"
            # duplication seen on Qwen3-Omni audio output.
            #
            # The .clone() snapshots the sampled value into a fresh
            # allocation that lives as long as the Python view, breaking
            # the alias.
            sampled = self.sampler.sample(request_ids, stacked_logits).clone()
            sampled_views = sampled.split(1)
            outputs = {
                rid: {"new_token": [view]}
                for rid, view in zip(request_ids, sampled_views, strict=True)
            }

            # Collect non-logit per-rid outputs (e.g. hidden states) only when
            # the captured graph actually produced any — for most AR models
            # (Orpheus included) it only emits logits, so the loop is skipped.
            if slot_data.has_non_logit_outputs:
                for i, rid in enumerate(request_ids):
                    dummy_rid = dummy_rids[i]
                    if dummy_rid not in static_output:
                        continue
                    # Captured dummy output keys are static (graph-compat); ask
                    # the submodule which keys this real request should actually
                    # receive (e.g. Thinker emits thinker_states inside the graph
                    # but drops it here when audio_output=False).
                    filtered = submodule.filter_batched_output(
                        per_request_info.get(rid), static_output[dummy_rid],
                    )
                    for out_key, val in filtered.items():
                        if out_key == "logits":
                            continue
                        if isinstance(val, list):
                            outputs[rid][out_key] = [t.clone() for t in val]
                        elif isinstance(val, torch.Tensor):
                            outputs[rid][out_key] = [val.clone()]
                        else:
                            outputs[rid][out_key] = val
            self._merge_unpacked(
                outputs, static_output, request_ids, inputs,
                per_request_info, submodule,
            )
            return outputs

        # Fallback: collect per-rid logits and concatenate.
        all_logits = []
        non_logit_keys: dict[str, list] = {}
        for i in range(len(request_ids)):
            dummy_rid = dummy_rids[i]
            if dummy_rid not in static_output:
                continue
            dummy_out = static_output[dummy_rid]
            for out_key, val in dummy_out.items():
                if out_key == "logits":
                    logits_t = val[0] if isinstance(val, list) else val
                    all_logits.append(logits_t)
                else:
                    non_logit_keys.setdefault(out_key, []).append((i, val))

        if all_logits:
            stacked_logits = torch.cat(all_logits, dim=0)
            # See clone() rationale in the fast path above — FlashInfer
            # reuses the sampling output buffer across calls, so the view
            # held in routing aliases iter-(N+k)'s value once that iter
            # samples.
            sampled = self.sampler.sample(request_ids, stacked_logits).clone()
            for i, rid in enumerate(request_ids):
                outputs[rid] = {"new_token": [sampled[i:i+1]]}
        else:
            for rid in request_ids:
                outputs[rid] = {}

        for out_key, entries in non_logit_keys.items():
            for idx, val in entries:
                rid = request_ids[idx]
                if isinstance(val, list):
                    outputs[rid][out_key] = [t.clone() for t in val]
                elif isinstance(val, torch.Tensor):
                    outputs[rid][out_key] = [val.clone()]
                else:
                    outputs[rid][out_key] = val

        self._merge_unpacked(
            outputs, static_output, request_ids, inputs,
            per_request_info, submodule,
        )
        return outputs

    def _merge_unpacked(
        self,
        outputs: dict,
        static_output: dict,
        request_ids: list[str],
        inputs: list[ARNodeInputs] | None,
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: ARNodeSubmodule,
    ) -> None:
        """Invoke ``submodule.unpack_packed_outputs`` and merge per-rid keys.

        No-op when ``inputs`` is None (legacy callers that haven't been wired
        through) or when the submodule's hook returns nothing (decode-style
        submodules whose captured forward already emits per-rid entries at
        fixed shape).
        """
        if inputs is None:
            return
        real_seq_lens = [inp.input_seq_len for inp in inputs]
        unpacked = submodule.unpack_packed_outputs(
            static_output=static_output,
            request_ids=request_ids,
            real_seq_lens=real_seq_lens,
            inputs=inputs,
            per_request_info=per_request_info,
        )
        if not unpacked:
            return
        for rid, rid_out in unpacked.items():
            outputs.setdefault(rid, {})
            for k, v in rid_out.items():
                outputs[rid][k] = v


class CodecCudaGraphRunner:
    """CUDA graph capture/replay for stateless batched submodules.

    Contract (matches the AR runner so submodules look similar across engines):

        preprocess(graph_walk, per_request_inputs, request_ids, per_request_info)
            Python-level prep that turns variable list-of-dicts inputs into a
            dict of fixed-shape packed tensors. Runs OUTSIDE the captured
            region both during capture (on dummy inputs built from the
            config's ``single_request_inputs``) and at replay (on real inputs).
            May return an empty dict to signal "can't be batched" — the
            engine falls back to the eager path in that case.

        cuda_graph_forward(**packed_tensors) -> dict[str, torch.Tensor]
            Pure-tensor call captured inside the graph. Output tensors must
            have batch-dim first, same size as the input batch dim, so the
            runner can slice ``[:actual_bs]`` and index per request.

        get_cuda_graph_configs(device) -> list[CudaGraphConfig]
            Each config's ``single_request_inputs`` is a single per-request
            ARNodeInputs (same shape as real runtime inputs). The runner
            clones it per capture batch slot, then feeds the resulting list
            to ``submodule.preprocess``.

    Warmup flow (per config × batch size):
        1. Clone dummy per-request inputs for ``bs`` slots.
        2. Call submodule.preprocess → packed tensors.
        3. Allocate matching static buffers, copy the packed tensors in.
        4. Run 2 warmup forwards outside the graph (kernel compilation).
        5. Capture ``cuda_graph_forward(**static_buffers)``.

    Runtime flow (per ``run(batch, submodule)`` call):
        1. submodule.preprocess on real inputs → packed tensors.
           If it returns {}, raise CodecGraphNotApplicableError so the engine
           can fall back to the eager path.
        2. Copy packed tensors into static buffers (slots [:actual_bs]).
        3. graph.replay().
        4. Slice outputs to [:actual_bs] and split per request.
    """

    DEFAULT_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def __init__(
        self,
        submodule_name: str,
        submodule: nn.Module,
        device: torch.device,
    ):
        self.submodule_name = submodule_name
        self.submodule = submodule
        self.device = device
        self.capture_configs: list[CudaGraphConfig] = (
            submodule.get_cuda_graph_configs(device) if submodule is not None else []
        )

        # Keyed by (graph_walk, padded_bs)
        self.graphs: dict[tuple[str, int], torch.cuda.CUDAGraph] = {}
        self.static_inputs: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.static_outputs: dict[tuple[str, int], Any] = {}
        self.dummy_rids: dict[tuple[str, int], list[str]] = {}
        self.memory_pool = None
        self.enable_nvtx = False

    def warmup_and_capture(self) -> None:
        if not torch.cuda.is_available() or self.device is None:
            logger.warning(
                "CUDA not available, skipping codec graph capture for %s",
                self.submodule_name,
            )
            return
        if not self.capture_configs:
            return

        # Pin the device so torch.cuda.graph's side stream lands on the
        # right GPU — without this, capture on cuda:N>0 dispatches on the
        # default cuda:0 and every captured kernel errors out.
        torch.cuda.set_device(self.device)

        # Warmup AND capture share one side stream. cuDNN/cuBLAS allocate
        # per-stream workspaces on their first kernel call; running warmup
        # on the default stream and then capture on a fresh side stream
        # triggers that allocation mid-capture, which fails with "operation
        # not permitted when stream is capturing". Same-stream warmup makes
        # the workspace land before capture_begin (matches sglang/vllm).
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self.memory_pool = torch.cuda.graphs.graph_pool_handle()

        for config in self.capture_configs:
            sizes = config.capture_batch_sizes or self.DEFAULT_CAPTURE_BATCH_SIZES
            for bs in reversed(sorted(sizes)):
                try:
                    self._capture_one(
                        bs, config, self.submodule
                    )
                    logger.info(
                        "Captured codec CUDA graph for %s: walk=%s bs=%d",
                        self.submodule_name, config.capture_graph_walk, bs,
                    )
                except Exception:
                    logger.warning(
                        "Failed to capture codec CUDA graph for %s: walk=%s bs=%d",
                        self.submodule_name, config.capture_graph_walk, bs, exc_info=True,
                    )

    def _capture_one(
        self, bs: int, config: CudaGraphConfig, submodule: NodeSubmodule
    ) -> None:
        if config.single_request_inputs is None:
            raise ValueError(
                f"{self.submodule_name}: CudaGraphConfig for walk "
                f"{config.capture_graph_walk!r} missing single_request_inputs"
            )

        # Build dummy per-request inputs (same format as real inputs) and
        # route them through the submodule's own preprocess — the AR runner
        # does the same, so the two code paths stay symmetric.
        template = config.single_request_inputs
        dummy_rids = [
            f"__codec_cg_{config.capture_graph_walk}_{i}__" for i in range(bs)
        ]
        dummy_inputs = [template.clone() for _ in dummy_rids]
        dummy_info = {
            rid: CurrentForwardPassInfo(
                request_id=rid,
                graph_walk=config.capture_graph_walk,
                requires_cfg=False,
                fwd_index=0,
                random_seed=0,
                max_tokens=1,
                sampling_config={}
            )
            for rid in dummy_rids
        }
        engine_inputs = ModelInputsFromEngine(
            request_ids=dummy_rids,
            per_request_info=dummy_info,
        )

        packed = submodule.preprocess(
            graph_walk=config.capture_graph_walk,
            engine_inputs=engine_inputs,
            inputs=dummy_inputs,
        )
        if not packed or not all(isinstance(v, torch.Tensor) for v in packed.values()):
            raise RuntimeError(
                f"{self.submodule_name}: preprocess returned non-tensor/empty packed "
                f"inputs during capture (walk={config.capture_graph_walk!r}); cannot capture"
            )

        # Static buffers match the preprocessed shapes (leading dim == bs).
        static_inputs: dict[str, torch.Tensor] = {}
        for name, t in packed.items():
            if t.dim() == 0 or t.shape[0] != bs:
                raise ValueError(
                    f"{self.submodule_name}: preprocess output {name!r} has shape "
                    f"{tuple(t.shape)}; expected leading dim {bs}"
                )
            static_inputs[name] = torch.zeros(t.shape, dtype=t.dtype, device=self.device)
            static_inputs[name].copy_(t)

        fwd = submodule.forward_batched
        if config.compile:
            fwd = torch.compile(
                fwd,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
                dynamic=False,
            )

        # Warmup and capture on the shared side stream (see warmup_and_capture
        # for why). The stream is created in warmup_and_capture before the
        # first _capture_one call.
        stream = self._capture_stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                fwd(
                    graph_walk=config.capture_graph_walk,
                    engine_inputs=engine_inputs,
                    **static_inputs
                )
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.memory_pool, stream=stream):
            static_output = fwd(
                graph_walk=config.capture_graph_walk,
                engine_inputs=engine_inputs,
                **static_inputs
            )
        stream.synchronize()

        for graph_walk in config.replay_graph_walks:
            key = (graph_walk, bs)
            self.graphs[key] = graph
            self.static_inputs[key] = static_inputs
            self.static_outputs[key] = static_output
            self.dummy_rids[key] = dummy_rids

    def _sizes_for(self, graph_walk: str) -> list[int]:
        for cfg in self.capture_configs:
            if cfg.capture_graph_walk == graph_walk:
                return cfg.capture_batch_sizes or self.DEFAULT_CAPTURE_BATCH_SIZES
        return self.DEFAULT_CAPTURE_BATCH_SIZES

    def _get_padded_batch_size(self, batch_size: int, graph_walk: str) -> int | None:
        sizes = self._sizes_for(graph_walk)
        idx = bisect.bisect_left(sizes, batch_size)
        if idx >= len(sizes):
            return None
        return sizes[idx]

    def can_run(self, batch_size: int, graph_walk: str = "decode") -> bool:
        if not self.graphs:
            return False
        padded = self._get_padded_batch_size(batch_size, graph_walk)
        if padded is None:
            return False
        return (graph_walk, padded) in self.graphs

    def run(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: NodeSubmodule,
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """End-to-end replay: preprocess + replay + per-rid output split.

        Argument shape matches ``CudaGraphRunner.run`` (AR) so the two
        runners present the same interface to their engines. ``submodule``
        is passed in at call time (rather than taken from ``self.submodule``)
        for the same reason AR does it — keeps the runtime call site
        self-contained and mirrors AR's contract exactly.
        """
        actual_bs = len(request_ids)
        padded_bs = self._get_padded_batch_size(actual_bs, graph_walk)
        if padded_bs is None:
            raise RuntimeError(
                f"{self.submodule_name}: no captured graph for walk={graph_walk!r}, "
                f"actual_bs={actual_bs}"
            )
        key = (graph_walk, padded_bs)
        static_inputs = self.static_inputs[key]
        static_output = self.static_outputs[key]
        dummy_rids = self.dummy_rids[key]

        engine_inputs = ModelInputsFromEngine(
            request_ids=request_ids,
            per_request_info=per_request_info,
        )

        if self.enable_nvtx:
            mark("gpu_thread.preprocess_start")
            range_push("gpu_thread.preprocess", synchronize=False)
            range_push("codec_cg.preprocess", synchronize=False)
        packed = submodule.preprocess(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)

        if self.enable_nvtx:
            range_push("codec_cg.copy_inputs", synchronize=False)
        for name, real_val in packed.items():
            static_buf = static_inputs.get(name)
            if static_buf is None:
                raise KeyError(
                    f"{self.submodule_name}: preprocess output {name!r} was not present "
                    f"at capture time (expected keys: {list(static_inputs.keys())})"
                )
            static_buf.zero_()
            static_buf[:actual_bs].copy_(real_val)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_pop(synchronize=False)
            mark("gpu_thread.preprocess_end")

        if self.enable_nvtx:
            mark("gpu_thread.cuda_graph_start")
            range_push("gpu_thread.cuda_graph", synchronize=False)
            range_push("codec_cg.replay", synchronize=False)
        self.graphs[key].replay()
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_pop(synchronize=False)
            mark("gpu_thread.cuda_graph_end")

        if not isinstance(static_output, dict):
            raise TypeError(
                f"{self.submodule_name}: cuda_graph_forward must return dict[str, Tensor] "
                f"(got {type(static_output).__name__}) so outputs can be split per request"
            )

        if self.enable_nvtx:
            mark("gpu_thread.postprocess_start")
            range_push("gpu_thread.postprocess", synchronize=False)
        outputs = {
            rid: {
                name: [
                    tensor.clone() for tensor in static_output[dummy_rids[i]][name]
                ] for name in static_output[dummy_rids[i]]
            } for i, rid in enumerate(request_ids)
        }
        if self.enable_nvtx:
            range_pop(synchronize=False)
            mark("gpu_thread.postprocess_end")

        return outputs


# ---------------------------------------------------------------------------
# PiecewiseCudaGraphRunner
# ---------------------------------------------------------------------------

class PiecewiseCudaGraphRunner:
    """Captures a transformer block-loop callable as one CUDA graph per batch-size bucket.

    Designed for the inner block loops of VJepa2 predictors
    (VisionTransformerPredictorAC with KV cache, and VJEPA2Predictor without).
    The caller supplies a ``fn_factory`` that builds the capturable
    ``fn(x) -> x`` closure given a static BatchedCacheManager and a dict of
    pre-allocated position-tensor buffers.  The runner owns those buffers;
    callers update them via ``.copy_()`` through the ``pos_bufs`` argument of
    ``run()``, making per-step position tensors visible to the captured GPU ops
    without creating new tensors inside the captured region.

    Key invariants (matching CudaGraphRunner):
    - FlashInfer wrappers are PERSISTENT (created once per bs bucket at capture).
    - plan_attention is called OUTSIDE the graph before each replay.
    - advance_seq_len is called OUTSIDE the graph after each replay.
    - KV state is swapped onto dummy slots before replay and restored after.
    """

    def __init__(
        self,
        fn_factory: Callable[
            [BatchedCacheManager | None, dict[str, torch.Tensor]],
            Callable[[torch.Tensor], torch.Tensor],
        ],
        embed_dim: int,
        capture_batch_sizes: list[int],
        capture_seq_len: int,
        device: torch.device,
        autocast_dtype: torch.dtype,
        pos_buf_shapes: dict[str, tuple[int, ...]] | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        alloc_manager: PagedAllocationManager | None = None,
        buffer_manager: WorkspaceBufferManager | None = None,
        cache_labels: list[str] | None = None,
    ):
        self.fn_factory = fn_factory
        self.embed_dim = embed_dim
        self.capture_batch_sizes = sorted(capture_batch_sizes)
        self.capture_seq_len = capture_seq_len
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.pos_buf_shapes: dict[str, tuple[int, ...]] = pos_buf_shapes or {}
        self.kv_cache_config = kv_cache_config
        self.alloc_manager = alloc_manager
        self.buffer_manager = buffer_manager
        self.cache_labels: list[str] = cache_labels or ["main"]

        self.graphs: dict[int, PiecewiseGraphData] = {}
        self.memory_pool = None

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def warmup_and_capture(self) -> None:
        if not torch.cuda.is_available() or self.device is None:
            logger.warning("CUDA not available — skipping PiecewiseCudaGraphRunner capture")
            return

        torch.cuda.set_device(self.device)
        self.memory_pool = torch.cuda.graphs.graph_pool_handle()

        for bs in reversed(self.capture_batch_sizes):
            try:
                self._capture_one(bs)
                logger.info("PiecewiseCudaGraphRunner: captured bs=%d seq_len=%d", bs, self.capture_seq_len)
            except Exception:
                logger.warning(
                    "PiecewiseCudaGraphRunner: failed to capture bs=%d", bs, exc_info=True
                )

    def _capture_one(self, bs: int) -> None:
        # Match autocast_dtype so copy_() at replay is a same-dtype memcpy
        # (no silent upcast from bfloat16 → float32 followed by an immediate
        # cast back inside the first linear).
        static_x = torch.zeros(
            bs, self.capture_seq_len, self.embed_dim,
            dtype=self.autocast_dtype, device=self.device,
        )

        # Position buffers stay float32: they hold scalar position indices
        # (frame id, height id, ...) and RoPE uses them as frequencies, where
        # float32 precision matters more than matching the hidden state dtype.
        static_pos_bufs: dict[str, torch.Tensor] = {
            name: torch.zeros(shape, dtype=torch.float32, device=self.device)
            for name, shape in self.pos_buf_shapes.items()
        }

        # KV cache support
        static_cm: BatchedCacheManager | None = None
        dummy_rids: list[str] = []
        if self.kv_cache_config is not None:
            assert self.alloc_manager is not None and self.buffer_manager is not None
            dummy_rids = [f"__pcgr_{bs}_{i}__" for i in range(bs)]
            for rid in dummy_rids:
                self.alloc_manager.add_request(rid, labels=self.cache_labels)

            plan_states = self._build_persistent_wrappers(bs)
            static_cm = BatchedCacheManager(
                request_ids=dummy_rids,
                active_labels_per_request={rid: self.cache_labels[0] for rid in dummy_rids},
                kv_cache=self.alloc_manager.kv_cache,
                alloc_manager=self.alloc_manager,
                buffer_manager=self.buffer_manager,
                kv_cache_config=self.kv_cache_config,
                device=self.device,
                cuda_graph_plan_states=plan_states,
            )

        fn = self.fn_factory(static_cm, static_pos_bufs)

        def _plan():
            if static_cm is not None:
                static_cm.plan_attention(
                    seq_lens=[self.capture_seq_len] * bs,
                    is_causal=False,
                )

        def _reset_dummy_states():
            for rid in dummy_rids:
                for label in self.cache_labels:
                    state = self.alloc_manager.get_state(rid, label)
                    state.seq_len = 0
                    state.position_id_start = 0

        _plan()

        # Warmup — 2 passes
        torch.cuda.synchronize()
        for _ in range(2):
            with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                fn(static_x)
            _reset_dummy_states()
            _plan()
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
            with torch.cuda.graph(graph, pool=self.memory_pool):
                static_out = fn(static_x)
        torch.cuda.synchronize()

        # Free dummy KV state so it doesn't accumulate across bs captures
        for rid in dummy_rids:
            for label in self.cache_labels:
                self.alloc_manager.reset_label(rid, label, free=True)

        self.graphs[bs] = PiecewiseGraphData(
            graph=graph,
            static_x=static_x,
            static_out=static_out,
            static_pos_bufs=static_pos_bufs,
            static_cache_manager=static_cm,
            dummy_rids=dummy_rids,
            bs=bs,
        )

    def _build_persistent_wrappers(self, bs: int) -> dict:
        from mminf.engine.cache_manager import _PlanState
        from mminf.utils.flashinfer_utils import FlashInferPrefillWrapper

        cfg = self.kv_cache_config
        plan_states: dict = {}
        for label in self.cache_labels:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(f"{label}_pcgr_{bs}"),
                num_qo_heads=cfg.num_qo_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                page_size=cfg.page_size,
                batch_size=bs,
                max_total_tokens=bs * self.capture_seq_len,
                max_num_pages=cfg.max_num_pages,
                device=self.device,
                use_cuda_graph=True,
            )
            plan_states[label] = _PlanState(wrapper=wrapper)
        return plan_states

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def can_run(self, batch_size: int) -> bool:
        return bool(self.graphs) and self._padded_bs(batch_size) is not None

    def _padded_bs(self, batch_size: int) -> int | None:
        idx = bisect.bisect_left(self.capture_batch_sizes, batch_size)
        if idx >= len(self.capture_batch_sizes):
            return None
        return self.capture_batch_sizes[idx]

    def run(
        self,
        x: torch.Tensor,                                    # [real_bs, seq_len, D]
        pos_bufs: dict[str, torch.Tensor] | None = None,   # updated into static buffers
        request_ids: list[str] | None = None,
        ) -> torch.Tensor:
        """Replay the captured graph for the given input.

        Steps (mirroring CudaGraphRunner._run_basic_batched):
          1. Copy real x into static buffer.
          2. Update position buffers via .copy_().
          3. Swap real KV states onto dummy slots + plan_attention.
          4. graph.replay().
          5. advance_seq_len (Python-only, outside graph).
          6. Restore dummy states.
          7. Return static_out[:real_bs].clone().
        """
        real_bs = x.size(0)
        padded_bs = self._padded_bs(real_bs)
        if padded_bs is None:
            raise RuntimeError(
                f"PiecewiseCudaGraphRunner: no captured graph for bs={real_bs}"
            )
        data = self.graphs[padded_bs]

        # --- 1: copy input ---
        data.static_x[:real_bs].copy_(x)
        if real_bs < padded_bs:
            data.static_x[real_bs:].zero_()

        # --- 2: update position buffers ---
        if pos_bufs:
            for name, val in pos_bufs.items():
                if name in data.static_pos_bufs:
                    data.static_pos_bufs[name].copy_(val)

        # --- 3: KV state swap + plan_attention ---
        if data.static_cache_manager is not None and request_ids is not None:
            for i, rid in enumerate(request_ids):
                dummy_rid = data.dummy_rids[i]
                for label in self.cache_labels:
                    real_state = self.alloc_manager.get_state(rid, label)
                    self.alloc_manager.get_state(dummy_rid, label)   # ensure slot exists
                    self.alloc_manager.request_states[dummy_rid][label] = real_state
            data.static_cache_manager.plan_attention(
                seq_lens=[self.capture_seq_len] * padded_bs,
                is_causal=False,
            )

        # --- 4: replay ---
        data.graph.replay()

        # --- 5: advance seq_len (Python-only, post-replay) ---
        if data.static_cache_manager is not None and request_ids is not None:
            data.static_cache_manager.advance_seq_len(n=self.capture_seq_len)

        # --- 6: restore dummy states ---
        if data.static_cache_manager is not None and request_ids is not None:
            for i, dummy_rid in enumerate(data.dummy_rids):
                for label in self.cache_labels:
                    self.alloc_manager.reset_label(dummy_rid, label, free=i >= real_bs)

        # --- 7: return output ---
        return data.static_out[:real_bs].clone()
