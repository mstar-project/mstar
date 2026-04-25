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

Also provides EncDecCudaGraphWrapper for stateless encoder/decoder submodules.
"""

import bisect
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
from mminf.engine.cuda_graph_config import BasicBatchedCudaGraphConfig, CudaGraphConfig, CudaGraphConfigType, FlashInferPackedCudaGraphConfig
from mminf.engine.kv_store import KVCacheConfig, PagedAllocationManager
from mminf.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, ARNodeSubmodule, NodeSubmodule
from mminf.utils.profiler import range_pop, range_push
from mminf.utils.sampling import Sampler

logger = logging.getLogger(__name__)


DEFAULT_AR_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class DummyCaptureInput:
    tensors: dict[str, list[torch.Tensor]]  # {tensor_name: [tensor(s)]}
    seq_len: int | None = field(default=None)


@dataclass
class CudaGraphData:
    graph: torch.cuda.CUDAGraph
    static_inputs: dict
    static_outputs: dict
    static_cache_manager: BatchedCacheManager
    config: CudaGraphConfig
    bs: int
    # Cached at capture time: True iff any dummy_rid in static_outputs has a
    # key besides "logits". When False, sample_and_remap can skip the
    # per-rid collection loop entirely.
    has_non_logit_outputs: bool = False


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
                            self._capture_one_basic_matched(
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

    def _create_persistent_wrappers(
        self, bs: int, config: CudaGraphConfig,
        total_tokens: int
    ) -> dict:
        """Create persistent FlashInfer wrappers for CUDA graph capture.

        Returns dict of label -> _PlanState with persistent wrappers.
        """
        from mminf.engine.cache_manager import _PlanState
        from mminf.utils.flashinfer_utils import (
            FlashInferDecodeWrapper,
            FlashInferPrefillWrapper,
        )

        is_decode = (total_tokens == bs)

        cfg = self.kv_cache_config

        # Allocate workspace buffer for CUDA graph wrappers.
        # Each label gets its own workspace to avoid conflicts during
        # multi-pass captures (e.g., main + cfg_img in same graph).
        plan_states = {}
        for label in config.labels:
            if is_decode:
                wrapper = FlashInferDecodeWrapper(
                    workspace_buffer=self.buffer_manager.get(f"{label}_cugraph"),
                    num_qo_heads=cfg.num_qo_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                    page_size=cfg.page_size,
                    batch_size=bs,
                    max_num_pages=cfg.max_num_pages,
                    device=self.device,
                    use_cuda_graph=True,
                )
            else:
                wrapper = FlashInferPrefillWrapper(
                    workspace_buffer=self.buffer_manager.get(f"{label}_cugraph"),
                    num_qo_heads=cfg.num_qo_heads,
                    num_kv_heads=cfg.num_kv_heads,
                    head_dim=cfg.head_dim,
                    page_size=cfg.page_size,
                    batch_size=bs,
                    max_total_tokens=total_tokens,
                    max_num_pages=cfg.max_num_pages,
                    device=self.device,
                    use_cuda_graph=True,
                )

            # Static pos_ids buffer for RoPE
            static_pos_ids = torch.zeros(
                total_tokens, dtype=torch.long, device=self.device
            )

            plan_states[label] = _PlanState(
                wrapper=wrapper,
                pos_ids=static_pos_ids,
            )

        return plan_states
    
    def _make_dummy_rids(self, config: CudaGraphConfig, bs: int):
        dummy_rids = [f"__cg_{config.capture_graph_walk}_{config.requires_cfg}_{i}__"
                      for i in range(bs)]

        # Add dummy requests with all needed labels
        for rid in dummy_rids:
            self.alloc_manager.add_request(rid, labels=config.labels)
        return dummy_rids
    
    def _free_dummy_rids(self, config: CudaGraphConfig, dummy_rids: list[str]):
        for rid in dummy_rids:
            for label in config.labels:
                self.alloc_manager.reset_label(rid, label, free=True)
    
    def _clone_template(self, template):
        if isinstance(template, torch.Tensor):
            return template.clone()
        elif isinstance(template, list):
            return [self._clone_template(t) for t in template]
        elif isinstance(template, dict):
            return {k: self._clone_template(v) for k, v in template.items()}
        else:
            raise ValueError(f"Unsupported template type: {type(template)}")
    
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
            auto_write_store=False
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
            cache_manager=cache_manager
        )
    
    def _make_dummy_seq_lens(
        self, bs: int,
        total_tokens: int, # total tokens per batch
    )-> list[int]:
        # must ensure that the seq lens array sums to total tokens
        seq_lens = [total_tokens // bs] * bs
        seq_lens[0] += total_tokens % bs
        return seq_lens
    
    def _postprocess_cuda_graph_output(
        self, output, config: CudaGraphConfig,
        key: CudaGraphKey, graph, static_inputs,
        cache_manager, bs
    ):
        # Inspect per-rid output keys once at capture time so sample_and_remap
        # can skip its per-rid collection loop when only logits are present.
        # Skip only the __batched_logits__ sentinel — the per-rid entries
        # (dummy_rids, also "__"-prefixed) ARE what we want to inspect.
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
        logger.info(
            "CudaGraphRunner: captured graph %s has_non_logit_outputs=%s",
            key, has_non_logit,
        )

        for graph_walk in config.replay_graph_walks:
            lookup_key = CudaGraphKey(
                graph_walk=graph_walk,
                requires_cfg=config.requires_cfg,
                bs=bs,
                num_tokens=key.num_tokens,
            )
            self.graphs[lookup_key] = CudaGraphData(
                graph=graph,
                static_inputs=static_inputs,
                static_outputs=output,
                static_cache_manager=cache_manager,
                config=config,
                bs=bs,
                has_non_logit_outputs=has_non_logit,
            )

        logger.debug("Captured graph %s, output keys: %s", key,
                        list(output.keys()) if isinstance(output, dict)
                        else type(output))

    def _capture_one_flashinfer_packed(
        self, key: CudaGraphKey,
        config: FlashInferPackedCudaGraphConfig,
        submodule: ARNodeSubmodule,
    ):
        """Capture one prefill graph for (bs, num_tokens) bucket.

        submodule.preprocess is NOT called at capture: config.num_token_to_inputs[num_tokens]
        is the packed input already in the format forward_batched expects post-preprocess.
        The runner plans FlashInfer attention/RoPE itself with synthetic seq_lens and
        captures forward_batched directly. At replay (_run_flashinfer_packed), preprocess IS
        called on real per-request inputs to fill the static buffers.
        """
        bs = key.bs

        # Create dummy request IDs
        dummy_rids = self._make_dummy_rids(config, bs)
        try:
            # TODO reuse buffers
            templates = self._clone_template(
                config.num_token_to_inputs[key.num_tokens]
            )

            plan_states = self._create_persistent_wrappers(
                bs, config, total_tokens=key.num_tokens
            )
            seq_lens = self._make_dummy_seq_lens(bs, key.num_tokens)

            # Build cache manager FIRST so plan_attention can close over it.
            engine_inputs = self._create_cache_mgr_and_dummy_engine_inputs(
                dummy_rids=dummy_rids,  plan_states=plan_states, config=config
            )
            cache_manager = engine_inputs.cache_manager

            # manually plan attention
            def plan_attention():
                for label in config.labels:
                    cache_manager.plan_attention(
                        seq_lens=seq_lens,
                        is_causal=config.causal_attention,
                        label=label, write_store=False
                    )
                    cache_manager.plan_rope(
                        seq_lens=seq_lens, label=label
                    )

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

            def run_forward():
                return forward(
                    graph_walk=config.capture_graph_walk,
                    engine_inputs=engine_inputs,
                    **templates
                )
            
            torch.cuda.set_device(self.device)
            # Warmup: 2 forward passes
            torch.cuda.synchronize()
            for _ in range(2):
                with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                    output = run_forward()
                # Reset seq_lens after warmup passes so capture starts clean
                for rid in dummy_rids:
                    for label in config.labels:
                        state = self.alloc_manager.get_state(rid, label)
                        state.seq_len = 0
                        state.position_id_start = 0
                # Re-plan after reset
                plan_attention()
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                with torch.cuda.graph(graph, pool=self.memory_pool):
                    output = run_forward()
            torch.cuda.synchronize()

            self._postprocess_cuda_graph_output(
                output=output, config=config, key=key,
                graph=graph, static_inputs={
                    "preprocessed": templates,
                    "static_input_keys": static_input_keys,
                    "dummy_rids": dummy_rids,
                    "dummy_metadata": engine_inputs.per_request_info,
                },
                cache_manager=cache_manager, bs=bs
            )

        finally:
            self._free_dummy_rids(config, dummy_rids)   


    def _capture_one_basic_matched(
        self, key: CudaGraphKey,
        config: BasicBatchedCudaGraphConfig,
        submodule: ARNodeSubmodule,
    ) -> None:
        """Capture one decode graph for (bs, single_request_inputs.input_seq_len * bs) bucket.

        submodule.preprocess IS called at capture with bs cloned single_request_inputs;
        its output gets captured as static buffers. This is the only path where preprocess
        logic (including plan_attention/plan_rope) runs inside the captured region — at
        replay those are re-planned outside the graph.
        """
        bs = key.bs

        # Create dummy request IDs
        dummy_rids = self._make_dummy_rids(config, bs)

        try:
            # Build dummy per-request inputs via the submodule's own
            # capture-input generator. This lets each submodule declare what
            # dummy tensors its preprocess() needs (e.g., Thinker decode needs
            # a dummy token, Talker decode needs dummy all_codes).
            template = config.single_request_inputs
            if template is None:
                logger.warning(
                    "%s.get_cuda_graph_configs returned a BasicBatchedCudaGraphConfig "
                    "with single_request_inputs=None for walk=%s — skipping capture",
                    self.submodule_name, config.capture_graph_walk,
                )
                return
            
            dummy_inputs = [template.clone() for _ in dummy_rids]

            plan_states = self._create_persistent_wrappers(
                bs, config, total_tokens=bs * template.input_seq_len
            )

            engine_inputs = self._create_cache_mgr_and_dummy_engine_inputs(
                dummy_rids=dummy_rids,  plan_states=plan_states, config=config
            )
            cache_manager = engine_inputs.cache_manager

            # Preprocess (plans attention+rope outside graph)
            preprocessed = submodule.preprocess(
                graph_walk=config.capture_graph_walk,
                engine_inputs=engine_inputs,
                inputs=dummy_inputs,
            )

            # Static input buffers for ALL tensor inputs in the preprocessed
            # dict. These are the slots that will be overwritten with real
            # inputs during replay. Non-tensor values (lists, ints) are kept
            # in `preprocessed` as-is since they're fixed at capture time.
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

            def run_forward():
                return forward(
                    graph_walk=config.capture_graph_walk,
                    engine_inputs=engine_inputs,
                    **preprocessed
                )

            torch.cuda.set_device(self.device)
            # Warmup: 2 forward passes
            torch.cuda.synchronize()
            for _ in range(2):
                with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                    output = run_forward()
                # Reset seq_lens after warmup passes so capture starts clean
                for rid in dummy_rids:
                    for label in config.labels:
                        state = self.alloc_manager.get_state(rid, label)
                        state.seq_len = 0
                        state.position_id_start = 0
                # Re-plan after reset
                submodule.preprocess(
                    graph_walk=config.capture_graph_walk,
                    engine_inputs=engine_inputs,
                    inputs=dummy_inputs,
                )
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                with torch.cuda.graph(graph, pool=self.memory_pool):
                    output = run_forward()
            torch.cuda.synchronize()

            self._postprocess_cuda_graph_output(
                output=output, config=config, key=key,
                graph=graph, static_inputs={
                    "preprocessed": preprocessed,
                    "capture_template": template,
                    "static_input_keys": static_input_keys,
                    "dummy_rids": dummy_rids,
                    "dummy_metadata": engine_inputs.per_request_info,
                },
                cache_manager=cache_manager, bs=bs
            )
           
        finally:
            self._free_dummy_rids(config, dummy_rids)

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


    def run(
        self,
        graph_walk: str,
        requires_cfg: bool,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: ARNodeSubmodule,
    ) -> dict:
        """Look up the matching captured graph and dispatch on config type.

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
        cfg_type = graph_data.config.get_config_type()
        if cfg_type == CudaGraphConfigType.BASIC_BATCHED:
            return self._run_basic_matched(
                key, graph_data, request_ids, inputs, per_request_info, submodule,
            )
        if cfg_type == CudaGraphConfigType.FLASH_INFER_PACKED:
            return self._run_flashinfer_packed(
                key, graph_data, request_ids, inputs, per_request_info, submodule,
            )
        raise ValueError(f"Unknown CudaGraphConfigType: {cfg_type}")

    def _run_basic_matched(
        self,
        key: CudaGraphKey,
        graph_data: CudaGraphData,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: ARNodeSubmodule,
    ) -> dict:
        """Decode-style replay. Pads real inputs to padded_bs by cloning the capture
        template, then routes through submodule.preprocess (which re-plans attention
        and RoPE on the static cache manager) and copies the resulting packed tensors
        into the static buffers before replay.
        """
        real_bs = len(request_ids)
        padded_bs = key.bs

        graph = graph_data.graph
        static = graph_data.static_inputs
        static_cm = graph_data.static_cache_manager
        static_output = graph_data.static_outputs

        preprocessed = static["preprocessed"]
        dummy_rids = static["dummy_rids"]
        static_input_keys = static["static_input_keys"]
        capture_template = static["capture_template"]
        config_labels = graph_data.config.labels

        # --- Step 1: Swap real request states onto dummy slots ---
        if self.enable_nvtx:
            range_push("cg.swap_states", synchronize=True)
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
            range_pop(synchronize=True)

        # --- Step 2: Pad inputs to padded_bs and re-plan via preprocess ---
        if self.enable_nvtx:
            range_push("cg.preprocess_replan", synchronize=True)
        real_inputs = list(inputs)
        # Padding slots reuse the capture_template so submodule.preprocess sees the
        # same input shape it saw at capture time and doesn't crash on missing keys.
        for _i in range(real_bs, padded_bs):
            real_inputs.append(capture_template.clone())

        real_metadata = self._build_replay_metadata(
            dummy_rids, request_ids, real_bs,
            per_request_info, static["dummy_metadata"],
        )
        engine_inputs = ModelInputsFromEngine(
            request_ids=dummy_rids,
            per_request_info=real_metadata,
            cache_manager=static_cm,
        )
        real_inputs = submodule.preprocess(
            graph_walk=key.graph_walk,
            engine_inputs=engine_inputs,
            inputs=real_inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 3: Copy real packed tensors into static buffers ---
        if self.enable_nvtx:
            range_push("cg.copy_inputs", synchronize=True)
        for k in static_input_keys:
            real_val = real_inputs.get(k)
            if real_val is None or not isinstance(real_val, torch.Tensor):
                continue
            static_buf = preprocessed[k]
            static_buf[:real_val.shape[0]].copy_(real_val)
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 4: Replay ---
        if self.enable_nvtx:
            range_push("cg.replay")
        graph.replay()
        if self.enable_nvtx:
            range_pop()

        # --- Step 5: Advance seq_lens on REAL request states (Python-only) ---
        # advance_seq_lens is not captured in the graph; we call it manually so
        # the real states (aliased onto dummy slots) move forward.
        if self.enable_nvtx:
            range_push("cg.advance_seq_lens", synchronize=True)
        for label in config_labels:
            static_cm.set_active_label(label)
            static_cm.advance_seq_lens()
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 6: Sample logits and remap dummy → real outputs ---
        if self.enable_nvtx:
            range_push("cg.sample_and_remap", synchronize=True)
        outputs = self._sample_and_remap(
            request_ids=request_ids,
            dummy_rids=dummy_rids,
            static_output=static_output,
            per_request_info=per_request_info,
            graph_data=graph_data,
            submodule=submodule,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 7: Restore dummy states ---
        self._restore_dummy_states(
            dummy_rids=dummy_rids,
            request_ids=request_ids,
            real_bs=real_bs,
            config_labels=config_labels,
            static_cm=static_cm,
        )

        return outputs

    def _run_flashinfer_packed(
        self,
        key: CudaGraphKey,
        graph_data: CudaGraphData,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: ARNodeSubmodule,
    ) -> dict:
        """Prefill-style replay (vox-serve pattern).

        Padding slots are zero-length ARNodeInputs — so qo_indptr (re-planned via
        cache_manager.plan_attention inside preprocess) sums to real_num_tokens,
        which FlashInfer's attention path actually walks. Trailing static-buffer
        slots [real_num_tokens : padded_num_tokens] keep their capture-time
        contents; non-attention compute over them is wasted work, not a correctness
        issue. State swap / advance_seq_lens / output remap mirror _run_basic_matched.
        """
        real_bs = len(request_ids)
        padded_bs = key.bs

        graph = graph_data.graph
        static = graph_data.static_inputs
        static_cm = graph_data.static_cache_manager
        static_output = graph_data.static_outputs

        templates = static["preprocessed"]
        dummy_rids = static["dummy_rids"]
        static_input_keys = static["static_input_keys"]
        config_labels = graph_data.config.labels

        # --- Step 1: Swap real request states onto dummy slots ---
        if self.enable_nvtx:
            range_push("cg.swap_states", synchronize=True)
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
            range_pop(synchronize=True)

        # --- Step 2: Build padded per-request inputs and re-plan via preprocess ---
        if self.enable_nvtx:
            range_push("cg.preprocess_replan", synchronize=True)
        # Unlike basic_matched, prefill captures don't expose a capture_template
        # ARNodeInputs (the config provides post-preprocess packed dicts instead).
        # Synthesize zero-length ARNodeInputs from the first real input's shape so
        # all required tensor fields exist as empty slices for the padding slots.
        padded_inputs = list(inputs)
        for _i in range(real_bs, padded_bs):
            padded_inputs.append(self._zero_padding_input(inputs[0]))

        real_metadata = self._build_replay_metadata(
            dummy_rids, request_ids, real_bs,
            per_request_info, static["dummy_metadata"],
        )
        engine_inputs = ModelInputsFromEngine(
            request_ids=dummy_rids,
            per_request_info=real_metadata,
            cache_manager=static_cm,
        )
        real_packed = submodule.preprocess(
            graph_walk=key.graph_walk,
            engine_inputs=engine_inputs,
            inputs=padded_inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 3: Copy real packed tensors into static buffers ---
        if self.enable_nvtx:
            range_push("cg.copy_inputs", synchronize=True)
        for k in static_input_keys:
            real_val = real_packed.get(k)
            if real_val is None or not isinstance(real_val, torch.Tensor):
                continue
            static_buf = templates[k]
            static_buf[:real_val.shape[0]].copy_(real_val)
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 4: Replay ---
        if self.enable_nvtx:
            range_push("cg.replay")
        graph.replay()
        if self.enable_nvtx:
            range_pop()

        # --- Step 5: Advance seq_lens on REAL request states (Python-only) ---
        if self.enable_nvtx:
            range_push("cg.advance_seq_lens", synchronize=True)
        for label in config_labels:
            static_cm.set_active_label(label)
            static_cm.advance_seq_lens()
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 6: Sample logits and remap dummy → real outputs ---
        if self.enable_nvtx:
            range_push("cg.sample_and_remap", synchronize=True)
        outputs = self._sample_and_remap(
            request_ids=request_ids,
            dummy_rids=dummy_rids,
            static_output=static_output,
            per_request_info=per_request_info,
            graph_data=graph_data,
            submodule=submodule,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 7: Restore dummy states ---
        self._restore_dummy_states(
            dummy_rids=dummy_rids,
            request_ids=request_ids,
            real_bs=real_bs,
            config_labels=config_labels,
            static_cm=static_cm,
        )

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
            range_push("cg.restore_states", synchronize=True)
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
            range_pop(synchronize=True)

    def _sample_and_remap(
        self,
        request_ids: list[str],
        dummy_rids: list[str],
        static_output: dict,
        per_request_info: dict[str, CurrentForwardPassInfo],
        graph_data: CudaGraphData,
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
            # Sampler.sample returns the raw [B] tokens tensor. FlashInfer
            # allocates a fresh output each call (not captured in the CUDA
            # graph), so the per-rid views are valid for the lifetime of the
            # Python reference — no .clone() needed.
            sampled = self.sampler.sample(request_ids, stacked_logits)
            sampled_views = sampled.split(1)
            outputs = {
                rid: {"new_token": [view]}
                for rid, view in zip(request_ids, sampled_views, strict=True)
            }

            # Collect non-logit per-rid outputs (e.g. hidden states) only when
            # the captured graph actually produced any — for most AR models
            # (Orpheus included) it only emits logits, so the loop is skipped.
            if graph_data.has_non_logit_outputs:
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
            sampled = self.sampler.sample(request_ids, stacked_logits)
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
            range_push("codec_cg.preprocess", synchronize=True)
        packed = submodule.preprocess(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        if self.enable_nvtx:
            range_push("codec_cg.copy_inputs", synchronize=True)
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
            range_pop(synchronize=True)

        if self.enable_nvtx:
            range_push("codec_cg.replay")
        self.graphs[key].replay()
        if self.enable_nvtx:
            range_pop()

        if not isinstance(static_output, dict):
            raise TypeError(
                f"{self.submodule_name}: cuda_graph_forward must return dict[str, Tensor] "
                f"(got {type(static_output).__name__}) so outputs can be split per request"
            )

        return {
            rid: {
                name: [
                    tensor.clone() for tensor in static_output[dummy_rids[i]][name]
                ] for name in static_output[dummy_rids[i]]
            } for i, rid in enumerate(request_ids)
        }
