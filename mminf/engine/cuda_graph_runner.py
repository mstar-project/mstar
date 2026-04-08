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
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
from mminf.engine.kv_store import KVCacheConfig, PagedAllocationManager

logger = logging.getLogger(__name__)


@dataclass
class CudaGraphConfig:
    """Defines what computation a captured graph represents."""
    graph_walk: str  # "decode"
    requires_cfg: bool  # whether CFG is active
    labels: list[str]  # cache labels used: ["main"] or ["main", "cfg_img"]


@dataclass
class CudaGraphData:
    graph: torch.cuda.CUDAGraph
    static_inputs: dict
    static_outputs: dict
    static_cache_manager: BatchedCacheManager
    config: CudaGraphConfig
    bs: int


# Pre-defined configs for Option A
# TODO: have the model declare this itself
DECODE_NO_CFG = CudaGraphConfig(
    graph_walk="decode", requires_cfg=False, labels=["main"]
)
DECODE_WITH_CFG = CudaGraphConfig(
    graph_walk="decode", requires_cfg=True, labels=["main", "cfg_img"]
)

# All configs to capture during warmup
CAPTURE_CONFIGS = [DECODE_NO_CFG, DECODE_WITH_CFG]


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

    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(
        self,
        submodule_name: str,
        submodule: nn.Module,
        kv_cache_config: KVCacheConfig,
        alloc_manager: PagedAllocationManager,
        buffer_manager: WorkspaceBufferManager,
        device: torch.device,
        autocast_dtype: torch.dtype
    ):
        self.submodule_name = submodule_name
        self.submodule = submodule
        self.kv_cache_config = kv_cache_config
        self.alloc_manager = alloc_manager
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.buffer_manager = buffer_manager

        # Keyed by (graph_walk, requires_cfg, batch_size)
        self.graphs: dict[tuple, CudaGraphData] = {}

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

        for config in CAPTURE_CONFIGS:
            for bs in reversed(self.CAPTURE_BATCH_SIZES):
                key = (config.graph_walk, config.requires_cfg, bs)
                try:
                    self._capture_one(bs, config, self.submodule)
                    logger.info("Captured CUDA graph for %s: %s bs=%d",
                                self.submodule_name, key, bs)
                except Exception:
                    logger.warning(
                        "Failed to capture CUDA graph for %s: %s bs=%d",
                        self.submodule_name, key, bs, exc_info=True)

    def _create_persistent_wrappers(
        self, bs: int, config: CudaGraphConfig
    ) -> dict:
        """Create persistent FlashInfer wrappers for CUDA graph capture.

        Returns dict of label -> _PlanState with persistent wrappers.
        """
        from mminf.engine.cache_manager import _PlanState
        from mminf.utils.flashinfer_utils import (
            FlashInferDecodeWrapper,
            FlashInferPrefillWrapper,
        )

        cfg = self.kv_cache_config
        # For decode: each request has 1 new token
        total_tokens = bs

        # Allocate workspace buffer for CUDA graph wrappers.
        # Each label gets its own workspace to avoid conflicts during
        # multi-pass captures (e.g., main + cfg_img in same graph).
        plan_states = {}
        for label in config.labels:
            if config.graph_walk == "decode":
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

    def _capture_one(
        self, bs: int, config: CudaGraphConfig, submodule
    ) -> None:
        """Capture a single CUDA graph for the given batch size and config."""
        from mminf.engine.cache_manager import BatchedCacheManager

        cfg = self.kv_cache_config
        key = (config.graph_walk, config.requires_cfg, bs)

        # Create dummy request IDs
        dummy_rids = [f"__cg_{config.graph_walk}_{config.requires_cfg}_{i}__"
                      for i in range(bs)]

        # Add dummy requests with all needed labels
        for rid in dummy_rids:
            self.alloc_manager.add_request(rid, labels=config.labels)

        try:
            # Create persistent wrappers
            plan_states = self._create_persistent_wrappers(bs, config)

            # Create BatchedCacheManager with CUDA graph plan states
            cache_manager = BatchedCacheManager(
                request_ids=dummy_rids,
                active_labels_per_request={rid: "main" for rid in dummy_rids},
                kv_cache=self.alloc_manager.kv_cache,
                alloc_manager=self.alloc_manager,
                buffer_manager=self.buffer_manager,
                kv_cache_config=cfg,
                device=self.device,
                cuda_graph_plan_states=plan_states,
                auto_write_store=False
            )

            # Build dummy per-request inputs
            dummy_inputs = [
                {"text_inputs": [torch.zeros(1, dtype=torch.long,
                                             device=self.device)]}
                for _ in dummy_rids
            ]

            # Build per-request metadata
            dummy_metadata = {
                rid: CurrentForwardPassInfo(
                    graph_walk=config.graph_walk,
                    requires_cfg=config.requires_cfg,
                    fwd_index=0,
                    random_seed=0,
                    per_label_seq_info={}
                ) for rid in dummy_rids
            }

            # Preprocess (plans attention+rope outside graph)
            preprocessed = submodule.preprocess(
                graph_walk=config.graph_walk,
                cache_manager=cache_manager,
                per_request_inputs=dummy_inputs,
                request_ids=dummy_rids,
                per_request_info=dummy_metadata,
            )

            # Static input buffer for the concatenated embeddings
            static_text_inputs = preprocessed["text_inputs"].clone()

            forward = torch.compile(
                submodule.forward_batched,
                mode="max-autotune-no-cudagraphs",
                fullgraph=False,
                dynamic=False,
            )

            def run_forward():
                return forward(
                    graph_walk=config.graph_walk,
                    cache_manager=cache_manager,
                    packed_inputs=preprocessed,
                    request_ids=dummy_rids,
                    per_request_info=dummy_metadata,
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
                        state.seq_len = max(0, state.seq_len - 1)
                        state.position_id_start = max(
                            0, state.position_id_start - 1)
                # Re-plan after reset
                submodule.preprocess(
                    graph_walk=config.graph_walk,
                    cache_manager=cache_manager,
                    per_request_inputs=dummy_inputs,
                    request_ids=dummy_rids,
                    per_request_info=dummy_metadata,
                )
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                with torch.cuda.graph(graph, pool=self.memory_pool):
                    output = run_forward()
            torch.cuda.synchronize()

            self.graphs[key] = CudaGraphData(
                graph=graph,
                static_inputs={
                    "preprocessed": preprocessed,
                    "static_text_inputs": static_text_inputs,
                    "dummy_rids": dummy_rids,
                    "dummy_metadata": dummy_metadata,
                },
                static_outputs=output,
                static_cache_manager=cache_manager,
                config=config,
                bs=bs
            )

            logger.debug("Captured graph %s, output keys: %s", key,
                         list(output.keys()) if isinstance(output, dict)
                         else type(output))
        finally:
            # Clean up dummy requests
            for rid in dummy_rids:
                for label in config.labels:
                    self.alloc_manager.reset_label(rid, label, free=True)

    def can_run(
        self,
        batch_size: int,
        graph_walk: str = "decode",
        requires_cfg: bool = False,
    ) -> bool:
        """Check if a captured graph exists for this configuration."""
        if not self.graphs:
            return False
        padded_bs = self._get_padded_batch_size(batch_size)
        if padded_bs is None:
            return False
        key = (graph_walk, requires_cfg, padded_bs)
        return key in self.graphs

    def _get_padded_batch_size(self, batch_size: int) -> int | None:
        """Find smallest captured batch size >= batch_size."""
        idx = bisect.bisect_left(self.CAPTURE_BATCH_SIZES, batch_size)
        if idx >= len(self.CAPTURE_BATCH_SIZES):
            return None
        return self.CAPTURE_BATCH_SIZES[idx]

    def run(
        self,
        graph_walk: str,
        requires_cfg: bool,
        request_ids: list[str],
        per_request_inputs: list[dict],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: Any,
    ) -> dict:
        """Run using a captured CUDA graph.

        Steps:
        1. Look up the right graph by (graph_walk, requires_cfg, padded_bs)
        2. Add real requests temporarily, re-plan wrappers with real pages
        3. Copy real input embeddings into static buffers
        4. graph.replay()
        5. advance_seq_lens on real request states (not captured)
        6. Clone outputs and remap dummy -> real request IDs
        7. Clean up temporary request states
        """
        batch_size = len(request_ids)
        padded_bs = self._get_padded_batch_size(batch_size)
        key = (graph_walk, requires_cfg, padded_bs)

        graph_data: CudaGraphData = self.graphs[key]
        graph = graph_data.graph
        static = graph_data.static_inputs
        static_cm = graph_data.static_cache_manager
        static_output = graph_data.static_outputs

        preprocessed = static["preprocessed"]
        dummy_rids = static["dummy_rids"]
        config_labels = graph_data.config.labels

        # --- Step 1: Set up real request states on dummy request IDs ---
        # Save the dummy states, swap in real request states
        for i, rid in enumerate(request_ids):
            dummy_rid = dummy_rids[i]
            for label in config_labels:
                real_state = self.alloc_manager.get_state(rid, label)
                # makes state if it doesn't exist
                self.alloc_manager.get_state(dummy_rid, label)
                self.alloc_manager.request_states[dummy_rid][label] = real_state

        # For padding slots (i >= batch_size), ensure dummy states exist
        for i in range(batch_size, padded_bs):
            dummy_rid = dummy_rids[i]
            for label in config_labels:
                # makes state if it doesn't exist
                self.alloc_manager.get_state(dummy_rid, label)

        # --- Step 2: Re-plan with real page tables (outside graph) ---
        # Build real per-request inputs for the real slots
        real_inputs = []
        for i in range(batch_size):
            real_inputs.append(per_request_inputs[i])
        # Pad with dummy inputs for remaining slots (use a dummy token tensor
        # so submodule.preprocess doesn't crash on empty lists)
        dummy_token = torch.zeros(1, dtype=torch.long, device=self.device)
        for _i in range(batch_size, padded_bs):
            real_inputs.append(
                {"text_inputs": [dummy_token]}
            )

        # Update metadata for real requests
        real_metadata = {}
        for i, dummy_rid in enumerate(dummy_rids):
            if i < batch_size:
                real_metadata[dummy_rid] = per_request_info[request_ids[i]]
            else:
                real_metadata[dummy_rid] = static["dummy_metadata"][dummy_rid]

        # Preprocess re-plans attention+rope with real page tables
        real_inputs = submodule.preprocess(
            graph_walk=graph_walk,
            cache_manager=static_cm,
            per_request_inputs=real_inputs,
            request_ids=dummy_rids,
            per_request_info=real_metadata,
        )

        # --- Step 3: Copy real embeddings to static buffer ---
        real_text = real_inputs["text_inputs"]
        preprocessed["text_inputs"][:real_text.shape[0]].copy_(real_text)

        # --- Step 4: Replay ---
        graph.replay()
        torch.cuda.default_stream().synchronize()

        # --- Step 5: Advance seq_lens on REAL request states ---
        # During replay, advance_seq_lens ran on dummy states (which point
        # to real states), so seq_lens are already advanced. But since
        # advance_seq_lens is Python-only and NOT captured in the graph,
        # we need to call it manually here.
        for label in config_labels:
            static_cm.set_active_label(label)
            # advance_seq_lens uses planned seq_lens (all 1 for decode)
            static_cm.advance_seq_lens()

        # --- Step 6: Sample from logits (outside graph) and remap outputs ---
        from mminf.utils.sampling import sample_tokens
        outputs = {}
        for i, rid in enumerate(request_ids):
            dummy_rid = dummy_rids[i]
            if dummy_rid in static_output:
                dummy_out = static_output[dummy_rid]
                outputs[rid] = {}
                for out_key, val in dummy_out.items():
                    if out_key == "logits":
                        # Sample token from logits (post-graph, CUDA-graph safe)
                        logits = val[0] if isinstance(val, list) else val
                        meta = per_request_info[rid].step_metadata
                        token = sample_tokens(
                            logits,
                            temperature=meta.get("temperature", 0.6),
                            top_k=meta.get("top_k", 0),
                            top_p=meta.get("top_p", 1.0),
                            repetition_penalty=meta.get("repetition_penalty", 1.0),
                            seen_token_ids=meta.get("seen_token_ids", None),
                        )
                        outputs[rid]["new_token"] = [token.clone()]
                    elif isinstance(val, list):
                        outputs[rid][out_key] = [t.clone() for t in val]
                    elif isinstance(val, torch.Tensor):
                        outputs[rid][out_key] = [val.clone()]
                    else:
                        outputs[rid][out_key] = val

        # --- Step 7: Restore dummy states ---
        for i, rid in enumerate(dummy_rids):
            for label in config_labels:
                self.alloc_manager.reset_label(
                    rid, label, free=i>=batch_size,
                )
        for rid in request_ids:
            for label in config_labels:
                ps = static_cm._plan_states.get(label)
                if ps is not None and ps.write_store:
                    self.alloc_manager.flush_to_store(rid, label)

        return outputs


class EncDecCudaGraphWrapper:
    """CUDA graph wrapper for stateless encoders/decoders (ViT, VAE).

    Simpler than CudaGraphRunner since EncDec models have fixed-shape inputs
    per batch size (no KV cache complications).
    """

    DEFAULT_CAPTURE_SIZES = [1, 2, 4, 8]

    def __init__(self, submodule: torch.nn.Module, device: torch.device):
        self.submodule = submodule
        self.device = device
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}
        self.memory_pool = None

    def warmup_and_capture(
        self, input_shape_template: tuple[int, ...]
    ) -> None:
        """Capture graphs for default batch sizes using a shape template."""
        if not torch.cuda.is_available():
            return

        self.memory_pool = torch.cuda.graphs.graph_pool_handle()

        for bs in reversed(self.DEFAULT_CAPTURE_SIZES):
            try:
                self._capture_one(bs, input_shape_template)
                logger.info("Captured EncDec CUDA graph at batch_size=%d", bs)
            except Exception:
                logger.warning(
                    "Failed to capture EncDec CUDA graph at batch_size=%d",
                    bs, exc_info=True)

    def _capture_one(
        self, bs: int, input_shape: tuple[int, ...]
    ) -> None:
        dummy_input = torch.randn(
            bs, *input_shape, dtype=torch.bfloat16, device=self.device
        )

        torch.cuda.synchronize()
        for _ in range(2):
            self.submodule(dummy_input)
        torch.cuda.synchronize()

        static_input = dummy_input.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.memory_pool):
            static_output = self.submodule(static_input)

        self.graphs[bs] = graph
        self.static_inputs[bs] = static_input
        self.static_outputs[bs] = static_output

    def can_run(self, batch_size: int) -> bool:
        if not self.graphs:
            return False
        return batch_size <= max(self.DEFAULT_CAPTURE_SIZES)

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        actual_bs = input_tensor.shape[0]

        idx = bisect.bisect_left(self.DEFAULT_CAPTURE_SIZES, actual_bs)
        if idx >= len(self.DEFAULT_CAPTURE_SIZES):
            return self.submodule(input_tensor)

        padded_bs = self.DEFAULT_CAPTURE_SIZES[idx]
        if padded_bs not in self.graphs:
            return self.submodule(input_tensor)

        self.static_inputs[padded_bs].zero_()
        self.static_inputs[padded_bs][:actual_bs] = input_tensor

        self.graphs[padded_bs].replay()

        return self.static_outputs[padded_bs][:actual_bs].clone()
