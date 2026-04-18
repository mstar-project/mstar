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
from mminf.engine.kv_store import KVCacheConfig, PagedAllocationManager
from mminf.utils.profiler import range_pop, range_push
from mminf.utils.sampling import Sampler

logger = logging.getLogger(__name__)


DEFAULT_AR_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


@dataclass
class CudaGraphConfig:
    """Defines what computation a captured graph represents."""
    graph_walk: str  # "decode"
    dummy_capture_inputs: list[dict[str, list[torch.Tensor]]] # [{tensor_name: [tensor(s)]}]
    requires_cfg: bool  = False# whether CFG is active
    labels: list[str]  = field(default_factory=lambda: ["main"]) # cache labels used: ["main"] or ["main", "cfg_img"]
    compile: bool = True
    # Per-config override for the set of batch sizes to capture. None → use the
    # runner's default (AR engine default: DEFAULT_AR_CAPTURE_BATCH_SIZES;
    # CodecCudaGraphRunner picks its own default). Useful for codec-style
    # submodules where memory cost per size is high, or for AR walks where a
    # small subset is enough.
    capture_batch_sizes: list[int] | None = None


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
        submodule: nn.Module,
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

        for config in self.capture_configs:
            sizes = config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES
            for bs in reversed(sizes):
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

            # Build dummy per-request inputs via the submodule's own
            # capture-input generator. This lets each submodule declare what
            # dummy tensors its preprocess() needs (e.g., Thinker decode needs
            # a dummy token, Talker decode needs dummy all_codes).
            capture_templates = config.dummy_capture_inputs
            if capture_templates is None:
                # Submodule opts out of CUDA graphs for this walk.
                logger.info("%s.get_cuda_graph_capture_inputs returned None, skipping...", self.submodule_name)
                return

            def _clone_template(tpl):
                """Deep-copy a capture template (dict of input_name -> list[Tensor])."""
                out = {}
                for k, v in tpl.items():
                    if isinstance(v, list):
                        out[k] = [t.clone() if isinstance(t, torch.Tensor) else t for t in v]
                    elif isinstance(v, torch.Tensor):
                        out[k] = v.clone()
                    else:
                        out[k] = v
                return out

            # Use the first template for each dummy request slot
            template = capture_templates[0]
            dummy_inputs = [_clone_template(template) for _ in dummy_rids]

            # Build per-request metadata
            dummy_metadata = {
                rid: CurrentForwardPassInfo(
                    graph_walk=config.graph_walk,
                    requires_cfg=config.requires_cfg,
                    fwd_index=0,
                    random_seed=0,
                    max_tokens=1,
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

            # Static input buffers for ALL tensor inputs in the preprocessed
            # dict. These are the slots that will be overwritten with real
            # inputs during replay. Non-tensor values (lists, ints) are kept
            # in `preprocessed` as-is since they're fixed at capture time.
            static_input_keys = [
                k for k, v in preprocessed.items()
                if isinstance(v, torch.Tensor)
            ]

            if config.compile:
                forward = torch.compile(
                    submodule.forward_batched,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=False,
                    dynamic=False,
                )
            else:
                forward = submodule.forward_batched

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

            self.graphs[key] = CudaGraphData(
                graph=graph,
                static_inputs={
                    "preprocessed": preprocessed,
                    "static_input_keys": static_input_keys,
                    "capture_template": template,
                    "dummy_rids": dummy_rids,
                    "dummy_metadata": dummy_metadata,
                },
                static_outputs=output,
                static_cache_manager=cache_manager,
                config=config,
                bs=bs,
                has_non_logit_outputs=has_non_logit,
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
        padded_bs = self._get_padded_batch_size(batch_size, graph_walk, requires_cfg)
        if padded_bs is None:
            return False
        key = (graph_walk, requires_cfg, padded_bs)
        return key in self.graphs

    def _sizes_for(self, graph_walk: str, requires_cfg: bool) -> list[int]:
        for cfg in self.capture_configs:
            if cfg.graph_walk == graph_walk and cfg.requires_cfg == requires_cfg:
                return cfg.capture_batch_sizes or self.CAPTURE_BATCH_SIZES
        return self.CAPTURE_BATCH_SIZES

    def _get_padded_batch_size(
        self,
        batch_size: int,
        graph_walk: str = "decode",
        requires_cfg: bool = False,
    ) -> int | None:
        """Find smallest captured batch size >= batch_size for this config."""
        sizes = self._sizes_for(graph_walk, requires_cfg)
        idx = bisect.bisect_left(sizes, batch_size)
        if idx >= len(sizes):
            return None
        return sizes[idx]

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
        padded_bs = self._get_padded_batch_size(batch_size, graph_walk, requires_cfg)
        key = (graph_walk, requires_cfg, padded_bs)

        graph_data: CudaGraphData = self.graphs[key]
        graph = graph_data.graph
        static = graph_data.static_inputs
        static_cm = graph_data.static_cache_manager
        static_output = graph_data.static_outputs

        preprocessed = static["preprocessed"]
        dummy_rids = static["dummy_rids"]
        static_input_keys = static["static_input_keys"]
        capture_template = static["capture_template"]
        config_labels = graph_data.config.labels

        # --- Step 1: Set up real request states on dummy request IDs ---
        if self.enable_nvtx:
            range_push("cg.swap_states", synchronize=True)
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
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 2: Re-plan with real page tables (outside graph) ---
        if self.enable_nvtx:
            range_push("cg.preprocess_replan", synchronize=True)
        # Build real per-request inputs for the real slots
        real_inputs = []
        for i in range(batch_size):
            real_inputs.append(per_request_inputs[i])
        # Pad with dummy inputs for remaining slots using the same capture
        # template that was used during capture. This ensures submodule.preprocess
        # doesn't crash on empty lists for any input key the submodule expects.
        def _clone_template_for_padding(tpl):
            out = {}
            for k, v in tpl.items():
                if isinstance(v, list):
                    out[k] = [t.clone() if isinstance(t, torch.Tensor) else t for t in v]
                elif isinstance(v, torch.Tensor):
                    out[k] = v.clone()
                else:
                    out[k] = v
            return out

        for _i in range(batch_size, padded_bs):
            real_inputs.append(_clone_template_for_padding(capture_template))

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
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 3: Copy real tensor inputs into static buffers ---
        # The static buffers were captured from the output of preprocess() at
        # capture time. At runtime, preprocess() produces fresh tensors for the
        # real inputs; we copy each tensor field into its corresponding static
        # buffer so the CUDA graph's captured pointers see the new data.
        if self.enable_nvtx:
            range_push("cg.copy_inputs", synchronize=True)
        for key in static_input_keys:
            real_val = real_inputs.get(key)
            if real_val is None or not isinstance(real_val, torch.Tensor):
                continue
            static_buf = preprocessed[key]
            static_buf[:real_val.shape[0]].copy_(real_val)
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 4: Replay ---
        if self.enable_nvtx:
            range_push("cg.replay")
        graph.replay()
        if self.enable_nvtx:
            range_pop()

        # --- Step 5: Advance seq_lens on REAL request states ---
        # During replay, advance_seq_lens ran on dummy states (which point
        # to real states), so seq_lens are already advanced. But since
        # advance_seq_lens is Python-only and NOT captured in the graph,
        # we need to call it manually here.
        if self.enable_nvtx:
            range_push("cg.advance_seq_lens", synchronize=True)
        for label in config_labels:
            static_cm.set_active_label(label)
            # advance_seq_lens uses planned seq_lens (all 1 for decode)
            static_cm.advance_seq_lens()
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 6: Batched sampling from logits and remap outputs ---
        if self.enable_nvtx:
            range_push("cg.sample_and_remap", synchronize=True)

        outputs = {}

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
            # One `split` call produces N [1] views in C++ (faster than N
            # Python-level slicing ops). Then zip + dict comprehension builds
            # the outputs dict without enumerate overhead.
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
                    for out_key, val in static_output[dummy_rid].items():
                        if out_key == "logits":
                            continue
                        if isinstance(val, list):
                            outputs[rid][out_key] = [t.clone() for t in val]
                        elif isinstance(val, torch.Tensor):
                            outputs[rid][out_key] = [val.clone()]
                        else:
                            outputs[rid][out_key] = val
        else:
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

        if self.enable_nvtx:
            range_pop(synchronize=True)

        # --- Step 7: Restore dummy states ---
        if self.enable_nvtx:
            range_push("cg.restore_states", synchronize=True)
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
        if self.enable_nvtx:
            range_pop(synchronize=True)

        return outputs


class CodecGraphNotApplicableError(RuntimeError):
    """Raised by CodecCudaGraphRunner.run when preprocess rejects the batch.

    The audio codec engine catches this and falls back to eager execution
    (either batched forward_batched or per-request sequential).
    """


class CodecCudaGraphRunner:
    """CUDA graph capture/replay for stateless batched submodules.

    Contract (matches the AR runner so submodules look similar across engines):

        preprocess(graph_walk, per_request_inputs, request_ids, per_request_info)
            Python-level prep that turns variable list-of-dicts inputs into a
            dict of fixed-shape packed tensors. Runs OUTSIDE the captured
            region both during capture (on dummy inputs built from the
            config's ``dummy_capture_inputs``) and at replay (on real inputs).
            May return an empty dict to signal "can't be batched" — the
            engine falls back to the eager path in that case.

        cuda_graph_forward(**packed_tensors) -> dict[str, torch.Tensor]
            Pure-tensor call captured inside the graph. Output tensors must
            have batch-dim first, same size as the input batch dim, so the
            runner can slice ``[:actual_bs]`` and index per request.

        get_cuda_graph_configs(device) -> list[CudaGraphConfig]
            Each config's ``dummy_capture_inputs`` is a list of per-request
            NameToTensorList entries (same shape as real runtime inputs).
            The runner clones one of those entries per capture batch slot,
            then feeds the whole list to ``submodule.preprocess``.

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

        fwd = getattr(self.submodule, "cuda_graph_forward", None)
        if not callable(fwd):
            logger.info(
                "Submodule %s has no cuda_graph_forward(), skipping codec graph capture",
                self.submodule_name,
            )
            return

        self.memory_pool = torch.cuda.graphs.graph_pool_handle()

        for config in self.capture_configs:
            sizes = config.capture_batch_sizes or self.DEFAULT_CAPTURE_BATCH_SIZES
            for bs in reversed(sizes):
                try:
                    self._capture_one(bs, config, fwd)
                    logger.info(
                        "Captured codec CUDA graph for %s: walk=%s bs=%d",
                        self.submodule_name, config.graph_walk, bs,
                    )
                except Exception:
                    logger.warning(
                        "Failed to capture codec CUDA graph for %s: walk=%s bs=%d",
                        self.submodule_name, config.graph_walk, bs, exc_info=True,
                    )

    def _clone_template(self, tpl: dict) -> dict:
        out = {}
        for k, v in tpl.items():
            if isinstance(v, list):
                out[k] = [t.clone() if isinstance(t, torch.Tensor) else t for t in v]
            elif isinstance(v, torch.Tensor):
                out[k] = v.clone()
            else:
                out[k] = v
        return out

    def _capture_one(self, bs: int, config: CudaGraphConfig, fwd) -> None:
        if not config.dummy_capture_inputs:
            raise ValueError(
                f"{self.submodule_name}: CudaGraphConfig for walk "
                f"{config.graph_walk!r} missing dummy_capture_inputs"
            )

        # Build dummy per-request inputs (same format as real inputs) and
        # route them through the submodule's own preprocess — the AR runner
        # does the same, so the two code paths stay symmetric.
        template = config.dummy_capture_inputs[0]
        dummy_rids = [
            f"__codec_cg_{config.graph_walk}_{i}__" for i in range(bs)
        ]
        dummy_inputs = [self._clone_template(template) for _ in dummy_rids]
        dummy_info = {
            rid: CurrentForwardPassInfo(
                graph_walk=config.graph_walk,
                requires_cfg=False,
                fwd_index=0,
                random_seed=0,
                max_tokens=1,
            )
            for rid in dummy_rids
        }

        packed = self.submodule.preprocess(
            graph_walk=config.graph_walk,
            per_request_inputs=dummy_inputs,
            request_ids=dummy_rids,
            per_request_info=dummy_info,
        )
        if not packed or not all(isinstance(v, torch.Tensor) for v in packed.values()):
            raise RuntimeError(
                f"{self.submodule_name}: preprocess returned non-tensor/empty packed "
                f"inputs during capture (walk={config.graph_walk!r}); cannot capture"
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

        torch.cuda.set_device(self.device)
        torch.cuda.synchronize()
        # Warmup (outside graph): compile kernels, prime caches
        for _ in range(2):
            fwd(**static_inputs)
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.memory_pool):
            static_output = fwd(**static_inputs)
        torch.cuda.synchronize()

        key = (config.graph_walk, bs)
        self.graphs[key] = graph
        self.static_inputs[key] = static_inputs
        self.static_outputs[key] = static_output

    def _sizes_for(self, graph_walk: str) -> list[int]:
        for cfg in self.capture_configs:
            if cfg.graph_walk == graph_walk:
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
        per_request_inputs: list[dict],
        per_request_info: dict[str, CurrentForwardPassInfo],
        submodule: nn.Module,
    ) -> dict[str, dict[str, list[torch.Tensor]]]:
        """End-to-end replay: preprocess + replay + per-rid output split.

        Argument shape matches ``CudaGraphRunner.run`` (AR) so the two
        runners present the same interface to their engines. ``submodule``
        is passed in at call time (rather than taken from ``self.submodule``)
        for the same reason AR does it — keeps the runtime call site
        self-contained and mirrors AR's contract exactly.

        Raises ``CodecGraphNotApplicableError`` when preprocess returns
        an empty dict (e.g. SNAC frame-count mismatch), so the engine can
        transparently fall back to the eager batched path.
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

        if self.enable_nvtx:
            range_push("codec_cg.preprocess", synchronize=True)
        packed = submodule.preprocess(
            graph_walk=graph_walk,
            per_request_inputs=per_request_inputs,
            request_ids=request_ids,
            per_request_info=per_request_info,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)
        if not packed:
            raise CodecGraphNotApplicableError(
                f"{self.submodule_name}: preprocess signaled non-batchable inputs"
            )

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
        # Per-request split: each output tensor's leading dim is the batch
        # dim, so outputs[rid][name] = [static_output[name][i]] (cloned to
        # detach from the static buffer on the next replay).
        return {
            rid: {name: [static_output[name][i].clone()] for name in static_output}
            for i, rid in enumerate(request_ids)
        }


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
