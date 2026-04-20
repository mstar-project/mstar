import bisect
from dataclasses import asdict, dataclass, field
import logging

from torch import nn
import torch

from mminf.engine.ar_engine import AREngine, KVManagement, SubmoduleManagement
from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
from mminf.engine.cuda_graph_runner import CudaGraphData
from mminf.engine.kv_store import KVCacheConfig, PagedAllocationManager
from mminf.model.base import NodeSubmodule
from mminf.utils.profiler import range_pop, range_push
from mminf.utils.sampling import Sampler


logger = logging.getLogger(__name__)


# TODO: this all feels a bit hacky and not generalizable. Ideally, we would like to modify our system
# to deal with the code predictor paradigm without making a code predictor engine (e.g., having an
# abstraction for injecting the cuda graph runner into the submodule execution path).


@dataclass
class CodePredictorCudaGraphConfig:
    """Defines what computation a captured graph represents."""

    # Unlike in other engines, the code predictor captures a subset of the forward
    # function in cuda graphs, instead of the whole forward_batched. So, the cuda graph
    # inputs are simple packed inputs to a function 
    dummy_capture_inputs: dict[str, torch.Tensor]
    graph_walk: str
    compile: bool = True
    # Per-config override for the set of batch sizes to capture.
    capture_batch_sizes: list[int] | None = None
    labels: list[str] = field(default_factory=lambda: ["main"])

    capture_function_name: str = "forward_cuda_graph"


class CodePredictorSubmodule(NodeSubmodule):
    def forward(
        self,
        request_info,
        **kwargs
    ):
        raise NotImplementedError(
            "Code predictor submodules must go through forward_batched."
        )
    
    def forward_cuda_graph(
        self, cache_manager: BatchedCacheManager,
        embed: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        This is the default function that will be captured by the cuda graph,
        if applicable. It takes in tensor keyword arguments, and returns a dict
        of tensors.
        """
        return {}
    
    def get_cuda_graph_configs(self, device: torch.device):
        raise NotImplementedError(
            "Code predictor submodules should be handled by the code predictor engine, "
            "which calls get_code_pred_cuda_graph_configs instead of get_cuda_graph_configs. "
        )

    def get_code_pred_cuda_graph_configs(
        self, device: torch.device
    ) -> list[CodePredictorCudaGraphConfig]:
        return []


class CodePredictorCudaGraphRunner:
    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def __init__(
        self,
        submodule_name: str,
        submodule: nn.Module,
        kv_cache_config: KVCacheConfig,
        alloc_manager: PagedAllocationManager,
        sampler: Sampler,
        buffer_manager: WorkspaceBufferManager,
        device: torch.device,
    ):
        self.submodule_name = submodule_name
        self.submodule = submodule
        self.capture_configs: list[CodePredictorCudaGraphConfig] = submodule.get_code_pred_cuda_graph_configs(device)
        self.kv_cache_config = kv_cache_config
        self.alloc_manager = alloc_manager
        self.sampler = sampler
        self.device = device
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

        for config in self.capture_configs:
            sizes = config.capture_batch_sizes or self.CAPTURE_BATCH_SIZES
            for bs in reversed(sizes):
                key = (config.graph_walk, bs)
                try:
                    self._capture_one(bs, config, self.submodule)
                    logger.info("Captured CUDA graph for %s: %s bs=%d",
                                self.submodule_name, key, bs)
                except Exception:
                    logger.warning(
                        "Failed to capture CUDA graph for %s: %s bs=%d",
                        self.submodule_name, key, bs, exc_info=True)

    def _create_persistent_wrappers(
        self, bs: int, config: CodePredictorCudaGraphConfig
    ) -> dict:
        """Create persistent FlashInfer wrappers for CUDA graph capture.

        Returns dict of label -> _PlanState with persistent wrappers.
        """
        from mminf.engine.cache_manager import _PlanState
        from mminf.utils.flashinfer_utils import FlashInferDecodeWrapper

        cfg = self.kv_cache_config
        # For decode: each request has 1 new token
        total_tokens = bs

        # Allocate workspace buffer for CUDA graph wrappers.
        # Each label gets its own workspace to avoid conflicts during
        # multi-pass captures (e.g., main + cfg_img in same graph).
        plan_states = {}
        for label in config.labels:
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
        self, bs: int, config: CodePredictorCudaGraphConfig, submodule
    ) -> None:
        """Capture a single CUDA graph for the given batch size and config."""
        from mminf.engine.cache_manager import BatchedCacheManager

        cfg = self.kv_cache_config
        key = (config.graph_walk, bs)

        # Create dummy request IDs
        dummy_rids = [f"__cg_{config.graph_walk}_{i}__"
                      for i in range(bs)]
        seq_lens = [1] * bs

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

            for label in config.labels:
                # For now, this attention planning is fixed. In the future,
                # we may need to generalize by pushing stuff to the submodule
                cache_manager.plan_attention(
                    seq_lens=seq_lens, is_causal=True, label=label
                )
                cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label=label)

            dummy_inputs = {}
            for k, val in config.dummy_capture_inputs.items():
                if val.shape[0] != 1:
                    val = val.unsqueeze(0)
                dummy_inputs[k] = val.repeat(bs, *([1] * (val.dim() -1)))
            static_input_keys = list(dummy_inputs.keys())

            forward = getattr(submodule, config.capture_function_name)
            if config.compile:
                forward = torch.compile(
                    forward,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=False,
                    dynamic=False,
                )

            def run_forward():
                return forward(
                    graph_walk=config.graph_walk,
                    cache_manager=cache_manager,
                    **dummy_inputs
                )

            torch.cuda.set_device(self.device)
            # Warmup: 2 forward passes
            torch.cuda.synchronize()
            for _ in range(2):
                output = run_forward()
                # Reset seq_lens after warmup passes so capture starts clean
                for rid in dummy_rids:
                    for label in config.labels:
                        state = self.alloc_manager.get_state(rid, label)
                        state.seq_len = max(0, state.seq_len - 1)
                        state.position_id_start = max(
                            0, state.position_id_start - 1)
                # Re-plan after reset
                for label in config.labels:
                    cache_manager.plan_attention(
                        seq_lens=seq_lens, is_causal=True, label=label
                    )
                    cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label=label)
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = run_forward()
            torch.cuda.synchronize()

            logger.info(
                "CodePredictorCudaGraphRunner: captured graph %s", key,
            )

            self.graphs[key] = CudaGraphData(
                graph=graph,
                static_inputs={
                    "dummy_inputs": dummy_inputs,
                    "static_input_keys": static_input_keys,
                    "dummy_rids": dummy_rids,
                },
                static_outputs=output,
                static_cache_manager=cache_manager,
                config=config,
                bs=bs,
                has_non_logit_outputs=True
            )

            logger.debug("Captured graph %s, output keys: %s", key,
                         list(output.keys()) if isinstance(output, dict)
                         else type(output))
        finally:
            # Clean up dummy requests
            for rid in dummy_rids:
                for label in config.labels:
                    self.alloc_manager.reset_label(rid, label, free=True)

    def _sizes_for(self, graph_walk: str) -> list[int]:
        for cfg in self.capture_configs:
            if cfg.graph_walk == graph_walk:
                return cfg.capture_batch_sizes or self.CAPTURE_BATCH_SIZES
        return self.CAPTURE_BATCH_SIZES

    def _get_padded_batch_size(
        self,
        batch_size: int,
        graph_walk: str = "decode",
    ) -> int | None:
        """Find smallest captured batch size >= batch_size for this config."""
        sizes = self._sizes_for(graph_walk)
        idx = bisect.bisect_left(sizes, batch_size)
        if idx >= len(sizes):
            return None
        return sizes[idx]

    @torch.compiler.disable()
    def run(
        self,
        graph_walk: str,
        packed_inputs: dict[str, torch.Tensor],
        request_ids: str,
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
        padded_bs = self._get_padded_batch_size(batch_size, graph_walk)
        key = (graph_walk, padded_bs)

        graph_data: CudaGraphData = self.graphs[key]
        graph = graph_data.graph
        static = graph_data.static_inputs
        static_cm = graph_data.static_cache_manager
        static_output = graph_data.static_outputs

        dummy_inputs = static["dummy_inputs"]
        dummy_rids = static["dummy_rids"]
        static_input_keys = static["static_input_keys"]
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
        seq_lens = [1] * batch_size
        for label in config_labels:
            static_cm.plan_attention(
                seq_lens=seq_lens, is_causal=True, label=label
            )
            static_cm.plan_rope(seq_lens=seq_lens, pos_ids=None, label=label)


        # --- Step 3: Copy real tensor inputs into static buffers ---
        for key in static_input_keys:
            real_val = packed_inputs.get(key)
            if real_val is None or not isinstance(real_val, torch.Tensor):
                continue
            static_buf = dummy_inputs[key]
            static_buf[:real_val.shape[0]].copy_(real_val)
        graph.replay()

        # --- Step 5: Advance seq_lens on REAL request states ---
        for label in config_labels:
            static_cm.set_active_label(label)
            # advance_seq_lens uses planned seq_lens (all 1 for decode)
            static_cm.advance_seq_len()

        # TODO: hardcoding "batch_size" here is maybe hacky
        outputs = {
            key: val[:batch_size] for key, val in static_output.items()
        }

        # --- Step 7: Restore dummy states ---
        for i, rid in enumerate(dummy_rids):
            for label in config_labels:
                self.alloc_manager.reset_label(
                    rid, label, free=i>=batch_size,
                )
        return outputs


class CodePredictorEngine(AREngine):
    def engine_type(self) -> EngineType:
        return EngineType.CODE_PREDICTOR
    
    def has_autocast(self):
        return False
    
    def warmup(self) -> None:
        """Compile submodules and capture CUDA graphs."""
        # CUDA graph capture for decode (Option A keying)
        for node_name, submodule_mgmt in self.submodule_management.items():
            kv_mgmt = submodule_mgmt.kv_management
            runner = CodePredictorCudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule_mgmt.submodule,
                kv_cache_config=kv_mgmt.kv_cache_config,
                alloc_manager=kv_mgmt.alloc_manager,
                sampler=submodule_mgmt.sampler,
                buffer_manager=kv_mgmt.buffer_manager,
                device=self.device,
            )
            runner.warmup_and_capture()
            if runner.graphs:
                submodule_mgmt.cuda_graph_runner = runner
                logger.info("AREngine: CUDA graphs captured for %s (%d configs)",
                            node_name, len(runner.graphs))
            # self._compile_submodules()

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute batch with BatchedCacheManager for true vectorized batching."""
        cache_manager = self._create_cache_manager(
            batch.request_ids, batch.node_name
        )

        for rid in batch.request_ids:
            cache_manager.reset_state(
                request_id=rid,
                keep_pages=True
            )

        # Preprocess all requests
        rids = list(batch.per_request_input_tensors.keys())
        seq_lens = {
            rid: cache_manager._get_state(rid, "main").seq_len for rid in rids
        }
        logger.debug(f"Execute batched {seq_lens}")
        input_tensors = [
            batch.per_request_input_tensors[rid] for rid in rids
        ]
        if self.enable_nvtx:
            range_push("code_pred.batched.preprocess", synchronize=True)
        preprocessed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            per_request_inputs=input_tensors,
            request_ids=rids,
            per_request_info=batch.per_request_info,
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)

        if self.enable_nvtx:
            range_push("code_pred.batched.forward")
        
        sampler = self.submodule_management[batch.node_name].sampler
        cuda_graph_runner = self.submodule_management[batch.node_name].cuda_graph_runner
        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            packed_inputs=preprocessed,
            sampler=sampler,
            cuda_graph_runner=cuda_graph_runner,
            request_ids=rids,
            per_request_info=batch.per_request_info,
        )
        output = NodeOutput(per_request_output_tensors=batched_output)
        if self.enable_nvtx:
            range_pop()
        return output
    
    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"engine.code_pred.{batch.node_name}.{batch.graph_walk}.bs{len(batch.request_ids)}")
        
        submod_mgmt = self.submodule_management[batch.node_name]
        submodule = submod_mgmt.submodule
        if self.enable_nvtx:
            range_push("code_pred.sampler_config", synchronize=True)
        for rid, info in batch.per_request_info.items():
            sampling_config = info.sampling_config.get(batch.node_name)
            sampling_config = {} if sampling_config is None else asdict(sampling_config) 
            submod_mgmt.sampler.set_config(rid, **sampling_config)
        if self.enable_nvtx:
            range_pop(synchronize=True)


        # NO autocast for float32 Code Predictor inference.  HF and
        # vllm-omni found that fused/autocast kernels degrade audio quality
        # for the small (5-layer) Code Predictor.
        with torch.no_grad():
            output = self._execute_batched(batch, submodule)
            for rid, info in batch.per_request_info.items():
                submodule.postprocess(
                    request_id=rid,
                    request_info=info,
                    outputs=output.per_request_output_tensors.get(rid, {})
                )
            return output
    
    def check_ready(
        self, *args, **kwargs
    ):
        return True