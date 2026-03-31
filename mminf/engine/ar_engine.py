import logging

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
from mminf.engine.cuda_graph_runner import CudaGraphRunner
from mminf.engine.kv_store import KVCacheConfig, MooncakeStoreConfig, PagedAllocationManager, TransferEngineInfo
from mminf.conductor.request_info import CurrentForwardPassInfo, SequenceInfo
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)

class AREngine(BaseEngine):
    """
    Autoregressive engine with paged KV cache.
    Uses FlashInfer for prefill/decode when available.
    Supports pause/resume for interleaved loops (LLM <-> flow).

    The engine provides cache infrastructure (FlashInfer, page tables, KV tensor)
    via CacheHandle objects. Submodules decide which caches to read/write, when
    to snapshot, and how to combine multi-cache outputs (e.g., CFG formula).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig | dict,
        autocast_dtype=torch.bfloat16,
        enable_nvtx: bool = False,
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        if isinstance(kv_cache_config, dict):
            kv_cache_config = KVCacheConfig(**kv_cache_config)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.kv_cache_config = kv_cache_config
        self.device = None
        self.autocast_dtype = autocast_dtype
        self.kv_cache = None  # [num_layers, max_pages, 2, page_size, num_kv_heads, head_dim]
        self.alloc_manager: PagedAllocationManager | None = None
        self.buffer_manager = None

        # CUDA graph runners (initialized in warmup())
        self.cuda_graph_runners: dict[str, "CudaGraphRunner"] = {}

    def engine_type(self) -> EngineType:
        return EngineType.AR

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
        mooncake_cfg: MooncakeStoreConfig,
        transfer_engine_info: TransferEngineInfo,
        kv_cache_type=torch.bfloat16,
    ) -> None:
        self.submodules = submodules
        self.device = device
        cfg = model_config.get(
            "kv_cache", self.kv_cache_config
        )
        if not cfg:
            return  # dummy mode without config
        if isinstance(cfg, dict):
            cfg = KVCacheConfig(**cfg)

        num_layers = cfg.num_layers
        max_num_pages = cfg.max_num_pages
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim

        self.kv_cache = torch.zeros(
            num_layers, max_num_pages, 2,
            page_size, num_kv_heads, head_dim,
            dtype=kv_cache_type, device=device,
        ).contiguous()
        self.alloc_manager = PagedAllocationManager(
            config=cfg,
            kv_cache=self.kv_cache,
            mooncake_cfg=mooncake_cfg,
            transfer_engine_info=transfer_engine_info
        )

        # 256MB workspace for FlashInfer
        self.buffer_manager = WorkspaceBufferManager(
            256 * 1024 * 1024, device=device
        )

    def _create_cache_manager(self, request_id: str) -> BatchedCacheManager:
        """Create a CacheHandle for a single request."""
        from mminf.engine.kv_store import StoreWritePolicy
        return BatchedCacheManager(
            request_ids=[request_id],
            active_labels_per_request={request_id: "main"},
            kv_cache=self.kv_cache,
            alloc_manager=self.alloc_manager,
            buffer_manager=self.buffer_manager,
            kv_cache_config=self.kv_cache_config,
            device=self.device,
            auto_write_store=self.alloc_manager.write_policy == StoreWritePolicy.ALWAYS,
        )

    def _compile_submodules(self) -> None:
        """Apply torch.compile to submodule forward paths.

        Uses mode="max-autotune-no-cudagraphs" (SGLang's approach) so compiled
        code gets baked into CUDA graphs when captured. Must be called BEFORE
        CUDA graph capture.
        """
        if not torch.cuda.is_available():
            return

        for node_name, submodule in self.submodules.items():
            try:
                submodule.forward = torch.compile(
                    submodule.forward,
                    fullgraph=False,
                    dynamic=True,
                )
                # TODO @nsagan refactor to just have one forward function that handles batched
                # and sequential
                if hasattr(submodule, 'forward_batched'):
                    submodule.forward_batched = torch.compile(
                        submodule.forward_batched,
                        fullgraph=False,
                        dynamic=True,
                    )
                logger.info("AREngine: torch.compile applied to %s language_model", node_name)
            except Exception:
                logger.warning("AREngine: torch.compile failed for %s, using eager mode",
                               node_name, exc_info=True)

    def warmup(self) -> None:
        """Compile submodules and capture CUDA graphs."""
        from mminf.engine.cuda_graph_runner import CudaGraphRunner

        if self.kv_cache is None or self.device is None:
            logger.info("AREngine: skipping warmup (no KV cache or device)")
            return

        # Step 2: CUDA graph capture for decode (Option A keying)
        for node_name in self.submodules:
            runner = CudaGraphRunner(
                submodule_name=node_name,
                submodule=self.submodules[node_name],
                kv_cache_config=self.kv_cache_config,
                alloc_manager=self.alloc_manager,
                buffer_manager=self.buffer_manager,
                device=self.device,
                autocast_dtype=self.autocast_dtype
            )
            runner.warmup_and_capture()
            if runner.graphs:
                self.cuda_graph_runners[node_name] = runner
                logger.info("AREngine: CUDA graphs captured for %s (%d configs)",
                            node_name, len(runner.graphs))
        # Step 1: torch.compile (before CUDA graph capture)
        self._compile_submodules()

    def _sample_decode_outputs(
        self,
        output: NodeOutput,
        per_request_info: dict[str, CurrentForwardPassInfo]
    ) -> NodeOutput:
        """Post-process decode outputs: sample tokens from logits.

        Called AFTER the model forward (and outside CUDA graph capture).
        Replaces 'logits' with 'new_token' in each request's output.
        """
        from mminf.utils.sampling import sample_tokens

        for rid, tensors in output.per_request_output_tensors.items():
            if "logits" not in tensors:
                continue
            logits = tensors["logits"][0]  # [1, vocab_size]
            meta = per_request_info.get(rid).step_metadata
            temperature = meta.get("temperature", 0.6)
            top_k = meta.get("top_k", 0)
            top_p = meta.get("top_p", 1.0)
            # TODO add random seed here
            token = sample_tokens(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            tensors["new_token"] = [token]
            del tensors["logits"]

        return output

    def _can_batch(self, batch: NodeBatch) -> bool:
        """Only batch when all requests share a batchable graph_walk path.

        image_gen with 3-pass CFG is too complex to batch initially due to
        multi-label switching and snapshot operations within the forward pass.
        """
        if len(batch.request_ids) <= 1:
            return False
        if batch.graph_walk not in ("decode", "prefill_text"):
            return False
        # Ensure the submodule supports batched forward
        submodule = self.submodules.get(batch.node_name)
        if submodule is None or not hasattr(submodule, "forward_batched"):
            return False
        return True

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute batch with BatchedCacheManager for true vectorized batching."""
        from mminf.engine.kv_store import StoreWritePolicy
        cache_manager = BatchedCacheManager(
            request_ids=batch.request_ids,
            active_labels_per_request={rid: "main" for rid in batch.request_ids},
            kv_cache=self.kv_cache,
            alloc_manager=self.alloc_manager,
            buffer_manager=self.buffer_manager,
            kv_cache_config=self.kv_cache_config,
            device=self.device,
            auto_write_store=self.alloc_manager.write_policy == StoreWritePolicy.ALWAYS,
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
        preprocessed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            per_request_inputs=input_tensors,
            request_ids=rids,
            per_request_info=batch.per_request_info,
        )

        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            packed_inputs=preprocessed,
            request_ids=rids,
            per_request_info=batch.per_request_info,
        )
        cache_manager.flush_to_store()

        output = NodeOutput(per_request_output_tensors=batched_output)
        if batch.graph_walk == "decode":
            output = self._sample_decode_outputs(output, batch.per_request_info)
        return output

    def _execute_sequential(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Original per-request execution with CacheHandle."""
        per_request_outputs = {}
        
        for rid in batch.request_ids:
            cache_manager = self._create_cache_manager(rid)
            inputs = batch.per_request_input_tensors.get(rid, {})
            metadata = {
                rid: batch.per_request_info[rid]
            }

            seq_lens = {
                rid: cache_manager._get_state(rid, "main").seq_len
            }
            logger.debug(f"Execute sequential {seq_lens}")

            preprocessed = submodule.preprocess(
                batch.graph_walk,
                cache_manager=cache_manager,
                per_request_inputs=[inputs],
                request_ids=[rid],
                per_request_info=metadata,
            )
            output = submodule(
                graph_walk=batch.graph_walk,
                cache_handle=cache_manager,
                request_info=metadata[rid],
                **preprocessed,
            )
            cache_manager.flush_to_store()
            per_request_outputs[rid] = output

        output = NodeOutput(per_request_output_tensors=per_request_outputs)
        if batch.graph_walk == "decode":
            output = self._sample_decode_outputs(output, batch.per_request_info)
        return output

    def _can_use_cuda_graph(self, batch: NodeBatch) -> bool:
        """Check if CUDA graph replay is available for this batch."""
        if batch.graph_walk != "decode":
            return False
        runner = self.cuda_graph_runners.get(batch.node_name)
        if runner is None:
            return False

        has_cfg = any(
            batch.per_request_info[rid].requires_cfg
            for rid in batch.request_ids
        )
        return runner.can_run(
            batch_size=len(batch.request_ids),
            graph_walk=batch.graph_walk,
            requires_cfg=has_cfg,
        )

    def _execute_with_cuda_graph(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute using a captured CUDA graph.

        The CudaGraphRunner handles:
        1. Creating a BatchedCacheManager with persistent CUDA graph wrappers
        2. Running preprocess (plan_attention/plan_rope outside the graph)
        3. Copying inputs to static buffers, replaying the graph
        4. Advancing seq_lens after replay (Python-only, not captured)
        5. Remapping outputs from dummy request IDs to real ones
        """
        runner = self.cuda_graph_runners[batch.node_name]

        # TODO: don't hardcode it like this
        has_cfg = any(
            batch.per_request_info[rid].requires_cfg
            for rid in batch.request_ids
        )
        rids = list(batch.per_request_input_tensors.keys())
        input_tensors = [
            batch.per_request_input_tensors[rid] for rid in rids
        ]

        batched_output = runner.run(
            graph_walk=batch.graph_walk,
            requires_cfg=has_cfg,
            request_ids=rids,
            per_request_inputs=input_tensors,
            per_request_info=batch.per_request_info,
            submodule=submodule,
        )

        return NodeOutput(per_request_output_tensors=batched_output)

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"engine.ar.{batch.node_name}.{batch.graph_walk}")

        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            output = NodeOutput(
                per_request_output_tensors={rid: {} for rid in batch.request_ids}
            )
            if self.enable_nvtx:
                range_pop()
            return output

        try:
            # Filter to only retrieve cache labels this node actually needs
            needed_labels = None
            if hasattr(submodule, 'get_needed_cache_labels'):
                needed = submodule.get_needed_cache_labels(
                    batch.graph_walk, batch.per_request_info
                )
                if needed is not None:
                    needed_labels = set(needed)

            for req_id, info in batch.per_request_info.items():
                per_label_seq_info = info.per_label_seq_info
                for label, seq_info in per_label_seq_info.items():
                    if needed_labels is not None and label not in needed_labels:
                        continue
                    self.alloc_manager.retrieve_from_store(
                        req_id, label, seq_info
                    )

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                    # Priority: CUDA graph > batched > sequential
                    if self._can_use_cuda_graph(batch):
                        return self._execute_with_cuda_graph(batch, submodule)
                    elif self._can_batch(batch):
                        return self._execute_batched(batch, submodule)
                    else:
                        return self._execute_sequential(batch, submodule)
        finally:
            for req_id in batch.request_ids:
                batch.per_request_info[req_id].per_label_seq_info = \
                    self.alloc_manager.get_per_label_seq_info(req_id)
            if self.enable_nvtx:
                range_pop()

    def add_request(
        self, request_id: str, cache_labels: list[str] | None = None,
    ) -> None:
        self.alloc_manager.add_request(request_id, cache_labels or ["main"])

    def remove_request(self, request_id: str) -> None:
        self.alloc_manager.remove_request(request_id)

    def pause_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """For interleaved loop: mark as paused, keep KV pages allocated."""
        self.alloc_manager.get_state(request_id, cache_label).is_paused = True

    def resume_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """Resume from paused state for next LLM step in loop."""
        self.alloc_manager.get_state(request_id, cache_label).is_paused = False

    def shutdown(self) -> None:
        self.kv_cache = None
        self.buffer_manager = None
        self.alloc_manager.cleanup()