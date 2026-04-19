import logging
from dataclasses import dataclass, field

import torch

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
from mminf.engine.cpu_page_pool import CPUPagePool
from mminf.engine.cuda_graph_runner import CudaGraphRunner
from mminf.engine.kv_store import KVCacheConfig, PagedAllocationManager, TransferEngineInfo
from mminf.utils.profiler import range_pop, range_push
from mminf.utils.sampling import Sampler

logger = logging.getLogger(__name__)


# Multiple nodes may share a KV cache
@dataclass
class KVManagement:
    kv_cache_config: KVCacheConfig
    kv_cache: torch.Tensor
    alloc_manager: PagedAllocationManager
    cpu_page_pool: CPUPagePool | None
    buffer_manager: WorkspaceBufferManager


@dataclass
class SubmoduleManagement:
    submodule: torch.nn.Module
    kv_management: KVManagement
    sampler: Sampler = field(default_factory=Sampler)
    cuda_graph_runner: CudaGraphRunner | None = None


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
        autocast_dtype=torch.bfloat16,
        enable_nvtx: bool = False,
    ):
        super().__init__(enable_nvtx=enable_nvtx)

        self.kv_management: dict[str, KVManagement] = {}
        self.submodule_management: dict[str, SubmoduleManagement] = {}

        self.device = None
        self.autocast_dtype = autocast_dtype

    def engine_type(self) -> EngineType:
        return EngineType.AR

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        kv_cache_config: list[KVCacheConfig],
        device: torch.device,
        transfer_engine_info: TransferEngineInfo,
        kv_cache_type=None,
    ) -> None:
        self.device = device
        if kv_cache_type is None:
            kv_cache_type = self.autocast_dtype

        node_to_kv_mgmt = {}
        for cfg in kv_cache_config:
            num_layers = cfg.num_layers
            max_num_pages = cfg.max_num_pages
            page_size = cfg.page_size
            num_kv_heads = cfg.num_kv_heads
            head_dim = cfg.head_dim

            kv_cache = torch.zeros(
                num_layers, max_num_pages, 2,
                page_size, num_kv_heads, head_dim,
                dtype=kv_cache_type, device=device,
            ).contiguous()

            cpu_page_pool = None
            if cfg.cpu_offload_pages > 0:
                cpu_page_pool = CPUPagePool(
                    kv_cache_config=cfg,
                    max_cpu_pages=cfg.cpu_offload_pages,
                    kv_cache_dtype=kv_cache_type,
                )
                logger.info(
                    "AREngine: CPU page pool for initialized with %d pages",
                    cfg.cpu_offload_pages,
                )

            kv_mgmt = KVManagement(
                kv_cache_config=cfg,
                kv_cache=kv_cache,
                alloc_manager=PagedAllocationManager(
                    config=cfg,
                    kv_cache=kv_cache,
                    transfer_engine_info=transfer_engine_info
                ),
                cpu_page_pool=cpu_page_pool,
                buffer_manager = WorkspaceBufferManager(
                    256 * 1024 * 1024, device=device
                ),
            )
            self.kv_management[cfg.get_node_str()] = kv_mgmt

            for node_name in (cfg.nodes or submodules.keys()):
                node_to_kv_mgmt[node_name] = kv_mgmt

        for node_name, submodule in submodules.items():
            self.submodule_management[node_name] = SubmoduleManagement(
                submodule=submodule,
                kv_management=node_to_kv_mgmt[node_name],
            )


    def _create_cache_manager(
        self, request_ids: list[str],
        node_name: str
    ) -> BatchedCacheManager:
        """Create a CacheHandle for a single request."""
        submod_mgmt = self.submodule_management[node_name]
        cache_mgmt = submod_mgmt.kv_management

        from mminf.engine.kv_store import StoreWritePolicy
        autowrite = (cache_mgmt.alloc_manager.write_policy == StoreWritePolicy.ALWAYS)

        return BatchedCacheManager(
            request_ids=request_ids,
            active_labels_per_request={rid: "main" for rid in request_ids},
            kv_cache=cache_mgmt.kv_cache,
            alloc_manager=cache_mgmt.alloc_manager,
            buffer_manager=cache_mgmt.buffer_manager,
            kv_cache_config=cache_mgmt.kv_cache_config,
            device=self.device,
            auto_write_store=autowrite,
        )

    def _compile_submodules(self) -> None:
        """Apply torch.compile to submodule forward paths.

        Uses mode="max-autotune-no-cudagraphs" (SGLang's approach) so compiled
        code gets baked into CUDA graphs when captured. Must be called BEFORE
        CUDA graph capture.
        """
        if not torch.cuda.is_available():
            return

        for node_name, submodule_mgmt in self.submodule_management.items():
            submodule = submodule_mgmt.submodule

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

        # CUDA graph capture for decode (Option A keying)
        for node_name, submodule_mgmt in self.submodule_management.items():
            kv_mgmt = submodule_mgmt.kv_management
            runner = CudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule_mgmt.submodule,
                kv_cache_config=kv_mgmt.kv_cache_config,
                alloc_manager=kv_mgmt.alloc_manager,
                sampler=submodule_mgmt.sampler,
                buffer_manager=kv_mgmt.buffer_manager,
                device=self.device,
                autocast_dtype=self.autocast_dtype
            )
            runner.enable_nvtx = self.enable_nvtx
            runner.warmup_and_capture()
            if runner.graphs:
                submodule_mgmt.cuda_graph_runner = runner
                logger.info("AREngine: CUDA graphs captured for %s (%d configs)",
                            node_name, len(runner.graphs))
        # Step 1: torch.compile (before CUDA graph capture)
        self._compile_submodules()

    def _sample_decode_outputs(
        self,
        node_name: str,
        output: NodeOutput,
    ) -> NodeOutput:
        """Post-process decode outputs: sample tokens from logits.

        Called AFTER the model forward (and outside CUDA graph capture).
        Replaces 'logits' with 'new_token' in each request's output.
        """

        for rid, tensors in output.per_request_output_tensors.items():
            # Guard against non-per-rid keys (e.g. the __batched_logits__
            # sentinel used as a CUDA-graph fast-path hint): their value is
            # a torch.Tensor, not a dict, so the `"logits" not in tensors`
            # check below would raise TypeError (Tensor.__contains__ calls
            # torch.eq on strings).
            if not isinstance(tensors, dict) or "logits" not in tensors:
                continue
            logits = tensors["logits"][0]  # [1, vocab_size]
            tensors["new_token"] = [
                self.submodule_management[node_name].sampler.sample(
                    request_ids=[rid], logits=logits
                )
            ]
            del tensors["logits"]

        return output

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute batch with BatchedCacheManager for true vectorized batching."""
        cache_manager = self._create_cache_manager(
            batch.request_ids, batch.node_name
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
            range_push("ar.batched.preprocess", synchronize=True)
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
            range_push("ar.batched.forward")
        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            packed_inputs=preprocessed,
            request_ids=rids,
            per_request_info=batch.per_request_info,
        )
        if self.enable_nvtx:
            range_pop()

        cache_manager.flush_to_store()

        # `__batched_logits__` is the stacked [B, V] logits the submodule
        # already produced for the batch. When present, sample once across
        # the whole batch instead of looping per-rid (matches the CUDA-graph
        # fast path in cuda_graph_runner.sample_and_remap).
        batched_logits = batched_output.pop("__batched_logits__", None)

        if self.enable_nvtx:
            range_push("ar.batched.sample", synchronize=True)
        if batched_logits is not None:
            sampler = self.submodule_management[batch.node_name].sampler
            sampled = sampler.sample(rids, batched_logits)
            for rid, view in zip(rids, sampled.split(1), strict=True):
                rid_out = batched_output[rid]
                rid_out["new_token"] = [view]
                del rid_out["logits"]
            output = NodeOutput(per_request_output_tensors=batched_output)
        else:
            output = NodeOutput(per_request_output_tensors=batched_output)
            output = self._sample_decode_outputs(batch.node_name, output)
        if self.enable_nvtx:
            range_pop(synchronize=True)

        # Apply per-rid output filter so submodules that emit a static set
        # of keys for CUDA-graph capture compat (e.g. Qwen3-Omni Thinker
        # always emits thinker_states) can drop keys per real request in
        # eager mode too, keeping both execution paths consistent.
        for rid in rids:
            rid_out = output.per_request_output_tensors.get(rid)
            if not isinstance(rid_out, dict):
                continue
            output.per_request_output_tensors[rid] = submodule.filter_batched_output(
                batch.per_request_info.get(rid), rid_out,
            )
        return output

    def _execute_sequential(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Original per-request execution with CacheHandle."""
        per_request_outputs = {}

        for rid in batch.request_ids:
            cache_manager = self._create_cache_manager([rid], batch.node_name)
            inputs = batch.per_request_input_tensors.get(rid, {})
            metadata = {
                rid: batch.per_request_info[rid]
            }

            seq_lens = {
                rid: cache_manager._get_state(rid, "main").seq_len
            }
            logger.debug(f"Execute sequential {seq_lens}")

            if self.enable_nvtx:
                range_push("ar.seq.preprocess", synchronize=True)
            preprocessed = submodule.preprocess(
                batch.graph_walk,
                cache_manager=cache_manager,
                per_request_inputs=[inputs],
                request_ids=[rid],
                per_request_info=metadata,
            )
            if self.enable_nvtx:
                range_pop(synchronize=True)

            if self.enable_nvtx:
                range_push("ar.seq.forward")
            output = submodule(
                graph_walk=batch.graph_walk,
                cache_handle=cache_manager,
                request_info=metadata[rid],
                **preprocessed,
            )
            if self.enable_nvtx:
                range_pop()

            cache_manager.flush_to_store()
            per_request_outputs[rid] = output

        if self.enable_nvtx:
            range_push("ar.seq.sample", synchronize=True)
        output = NodeOutput(per_request_output_tensors=per_request_outputs)
        output = self._sample_decode_outputs(
            batch.node_name, output
        )
        if self.enable_nvtx:
            range_pop(synchronize=True)
        return output

    def _can_use_cuda_graph(self, batch: NodeBatch) -> bool:
        """Check if CUDA graph replay is available for this batch.

        Delegates the eligibility check to the submodule via
        ``submodule.can_use_cuda_graphs(batch)``. The default
        implementation on NodeSubmodule derives this from
        ``get_cuda_graph_configs`` (graph_walk membership).
        """
        submod_mgmt = self.submodule_management[batch.node_name]
        submodule = submod_mgmt.submodule
        if submodule is None:
            return False
        if not submodule.can_use_cuda_graphs(batch):
            return False
        runner = submod_mgmt.cuda_graph_runner
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
        runner = self.submodule_management[batch.node_name].cuda_graph_runner

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
            range_push(f"engine.ar.{batch.node_name}.{batch.graph_walk}.bs{len(batch.request_ids)}")

        submod_mgmt = self.submodule_management[batch.node_name]
        cache_mgmt = submod_mgmt.kv_management
        kv_cache_string = cache_mgmt.kv_cache_config.get_node_str()
        submodule = submod_mgmt.submodule
        try:
            needed_labels = self._get_needed_labels(
                batch.node_name, batch.graph_walk, batch.per_request_info
            )
            cache_mgmt.alloc_manager.alloc_status.reset()
            try:
                if self.enable_nvtx:
                    range_push("ar.kv_sync_retrieve", synchronize=True)
                for req_id, info in batch.per_request_info.items():
                    for label, seq_info in info.per_label_seq_info.get(kv_cache_string).items():
                        if needed_labels is not None and label not in needed_labels:
                            continue
                        cache_mgmt.alloc_manager.sync_retrieve(
                            req_id, label, seq_info
                        )
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                if self.enable_nvtx:
                    range_push("ar.sampler_config", synchronize=True)
                for rid, info in batch.per_request_info.items():
                    submod_mgmt.sampler.set_config(rid, **info.step_metadata)
                if self.enable_nvtx:
                    range_pop(synchronize=True)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                        # Priority: CUDA graph > batched > sequential
                        if self._can_use_cuda_graph(batch):
                            if self.enable_nvtx:
                                range_push("ar.cuda_graph_path", synchronize=True)
                            try:
                                output = self._execute_with_cuda_graph(batch, submodule)
                            finally:
                                if self.enable_nvtx:
                                    range_pop(synchronize=True)
                        elif submodule.can_batch(batch):
                            if self.enable_nvtx:
                                range_push("ar.batched_path", synchronize=True)
                            try:
                                output = self._execute_batched(batch, submodule)
                            finally:
                                if self.enable_nvtx:
                                    range_pop(synchronize=True)
                        else:
                            if self.enable_nvtx:
                                range_push("ar.sequential_path", synchronize=True)
                            try:
                                output = self._execute_sequential(batch, submodule)
                            finally:
                                if self.enable_nvtx:
                                    range_pop(synchronize=True)
                        for rid, info in batch.per_request_info.items():
                            submodule.postprocess(
                                request_info=info,
                                outputs=output.per_request_output_tensors.get(rid, {})
                            )
                        return output
            except RuntimeError:
                if not cache_mgmt.alloc_manager.alloc_status.success:
                    status = cache_mgmt.alloc_manager.alloc_status
                    logger.warning(
                        "KV cache page allocation failed for batch "
                        "(node=%s, walk=%s, request=%s, label=%s, "
                        "pages_short=%d)",
                        batch.node_name, batch.graph_walk,
                        status.request_id, status.label,
                        status.pages_short,
                    )
                    return NodeOutput(
                        per_request_output_tensors={
                            rid: {} for rid in batch.request_ids
                        },
                        allocation_failed=True,
                        alloc_pages_short=status.pages_short,
                        alloc_failed_request_id=status.request_id,
                    )
                raise
        finally:
            for req_id in batch.request_ids:
                batch.per_request_info[req_id].per_label_seq_info.add(
                    kv_cache_string,
                    cache_mgmt.alloc_manager.get_per_label_seq_info(req_id)
                )
            if self.enable_nvtx:
                range_pop()

    def _get_needed_labels(
        self, node_name: str, graph_walk: str,
        request_info: dict[str, CurrentForwardPassInfo]
    ):
        submodule = self.submodule_management[node_name].submodule
        needed_labels = None
        if hasattr(submodule, 'get_needed_cache_labels'):
            needed = submodule.get_needed_cache_labels(
                graph_walk, request_info)
            if needed is not None:
                needed_labels = set(needed)
        return needed_labels

    def check_ready(
        self, node_name: str, request_id: str,
        request_info: CurrentForwardPassInfo,
    ):
        submod_mgmt = self.submodule_management[node_name]
        cache_mgmt = submod_mgmt.kv_management
        # If this request was offloaded to CPU, try reloading first
        if cache_mgmt.cpu_page_pool is not None and cache_mgmt.cpu_page_pool.is_offloaded(request_id):
            try:
                cache_mgmt.alloc_manager.reload_request(request_id, cache_mgmt.cpu_page_pool)
                logger.info("Reloaded offloaded request %s from CPU", request_id)
            except RuntimeError:
                return False  # can't reload yet, not ready

        needed_labels = self._get_needed_labels(
            node_name, request_info.graph_walk, {
                    request_id: request_info
            }
        )

        labels_to_check = []
        try:
            for label, seq_info in request_info.per_label_seq_info.get(
                cache_mgmt.kv_cache_config.get_node_str()
            ).items():
                if needed_labels is not None and label not in needed_labels:
                    continue
                cache_mgmt.alloc_manager.start_async_retrieve(
                    request_id, label, seq_info
                )
                labels_to_check.append(label)
        except RuntimeError:
            # Not enough pages to allocate for retrieval — not ready
            return False

        ar_ready = all([
            cache_mgmt.alloc_manager.check_retrieve_ready(request_id, label)
            for label in labels_to_check
        ])
        if not ar_ready:
            return False
        return super().check_ready(node_name, request_id, request_info)

    def add_request(
        self, request_id: str, cache_labels: list[str] | None = None,
    ) -> None:
        for submodule_mgmt in self.submodule_management.values():
            submodule_mgmt.kv_management.alloc_manager.add_request(request_id, cache_labels or ["main"])
            submodule_mgmt.sampler.add_request(request_id)

    def remove_request(self, request_id: str) -> None:
        for submodule_mgmt in self.submodule_management.values():
            cache_mgmt = submodule_mgmt.kv_management
            if cache_mgmt.cpu_page_pool is not None:
                cache_mgmt.cpu_page_pool.remove_request(request_id)
            cache_mgmt.alloc_manager.remove_request(request_id)
            submodule_mgmt.sampler.remove_request(request_id)

    def pause_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """For interleaved loop: mark as paused, keep KV pages allocated."""
        for submodule_mgmt in self.submodule_management.values():
            cache_mgmt = submodule_mgmt.kv_management
            cache_mgmt.alloc_manager.get_state(request_id, cache_label).is_paused = True

    def resume_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """Resume from paused state for next LLM step in loop."""
        for submodule_mgmt in self.submodule_management.values():
            cache_mgmt = submodule_mgmt.kv_management
            cache_mgmt.alloc_manager.get_state(request_id, cache_label).is_paused = False

    def shutdown(self) -> None:
        for submodule_mgmt in self.submodule_management.values():
            cache_mgmt = submodule_mgmt.kv_management
            cache_mgmt.kv_cache = None
            cache_mgmt.buffer_manager = None
            cache_mgmt.alloc_manager.cleanup()
