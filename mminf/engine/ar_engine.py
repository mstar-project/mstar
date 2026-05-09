import logging
import os
from dataclasses import asdict, dataclass, field

import torch

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cache_manager import BatchedCacheManager, WorkspaceBufferManager
from mminf.engine.cpu_page_pool import CPUPagePool
from mminf.engine.cuda_graph_runner import CudaGraphRunner
from mminf.engine.kv_store import KVCacheConfig, PagedAllocationManager, TransferEngineInfo
from mminf.model.submodule_base import ARNodeInputs, ARNodeSubmodule, ModelInputsFromEngine
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
    submodule: ARNodeSubmodule
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

        # Dedup set for "cuda graphs captured but not usable for this shape"
        # warnings — each unique miss shape is logged at most once.
        self._logged_graph_misses: set[tuple] = set()

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
                    int(os.environ.get("MMINF_WORKSPACE_BUFFER_MB", "512")) * 1024 * 1024,
                    device=device,
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
            enable_nvtx=self.enable_nvtx,
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
        from mminf.engine.cuda_graph_runner import (
            CudaGraphRunner, PiecewiseCudaGraphRunner, DEFAULT_AR_CAPTURE_BATCH_SIZES,
        )

        for node_name, submodule_mgmt in self.submodule_management.items():
            kv_mgmt = submodule_mgmt.kv_management
            submodule = submodule_mgmt.submodule

            # Standard AR decode CUDA graph (CudaGraphRunner).
            runner = CudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
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

            # Piecewise CUDA graph for transformer block loops (e.g. VJepa2 AC rollout).
            # Submodules opt in by implementing get_piecewise_runner_config().
            pcgr_config = getattr(submodule, "get_piecewise_runner_config", lambda: None)()
            if pcgr_config is not None:
                pcgr = PiecewiseCudaGraphRunner(
                    fn_factory=pcgr_config["fn_factory"],
                    embed_dim=pcgr_config["embed_dim"],
                    capture_batch_sizes=pcgr_config.get("capture_batch_sizes", DEFAULT_AR_CAPTURE_BATCH_SIZES),
                    capture_seq_len=pcgr_config["capture_seq_len"],
                    device=self.device,
                    autocast_dtype=self.autocast_dtype,
                    pos_buf_shapes=pcgr_config.get("pos_buf_shapes"),
                    kv_cache_config=kv_mgmt.kv_cache_config,
                    alloc_manager=kv_mgmt.alloc_manager,
                    buffer_manager=kv_mgmt.buffer_manager,
                    cache_labels=pcgr_config.get("cache_labels", ["main"]),
                )
                pcgr.warmup_and_capture()
                if pcgr.graphs:
                    submodule.set_piecewise_runner(pcgr)
                    logger.info(
                        "AREngine: PiecewiseCudaGraphRunner installed for %s (%d bs buckets)",
                        node_name, len(pcgr.graphs),
                    )

        # torch.compile applied after CUDA graph capture so compiled kernels
        # are baked into the graphs.
        self._compile_submodules()

    def get_max_batch_size(self, node_name, graph_walk):
        if node_name not in self.submodule_management:
            return
        submod_max_bs = self.submodule_management[node_name].submodule.max_batch_size(graph_walk)
        submod_mg = self.submodule_management[node_name]
        if submod_mg.cuda_graph_runner is None:
            return submod_max_bs
        
        runner = submod_mg.cuda_graph_runner
        configs = [
            cfg for cfg in runner.capture_configs \
                if graph_walk in cfg.replay_graph_walks
        ]
        if not configs:
            return submod_max_bs
        max_cuda_graph_bs = max([
            max(cfg.capture_batch_sizes or runner.CAPTURE_BATCH_SIZES) for cfg in configs
        ])
        if submod_max_bs is not None:
            return min(max_cuda_graph_bs, submod_max_bs)
        return max_cuda_graph_bs

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
            # Clone for the same reason as the cuda_graph_runner sampler
            # paths: FlashInfer's sampling reuses the output buffer and
            # speculation chains expose the alias as token doubling.
            tensors["new_token"] = [
                self.submodule_management[node_name].sampler.sample(
                    request_ids=[rid], logits=logits
                ).clone()
            ]
            del tensors["logits"]

        return output

    def _execute_batched(
        self, batch: NodeBatch, submodule: ARNodeSubmodule,
        inputs: list[ARNodeInputs], sampler: Sampler,
    ) -> NodeOutput:
        """Execute batch with BatchedCacheManager for true vectorized batching."""
        cache_manager = self._create_cache_manager(
            batch.request_ids, batch.node_name
        )
        engine_inputs = ModelInputsFromEngine(
            request_ids=batch.request_ids,
            per_request_info=batch.per_request_info,
            cache_manager=cache_manager,
            sampler=sampler
        )
        if self.enable_nvtx:
            range_push("ar.batched.preprocess", synchronize=False)
        preprocessed = submodule.preprocess(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            inputs=inputs,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)

        if self.enable_nvtx:
            range_push("ar.batched.forward")
        # Signal the main thread that we're about to enter CUDA launch
        # code. PyTorch drops the GIL inside the C++ kernel-launch path,
        # so main can resume Python-heavy postprocess in parallel.
        launch_started_event = batch.metadata.get("launch_started_event")
        if launch_started_event is not None:
            launch_started_event.set()
        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            engine_inputs=engine_inputs,
            **preprocessed
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
            range_push("ar.batched.sample", synchronize=False)
        if batched_logits is not None:
            sampler = self.submodule_management[batch.node_name].sampler
            sampled = sampler.sample(batch.request_ids, batched_logits)
            for rid, view in zip(batch.request_ids, sampled.split(1), strict=True):
                rid_out = batched_output[rid]
                rid_out["new_token"] = [view]
                del rid_out["logits"]
            output = NodeOutput(per_request_output_tensors=batched_output)
        else:
            output = NodeOutput(per_request_output_tensors=batched_output)
            output = self._sample_decode_outputs(batch.node_name, output)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # Apply per-rid output filter so submodules that emit a static set
        # of keys for CUDA-graph capture compat (e.g. Qwen3-Omni Thinker
        # always emits thinker_states) can drop keys per real request in
        # eager mode too, keeping both execution paths consistent.
        for rid in batch.request_ids:
            rid_out = output.per_request_output_tensors.get(rid)
            if not isinstance(rid_out, dict):
                continue
            output.per_request_output_tensors[rid] = submodule.filter_batched_output(
                batch.per_request_info.get(rid), rid_out,
            )
        return output

    def _execute_sequential(
        self, batch: NodeBatch,
        submodule: ARNodeSubmodule,
        inputs: list[ARNodeInputs],
        sampler: Sampler,
    ) -> NodeOutput:
        """Original per-request execution with CacheHandle."""
        per_request_outputs = {}

        for rid, node_inputs in zip(batch.request_ids, inputs, strict=True):
            cache_manager = self._create_cache_manager([rid], batch.node_name)
            inputs = batch.per_request_input_tensors.get(rid, {})
            engine_inputs = ModelInputsFromEngine(
                request_ids=[rid],
                per_request_info={
                    rid: batch.per_request_info[rid]
                },
                cache_manager=cache_manager,
                sampler=sampler,
            )

            if self.enable_nvtx:
                range_push("ar.seq.preprocess", synchronize=False)
            preprocessed = submodule.preprocess(
                batch.graph_walk,
                engine_inputs=engine_inputs,
                inputs=[node_inputs],
            )
            if self.enable_nvtx:
                range_pop(synchronize=False)

            if self.enable_nvtx:
                range_push("ar.seq.forward")
            # Signal on the first rid only — subsequent forwards for
            # other rids continue to release the GIL inside PyTorch C++.
            launch_started_event = batch.metadata.get("launch_started_event")
            if launch_started_event is not None and not launch_started_event.is_set():
                launch_started_event.set()
            output = submodule.forward(
                graph_walk=batch.graph_walk,
                engine_inputs=engine_inputs,
                **preprocessed,
            )
            if self.enable_nvtx:
                range_pop()

            cache_manager.flush_to_store()
            per_request_outputs[rid] = output

        if self.enable_nvtx:
            range_push("ar.seq.sample", synchronize=False)
        output = NodeOutput(per_request_output_tensors=per_request_outputs)
        output = self._sample_decode_outputs(
            batch.node_name, output
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
        return output

    def _can_use_cuda_graph(self, batch: NodeBatch, inputs: list[ARNodeInputs]) -> bool:
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
        runner = submod_mgmt.cuda_graph_runner
        if runner is None:
            return False

        has_cfg = any(
            batch.per_request_info[rid].requires_cfg
            for rid in batch.request_ids
        )
        bs = len(batch.request_ids)
        #TODO: Remove in production
        num_tokens = sum(inp.input_seq_len for inp in inputs)

        if not submodule.can_use_cuda_graphs(batch, inputs):
            self._log_graph_miss(
                node_name=batch.node_name,
                graph_walk=batch.graph_walk,
                bs=bs, num_tokens=num_tokens, requires_cfg=has_cfg,
                runner=runner,
                reason="submodule.can_use_cuda_graphs() returned False",
            )
            return False

        if not runner.can_run(
            batch_size=bs,
            num_tokens=num_tokens,
            graph_walk=batch.graph_walk,
            requires_cfg=has_cfg,
        ):
            self._log_graph_miss(
                node_name=batch.node_name,
                graph_walk=batch.graph_walk,
                bs=bs, num_tokens=num_tokens, requires_cfg=has_cfg,
                runner=runner,
                reason="no captured graph matches this (bs, num_tokens, graph_walk, requires_cfg)",
            )
            return False
        return True

    def _log_graph_miss(
        self,
        node_name: str,
        graph_walk: str,
        bs: int,
        num_tokens: int,
        requires_cfg: bool,
        runner: CudaGraphRunner,
        reason: str,
    ) -> None:
        """Warn (once per unique miss shape) when a runner exists but the
        current request can't use a captured graph. Helps diagnose decode
        slowness from unexpected eager fallbacks.
        """
        if not runner.graphs:
            return  # nothing was ever captured — not actionable, skip noise
        miss_key = (node_name, graph_walk, bs, num_tokens, requires_cfg, reason)
        if miss_key in self._logged_graph_misses:
            return
        self._logged_graph_misses.add(miss_key)

        captured_for_walk = sorted(
            {(k.bs, k.num_tokens) for k in runner.graphs.keys()
             if k.graph_walk == graph_walk and k.requires_cfg == requires_cfg}
        )
        captured_walks = sorted({k.graph_walk for k in runner.graphs.keys()})
        logger.warning(
            "[cuda-graph miss] node=%s graph_walk=%s requested=(bs=%d, num_tokens=%d, requires_cfg=%s) "
            "reason='%s' captured_shapes_for_walk=%s captured_walks=%s — falling back to eager.",
            node_name, graph_walk, bs, num_tokens, requires_cfg,
            reason, captured_for_walk or "<none>", captured_walks,
        )

    def _execute_with_cuda_graph(
        self, batch: NodeBatch, submodule: ARNodeSubmodule,
        inputs: list[ARNodeInputs]
    ) -> NodeOutput:
        """Execute using a captured CUDA graph.

        The CudaGraphRunner handles:
        1. Creating a BatchedCacheManager with persistent CUDA graph wrappers
        2. Running preprocess (plan_attention/plan_rope outside the graph)
        3. Copying inputs to static buffers, replaying the graph
        4. Advancing seq_lens after replay (Python-only, not captured)
        5. Remapping outputs from dummy request IDs to real ones
        """
        runner = self.submodule_management[batch.node_name].cuda_graph_runner

        has_cfg = any(
            batch.per_request_info[rid].requires_cfg
            for rid in batch.request_ids
        )

        batched_output = runner.run(
            graph_walk=batch.graph_walk,
            requires_cfg=has_cfg,
            request_ids=batch.request_ids,
            inputs=inputs,
            per_request_info=batch.per_request_info,
            submodule=submodule,
            slot=batch.metadata.get("cuda_graph_slot"),
            advance_event=batch.metadata.get("advance_event"),
            launch_started_event=batch.metadata.get("launch_started_event"),
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
                    range_push("ar.kv_sync_retrieve", synchronize=False)
                for req_id, info in batch.per_request_info.items():
                    for label, seq_info in info.per_label_seq_info.get(kv_cache_string).items():
                        if needed_labels is not None and label not in needed_labels:
                            continue
                        cache_mgmt.alloc_manager.sync_retrieve(
                            req_id, label, seq_info
                        )
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                if self.enable_nvtx:
                    range_push("ar.sampler_config", synchronize=False)
                runner = submod_mgmt.cuda_graph_runner
                for rid, info in batch.per_request_info.items():
                    sampling_config = info.sampling_config.get(batch.node_name)
                    sampling_kwargs = {} if sampling_config is None else asdict(sampling_config)
                    submod_mgmt.sampler.set_config(rid, **sampling_kwargs)
                    # Mirror into the cuda-graph runner's master GPU buffers.
                    # update_request_config is change-detected, so steady-state
                    # requests pay only a dict comparison here — no GPU work.
                    if runner is not None and sampling_config is not None:
                        runner.update_request_config(rid, sampling_config)
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=True, dtype=self.autocast_dtype):
                        # run prepare inputs
                        node_inputs: list[ARNodeInputs] = []
                        if self.enable_nvtx:
                            range_push(
                                f"ar.prepare_inputs"
                            )
                        for rid in batch.request_ids:
                            labels = cache_mgmt.alloc_manager.get_labels(rid)
                            pos_info = {
                                label: cache_mgmt.alloc_manager.get_state(
                                    rid, label
                                ).get_pos_info() for label in labels
                            }
                            node_inputs.append(
                                submodule.prepare_inputs(
                                    graph_walk=batch.graph_walk,
                                    fwd_info=batch.per_request_info[rid],
                                    inputs=batch.per_request_input_tensors[rid],
                                    pos_info=pos_info
                                )
                            )
                        if self.enable_nvtx:
                            range_pop(synchronize=True)

                        # Priority: CUDA graph > batched > sequential
                        if self._can_use_cuda_graph(batch, node_inputs):
                            if self.enable_nvtx:
                                range_push("ar.cuda_graph_path", synchronize=False)
                            try:
                                output = self._execute_with_cuda_graph(
                                    batch, submodule, node_inputs
                                )
                            finally:
                                if self.enable_nvtx:
                                    range_pop(synchronize=False)
                        elif submodule.can_batch(batch, node_inputs):
                            if self.enable_nvtx:
                                range_push("ar.batched_path", synchronize=False)
                            try:
                                output = self._execute_batched(
                                    batch, submodule, node_inputs,
                                    sampler=submod_mgmt.sampler
                                )
                            finally:
                                if self.enable_nvtx:
                                    range_pop(synchronize=False)
                        else:
                            if self.enable_nvtx:
                                range_push("ar.sequential_path", synchronize=False)
                            try:
                                output = self._execute_sequential(
                                    batch, submodule, node_inputs,
                                    sampler=submod_mgmt.sampler
                                )
                            finally:
                                if self.enable_nvtx:
                                    range_pop(synchronize=False)
                        for rid, info in batch.per_request_info.items():
                            submodule.postprocess(
                                request_id=rid,
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

    def check_stop_for_batch(
        self, batch: NodeBatch, output: NodeOutput
    ) -> dict[str, set[str]]:
        """Delegate to each rid's submodule.check_stop. Worker calls this on
        the slow-postprocess path so the .item() / .cpu() reads no longer
        block ``execute_batch`` on the GPU thread."""
        if batch.node_name not in self.submodule_management:
            return {}
        submodule = self.submodule_management[batch.node_name].submodule
        result: dict[str, set[str]] = {}
        for rid in batch.request_ids:
            req_outputs = output.per_request_output_tensors.get(rid, {})
            if not req_outputs:
                continue
            req_info = batch.per_request_info.get(rid)
            if req_info is None:
                continue
            stops = submodule.check_stop(rid, req_info, req_outputs)
            if stops:
                result[rid] = stops
        return result

    def reserve_replay_slot(self, batch: NodeBatch) -> int | None:
        """Allocate the next double-buffer slot for this batch and stash it
        on ``batch.metadata['cuda_graph_slot']``.

        Phase 3: Worker's main thread calls this on the speculative path
        BEFORE submitting both pre-plan and replay so they target the same
        slot (and the OPPOSITE slot from the in-flight replay). Returns the
        slot index, or ``None`` if no captured graph matches (eager path).
        """
        runner = self.submodule_management[batch.node_name].cuda_graph_runner
        if runner is None or not runner.graphs:
            return None
        has_cfg = any(
            info.requires_cfg for info in batch.per_request_info.values()
        )
        bs = len(batch.request_ids)
        # Don't pass num_tokens — the runner derives it from the captured
        # BASIC_BATCHED config. Non-decode (prefill) batches don't speculate
        # and don't pre-reserve, so they go through ``run`` which advances
        # the per-key counter itself.
        slot = runner.reserve_slot(
            graph_walk=batch.graph_walk,
            requires_cfg=has_cfg,
            batch_size=bs,
        )
        if slot is not None:
            batch.metadata["cuda_graph_slot"] = slot
        return slot

    def reset_pre_plan_for_batch(self, batch: NodeBatch) -> None:
        """Clear pre-plan state on the slot that ``pre_plan_for_batch``
        targeted for this batch. Used to recover from speculation drops
        or pre-plan failures without disturbing other slots' valid
        pre-plan state. No-op if no captured graph matches.
        """
        runner = self.submodule_management[batch.node_name].cuda_graph_runner
        if runner is None or not runner.graphs:
            return
        has_cfg = any(
            info.requires_cfg for info in batch.per_request_info.values()
        )
        bs = len(batch.request_ids)
        slot = batch.metadata.get("cuda_graph_slot")
        runner.reset_pre_plan_state_for_slot(
            graph_walk=batch.graph_walk,
            requires_cfg=has_cfg,
            batch_size=bs,
            slot=slot,
        )

    def pre_plan_for_batch(
        self,
        batch: NodeBatch,
        prev_completion_event: "torch.cuda.Event | None" = None,
    ) -> bool:
        """Phase 3: pre-plan FlashInfer attention for a batch on the
        Worker.plan_executor thread, so the GPU thread's preprocess can skip
        the GIL-contended plan() call.

        With double-buffer, the slot has already been reserved by
        ``reserve_replay_slot`` and lives on ``batch.metadata['cuda_graph_slot']``.
        We forward it to the runner so plan() targets the inactive slot's
        wrapper (the one replay(N) is NOT using).

        Returns True if pre-planning was applied (caller's GPU thread should
        wait on the plan future before running this batch). False if no
        captured graph matches, in which case the GPU thread plans inline.
        """
        runner = self.submodule_management[batch.node_name].cuda_graph_runner
        if runner is None or not runner.graphs:
            return False
        has_cfg = any(
            info.requires_cfg for info in batch.per_request_info.values()
        )
        slot = batch.metadata.get("cuda_graph_slot")
        return runner.pre_plan_for_batch(
            graph_walk=batch.graph_walk,
            requires_cfg=has_cfg,
            request_ids=list(batch.request_ids),
            per_request_info=batch.per_request_info,
            prev_completion_event=prev_completion_event,
            slot=slot,
        )

    def add_request(
        self, request_id: str, cache_labels: list[str] | None = None,
    ) -> None:
        for submodule_mgmt in self.submodule_management.values():
            submodule_mgmt.kv_management.alloc_manager.add_request(request_id, cache_labels or ["main"])
            submodule_mgmt.sampler.add_request(request_id)
            # Mirror into the cuda-graph runner's master sampler buffers so
            # the per-step path can index_select instead of rebuilding from
            # Python (see ``SamplerBuffers.gather_for_request_ids``).
            if submodule_mgmt.cuda_graph_runner is not None:
                submodule_mgmt.cuda_graph_runner.register_request(request_id)

    def remove_request(self, request_id: str) -> None:
        for submodule_mgmt in self.submodule_management.values():
            cache_mgmt = submodule_mgmt.kv_management
            if cache_mgmt.cpu_page_pool is not None:
                cache_mgmt.cpu_page_pool.remove_request(request_id)
            cache_mgmt.alloc_manager.remove_request(request_id)
            submodule_mgmt.sampler.remove_request(request_id)
            if submodule_mgmt.cuda_graph_runner is not None:
                submodule_mgmt.cuda_graph_runner.unregister_request(request_id)

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
