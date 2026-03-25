import logging
from dataclasses import dataclass

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cuda_graph_runner import CudaGraphRunner
from mminf.engine.pd_disaggregation import KVCacheSender
from mminf.engine.paged_attention import KVCacheConfig, KVRequestState, PageAllocator
from mminf.utils.flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)

class WorkspaceBufferManager:
    def __init__(
        self, size, device
    ):
        self.size = size
        self.device = device
        self.buffers = {}
    
    def get(self, label: str="main"):
        if label not in self.buffers:
            self.buffers[label] = torch.empty(
                self.size, dtype=torch.uint8, device=self.device
            )
        return self.buffers[label]
            

@dataclass
class _PlanState:
    """Pre-computed state from plan_attention/plan_rope for a single cache label.

    Stored per-label so that preprocess can plan for all relevant labels
    upfront (plan operations are CUDA graph incompatible). During forward,
    run_attention/apply_rope look up the active label's plan state.

    In CUDA graph mode, wrapper is a persistent FlashInferPrefillWrapper or
    FlashInferDecodeWrapper created once during capture. plan_attention()
    calls wrapper.plan() which updates static buffers via .copy_().
    """
    wrapper: FlashInferPrefillWrapper | FlashInferDecodeWrapper | None = None
    page_indices: torch.Tensor | None = None
    per_req_page_indices: dict[str, torch.Tensor] | None = None
    seq_len_at_start: int | None = None
    page_offsets: torch.Tensor | None = None
    token_offsets: torch.Tensor | None = None
    pos_ids: torch.Tensor | None = None
    seq_lens: list[int] | None = None


class BatchedCacheManager:
    """Manages batched FlashInfer attention for multiple requests simultaneously.

    Replaces per-request CacheHandle for decode and simple prefill batches where
    all requests use the same graph_walk. Constructs batch-level FlashInfer index
    tensors (qo_indptr, paged_kv_indptr, paged_kv_indices) and issues a single
    FlashInfer call per layer instead of N separate calls.

    Keep existing CacheHandle for backward compatibility — complex paths like
    image_gen (3-pass CFG with label switching) continue using per-request
    CacheHandle.
    """

    def __init__(
        self,
        request_ids: list[str],
        active_labels_per_request: dict[str, str],
        kv_cache: torch.Tensor,
        page_allocator: PageAllocator,
        request_states: dict[tuple[str, str], KVRequestState],
        buffer_manager: WorkspaceBufferManager,
        kv_cache_config: KVCacheConfig,
        device,
        cuda_graph_plan_states: dict[str, _PlanState] | None = None,
        kv_cache_senders: dict[str, KVCacheSender] = {}
    ):
        self.request_ids = request_ids
        self.active_labels = active_labels_per_request  # {req_id: label}
        self.kv_cache = kv_cache
        self.page_allocator = page_allocator
        self.request_states = request_states
        self.buffer_manager = buffer_manager
        self.kv_cache_config = kv_cache_config
        self.device = device
        self.layer_idx = 0

        self.kv_cache_senders = kv_cache_senders

        # CUDA graph mode: persistent wrappers passed in from CudaGraphRunner.
        # When set, plan_attention() uses the persistent wrapper's plan()
        # method instead of creating a new wrapper each call.
        self._cuda_graph_mode = cuda_graph_plan_states is not None
        if self._cuda_graph_mode:
            for req_id in kv_cache_senders:
                kv_cache_senders[req_id].delayed_transfer = True

        # Per-label plan state: plan_attention/plan_rope store results here,
        # run_attention/apply_rope look up by active label.
        if cuda_graph_plan_states is not None:
            self._plan_states: dict[str, _PlanState] = cuda_graph_plan_states
        else:
            self._plan_states: dict[str, _PlanState] = {}

        self.base_pos_ids = torch.arange(
            kv_cache_config.max_seq_len, dtype=torch.long, device=device
        )

    @torch.compiler.disable
    def _get_state(self, request_id: str, label: str | None = None) -> KVRequestState:
        label = label or self.active_labels.get(request_id, "main")
        key = (request_id, label)
        if key not in self.request_states:
            self.request_states[key] = KVRequestState()
        return self.request_states[key]

    @torch.compiler.disable
    def set_active_labels(self, labels: dict[str, str]) -> None:
        """Switch active cache labels for all requests at once."""
        self.active_labels = labels

    @torch.compiler.disable
    def set_active_label(self, label: str) -> None:
        """Switch all requests to the same cache label."""
        self.active_labels = {rid: label for rid in self.request_ids}
    
    @torch.compiler.disable
    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = layer_idx

    def plan_attention(
        self,
        seq_lens: list[int] | None = None,
        dtype=torch.bfloat16,
        is_causal=True,
        label: str | None = None,
    ):
        """Pre-compute FlashInfer plan and page positions for a cache label.

        Allocates pages, computes page_indices/page_offsets/token_offsets for
        vectorized KV writes, builds FlashInfer index tensors, and plans the
        wrapper. All state is stored in _plan_states[label].

        In CUDA graph mode, uses the persistent wrapper from _plan_states
        (pre-built by CudaGraphRunner) and calls its plan() method which
        updates static buffers via .copy_(). In eager mode, creates a new
        wrapper each call.

        Args:
            seq_lens: number of new tokens per request.
            dtype: query data type for FlashInfer.
            is_causal: whether attention is causal.
            label: cache label to plan for. If None, uses the current active label.
        """
        assert self.kv_cache is not None

        effective_label = label
        if effective_label is None:
            labels = list(self.active_labels.values())
            assert len(set(labels)) == 1, f"All active labels must be the same, got {labels}"
            effective_label = next(iter(self.active_labels.values()))

        cfg = self.kv_cache_config
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim
        num_qo_heads = cfg.num_qo_heads
        device = self.device

        qo_indptr_list = [0]
        kv_indptr_list = [0]
        all_page_indices = []
        kv_last_page_lens = []

        page_indices_all = []
        page_offsets_all = []

        per_req_page_indices = {}
        for i, rid in enumerate(self.request_ids):
            state = self._get_state(rid, effective_label)
            sl = seq_lens[i]

            # Allocate pages if needed
            total_len = state.seq_len + sl
            num_pages_needed = (total_len + page_size - 1) // page_size
            num_new_pages = num_pages_needed - len(state.page_indices)
            if num_new_pages > 0:
                new_pages = self.page_allocator.allocate(num_new_pages)
                state.page_indices.extend(new_pages)

            # Compute positions for this request
            pos = torch.arange(state.seq_len, state.seq_len + sl, device=self.device)

            page_idx = torch.tensor(state.page_indices, device=self.device)[pos // page_size]
            page_offset = pos % page_size

            page_indices_all.append(page_idx)
            per_req_page_indices[rid] = page_idx
            page_offsets_all.append(page_offset)

            # Build indptr entries
            qo_indptr_list.append(qo_indptr_list[-1] + sl)
            all_page_indices.extend(state.page_indices)
            kv_indptr_list.append(kv_indptr_list[-1] + len(state.page_indices))

            last_page_len = total_len % page_size or page_size
            kv_last_page_lens.append(last_page_len)

        page_indices = torch.cat(page_indices_all)
        page_offsets = torch.cat(page_offsets_all)
        token_offsets = torch.arange(page_indices.numel(), device=self.device)

        # Build batched FlashInfer index tensors
        qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device=device)
        paged_kv_indices = torch.tensor(all_page_indices, dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32, device=device)


        is_decode = all([sl == 1 for sl in seq_lens])
        ps = self._plan_states.get(effective_label)
        if ps is not None and ps.wrapper is not None:
            wrapper = ps.wrapper
        elif is_decode:
            wrapper = FlashInferDecodeWrapper(
                workspace_buffer=self.buffer_manager.get(effective_label),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
            )
            ps = _PlanState(wrapper=wrapper)
            self._plan_states[effective_label] = ps
        else:
            wrapper = FlashInferPrefillWrapper(
                workspace_buffer=self.buffer_manager.get(effective_label),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
            )
            ps = _PlanState(wrapper=wrapper)
            self._plan_states[effective_label] = ps

        if isinstance(wrapper, FlashInferDecodeWrapper):
            wrapper.plan(
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                dtype=dtype,
            )
        else:
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                causal=is_causal,
                dtype=dtype,
            )
        # Note: plain assignment (not .copy_()) is intentional here.
        # These fields are NOT read inside the CUDA graph — run_attention()
        # uses wrapper.set_kv_cache()/run() which rely on the wrapper's own
        # internal static tensors (updated via .copy_() inside wrapper.plan()).
        # These are stored on _PlanState only for bookkeeping/debugging.
        ps.page_indices = page_indices
        ps.page_offsets = page_offsets
        ps.token_offsets = token_offsets
        ps.seq_lens = seq_lens
        ps.per_req_page_indices = per_req_page_indices
        ps.seq_len_at_start = self._get_state(rid, effective_label).seq_len
    
    def plan_rope(
        self,
        seq_lens: list[int],
        pos_ids: torch.Tensor | None = None,
        label: str | None = None,
    ):
        """Pre-compute position IDs for RoPE for a cache label.

        In CUDA graph mode, updates the static pos_ids tensor via .copy_()
        so that the same GPU address is used during graph replay.

        Args:
            seq_lens: number of new tokens per request.
            pos_ids: explicit position IDs. If None, computed from
                each request's position_id_start.
            label: cache label. If None, uses the current active label.
        """
        effective_label = label
        if effective_label is None:
            labels = list(self.active_labels.values())
            assert len(set(labels)) == 1, f"All active labels must be the same, got {labels}"
            effective_label = next(iter(self.active_labels.values()))

        if effective_label not in self._plan_states:
            self._plan_states[effective_label] = _PlanState()

        computed_pos_ids = pos_ids
        if computed_pos_ids is None:
            computed_pos_ids = torch.cat([
                torch.arange(sl, device=self.device, dtype=torch.long)
                + self._get_state(rid, effective_label).position_id_start
                for rid, sl in zip(self.request_ids, seq_lens)
            ])

        if self._cuda_graph_mode:
            # Update static buffer via .copy_() for CUDA graph compatibility
            ps = self._plan_states[effective_label]
            if ps.pos_ids is not None:
                n = computed_pos_ids.shape[0]
                ps.pos_ids[:n].copy_(computed_pos_ids)
            else:
                ps.pos_ids = computed_pos_ids
        else:
            self._plan_states[effective_label].pos_ids = computed_pos_ids

    @torch.compiler.disable
    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int | None=None,
    ) -> torch.Tensor:
        """Run pre-planned FlashInfer attention with KV cache write.

        Uses the active label's plan state (set up by a prior plan_attention
        call). Writes K and V to the paged KV cache at pre-computed page
        positions, then runs the FlashInfer wrapper for batched attention.

        In CUDA graph mode, uses wrapper.set_kv_cache() + wrapper.run()
        which operates on pre-computed token_to_page/token_to_cache or
        kv_cache_locations tensors (static GPU addresses).

        In eager mode, uses direct fancy indexing for KV writes and
        the raw FlashInfer wrapper's run().

        Args:
            q: [total_tokens, num_q_heads, head_dim]
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
            layer_idx: transformer layer index
        Returns:
            output: [total_tokens, num_q_heads, head_dim]
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        labels = list(self.active_labels.values())
        assert len(set(labels)) == 1, f"All active labels must be the same, got {labels}"
        label = next(iter(self.active_labels.values()))

        ps = self._plan_states[label]
        assert self.kv_cache is not None and ps.wrapper is not None

        ps.wrapper.set_kv_cache(self.kv_cache[layer_idx], k, v)

        for req_id in self.kv_cache_senders:
            self.kv_cache_senders[req_id].transfer_layer(
                layer_idx=layer_idx,
                label=label,
                page_idxs=ps.per_req_page_indices[req_id],
                seq_len_at_start=ps.seq_len_at_start
            )

        return ps.wrapper.run(q, self.kv_cache[layer_idx])

    @torch.compiler.disable
    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
        rope_dtype=None
    ):
        """Apply RoPE using the active label's pre-computed position IDs."""
        # Assert all active labels are the same
        labels = list(self.active_labels.values())
        assert len(set(labels)) == 1, f"All active labels must be the same, got {labels}"
        label = next(iter(self.active_labels.values()))
    
        ps = self._plan_states[label]
        assert ps.pos_ids is not None

        orig_dtype = q.dtype

        if rope_dtype is not None:
            q, k = q.to(rope_dtype), k.to(rope_dtype)
        elif torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            q, k = q.to(dtype), k.to(dtype)

        import flashinfer
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q, k, ps.pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return q.to(orig_dtype), k.to(orig_dtype)
    
    @torch.compiler.disable
    def apply_rope_llama(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
        rope_dtype=None
    ):
        """Apply RoPE using the active label's pre-computed position IDs."""
        labels = list(self.active_labels.values())
        assert len(set(labels)) == 1, f"All active labels must be the same, got {labels}"
        label = next(iter(self.active_labels.values()))
    
        ps = self._plan_states[label]
        assert ps.pos_ids is not None

        orig_dtype = q.dtype
        if rope_dtype is not None:
            q, k = q.to(rope_dtype), k.to(rope_dtype)
        elif torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            q, k = q.to(dtype), k.to(dtype)

        import flashinfer
        flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
            q, k, ps.pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return q.to(orig_dtype), k.to(orig_dtype)

    @torch.compiler.disable
    def advance_seq_len(self, n: int | None = None, pos_id_n: int | None = None) -> None:
        """Advance seq_len for all requests.

        Uses provided n and/or pos_id_n if they exist, then falls back to
        per-request seq_lens from the last plan_attention call. Errors if
        both n and planned seq_lens are None.
        """
        if n is None:
            return self.advance_seq_lens(pos_id_n)
        for rid in self.request_ids:
            state = self._get_state(rid)
            state.seq_len += n
            state.position_id_start += (pos_id_n if pos_id_n is not None else n)
    
    @torch.compiler.disable
    def advance_seq_lens(self, pos_id_ns: list[int] | int | None = None) -> None:
        """Advance seq_len for each request by different amounts."""
        for i, rid in enumerate(self.request_ids):
            n = self._plan_states[self.active_labels[rid]].seq_lens[i]
            state = self._get_state(rid)
            state.seq_len += n
            if pos_id_ns is None:
                state.position_id_start += n
            elif isinstance(pos_id_ns, int):
                state.position_id_start += pos_id_ns
            else:
                state.position_id_start += pos_id_ns[i]

    def snapshot_all(self, from_label: str, to_label: str) -> None:
        """Snapshot KV cache for all requests in batch."""
        for rid in self.request_ids:
            from_state = self._get_state(rid, from_label)
            to_key = (rid, to_label)

            # Free old pages if target already exists
            if to_key in self.request_states:
                old_state = self.request_states[to_key]
                if old_state.page_indices:
                    self.page_allocator.free(old_state.page_indices)

            num_pages = len(from_state.page_indices)
            new_pages = self.page_allocator.allocate(num_pages) if num_pages > 0 else []

            for src_page, dst_page in zip(from_state.page_indices, new_pages, strict=False):
                self.kv_cache[:, dst_page] = self.kv_cache[:, src_page]

            self.request_states[to_key] = KVRequestState(
                page_indices=new_pages,
                seq_len=from_state.seq_len,
                position_id_start=from_state.position_id_start,
            )
    
    def finish_transfers(self):
        for rid in self.kv_cache_senders:
            final_seq_len = {}
            next_pos_id = {}
            seq_len_at_start = {}
            labels = []

            for label, plan_state in self._plan_states.items():
                state = self._get_state(rid, label)
                if state.seq_len - plan_state.seq_len_at_start == 0:
                    continue
                labels.append(label)
                final_seq_len[label] = state.seq_len
                next_pos_id[label] = state.position_id_start
                seq_len_at_start[label] = plan_state.seq_len_at_start

            self.kv_cache_senders[rid].finish_transfer(
                request_id=rid,
                labels=labels,
                num_layers=self.kv_cache_config.num_layers,
                final_seq_len=final_seq_len,
                next_pos_id=next_pos_id,
                seq_len_at_start=seq_len_at_start
            )


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
        self.page_allocator: PageAllocator | None = None
        # Keyed by (request_id, cache_label) to support multiple KV caches
        # per request (needed for BAGEL's classifier-free guidance which
        # maintains "main", "cfg_text", and "cfg_img" caches).
        self.request_states: dict[tuple[str, str], KVRequestState] = {}
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
        )
        self.page_allocator = PageAllocator(max_num_pages)

        # 256MB workspace for FlashInfer
        self.buffer_manager = WorkspaceBufferManager(
            256 * 1024 * 1024, device=device
        )

    def _create_cache_manager(self, request_id: str) -> BatchedCacheManager:
        """Create a CacheHandle for a single request."""
        return BatchedCacheManager(
            request_ids=[request_id],
            active_labels_per_request={request_id: "main"},
            kv_cache=self.kv_cache,
            page_allocator=self.page_allocator,
            request_states=self.request_states,
            buffer_manager=self.buffer_manager,
            kv_cache_config=self.kv_cache_config,
            device=self.device
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
            runner = CudaGraphRunner(self, node_name, self.kv_cache_config)
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
        per_request_metadata: dict[str, dict],
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
            meta = per_request_metadata.get(rid, {})
            temperature = meta.get("temperature", 0.6)
            top_k = meta.get("top_k", 0)
            top_p = meta.get("top_p", 1.0)
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
        cache_manager = BatchedCacheManager(
            request_ids=batch.request_ids,
            active_labels_per_request={rid: "main" for rid in batch.request_ids},
            kv_cache=self.kv_cache,
            page_allocator=self.page_allocator,
            request_states=self.request_states,
            buffer_manager=self.buffer_manager,
            kv_cache_config=self.kv_cache_config,
            device=self.device,
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
            per_request_metadata=batch.per_request_metadata,
        )

        batched_output = submodule.forward_batched(
            graph_walk=batch.graph_walk,
            cache_manager=cache_manager,
            packed_inputs=preprocessed,
            request_ids=rids,
            per_request_metadata=batch.per_request_metadata,
        )

        output = NodeOutput(per_request_output_tensors=batched_output)
        if batch.graph_walk == "decode":
            output = self._sample_decode_outputs(output, batch.per_request_metadata)
        return output

    def _execute_sequential(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Original per-request execution with CacheHandle."""
        per_request_outputs = {}
        
        for rid in batch.request_ids:
            cache_handle = self._create_cache_manager(rid)
            inputs = batch.per_request_input_tensors.get(rid, {})
            metadata = {rid: batch.per_request_metadata.get(rid, {})}

            seq_lens = {
                rid: cache_handle._get_state(rid, "main").seq_len
            }
            logger.debug(f"Execute sequential {seq_lens}")

            preprocessed = submodule.preprocess(
                batch.graph_walk,
                cache_manager=cache_handle,
                per_request_inputs=[inputs],
                request_ids=[rid],
                per_request_metadata=metadata,
            )
            output = submodule(
                graph_walk=batch.graph_walk,
                cache_handle=cache_handle,
                **preprocessed,
                **metadata[rid],
            )
            per_request_outputs[rid] = output

        output = NodeOutput(per_request_output_tensors=per_request_outputs)
        if batch.graph_walk == "decode":
            output = self._sample_decode_outputs(output, batch.per_request_metadata)
        return output

    def _can_use_cuda_graph(self, batch: NodeBatch) -> bool:
        """Check if CUDA graph replay is available for this batch."""
        if batch.graph_walk != "decode":
            return False
        runner = self.cuda_graph_runners.get(batch.node_name)
        if runner is None:
            return False

        has_cfg = any(
            batch.per_request_metadata.get(rid, {}).get("requires_cfg", False)
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
            batch.per_request_metadata.get(rid, {}).get("requires_cfg", False)
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
            per_request_metadata=batch.per_request_metadata,
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
            if self.enable_nvtx:
                range_pop()

    def add_request(
        self, request_id: str, cache_labels: list[str] | None = None,
    ) -> None:
        labels = cache_labels or ["main"]
        for label in labels:
            self.request_states[(request_id, label)] = KVRequestState()

    def remove_request(self, request_id: str) -> None:
        keys_to_remove = [
            k for k in self.request_states if k[0] == request_id
        ]
        for key in keys_to_remove:
            if self.page_allocator is not None:
                self.page_allocator.free(self.request_states[key].page_indices)
            del self.request_states[key]

    def pause_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """For interleaved loop: mark as paused, keep KV pages allocated."""
        key = (request_id, cache_label)
        if key in self.request_states:
            self.request_states[key].is_paused = True

    def resume_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """Resume from paused state for next LLM step in loop."""
        key = (request_id, cache_label)
        if key in self.request_states:
            self.request_states[key].is_paused = False

    def shutdown(self) -> None:
        self.kv_cache = None
        self.buffer_manager = None
        self.request_states.clear()
