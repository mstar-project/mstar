import logging
import queue
from dataclasses import dataclass, field

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.engine.cuda_graph_runner import CudaGraphRunner
from mminf.utils.flashinfer_utils import FlashInferDecodeWrapper, FlashInferPrefillWrapper
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)

@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    num_qo_heads: int | None = None  # Optional, defaults to num_kv_heads

    def __post_init__(self):
        if self.num_qo_heads is None:
            self.num_qo_heads = self.num_kv_heads


@dataclass
class KVRequestState:
    """Per-request KV cache state for the AR engine."""
    page_indices: list[int] = field(default_factory=list)
    seq_len: int = 0
    position_id_start: int = 0
    is_paused: bool = False


class PageAllocator:
    """Simple page allocator using a FIFO queue of free page indices."""

    def __init__(self, max_num_pages: int):
        self.max_num_pages = max_num_pages
        self.free_pages: queue.Queue[int] = queue.Queue()
        for i in range(max_num_pages):
            self.free_pages.put(i)

    def allocate(self, n: int) -> list[int]:
        if self.free_pages.qsize() < n:
            raise RuntimeError(
                f"Not enough free pages: requested {n}, "
                f"available {self.free_pages.qsize()}"
            )
        return [self.free_pages.get() for _ in range(n)]

    def free(self, pages: list[int]) -> None:
        for page in pages:
            self.free_pages.put(page)

    @property
    def num_free(self) -> int:
        return self.free_pages.qsize()


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
        workspace_buffer: torch.Tensor,
        kv_cache_config: KVCacheConfig,
        device,
        cuda_graph_plan_states: dict[str, _PlanState] | None = None,
    ):
        self.request_ids = request_ids
        self.active_labels = active_labels_per_request  # {req_id: label}
        self.kv_cache = kv_cache
        self.page_allocator = page_allocator
        self.request_states = request_states
        self.workspace_buffer = workspace_buffer
        self.kv_cache_config = kv_cache_config
        self.device = device
        self.layer_idx = 0

        # CUDA graph mode: persistent wrappers passed in from CudaGraphRunner.
        # When set, plan_attention() uses the persistent wrapper's plan()
        # method instead of creating a new wrapper each call.
        self._cuda_graph_mode = cuda_graph_plan_states is not None

        # Per-label plan state: plan_attention/plan_rope store results here,
        # run_attention/apply_rope look up by active label.
        if cuda_graph_plan_states is not None:
            self._plan_states: dict[str, _PlanState] = cuda_graph_plan_states
        else:
            self._plan_states: dict[str, _PlanState] = {}

        self.base_pos_ids = torch.arange(
            kv_cache_config.max_seq_len, dtype=torch.long, device=device
        )

    def _get_state(self, request_id: str, label: str | None = None) -> KVRequestState:
        label = label or self.active_labels.get(request_id, "main")
        key = (request_id, label)
        if key not in self.request_states:
            self.request_states[key] = KVRequestState()
        return self.request_states[key]

    def set_active_labels(self, labels: dict[str, str]) -> None:
        """Switch active cache labels for all requests at once."""
        self.active_labels = labels

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

        if self._cuda_graph_mode:
            # CUDA graph mode: use persistent wrapper, call its plan() method.
            # The wrapper was created by CudaGraphRunner and stored in _plan_states.
            ps = self._plan_states[effective_label]
            wrapper = ps.wrapper

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
        else:
            # Eager mode: reuse wrapper if one already exists for this label.
            # Avoids recreating FlashInfer wrapper objects on every call.
            ps = self._plan_states.get(effective_label)
            if ps is not None and ps.wrapper is not None:
                wrapper = ps.wrapper
            else:
                import flashinfer
                wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer, "NHD"
                )
                ps = _PlanState(wrapper=wrapper)
                self._plan_states[effective_label] = ps

            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=is_causal,
                q_data_type=dtype,
            )

            ps.page_indices = page_indices
            ps.page_offsets = page_offsets
            ps.token_offsets = token_offsets
            ps.seq_lens = seq_lens
    
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
        label = next(iter(self.active_labels.values()))
        ps = self._plan_states[label]
        assert self.kv_cache is not None and ps.wrapper is not None

        if self._cuda_graph_mode:
            # CUDA graph mode: use wrapper's set_kv_cache + run
            ps.wrapper.set_kv_cache(self.kv_cache[layer_idx], k, v)
            return ps.wrapper.run(q, self.kv_cache[layer_idx])
        else:
            # Eager mode: direct fancy indexing for KV writes
            self.kv_cache[layer_idx, ps.page_indices, 0, ps.page_offsets] = k[ps.token_offsets]
            self.kv_cache[layer_idx, ps.page_indices, 1, ps.page_offsets] = v[ps.token_offsets]
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
    ):
        """Apply RoPE using the active label's pre-computed position IDs."""
        label = next(iter(self.active_labels.values()))
    
        ps = self._plan_states[label]
        assert ps.pos_ids is not None

        import flashinfer
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q, k, ps.pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return q, k

    def advance_seq_len(self, n: int | None = None, pos_id_n: int | None = None) -> None:
        """Advance seq_len for all requests.

        Uses provided n and/or pos_id_n if they exist, then falls back to
        per-request seq_lens from the last plan_attention call. Errors if
        both n and planned seq_lens are None.
        """
        if n is not None:
            for rid in self.request_ids:
                state = self._get_state(rid)
                state.seq_len += n
                state.position_id_start += (pos_id_n if pos_id_n is not None else n)
        else:
            # Fall back to planned seq_lens from active label's plan state
            label = next(iter(self.active_labels.values()))
            ps = self._plan_states.get(label)
            planned = ps.seq_lens if ps is not None else None
            if planned is None:
                raise ValueError(
                    "advance_seq_len: both n and planned seq_lens are None"
                )
            for i, rid in enumerate(self.request_ids):
                state = self._get_state(rid)
                state.seq_len += planned[i]
                state.position_id_start += (
                    pos_id_n if pos_id_n is not None else planned[i]
                )

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
        enable_nvtx: bool = False,
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        if isinstance(kv_cache_config, dict):
            kv_cache_config = KVCacheConfig(**kv_cache_config)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.kv_cache_config = kv_cache_config
        self.device = None
        self.kv_cache = None  # [num_layers, max_pages, 2, page_size, num_kv_heads, head_dim]
        self.page_allocator: PageAllocator | None = None
        # Keyed by (request_id, cache_label) to support multiple KV caches
        # per request (needed for BAGEL's classifier-free guidance which
        # maintains "main", "cfg_text", and "cfg_img" caches).
        self.request_states: dict[tuple[str, str], KVRequestState] = {}

        # FlashInfer wrappers (initialized in load_model if available)
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.workspace_buffer = None

        # CUDA graph runners (initialized in warmup())
        self.cuda_graph_runners: dict[str, "CudaGraphRunner"] = {}

    def engine_type(self) -> EngineType:
        return EngineType.AR

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
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
            dtype=torch.bfloat16, device=device,
        )
        self.page_allocator = PageAllocator(max_num_pages)

        # 256MB workspace for FlashInfer
        self.workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device
        )

        try:
            import flashinfer
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
        except ImportError:
            pass  # Dummy mode — no FlashInfer

    def _create_cache_manager(self, request_id: str) -> BatchedCacheManager:
        """Create a CacheHandle for a single request."""
        return BatchedCacheManager(
            request_ids=[request_id],
            active_labels_per_request={request_id: "main"},
            kv_cache=self.kv_cache,
            page_allocator=self.page_allocator,
            request_states=self.request_states,
            workspace_buffer=self.workspace_buffer,
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
            workspace_buffer=self.workspace_buffer,
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

        with torch.no_grad():
            batched_output = submodule.forward_batched(
                graph_walk=batch.graph_walk,
                cache_manager=cache_manager,
                packed_inputs=preprocessed,
                request_ids=rids,
                per_request_metadata=batch.per_request_metadata,
            )

        return NodeOutput(per_request_output_tensors=batched_output)

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
            with torch.no_grad():
                output = submodule(
                    graph_walk=batch.graph_walk,
                    cache_handle=cache_handle,
                    **preprocessed,
                    **metadata[rid],
                )
            per_request_outputs[rid] = output

        return NodeOutput(per_request_output_tensors=per_request_outputs)

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

        with torch.no_grad():
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
        self.workspace_buffer = None
        self.request_states.clear()
