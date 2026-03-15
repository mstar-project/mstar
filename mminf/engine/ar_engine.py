import logging
import queue
from dataclasses import dataclass, field

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput

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


class CacheHandle:
    """Per-request cache handle. Created by AR engine, passed to submodule.forward().

    Provides an interface between the engine's cache infrastructure (FlashInfer,
    page tables, KV tensor) and the submodule's cache semantics (which caches
    to read/write, when to snapshot, how to combine multi-cache outputs).

    The active_label determines which KV cache subsequent run_attention calls
    operate on. For single-cache models this is always "main"; for CFG models
    like BAGEL it switches between "main", "cfg_text", "cfg_img".
    """

    def __init__(
        self,
        request_id: str,
        kv_cache: torch.Tensor | None,
        page_allocator: PageAllocator | None,
        request_states: dict[tuple[str, str], KVRequestState],
        workspace_buffer: torch.Tensor | None,
        kv_cache_config: KVCacheConfig,
        device,
    ):
        self.request_id = request_id
        self.kv_cache = kv_cache
        self.page_allocator = page_allocator
        self.request_states = request_states
        self.workspace_buffer = workspace_buffer
        self.kv_cache_config = kv_cache_config
        self.max_seq_len = kv_cache_config.max_seq_len
        self.active_label = "main"

        # for RoPE, may be moved!
        self.base_pos_ids = torch.arange(
            self.max_seq_len, dtype=torch.long, device=device
        )
        self.device = device

    def _get_state(self, label: str | None = None) -> KVRequestState:
        label = label or self.active_label
        key = (self.request_id, label)
        if key not in self.request_states:
            self.request_states[key] = KVRequestState()
        return self.request_states[key]

    def set_active_label(self, label: str) -> None:
        """Switch which KV cache subsequent run_attention calls use."""
        self.active_label = label

    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        is_causal: bool = True,
        write_cache: bool = True,
    ) -> torch.Tensor:
        """Run paged attention for the active cache label at this layer.

        Reads existing K,V from the active label's cache pages, concatenates
        with the new k,v, computes attention, and optionally writes new k,v
        to the cache.

        write_cache=False: use prefix cache for attention but don't persist
        new k,v. Used during flow matching where latent tokens are re-processed
        fresh each step against a frozen prefix.

        Args:
            q: [seq_len, num_q_heads, head_dim]
            k: [seq_len, num_kv_heads, head_dim]
            v: [seq_len, num_kv_heads, head_dim]
            layer_idx: transformer layer index
            is_causal: whether to apply causal masking
            write_cache: whether to persist new k,v to cache pages

        Returns:
            Attention output tensor [seq_len, num_q_heads, head_dim]
        """
        # if self.kv_cache is None:
        #     # Dummy mode: return zeros shaped like q
        #     return torch.zeros_like(q)

        assert self.kv_cache is not None

        state = self._get_state()
        cfg = self.kv_cache_config
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim
        num_qo_heads = cfg.num_qo_heads
        seq_len = q.shape[0]

        # Allocate pages if needed for the new tokens
        total_len = state.seq_len + seq_len
        num_pages_needed = (total_len + page_size - 1) // page_size
        num_new_pages = num_pages_needed - len(state.page_indices)
        if num_new_pages > 0:
            new_pages = self.page_allocator.allocate(num_new_pages)
            state.page_indices.extend(new_pages)

        # Write K,V into cache pages at this layer
        for i in range(seq_len):
            pos = state.seq_len + i
            page_idx = state.page_indices[pos // page_size]
            offset = pos % page_size
            self.kv_cache[layer_idx, page_idx, 0, offset] = k[i]
            self.kv_cache[layer_idx, page_idx, 1, offset] = v[i]

        # logger.warning(f"write_cache is True. num_pages_needed={num_pages_needed}, num_new_pages={num_new_pages}")

        # Build FlashInfer single-request prefill args
        device = q.device
        kv_indptr = torch.tensor(
            [0, len(state.page_indices)], dtype=torch.int32, device=device
        )
        kv_indices = torch.tensor(
            state.page_indices, dtype=torch.int32, device=device
        )

        # logger.warning(f"kv_indptr={kv_indptr}, kv_indices={kv_indices}")
        # logger.warning(f"q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}")

        total_len = state.seq_len + seq_len

        last_page_len = total_len % page_size or page_size
        kv_last_page_len = torch.tensor(
            [last_page_len], dtype=torch.int32, device=device
        )

        try:
            import flashinfer
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )

            qo_indptr = torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=kv_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=is_causal,
                q_data_type=q.dtype,
            )

            # q shape for FlashInfer: [total_tokens, num_heads, head_dim]
            output = wrapper.run(q, self.kv_cache[layer_idx])
        except ImportError as e:
            # No FlashInfer: naive attention fallback (for testing)
            logger.error("Could not run flashinfer. Outputting zeros from run_attention()")
            raise e

        return output


    def apply_rope_default(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
    ):
        offset = self._get_state().position_id_start
        pos_ids = self.base_pos_ids[:q.shape[0]] + offset

        import flashinfer
        return flashinfer.rope.apply_rope_pos_ids(
            q, k, pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

    def apply_rope_custom_pos_ids(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_ids: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
    ):
        import flashinfer
        return flashinfer.rope.apply_rope_pos_ids(
            q, k, pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

    def snapshot(self, from_label: str, to_label: str) -> None:
        """Deepcopy KV cache state (all layers, all pages) from one label to another.

        Allocates new pages for the target label and copies the KV data.
        """
        from_state = self._get_state(from_label)
        to_key = (self.request_id, to_label)

        if self.kv_cache is not None and self.page_allocator is not None:
            # Free old pages if target already exists (prevent page leak)
            if to_key in self.request_states:
                old_state = self.request_states[to_key]
                if old_state.page_indices:
                    self.page_allocator.free(old_state.page_indices)

            # Allocate fresh pages for the target
            num_pages = len(from_state.page_indices)
            new_pages = self.page_allocator.allocate(num_pages) if num_pages > 0 else []

            # Copy KV data page by page across all layers
            for src_page, dst_page in zip(from_state.page_indices, new_pages, strict=False):
                self.kv_cache[:, dst_page] = self.kv_cache[:, src_page]

            self.request_states[to_key] = KVRequestState(
                page_indices=new_pages,
                seq_len=from_state.seq_len,
                position_id_start=from_state.position_id_start,
            )
        else:
            # Dummy mode: just copy the state metadata
            self.request_states[to_key] = KVRequestState(
                page_indices=list(from_state.page_indices),
                seq_len=from_state.seq_len,
                position_id_start=from_state.position_id_start,
            )

    def advance_seq_len(self, n: int, pos_id_n: int | None=None) -> None:
        """Advance the active label's seq_len by n tokens.

        Must be called once after a full forward pass (all layers), not
        per-layer. This keeps seq_len consistent across layers for KV
        cache writes, RoPE offsets, and page allocation.
        """
        self._get_state().seq_len += n
        if pos_id_n is None:
            self._get_state().position_id_start += n
        else:
            self._get_state().position_id_start += pos_id_n

    def save_seq_position(self) -> int:
        """Return current seq_len for the active label (for flow matching rewind)."""
        return self._get_state().seq_len

    def restore_seq_position(self, pos: int) -> None:
        """Rewind the active label's seq_len (discard latent token KV entries).

        Does not deallocate pages — they will be overwritten on next write.
        """
        self._get_state().seq_len = pos


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
        kv_cache_config: KVCacheConfig | dict
    ):
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

    def _create_cache_handle(self, request_id: str) -> CacheHandle:
        """Create a CacheHandle for a single request."""
        return CacheHandle(
            request_id=request_id,
            kv_cache=self.kv_cache,
            page_allocator=self.page_allocator,
            request_states=self.request_states,
            workspace_buffer=self.workspace_buffer,
            kv_cache_config=self.kv_cache_config,
            device=self.device
        )

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            # Dummy mode: return empty output per request
            return NodeOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        # Per-request execution with CacheHandle
        per_request_outputs = {}
        for rid in batch.request_ids:
            cache_handle = self._create_cache_handle(rid)
            inputs = batch.per_request_input_tensors.get(rid, {})
            metadata = batch.per_request_metadata.get(rid, {})

            preprocessed = submodule.preprocess(batch.graph_walk, **inputs)
            with torch.no_grad():
                output = submodule(
                    graph_walk=batch.graph_walk,
                    cache_handle=cache_handle,
                    **preprocessed,
                    **metadata,
                )
            per_request_outputs[rid] = output

        return NodeOutput(per_request_output_tensors=per_request_outputs)

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
