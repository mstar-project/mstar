import queue
from dataclasses import dataclass, field

import torch

from mminf.engine.base import BaseEngine, EngineType, StageBatch, StageOutput


@dataclass
class KVRequestState:
    """Per-request KV cache state for the AR engine."""
    page_indices: list[int] = field(default_factory=list)
    seq_len: int = 0
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


class AREngine(BaseEngine):
    """
    Autoregressive engine with paged KV cache.
    Uses FlashInfer for prefill/decode when available.
    Supports pause/resume for interleaved loops (LLM <-> flow).
    """

    def __init__(
        self,
        model: torch.nn.Module | None = None,
        kv_cache_config: dict | None = None,
    ):
        self.model = model
        self.kv_cache_config = kv_cache_config or {}
        self.device = None
        self.kv_cache = None  # [num_layers, max_pages, 2, page_size, num_kv_heads, head_dim]
        self.page_allocator: PageAllocator | None = None
        self.request_states: dict[str, KVRequestState] = {}

        # FlashInfer wrappers (initialized in load_model if available)
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.workspace_buffer = None

    def engine_type(self) -> EngineType:
        return EngineType.AR

    def load_model(self, model_config: dict, device: torch.device) -> None:
        self.device = device
        cfg = model_config.get("kv_cache", self.kv_cache_config)
        if not cfg:
            return  # dummy mode without config

        num_layers = cfg["num_layers"]
        max_num_pages = cfg["max_num_pages"]
        page_size = cfg["page_size"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]

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

    def execute_batch(self, batch: StageBatch) -> StageOutput:
        if self.model is None:
            # Dummy mode: return empty output per request
            return StageOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        is_prefill = batch.metadata.get("is_prefill", False)

        if is_prefill:
            return self._execute_prefill(batch)
        else:
            return self._execute_decode(batch)

    def _execute_prefill(self, batch: StageBatch) -> StageOutput:
        """Run prefill (prompt processing) for a batch of requests."""
        if self.prefill_wrapper is None or self.kv_cache is None:
            return StageOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        cfg = self.kv_cache_config
        page_size = cfg["page_size"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        num_qo_heads = cfg.get("num_qo_heads", num_kv_heads)

        # Build FlashInfer arguments from request states
        qo_indptr = [0]
        paged_kv_indptr = [0]
        paged_kv_indices = []
        paged_kv_last_page_len = []
        all_q = []

        for rid in batch.request_ids:
            state = self.request_states[rid]
            inputs = batch.per_request_input_tensors.get(rid, {})
            q = inputs.get("hidden_states", inputs.get("text_emb"))
            if q is None:
                continue
            seq_len = q.shape[0]

            # Allocate pages for this sequence
            num_pages_needed = (state.seq_len + seq_len + page_size - 1) // page_size
            num_new_pages = num_pages_needed - len(state.page_indices)
            if num_new_pages > 0:
                new_pages = self.page_allocator.allocate(num_new_pages)
                state.page_indices.extend(new_pages)

            state.seq_len += seq_len
            last_page_len = state.seq_len % page_size or page_size

            qo_indptr.append(qo_indptr[-1] + seq_len)
            paged_kv_indptr.append(paged_kv_indptr[-1] + len(state.page_indices))
            paged_kv_indices.extend(state.page_indices)
            paged_kv_last_page_len.append(last_page_len)
            all_q.append(q)

        if not all_q:
            return StageOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        q_tensor = torch.cat(all_q, dim=0)
        qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device=self.device)
        kv_indptr_t = torch.tensor(paged_kv_indptr, dtype=torch.int32, device=self.device)
        kv_indices_t = torch.tensor(paged_kv_indices, dtype=torch.int32, device=self.device)
        kv_last_page_t = torch.tensor(paged_kv_last_page_len, dtype=torch.int32, device=self.device)

        self.prefill_wrapper.plan(
            qo_indptr_t, kv_indptr_t, kv_indices_t, kv_last_page_t,
            num_qo_heads, num_kv_heads, head_dim, page_size, causal=True,
        )

        # Run model forward (attention handled by FlashInfer wrapper)
        with torch.no_grad():
            output = self.model(q_tensor, self.kv_cache, self.prefill_wrapper)

        # Split outputs back per request
        per_request_outputs = {}
        offset = 0
        for rid in batch.request_ids:
            seq_len = qo_indptr[batch.request_ids.index(rid) + 1] - qo_indptr[batch.request_ids.index(rid)]
            per_request_outputs[rid] = {"hidden_states": output[offset:offset + seq_len]}
            offset += seq_len

        return StageOutput(per_request_output_tensors=per_request_outputs)

    def _execute_decode(self, batch: StageBatch) -> StageOutput:
        """Run decode (single token generation) for a batch of requests."""
        if self.decode_wrapper is None or self.kv_cache is None:
            return StageOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        cfg = self.kv_cache_config
        page_size = cfg["page_size"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        num_qo_heads = cfg.get("num_qo_heads", num_kv_heads)

        indptr = [0]
        indices = []
        last_page_len = []
        all_q = []

        for rid in batch.request_ids:
            state = self.request_states[rid]
            inputs = batch.per_request_input_tensors.get(rid, {})
            q = inputs.get("hidden_states", inputs.get("text_emb"))
            if q is None:
                continue

            # Allocate one more page if needed
            state.seq_len += 1
            num_pages_needed = (state.seq_len + page_size - 1) // page_size
            num_new_pages = num_pages_needed - len(state.page_indices)
            if num_new_pages > 0:
                new_pages = self.page_allocator.allocate(num_new_pages)
                state.page_indices.extend(new_pages)

            lpl = state.seq_len % page_size or page_size

            indptr.append(indptr[-1] + len(state.page_indices))
            indices.extend(state.page_indices)
            last_page_len.append(lpl)
            all_q.append(q)

        if not all_q:
            return StageOutput(
                per_request_output_tensors={
                    rid: {} for rid in batch.request_ids
                }
            )

        q_tensor = torch.cat(all_q, dim=0)
        indptr_t = torch.tensor(indptr, dtype=torch.int32, device=self.device)
        indices_t = torch.tensor(indices, dtype=torch.int32, device=self.device)
        last_page_t = torch.tensor(last_page_len, dtype=torch.int32, device=self.device)

        self.decode_wrapper.plan(
            indptr_t, indices_t, last_page_t,
            num_qo_heads, num_kv_heads, head_dim, page_size,
        )

        with torch.no_grad():
            output = self.model(q_tensor, self.kv_cache, self.decode_wrapper)

        per_request_outputs = {}
        for i, rid in enumerate(batch.request_ids):
            per_request_outputs[rid] = {"hidden_states": output[i:i + 1]}

        return StageOutput(per_request_output_tensors=per_request_outputs)

    def add_request(self, request_id: str) -> None:
        self.request_states[request_id] = KVRequestState()

    def remove_request(self, request_id: str) -> None:
        if request_id in self.request_states:
            if self.page_allocator is not None:
                self.page_allocator.free(self.request_states[request_id].page_indices)
            del self.request_states[request_id]

    def pause_request(self, request_id: str) -> None:
        """For interleaved loop: mark as paused, keep KV pages allocated."""
        if request_id in self.request_states:
            self.request_states[request_id].is_paused = True

    def resume_request(self, request_id: str) -> None:
        """Resume from paused state for next LLM step in loop."""
        if request_id in self.request_states:
            self.request_states[request_id].is_paused = False

    def shutdown(self) -> None:
        self.kv_cache = None
        self.workspace_buffer = None
        self.request_states.clear()
