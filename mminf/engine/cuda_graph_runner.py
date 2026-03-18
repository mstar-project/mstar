"""CUDA Graph capture and replay for AR decode and EncDec engines.

Implements SGLang-style CudaGraphRunner for autoregressive decode batches:
- Capture CUDA graphs at discrete batch sizes (1, 2, 4, 8, 16, 32, 64)
- At runtime, binary search for the smallest sufficient captured graph
- Pad inputs to match the captured size, replay graph, slice output

Also provides EncDecCudaGraphWrapper for stateless encoder/decoder submodules
(ViT, VAE) with fixed-shape inputs.

Key requirements for CUDA graph compatibility:
- FlashInfer's BatchDecodeWithPagedKVCacheWrapper must use use_cuda_graph=True
- Static buffers for page indices, seq_lens (updated via .copy_(), not reassignment)
- No dynamic memory allocation inside captured region
- No Python control flow that changes between replays
"""

import bisect
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CudaGraphRunner:
    """Captures and replays CUDA graphs for AR decode batches.

    Follows the SGLang pattern:
    1. At warmup, capture graphs for each CAPTURE_BATCH_SIZES (largest first
       for memory reuse via shared memory pool).
    2. At runtime, binary search for smallest captured size >= actual batch size.
    3. Copy real inputs into static buffers, pad remainder, replay graph.
    4. Slice output to actual batch size.
    """

    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def __init__(
        self,
        ar_engine: Any,
        submodule_name: str,
        kv_cache_config: Any,
    ):
        self.ar_engine = ar_engine
        self.submodule_name = submodule_name
        self.kv_cache_config = kv_cache_config
        self.device = ar_engine.device

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: dict[int, dict] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}

        # Shared memory pool for CUDA graph capture
        self.memory_pool = None

    def warmup_and_capture(self) -> None:
        """Capture graphs for each batch size (reverse order for memory reuse).

        Reverse order ensures larger graphs allocate memory first, and smaller
        graphs can reuse the same memory pool (CUDA graph memory sharing).
        """
        if self.device is None or not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph capture for %s", self.submodule_name)
            return

        submodule = self.ar_engine.submodules.get(self.submodule_name)
        if submodule is None:
            logger.warning("Submodule %s not found, skipping graph capture", self.submodule_name)
            return

        if not hasattr(submodule, 'forward_batched'):
            logger.info("Submodule %s does not support batched forward, skipping CUDA graph capture",
                        self.submodule_name)
            return

        logger.info("Capturing CUDA graphs for %s at batch sizes %s",
                     self.submodule_name, self.CAPTURE_BATCH_SIZES)

        # Get memory pool for sharing across batch sizes
        self.memory_pool = torch.cuda.graphs.graph_pool_handle()

        for bs in reversed(self.CAPTURE_BATCH_SIZES):
            try:
                self._capture_one(bs, submodule)
                logger.info("Captured CUDA graph for %s at batch_size=%d", self.submodule_name, bs)
            except Exception:
                logger.warning("Failed to capture CUDA graph for %s at batch_size=%d",
                               self.submodule_name, bs, exc_info=True)

    def _capture_one(self, bs: int, submodule: torch.nn.Module) -> None:
        """Capture a single CUDA graph for the given batch size.

        Steps:
        1. Create dummy inputs of shape [bs, hidden]
        2. Create dummy BatchedCacheManager with bs requests
        3. Warmup: 2 forward passes to trigger lazy initializations
        4. Capture the graph
        """
        from mminf.engine.ar_engine import BatchedCacheManager

        cfg = self.kv_cache_config
        hidden_size = cfg.num_qo_heads * cfg.head_dim

        # Create dummy request IDs and states
        dummy_request_ids = [f"__cuda_graph_dummy_{i}__" for i in range(bs)]

        # Add dummy requests to engine
        for rid in dummy_request_ids:
            self.ar_engine.add_request(rid)

        try:
            # Create dummy inputs (single token per request for decode)
            dummy_token_ids = torch.zeros(bs, dtype=torch.long, device=self.device)

            # Create BatchedCacheManager for dummy requests
            cache_manager = BatchedCacheManager(
                request_ids=dummy_request_ids,
                active_labels_per_request={rid: "main" for rid in dummy_request_ids},
                kv_cache=self.ar_engine.kv_cache,
                page_allocator=self.ar_engine.page_allocator,
                request_states=self.ar_engine.request_states,
                workspace_buffer=self.ar_engine.workspace_buffer,
                kv_cache_config=cfg,
                device=self.device,
            )

            # Build dummy preprocessed inputs
            dummy_per_request_inputs = {
                "text_inputs": torch.zeros(len(dummy_request_ids), dtype=torch.long, device=self.device),
            }

            def run_once():
                return submodule.forward_batched(
                    graph_walk="decode",
                    cache_manager=cache_manager,
                    per_request_inputs=dummy_per_request_inputs,
                    per_request_metadata={},
                )

            # Warmup: 2 forward passes
            torch.cuda.synchronize()
            for _ in range(2):
                run_once()
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.memory_pool):
                output = run_once()

            self.graphs[bs] = graph
            self.static_outputs[bs] = output
            self.static_inputs[bs] = {
                "cache_manager": cache_manager,
                "per_request_inputs": dummy_per_request_inputs,
            }
        finally:
            # Clean up dummy requests
            for rid in dummy_request_ids:
                self.ar_engine.remove_request(rid)

    def can_run(self, batch_size: int) -> bool:
        """Check if we have a captured graph that can handle this batch size."""
        if not self.graphs:
            return False
        return batch_size <= max(self.CAPTURE_BATCH_SIZES)

    def get_padded_batch_size(self, batch_size: int) -> int | None:
        """Find the smallest captured graph size >= batch_size."""
        idx = bisect.bisect_left(self.CAPTURE_BATCH_SIZES, batch_size)
        if idx >= len(self.CAPTURE_BATCH_SIZES):
            return None
        padded_bs = self.CAPTURE_BATCH_SIZES[idx]
        return padded_bs if padded_bs in self.graphs else None

    def run(
        self,
        batch_size: int,
        per_request_inputs: dict[str, dict],
        per_request_metadata: dict[str, dict],
        cache_manager: Any,
    ) -> dict[str, dict]:
        """Run using a captured CUDA graph.

        Binary searches for the smallest sufficient graph, copies real inputs
        into static buffers, replays graph, and slices output.
        """
        padded_bs = self.get_padded_batch_size(batch_size)
        if padded_bs is None:
            raise RuntimeError(
                f"No CUDA graph available for batch_size={batch_size}"
            )

        static_inputs = self.static_inputs[padded_bs]

        # Copy real inputs into static buffer positions
        static_per_req = static_inputs["per_request_inputs"]
        request_ids = list(per_request_inputs.keys())

        for i, rid in enumerate(request_ids):
            dummy_rid = f"__cuda_graph_dummy_{i}__"
            if dummy_rid in static_per_req:
                for key, val in per_request_inputs[rid].items():
                    if isinstance(val, torch.Tensor) and key in static_per_req[dummy_rid]:
                        static_per_req[dummy_rid][key].copy_(val)

        # Replay graph
        self.graphs[padded_bs].replay()

        # Extract outputs for real requests only
        static_output = self.static_outputs[padded_bs]
        outputs = {}
        for i, rid in enumerate(request_ids):
            dummy_rid = f"__cuda_graph_dummy_{i}__"
            if dummy_rid in static_output:
                outputs[rid] = {}
                for key, val in static_output[dummy_rid].items():
                    if isinstance(val, list):
                        outputs[rid][key] = [t.clone() for t in val]
                    elif isinstance(val, torch.Tensor):
                        outputs[rid][key] = [val.clone()]
                    else:
                        outputs[rid][key] = val

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

    def warmup_and_capture(self, input_shape_template: tuple[int, ...]) -> None:
        """Capture graphs for default batch sizes using a shape template.

        input_shape_template: shape of a SINGLE input (without batch dim).
        E.g., for ViT: (num_patches, patch_dim)
        """
        if not torch.cuda.is_available():
            return

        self.memory_pool = torch.cuda.graphs.graph_pool_handle()

        for bs in reversed(self.DEFAULT_CAPTURE_SIZES):
            try:
                self._capture_one(bs, input_shape_template)
                logger.info("Captured EncDec CUDA graph at batch_size=%d", bs)
            except Exception:
                logger.warning("Failed to capture EncDec CUDA graph at batch_size=%d",
                               bs, exc_info=True)

    def _capture_one(self, bs: int, input_shape: tuple[int, ...]) -> None:
        """Capture one graph for the given batch size."""
        dummy_input = torch.randn(bs, *input_shape, dtype=torch.bfloat16, device=self.device)

        # Warmup
        torch.cuda.synchronize()
        for _ in range(2):
            self.submodule(dummy_input)
        torch.cuda.synchronize()

        # Capture
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
        """Run with CUDA graph, padding if necessary."""
        actual_bs = input_tensor.shape[0]

        # Find smallest sufficient graph
        idx = bisect.bisect_left(self.DEFAULT_CAPTURE_SIZES, actual_bs)
        if idx >= len(self.DEFAULT_CAPTURE_SIZES):
            # Fallback to eager
            return self.submodule(input_tensor)

        padded_bs = self.DEFAULT_CAPTURE_SIZES[idx]
        if padded_bs not in self.graphs:
            return self.submodule(input_tensor)

        # Copy real input into static buffer (pad remainder with zeros)
        self.static_inputs[padded_bs].zero_()
        self.static_inputs[padded_bs][:actual_bs] = input_tensor

        # Replay
        self.graphs[padded_bs].replay()

        # Slice output to actual batch size
        return self.static_outputs[padded_bs][:actual_bs].clone()
