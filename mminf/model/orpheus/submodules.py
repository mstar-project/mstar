import bisect
import logging

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.engine.ar_engine import BatchedCacheManager
from mminf.engine.cuda_graph_runner import CudaGraphConfig
from mminf.model.base import NodeSubmodule
from mminf.model.orpheus.config import OrpheusModelConfig
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)


class SNACCudaGraphRunner:
    """CUDA graph capture/replay for the SNAC decoder.

    Captures separate graphs per batch size. All captures assume
    the standard streaming window of ``num_frames`` frames (typically 1)
    so the time dimension is fixed.

    Warmup flow:
        For each batch size (largest first for memory-pool reuse):
            1. Create static input code buffers [codes_0, codes_1, codes_2]
            2. Run 2 warmup passes
            3. Capture the graph

    Runtime flow:
        1. Pad batch to next captured size
        2. Copy real codes into static buffers
        3. graph.replay()
        4. Clone & slice outputs for real batch size
    """

    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def __init__(
        self,
        snac_model: nn.Module,
        config: OrpheusModelConfig,
        device: torch.device,
    ):
        self.snac_model = snac_model
        self.config = config
        self.device = device

        # num_windows for standard streaming window: _tokens_to_codes pads to
        # multiples of 28 then does view(-1, 4, 7), so the first dim is
        # num_tokens / 28. For the standard 28-token window, this is 1.
        self.num_frames = config.snac_window_tokens // (4 * config.tokens_per_frame)

        # Keyed by padded batch size
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_codes: dict[int, list[torch.Tensor]] = {}  # [codes_0, codes_1, codes_2]
        self.static_outputs: dict[int, torch.Tensor] = {}
        self.memory_pool = None

    def warmup_and_capture(self) -> None:
        if not torch.cuda.is_available():
            return

        self.memory_pool = torch.cuda.graphs.graph_pool_handle()
        N = self.num_frames  # frames per request

        for bs in reversed(self.CAPTURE_BATCH_SIZES):
            try:
                self._capture_one(bs, N)
                logger.info(
                    "Captured SNAC CUDA graph: bs=%d, frames=%d", bs, N,
                )
            except Exception:
                logger.warning(
                    "Failed to capture SNAC CUDA graph: bs=%d",
                    bs, exc_info=True,
                )

    def _capture_one(self, bs: int, num_frames: int) -> None:
        # Static input buffers — code shapes for num_frames frames
        codes_0 = torch.zeros(bs, num_frames * 4, dtype=torch.long, device=self.device)
        codes_1 = torch.zeros(bs, num_frames * 8, dtype=torch.long, device=self.device)
        codes_2 = torch.zeros(bs, num_frames * 16, dtype=torch.long, device=self.device)

        torch.cuda.synchronize()
        # Warmup
        for _ in range(2):
            self.snac_model.decode([codes_0, codes_1, codes_2])
        torch.cuda.synchronize()

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.memory_pool):
            static_output = self.snac_model.decode([codes_0, codes_1, codes_2])
        torch.cuda.synchronize()

        self.graphs[bs] = graph
        self.static_codes[bs] = [codes_0, codes_1, codes_2]
        self.static_outputs[bs] = static_output

    def can_run(self, batch_size: int, num_frames: int) -> bool:
        """Check if we have a captured graph for this configuration."""
        if not self.graphs:
            return False
        if num_frames != self.num_frames:
            return False
        padded = self._get_padded_bs(batch_size)
        return padded is not None and padded in self.graphs

    def _get_padded_bs(self, batch_size: int) -> int | None:
        idx = bisect.bisect_left(self.CAPTURE_BATCH_SIZES, batch_size)
        if idx >= len(self.CAPTURE_BATCH_SIZES):
            return None
        return self.CAPTURE_BATCH_SIZES[idx]

    def run(
        self,
        codes_0: torch.Tensor,
        codes_1: torch.Tensor,
        codes_2: torch.Tensor,
        actual_bs: int,
    ) -> torch.Tensor:
        """Replay a captured CUDA graph.

        Args:
            codes_0/1/2: real code tensors of shape ``(actual_bs, T_i)``.
            actual_bs: number of real requests in the batch.

        Returns:
            Audio output tensor of shape ``(actual_bs, 1, audio_len)``.
        """
        padded_bs = self._get_padded_bs(actual_bs)
        static_c0, static_c1, static_c2 = self.static_codes[padded_bs]

        # Zero then copy real codes into static buffers
        static_c0.zero_()
        static_c1.zero_()
        static_c2.zero_()
        static_c0[:actual_bs].copy_(codes_0)
        static_c1[:actual_bs].copy_(codes_1)
        static_c2[:actual_bs].copy_(codes_2)

        self.graphs[padded_bs].replay()

        # Return only the real batch slice, cloned to detach from static buffer
        return self.static_outputs[padded_bs][:actual_bs].clone()


class OrpheusLLMSubmodule(NodeSubmodule):
    """Llama 3.2 3B wrapper for Orpheus TTS.

    Dispatches on graph_walk:
      - prefill: embed text tokens, fill KV cache
      - decode: embed previous token, generate next audio token
    """

    def __init__(
        self,
        language_model: nn.Module,
        config: OrpheusModelConfig,
    ):
        super().__init__()
        self.language_model = language_model
        self.embed_tokens = language_model.model.embed_tokens
        self.lm_head = language_model.lm_head
        self.config = config

    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        return [
            CudaGraphConfig(
                graph_walk="decode", requires_cfg=False, labels=["main"],
                dummy_capture_inputs=[{"text_inputs": [torch.zeros(1, dtype=torch.long, device=device)]}]
            ),
        ]

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        per_request_info: dict | None = None,
        per_request_metadata: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_lens = []
        if graph_walk == "prefill":
            result = {
                "text_inputs": [inp["text_inputs"][0] for inp in per_request_inputs],
            }
            seq_lens = [inp.shape[0] for inp in result["text_inputs"]]
        elif graph_walk == "decode":
            result = {
                "text_inputs": [inp["text_inputs"][0] for inp in per_request_inputs],
            }
            seq_lens = [1] * len(per_request_inputs)
        else:
            raise ValueError(f"Unknown graph walk for OrpheusLLM: {graph_walk!r}")

        # Plan attention and rope for the main cache label
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")

        result = {
            key: torch.cat(val) if isinstance(val, list) and isinstance(val[0], torch.Tensor) else val
            for key, val in result.items()
        }
        result["seq_lens"] = seq_lens
        return result

    def forward(self, graph_walk: str, cache_handle=None, **kwargs) -> NameToTensorList:
        if graph_walk == "prefill":
            return self._forward_prefill(cache_handle=cache_handle, **kwargs)
        elif graph_walk == "decode":
            return self._forward_decode(cache_handle=cache_handle, **kwargs)
        else:
            raise ValueError(f"Unknown graph walk for OrpheusLLM: {graph_walk!r}")

    def postprocess(
        self, request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]]
    ):
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]
        token = outputs["new_token"][0].item()
        eos_token_id = self.config.stop_token_id
        if (eos_token_id is not None and eos_token_id == token) or \
                (request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1 >= request_info.max_tokens):
            request_info.register_loop_stop("decode_loop")

    def _forward_prefill(
        self,
        text_inputs: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> NameToTensorList:
        """Embed text tokens, fill KV cache, and sample the first audio token."""
        kwargs.pop("is_prefill", None)
        emb = self.embed_tokens(text_inputs)
        if cache_handle is not None:
            cache_handle.set_active_label("main")
        hidden = self.language_model(emb, cache_handle=cache_handle, **kwargs)

        logits = self.lm_head(hidden[-1:])
        return {"logits": [logits]}

    def _forward_decode(
        self,
        text_inputs: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> NameToTensorList:
        """Embed previous token, run LLM forward, return logits for sampling."""
        kwargs.pop("is_prefill", None)
        emb = self.embed_tokens(text_inputs)
        if cache_handle is not None:
            cache_handle.set_active_label("main")
        hidden = self.language_model(emb, cache_handle=cache_handle, **kwargs)

        logits = self.lm_head(hidden[-1:])
        return {"logits": [logits]}

    def can_batch(self, batch) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        packed_inputs: dict[str, torch.Tensor],
        per_request_info: dict | None = None,
        per_request_metadata: dict | None = None,
    ) -> dict[str, NameToTensorList]:
        """Batched forward pass for prefill and decode."""
        if graph_walk == "decode":
            return self._forward_decode_batched(
                cache_manager=cache_manager,
                request_ids=request_ids,
                packed_inputs=packed_inputs,
            )
        elif graph_walk == "prefill":
            result = self._forward_prefill(cache_handle=cache_manager, **packed_inputs)
            # Each request gets the same first token (single-request prefill)
            return {rid: result for rid in request_ids}
        else:
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

    def _forward_decode_batched(
        self,
        cache_manager: BatchedCacheManager,
        request_ids: list[str],
        packed_inputs: dict[str, torch.Tensor],
    ) -> dict[str, NameToTensorList]:
        request_ids = cache_manager.request_ids
        embs = self.embed_tokens(packed_inputs["text_inputs"])

        cache_manager.set_active_label("main")
        hidden = self.language_model(embs, cache_handle=cache_manager)

        logits = self.lm_head(hidden)

        # Expose the stacked [B, V] tensor under a sentinel key so the CUDA
        # graph runner can sample directly without concatenating per-rid slices.
        out: dict = {
            rid: {"logits": [logits[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }
        out["__batched_logits__"] = logits
        return out


class SNACDecoderSubmodule(NodeSubmodule):
    """SNAC 24kHz streaming decoder submodule.

    Receives a window of raw audio token tensors from StreamBuffer
    (via normal graph input routing), converts to SNAC codes, and
    decodes the middle region of the audio for low-latency output.

    Supports batched inference: multiple requests are decoded in a
    single SNAC forward pass when all windows have the same frame count.
    """

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        device = next(self.snac_model.parameters()).device
        self.idx_14 = torch.tensor([1, 4], dtype=torch.long, device=device)
        self.idx_2356 = torch.tensor([2, 3, 5, 6], dtype=torch.long, device=device)
        self.config = config
        self.cuda_graph_runner: SNACCudaGraphRunner | None = None

    def _tokens_to_codes(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pad raw token IDs to a multiple of 28 and convert to SNAC codes.

        Args:
            tokens: flat 1-D token tensor for one request.

        Returns:
            SNAC codes tensor of shape ``(num_frames, 4, 7)``.
        """
        remainder = tokens.numel() % 28
        if remainder != 0:
            pad_len = 28 - remainder
            pad = tokens[-1].repeat(pad_len)
            tokens = torch.cat([tokens, pad], dim=0)

        tokens = tokens.view(-1, 4, 7)
        return (tokens - self.config.custom_token_base_id - 10) % 4096

    def _extract_snac_codes(self, mf: torch.Tensor):
        """Split (N, 4, 7) codes into the three codebook levels.

        Returns:
            (codes_0, codes_1, codes_2) with shapes
            ``(N, 4)``, ``(N, 8)``, ``(N, 16)``.
        """
        codes_0 = mf[:, :, 0]
        c1 = torch.index_select(mf, dim=2, index=self.idx_14)
        codes_1 = c1.reshape(mf.shape[0], -1)
        c2 = torch.index_select(mf, dim=2, index=self.idx_2356)
        codes_2 = c2.reshape(mf.shape[0], -1)
        return codes_0, codes_1, codes_2

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict | None = None,
        cache_manager: BatchedCacheManager = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        assert len(request_ids) == 1, "SNAC decoder preprocess: use can_batch/forward_batched for multiple requests"
        request_id = request_ids[0]
        inputs = per_request_inputs[0]

        tokens = inputs["new_token"][0].flatten()
        snac_codes = self._tokens_to_codes(tokens)

        return {
            "request_id": request_id,
            "audio_token_ids": snac_codes,
        }

    def can_batch(self, batch) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        per_request_inputs: list[NameToTensorList],
        per_request_info: dict | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Batched SNAC decode: stack codes from all requests and decode in one pass."""
        # Convert each request's tokens to SNAC codes
        per_request_codes = []
        for inputs in per_request_inputs:
            tokens = inputs["new_token"][0].flatten()
            codes = self._tokens_to_codes(tokens)
            per_request_codes.append(codes)

        # Check if all requests have the same frame count (required for batching)
        frame_counts = [c.shape[0] for c in per_request_codes]
        if len(set(frame_counts)) != 1:
            # Fall back to sequential decode when frame counts differ
            logger.debug(
                "SNAC batched: frame counts differ %s, falling back to sequential",
                frame_counts,
            )
            outputs = {}
            for rid, codes in zip(request_ids, per_request_codes, strict=True):
                if codes.numel() < 7:
                    outputs[rid] = {}
                else:
                    outputs[rid] = self._decode_single(codes)
            return outputs

        # Stack: (B, num_frames, 4, 7)
        stacked = torch.stack(per_request_codes, dim=0)
        B, N = stacked.shape[0], stacked.shape[1]

        # Merge batch and frame dims for code extraction: (B*N, 4, 7)
        flat = stacked.reshape(B * N, 4, 7)
        codes_0, codes_1, codes_2 = self._extract_snac_codes(flat)

        # Reshape to (B, T_i) for each codebook level
        codes_0 = codes_0.reshape(B, N * 4)
        codes_1 = codes_1.reshape(B, N * 8)
        codes_2 = codes_2.reshape(B, N * 16)

        # CUDA graph path or eager decode
        runner = self.cuda_graph_runner
        if runner is not None and runner.can_run(B, N):
            range_push("snac.cuda_graph_replay")
            audio_hat = runner.run(codes_0, codes_1, codes_2, actual_bs=B)
            range_pop()
        else:
            range_push("snac.eager_decode")
            audio_hat = self.snac_model.decode([codes_0, codes_1, codes_2])
            range_pop()

        # Slice middle region and convert to int16 per request
        audio_slice = audio_hat[:, :, self.config.snac_audio_slice_start:self.config.snac_audio_slice_end]
        audio_int16 = (audio_slice.clamp(-1, 1) * 32767).to(torch.int16)

        outputs = {}
        for i, rid in enumerate(request_ids):
            chunk = audio_int16[i].squeeze().detach()
            outputs[rid] = {"audio_chunk": [chunk]}

        return outputs

    def forward(self, request_id: str, audio_token_ids: torch.Tensor, **kwargs) -> NameToTensorList:
        if audio_token_ids is None or audio_token_ids.numel() < 7:
            logger.warning(
                "SNAC forward: skipping chunk with %d token IDs (need >=7) for request %s",
                audio_token_ids.numel() if audio_token_ids is not None else 0, request_id,
            )
            return {}
        result = self._decode_single(audio_token_ids)
        if not result:
            logger.warning(
                "SNAC decode returned empty for request %s (codes may be out of range)",
                request_id,
            )
        else:
            logger.debug(
                "SNAC produced audio for request %s (%d samples)",
                request_id, result["audio_chunk"][0].numel(),
            )
        return result

    def _decode_single(self, mf: torch.Tensor) -> NameToTensorList:
        """Decode a single request's SNAC codes into PCM audio (middle region)."""
        codes_0, codes_1, codes_2 = self._extract_snac_codes(mf)
        codes = [codes_0, codes_1, codes_2]
        audio_hat = self.snac_model.decode(codes)
        audio_slice = audio_hat[:, :, self.config.snac_audio_slice_start:self.config.snac_audio_slice_end]
        audio_int16 = (audio_slice.clamp(-1, 1) * 32767).to(torch.int16).squeeze().detach()
        return {"audio_chunk": [audio_int16]}
