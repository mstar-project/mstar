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
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs
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

    # _tokens_to_codes pads to multiples of 28 tokens then reshapes to
    # (N_frames, 4, 7); for a single streaming window this is 1 frame.
    @property
    def _num_frames(self) -> int:
        return self.config.snac_window_tokens // (4 * self.config.tokens_per_frame)

    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        """Declare the SNAC decode capture.

        ``dummy_capture_inputs`` uses the *pre-preprocess* layout — one
        per-request ``{"new_token": [tokens]}`` entry, exactly matching the
        real runtime inputs. CodecCudaGraphRunner clones this entry per
        capture-batch slot and pushes the whole list through
        ``preprocess`` before the graph is captured, so the capture path
        and the runtime path share the same Python-level prep.
        """
        # One streaming window is ``snac_window_tokens`` raw tokens
        # (4 frames × 7 codes). Make the dummy tokens a multiple of 28 so
        # ``_tokens_to_codes`` doesn't take the pad branch during capture
        # (the pad branch creates a fresh tensor that would be graph-time
        # churn we don't want in the cached trace).
        tokens_per_window = self.config.snac_window_tokens
        dummy = [{
            "new_token": [
                torch.zeros(tokens_per_window, dtype=torch.long, device=device)
            ],
        }]
        return [
            CudaGraphConfig(
                graph_walk="snac_chunk",
                dummy_capture_inputs=dummy,
                capture_batch_sizes=[1, 2, 4, 8, 16],
            ),
        ]

    def cuda_graph_forward(
        self,
        codes_0: torch.Tensor,
        codes_1: torch.Tensor,
        codes_2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """SNAC decode + middle-region slice + int16 conversion.

        Called inside the captured CUDA graph by CodecCudaGraphRunner and
        directly by ``forward_batched`` for the eager path. All ops are
        pointwise / fixed-shape so the whole thing is graphable.

        Returns ``{"audio_chunk": int16 [B, slice_len]}`` — the ``squeeze(1)``
        happens inside the graph so the runner can split per-rid uniformly
        (each rid gets a 1-D tensor).
        """
        audio_hat = self.snac_model.decode([codes_0, codes_1, codes_2])
        audio_slice = audio_hat[
            :, :, self.config.snac_audio_slice_start:self.config.snac_audio_slice_end,
        ]
        audio_int16 = (audio_slice.clamp(-1, 1) * 32767).to(torch.int16)
        return {"audio_chunk": audio_int16.squeeze(1)}

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
        """Pack per-request tokens into stacked SNAC code tensors.

        Returns ``{"codes_0": [B, N*4], "codes_1": [B, N*8], "codes_2": [B, N*16]}``
        when every request has the same frame count (the only case that
        batches). Returns ``{}`` to signal "can't batch" — the engine falls
        back to the sequential path.
        """
        per_request_codes = []
        for inputs in per_request_inputs:
            tokens = inputs["new_token"][0].flatten()
            # Need at least one full codebook row (7 tokens) to form a
            # meaningful frame; below that, _tokens_to_codes' pad branch
            # either crashes on the empty tensor or emits all-pad codes.
            if tokens.numel() < self.config.tokens_per_frame:
                logger.debug(
                    "SNAC preprocess: only %d tokens — signaling skip",
                    tokens.numel(),
                )
                return {}
            per_request_codes.append(self._tokens_to_codes(tokens))

        frame_counts = {c.shape[0] for c in per_request_codes}
        if len(frame_counts) != 1:
            logger.debug(
                "SNAC preprocess: frame counts differ %s, signaling fallback",
                sorted(frame_counts),
            )
            return {}

        stacked = torch.stack(per_request_codes, dim=0)
        B, N = stacked.shape[0], stacked.shape[1]
        flat = stacked.reshape(B * N, 4, 7)
        codes_0, codes_1, codes_2 = self._extract_snac_codes(flat)
        return {
            "codes_0": codes_0.reshape(B, N * 4),
            "codes_1": codes_1.reshape(B, N * 8),
            "codes_2": codes_2.reshape(B, N * 16),
        }

    def can_batch(self, batch) -> bool:
        return True

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        packed_inputs: dict[str, torch.Tensor],
        per_request_info: dict | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Eager batched decode: runs ``cuda_graph_forward`` without a graph
        and splits the stacked output per request.

        The CUDA-graph-backed path lives in ``AudioCodecEngine``, which
        calls ``CodecCudaGraphRunner.run`` directly.
        """
        range_push("snac.eager_decode")
        stacked = self.cuda_graph_forward(**packed_inputs)
        range_pop()

        return {
            rid: {name: [stacked[name][i].detach()] for name in stacked}
            for i, rid in enumerate(request_ids)
        }

    def forward(
        self,
        codes_0: torch.Tensor,
        codes_1: torch.Tensor,
        codes_2: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        """Single-request forward — wraps cuda_graph_forward with B=1.

        Used by the engine's sequential path. Caller guarantees leading dim
        is 1 (preprocess always stacks to at least B=1).
        """
        stacked = self.cuda_graph_forward(codes_0, codes_1, codes_2)
        return {name: [stacked[name][0].detach()] for name in stacked}
