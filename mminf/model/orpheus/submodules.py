import logging

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.engine.ar_engine import BatchedCacheManager
from mminf.model.base import NodeSubmodule
from mminf.model.orpheus.config import OrpheusModelConfig

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

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_metadata: dict[str, dict],
        cache_manager: BatchedCacheManager,
    ) -> dict[str, torch.Tensor]:
        seq_lens = []

        if graph_walk == "prefill":
            result = {
                "text_inputs": [inp["text_inputs"][0] for inp in per_request_inputs],
            }
            seq_lens = [inp.shape[0] for inp in result["text_inputs"]]
        elif graph_walk in ("decode", "decode_llm"):
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
        elif graph_walk in ("decode", "decode_llm"):
            return self._forward_decode(cache_handle=cache_handle, **kwargs)
        else:
            raise ValueError(f"Unknown graph walk for OrpheusLLM: {graph_walk!r}")

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
        token = torch.argmax(logits, dim=-1)
        return {"new_token": [token]}

    def _forward_decode(
        self,
        text_inputs: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> NameToTensorList:
        """Embed previous token, run LLM forward, sample next token."""
        kwargs.pop("is_prefill", None)
        emb = self.embed_tokens(text_inputs)
        if cache_handle is not None:
            cache_handle.set_active_label("main")
        hidden = self.language_model(emb, cache_handle=cache_handle, **kwargs)

        logits = self.lm_head(hidden[-1:])
        token = torch.argmax(logits, dim=-1)
        return {"new_token": [token]}

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        packed_inputs: dict[str, torch.Tensor],
        per_request_metadata: dict[str, dict],
    ) -> dict[str, NameToTensorList]:
        """Batched forward pass for prefill and decode."""
        if graph_walk in ("decode", "decode_llm"):
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
        tokens = torch.argmax(logits, dim=-1)

        return {
            rid: {"new_token": [tokens[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


class SNACDecoderSubmodule(NodeSubmodule):
    """SNAC 24kHz decoder submodule.

    Supports two modes:
      - **Full decode** (audio_gen walk): receives all audio token IDs via
        per_request_metadata and decodes the full utterance.
      - **Streaming decode** (snac_chunk walk): reads a 28-token sliding window
        from the streaming buffer and decodes the middle region of the audio.
    """

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        self.config = config

    def check_streaming_ready(
        self,
        streaming_buffer: dict[str, list[torch.Tensor]],
        request_info,
    ) -> bool:
        """Check if the streaming buffer has enough tokens for a chunk."""
        tokens = streaming_buffer.get("streaming_token", [])
        window = self.config.snac_window_tokens
        return len(tokens) >= window

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_metadata: dict[str, dict] | None = None,
        per_request_info: dict | None = None,
        cache_manager: BatchedCacheManager = None,
    ) -> dict[str, torch.Tensor]:
        assert len(request_ids) == 1, "SNAC decoder processes one request at a time"
        request_id = request_ids[0]

        if graph_walk == "snac_chunk":
            return self._preprocess_streaming(
                request_id, per_request_info or {},
            )
        else:
            meta = (per_request_metadata or {}).get(request_id, {})
            if not meta and per_request_info:
                meta = per_request_info.get(request_id, {})
                if hasattr(meta, 'step_metadata'):
                    meta = meta.step_metadata
            audio_token_ids = meta.get("audio_token_ids", [])
            return {
                "request_id": request_id,
                "audio_token_ids": audio_token_ids,
                "graph_walk": graph_walk,
            }

    def _preprocess_streaming(
        self, request_id: str, per_request_info: dict,
    ) -> dict[str, torch.Tensor]:
        """Extract a sliding window of tokens from the streaming buffer."""
        fwd_info = per_request_info.get(request_id)
        step_meta = fwd_info.step_metadata if fwd_info else {}

        streaming_buffer = step_meta.get("_streaming_buffer", {})
        token_tensors = streaming_buffer.get("streaming_token", [])

        window_start = step_meta.get("window_start", 0)
        window_size = step_meta.get("window_size", self.config.snac_window_tokens)

        # Convert token tensors to audio codes
        audio_codes = []
        for tensor in token_tensors:
            vals = tensor.cpu().numpy().tolist()
            if isinstance(vals, list):
                audio_codes.extend(vals)
            else:
                audio_codes.append(int(vals))

        # Extract the window
        window_end = window_start + window_size
        window_tokens = audio_codes[window_start:window_end]

        # Convert LLM token IDs to SNAC codes
        snac_codes = []
        for idx, t in enumerate(window_tokens):
            code = t - 10 - ((idx % 7) * 4096)
            snac_codes.append(code)

        return {
            "request_id": request_id,
            "audio_token_ids": snac_codes,
            "graph_walk": "snac_chunk",
        }

    def forward(
        self, request_id: str, audio_token_ids: list[int],
        graph_walk: str = "audio_gen", **kwargs,
    ) -> NameToTensorList:
        if not audio_token_ids or len(audio_token_ids) < 7:
            logger.warning(
                "SNAC forward: no/insufficient token IDs for request %s (got %d)",
                request_id, len(audio_token_ids) if audio_token_ids else 0,
            )
            return {}
        device = next(self.snac_model.parameters()).device
        result = self._decode_tokens(audio_token_ids, device, streaming=(graph_walk == "snac_chunk"))
        if not result:
            logger.warning(
                "SNAC decode returned empty for request %s (codes may be out of range)",
                request_id,
            )
        else:
            logger.debug(
                "SNAC produced audio for request %s (%d samples, walk=%s)",
                request_id, result["audio_chunk"][0].numel(), graph_walk,
            )
        return result

    def _decode_tokens(
        self, token_ids: list[int], device: torch.device,
        streaming: bool = False,
    ) -> NameToTensorList:
        """Decode SNAC token sequence into PCM audio.

        When ``streaming=True``, only the middle region of the decoded audio
        is returned (sliding window overlap strategy).
        """
        num_frames = len(token_ids) // 7
        if num_frames == 0:
            return {}
        frame_tokens = token_ids[: num_frames * 7]

        codes_0 = []
        codes_1 = []
        codes_2 = []

        for j in range(num_frames):
            i = 7 * j
            codes_0.append(frame_tokens[i])
            codes_1.extend([frame_tokens[i + 1], frame_tokens[i + 4]])
            codes_2.extend([frame_tokens[i + 2], frame_tokens[i + 3], frame_tokens[i + 5], frame_tokens[i + 6]])

        codes_0_t = torch.tensor(codes_0, device=device, dtype=torch.int32).unsqueeze(0)
        codes_1_t = torch.tensor(codes_1, device=device, dtype=torch.int32).unsqueeze(0)
        codes_2_t = torch.tensor(codes_2, device=device, dtype=torch.int32).unsqueeze(0)

        # Validate codes are in range
        if (
            torch.any(codes_0_t < 0)
            or torch.any(codes_0_t > 4096)
            or torch.any(codes_1_t < 0)
            or torch.any(codes_1_t > 4096)
            or torch.any(codes_2_t < 0)
            or torch.any(codes_2_t > 4096)
        ):
            return {}

        codes = [codes_0_t, codes_1_t, codes_2_t]

        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)

        if streaming:
            # Streaming mode: take the middle region of the decoded audio
            audio_slice = audio_hat[:, :, self.config.snac_audio_slice_start:self.config.snac_audio_slice_end]
            audio_int16 = (audio_slice.clamp(-1, 1) * 32767).to(torch.int16).squeeze().detach()
        else:
            audio_int16 = (audio_hat.clamp(-1, 1) * 32767).to(torch.int16).squeeze().detach()

        return {"audio_chunk": [audio_int16]}
