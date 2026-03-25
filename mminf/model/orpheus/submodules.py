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
        return {
            "new_token": [token],
            "audio_token": [token],
        }

    def forward_batched(
        self,
        graph_walk: str,
        request_ids: list[str],
        cache_manager: BatchedCacheManager,
        packed_inputs: dict[str, torch.Tensor],
        per_request_metadata: dict[str, dict],
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
        tokens = torch.argmax(logits, dim=-1)

        return {
            rid: {"new_token": [tokens[i : i + 1]], "audio_token": [tokens[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


class SNACDecoderSubmodule(NodeSubmodule):
    """SNAC 24kHz decoder submodule.

    Accumulates audio tokens per-request and decodes a full frame (7 tokens)
    into PCM audio when ready.
    """

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        self.config = config
        self._buffers: dict[str, list[int]] = {}

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_metadata: dict[str, dict],
        cache_manager: BatchedCacheManager = None,
    ) -> dict[str, torch.Tensor]:
        assert len(per_request_inputs) == 1, "SNAC decoder processes one request at a time"
        request_id = request_ids[0]
        audio_token = per_request_inputs[0]["audio_token"][0]
        return {
            "request_id": request_id,
            "audio_token": audio_token,
        }

    def forward(self, request_id: str, audio_token: torch.Tensor, **kwargs) -> NameToTensorList:
        device = audio_token.device
        token_id = audio_token.item()

        # Initialize buffer for new requests
        if request_id not in self._buffers:
            self._buffers[request_id] = []

        buf = self._buffers[request_id]
        pos = len(buf)

        # Convert token ID to SNAC code: token_id - 10 - ((pos % 7) * 4096)
        code = token_id - 10 - ((pos % 7) * 4096)

        if code < 0 or code > 4096:
            # Invalid code, skip but still track position
            buf.append(code)
            return {}

        buf.append(code)

        # Check if we have a complete frame (multiple of 7 tokens, at least 7)
        if len(buf) >= 7 and len(buf) % 7 == 0:
            return self._decode_latest_frames(buf, device)

        return {}

    def _decode_latest_frames(self, buf: list[int], device: torch.device) -> NameToTensorList:
        """Decode the last 4 frames (28 tokens) worth of audio, return slice [2048:4096]."""
        # Use last 28 tokens (4 frames) for context, matching reference implementation
        num_context_tokens = min(len(buf), 28)
        frame_tokens = buf[-num_context_tokens:]
        num_frames = len(frame_tokens) // 7

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

        # Slice [2048:4096] to get the relevant audio chunk (matching reference)
        audio_slice = audio_hat[:, :, 2048:4096].detach()
        # Convert to int16 PCM
        audio_int16 = (audio_slice * 32767).to(torch.int16).squeeze()
        return {"audio_chunk": [audio_int16]}

    def clear_request(self, request_id: str):
        """Clean up buffer when request completes."""
        self._buffers.pop(request_id, None)
