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
        tokens = torch.argmax(logits, dim=-1)

        return {
            rid: {"new_token": [tokens[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


class SNACDecoderSubmodule(NodeSubmodule):
    """SNAC 24kHz streaming decoder submodule.

    Operates in streaming mode (snac_chunk walk): reads a 28-token sliding
    window from the streaming buffer and decodes the middle region of the
    audio for low-latency output.
    """

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        self.config = config
        # Track last emitted valid-code count per request to avoid duplicate chunks
        self._last_emitted_count: dict[str, int] = {}

    def check_streaming_ready(
        self,
        streaming_buffer: dict[str, list[torch.Tensor]],
        request_info,
    ) -> bool:
        """Check if the streaming buffer has enough tokens for a chunk."""
        tokens = streaming_buffer.get("new_token", [])
        window = self.config.snac_window_tokens
        return len(tokens) >= window

    def preprocess(
        self,
        graph_walk: str,
        per_request_inputs: list[NameToTensorList],
        request_ids: list[str],
        per_request_info: dict | None = None,
        cache_manager: BatchedCacheManager = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        assert len(request_ids) == 1, "SNAC decoder processes one request at a time"
        request_id = request_ids[0]
        return self._preprocess_streaming(request_id, per_request_info or {})

    def _preprocess_streaming(
        self, request_id: str, per_request_info: dict,
    ) -> dict[str, torch.Tensor]:
        """Convert all buffered tokens to codes and emit at frame boundaries.

        Matches the reference decoder exactly:
        - Each token is converted with a global running count
        - Audio is only produced when count % 7 == 0 and count > 27
        - The last 28 valid codes are used (buffer[-28:])
        - Each frame boundary emits exactly once (no duplicates)
        """
        fwd_info = per_request_info.get(request_id)
        step_meta = fwd_info.step_metadata if fwd_info else {}

        streaming_buffer = step_meta.get("_streaming_buffer", {})
        token_tensors = streaming_buffer.get("new_token", [])

        # Flatten all token tensors into raw vocab IDs
        all_token_ids = []
        for tensor in token_tensors:
            vals = tensor.cpu().numpy().tolist()
            if isinstance(vals, list):
                all_token_ids.extend(vals)
            else:
                all_token_ids.append(int(vals))

        # Convert ALL tokens to codes with a global running count
        # (matching reference decoder's turn_token_into_id + filter).
        base_id = self.config.custom_token_base_id
        min_audio_token = base_id + 10
        all_codes = []
        count = 0
        for t in all_token_ids:
            if t < min_audio_token:
                continue
            code = (t - base_id) - 10 - ((count % 7) * 4096)
            if 0 <= code <= 4096:
                all_codes.append(code)
                count += 1

        n_valid = len(all_codes)
        last_emitted = self._last_emitted_count.get(request_id, 0)
        window = self.config.snac_window_tokens  # 28

        # Match reference: only emit when at a frame boundary (count % 7 == 0)
        # and we have enough codes, and this is a NEW boundary we haven't emitted.
        if n_valid < window or n_valid % 7 != 0 or n_valid <= last_emitted:
            logger.debug(
                "SNAC preprocess: skip (valid=%d, last_emitted=%d, %%7=%d)",
                n_valid, last_emitted, n_valid % 7,
            )
            return {"request_id": request_id, "audio_token_ids": []}

        # Take last 28 codes (reference's buffer[-28:])
        snac_codes = all_codes[-window:]
        self._last_emitted_count[request_id] = n_valid

        logger.debug(
            "SNAC preprocess: raw=%d, valid=%d, emitting chunk, first=%s",
            len(all_token_ids), n_valid, snac_codes[:7],
        )

        return {
            "request_id": request_id,
            "audio_token_ids": snac_codes,
        }

    def cleanup_request(self, request_id: str):
        """Clean up per-request state."""
        self._last_emitted_count.pop(request_id, None)

    def forward(self, request_id: str, audio_token_ids: list[int], **kwargs) -> NameToTensorList:
        if not audio_token_ids or len(audio_token_ids) < 7:
            logger.debug(
                "SNAC forward: skipping chunk with %d token IDs (need >=7) for request %s",
                len(audio_token_ids) if audio_token_ids else 0, request_id,
            )
            return {}
        device = next(self.snac_model.parameters()).device
        result = self._decode_tokens(audio_token_ids, device)
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

    def _decode_tokens(self, token_ids: list[int], device: torch.device) -> NameToTensorList:
        """Decode a SNAC token window into PCM audio (middle region)."""
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

        # Take the middle region of the decoded audio (sliding window overlap strategy)
        audio_slice = audio_hat[:, :, self.config.snac_audio_slice_start:self.config.snac_audio_slice_end]
        audio_int16 = (audio_slice.clamp(-1, 1) * 32767).to(torch.int16).squeeze().detach()
        return {"audio_chunk": [audio_int16]}
