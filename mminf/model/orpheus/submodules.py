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
        self._seen_ids: dict[str, list[int]] = {}

    def cleanup_request(self, request_id: str):
        """Clean up per-request repetition penalty state."""
        self._seen_ids.pop(request_id, None)

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
            # Seed seen-token history with prompt ids for repetition penalty
            for rid, inp in zip(request_ids, per_request_inputs):
                self._seen_ids[rid] = inp["text_inputs"][0].tolist()
        elif graph_walk == "decode":
            result = {
                "text_inputs": [inp["text_inputs"][0] for inp in per_request_inputs],
            }
            seq_lens = [1] * len(per_request_inputs)
            # Append incoming token to seen history and expose to sampler
            for rid, inp in zip(request_ids, per_request_inputs):
                token_ids = inp["text_inputs"][0].tolist()
                self._seen_ids.setdefault(rid, []).extend(
                    token_ids if isinstance(token_ids, list) else [token_ids]
                )
                if per_request_info and rid in per_request_info:
                    meta = per_request_info[rid].step_metadata
                    meta["seen_token_ids"] = self._seen_ids[rid]
                    meta["repetition_penalty"] = self.config.repetition_penalty
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

        return {
            rid: {"logits": [logits[i : i + 1]]}
            for i, rid in enumerate(request_ids)
        }


class SNACDecoderSubmodule(NodeSubmodule):
    """SNAC 24kHz streaming decoder submodule.

    Receives a window of raw audio token tensors from StreamBuffer
    (via normal graph input routing), converts to SNAC codes, and
    decodes the middle region of the audio for low-latency output.
    """

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        self.config = config
        # Per-request accumulated valid codes and running count.
        # We process only the NEW tokens (stride) each chunk, append
        # to the running code list, and take the last `window` codes.
        self._all_codes: dict[str, list[int]] = {}
        self._valid_count: dict[str, int] = {}
        self._raw_consumed: dict[str, int] = {}

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
        inputs = per_request_inputs[0]

        # The chunk tensor from StreamBuffer: [window_size] of stacked token tensors
        chunk_tensor = inputs.get("new_token", [None])[0]
        if chunk_tensor is None:
            return {"request_id": request_id, "audio_token_ids": []}

        # Flatten to raw token IDs
        all_token_ids = chunk_tensor.cpu().numpy().flatten().tolist()

        # How many raw tokens have we already processed?
        prev_raw = self._raw_consumed.get(request_id, 0)
        # Only process the NEW tokens (skip the overlapping prefix)
        new_start = prev_raw  # global position of first new token
        # The chunk covers [chunk_start, chunk_start+len). We only need
        # tokens from index `prev_raw - chunk_start` onwards.
        # With sliding window: chunk covers raw positions [consumed_before, consumed_before+window)
        # and consumed_before = chunk_index * stride. prev_raw should equal consumed_before
        # for the first chunk (0), then stride for subsequent.
        # Since StreamBuffer advances consumed by stride, the overlap is window-stride.
        overlap = max(0, prev_raw - (len(all_token_ids) - self.config.snac_stride_tokens))
        # Simpler: we know stride=7, window=28. First chunk: all 28 are new.
        # Subsequent chunks: first 21 are old, last 7 are new.
        stride = self.config.snac_stride_tokens
        if prev_raw == 0:
            new_tokens = all_token_ids  # first chunk: all tokens are new
        else:
            new_tokens = all_token_ids[-stride:]  # subsequent: only stride new tokens

        # Convert new tokens to SNAC codes using running count
        base_id = self.config.custom_token_base_id
        min_audio_token = base_id + 10
        count = self._valid_count.get(request_id, 0)
        codes = self._all_codes.get(request_id, [])

        for t in new_tokens:
            if t < min_audio_token:
                continue
            code = (t - base_id) - 10 - ((count % 7) * 4096)
            if code > 0:
                codes.append(code)
                count += 1

        self._valid_count[request_id] = count
        self._all_codes[request_id] = codes
        self._raw_consumed[request_id] = prev_raw + len(new_tokens)

        window = self.config.snac_window_tokens
        if len(codes) < 7:
            return {"request_id": request_id, "audio_token_ids": []}

        # Take the last `window` codes (matching reference's buffer[-28:])
        snac_codes = codes[-window:] if len(codes) >= window else codes

        logger.debug(
            "SNAC preprocess: new=%d, total_codes=%d, emitting %d codes",
            len(new_tokens), len(codes), len(snac_codes),
        )

        return {
            "request_id": request_id,
            "audio_token_ids": snac_codes,
        }

    def cleanup_request(self, request_id: str):
        """Clean up per-request state."""
        self._valid_count.pop(request_id, None)
        self._all_codes.pop(request_id, None)
        self._raw_consumed.pop(request_id, None)

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
