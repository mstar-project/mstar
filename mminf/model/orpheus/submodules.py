import logging

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.engine.ar_engine import BatchedCacheManager
from mminf.engine.cuda_graph_runner import CudaGraphConfig
from mminf.model.base import NodeSubmodule
from mminf.model.orpheus.config import OrpheusModelConfig
from mminf.utils.sampling import Sampler

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
    
    def get_cuda_graph_configs(self, device: torch.device) -> list[CudaGraphConfig]:
        """Return dummy inputs for CUDA graph capture, or None if this walk
        doesn't support CUDA graphs.

        Default: returns text_inputs for "decode" walks. Override in subclasses
        for walks with different input names (e.g., Qwen3-Omni Thinker uses
        "input_embeds" and "cos_sin_3d"; Talker uses "input_embeds").
        """
        return [
            CudaGraphConfig(
                graph_walk="decode", requires_cfg=False, labels=["main"],
                dummy_capture_inputs=[{"text_inputs": [torch.zeros(1, dtype=torch.long, device=device)]}]
            ),
        ]


class SNACDecoderSubmodule(NodeSubmodule):
    """SNAC 24kHz streaming decoder submodule.

    Receives a window of raw audio token tensors from StreamBuffer
    (via normal graph input routing), converts to SNAC codes, and
    decodes the middle region of the audio for low-latency output.
    """

    def __init__(self, snac_model: nn.Module, config: OrpheusModelConfig):
        super().__init__()
        self.snac_model = snac_model
        device = next(self.snac_model.parameters()).device
        self.idx_14 = torch.tensor([1, 4], dtype=torch.long, device=device)
        self.idx_2356 = torch.tensor([2, 3, 5, 6], dtype=torch.long, device=device)
        self.config = config

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

        tokens = inputs["new_token"][0].flatten()

        # Compute how many tokens we need to add
        remainder = tokens.numel() % 28
        if remainder != 0:
            pad_len = 28 - remainder
            pad = tokens[-1].repeat(pad_len)  # repeat last token
            tokens = torch.cat([tokens, pad], dim=0)

        tokens = tokens.view(-1, 4, 7)
        snac_codes = (tokens - 128256 - 10) % 4096

        return {
            "request_id": request_id,
            "audio_token_ids": snac_codes,
        }

    def forward(self, request_id: str, audio_token_ids: torch.Tensor, **kwargs) -> NameToTensorList:
        if audio_token_ids is None or audio_token_ids.numel() < 7:
            logger.warning(
                "SNAC forward: skipping chunk with %d token IDs (need >=7) for request %s",
                audio_token_ids.numel() if audio_token_ids is not None else 0, request_id,
            )
            return {}
        result = self._decode_tokens(audio_token_ids)
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

    def _decode_tokens(self, mf: torch.Tensor) -> NameToTensorList:
        """Decode a SNAC token window into PCM audio (middle region)."""

        codes_0 = mf[:, :, 0]

        c1 = torch.index_select(mf, dim=2, index=self.idx_14)
        codes_1 = c1.reshape(-1, 8)

        c2 = torch.index_select(mf, dim=2, index=self.idx_2356)
        codes_2 = c2.reshape(-1, 16)

        codes = [codes_0, codes_1, codes_2]
        audio_hat = self.snac_model.decode(codes)
        # Take the middle region of the decoded audio (sliding window overlap strategy)
        audio_slice = audio_hat[:, :, self.config.snac_audio_slice_start:self.config.snac_audio_slice_end]
        audio_int16 = (audio_slice.clamp(-1, 1) * 32767).to(torch.int16).squeeze().detach()
        return {"audio_chunk": [audio_int16]}
