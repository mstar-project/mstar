"""MiniCPM-o's speech-token LM (``MiniCPMTTS``): a 20-layer Llama that reads the
LLM's reply and writes 25 Hz s3tokenizer codes.

Ported from ``MiniCPMTTS`` in the checkpoint's ``modeling_minicpmo.py``
(Apache-2.0), half-duplex ``chat`` path. The condition for a reply's text
token ``t`` with post-norm LLM hidden ``h`` is
``emb_text(t) + L2norm(projector_semantic(h))``; the sequence is the
condition rows, then ``emb_text(text_eos)``, ``emb_text(audio_bos)``, then
one ``emb_code`` row per generated code. Positions are plain 1D from 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.components.norm import RMSNorm


@dataclass
class TTSConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 20
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    rms_norm_eps: float = 1e-6
    # LlamaConfig's default; upstream never sets it
    rope_theta: float = 10_000.0
    max_position_embeddings: int = 4096
    num_text_tokens: int = 152064
    # the last code is EOS
    num_audio_tokens: int = 6562
    llm_dim: int = 4096
    audio_bos_token_id: int = 151687
    text_eos_token_id: int = 151692
    normalize_projected_hidden: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def eos_code(self) -> int:
        return self.num_audio_tokens - 1

    @classmethod
    def from_hf(cls, tts: dict) -> "TTSConfig":
        if tts.get("backbone_model", "llama") != "llama" or tts.get("condition_type") != "hidden_text_merge":
            raise ValueError(
                "MiniCPM-o's TTS is ported for the llama backbone with hidden_text_merge conditioning; "
                f"got {tts.get('backbone_model')!r} / {tts.get('condition_type')!r}"
            )
        if tts.get("num_vq", 1) != 1 or tts.get("projector_type", "mlp") != "mlp":
            raise ValueError("MiniCPM-o's TTS is ported for one codebook and an MLP projector")
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in tts.items() if k in names})


class Projector(nn.Module):
    """``projector_semantic``: linear, ReLU, linear (upstream's ``MultiModalProjector``)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class TTSBackbone(nn.Module):
    """``tts.model.*``: the Llama layers and final norm (its own token
    embedding is unused; every input is an embedding)."""

    def __init__(self, config: TTSConfig, *, attn_key: str, kv_key: str, pos_key: str):
        super().__init__()
        comm_group = CommGroup.trivial()
        self.layers = nn.ModuleList(
            DecoderLayer(
                self_attn=ParallelAttention(
                    comm_group=comm_group,
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    attn_key=attn_key,
                    kv_key=kv_key,
                    pos_key=pos_key,
                ),
                mlp=ParallelGatedMLP(
                    comm_group=comm_group,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    activation="silu",
                ),
                input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, *, label: str) -> torch.Tensor:
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = layer(hidden_states=hidden_states)
        return self.norm(hidden_states)


class MiniCPMTTS(nn.Module):
    """Parameter paths mirror the checkpoint's ``tts.*`` (``emb_code.0`` and
    ``head_code.0`` flattened, the head's weight norm folded at load)."""

    def __init__(self, config: TTSConfig, *, attn_key: str, kv_key: str, pos_key: str):
        super().__init__()
        self.config = config
        self.model = TTSBackbone(config, attn_key=attn_key, kv_key=kv_key, pos_key=pos_key)
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.emb_code = nn.Embedding(config.num_audio_tokens, config.hidden_size)
        self.head_code = nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False)
        self.projector_semantic = Projector(config.llm_dim, config.hidden_size)

    def condition(self, ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """``[n]`` reply token ids and their ``[n, llm_dim]`` LLM hiddens ->
        ``[n + 2, hidden]``: the condition rows, then text EOS and audio BOS."""
        projected = self.projector_semantic(hidden.to(self.emb_text.weight.dtype))
        if self.config.normalize_projected_hidden:
            projected = F.normalize(projected, p=2, dim=-1)
        tail = torch.tensor(
            [self.config.text_eos_token_id, self.config.audio_bos_token_id],
            dtype=torch.long, device=ids.device,
        )
        return torch.cat([self.emb_text(ids) + projected, self.emb_text(tail)])

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # upstream computes the head into a float32 buffer
        return self.head_code(hidden).float()

    def load_weights(self, weights) -> None:
        """Stream ``tts.``-relative ``(name, tensor)`` pairs in; raise if any
        parameter is left unloaded."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        head_g = head_v = None
        rest = []
        for name, tensor in weights:
            if name == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
            elif name == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
            elif name.startswith(("model.embed_tokens.", "projector_spk.")):
                # unused by the chat path: inputs are embeddings, and the
                # speaker embedding is empty
                continue
            else:
                rest.append((name.replace("emb_code.0.", "emb_code."), tensor))
        loaded = load_hf_weights(self, iter(rest), stacked_params=LLAMA_STACKED_PARAMS)
        if head_g is None or head_v is None:
            raise RuntimeError("MiniCPM-o checkpoint has no tts.head_code.0 weight-norm pair")
        # torch weight_norm over dim 0: weight = g * v / ||v|| per output row
        v = head_v.float()
        weight = head_g.float() * v / v.norm(dim=1, keepdim=True)
        with torch.no_grad():
            self.head_code.weight.copy_(weight.to(self.head_code.weight.dtype))
        loaded.add("head_code.weight")
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(
                f"MiniCPM-o checkpoint left {len(missing)} TTS parameter(s) unloaded, e.g. {missing[:5]}"
            )


def next_history(history: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """``history [B, W + 1]`` (the last ``W`` codes, -1 for none yet, then the number
    generated so far) after sampling ``codes [B]``."""
    recent, generated = history[:, :-1], history[:, -1:]
    return torch.cat([recent[:, 1:], codes.reshape(-1, 1).to(history.dtype), generated + 1], dim=1)


def windowed_frequency_penalty(
    logits: torch.Tensor, recent: torch.Tensor, penalty: float,
) -> torch.Tensor:
    """Upstream's ``CustomRepetitionPenaltyLogitsProcessorRepeat``: each code's
    logit is divided (positive) or multiplied (negative) by ``penalty ** n``,
    ``n`` its count among the ``recent`` codes (``[B, W]``, -1 for empty).
    Pure tensor ops, so it runs inside a captured graph."""
    counts = torch.zeros_like(logits)
    valid = recent >= 0
    counts.scatter_add_(1, recent.clamp(min=0), valid.to(logits.dtype))
    alpha = torch.pow(torch.full_like(counts, penalty), counts)
    return torch.where(logits > 0, logits / alpha, logits * alpha)
