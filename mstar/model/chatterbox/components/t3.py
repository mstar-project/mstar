"""T3: Chatterbox's text-to-speech-token transformer, on M* components.

T3 is a causal transformer that reads ``[conditioning | text | speech...]``
and predicts the next S3 speech token. The conditioning prefix is a speaker
embedding projected to the model width, the reference clip's speech tokens
(resampled to 32 vectors by a small perceiver for Chatterbox, raw for Turbo)
and, for Chatterbox, one emotion/exaggeration vector. Chatterbox adds learned
absolute position tables to the text and speech segments (indexed within each
segment) on top of the backbone's RoPE; Turbo's GPT-2 backbone carries one
absolute table over the whole sequence instead.

Two backbones share the layer stack driver:

* ``llama`` (Chatterbox, 30 layers x 1024): ``DecoderLayer`` of
  ``ParallelAttention`` (RoPE through the position resource, Llama-3 scaling)
  and ``ParallelGatedMLP`` with ``RMSNorm``.
* ``gpt2`` (Turbo, 24 layers x 1024): the same ``DecoderLayer`` with
  ``nn.LayerNorm``, biased QKV/output projections, a GELU(tanh) MLP and a
  ``wpe`` table looked up on the position ids the position resource plans.

Learned from ``chatterbox/models/t3/t3.py``, ``modules/cond_enc.py``,
``modules/perceiver.py``, ``modules/learned_pos_emb.py``, ``llama_configs.py``
and the HF Llama / GPT-2 implementations; nothing is copied.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.chatterbox.config import (
    T3_ATTN,
    T3_KV,
    T3_POS,
    T3BackboneConfig,
    T3Config,
)
from mstar.model.chatterbox.loader import load_component
from mstar.model.components import DecoderLayer, RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelAttention,
    ParallelGatedMLP,
    RowParallelLinear,
)
from mstar.model.loader import LLAMA_STACKED_PARAMS

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class GPT2MLP(nn.Module):
    """``c_proj(gelu_tanh(c_fc(x)))``; both projections carry a bias."""

    def __init__(self, hidden_size: int, intermediate_size: int, comm_group: CommGroup):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            comm_group, hidden_size, intermediate_size, bias=True, gather_output=False,
        )
        self.fc2 = RowParallelLinear(
            comm_group, intermediate_size, hidden_size, bias=True,
            input_is_parallel=True, reduce_results=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def _build_layer(config: T3BackboneConfig, comm_group: CommGroup) -> DecoderLayer:
    if config.is_gpt2:
        return DecoderLayer(
            self_attn=ParallelAttention(
                comm_group=comm_group,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                qkv_bias=True,
                o_bias=True,
                attn_key=T3_ATTN,
                kv_key=T3_KV,
                # absolute positions are added at the input; no rotation
                pos_key=None,
            ),
            mlp=GPT2MLP(config.hidden_size, config.intermediate_size, comm_group),
            input_layernorm=nn.LayerNorm(config.hidden_size, eps=config.norm_eps),
            post_attention_layernorm=nn.LayerNorm(config.hidden_size, eps=config.norm_eps),
        )
    scaling = config.rope_scaling
    return DecoderLayer(
        self_attn=ParallelAttention(
            comm_group=comm_group,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_scale=scaling["factor"],
            rope_low_freq_factor=scaling["low_freq_factor"],
            rope_high_freq_factor=scaling["high_freq_factor"],
            rope_old_context_len=scaling["original_max_position_embeddings"],
            attn_key=T3_ATTN,
            kv_key=T3_KV,
            pos_key=T3_POS,
        ),
        mlp=ParallelGatedMLP(
            comm_group=comm_group,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
        ),
        input_layernorm=RMSNorm(config.hidden_size, eps=config.norm_eps),
        post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.norm_eps),
    )


class T3Backbone(nn.Module):
    """The causal transformer. Parameter paths: ``layers.N.*``, ``norm``, and
    ``wpe`` for the GPT-2 variant."""

    def __init__(self, config: T3BackboneConfig, comm_group: CommGroup | None = None):
        super().__init__()
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.config = config
        self.layers = nn.ModuleList(
            [_build_layer(config, comm_group) for _ in range(config.num_hidden_layers)]
        )
        if config.is_gpt2:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.wpe = None
            self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[N, D] -> [N, D]`` over the packed step.

        ``position_ids`` are the absolute positions the position resource
        planned for this step; the GPT-2 table needs them, the Llama stack
        rotates inside the attention resource and ignores them.
        """
        hidden = input_embeds
        if self.wpe is not None:
            if position_ids is None:
                raise ValueError("the GPT-2 backbone needs position_ids")
            hidden = hidden + self.wpe(position_ids[: hidden.shape[0]])
        # The label and layer index are cursors on the shared resources: bind
        # once, advance per layer (an int argument would make inductor
        # specialize the layer body per layer).
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden = layer(hidden)
        return self.norm(hidden)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


class PerceiverAttention(nn.Module):
    """One pre-norm attention block with a residual on the query side.

    Queries and keys/values may be different sequences; both go through the
    same LayerNorm. Used twice by ``Perceiver``: once as cross-attention from
    the learned queries onto the prompt, once as self-attention.
    """

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"{channels=} is not divisible by {num_heads=}")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = queries.shape
        q = self.to_q(self.norm(queries))
        kv_in = self.norm(context)
        k = self.to_k(kv_in)
        v = self.to_v(kv_in)

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        attn = attn.transpose(1, 2).reshape(bsz, q_len, -1)
        return queries + self.proj_out(attn)


class Perceiver(nn.Module):
    """Resample a variable-length prompt to ``num_queries`` vectors."""

    def __init__(self, dim: int, num_queries: int, num_heads: int):
        super().__init__()
        self.pre_attention_query = nn.Parameter(torch.empty(1, num_queries, dim))
        self.attn = PerceiverAttention(dim, num_heads)

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        queries = self.pre_attention_query.expand(prompt.shape[0], -1, -1)
        latents = self.attn(queries, prompt)
        return self.attn(latents, latents)


class T3CondEncoder(nn.Module):
    """``[speaker | prompt | emotion]`` conditioning prefix, in that order."""

    def __init__(self, config: T3Config):
        super().__init__()
        dim = config.hidden_size
        self.speaker_embed_size = config.speaker_embed_size
        self.spkr_enc = nn.Linear(config.speaker_embed_size, dim)
        self.perceiver = (
            Perceiver(dim, config.perceiver_num_queries, config.perceiver_num_heads)
            if config.use_perceiver_resampler else None
        )
        self.emotion_adv_fc = nn.Linear(1, dim, bias=False) if config.emotion_adv else None

    def forward(
        self,
        speaker_emb: torch.Tensor,
        prompt_speech_emb: torch.Tensor | None,
        emotion_adv: torch.Tensor | None,
    ) -> torch.Tensor:
        weight = self.spkr_enc.weight
        spk = self.spkr_enc(
            speaker_emb.to(weight.dtype).view(-1, self.speaker_embed_size)
        )[:, None]
        parts = [spk]
        if prompt_speech_emb is not None:
            if self.perceiver is not None:
                prompt_speech_emb = self.perceiver(prompt_speech_emb)
            parts.append(prompt_speech_emb)
        if self.emotion_adv_fc is not None:
            if emotion_adv is None:
                raise ValueError("this T3 variant needs an emotion_adv value")
            emo = emotion_adv.to(weight.dtype).view(-1, 1, 1)
            parts.append(self.emotion_adv_fc(emo))
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class T3Model(nn.Module):
    def __init__(self, config: T3Config, comm_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.text_emb = nn.Embedding(config.text_vocab_size, dim)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, dim)
        if config.learned_pos_emb:
            self.text_pos_emb = nn.Embedding(config.text_pos_table_size, dim)
            self.speech_pos_emb = nn.Embedding(config.speech_pos_table_size, dim)
        else:
            self.text_pos_emb = None
            self.speech_pos_emb = None
        self.cond_enc = T3CondEncoder(config)
        self.backbone = T3Backbone(config.backbone, comm_group=comm_group)
        self.speech_head = nn.Linear(dim, config.speech_vocab_size, bias=config.speech_head_bias)

    # -- embeddings ---------------------------------------------------------

    def embed_prompt_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, P]`` reference speech tokens -> ``[B, P, D]``; Chatterbox adds
        the speech position table from index 0."""
        emb = self.speech_emb(tokens)
        if self.speech_pos_emb is not None:
            emb = emb + self.speech_pos_emb(
                torch.arange(tokens.shape[1], device=tokens.device)
            )
        return emb

    def conditioning(
        self,
        speaker_emb: torch.Tensor,
        prompt_tokens: torch.Tensor | None,
        emotion_adv: torch.Tensor | None,
    ) -> torch.Tensor:
        """``[B, cond_len, D]``."""
        prompt = None if prompt_tokens is None else self.embed_prompt_tokens(prompt_tokens)
        return self.cond_enc(speaker_emb, prompt, emotion_adv)

    def embed_text(self, ids: torch.Tensor, text_positions: torch.Tensor) -> torch.Tensor:
        emb = self.text_emb(ids)
        if self.text_pos_emb is not None:
            emb = emb + self.text_pos_emb(text_positions)
        return emb

    def embed_speech(self, ids: torch.Tensor, speech_positions: torch.Tensor) -> torch.Tensor:
        emb = self.speech_emb(ids)
        if self.speech_pos_emb is not None:
            emb = emb + self.speech_pos_emb(speech_positions)
        return emb

    def build_prefill_embeds(
        self, cond_emb: torch.Tensor, text_ids: torch.Tensor, *, uncond: bool = False,
    ) -> torch.Tensor:
        """One request's prefill: ``[cond | text | BOS]`` (Chatterbox repeats
        the BOS, both at speech position 0, as the reference sampler does).

        The unconditional branch of classifier-free guidance zeroes the text
        token embedding *before* the position table is added, so its text rows
        are the position embeddings alone -- matching the reference, where
        ``text_emb[1].zero_()`` precedes ``+ text_pos_emb``.
        """
        device = text_ids.device
        text = self.text_emb(text_ids)
        if uncond:
            text = torch.zeros_like(text)
        if self.text_pos_emb is not None:
            text = text + self.text_pos_emb(torch.arange(text_ids.shape[0], device=device))
        bos = torch.full((1,), self.config.start_speech_token, dtype=torch.long, device=device)
        speech = self.embed_speech(bos, torch.zeros(1, dtype=torch.long, device=device))
        parts = [cond_emb.to(text.dtype), text, speech]
        if self.config.duplicate_bos_in_prefill:
            parts.append(speech)
        return torch.cat(parts, dim=0)

    # -- transformer --------------------------------------------------------

    def hidden(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone(input_embeds, label=label, position_ids=position_ids)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.speech_head(hidden)

    # -- weights ------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream the reference checkpoint (``t3_cfg.safetensors`` or
        ``t3_turbo_v1.safetensors``) into this module, exactly."""
        if self.config.backbone.is_gpt2:
            weights = _split_gpt2_projections(weights)
            remapper = _remap_gpt2_name
        else:
            remapper = _remap_llama_name
        return load_component(
            self,
            weights,
            component=f"T3 ({self.config.backbone.kind})",
            name_remapper=remapper,
            stacked_params=LLAMA_STACKED_PARAMS,
        )


# Checkpoint keys that hold no parameter of the serving model: the backbone's
# unused token table, the text head (training only) and Turbo's GPT-2 ``wte``.
_DROPPED_PREFIXES = ("tfmr.embed_tokens.", "tfmr.wte.", "text_head.")


def _remap_llama_name(name: str) -> str | None:
    if name.startswith(_DROPPED_PREFIXES):
        return None
    if name.startswith("tfmr."):
        return "backbone." + name[len("tfmr."):]
    if name in ("text_pos_emb.emb.weight", "speech_pos_emb.emb.weight"):
        return name.replace(".emb.weight", ".weight")
    return name


_GPT2_SUFFIXES = {
    "ln_1": "input_layernorm",
    "ln_2": "post_attention_layernorm",
    "attn.c_proj": "self_attn.o_proj",
    "mlp.c_fc": "mlp.fc1",
    "mlp.c_proj": "mlp.fc2",
}


def _remap_gpt2_name(name: str) -> str | None:
    if name.startswith(_DROPPED_PREFIXES):
        return None
    if name == "tfmr.wpe.weight":
        return "backbone.wpe.weight"
    if name.startswith("tfmr.ln_f."):
        return "backbone.norm." + name[len("tfmr.ln_f."):]
    if name.startswith("tfmr.h."):
        _, _, layer, rest = name.split(".", 3)
        module, _, param = rest.rpartition(".")
        module = _GPT2_SUFFIXES.get(module, module)
        return f"backbone.layers.{layer}.{module}.{param}"
    return name


def _split_gpt2_projections(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """GPT-2 stores its projections as ``Conv1D`` (``[in, out]``) and its QKV
    fused; hand the stream over as Linear-layout q/k/v/o and MLP tensors so
    the standard stacked-shard rules do the rest."""
    for name, tensor in weights:
        if ".attn.c_attn." in name:
            prefix, _, param = name.rpartition(".")
            prefix = prefix[: -len("attn.c_attn")]
            if param == "weight":
                tensor = tensor.t().contiguous()
            for proj, shard in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0), strict=True):
                yield f"{prefix}self_attn.{proj}.{param}", shard.contiguous()
            continue
        if name.endswith(".weight") and any(
            key in name for key in (".attn.c_proj.", ".mlp.c_fc.", ".mlp.c_proj.")
        ):
            tensor = tensor.t().contiguous()
        yield name, tensor
