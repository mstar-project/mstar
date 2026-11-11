"""Dense Qwen3 causal language model from the shared components.

Qwen3 = Llama-style pre-norm decoder with GQA, per-head QK RMSNorm, SwiGLU
MLP and plain RoPE (``rope_theta`` 1e6, no Llama-3 scaling). Several
audio-LLM checkpoints ship one as their text decoder (Qwen3-ASR, Higgs-Audio,
Qwen3-Omni's dense layers); this is the one place to build it from
``ParallelAttention`` / ``ParallelGatedMLP`` / ``RMSNorm``, bound to the
array of the node that owns it, or to its KV, attention and position resources
on a model not yet on the state contract (Higgs-Audio, MiniCPM-o).

Parameter paths mirror HF's ``Qwen3ForCausalLM`` with the ``model.`` prefix
dropped (``embed_tokens.*``, ``layers.N.*``, ``norm.*``, ``lm_head.*``); a
checkpoint that ties ``lm_head`` to the embedding loads without the head key.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.state import PagedNHD, PagedNHDTarget, positions, state_op
from mstar.model.components.decoder_layer import DecoderLayer, no_cursor, run_layers
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelAttention,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.components.norm import RMSNorm
from mstar.model.components.qk_norm_rope import norm_rotate, rope_cos_sin


def _norm_rope_attend(
    kv, layer, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    pos_ids: torch.Tensor, theta: float, q_eps: float, k_eps: float,
) -> torch.Tensor:
    cos, sin = rope_cos_sin(pos_ids[: q.shape[0]], q.shape[-1], theta)
    cos, sin = cos[:, None], sin[:, None]
    # the angles are fp32, and a backend that takes K/V at its read refuses a k of another dtype than its v
    rotated_k = norm_rotate(k, k_weight, k_eps, cos, sin).to(k.dtype)
    q.copy_(norm_rotate(q, q_weight, q_eps, cos, sin))
    kv.add(layer, k=rotated_k, v=v)
    return kv.read(layer, q)


# The kernel is CUDA's, so any other device takes the reference.
@state_op(
    adds=("kv",), reads=("kv",), mutates=("q",), formats=(PagedNHD(devices={"cuda"}),), reference=_norm_rope_attend,
)
def qk_norm_rope_attend(
    kv, layer, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    pos_ids: torch.Tensor, theta: float, q_eps: float, k_eps: float, *, target: PagedNHDTarget,
) -> torch.Tensor:
    """Per-head q/k norm and plain RoPE at ``pos_ids``, in place on ``q``, and this layer's K/V write, in one launch,
    then the layer's read."""
    from mstar.utils import attention

    attention.qk_norm_rope_kv_write(
        q, k, v, q_weight, k_weight, pos_ids, target.cache, target.page_of_token, target.slot_of_token,
        min(target.count, q.shape[0]), q_eps, k_eps, theta,
    )
    return kv.read(layer, q)


def _plain_theta(rope, head_dim: int) -> float | None:
    """The theta of a declared rope the fused kernel computes itself: rotate-half over the whole head, unscaled."""
    plain = rope is not None and (
        rope.scale == 1.0 and not rope.interleave and rope.rotary_dim in (None, head_dim)
        and rope.low_freq_factor is None and rope.dtype is None and rope.table_dtype is None
    )
    return float(rope.theta) if plain else None


class Qwen3Attention(ParallelAttention):
    """``ParallelAttention`` with Qwen3's per-head q/k RMSNorm. Where the array declares plain RoPE, the norm, the
    rotation and the step's K/V write run in one kernel (`qk_norm_rope_kv_write`): at decode width a step is a few small
    kernels per layer, and these three were a sizeable share of it. Anything else rotates through the array. On
    resources the kernel runs where the position resource applies plain RoPE, on CUDA."""

    _fuse_qk = False
    _fuse = False

    def bind_resources(self, resources: dict) -> None:
        super().bind_resources(resources)
        # the kernel is plain non-interleaved RoPE over the whole head dim;
        # anything else the position resource would apply stays unfused
        self._rope_theta = None if self.pos is None else self.pos.fusable_rope_theta(self.head_dim)
        self._fuse_qk = (
            self._rope_theta is not None and self.q_norm is not None and self.rope_scale == 1.0
        )

    def bind_state(self, handles: dict) -> None:
        super().bind_state(handles)
        declared = None if self.state is None else self.state.array.positions
        self._theta = None if declared is None else _plain_theta(declared.rope, self.head_dim)
        self._fuse = self._theta is not None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.state is not None:
            return self._forward_state(hidden_states)
        pos_ids = self.pos.pos_ids(self.attend.label) if self._fuse_qk else None
        if pos_ids is None or not hidden_states.is_cuda:
            return super().forward(hidden_states)
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        attn_output = self.attend.norm_rope_attend(q, k, v, self.q_norm, self.k_norm, pos_ids, self._rope_theta)
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def _forward_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._fuse:
            return super().forward(hidden_states)
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        out = qk_norm_rope_attend(
            self.state, self.layer, q, k, v, self.q_norm.weight, self.k_norm.weight, positions(self.state),
            self._theta, float(self.q_norm.variance_epsilon), float(self.k_norm.variance_epsilon),
        )
        return self.o_proj(out.reshape(num_tokens, self.num_heads * self.head_dim))


@dataclass
class Qwen3LMConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, hf: dict) -> "Qwen3LMConfig":
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in hf.items() if k in names}
        values.setdefault("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
        return cls(**values)


class Qwen3DenseLM(nn.Module):
    def __init__(
        self,
        config: Qwen3LMConfig,
        *,
        attn_key: str | None = None,
        kv_key: str | None = None,
        pos_key: str | None = None,
        array: str | None = None,
        comm_group: CommGroup | None = None,
    ):
        """``array`` is what the layers attend through on a model on the state contract, and the keys are the resources
        a model still on resources names."""
        super().__init__()
        self.config = config
        comm_group = comm_group or CommGroup.trivial()
        keys = {"attn_key": attn_key, "kv_key": kv_key, "pos_key": pos_key} if array is None else {"kv_key": array}
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            comm_group=comm_group,
        )
        self.layers = nn.ModuleList([
            DecoderLayer(
                self_attn=Qwen3Attention(
                    comm_group=comm_group,
                    hidden_size=config.num_attention_heads * config.head_dim,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    qk_norm=True,
                    rms_norm_eps=config.rms_norm_eps,
                    rope_theta=config.rope_theta,
                    input_hidden_size=config.hidden_size,
                    layer=None if array is None else i,
                    **keys,
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
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=True,
        )

    def load_weights(self, weights) -> set[str]:
        """Stream HF ``(name, tensor)`` pairs (``model.`` prefix already
        dropped); tie the head to the embedding when the checkpoint does;
        raise on anything left unloaded."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        loaded = load_hf_weights(self, weights, stacked_params=LLAMA_STACKED_PARAMS)
        if "lm_head.weight" not in loaded:
            if not self.config.tie_word_embeddings:
                raise RuntimeError("Qwen3 checkpoint has no lm_head.weight and does not tie embeddings")
            # Both shard the vocab dim by TP rank over the same rows, so the
            # local embedding shard is this rank's head shard.
            self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
            loaded.add("lm_head.weight")
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(
                f"Qwen3 checkpoint left {len(missing)} parameter(s) unloaded, e.g. {missing[:5]}"
            )
        return loaded

    @property
    def embed_dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_embeds: torch.Tensor, *, label: str | None = None) -> torch.Tensor:
        """``(tokens, hidden)`` -> ``(tokens, hidden)`` after the final norm.

        On resources, the label and layer index are cursors on the shared
        resources: bind the label once, advance the index per layer. Passing
        them as arguments instead would make inductor specialize on the int.
        On the state contract each row's label comes from the step's segments,
        and ``label`` goes unread.
        """
        attention = self.layers[0].self_attn
        if attention.state is not None:
            return run_layers(self.layers, input_embeds, self.norm, no_cursor)
        attention.attend.bind_step(label)
        return run_layers(self.layers, input_embeds, self.norm)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
