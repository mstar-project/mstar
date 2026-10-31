"""Neural components for the Qwen3-TTS Talker.

The checkpoint has two coupled autoregressive axes:

* Across time, the 28-layer Talker predicts codec group 0 at 12 Hz and uses
  M*'s paged KV cache. Its attention/MLP projections are tensor-parallel.
* Within one time step, the 5-layer CodePredictor walks codec groups 1-15:
  the shared ``ResidualCodePredictor``, over a fixed local KV tensor, kept
  replicated. Its inputs (the Talker hidden state and every codec embedding)
  live in the Talker width; ``small_to_mtp_projection`` maps them into the
  predictor width when the two differ (1.7B: 2048 -> 1024; 0.6B: identity).

Class/module names intentionally follow Hugging Face checkpoint namespaces so
``load_hf_weights`` can stream parameters without a model-specific state-dict
rewrite beyond the standard fused Q/K/V and gate/up rules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components import RMSNorm
from mstar.model.components.decoder_layer import DecoderLayer, run_layers
from mstar.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mstar.model.qwen3_tts.config import (
    TALKER_ATTN,
    TALKER_KV,
    TALKER_POS,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)

# ---------------------------------------------------------------------------
# Talker backbone: autoregression across audio frames
# ---------------------------------------------------------------------------


class Qwen3TTSResizeMLP(nn.Module):
    """Two-layer text-embedding projection used by the official Talker."""

    def __init__(self, input_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(hidden_states)))


class Qwen3TTSTalkerLayer(DecoderLayer):
    """One pre-norm Talker block backed by M* tensor-parallel layers."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__(
            self_attn=ParallelAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rope_theta=config.rope_theta,
                qk_norm=True,
                rms_norm_eps=config.rms_norm_eps,
                comm_group=comm_group,
                attn_key=TALKER_ATTN,
                kv_key=TALKER_KV,
                pos_key=TALKER_POS,
            ),
            mlp=ParallelGatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                activation="silu",
                comm_group=comm_group,
            ),
            input_layernorm=RMSNorm(config.hidden_size, config.rms_norm_eps),
            post_attention_layernorm=RMSNorm(config.hidden_size, config.rms_norm_eps),
        )


class Qwen3TTSTalkerLanguageModel(nn.Module):
    """Backbone matching ``talker.model.*`` and the engine-managed KV cache."""

    def __init__(
        self,
        config: Qwen3TTSTalkerConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            Qwen3TTSTalkerLayer(config, comm_group=comm_group)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        # Which of the node's KV streams the layers write (`label`) and which
        # layer of it are cursors on the shared resources: bind the label once,
        # advance the index per layer. Passing them as arguments instead would
        # make inductor specialize on the int. The resources own the pages, so
        # module code stays clear of the allocator.
        self.layers[0].self_attn.attend.bind_step(label)
        # The residual stream is updated in place, so it starts from a copy:
        # ``input_embeds`` is the caller's, a packed replay's static input
        # whose padding rows no copy refreshes.
        return run_layers(self.layers, input_embeds.clone(), self.norm)


class Qwen3TTSTalkerModel(nn.Module):
    """Talker backbone, text projection, and codec-group-0 output head."""

    def __init__(
        self,
        config: Qwen3TTSModelConfig,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        talker = config.talker
        self.model = Qwen3TTSTalkerLanguageModel(talker, comm_group=comm_group)
        self.text_projection = Qwen3TTSResizeMLP(
            talker.text_hidden_size,
            talker.text_hidden_size,
            talker.hidden_size,
        )
        self.codec_head = nn.Linear(
            talker.hidden_size, talker.vocab_size, bias=False
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        return self.model(input_embeds, label=label)
