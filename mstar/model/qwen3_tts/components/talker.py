"""Neural components for Qwen3-TTS Talker and residual CodePredictor.

The checkpoint has two coupled autoregressive axes:

* Across time, the 28-layer Talker predicts codec group 0 at 12 Hz and uses
  M*'s paged KV cache. Its attention/MLP projections are tensor-parallel.
* Within one time step, the 5-layer CodePredictor walks codec groups 1-15.
  This short depth axis uses a fixed local KV tensor and is kept replicated.
  Its inputs (the Talker hidden state and every codec embedding) live in the
  Talker width; ``small_to_mtp_projection`` maps them into the predictor
  width when the two differ (1.7B: 2048 -> 1024; 0.6B: identity).

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
from mstar.model.components.code_predictor_kernels import (
    STREAM_TILE,
    DepthAttention,
    DepthKV,
    DepthWeights,
    cp_linear,
    fold_depth_weights,
    run_depth_layers,
)
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


# ---------------------------------------------------------------------------
# CodePredictor: autoregression across codec groups within one frame
# ---------------------------------------------------------------------------
# Qwen3-Omni's code predictor has the same layers and attends through the same
# ``cp_attend``; its checkpoint namespace, sampling and capture differ, and its
# first depth step is two tokens wide, so it still runs ``run_layers``.


class Qwen3TTSCodePredictorLayer(DecoderLayer):
    """Parameter container for one replicated residual-code decoder block."""

    def __init__(self, config: Qwen3TTSModelConfig, depth: DepthKV) -> None:
        cp = config.code_predictor
        super().__init__(
            self_attn=DepthAttention(
                depth,
                hidden_size=cp.hidden_size,
                num_heads=cp.num_attention_heads,
                num_kv_heads=cp.num_key_value_heads,
                head_dim=cp.head_dim,
                rope_theta=cp.rope_theta,
                rms_norm_eps=cp.rms_norm_eps,
            ),
            mlp=ParallelGatedMLP(
                hidden_size=cp.hidden_size,
                intermediate_size=cp.intermediate_size,
                activation="silu",
            ),
            input_layernorm=RMSNorm(cp.hidden_size, cp.rms_norm_eps),
            post_attention_layernorm=RMSNorm(cp.hidden_size, cp.rms_norm_eps),
        )


class Qwen3TTSCodePredictorInnerModel(nn.Module):
    """Depth decoder and one embedding table per residual codec group."""

    def __init__(self, config: Qwen3TTSModelConfig, depth: DepthKV) -> None:
        super().__init__()
        cp = config.code_predictor
        self.layers = nn.ModuleList([
            Qwen3TTSCodePredictorLayer(config, depth)
            for _ in range(cp.num_hidden_layers)
        ])
        self.norm = RMSNorm(cp.hidden_size, cp.rms_norm_eps)
        self.codec_embedding = nn.ModuleList([
            nn.Embedding(cp.vocab_size, config.talker.hidden_size)
            for _ in range(config.num_code_groups - 1)
        ])


class Qwen3TTSCodePredictor(nn.Module):
    """Five-layer decoder that predicts codec groups 1 through 15.

    ``depth_step`` performs one position on the group-depth axis. Callers
    first run the Talker hidden state at position 0 (``depth_input``), then
    feed the preceding codec embedding at positions 1-15. Keeping this as
    tensor-only code allows the complete depth loop to be CUDA-graph captured.

    Every depth input arrives in the Talker width and passes through
    ``small_to_mtp_projection`` (a biased linear layer on the 1.7B
    checkpoints, identity on the 0.6B where both widths are 1024), exactly as
    the reference ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``
    does before its decoder: the Talker hidden state at step 0, the embedding
    tables already projected (``input_tables``) after it.

    The layers are parameter containers (checkpoint names, loading): a step
    runs them through the shared fused kernels (``run_depth_layers``), five
    launches a layer.
    """

    def __init__(self, config: Qwen3TTSModelConfig) -> None:
        super().__init__()
        cp = config.code_predictor
        self.config = cp
        # the frame-local KV cache its layers attend over; see depth_step
        self.depth = DepthKV()
        self.model = Qwen3TTSCodePredictorInnerModel(config, self.depth)
        self.small_to_mtp_projection: nn.Module = (
            nn.Linear(config.talker.hidden_size, cp.hidden_size, bias=True)
            if cp.hidden_size != config.talker.hidden_size
            else nn.Identity()
        )
        self.lm_head = nn.ModuleList([
            nn.Linear(cp.hidden_size, cp.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ])
        # projected embedding tables and the fused layers' weights, built after
        # weight loading
        self.input_tables: list[torch.Tensor] | None = None
        self.fused: DepthWeights | None = None
        self.register_buffer(
            # Populated after weight loading. A single contiguous tensor lets
            # the piecewise loop select a residual head without traversing a
            # Python ModuleList during CUDA Graph replay.
            "lm_head_weight",
            torch.empty(
                config.num_code_groups - 1,
                cp.vocab_size,
                cp.hidden_size,
            ),
            persistent=False,
        )

    def consolidate_stacked_weights(self, talker_codec_embedding: torch.Tensor) -> None:
        """Pack the 15 loaded LM heads into graph-friendly contiguous storage,
        and, on checkpoints that project the depth inputs, every embedding
        table the depth loop reads already projected (``input_tables``): a
        step's input is then a row lookup, not a GEMM over the looked-up
        row. ``talker_codec_embedding`` embeds group 0, whose embedding is the
        depth sequence's second input."""
        self.lm_head_weight = torch.stack([
            head.weight.data for head in self.lm_head
        ]).contiguous()
        self.fused = fold_depth_weights(self.model.layers, self.model.norm, self.lm_head_weight)
        self.input_tables = None
        if isinstance(self.small_to_mtp_projection, nn.Linear):
            with torch.no_grad():
                self.input_tables = [
                    self.small_to_mtp_projection(table).contiguous()
                    for table in [talker_codec_embedding]
                    + [emb.weight for emb in self.model.codec_embedding[:-1]]
                ]

    def depth_input(self, last_hidden: torch.Tensor, partials: torch.Tensor) -> tuple[torch.Tensor, int]:
        """The depth sequence's first input, ``last_hidden (bs, talker_hidden)``
        projected to the predictor width (a new tensor, so the step may
        overwrite it), and how many row partials it left in ``partials`` for
        ``depth_step``."""
        proj = self.small_to_mtp_projection
        if isinstance(proj, nn.Linear):
            x = cp_linear(last_hidden, proj.weight, bias=proj.bias, partials=partials)
            return x, proj.weight.shape[0] // STREAM_TILE
        return last_hidden.clone(), 0

    def depth_step(
        self, x: torch.Tensor, partials: torch.Tensor, n_part: int = 0, head: int | None = None,
    ) -> torch.Tensor | None:
        """One depth step over ``x (bs, predictor_hidden)``, bound first with
        ``depth.bind_step``, once per call. ``x`` holds the residual stream and
        is overwritten; ``n_part`` says how many of its row sums of squares a
        producer left in ``partials`` (0: none). Returns the logits of LM head
        ``head`` (residual group ``head + 1``), or None.

        Unlike the Talker's, the cache it binds is frame-local scratch space:
        every generated frame starts at ``cache_pos=0`` and overwrites it. The
        depth loop binds the step, not this frame, which would compile once
        per ``cache_pos``."""
        return run_depth_layers(
            x, self.fused, self.depth, [layer.self_attn for layer in self.model.layers], partials,
            n_part=n_part, head=head,
        )
