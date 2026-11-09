"""Assembled Kimi K3 text backbone: embedding, 93 hybrid layers on attention residuals, the
output read, final norm and LM head."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, VocabParallelEmbedding
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.components import attn_res
from mstar.model.kimi_k3.components.attn_res import attn_res_read_norm
from mstar.model.kimi_k3.components.decoder_layer import KimiK3DecoderLayer
from mstar.model.kimi_k3.config import KimiK3ModelConfig


class KimiK3LanguageModel(nn.Module):
    def __init__(self, config: KimiK3ModelConfig, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        self.eps = config.rms_norm_eps
        self.num_attn_res_blocks = config.num_attn_res_blocks
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size,
            comm_group=comm_group, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            KimiK3DecoderLayer(config, idx, comm_group=comm_group)
            for idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.attn_res_flashinfer = config.attn_res_backend == "auto"

    def forward(self, input_ids: torch.Tensor, rows: torch.Tensor | None = None,
                aux_layers: tuple[int, ...] | None = None,
                aux_sink: Callable[[int, torch.Tensor], None] | None = None,
                num_valid_tokens: torch.Tensor | None = None):
        """``input_ids (T,)`` flat batch -> final normed hidden ``(T, hidden)``, or only its
        ``rows``. With ``aux_layers`` (0-based, ascending, before the last layer) each of those
        layers' output stream goes to ``aux_sink(j, state)`` (``j`` its index in ``aux_layers``),
        for the DSpark drafter's context: under block attention residuals, the running partial sum
        of the block (the layer's input to the next one's reads, residual and MLP added), as PR #257
        reads it for K3's drafter (vLLM captures a DSpark drafter's ``target_layer_ids`` + 1;
        which stream it takes under attention residuals is a hypothesis)."""
        prefix = self.embed_tokens(input_ids)
        # the residual stack, token-major (FlashInfer's layout): each block start writes its block in place
        stack = prefix.new_empty(prefix.shape[0], self.num_attn_res_blocks, prefix.shape[1])
        pending = None
        last = len(self.layers) - 1
        taps = {layer: j for j, layer in enumerate(aux_layers or ())}
        if taps and (list(taps) != sorted(taps) or max(taps) >= last or aux_sink is None):
            raise ValueError(f"aux_layers {aux_layers}: ascending, before layer {last}, with a sink")
        # a padded step's real rows lead it: the MoE blocks skip the rest (moe_mask_padding)
        token_valid = (None if num_valid_tokens is None else
                       torch.arange(input_ids.shape[0], device=input_ids.device) < num_valid_tokens)
        for idx, layer in enumerate(self.layers):
            prefix, pending, stack = layer(prefix, pending, stack, rows=rows if idx == last else None,
                                           token_valid=token_valid)
            if idx in taps:
                # the tap reads the stream: add the MLP's output here rather than in the next read
                prefix, pending = prefix + pending, None
                aux_sink(taps[idx], prefix)
                if self.attn_res_flashinfer and attn_res.fi_serves(stack, prefix, ()):
                    # FlashInfer's next read adds into the stream in place: not into the tensor the sink holds
                    prefix = prefix.clone()
        return self.read_out(prefix, pending, stack)

    def read_out(self, prefix: torch.Tensor, pending: torch.Tensor | None, stack: torch.Tensor) -> torch.Tensor:
        """The output read (over the whole stack, adding ``pending``) and the final norm."""
        out, _ = attn_res_read_norm(
            stack, self.layers[-1].blocks_out, prefix, self.output_attn_res_norm.weight,
            self.output_attn_res_proj.weight, self.eps, self.norm, addend=pending,
            flashinfer=self.attn_res_flashinfer)
        return out


class KimiK3ForCausalLM(nn.Module):
    """The ``LLM`` node's weights root: ``model.*`` and ``lm_head`` under the checkpoint's
    ``language_model.`` prefix."""

    def __init__(self, config: KimiK3ModelConfig, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = KimiK3LanguageModel(config, comm_group=comm_group)
        self.lm_head = ColumnParallelLinear(
            comm_group or CommGroup.trivial(), input_size=config.hidden_size,
            output_size=config.vocab_size, bias=False, gather_output=True)

    def forward(self, input_ids: torch.Tensor, rows: torch.Tensor | None = None,
                num_valid_tokens: torch.Tensor | None = None) -> torch.Tensor:
        if num_valid_tokens is None:
            return self.lm_head(self.model(input_ids, rows=rows))
        return self.lm_head(self.model(input_ids, rows=rows, num_valid_tokens=num_valid_tokens))

    def kda_conv_dtype(self) -> torch.dtype:
        """The KDA projections' dtype, which the pool's conv tail must match."""
        for layer in self.model.layers:
            if layer.is_linear_attention:
                return layer.self_attn.q_proj.weight.dtype
        raise RuntimeError("no KDA layer in the schedule")  # unreachable: the config refuses it

    def load_weights(self, weights) -> set[str]:
        from mstar.model.kimi_k3.weight_loader import load_kimi_k3_weights

        return load_kimi_k3_weights(self, weights, self.config)
