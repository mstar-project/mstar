"""T5EncoderBlockByT5Mapper — Ming's per-block T5 stack mapping byt5 features
onto the DiT condition space.

Native mminf port of vllm-omni's ``t5_block_mapper.py``. The upstream version
builds on vllm-omni's TP-fused ``T5Block`` (fused ``qkv_proj`` / ``wi``) and
therefore needs a stacked-weight remap at load time. We instead build on
HuggingFace's stock ``T5Block``, whose submodule layout (``SelfAttention.q/k/v/o``
+ ``DenseReluDense.wi_0/wi_1/wo``) is byte-for-byte what Ming's
``byt5_mapper.pt`` ships — so the checkpoint loads with a plain
``load_state_dict`` (no fused mapping). This keeps the port pure-torch + stock
transformers, consistent with the rest of the mminf modeling tree.

The mapper stacks ``num_layers`` encoder blocks on the byt5 features, RMSNorms,
then projects ``d_model -> sdxl_channels`` (Ming's ``diffusion_c_input_dim``).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5Block, T5LayerNorm


class T5EncoderBlockByT5Mapper(nn.Module):
    """Stacks ``num_layers`` HF T5 encoder blocks on top of byt5 features and
    projects them to ``sdxl_channels``.

    Args:
        byte5_config: an HF ``T5Config`` (``text_encoder.config`` from the
            loaded byt5 backbone). Supplies ``d_model`` / ``num_heads`` /
            ``layer_norm_epsilon`` / relative-attention knobs.
        num_layers: number of T5 encoder blocks to stack (0 ⇒ norm + project
            only). Only the first block carries the relative-attention bias;
            the rest reuse the position_bias it emits (standard T5 weight
            sharing).
        sdxl_channels: output projection width. ``None`` ⇒ no projection
            (returns ``d_model``-wide features after the first RMSNorm).
    """

    def __init__(self, byte5_config, num_layers: int, sdxl_channels: int | None = None) -> None:
        super().__init__()
        if num_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    T5Block(byte5_config, has_relative_attention_bias=(i == 0))
                    for i in range(num_layers)
                ]
            )
        else:
            self.blocks = None
        self.layer_norm = T5LayerNorm(byte5_config.d_model, eps=byte5_config.layer_norm_epsilon)
        if sdxl_channels is not None:
            self.channel_mapper = nn.Linear(byte5_config.d_model, sdxl_channels)
            self.final_layer_norm = T5LayerNorm(sdxl_channels, eps=byte5_config.layer_norm_epsilon)
        else:
            self.channel_mapper = None
            self.final_layer_norm = None

    @staticmethod
    def get_extended_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Turn a {0,1} pad mask into an additive (-inf on pad) attention bias.

        Mirrors the upstream helper: accepts a 2-D ``[B, S]`` or pre-broadcast
        3-D ``[B, S, S]`` mask and returns ``[B, 1, *, S]`` with ``0`` on keep
        positions and ``finfo.min`` on pad positions, ready to add to the
        attention logits inside ``T5Block``.
        """
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Unexpected attention_mask shape {tuple(attention_mask.shape)}")
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        extended_mask = self.get_extended_attention_mask(attention_mask, dtype=inputs_embeds.dtype)

        hidden_states = inputs_embeds
        position_bias = None

        if self.blocks is not None:
            for block in self.blocks:
                # HF T5Block returns (hidden_states, position_bias) with
                # use_cache=False; the first block computes position_bias from
                # its relative-attention table and later blocks reuse it.
                hidden_states, position_bias = block(
                    hidden_states,
                    attention_mask=extended_mask,
                    position_bias=position_bias,
                    use_cache=False,
                )

        hidden_states = self.layer_norm(hidden_states)
        if self.channel_mapper is not None:
            hidden_states = self.channel_mapper(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Ming's HF-format ``byt5_mapper.pt`` directly.

        Because we build on stock HF ``T5Block`` (unfused q/k/v/o, wi_0/wi_1/wo)
        the source and target names already match — no stacked-param remap like
        the vllm-omni port needs. Names present in the checkpoint but absent
        from the module (or vice versa) are skipped and reported via the return
        value, so callers can assert full coverage.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            if param.shape != loaded_weight.shape:
                raise ValueError(
                    f"Shape mismatch loading byt5 mapper weight {name}: "
                    f"param {tuple(param.shape)} vs checkpoint {tuple(loaded_weight.shape)}"
                )
            with torch.no_grad():
                param.copy_(loaded_weight)
            loaded_params.add(name)
        return loaded_params


__all__ = ["T5EncoderBlockByT5Mapper"]
