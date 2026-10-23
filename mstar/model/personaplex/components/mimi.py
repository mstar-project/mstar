# The conversion table and q/k permutation below are adapted from
# huggingface/transformers v4.45.0
# src/transformers/models/mimi/convert_mimi_checkpoint_to_pytorch.py:52-121:
#
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for M* on 2026-10-01: returns a converted state dict for an existing
# model instead of saving one, and takes shapes as arguments.
"""Mimi codec for PersonaPlex through ``transformers.MimiModel`` submodules.

NVIDIA runs Mimi one 80 ms frame at a time in streaming mode ([PP] offline.py:
222-255, models/lm.py:153-188). Causal convolutions make a whole-sequence pass
equal to that, but the streaming transformers do not: NVIDIA's
``RingKVCache`` (context 250, two 25 Hz steps per frame) masks one extra key
once full, and HF's public ``encode``/``decode`` use full causal attention.
``MimiCodec`` therefore composes HF's submodules and passes a 4D mask that
reproduces the ring (``ring_attention_mask``); the two diverge beyond 124
frames. The conversion of NVIDIA's
moshi-named weights is vendored because the HF script is absent from
transformers 4.50-4.57.3 and is not an importable API in any version.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import AUDIO_CODEBOOKS_PER_STREAM, FRAME_SIZE

MIMI_CONTEXT = 250  # [PP] models/loaders.py:81 (_transformer_kwargs["context"])
MIMI_STEPS_PER_FRAME = 2  # 24 kHz / hop 960 = 25 Hz transformer rate vs 12.5 Hz frames

# (substring, replacement), applied in order ([HF v4.45.0] convert_mimi_checkpoint_to_pytorch.py:52-82).
CONVERT_LIST = (
    ("conv.conv.conv", "conv"),
    ("convtr.convtr.convtr", "conv"),
    ("conv.conv", "conv"),
    ("convtr.convtr", "conv"),
    ("quantizer.rvq_first.vq", "quantizer.semantic_residual_vector_quantizer"),
    ("quantizer.rvq_first", "quantizer.semantic_residual_vector_quantizer"),
    ("quantizer.rvq_rest.vq", "quantizer.acoustic_residual_vector_quantizer"),
    ("quantizer.rvq_rest", "quantizer.acoustic_residual_vector_quantizer"),
    ("_codebook", "codebook"),
    ("_initialized", "initialized"),
    ("embedding_sum", "embed_sum"),
    ("encoder.model", "encoder.layers"),
    ("decoder.model", "decoder.layers"),
    ("encoder_transformer.transformer", "encoder_transformer"),
    ("decoder_transformer.transformer", "decoder_transformer"),
    ("linear1", "mlp.fc1"),
    ("linear2", "mlp.fc2"),
    ("self_attn.out_proj", "self_attn.o_proj"),
    ("norm1", "input_layernorm"),
    ("norm2", "post_attention_layernorm"),
    ("layer_scale_1", "self_attn_layer_scale"),
    ("layer_scale_2", "mlp_layer_scale"),
)


def permute_for_rotate_half(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Reorder each head's rows from adjacent pairs to halves ([HF v4.45.0] :94-96).

    Moshi rotates pairs ``(2i, 2i+1)``; HF's ``rotate_half`` rotates
    ``(i, i + D/2)``. Permuting q/k rows makes the two rotations agree.
    """
    dim1, dim2 = w.shape
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_mimi_state_dict(
    state_dict: dict[str, torch.Tensor], expected: dict[str, torch.Size], *, num_heads: int, num_kv_heads: int
) -> dict[str, torch.Tensor]:
    """Rename and split NVIDIA Mimi weights for ``transformers.MimiModel`` ([HF v4.45.0] :85-121).

    ``expected`` is the target model's ``state_dict`` shapes. Extra or missing
    keys and shape mismatches raise, as the source script does.
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in CONVERT_LIST:
            if old in new_key:
                new_key = new_key.replace(old, new)
        if "in_proj_weight" in new_key:
            size = value.shape[0] // 3
            q, k, v = value[:size], value[size : 2 * size], value[2 * size :]
            converted[new_key.replace("in_proj_weight", "q_proj.weight")] = permute_for_rotate_half(q, num_heads)
            converted[new_key.replace("in_proj_weight", "k_proj.weight")] = permute_for_rotate_half(k, num_kv_heads)
            converted[new_key.replace("in_proj_weight", "v_proj.weight")] = v
        else:
            converted[new_key] = value
    extra, missing = sorted(converted.keys() - expected.keys()), sorted(expected.keys() - converted.keys())
    if extra or missing:
        raise ValueError(f"Mimi conversion mismatch: extra keys {extra[:8]}, missing keys {missing[:8]}")
    bad = [k for k, v in converted.items() if v.shape != expected[k]]
    if bad:
        raise ValueError(f"Mimi conversion shape mismatch for {bad[:8]}")
    return converted


def mimi_config_kwargs(seanet: dict, transformer: dict, quantizer: dict) -> dict:
    """``MimiConfig`` kwargs for NVIDIA's Mimi kwargs ([PP] models/loaders.py:48-90).

    Raises on any reference setting this mapping does not represent.
    """
    fixed = {
        ("seanet", "channels"): 1, ("seanet", "causal"): True, ("seanet", "activation"): "ELU",
        ("seanet", "norm"): "none", ("seanet", "true_skip"): True, ("seanet", "disable_norm_outer_blocks"): 0,
        ("transformer", "causal"): True, ("transformer", "gating"): "none", ("transformer", "norm"): "layer_norm",
        ("transformer", "positional_embedding"): "rope", ("transformer", "conv_layout"): True,
        ("transformer", "context"): MIMI_CONTEXT,
    }
    groups = {"seanet": seanet, "transformer": transformer, "quantizer": quantizer}
    wrong = {f"{g}.{k}": groups[g].get(k) for (g, k), v in fixed.items() if groups[g].get(k) != v}
    if wrong:
        raise ValueError(f"unsupported Mimi settings: {wrong}")
    dim = seanet["dimension"]
    if (transformer["d_model"], transformer["input_dimension"], list(transformer["output_dimensions"])) != (
        dim, dim, [dim]
    ) or (quantizer["input_dimension"], quantizer["output_dimension"]) != (dim, dim):
        raise ValueError("Mimi transformer/quantizer widths must equal the SEANet dimension")
    return dict(
        sampling_rate=24000, frame_rate=12.5, audio_channels=1, hidden_size=dim,
        num_filters=seanet["n_filters"], num_residual_layers=seanet["n_residual_layers"],
        upsampling_ratios=list(seanet["ratios"]), kernel_size=seanet["kernel_size"],
        last_kernel_size=seanet["last_kernel_size"], residual_kernel_size=seanet["residual_kernel_size"],
        dilation_growth_rate=seanet["dilation_base"], use_causal_conv=True, pad_mode=seanet["pad_mode"],
        compress=seanet["compress"], trim_right_ratio=1.0, codebook_size=quantizer["bins"],
        codebook_dim=quantizer["dimension"], num_quantizers=quantizer["n_q"], use_conv_shortcut=False,
        vector_quantization_hidden_dimension=quantizer["dimension"], num_semantic_quantizers=1,
        upsample_groups=dim, num_hidden_layers=transformer["num_layers"],
        intermediate_size=transformer["dim_feedforward"], num_attention_heads=transformer["num_heads"],
        num_key_value_heads=transformer["num_heads"], head_dim=dim // transformer["num_heads"],
        hidden_act="gelu", norm_eps=1e-5, rope_theta=float(transformer["max_period"]),
        sliding_window=transformer["context"], layer_scale_initial_scale=transformer["layer_scale"],
        attention_bias=False, use_cache=False, use_streaming=False,
    )


def ring_attention_mask(
    steps: int, *, dtype: torch.dtype, device: torch.device | str = "cpu",
    capacity: int = MIMI_CONTEXT, chunk: int = MIMI_STEPS_PER_FRAME,
) -> torch.Tensor:
    """Additive ``[1, 1, T, T]`` mask equal to NVIDIA's streamed ``RingKVCache``.

    Query ``p`` is processed in a call of ``chunk`` steps ending at
    ``E = (p // chunk + 1) * chunk`` and sees keys ``max(0, E - C + 1) <= k <= p``
    ([PP] modules/transformer.py:263-299, 425-434).
    """
    p = torch.arange(steps, device=device).view(-1, 1)
    k = torch.arange(steps, device=device).view(1, -1)
    end = (p // chunk + 1) * chunk
    visible = (k <= p) & (k >= (end - capacity + 1).clamp(min=0))
    mask = torch.zeros(steps, steps, dtype=dtype, device=device)
    return mask.masked_fill(~visible, torch.finfo(dtype).min).view(1, 1, steps, steps)


def build_hf_mimi(config_kwargs: dict | None = None) -> nn.Module:
    """``transformers.MimiModel`` on SDPA (a flash backend would drop the custom mask)."""
    from transformers import MimiConfig, MimiModel

    config = MimiConfig(**(config_kwargs or {}))
    config._attn_implementation = "sdpa"
    return MimiModel(config).eval()


class MimiCodec(nn.Module):
    """Whole-sequence encode/decode equal to NVIDIA's per-frame streaming Mimi."""

    def __init__(self, model: nn.Module, num_codebooks: int = AUDIO_CODEBOOKS_PER_STREAM):
        super().__init__()
        self.model = model
        self.num_codebooks = num_codebooks

    def _transformer(self, transformer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        mask = ring_attention_mask(h.shape[1], dtype=h.dtype, device=h.device)
        return transformer(h, attention_mask=mask, return_dict=True)[0].transpose(1, 2)

    def encode_latent(self, pcm: torch.Tensor) -> torch.Tensor:
        """``[B, 1, N*1920]`` PCM -> unquantized ``[B, D, N]`` latent."""
        if pcm.dim() != 3 or pcm.shape[1] != 1 or pcm.shape[-1] % FRAME_SIZE:
            raise ValueError(f"expected [B, 1, N*{FRAME_SIZE}] mono PCM, got {tuple(pcm.shape)}")
        m = self.model
        return m.downsample(self._transformer(m.encoder_transformer, m.encoder(pcm)))

    def encode(self, pcm: torch.Tensor) -> torch.Tensor:
        """``[B, 1, N*1920]`` PCM -> ``[B, 8, N]`` codes."""
        return self.model.quantizer.encode(self.encode_latent(pcm), self.num_codebooks).transpose(0, 1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, 8, N]`` codes -> ``[B, 1, N*1920]`` PCM."""
        if codes.dim() != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(f"expected [B, {self.num_codebooks}, N] codes, got {tuple(codes.shape)}")
        m = self.model
        x = m.upsample(m.quantizer.decode(codes))
        return m.decoder(self._transformer(m.decoder_transformer, x))


def load_mimi_codec(
    state_dict: dict[str, torch.Tensor], config_kwargs: dict | None = None, device: torch.device | str = "cpu"
) -> MimiCodec:
    """Build ``MimiCodec`` from NVIDIA-named Mimi weights (strict), in fp32 on ``device``."""
    model = build_hf_mimi(config_kwargs)
    config = model.config
    expected = {name: tensor.shape for name, tensor in model.state_dict().items()}
    converted = convert_mimi_state_dict(
        state_dict, expected, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads
    )
    model.load_state_dict(converted, strict=True)
    return MimiCodec(model.to(device=device, dtype=torch.float32).eval())


def pad_to_frames(pcm: torch.Tensor) -> torch.Tensor:
    """Zero-pad the last frame to 1920 samples, as ``_iterate_audio(pad=True)`` does ([PP] models/lm.py:131-150)."""
    remainder = pcm.shape[-1] % FRAME_SIZE
    if remainder == 0:
        return pcm
    return torch.nn.functional.pad(pcm, (0, FRAME_SIZE - remainder))
