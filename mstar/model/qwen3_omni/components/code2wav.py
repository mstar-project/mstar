"""Code2Wav vocoder wrapper for Qwen3-Omni.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from mstar.model.qwen3_omni.config import Code2WavConfig
from mstar.utils.attention import apply_rope_pos_ids, rms_norm, sliding_window_attn

logger = logging.getLogger(__name__)


class Qwen3OmniMoeCausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
    ):
        super().__init__()
        if stride != 1:
            raise NotImplementedError(
                "Qwen3OmniMoeCausalConvNet inference path assumes stride=1; "
                f"got stride={stride}. Re-add the runtime extra-padding calc "
                "if you need stride>1."
            )
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        # For stride=1 the original ``_get_extra_padding_for_conv1d`` always
        # returns 0, so we only need a constant left pad.
        self.padding = self.kernel_size - self.stride

    def forward(self, hidden_state):
        hidden_state = F.pad(hidden_state, (self.padding, 0))
        return self.conv(hidden_state)


class Qwen3OmniMoeCausalTransConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # Original implementation ran a no-padding ConvTranspose1d and then
        # sliced ``[pad : L - pad]`` off the temporal dim, which produced a
        # non-contiguous view that had to be ``.contiguous()``'d (a full
        # memcpy of the upsampled tensor — hundreds of MB on the deeper
        # decoder blocks). ConvTranspose1d's own ``padding`` arg crops the
        # output symmetrically by the same amount, mathematically
        # identical, with no extra kernel.
        pad = kernel_size - stride
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=pad,
        )

    def forward(self, hidden_state):
        return self.conv(hidden_state)


class Qwen3OmniMoeConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3OmniMoeCausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states):
        input = hidden_states

        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)

        hidden_states = self.gamma * hidden_states

        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = input + hidden_states

        return hidden_states


class Qwen3OmniMoeCode2WavAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Code2WavConfig, layer_idx):
        super().__init__()
        self.Code2WavConfig = Code2WavConfig
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_size = config.num_attention_heads * self.head_dim
        self.kv_size = config.num_key_value_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, self.q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=config.attention_bias)
        # q_norm / k_norm are identity in code2wav; intentionally not called in forward.
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window
        self.rope_theta = config.rope_theta

        # Populated by ``set_qkv_proj_weight()`` after weight load.
        self.qkv_proj_bias = None

    def set_qkv_proj_weight(self) -> None:
        """Concat q/k/v weights into a single buffer and drop the originals."""
        if self.q_proj is None:
            return
        qkv_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0,
        ).contiguous()
        self.register_buffer("qkv_proj_weight", qkv_weight, persistent=False)
        if self.q_proj.bias is not None:
            qkv_bias = torch.cat(
                (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), dim=0,
            ).contiguous()
            self.register_buffer("qkv_proj_bias", qkv_bias, persistent=False)
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

        self.rope_theta = float(self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, _ = hidden_states.shape

        qkv = F.linear(hidden_states, self.qkv_proj_weight, self.qkv_proj_bias)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.view(bsz, seq_len, self.config.num_attention_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.config.num_key_value_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.config.num_key_value_heads, self.head_dim)

        q_flat = q.view(bsz * seq_len, -1, self.head_dim)
        k_flat = k.view(bsz * seq_len, -1, self.head_dim)

        flat_pos = position_ids.reshape(-1).to(torch.int32)
        apply_rope_pos_ids(
            q_flat, k_flat, flat_pos,
            rope_theta=self.rope_theta,
        )

        attn_output = sliding_window_attn(
            q, k, v,
            window=self.sliding_window,
            scale=self.scaling,
        ).reshape(bsz, seq_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen3OmniMoeCode2WavMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3OmniMoeCode2WavRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3OmniMoeCode2WavRMSNorm is equivalent to T5LayerNorm. HF reference
        computes the variance in fp32 to preserve precision.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        out = rms_norm(
            hidden_states.reshape(-1, orig_shape[-1]),
            self.weight,
            eps=self.variance_epsilon,
        )
        return out.view(orig_shape)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3OmniMoeCode2WavLayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://huggingface.co/papers/2103.17239).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.
    """

    def __init__(self, config: Code2WavConfig):
        super().__init__()
        self.config = config
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.scale * x


class Qwen3OmniMoeCode2WavTransformerLayer(nn.Module):
    def __init__(self, config: Code2WavConfig, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3OmniMoeCode2WavAttention(config, layer_idx)
        self.mlp = Qwen3OmniMoeCode2WavMlp(config)
        self.input_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)
        self.mlp_layer_scale = Qwen3OmniMoeCode2WavLayerScale(config)
        self.attention_type = "sliding_attention"

    def consolidate(self) -> None:
        """Apply all weight-time consolidations for this layer."""
        self.self_attn.set_qkv_proj_weight()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            **kwargs,
        )
        # LayerScale folded into o_proj at load time -> Identity here.
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # LayerScale folded into down_proj at load time -> Identity here.
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states


class Qwen3OmniMoeCode2WavTransformerModel(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3OmniMoeCode2WavTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3OmniMoeCode2WavRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope_theta = config.rope_theta
        self.gradient_checkpointing = False
        self.window_size = config.sliding_window

    def consolidate(self) -> None:
        """Run all per-layer weight consolidations. Call after weight load."""
        for layer in self.layers:
            layer.consolidate()

    @torch.compiler.disable
    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = input_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                **kwargs,
            )

        orig_shape = hidden_states.shape
        hidden_states = rms_norm(
            hidden_states.reshape(-1, orig_shape[-1]),
            self.norm.weight,
            eps=self.norm.variance_epsilon
        ).reshape(orig_shape)
        return hidden_states


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://huggingface.co/papers/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = nn.Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def consolidate_snake_params(self) -> None:
        """Pre-compute ``exp(alpha)`` and ``1/(exp(beta)+eps)`` at load time.

        At inference these are constants, so recomputing them every call is
        wasted work (and extra ops for any tracer). After this, forward is
        ``x + inv_beta_safe * sin(x * exp_alpha)**2``.
        """
        if self.alpha is None:
            return
        exp_alpha = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1).contiguous()
        inv_beta_safe = (
            1.0 / (torch.exp(self.beta) + self.no_div_by_zero)
        ).unsqueeze(0).unsqueeze(-1).contiguous()
        self.register_buffer("exp_alpha", exp_alpha, persistent=False)
        self.register_buffer("inv_beta_safe", inv_beta_safe, persistent=False)
        # nn.Module.__setattr__ removes these from _parameters when set to None.
        self.alpha = None
        self.beta = None

    def forward(self, hidden_states):
        """SnakeBeta := x + 1/b * sin^2(x*a)"""
        return hidden_states + self.inv_beta_safe * torch.pow(
            torch.sin(hidden_states * self.exp_alpha), 2
        )


class Qwen3OmniMoeCode2WavDecoderResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3OmniMoeCausalConvNet(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Qwen3OmniMoeCode2WavDecoderBlock(nn.Module):
    def __init__(self, config: Code2WavConfig, layer_idx):
        super().__init__()
        self.config = config
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block = [
            SnakeBeta(in_dim),
            Qwen3OmniMoeCausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]

        for dilation in (1, 3, 9):
            block.append(Qwen3OmniMoeCode2WavDecoderResidualUnit(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden, **kwargs):
        for block in self.block:
            hidden = block(hidden)
        return hidden


class Qwen3OmniMoeCode2Wav(nn.Module):
    def __init__(self, config: Code2WavConfig):
        super().__init__()
        self.config = config
        self.total_upsample = np.prod(config.upsample_rates + config.upsampling_ratios)
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel(config)
        self.code_embedding = nn.Embedding(config.codebook_size * config.num_quantizers, config.hidden_size)
        self.register_buffer(
            "code_offset", torch.arange(config.num_quantizers).view(1, -1, 1) * config.codebook_size, persistent=False
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(config.hidden_size, config.hidden_size, factor, factor),
                        Qwen3OmniMoeConvNeXtBlock(config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3OmniMoeCausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)
    
    def consolidate(self):
        self.pre_transformer.consolidate()
        # Pre-fold SnakeBeta exp(alpha) / 1/(exp(beta)+eps) on every instance
        # in the upsample + decoder stacks. self.modules() walks recursively.
        for module in self.modules():
            if isinstance(module, SnakeBeta):
                module.consolidate_snake_params()
        # Compile the forward to fuse the SnakeBeta + Conv1d + F.pad
        # element-wise chains in the upsample/decoder stacks (the dominant
        # cost). The pre_transformer carries ``@torch.compiler.disable`` so
        # it remains an eager graph break and its custom RoPE/RMSNorm/SDPA
        # kernels are unaffected.
        self.forward = torch.compile(self.forward, dynamic=False)

    def forward(
        self, codes: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs
    ):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        hidden = self.pre_transformer(input_embeds=hidden, position_ids=position_ids)
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)
    
    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        position_ids: torch.Tensor,
        left_context_size: list[int],
    ) -> list[torch.Tensor]:
        """Streaming vocoder decode with per-request left-context trimming.

        Each batch element may have a different number of context frames
        prepended to its input -- 0 for the first chunk, ``N`` for subsequent
        chunks where ``N`` codec frames of overlap were carried over from the
        prior chunk's tail. This method runs the ConvNet vocoder once on the
        full batched input and then trims ``left_context_size[i] * total_upsample``
        samples from the start of request ``i``'s waveform.

        Mirrors vllm-omni's ``Qwen3OmniMoeCode2Wav.chunked_decode_streaming``:
        the model forward is unchanged, trimming is per-request on the output.

        Args:
            codes: ``[batch, num_quantizers, T]`` RVQ codes where ``T`` is
                the total codec-frame length (already includes any prepended
                left-context frames).
            left_context_size: list of length ``batch`` giving the number of
                leading codec frames to treat as context and trim from the
                emitted waveform for each request.

        Returns:
            List of waveforms, one per batch element, each shape
            ``[1, (T - left_context_size[i]) * total_upsample]``.
        """
        batch_size = codes.shape[0]
        if len(left_context_size) != batch_size:
            raise ValueError(
                f"left_context_size length {len(left_context_size)} "
                f"does not match batch size {batch_size}"
            )

        wav = self(codes, position_ids)  # [batch, 1, T * total_upsample]

        outputs: list[torch.Tensor] = []
        for i in range(batch_size):
            trim = left_context_size[i] * self.total_upsample
            outputs.append(wav[i, :, trim:])
        return outputs
