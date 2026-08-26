"""Vision + audio projectors for Ming-flash-omni-2.0.

Ports the two ``nn.Sequential`` blocks built inline in
``modeling_bailingmm2.py:BailingMM2NativeForConditionalGeneration.__init__``
(lines 66-88 of the Ming source repo) into standalone modules that mstar
can load weights into directly. The released checkpoint stores the
weights under the top-level prefixes ``linear_proj.*`` (vision) and
``linear_proj_audio.*`` (audio):

  * Vision (mlp_depth=2):
      linear_proj.0.{weight,bias}   -> Linear(vision_out_hidden, llm_hidden)
      [GELU at index 1, no params]
      linear_proj.2.{weight,bias}   -> Linear(llm_hidden, llm_hidden)

  * Audio (mlp_depth=2):
      linear_proj_audio.0.{weight,bias}   -> Conv1d(audio_d_model, llm_hidden, ds_kernel_size, ds_stride)
      [Transpose at index 1, GELU at index 2, no params]
      linear_proj_audio.3.{weight,bias}   -> Linear(llm_hidden, llm_hidden)
      [Transpose at index 4, no params]

We mirror the upstream layer ordering exactly so the
``linear_proj.*`` / ``linear_proj_audio.*`` keys from the checkpoint land
on the right ``nn.Module`` slot via plain index-based lookup.
"""

from __future__ import annotations

import torch
from torch import nn


class _Transpose(nn.Module):
    """Used inside ``nn.Sequential`` chains (modeling_utils.py:Transpose)."""

    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class MingVisionProjector(nn.Module):
    """MLP projector: vision encoder output -> LLM hidden space.

    Args:
        vision_dim: ``VisionEncoderConfig.out_hidden_size`` (4096 on the
            released ckpt — the vision encoder already projects internally
            via its ``PatchMerger``).
        llm_dim:    ``ThinkerLLMConfig.hidden_size`` (4096).
        mlp_depth:  ``MingFlashOmniModelConfig.mlp_depth`` (2 on the
            released ckpt). depth=1 yields a single Linear; depth=N adds
            (N-1) GELU+Linear pairs after it.
    """

    def __init__(self, vision_dim: int, llm_dim: int, mlp_depth: int = 2) -> None:
        super().__init__()
        if mlp_depth < 1:
            raise ValueError(f"mlp_depth must be >= 1, got {mlp_depth}")
        layers: list[nn.Module] = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        # Expose as ``proj`` (not raw ``nn.Sequential``) so subclassing /
        # surgery has a stable name. Weight loading walks ``proj.<idx>.*``.
        self.proj = nn.Sequential(*layers)

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        """Project vision tokens.

        Args:
            vision_embeds: (N_tokens, vision_dim) or (B, N_tokens, vision_dim).

        Returns:
            Same shape with the last dim replaced by ``llm_dim``.
        """
        return self.proj(vision_embeds)


class MingAudioProjector(nn.Module):
    """Conv1d-downsample + MLP projector: Whisper encoder -> LLM hidden space.

    Layer ordering matches ``modeling_bailingmm2.py`` exactly so the
    released ckpt's ``linear_proj_audio.0`` / ``.3`` keys hit the Conv1d
    and Linear by integer index.

    Args:
        audio_dim:     ``AudioEncoderConfig.d_model`` (= whisper n_state,
                       1280 on the released ckpt).
        llm_dim:       ``ThinkerLLMConfig.hidden_size``.
        ds_kernel_size: temporal kernel for the down-sample conv (3 on
                       the released ckpt).
        ds_stride:     temporal stride (2 on the released ckpt).
        mlp_depth:     ``MingFlashOmniModelConfig.mlp_depth`` (2 on the
                       released ckpt; depth=N adds (N-1) GELU+Linear pairs
                       after the conv).
    """

    def __init__(
        self,
        audio_dim: int,
        llm_dim: int,
        ds_kernel_size: int = 3,
        ds_stride: int = 2,
        mlp_depth: int = 2,
    ) -> None:
        super().__init__()
        if mlp_depth < 1:
            raise ValueError(f"mlp_depth must be >= 1, got {mlp_depth}")
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride
        self.audio_dim = audio_dim
        self.llm_dim = llm_dim

        layers: list[nn.Module] = [
            nn.Conv1d(
                audio_dim,
                llm_dim,
                kernel_size=ds_kernel_size,
                stride=ds_stride,
                padding=ds_kernel_size // 2,
            ),
            # Conv1d output is (B, llm_dim, T'); MLP wants (B, T', llm_dim).
            _Transpose(-1, -2),
        ]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        # Trailing transpose flips back to (B, llm_dim, T') — that's the
        # shape upstream callers expect after the projector.
        layers.append(_Transpose(-1, -2))
        self.proj = nn.Sequential(*layers)

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        """Project a packed (B, T, audio_dim) tensor.

        Args:
            audio_embeds: (B, T, audio_dim) Whisper encoder output, channels-last.

        Returns:
            (B, llm_dim, T') tensor, where
            ``T' = (T - ds_kernel_size + 2*(ds_kernel_size//2)) // ds_stride + 1``.
        """
        # Conv1d expects (B, C, T) — flip first.
        x = audio_embeds.transpose(-1, -2)
        return self.proj(x)

    def compute_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Output sequence length after Whisper conv stems + this projector.

        Mirrors :func:`projectors.AudioProjector.compute_output_length` from
        vllm-omni: the Whisper encoder has two fixed Conv1d stems (kernel=3,
        stride=2 then stride=1 -> see ``whisper_encoder``); we then apply
        ``Conv1d(ds_kernel_size, ds_stride)``. The Whisper stem formula
        ``(L - 3 + 2) // 2 + 1`` applies once, then the projector conv.
        """
        # Whisper encoder stem (conv1: kernel=3, pad=1, stride=2)
        length = (input_length - 3 + 2 * 1) // 2 + 1
        # Projector conv (kernel=ds_kernel_size, pad=ds_kernel_size//2, stride=ds_stride)
        length = (length - self.ds_kernel_size + 2 * (self.ds_kernel_size // 2)) // self.ds_stride + 1
        return length


__all__ = ["MingVisionProjector", "MingAudioProjector"]
