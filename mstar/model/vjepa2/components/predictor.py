"""V-JEPA 2 masked latent predictor.

Ports ``VJEPA2PredictorEmbeddings`` and ``VJEPA2Predictor`` from HuggingFace
``transformers/models/vjepa2/modeling_vjepa2.py``.

Takes encoder hidden states + ``context_mask`` (positions the predictor sees)
+ ``target_mask`` (positions to predict) and emits predicted embeddings at
the target positions.  NOT autoregressive — a single forward predicts every
target token in parallel.

Weight layout (prefix ``predictor.``):
    predictor.embeddings.predictor_embeddings.{weight,bias}
    predictor.embeddings.mask_tokens
    predictor.layer.{N}.*
    predictor.layernorm.{weight,bias}
    predictor.proj.{weight,bias}
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from mstar.model.vjepa2.components.layers import VJEPA2Layer
from mstar.model.vjepa2.components.vit_encoder import apply_masks
from mstar.model.vjepa2.config import VJepa2Config


class VJEPA2PredictorEmbeddings(nn.Module):
    """Project encoder hidden states into predictor space and concatenate
    learned mask tokens at the target positions."""

    def __init__(self, config: VJepa2Config):
        super().__init__()
        self.config = config
        self.predictor_embeddings = nn.Linear(config.hidden_size, config.pred_hidden_size)
        self.num_mask_tokens = config.pred_num_mask_tokens
        self.zero_init_mask_tokens = config.pred_zero_init_mask_tokens
        self.mask_tokens = nn.Parameter(torch.zeros(self.num_mask_tokens, 1, 1, config.pred_hidden_size))
        self.patch_size = config.patch_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
        mask_index: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = hidden_states.size(0)
        context = self.predictor_embeddings(hidden_states)

        mask_index = mask_index % self.num_mask_tokens
        target = self.mask_tokens[mask_index]

        # Max patch id in the target mask determines how many mask tokens to
        # materialize before gather.  (Enables running predictor with more
        # tokens than the config's frames_per_clip suggests.)
        max_patch_num = target_mask[0].max() + 1
        target = target.repeat(B, max_patch_num, 1)
        target = apply_masks(target, target_mask)

        context = context.repeat(len(context_mask), 1, 1)
        embeddings = torch.cat([context, target], dim=1)

        cm = torch.cat(context_mask, dim=0)
        tm = torch.cat(target_mask, dim=0)
        masks = torch.cat([cm, tm], dim=1)

        return embeddings, masks


class VJEPA2Predictor(nn.Module):
    def __init__(self, config: VJepa2Config):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA2PredictorEmbeddings(config)
        self.layer = nn.ModuleList(
            [
                VJEPA2Layer(
                    config,
                    hidden_size=config.pred_hidden_size,
                    num_attention_heads=config.pred_num_attention_heads,
                    mlp_ratio=config.pred_mlp_ratio,
                )
                for _ in range(config.pred_num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.pred_hidden_size, eps=config.layer_norm_eps)
        self.proj = nn.Linear(config.pred_hidden_size, config.hidden_size, bias=True)

    @staticmethod
    def _sort_tokens(
        hidden_states: torch.Tensor,
        position_masks: torch.Tensor,
        argsort: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        argsort = argsort.to(position_masks.device)
        position_masks = torch.gather(position_masks, dim=1, index=argsort)
        argsort = argsort.to(hidden_states.device)
        gather_idx = argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        hidden_states = torch.gather(hidden_states, dim=1, index=gather_idx)
        return hidden_states, position_masks

    @staticmethod
    def _unsort_tokens(hidden_states: torch.Tensor, argsort: torch.Tensor) -> torch.Tensor:
        argsort = argsort.to(hidden_states.device)
        reverse_argsort = torch.argsort(argsort, dim=1)
        gather_idx = reverse_argsort.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        return torch.gather(hidden_states, dim=1, index=gather_idx)

    def _run_forward_piecewise(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Preamble for PiecewiseCudaGraphRunner: embed, sort, and return layer-loop input.

        Returns ``(hidden_states, n_ctxt, argsort)`` so the postamble
        ``_finalize_forward_piecewise`` can unsort and slice correctly.
        All ops here run eagerly (outside any CUDA-graph-captured region).
        """
        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        _, n_ctxt, _ = encoder_hidden_states.shape
        hidden_states, position_masks = self.embeddings(encoder_hidden_states, context_mask, target_mask)
        argsort = torch.argsort(position_masks, dim=1)
        hidden_states, _ = self._sort_tokens(hidden_states, position_masks, argsort)
        return hidden_states, n_ctxt, argsort

    def _finalize_forward_piecewise(
        self,
        hidden_states: torch.Tensor,
        n_ctxt: int,
        argsort: torch.Tensor,
    ) -> torch.Tensor:
        """Postamble for PiecewiseCudaGraphRunner: layernorm, unsort, slice, project."""
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self._unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, n_ctxt:]
        return self.proj(hidden_states)

    def make_layer_loop_fn(
        self,
        static_cm,                           # always None for masked predictor (no KV cache)
        static_pos_bufs: dict[str, torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a closure over the layer loop for PiecewiseCudaGraphRunner capture.

        ``static_pos_bufs["position_mask"]`` must already be filled with the
        static ``[n_seq]`` position IDs for this rollout config before this
        method is called (the fn_factory in get_piecewise_runner_config does
        this via ``.copy_()``).  At replay, the runner never updates this buffer
        (callers pass ``pos_bufs=None``), so the captured ops always see the
        same position IDs.

        The ``unsqueeze(0)`` inside ``fn`` is a zero-copy view, CUDA-graph
        compatible.  ``VJEPA2RopeAttention.get_position_ids`` broadcasts
        ``[1, H, N]`` IDs against ``[B, H, N, D]`` Q/K tensors correctly.
        """
        layers = self.layer
        pm = static_pos_bufs["position_mask"]   # [n_seq], device-resident

        def fn(x: torch.Tensor) -> torch.Tensor:
            position_mask = pm.unsqueeze(0)      # [1, n_seq] — view, no allocation
            for layer in layers:
                x = layer(x, position_mask=position_mask)
            return x

        return fn

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        context_mask: list[torch.Tensor],
        target_mask: list[torch.Tensor],
    ) -> torch.Tensor:
        # Caller passes full encoder output; we subselect the context positions.
        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        _, n_ctxt, _ = encoder_hidden_states.shape

        hidden_states, position_masks = self.embeddings(encoder_hidden_states, context_mask, target_mask)

        # Sort tokens so that RoPE-derived position ids are monotone within
        # the sequence (required for the 3D RoPE projection to line up).
        argsort = torch.argsort(position_masks, dim=1)
        hidden_states, position_masks = self._sort_tokens(hidden_states, position_masks, argsort)

        for layer in self.layer:
            hidden_states = layer(hidden_states, position_mask=position_masks)

        hidden_states = self.layernorm(hidden_states)
        hidden_states = self._unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, n_ctxt:]
        return self.proj(hidden_states)
