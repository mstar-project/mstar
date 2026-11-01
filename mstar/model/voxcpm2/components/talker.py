# Copyright (c) 2026 OpenBMB.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: both LMs and the per-patch heads
# fused into one module stepping one patch per call.
#
# Original file (OpenBMB/VoxCPM, src/voxcpm/model/voxcpm2.py) was released
# under Apache-2.0, with the full license text available at
# https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""The VoxCPM2 generator: both LMs plus the per-patch heads, as one module.

One autoregressive step turns the previous latent patch into the next:

    embed   = enc_to_lm_proj(feat_encoder(prev_patch))
    lm_h    = fsq(base_lm(embed))                          # paged KV, LongRoPE
    res_h   = residual_lm(fusion_concat_proj([lm_h, embed]))  # paged KV, no RoPE
    patch   = CFM(feat_decoder.estimator; mu=[lm_to_dit(lm_h), res_to_dit(res_h)],
                  cond=prev_patch)
    stop    = argmax(stop_head(silu(stop_proj(lm_h))))

Parameter paths match ``model.safetensors`` (``base_lm.*``, ``feat_decoder.estimator.*`` ...).
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.voxcpm2.components.local import LocalDiT, LocalEncoder, ScalarQuantization, solve_cfm
from mstar.model.voxcpm2.components.minicpm import MiniCPMLM
from mstar.model.voxcpm2.config import BASE_ATTN, BASE_KV, BASE_POS, RES_ATTN, RES_KV, VoxCPM2Config


class _FeatDecoder(nn.Module):
    """Holds the estimator under the checkpoint's ``feat_decoder.estimator`` path."""

    def __init__(self, estimator: LocalDiT):
        super().__init__()
        self.estimator = estimator


class VoxCPM2Generator(nn.Module):
    def __init__(self, config: VoxCPM2Config, comm_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        lm = config.lm
        hidden, dit_hidden = lm.hidden_size, config.dit.hidden_size
        self.base_lm = MiniCPMLM(
            lm,
            attn_key=BASE_ATTN,
            kv_key=BASE_KV,
            pos_key=BASE_POS,
            max_len=config.max_length,
            comm_group=comm_group,
        )
        self.residual_lm = MiniCPMLM(
            config.residual,
            attn_key=RES_ATTN,
            kv_key=RES_KV,
            pos_key=None if config.residual.no_rope else BASE_POS,
            max_len=config.max_length,
            comm_group=comm_group,
        )
        self.feat_encoder = LocalEncoder(config.encoder, config.feat_dim, config.patch_size)
        self.feat_decoder = _FeatDecoder(LocalDiT(config.dit, config.feat_dim, config.patch_size))
        self.fsq_layer = ScalarQuantization(hidden, config.fsq_latent_dim, config.fsq_scale)
        self.enc_to_lm_proj = nn.Linear(config.encoder.hidden_size, hidden)
        self.lm_to_dit_proj = nn.Linear(hidden, dit_hidden)
        self.res_to_dit_proj = nn.Linear(hidden, dit_hidden)
        self.fusion_concat_proj = nn.Linear(2 * hidden, hidden)
        self.stop_proj = nn.Linear(hidden, hidden)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(hidden, 2, bias=False)

    def materialize_buffers(self) -> None:
        self.base_lm.materialize_buffers()
        self.residual_lm.materialize_buffers()
        self.feat_encoder.encoder.materialize_buffers()
        self.feat_decoder.estimator.decoder.materialize_buffers()

    def load_weights(self, weights) -> set[str]:
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        return load_hf_weights(self, weights, stacked_params=LLAMA_STACKED_PARAMS)

    def embed_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.base_lm.embed_tokens(token_ids)

    def embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """``[rows, patch, feat]`` -> LM input embeddings ``[rows, hidden]``."""
        return self.enc_to_lm_proj(self.feat_encoder(patches))

    def stop_logits(self, lm_hidden: torch.Tensor) -> torch.Tensor:
        return self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))

    def predict_patch(
        self,
        lm_hidden: torch.Tensor,
        res_hidden: torch.Tensor,
        cond_patch: torch.Tensor,
        noise: torch.Tensor,
        cfg_value: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """One CFM solve per row: ``[rows, patch, feat]`` from the two LM states,
        the previous patch (``cond_patch``) and ``noise`` (``[rows, feat, patch]``)."""
        mu = torch.cat([self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(res_hidden)], dim=-1)
        patch = solve_cfm(
            self.feat_decoder.estimator,
            noise,
            mu,
            cond_patch.transpose(1, 2).contiguous(),
            cfg_value,
            t_span,
            mean_mode=self.config.dit_mean_mode,
        )
        return patch.transpose(1, 2)

    def decode_step(
        self,
        prev_patch: torch.Tensor,
        noise: torch.Tensor,
        cfg_value: torch.Tensor,
        t_span: torch.Tensor,
        *,
        label: str = "main",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One AR step for a batch of one-token rows: ``(patch, stop_logits)``."""
        embed = self.embed_patches(prev_patch)
        lm_hidden = self.fsq_layer(self.base_lm(embed, label=label))
        res_hidden = self.residual_lm(
            self.fusion_concat_proj(torch.cat([lm_hidden, embed], dim=-1)),
            label=label,
        )
        patch = self.predict_patch(lm_hidden, res_hidden, prev_patch, noise, cfg_value, t_span)
        return patch, self.stop_logits(lm_hidden)

    def prefill(
        self,
        token_ids: torch.Tensor,
        text_mask: torch.Tensor,
        audio_patches: torch.Tensor,
        audio_index: torch.Tensor,
        last_token_indices: torch.Tensor,
        cond_patch: torch.Tensor,
        noise: torch.Tensor,
        cfg_value: torch.Tensor,
        t_span: torch.Tensor,
        *,
        label: str = "main",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Packed prompts -> first patch and stop logits per request.

        ``token_ids``/``text_mask`` cover every packed position; the prompt's
        audio positions (reference / continuation patches) are the
        ``audio_index`` rows, embedded from ``audio_patches`` by the local
        encoder. Mirrors ``VoxCPM2Model._inference`` up to its first patch.
        """
        lm_hidden, res_hidden = self.prefill_states(
            token_ids, text_mask, audio_patches, audio_index, last_token_indices, label=label
        )
        return self.first_patch(lm_hidden, res_hidden, cond_patch, noise, cfg_value, t_span)

    def first_patch(
        self,
        lm_hidden: torch.Tensor,
        res_hidden: torch.Tensor,
        cond_patch: torch.Tensor,
        noise: torch.Tensor,
        cfg_value: torch.Tensor,
        t_span: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The prefill's per-row tail: one CFM solve and the stop head on each
        row's last position. Its shapes depend on the row count only."""
        patch = self.predict_patch(lm_hidden, res_hidden, cond_patch, noise, cfg_value, t_span)
        return patch, self.stop_logits(lm_hidden)

    def prefill_states(
        self,
        token_ids: torch.Tensor,
        text_mask: torch.Tensor,
        audio_patches: torch.Tensor,
        audio_index: torch.Tensor,
        last_token_indices: torch.Tensor,
        *,
        label: str = "main",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both LMs over the packed prompts: each row's last-position
        ``(lm_hidden, res_hidden)``, ``[rows, hidden]``."""
        text_mask = text_mask.to(self.fusion_concat_proj.weight.dtype).unsqueeze(-1)
        audio_mask = 1 - text_mask
        text_embed = self.embed_text(token_ids)
        feat_embed = torch.zeros_like(text_embed)
        if audio_patches.shape[0]:
            feat_embed = feat_embed.index_copy(0, audio_index, self.embed_patches(audio_patches))
        hidden = self.base_lm(text_mask * text_embed + audio_mask * feat_embed, label=label)
        hidden = self.fsq_layer(hidden) * audio_mask + hidden * text_mask
        res = self.residual_lm(
            self.fusion_concat_proj(torch.cat([hidden, audio_mask * feat_embed], dim=-1)),
            label=label,
        )
        return hidden.index_select(0, last_token_indices), res.index_select(0, last_token_indices)
