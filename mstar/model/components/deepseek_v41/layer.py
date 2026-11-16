"""DeepSeek V4.1 layer's one-site-lagged hyper-connection schedule."""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.components import RMSNorm
from mstar.model.glm5_next.mhc import Glm5NextHyperConnection, update_streams


class DeepSeekV41Layer(nn.Module):
    """Attention and FFN over mHC streams, carrying the FFN pre-mix to the next layer.

    The attention and FFN modules are supplied by the model graph. The published
    block uses the *incoming* pre-mix for attention and this layer's attention
    pre-mix for FFN; each site's post/comb comes from its own input streams.
    """

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        attention: nn.Module,
        ffn: nn.Module,
        *,
        hc_mult: int = 4,
        norm_eps: float = 1e-20,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn = attention
        self.ffn = ffn
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attn_hc = Glm5NextHyperConnection(
            hidden_size,
            hc_mult=hc_mult,
            hc_eps=hc_eps,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            rms_norm_eps=norm_eps,
        )
        self.ffn_hc = Glm5NextHyperConnection(
            hidden_size,
            hc_mult=hc_mult,
            hc_eps=hc_eps,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            rms_norm_eps=norm_eps,
        )
        self._keep_hc_fp32()

    def _keep_hc_fp32(self) -> None:
        # The published Block allocates all six hc_* parameters under fp32,
        # even when its ordinary projection weights are bf16.
        for site in (self.attn_hc, self.ffn_hc):
            for name in ("fn", "base", "scale"):
                param = getattr(site, name)
                if param.dtype != torch.float32:
                    param.data = param.data.float()
            site._fn_fp32 = None

    def _apply(self, fn, recurse=True):
        original = [
            (site, name, getattr(site, name).detach())
            for site in (self.attn_hc, self.ffn_hc)
            for name in ("fn", "base", "scale")
        ]
        result = super()._apply(fn, recurse=recurse)
        for site, name, value in original:
            target = getattr(site, name)
            # A post-load .to(bf16) must not round the published fp32 mHC
            # values before restoring the dtype. Move the original values to
            # the target device instead. Meta tensors have no values to move.
            target.data = (
                target.data.float()
                if value.is_meta else value.to(device=target.device, dtype=torch.float32)
            )
            site._fn_fp32 = None
        return result

    @staticmethod
    def _collapse(streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return (pre.unsqueeze(-1) * streams.float()).sum(dim=2).to(streams.dtype)

    @staticmethod
    def _post(
        streams: torch.Tensor,
        out: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        # DeepSeek multiplies fp32 mHC coefficients by fp32 residuals before
        # casting the result back. GLM's ordinary path casts coefficients first.
        return update_streams(streams.float(), out.float(), post, comb).to(out.dtype)

    def forward(
        self,
        streams: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        *attention_args,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pre_mix.shape != streams.shape[:3]:
            raise ValueError("incoming pre-mix must have one value per stream and token")
        attn_pre, attn_post, attn_comb = self.attn_hc.mixes(streams)
        attn_input = self.attn_norm(self._collapse(streams, pre_mix))
        attn_out = self.attn(attn_input, start_pos, *attention_args)
        streams = self._post(streams, attn_out, attn_post, attn_comb)

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc.mixes(streams)
        ffn_input = self.ffn_norm(self._collapse(streams, attn_pre))
        ffn_out = self.ffn(ffn_input, image_mask)
        return self._post(streams, ffn_out, ffn_post, ffn_comb), ffn_pre
