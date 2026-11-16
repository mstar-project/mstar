"""DeepSeek V4.1's learned compressed KV, with caller-owned partial groups."""

from __future__ import annotations

import torch
from torch import nn


class _ReferenceRMSNorm(nn.Module):
    """Keep the reference's fp32 multiply before the cast back to BF16.

    The shared RMSNorm casts its weight to the input dtype. That changes
    DeepSeek's BF16 results, even on CPU.
    """

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepSeekV41Compressor(nn.Module):
    """Pool complete groups while returning the incomplete group's state.

    The graph owns ``(kv_state, score_state)`` for each request. The module has
    no cross-request buffers; a future Value array can retain these tensors.
    ``start_pos`` follows the reference's prefill-at-zero, one-token-decode
    schedule. The returned latent is before compressed-position RoPE.
    """

    def __init__(self, hidden_size: int, head_dim: int, ratio: int, *, norm_eps: float = 1e-20) -> None:
        super().__init__()
        if hidden_size < 1 or head_dim < 1 or ratio < 1:
            raise ValueError("compressor dimensions and ratio must be positive")
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.ratio = ratio
        self.norm = _ReferenceRMSNorm(head_dim, eps=norm_eps)
        self.wkv = nn.Linear(
            hidden_size, head_dim, bias=False,
            dtype=torch.float32 if ratio > 1 else torch.bfloat16,
        )
        self.wgate = nn.Linear(hidden_size, head_dim, bias=False, dtype=torch.float32) if ratio > 1 else None

    def _apply(self, fn, recurse=True):
        # Parent casts must not change checkpoint storage dtypes.
        kept = [(self.norm, "weight", self.norm.weight.detach(), torch.float32)]
        if self.ratio > 1:
            kept.extend((module, "weight", module.weight.detach(), torch.float32) for module in (self.wkv, self.wgate))
        else:
            kept.append((self.wkv, "weight", self.wkv.weight.detach(), torch.bfloat16))
        result = super()._apply(fn, recurse=recurse)
        for module, name, original, dtype in kept:
            param = getattr(module, name)
            param.data = param.data.to(dtype) if original.is_meta else original.to(param.device, dtype=dtype)
        return result

    def new_state(self, batch: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 1 or batch < 1:
            raise ValueError("partial state needs a ratio above one and a positive batch")
        shape = (batch, self.ratio, self.head_dim)
        return torch.zeros(shape, dtype=torch.float32, device=device), torch.full(
            shape, -torch.inf, dtype=torch.float32, device=device,
        )

    def _state(
        self, state: tuple[torch.Tensor, torch.Tensor] | None, batch: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            return self.new_state(batch, device)
        kv, score = state
        shape = (batch, self.ratio, self.head_dim)
        if (kv.shape != shape or score.shape != shape or kv.dtype != torch.float32 or
                score.dtype != torch.float32 or kv.device != device or score.device != device):
            raise ValueError("compressor partial state has the wrong shape, dtype or device")
        return kv.clone(), score.clone()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        if x.ndim != 3 or x.shape[-1] != self.hidden_size or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("compressor input must be a nonempty [batch, tokens, hidden] tensor")
        if type(start_pos) is not int or start_pos < 0 or (start_pos and x.shape[1] != 1):
            raise ValueError("compressor takes a prefill at zero or one token at a later position")
        if self.ratio == 1:
            if state is not None:
                raise ValueError("ratio-one compression has no partial state")
            return self.norm(self.wkv(x)), None
        if start_pos and state is None:
            raise ValueError("decode compression needs the request's partial state")

        kv_state, score_state = self._state(state, x.shape[0], x.device)
        projected, scores = self.wkv(x.float()), self.wgate(x.float())
        if start_pos == 0:
            complete, remainder = divmod(x.shape[1], self.ratio)
            cutoff = complete * self.ratio
            if remainder:
                kv_state[:, :remainder] = projected[:, cutoff:]
                score_state[:, :remainder] = scores[:, cutoff:]
            if not complete:
                return None, (kv_state, score_state)
            grouped_kv = projected[:, :cutoff].unflatten(1, (complete, self.ratio))
            grouped_scores = scores[:, :cutoff].unflatten(1, (complete, self.ratio))
            pooled = (grouped_kv * grouped_scores.softmax(dim=2)).sum(dim=2)
        else:
            slot = start_pos % self.ratio
            kv_state[:, slot] = projected[:, 0]
            score_state[:, slot] = scores[:, 0]
            if (start_pos + 1) % self.ratio:
                return None, (kv_state, score_state)
            pooled = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
        return self.norm(pooled.to(x.dtype)), (kv_state, score_state)
