import glob
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub.constants import HF_HUB_CACHE
from torch import nn

_HUB = Path(HF_HUB_CACHE)  # honours HF_HUB_CACHE and HF_HOME
GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)

needs_checkpoint = pytest.mark.skipif(not (GR00T.exists() and BACKBONE), reason="checkpoint not cached")


def randomize(module: nn.Module, seed: int, scale: float) -> None:
    """Fill every parameter from one seeded stream: default inits (identity LayerNorms,
    ``torch.empty`` tables) would hide wiring mistakes or read garbage."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(scale * torch.randn(parameter.shape, generator=generator))


class SegmentAttention:
    """Stands in for the ragged attention resource on SDPA: each segment attends only within
    itself; one segment over every row when ``seq_lengths`` is None."""

    def __init__(self, seq_lengths: tuple[int, ...] | None = None, causal: bool = False, gqa: bool = False):
        self.seq_lengths, self.causal, self.gqa = seq_lengths, causal, gqa

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out, start = [], 0
        for span in (q.shape[0],) if self.seq_lengths is None else self.seq_lengths:
            rows = slice(start, start + span)
            # the kernel's layout is [tokens, heads, dim]; SDPA wants [batch, heads, tokens, dim],
            # and without the batch axis every fused kernel refuses and it falls back to math,
            # which rounds unlike the reference's call (1.21x on GH200's vision bar)
            out.append(
                F.scaled_dot_product_attention(
                    q[rows].transpose(0, 1)[None], k[rows].transpose(0, 1)[None], v[rows].transpose(0, 1)[None],
                    is_causal=self.causal, enable_gqa=self.gqa,
                )[0].transpose(0, 1)
            )
            start += span
        return torch.cat(out)


def relative_gap(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    return ((ours - theirs).abs().max() / theirs.abs().max()).item()
