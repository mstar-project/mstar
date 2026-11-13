"""Controls for the reference comparison: a deliberate bug it must catch, and the reference's own spread."""

import torch

from mstar.model.personaplex.components.temporal import DenseTemporalCache


def candidates(logits):
    values, ids = logits.float().topk(2, dim=-1)
    return dict(tokens=ids[0].tolist(), logits=values[0].tolist(), margin=float(values[0, 0] - values[0, 1]))


def first_reference_decision(records, other):
    """The first sampled (unforced) choice where two observed NVIDIA runs pick different tokens."""
    for step, (mine, theirs) in enumerate(zip(records, other, strict=True)):
        if mine["phase"] != "generation":
            continue
        decisions = [("text", None, mine["text_logits"].reshape(1, -1), theirs["text_logits"].reshape(1, -1),
                      bool(mine["provided"][0, 0, 0]))]
        decisions += [("depth", d, mine["depth_logits"][:, d], theirs["depth_logits"][:, d],
                       bool(mine["depth_provided"][0, d])) for d in range(16)]
        for kind, slot, first, second, forced in decisions:
            if not forced and not torch.equal(first.argmax(-1), second.argmax(-1)):
                return dict(record=step, kind=kind, depth_step=slot, default=candidates(first), math=candidates(second))
    return None


class HalfSplitRope(DenseTemporalCache):
    """The negative control's deliberate bug: RoPE over split halves ``(i, i + D/2)``, not adjacent pairs.

    This is the layout mix-up ``interleave=True`` guards against. (A uniform
    position offset would not do: RoPE depends only on relative position, so it
    leaves attention unchanged.)
    """

    def rope(self, q, k):
        if q.dim() == 3:  # the base class re-enters rope with a time axis
            return super().rope(q, k)
        half = q.shape[-1] // 2
        pairs = torch.stack((torch.arange(half), torch.arange(half) + half), dim=-1).flatten().to(q.device)
        inverse = torch.argsort(pairs)
        q, k = super().rope(q[..., pairs], k[..., pairs])
        return q[..., inverse], k[..., inverse]
