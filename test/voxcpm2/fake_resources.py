"""Dense stand-ins for the Talker's paged KV, attention and position resources.

``DenseKVAttention`` keeps each request's K/V history per (label, layer) and
attends it with causal SDPA, which is what the paged kernel computes, on
whatever device the activations live. ``DensePositions`` plans absolute
positions from per-request counters (the Talker applies LongRoPE itself from
``pos_ids``). ``bind_dense_resources`` binds both LMs of a ``VoxCPM2Generator``
and returns a ``plan(seq_lens)`` to call before each step.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.voxcpm2.config import BASE_ATTN, BASE_KV, BASE_POS, RES_ATTN, RES_KV


class DenseKVAttention:
    requires_kv_write = True

    def __init__(self):
        self._label = "main"
        self._layer_idx = 0
        self._seq_lens: list[int] = [1]
        self._kv: dict[tuple[str, int], list[tuple[list[torch.Tensor], list[torch.Tensor]]]] = {}
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def default_label(self) -> str:
        return self._label

    def set_default_label(self, label: str) -> None:
        self._label = label

    def set_default_layer_idx(self, layer_idx: int) -> None:
        self._layer_idx = layer_idx

    def layer_view(self):
        return self._layer_idx

    def plan(self, seq_lens: list[int]) -> None:
        self._seq_lens = list(seq_lens)

    def reset(self) -> None:
        self._kv.clear()
        self._pending = None

    def write_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self._pending = (k, v)

    def run(self, q: torch.Tensor, kv_cache_layer, k=None, v=None) -> torch.Tensor:
        del k, v
        k_all, v_all = self._pending
        self._pending = None
        histories = self._kv.setdefault((self._label, kv_cache_layer), [([], []) for _ in self._seq_lens])
        outputs = []
        for i, (q_i, k_i, v_i) in enumerate(
            zip(
                torch.split(q, self._seq_lens),
                torch.split(k_all, self._seq_lens),
                torch.split(v_all, self._seq_lens),
                strict=True,
            )
        ):
            histories[i][0].append(k_i)
            histories[i][1].append(v_i)
            keys = torch.cat(histories[i][0])
            vals = torch.cat(histories[i][1])
            n_q, n_k = q_i.shape[0], keys.shape[0]
            q_pos = torch.arange(n_k - n_q, n_k, device=q.device).unsqueeze(1)
            mask = torch.arange(n_k, device=q.device).unsqueeze(0) <= q_pos
            out = F.scaled_dot_product_attention(
                q_i.transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                vals.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                enable_gqa=q_i.shape[1] != keys.shape[1],
            )
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)


class DensePositions:
    def __init__(self):
        self._counters: list[int] = []
        self._pos_ids: torch.Tensor | None = None

    def plan(self, seq_lens: list[int], device=None) -> None:
        if len(self._counters) != len(seq_lens):
            self._counters = [0] * len(seq_lens)
        ids = []
        for i, n in enumerate(seq_lens):
            ids.extend(range(self._counters[i], self._counters[i] + n))
            self._counters[i] += n
        self._pos_ids = torch.tensor(ids, dtype=torch.long, device=device)

    def reset(self) -> None:
        self._counters = []
        self._pos_ids = None

    def pos_ids(self, label: str = "main") -> torch.Tensor:
        del label
        return self._pos_ids


def bind_dense_resources(generator):
    """Bind dense resources to both LMs; returns ``(plan, reset)``."""
    base, res, pos = DenseKVAttention(), DenseKVAttention(), DensePositions()
    resources = {BASE_KV: base, BASE_ATTN: base, BASE_POS: pos, RES_KV: res, RES_ATTN: res}
    for module in generator.modules():
        if module is not generator and hasattr(module, "bind_resources"):
            module.bind_resources(resources)
    device = next(generator.parameters()).device

    def plan(seq_lens: list[int]) -> None:
        base.plan(seq_lens)
        res.plan(seq_lens)
        pos.plan(seq_lens, device=device)

    def reset() -> None:
        base.reset()
        res.reset()
        pos.reset()

    return plan, reset
