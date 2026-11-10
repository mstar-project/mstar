"""Dense stand-ins for the Talker's paged KV, attention and position resources.

``DenseKVAttention`` keeps each request's K/V history per (label, layer) and
attends it with causal SDPA, which is what the paged kernel computes, on
whatever device the activations live. ``DensePositions`` plans absolute
positions from per-request counters (the Talker applies LongRoPE itself from
``pos_ids``). ``bind_dense_resources`` binds both LMs of a ``VoxCPM2Generator``
under real handles and returns a ``plan(seq_lens)`` to call before each step.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import StateSetup, node_handles
from mstar.model.voxcpm2.config import BASE_ATTN, BASE_KV, RES_ATTN, RES_KV
from mstar.model.voxcpm2.voxcpm2_model import VoxCPM2Model


class DenseKVAttention:
    requires_kv_write = True

    def __init__(self):
        self._seq_lens: list[int] = [1]
        self._kv: dict[tuple[str, int], list[tuple[list[torch.Tensor], list[torch.Tensor]]]] = {}
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    def layer_view(self, layer_idx: int) -> int:
        return layer_idx

    def opens_label(self, request_id, label: str) -> bool:
        del request_id, label
        return True

    def plan(self, seq_lens: list[int]) -> None:
        self._seq_lens = list(seq_lens)

    def reset(self) -> None:
        self._kv.clear()
        self._pending = None

    def write_kv(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, label: str) -> None:
        del layer_idx, label
        self._pending = (k, v)

    def run(self, q: torch.Tensor, label: str, kv_cache_layer: int, k=None, v=None, layer_idx=None) -> torch.Tensor:
        del k, v, layer_idx
        k_all, v_all = self._pending
        self._pending = None
        histories = self._kv.setdefault((label, kv_cache_layer), [([], []) for _ in self._seq_lens])
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


def bind_dense_resources(root):
    """Bind real handles over dense resources to ``root``, a ``VoxCPM2Generator`` or the Talker that holds one, and to
    the layers under it; returns ``(plan, reset)``. ``plan`` opens the arrays' handles over each step's spans."""
    base, res, pos = DenseKVAttention(), DenseKVAttention(), DensePositions()
    resources = {BASE_KV: base, BASE_ATTN: base, "base_pos": pos, RES_KV: res, RES_ATTN: res}
    declarations = VoxCPM2Model.declare_state(SimpleNamespace(config=root.config))
    handles = node_handles(
        SimpleNamespace(declare_state=lambda: declarations), "Talker", resources, setup=StateSetup(),
    )
    root.state_handles = handles
    for module in root.modules():
        bind = getattr(module, "bind_state", None)
        if bind is not None:
            bind(handles)
    device = next(root.parameters()).device

    def plan(seq_lens: list[int]) -> None:
        base.plan(seq_lens)
        res.plan(seq_lens)
        pos.plan(seq_lens, device=device)
        step = SubmoduleStep(
            segments=[Segment(row, "main", n) for row, n in enumerate(seq_lens)],
            state={BASE_KV: ArrayStep(), RES_KV: ArrayStep()},
        )
        step.set_ctx(StepContext(tuple(range(len(seq_lens))), "talker_decode", 0, False))
        for key in (BASE_KV, RES_KV):
            handles[key].end_step(step)
            handles[key].begin_step(step)

    def reset() -> None:
        base.reset()
        res.reset()
        pos.reset()

    return plan, reset
