"""A CPU stand-in for the array handle a block attends through.

Lets the AC predictor's paged path run without FlashInfer or a GPU: the
accumulated K/V history is what the real path holds in a request's pages,
and F.scaled_dot_product_attention over it is what the kernel computes, with no
mask since the array's read is ``Bidirectional``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.vjepa2.config import KV_CACHE


class FakeKVAttention:
    """Stands in for the array handle every block holds. Per layer and per
    request we keep the K/V tokens seen across time steps, which is what the
    real paged path holds in that request's pages.

    ``plan`` mirrors what preprocess does in production: it says how to split
    the flattened ``[sum(seq_lens), H, D]`` batch back into per-request slices.
    """

    def __init__(self):
        self._seq_lens: list[int] = []
        # layer -> per-request (list[k], list[v])
        self._kv: dict[int, list[tuple[list[torch.Tensor], list[torch.Tensor]]]] = {}
        # this step's add, consumed by the next `read`
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    # --- planning

    def plan(self, seq_lens: list[int]) -> None:
        """Record per-request token counts for the upcoming forward pass."""
        self._seq_lens = list(seq_lens)
        n_req = len(seq_lens)
        for layer, req_kvs in self._kv.items():
            if len(req_kvs) != n_req:
                self._kv[layer] = [([], []) for _ in range(n_req)]

    # --- the two calls a block makes

    def add(self, layer: torch.Tensor, *, k: torch.Tensor, v: torch.Tensor) -> None:
        self._pending = (k, v)

    def read(self, layer: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        layer = int(layer)
        assert self._pending is not None, "read without a preceding add"
        k_all, v_all = self._pending
        self._pending = None

        # Fall back to single-request when `plan` was not called.
        seq_lens = self._seq_lens if self._seq_lens else [q.shape[0]]
        n_req = len(seq_lens)

        if layer not in self._kv:
            self._kv[layer] = [([], []) for _ in range(n_req)]

        req_kvs = self._kv[layer]
        q_splits = torch.split(q, seq_lens)
        k_splits = torch.split(k_all, seq_lens)
        v_splits = torch.split(v_all, seq_lens)

        outputs: list[torch.Tensor] = []
        for i, (q_i, k_i, v_i) in enumerate(zip(q_splits, k_splits, v_splits, strict=False)):
            req_kvs[i][0].append(k_i)
            req_kvs[i][1].append(v_i)

            all_k = torch.cat(req_kvs[i][0], dim=0)  # [ctx_tokens, H, D]
            all_v = torch.cat(req_kvs[i][1], dim=0)

            # SDPA: [1, H, L, D] x [1, H, S, D]
            q_t = q_i.permute(1, 0, 2).unsqueeze(0)
            k_t = all_k.permute(1, 0, 2).unsqueeze(0)
            v_t = all_v.permute(1, 0, 2).unsqueeze(0)
            out_i = F.scaled_dot_product_attention(q_t, k_t, v_t)
            outputs.append(out_i.squeeze(0).permute(1, 0, 2))  # [L, H, D]

        return torch.cat(outputs, dim=0)  # [sum(seq_lens), H, D]


def bind_fake_resources(model) -> FakeKVAttention:
    """Give every block the same fake handle, as the engine does with the real
    one; the layer id is what keeps their histories apart."""
    handle = FakeKVAttention()
    for block in model.predictor_blocks:
        block.attn.bind_state({KV_CACHE: handle})
    return handle
