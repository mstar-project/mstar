"""A CPU stand-in for the KV + attention resources a block attends through.

Lets the AC predictor's paged path run without FlashInfer or a GPU: the
accumulated K/V history is what the real path holds in a request's pages,
and F.scaled_dot_product_attention over it is what the kernel computes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class FakeKVAttention:
    """Stands in for the KV and attention resources a block attends through.

    One object serves as both: ``AttentionCallable`` writes K/V through the KV
    resource and then calls the attention one, and the accumulated history is
    the same state either way. Per layer and per request we keep the K/V tokens
    seen across time steps, which is what the real paged path holds in that
    request's pages.

    ``plan`` mirrors what preprocess does in production: it says how to split
    the flattened ``[sum(seq_lens), H, D]`` batch back into per-request slices.
    """

    requires_kv_write = True

    def __init__(self):
        self._label = "main"
        self._layer_idx = 0
        self._seq_lens: list[int] = []
        # layer -> per-request (list[k], list[v])
        self._kv: dict[int, list[tuple[list[torch.Tensor], list[torch.Tensor]]]] = {}
        # this step's writes, consumed by the next `run`
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    # --- resource cursors (see AttentionCallable)

    @property
    def default_label(self) -> str:
        return self._label

    def set_default_label(self, label: str) -> None:
        self._label = label

    def set_default_layer_idx(self, layer_idx: int) -> None:
        self._layer_idx = layer_idx

    def layer_view(self):
        return self._layer_idx

    # --- planning

    def plan(self, seq_lens: list[int]) -> None:
        """Record per-request token counts for the upcoming forward pass."""
        self._seq_lens = list(seq_lens)
        n_req = len(seq_lens)
        for layer, req_kvs in self._kv.items():
            if len(req_kvs) != n_req:
                self._kv[layer] = [([], []) for _ in range(n_req)]

    # --- the two calls a block makes

    def write_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self._pending = (k, v)

    def run(self, q: torch.Tensor, kv_cache_layer, k=None, v=None) -> torch.Tensor:
        del k, v  # taken from the write above, as the paged backends do
        layer = kv_cache_layer
        assert self._pending is not None, "run without a preceding write_kv"
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
    """Give every block the same fake pair, as the engine does with the real
    resources; the layer cursor is what keeps their histories apart."""
    resources = FakeKVAttention()
    for block in model.predictor_blocks:
        block.attn.bind_resources({"attn": resources, "kv": resources})
    return resources
