"""Test-support helpers for the GLM-5.2 fp8-block path.

NOT part of the serving path (the real checkpoint arrives pre-quantized).
These fabricate a synthetic fp8-block checkpoint and its exact bf16
reference so goldens can assert the load path (``quantization.py`` +
``weight_loader.py``) reproduces the reference bit-for-bit — the same role
``kimi_k2_7/_testing.py`` plays for compressed-tensors.
"""
from __future__ import annotations

import torch

from mstar.model.glm52.quantization import FP8_DTYPE, dequantize_fp8_block_weight


def fake_quantize_fp8_block(
    weight: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fp8 weight, fp32 scale_inv, exact bf16 dequantized reference).

    Per block: scale = amax / 448 (e4m3 max normal), quantize w / scale to
    e4m3, store scale as ``weight_scale_inv`` (the multiply-back convention).
    """
    out_f, in_f = weight.shape
    bo, bi = block_size
    n_bo, n_bi = -(-out_f // bo), -(-in_f // bi)

    w = weight.to(torch.float32)
    padded = torch.zeros(n_bo * bo, n_bi * bi, dtype=torch.float32)
    padded[:out_f, :in_f] = w
    blocks = padded.view(n_bo, bo, n_bi, bi)
    amax = blocks.abs().amax(dim=(1, 3))  # (n_bo, n_bi)
    scale_inv = amax / 448.0  # e4m3 max normal value
    scale_inv = torch.where(scale_inv == 0, torch.ones_like(scale_inv), scale_inv)

    scale_bc = scale_inv.repeat_interleave(bo, dim=0)[:out_f]
    scale_bc = scale_bc.repeat_interleave(bi, dim=1)[:, :in_f]
    w_fp8 = (w / scale_bc).to(FP8_DTYPE)

    dequant = dequantize_fp8_block_weight(w_fp8, scale_inv, block_size=block_size)
    return w_fp8, scale_inv, dequant


class ReferenceCacheHandle:
    """CPU stand-in for the naive-path ``BatchedCacheManager`` contract the
    GLM-5.2 model consumes: position-addressed per-(request, layer) K/V rows
    (a dict emulating the paged layout — writes at a position OVERWRITE, so
    rewound tails heal exactly like pages do), dense causal attention per
    query row, and the plan/advance/rewind position bookkeeping. Enough to
    drive real multi-step decode numerics — including the M3 draft loop's
    extra layer plane — without an engine or GPU."""

    class _State:
        def __init__(self):
            self.position_id_start = 0
            self.seq_len = 0
            self.page_indices: list[int] = []

    def __init__(self, request_ids: list[str]):
        self.request_ids = list(request_ids)
        self.layer_idx = 0
        self._states = {rid: self._State() for rid in self.request_ids}
        self._store: dict[tuple[str, int], dict[int, tuple]] = {}
        self._plan: tuple[list[int], list[int]] | None = None

    # -- bookkeeping ------------------------------------------------------
    def set_active_label(self, label):
        pass

    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = layer_idx

    def get_qo_indptr_buf(self, label):
        return None

    def _get_state(self, request_id: str, label=None):
        return self._states[request_id]

    def plan_attention(self, seq_lens, is_causal=True, label=None):
        # Snapshot write positions at plan time, like the paged backends.
        self._plan = (
            list(seq_lens),
            [self._states[rid].position_id_start for rid in self.request_ids],
        )

    def plan_rope(self, seq_lens, pos_ids=None, label=None):
        pass  # the naive attention path ropes in-module

    def advance_seq_lens(self, pos_id_ns=None):
        seq_lens, _ = self._plan
        for rid, sl in zip(self.request_ids, seq_lens, strict=True):
            st = self._states[rid]
            st.seq_len += sl
            st.position_id_start += sl

    def rewind_seq_lens(self, ns):
        for i, rid in enumerate(self.request_ids):
            n = ns if isinstance(ns, int) else ns[i]
            st = self._states[rid]
            assert 0 <= n <= st.seq_len
            st.seq_len -= n
            st.position_id_start -= n

    # -- attention --------------------------------------------------------
    def run_attention(self, q, k, v, layer_idx=None):
        layer = self.layer_idx if layer_idx is None else layer_idx
        seq_lens, starts = self._plan
        scale = q.shape[-1] ** -0.5
        outs = []
        row = 0
        for i, rid in enumerate(self.request_ids):
            sl, start = seq_lens[i], starts[i]
            store = self._store.setdefault((rid, layer), {})
            for j in range(sl):
                store[start + j] = (k[row + j], v[row + j])
            for j in range(sl):
                pos = sorted(p for p in store if p <= start + j)
                keys = torch.stack([store[p][0] for p in pos]).float()
                vals = torch.stack([store[p][1] for p in pos]).float()
                qj = q[row + j].float()  # (H, D)
                scores = torch.einsum("hd,thd->ht", qj, keys) * scale
                attn = torch.softmax(scores, dim=-1)
                outs.append(torch.einsum("ht,thd->hd", attn, vals).to(q.dtype))
            row += sl
        return torch.stack(outs)

    # -- test introspection ----------------------------------------------
    def committed_rows(self, request_id: str, layer: int):
        """K/V rows at positions below the counter — the verified stream."""
        st = self._states[request_id]
        store = self._store.get((request_id, layer), {})
        return {p: store[p] for p in store if p < st.position_id_start}
