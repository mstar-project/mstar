"""GLM-5.3-Flash mHC: Sinkhorn manifold, shapes, stack parity, dtype discipline.

The parity test does NOT call ``mhc.py`` against itself: the expected path is
an independent numpy fp64 port of the HF reference computation written inline
here (explicit einsums / division-form RMSNorm), so a transcription bug in
``mhc.py`` cannot regenerate its own expectation. The dtype test runs the whole
forward under a ``TorchDispatchMode`` recorder and asserts no aten op ever
produced fp64 — and, as a decode-path canary, that no host-sync
(``_local_scalar_dense``) or data-dependent-shape op was dispatched (lane
ground rule 2). The KDA forget gate is imported from ``kda.py`` — the ONE
implementation the model graph uses (the duplicate that once lived in
``mhc.py`` diverged in projection dtype). Pure torch + numpy; imports without
flashinfer/triton on any machine (lane ground rule 4).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mstar.model.glm5_next.kda import (
    Glm5NextForgetGate,
    Glm5NextKdaConfig,
)
from mstar.model.glm5_next.mhc import (
    Glm5NextHyperConnection,
    Glm5NextHyperHead,
    expand_streams,
    sinkhorn_normalize,
    update_streams,
)

HC_EPS = 1e-6
RMS_EPS = 1e-5
SINKHORN_ITERS = 20


# --- independent numpy fp64 reference (ported from the HF computation) ------


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_softmax_last(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _np_hyper_connection(streams, fn, base, scale, hc):
    """pre/post/comb + collapse, division-form norm and explicit slicing."""
    b, s, h, d = streams.shape
    flat = streams.reshape(b, s, h * d)
    flat = flat / np.sqrt((flat**2).mean(axis=-1, keepdims=True) + RMS_EPS)
    mixl = flat @ fn.T
    pre_w, post_w, comb_w = mixl[..., :hc], mixl[..., hc:2 * hc], mixl[..., 2 * hc:]

    pre = _np_sigmoid(pre_w * scale[0] + base[:hc]) + HC_EPS
    post = 2.0 * _np_sigmoid(post_w * scale[1] + base[hc:2 * hc])
    logits = comb_w.reshape(b, s, hc, hc) * scale[2] + base[2 * hc:].reshape(hc, hc)
    comb = _np_softmax_last(logits) + HC_EPS
    comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + HC_EPS)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + HC_EPS)
    collapsed = (pre[..., None] * streams).sum(axis=2)
    return post, comb, collapsed


def _np_update_streams(residual, sublayer_out, post, comb):
    """streams = post . out + comb^T @ residual, as one explicit einsum."""
    place = post[..., :, None] * sublayer_out[..., None, :]
    mix = np.einsum("bsji,bsjd->bsid", comb, residual)
    return place + mix


# --- (1) Sinkhorn: rows AND cols sum to 1 after 20 iterations ---------------


def test_sinkhorn_rows_and_cols_sum_to_one():
    torch.manual_seed(1)
    logits = torch.randn(5, 7, 4, 4)
    start = torch.softmax(logits, dim=-1) + HC_EPS
    ones = torch.ones(5, 7, 4)
    # The softmax start is row-stochastic but nowhere near column-stochastic.
    assert not torch.allclose(start.sum(dim=-2), ones, atol=1e-2)

    out = sinkhorn_normalize(start, SINKHORN_ITERS, HC_EPS)

    assert out.shape == start.shape
    assert bool((out > 0).all())
    assert torch.allclose(out.sum(dim=-1), ones, atol=1e-4)
    assert torch.allclose(out.sum(dim=-2), ones, atol=1e-4)


def test_sinkhorn_near_permutation_regime():
    """Sharpened logits (x3): near-permutation matrices converge slower.

    Columns are exact regardless (the last op is a column normalize; only the
    eps guard offsets them, ~4e-6); rows land within 5e-2 at 20 iterations
    (measured max 2.7e-2 at this seed) and keep shrinking with more rounds.
    """
    torch.manual_seed(1)
    start = torch.softmax(torch.randn(5, 7, 4, 4) * 3.0, dim=-1) + HC_EPS
    ones = torch.ones(5, 7, 4)

    out = sinkhorn_normalize(start, SINKHORN_ITERS, HC_EPS)
    assert torch.allclose(out.sum(dim=-2), ones, atol=1e-4)
    row_dev = (out.sum(dim=-1) - 1).abs().max()
    assert row_dev < 5e-2

    more = sinkhorn_normalize(start, 4 * SINKHORN_ITERS, HC_EPS)
    assert (more.sum(dim=-1) - 1).abs().max() < row_dev


# --- (2) mixing-matrix shapes at hc_mult 4, hidden 4096 ---------------------


def test_mixing_shapes_at_checkpoint_size():
    torch.manual_seed(2)
    hc = Glm5NextHyperConnection(hidden_size=4096, hc_mult=4)
    # mix = (2 + 4) * 4 = 24 rows over the flattened 4 * 4096 streams.
    assert hc.fn.shape == (24, 16384)
    assert hc.base.shape == (24,)
    assert hc.scale.shape == (3,)

    streams = expand_streams(torch.randn(2, 3, 4096, dtype=torch.bfloat16))
    assert streams.shape == (2, 3, 4, 4096)
    post, comb, collapsed = hc(streams)
    assert post.shape == (2, 3, 4)
    assert comb.shape == (2, 3, 4, 4)
    assert collapsed.shape == (2, 3, 4096)
    # The comb the module hands out is on the doubly-stochastic manifold.
    # Random-init fn at hidden 4096 gives O(2.5) comb logits, so the slowest
    # rows sit ~1e-3 from 1 after the 20 fixed iterations (cols are exact).
    ones = torch.ones(2, 3, 4)
    assert torch.allclose(comb.sum(dim=-1), ones, atol=5e-3)
    assert torch.allclose(comb.sum(dim=-2), ones, atol=1e-4)

    updated = update_streams(streams, collapsed, post, comb)
    assert updated.shape == streams.shape
    assert Glm5NextHyperHead()(updated).shape == (2, 3, 4096)

    # The post-load fp32 ``fn`` cache must be bit-identical to the
    # per-call ``.float()`` fallback it replaces.
    hc.process_weights_after_loading(torch.device("cpu"))
    assert hc._fn_fp32 is not None
    post2, comb2, collapsed2 = hc(streams)
    assert torch.equal(post2, post)
    assert torch.equal(comb2, comb)
    assert torch.equal(collapsed2, collapsed)


# --- (3) two-layer toy stack vs the independent reference -------------------


def test_two_layer_stack_matches_independent_reference():
    torch.manual_seed(3)
    b, s, h, d = 2, 3, 4, 8

    sites = []  # per layer, per site: (module, fn, base, scale)
    for _ in range(2):
        layer_sites = []
        for _ in range(2):
            mod = Glm5NextHyperConnection(hidden_size=d, hc_mult=h)
            fn = torch.randn(24, h * d) * 0.5
            base = torch.randn(24) * 0.3
            scale = 1.0 + 0.25 * torch.randn(3)
            with torch.no_grad():
                mod.fn.copy_(fn)
                mod.base.copy_(base)
                mod.scale.copy_(scale)
            layer_sites.append((mod, fn, base, scale))
        sites.append(layer_sites)
    w_attn = [torch.randn(d, d) * 0.5 for _ in range(2)]
    w_mlp = [torch.randn(d, d) * 0.5 for _ in range(2)]

    # mhc.py path, fp32 end to end (post/comb casts are no-ops there).
    x = torch.randn(b, s, d)
    streams = expand_streams(x, hc_mult=h)
    with torch.no_grad():
        for layer in range(2):
            residual = streams
            post, comb, col = sites[layer][0][0](streams)
            streams = update_streams(residual, torch.tanh(col @ w_attn[layer]), post, comb)
            residual = streams
            post, comb, col = sites[layer][1][0](streams)
            z = col @ w_mlp[layer]
            streams = update_streams(residual, z * torch.sigmoid(z), post, comb)
        out = Glm5NextHyperHead()(streams)

    # Independent fp64 reference over the SAME weights and toy sublayers.
    f64 = np.float64
    streams_np = np.broadcast_to(
        x.numpy().astype(f64)[:, :, None, :], (b, s, h, d)).copy()
    for layer in range(2):
        for site in range(2):
            _, fn, base, scale = sites[layer][site]
            residual_np = streams_np
            post_np, comb_np, col_np = _np_hyper_connection(
                residual_np, fn.numpy().astype(f64), base.numpy().astype(f64),
                scale.numpy().astype(f64), h)
            if site == 0:
                sub_np = np.tanh(col_np @ w_attn[layer].numpy().astype(f64))
            else:
                z_np = col_np @ w_mlp[layer].numpy().astype(f64)
                sub_np = z_np * _np_sigmoid(z_np)
            streams_np = _np_update_streams(residual_np, sub_np, post_np, comb_np)
    expected = streams_np.mean(axis=2)

    assert out.dtype == torch.float32
    np.testing.assert_allclose(
        out.numpy().astype(f64), expected, rtol=1e-5, atol=1e-5)


# --- (4) dtype stability: no silent fp64, no host syncs ---------------------


class _OpRecorder(TorchDispatchMode):
    """Record every aten op name and every produced tensor dtype."""

    def __init__(self):
        super().__init__()
        self.ops = set()
        self.dtypes = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self.ops.add(func.overloadpacket.__name__)
        stack = [out]
        while stack:
            item = stack.pop()
            if isinstance(item, torch.Tensor):
                self.dtypes.add(item.dtype)
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        return out


def test_dtype_stability_no_silent_fp64():
    torch.manual_seed(4)
    d = 32
    hc = Glm5NextHyperConnection(hidden_size=d, hc_mult=4).to(torch.bfloat16)
    gate = Glm5NextForgetGate(
        Glm5NextKdaConfig(hidden_size=d, linear_num_heads=2, linear_head_dim=8),
        dtype=torch.bfloat16)
    # Checkpoint dtypes: projections bf16, dt_bias / A_log strict fp32.
    assert gate.f_a_proj.weight.dtype == torch.bfloat16
    assert gate.f_b_proj.weight.dtype == torch.bfloat16
    assert gate.dt_bias.dtype == torch.float32
    assert gate.A_log.dtype == torch.float32

    streams = expand_streams(torch.randn(2, 3, d, dtype=torch.bfloat16))
    rec = _OpRecorder()
    with torch.no_grad(), rec:
        post, comb, collapsed = hc(streams)
        updated = update_streams(streams, collapsed, post, comb)
        head = Glm5NextHyperHead()(updated)
        g = gate(collapsed)

    # Compute in fp32, carry in bf16 — nothing ever widens to fp64.
    assert post.dtype == torch.float32
    assert comb.dtype == torch.float32
    assert collapsed.dtype == torch.bfloat16
    assert updated.dtype == torch.bfloat16
    assert head.dtype == torch.bfloat16
    assert g.dtype == torch.float32
    assert torch.float64 not in rec.dtypes

    # Decode-path canaries: no host syncs, no data-dependent shapes.
    forbidden = {"_local_scalar_dense", "item", "nonzero", "masked_select", "unique"}
    assert not (rec.ops & forbidden), rec.ops & forbidden


# --- forget gate: the live branch vs the independent reference --------------


def test_forget_gate_matches_independent_reference():
    torch.manual_seed(5)
    b, s, hidden, heads, head_dim = 2, 3, 16, 2, 4
    cfg = Glm5NextKdaConfig(
        hidden_size=hidden, linear_num_heads=heads, linear_head_dim=head_dim)
    x = torch.randn(b, s, hidden)
    w_a = torch.randn(head_dim, hidden) * 0.5
    w_b = torch.randn(heads * head_dim, head_dim) * 0.5
    # Large positive dt_bias entries drive the sigmoid to saturation so the
    # closed bound is exercised, not just the interior.
    dt_bias = torch.randn(heads * head_dim) * 0.5
    dt_bias[::3] += 30.0
    a_log = torch.randn(heads) * 0.3

    f64 = np.float64
    g_np = (
        x.numpy().astype(f64)
        @ w_a.numpy().astype(f64).T
        @ w_b.numpy().astype(f64).T
        + dt_bias.numpy().astype(f64)
    ).reshape(b, s, heads, head_dim)
    decay_np = np.exp(a_log.numpy().astype(f64))[None, None, :, None]

    gate = Glm5NextForgetGate(cfg, dtype=torch.float32)
    with torch.no_grad():
        gate.f_a_proj.weight.copy_(w_a)
        gate.f_b_proj.weight.copy_(w_b)
        gate.dt_bias.copy_(dt_bias)
        gate.A_log.copy_(a_log)
        out = gate(x)

    assert out.shape == (b, s, heads, head_dim)
    assert out.dtype == torch.float32
    expected = cfg.gate_lower_bound * _np_sigmoid(decay_np * g_np)
    # Saturated channels (dt_bias +30) sit exactly at the bound.
    assert bool((out < 0).all()) and bool((out >= cfg.gate_lower_bound).all())
    np.testing.assert_allclose(
        out.numpy().astype(f64), expected, rtol=1e-4, atol=1e-5)

    # The HF softplus branch (gate_lower_bound=None) is dead for this
    # checkpoint: the one live implementation refuses it loudly instead of
    # shipping a second, untested math path.
    with pytest.raises(ValueError, match="softplus"):
        Glm5NextForgetGate(
            Glm5NextKdaConfig(
                hidden_size=hidden, linear_num_heads=heads,
                linear_head_dim=head_dim, gate_lower_bound=None))
