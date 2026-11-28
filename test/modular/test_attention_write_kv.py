"""``write_kv``: the K/V an attention layer's ``forward`` would write, without attending."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.model.components.attention import Attention
from mstar.model.components.distributed.attention import ParallelAttention
from mstar.model.components.dspark.context_write import project_kv, write_context, write_layer_kv


class _KV:
    """A KV resource that records what it is asked to write."""

    default_label = "main"

    def __init__(self):
        self.written = []

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def layer_view(self):
        return None

    def write_kv(self, k, v):
        self.written.append((k.clone(), v.clone()))


class _Attn:
    requires_kv_write = True

    def set_default_label(self, label):
        pass

    def set_default_layer_idx(self, idx):
        pass

    def run(self, q, kv_cache_layer=None, k=None, v=None):
        return torch.zeros_like(q)


class _Pos:
    """1D RoPE stand-in: a per-token rotation of q and k, so a write that skips it differs."""

    def apply_qk(self, q, k, label=None, **kwargs):
        scale = torch.arange(1, q.shape[0] + 1, dtype=q.dtype)[:, None, None]
        return q * scale, k * scale


@pytest.mark.parametrize("cls", [Attention, ParallelAttention], ids=lambda c: c.__name__)
def test_write_kv_writes_what_forward_writes(cls):
    torch.manual_seed(0)
    kwargs = dict(hidden_size=32, num_heads=4, num_kv_heads=2, head_dim=8, qk_norm=True,
                  attn_key="attn", kv_key="kv", pos_key="rope")
    layer = cls(**kwargs) if cls is Attention else cls(comm_group=None, **kwargs)
    # ParallelAttention's projections come from torch.empty; NaN would make torch.equal fail at random
    with torch.no_grad():
        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.02)
    kv = _KV()
    layer.bind_resources({"attn": _Attn(), "kv": kv, "rope": _Pos()})
    x = torch.randn(5, 32)
    with torch.no_grad():
        layer(x)
        layer.write_kv(x)
    (k_fwd, v_fwd), (k_w, v_w) = kv.written
    assert torch.equal(k_fwd, k_w) and torch.equal(v_fwd, v_w)


@pytest.mark.parametrize("write", ["stack", "dspark"])
def test_a_stacks_write_kv_moves_each_layers_cursor_to_its_own_index_under_its_label(write):
    from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig

    config = Qwen3LMConfig(vocab_size=16, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6,
                           rope_theta=1e4, max_position_embeddings=64)
    lm = Qwen3DenseLM(config, attn_key="attn", kv_key="kv", pos_key="rope")
    seen = []
    kv = _KV()
    kv.set_default_label = kv.set_default_layer_idx = seen.append
    for layer in lm.layers:
        layer.self_attn.bind_resources({"attn": _Attn(), "kv": kv, "rope": SimpleNamespace(
            apply_qk=_Pos().apply_qk, fusable_rope_theta=lambda head_dim: None)})
    with torch.no_grad():
        if write == "stack":
            lm.write_kv(torch.randn(4, 32), label="other")
        else:
            write_context(lm, torch.randn(4, 32), label="other")
    # the label is bound before the first layer writes, though another was bound last
    assert seen == ["other", 0, 1, 2] and len(kv.written) == 3


# ── DSpark's fused context write on Qwen3 layers ──


class _PagedKV(_KV):
    """An NHD layer of pages a fused kernel writes into (`write_target`), token t at page t // page, slot
    t % page."""

    def __init__(self, tokens: int, heads: int, head_dim: int, device, dtype, page: int = 16):
        super().__init__()
        pages = -(-tokens // page)
        self.cache = torch.zeros(pages, 2, page, heads, head_dim, device=device, dtype=dtype)
        t = torch.arange(tokens, device=device)
        self.page_idx, self.slot_idx, self.tokens = (t // page).int(), (t % page).int(), tokens

    def write_target(self):
        return self.cache, self.page_idx, self.slot_idx, self.tokens

    def layer_view(self):
        return self.cache


class _CachedAttn(_Attn):
    def reads_step_kv(self, label=None):
        return False


class _Positions:
    def __init__(self, pos_ids):
        self._pos_ids = pos_ids

    def fusable_rope_theta(self, head_dim):
        return 1e6

    def pos_ids(self, label):
        return self._pos_ids

    def apply_qk(self, q, k, label=None, **kwargs):
        if not q.is_cuda:
            return _Pos().apply_qk(q, k)
        from mstar.utils.attention import apply_rope_pos_ids

        return apply_rope_pos_ids(q, k, self._pos_ids, rope_theta=1e6)


def _qwen3_attention(device, dtype, hidden=128, heads=4, kv_heads=2, head_dim=32):
    from mstar.model.components.qwen3_lm import Qwen3Attention

    layer = Qwen3Attention(comm_group=None, hidden_size=hidden, num_heads=heads, num_kv_heads=kv_heads,
                           head_dim=head_dim, qk_norm=True, rope_theta=1e6, attn_key="attn", kv_key="kv",
                           pos_key="rope")
    with torch.no_grad():
        for p in layer.parameters():
            torch.nn.init.normal_(p, std=0.05)
    return layer.to(device=device, dtype=dtype)


def test_qwen3s_kv_projection_is_the_fused_projections_kv_rows():
    torch.manual_seed(0)
    layer = _qwen3_attention("cpu", torch.float64)
    x = torch.randn(7, 128, dtype=torch.float64)
    with torch.no_grad():
        _, k, v = layer._project_qkv(x)
        k_alone, v_alone = project_kv(layer, x)
    torch.testing.assert_close(k_alone, k, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(v_alone, v, rtol=1e-12, atol=1e-12)


def test_off_cuda_the_context_write_takes_the_unfused_path():
    torch.manual_seed(0)
    layer = _qwen3_attention("cpu", torch.float64)
    kv = _KV()
    layer.bind_resources({"attn": _Attn(), "kv": kv, "rope": _Positions(torch.arange(5))})
    x = torch.randn(5, 128, dtype=torch.float64)
    with torch.no_grad():
        layer(x)
        write_layer_kv(layer, x)
    (k_fwd, v_fwd), (k_w, v_w) = kv.written
    assert torch.equal(k_fwd, k_w) and torch.equal(v_fwd, v_w)


_TOY, _QWEN3_8B = (256, 4, 2, 128), (4096, 32, 8, 128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel runs on CUDA")
@pytest.mark.parametrize(
    ("shape", "tokens", "exact"),
    [(_TOY, 8, True), (_TOY, 64, True), (_TOY, 512, True),
     (_QWEN3_8B, 8, False), (_QWEN3_8B, 64, False), (_QWEN3_8B, 512, True)],
    ids=["toy-8", "toy-64", "toy-512", "qwen3_8b-8", "qwen3_8b-64", "qwen3_8b-512"],
)
def test_the_fused_context_write_is_what_qwen3s_fused_forward_writes(shape, tokens, exact):
    """On CUDA in bf16: the context write's K/V-only projection and its one kernel over no query heads write the
    pages the fused ``forward`` writes, and the unfused write differs from both only by its extra roundings.

    Bit for bit where cuBLAS runs the same GEMM kernel for the K/V rows as for the whole fused projection. At
    Qwen3-8B's width with few tokens it picks its kernel by N (2048 K/V columns against 6144), whose different
    summation order moves an element by up to the rounding of the tensor's largest values, so the bound there is
    one bf16 ulp at the tensor's top: 2^-7 of its largest magnitude, for every element. This K/V is the drafter's
    own: it changes only the drafts, never the target's replies, which verification keeps exact."""
    hidden, heads, kv_heads, head_dim = shape
    torch.manual_seed(0)
    layer = _qwen3_attention("cuda", torch.bfloat16, hidden, heads, kv_heads, head_dim)
    pos = torch.randint(0, 4000, (tokens,), device="cuda")
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    caches = {}
    for how in ("forward", "write_kv", "unfused"):
        kv = _PagedKV(tokens, kv_heads, head_dim, "cuda", torch.bfloat16)
        layer.bind_resources({"attn": _CachedAttn(), "kv": kv, "rope": _Positions(pos)})
        with torch.no_grad():
            if how == "forward":
                layer(x)
            elif how == "write_kv":
                write_layer_kv(layer, x)
            else:
                ParallelAttention.write_kv(layer, x)
                (k, v), = kv.written
                kv.cache[kv.page_idx, 0, kv.slot_idx] = k
                kv.cache[kv.page_idx, 1, kv.slot_idx] = v
        caches[how] = kv.cache
    diff = (caches["write_kv"].float() - caches["forward"].float()).abs().max().item()
    top = caches["forward"].float().abs().max().item()
    seen = f"max |diff| {diff}, max |forward| {top}"
    if exact:
        assert torch.equal(caches["write_kv"], caches["forward"]), seen
    else:
        assert diff <= 2**-7 * top, seen
    torch.testing.assert_close(caches["unfused"], caches["forward"], rtol=2e-2, atol=2e-2)
