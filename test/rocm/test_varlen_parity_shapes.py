"""Attention at pi05's and V-JEPA2's shapes on an AMD GPU, against fp32 SDPA.

One case per attention call the two models make, at their head dims, lengths,
dtypes and masks:

- pi05 SigLIP so400m tower: HF SDPA in fp32, 16 heads of 72, 256 tokens per
  camera, no mask, eager and compiled.
- pi05 PaliGemma prefix and action expert: paged prefix-LM (non-causal) at head
  dim 256 with 8 query heads on one KV head, page 128, through
  ``AiterPagedAttentionManager`` (which hands these steps to FlashInfer's fa2),
  eager and captured, plus RoPE over the full head dim.
- V-JEPA2 ViT-L encoder: math SDPA in bf16, 16 heads of 64, 1024 tokens, no
  mask, eager and compiled.
- ``mstar::varlen_attention`` at the encoder head dims (64, 72, 88).

    pytest test/rocm/test_varlen_parity_shapes.py
"""

import random

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutput, KVPlanOutputs, SequenceView, build_paged_indptrs
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = torch.device("cuda")
PAGE = 128
NUM_PAGES = 96
# pi05: 3 cameras x 256 image tokens plus the tokenized task and state
PI05_PREFIX = (865, 908, 841, 930)
PI05_HORIZON = 50


def _rel(out, ref):
    """Relative L2 error and the count of non-finite outputs."""
    o = out[: ref.shape[0]].double()
    bad = int((~torch.isfinite(o)).sum())
    return ((o.nan_to_num() - ref.double()).norm() / ref.double().norm()).item(), bad


def _sdpa_ref(q, k, v, scale=None):
    """fp32 math SDPA over [B, H, N, D]."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale)


# --- encoders: SDPA as the models call it ----------------------------------------


def _bhnd(b, h, n, d, dtype, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(b, h, n, d, generator=g, device=DEV).to(dtype) for _ in range(3)]


@pytest.mark.parametrize("compiled", [False, True])
def test_siglip_fp32_sdpa_hd72(compiled):
    """HF's SiglipAttention on SDPA, fp32 (pi05 keeps the tower in fp32)."""
    q, k, v = _bhnd(3, 16, 256, 72, torch.float32, 0)

    def attn(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, scale=72 ** -0.5)

    fn = torch.compile(attn, fullgraph=True) if compiled else attn
    err, bad = _rel(fn(q, k, v), _sdpa_ref(q, k, v, 72 ** -0.5))
    assert bad == 0 and err < 1e-4, (err, bad)


@pytest.mark.parametrize("compiled", [False, True])
def test_vjepa2_bf16_math_sdpa_hd64(compiled):
    """V-JEPA2's encoder attention: math SDPA in bf16 over 1024 tokens."""
    q, k, v = _bhnd(1, 16, 1024, 64, torch.bfloat16, 1)

    def attn(q, k, v):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            return F.scaled_dot_product_attention(q, k, v)

    fn = torch.compile(attn) if compiled else attn
    err, bad = _rel(fn(q, k, v), _sdpa_ref(q, k, v))
    assert bad == 0 and err < 1e-2, (err, bad)


@pytest.mark.parametrize("d", [64, 72, 88])
def test_varlen_attention_encoder_head_dims(d):
    """``mstar::varlen_attention`` (AITER varlen) at encoder head dims, packed images."""
    from mstar.model.components.hf_attention import varlen_attention, varlen_attention_available

    if not varlen_attention_available():
        pytest.skip("AITER is not installed")
    lens = [256, 256, 1024]
    cu = torch.tensor([0, 256, 512, 1536], dtype=torch.int32, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(d)
    q, k, v = (torch.randn(sum(lens), 16, d, generator=g, device=DEV).to(torch.bfloat16) for _ in range(3))
    out = varlen_attention(q, k, v, cu, cu, max(lens), max(lens), False, d ** -0.5)
    ref = torch.cat([
        _sdpa_ref(*(t[a:b].transpose(0, 1)[None] for t in (q, k, v)), d ** -0.5)[0].transpose(0, 1)
        for a, b in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True)
    ])
    err, bad = _rel(out, ref)
    assert bad == 0 and err < 1e-2, (err, bad)


# --- pi05 LLM: paged prefix-LM attention ---------------------------------------


class _Cache:
    """One layer of a paged cache in mstar's layout, ``[pages, 2, page, Hkv, D]``."""

    def __init__(self, hkv, d, seed=0, fill=0.0):
        self.hkv, self.d = hkv, d
        self.layer = torch.full((NUM_PAGES, 2, PAGE, hkv, d), fill, dtype=torch.bfloat16, device=DEV)
        self.free = list(range(1, NUM_PAGES))
        random.Random(seed).shuffle(self.free)
        self.gen = torch.Generator(device=DEV).manual_seed(seed)

    def request(self, length):
        pages = [self.free.pop() for _ in range(-(-length // PAGE))]
        k, v = (torch.randn(length, self.hkv, self.d, generator=self.gen, device=DEV).to(torch.bfloat16)
                for _ in range(2))
        pos = torch.arange(length, device=DEV)
        pg = torch.tensor(pages, device=DEV)[pos // PAGE]
        self.layer[pg, 0, pos % PAGE] = k
        self.layer[pg, 1, pos % PAGE] = v
        return pages, k, v


def _layout(cache, mix):
    views, rows = [], []
    for i, (ctx_len, q_len) in enumerate(mix):
        pages, k, v = cache.request(ctx_len + q_len)
        views.append(SequenceView(f"r{i}", "main", pages, ctx_len + q_len, q_len))
        rows.append((q_len, k, v))
    return views, rows


def _paged_ref(q, rows, scale):
    """Non-causal fp32 attention of each row's queries over all of its keys."""
    outs, at = [], 0
    for q_len, k, v in rows:
        g = q.shape[1] // k.shape[1]
        qi = q[at:at + q_len].transpose(0, 1)[None]
        ki, vi = (t.repeat_interleave(g, dim=1).transpose(0, 1)[None] for t in (k, v))
        outs.append(_sdpa_ref(qi, ki, vi, scale)[0].transpose(0, 1))
        at += q_len
    return torch.cat(outs)


def _ctx(views, lease=None):
    ctx = StepContext(
        request_ids=[v.request_id for v in views], graph_walk="w", slot=0, capture=False,
        is_preplan=False, slot_lease=lease,
    )
    ctx.plan_results = {"kv": KVPlanOutputs(
        {"main": KVPlanOutput(cpu_indptrs=build_paged_indptrs(views, PAGE), views=views)},
    )}
    return ctx


def _manager(hq=8, hkv=1, d=256):
    from mstar.engine.resources.attn.aiter import AiterPagedAttentionManager, flashinfer_rocm_unavailable_reason

    if flashinfer_rocm_unavailable_reason() is not None:
        pytest.skip("AMD's FlashInfer build is not installed")
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=hkv, head_dim=d, num_qo_heads=hq,
        max_seq_len=2048, max_num_pages=NUM_PAGES, page_size=PAGE,
    )
    return AiterPagedAttentionManager("kv", DEV, torch.bfloat16, cfg)


PI05_MIXES = {
    "prefill": [(0, n) for n in PI05_PREFIX[:1]],
    "prefill-bs4": [(0, n) for n in PI05_PREFIX],
    "action": [(PI05_PREFIX[0], PI05_HORIZON)],
    "action-bs4": [(n, PI05_HORIZON) for n in PI05_PREFIX],
}


@pytest.mark.parametrize("mix", list(PI05_MIXES))
@pytest.mark.parametrize("fill", [0.0, float("nan")])
def test_pi05_prefix_lm_paged_eager(mix, fill):
    """``fill`` is what the pages held before: a non-finite page tail must not leak in."""
    cache = _Cache(1, 256, fill=fill)
    views, rows = _layout(cache, PI05_MIXES[mix])
    m = _manager()
    m.plan(AttentionStep(causal=False), _ctx(views))
    q = torch.randn(sum(r[0] for r in rows), 8, 256, device=DEV).to(torch.bfloat16)
    err, bad = _rel(m.run(q, "main", cache.layer), _paged_ref(q, rows, 256 ** -0.5))
    assert bad == 0 and err < 1e-2, (err, bad)


def _capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


@pytest.mark.parametrize("walk", ["prefill", "action"])
@pytest.mark.parametrize("bs", [1, 2, 4])
def test_pi05_prefix_lm_paged_captured(walk, bs):
    """Captured like pi05's buckets: packed prefill at 1024/1800 tokens with
    zero-length padding rows, action_gen at bs x 50 with padding rows of 50."""
    if walk == "prefill" and bs == 4:
        pytest.skip("four pi05 prefixes don't fit the largest prefill bucket")
    m = _manager()
    tokens = (1024 if bs == 1 else 1800) if walk == "prefill" else bs * PI05_HORIZON
    lease = SlotLease(slot=0, bucket=BucketKey(walk, bs, tokens))
    q_buf = torch.zeros(tokens, 8, 256, dtype=torch.bfloat16, device=DEV)

    def pad(views):
        span = 0 if walk == "prefill" else PI05_HORIZON
        return views + [
            SequenceView(f"pad{i}", "main", [SINK_PAGE] * -(-span // PAGE), span, span)
            for i in range(bs - len(views))
        ]

    capture_cache = _Cache(1, 256, seed=7)
    if walk == "prefill":
        cap_views, _ = _layout(capture_cache, [(0, tokens // bs)] * bs)
    else:
        cap_views, _ = _layout(capture_cache, [(PI05_PREFIX[0], PI05_HORIZON)] * bs)
    m.plan(AttentionStep(causal=False), _ctx(pad(cap_views), lease))
    cache = _Cache(1, 256, seed=1, fill=float("nan"))
    graph, out = _capture(lambda: m.run(q_buf, "main", cache.layer))

    prefix = PI05_PREFIX[:bs]
    mixes = ([[(0, n) for n in prefix], [(0, 600)]] if walk == "prefill"
             else [[(n, PI05_HORIZON) for n in prefix], [(700, PI05_HORIZON)]])
    for i, mix in enumerate(mixes):
        cache.free = list(range(1, NUM_PAGES))
        random.Random(i).shuffle(cache.free)
        views, rows = _layout(cache, mix)
        m.plan(AttentionStep(causal=False), _ctx(pad(views), lease))
        n = sum(r[0] for r in rows)
        q = torch.randn(n, 8, 256, device=DEV).to(torch.bfloat16)
        q_buf.zero_()
        q_buf[:n].copy_(q)
        graph.replay()
        torch.cuda.synchronize()
        err, bad = _rel(out, _paged_ref(q, rows, 256 ** -0.5))
        assert bad == 0 and err < 1e-2, (i, err, bad)


# what ParallelAttention passes on every call, which picks the on-the-fly kernel
_LAYER_ROPE_KWARGS = dict(
    rope_theta=10000.0, rope_scale=1.0, low_freq_factor=1.0, high_freq_factor=1.0, old_context_len=8192,
)


@pytest.mark.parametrize("layer_kwargs", [False, True])
@pytest.mark.parametrize("tokens", [50, 865])
def test_pi05_rope_full_head_dim(tokens, layer_kwargs):
    """Gemma's RoPE: theta 1e4 over all 256 dims, 8 query heads and one KV head,
    on the strided q/k views of the fused QKV projection."""
    from mstar.engine.resources.position.config import PositionConfig
    from mstar.engine.resources.position.manager import RopeManager

    heads, kv_heads, d = 8, 1, 256
    config = PositionConfig(kv_cache="kv", rope_theta=10000.0)
    manager = RopeManager(config, DEV, max_seq_len=2048, head_dim=d)
    pos = torch.arange(tokens, device=DEV) + (865 if tokens == 50 else 0)
    manager._current_pos_ids["main"] = pos
    qkv = torch.randn(tokens, (heads + 2 * kv_heads) * d, device=DEV, dtype=torch.bfloat16)
    q = qkv[:, : heads * d].view(tokens, heads, d)
    k = qkv[:, heads * d:(heads + kv_heads) * d].view(tokens, kv_heads, d)
    inv = 1.0 / 10000.0 ** (torch.arange(0, d, 2, dtype=torch.float64, device=DEV) / d)
    ang = pos.double()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]

    def ref(x):
        a, b = x.double()[..., : d // 2], x.double()[..., d // 2:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)

    ref_q, ref_k = ref(q), ref(k)
    out_q, out_k = manager.apply_qk(q, k, "main", **(_LAYER_ROPE_KWARGS if layer_kwargs else {}))
    for out, r in ((out_q, ref_q), (out_k, ref_k)):
        err, bad = _rel(out, r)
        assert bad == 0 and err < 6e-3, (err, bad)


# --- patch embeddings: MIOpen convs under the worker's environment ---------------

_CONV_CHECK = r"""
import json, sys
import torch
cin, cout, kernel, shape, dtype = json.loads(sys.argv[1])
dtype = getattr(torch, dtype)
g = torch.Generator().manual_seed(0)
conv = (torch.nn.Conv2d if len(kernel) == 2 else torch.nn.Conv3d)(cin, cout, kernel, stride=kernel)
with torch.no_grad():
    for p in conv.parameters():
        p.copy_(torch.randn(p.shape, generator=g) * 0.05)
x = torch.randn(shape, generator=g)
conv = conv.cuda().to(dtype)
x = x.cuda().to(dtype)
with torch.no_grad():
    out = conv(x).double()
    # stride == kernel, so the conv is one GEMM over non-overlapping patches
    w, b = conv.weight.double(), conv.bias.double()
    p = x.double()
    for dim, k in enumerate(kernel):
        p = p.unfold(2 + dim, k, k)
    p = p.permute(0, *range(2, 2 + len(kernel)), 1, *range(2 + len(kernel), 2 + 2 * len(kernel)))
    ref = p.reshape(*p.shape[: 1 + len(kernel)], -1) @ w.reshape(cout, -1).T + b
    ref = ref.permute(0, -1, *range(1, 1 + len(kernel)))
bad = int((~torch.isfinite(out)).sum())
err = ((out.nan_to_num() - ref).norm() / ref.norm()).item()
print(json.dumps({"err": err, "bad": bad}))
"""

_PATCH_CONVS = {
    # SigLIP so400m/14 at 224, 3 cameras, fp32
    "siglip-fp32": (3, 1152, [14, 14], [3, 3, 224, 224], "float32", 1e-5),
    # V-JEPA2 tubelet 2x16x16 over 8 frames at 256, bf16
    "vjepa2-bf16": (3, 1024, [2, 16, 16], [1, 3, 8, 256, 256], "bfloat16", 1e-2),
}


@pytest.mark.parametrize("worker_env", [False, True])
@pytest.mark.parametrize("case", list(_PATCH_CONVS))
def test_patch_embed_conv(case, worker_env):
    """The patch-embedding conv, in a fresh process with or without the env the
    worker sets before its first HIP call (MIOpen reads it at init)."""
    import json
    import os
    import subprocess
    import sys

    from mstar.platforms import _built_for

    cin, cout, kernel, shape, dtype, tol = _PATCH_CONVS[case]
    env = dict(os.environ)
    for key in _built_for().process_env_defaults():
        env.pop(key, None)
    if worker_env:
        env.update(_built_for().process_env_defaults())
    res = subprocess.run(
        [sys.executable, "-c", _CONV_CHECK, json.dumps([cin, cout, kernel, shape, dtype])],
        env=env, capture_output=True, text=True, timeout=300, check=True,
    )
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["bad"] == 0 and out["err"] < tol, out
