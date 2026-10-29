"""The ragged attention resource's AITER path at the served DiT shapes, on an AMD GPU.

One case per attention the image and video models declare: FLUX.2-klein 4B and 9B
(24 and 32 heads of 128 over 512 text + 4096 image tokens, batches of 1 and 4),
Z-Image-Turbo (30 heads; the refiners' image-only and caption-only spans next to the
main span, three labels in one step) and Wan2.2 at 480x832x33 (24 heads; the two
guidance branches as two segments of 3510 tokens, plus each branch's cross-attention
over 512 text tokens). Every case plans its step through the manager, eagerly and
leased like a captured bucket padded with empty segments, and scores the kernel's
output per segment against fp64 SDPA, next to torch's own bf16 SDPA on the same
inputs, which is what these models served before.

    pytest test/rocm/test_ragged_dit_shapes.py
"""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged import aiter as rocm_ragged
from mstar.engine.resources.attn.ragged.aiter import AiterRaggedCrossManager, AiterRaggedManager
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedCrossAttentionStep,
    cross_label,
)
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available() and current_platform.is_rocm()
        and rocm_ragged.aiter_varlen_unavailable_reason() is None
    ),
    reason="needs a ROCm GPU with AITER's varlen flash attention",
)

DEV = torch.device("cuda")
HEAD_DIM = 128
FLUX_TOKENS = 512 + 4096  # the 1024x1024 bucket: text + packed image tokens
ZIMAGE_IMAGE, ZIMAGE_CAPTION = 4096, 64
WAN_TOKENS, WAN_TEXT = 3510, 512  # 480x832x33 -> (9, 30, 52) latents, patch 1x2x2


def _sdpa_ref(q, k, v):
    """Math SDPA in fp64 over [N, H, D] tensors (importing mstar.engine sets fp32
    matmuls to the reduced-precision path, which the math backend would use)."""
    q, k, v = (x.transpose(0, 1).unsqueeze(0).double() for x in (q, k, v))
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v)
    return out.squeeze(0).transpose(0, 1)


def _sdpa_bf16(q, k, v):
    """torch's own bf16 SDPA, the kernel the models served through before."""
    q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1)


def _rel(out, ref):
    return ((out.double() - ref).norm() / ref.norm()).item()


def _inputs(q_tokens, kv_tokens, heads, peak, seed):
    """Unit-RMS q/k/v per head like the QK-normed DiTs, q scaled by ``peak``."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(q_tokens, heads, HEAD_DIM, device=DEV, generator=g) * peak
    k = torch.randn(kv_tokens, heads, HEAD_DIM, device=DEV, generator=g)
    v = torch.randn(kv_tokens, heads, HEAD_DIM, device=DEV, generator=g)
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _config(heads, segments_per_request=1):
    return RaggedAttentionConfig(
        num_qo_heads=heads, num_kv_heads=heads, head_dim=HEAD_DIM,
        max_segments_per_request=segments_per_request, dtype=torch.bfloat16,
    )


def _ctx(request_ids, lease=None):
    return StepContext(request_ids=list(request_ids), graph_walk="w", slot=0, capture=False, slot_lease=lease)


def _lease(bs, num_tokens):
    return SlotLease(slot=0, bucket=BucketKey(graph_walk="w", bs=bs, num_tokens=num_tokens))


def _check_segments(out, q, k, v, spans_q, spans_k):
    """Score each segment of the packed output against fp64 SDPA; the kernel may not
    be much worse than torch's bf16 SDPA on the same inputs (bf16 output rounding
    alone is ~2e-3 relative)."""
    assert torch.isfinite(out[: sum(spans_q)]).all()
    q0 = k0 = 0
    for nq, nk in zip(spans_q, spans_k, strict=True):
        qs, ks, vs = q[q0:q0 + nq], k[k0:k0 + nk], v[k0:k0 + nk]
        ref = _sdpa_ref(qs, ks, vs)
        err, base = _rel(out[q0:q0 + nq], ref), _rel(_sdpa_bf16(qs, ks, vs), ref)
        assert err <= max(1.5 * base, 4e-3), f"aiter {err:.3e} vs torch bf16 {base:.3e} on segment of {nq}"
        q0, k0 = q0 + nq, k0 + nk


@pytest.mark.parametrize("name,heads,rows,peak", [
    ("flux2_klein_4b", 24, 1, 1.0),
    ("flux2_klein_4b", 24, 1, 3.0),
    ("flux2_klein_4b", 24, 4, 1.0),
    ("flux2_klein_9b", 32, 1, 1.0),
    ("flux2_klein_9b", 32, 2, 3.0),
])
def test_joint_attention_matches_sdpa(name, heads, rows, peak):
    """One segment per request over [txt | img], eager and under a leased plan."""
    m = AiterRaggedManager(device=DEV, dtype=torch.bfloat16, config=_config(heads))
    rids = [f"r{i}" for i in range(rows)]
    segments = tuple(Segment(rid, "main", FLUX_TOKENS) for rid in rids)
    q, k, v = _inputs(rows * FLUX_TOKENS, rows * FLUX_TOKENS, heads, peak, seed=heads * 100 + rows * 10 + int(peak))
    spans = [FLUX_TOKENS] * rows

    m.plan(AttentionStep(segments=segments, causal=False), _ctx(rids))
    eager = m.run(q, k, v, "main")
    _check_segments(eager, q, k, v, spans, spans)

    # the leased layout is the bucket's: one more row than live, padded with an empty segment
    bucket = _lease(rows + 1, (rows + 1) * FLUX_TOKENS)
    m.plan(AttentionStep(segments=segments, causal=False), _ctx(rids, bucket))
    pad = FLUX_TOKENS
    qb, kb, vb = (torch.cat([t, torch.randn(pad, heads, HEAD_DIM, device=DEV).bfloat16()]) for t in (q, k, v))
    leased = m.run(qb, kb, vb, "main")
    _check_segments(leased, qb, kb, vb, spans, spans)
    # the bucket's token ceiling sizes the kernel's grid, which may change its split and so the
    # reduction order, but not the real rows' result beyond rounding
    n = rows * FLUX_TOKENS
    assert _rel(leased[:n], eager.double()) <= 4e-3


def test_each_layers_output_survives_the_next_layer():
    """The output is the kernel's own allocation per call: under a captured bucket a
    layer's result is not overwritten by the next layer's attention."""
    m = AiterRaggedManager(device=DEV, dtype=torch.bfloat16, config=_config(24))
    segments = (Segment("r0", "main", FLUX_TOKENS),)
    m.plan(AttentionStep(segments=segments, causal=False), _ctx(["r0"], _lease(1, FLUX_TOKENS)))
    q1, k1, v1 = _inputs(FLUX_TOKENS, FLUX_TOKENS, 24, 1.0, seed=1)
    q2, k2, v2 = _inputs(FLUX_TOKENS, FLUX_TOKENS, 24, 1.0, seed=2)
    first = m.run(q1, k1, v1, "main")
    kept = first.clone()
    second = m.run(q2, k2, v2, "main")
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, kept)
    assert not torch.equal(first, second)


def test_z_image_spans_plan_one_layout_per_label():
    """Z-Image's step: the refiners attend the image tokens and the caption alone, the
    main blocks both; one manager, three labels, each its own layout."""
    heads = 30
    m = AiterRaggedManager(device=DEV, dtype=torch.bfloat16, config=_config(heads))
    total = ZIMAGE_IMAGE + ZIMAGE_CAPTION
    segments = (
        Segment("r0", "image", ZIMAGE_IMAGE), Segment("r0", "caption", ZIMAGE_CAPTION), Segment("r0", "main", total),
    )
    m.plan(AttentionStep(segments=segments, causal=False), _ctx(["r0"], _lease(1, total)))
    for label, n in (("image", ZIMAGE_IMAGE), ("caption", ZIMAGE_CAPTION), ("main", total)):
        q, k, v = _inputs(n, n, heads, 1.0, seed=n)
        _check_segments(m.run(q, k, v, label), q, k, v, [n], [n])


def test_wan_cfg_branches_attend_within_themselves():
    """Wan2.2's batch-2 CFG forward: two segments of one request under one label."""
    heads = 24
    m = AiterRaggedManager(device=DEV, dtype=torch.bfloat16, config=_config(heads, segments_per_request=2))
    segments = (Segment("r0", "video", WAN_TOKENS), Segment("r0", "video", WAN_TOKENS))
    m.plan(AttentionStep(segments=segments, causal=False), _ctx(["r0"]))
    q, k, v = _inputs(2 * WAN_TOKENS, 2 * WAN_TOKENS, heads, 1.0, seed=33)
    _check_segments(m.run(q, k, v, "video"), q, k, v, [WAN_TOKENS] * 2, [WAN_TOKENS] * 2)


def test_wan_cross_attention_reads_each_branchs_text():
    """Each guidance branch's video tokens over that branch's 512 text tokens."""
    heads = 24
    m = AiterRaggedCrossManager(device=DEV, dtype=torch.bfloat16, config=_config(heads, segments_per_request=2))
    step = RaggedCrossAttentionStep(
        segments=(
            Segment("r0", "video", WAN_TOKENS), Segment("r0", "video", WAN_TOKENS),
            Segment("r0", "text", WAN_TEXT), Segment("r0", "text", WAN_TEXT),
        ),
        pairs=(("video", "text"),),
    )
    m.plan(step, _ctx(["r0"]))
    q, k, v = _inputs(2 * WAN_TOKENS, 2 * WAN_TEXT, heads, 1.0, seed=81)
    out = m.run(q, k, v, cross_label("video", "text"))
    _check_segments(out, q, k, v, [WAN_TOKENS] * 2, [WAN_TEXT] * 2)
