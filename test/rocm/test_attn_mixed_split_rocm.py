"""A split walk's mixed step on AMD's FlashInfer build, against fp32 attention.

`FlashInferMixedWrapper` plans the one-token rows on the decode wrapper and
the prompt rows on the prefill wrapper, with the backend and decode
tensor-core choice ``flashinfer_kwargs`` gives ROCm (fa2). Checked eagerly and
as the engine runs it: planned at capture, captured, then replanned for other
mixes and replayed.

    pytest test/rocm/test_attn_mixed_split_rocm.py
"""

import math

import pytest
import torch

from mstar.engine.resources.attn.wrappers import FlashInferMixedWrapper
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens
from mstar.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = torch.device("cuda")
QO_HEADS, KV_HEADS, DIM, PAGE = 8, 2, 256, 16
BUCKET_BS, BUCKET_TOKENS, MAX_PROMPTS = 8, 64, 4
NUM_PAGES = 64
# the engine's default workspace (MSTAR_WORKSPACE_BUFFER_MB); fa2's split-KV
# scratch does not fit the 128 MB the NVIDIA test uses
WS = 512 << 20

# (new tokens, cached tokens before this step), real rows only: the bucket's
# other rows are zero-span padding on the sink page. Decode rows, a
# continuing prompt chunk, a fresh prompt and a one-token prompt.
CAPTURE_ROWS = [(1, 37), (1, 5), (9, 20), (1, 64), (6, 0), (1, 0)]
REPLAY_ROWS = [
    [(1, 90), (12, 3), (1, 17), (1, 1), (20, 0), (5, 40), (1, 60)],
    [(30, 0), (1, 120), (1, 7)],
]


def _layout(rows):
    """Host index arrays as the KV plan builds them, padding rows on the sink."""
    qo, kvp, idx, last, lens = [0], [0], [], [], []
    page = SINK_PAGE + 1
    for new, cached in rows:
        n = new + cached
        pages = -(-n // PAGE)
        qo.append(qo[-1] + new)
        idx.extend(range(page, page + pages))
        page += pages
        kvp.append(kvp[-1] + pages)
        last.append((n - 1) % PAGE + 1)
        lens.append(n)
    assert page <= NUM_PAGES and qo[-1] <= BUCKET_TOKENS
    for _ in range(BUCKET_BS - len(rows)):
        qo.append(qo[-1])
        idx.append(SINK_PAGE)
        kvp.append(kvp[-1] + 1)
        last.append(1)
        lens.append(1)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    return dict(
        qo_indptr=i32(qo), paged_kv_indptr=i32(kvp), paged_kv_indices=i32(idx),
        paged_kv_last_page_len=i32(last), kv_lens=i32(lens).as_subclass(HostLens),
    )


def _reference(q, cache, rows, plan):
    """Causal fp32 attention of each real row's new tokens over its pages."""
    out = torch.zeros(q.shape[0], QO_HEADS, DIM, dtype=torch.float32, device=DEV)
    kvp, idx, qo = plan["paged_kv_indptr"], plan["paged_kv_indices"], plan["qo_indptr"]
    for r, (new, cached) in enumerate(rows):
        pages = idx[kvp[r]:kvp[r + 1]].long().to(DEV)
        kv = cache[pages].float()  # [pages, 2, PAGE, KV_HEADS, DIM]
        k = kv[:, 0].reshape(-1, KV_HEADS, DIM)[: new + cached].repeat_interleave(QO_HEADS // KV_HEADS, 1)
        v = kv[:, 1].reshape(-1, KV_HEADS, DIM)[: new + cached].repeat_interleave(QO_HEADS // KV_HEADS, 1)
        qi = q[qo[r]:qo[r + 1]].float()
        scores = torch.einsum("qhd,khd->hqk", qi, k) / math.sqrt(DIM)
        allowed = torch.arange(new + cached, device=DEV)[None, :] <= cached + torch.arange(new, device=DEV)[:, None]
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        out[qo[r]:qo[r + 1]] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    return out


def _check(out, q, cache, rows, plan):
    real = int(plan["qo_indptr"][len(rows)])
    ref = _reference(q, cache, rows, plan)[:real]
    err = (out[:real].float() - ref).abs().max().item()
    assert err <= 2e-2, f"max err {err} over {real} tokens, max |ref| {ref.abs().max().item():.3g}"


def _mixed():
    from mstar.engine.resources.attn.aiter import flashinfer_rocm_unavailable_reason
    from mstar.engine.resources.attn.base import flashinfer_kwargs

    reason = flashinfer_rocm_unavailable_reason()
    if reason is not None:
        pytest.skip(f"AMD's FlashInfer build is not importable: {reason}")
    kv = PagedKVConfig(
        num_layers=1, num_kv_heads=KV_HEADS, head_dim=DIM, num_qo_heads=QO_HEADS,
        max_seq_len=2048, max_num_pages=NUM_PAGES, page_size=PAGE,
    )
    platform = flashinfer_kwargs(kv, "auto")
    return FlashInferMixedWrapper(
        decode_workspace=torch.empty(WS, dtype=torch.uint8, device=DEV),
        prefill_workspace=torch.empty(WS, dtype=torch.uint8, device=DEV),
        num_qo_heads=QO_HEADS, num_kv_heads=KV_HEADS, head_dim=DIM, page_size=PAGE,
        batch_size=BUCKET_BS, max_total_tokens=BUCKET_TOKENS,
        max_num_pages=BUCKET_BS * 16, max_prefill_rows=MAX_PROMPTS, device=DEV,
        backend=platform["backend"], use_tensor_cores=platform["decode_tensor_cores"],
    )


def _cache(seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(NUM_PAGES, 2, PAGE, KV_HEADS, DIM, generator=gen, device=DEV).to(torch.bfloat16)


def test_eager_mixed_step_matches_fp32_attention():
    mixed = _mixed()
    cache = _cache(0)
    q = torch.randn(BUCKET_TOKENS, QO_HEADS, DIM, device=DEV).to(torch.bfloat16)
    plan = _layout(CAPTURE_ROWS)
    mixed.plan(causal=True, dtype=torch.bfloat16, **plan)
    out = mixed.run(q, cache)
    _check(out, q, cache, CAPTURE_ROWS, plan)
    assert torch.equal(mixed._qo_indptr_buf.cpu(), plan["qo_indptr"])
    # a second plan of the same layout on the same buffers gives the same answer
    mixed.plan(causal=True, dtype=torch.bfloat16, **plan)
    assert torch.equal(mixed.run(q, cache), out)


def test_captured_mixed_step_replays_other_mixes():
    mixed = _mixed()
    cache = _cache(1)
    q_buf = torch.zeros(BUCKET_TOKENS, QO_HEADS, DIM, dtype=torch.bfloat16, device=DEV)
    mixed.plan(causal=True, dtype=torch.bfloat16, **_layout(CAPTURE_ROWS))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            mixed.run(q_buf, cache)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mixed.run(q_buf, cache)

    for i, rows in enumerate(REPLAY_ROWS):
        plan = _layout(rows)
        mixed.plan(causal=True, dtype=torch.bfloat16, **plan)
        q = torch.randn(BUCKET_TOKENS, QO_HEADS, DIM, generator=torch.Generator(device=DEV).manual_seed(i),
                        device=DEV).to(torch.bfloat16)
        q_buf.copy_(q)
        graph.replay()
        torch.cuda.synchronize()
        _check(out, q, cache, rows, plan)
