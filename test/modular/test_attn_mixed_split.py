"""`FlashInferMixedWrapper` attends as the packed prefill wrapper does.

A mixed step plans its one-token rows on the decode wrapper and its prompt
rows on the prefill wrapper. Against the prefill wrapper over every row of
the same bucket: the same output at every real token, and the same full
qo_indptr for `select_last_hidden`.
"""

from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.attn.wrappers import (
    FlashInferMixedWrapper,
    FlashInferPrefillWrapper,
)
from mstar.engine.resources.kv.plan import SINK_PAGE, HostLens
from mstar.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or current_platform.is_rocm(),
    reason="needs an NVIDIA GPU (test/rocm/test_attn_mixed_split_rocm.py covers AMD)",
)

QO_HEADS, KV_HEADS, DIM, PAGE = 8, 2, 256, 16
BUCKET_BS, BUCKET_TOKENS, MAX_PROMPTS = 8, 64, 4
WS = 128 << 20

# (new tokens, cached tokens before this step): decode rows, a continuing
# prompt chunk, a fresh prompt, a one-token prompt, then padding rows
ROWS = [(1, 37), (1, 5), (9, 20), (1, 64), (6, 0), (1, 0)]


def _layout():
    """Host index arrays as the KV plan builds them, padding rows on the sink."""
    qo, kvp, idx, last, lens = [0], [0], [], [], []
    page = SINK_PAGE + 1
    for new, cached in ROWS:
        n = new + cached
        pages = -(-n // PAGE)
        qo.append(qo[-1] + new)
        idx.extend(range(page, page + pages))
        page += pages
        kvp.append(kvp[-1] + pages)
        last.append((n - 1) % PAGE + 1)
        lens.append(n)
    for _ in range(BUCKET_BS - len(ROWS)):  # zero-span padding on the sink page
        qo.append(qo[-1])
        idx.append(SINK_PAGE)
        kvp.append(kvp[-1] + 1)
        last.append(1)
        lens.append(1)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    return dict(
        qo_indptr=i32(qo), paged_kv_indptr=i32(kvp), paged_kv_indices=i32(idx),
        paged_kv_last_page_len=i32(last), kv_lens=i32(lens).as_subclass(HostLens),
    ), page


def _kw(device):
    return dict(
        num_qo_heads=QO_HEADS, num_kv_heads=KV_HEADS, head_dim=DIM, page_size=PAGE,
        max_num_pages=BUCKET_BS * 16, device=device,
    )


@requires_gpu
def test_mixed_split_attends_as_the_packed_prefill_does():
    device = torch.device("cuda")
    plan, num_pages = _layout()
    torch.manual_seed(0)
    cache = torch.randn(num_pages, 2, PAGE, KV_HEADS, DIM, device=device, dtype=torch.bfloat16)
    q = torch.randn(BUCKET_TOKENS, QO_HEADS, DIM, device=device, dtype=torch.bfloat16)
    real = int(plan["qo_indptr"][len(ROWS)])

    # eager, over the real rows only: what the packed path attends
    n = len(ROWS)
    kvp = plan["paged_kv_indptr"]
    ref = FlashInferPrefillWrapper(
        workspace_buffer=torch.empty(WS, dtype=torch.uint8, device=device), **_kw(device),
    )
    ref.plan(
        causal=True, dtype=torch.bfloat16,
        qo_indptr=plan["qo_indptr"][: n + 1].clone(),
        paged_kv_indptr=kvp[: n + 1].clone(),
        paged_kv_indices=plan["paged_kv_indices"][: int(kvp[n])].clone(),
        paged_kv_last_page_len=plan["paged_kv_last_page_len"][:n].clone(),
    )
    ref_out = ref.run(q[:real], cache)

    mixed = FlashInferMixedWrapper(
        decode_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
        prefill_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
        batch_size=BUCKET_BS, max_total_tokens=BUCKET_TOKENS,
        max_prefill_rows=MAX_PROMPTS, **_kw(device),
    )
    mixed.plan(causal=True, dtype=torch.bfloat16, **plan)
    out = mixed.run(q, cache)

    err = (out[:real].float() - ref_out[:real].float()).abs().max().item()
    print(f"max err {err:.3g} over {real} tokens, max |ref| {ref_out.float().abs().max().item():.3g}")
    assert err <= 2e-2, f"max err {err}"
    assert torch.equal(mixed._qo_indptr_buf.cpu(), plan["qo_indptr"])

    # a second plan of the same layout on the same buffers gives the same answer
    mixed.plan(causal=True, dtype=torch.bfloat16, **plan)
    assert torch.equal(mixed.run(q, cache)[:real], out[:real])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="plan() records a CUDA event before the row check",
)
def test_too_many_prompt_rows_for_the_bucket_is_refused():
    plan, _ = _layout()
    mixed = FlashInferMixedWrapper.__new__(FlashInferMixedWrapper)
    mixed._max_prefill_rows, mixed._tokens, mixed._done = 1, BUCKET_TOKENS, None
    mixed._page_size, mixed._sink_page = PAGE, SINK_PAGE
    with pytest.raises(RuntimeError, match="prompt rows"):
        mixed.plan(**plan)
