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

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer needs a GPU")

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
@pytest.mark.parametrize("window_left", [-1, 7])
def test_mixed_split_attends_as_the_packed_prefill_does(window_left):
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
        causal=True, dtype=torch.bfloat16, window_left=window_left,
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
    mixed.plan(causal=True, dtype=torch.bfloat16, window_left=window_left, **plan)
    out = mixed.run(q, cache)

    err = (out[:real].float() - ref_out[:real].float()).abs().max().item()
    print(f"max err {err:.3g} over {real} tokens, max |ref| {ref_out.float().abs().max().item():.3g}")
    assert err <= 2e-2, f"max err {err}"
    assert torch.equal(mixed._qo_indptr_buf.cpu(), plan["qo_indptr"])

    # a second plan of the same layout on the same buffers gives the same answer
    mixed.plan(causal=True, dtype=torch.bfloat16, window_left=window_left, **plan)
    assert torch.equal(mixed.run(q, cache)[:real], out[:real])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="plan() records a CUDA event before the row check",
)
def test_too_many_prompt_rows_for_the_bucket_is_refused():
    plan, _ = _layout()
    mixed = FlashInferMixedWrapper.__new__(FlashInferMixedWrapper)
    mixed._max_prefill_rows, mixed._tokens, mixed._done = 1, BUCKET_TOKENS, None
    mixed._page_size, mixed._sink_page, mixed._bs = PAGE, SINK_PAGE, BUCKET_BS
    with pytest.raises(RuntimeError, match="prompt rows"):
        mixed.plan(**plan)


def _random_layout(rng, prompts):
    """A bucket of one-token rows with ``prompts`` prompt rows among them, then
    zero-span padding rows on the sink page, as the KV plan lays one out."""
    real = int(rng.integers(prompts + 1, BUCKET_BS + 1))
    spans = [1] * real
    budget = BUCKET_TOKENS - real
    for r in rng.choice(real, prompts, replace=False):
        spans[r] = 1 + int(rng.integers(1, budget // prompts + 1))
    qo, kvp, idx, last, lens = [0], [0], [], [], []
    page = SINK_PAGE + 1
    for new in spans:
        n = new + int(rng.integers(0, 3 * PAGE))
        pages = -(-n // PAGE)
        qo.append(qo[-1] + new)
        idx.extend(range(page, page + pages))
        page += pages
        kvp.append(kvp[-1] + pages)
        last.append((n - 1) % PAGE + 1)
        lens.append(n)
    for _ in range(BUCKET_BS - real):
        qo.append(qo[-1])
        idx.append(SINK_PAGE)
        kvp.append(kvp[-1] + 1)
        last.append(1)
        lens.append(1)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32)  # noqa: E731
    return dict(
        qo_indptr=i32(qo), paged_kv_indptr=i32(kvp), paged_kv_indices=i32(idx),
        paged_kv_last_page_len=i32(last), kv_lens=i32(lens).as_subclass(HostLens),
    ), page, real


@requires_gpu
def test_later_plans_match_a_first_plan():
    """Later plans of a wrapper skip FlashInfer's ``plan`` for its planner;
    each must attend exactly as a new wrapper's first plan of the same step,
    and as the packed prefill wrapper over the real rows."""
    import numpy as np

    device = torch.device("cuda")
    rng = np.random.default_rng(0)

    def mixed():
        return FlashInferMixedWrapper(
            decode_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
            prefill_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
            batch_size=BUCKET_BS, max_total_tokens=BUCKET_TOKENS,
            max_prefill_rows=MAX_PROMPTS, **_kw(device),
        )

    reused = mixed()
    torch.manual_seed(0)
    cache = torch.randn(BUCKET_BS * 16, 2, PAGE, KV_HEADS, DIM, device=device, dtype=torch.bfloat16)
    for step in range(16):
        plan, _, n = _random_layout(rng, prompts=step % (MAX_PROMPTS + 1))
        q = torch.randn(BUCKET_TOKENS, QO_HEADS, DIM, device=device, dtype=torch.bfloat16)
        first = mixed()
        first.plan(causal=True, dtype=torch.bfloat16, **plan)
        reused.plan(causal=True, dtype=torch.bfloat16, **plan)
        if step > 0:
            assert reused._prefill._planner_args is not None
            assert reused._decode._planner_args is not None
        for a, b in ((first._prefill, reused._prefill), (first._decode, reused._decode)):
            assert list(a.attn_wrapper._plan_info) == list(b.attn_wrapper._plan_info)
        real = int(plan["qo_indptr"][n])
        out = reused.run(q, cache)
        assert torch.equal(first.run(q, cache)[:real], out[:real])

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
        err = (out[:real].float() - ref.run(q[:real], cache).float()).abs().max().item()
        assert err <= 2e-2, f"step {step}: max err {err}"
