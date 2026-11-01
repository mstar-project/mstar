"""`FlashInferMixedWrapper` attends as the packed prefill wrapper does.

A mixed step plans its leading run of one-token rows on the decode wrapper
and every other row on the prefill wrapper, both reading the step's queries
in place. Against the prefill wrapper over every row of the same bucket: the
same output at every real token, and the same full qo_indptr for
`select_last_hidden`. The engine orders a mixed step decode first; in any
other order a one-token row after the run is a prefill row.
"""

from __future__ import annotations

import numpy as np
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

# (new tokens, cached tokens before this step), then padding rows
LAYOUTS = {
    # as the engine orders a mixed step: decode rows, then a continuing prompt
    # chunk, a fresh prompt and a one-token prompt
    "decode_first": [(1, 37), (1, 5), (1, 64), (9, 20), (6, 0), (1, 0)],
    # a decode row after a prompt row runs on the prefill wrapper
    "interleaved": [(1, 37), (1, 5), (9, 20), (1, 64), (6, 0), (1, 0)],
    "no_decode": [(9, 20), (6, 0), (1, 0)],
}
ROWS = LAYOUTS["interleaved"]


def _layout(ROWS=ROWS):
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
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("window_left", [-1, 7])
@pytest.mark.parametrize("tensor_cores", [True, False])
def test_mixed_split_attends_as_the_packed_prefill_does(window_left, layout, tensor_cores):
    device = torch.device("cuda")
    ROWS = LAYOUTS[layout]
    plan, num_pages = _layout(ROWS)
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
        max_prefill_rows=MAX_PROMPTS, use_tensor_cores=tensor_cores, **_kw(device),
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
    mixed._page_size, mixed._sink_page = PAGE, SINK_PAGE
    with pytest.raises(RuntimeError, match="prompt rows"):
        mixed.plan(**plan)


@requires_gpu
def test_a_replay_of_another_layout_attends_as_the_packed_prefill_does():
    """Captured over one layout and replayed over another: the run's length,
    the prompt rows and the decode mask all come from the plan."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_pages = 64
    cache = torch.randn(num_pages, 2, PAGE, KV_HEADS, DIM, device=device, dtype=torch.bfloat16)
    q = torch.randn(BUCKET_TOKENS, QO_HEADS, DIM, device=device, dtype=torch.bfloat16)
    mixed = FlashInferMixedWrapper(
        decode_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
        prefill_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
        batch_size=BUCKET_BS, max_total_tokens=BUCKET_TOKENS,
        max_prefill_rows=MAX_PROMPTS, **_kw(device),
    )
    mixed.plan(causal=True, dtype=torch.bfloat16, **_layout(LAYOUTS["no_decode"])[0])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mixed.run(q, cache)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mixed.run(q, cache)

    rows = LAYOUTS["decode_first"]
    plan, _ = _layout(rows)
    mixed.plan(causal=True, dtype=torch.bfloat16, **plan)
    graph.replay()

    n = len(rows)
    real = int(plan["qo_indptr"][n])
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
    err = (out[:real].float() - ref_out.float()).abs().max().item()
    assert err <= 2e-2, f"max err {err}"


def _random_rows(rng, prompts):
    """A decode-first bucket: one-token rows, then ``prompts`` prompt rows."""
    decode = int(rng.integers(0 if prompts else 1, BUCKET_BS - prompts + 1))
    rows = [(1, int(rng.integers(0, 4 * PAGE))) for _ in range(decode)]
    budget = BUCKET_TOKENS - decode
    for _ in range(prompts):
        rows.append((1 + int(rng.integers(1, budget // prompts)), int(rng.integers(0, 2 * PAGE))))
    return rows


@requires_gpu
@pytest.mark.parametrize("tensor_cores", [True, False])
def test_later_plans_match_a_first_plan(tensor_cores):
    """Later plans of a wrapper skip FlashInfer's ``plan`` for its planner;
    each must attend exactly as a new wrapper's first plan of the same step,
    and as the packed prefill wrapper over the real rows."""
    device = torch.device("cuda")
    rng = np.random.default_rng(0)

    def mixed():
        return FlashInferMixedWrapper(
            decode_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
            prefill_workspace=torch.empty(WS, dtype=torch.uint8, device=device),
            batch_size=BUCKET_BS, max_total_tokens=BUCKET_TOKENS,
            max_prefill_rows=MAX_PROMPTS, use_tensor_cores=tensor_cores, **_kw(device),
        )

    reused = mixed()
    torch.manual_seed(0)
    cache = torch.randn(BUCKET_BS * 16, 2, PAGE, KV_HEADS, DIM, device=device, dtype=torch.bfloat16)
    for step in range(16):
        rows = _random_rows(rng, prompts=step % (MAX_PROMPTS + 1))
        plan, _ = _layout(rows)
        n = len(rows)
        q = torch.randn(BUCKET_TOKENS, QO_HEADS, DIM, device=device, dtype=torch.bfloat16)
        first = mixed()
        first.plan(causal=True, dtype=torch.bfloat16, **plan)
        reused.plan(causal=True, dtype=torch.bfloat16, **plan)
        if step > 0:
            assert reused._prefill._planner_args is not None
            # FlashInfer plans the CUDA-core decode itself (`_decode_plan_is_direct`)
            assert (reused._decode._planner_args is not None) == tensor_cores
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
