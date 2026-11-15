"""Block-causal attention over a paged KV cache: the per-block layout, the wrapper
eager and under CUDA-graph replay across layouts, and the resource's refusals. The
reference is SDPA with a dense block-causal mask over each row's whole cache."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.attn.block_causal import (
    PagedBlockCausalWrapper,
    max_blocks,
    paged_block_causal_layout,
)
from mstar.engine.resources.attn.config import AttentionConfig, AttnBackend, BlockCausalAttentionSpec
from mstar.engine.resources.attn.workspace import Workspace

HEADS, HEAD_DIM, BLOCK, PAGE = 4, 64, 5, 8
DTYPE = torch.bfloat16
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer's kernels need a GPU")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_each_block_reads_its_rows_prefix_pages():
    # a row with 10 cached and 7 new, on pages 3, 9, 4; a row with nothing new; 3 new on page 7
    qo, kv, idx, last = paged_block_causal_layout(
        [0, 7, 7, 10], [0, 3, 4, 5], np.array([3, 9, 4, 1, 7]), [17, 5, 3], BLOCK, PAGE,
    )
    assert qo == [0, 5, 7, 10]  # blocks at 10..15, 15..17, 0..3
    assert kv == [0, 2, 5, 6]  # prefixes of 15, 17 and 3 keys
    assert idx.tolist() == [3, 9, 3, 9, 4, 7]
    assert last == [7, 1, 3]


def test_a_row_starting_inside_a_block_is_refused():
    with pytest.raises(ValueError, match="inside a block"):
        paged_block_causal_layout([0, 3], [0, 1], np.array([0]), [7], BLOCK, PAGE)


def test_nothing_new_is_no_block():
    qo, kv, idx, last = paged_block_causal_layout([0, 0], [0, 1], np.array([2]), [8], BLOCK, PAGE)
    assert qo == [0] and kv == [0] and len(idx) == 0
    assert last == []


def test_block_ceiling_counts_a_partial_block_per_row():
    assert max_blocks(12, 3, BLOCK) == 5
    # rows of 1, 1 and 10 tokens: 1 + 1 + 2 blocks
    assert sum(-(-n // BLOCK) for n in (1, 1, 10)) <= max_blocks(12, 3, BLOCK)


def test_the_spec_runs_on_flashinfer_only_and_needs_a_block():
    config = AttentionConfig(kv_cache="kv")
    with pytest.raises(ValueError, match="block_size"):
        BlockCausalAttentionSpec(resource_key="a", nodes={"n"}, config=config)
    with pytest.raises(ValueError, match="flashinfer"):
        BlockCausalAttentionSpec(
            resource_key="a", nodes={"n"}, block_size=BLOCK,
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.DENSE),
        )


def _keys(total, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return tuple(torch.randn(total, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE, generator=gen) for _ in range(3))


class _Pool:
    """Rows on shuffled pages of one layer."""

    def __init__(self, rows):
        pages = sum(-(-n // PAGE) for n in rows)
        order = torch.randperm(pages, generator=torch.Generator().manual_seed(1)).tolist()
        self.k = torch.zeros(pages, PAGE, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        self.v = torch.zeros_like(self.k)
        self.row_pages = []
        at = 0
        for n in rows:
            num = -(-n // PAGE)
            self.row_pages.append(order[at : at + num])
            at += num

    def write(self, row, first, k, v):
        for i in range(k.shape[0]):
            f = first + i
            p = self.row_pages[row][f // PAGE]
            self.k[p, f % PAGE] = k[i]
            self.v[p, f % PAGE] = v[i]

    def plan_args(self, spans):
        """A KV plan's rows: spans are (cached before the step, new)."""
        qo, kv, indices, last = [0], [0], [], []
        lens = []
        for row, (before, new) in enumerate(spans):
            end = before + new
            pages = -(-end // PAGE) if end else 1
            qo.append(qo[-1] + new)
            kv.append(kv[-1] + pages)
            indices += self.row_pages[row][:pages]
            last.append(end - (pages - 1) * PAGE if end else 1)
            lens.append(end)
        i32 = dict(dtype=torch.int32)
        return dict(
            qo_indptr=torch.tensor(qo, **i32), paged_kv_indptr=torch.tensor(kv, **i32),
            paged_kv_indices=torch.tensor(indices, **i32), paged_kv_last_page_len=torch.tensor(last, **i32),
            kv_lens=lens, causal=False, dtype=DTYPE,
        )


def _reference(q, k, v, first):
    """New tokens' output: ``q`` ``[n, H, D]`` from frame ``first``, ``k``/``v`` the whole row so far."""
    total = k.shape[0]
    q_block = (first + torch.arange(q.shape[0], device=DEVICE)) // BLOCK
    key_block = torch.arange(total, device=DEVICE) // BLOCK
    mask = key_block[None, :] <= q_block[:, None]
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).float(), k.transpose(0, 1).float(), v.transpose(0, 1).float(),
        attn_mask=mask, scale=HEAD_DIM ** -0.5,
    )
    return out.transpose(0, 1)


def _wrapper(**overrides):
    kwargs = dict(
        block_size=BLOCK, max_seq_len=96,
        workspace_buffer=Workspace(torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE), "test"),
        num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=HEAD_DIM, page_size=PAGE, device=DEVICE,
        sm_scale=HEAD_DIM ** -0.5,
    )
    return PagedBlockCausalWrapper(**(kwargs | overrides))


def _stream_rows(rows):
    """Each row's whole q/k/v and a schedule of (row, first, new) steps."""
    return [_keys(n, seed=10 + i) for i, n in enumerate(rows)]


@cuda
@pytest.mark.parametrize(
    "spans", [[(0, 13)], [(0, 7), (5, 11)], [(10, 3), (0, 1), (15, 25)], [(20, 5), (40, 10)]],
    ids=["one_row", "first_and_later", "mixed", "aligned_prefixes"],
)
def test_eager_matches_the_dense_mask(spans):
    rows = [before + new for before, new in spans]
    data = _stream_rows(rows)
    pool = _Pool(rows)
    for row, (_, k, v) in enumerate(data):
        pool.write(row, 0, k, v)
    wrapper = _wrapper()
    wrapper.plan(**pool.plan_args(spans))
    qs = torch.cat([q[before:] for (q, _, _), (before, _) in zip(data, spans, strict=True)])
    out = wrapper.run(qs, (pool.k, pool.v))
    at = 0
    for (q, k, v), (before, new) in zip(data, spans, strict=True):
        want = _reference(q[before:], k, v, before)
        err = (out[at : at + new].float() - want).abs().max().item()
        assert err < 0.02, f"row at {before}+{new}: {err}"
        at += new


@cuda
def test_graph_replay_matches_across_layouts_and_prefixes():
    """Captured with no prefix, replayed with long ones: the plan's ceilings were primed."""
    rows, max_tokens = 2, 30
    data = _stream_rows([80, 80])
    pool = _Pool([80, 80])
    for row, (_, k, v) in enumerate(data):
        pool.write(row, 0, k, v)
    wrapper = _wrapper(
        batch_size=max_blocks(max_tokens, rows, BLOCK), max_total_tokens=max_tokens,
        max_num_pages=max_blocks(max_tokens, rows, BLOCK) * 12, accelerator_graph=True,
    )
    wrapper.plan(**pool.plan_args([(0, 15), (0, 15)]))
    q_buf = torch.zeros(max_tokens, HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = wrapper.run(q_buf, (pool.k, pool.v))
    torch.cuda.synchronize()
    for spans in ([(0, 15), (0, 15)], [(40, 12), (75, 5)], [(70, 10), (10, 20)], [(0, 1), (5, 2)]):
        wrapper.plan(**pool.plan_args(spans))
        qs = torch.cat([q[b : b + n] for (q, _, _), (b, n) in zip(data, spans, strict=True)])
        q_buf.zero_()
        q_buf[: qs.shape[0]] = qs
        graph.replay()
        torch.cuda.synchronize()
        at = 0
        for (q, k, v), (before, new) in zip(data, spans, strict=True):
            want = _reference(q[before : before + new], k[: before + new], v[: before + new], before)
            err = (out[at : at + new].float() - want).abs().max().item()
            assert err < 0.02, f"{spans}, row at {before}+{new}: {err}"
            at += new


@cuda
def test_too_many_blocks_for_the_bucket_are_refused():
    wrapper = _wrapper(
        batch_size=max_blocks(10, 1, BLOCK), max_total_tokens=10,
        max_num_pages=max_blocks(10, 1, BLOCK) * 12, accelerator_graph=True,
    )
    with pytest.raises(ValueError, match="exceed"):
        wrapper._plan_blocks(list(range(0, 41, 5)), list(range(9)), np.zeros(8, dtype=np.int32), [5] * 8, DTYPE)
