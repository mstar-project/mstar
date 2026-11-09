"""The MLA attention resource's plan for causal rows off a 64-key boundary (a verify block, a drafter's
block, a prompt chunk the scheduler cut mid-way): single-query rows up to the boundary, then the rest as
one causal row, since FlashInfer 0.7.2's planned MLA backends refuse a causal row whose query/key offset
puts a tile boundary on a KV tile's first key at a head count their query tile doesn't divide (GPU job
1529: "fa3 MLA causal tile boundary is unsupported: ... query=2, key=64, native KV tile=64"; the B300
pre-flight audit: a cut prompt's next chunk at 12 heads a rank)."""
from __future__ import annotations

import torch

from mstar.engine.resources.attn.flashinfer_mla import (
    FlashInferMLAWrapper,
    rowwise_causal_indptrs,
)
from mstar.engine.resources.kv.plan import PagedIndptrs

PAGE = 16


def _indptrs(rows: list[tuple[int, list[int], int]]) -> PagedIndptrs:
    """``rows``: (queries, pages, kv length) per row."""
    qo, kv, pages, last, lens = [0], [0], [], [], []
    for q, row_pages, m in rows:
        qo.append(qo[-1] + q)
        pages.extend(row_pages)
        kv.append(kv[-1] + len(row_pages))
        last.append(m - (len(row_pages) - 1) * PAGE)
        lens.append(m)

    def ints(v):
        return torch.tensor(v, dtype=torch.int32)

    return PagedIndptrs(ints(qo), ints(kv), ints(pages), ints(last), ints(lens))


def test_a_short_causal_row_becomes_one_row_per_query_over_its_own_keys():
    # a 4-query verify block over 70 keys (5 pages), a decode row, a 40-query prefill chunk
    out = rowwise_causal_indptrs(_indptrs([(4, [7, 3, 9, 1, 5], 70), (1, [2, 8], 20), (40, [4, 6, 0], 40)]), PAGE)
    assert out.qo_indptr.tolist() == [0, 1, 2, 3, 4, 5, 45]
    assert out.kv_lens.tolist() == [67, 68, 69, 70, 20, 40]
    kv, pages = out.paged_kv_indptr.tolist(), out.paged_kv_indices.tolist()
    assert [pages[kv[i]:kv[i + 1]] for i in range(6)] == [[7, 3, 9, 1, 5]] * 4 + [[2, 8], [4, 6, 0]]
    assert out.paged_kv_last_page_len.tolist() == [3, 4, 5, 6, 4, 8]
    # an offset that crosses a page: 3 queries over 33 keys, the first two (31, 32 keys) fit two pages
    out = rowwise_causal_indptrs(_indptrs([(3, [1, 2, 3], 33)]), PAGE)
    kv, pages = out.paged_kv_indptr.tolist(), out.paged_kv_indices.tolist()
    assert [pages[kv[i]:kv[i + 1]] for i in range(3)] == [[1, 2], [1, 2], [1, 2, 3]]
    assert out.kv_lens.tolist() == [31, 32, 33]
    # nothing to split: decode rows and rows starting on a 64-key boundary come back as they were
    plain = _indptrs([(1, [1], 5), (17, [2, 3, 4, 5, 6, 7], 81), (40, [4, 6, 0], 40)])
    assert rowwise_causal_indptrs(plain, PAGE) is plain
    # a prompt's next chunk at offset 75 (11 past a boundary): 53 single rows to key 128, then one row
    out = rowwise_causal_indptrs(_indptrs([(500, list(range(36)), 575)]), PAGE)
    assert out.qo_indptr.tolist() == list(range(54)) + [500]
    assert out.kv_lens.tolist() == list(range(76, 129)) + [575]


def test_the_split_rows_attend_as_the_causal_rows_did():
    """Through the resource's reference attention: a step of a 4-query verify block, a decode row, a
    short prefill chunk and a cut prompt's 40-query chunk at offset 50, planned causal as it was and
    split; the outputs and LSEs agree."""
    torch.manual_seed(0)
    heads, rank, rope = 4, 32, 8
    cache = torch.randn(12, PAGE, rank + rope)
    rows = [(4, [7, 3, 9, 1, 5], 70), (1, [2, 8], 20), (6, [4, 6, 0], 40), (40, [10, 11, 1, 2, 8, 6], 90)]
    original = _indptrs(rows)
    tokens = sum(q for q, _, _ in rows)
    q_nope, q_pe = torch.randn(tokens, heads, rank), torch.randn(tokens, heads, rope)
    outs = []
    for indptrs in (original, rowwise_causal_indptrs(original, PAGE)):
        wrapper = FlashInferMLAWrapper(None, num_qo_heads=heads, kv_lora_rank=rank, qk_rope_head_dim=rope,
                                       page_size=PAGE, sm_scale=0.2, device=torch.device("cpu"))
        wrapper.plan(causal=True, dtype=torch.float32, **indptrs.to_kwargs_dict())
        outs.append(wrapper.run(q_nope, q_pe, cache, return_lse=True))
    (o1, l1), (o2, l2) = outs
    torch.testing.assert_close(o2, o1, atol=1e-5, rtol=0)
    torch.testing.assert_close(l2, l1, atol=1e-5, rtol=0)


def test_only_a_head_count_the_query_tile_does_not_divide_splits_and_consumers_see_requests(monkeypatch):
    """The resource splits on the kernel (not its CPU reference, which takes any causal offset),
    for 96 heads (Kimi K3) and not 64 or 128; the consumers (``select_last_hidden``) still read
    the step's requests' query offsets and kv lengths, not the kernel's rows."""
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
    from mstar.engine.resources.kv.config import KVLayout, PagedKVConfig

    def manager(heads):
        cfg = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=576, max_seq_len=64, num_qo_heads=heads,
                            layout=KVLayout.MLA, kv_lora_rank=512, qk_rope_head_dim=64)
        return FlashInferMLAManager("kv", torch.device("cpu"), torch.bfloat16, cfg, sm_scale=0.1)

    monkeypatch.setattr(FlashInferMLAManager, "uses_kernel", property(lambda self: True))
    assert manager(96)._splits_short_rows and not manager(64)._splits_short_rows
    assert not manager(128)._splits_short_rows
    monkeypatch.setattr(FlashInferMLAManager, "uses_kernel", property(lambda self: False))
    assert not manager(96)._splits_short_rows

    original = _indptrs([(4, [7, 3, 9, 1, 5], 70), (1, [2, 8], 20)])
    wrapper = FlashInferMLAWrapper(None, num_qo_heads=4, kv_lora_rank=32, qk_rope_head_dim=8, page_size=PAGE,
                                   sm_scale=0.2, device=torch.device("cpu"))
    wrapper.plan(causal=True, dtype=torch.float32, requests=original,
                 **rowwise_causal_indptrs(original, PAGE).to_kwargs_dict())
    assert wrapper.request_qo_indptr.tolist() == [0, 4, 5] and wrapper.request_kv_len.tolist() == [70, 20]
    assert wrapper._qo_indptr_buf.tolist() == [0, 1, 2, 3, 4, 5]


def _refused(q_lens, kv_lens, heads, tile_kv):
    """FlashInfer 0.7.2's ``_validate_fa_causal_tile_bound`` (mla/_batch_mla/_backends/_fa_common.py) as a
    predicate: its query tile from the batch, then each row's tile boundaries."""
    tile_q = 64 * (2 if sum(q_lens) * heads // len(q_lens) > 64 else 1)
    if tile_q % heads == 0:
        return False
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        for end in range(tile_q, q_len * heads, tile_q):
            query, rem = divmod(end, heads)
            key = kv_len - q_len + query
            if rem and query < q_len and 0 <= key < kv_len and key % tile_kv == 0:
                return True
    return False


def test_no_planned_row_trips_the_tile_rule_at_any_head_count_or_offset():
    """Every causal row the resource plans passes FlashInfer's tile rule: K3.1's 12 heads a rank (TP8)
    up to its 96, odd counts, KV tiles of 16, 32 and 64, each offset in two 64-key periods, short and
    long rows, alone or beside decode rows (which move the batch's query tile). The rows as the
    scheduler hands them (a cut prompt's chunk at offset 75, 12 heads) are refused."""
    assert _refused([500], [575], 12, 64)
    for heads in (12, 24, 48, 96, 7, 10):
        for tile_kv in (16, 32, 64):
            for offset in range(128):
                for q in (*range(2, 34), 63, 64, 65, 129, 500):
                    for decode_rows in (0, 20):
                        rows = [(1, [0], 1)] * decode_rows + [(q, list(range(-(-(offset + q) // PAGE))), offset + q)]
                        out = rowwise_causal_indptrs(_indptrs(rows), PAGE)
                        qo = out.qo_indptr.tolist()
                        q_lens = [b - a for a, b in zip(qo, qo[1:], strict=False)]
                        assert not _refused(q_lens, out.kv_lens.tolist(), heads, tile_kv), (heads, tile_kv, offset, q)
