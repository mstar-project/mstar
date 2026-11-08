"""MLA's captured one-token decode on trtllm-gen (``decode_fmha`` on ``FLASHINFER_MLA``): the
block table it reads 64-token sub-pages from, which captures take it, and on Blackwell that it
gives the paged MLA kernel's output."""
import numpy as np
import pytest
import torch

from mstar.engine.resources.attn import flashinfer_mla
from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager, expand_mla_block_table
from mstar.engine.resources.kv.config import KVLayout, PagedKVConfig
from mstar.engine.resources.step import BucketKey, SlotLease


def test_each_page_becomes_its_sub_pages():
    table = np.full((3, 8), -1, dtype=np.int32)
    lens = np.zeros(3, dtype=np.int32)
    expand_mla_block_table(np.array([0, 1, 3, 3]), np.array([4, 0, 2]), np.array([5, 128, 9]), 128, table, lens)
    assert table[0, :2].tolist() == [8, 9] and (table[0, 2:] == -1).all()
    assert table[1, :4].tolist() == [0, 1, 4, 5]
    # a row without pages (padding) reads one token of whatever its row holds; its output is dropped
    assert lens.tolist() == [5, 256, 1]


def _manager(monkeypatch, why):
    monkeypatch.setattr(flashinfer_mla, "fmha_decode_applies", lambda device: True)
    monkeypatch.setattr(FlashInferMLAManager, "uses_kernel", property(lambda self: True))
    probes = []
    monkeypatch.setattr(flashinfer_mla, "mla_fmha_matches", lambda *a: probes.append(a) or why)

    class Fake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFmha(Fake):
        pass

    monkeypatch.setattr(flashinfer_mla, "FlashInferMLAWrapper", Fake)
    monkeypatch.setattr(flashinfer_mla, "FlashInferMLAFmhaDecodeWrapper", FakeFmha)
    kv = PagedKVConfig(num_layers=2, num_kv_heads=1, head_dim=576, max_seq_len=2048, num_qo_heads=16,
                       layout=KVLayout.MLA, kv_lora_rank=512, qk_rope_head_dim=64)
    manager = FlashInferMLAManager("kv", torch.device("cpu"), torch.bfloat16, kv, sm_scale=0.07,
                                   decode_fmha=True)
    return manager, FakeFmha, probes


def test_one_token_decode_captures_take_the_fmha_wrapper(monkeypatch):
    manager, fmha, probes = _manager(monkeypatch, None)
    decode = manager._cg_wrapper(SlotLease(slot=0, bucket=BucketKey("decode", 8, 8)), "main", 8)
    assert isinstance(decode, fmha)
    assert decode.kwargs["batch_size"] == 8 and decode.kwargs["page_size"] == 128
    assert decode.kwargs["max_seq_len"] == 2048 and decode.kwargs["sm_scale"] == 0.07
    prefill = manager._cg_wrapper(SlotLease(slot=0, bucket=BucketKey("prefill", 4, 256)), "main", 4)
    assert not isinstance(prefill, fmha)
    packed = manager._cg_wrapper(SlotLease(slot=0, bucket=BucketKey("prefill", 4, 4)), "main", 4, True)
    assert not isinstance(packed, fmha)
    assert probes and probes[0][1:] == (16, 512, 64, 128, 0.07)


def test_a_failed_probe_keeps_the_paged_kernel(monkeypatch):
    manager, fmha, probes = _manager(monkeypatch, "relative L2 0.9")
    for bs in (1, 8):
        wrapper = manager._cg_wrapper(SlotLease(slot=0, bucket=BucketKey("decode", bs, bs)), "main", bs)
        assert not isinstance(wrapper, fmha)
    assert len(probes) == 1


def _blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.get_device_capability()[0] == 10


@pytest.mark.skipif(not _blackwell(), reason="trtllm-gen MLA decode runs on Blackwell (sm_100 / sm_103)")
@pytest.mark.parametrize("heads,rope,taken", [(16, 64, True), (32, 0, False)])
def test_trtllm_gen_matches_the_paged_kernel_where_it_runs(heads, rope, taken):
    """GLM-5.3 at TP4 (16 heads, 512 + 64) takes it; GLM-5.3-Flash's NoPE latent at TP2 does not (trtllm-gen
    decodes a rope-less latent only in its sparse mode), so it keeps the paged kernel. Both as on 4x B300."""
    why = flashinfer_mla.mla_fmha_matches(torch.device("cuda"), heads, 512, rope, 128, 192 ** -0.5)
    assert (why is None) is taken, why


def _trtllm_mla_reference(query, kv_cache, workspace_buffer, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim,
                          block_tables, seq_lens, max_seq_len, bmm1_scale, bmm2_scale, **kwargs):
    """The kernel's documented semantics in torch: each row reads its first seq_len tokens through
    its block table of ``kv_cache.shape[1]``-token pages."""
    block = kv_cache.shape[1]
    out = torch.empty(*query.shape[:-1], kv_lora_rank, dtype=query.dtype)
    for b in range(query.shape[0]):
        n = int(seq_lens[b])
        pages = block_tables[b, : -(-n // block)].long()
        kv = kv_cache[pages].reshape(-1, kv_cache.shape[-1])[:n].float()
        scores = torch.einsum("qhd,kd->qhk", query[b].float(), kv) * bmm1_scale
        out[b] = (torch.softmax(scores, dim=-1) @ kv[:, :kv_lora_rank] * bmm2_scale).to(out.dtype)
    return out


def test_the_wrapper_reads_what_the_paged_plan_names(monkeypatch):
    """On host tensors with the kernel replaced by its semantics: the sub-page view, the expanded
    table and the lengths give the paged MLA reference's output (FlashInferMLAWrapper's fallback)."""
    import sys
    import types

    fi = types.ModuleType("flashinfer")
    fi.utils = types.SimpleNamespace(
        _get_trtllm_gen_multi_ctas_kv_counter_buffer=lambda *a: torch.zeros(1),
        get_device_sm_count=lambda device: 1)
    fi.mla = types.SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=_trtllm_mla_reference)
    for name, module in (("flashinfer", fi), ("flashinfer.utils", fi.utils), ("flashinfer.mla", fi.mla)):
        monkeypatch.setitem(sys.modules, name, module)
    from mstar.engine.resources.attn.workspace import Workspace

    workspace = Workspace(torch.empty(0, dtype=torch.uint8), "test")
    gen = torch.Generator().manual_seed(1)
    page, heads, rows = 128, 4, 3
    cache = torch.randn(6, page, 576, generator=gen).to(torch.bfloat16)
    q_nope = torch.randn(rows, heads, 512, generator=gen).to(torch.bfloat16)
    q_pe = torch.randn(rows, heads, 64, generator=gen).to(torch.bfloat16)
    plan = dict(qo_indptr=torch.arange(rows + 1, dtype=torch.int32),
                paged_kv_indptr=torch.tensor([0, 1, 3, 6], dtype=torch.int32),
                paged_kv_indices=torch.tensor([4, 0, 2, 5, 1, 3], dtype=torch.int32),
                paged_kv_last_page_len=torch.tensor([40, 128, 7], dtype=torch.int32))
    ref = flashinfer_mla.FlashInferMLAWrapper(workspace, heads, 512, 64, page, 0.07, device=torch.device("cpu"))
    ref.plan(**plan)
    fmha = flashinfer_mla.FlashInferMLAFmhaDecodeWrapper(workspace, heads, 512, 64, page, 0.07, rows, 3 * page,
                                                         torch.device("cpu"))
    fmha.plan(**plan)
    want, got = ref.run(q_nope, q_pe, cache).float(), fmha.run(q_nope, q_pe, cache).float()
    assert got.shape == want.shape == (rows, heads, 512)
    assert ((got - want).norm() / want.norm()) < 0.01
    with pytest.raises(ValueError, match="one token each"):
        fmha.plan(**{**plan, "qo_indptr": torch.tensor([0, 2, 3, 4], dtype=torch.int32)})
