"""CPU tests for the MLA KV layout: storage shape, latent write/read, chunk pointers,
page copies, and the manager's plan-driven write path."""
import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    KVStep,
    Segment,
    StepContext,
    SubmoduleStep,
)
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache

PAGE = 4


def _cfg(pages=6):
    return KVConfig(
        num_layers=2, num_kv_heads=96, head_dim=999, max_seq_len=pages * PAGE,
        max_num_pages=pages, page_size=PAGE, num_qo_heads=96,
        layout=KVLayout.MLA, kv_lora_rank=8, qk_rope_head_dim=4,
    )


def test_config_and_storage_shape():
    cfg = _cfg()
    assert cfg.num_kv_heads == 1 and cfg.head_dim == 12 and cfg.latent_dim == 12
    cfg.shard(4)
    assert cfg.num_kv_heads == 1 and cfg.num_qo_heads == 24  # latent replicated, q heads split
    cache = KVCache(cfg, torch.device("cpu"), torch.float32)
    assert cache.tensor.shape == (2, 6, PAGE, 12)
    assert cache.layer_view(1).shape == (6, PAGE, 12)
    # strided ckv / kpe views keep a contiguous last dim and explicit page/token strides
    layer = cache.layer_view(0)
    ckv, kpe = layer[..., :8], layer[..., 8:]
    assert ckv.stride(-1) == 1 and kpe.stride(-1) == 1 and ckv.stride(1) == 12 and kpe.stride(0) == PAGE * 12


def test_write_read_and_pointers():
    cfg = _cfg()
    cache = KVCache(cfg, torch.device("cpu"), torch.float32)
    latent = torch.randn(5, 12)
    pages = torch.tensor([2, 2, 2, 2, 3])
    offs = torch.tensor([0, 1, 2, 3, 0])
    cache.write_tokens(1, latent, None, pages, offs)
    back = cache.read_tokens(1, pages, offs)
    assert torch.equal(back, latent)
    assert torch.equal(cache.layer_view(1)[2], latent[:4]) and torch.equal(cache.layer_view(1)[3, 0], latent[4])
    ptrs, nbytes = cache.chunk_ptrs(1, 2, 1, 3)
    assert len(ptrs) == 1 and nbytes == 2 * 12 * 4
    expect = cache.data_ptr() + (1 * cache.tensor.stride(0) + 2 * cache.tensor.stride(1) + 1 * 12) * 4
    assert ptrs[0] == expect
    assert torch.equal(cache.chunk_view(1, 2, 1, 3), latent[1:3])
    cache.copy_pages([2], [5])
    assert torch.equal(cache.layer_view(1)[5], latent[:4])


class _StubTransfer:
    def __init__(self, *a, **k):
        pass

    def cleanup(self):
        pass


def test_manager_plans_and_writes_latents(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    cfg = _cfg(pages=8)
    mgr = manager_mod.KVManager(cfg=cfg, name="mla_kv", joint_comm_group=None, transfer_engine_info=None,
                                device=torch.device("cpu"), dtype=torch.float32)
    mgr.ingest_request("a")
    mgr.ingest_request("b")
    step = SubmoduleStep(segments=[Segment("a", "main", 5), Segment("b", "main", 3)], steps={"mla_kv": KVStep()})
    ctx = StepContext(request_ids=["a", "b"], graph_walk="prefill", slot=0, capture=False)
    step.set_ctx(ctx)
    assert mgr.admit(step.get("mla_kv"), ctx).ok
    out = mgr.plan(step.get("mla_kv"), ctx)["main"]
    assert out.cpu_indptrs.qo_indptr.tolist() == [0, 5, 8]
    latents = torch.randn(8, 12)
    mgr.set_default_layer_idx(1)
    mgr.write_kv(latents)  # v=None: MLA writes one latent per token
    assert torch.equal(mgr.read_kv(), latents)
    mgr.commit(step.get("mla_kv"), ctx)
    # decode step appends one token per request after the resident 5 / 3
    step2 = SubmoduleStep(segments=[Segment("a", "main", 1), Segment("b", "main", 1)], steps={"mla_kv": KVStep()})
    ctx2 = StepContext(request_ids=["a", "b"], graph_walk="decode", slot=0, capture=False)
    step2.set_ctx(ctx2)
    assert mgr.admit(step2.get("mla_kv"), ctx2).ok
    out2 = mgr.plan(step2.get("mla_kv"), ctx2)["main"]
    assert out2.is_decode and out2.cpu_indptrs.paged_kv_last_page_len.tolist() == [2, 4]
    # the MLA attention spec resolves against this cache and needs sm_scale
    spec = AttentionConfig(kv_cache="mla_kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=192 ** -0.5)
    assert spec.backend is AttnBackend.FLASHINFER_MLA and KVSpec is not None
