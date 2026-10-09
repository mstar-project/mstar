"""When the FlashInfer MLA kernel serves a plan, and what happens when it cannot."""
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.attn import flashinfer_mla as fm
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttnBackend
from mstar.engine.resources.kv.config import KVLayout, PagedKVConfig
from mstar.engine.resources.step import SlotLease


@pytest.fixture
def fake_flashinfer(monkeypatch):
    fake = types.ModuleType("flashinfer")
    monkeypatch.setitem(sys.modules, "flashinfer", fake)
    fm._flashinfer_at_least.cache_clear()
    yield fake
    fm._flashinfer_at_least.cache_clear()


@pytest.mark.parametrize("version, kpe0", [
    ("0.6.14", False), ("0.6.17", False), ("0.6.18", True), ("0.6.18.post1", True), ("0.7.0", True),
])
def test_kpe0_needs_flashinfer_0_6_18(fake_flashinfer, version, kpe0):
    fake_flashinfer.__version__ = version
    assert fm.flashinfer_mla_supports(512, 0) is kpe0
    assert fm.flashinfer_mla_supports(512, 64)
    assert not fm.flashinfer_mla_supports(256, 0)


def test_fallback_refuses_a_capture_lease():
    kv = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=32, max_seq_len=64, page_size=16,
                  max_num_pages=8, num_qo_heads=2, layout=KVLayout.MLA,
                  kv_lora_rank=32, qk_rope_head_dim=0)
    attn = fm.FlashInferMLAManager("kv", torch.device("cpu"), torch.float32, kv, sm_scale=0.1)
    assert not attn.uses_kernel
    with pytest.raises(RuntimeError, match="CUDA graph"):
        attn._cg_wrapper(SlotLease(slot=0, bucket=None), "main", num_rows=1)


def test_build_takes_the_mla_backend_on_a_paged_cache():
    """``AttentionManager.build`` checks each backend against its KV config type
    before dispatching; the MLA backend has to be in that table."""
    kv = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=32, max_seq_len=64, page_size=16,
                       max_num_pages=8, num_qo_heads=2, layout=KVLayout.MLA,
                       kv_lora_rank=32, qk_rope_head_dim=0)
    info = SimpleNamespace(
        dependency=lambda key: SimpleNamespace(config=kv),
        device=torch.device("cpu"), kv_dtype=torch.float32, joint_comm_group=None,
    )
    spec = AttentionSpec(
        resource_key="attn", nodes={"LLM"},
        config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=0.1),
    )
    assert isinstance(AttentionManager.build(spec, info), fm.FlashInferMLAManager)


def test_eager_wrappers_are_per_slot():
    """A plan stages into a pinned buffer the wrapper holds and copies it to the
    device without blocking: one wrapper per slot keeps plan(N+1) off step N's
    queued copy, as FlashInferManager does since #243."""
    kv = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=32, max_seq_len=64, page_size=16,
                       max_num_pages=8, num_qo_heads=2, layout=KVLayout.MLA,
                       kv_lora_rank=32, qk_rope_head_dim=0)
    attn = fm.FlashInferMLAManager("kv", torch.device("cpu"), torch.float32, kv, sm_scale=0.1)
    assert attn.force_double_buffer
    w0 = attn._eager_wrapper("main", 0)
    assert attn._eager_wrapper("main", 1) is not w0
    assert attn._eager_wrapper("main", 0) is w0


@pytest.mark.parametrize("captured", [True, False])
def test_flashmla_serves_eager_decode_plans_only(captured, monkeypatch):
    """A captured bucket replays the kernel it was captured with and pads with rows of
    no query, which FlashMLA cannot plan; switching per plan replayed stale buffers."""
    kv = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=576, max_seq_len=128, page_size=64,
                       max_num_pages=8, num_qo_heads=2, layout=KVLayout.MLA,
                       kv_lora_rank=512, qk_rope_head_dim=64)
    attn = fm.FlashInferMLAManager("kv", torch.device("cpu"), torch.float32, kv, sm_scale=0.1)
    attn._flashmla = True
    chose = []

    def wrapper(*args):
        chose.append(args[-1])
        return SimpleNamespace(plan=lambda **kw: None)

    monkeypatch.setattr(attn, "_cg_wrapper", wrapper)
    monkeypatch.setattr(attn, "_eager_wrapper", wrapper)
    indptrs = SimpleNamespace(qo_indptr=torch.arange(3), to_kwargs_dict=dict)
    ctx = SimpleNamespace(
        slot_lease=SlotLease(slot=0, bucket=None) if captured else None, slot=0,
        is_preplan=False, plan_results={"kv": {"main": SimpleNamespace(cpu_indptrs=indptrs)}})
    attn.plan(SimpleNamespace(causal=True, context_only=False, segments=None), ctx)
    assert chose == [not captured]


def test_mla_attention_needs_the_query_heads():
    # defaulted to the one latent KV head, the kernel planned a single head; a cache only
    # stored (an index-key cache) needs none
    kv = PagedKVConfig(num_layers=1, num_kv_heads=1, head_dim=32, max_seq_len=64, page_size=16,
                       max_num_pages=8, layout=KVLayout.MLA, kv_lora_rank=32, qk_rope_head_dim=0)
    with pytest.raises(ValueError, match="num_qo_heads"):
        fm.FlashInferMLAManager("kv", torch.device("cpu"), torch.float32, kv, sm_scale=0.1)


def test_only_the_mla_backend_takes_sm_scale():
    kv = PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=96, max_seq_len=64, page_size=16,
                       max_num_pages=8)
    info = SimpleNamespace(
        dependency=lambda key: SimpleNamespace(config=kv),
        device=torch.device("cpu"), kv_dtype=torch.float32, joint_comm_group=None,
    )
    spec = AttentionSpec(
        resource_key="attn", nodes={"LLM"},
        config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER, sm_scale=0.1),
    )
    with pytest.raises(ValueError, match="ignores sm_scale"):
        AttentionManager.build(spec, info)
