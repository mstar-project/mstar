"""KV cache quantization: config, YAML, the write and read paths, scales, and what refuses it."""
import types

import pytest
import torch

from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttnBackend,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.attn.cross import CrossAttentionManager
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.calibrate import main as calibrate_main
from mstar.engine.resources.kv.calibrate import merge_observed
from mstar.engine.resources.kv.config import (
    KVSpec,
    PagedKVConfig,
    RingKVConfig,
    RingKVLayerConfig,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.quant import (
    KVObserver,
    KVQuantConfig,
    dequantize_kv,
    load_kv_scales,
    local_heads,
    quantize_kv,
)
from mstar.engine.resources.kv.transfer import ShmKVTransferInfo, check_storage

FP8 = torch.float8_e4m3fn
UNIT = KVQuantConfig("fp8_e4m3")


def _cfg(**overrides) -> PagedKVConfig:
    cfg = dict(
        num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=64,
        max_num_pages=4, page_size=8,
    )
    cfg.update(overrides)
    return PagedKVConfig(**cfg)


def _write_scales(path, layers=2, heads=2, k=None, v=None):
    from safetensors.torch import save_file

    k = k if k is not None else torch.rand(layers, heads) + 0.01
    v = v if v is not None else torch.rand(layers, heads) + 0.01
    save_file({"k_scale": k.float(), "v_scale": v.float()}, str(path))
    return str(path)


# ---- config and YAML ----

def test_config_validation():
    with pytest.raises(ValueError, match="unknown KV quant format"):
        KVQuantConfig("int8")
    with pytest.raises(NotImplementedError, match="per-token scales"):
        KVQuantConfig("fp8_e4m3", scales="dynamic_head")
    with pytest.raises(ValueError, match="need scales_path"):
        KVQuantConfig("fp8_e4m3", scales="static")
    with pytest.raises(ValueError, match="only for static"):
        KVQuantConfig("fp8_e4m3", scales_path="x.safetensors")
    with pytest.raises(ValueError, match="records amax only"):
        KVQuantConfig("fp8_e4m3", scale_rule="mse")
    assert UNIT.value_dtype is FP8


def test_yaml_sets_changes_and_removes_quant():
    spec = KVSpec(resource_key="kv", nodes={"n"}, config=_cfg())
    spec.apply_yaml_overrides(quant={"format": "fp8_e4m3", "scales": "unit"})
    assert spec.config.quant == UNIT
    # an omitted key keeps it
    spec.apply_yaml_overrides(max_num_pages=8)
    assert spec.config.quant == UNIT
    spec.apply_yaml_overrides(quant=None)
    assert spec.config.quant is None
    with pytest.raises(ValueError, match=r"resources\.kv\.quant: unknown key"):
        spec.apply_yaml_overrides(quant={"format": "fp8_e4m3", "scale": "unit"})
    with pytest.raises(ValueError, match=r"resources\.kv\.quant: format is required"):
        spec.apply_yaml_overrides(quant={"scales": "unit"})


def test_ring_kv_refuses_quant():
    ring = RingKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, tokens_per_frame=4,
        layers=(RingKVLayerConfig(2, 1, 1),),
    )
    with pytest.raises(ValueError, match="ring KV cannot be quantized"):
        ring.apply_yaml_overrides(quant={"format": "fp8_e4m3"})


# ---- the cache ----

@pytest.mark.parametrize("static", [False, True])
def test_cache_stores_fp8_and_reads_back_dequantized(tmp_path, static):
    quant, scales = UNIT, None
    if static:
        quant = KVQuantConfig("fp8_e4m3", "static", scales_path=_write_scales(tmp_path / "s.safetensors"))
        scales = load_kv_scales(quant, 2, 2, 0, 1, torch.device("cpu"))
    cache = KVCache(_cfg(), torch.device("cpu"), torch.bfloat16, quant=quant, scales=scales)
    assert cache.dtype is FP8 and cache.value_dtype is torch.bfloat16
    k, v = torch.randn(5, 2, 8, dtype=torch.bfloat16), torch.randn(5, 2, 8, dtype=torch.bfloat16)
    pages, slots = torch.tensor([1, 1, 2, 2, 3]), torch.tensor([0, 1, 0, 7, 3])
    cache.write_tokens(1, k, v, pages, slots)
    ks = scales.k[1] if static else None
    vs = scales.v[1] if static else None
    stored = cache.tensor.view(torch.uint8)[1, pages, :, slots]
    assert torch.equal(stored[:, 0], quantize_kv(k, ks, FP8).view(torch.uint8))
    assert torch.equal(stored[:, 1], quantize_kv(v, vs, FP8).view(torch.uint8))
    out = cache.read_tokens(1, pages, slots)
    assert out.dtype is torch.bfloat16
    assert torch.equal(out[:, 0], dequantize_kv(quantize_kv(k, ks, FP8), ks, torch.bfloat16))
    # fp8 e4m3 keeps 3 mantissa bits
    torch.testing.assert_close(out[:, 0].float(), k.float(), rtol=0.07, atol=0.02 * float(ks.max()) if static else 0.02)


def test_unit_scale_saturates_instead_of_overflowing():
    x = torch.tensor([[[1e4, -1e4, 1.0]]])
    assert quantize_kv(x, None, FP8).float().tolist() == [[[448.0, -448.0, 1.0]]]


def test_storage_id_names_format_and_scales(tmp_path):
    plain = KVCache(_cfg(), torch.device("cpu"), torch.bfloat16)
    unit = KVCache(_cfg(), torch.device("cpu"), torch.bfloat16, quant=UNIT)
    path = _write_scales(tmp_path / "a.safetensors")
    static = KVQuantConfig("fp8_e4m3", "static", scales_path=path)
    a = KVCache(_cfg(), torch.device("cpu"), torch.bfloat16, quant=static,
                scales=load_kv_scales(static, 2, 2, 0, 1, torch.device("cpu")))
    ids = {plain.storage_id, unit.storage_id, a.storage_id}
    assert len(ids) == 3
    # same path, different contents: a different id
    _write_scales(tmp_path / "a.safetensors")
    assert static.digest() != a.storage_id.split("/")[1]


def test_static_needs_scales_and_only_static_takes_them():
    with pytest.raises(ValueError, match="static KV quantization needs scales"):
        KVCache(_cfg(), torch.device("cpu"), torch.bfloat16, quant=UNIT,
                scales=types.SimpleNamespace(k=None, v=None))


# ---- scales ----

def test_local_heads_match_the_qkv_split():
    assert local_heads(8, 1, 4) == slice(2, 4)
    # fewer heads than ranks: rank r holds head r // replicas
    assert [local_heads(2, r, 4) for r in range(4)] == [slice(0, 1), slice(0, 1), slice(1, 2), slice(1, 2)]


def test_load_kv_scales_slices_by_rank_and_checks_shape(tmp_path):
    k = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    path = _write_scales(tmp_path / "s.safetensors", heads=4, k=k, v=k * 2)
    quant = KVQuantConfig("fp8_e4m3", "static", scales_path=path)
    scales = load_kv_scales(quant, 2, 4, 1, 2, torch.device("cpu"))
    assert scales.k.tolist() == [[3, 4], [7, 8]] and scales.v.tolist() == [[6, 8], [14, 16]]
    with pytest.raises(ValueError, match=r"k_scale is \(2, 4\), the cache has 3 layers"):
        load_kv_scales(quant, 3, 4, 0, 1, torch.device("cpu"))
    bad = KVQuantConfig("fp8_e4m3", "static", scales_path=_write_scales(
        tmp_path / "bad.safetensors", k=torch.zeros(2, 2)))
    with pytest.raises(ValueError, match="non-positive"):
        load_kv_scales(bad, 2, 2, 0, 1, torch.device("cpu"))


def test_observer_and_calibrate_round_trip(tmp_path):
    obs = tmp_path / "obs"
    # TP2 over 4 heads: each rank observes its two
    for rank in range(2):
        o = KVObserver(str(obs), "kv", 2, 2, 4, rank, 2, torch.device("cpu"))
        k = torch.zeros(3, 2, 8)
        k[1, 0, 2] = -10.0 * (rank + 1)
        k[0, 1, 0] = 5.0
        o.observe(0, k, k * 0.5)
        o.observe(1, k * 2, k)
        o.save(force=True)
    amax = merge_observed(str(obs), "kv")
    assert amax[0, 0].tolist() == [10, 5, 20, 5]
    assert amax[1, 1].tolist() == [10, 5, 20, 5]
    out = tmp_path / "scales.safetensors"
    calibrate_main([str(obs), "--out", str(out), "--headroom", "1.5"])
    quant = KVQuantConfig("fp8_e4m3", "static", scales_path=str(out))
    scales = load_kv_scales(quant, 2, 4, 0, 1, torch.device("cpu"))
    torch.testing.assert_close(scales.k[0], 1.5 * torch.tensor([10, 5, 20, 5]) / 448)


# ---- moving pages ----

def test_transfer_refuses_pages_in_another_encoding():
    cache = KVCache(_cfg(), torch.device("cpu"), torch.bfloat16, quant=UNIT)
    info = ShmKVTransferInfo(path="", page_indices=(), layout=cache.layout, storage="torch.bfloat16")
    with pytest.raises(ValueError, match="stored as 'torch.bfloat16'"):
        check_storage(info, cache)
    check_storage(ShmKVTransferInfo("", (), cache.layout, storage=cache.storage_id), cache)


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass


def _manager(monkeypatch, quant):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    return KVManager(
        cfg=_cfg(quant=quant), name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.bfloat16,
    )


def test_prefix_root_folds_in_the_quant(monkeypatch, tmp_path):
    static = KVQuantConfig("fp8_e4m3", "static", scales_path=_write_scales(tmp_path / "s.safetensors"))
    prints = {_manager(monkeypatch, q).fingerprint() for q in (None, UNIT, static)}
    assert len(prints) == 3


def test_offload_pool_holds_fp8_pages(monkeypatch):
    mgr = _manager(monkeypatch, UNIT)
    assert mgr.kv_cache.tensor.dtype is FP8
    assert mgr.kv_cache.nbytes == _manager(monkeypatch, None).kv_cache.nbytes // 2


# ---- what reads it ----

def _info(kv_spec):
    return EngineResourceInfo(
        device=torch.device("cpu"), default_dtype=torch.bfloat16,
        dependencies={kv_spec.resource_key: kv_spec},
    )


def test_flashinfer_plans_fp8_pages_and_folds_static_scales(tmp_path):
    path = _write_scales(tmp_path / "s.safetensors", k=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    kv = KVSpec(resource_key="kv", nodes={"n"}, config=_cfg(
        num_qo_heads=4, quant=KVQuantConfig("fp8_e4m3", "static", scales_path=path),
    ))
    attn = AttentionManager.build(AttentionSpec(
        resource_key="attn", nodes={"n"},
        config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER),
    ), _info(kv))
    assert attn._kv_dtype is FP8 and attn._dtype is torch.bfloat16
    # two q heads per KV head
    assert attn._layer_scales[1].k_qo[:, 0].tolist() == [3.0, 3.0, 4.0, 4.0]
    assert attn._layer_scales[1].k.tolist() == [3.0, 4.0]


@pytest.mark.parametrize("backend", [AttnBackend.FLEX, AttnBackend.XPU_PAGED])
def test_backends_without_an_fp8_read_refuse(backend):
    kv = KVSpec(resource_key="kv", nodes={"n"}, config=_cfg(quant=UNIT))
    spec = AttentionSpec(resource_key="attn", nodes={"n"}, config=AttentionConfig(kv_cache="kv", backend=backend))
    with pytest.raises((ValueError, TypeError), match="reads no quantized KV|requires a RingKVConfig"):
        AttentionManager.build(spec, _info(kv))


def test_cross_attention_refuses_a_quantized_context():
    kv = KVSpec(resource_key="kv", nodes={"n"}, config=_cfg(quant=UNIT))
    spec = CrossAttentionSpec(resource_key="cross", nodes={"n"}, config=CrossAttentionConfig(kv_cache="kv"))
    with pytest.raises(ValueError, match="cross attention 'cross' reads no quantized KV"):
        CrossAttentionManager.build(spec, _info(kv))


# ---- CUDA: the Triton write and FlashInfer read ----

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("static", [False, True])
def test_triton_scatter_matches_the_reference(tmp_path, static):
    dev = torch.device("cuda")
    cfg = _cfg(num_kv_heads=4, head_dim=128, max_num_pages=16, page_size=16)
    quant, scales = UNIT, None
    if static:
        quant = KVQuantConfig("fp8_e4m3", "static", scales_path=_write_scales(tmp_path / "s.safetensors", heads=4))
        scales = load_kv_scales(quant, 2, 4, 0, 1, dev)
    cache = KVCache(cfg, dev, torch.bfloat16, quant=quant, scales=scales)
    n = 100
    k = torch.randn(n, 4, 128, device=dev, dtype=torch.bfloat16) * 3
    v = torch.randn(n, 4, 128, device=dev, dtype=torch.bfloat16) * 3
    flat = torch.randperm(16 * 16, device=dev)[:n]
    pages, slots = flat // 16, flat % 16
    cache.write_tokens(0, k, v, pages, slots)
    ks = scales.k[0] if static else None
    vs = scales.v[0] if static else None
    got = cache.tensor.view(torch.uint8)[0, pages, :, slots]
    assert torch.equal(got[:, 0], quantize_kv(k, ks, FP8).view(torch.uint8))
    assert torch.equal(got[:, 1], quantize_kv(v, vs, FP8).view(torch.uint8))


def _reference_attention(q, k, v, causal):
    """q [Tq, Hq, D], k/v [Tk, Hk, D]: fp32 attention, queries at the end."""
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1).float()
    v = v.repeat_interleave(group, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) / q.shape[-1] ** 0.5
    if causal:
        tq, tk = q.shape[0], k.shape[0]
        mask = torch.arange(tk, device=q.device)[None] > (torch.arange(tq, device=q.device)[:, None] + tk - tq)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


@cuda
@pytest.mark.parametrize("static", [False, True])
def test_flashinfer_decode_reads_fp8_pages(tmp_path, static):
    """Decode keeps bf16 q: equal to attention over the dequantized pages, within bf16 rounding."""
    from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper

    dev = torch.device("cuda")
    heads, qo_heads, dim, page = 2, 8, 128, 16
    cfg = _cfg(num_kv_heads=heads, num_qo_heads=qo_heads, head_dim=dim, max_num_pages=8, page_size=page)
    quant, scales = UNIT, None
    if static:
        quant = KVQuantConfig("fp8_e4m3", "static", scales_path=_write_scales(
            tmp_path / "s.safetensors", k=torch.tensor([[0.02, 0.05]] * 2), v=torch.tensor([[0.01, 0.03]] * 2)))
        scales = load_kv_scales(quant, 2, heads, 0, 1, dev)
    cache = KVCache(cfg, dev, torch.bfloat16, quant=quant, scales=scales)
    length = 40
    k = torch.randn(length, heads, dim, device=dev, dtype=torch.bfloat16) * 2
    v = torch.randn(length, heads, dim, device=dev, dtype=torch.bfloat16) * 2
    pages = torch.tensor([3, 5, 6], device=dev)
    pos = torch.arange(length, device=dev)
    cache.write_tokens(1, k, v, pages[pos // page], pos % page)

    n_q = 1
    q = torch.randn(n_q, qo_heads, dim, device=dev, dtype=torch.bfloat16)
    kv_kwargs = dict(
        paged_kv_indptr=torch.tensor([0, 3], dtype=torch.int32),
        paged_kv_indices=pages.int().cpu(),
        paged_kv_last_page_len=torch.tensor([length - 2 * page], dtype=torch.int32),
    )
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    common = dict(num_qo_heads=qo_heads, num_kv_heads=heads, head_dim=dim, page_size=page, device=dev)
    wrapper = FlashInferDecodeWrapper(workspace, **common)
    wrapper.plan(dtype=torch.bfloat16, kv_dtype=FP8, **kv_kwargs)

    from mstar.engine.resources.kv.quant import layer_kv_scales

    manager = types.SimpleNamespace(
        _current_plan_states={"main": wrapper}, _layer_scales=layer_kv_scales(scales, qo_heads),
    )
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    out = FlashInferManager._attend(manager, q, "main", cache.layer_view(1), 1)
    assert out.dtype is torch.bfloat16

    stored = cache.read_tokens(1, pages[pos // page], pos % page)
    expected = _reference_attention(q, stored[:, 0], stored[:, 1], causal=True)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test_observer_ignores_non_finite_and_resets(tmp_path):
    o = KVObserver(str(tmp_path), "kv", 1, 2, 2, 0, 1, torch.device("cpu"))
    # a head whose write holds NaN or inf is skipped for that write
    o.observe(0, torch.tensor([[[float("nan"), 1.0], [2.0, 3.0]]]), torch.ones(1, 2, 2))
    o.observe(0, torch.tensor([[[0.5, 1.0], [float("inf"), 9.0]]]), torch.ones(1, 2, 2))
    assert o.amax[0, 0].tolist() == [1.0, 3.0]
    o.reset()
    assert not o.amax.any()


def test_copy_pages_moves_fp8_bytes():
    cache = KVCache(_cfg(), torch.device("cpu"), torch.bfloat16, quant=UNIT)
    raw = cache.tensor.view(torch.uint8)
    raw.copy_(torch.randint(0, 120, raw.shape, dtype=torch.uint8))
    before = raw[:, 1].clone()
    cache.copy_pages([1], [3])
    assert torch.equal(raw[:, 3], before)


@cuda
def test_offload_reload_keeps_fp8_pages(monkeypatch):
    from mstar.engine.resources.kv.config import KVStep
    from mstar.engine.resources.step import Segment, StepContext

    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    mgr = KVManager(
        cfg=_cfg(quant=UNIT, cpu_offload_pages=4), name="kv", joint_comm_group=None,
        transfer_engine_info=None, device=torch.device("cuda"), dtype=torch.bfloat16,
    )
    mgr.ingest_request("r0")
    step = KVStep(segments=(Segment("r0", "main", 16),))
    ctx = StepContext(request_ids=("r0",), graph_walk="walk", slot=0, capture=False)
    assert mgr.admit(step, ctx).ok
    mgr.plan(step, ctx)
    mgr.commit(step, ctx)
    pages = list(mgr._streams["r0"]["main"].page_indices)
    raw = mgr.kv_cache.tensor.view(torch.uint8)
    raw[:, pages] = torch.randint(0, 120, raw[:, pages].shape, dtype=torch.uint8, device="cuda")
    expected = raw[:, pages].clone()
    assert mgr.offload("r0") == len(pages)
    raw.fill_(0)
    assert mgr.reload("r0")
    assert torch.equal(raw[:, mgr._streams["r0"]["main"].page_indices], expected)


# ---- prefill on FA3 ----

def test_fp8_kv_refuses_a_pinned_non_fa3_backend():
    kv = KVSpec(resource_key="kv", nodes={"n"}, config=_cfg(quant=UNIT))
    spec = AttentionSpec(resource_key="attn", nodes={"n"}, config=AttentionConfig(
        kv_cache="kv", backend=AttnBackend.FLASHINFER, flashinfer_backend="fa2"))
    with pytest.raises(ValueError, match="fp8 KV runs on FA3"):
        AttentionManager.build(spec, _info(kv))


@cuda
@pytest.mark.parametrize("static", [False, True])
def test_fa3_fp8_prefill_reads_fp8_pages(tmp_path, static):
    """Prefill over fp8 pages quantizes q and runs FA3; close to the reference within fp8 rounding."""
    from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper
    from mstar.engine.resources.kv.quant import layer_kv_scales

    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FA3 needs sm90")
    dev = torch.device("cuda")
    heads, qo_heads, dim, page = 2, 8, 128, 16
    cfg = _cfg(num_kv_heads=heads, num_qo_heads=qo_heads, head_dim=dim, max_num_pages=8, page_size=page)
    quant, scales = UNIT, None
    if static:
        quant = KVQuantConfig("fp8_e4m3", "static", scales_path=_write_scales(
            tmp_path / "s.safetensors", k=torch.tensor([[0.02, 0.05]] * 2), v=torch.tensor([[0.01, 0.03]] * 2)))
        scales = load_kv_scales(quant, 2, heads, 0, 1, dev)
    cache = KVCache(cfg, dev, torch.bfloat16, quant=quant, scales=scales)
    length, n_q = 40, 7
    k = torch.randn(length, heads, dim, device=dev, dtype=torch.bfloat16) * 2
    v = torch.randn(length, heads, dim, device=dev, dtype=torch.bfloat16) * 2
    pages = torch.tensor([3, 5, 6], device=dev)
    pos = torch.arange(length, device=dev)
    cache.write_tokens(1, k, v, pages[pos // page], pos % page)
    q = torch.randn(n_q, qo_heads, dim, device=dev, dtype=torch.bfloat16) * 0.5

    wrapper = FlashInferPrefillWrapper(
        torch.empty(128 << 20, dtype=torch.uint8, device=dev), num_qo_heads=qo_heads, num_kv_heads=heads,
        head_dim=dim, page_size=page, device=dev,
    )
    wrapper.plan(
        qo_indptr=torch.tensor([0, n_q], dtype=torch.int32), paged_kv_indptr=torch.tensor([0, 3], dtype=torch.int32),
        paged_kv_indices=pages.int().cpu(), paged_kv_last_page_len=torch.tensor([length - 2 * page], dtype=torch.int32),
        causal=True, dtype=torch.bfloat16, kv_dtype=FP8,
    )
    assert wrapper.attn_wrapper._backend == "fa3"
    layer = layer_kv_scales(scales, qo_heads)[1] if static else None
    out = wrapper.run(q, cache.layer_view(1), layer)
    assert out.dtype is torch.bfloat16

    stored = cache.read_tokens(1, pages[pos // page], pos % page)
    expected = _reference_attention(q, stored[:, 0], stored[:, 1], causal=True)
    # static scales store K near 448, where q at full fp8 range lost 40% here
    assert ((out.float() - expected).norm() / expected.norm()).item() < 0.05
