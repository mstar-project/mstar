"""KV cache quantization: config, YAML, the write and read paths, scales, and what refuses it."""
import types

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.calibrate import main as calibrate_main
from mstar.engine.resources.kv.calibrate import merge_observed
from mstar.engine.resources.kv.config import (
    PagedKVConfig,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.quant import (
    KVObserver,
    KVQuantConfig,
    dequantize_kv,
    layer_kv_scales,
    load_kv_scales,
    local_heads,
    quantize_kv,
)

FP8 = torch.float8_e4m3fn
UNIT = KVQuantConfig("fp8_e4m3")
CPU = torch.device("cpu")


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


def _static(path) -> KVQuantConfig:
    return KVQuantConfig("fp8_e4m3", "static", scales_path=str(path))


# ---- config and YAML ----

@pytest.mark.parametrize("static", [False, True])
def test_cache_stores_fp8_and_reads_back_dequantized(tmp_path, static):
    quant, scales = UNIT, None
    if static:
        quant = _static(_write_scales(tmp_path / "s.safetensors"))
        scales = load_kv_scales(quant, 2, 2, 0, 1, CPU)
    cache = KVCache(_cfg(), CPU, torch.bfloat16, quant=quant, scales=scales)
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


def test_a_grouped_pool_stores_fp8():
    cfg = _cfg(num_layers=2, max_seq_len=64)
    cfg.keep_window(8, (1,))
    cache = KVCache(cfg, CPU, torch.bfloat16, quant=UNIT)
    assert cache.dtype is FP8 and cache.layer_view(1).dtype is FP8
    k = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    cache.write_tokens(1, k, k, torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]))
    out = cache.read_tokens(1, torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]))
    torch.testing.assert_close(out[:, 0].float(), k.float(), rtol=0.07, atol=0.02)


def test_storage_id_names_format_and_scales(tmp_path):
    plain = KVCache(_cfg(), CPU, torch.bfloat16)
    unit = KVCache(_cfg(), CPU, torch.bfloat16, quant=UNIT)
    static = _static(_write_scales(tmp_path / "a.safetensors"))
    a = KVCache(_cfg(), CPU, torch.bfloat16, quant=static, scales=load_kv_scales(static, 2, 2, 0, 1, CPU))
    assert len({plain.storage_id, unit.storage_id, a.storage_id}) == 3
    # same path, different contents: a different id
    _write_scales(tmp_path / "a.safetensors")
    assert static.digest() != a.storage_id.split("/")[1]


def test_copy_pages_moves_fp8_bytes():
    cache = KVCache(_cfg(), CPU, torch.bfloat16, quant=UNIT)
    raw = cache.tensor.view(torch.uint8)
    raw.copy_(torch.randint(0, 120, raw.shape, dtype=torch.uint8))
    before = raw[:, 1].clone()
    cache.copy_pages([1], [3])
    assert torch.equal(raw[:, 3], before)


# ---- scales ----

def test_observer_and_calibrate_round_trip(tmp_path):
    obs = tmp_path / "obs"
    # TP2 over 4 heads: each rank observes its two
    for rank in range(2):
        o = KVObserver(str(obs), "kv", 2, 2, 4, rank, 2, CPU)
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
    scales = load_kv_scales(_static(out), 2, 4, 0, 1, CPU)
    torch.testing.assert_close(scales.k[0], 1.5 * torch.tensor([10, 5, 20, 5]) / 448)


def test_replicas_on_the_same_tp_rank_merge_to_the_max(tmp_path):
    """Data-parallel replicas each write their own file for TP rank 0; neither is lost."""
    obs = tmp_path / "obs"
    for replica, value in enumerate((3.0, 7.0)):
        o = KVObserver(str(obs), "kv", 1, 1, 1, 0, 1, CPU)
        # one process stands in for two: a replica's file is its pid's
        o.path = str(obs / f"kv.rank0.{1000 + replica}.pt")
        k = torch.full((1, 1, 2), value)
        o.observe(0, k, k)
        o.save(force=True)
    assert merge_observed(str(obs), "kv")[:, 0, 0].tolist() == [7.0, 7.0]


def test_observer_ignores_non_finite_and_resets(tmp_path):
    o = KVObserver(str(tmp_path), "kv", 1, 2, 2, 0, 1, CPU)
    # a head whose write holds NaN or inf is skipped for that write
    o.observe(0, torch.tensor([[[float("nan"), 1.0], [2.0, 3.0]]]), torch.ones(1, 2, 2))
    o.observe(0, torch.tensor([[[0.5, 1.0], [float("inf"), 9.0]]]), torch.ones(1, 2, 2))
    assert o.amax[0, 0].tolist() == [1.0, 3.0]
    o.reset()
    assert not o.amax.any()


# ---- moving pages, and the manager ----

class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass


def _manager(monkeypatch, quant, tmp_path=None, **cfg):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    scales = None
    if quant is not None and quant.scales == "static":
        scales = load_kv_scales(quant, 2, 2, 0, 1, CPU)
    return KVManager(
        cfg=_cfg(quant=quant, **cfg), name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=CPU, dtype=torch.bfloat16, kv_scales=scales,
    )


def test_prefix_root_folds_in_the_quant(monkeypatch, tmp_path):
    static = _static(_write_scales(tmp_path / "s.safetensors"))
    prints = {_manager(monkeypatch, q).fingerprint() for q in (None, UNIT, static)}
    assert len(prints) == 3


def test_a_quantized_cache_takes_norm_rope_attend_unfused(monkeypatch):
    """The fused norm/RoPE kernels would write raw values into fp8 pages; a quantized
    cache takes the unfused path, whose ``write_kv`` quantizes."""
    from mstar.engine.resources import convenience
    from mstar.utils import attention as attention_utils

    calls = []
    monkeypatch.setattr(attention_utils, "qk_norm_rope_inplace", lambda *a: calls.append("inplace"))
    monkeypatch.setattr(attention_utils, "qk_norm_rope_kv_write", lambda *a: calls.append("fused"))
    kv = types.SimpleNamespace(quantized=True, write_kv=lambda k, v: calls.append("write_kv"), layer_view=lambda: None)
    attn = types.SimpleNamespace(requires_kv_write=True, run=lambda q, **kw: q)
    norm = types.SimpleNamespace(weight=torch.ones(8), variance_epsilon=1e-6)
    q = torch.zeros(1, 2, 8)
    convenience.AttentionCallable(kv, attn).norm_rope_attend(q, q, q, norm, norm, torch.zeros(1), 1e4)
    assert calls == ["inplace", "write_kv"]


# ---- what reads it ----

@pytest.mark.parametrize("static", [False, True])
def test_dense_dequantizes_the_prefix_it_gathers(tmp_path, static):
    from mstar.engine.resources.attn.dense import DenseAttentionManager
    from mstar.engine.resources.attn.ragged.dense_wrappers import DensePrefixedPrefillWrapper

    quant, scales = UNIT, None
    if static:
        quant = _static(_write_scales(tmp_path / "s.safetensors"))
        scales = load_kv_scales(quant, 2, 2, 0, 1, CPU)
    cfg = _cfg(quant=quant)
    cache = KVCache(cfg, CPU, torch.bfloat16, quant=quant, scales=scales)
    k, v = torch.randn(10, 2, 8, dtype=torch.bfloat16), torch.randn(10, 2, 8, dtype=torch.bfloat16)
    pos = torch.arange(10)
    pages = torch.tensor([2, 3])
    cache.write_tokens(1, k, v, pages[pos // 8], pos % 8)
    dense = DenseAttentionManager(
        kv_cache="kv", device=CPU, dtype=torch.bfloat16, kv_config=cfg,
        wrapper_class=DensePrefixedPrefillWrapper, kv_scales=scales,
    )
    segment = types.SimpleNamespace(request_id="r", label="main", generation=0, prefix_len=10, pages=[2, 3])
    k_pref, v_pref = dense._prefix_kv(segment, 1, cache.layer_view(1))
    stored = cache.read_tokens(1, pages[pos // 8], pos % 8)
    assert k_pref.dtype is torch.bfloat16
    assert torch.equal(k_pref, stored[:, 0]) and torch.equal(v_pref, stored[:, 1])


# ---- CUDA: the Triton write, and FlashInfer reading fp8 pages ----

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("static", [False, True])
def test_triton_scatter_matches_the_reference(tmp_path, static):
    dev = torch.device("cuda")
    cfg = _cfg(num_kv_heads=4, head_dim=128, max_num_pages=16, page_size=16)
    quant, scales = UNIT, None
    if static:
        quant = _static(_write_scales(tmp_path / "s.safetensors", heads=4))
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
@pytest.mark.parametrize("kind", ["prefill", "decode_tensor_cores", "decode_cuda_cores"])
def test_flashinfer_reads_fp8_pages(tmp_path, static, kind):
    """q stays in bf16 over fp8 pages, with static scales folded around the kernel: equal to
    fp32 attention over the dequantized pages, within bf16 rounding."""
    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.attn.workspace import Workspace
    from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper, FlashInferPrefillWrapper

    dev = torch.device("cuda")
    heads, qo_heads, dim, page = 2, 8, 128, 16
    cfg = _cfg(num_kv_heads=heads, num_qo_heads=qo_heads, head_dim=dim, max_num_pages=8, page_size=page)
    quant, scales = UNIT, None
    if static:
        quant = _static(_write_scales(
            tmp_path / "s.safetensors", k=torch.tensor([[0.02, 0.05]] * 2), v=torch.tensor([[0.01, 0.03]] * 2)))
        scales = load_kv_scales(quant, 2, heads, 0, 1, dev)
    cache = KVCache(cfg, dev, torch.bfloat16, quant=quant, scales=scales)
    length = 40
    k = torch.randn(length, heads, dim, device=dev, dtype=torch.bfloat16) * 2
    v = torch.randn(length, heads, dim, device=dev, dtype=torch.bfloat16) * 2
    pages = torch.tensor([3, 5, 6], device=dev)
    pos = torch.arange(length, device=dev)
    cache.write_tokens(1, k, v, pages[pos // page], pos % page)

    n_q = 7 if kind == "prefill" else 1
    q = torch.randn(n_q, qo_heads, dim, device=dev, dtype=torch.bfloat16) * 0.5
    kv_kwargs = dict(
        paged_kv_indptr=torch.tensor([0, 3], dtype=torch.int32),
        paged_kv_indices=pages.int().cpu(),
        paged_kv_last_page_len=torch.tensor([length - 2 * page], dtype=torch.int32),
    )
    workspace = Workspace(torch.empty(128 << 20, dtype=torch.uint8, device=dev), "kv")
    common = dict(num_qo_heads=qo_heads, num_kv_heads=heads, head_dim=dim, page_size=page, device=dev, kv_dtype=FP8)
    if kind == "prefill":
        wrapper = FlashInferPrefillWrapper(workspace, **common)
        wrapper.plan(qo_indptr=torch.tensor([0, n_q], dtype=torch.int32), causal=True, dtype=torch.bfloat16,
                     **kv_kwargs)
    else:
        wrapper = FlashInferDecodeWrapper(workspace, use_tensor_cores=kind == "decode_tensor_cores", **common)
        wrapper.plan(dtype=torch.bfloat16, **kv_kwargs)

    manager = types.SimpleNamespace(
        _current_plan_states={"main": wrapper}, _layer_scales=layer_kv_scales(scales, qo_heads),
        _layers=None, _default_layer_idx=None, _kv_cache_name="kv",
    )
    out = FlashInferManager._attend(manager, q, "main", cache.layer_view(1), 1)
    assert out.dtype is torch.bfloat16

    stored = cache.read_tokens(1, pages[pos // page], pos % page)
    expected = _reference_attention(q, stored[:, 0], stored[:, 1], causal=True)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


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
    assert torch.equal(raw[:, list(mgr._streams["r0"]["main"].page_indices)], expected)


# ---- parallelism ----

def _tp(rank, world_size, sp_size=1):
    """A joint group of ``world_size`` TP ranks (by ``sp_size`` SP ranks), as `kv_scales_for` reads it."""
    return types.SimpleNamespace(
        tp_group=types.SimpleNamespace(rank=rank, world_size=world_size),
        sp_group=types.SimpleNamespace(rank=0, world_size=sp_size),
        world_size=world_size * sp_size, rank=rank,
    )


@pytest.mark.parametrize("total_kv, tp", [(8, 1), (8, 2), (8, 4), (2, 4), (1, 2), (4, 8)])
def test_each_tp_rank_gets_the_scales_of_the_kv_heads_its_qkv_projection_loads(tmp_path, total_kv, tp):
    """`kv_scales_for` slices the scales file by the same heads `QKVParallelLinear` hands a rank,
    replicated heads included, so a rank divides its K by its own heads' scales."""
    from mstar.engine.resources.kv.quant import kv_scales_for
    from mstar.model.components.distributed.linear import QKVParallelLinear

    head_dim = 2
    heads = torch.arange(total_kv, dtype=torch.float32)
    path = _write_scales(tmp_path / "s.safetensors", layers=1, heads=total_kv,
                         k=(heads + 1)[None], v=(heads + 1)[None] * 10)
    # each K row of the checkpoint carries its head's index
    k_weight = heads.repeat_interleave(head_dim)[:, None].expand(-1, 4).contiguous()
    qo_heads = max(tp, total_kv) * 2
    for rank in range(tp):
        group = types.SimpleNamespace(rank=rank, world_size=tp)
        qkv = QKVParallelLinear(group, 4, head_dim, total_num_heads=qo_heads, total_num_kv_heads=total_kv, bias=False)
        qkv.weight_loader(qkv.weight, k_weight, "k")
        start = qkv.num_heads * head_dim
        loaded = qkv.weight.data[start:start + qkv.num_kv_heads * head_dim, 0][::head_dim]

        cfg = _cfg(num_layers=1, num_kv_heads=total_kv, num_qo_heads=qo_heads, quant=_static(path))
        cfg.shard(tp)
        scales = kv_scales_for(cfg, EngineResourceInfo(device=CPU, joint_comm_group=_tp(rank, tp)))
        assert scales.k.shape == (1, cfg.num_kv_heads)
        assert (scales.k[0] - 1).tolist() == loaded.tolist(), (rank, tp)


def test_shm_transfer_moves_fp8_pages_between_workers(tmp_path):
    """A node on another worker reads fp8 pages byte for byte and reads them back with the same
    scales; a worker whose pages are encoded otherwise refuses them."""
    from mstar.engine.resources.kv.transfer import KVReadInfo, ShmKVTransferEngine

    quant = _static(_write_scales(tmp_path / "s.safetensors"))
    scales = load_kv_scales(quant, 2, 2, 0, 1, CPU)
    source = KVCache(_cfg(), CPU, torch.bfloat16, quant=quant, scales=scales)
    destination = KVCache(_cfg(), CPU, torch.bfloat16, quant=quant, scales=scales)
    k = torch.randn(12, 2, 8, dtype=torch.bfloat16)
    pos = torch.arange(12)
    pages = torch.tensor([1, 3])
    source.write_tokens(1, k, -k, pages[pos // 8], pos % 8)
    producer = ShmKVTransferEngine(source, "producer", str(tmp_path / "shm"))
    consumer = ShmKVTransferEngine(destination, "consumer", str(tmp_path / "shm"))
    info = producer.get_kv_transfer_info(request_id="r", label="main", page_indices=[1, 3], seq_len=12)
    reads = [KVReadInfo(layer, local, remote, 0, 8) for layer in range(2) for local, remote in ((0, 1), (2, 3))]
    consumer.read_batched_async(info, reads)
    raw_src, raw_dst = source.tensor.view(torch.uint8), destination.tensor.view(torch.uint8)
    assert torch.equal(raw_dst[:, 0], raw_src[:, 1]) and torch.equal(raw_dst[:, 2], raw_src[:, 3])
    torch.testing.assert_close(
        destination.read_tokens(1, torch.tensor([0, 2])[pos // 8], pos % 8),
        source.read_tokens(1, pages[pos // 8], pos % 8),
    )
    plain = ShmKVTransferEngine(KVCache(_cfg(), CPU, torch.bfloat16), "plain", str(tmp_path / "shm"))
    failed = plain.read_batched_async(info, reads)
    with pytest.raises(ValueError, match="stored as"):
        failed.result()


@cuda
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_tensor_parallel_ranks_together_compute_the_single_rank_attention(tmp_path, tp):
    """Each TP rank holds its KV heads' fp8 pages, written and read with its own slice of the
    static scales; the ranks' outputs side by side are the single-rank output. At tp=8 over
    4 KV heads, ranks share heads (replicated), each with that head's scales."""
    from mstar.engine.resources.attn.flashinfer import FlashInferManager
    from mstar.engine.resources.attn.workspace import Workspace
    from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper, FlashInferPrefillWrapper
    from mstar.engine.resources.kv.quant import kv_scales_for

    dev = torch.device("cuda")
    kv_heads, qo_heads, dim, page, length = 4, 16, 128, 16, 40
    path = _write_scales(tmp_path / "s.safetensors", layers=2, heads=kv_heads,
                         k=torch.tensor([[0.02, 0.05, 0.03, 0.04]] * 2),
                         v=torch.tensor([[0.01, 0.03, 0.02, 0.05]] * 2))
    k = torch.randn(length, kv_heads, dim, device=dev, dtype=torch.bfloat16) * 2
    v = torch.randn(length, kv_heads, dim, device=dev, dtype=torch.bfloat16) * 2
    pos = torch.arange(length, device=dev)
    pages = torch.tensor([3, 5, 6], device=dev)
    kv_kwargs = dict(
        paged_kv_indptr=torch.tensor([0, 3], dtype=torch.int32), paged_kv_indices=pages.int().cpu(),
        paged_kv_last_page_len=torch.tensor([length - 2 * page], dtype=torch.int32),
    )

    def attend(rank, world, q, decode):
        cfg = _cfg(num_kv_heads=kv_heads, num_qo_heads=qo_heads, head_dim=dim, max_num_pages=8, page_size=page,
                   quant=_static(path))
        cfg.shard(world)
        group = _tp(rank, world) if world > 1 else None
        scales = kv_scales_for(cfg, EngineResourceInfo(device=dev, joint_comm_group=group))
        cache = KVCache(cfg, dev, torch.bfloat16, quant=cfg.quant, scales=scales)
        heads = local_heads(kv_heads, rank, world)
        cache.write_tokens(1, k[:, heads].contiguous(), v[:, heads].contiguous(), pages[pos // page], pos % page)
        workspace = Workspace(torch.empty(128 << 20, dtype=torch.uint8, device=dev), "kv")
        common = dict(num_qo_heads=cfg.num_qo_heads, num_kv_heads=cfg.num_kv_heads, head_dim=dim,
                      page_size=page, device=dev, kv_dtype=FP8)
        if decode:
            wrapper = FlashInferDecodeWrapper(workspace, **common)
            wrapper.plan(dtype=torch.bfloat16, **kv_kwargs)
        else:
            wrapper = FlashInferPrefillWrapper(workspace, **common)
            wrapper.plan(qo_indptr=torch.tensor([0, q.shape[0]], dtype=torch.int32), causal=True,
                         dtype=torch.bfloat16, **kv_kwargs)
        manager = types.SimpleNamespace(
            _current_plan_states={"main": wrapper}, _layer_scales=layer_kv_scales(scales, cfg.num_qo_heads),
            _layers=None, _default_layer_idx=None, _kv_cache_name="kv",
        )
        per = qo_heads // world
        return FlashInferManager._attend(manager, q[:, rank * per:(rank + 1) * per].contiguous(), "main",
                                         cache.layer_view(1), 1)

    for decode, n_q in ((True, 1), (False, 7)):
        q = torch.randn(n_q, qo_heads, dim, device=dev, dtype=torch.bfloat16) * 0.5
        whole = attend(0, 1, q, decode)
        parts = torch.cat([attend(rank, tp, q, decode) for rank in range(tp)], dim=1)
        torch.testing.assert_close(parts.float(), whole.float(), rtol=1e-2, atol=1e-2)
