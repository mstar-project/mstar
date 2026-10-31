"""The fused MLA pre-attention (mla_prep.py) at one GLM-5.2 TP8 rank's dims: bit for bit
against FlashInfer's rmsnorm, the position resource's RoPE and the cache scatter, then a
compiled attention layer with the fused prep against the same layer without it."""
import copy
import importlib
import sys

import pytest
import torch

from mstar.model.glm52.components.attention import Glm52MLAAttention
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.utils import mla_prep

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need CUDA")

Q, L, R, NOPE, H = 2048, 512, 64, 192, 8  # q_lora, kv_lora, rope, nope; heads per TP8 rank
EPS = 1e-5
PAGE = 128
TOKENS = [1, 8, 32, 64, 1100]
MAX_POS = 1 << 18


_REAL_FLASHINFER = []


@pytest.fixture(autouse=True)
def _real_flashinfer(monkeypatch):
    """test_glm52_mla_absorb leaves a CPU flashinfer stand-in in sys.modules; these tests
    compare against the real kernels, imported once (a re-import loses its submodules)."""
    fi = sys.modules.get("flashinfer")
    if fi is None or hasattr(fi, "__path__") or not torch.cuda.is_available():
        return
    if not _REAL_FLASHINFER:
        monkeypatch.delitem(sys.modules, "flashinfer")
        _REAL_FLASHINFER.append(importlib.import_module("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer", _REAL_FLASHINFER[0])


def _bits(x):
    return x.view(torch.int16)


def _fused_rows(T, seed):
    """[q_a | kv_a | k_pe] rows at scales spread over e^(+-4), with outlier columns."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, Q + L + R, generator=g, device="cuda")
    x = x * torch.exp(torch.randn(T, 1, generator=g, device="cuda") * 2)
    x[:, ::97] *= 30
    return x.to(torch.bfloat16)


def _norm_weight(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (1.0 + 0.1 * torch.randn(n, generator=g, device="cuda")).to(torch.bfloat16)


_POSITIONS = []


def _positions():
    """GLM-5.2's position resource on the GPU, its fp32 table over MAX_POS positions."""
    from mstar.engine.resources.position.manager import RopeManager

    if not _POSITIONS:
        _POSITIONS.append(RopeManager(
            Glm52ModelConfig().position_config(), torch.device("cuda"), max_seq_len=MAX_POS))
    return _POSITIONS[0]


def _rope_inputs(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(T, H, NOPE + R, generator=g, device="cuda") * 2).to(torch.bfloat16)
    pos = torch.randint(0, MAX_POS, (T,), generator=g, device="cuda")
    return q, _positions().cos_sin_table, pos


def _slots(T, pages, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    flat = torch.randperm(pages * PAGE, generator=g, device="cuda")[:T]
    return flat // PAGE, flat % PAGE


def _unfused_rope(q_pe, k_pe, pos):
    """The unfused path's RoPE: the position resource's (FlashInfer's cos/sin-cache kernel)."""
    return _positions().apply_qk(
        q_pe.clone(memory_format=torch.contiguous_format),
        k_pe.clone(memory_format=torch.contiguous_format), pos_ids=pos)


@gpu
@pytest.mark.parametrize("T", TOKENS)
def test_q_norm_is_flashinfers_rmsnorm_bit_for_bit(T):
    from mstar.engine.resources.rms_norm.rms_norm import run_rms_norm

    fused = _fused_rows(T, T)
    w = _norm_weight(Q, 1)
    out = mla_prep.q_norm(fused, w, EPS)
    assert out.shape == (T, Q) and out.is_contiguous()
    assert torch.equal(_bits(out), _bits(run_rms_norm(fused[:, :Q].contiguous(), w, eps=EPS)))


@gpu
@pytest.mark.parametrize("T", TOKENS)
def test_kv_write_q_rope_is_the_unfused_path_bit_for_bit(T):
    """Against FlashInfer's rmsnorm, the position resource's RoPE and the advanced-index
    scatter; the whole cache compares, so a stray write shows too."""
    from mstar.engine.resources.rms_norm.rms_norm import run_rms_norm

    fused = _fused_rows(T, 100 + T)
    w = _norm_weight(L, 2)
    q, table, pos = _rope_inputs(T, 200 + T)
    num_pages = -(-T // PAGE) + 3
    cache = torch.randn(num_pages, PAGE, L + R, device="cuda").to(torch.bfloat16)
    pages, offsets = _slots(T, num_pages, 300 + T)

    kv_c = run_rms_norm(fused[:, Q:Q + L].clone(), w, eps=EPS)
    q_pe, k_pe = _unfused_rope(q[..., NOPE:], fused[:, Q + L:].view(T, 1, R), pos)
    expect = cache.clone()
    expect[pages, offsets] = torch.cat([kv_c, k_pe.squeeze(1)], dim=-1)

    out = mla_prep.kv_write_q_rope(fused, w, EPS, table, pos, q, cache, pages, offsets)
    assert out.shape == (T, H, R) and out.is_contiguous()
    assert torch.equal(_bits(cache), _bits(expect))
    assert torch.equal(_bits(out), _bits(q_pe))


@gpu
def test_kv_write_stops_at_the_planned_tokens():
    """Rows past ``len(pages)`` (the plan's token count) get q_pe but no cache write."""
    T, n = 8, 5
    fused = _fused_rows(T, 11)
    q, table, pos = _rope_inputs(T, 12)
    cache = torch.randn(2, PAGE, L + R, device="cuda").to(torch.bfloat16)
    pages, offsets = _slots(T, 2, 13)
    before = cache.clone()
    mla_prep.kv_write_q_rope(fused, _norm_weight(L, 4), EPS, table, pos, q, cache, pages[:n],
                             offsets[:n])
    unwritten = torch.ones(2, PAGE, dtype=torch.bool, device="cuda")
    unwritten[pages[:n], offsets[:n]] = False
    assert torch.equal(cache[unwritten], before[unwritten])
    assert not torch.equal(cache[pages[:n], offsets[:n]], before[pages[:n], offsets[:n]])


# ── the layer: compiled, fused prep vs op by op, through the real MLA cache and attention ──


def test_fused_prep_needs_the_absorbed_path_and_supported_widths():
    cfg = Glm52ModelConfig.reduced()
    assert not Glm52MLAAttention(cfg).mla_fused_prep  # naive path, q_lora_rank 48
    cfg.mla_absorb = True
    assert not Glm52MLAAttention(cfg).mla_fused_prep  # q_lora_rank 48
    cfg.q_lora_rank, cfg.kv_lora_rank, cfg.qk_rope_head_dim = 256, 256, 8
    assert Glm52MLAAttention(cfg).mla_fused_prep
    cfg.kernels = "reference"  # the numerics anchor runs the ops
    assert not Glm52MLAAttention(cfg).mla_fused_prep


def _gpu_resources(cfg, num_pages):
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVLayout, PagedKVConfig
    from mstar.engine.resources.position.manager import RopeManager
    from mstar.engine.resources.runner import StepRunner
    from mstar.model.glm52._testing import _StubTransferManager
    from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE, POS_RESOURCE

    dev = torch.device("cuda")
    kv_cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=cfg.cache_latent_dim,
        max_seq_len=cfg.max_seq_len, max_num_pages=num_pages, page_size=PAGE,
        num_qo_heads=cfg.num_attention_heads, layout=KVLayout.MLA,
        kv_lora_rank=cfg.kv_lora_rank, qk_rope_head_dim=cfg.qk_rope_head_dim)
    stub = manager_mod.KVTransferManager
    manager_mod.KVTransferManager = _StubTransferManager
    try:
        kv = manager_mod.KVManager(cfg=kv_cfg, name=KV_RESOURCE, joint_comm_group=None,
                                   transfer_engine_info=None, device=dev, dtype=torch.bfloat16)
    finally:
        manager_mod.KVTransferManager = stub
    attn = FlashInferMLAManager(kv_cache=KV_RESOURCE, device=dev, dtype=torch.bfloat16,
                                kv_config=kv_cfg, sm_scale=cfg.softmax_scale)
    pos = RopeManager(cfg.position_config(), dev, max_seq_len=cfg.max_seq_len)
    resources = {KV_RESOURCE: kv, ATTN_RESOURCE: attn, POS_RESOURCE: pos}
    runner = StepRunner(resources, node_resources={"LLM": list(resources)})
    runner.ingest_request("r0")
    return resources, runner


def _step(layer, fwd, runner, h, start):
    from mstar.engine.resources import AttentionStep, KVStep, Segment, StepContext, SubmoduleStep
    from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE

    n = h.shape[0]
    step = SubmoduleStep(
        segments=[Segment("r0", "main", n)],
        steps={KV_RESOURCE: KVStep(), ATTN_RESOURCE: AttentionStep(causal=True)})
    step.set_ctx(StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)
    pos = torch.arange(start, start + n, device="cuda")
    with torch.no_grad():
        out = fwd(h, pos)
    runner.commit(step)
    return out


@gpu
def test_compiled_layer_with_fused_prep_is_the_unfused_layer_bit_for_bit():
    """A 1100-token prefill then three decode steps: same outputs, same cache bytes."""
    from mstar.model.glm52.config import KV_RESOURCE

    cfg = Glm52ModelConfig(num_hidden_layers=1, num_attention_heads=H)
    torch.manual_seed(0)
    with torch.device("cuda"):
        base = Glm52MLAAttention(cfg, layer_idx=0).to(torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        for lin in (base.q_a_proj, base.q_b_proj, base.kv_a_proj_with_mqa, base.kv_b_proj,
                    base.o_proj):
            lin.weight.normal_(0, 0.02)
        for norm in (base.q_a_layernorm, base.kv_a_layernorm):
            norm.weight.normal_(1.0, 0.1)
    base.process_weights_after_loading()
    assert base.mla_fused_prep  # where it runs, the layer takes it
    fused = copy.deepcopy(base)
    base.mla_fused_prep = False

    runs = []
    for layer in (base, fused):
        resources, runner = _gpu_resources(cfg, num_pages=16)
        layer.bind_resources(resources)
        torch._dynamo.reset()
        fwd = torch.compile(layer)
        g = torch.Generator(device="cuda").manual_seed(1)
        h = torch.randn(1103, cfg.hidden_size, generator=g, device="cuda").to(torch.bfloat16)
        outs = [_step(layer, fwd, runner, h[:1100], 0)]
        outs += [_step(layer, fwd, runner, h[t:t + 1], t) for t in range(1100, 1103)]
        runs.append((outs, resources[KV_RESOURCE].kv_cache.tensor.clone()))
    (base_outs, base_cache), (fused_outs, fused_cache) = runs
    assert torch.equal(_bits(fused_cache), _bits(base_cache))
    for got, want in zip(fused_outs, base_outs, strict=True):
        assert torch.equal(_bits(got), _bits(want))
