"""GPU tests for the paged execution paths (skipped without CUDA):

* the FlashInfer MLA wrapper over a ``KVLayout.MLA`` cache vs. the dense reference,
* the whole tiny model through the resource-driven paged forward (KDA state resource +
  MLA cache + attention) vs. the dense forward, for a packed prefill and a decode step.
"""
import pytest
import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    KVStep,
    LinearAttnStep,
    RecurrentStep,
    SamplerStep,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.kda import KDAPlan
from mstar.model.kimi_k3.config import KDA_ATTN, KDA_STATE, MLA_ATTN, MLA_KV, SAMPLER, KimiK3Config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


def _build(specs, device=DEV):
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo

    by_key = resolve_spec_dependencies(specs)
    resources = {}
    transfer = TransferEngineInfo("local", "local", LocalTransferEngine("localhost"))
    for key, spec in by_key.items():
        info = EngineResourceInfo(device=device, kv_dtype=torch.bfloat16, transfer_engine_info=transfer,
                                  dependencies={d: by_key[d] for d in spec.depends_on()})
        resources[key] = build_resource(spec, info)
    return resources, StepRunner(resources)




def _production_shapes(cfg) -> bool:
    return cfg.kv_lora_rank == 512 or cfg.kda_head_dim == 128


def _assert_logits_agree(logits, dense, cfg):
    """Exact top-1 agreement for the fp32-reference tiny model; production-shape checkpoints
    run the bf16 kernels (FlashInfer MLA, fla/FlashKDA at head_dim 128), which deviate from the
    fp32 reference by ~0.5% per layer: on random weights with near-uniform logits that flips
    argmaxes, so those only get a statistical check (the kernels are unit-tested separately)."""
    logits, dense = logits.float(), dense.float()
    if _production_shapes(cfg) and logits.is_cuda:
        top1 = (logits.argmax(-1) == dense.argmax(-1)).float().mean().item()
        rel = ((logits - dense).norm() / dense.norm()).item()
        # near-uniform random-weight logits: routing flips compound through the layers, so
        # only a loose global bound is meaningful (and top-1 only over enough positions)
        assert rel < 0.5, f"logits rel {rel:.3f}, top-1 agreement {top1:.2f}"
        assert logits.shape[0] < 8 or top1 >= 0.5, f"top-1 agreement {top1:.2f} (rel {rel:.3f})"
    else:
        torch.testing.assert_close(logits, dense, rtol=5e-2, atol=5e-1)
        assert torch.equal(logits.argmax(-1), dense.argmax(-1))


def _assert_rel(actual, expected, tol, what):
    """Relative RMS check: bf16 chunked kernels deviate from the fp32 reference by ~0.5% RMS
    with occasional larger single elements, which elementwise tolerances misjudge."""
    a, e = actual.float(), expected.float()
    rel = ((a - e).norm() / (e.norm() + 1e-9)).item()
    max_abs = (a - e).abs().max().item()
    # near-zero references (a decayed recurrent read after a long random prefix) make the
    # relative error meaningless; an absolute bound covers them
    assert rel < tol or max_abs < 5e-3, f"{what}: rel RMS {rel:.4f} >= {tol}, max abs {max_abs:.4f}"

def _fresh_steps():
    """New per-resource step objects for every batch: ``SubmoduleStep`` binds the batch's
    segments into a step only if it has none, so reusing them would keep the first batch's."""
    return {MLA_KV: KVStep(), MLA_ATTN: AttentionStep(causal=True), KDA_STATE: RecurrentStep(),
            KDA_ATTN: LinearAttnStep(),
            SAMPLER: SamplerStep(apply_penalty=False)}

def _step(runner, keys_steps, segs, walk):
    step = SubmoduleStep(segments=segs, steps=keys_steps)
    ctx = StepContext(request_ids=[s.request_id for s in segs], graph_walk=walk, slot=0, capture=False)
    step.set_ctx(ctx)
    assert runner.admit(step).ok
    runner.plan(step)
    return step


@cuda
@pytest.mark.parametrize(
    "heads,lora,rope,page", [(4, 512, 64, 64), (8, 512, 64, 64), (12, 512, 64, 16), (4, 64, 16, 16)])
def test_mla_wrapper_matches_dense_reference(heads, lora, rope, page):
    """FlashInfer's Hopper MLA kernel at the real latent shape (two head counts / page
    sizes) and the torch fallback at a small shape, both against dense attention."""
    torch.manual_seed(0)
    kv_cfg = KVConfig(num_layers=1, num_kv_heads=1, head_dim=lora + rope, max_seq_len=1024, max_num_pages=64,
                      page_size=page, num_qo_heads=heads, layout=KVLayout.MLA, kv_lora_rank=lora, qk_rope_head_dim=rope)
    scale = (128 + rope) ** -0.5
    specs = [
        KVSpec(resource_key="kv", nodes={"n"}, config=kv_cfg),
        AttentionSpec(resource_key="attn", nodes={"n"},
                      config=AttentionConfig(kv_cache="kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=scale)),
    ]
    res, runner = _build(specs)
    kv, attn = res["kv"], res["attn"]
    runner.ingest_request("a")
    runner.ingest_request("b")
    lens = [37, 5]
    t = sum(lens)
    step = _step(runner, {"kv": KVStep(), "attn": AttentionStep(causal=True)},
                 [Segment("a", "main", lens[0]), Segment("b", "main", lens[1])], "prefill")
    latent = torch.randn(t, lora + rope, device=DEV, dtype=torch.bfloat16)
    q_lat = torch.randn(t, heads, lora, device=DEV, dtype=torch.bfloat16)
    q_pe = torch.randn(t, heads, rope, device=DEV, dtype=torch.bfloat16)
    kv.set_default_layer_idx(0)
    attn.set_default_layer_idx(0)
    kv.set_default_label("main")
    attn.set_default_label("main")
    kv.write_kv(latent)
    out = attn.run(q_lat, kv_cache_layer=kv.layer_view(), q_pe=q_pe)
    runner.commit(step)

    def dense(ql, qp, lat):
        c, kp = lat.float().split([lora, rope], dim=-1)
        s = (torch.einsum("qhl,kl->hqk", ql.float(), c) + torch.einsum("qhr,kr->hqk", qp.float(), kp)) * scale
        n, m = ql.shape[0], lat.shape[0]
        mask = torch.arange(m, device=DEV)[None, :] <= (torch.arange(n, device=DEV)[:, None] + (m - n))
        s = s.masked_fill(~mask[None], float("-inf"))
        return torch.einsum("hqk,kl->qhl", torch.softmax(s, -1), c)

    ref_a = dense(q_lat[: lens[0]], q_pe[: lens[0]], latent[: lens[0]])
    ref_b = dense(q_lat[lens[0]:], q_pe[lens[0]:], latent[lens[0]:])
    torch.testing.assert_close(out[: lens[0]].float(), ref_a, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out[lens[0]:].float(), ref_b, rtol=2e-2, atol=2e-2)
    # decode: one new token per request against the resident context
    step2 = _step(runner, {"kv": KVStep(), "attn": AttentionStep(causal=True)},
                  [Segment("a", "main", 1), Segment("b", "main", 1)], "decode")
    lat2 = torch.randn(2, lora + rope, device=DEV, dtype=torch.bfloat16)
    ql2 = torch.randn(2, heads, lora, device=DEV, dtype=torch.bfloat16)
    qp2 = torch.randn(2, heads, rope, device=DEV, dtype=torch.bfloat16)
    kv.write_kv(lat2)
    out2 = attn.run(ql2, kv_cache_layer=kv.layer_view(), q_pe=qp2)
    runner.commit(step2)
    ref2a = dense(ql2[:1], qp2[:1], torch.cat([latent[: lens[0]], lat2[:1]]))
    ref2b = dense(ql2[1:], qp2[1:], torch.cat([latent[lens[0]:], lat2[1:]]))
    torch.testing.assert_close(out2[:1].float(), ref2a, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out2[1:].float(), ref2b, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_tiny_model_paged_forward_matches_dense(tiny_dir, device):
    """The paged model path (resources bound like the engine does) vs. forward_dense;
    on CPU it runs the torch KDA kernels and the MLA fallback, so the addressing logic
    is covered without a GPU."""
    device = torch.device(device)
    from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM
    from mstar.model.loader import load_weights
    from mstar.model.registry import get_model_class

    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    with torch.device("meta"):
        lm = KimiK3ForCausalLM(cfg)
    lm = lm.to(torch.bfloat16)
    lm.to_empty(device=device)
    for name, p in lm.named_parameters():
        if name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
            p.data = p.data.to(torch.float32)
    load_weights(lm, tiny_dir, device=device)
    lm.eval()
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir))
    res, runner = _build(model.get_node_resources(), device)
    # bind the resources into the layers like the engine does
    for mod in lm.modules():
        if hasattr(mod, "bind_resources"):
            mod.bind_resources(res)
    # this test covers the addressing (pool slots, packed rows, state carried into the decode) with the
    # torch reference kernels on both devices; the fla kernels have their own tests below
    from mstar.model.kimi_k3.components.kda import TorchKDAKernels

    res[KDA_ATTN].set_kernels(TorchKDAKernels())
    steps = _fresh_steps
    torch.manual_seed(0)
    ids_a = torch.randint(0, 1000, (23,), device=device)
    ids_b = torch.randint(0, 1000, (9,), device=device)
    runner.ingest_request("a")
    runner.ingest_request("b")
    with torch.no_grad():
        dense_a, st_a = lm.forward_dense(ids_a)
        dense_b, st_b = lm.forward_dense(ids_b)
        step = _step(runner, steps(), [Segment("a", "main", 23), Segment("b", "main", 9)], "prefill")
        hidden = lm.model(lm.model.embed_tokens(torch.cat([ids_a, ids_b])), label="main")
        logits = lm.lm_head(hidden)
        runner.commit(step)
    _assert_logits_agree(logits[:23], dense_a, cfg)
    _assert_logits_agree(logits[23:], dense_b, cfg)
    # decode one token each
    nxt_a, nxt_b = dense_a[-1].argmax().view(1), dense_b[-1].argmax().view(1)
    with torch.no_grad():
        d2a, _ = lm.forward_dense(nxt_a, st_a)
        d2b, _ = lm.forward_dense(nxt_b, st_b)
        step2 = _step(runner, steps(), [Segment("b", "main", 1), Segment("a", "main", 1)], "decode")
        hidden2 = lm.model(lm.model.embed_tokens(torch.cat([nxt_b, nxt_a])), label="main")
        logits2 = lm.lm_head(hidden2)
        runner.commit(step2)
    _assert_logits_agree(logits2[0:1], d2b, cfg)
    _assert_logits_agree(logits2[1:2], d2a, cfg)


@cuda
def test_fla_kernels_match_torch_reference(tiny_dir):
    """The slot-indexed fla kernels (prefill chunk + decode recurrent, conv update) agree
    with the torch reference kernels on the same resource-style state tensors."""
    from mstar.engine.resources.linear_attn.kda_kernels import FLAKDAKernels
    from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM
    from mstar.model.kimi_k3.config import KimiK3Config
    from mstar.model.loader import load_weights

    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    with torch.device("meta"):
        lm = KimiK3ForCausalLM(cfg)
    lm = lm.to(torch.bfloat16)
    lm.to_empty(device=DEV)
    for name, p in lm.named_parameters():
        if name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
            p.data = p.data.to(torch.float32)
    load_weights(lm, tiny_dir, device=DEV)
    layer = lm.model.layers[0].self_attn
    p = layer.params()
    torch.manual_seed(3)
    lens = [19, 6, 33]
    t = sum(lens)
    x = torch.randn(t, cfg.hidden_size, device=DEV, dtype=torch.bfloat16)
    qkv, g_raw, beta_raw, _ = layer._project(x)
    n_slots = 5
    conv_t = torch.zeros(n_slots, 3 * layer.projection_size, cfg.kda_conv_kernel_size - 1, device=DEV,
                         dtype=torch.bfloat16)
    rec_t = torch.zeros(n_slots, layer.num_heads, layer.head_dim, layer.head_dim, device=DEV)
    conv_f, rec_f = conv_t.clone(), rec_t.clone()
    cu = [0, lens[0], lens[0] + lens[1], t]
    slots, has = [1, 3, 4], [False, False, False]
    plan = KDAPlan(
        slot_ids=torch.tensor(slots, dtype=torch.int32, device=DEV), has_state=torch.tensor(has, device=DEV),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=DEV), cu_seqlens_cpu=cu, num_rows=3, num_tokens=t,
        is_decode=False,
    )
    o_t = layer.kernels.run_paged(qkv, g_raw, beta_raw, plan, conv_t, rec_t, p)
    o_f = FLAKDAKernels().run_paged(qkv, g_raw, beta_raw, plan, conv_f, rec_f, p)
    _assert_rel(o_f, o_t, 2e-2, "prefill output")
    _assert_rel(rec_f[slots], rec_t[slots], 2e-2, "prefill recurrent state")
    _assert_rel(conv_f[slots], conv_t[slots], 1e-2, "prefill conv state")
    # decode: one token for rows in slots 4 and 1 (reversed order), states resident
    x2 = torch.randn(2, cfg.hidden_size, device=DEV, dtype=torch.bfloat16)
    qkv2, g2, b2, _ = layer._project(x2)
    slots2, has2, cu2 = [4, 1], [True, True], [0, 1, 2]
    plan2 = KDAPlan(
        slot_ids=torch.tensor(slots2, dtype=torch.int32, device=DEV), has_state=torch.tensor(has2, device=DEV),
        cu_seqlens=torch.tensor(cu2, dtype=torch.int32, device=DEV), cu_seqlens_cpu=cu2, num_rows=2, num_tokens=2,
        is_decode=True,
    )
    o2_t = layer.kernels.run_paged(qkv2, g2, b2, plan2, conv_t, rec_t, p)
    o2_f = FLAKDAKernels().run_paged(qkv2, g2, b2, plan2, conv_f, rec_f, p)
    _assert_rel(o2_f, o2_t, 2e-2, "decode output")
    _assert_rel(rec_f[slots2], rec_t[slots2], 2e-2, "decode recurrent state")
    _assert_rel(conv_f[slots2], conv_t[slots2], 1e-2, "decode conv state")


@cuda
def test_tiny_model_paged_forward_with_fla_kernels(tiny_dir):
    """Whole-model paged forward on the fla kernels vs the dense reference."""
    from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM, select_kda_kernels
    from mstar.model.loader import load_weights
    from mstar.model.registry import get_model_class

    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    with torch.device("meta"):
        lm = KimiK3ForCausalLM(cfg)
    lm = lm.to(torch.bfloat16)
    lm.to_empty(device=DEV)
    for name, p in lm.named_parameters():
        if name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
            p.data = p.data.to(torch.float32)
    load_weights(lm, tiny_dir, device=DEV)
    lm.eval()
    assert select_kda_kernels(lm, DEV) is True
    model = get_model_class("kimi_k3")(model_path_hf=str(tiny_dir))
    res, runner = _build(model.get_node_resources())
    for mod in lm.modules():
        if hasattr(mod, "bind_resources"):
            mod.bind_resources(res)
    steps = _fresh_steps
    torch.manual_seed(5)
    ids = torch.randint(0, 1000, (31,), device=DEV)
    runner.ingest_request("a")
    with torch.no_grad():
        dense, st = lm.forward_dense(ids)
        step = _step(runner, steps(), [Segment("a", "main", 31)], "prefill")
        logits = lm.lm_head(lm.model(lm.model.embed_tokens(ids), label="main"))
        runner.commit(step)
        _assert_logits_agree(logits, dense, cfg)
        nxt = dense[-1].argmax().view(1)
        d2, _ = lm.forward_dense(nxt, st)
        step2 = _step(runner, steps(), [Segment("a", "main", 1)], "decode")
        l2 = lm.lm_head(lm.model(lm.model.embed_tokens(nxt), label="main"))
        runner.commit(step2)
    _assert_logits_agree(l2[0:1], d2, cfg)


@cuda
def test_flashkda_prefill_matches_fla_kernels():
    """FlashKDA prefill (D=128, H=4) vs the fla chunk kernel on the same random layer."""
    pytest.importorskip("flash_kda")
    from mstar.engine.resources.linear_attn.kda_kernels import FLAKDAKernels, FlashKDAKernels
    from mstar.model.kimi_k3.components.kda import ParallelKDAAttention

    torch.manual_seed(7)
    hidden, h, d = 256, 4, 128
    layer = ParallelKDAAttention(hidden_size=hidden, num_heads=h, head_dim=d, gate_lower_bound=-5.0)
    layer = layer.to(DEV, torch.bfloat16)
    for prm in layer.parameters():
        with torch.no_grad():
            prm.normal_(std=0.05)
    with torch.no_grad():
        layer.A_log.copy_(torch.log(torch.empty(h, device=DEV).uniform_(1, 16)))
        layer.dt_bias.normal_(std=0.5)
        layer.A_log.data = layer.A_log.data.float()
        layer.dt_bias.data = layer.dt_bias.data.float()
    p = layer.params()
    lens = [40, 3, 70]
    t = sum(lens)
    x = torch.randn(t, hidden, device=DEV, dtype=torch.bfloat16)
    qkv, g_raw, beta_raw, _ = layer._project(x)
    n_slots = 4
    conv_a = torch.zeros(n_slots, 3 * h * d, 3, device=DEV, dtype=torch.bfloat16)
    rec_a = torch.zeros(n_slots, h, d, d, device=DEV)
    conv_b, rec_b = conv_a.clone(), rec_a.clone()
    # give slot 1 a resident state so the initial-state path is exercised
    rec_a[1].normal_(std=0.1)
    rec_b.copy_(rec_a)
    cu = [0, lens[0], lens[0] + lens[1], t]
    slots, has = [1, 2, 3], [True, False, False]
    plan = KDAPlan(
        slot_ids=torch.tensor(slots, dtype=torch.int32, device=DEV), has_state=torch.tensor(has, device=DEV),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=DEV), cu_seqlens_cpu=cu, num_rows=3, num_tokens=t,
        is_decode=False,
    )
    o_fla = FLAKDAKernels().run_paged(qkv, g_raw, beta_raw, plan, conv_a, rec_a, p)
    o_fk = FlashKDAKernels().run_paged(qkv, g_raw, beta_raw, plan, conv_b, rec_b, p)
    torch.testing.assert_close(o_fk.float(), o_fla.float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(rec_b[slots], rec_a[slots], rtol=3e-2, atol=3e-2)


@cuda
def test_kda_prefill_layers_do_not_sync_with_the_host():
    """A prefill step's KDA layers run without a host sync: the int64 indices, the state mask and fla's
    chunk table are built once per plan and every layer reads them (FlashKDA's first layer of a step
    included; the fla bundle's first layer may still sync inside fla's own chunk bookkeeping). One
    sync per layer had stalled the eager prefill at all 69 KDA layers of Kimi K3."""
    from mstar.engine.resources.linear_attn.kda_kernels import FLAKDAKernels, default_kernels
    from mstar.model.kimi_k3.components.kda import ParallelKDAAttention

    torch.manual_seed(11)
    hidden, h, d = 256, 4, 128
    layer = ParallelKDAAttention(hidden_size=hidden, num_heads=h, head_dim=d, gate_lower_bound=-5.0)
    layer = layer.to(DEV, torch.bfloat16)
    with torch.no_grad():
        for prm in layer.parameters():
            prm.normal_(std=0.05)
        layer.A_log.data = torch.log(torch.empty(h, device=DEV).uniform_(1, 16))
        layer.dt_bias.data = torch.randn(h * d, device=DEV) * 0.5
    p = layer.params()
    lens = [40, 3, 70]
    t = sum(lens)
    x = torch.randn(t, hidden, device=DEV, dtype=torch.bfloat16)
    qkv, g_raw, beta_raw, _ = layer._project(x)
    cu = [0, lens[0], lens[0] + lens[1], t]

    def plan():
        return KDAPlan(
            slot_ids=torch.tensor([1, 2, 3], dtype=torch.int32, device=DEV),
            has_state=torch.tensor([True, False, False], device=DEV),
            cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=DEV), cu_seqlens_cpu=cu, num_rows=3,
            num_tokens=t, is_decode=False,
        )

    served = default_kernels(DEV)  # FlashKDA where it is built, fla otherwise
    bundles = [(FLAKDAKernels(), 1)] + ([(served, 0)] if type(served) is not FLAKDAKernels else [])
    for kernels, warm_layers in bundles:
        conv = torch.zeros(4, 3 * h * d, 3, device=DEV, dtype=torch.bfloat16)
        rec = torch.zeros(4, h, d, d, device=DEV)
        rec[1].normal_(std=0.1)
        want = kernels.run_paged(qkv, g_raw, beta_raw, plan(), conv.clone(), rec.clone(), p)  # JIT, autotuning
        torch.cuda.synchronize()
        step = plan()
        for _ in range(warm_layers):
            kernels.run_paged(qkv, g_raw, beta_raw, step, conv.clone(), rec.clone(), p)
        torch.cuda.set_sync_debug_mode("error")
        try:
            got = kernels.run_paged(qkv, g_raw, beta_raw, step, conv, rec, p)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        torch.testing.assert_close(got.float(), want.float(), rtol=1e-3, atol=1e-3, msg=type(kernels).__name__)
