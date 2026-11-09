"""Kimi K3's new pieces one at a time against the reference math (``reference.py``, from the
checkpoint's ``modeling_kimi_linear.py``): SiTU-GLU on every dispatch path, the attention
residual read, and the latent MoE block."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.moe import dispatch_experts_fused
from mstar.model.kimi_k3.components.attn_res import attn_res_read
from mstar.model.kimi_k3.components.moe import KimiLatentMoE
from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.tests import reference
from mstar.model.kimi_k3.weight_loader import build_stacked_params, kimi_k3_name_remapper
from mstar.model.loader import load_weights_into
from mstar.utils.situ import SiTU, situ_and_mul


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(2)


def test_situ_matches_the_reference_and_bends_the_up_half():
    g = torch.Generator().manual_seed(0)
    gate, up = torch.randn(64, 32, generator=g) * 8, torch.randn(64, 32, generator=g) * 40
    act = SiTU(4.0, 25.0)
    torch.testing.assert_close(act(gate, up), reference.situ(gate, up, 4.0, 25.0), atol=0, rtol=0)
    # without linear_beta the up half passes through: SiLU-like gate times up
    plain = situ_and_mul(gate, up, 4.0, None)
    torch.testing.assert_close(plain, 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * up)
    assert not torch.allclose(plain, act(gate, up))  # the linear beta bites at |up| ~ 40
    # bf16 in, bf16 out, fp32 inside
    assert act(gate.bfloat16(), up.bfloat16()).dtype == torch.bfloat16
    with pytest.raises(ValueError):
        SiTU(0.0)


def test_naive_expert_dispatch_takes_the_activation():
    g = torch.Generator().manual_seed(1)
    e, hidden, inter, tokens = 4, 16, 8, 5
    w13, w2 = torch.randn(e, 2 * inter, hidden, generator=g), torch.randn(e, hidden, inter, generator=g)
    x = torch.randn(tokens, hidden, generator=g)
    ids = torch.tensor([[0, 2], [1, 3], [2, 2], [3, 0], [1, 1]])
    w = torch.rand(tokens, 2, generator=g)
    act = SiTU(4.0, 25.0)
    got = dispatch_experts_fused(x, w13, w2, e, ids, w, activation=act)
    want = torch.zeros(tokens, hidden)
    for t in range(tokens):
        for k in range(2):
            gate, up = (x[t] @ w13[ids[t, k]].T).chunk(2)
            want[t] += w[t, k] * (act(gate, up) @ w2[ids[t, k]].T)
    torch.testing.assert_close(got, want)
    # silu stays the default
    assert not torch.allclose(dispatch_experts_fused(x, w13, w2, e, ids, w), got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel needs CUDA")
def test_triton_situ_activation_matches_torch():
    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    g = torch.Generator(device="cuda").manual_seed(0)
    gate_up = (torch.randn(37, 2 * 384, device="cuda", generator=g) * 20).bfloat16()
    out = torch.empty(37, 384, device="cuda", dtype=torch.bfloat16)
    act = SiTU(4.0, 25.0)
    act_and_mul_triton(gate_up, out, activation=act)
    gate, up = gate_up.chunk(2, dim=-1)
    torch.testing.assert_close(out, act(gate, up), atol=1e-2, rtol=1e-2)


def test_attn_res_read_matches_the_reference():
    g = torch.Generator().manual_seed(2)
    t, d = 6, 32
    prefix, addend = torch.randn(t, d, generator=g), torch.randn(t, d, generator=g)
    stack = [torch.randn(t, d, generator=g) for _ in range(3)]
    norm_w, proj_w = 1 + 0.1 * torch.randn(d, generator=g), torch.randn(1, d, generator=g)
    blocks = torch.stack(stack + [torch.full((t, d), float("nan"))])  # a slot past m is never read
    got, stream = attn_res_read(blocks, 3, prefix, norm_w, proj_w, 1e-5)
    assert stream is prefix
    torch.testing.assert_close(got, reference.attn_res(prefix, stack, norm_w, proj_w, 1e-5))
    # the residual add ahead of the read: the stream is the sum, and the read mixes it
    got, stream = attn_res_read(blocks, 3, prefix, norm_w, proj_w, 1e-5, addend=addend)
    torch.testing.assert_close(stream, prefix + addend, atol=0, rtol=0)
    torch.testing.assert_close(got, reference.attn_res(prefix + addend, stack, norm_w, proj_w, 1e-5))
    # an empty stack is the stream itself: a softmax over one candidate
    got, stream = attn_res_read(None, 0, prefix, norm_w, proj_w, 1e-5)
    assert got is prefix and stream is prefix
    # a mixture of the candidates: equal scores average them
    flat, _ = attn_res_read(blocks, 3, prefix, torch.zeros(d), proj_w, 1e-5)
    torch.testing.assert_close(flat, torch.stack(stack + [prefix]).mean(0))


def test_latent_moe_matches_the_reference():
    cfg = KimiK3ModelConfig.reduced()
    sd = reference.random_checkpoint(cfg, seed=3)
    prefix = "model.layers.1.block_sparse_moe."
    parent = nn.Module()
    parent.block_sparse_moe = KimiLatentMoE(cfg)
    stream = ((k[len("model.layers.1."):], v) for k, v in sd.items() if k.startswith(prefix))
    loaded = load_weights_into(parent, stream, stacked_params=build_stacked_params(cfg.num_experts),
                               name_remapper=kimi_k3_name_remapper(False), strict=True)
    assert loaded == {n for n, _ in parent.named_parameters()}
    x = torch.randn(7, cfg.hidden_size, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        got = parent.block_sparse_moe(x)
    torch.testing.assert_close(got, reference.latent_moe(sd, prefix, x, cfg), atol=1e-5, rtol=1e-5)


def _loaded_latent_moe(cfg, prefix="model.layers.1.block_sparse_moe."):
    sd = reference.random_checkpoint(cfg, seed=3)
    parent = nn.Module()
    parent.block_sparse_moe = KimiLatentMoE(cfg)
    stream = ((k[len("model.layers.1."):], v) for k, v in sd.items() if k.startswith(prefix))
    load_weights_into(parent, stream, stacked_params=build_stacked_params(cfg.num_experts),
                      name_remapper=kimi_k3_name_remapper(False), strict=True)
    return parent.block_sparse_moe, sd


def _fake_flashinfer(monkeypatch) -> list:
    """Stand-ins for the FlashInfer entry points ``TrtllmFp4Experts`` calls, off a B200/B300: the routed MoE
    records its arguments (and returns zeros), ``mxfp8_quantize`` passes the activations through."""
    import enum
    import sys
    import types

    class Calls(list):
        tuning: list

    calls = Calls()

    def routed_moe(*args, **kwargs):
        calls.append((args, kwargs))
        tokens, top_k = args[2].shape[0], args[17]
        if kwargs.get("do_finalize", True):
            return [torch.zeros(args[2].shape, dtype=torch.bfloat16)]
        # as the launcher (RoutingMetadataBuffers): GEMM2's padded permuted rows, the routing weights
        # [tokens, top_k] and the expanded index flat, [tokens * top_k]
        return [torch.zeros(tokens * top_k + 8, args[2].shape[1], dtype=torch.bfloat16),
                torch.zeros(tokens, top_k, dtype=torch.bfloat16), torch.zeros(tokens * top_k, dtype=torch.int32)]

    class RoutingMethodType(enum.IntEnum):
        Default, Renormalize, DeepSeekV3, Llama4, RenormalizeNaive, TopK = range(6)

    class SfLayout(enum.IntEnum):
        layout_128x4, layout_8x4, layout_linear = range(3)

    names = ("flashinfer", "flashinfer.fused_moe", "flashinfer.fused_moe.core", "flashinfer.tllm_enums",
             "flashinfer.quantization", "flashinfer.quantization.fp8_quantization", "flashinfer.autotuner",
             "flashinfer.fused_moe.utils")
    mods = {name: types.ModuleType(name) for name in names}
    mods["flashinfer"].__version__ = "0.7.2"
    mods["flashinfer.fused_moe.core"].trtllm_fp4_block_scale_routed_moe = routed_moe
    mods["flashinfer.fused_moe.core"].trtllm_fp4_block_scale_moe = routed_moe  # the monolithic op: same recorder
    tuning: list = []

    class autotune:  # noqa: N801 - FlashInfer's name
        def __init__(self, tune_mode=True, cache=None, tuning_buckets=None):
            self.args = dict(tune_mode=tune_mode, cache=cache, tuning_buckets=tuning_buckets)

        def __enter__(self):
            tuning.append(self.args)
            self.args["calls_before"] = len(calls)

        def __exit__(self, *exc):
            self.args["calls_inside"] = len(calls) - self.args["calls_before"]

    mods["flashinfer.autotuner"].autotune = autotune
    # FlashInfer 0.7.2's hybrid buckets under 8192 tokens
    mods["flashinfer.fused_moe.utils"].get_hybrid_num_tokens_buckets = lambda max_tokens: (
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 8192)
    calls.tuning = tuning
    mods["flashinfer.tllm_enums"].RoutingMethodType = RoutingMethodType
    mods["flashinfer.tllm_enums"].SfLayout = SfLayout
    mods["flashinfer.quantization.fp8_quantization"].mxfp8_quantize = (
        lambda x, is_sf_swizzled_layout=True: (x, torch.zeros(x.shape[0] * -(-x.shape[1] // 32), dtype=torch.uint8)))
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_flashinfers_fp4_moe_gets_the_routers_weights_as_the_hf_reference_computes_them(monkeypatch):
    """K3.1 routes noaux_tc: sigmoid scores, the top-k by score + bias, their unbiased scores renormalized
    (``moe_renormalize``: true in the release) and times ``routed_scaling_factor`` (1.0 in the release; 2.5 here,
    so a dropped or doubled factor shows), as the HF reference does. FlashInfer's routed MoE gets those ids and
    weights as computed, in its precomputed form, where its routing runs only the post-top-k pipeline and no
    routing method's softmax or renormalization touches them; the wrapper names the method that means that,
    ``TopK``. With the EP sentinel id the expert count it passes stays a multiple of 4 past the sentinel."""
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from mstar.utils.fused_moe import fp4_flashinfer

    release = json.loads((Path(reference.__file__).parent / "fixtures" / "k31_config.json").read_text())
    k31 = KimiK3ModelConfig.from_hf_config(release, strict=False)
    assert k31.moe_renormalize and k31.routed_scaling_factor == 1.0

    cfg = KimiK3ModelConfig.reduced(routed_scaling_factor=2.5)
    block, sd = _loaded_latent_moe(cfg)
    x = torch.randn(7, cfg.hidden_size, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        weights, ids, _ = block.gate(x)
    ref_ids, ref_w = reference.route(sd, "model.layers.1.block_sparse_moe.", x, cfg)

    def dense(i, w):
        return torch.zeros(x.shape[0], cfg.num_experts).scatter(1, i.long(), w.float())

    torch.testing.assert_close(dense(ids, weights), dense(ref_ids, ref_w), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights.float().sum(-1), torch.full((7,), 2.5))

    calls = _fake_flashinfer(monkeypatch)
    e, two_inter, latent = cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.routed_expert_hidden_size
    u8 = torch.zeros(1, dtype=torch.uint8)
    shim = SimpleNamespace(
        experts=SimpleNamespace(gate_up_proj_packed=u8, down_proj_packed=u8, gate_up_proj_scale=u8,
                                down_proj_scale=u8),
        _experts_dims=lambda: (e, two_inter, latent))
    kernel = fp4_flashinfer.TrtllmFp4Experts(shim, "mxfp4", "mxfp8")
    kernel.gated = (10, torch.full((e,), 4.0), torch.full((e,), 25.0))
    kernel.scalars = (None, None, None)
    z = torch.randn(7, latent).bfloat16()
    kernel(z, weights, ids, False)
    (args, kwargs), = calls
    got_ids, got_w = args[0]
    assert got_ids.dtype == torch.int32 and torch.equal(got_ids, ids.to(torch.int32))
    assert got_w.dtype == torch.float32 and torch.equal(got_w, weights.float())
    assert kwargs["routing_method_type"] == 5  # TopK: no pre- or post-processing of the given weights
    assert kwargs["hidden_states_scale_layout"] == 2  # linear, as _hidden quantizes them
    assert (args[16], args[17], args[21], args[22]) == (e, cfg.num_experts_per_tok, 0, e)

    kernel(z, weights, ids, True)
    sentinel_count = calls[-1][0][16]
    assert sentinel_count % 4 == 0 and sentinel_count > e and calls[-1][0][22] == e
    assert fp4_flashinfer.routing_experts(112, True) == 116  # EP8: 113 would fail the routing kernel's check
    assert fp4_flashinfer.routing_experts(896, False) == 896


def test_the_fp4_moe_is_tuned_once_at_load_over_the_buckets_its_calls_look_up(monkeypatch, tmp_path):
    """vLLM's boot autotune: one call at the largest bucket inside ``flashinfer.autotune`` over the served buckets
    (default: every FlashInfer bucket, decode 1..256 and prefill 512..8192), the tactics saved per GPU; every
    call, the tuning one and the served ones, passes the same ``tune_max_num_tokens``, so a call's bucket is the
    key it was tuned under. Once per process and shape (~90 MoE layers share it); a count that is no FlashInfer
    bucket is refused."""
    from types import SimpleNamespace

    from mstar.utils.fused_moe import fp4_flashinfer

    calls = _fake_flashinfer(monkeypatch)
    monkeypatch.setattr(fp4_flashinfer, "_TUNED", {})
    monkeypatch.setattr(fp4_flashinfer, "_tune_cache_path", lambda key, device: str(tmp_path / f"{key}.json"))
    e, two_inter, latent = 16, 256, 128
    u8 = torch.zeros(1, dtype=torch.uint8)
    block = SimpleNamespace(
        experts=SimpleNamespace(gate_up_proj_packed=u8, down_proj_packed=u8, gate_up_proj_scale=u8,
                                down_proj_scale=u8),
        _experts_dims=lambda: (e, two_inter, latent), num_experts_per_tok=4)
    kernel = fp4_flashinfer.TrtllmFp4Experts(block, "mxfp4", "mxfp8")
    kernel.gated = (10, torch.full((e,), 4.0), torch.full((e,), 25.0))
    kernel.scalars = (None, None, None)
    cpu = torch.device("cpu")

    summary = fp4_flashinfer.autotune(block, kernel, cpu)
    assert summary.startswith("autotuned 21 buckets 1..8192"), summary
    assert 3072 in fp4_flashinfer.TUNE_TOKENS and fp4_flashinfer.tune_buckets() == fp4_flashinfer.TUNE_TOKENS
    (tuned,) = calls.tuning
    assert tuned["tune_mode"] and tuned["tuning_buckets"] == fp4_flashinfer.TUNE_TOKENS
    assert tuned["cache"] == str(tmp_path / "mxfp4xmxfp8_E16_I128_H128_K4.json") and tuned["calls_inside"] == 1
    args, kwargs = calls[0]
    assert args[2].shape == (8192, latent) and args[0][0].shape == (8192, 4)
    assert kwargs["tune_max_num_tokens"] == fp4_flashinfer.TUNE_MAX_TOKENS

    assert fp4_flashinfer.autotune(block, kernel, cpu) == summary and len(calls) == 1  # once per shape
    kernel(torch.randn(3, latent).bfloat16(), torch.rand(3, 4).softmax(-1), torch.randint(0, e, (3, 4)), False)
    assert calls[-1][1]["tune_max_num_tokens"] == fp4_flashinfer.TUNE_MAX_TOKENS  # served: the same lookup

    fp4_flashinfer.autotune(block, kernel, cpu, tokens=(16, 64), skip_invalid=True)
    assert calls.tuning[-1]["tuning_buckets"] == (16, 64) and calls[-1][0][2].shape[0] == 64
    assert calls.tuning[-1]["cache"].endswith("_ep.json")
    assert "autotune skipped" in fp4_flashinfer.autotune(block, kernel, cpu, tokens=(100,))

    # routing in the kernel: the served call is from_logits; its unfinalized form (the finalize fold's) is
    # another key in FlashInfer's tuner, tuned into the same file (a cache file on entry drops any other's)
    kernel.routing = fp4_flashinfer.KernelRouting(torch.zeros(e), 4, 1.0)
    fp4_flashinfer.autotune(block, kernel, cpu, tokens=(1, 2, 4))
    assert calls[-1][1]["do_finalize"] is True and calls.tuning[-1]["cache"].endswith("_K4_dsv3.json")
    assert "autotuned 3 buckets" in fp4_flashinfer.autotune(block, kernel, cpu, tokens=(1, 2, 4), finalize=False)
    assert calls[-1][1]["do_finalize"] is False and calls.tuning[-1]["cache"].endswith("_K4_dsv3.json")
    assert calls.tuning[-1]["tuning_buckets"] == (1, 2, 4) and calls[-1][0][2].shape[0] == 4
    kernel.routing = None
    assert "autotune skipped" in fp4_flashinfer.autotune(block, kernel, cpu, finalize=False)


def test_the_load_tunes_the_fp4_moe_with_the_blocks_settings(monkeypatch):
    """``experts_after_load`` tunes the converted kernel when the block asks (Kimi K3: model_kwargs
    ``routed_experts_autotune``, default on, over ``routed_experts_tune_tokens``) and puts the summary in the
    backend line; an expert parallel block tunes the sentinel call it serves."""
    from mstar.model.components import moe
    from mstar.utils.fused_moe import fp4_flashinfer

    cfg = KimiK3ModelConfig.reduced(routed_experts_tune_tokens=[16, 64])
    assert cfg.routed_experts_autotune and cfg.routed_experts_tune_tokens == (16, 64)
    block = KimiLatentMoE(cfg)
    assert block.routed_experts_autotune and block.routed_experts_tune_tokens == (16, 64)
    assert KimiLatentMoE(KimiK3ModelConfig.reduced()).routed_experts_tune_tokens is None

    seen, lines = [], []
    from types import SimpleNamespace

    kernel = SimpleNamespace(routing=None)
    monkeypatch.setattr(fp4_flashinfer, "choose", lambda block_, device: ("mxfp8", "probe ok"))
    monkeypatch.setattr(fp4_flashinfer, "probe_routing", lambda block_, device, activation, routing: (None, "ok"))
    monkeypatch.setattr(fp4_flashinfer.TrtllmFp4Experts, "convert",
                        classmethod(lambda cls, block_, device, activation="bf16": kernel))
    monkeypatch.setattr(fp4_flashinfer, "autotune", lambda block_, kernel, device, tokens, skip_invalid: (
        seen.append((kernel, tokens, skip_invalid)) or "autotuned 2 buckets"))
    monkeypatch.setattr(moe, "_EXPERTS_BACKEND_LOGGED", set())
    monkeypatch.setattr(moe.logger, "info", lambda msg, *a: lines.append(msg % a))
    block.experts_packed = type("Fmt", (), {"name": "mxfp4"})()
    monkeypatch.setattr(type(block), "_experts_dims", lambda self: (cfg.num_experts, 2 * cfg.moe_intermediate_size,
                                                                    cfg.routed_expert_hidden_size))
    block.experts_after_load()
    # the router in the kernel: its calls carry no sentinel (the padding mask can't reach them), tuned without
    assert seen == [(kernel, (16, 64), False)] and block._experts_kernel is kernel
    assert kernel.routing is not None and kernel.routing.top_k == cfg.num_experts_per_tok
    assert "probe ok; routing in the kernel (DeepSeekV3, one group); autotuned 2 buckets" in lines[0], lines
    block.routed_experts_autotune = False
    block.experts_after_load()
    assert len(seen) == 1


def test_the_kernels_dsv3_routing_is_k31s_router():
    """FlashInfer's DeepSeekV3 routing with one expert group (routingCustom: SigmoidBias, ScaledSumNormalize) is
    K3.1's noaux_tc exactly: experts by sigmoid + e_score_correction_bias, weights the selected sigmoid scores
    renormalized times routed_scaling_factor (2.5 here), as the HF reference routes (``deepseek_v3_route`` is the
    formula the load probe checks the kernel against on the GPU)."""
    from mstar.utils.fused_moe import fp4_flashinfer

    cfg = KimiK3ModelConfig.reduced(routed_scaling_factor=2.5)
    block, sd = _loaded_latent_moe(cfg)
    with torch.no_grad():
        block.gate.e_score_correction_bias.copy_(torch.randn(cfg.num_experts) * 0.5)
        sd["model.layers.1.block_sparse_moe.gate.e_score_correction_bias"] = block.gate.e_score_correction_bias.clone()
    x = torch.randn(9, cfg.hidden_size, generator=torch.Generator().manual_seed(5))
    ref_ids, ref_w = reference.route(sd, "model.layers.1.block_sparse_moe.", x, cfg)
    logits = F.linear(x.float(), block.gate.weight.float())
    w, ids = fp4_flashinfer.deepseek_v3_route(logits, block.gate.e_score_correction_bias, cfg.num_experts_per_tok, 2.5)

    def dense(i, v):
        return torch.zeros(x.shape[0], cfg.num_experts).scatter(1, i.long(), v.float())

    torch.testing.assert_close(dense(ids, w), dense(ref_ids, ref_w), atol=1e-6, rtol=1e-6)
    routing, why = fp4_flashinfer.kernel_routing(block)
    assert why == "" and routing.top_k == cfg.num_experts_per_tok and routing.scale == 2.5
    assert torch.equal(routing.bias, block.gate.e_score_correction_bias.float())


def test_kernel_routing_takes_only_a_tp_block_with_a_renormalizing_noaux_tc_router():
    """DeepSeekV3 routing always renormalizes, sees every expert's logit and takes no sentinel: a router that
    doesn't renormalize, an expert parallel block, or a top-k past the kernel's (22 over 384 experts) keep the
    block's own routing."""
    from types import SimpleNamespace

    from mstar.utils.fused_moe import fp4_flashinfer

    assert fp4_flashinfer.kernel_routing(KimiLatentMoE(KimiK3ModelConfig.reduced(moe_renormalize=False)))[0] is None
    gate = SimpleNamespace(e_score_correction_bias=torch.zeros(896), routed_scaling_factor=1.0, norm_topk_prob=True,
                           top_k=16)
    tp = SimpleNamespace(gate=gate, num_experts=896, _experts_dims=lambda: (896, 768, 3584))
    assert fp4_flashinfer.kernel_routing(tp)[0] is not None  # K3.1 TP8: 896 experts, top-16
    ep = SimpleNamespace(gate=gate, num_experts=896, num_local_experts=112, _experts_dims=lambda: (112, 6144, 3584))
    assert "expert parallel" in fp4_flashinfer.kernel_routing(ep)[1]
    wide = SimpleNamespace(gate=SimpleNamespace(**{**vars(gate), "top_k": 24}), num_experts=896,
                           _experts_dims=lambda: (896, 768, 3584))
    assert fp4_flashinfer.kernel_routing(wide)[0] is None
    softmax = SimpleNamespace(gate=SimpleNamespace(top_k=8), num_experts=64, _experts_dims=lambda: (64, 256, 128))
    assert fp4_flashinfer.kernel_routing(softmax)[0] is None


def test_from_logits_hands_the_routing_to_flashinfers_dsv3_kernel(monkeypatch):
    """``from_logits``: FlashInfer's monolithic FP4 MoE routes from the router's fp32 logits with the DeepSeekV3
    method, one group (n_group = topk_group = 1), the router's bias, top-k and scaling, over every expert of the
    rank, with the tuning range every call shares."""
    from types import SimpleNamespace

    from mstar.utils.fused_moe import fp4_flashinfer

    calls = _fake_flashinfer(monkeypatch)
    e, two_inter, latent, top_k = 896, 768, 128, 16
    u8 = torch.zeros(1, dtype=torch.uint8)
    shim = SimpleNamespace(
        experts=SimpleNamespace(gate_up_proj_packed=u8, down_proj_packed=u8, gate_up_proj_scale=u8,
                                down_proj_scale=u8),
        _experts_dims=lambda: (e, two_inter, latent))
    kernel = fp4_flashinfer.TrtllmFp4Experts(shim, "mxfp4", "mxfp8")
    kernel.gated = (10, torch.full((e,), 4.0), torch.full((e,), 25.0))
    kernel.scalars = (None, None, None)
    bias = torch.randn(e)
    kernel.routing = fp4_flashinfer.KernelRouting(bias, top_k, 1.0)
    packed = torch.randn(3, e + 40)  # the logits as the packed GEMM leaves them: a column slice
    kernel.from_logits(torch.randn(3, latent).bfloat16(), packed[:, :e])
    (args, kwargs), = calls
    assert args[0].dtype == torch.float32 and args[0].is_contiguous() and torch.equal(args[0], packed[:, :e])
    assert args[1] is bias
    assert (args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23]) == (
        e, top_k, 1, 1, two_inter // 2, 0, e, 1.0)
    assert kwargs["routing_method_type"] == 2  # DeepSeekV3
    assert kwargs["tune_max_num_tokens"] == fp4_flashinfer.TUNE_MAX_TOKENS and kwargs["do_finalize"] is True


def test_the_latent_moe_leaves_routing_to_a_kernel_that_routes(monkeypatch):
    """With the experts kernel routing from the logits, the block's router never runs: the packed input GEMM's
    fp32 logits go to ``from_logits`` (no top-k, gather or renormalize launches); without it, the router routes
    as before."""
    from types import SimpleNamespace

    cfg = KimiK3ModelConfig.reduced()
    block, _ = _loaded_latent_moe(cfg)
    block.process_weights_after_loading(None)
    assert block._in_weight is not None
    x = torch.randn(5, cfg.hidden_size, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        want = block(x)
    got_logits = []

    def from_logits(z, logits):
        got_logits.append(logits)
        routing_weights, selected, _ = block.gate.__class__.route(block.gate, logits)
        return block.experts_forward(z, routing_weights, selected, reduce_results=False)

    def no_router(*a, **k):
        raise AssertionError("the block's router ran")

    block._experts_kernel = SimpleNamespace(routing=object(), from_logits=from_logits)
    monkeypatch.setattr(block.gate, "route", no_router)
    with torch.no_grad():
        got = block(x)
    (logits,) = got_logits
    torch.testing.assert_close(logits, F.linear(x, block.gate.weight), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def _finalize_ref(permuted, idx, weights):
    """The MoE finalize: each token's top-k rows of the permuted buffer, weighted, summed."""
    return (permuted[idx.long()].float() * weights.float()[..., None]).sum(1)


def test_the_routed_finalize_fold_is_one_flashinfer_call(monkeypatch):
    """``ar_fusion.allreduce_moe_finalize_rmsnorm``: FlashInfer's kMoEFinalizeARResidualRMSNorm on its TRT-LLM
    workspace, the MoE's permuted rows with their expanded index and routing weights, a zero residual, the norm's
    weight and eps; no PDL (its trigger can't be held to the end)."""
    import sys
    import types

    from mstar.distributed import ar_fusion

    calls = []
    comm = types.ModuleType("flashinfer.comm")
    comm.AllReduceFusionPattern = types.SimpleNamespace(kMoEFinalizeARResidualRMSNorm=7)
    comm.allreduce_fusion = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.comm", comm)
    monkeypatch.setattr(ar_fusion, "_WORKSPACES", ["trtllm ws"])
    permuted, idx = torch.randn(40, 48).bfloat16(), torch.arange(12, dtype=torch.int32).view(3, 4)
    scales, residual, weight = torch.rand(3, 4).bfloat16(), torch.zeros(3, 48).bfloat16(), torch.ones(48).bfloat16()
    out = ar_fusion.allreduce_moe_finalize_rmsnorm(permuted, idx, scales, residual, weight, 1e-6, 0.0, 0)
    (kw,) = calls
    assert kw["pattern"] == 7 and kw["workspace"] == "trtllm ws" and kw["launch_with_pdl"] is False
    assert kw["input"] is permuted and kw["expanded_idx_to_permuted_idx"] is idx and kw["expert_scale_factor"] is scales
    assert kw["residual_in"] is residual and kw["rms_gamma"] is weight and kw["rms_eps"] == 1e-6
    assert kw["norm_out"] is out and out.shape == (3, 48)


def test_the_comm_group_folds_only_into_its_own_workspace(monkeypatch):
    """``CommGroup.allreduce_moe_finalize_rmsnorm`` takes its TRT-LLM workspace's zero residual, and says None
    (the caller finalizes in the MoE) without one or past its tokens."""
    from mstar.distributed import ar_fusion
    from mstar.distributed.communication import ArFusionWorkspace, CommGroup

    group = CommGroup.trivial()
    group.world_size = 2
    permuted, idx = torch.randn(40, 48).bfloat16(), torch.zeros(3, 4, dtype=torch.int32)
    scales, weight = torch.rand(3, 4).bfloat16(), torch.ones(48).bfloat16()
    assert group.allreduce_moe_finalize_rmsnorm(permuted, idx, scales, weight, 1e-6) is None
    assert group.moe_finalize_max_tokens(48, torch.bfloat16) == 0
    zeros = torch.zeros(4, 48, dtype=torch.bfloat16)
    group._moe_finalize[(torch.bfloat16, 48)] = ArFusionWorkspace(5, 4, zeros)
    seen = []
    monkeypatch.setattr(ar_fusion, "allreduce_moe_finalize_rmsnorm",
                        lambda *a: seen.append(a) or torch.full((a[1].shape[0], 48), 2.0))
    out = group.allreduce_moe_finalize_rmsnorm(permuted, idx, scales, weight, 1e-6, 0.0)
    (args,) = seen
    assert out.shape == (3, 48) and args[0] is permuted and args[3].data_ptr() == zeros.data_ptr()
    assert args[3].shape == (3, 48) and args[5:] == (1e-6, 0.0, 5)
    assert group.allreduce_moe_finalize_rmsnorm(permuted, torch.zeros(5, 4, dtype=torch.int32), torch.rand(5, 4),
                                                weight, 1e-6) is None  # past the workspace's 4 tokens


def test_the_fold_refuses_what_its_kernel_would_index_past(monkeypatch):
    """``CommGroup.moe_finalize_unfit``, on the host before any launch (a bad index in FlashInfer's kernel is a
    sticky illegal address): a flat expanded index (the kernel would take tokens x top_k as top_k), routing
    weights of another dtype or shape, fewer permuted rows than token slots, strided rows, a norm weight of
    another dtype, a group size the kernel has no instance for. ``allreduce_moe_finalize_rmsnorm`` raises on
    any of them without calling the op."""
    from mstar.distributed import ar_fusion
    from mstar.distributed.communication import ArFusionWorkspace, CommGroup

    group = CommGroup.trivial()
    group.world_size = 8
    group._moe_finalize[(torch.bfloat16, 48)] = ArFusionWorkspace(5, 4, torch.zeros(4, 48, dtype=torch.bfloat16))
    seen = []
    monkeypatch.setattr(ar_fusion, "allreduce_moe_finalize_rmsnorm", lambda *a: seen.append(a))
    permuted, idx = torch.randn(40, 48).bfloat16(), torch.zeros(3, 4, dtype=torch.int32)
    scales, weight = torch.rand(3, 4).bfloat16(), torch.ones(48).bfloat16()
    assert group.moe_finalize_unfit(permuted, idx, scales, weight) is None
    bad = {
        "expanded index": (permuted, idx.flatten(), scales, weight),  # slot 8: [1 x 16] read as 16 tokens x 16
        "routing weights": (permuted, idx, scales.float(), weight),
        "permuted rows": (permuted[:11], idx, scales, weight),
        "not [rows, hidden] dense": (torch.randn(48, 40).bfloat16().t(), idx, scales, weight),
        "norm weight": (permuted, idx, scales, weight.float()),
    }
    for what, args in bad.items():
        assert what in group.moe_finalize_unfit(*args), what
        with pytest.raises(ValueError, match="allreduce_moe_finalize_rmsnorm"):
            group.allreduce_moe_finalize_rmsnorm(*args, 1e-6)
    assert seen == []
    group.world_size = 6
    assert "6 ranks" in group.moe_finalize_unfit(permuted, idx, scales, weight)


def test_from_logits_can_leave_the_finalize_to_the_caller(monkeypatch):
    """``from_logits(finalize=False)``: the MoE stops before its top-k sum (do_finalize False) and hands back
    its permuted rows, routing weights and expanded index, the last as ``[tokens, top_k]`` (FlashInfer's is flat,
    and the fold kernel takes top_k off its last dim: B300 slot 8 read past it at 1 token)."""
    from types import SimpleNamespace

    from mstar.utils.fused_moe import fp4_flashinfer

    calls = _fake_flashinfer(monkeypatch)
    e, two_inter, latent = 64, 256, 128
    u8 = torch.zeros(1, dtype=torch.uint8)
    shim = SimpleNamespace(experts=SimpleNamespace(gate_up_proj_packed=u8, down_proj_packed=u8,
                                                   gate_up_proj_scale=u8, down_proj_scale=u8),
                           _experts_dims=lambda: (e, two_inter, latent))
    kernel = fp4_flashinfer.TrtllmFp4Experts(shim, "mxfp4", "mxfp8")
    kernel.gated, kernel.scalars = (10, None, None), (None, None, None)
    kernel.routing = fp4_flashinfer.KernelRouting(torch.zeros(e), 4, 1.0)
    out = kernel.from_logits(torch.randn(3, latent).bfloat16(), torch.randn(3, e), finalize=False)
    assert calls[-1][1]["do_finalize"] is False
    permuted, weights, idx = out
    assert weights.shape == idx.shape == (3, 4) and idx.dtype == torch.int32 and permuted.shape == (20, latent)
    assert kernel.from_logits(torch.randn(3, latent).bfloat16(), torch.randn(3, e)).shape == (3, latent)
    assert calls[-1][1]["do_finalize"] is True
    # z already MXFP8 from the cast (``takes_mxfp8``: values, scales): the tokens come from the logits
    z = (torch.zeros(3, latent, dtype=torch.float8_e4m3fn), torch.zeros(3, latent // 32, dtype=torch.uint8))
    _, weights, idx = kernel.from_logits(z, torch.randn(3, e), finalize=False)
    assert weights.shape == idx.shape == (3, 4)


class _FoldGroup:
    """A two-rank TP group's surface for the fold, run on one process: the all-reduce is the identity."""

    world_size = 2

    def __init__(self, routes=(1,), ok=True):
        self.votes, self.enabled, self.folded, self._routes, self._ok = [], [], [], list(routes), ok
        self.unfit = None  # this rank's moe_finalize_unfit answer

    def agree(self, values):
        self.votes.append(values)
        return [min(values + [self._routes.pop(0) if self._routes else 1])]

    def enable_moe_finalize_fusion(self, hidden, dtype, max_tokens):
        self.enabled.append((hidden, dtype, max_tokens))
        return self._ok

    def all_reduce(self, x):
        return x

    def allreduce_rmsnorm(self, x, weight, eps, weight_bias=0.0):
        from mstar.model.components.norm import rms_norm

        return rms_norm(x, weight, eps, False)

    def moe_finalize_unfit(self, permuted, idx, weights, weight):
        return self.unfit

    def allreduce_moe_finalize_rmsnorm(self, permuted, idx, weights, weight, eps, weight_bias=0.0):
        from mstar.model.components.norm import rms_norm

        self.folded.append(idx.shape[0])
        return rms_norm(_finalize_ref(permuted, idx, weights).to(weight.dtype), weight, eps, False)


def _fold_block(monkeypatch, group, fold=True):
    """A reduced K3.1 latent MoE whose experts kernel routes from the logits and (``finalize=False``) hands back
    a permuted buffer of its top-k rows, under ``group``."""
    from types import SimpleNamespace

    from mstar.utils.fused_moe import fp4_flashinfer

    cfg = KimiK3ModelConfig.reduced()
    block, _ = _loaded_latent_moe(cfg)
    block.process_weights_after_loading(None)
    route = fp4_flashinfer.kernel_routing(block)[0]
    calls = []

    def from_logits(z, logits, finalize=True):
        calls.append(finalize)
        w, _ = fp4_flashinfer.deepseek_v3_route(logits, route.bias, route.top_k, route.scale)
        k = route.top_k
        # stand-ins for the experts' outputs: each token's k rows of the permuted buffer, all different
        rows = z.repeat_interleave(k, 0) * torch.arange(1, k + 1, dtype=z.dtype).repeat(z.shape[0])[:, None]
        idx = torch.arange(z.shape[0] * k, dtype=torch.int32).view(z.shape[0], k)
        if not finalize:
            return rows, w, idx
        return _finalize_ref(rows, idx, w).to(z.dtype)

    block._experts_kernel = SimpleNamespace(routing=route, from_logits=from_logits)
    monkeypatch.setattr(block, "comm_group", group)
    block._fold_finalize = fold
    return cfg, block, calls


def test_the_latent_moe_folds_its_finalize_into_the_routed_all_reduce(monkeypatch):
    """With the fold, the MoE stops before its finalize and the routed all-reduce + RMSNorm sums its top-k
    (one call, its routing weights and expanded index), up to FOLD_MAX_TOKENS; past them the MoE finalizes and
    the fused all-reduce + RMSNorm reduces. Either way the block's output is the unfolded one."""
    from mstar.model.kimi_k3.components import moe as kmoe

    monkeypatch.setattr(kmoe, "FOLD_MAX_TOKENS", 4)
    group = _FoldGroup()
    cfg, block, calls = _fold_block(monkeypatch, group)
    x = torch.randn(3, cfg.hidden_size, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        folded = block(x)
    assert calls == [False] and group.folded == [3]
    block._fold_finalize = False
    with torch.no_grad():
        plain = block(x)
    assert calls == [False, True]
    torch.testing.assert_close(folded, plain, atol=2e-2, rtol=2e-2)
    block._fold_finalize = True
    with torch.no_grad():
        block(torch.randn(6, cfg.hidden_size))  # past FOLD_MAX_TOKENS: the MoE finalizes
    assert calls[-1] is True and group.folded == [3]


def test_the_fold_needs_every_rank_routing_in_the_kernel_and_a_probe(monkeypatch):
    """``_setup_finalize_fold`` (collective): every rank must route in its kernel (one vote per block), the
    TRT-LLM workspace must build and the probe agree, decided once on the first block; a single rank, no
    norm, ``routed_experts_finalize: kernel`` or a rank routing in torch keep the MoE's finalize."""
    from mstar.model.kimi_k3.components import moe as kmoe

    monkeypatch.setattr(kmoe, "_FOLD_VERDICT", {})
    group = _FoldGroup()
    _, block, _ = _fold_block(monkeypatch, group, fold=False)
    block._setup_finalize_fold()
    assert not block._fold_finalize and not group.votes  # the default ("kernel"): no vote, no probe
    block.routed_experts_finalize = "fused"
    block.experts_packed = type("Fmt", (), {"name": "mxfp4"})()
    probes = []
    monkeypatch.setattr(type(block), "_probe_fold", lambda self: probes.append(1) or (True, "probe ok"))
    monkeypatch.setattr(type(block), "_tune_unfinalized", lambda self: "tuned")
    block._setup_finalize_fold()
    assert block._fold_finalize and group.enabled == [(48, torch.float32, kmoe.FOLD_MAX_TOKENS)] and probes == [1]
    block._setup_finalize_fold()  # a later block: its vote, not another workspace or probe
    assert block._fold_finalize and len(group.enabled) == 1 and probes == [1] and len(group.votes) == 2
    group._routes = [0]  # another rank's kernel routes in torch
    block._setup_finalize_fold()
    assert not block._fold_finalize
    block.routed_experts_finalize = "kernel"
    block._setup_finalize_fold()
    assert not block._fold_finalize and len(group.votes) == 3  # no vote: the config is every rank's
    with pytest.raises(ValueError, match="routed_experts_finalize"):
        KimiK3ModelConfig.reduced(routed_experts_finalize="maybe")
    assert KimiLatentMoE(KimiK3ModelConfig.reduced()).routed_experts_finalize == "kernel"


def test_the_fold_probe_launches_nothing_a_rank_could_fault_on(monkeypatch):
    """``_probe_fold`` (collective) launches the fold only once every rank's unfinalized MoE output passed the
    kernel's contract on the host and its expanded index (each a permuted row or -1) on the device, by vote:
    B300 slot 8's probe ran the kernel on a flat index, faulted (sticky), and the "fallback" died with it. A
    rank that says no, or another rank's no, launches nothing; a device fault stops the load, saying so."""
    from mstar.model.kimi_k3.components import moe as kmoe

    monkeypatch.setattr(kmoe, "FOLD_MAX_TOKENS", 4)
    group = _FoldGroup()
    _, block, calls = _fold_block(monkeypatch, group, fold=False)
    ok, detail = block._probe_fold()
    assert ok and group.folded == [1, 32, 4] and "probe rel L2" in detail, detail
    assert len(group.votes) == 4  # one before each launch, one on the result

    group = _FoldGroup()
    group.unfit = "expanded index (16,) torch.int32, not [tokens, top_k] int32"
    monkeypatch.setattr(block, "comm_group", group)
    ok, detail = block._probe_fold()
    assert not ok and group.folded == [] and "not launched at 1 tokens: expanded index (16,)" in detail
    group = _FoldGroup(routes=(0,))  # another rank said no
    monkeypatch.setattr(block, "comm_group", group)
    ok, detail = block._probe_fold()
    assert not ok and group.folded == [] and detail.endswith("another rank")

    route = block._experts_kernel.routing
    good = block._experts_kernel.from_logits

    def wild_index(z, logits, finalize=True):
        out = good(z, logits, finalize)
        return out if finalize else (out[0], out[1], out[2] + 10_000)

    group = _FoldGroup()
    monkeypatch.setattr(block, "comm_group", group)
    block._experts_kernel = type("K", (), {"routing": route, "from_logits": staticmethod(wild_index)})()
    ok, detail = block._probe_fold()
    assert not ok and group.folded == [] and "outside the" in detail

    def faulted(z, logits, finalize=True):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    block._experts_kernel = type("K", (), {"routing": route, "from_logits": staticmethod(faulted)})()
    with pytest.raises(RuntimeError, match="routed_experts_finalize: kernel"):
        block._probe_fold()


def test_the_fold_tunes_its_unfinalized_call(monkeypatch):
    """The unfinalized MoE call is its own entry in FlashInfer's tuner (a ``(tokens, 0)`` output): with boot
    tuning on, the fold tunes it at the tune tokens it serves (up to FOLD_MAX_TOKENS), finalize False."""
    from mstar.model.kimi_k3.components import moe as kmoe
    from mstar.utils.fused_moe import fp4_flashinfer

    _, block, _ = _fold_block(monkeypatch, _FoldGroup(), fold=False)
    seen = []
    monkeypatch.setattr(fp4_flashinfer, "autotune",
                        lambda blk, kernel, device, tokens, finalize=True: seen.append((tokens, finalize)) or "ok")
    block.routed_experts_autotune, block.routed_experts_tune_tokens = True, None
    assert block._tune_unfinalized() == "unfinalized call ok"
    assert seen == [([t for t in fp4_flashinfer.TUNE_TOKENS if t <= kmoe.FOLD_MAX_TOKENS], False)]
    block.routed_experts_autotune = False
    assert block._tune_unfinalized() == "" and len(seen) == 1


def test_routed_experts_backend_is_a_serving_switch():
    """``routed_experts_backend`` (model_kwargs): auto, or trtllm / marlin to force one; the MoE block carries it
    to ``QuantExperts.experts_after_load``."""
    assert KimiK3ModelConfig.reduced().routed_experts_backend == "auto"
    for value in ("auto", "trtllm", "marlin"):
        assert KimiLatentMoE(KimiK3ModelConfig.reduced(routed_experts_backend=value)).routed_experts_backend == value
    with pytest.raises(ValueError, match="routed_experts_backend"):
        KimiK3ModelConfig.reduced(routed_experts_backend="cutlass")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton MXFP4 experts kernel needs CUDA")
def test_triton_mxfp4_experts_take_situ():
    """The quantization API's MXFP4 experts kernel (``fused_experts_w4a16``, e2m1 dequantized in
    registers) with SiTU's spec, against the dequantized experts run in torch with SiTU."""
    from mstar.model.components.int4_formats import KIND_E2M1
    from mstar.model.components.moe import dispatch_experts_fused
    from mstar.model.kimi_k3.truncate import dequant_mxfp4
    from mstar.utils.fused_moe.w4a16 import fused_experts_w4a16

    g = torch.Generator(device="cuda").manual_seed(0)
    e, hidden, inter, tokens, top_k = 8, 256, 128, 37, 4
    w1 = torch.randint(0, 256, (e, 2 * inter, hidden // 2), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (e, hidden, inter // 2), dtype=torch.uint8, device="cuda", generator=g)
    s1 = torch.randint(120, 125, (e, 2 * inter, hidden // 32), dtype=torch.uint8, device="cuda", generator=g)
    s2 = torch.randint(120, 125, (e, hidden, inter // 32), dtype=torch.uint8, device="cuda", generator=g)
    x = torch.randn(tokens, hidden, device="cuda", generator=g).bfloat16()
    ids = torch.stack([torch.randperm(e, device="cuda", generator=g)[:top_k] for _ in range(tokens)])
    weights = torch.rand(tokens, top_k, device="cuda", generator=g).softmax(-1)
    act = SiTU(4.0, 25.0)
    out = fused_experts_w4a16(x, w1, w2, s1, s2, None, None, weights, ids, KIND_E2M1, 32, act.spec, False)
    d1 = torch.stack([dequant_mxfp4(w1[i].cpu(), s1[i].cpu()) for i in range(e)]).cuda()
    d2 = torch.stack([dequant_mxfp4(w2[i].cpu(), s2[i].cpu()) for i in range(e)]).cuda()
    ref = dispatch_experts_fused(x.float(), d1.float(), d2.float(), e, ids, weights, activation=act)
    torch.testing.assert_close(out.float(), ref, atol=5e-2 * ref.abs().max().item(), rtol=5e-2)
    silu = fused_experts_w4a16(x, w1, w2, s1, s2, None, None, weights, ids, KIND_E2M1, 32, "silu", False)
    assert (silu.float() - ref).abs().max() > 0.1 * ref.abs().max()


def test_blackwell_fp4_moe_takes_situ_with_both_betas():
    """FlashInfer's TRT-LLM FP4 MoE (B200/B300) runs SwiGLU and SiTU with both betas (its
    activation_type Situ); any other activation keeps MXFP4 experts on the Triton kernel. So does
    SiTU on MXFP4 with bf16 activations, Kimi's W4A16: trtllm-gen has no such kernel (B300 04c)."""
    from mstar.utils.fused_moe.fp4_flashinfer import unsupported_activation

    assert unsupported_activation("silu") is None
    assert unsupported_activation(SiTU(4.0, 25.0), "mxfp4", "bf16") is not None
    assert unsupported_activation(SiTU(4.0, 25.0)) is not None
    assert unsupported_activation(SiTU(4.0, 25.0), "mxfp4", "mxfp8") is None
    assert unsupported_activation(SiTU(4.0, 25.0), "nvfp4") is None
    assert unsupported_activation("silu", "mxfp4", "bf16") is None
    assert unsupported_activation(SiTU(4.0, None)) is not None
    assert unsupported_activation("gelu") is not None
    assert SiTU.parse(SiTU(4.0, 25.0).spec) == SiTU(4.0, 25.0) and SiTU.parse("silu") is None
