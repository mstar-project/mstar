"""GLM-5.2 prefill: capture buckets from model_kwargs, the router under autocast."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.cuda_graph_config import PackedCudaGraphConfig  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


def _submodule(config) -> Glm52LLMSubmodule:
    # get_cuda_graph_configs reads only self.config
    sub = object.__new__(Glm52LLMSubmodule)
    sub.config = config
    return sub


def _prefill_config(sub) -> PackedCudaGraphConfig:
    (packed,) = [c for c in sub.get_cuda_graph_configs(torch.device("cpu"))
                 if c.capture_graph_walk == "prefill"]
    return packed


def test_prefill_buckets_from_model_kwargs():
    model = Glm52Model("", tokenizer_mode="byte", config_variant="reduced",
                       prefill_token_buckets=[16, 32, 64],
                       prefill_capture_batch_sizes=[1, 4])
    assert model.config.prefill_token_buckets == [16, 32, 64]
    assert model.config.prefill_capture_batch_sizes == [1, 4]
    packed = _prefill_config(_submodule(model.config))
    assert packed.get_total_tokens(1) == [16, 32, 64]
    assert packed.capture_batch_sizes == [1, 4]


def test_prefill_buckets_default_without_kwargs():
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full")
    assert model.config.prefill_token_buckets is None
    packed = _prefill_config(_submodule(model.config))
    assert packed.get_total_tokens(1) == Glm52LLMSubmodule.PREFILL_TOKEN_BUCKETS
    assert packed.capture_batch_sizes == Glm52LLMSubmodule.PREFILL_CAPTURE_BATCH_SIZES


def test_config_default_buckets_unchanged():
    assert Glm52ModelConfig().prefill_token_buckets is None
    assert Glm52ModelConfig().prefill_capture_batch_sizes is None


# ── router under the engine's autocast ──


def _gate(hidden=256, experts=32, device="cpu"):
    from mstar.model.glm52.components.moe import Glm52MoEGate

    torch.manual_seed(3)
    gate = Glm52MoEGate(hidden, experts, 8, routed_scaling_factor=2.5).to(device)
    with torch.no_grad():
        gate.weight.normal_(0.0, 0.05)
        gate.e_score_correction_bias.normal_(0.0, 0.5)
    gate.weight.data = gate.weight.data.to(torch.bfloat16)
    gate.finalize_weights()
    return gate


def _fp32_reference(gate, x):
    """The HF router: fp32 logits from the bf16 weight, no autocast; also each row's gap
    between the 8th and 9th biased score (summation order can swap a closer pair)."""
    scores = torch.sigmoid(x.double() @ gate.weight.double().t()).float()
    biased = scores + gate.e_score_correction_bias
    top = torch.topk(biased, k=gate.top_k + 1, dim=-1)
    ids = top.indices[:, :gate.top_k]
    w = scores.gather(1, ids)
    gap = top.values[:, gate.top_k - 1] - top.values[:, gate.top_k]
    return w / w.sum(-1, keepdim=True) * gate.routed_scaling_factor, ids, gap


def _check(gate, x, device):
    with torch.autocast(device, dtype=torch.bfloat16):
        w, ids = gate(x)
    w_ref, ids_ref, gap = _fp32_reference(gate, x)
    clear = gap > 1e-5
    assert clear.float().mean() > 0.99
    ids, order = ids.sort(dim=-1)
    ids_ref, order_ref = ids_ref.sort(dim=-1)
    assert w.dtype == torch.float32
    torch.testing.assert_close(ids[clear], ids_ref[clear], rtol=0, atol=0)
    # fp32 accumulation over 6144 terms is off a float64 reference by ~1e-5; bf16 logits
    # (the autocast demotion) put the weights off by ~1e-2
    torch.testing.assert_close(w.gather(1, order)[clear], w_ref.gather(1, order_ref)[clear],
                               rtol=1e-4, atol=1e-5)


def test_router_stays_fp32_under_bf16_autocast():
    gate = _gate()
    x = torch.randn(64, 256, dtype=torch.bfloat16)
    _check(gate, x, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 GEMM with fp32 output on CUDA")
@pytest.mark.parametrize("tokens", [1, 65, 300, 1100])
def test_router_bf16_gemm_matches_fp32_on_cuda(tokens):
    gate = _gate(hidden=6144, experts=256, device="cuda")
    x = torch.randn(tokens, 6144, dtype=torch.bfloat16, device="cuda")
    _check(gate, x, "cuda")


# ── the engine's compiled fallback ──


def _cpu_flashinfer(monkeypatch):
    import types

    def rmsnorm(x, weight, eps=1e-6):
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    fi = types.ModuleType("flashinfer")
    fi.norm = types.SimpleNamespace(rmsnorm=rmsnorm)
    monkeypatch.setitem(sys.modules, "flashinfer", fi)


def test_compiled_forward_does_not_recompile_per_request(monkeypatch):
    """The engine compiles forward_batched (dynamic=None) for uncaptured steps; a new
    request's prefill at an already-compiled length must reuse the compiled frames."""
    if not hasattr(sys.modules.get("triton"), "__path__"):
        pytest.skip("dynamo needs the real triton package (conftest stubs it on macOS)")
    from types import SimpleNamespace

    from torch._dynamo.testing import CompileCounter

    from mstar.engine.resources import StepContext
    from mstar.model.glm52._testing import build_cpu_resources
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.config import SAMPLER_RESOURCE
    from mstar.model.glm52.quantization import process_weights_after_loading
    from mstar.model.submodule_base import ModelInputsFromEngine

    _cpu_flashinfer(monkeypatch)
    cfg = Glm52ModelConfig.reduced()
    torch.manual_seed(0)
    lm = Glm52ForCausalLM(cfg)
    for p in lm.parameters():
        p.data.normal_(0, 0.05)
    process_weights_after_loading(lm, torch.device("cpu"))
    sub = Glm52LLMSubmodule(lm.eval(), cfg)
    rids = ["a", "b", "c", "d"]
    resources, runner = build_cpu_resources(cfg, rids)
    sub.bind_node_resources(resources)
    counter = CompileCounter()
    torch._dynamo.reset()
    fwd = torch.compile(sub.forward_batched, backend=counter, fullgraph=False, dynamic=None)

    def prefill(rid):
        info = SimpleNamespace(
            request_id=rid, max_tokens=8, dynamic_loop_iter_counts={},
            resource_configs={SAMPLER_RESOURCE: SimpleNamespace(
                ignore_eos=True, temperature=0.0, repetition_penalty=1)})
        inputs = [sub.prepare_inputs("prefill", info, {"text_inputs": [torch.arange(6) + 3]})]
        step = sub.declare_step("prefill", [rid], inputs)
        step.set_ctx(StepContext(request_ids=(rid,), graph_walk="prefill", slot=0,
                                 capture=False))
        assert runner.admit(step).ok
        runner.plan(step)
        ei = ModelInputsFromEngine(request_ids=[rid], per_request_info={rid: info},
                                   resources=resources, step=step)
        with torch.no_grad():
            out = fwd("prefill", ei, **sub.preprocess("prefill", ei, inputs))
        runner.commit(step)
        return out

    try:
        # the second call recompiles once as the RoPE cache fills
        for rid in ("a", "b"):
            assert set(prefill(rid)) == {rid}
        frames = counter.frame_count
        for rid in ("c", "d"):
            assert set(prefill(rid)) == {rid}
        assert counter.frame_count == frames
    finally:
        torch._dynamo.reset()


def test_prefill_buckets_past_the_context_limit_skip_small_batches():
    # a 4096-token bucket needs two rows under the 2048-token limit
    cfg = Glm52ModelConfig(prefill_token_buckets=[1024, 2048, 3072, 4096],
                           prefill_capture_batch_sizes=[1, 2, 4])
    packed = _prefill_config(_submodule(cfg))
    assert packed.get_total_tokens(1) == [1024, 2048]
    assert packed.get_total_tokens(2) == [1024, 2048, 3072, 4096]
    assert packed.get_total_tokens(4) == [1024, 2048, 3072, 4096]


def test_last_rows_is_compiler_disabled():
    # traced, its resume frame takes one dynamo entry per (tokens, rows) capture bucket
    fn = Glm52LLMSubmodule._last_rows
    assert hasattr(fn, "_torchdynamo_disable") or hasattr(fn, "_torchdynamo_orig_callable")


def test_batched_prefill_buckets_get_their_own_capture():
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full",
                       prefill_token_buckets=[64, 128, 192],
                       prefill_batched_token_buckets=[1024, 2048],
                       prefill_capture_batch_sizes=[1, 4, 16])
    assert model.config.prefill_batched_token_buckets == [1024, 2048]
    prefill = [c for c in _submodule(model.config).get_cuda_graph_configs(torch.device("cpu"))
               if c.capture_graph_walk == "prefill"]
    assert [(c.capture_batch_sizes, c.get_total_tokens(c.capture_batch_sizes[0]))
            for c in prefill] == [([1], [64, 128, 192]), ([4, 16], [1024, 2048])]


# ── the router's top-k kernel (moe_router_kernel) ──


def test_router_kernel_flag_defaults_off_and_reaches_the_block():
    from mstar.model.glm52.components.moe import Glm52SparseMoeBlock

    assert Glm52ModelConfig().moe_router_kernel is False
    model = Glm52Model("", tokenizer_mode="byte", config_variant="reduced_fp8",
                       moe_router_kernel=True)
    assert model.config.moe_router_kernel is True
    block = Glm52SparseMoeBlock(model.config)
    assert block._router_kernel is True
    block.process_weights_after_loading("cpu")  # no fused path on the host: torch top-k
    assert block.gate._topk_op is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton top-k kernel on CUDA")
@pytest.mark.parametrize("tokens", [1, 65, 1100])
def test_router_kernel_matches_the_torch_top_k(tokens):
    from mstar.utils.fused_moe import decode as moe_decode

    gate = _gate(hidden=6144, experts=256, device="cuda")
    x = torch.randn(tokens, 6144, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():  # as the engine runs it
        w_ref, ids_ref = gate(x)
        gate._topk_op = moe_decode.router_topk
        w, ids = torch.compile(gate, fullgraph=True)(x)  # an opaque op: no graph break
    _, _, gap = _fp32_reference(gate, x)
    clear = gap > 1e-5
    ids, order = ids.sort(dim=-1)
    ids_ref, order_ref = ids_ref.sort(dim=-1)
    torch.testing.assert_close(ids[clear], ids_ref[clear], rtol=0, atol=0)
    torch.testing.assert_close(w.gather(1, order)[clear], w_ref.gather(1, order_ref)[clear],
                               rtol=1e-6, atol=1e-7)


# ── the last layer's FFN on the sampled rows (prefill_last_layer_rows) ──


def test_last_layer_rows_flag_defaults_off_and_reaches_the_config():
    assert Glm52ModelConfig().prefill_last_layer_rows is False
    model = Glm52Model("", tokenizer_mode="byte", config_variant="reduced",
                       prefill_last_layer_rows=True)
    assert model.config.prefill_last_layer_rows is True


def test_last_layer_ffn_runs_on_the_sampled_rows(monkeypatch):
    """Two packed requests: the last layer's MLP sees their two last rows, every other
    layer all of them, and the sampled tokens and logits match the full forward."""
    from types import SimpleNamespace

    from mstar.engine.resources import StepContext
    from mstar.model.glm52._testing import build_cpu_resources
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.config import SAMPLER_RESOURCE
    from mstar.model.glm52.quantization import process_weights_after_loading
    from mstar.model.submodule_base import ModelInputsFromEngine

    _cpu_flashinfer(monkeypatch)
    cfg = Glm52ModelConfig.reduced()
    torch.manual_seed(0)
    lm = Glm52ForCausalLM(cfg)
    for p in lm.parameters():
        p.data.normal_(0, 0.05)
    process_weights_after_loading(lm, torch.device("cpu"))
    lm.eval()
    rows_seen = []
    for layer in lm.model.layers:
        layer.mlp.register_forward_pre_hook(lambda m, a: rows_seen.append(a[0].shape[0]))
    logits = []
    lm.lm_head.register_forward_hook(lambda m, a, out: logits.append(out))

    def prefill(flag, rids):
        cfg.prefill_last_layer_rows = flag
        sub = Glm52LLMSubmodule(lm, cfg)
        resources, runner = build_cpu_resources(cfg, rids)
        sub.bind_node_resources(resources)
        info = {rid: SimpleNamespace(
            request_id=rid, max_tokens=8, dynamic_loop_iter_counts={},
            resource_configs={SAMPLER_RESOURCE: SimpleNamespace(
                ignore_eos=True, temperature=0.0, repetition_penalty=1)}) for rid in rids}
        prompts = [torch.arange(5) + 3, torch.arange(7) + 11]
        inputs = [sub.prepare_inputs("prefill", info[rid], {"text_inputs": [p]})
                  for rid, p in zip(rids, prompts, strict=True)]
        step = sub.declare_step("prefill", rids, inputs)
        step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0,
                                 capture=False))
        assert runner.admit(step).ok
        runner.plan(step)
        ei = ModelInputsFromEngine(request_ids=rids, per_request_info=info,
                                   resources=resources, step=step)
        rows_seen.clear()
        with torch.no_grad():
            out = sub.forward_batched("prefill", ei, **sub.preprocess("prefill", ei, inputs))
        return [out[rid]["new_token"][0] for rid in rids], list(rows_seen)

    full_tokens, full_rows = prefill(False, ["a", "b"])
    rows_tokens, rows_rows = prefill(True, ["c", "d"])
    assert full_rows == [12] * cfg.num_hidden_layers
    assert rows_rows == [12] * (cfg.num_hidden_layers - 1) + [2]
    assert [int(t) for t in rows_tokens] == [int(t) for t in full_tokens]
    torch.testing.assert_close(logits[1], logits[0], rtol=1e-5, atol=1e-5)


def test_capture_configs_raise_the_recompile_limit_to_fit(monkeypatch):
    # every captured token count costs a decoder layer's frames up to three entries
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 84)
    buckets = list(range(64, 2049, 64))
    cfg = Glm52ModelConfig(prefill_token_buckets=buckets,
                           prefill_batched_token_buckets=[2304, 2560, 3072, 3584, 4096],
                           prefill_capture_batch_sizes=[1, 4, 16])
    _submodule(cfg).get_cuda_graph_configs(torch.device("cpu"))
    shapes = len(buckets) + 5 + 7  # + the decode batch sizes
    assert torch._dynamo.config.recompile_limit >= 3 * shapes
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 84)
    _submodule(Glm52ModelConfig()).get_cuda_graph_configs(torch.device("cpu"))
    assert torch._dynamo.config.recompile_limit == 84  # the default buckets fit


def test_recompile_limit_reaches_the_thread_that_runs_steps(monkeypatch):
    # dynamo's config is per thread, and the engine runs steps on a thread of its own
    import threading

    from mstar.engine import apply_torch_config

    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 84)
    cfg = Glm52ModelConfig(prefill_token_buckets=list(range(64, 2049, 64)),
                           prefill_capture_batch_sizes=[1, 4, 16])
    sub = _submodule(cfg)
    sub.get_cuda_graph_configs(torch.device("cpu"))
    need = torch._dynamo.config.recompile_limit
    assert need > 84

    class _StopError(Exception):
        pass

    def _kv(engine_inputs=None):
        raise _StopError

    monkeypatch.setattr(sub, "_kv", _kv)
    seen = []

    def step():
        apply_torch_config()
        with pytest.raises(_StopError):
            sub.preprocess("prefill", None, [])
        seen.append(torch._dynamo.config.recompile_limit)

    thread = threading.Thread(target=step)
    thread.start()
    thread.join()
    assert seen == [need]
