"""GLM-5.2 prefill: capture batch sizes from model_kwargs, the router under autocast."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.accelerator_graph_config import PackedAcceleratorGraphConfig  # noqa: E402
from mstar.model.glm52.config import Glm52ModelConfig  # noqa: E402
from mstar.model.glm52.glm52_model import Glm52Model  # noqa: E402
from mstar.model.glm52.submodules import Glm52LLMSubmodule  # noqa: E402


def _submodule(config) -> Glm52LLMSubmodule:
    # get_accelerator_graph_configs reads the config and the ladder, not the weights
    language_model = torch.nn.Module()
    language_model.lm_head = torch.nn.Identity()
    return Glm52LLMSubmodule(language_model, config)


def _prefill_config(sub) -> PackedAcceleratorGraphConfig:
    (packed,) = [c for c in sub.get_accelerator_graph_configs(torch.device("cpu"))
                 if c.capture_graph_walk == "prefill"]
    return packed


def test_prefill_capture_batch_sizes_from_model_kwargs():
    model = Glm52Model("", tokenizer_mode="byte", config_variant="reduced",
                       prefill_capture_batch_sizes=[1, 4])
    assert model.config.prefill_capture_batch_sizes == [1, 4]
    assert _prefill_config(_submodule(model.config)).capture_batch_sizes == [1, 4]


def test_prefill_buckets_default_without_kwargs():
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full")
    packed = _prefill_config(_submodule(model.config))
    assert packed.get_total_tokens(1) == Glm52LLMSubmodule.PREFILL_TOKEN_BUCKETS
    assert packed.capture_batch_sizes == Glm52LLMSubmodule.PREFILL_CAPTURE_BATCH_SIZES


@pytest.mark.parametrize("kwargs", [
    {"prefill_capture_batch_sizes": [0, 1]}, {"prefill_capture_batch_sizes": [-4, 1]},
    {"prefill_capture_batch_sizes": []}, {"graph_compile": "false"},
    {"mtp_num_draft_tokens": True}, {"mtp_num_draft_tokens": 2.5}, {"mtp_num_draft_tokens": 1023},
])
def test_a_bad_serving_knob_is_refused_at_load(kwargs):
    """Each loaded before: the batch sizes failed the capture build (a ZeroDivisionError or an
    IndexError), a 0-token bucket was captured, ``"false"`` compiled, and k=1023 refused every
    prompt as longer than a negative limit."""
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        Glm52Model("", tokenizer_mode="byte", config_variant="reduced", **kwargs)


def test_config_default_batch_sizes_unchanged():
    assert Glm52ModelConfig().prefill_capture_batch_sizes is None


# ── router under the engine's autocast ──


def _gate(hidden=256, experts=32, device="cpu"):
    from mstar.model.components.moe import NoAuxTcRouter

    torch.manual_seed(3)
    gate = NoAuxTcRouter(hidden, experts, 8, routed_scaling_factor=2.5).to(device)
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
        w, ids, _ = gate(x)
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
    from mstar.model.glm52._testing import install_cpu_rope

    install_cpu_rope(monkeypatch)


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
    from mstar.model.submodule_base import ModelInputsFromEngine
    from mstar.model.utils import process_weights_after_loading

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
            assert prefill(rid).row_outputs["new_token"].shape == (1,)
        frames = counter.frame_count
        for rid in ("c", "d"):
            assert prefill(rid).row_outputs["new_token"].shape == (1,)
        assert counter.frame_count == frames
    finally:
        torch._dynamo.reset()


def test_prefill_buckets_past_the_context_limit_skip_small_batches(monkeypatch):
    # a 4096-token bucket needs two rows under the 2048-token limit
    monkeypatch.setattr(Glm52LLMSubmodule, "PREFILL_TOKEN_BUCKETS", [1024, 2048, 3072, 4096])
    packed = _prefill_config(_submodule(Glm52ModelConfig(prefill_capture_batch_sizes=[1, 2, 4])))
    assert packed.get_total_tokens(1) == [1024, 2048]
    assert packed.get_total_tokens(2) == [1024, 2048, 3072, 4096]
    assert packed.get_total_tokens(4) == [1024, 2048, 3072, 4096]


@pytest.mark.parametrize("name", ["glm52_tp8.yaml"])
def test_a_shipped_config_s_ladders_reach_their_batch_sizes(name):
    """The YAML's ``prefill_token_buckets`` mapping passes the load-time checks and, as the
    engine applies it, captures one request's prefill on the fine ladder and a batch's on
    the coarse one."""
    import yaml

    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
    from mstar.worker.engine_manager import _prefill_token_buckets, _split_token_ladders

    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / name).read_text())
    model = Glm52Model("", tokenizer_mode="byte", config_variant="full", **config["model_kwargs"])
    walks, regions = _split_token_ladders(_prefill_token_buckets(config), model)
    assert (walks, regions) == (config["prefill_token_buckets"], {})
    runner = AcceleratorGraphRunner(
        submodule_name="LLM", submodule=_submodule(model.config), resources={}, step_runner=None,
        device=torch.device("cpu"), autocast_dtype=None, joint_comm_group=SimpleNamespace(world_size=1),
        num_slots=1, token_ladders=config["prefill_token_buckets"]["LLM"],
    )
    ladders = config["prefill_token_buckets"]["LLM"]["prefill"]
    captured = {}
    for spec in runner._get_slot_specs():
        if spec.bucket.graph_walk == "prefill":
            captured.setdefault(spec.bucket.bs, []).append(spec.bucket.num_tokens)
    assert captured == {1: ladders[1], 4: ladders[2], 16: ladders[2]}


# ── the router's top-k kernel (kernels: auto) ──


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton top-k kernel on CUDA")
@pytest.mark.parametrize("tokens", [1, 65, 1100])
def test_router_kernel_matches_the_torch_top_k(tokens):
    gate = _gate(hidden=6144, experts=256, device="cuda")
    x = torch.randn(tokens, 6144, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():  # as the engine runs it
        w_ref, ids_ref, _ = gate(x)
        gate.finalize_weights(topk_kernel=True)
        w, ids, _ = torch.compile(gate, fullgraph=True)(x)  # an opaque op: no graph break
    _, _, gap = _fp32_reference(gate, x)
    clear = gap > 1e-5
    ids, order = ids.sort(dim=-1)
    ids_ref, order_ref = ids_ref.sort(dim=-1)
    torch.testing.assert_close(ids[clear], ids_ref[clear], rtol=0, atol=0)
    torch.testing.assert_close(w.gather(1, order)[clear], w_ref.gather(1, order_ref)[clear],
                               rtol=1e-6, atol=1e-7)


# ── the last layer's FFN on the sampled rows (kernels: auto) ──


def test_last_layer_ffn_runs_on_the_sampled_rows(monkeypatch):
    """Under kernels: auto the last layer's MLP sees only the sampled rows, and the tokens and
    logits match the full forward."""
    from types import SimpleNamespace

    from mstar.engine.resources import StepContext
    from mstar.model.glm52._testing import build_cpu_resources, per_request_outputs
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.config import SAMPLER_RESOURCE
    from mstar.model.submodule_base import ModelInputsFromEngine
    from mstar.model.utils import process_weights_after_loading

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

    def prefill(rows_only, rids):
        cfg.kernels = "auto" if rows_only else "reference"
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
            out = per_request_outputs(
                sub.forward_batched("prefill", ei, **sub.preprocess("prefill", ei, inputs)), rids)
        return [out[rid]["new_token"][0] for rid in rids], list(rows_seen)

    full_tokens, full_rows = prefill(False, ["a", "b"])
    rows_tokens, rows_rows = prefill(True, ["c", "d"])
    assert full_rows == [12] * cfg.num_hidden_layers
    assert rows_rows == [12] * (cfg.num_hidden_layers - 1) + [2]
    assert [int(t) for t in rows_tokens] == [int(t) for t in full_tokens]
    torch.testing.assert_close(logits[1], logits[0], rtol=1e-5, atol=1e-5)


def _recompile_limit(ladders):
    """The recompile limit the capture runner of an LLM capturing prefill at batch sizes 1,
    4 and 16 asks for under the serving config's ``ladders``."""
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner

    return AcceleratorGraphRunner(
        submodule_name="LLM", submodule=_submodule(Glm52ModelConfig(prefill_capture_batch_sizes=[1, 4, 16])),
        resources={}, step_runner=None, device=torch.device("cpu"), autocast_dtype=None,
        joint_comm_group=SimpleNamespace(world_size=1), num_slots=1, token_ladders=ladders,
    ).recompile_limit


def test_the_runner_sizes_the_recompile_limit_for_a_config_s_ladder():
    """Every captured token count costs a decoder layer's frames up to three entries; the
    runner counts the serving config's ladder, not the model's default."""
    # past dynamo's default 256 entries per code object, which the wrapper raises too
    buckets = list(range(64, 4097, 32))
    shapes = len(buckets) + 7  # + the decode batch sizes
    assert _recompile_limit({"prefill": buckets}) >= 3 * shapes > 256


def test_a_token_count_captured_at_several_batch_sizes_costs_entries_at_each():
    """A frame is traced per captured batch size too (the last layer runs on a row per
    request), so one request's ladder and a batch's each count at every batch size."""
    fine, coarse = list(range(64, 1025, 64)), list(range(256, 2049, 256))
    shapes = len(fine) + 2 * len(coarse) + 7  # bs 1 on the fine ladder, 4 and 16 on the coarse
    assert _recompile_limit({"prefill": {1: fine, 2: coarse}}) >= 3 * shapes
