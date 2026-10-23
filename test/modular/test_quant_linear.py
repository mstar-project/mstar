"""Step 5 linear quantization: config and YAML, scale rules, the load pass, and
the kernels (CPU via the emulated kernel; CUDA cases skip without a GPU)."""
import types

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import CommGroup, WorkerParallelGroups
from mstar.engine.precision import (
    NodePrecision,
    apply_precision,
    apply_yaml_precision,
    fp32_names,
    precision_mismatches,
)
from mstar.engine.quant import QuantConfig, quantize_linears
from mstar.engine.quant.formats import FORMATS
from mstar.engine.quant.kernels import KERNELS
from mstar.engine.quant.linear import QuantizedLinear, choose_kernel
from mstar.engine.quant.ops import FP8, _quant_per_token_portable
from mstar.engine.quant.scale_rules import fake_quant, get_scale_rule
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.components.distributed.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils.device import DeviceCapability
from mstar.worker.engine_manager import EngineManager

CPU = DeviceCapability.of("cpu")
EMULATED = QuantConfig("fp8_channel", activations="dynamic_token", kernel="fp8_emulated")
TP1 = CommGroup(0, 0, [0])


class _Node(NodeSubmodule):
    def __init__(self):
        super().__init__()
        self.qkv = FusedColumnLinear(64, {"q": 64, "k": 32, "v": 32}, bias=True)
        self.gate_up = MergedColumnParallelLinear(TP1, 64, [128, 128], bias=False)
        self.down = RowParallelLinear(TP1, 128, 64, bias=True)
        self.out = torch.nn.Linear(128, 64)
        self.head = torch.nn.Linear(64, 32, bias=False)

    def forward(self, x):
        h = self.out(self.qkv(x))
        gate, up = self.gate_up(h).chunk(2, dim=-1)
        h = self.down(torch.nn.functional.silu(gate) * up)
        return self.head(h)

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError


def _node(dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    node = _Node().requires_grad_(False)
    for param in node.parameters():
        param.normal_(std=0.1)
    return apply_precision(node, NodePrecision(dtype))


def _quantize(node, config=EMULATED, skip=(), islands=(), device=CPU, dtype=torch.bfloat16):
    quantize_linears(node, "llm", tuple(skip), config, fp32_names(node, tuple(islands)), device, dtype)
    return node


# --- config and YAML -----------------------------------------------------------------


def test_config_refuses_unknown_and_unbuilt_choices():
    with pytest.raises(ValueError, match="unknown quant format 'fp9'"):
        QuantConfig("fp9")
    with pytest.raises(ValueError, match="unknown scale rule 'p99'"):
        QuantConfig("fp8_channel", weight_scale="p99")
    with pytest.raises(NotImplementedError, match="'static' is not implemented"):
        QuantConfig("fp8_channel", activations="static")
    with pytest.raises(NotImplementedError, match="checkpoint reader"):
        QuantConfig("fp8_channel", source="checkpoint")
    with pytest.raises(ValueError, match="unknown quant kernel"):
        QuantConfig("fp8_channel", kernel="marlin")
    with pytest.raises(TypeError, match="tuple of globs"):
        NodePrecision(torch.bfloat16, quant_skip="lm_head")


def test_yaml_sets_changes_and_removes_quant():
    model = NodePrecision(torch.bfloat16, quant_skip=("head",))
    block = {"quant": {"weights": "fp8_channel", "activations": "dynamic_token", "weight_scale": "mse",
                       "skip": ["out"]}}
    quantized = apply_yaml_precision(model, "llm", block)
    assert quantized.quant == QuantConfig("fp8_channel", weight_scale="mse", activations="dynamic_token",
                                          skip=("out",))
    assert quantized.quant_skip == ("head",)
    assert apply_yaml_precision(quantized, "llm", {"quant": None}).quant is None
    assert apply_yaml_precision(quantized, "llm", {"fp32_modules": ["out"]}).quant == quantized.quant


def test_yaml_quant_errors_name_the_node():
    with pytest.raises(ValueError, match=r"nodes.llm.precision.quant: unknown key\(s\) \['format'\]"):
        apply_yaml_precision(NodePrecision(torch.bfloat16), "llm", {"quant": {"format": "fp8_channel"}})
    with pytest.raises(ValueError, match="nodes.llm.precision.quant: weights"):
        apply_yaml_precision(NodePrecision(torch.bfloat16), "llm", {"quant": {"activations": "dynamic_token"}})
    with pytest.raises(ValueError, match="nodes.llm.precision.quant: unknown quant format"):
        apply_yaml_precision(NodePrecision(torch.bfloat16), "llm", {"quant": {"weights": "int3"}})


def test_quant_changes_the_policy_repr_a_prefix_root_folds_in():
    plain = NodePrecision(torch.bfloat16)
    assert repr(plain) != repr(NodePrecision(torch.bfloat16, quant=QuantConfig("fp8_channel")))
    assert repr(NodePrecision(torch.bfloat16, quant=QuantConfig("fp8_channel"))) != repr(
        NodePrecision(torch.bfloat16, quant=QuantConfig("fp8_channel", weight_scale="mse"))
    )


# --- scale rules -------------------------------------------------------------------------


def test_amax_never_clips_and_mse_never_does_worse():
    torch.manual_seed(0)
    # heavy tails: a few outliers per row make clipping pay
    blocks = torch.randn(64, 512) * torch.where(torch.rand(64, 512) < 0.01, 20.0, 1.0)
    amax = get_scale_rule("amax")(blocks, FP8)
    torch.testing.assert_close(amax * torch.finfo(FP8).max, blocks.abs().amax(-1))
    mse = get_scale_rule("mse")(blocks, FP8)
    err = {name: (fake_quant(blocks, s, FP8) - blocks).square().sum(-1) for name, s in (("amax", amax), ("mse", mse))}
    assert torch.all(err["mse"] <= err["amax"])
    assert torch.all(mse <= amax)


def test_zero_rows_quantize_to_zero():
    values, scale = FORMATS["fp8_channel"].quantize(torch.zeros(4, 32), "amax")
    assert torch.all(values.float() == 0) and torch.all(scale > 0)
    q, s = _quant_per_token_portable(torch.zeros(3, 32))
    assert torch.all(q.float() == 0) and torch.all(s > 0)


# --- the load pass -------------------------------------------------------------------------


def test_pass_quantizes_every_linear_kind_and_keeps_output_close():
    ref = _node()
    node = _quantize(_node())
    for name, width in (("qkv", 64), ("gate_up", 64), ("down", 128), ("out", 128), ("head", 64)):
        layer = getattr(node, name)
        assert layer.weight.dtype is FP8, name
        assert layer.weight_scale.dtype is torch.float32 and layer.weight_scale.shape == (layer.weight.shape[0], 1)
        assert layer.linear_quant.kernel.name == "fp8_emulated"
        x = torch.randn(64, width, dtype=torch.bfloat16)
        want, got = getattr(ref, name)(x).float(), layer(x).float()
        # e4m3 W8A8 on Gaussian data: about 3.5% per layer
        assert ((got - want).norm() / want.norm()) < 0.05, name
    assert type(node.out) is QuantizedLinear
    assert node.qkv.bias.dtype is torch.bfloat16


def test_precision_cast_leaves_scales_alone_and_casts_bias():
    node = _quantize(_node())
    precision = NodePrecision(torch.float16)
    apply_precision(node, precision)
    assert node.out.weight.dtype is FP8 and node.out.weight_scale.dtype is torch.float32
    assert node.out.bias.dtype is torch.float16
    assert precision_mismatches(node, precision) == []


def test_skip_globs_and_fp32_islands_are_never_quantized():
    node = _node()
    apply_precision(node, NodePrecision(torch.bfloat16, fp32_modules=("down",)))
    _quantize(node, skip=("head",), islands=("down",))
    assert node.head.weight.dtype is torch.bfloat16 and type(node.head) is torch.nn.Linear
    assert node.down.weight.dtype is torch.float32 and node.down.linear_quant is None
    assert node.out.weight.dtype is FP8
    with pytest.raises(ValueError, match="quant.skip glob 'nope' matches no module"):
        _quantize(_node(), config=QuantConfig("fp8_channel", activations="dynamic_token",
                                              kernel="fp8_emulated", skip=("nope",)))


def test_tied_weight_is_refused():
    node = _node()
    node.extra = torch.nn.Linear(64, 32, bias=False)
    node.extra.weight = node.head.weight
    with pytest.raises(ValueError, match="shared with another param"):
        _quantize(node)


def test_shared_linear_must_be_quantized_alike():
    a = _quantize(_node())
    b = _node()
    b.out = a.out
    _quantize(b)  # same config: nothing to do for the shared layer
    assert b.out is a.out
    c = _node()
    c.out = a.out
    with pytest.raises(ValueError, match="shared with a node that quantized it"):
        _quantize(c, config=QuantConfig("fp8_channel", weight_scale="mse", activations="dynamic_token",
                                        kernel="fp8_emulated"))


def test_engine_refuses_a_linear_shared_with_a_node_that_does_not_quantize_it():
    from mstar.worker.engine_manager import _refuse_conflicting_shared_modules

    a = _quantize(_node())
    b = _node()
    b.out = a.out
    quantized = NodePrecision(torch.bfloat16, quant=EMULATED)
    _refuse_conflicting_shared_modules({"a": a, "b": b}, {"a": quantized, "b": quantized})
    for policy in (NodePrecision(torch.bfloat16), NodePrecision(torch.bfloat16, quant_skip=("out",), quant=EMULATED)):
        with pytest.raises(ValueError, match=r"""cast or quantized differently there: \["out \(quantized as"""):
            _refuse_conflicting_shared_modules({"a": a, "b": b}, {"a": quantized, "b": policy})


def test_kernel_choice():
    # the reference kernel is never picked unpinned; the sm90 kernel says why it can't run
    with pytest.raises(ValueError, match="scaled_mm_rowwise: needs CUDA sm90"):
        choose_kernel(QuantConfig("fp8_channel", activations="dynamic_token"), CPU, torch.bfloat16)
    assert choose_kernel(QuantConfig("fp8_channel"), CPU, torch.bfloat16).name == "dequant"
    with pytest.raises(ValueError, match="dequant: does not run activations='dynamic_token'"):
        choose_kernel(QuantConfig("fp8_channel", activations="dynamic_token", kernel="dequant"), CPU, torch.bfloat16)


def test_weight_only_matches_dequantized_weights():
    node = _quantize(_node(), config=QuantConfig("fp8_tensor"))
    assert node.out.weight_scale.shape == (1, 1)
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    w = (node.out.weight.float() * node.out.weight_scale).to(torch.bfloat16)
    torch.testing.assert_close(node.out(x), torch.nn.functional.linear(x, w, node.out.bias))


def test_lora_merge_into_a_quantized_weight_is_refused():
    from mstar.model.components.diffusion.lora import _merge_delta

    node = _quantize(_node())
    with pytest.raises(ValueError, match="merge before quantizing"):
        _merge_delta(node.out.weight, torch.zeros(64, 128), node.out, None)


# --- through the engine manager ----------------------------------------------------------


class _Model:
    def get_node_precision(self, node_name):
        return NodePrecision(torch.bfloat16, quant_skip=("head",))

    def get_submodule(self, node_name, device="cpu", tp_group=None, autocast_dtype=None):
        return _Node()

    def get_node_resources(self):
        return []

    def prefix_key_streams(self):
        return {}

    def get_graph_walk_graphs(self):
        return {}

    def combined_walk_of(self):
        return {}

    def checkpoint_path(self):
        return None

    def preprocess_fingerprint(self):
        return ""


def test_yaml_block_quantizes_one_node(caplog):
    quant = {"weights": "fp8_channel", "activations": "dynamic_token", "kernel": "fp8_emulated"}
    with caplog.at_level("INFO"):
        manager = EngineManager.build(
            node_names={"llm", "talker"},
            device=torch.device("cpu"),
            model_config={"nodes": {"llm": {"precision": {"quant": quant}}}},
            parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
            transfer_engine_info=TransferEngineInfo(
                my_entity_id="test", my_session_id="test", transfer_engine=LocalTransferEngine("test"),
            ),
            model=_Model(),
            graph_runtime=types.SimpleNamespace(),
        )
    try:
        llm = manager.engine._submodules["llm"].submodule
        talker = manager.engine._submodules["talker"].submodule
        assert llm.out.weight.dtype is FP8 and llm.head.weight.dtype is torch.bfloat16
        assert talker.out.weight.dtype is torch.bfloat16
        assert manager.engine._node_precisions["llm"].quant == QuantConfig(**quant)
        assert "4 linears quantized, 1 left as is" in caplog.text
    finally:
        manager.shutdown()


# --- CUDA ----------------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), reason="needs sm90",
)


@cuda
@pytest.mark.parametrize("shape", [(1, 64), (7, 1000), (33, 4096), (2, 3, 8192), (5, 10000)])
def test_triton_quant_matches_portable_bit_for_bit(shape):
    x = (torch.randn(shape, device="cuda") * 3).to(torch.bfloat16)
    x[0, ..., 0] = 0
    q, s = torch.ops.mstar.fp8_quant_per_token(x)
    q_ref, s_ref = _quant_per_token_portable(x)
    assert torch.equal(s, s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


@sm90
@pytest.mark.parametrize("rows", [9, 200])  # split-K Triton, then scaled_mm
def test_scaled_mm_matches_emulated_and_captures(rows):
    device = DeviceCapability.of("cuda")
    node = _quantize(_node().cuda(), config=QuantConfig("fp8_channel", activations="dynamic_token"),
                     device=device)
    ref = _quantize(_node().cuda(), device=device)
    assert node.out.linear_quant.kernel.name == "scaled_mm_rowwise"
    # per layer: chained, a bf16 rounding flip requantizes and grows to a few percent
    for name, width in (("qkv", 64), ("gate_up", 64), ("down", 128), ("out", 128), ("head", 64)):
        x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
        got, want = getattr(node, name)(x).float(), getattr(ref, name)(x).float()
        assert (got - want).norm() / want.norm() < 5e-3, name

    static = torch.zeros(rows, 64, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        node(static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = node(static)
    static.copy_(torch.randn_like(static))
    graph.replay()
    torch.testing.assert_close(out, node(static), atol=0, rtol=0)


@cuda
@pytest.mark.parametrize("m,n,k", [(1, 36, 64), (16, 2048, 6144), (64, 300, 1000)])
def test_small_m_gemm_matches_its_reference(m, n, k):
    a, sa = torch.ops.mstar.fp8_quant_per_token(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
    b = (torch.randn(n + 12, k, device="cuda") * 0.1).to(FP8)  # padded rows past n_out are ignored
    sb = torch.rand(1, n + 12, device="cuda") + 0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    got = torch.ops.mstar.fp8_mm_small_m(a, sa, b, sb, bias, n, torch.bfloat16)
    # the op's portable reference, on CPU
    want = torch.ops.mstar.fp8_mm_small_m(a.cpu(), sa.cpu(), b.cpu(), sb.cpu(), bias.cpu(), n, torch.bfloat16)
    assert got.shape == (m, n)
    torch.testing.assert_close(got.cpu(), want, atol=0.02, rtol=0.01)


@sm90
@pytest.mark.parametrize("rows", [10, 100])
@pytest.mark.parametrize("bias", [False, True])
def test_scaled_mm_pads_an_output_width_off_16(bias, rows):
    device = DeviceCapability.of("cuda")
    torch.manual_seed(0)
    ref = torch.nn.Linear(64, 36, bias=bias).cuda().to(torch.bfloat16)
    layers = {}
    for kernel in ("scaled_mm_rowwise", "fp8_emulated"):
        layer = torch.nn.Sequential(torch.nn.Linear(64, 36, bias=bias)).cuda().to(torch.bfloat16)
        layer.load_state_dict(torch.nn.Sequential(ref).state_dict())
        config = QuantConfig("fp8_channel", activations="dynamic_token", kernel=kernel)
        quantize_linears(layer, "llm", (), config, set(), device, torch.bfloat16)
        layers[kernel] = layer
    assert layers["scaled_mm_rowwise"][0].weight.shape == (48, 64)
    x = torch.randn(2, rows // 2, 64, device="cuda", dtype=torch.bfloat16)
    got = layers["scaled_mm_rowwise"](x)
    assert got.shape == (2, rows // 2, 36)
    torch.testing.assert_close(got, layers["fp8_emulated"](x), atol=0.02, rtol=0.02)


@sm90
def test_compiles_without_graph_breaks():
    node = _quantize(_node().cuda(), config=QuantConfig("fp8_channel", activations="dynamic_token"),
                     device=DeviceCapability.of("cuda"))
    x = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.reset()
    compiled = torch.compile(node, fullgraph=True)
    torch.testing.assert_close(compiled(x), node(x), atol=0.05, rtol=0.02)
    assert set(KERNELS) >= {"scaled_mm_rowwise", "dequant", "fp8_emulated"}


# --- step 5b: fusing the activation quantize into its producer ---------------------------


def test_fused_add_norm_quant_reference_is_the_unfused_pair():
    from mstar.engine.quant.ops import _quant_per_token_portable

    torch.manual_seed(0)
    h, r = torch.randn(5, 96, dtype=torch.bfloat16), torch.randn(5, 96, dtype=torch.bfloat16)
    w = torch.randn(96, dtype=torch.bfloat16) * 0.1
    h2, r2 = h.clone(), r.clone()
    torch.ops.mstar.fused_add_rms_norm(h2, r2, w, 1e-6, True)
    q, s = torch.ops.mstar.fused_add_rms_norm_fp8_quant(h, r, w, 1e-6, True)
    q2, s2 = _quant_per_token_portable(h2)
    assert torch.equal(h, h2) and torch.equal(r, r2)
    assert torch.equal(s, s2) and torch.equal(q.view(torch.uint8), q2.view(torch.uint8))


def test_fusion_is_off_with_the_env_switch(monkeypatch):
    import torch._inductor.config as inductor_config

    from mstar.engine.quant import fusion

    monkeypatch.setattr(inductor_config, "post_grad_custom_post_pass", None)
    monkeypatch.setenv("MSTAR_QUANT_FUSE", "0")
    fusion.install()
    assert inductor_config.post_grad_custom_post_pass is None
    monkeypatch.delenv("MSTAR_QUANT_FUSE")
    fusion.install()
    fusion.install()
    assert isinstance(inductor_config.post_grad_custom_post_pass, fusion.QuantFusionPass)


@cuda
@pytest.mark.parametrize("m,k", [(1, 2048), (37, 5120)])
def test_fused_add_norm_quant_kernel_matches_flashinfer_then_quantize(m, k):
    torch.manual_seed(0)
    h = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(h) * 4
    w = torch.randn(k, device="cuda", dtype=torch.bfloat16) * 0.1
    h2, r2 = h.clone(), r.clone()
    torch.ops.mstar.fused_add_rms_norm(h2, r2, w, 1e-6, True)
    q2, s2 = torch.ops.mstar.fp8_quant_per_token(h2)
    q, s = torch.ops.mstar.fused_add_rms_norm_fp8_quant(h, r, w, 1e-6, True)
    assert torch.equal(r, r2)
    # the sum of squares reduces in another order than FlashInfer's: a rare bf16 ulp
    assert (h != h2).float().mean() < 1e-4
    torch.testing.assert_close(h, h2, rtol=8e-3, atol=0)
    torch.testing.assert_close(s, s2, rtol=1e-5, atol=0)
    assert (q.view(torch.uint8) != q2.view(torch.uint8)).float().mean() < 1e-4


class _Decoder(torch.nn.Module):
    """qwen3_5's pattern: ATen norm, gated MLP, fused add + norm, gated MLP."""

    def __init__(self, hidden=256, inter=512):
        from mstar.model.components.distributed.mlp import ParallelGatedMLP
        from mstar.model.components.norm import RMSNorm

        super().__init__()
        self.n1, self.n2 = RMSNorm(hidden, gemma_mode=True), RMSNorm(hidden, gemma_mode=True)
        self.mlp1 = ParallelGatedMLP(hidden_size=hidden, intermediate_size=inter, comm_group=TP1)
        self.mlp2 = ParallelGatedMLP(hidden_size=hidden, intermediate_size=inter, comm_group=TP1)

    def forward(self, x):
        h, r = self.n2.forward_residual(self.mlp1(self.n1(x)), x * 1)
        return self.mlp2(h) + r


def _compiled_kernels(fuse: bool, x):
    import torch._inductor.config as inductor_config
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(0)
    node = _Decoder().cuda().to(torch.bfloat16).requires_grad_(False)
    for param in node.parameters():
        param.normal_(std=0.05)
    _quantize(node, config=QuantConfig("fp8_channel", activations="dynamic_token"), device=DeviceCapability.of("cuda"))
    saved = inductor_config.post_grad_custom_post_pass
    if not fuse:
        inductor_config.post_grad_custom_post_pass = None
    try:
        torch._dynamo.reset()
        compiled = torch.compile(node, dynamic=False)
        with torch.inference_mode():
            out = compiled(x)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                compiled(x)
                torch.cuda.synchronize()
    finally:
        inductor_config.post_grad_custom_post_pass = saved
    return out, [e.name for e in prof.events() if e.device_type.name == "CUDA"]


@sm90
@pytest.mark.parametrize("rows", [8, 300])
def test_compiled_node_has_no_standalone_quantize(rows, caplog, monkeypatch):
    import torch._inductor.config as inductor_config

    # a cached graph skips the pass, and its log line
    monkeypatch.setattr(inductor_config, "fx_graph_cache", False)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    with caplog.at_level("INFO", logger="mstar.engine.quant.fusion"):
        fused, kernels = _compiled_kernels(True, x)
    assert "quant fusion: {'norm': 1, 'aten': 3, 'kept': 0}" in caplog.text
    unfused, unfused_kernels = _compiled_kernels(False, x)
    assert not any("_quant_per_token_kernel" in k for k in kernels)
    assert sum("_quant_per_token_kernel" in k for k in unfused_kernels) == 4
    assert sum("add_rms_norm_quant" in k for k in kernels) == 1
    # one rounding (fp32 to fp8) where unfused rounds to bf16 first; grows over four layers
    assert (fused.float() - unfused.float()).norm() / unfused.float().norm() < 0.05


@cuda
def test_quantize_of_a_graph_input_is_kept(caplog, monkeypatch):
    import torch._inductor.config as inductor_config

    from mstar.engine.quant import fusion

    monkeypatch.setattr(inductor_config, "fx_graph_cache", False)
    fusion.install()
    torch._dynamo.reset()
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    with caplog.at_level("INFO", logger="mstar.engine.quant.fusion"), torch.inference_mode():
        q, s = torch.compile(torch.ops.mstar.fp8_quant_per_token, dynamic=False)(x)
    assert "'kept': 1" in caplog.text
    q2, s2 = _quant_per_token_portable(x)
    assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s, s2)


@cuda
def test_fused_add_norm_op_keeps_the_compiled_graph_whole():
    from mstar.model.components.norm import RMSNorm

    norm = RMSNorm(256, gemma_mode=True).cuda().to(torch.bfloat16).requires_grad_(False)
    h = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(h)
    with torch.inference_mode():
        explain = torch._dynamo.explain(norm.forward_residual)(h.clone(), r.clone())
        assert explain.graph_break_count == 0
        h2, r2 = h.clone(), r.clone()
        from flashinfer.norm import gemma_fused_add_rmsnorm

        gemma_fused_add_rmsnorm(h2, r2, norm.weight, norm.variance_epsilon)
        got_h, got_r = norm.forward_residual(h.clone(), r.clone())
    assert torch.equal(got_h, h2) and torch.equal(got_r, r2)
