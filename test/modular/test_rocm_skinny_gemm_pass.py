"""The graph passes that send decode-sized linears to ``mstar::skinny_gemm``
or ``mstar::decode_gemv`` (and gated activations to ``mstar::decode_gemv_silu_mul``),
and when the ROCm platform installs them. CPU only: graphs are traced with fake
tensors and only rewritten, never run."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from mstar.kernels import Choice
from mstar.platforms import RocmPlatform, rocm_gemm
from mstar.platforms.rocm_gemm import SkinnyGemmPass, SkinnyLinearPrePass


def _linear_graph(rows, n=64, k=32, bias=False, dtype=torch.bfloat16, w_t=False):
    x = torch.randn(rows, k, dtype=dtype)
    w = torch.randn(k, n, dtype=dtype).T if w_t else torch.randn(n, k, dtype=dtype)
    b = torch.randn(n, dtype=dtype) if bias else None
    return make_fx(F.linear, tracing_mode="fake")(x, w, b).graph


def _targets(graph):
    return [n.target for n in graph.nodes if n.op == "call_function"]


def _skinny_calls(graph):
    """one-row GEMMs go to the layout-free twin of the op"""
    ops = (torch.ops.mstar.skinny_gemm.default, torch.ops.mstar.skinny_gemm_any.default)
    return sum(t in ops for t in _targets(graph))


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_small_linears_are_rewritten(rows, bias):
    graph = _linear_graph(rows, bias=bias)
    SkinnyGemmPass()(graph)
    targets = _targets(graph)
    assert _skinny_calls(graph) == 1
    assert torch.ops.aten.mm.default not in targets
    assert torch.ops.aten.addmm.default not in targets
    graph.lint()


def test_residual_folded_into_addmm_is_rewritten():
    """inductor turns ``x + linear(h)`` into ``addmm(x, h, w.T)``"""
    h, w, x = torch.randn(2, 32, dtype=torch.bfloat16), torch.randn(64, 32, dtype=torch.bfloat16), torch.randn(
        2, 64, dtype=torch.bfloat16)
    graph = make_fx(lambda h, w, x: torch.addmm(x, h, w.t()), tracing_mode="fake")(h, w, x).graph
    SkinnyGemmPass()(graph)
    assert _skinny_calls(graph) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(rows=3),                    # past the row limit
        dict(rows=4),
        dict(rows=1, k=36),              # K not a multiple of 8
        dict(rows=1, n=8),               # too few output features
        dict(rows=1, dtype=torch.float32),
        dict(rows=1, w_t=True),          # weight not a contiguous [N, K]
    ],
    ids=["rows3", "rows4", "k36", "n8", "fp32", "transposed-weight"],
)
def test_other_gemms_are_left_alone(kwargs):
    graph = _linear_graph(**kwargs)
    before = _targets(graph)
    SkinnyGemmPass()(graph)
    assert _targets(graph) == before


class _Linears(torch.nn.Module):
    """Two functional linears, as dynamo records ``nn.Linear`` calls."""

    def __init__(self, k=32, n=64, bias=False):
        super().__init__()
        bf16 = dict(dtype=torch.bfloat16)
        self.wa = torch.nn.Parameter(torch.randn(n, k, **bf16))
        self.wb = torch.nn.Parameter(torch.randn(k, n, **bf16))
        self.ba = torch.nn.Parameter(torch.randn(n, **bf16)) if bias else None
        self.bb = torch.nn.Parameter(torch.randn(k, **bf16)) if bias else None

    def forward(self, x):
        h = F.silu(F.linear(x, self.wa, self.ba))
        return F.linear(h, self.wb, self.bb)


def _torch_level_graph(rows, **kwargs):
    """A graph at the level dynamo hands pre-grad passes: ``linear`` calls with fake values."""
    gm = torch.fx.symbolic_trace(_Linears(**kwargs))
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    FakeTensorProp(gm, mode).propagate(torch.randn(rows, kwargs.get("k", 32), dtype=torch.bfloat16))
    return gm.graph


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_pre_grad_pass_rewrites_small_linears(rows, bias):
    graph = _torch_level_graph(rows, bias=bias)
    SkinnyLinearPrePass()(graph)
    assert _skinny_calls(graph) == 2
    assert F.linear not in _targets(graph)
    graph.lint()


def test_pre_grad_pass_leaves_larger_buckets_alone():
    graph = _torch_level_graph(4)
    before = _targets(graph)
    SkinnyLinearPrePass()(graph)
    assert _targets(graph) == before


def test_pass_has_a_stable_cache_key():
    assert SkinnyGemmPass().uuid() == SkinnyGemmPass().uuid()
    assert SkinnyLinearPrePass().uuid() == SkinnyLinearPrePass().uuid()


@pytest.mark.parametrize(
    ("num_tokens", "skinny"),
    [(1, True), (2, True), (3, True), (16, True), (17, False), (32, False), (None, False)],
)
def test_rocm_installs_the_passes_for_sixteen_rows_or_fewer(monkeypatch, num_tokens, skinny):
    pre, post = SkinnyLinearPrePass(), SkinnyGemmPass()
    monkeypatch.setattr(rocm_gemm, "skinny_gemm_passes", lambda: (pre, post))
    overrides = RocmPlatform().inductor_overrides(num_tokens)
    assert (overrides.get("pre_grad_custom_pass") is pre) == skinny
    assert (overrides.get("post_grad_custom_post_pass") is post) == skinny
    if num_tokens is not None and num_tokens <= RocmPlatform.ATEN_GEMM_MAX_TOKENS:
        assert overrides["max_autotune_gemm_backends"] == "ATEN"


def test_rocm_without_the_kernel_keeps_hipblaslt_only(monkeypatch):
    monkeypatch.setattr(rocm_gemm, "skinny_gemm_passes", lambda: None)
    assert RocmPlatform().inductor_overrides(1) == {"max_autotune_gemm_backends": "ATEN"}


def test_wide_decode_buckets_get_the_passes_only_where_the_table_tunes_them(monkeypatch):
    pre, post = SkinnyLinearPrePass(), SkinnyGemmPass()
    monkeypatch.setattr(rocm_gemm, "skinny_gemm_passes", lambda: (pre, post))
    monkeypatch.setattr(rocm_gemm, "tunes_wide_decode", lambda: True)
    platform = RocmPlatform()
    assert platform.inductor_overrides(128, rows=128)["post_grad_custom_post_pass"] is post
    # a packed prefill bucket of the same token count is not a decode
    assert "post_grad_custom_post_pass" not in platform.inductor_overrides(32, rows=1)
    assert platform.inductor_overrides(256, rows=256) == {}
    monkeypatch.setattr(rocm_gemm, "tunes_wide_decode", lambda: False)
    assert "post_grad_custom_post_pass" not in platform.inductor_overrides(64, rows=64)
    assert platform.inductor_overrides(8, rows=8)["post_grad_custom_post_pass"] is post


def _route(monkeypatch, linear, gated=None):
    """The ``decode_gemm`` table answering ``linear[rows]`` for a linear and
    ``gated`` for a gate/up projection (None: no entry)."""

    def choose(shape):
        impl = gated if "gated" in shape else linear.get(shape["rows"])
        return None if impl is None else Choice(impl, {})

    monkeypatch.setattr(rocm_gemm.DECODE_GEMM, "choose", choose)
    monkeypatch.setattr(rocm_gemm, "_have_aiter", lambda: True)
    monkeypatch.setattr(rocm_gemm, "_solutions_apply", lambda: True)


def _calls(graph, op):
    return sum(t is op for t in _targets(graph))


def _passed(graph):
    SkinnyGemmPass()(graph)
    return graph


@pytest.mark.parametrize(
    ("rows", "entry", "op"),
    [
        (1, "triton", torch.ops.mstar.decode_gemv.default),
        (4, "triton", torch.ops.mstar.decode_gemv.default),
        (16, "triton", torch.ops.mstar.decode_gemv.default),
        (2, "aiter", torch.ops.mstar.skinny_gemm.default),
        (1, "aiter", torch.ops.mstar.skinny_gemm_any.default),
        (8, "hipblaslt", torch.ops.aten.mm.default),
        (8, "hipblaslt_tuned", torch.ops.mstar.decode_linear.default),
        (64, "hipblaslt_tuned", torch.ops.mstar.decode_linear.default),
        (4, None, torch.ops.aten.mm.default),
        (2, None, torch.ops.mstar.skinny_gemm.default),
    ],
    ids=["1-triton", "4-triton", "16-triton", "2-aiter", "1-aiter", "8-hipblaslt", "8-tuned", "64-tuned", "4-no-table",
         "2-no-table"],
)
def test_the_table_picks_each_shapes_kernel(monkeypatch, rows, entry, op):
    _route(monkeypatch, {rows: entry})
    graph = _linear_graph(rows)
    SkinnyGemmPass()(graph)
    assert _calls(graph, op) == 1
    graph.lint()


def test_a_tuned_solution_needs_its_stack_no_bias_and_bf16(monkeypatch):
    _route(monkeypatch, {8: "hipblaslt_tuned"})
    assert _calls(_passed(_linear_graph(8, dtype=torch.float16)), torch.ops.mstar.decode_linear.default) == 0
    assert _calls(_passed(_linear_graph(8, bias=True)), torch.ops.mstar.decode_linear.default) == 0
    monkeypatch.setattr(rocm_gemm, "_solutions_apply", lambda: False)
    assert _calls(_passed(_linear_graph(8)), torch.ops.mstar.decode_linear.default) == 0


def test_aiter_missing_leaves_its_shapes_on_hipblaslt(monkeypatch):
    _route(monkeypatch, {1: "aiter"})
    monkeypatch.setattr(rocm_gemm, "_have_aiter", lambda: False)
    graph = _linear_graph(1)
    SkinnyGemmPass()(graph)
    assert _calls(graph, torch.ops.aten.mm.default) == 1


def test_an_unknown_implementation_is_refused(monkeypatch):
    _route(monkeypatch, {1: "cublas"})
    with pytest.raises(ValueError, match="unknown implementation"):
        SkinnyGemmPass()(_linear_graph(1))


class _GatedMLP(torch.nn.Module):
    """A gated MLP as dynamo records ``ParallelGatedMLP``: one gate/up linear,
    a split, ``silu(gate) * up``, the down linear."""

    def __init__(self, hidden=32, inter=48, bias=False, leak=False):
        super().__init__()
        bf16 = dict(dtype=torch.bfloat16)
        self.inter, self.leak = inter, leak
        self.gate_up = torch.nn.Parameter(torch.randn(2 * inter, hidden, **bf16))
        self.down = torch.nn.Parameter(torch.randn(hidden, inter, **bf16))
        self.bias = torch.nn.Parameter(torch.randn(2 * inter, **bf16)) if bias else None

    def forward(self, x):
        gate_up = F.linear(x, self.gate_up, self.bias)
        gate, up = gate_up.split(self.inter, dim=-1)
        out = F.linear(F.silu(gate) * up, self.down)
        return (out, gate_up) if self.leak else out


def _mlp_graph(rows, **kwargs):
    gm = torch.fx.symbolic_trace(_GatedMLP(**kwargs))
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    FakeTensorProp(gm, mode).propagate(torch.randn(rows, 32, dtype=torch.bfloat16))
    return gm.graph


@pytest.mark.parametrize("rows", [1, 4, 16])
def test_a_gated_activation_folds_into_its_gate_up_projection(monkeypatch, rows):
    _route(monkeypatch, {rows: "triton"}, gated="fused")
    graph = _mlp_graph(rows)
    SkinnyLinearPrePass()(graph)
    targets = _targets(graph)
    assert _calls(graph, torch.ops.mstar.decode_gemv_silu_mul.default) == 1
    assert _calls(graph, torch.ops.mstar.decode_gemv.default) == 1, "the down projection"
    assert F.silu not in targets and F.linear not in targets
    graph.lint()


@pytest.mark.parametrize(
    ("kwargs", "entry"),
    [({}, "unfused"), ({}, None), ({"bias": True}, "fused"), ({"leak": True}, "fused")],
    ids=["table-says-unfused", "no-table", "biased", "gate-up-used-elsewhere"],
)
def test_a_gated_activation_stays_unfused_otherwise(monkeypatch, kwargs, entry):
    _route(monkeypatch, {4: "triton"}, gated=entry)
    graph = _mlp_graph(4, **kwargs)
    SkinnyLinearPrePass()(graph)
    assert _calls(graph, torch.ops.mstar.decode_gemv_silu_mul.default) == 0
    assert F.silu in _targets(graph)
    graph.lint()
