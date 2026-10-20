"""The graph passes that send 1-2 row linears to ``mstar::skinny_gemm``, and
when the ROCm platform installs them. CPU only: graphs are traced with fake
tensors and only rewritten, never run."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

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
    ("num_tokens", "skinny"), [(1, True), (2, True), (3, False), (4, False), (32, False), (None, False)],
)
def test_rocm_installs_the_passes_for_two_rows_or_fewer(monkeypatch, num_tokens, skinny):
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
