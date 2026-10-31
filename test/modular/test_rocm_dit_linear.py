"""The pass that sends a flow transformer's large GEMMs to ``mstar::dit_linear`` (a tuned hipBLASLt solution per
shape), and when the platforms install it. CPU only: graphs are traced with fake tensors and only rewritten."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from mstar.kernels import configs
from mstar.model.components.diffusion import compile_utils
from mstar.platforms import RocmPlatform, rocm_dit_gemm
from mstar.platforms.interface import Platform
from mstar.platforms.rocm_dit_gemm import DitLinearPass


def _graph(rows, n=64, k=32):
    x, w = torch.randn(rows, k, dtype=torch.bfloat16), torch.randn(n, k, dtype=torch.bfloat16)
    return make_fx(F.linear, tracing_mode="fake")(x, w, None).graph


def _calls(graph):
    return sum(n.op == "call_function" and n.target is torch.ops.mstar.dit_linear.default for n in graph.nodes)


def _table(monkeypatch, entries):
    monkeypatch.setattr(rocm_dit_gemm, "dit_linear_table", lambda device_index=0: entries)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_only_the_shapes_the_table_names_are_rewritten(monkeypatch):
    _table(monkeypatch, {"rows8-n64-k32": {"impl": "hipblaslt", "solution": 7}})
    named, other = _graph(8), _graph(16)
    DitLinearPass()(named)
    DitLinearPass()(other)
    assert _calls(named) == 1 and torch.ops.aten.mm.default not in [n.target for n in named.nodes]
    assert _calls(other) == 0
    named.lint()


def test_the_table_is_part_of_the_compile_key(monkeypatch):
    _table(monkeypatch, {"rows8-n64-k32": {"impl": "hipblaslt", "solution": 7}})
    first = DitLinearPass().uuid()
    _table(monkeypatch, {"rows8-n64-k32": {"impl": "hipblaslt", "solution": 8}})
    assert DitLinearPass().uuid() != first


def test_a_table_from_another_stack_is_ignored(monkeypatch):
    rocm_dit_gemm.dit_linear_table.cache_clear()
    facts = type("F", (), {"arch": "gfx942", "product": "mi325x"})()
    platform = type("P", (), {"device_facts": lambda self, device=None: facts})()
    monkeypatch.setattr("mstar.platforms.current_platform", platform)
    entry = {"rows8-n64-k32": {"impl": "hipblaslt", "solution": 7}, "default": {}}
    monkeypatch.setattr(configs, "load_config", lambda kernel, arch, product=None, root=None: {
        "classes": entry, "provenance": {"stack": "torch-0-hip-0-triton-0"}})
    try:
        assert rocm_dit_gemm.dit_linear_table(0) == {}
        rocm_dit_gemm.dit_linear_table.cache_clear()
        monkeypatch.setattr(rocm_dit_gemm, "_stack", lambda: "torch-0-hip-0-triton-0")
        assert rocm_dit_gemm.dit_linear_table(0) == {"rows8-n64-k32": entry["rows8-n64-k32"]}
    finally:
        rocm_dit_gemm.dit_linear_table.cache_clear()


def test_only_rocm_installs_the_pass(monkeypatch):
    tuned = DitLinearPass()
    monkeypatch.setattr(rocm_dit_gemm, "dit_linear_pass", lambda: tuned)
    assert RocmPlatform().dit_inductor_overrides() == {"post_grad_custom_post_pass": tuned}
    monkeypatch.setattr(rocm_dit_gemm, "dit_linear_pass", lambda: None)
    assert RocmPlatform().dit_inductor_overrides() == {}
    assert Platform.dit_inductor_overrides(object()) == {}


@pytest.mark.parametrize("tuned_linears", [False, True])
def test_compile_site_applies_the_overrides_only_when_a_model_opts_in(monkeypatch, tuned_linears):
    seen = {}

    def wrap(compiled, *, eager_rounding, overrides=None):
        seen["overrides"] = overrides
        return compiled

    monkeypatch.setattr(compile_utils, "with_inductor_rounding", wrap)
    monkeypatch.setattr(compile_utils.torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setattr(compile_utils.current_platform, "dit_inductor_overrides", lambda: {"a": 1}, raising=False)
    compile_utils.compile_transformer_forward(torch.nn.Linear(2, 2), tuned_linears=tuned_linears)
    assert seen["overrides"] == ({"a": 1} if tuned_linears else None)
