"""The pass that sends decode linears past the skinny rows to ``mstar::decode_linear``
(a tuned hipBLASLt solution per shape), and when the ROCm platform installs it.
CPU only: graphs are traced with fake tensors and only rewritten, never run."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from mstar.kernels import configs
from mstar.platforms import RocmPlatform, rocm_gemm
from mstar.platforms.rocm_gemm import DecodeLinearPass


def _graph(rows, n=64, k=32):
    x, w = torch.randn(rows, k, dtype=torch.bfloat16), torch.randn(n, k, dtype=torch.bfloat16)
    return make_fx(F.linear, tracing_mode="fake")(x, w, None).graph


def _calls(graph):
    return sum(n.op == "call_function" and n.target is torch.ops.mstar.decode_linear.default for n in graph.nodes)


def _table(monkeypatch, entries):
    monkeypatch.setattr(rocm_gemm, "decode_linear_table", lambda device_index=0: entries)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_only_the_shapes_the_table_names_are_rewritten(monkeypatch):
    _table(monkeypatch, {"rows8-n64-k32": {"impl": "hipblaslt", "solution": 7}})
    named, other = _graph(8), _graph(16)
    DecodeLinearPass()(named)
    DecodeLinearPass()(other)
    assert _calls(named) == 1 and torch.ops.aten.mm.default not in [n.target for n in named.nodes]
    assert _calls(other) == 0
    named.lint()


def test_a_table_from_another_stack_is_ignored(monkeypatch):
    rocm_gemm.decode_linear_table.cache_clear()
    facts = type("F", (), {"arch": "gfx942", "product": "mi325x"})()
    platform = type("P", (), {"device_facts": lambda self, device=None: facts})()
    monkeypatch.setattr("mstar.platforms.current_platform", platform)
    entry = {"rows8-n64-k32": {"impl": "hipblaslt", "solution": 7}, "default": {}}
    monkeypatch.setattr(configs, "load_config", lambda kernel, arch, product=None, root=None: {
        "classes": entry, "provenance": {"stack": "torch-0-hip-0-triton-0"}})
    try:
        assert rocm_gemm.decode_linear_table(0) == {}
        rocm_gemm.decode_linear_table.cache_clear()
        monkeypatch.setattr(rocm_gemm, "_stack", lambda: "torch-0-hip-0-triton-0")
        assert rocm_gemm.decode_linear_table(0) == {"rows8-n64-k32": entry["rows8-n64-k32"]}
    finally:
        rocm_gemm.decode_linear_table.cache_clear()


def test_rocm_installs_it_for_wider_decode_buckets_only(monkeypatch):
    tuned = DecodeLinearPass()
    monkeypatch.setattr(rocm_gemm, "decode_linear_pass", lambda: tuned)
    monkeypatch.setattr(rocm_gemm, "skinny_gemm_passes", lambda: ("pre", "post"))
    platform = RocmPlatform()
    assert platform.inductor_overrides(2, rows=2)["post_grad_custom_post_pass"] == "post"
    # up to 16 rows the decode kernels' pass runs first
    assert platform.inductor_overrides(8, rows=8)["post_grad_custom_post_pass"].passes == ("post", tuned)
    assert platform.inductor_overrides(128, rows=128)["post_grad_custom_post_pass"] is tuned
    # a packed prefill bucket of the same token count is not a decode
    assert platform.inductor_overrides(8, rows=1)["post_grad_custom_post_pass"] == "post"
    assert platform.inductor_overrides(256, rows=256) == {}
    monkeypatch.setattr(rocm_gemm, "decode_linear_pass", lambda: None)
    assert platform.inductor_overrides(8, rows=8)["post_grad_custom_post_pass"] == "post"


def test_the_table_takes_only_what_the_decode_kernels_leave(monkeypatch):
    # both name these shapes; the kernel the decode config picks keeps its own
    entry = {"impl": "hipblaslt", "solution": 7}
    _table(monkeypatch, {"rows8-n64-k32": entry, "rows8-n128-k32": entry})
    monkeypatch.setattr(rocm_gemm, "gemv_impl", lambda rows, n, k: "triton" if n == 128 else "hipblaslt")
    chain = rocm_gemm.ChainedGraphPass(rocm_gemm.SkinnyGemmPass(), DecodeLinearPass())
    left, taken = _graph(8), _graph(8, n=128)
    chain(left)
    chain(taken)
    assert _calls(left) == 1
    assert _calls(taken) == 0
    assert torch.ops.mstar.decode_gemv.default in [n.target for n in taken.nodes]
