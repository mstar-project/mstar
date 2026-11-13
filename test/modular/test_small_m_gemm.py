"""``mstar.utils.small_m_gemm``: without a table every product is ``F.linear`` / ``torch.mm``; a table routes a
shape and M to its backend for that GPU, the first measured M at or above the rows', and only where the backend
takes the output dtype and M."""
import json

import pytest
import torch
import torch.nn.functional as F

from mstar.utils import small_m_gemm


@pytest.fixture(autouse=True)
def _no_tables(monkeypatch):
    monkeypatch.setattr(small_m_gemm, "_TABLES", {})


def test_without_a_table_it_is_f_linear():
    x, w = torch.randn(3, 64), torch.randn(32, 64)
    assert torch.equal(small_m_gemm.linear(x, w), F.linear(x, w))
    out = torch.empty(3, 32)
    assert small_m_gemm.linear(x, w, out=out) is out and torch.equal(out, F.linear(x, w))
    assert small_m_gemm.backend_for(1, 32, 64, torch.bfloat16, torch.device("cpu")) == "torch"


def test_a_table_routes_shapes_by_capability_and_m(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"capability": [10, 3], "entries": [
        {"n": 6288, "k": 7168, "out": "bf16", "m": 1, "backend": "fi_cute_dsl"},
        {"n": 6288, "k": 7168, "out": "bf16", "m": 8, "backend": "fi_tgv"},
        {"n": 6288, "k": 7168, "out": "bf16", "m": 64, "backend": "fi_cute_dsl"},
        {"n": 6016, "k": 7168, "out": "fp32", "m": 16, "backend": "fi_cublaslt"},
    ]}))
    small_m_gemm.load_table(str(path))
    cuda = torch.device("cuda", 0)
    monkeypatch.setattr(small_m_gemm, "_capability", lambda device: (10, 3))
    pick = small_m_gemm.backend_for
    assert pick(1, 6288, 7168, torch.bfloat16, cuda) == "fi_cute_dsl"
    assert pick(2, 6288, 7168, torch.bfloat16, cuda) == "fi_tgv"  # the first measured M at or above 2: 8
    assert pick(64, 6288, 7168, torch.bfloat16, cuda) == "torch"  # cute-dsl takes M <= 32 only
    assert pick(128, 6288, 7168, torch.bfloat16, cuda) == "torch"  # past the table
    assert pick(4, 6016, 7168, torch.float32, cuda) == "fi_cublaslt"
    assert pick(4, 6016, 7168, torch.bfloat16, cuda) == "torch"  # another output dtype
    assert pick(1, 1536, 7168, torch.bfloat16, cuda) == "torch"  # a shape it doesn't name
    monkeypatch.setattr(small_m_gemm, "_capability", lambda device: (9, 0))
    assert pick(1, 6288, 7168, torch.bfloat16, cuda) == "torch"  # another GPU
    assert pick(1, 6288, 7168, torch.bfloat16, torch.device("cpu")) == "torch"


def test_a_table_naming_an_unknown_backend_is_refused(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"capability": [10, 3], "entries": [
        {"n": 1, "k": 1, "out": "bf16", "m": 1, "backend": "dsv3_fused_a"}]}))
    with pytest.raises(ValueError, match="unknown backend"):
        small_m_gemm.load_table(str(path))
