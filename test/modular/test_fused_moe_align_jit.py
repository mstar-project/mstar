"""The align op's JIT build is keyed by source content, not checkout path.

cpp_extension's ninja file names the source by absolute path, so compiling each checkout's
own .cu rebuilt the op whenever a different worktree booted.
"""
import pytest

from mstar.utils.fused_moe import align


@pytest.fixture
def ext_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    return tmp_path


def _checkout(root, name, text):
    src = root / name / "moe_align_block_size.cu"
    src.parent.mkdir(parents=True)
    src.write_text(text)
    return str(src)


def test_two_checkouts_of_one_source_share_a_build(ext_root, monkeypatch):
    monkeypatch.setattr(align, "_CSRC", _checkout(ext_root, "wt_a", "// op v1\n"))
    first = align._jit_source()
    monkeypatch.setattr(align, "_CSRC", _checkout(ext_root, "wt_b", "// op v1\n"))
    assert align._jit_source() == first
    name, path = first
    assert name.startswith("_mstar_moe_C_") and open(path).read() == "// op v1\n"
    assert "wt_a" not in path and "wt_b" not in path


def test_a_changed_source_gets_its_own_build(ext_root, monkeypatch):
    monkeypatch.setattr(align, "_CSRC", _checkout(ext_root, "wt_a", "// op v1\n"))
    v1 = align._jit_source()
    monkeypatch.setattr(align, "_CSRC", _checkout(ext_root, "wt_b", "// op v2\n"))
    v2 = align._jit_source()
    assert v1[0] != v2[0] and v1[1] != v2[1]
    assert open(v1[1]).read() == "// op v1\n"  # the other version's source stays
