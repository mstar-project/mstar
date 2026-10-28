"""The kernel-op registry and its tuned-config tables (mstar/kernels)."""

from __future__ import annotations

import json

import pytest
import torch

from mstar.kernels import Backend, DeviceKey, KernelOp, registry, table
from mstar.kernels.table import TABLE_DIR, TableEntry, file_stem, find_entry, parse_table, write_entry

PROV = {"device": "Test GPU", "date": "2026-10-28", "harness": "test"}


def _table(op: str, *entries: dict) -> dict:
    return {"format": 1, "op": op, "entries": list(entries)}


def _entry(shape=None, rows=None, dtype="bfloat16", backend="triton", **extra) -> dict:
    return {"dtype": dtype, "shape": shape or {"E": 8, "N": 64, "K": 32, "top_k": 2}, "backend": backend,
            "provenance": dict(PROV), "rows": rows or {"1": {"x": 1}, "64": {"x": 64}}, **extra}


def _write(directory, op, stem, data) -> None:
    path = directory / op / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(table.TABLES_ENV, raising=False)
    registry.reset_lookups()
    yield
    registry.reset_lookups()


def test_file_stem_matches_device_names():
    assert file_stem("NVIDIA H100 80GB HBM3") == "NVIDIA_H100_80GB_HBM3"
    assert file_stem("AMD Instinct MI300X") == "AMD_Instinct_MI300X"
    assert DeviceKey("NVIDIA H100 80GB HBM3", "sm_90").stems() == ("NVIDIA_H100_80GB_HBM3", "sm_90")


def test_row_takes_the_nearest_bucket_and_the_smaller_on_a_tie():
    (entry,) = parse_table(_table("op", _entry(rows={"1": {"x": 1}, "4": {"x": 4}, "16": {"x": 16}})), "op", "t")
    assert [entry.row(m)["x"] for m in (1, 2, 3, 6, 10, 11, 1000)] == [1, 1, 4, 4, 4, 16, 16]
    (single,) = parse_table(_table("op", _entry(rows={"*": {"x": 0}})), "op", "t")
    assert single.row() == {"x": 0} and single.row(5) == {"x": 0}


@pytest.mark.parametrize("bad, match", [
    ({"format": 2, "op": "op", "entries": []}, "format"),
    (_table("other", _entry()), "for op 'other'"),
    (_table("op", _entry(rows={"one": {}})), "row key"),
    (_table("op", _entry(rows={"*": {}, "1": {}})), "only row"),
    (_table("op", _entry(shape={"E": "8"})), "integers"),
    (_table("op", _entry(dtype="bf17")), "torch dtype"),
    (_table("op", _entry(), _entry()), "second entry"),
    (_table("op", {**_entry(), "provenance": {"device": "x"}}), "provenance lacks"),
])
def test_malformed_tables_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        parse_table(bad, "op", "t")


def test_lookup_order_extra_dirs_then_package_and_gpu_then_arch(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    dev = DeviceKey("Test GPU", "sm_00")
    shape = (("E", 8), ("K", 32), ("N", 64), ("top_k", 2))
    _write(second, "op", "Test_GPU", _table("op", _entry(rows={"1": {"from": "b-gpu"}})))
    _write(first, "op", "sm_00", _table("op", _entry(rows={"1": {"from": "a-arch"}})))
    assert find_entry("op", dev, "bfloat16", shape, "triton", (first, second)).row(1) == {"from": "a-arch"}
    _write(first, "op", "Test_GPU", _table("op", _entry(rows={"1": {"from": "a-gpu"}})))
    table.load_table.cache_clear()
    assert find_entry("op", dev, "bfloat16", shape, "triton", (first, second)).row(1) == {"from": "a-gpu"}
    # dtype, backend and every shape key must match
    assert find_entry("op", dev, "float16", shape, "triton", (first, second)) is None
    assert find_entry("op", dev, "bfloat16", shape, "aiter", (first, second)) is None
    assert find_entry("op", dev, "bfloat16", (*shape[:3], ("top_k", 4)), "triton", (first, second)) is None


def test_table_dirs_put_the_env_dirs_first(tmp_path, monkeypatch):
    monkeypatch.setenv(table.TABLES_ENV, f"{tmp_path / 'x'}:{tmp_path / 'y'}")
    assert table.table_dirs() == (tmp_path / "x", tmp_path / "y", TABLE_DIR)


def test_write_entry_replaces_its_key_and_keeps_the_rest(tmp_path):
    path = tmp_path / "op" / "Test_GPU.json"

    def entry(top_k, x):
        return TableEntry(op="op", dtype="bfloat16", shape=(("E", 8), ("top_k", top_k)), backend="triton",
                          rows=((1, {"x": x}), (8, {"x": x, "down": {"y": 1}})), provenance=PROV, source="",
                          measured={1: {"us": 2.0, "default_us": 3.0}})

    write_entry(path, entry(4, 1))
    write_entry(path, entry(2, 2))
    write_entry(path, entry(4, 3))
    entries = parse_table(json.loads(path.read_text()), "op", str(path))
    assert [(dict(e.shape)["top_k"], e.row(1)["x"]) for e in entries] == [(2, 2), (4, 3)]
    assert entries[0].measured == {1: {"us": 2.0, "default_us": 3.0}}
    # one line per row, so a re-sweep diffs bucket by bucket
    assert sum('"8": {' in line for line in path.read_text().splitlines()) == 2


OP = KernelOp(name="test_op", shape=("E", "N", "K", "top_k"), bucket="M",
              backends=(Backend("triton"), Backend("never", lambda d, t, s: "not here")))


def test_shape_keys_must_match_the_op():
    with pytest.raises(ValueError, match="keyed by"):
        OP.shape_key({"E": 8, "N": 64})
    assert OP.format_shape({"top_k": 2, "K": 32, "N": 64, "E": 8}) == "E=8,N=64,K=32,top_k=2"
    assert OP.backend("never").unavailable(torch.device("cpu"), torch.bfloat16, {}) == "not here"
    with pytest.raises(KeyError, match="no backend"):
        OP.backend("cuda")


def test_lookup_resolves_once_and_logs_the_sweep_command(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(table.TABLES_ENV, str(tmp_path))
    monkeypatch.setattr(registry, "_device_index", lambda device: 0)
    monkeypatch.setattr(registry, "device_key", lambda index: DeviceKey("Test GPU", "sm_00"))
    shape = {"E": 8, "N": 64, "K": 32, "top_k": 2}
    with caplog.at_level("INFO", logger="mstar.kernels.registry"):
        assert OP.lookup("triton", torch.bfloat16, shape) is None
    assert "python -m benchmark.kernels.sweep test_op --dtype bfloat16 --shape E=8,N=64,K=32,top_k=2" in caplog.text
    _write(tmp_path, "test_op", "Test_GPU", _table("test_op", _entry()))
    # resolved once per process: a table written later is not seen until reset
    assert OP.lookup("triton", torch.bfloat16, shape) is None
    registry.reset_lookups()
    assert OP.lookup("triton", torch.bfloat16, shape).row(3) == {"x": 1}


def test_lookup_off_gpu_finds_nothing():
    assert OP.lookup("triton", torch.bfloat16, {"E": 8, "N": 64, "K": 32, "top_k": 2},
                     torch.device("cpu")) is None


def test_ops_register_once_by_name():
    assert registry.get_op("moe_gemm").shape == ("E", "N", "K", "top_k")
    assert "moe_gemm" in registry.op_names()
    registry.register(OP)
    assert registry.register(OP) is OP
    with pytest.raises(ValueError, match="already registered"):
        registry.register(KernelOp(name="test_op", shape=("E",), bucket=None, backends=()))
    with pytest.raises(KeyError, match="no kernel op"):
        registry.get_op("nope")


MOE_ROW_KEYS = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"}


@pytest.mark.parametrize("path", sorted(TABLE_DIR.glob("*/*.json")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shipped_tables_are_well_formed(path):
    """Every table in the package parses, is filed under its op, and keys its entries by
    that op's shape. moe_gemm rows hold integer tiles the kernel takes."""
    op = registry.get_op(path.parent.name)
    entries = parse_table(json.loads(path.read_text()), op.name, str(path))
    assert entries
    for entry in entries:
        assert {name for name, _ in entry.shape} == set(op.shape)
        op.backend(entry.backend)
        assert file_stem(entry.provenance["device"]) == path.stem or entry.provenance.get("arch") == path.stem
        if op.name == "moe_gemm":
            for _, row in entry.rows:
                assert MOE_ROW_KEYS <= set(row)
                down = row.get("down", {})
                assert "BLOCK_SIZE_M" not in down
                assert all(type(v) is int for k, v in {**row, **down}.items() if k != "down")
