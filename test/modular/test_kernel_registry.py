"""The kernel-op registry: tables of tuned configs, the once-per-shape lookup, and the sweep that
writes them (mstar/kernels, benchmark/kernels)."""

from __future__ import annotations

import json
import math

import pytest
import torch

from mstar.kernels import Choice, DeviceKey, Impl, KernelOp, Tolerance, registry, table
from mstar.kernels.table import TABLE_DIR, TableEntry, file_stem, find_entries, parse_table, write_entry

PROV = {"device": "Test GPU", "date": "2026-10-30", "harness": "test"}
GPU = torch.device("cuda", 0)


def _table(op: str, *entries: dict) -> dict:
    return {"format": 2, "op": op, "entries": list(entries)}


def _entry(shape=None, rows=None, dtype="bfloat16", **extra) -> dict:
    return {"dtype": dtype, "shape": shape or {"N": 64, "K": 32}, "provenance": dict(PROV),
            "rows": rows or {"1": {"impl": "fast", "config": {"x": 1}}, "*": {"impl": "big"}}, **extra}


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


@pytest.fixture
def on_gpu(tmp_path, monkeypatch):
    """Lookups on a fake GPU named "Test GPU" (sm_00), reading tables from tmp_path only."""
    monkeypatch.setenv(table.TABLES_ENV, str(tmp_path))
    monkeypatch.setattr(table, "TABLE_DIR", tmp_path / "package")
    monkeypatch.setattr(registry, "_device", lambda device: GPU)
    monkeypatch.setattr(registry, "device_key", lambda device: DeviceKey("Test GPU", "sm_00"))
    registry.reset_lookups()
    return tmp_path


def _op(name="test_op") -> KernelOp:
    return KernelOp(name=name, shape=("N", "K"), bucket="M",
                    impls=(Impl("fast", tolerance=Tolerance(2**-6)), Impl("big", tolerance=Tolerance(2**-3))))


SHAPE = {"N": 64, "K": 32}


# --- tables ------------------------------------------------------------------------------------

def test_rows_name_their_implementation_and_star_sorts_last():
    (entry,) = parse_table(_table("op", _entry(rows={
        "*": {"impl": "big"}, "16": {"impl": "fast", "config": {"x": 16}}, "2": {"impl": "fast"}})), "op", "t")
    assert entry.buckets() == (2, 16, None)
    assert entry.rows[0][1] == Choice("fast") and dict(entry.rows[1][1].config) == {"x": 16}
    assert entry.impls() == ("fast", "big")


@pytest.mark.parametrize("bad, match", [
    ({"format": 1, "op": "op", "entries": []}, "format"),
    (_table("other", _entry()), "for op 'other'"),
    (_table("op", _entry(rows={"one": {"impl": "fast"}})), "rows key"),
    (_table("op", _entry(rows={"1": {"x": 1}})), "naming its 'impl'"),
    (_table("op", _entry(rows={"1": {"impl": "fast", "config": [1]}})), "'config' must be an object"),
    (_table("op", _entry(rows={"1": {"impl": "fast", "tiles": {}}})), "unknown keys"),
    (_table("op", _entry(backend="triton")), "unknown keys"),
    (_table("op", _entry(shape={"N": "8"})), "integers"),
    (_table("op", _entry(dtype="bf17")), "torch dtype"),
    (_table("op", _entry(), _entry()), "second entry"),
    (_table("op", {**_entry(), "provenance": {"device": "x"}}), "provenance lacks"),
    (_table("op", _entry(measured={"1": {"us": "fast"}})), "map names to numbers"),
])
def test_malformed_tables_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        parse_table(bad, "op", "t")


def test_lookup_order_extra_dirs_then_package_and_gpu_then_arch(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    dev = DeviceKey("Test GPU", "sm_00")
    shape = (("K", 32), ("N", 64))

    def rows(label):
        return {"*": {"impl": "fast", "config": {"from": label}}}

    _write(second, "op", "Test_GPU", _table("op", _entry(rows=rows("b-gpu"))))
    _write(first, "op", "sm_00", _table("op", _entry(rows=rows("a-arch"))))
    _write(first, "op", "Test_GPU", _table("op", _entry(rows=rows("a-gpu"))))
    found = [e.rows[0][1].config["from"] for e in find_entries("op", dev, "bfloat16", shape, (first, second))]
    assert found == ["a-gpu", "a-arch", "b-gpu"]
    # dtype and every shape key must match
    assert not list(find_entries("op", dev, "float16", shape, (first, second)))
    assert not list(find_entries("op", dev, "bfloat16", (("K", 32), ("N", 65)), (first, second)))


def test_table_dirs_put_the_env_dirs_first(tmp_path, monkeypatch):
    monkeypatch.setenv(table.TABLES_ENV, f"{tmp_path / 'x'}:{tmp_path / 'y'}")
    assert table.table_dirs() == (tmp_path / "x", tmp_path / "y", TABLE_DIR)


def test_write_entry_replaces_its_shape_and_keeps_the_rest(tmp_path):
    path = tmp_path / "op" / "Test_GPU.json"

    def entry(k, x):
        return TableEntry(op="op", dtype="bfloat16", shape=(("K", k), ("N", 64)), provenance=PROV, source="",
                          rows=((2, Choice("fast", {"x": x})), (None, Choice("big"))),
                          measured={2: {"us": 2.0, "default_us": 3.0}})

    write_entry(path, entry(64, 1))
    write_entry(path, entry(32, 2))
    write_entry(path, entry(64, 3))
    entries = parse_table(json.loads(path.read_text()), "op", str(path))
    assert [(dict(e.shape)["K"], e.rows[0][1].config["x"]) for e in entries] == [(32, 2), (64, 3)]
    assert entries[0].measured == {2: {"us": 2.0, "default_us": 3.0}}
    assert entries[0].rows[1] == (None, Choice("big"))
    # one line per row, so a re-sweep diffs bucket by bucket
    assert sum('"2": {"impl"' in line for line in path.read_text().splitlines()) == 2


# --- ops and their rows --------------------------------------------------------------------------

def test_an_op_names_distinct_implementations():
    with pytest.raises(ValueError, match="distinct names"):
        KernelOp(name="x", shape=("N",), bucket="M", impls=(Impl("a"), Impl("a")))
    with pytest.raises(ValueError, match="distinct names"):
        KernelOp(name="x", shape=("N",), bucket="M", impls=())
    op = _op()
    with pytest.raises(ValueError, match="keyed by"):
        op.shape_key({"N": 64})
    assert op.format_shape({"K": 32, "N": 64}) == "N=64,K=32"
    with pytest.raises(KeyError, match="no implementation"):
        op.impl("cuda")


def _rows(*pairs):
    return tuple((key, Choice(name)) for key, name in pairs)


def test_rows_take_counts_up_to_their_key_and_star_the_rest():
    op = _op()
    rows = _rows((2, "fast"), (512, "big"), (None, "fast"))
    assert registry.row_bounds(rows) == (2, 512)
    assert [op.row_for(rows, m).impl for m in (0, 1, 2, 3, 512, 513, 10**6)] == \
        ["fast", "fast", "fast", "big", "big", "fast", "fast"]


@pytest.mark.parametrize("rows, match", [
    ((), "at least one row"),
    (_rows((2, "fast"), (None, "nope")), "no implementation"),
    # a numeric last row would take every larger count, measured or not
    (_rows((2, "fast"), (64, "big")), "must be '\\*'"),
    (_rows((None, "fast"), (2, "big")), "must be '\\*'"),
    (_rows((2, "fast"), (2, "big"), (None, "big")), "ascend"),
    (_rows((8, "fast"), (2, "big"), (None, "big")), "ascend"),
])
def test_rows_must_end_with_star_and_ascend(rows, match):
    with pytest.raises((ValueError, KeyError), match=match):
        _op().check_rows(rows, "t")


def test_an_op_without_a_bucket_has_one_star_row():
    op = KernelOp(name="x", shape=("N",), bucket=None, impls=(Impl("a"),))
    op.check_rows(_rows((None, "a")), "t")
    with pytest.raises(ValueError, match="only row"):
        op.check_rows(_rows((2, "a"), (None, "a")), "t")


# --- lookup -----------------------------------------------------------------------------------

def test_lookup_takes_the_gpu_table_then_the_arch_table(on_gpu, caplog):
    op = _op()
    with caplog.at_level("INFO", logger="mstar.kernels.registry"):
        assert op.lookup(torch.bfloat16, SHAPE) is None
    assert "python -m benchmark.kernels.sweep test_op --dtype bfloat16 --shape N=64,K=32" in caplog.text
    registry.reset_lookups()
    _write(on_gpu, "test_op", "sm_00", _table("test_op", _entry(rows={"*": {"impl": "big"}})))
    assert op.lookup(torch.bfloat16, SHAPE).rows == ((None, Choice("big")),)
    _write(on_gpu, "test_op", "Test_GPU", _table("test_op", _entry(rows={
        "2": {"impl": "fast", "config": {"x": 1}}, "*": {"impl": "big"}})))
    # looked up once per process: the GPU's table is not seen until a reset
    assert op.lookup(torch.bfloat16, SHAPE).rows == ((None, Choice("big")),)
    registry.reset_lookups()
    entry = op.lookup(torch.bfloat16, SHAPE)
    assert entry.source.endswith("Test_GPU.json")
    assert [op.row_for(entry.rows, m) for m in (2, 3)] == [Choice("fast", {"x": 1}), Choice("big")]


@pytest.mark.parametrize("rows, error, match", [
    ({"*": {"impl": "fastest"}}, KeyError, "no implementation 'fastest'"),
    ({"2": {"impl": "fast"}, "512": {"impl": "big"}}, ValueError, "must be '\\*'"),
])
def test_a_table_the_op_cant_read_raises_at_lookup(on_gpu, rows, error, match):
    _write(on_gpu, "test_op", "Test_GPU", _table("test_op", _entry(rows=rows)))
    with pytest.raises(error, match=match):
        _op().lookup(torch.bfloat16, SHAPE)


def test_lookup_off_gpu_finds_nothing():
    assert _op().lookup(torch.bfloat16, SHAPE, torch.device("cpu")) is None


def test_ops_register_once_by_name():
    assert registry.get_op("moe_gemm").shape == ("E", "N", "K", "top_k")
    assert {"moe_gemm", "moe_decode"} <= set(registry.op_names())
    op = _op(name="test_registered")
    registry.register(op)
    assert registry.register(op) is op
    with pytest.raises(ValueError, match="already registered"):
        registry.register(KernelOp(name="test_registered", shape=("E",), bucket=None, impls=(Impl("a"),)))
    with pytest.raises(KeyError, match="no kernel op"):
        registry.get_op("nope")


def test_tolerance_is_relative_to_the_references_largest_value():
    ref = torch.tensor([1.0, -4.0, 0.5])
    tol = Tolerance(2**-3)
    assert tol.error(ref, ref) == 0.0
    assert tol.error(ref + 0.25, ref) == pytest.approx(0.5)  # 0.25 of an allowed 0.5
    assert tol.error(ref + 1.0, ref) > 1.0
    assert math.isinf(tol.error(torch.tensor([1.0, float("nan"), 0.5]), ref))
    assert math.isinf(tol.error(ref[:2], ref))


# --- the sweep's choices -------------------------------------------------------------------

def test_a_bucket_keeps_its_current_row_unless_the_best_beats_it_by_min_gain():
    from benchmark.kernels.sweep import BucketResult, pick_row

    def result(current_us, best_us):
        return BucketResult(bucket=4, current=Choice("fast"), current_us=current_us, default_us=9.0,
                            best=None if math.isnan(best_us) else Choice("big"), best_us=best_us)

    assert pick_row(result(10.0, 9.9), 0.02)[0] == Choice("fast")
    assert pick_row(result(10.0, 9.0), 0.02)[0] == Choice("big")
    assert pick_row(result(10.0, math.nan), 0.02)[0] == Choice("fast")
    # what runs now failed the reference check, so the best replaces it whatever min_gain says
    assert pick_row(result(math.nan, 20.0), 10.0)[0] == Choice("big")
    assert pick_row(result(math.nan, math.nan), 0.02) is None


def test_winners_become_rows_ending_with_star():
    """A run of equal winners is one row keyed by its largest bucket; counts past the last bucket
    take the last winner."""
    from benchmark.kernels.sweep import assemble_rows

    fast1, fast2, big = Choice("fast", {"x": 1}), Choice("fast", {"x": 2}), Choice("big")
    chosen = [(1, fast1), (2, fast2), (3, big), (8, big), (512, big), (1024, fast1), (2048, fast1)]
    rows = assemble_rows(chosen)
    assert rows == ((1, fast1), (2, fast2), (512, big), (None, fast1))
    _op().check_rows(rows, "t")


def test_a_moe_gemm_row_launches_the_tiles_it_was_written_from():
    """The sweep writes a (gate/up, down) pair as one row and launches a row as that pair again,
    the way fused_experts reads it."""
    pytest.importorskip("triton")
    from benchmark.kernels.moe_gemm import MoeGemmSweep, as_row
    from mstar.utils.fused_moe import kernels

    up = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4}
    down = {**up, "BLOCK_SIZE_N": 64, "num_warps": 8}
    row = as_row(up, down)
    assert row.config["down"] == {"BLOCK_SIZE_N": 64, "num_warps": 8}
    assert MoeGemmSweep.__new__(MoeGemmSweep).tiles(row) == (up, down)
    default = kernels.get_default_config(4, 8, 64, 32, 2)
    assert MoeGemmSweep.__new__(MoeGemmSweep).tiles(as_row(default, default)) == (default, default)
    entry = TableEntry(op="moe_gemm", dtype="bfloat16", shape=(), provenance=PROV, source="t",
                       rows=((None, row),))
    assert kernels._entry_tiles(entry) == ((), ((tuple(up.items()), tuple(down.items())),))


# --- the sweep end to end, its GPU work faked: what it writes, and what it refuses to ---------

def _won(m, best):
    """A sweep plugin's ``run_bucket`` result whose winner is ``best``."""
    from benchmark.kernels.sweep import BucketResult

    return BucketResult(bucket=m, current=GROUPED, current_us=math.nan, default_us=math.nan,
                        best=best, best_us=1.0)


def _pair(bj):
    return Choice("triton", {"path": "pair", "gate_up": {"bj": bj}, "down": {"bn": 16}, "router": {"bk": 7}})


GROUPED = Choice("triton", {"path": "grouped", "router": {"bk": 7}})


@pytest.fixture
def fake_moe_decode_sweep(on_gpu, monkeypatch):
    """benchmark.kernels.sweep.sweep over the real moe_decode op and its plugin's row assembly and
    check, with canned winners in place of the GPU timings."""
    pytest.importorskip("triton")
    from benchmark.kernels import moe_decode as plugin
    from benchmark.kernels import sweep as sweep_mod
    from mstar.utils.fused_moe import decode

    winners = {}

    class Sweep(plugin.MoeDecodeSweep):
        def __init__(self, shape, dtype, device, quick=False):
            self.builtin = decode._H200

        def run_bucket(self, m, current):
            return _won(m, winners[m])

    monkeypatch.setattr(sweep_mod, "_sweeper", lambda op: Sweep)
    monkeypatch.setattr(sweep_mod, "device_key", lambda device: DeviceKey("Test GPU", "sm_00"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def run(chosen):
        winners.clear()
        winners.update(chosen)
        return sweep_mod.sweep("moe_decode", {"H": 64, "E": 8, "I": 32, "K": 2}, torch.float8_e4m3fn,
                               sorted(chosen), quick=True, note=None, out_dir=on_gpu / "out", report=[],
                               min_gain=0.02)

    return run, on_gpu / "out" / "moe_decode" / "Test_GPU.json", decode


def test_the_moe_decode_sweep_writes_per_pair_rows_then_the_grouped_star_row(fake_moe_decode_sweep, capsys):
    run, path, decode = fake_moe_decode_sweep
    # per-pair to 3 tokens (two tilings), grouped from 4; a per-pair win past that runs grouped
    assert run({1: _pair(8), 2: _pair(16), 3: _pair(16), 4: GROUPED, 5: _pair(8), 6: GROUPED})
    assert "buckets [5]: per-pair won after the grouped kernels took over at 4" in capsys.readouterr().out
    (entry,) = parse_table(json.loads(path.read_text()), "moe_decode", str(path))
    assert [(key, choice.config["path"]) for key, choice in entry.rows] == [(1, "pair"), (3, "pair"), (None, "grouped")]
    # every row carries the built-in router config, which the sweep doesn't time
    assert {json.dumps(dict(c.config["router"])) for _, c in entry.rows} == {json.dumps(decode._H200.router)}
    tuning = decode._from_entry(entry)
    assert tuning.pair_max_tokens == 3 and decode._tiles(2, tuning)[0] == {"bj": 16}


@pytest.mark.parametrize("winners, match", [
    ({1: _pair(8), 2: _pair(8), 4: _pair(16), 8: _pair(16)}, "grouped kernels won none"),
    ({1: GROUPED, 2: GROUPED}, "won the smallest bucket"),
])
def test_the_moe_decode_sweep_refuses_rows_the_kernels_would_refuse(fake_moe_decode_sweep, capsys, winners, match):
    """Without a grouped winner the crossover is unknown, and an entry the kernels refuse at their
    first call is not written."""
    run, path, _ = fake_moe_decode_sweep
    assert not run(winners)
    assert match in capsys.readouterr().out
    assert not path.exists()


# --- the shipped tables and the ops that read them ------------------------------------------

MOE_ROW_KEYS = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"}


@pytest.mark.parametrize("path", sorted(TABLE_DIR.glob("*/*.json")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shipped_tables_are_well_formed(path):
    """Every table in the package parses, is filed under its op, keys its entries by that op's
    shape, ends each with a '*' row, and holds configs the op's kernels take."""
    op = registry.get_op(path.parent.name)
    entries = parse_table(json.loads(path.read_text()), op.name, str(path))
    assert entries
    for entry in entries:
        assert {name for name, _ in entry.shape} == set(op.shape)
        op.check_rows(entry.rows, entry.source)
        assert file_stem(entry.provenance["device"]) == path.stem or entry.provenance.get("arch") == path.stem
        if op.name == "moe_gemm":
            from mstar.utils.fused_moe import kernels

            kernels._entry_tiles(entry)
            for _, choice in entry.rows:
                assert MOE_ROW_KEYS <= set(choice.config)
                down = choice.config.get("down", {})
                assert "BLOCK_SIZE_M" not in down
                assert all(type(v) is int for k, v in {**choice.config, **down}.items() if k != "down")
        if op.name == "moe_decode":
            pytest.importorskip("triton")
            from mstar.utils.fused_moe import decode

            tuning = decode._from_entry(entry)
            for tokens in range(1, tuning.pair_max_tokens + 1):
                gate_up, down = decode._tiles(tokens, tuning)
                assert set(gate_up) == {"bj", "bk", "bjs", "bks", "warps"} and set(down) == {"bn", "bc", "warps"}
            assert set(tuning.router) == {"bk", "splits", "warps"}


# --- moe_decode: fused_moe.decode's tiles from the registry, the built-in table otherwise -----

DECODE_SHAPE = (6144, 256, 256, 8)  # GLM-5.2-FP8's experts at TP8: H, E, I, K


@pytest.fixture
def decode_on(on_gpu, monkeypatch):
    """``fused_moe.decode._tuning`` as it looks up on a GPU of the given name, with the tables
    in tmp_path."""
    pytest.importorskip("triton")
    from mstar.utils.fused_moe import decode

    def on(name: str):
        monkeypatch.setattr(registry, "device_key", lambda device: DeviceKey(name, "sm_90"))
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
        registry.reset_lookups()
        return decode

    yield on
    decode._tuning.cache_clear()


def _decode_row(path, bj=8):
    config = {"path": path, "router": {"bk": 256, "splits": 16, "warps": 2}}
    if path == "pair":
        config.update(gate_up={"bj": bj, "bk": 512, "bjs": 32, "bks": 256, "warps": 2},
                      down={"bn": 16, "bc": 256, "warps": 8})
    return {"impl": "triton", "config": config}


def test_moe_decode_takes_a_tables_entry_on_its_gpu(decode_on, on_gpu):
    _write(on_gpu, "moe_decode", "NVIDIA_H200", _table("moe_decode", _entry(
        dict(zip("HEIK", DECODE_SHAPE, strict=True)), dtype="float8_e4m3fn",
        rows={"1": _decode_row("pair"), "8": _decode_row("pair", bj=16), "*": _decode_row("grouped")})))
    decode = decode_on("NVIDIA H200")
    tuning = decode._tuning(GPU, DECODE_SHAPE)
    assert tuning.pair_max_tokens == 8
    assert decode._tiles(1, tuning)[0]["bj"] == 8 and decode._tiles(5, tuning)[0]["bj"] == 16
    assert tuning.router == {"bk": 256, "splits": 16, "warps": 2}


def test_moe_decode_keeps_the_builtin_table_where_none_has_the_shape(decode_on):
    decode = decode_on("NVIDIA H200")
    assert decode._tuning(GPU, DECODE_SHAPE) is decode._H200
    assert decode._tuning(GPU) is decode._H200  # route(): no shape
    decode = decode_on("NVIDIA H100 80GB HBM3")
    assert decode._tuning(GPU, DECODE_SHAPE) is decode._H100


def test_the_moe_decode_sweep_writes_the_tiles_the_builtin_table_launches():
    """A bucket that keeps the built-in table is written as the tiles that table launches there,
    and such a row launches them again."""
    pytest.importorskip("triton")
    from benchmark.kernels.moe_decode import as_row, as_tuning
    from mstar.utils.fused_moe import decode

    for builtin in (decode._H200, decode._H100):
        for m in range(1, 12):
            row = as_row(builtin, m)
            assert row.config is not None
            tuning = as_tuning(row, builtin)
            if m <= builtin.pair_max_tokens:
                assert row.config["path"] == "pair" and decode._tiles(m, tuning) == decode._tiles(m, builtin)
            else:
                assert row.config["path"] == "grouped" and m > tuning.pair_max_tokens
            assert tuning.router == builtin.router


@pytest.mark.parametrize("rows, match", [
    ({"1": "pair"}, "must be '\\*'"),
    ({"*": "pair"}, "ends with the '\\*' row"),
    ({"4": "grouped", "*": "grouped"}, "per-pair ones"),
    ({"1": "pair", "4": "grouped", "8": "pair", "*": "grouped"}, "per-pair ones"),
    ({"*": "grouped"}, "per-pair ones, at least one"),
    ({"1": "pair", "*": {"impl": "triton", "config": {"path": "grouped", "router": {"bk": 1}}}},
     "one router config"),
])
def test_moe_decode_entries_are_per_pair_rows_then_the_grouped_star_row(rows, match):
    pytest.importorskip("triton")
    from mstar.utils.fused_moe import decode

    rows = {k: _decode_row(v) if isinstance(v, str) else v for k, v in rows.items()}
    entry = parse_table(_table("moe_decode", _entry({"H": 64, "E": 8, "I": 32, "K": 2}, rows,
                                                    dtype="float8_e4m3fn")), "moe_decode", "t")[0]
    with pytest.raises(ValueError, match=match):
        registry.get_op("moe_decode").check_rows(entry.rows, "t")
        decode._from_entry(entry)
