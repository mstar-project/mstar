"""The kernel harness on fake kernels, on the CPU.

A fake op (scale a tensor) with candidates that are right, round by
truncation, raise, or fail to build, timed by a fake clock each candidate
advances by its own cost: the oracle keeps the right ones, the search ranks
them by cost, the store keeps and merges records under arch/product/stack and
seeds the next search with its winners, and promotion writes a config the
registry reads. Then the knob-space rules, the roofline, the ISA parsers and
the trace tools on synthetic inputs.
"""

import json

import pytest
import torch

from benchmark.kernels import bench as bench_module
from benchmark.kernels import candidates as cands
from benchmark.kernels import isa, search
from benchmark.kernels.bench import Bench, Timing
from benchmark.kernels.roofline import roofline
from benchmark.kernels.spec import KernelSpec, load_shapes, shape_product
from benchmark.kernels.store import ResultsStore, shape_key
from benchmark.kernels.trace import harvest, kseq, ktrace, kwindow
from mstar.kernels import configs
from mstar.kernels.numerics import Tolerance
from mstar.platforms.facts import DeviceFacts, facts_for

MI300X = facts_for("rocm", name="AMD Instinct MI300X", arch="gfx942", compute_units=304, warp_size=64)
MI355X = facts_for("rocm", name="AMD Instinct MI355X", arch="gfx950", compute_units=256, warp_size=64)
H100 = facts_for("cuda", name="NVIDIA H100 80GB HBM3", arch="sm_90", compute_units=132, warp_size=32)


class Clock:
    """Seconds that pass only when a fake kernel runs."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _inputs(shape, dtype, device, generator=None):
    x = torch.randn(shape["rows"], shape["hidden"], dtype=dtype, device=device, generator=generator)
    return x, 3.0


def _reference(x, scale):
    return x * scale


def _truncate(x: torch.Tensor) -> torch.Tensor:
    return (x.float().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


def _fake_candidates(clock: Clock):
    def kernel(cost_us, truncating=False):
        def build():
            def run(x, scale):
                clock.now += cost_us * 1e-6
                y = x.float() * scale
                return _truncate(y) if truncating else y.to(x.dtype)

            return run

        return build

    def broken():
        def run(x, scale):
            raise RuntimeError("illegal memory access")

        return run

    def unbuildable():
        raise OSError("hipcc: not found")

    def make(facts, budget=None, seeds=None):
        found = [
            cands.Candidate("torch", {}, kernel(5.0)),
            cands.Candidate("truncating", {}, kernel(0.5, truncating=True)),
            cands.Candidate("broken", {}, broken),
            cands.Candidate("unbuildable", {}, unbuildable),
        ]
        points = cands.space({"num_warps": (1, 2, 4, 8), "waves_per_eu": (0, 2)}, facts)
        for knobs in cands.search_order(points, (seeds or {}).get("triton", ()), budget):
            cost = 1.0 + abs(knobs["num_warps"] - 4) + knobs["waves_per_eu"]
            found.append(cands.Candidate("triton", knobs, kernel(cost)))
        return found

    return make


def _spec(clock, shapes=None):
    return KernelSpec(
        name="scale", shapes=shapes or shape_product(rows=(1, 64), hidden=(256,)), dtypes=(torch.bfloat16,),
        inputs=_inputs, reference=_reference, tolerance=Tolerance.rounded_once(),
        bytes_moved=lambda s, d: 4.0 * s["rows"] * s["hidden"], flops=lambda s, d: 0.0,
        candidates=_fake_candidates(clock), shape_class=lambda s: "decode" if s["rows"] <= 16 else "prefill",
    )


@pytest.fixture
def clock():
    return Clock()


def _run(clock, tmp_path, budget=None, shapes=None):
    store = ResultsStore(tmp_path / "store", stack="torch-test-hip-test-triton-test")
    bench = Bench(torch.device("cpu"), calls=8, rounds=3, rotate_bytes=4096, timer=clock)
    records = search.run(_spec(clock, shapes), budget=budget, store=store, bench=bench, facts=MI300X,
                         meta={"job": "test", "sha": "abc"})
    return store, records


def test_the_oracle_keeps_the_right_candidates_and_the_search_ranks_them(clock, tmp_path):
    _, records = _run(clock, tmp_path)
    assert len(records) == 2
    for record in records:
        c = record["candidates"]
        assert c["truncating"]["oracle"]["ok"] is False and "us" not in c["truncating"]
        assert c["truncating"]["oracle"]["mismatched"] == 0, "within an ulp, wrong on the norm"
        assert "illegal memory access" in c["broken"]["oracle"]["error"]
        assert c["unbuildable"]["error"] == "build: OSError: hipcc: not found"
        assert record["best"] == "triton[num_warps=4,waves_per_eu=0]"
        assert c[record["best"]]["us"] == pytest.approx(1.0)
        assert c["torch"]["us"] == pytest.approx(5.0)
        assert c["torch"]["job"] == "test" and c["torch"]["sha"] == "abc"
        floor = 4.0 * record["shape"]["rows"] * 256 / 5300e3
        assert record["floor_us"] == pytest.approx(floor)
        assert c["torch"]["pct_of_floor"] == pytest.approx(100 * floor / 5.0)


def test_records_are_keyed_by_arch_product_stack_and_shape(clock, tmp_path):
    store, _ = _run(clock, tmp_path)
    shape = {"rows": 64, "hidden": 256}
    path = store.path(MI300X, "scale", shape, torch.bfloat16)
    assert path.relative_to(tmp_path / "store").parts == (
        "gfx942", "mi300x", "torch-test-hip-test-triton-test", "scale", "rows64-hidden256-bfloat16.json",
    )
    record = json.loads(path.read_text())
    assert record["kernel"] == "scale" and record["shape_class"] == "prefill"
    assert store.read(MI355X, "scale", shape, torch.bfloat16) is None, "another product keeps its own"


def test_a_second_search_merges_and_is_seeded_by_the_winners(clock, tmp_path):
    store, _ = _run(clock, tmp_path)
    assert store.seeds(MI300X, "scale", "triton")[0] == {"num_warps": 4, "waves_per_eu": 0}
    # a budget of one tries the stored winner, and the record keeps every earlier candidate
    _, records = _run(clock, tmp_path, budget=1)
    assert [n for n in records[0]["candidates"] if n.startswith("triton")] == ["triton[num_warps=4,waves_per_eu=0]"]
    stored = store.read(MI300X, "scale", {"rows": 1, "hidden": 256}, torch.bfloat16)
    assert len([n for n in stored["candidates"] if n.startswith("triton")]) == 8
    assert stored["best"] == "triton[num_warps=4,waves_per_eu=0]"


def test_promotion_writes_a_config_the_registry_reads(clock, tmp_path):
    store, _ = _run(clock, tmp_path)
    path = store.promote(MI300X, "scale", "triton", lambda s: "decode" if s["rows"] <= 16 else "prefill",
                         root=tmp_path / "configs", provenance={"job": "test"})
    assert path == tmp_path / "configs" / "gfx942" / "scale.json"
    configs._read.cache_clear()
    config = configs.load_config("scale", "gfx942", "mi300x", root=tmp_path / "configs")
    assert configs.select(config, "decode") == {"num_warps": 4, "waves_per_eu": 0}
    assert config["provenance"]["job"] == "test"
    with pytest.raises(ValueError, match="no passing broken results"):
        store.promote(MI300X, "scale", "broken", lambda s: None, root=tmp_path / "configs")


def test_the_table_compares_the_best_with_the_baseline(clock, tmp_path):
    _, records = _run(clock, tmp_path)
    rows = [search.summarize(r) for r in records]
    assert rows[0]["baseline"] == "torch" and rows[0]["speedup"] == pytest.approx(5.0)
    assert set(rows[0]["failed"]) == {"broken", "truncating", "unbuildable"}
    text = search.table(rows, "scale on a fake device")
    assert text.splitlines()[0] == "scale on a fake device"
    assert "| rows1-hidden256-bfloat16 |" in text and "| torch | 5.00 |" in text


# --- knob spaces ---------------------------------------------------------------


def test_knob_rules_follow_the_device():
    assert cands.valid_for({"num_stages": 3}, MI300X) == "more than two stages need 160 KB of LDS"
    assert cands.valid_for({"num_stages": 3}, MI355X) is None
    assert cands.valid_for({"kpack": 2}, MI355X) == "kpack is a gfx942 knob"
    assert cands.valid_for({"kpack": 2}, MI300X) is None
    assert cands.valid_for({"async_lds": True}, MI300X) == "direct-to-LDS loads are gfx950's"
    assert cands.valid_for({"waves_per_eu": 2}, H100) == "waves_per_eu is an AMD knob"
    assert cands.valid_for({"xcd_remap": True}, H100) == "xcd_remap is an AMD knob"
    assert cands.valid_for({"xcd_remap": False, "kpack": 1}, H100) is None
    assert cands.valid_for({"waves_per_wg": 32}, MI300X) == "more than 1024 threads in a workgroup"
    assert cands.valid_for({"num_warps": 32}, H100) is None


def test_the_plans_knob_spaces_are_complete():
    assert {"BLOCK_M", "num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim", "kpack", "xcd_remap",
            "schedule_hint", "split_k"} <= set(cands.TRITON_KNOBS)
    assert {"tile", "waves_per_wg", "split_k", "xcd_map", "lds_layout", "unroll", "prefetch", "setprio",
            "sched_group_barrier", "async_lds"} <= set(cands.HIP_KNOBS)
    stages = {p["num_stages"] for p in cands.space({"num_stages": cands.TRITON_KNOBS["num_stages"]}, MI300X)}
    assert stages == {1, 2}


def test_search_order_puts_known_winners_first_under_a_budget():
    points = cands.space({"a": (1, 2, 3, 4)}, MI300X)
    order = cands.search_order(points, seeds=[{"a": 3}, {"a": 9}], budget=2)
    assert order[0] == {"a": 3} and len(order) == 2
    assert cands.search_order(points, seed=1) == cands.search_order(points, seed=1)
    assert cands.label("triton", {"b": 2, "a": 1}) == "triton[a=1,b=2]"


# --- timing, roofline, ISA ---------------------------------------------------------


def test_operand_copies_rotate_past_the_cache():
    assert bench_module.copies_needed(100 << 20, 512 << 20, max_copies=64, max_bytes=4 << 30) == 7
    assert bench_module.copies_needed(2 << 30, 512 << 20, max_copies=64, max_bytes=4 << 30) == 1
    assert bench_module.copies_needed(1 << 10, 512 << 20, max_copies=64, max_bytes=4 << 30) == 64
    assert bench_module.copies_needed(1 << 30, 512 << 20, max_copies=64, max_bytes=1 << 30) == 1


def test_the_cpu_bench_times_with_its_clock(clock):
    def cost(us):
        def run(x):
            clock.now += us * 1e-6

        return run

    bench = Bench(torch.device("cpu"), calls=10, rounds=3, timer=clock)
    timings = bench.time({"fast": cost(2.0), "slow": cost(7.0)}, [(torch.zeros(1),), (torch.zeros(1),)])
    assert isinstance(timings["fast"], Timing) and timings["fast"].us == pytest.approx(2.0)
    assert timings["slow"].us == pytest.approx(7.0) and timings["slow"].calls == 10
    assert bench.node_floor_us() is None


def test_the_roofline_bound():
    memory = roofline(5.3e9, 0.0, torch.bfloat16, MI300X)
    assert memory.floor_us == pytest.approx(1000.0) and memory.bound == "memory"
    assert memory.pct_of_floor(2000.0) == pytest.approx(50.0)
    compute = roofline(1e3, 2 * 1307.4e9, torch.bfloat16, MI300X)
    assert compute.bound == "compute" and compute.floor_us == pytest.approx(2000.0)
    unknown = roofline(1e9, 0.0, torch.bfloat16, DeviceFacts(platform="rocm"))
    assert unknown.floor_us is None and unknown.pct_of_floor(1.0) is None


def test_isa_listing_and_metadata_parsers():
    listing = "; NumVgprs: 52\n; NumAgprs: 0\n; NumSgprs: 24\n; ScratchSize: 0\n; Occupancy: 8\n; LDSByteSize: 512"
    assert isa.parse_listing(listing) == {
        "vgprs": 52, "agprs": 0, "sgprs": 24, "scratch_bytes": 0, "occupancy": 8, "lds_bytes": 512,
    }
    notes = """  AMDGPU Metadata: ---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .name:           x
        .size:           8
    .group_segment_fixed_size: 1024
    .name:           _Z15rms_norm_kernelILi256ELi4EEvv
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .vgpr_count:     22
    .vgpr_spill_count: 0
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
...
"""
    assert isa.parse_notes(notes) == {
        "_Z15rms_norm_kernelILi256ELi4EEvv": {
            "vgprs": 22, "agprs": 0, "sgprs": 18, "vgpr_spills": 0, "lds_bytes": 1024, "scratch_bytes": 0,
        },
    }
    assert isa.triton_stats(None) == {"error": "no compiled kernel"}


# --- served-step tools ------------------------------------------------------------


def _trace(steps=5):
    """A chrome trace of `steps` decode steps of embed, gemm, norm, launched by graph replay."""
    events, ts = [], 0.0
    for step in range(steps):
        events.append({"cat": "cuda_runtime", "name": "hipGraphLaunch", "args": {"correlation": step}})
        for name, dur in (("embedding_kernel", 2.0), ("Cijk_gemm", 30.0), ("RMSNormKernel", 3.0)):
            events.append({"cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"correlation": step}})
            ts += dur + 1.0
        events.append({"cat": "cpu_op", "name": "aten::mm", "ts": ts, "dur": 1.0,
                       "args": {"Input Dims": [[2, 4096], [4096, 12288]]}})
        events.append({"cat": "cpu_op", "name": "mstar::rms_norm", "ts": ts, "dur": 1.0,
                       "args": {"Input Dims": [[2, 4096], [4096], [], []]}})
        ts += 10.0
    return {"traceEvents": events}


def test_ktrace_cuts_steps_and_tabulates_categories(tmp_path):
    (tmp_path / "w.json").write_text(json.dumps(_trace()))
    text, result = ktrace.report(str(tmp_path), anchor="embedding")
    table = result["kinds"]["3"]
    assert table["steps"] == 4 and table["kernels"] == 3
    by_cat = {r["category"]: r for r in table["categories"]}
    assert by_cat["gemm"]["us"] == pytest.approx(30.0) and by_cat["add_norm"]["kernels"] == 1
    assert table["launches"] == {"graph": {"kernels": 3, "us": pytest.approx(35.0)}}
    assert table["wall_us"] == pytest.approx(48.0) and table["busy_us"] == pytest.approx(35.0)
    assert "== step kind: 3 kernels, 4 steps ==" in text
    path = tmp_path / "w.json"
    assert [name for _, name in kseq.sequence(str(path), "embedding")] == [
        "embedding_kernel", "Cijk_gemm", "RMSNormKernel",
    ]
    assert "graph" in kwindow.totals(str(path))


def test_harvest_from_a_config_a_server_log_and_a_trace(tmp_path):
    config = {"text_config": {
        "model_type": "qwen3_5_text", "hidden_size": 4096, "intermediate_size": 12288, "vocab_size": 248320,
        "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256, "attn_output_gate": True,
        "linear_num_value_heads": 32, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }}
    decode = harvest.qwen3_5_decode(config, rows=[1])
    linears = {dict(k)["n"]: c for k, c in decode["linear"].items()}
    assert linears == {12352: 3, 4096: 4, 10240: 1, 24576: 4, 248320: 1}
    log = harvest.from_server_log("x skinny GEMM: rows=2 k=4096 n=12352 bias=False\n" * 2)
    trace = harvest.from_trace(_trace(2)["traceEvents"])
    merged = harvest.merge(decode, log, trace)
    assert {"rows": 2, "n": 12352, "k": 4096, "count": 2} in merged["linear"]
    assert {"rows": 2, "n": 12288, "k": 4096, "count": 2} in merged["linear"]
    assert {"rows": 2, "hidden": 4096, "count": 2} in merged["rms_norm"]
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps({"model": "m", "sources": [], "ops": merged}))
    assert load_shapes(path, "rms_norm")[0] == {"rows": 1, "hidden": 4096}
    assert harvest.describe("/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/c2022362357621/config.json") == (
        "Qwen/Qwen3.5-9B@c20223623576"
    )


def test_the_built_in_specs_list_their_candidates(capsys):
    assert search.main(["--spec", "rms_norm", "--list", "--families", "registry,triton", "--budget", "2"]) == 0
    names = capsys.readouterr().out.split()
    assert names[0] == "registry" and len([n for n in names if n.startswith("triton[")]) == 2
    assert search.main(["--spec", "linear", "--list"]) == 0
    assert shape_key({"rows": 1, "n": 4096}, torch.bfloat16) == "rows1-n4096-bfloat16"


def _timed(family, us, **knobs):
    return {"family": family, "knobs": knobs, "us": us, "oracle": {"ok": True}}


def test_best_promotion_picks_an_implementation_per_shape(tmp_path):
    store = ResultsStore(tmp_path / "store", stack="torch-test-hip-test-triton-test")
    results = {
        # the tuned kernel wins by more than the margin: its knobs, as the class's entry
        1: [_timed("vendor", 10.0), _timed("aiter", 9.0), _timed("tuned_a", 8.0, block=8)],
        2: [_timed("vendor", 10.0), _timed("aiter", 9.5), _timed("tuned_a", 8.5, block=8)],
        # within the margin of the vendor kernel: the vendor kernel, as a shape entry
        4: [_timed("vendor", 10.0), _timed("tuned_a", 9.9, block=8), _timed("tuned_b", 9.95, block=16)],
        # a failing tuned candidate does not count
        8: [_timed("vendor", 10.0), {**_timed("tuned_a", 1.0, block=8), "oracle": {"ok": False}}],
    }
    for rows, timed in results.items():
        record = {"shape": {"rows": rows, "n": 64, "k": 32}, "candidates": {f"c{i}": c for i, c in enumerate(timed)}}
        store.write(MI300X, "gemv", record["shape"], torch.bfloat16, record)
    impls = {"vendor": ("vendor", {}), "aiter": ("aiter", {}), "tuned_a": ("triton", {"dot": False}),
             "tuned_b": ("hip", {"dot": True})}
    path = store.promote_best(MI300X, "gemv", impls.get, lambda s: "small" if s["rows"] <= 4 else "large",
                              tuned=("triton", "hip"), margin=0.02, root=tmp_path / "configs",
                              provenance={"job": "test"})
    assert path == tmp_path / "configs" / "gfx942" / "mi300x" / "gemv.json"
    config = json.loads(path.read_text())
    assert config["classes"] == {"small": {"impl": "triton", "block": 8, "dot": False}, "large": {"impl": "vendor"}}
    assert config["shapes"] == {"k32-n64-rows4": {"impl": "vendor"}}
    configs._read.cache_clear()
    loaded = configs.load_config("gemv", "gfx942", "mi300x", root=tmp_path / "configs")
    assert configs.select(loaded, "small", {"rows": 4, "n": 64, "k": 32}) == {"impl": "vendor"}
    assert configs.select(loaded, "small", {"rows": 2, "n": 64, "k": 32})["impl"] == "triton"
    # with a default per class, every measured shape that chose otherwise is its own entry
    path = store.promote_best(MI300X, "gemv", impls.get, lambda s: "small" if s["rows"] <= 4 else "large",
                              tuned=("triton", "hip"), class_default=lambda klass: {"impl": "vendor"},
                              root=tmp_path / "configs")
    config = json.loads(path.read_text())
    assert config["classes"] == {"small": {"impl": "vendor"}, "large": {"impl": "vendor"}}
    assert set(config["shapes"]) == {"k32-n64-rows1", "k32-n64-rows2"}
    # aiter is 10% faster than the default at rows 1 but only 5% at rows 2: rows 2 keeps the default
    path = store.promote_best(MI300X, "gemv", {"vendor": ("vendor", {}), "aiter": ("aiter", {})}.get,
                              lambda s: "all", class_default=lambda klass: {"impl": "vendor"}, margin=0.08,
                              root=tmp_path / "configs")
    assert json.loads(path.read_text())["shapes"] == {"k32-n64-rows1": {"impl": "aiter"}}
    with pytest.raises(ValueError, match="no passing results"):
        store.promote_best(MI300X, "gemv", lambda family: None, lambda s: None, root=tmp_path / "configs")
