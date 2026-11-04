"""The simulated GPU's cost model: roofline regimes, free ops, and the profile DB."""

import json
import os

import pytest
import torch

from mstar_sim import cost, gpus, tables

aten = torch.ops.aten


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _seconds(func, *args, **kwargs):
    return cost.op_seconds(func, args, kwargs, func(*args, **kwargs))


@pytest.fixture(autouse=True)
def _h100():
    cost.configure("h100", model="roofline")
    yield
    cost.stop_trace()
    cost.configure("h100", model="roofline")


def test_large_gemm_is_compute_bound_near_peak():
    h100 = gpus.lookup("h100")
    n = 8192
    ideal = 2 * n**3 / h100.bf16
    t = _seconds(aten.mm.default, _meta(n, n), _meta(n, n))
    assert ideal < t < 2 * ideal
    assert t > 2 * 3 * n * n * 2 / h100.hbm_bw  # far above its memory time


def test_large_elementwise_is_bandwidth_bound():
    h100 = gpus.lookup("h100")
    n = 1 << 28
    ideal = 3 * n * 2 / h100.hbm_bw  # read two bf16 tensors, write one
    assert ideal < _seconds(aten.add.Tensor, _meta(n), _meta(n)) < 1.5 * ideal


def test_h200_beats_h100_on_decode_gemv():
    args = (_meta(1, 8192), _meta(8192, 8192))
    h100 = _seconds(aten.mm.default, *args)
    cost.configure("H200")
    assert _seconds(aten.mm.default, *args) < 0.8 * h100


@pytest.mark.parametrize(
    "func, args",
    [
        (aten.view.default, (_meta(4, 8), [32])),
        (aten.t.default, (_meta(4, 8),)),
        (aten.expand.default, (_meta(1, 8), [4, 8])),
        (aten.slice.Tensor, (_meta(4, 8), 0, 1, 3)),
        (aten._unsafe_view.default, (_meta(4, 8), [32])),
        (aten.empty_strided.default, ([4, 8], [8, 1])),
    ],
)
def test_ops_without_a_kernel_cost_nothing(func, args):
    assert _seconds(func, *args) == 0.0


def test_kv_cache_write_costs_the_slot_not_the_cache():
    cache, rows = _meta(1 << 20, 8, 128), _meta(4, 8, 128)
    index = torch.zeros(4, dtype=torch.long, device="meta")
    small = _seconds(aten.add.Tensor, rows, rows)
    assert _seconds(aten.index_put_.default, cache, [index], rows) < 2 * small


def test_profile_db_overrides_and_scales_the_roofline(tmp_path):
    trace = tmp_path / "trace.jsonl"
    cost.start_trace(str(trace))
    roofline = _seconds(aten.mm.default, _meta(128, 4096), _meta(4096, 4096))
    cost.stop_trace()
    (entry,) = [json.loads(line) for line in trace.read_text().splitlines()]
    db = tmp_path / "h100.json"
    measured = 2 * entry["seconds"]
    db.write_text(json.dumps({
        "format": cost.FORMAT, "gpu": gpus.lookup("h100").name,
        "entries": {entry["key"]: {"seconds": measured, "roofline": entry["seconds"], "spec": entry["spec"]}},
    }))
    launch = roofline - entry["seconds"]
    other_shape = _seconds(aten.mm.default, _meta(512, 4096), _meta(4096, 4096))
    cost.configure("h100", profile_db=str(db))
    assert _seconds(aten.mm.default, _meta(128, 4096), _meta(4096, 4096)) == pytest.approx(measured + launch)
    # an unmeasured token count of the same family takes the measured/roofline ratio
    scaled = _seconds(aten.mm.default, _meta(512, 4096), _meta(4096, 4096))
    assert scaled == pytest.approx(2 * (other_shape - launch) + launch)


# --- measured AIConfigurator tables ------------------------------------------

needs_tables = pytest.mark.skipif(not os.path.isdir(tables.data_dir()), reason="no AIConfigurator data")


def _aic(family: str):
    """The H100 table the tables module reads for ``family``."""
    pd = pytest.importorskip("pandas")
    return pd.read_parquet(tables._sources(tables.data_dir(), "h100_sxm")[family])


def _gemm_row(m, n, k):
    d = _aic("gemm")
    row = d[(d.kernel_source == "sglang_torch_linear") & (d.m == m) & (d.n == n) & (d.k == k)]
    return float(row.latency.iloc[0]) * 1e-3


def _decode_attention(batch, kv_len, heads=8, kv_heads=2, head_dim=256):
    features = {"rows": [(1, kv_len, kv_len)] * batch, "heads": heads, "kv_heads": kv_heads,
                "head_dim": head_dim, "kv_dtype": torch.bfloat16}
    flops = 4.0 * batch * kv_len * heads * head_dim
    nbytes = batch * (4 * heads * head_dim + 4 * kv_len * kv_heads * head_dim)
    return cost.kernel_seconds("flashinfer.batch_paged_attention", flops=flops, bytes=nbytes,
                               features=features) - cost._S.gpu.launch_s


@needs_tables
def test_tables_return_a_measured_gemm_exactly_and_blend_between_grid_points():
    cost.configure("h100", model="tables")
    launch = gpus.lookup("h100").launch_s
    # m = 1025 is a measured tile cliff (~13% over m = 1024): kept, not smoothed
    exact = _seconds(aten.mm.default, _meta(1025, 1024), _meta(1024, 1024)) - launch
    assert exact == pytest.approx(_gemm_row(1025, 1024, 1024), rel=1e-9)
    lo, hi = sorted((_gemm_row(257, 1024, 1024), _gemm_row(512, 1024, 1024)))
    assert lo <= _seconds(aten.mm.default, _meta(384, 1024), _meta(1024, 1024)) - launch <= hi


@needs_tables
def test_decode_attention_never_beats_the_measured_floor():
    d = _aic("generation")
    s = d[(d.num_key_value_heads == 2) & (d.head_dim == 256) & (d.kv_cache_dtype == "bfloat16")
          & (d.window_size == 0) & (d.kernel_source == "fa3")]
    cost.configure("h100", model="tables")
    assert _decode_attention(1, 1) >= s.latency.min() * 1e-3 > 5e-6  # the roofline says ~2 us


@needs_tables
def test_shapes_the_tables_lack_cost_the_roofline():
    fp32 = torch.float32
    cases = [lambda: _decode_attention(4, 512, head_dim=96),
             lambda: _seconds(aten.mm.default, _meta(64, 1024, dtype=fp32), _meta(1024, 1024, dtype=fp32))]
    roofline = [case() for case in cases]
    cost.configure("h100", model="tables")
    assert [case() for case in cases] == roofline


@needs_tables
def test_b200_tables_beat_h100():
    cost.configure("h100", model="tables")
    h100 = (_decode_attention(64, 4096), _seconds(aten.mm.default, _meta(4096, 4096), _meta(4096, 4096)))
    cost.configure("b200", model="tables")
    b200 = (_decode_attention(64, 4096), _seconds(aten.mm.default, _meta(4096, 4096), _meta(4096, 4096)))
    assert all(b < 0.8 * h for b, h in zip(b200, h100, strict=True))
