"""The Mamba-2 resource on the recurrent state pool: the plan is a narrow of
the pool's addressing, the step kernel advances each row's slot in place and
agrees with the sequential scan, prefill scatters its final state so a decode
step continues from it, and padding rows leave live slots alone.

The torch reference path runs on CPU; the Triton kernel is checked against it
on a GPU when one is present (Nemotron-H's geometry among the cases).
"""
import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnSpec, LinearAttnVariant
from mstar.engine.resources.linear_attn.mamba2 import Mamba2Manager
from mstar.engine.resources.linear_attn.mamba2_kernels import (
    PAD_SLOT_ID,
    mamba2_scan,
    mamba2_state_update,
    mamba2_state_update_torch,
)
from mstar.engine.resources.recurrent.config import (
    Mamba2Geometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext

POOL, MAMBA = "mamba_state", "mamba"
GEOM = Mamba2Geometry(num_heads=4, head_dim=8, state_size=16, n_groups=2, conv_kernel_size=4)
NUM_LAYERS = 2


def pool_spec(max_slots=8) -> RecurrentStateSpec:
    return RecurrentStateSpec(
        POOL, {"llm"},
        RecurrentStateConfig(num_layers=NUM_LAYERS, blocks=GEOM.to_blocks(), max_slots=max_slots),
    )


def build(device="cpu"):
    info = EngineResourceInfo(device=torch.device(device))
    pool = RecurrentStatePool.build(pool_spec(), info)
    manager = LinearAttnManager.build(
        LinearAttnSpec(MAMBA, {"llm"}, LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.MAMBA2)),
        EngineResourceInfo(device=torch.device(device), dependencies={POOL: pool_spec()}),
    )
    return pool, manager


def ctx(rids, walk="decode") -> StepContext:
    return StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False)


def steps(rids, spans):
    segs = tuple(Segment(rid, "main", span) for rid, span in zip(rids, spans, strict=True))
    return RecurrentStep(segments=segs), segs


def plan(pool, manager, rids, spans, walk="decode"):
    """admit + plan both resources for one step, the way the runner does."""
    from mstar.engine.resources.linear_attn.config import LinearAttnStep

    rstep, segs = steps(rids, spans)
    c = ctx(rids, walk)
    for rid in rids:
        pool.ingest_request(rid)
    assert pool.admit(rstep, c).ok
    c.plan_results[POOL] = pool.plan(rstep, c)
    manager.plan(LinearAttnStep(segments=segs), c)
    return rstep, c


def params(device="cpu", gen=None):
    g = gen or torch.Generator().manual_seed(0)
    h = GEOM.num_heads
    return dict(
        a_log=torch.randn(h, generator=g).to(device),
        d=torch.randn(h, generator=g).to(device),
        dt_bias=torch.randn(h, generator=g).to(device),
    )


def inputs(rows, seq=1, device="cpu", gen=None, dtype=torch.float32):
    g = gen or torch.Generator().manual_seed(1)
    n = rows * seq
    return (
        torch.randn(n, GEOM.num_heads, GEOM.head_dim, generator=g).to(device, dtype),
        torch.randn(n, GEOM.num_heads, generator=g).to(device, dtype),
        torch.randn(n, GEOM.n_groups, GEOM.state_size, generator=g).to(device, dtype),
        torch.randn(n, GEOM.n_groups, GEOM.state_size, generator=g).to(device, dtype),
    )


def test_build_dispatches_on_the_variant_and_reads_the_geometry():
    pool, manager = build()
    assert isinstance(manager, Mamba2Manager)
    assert manager.geometry == GEOM and manager.num_layers == NUM_LAYERS
    assert manager.state_dtype is torch.float32 and manager.depends_on() == {POOL}
    assert pool.block("ssm", 1).shape == (8, 4, 8, 16) and pool.block("conv", 0).shape == (8, GEOM.conv_dim, 3)


def test_plan_is_a_narrow_of_the_pools_addressing():
    pool, manager = build()
    _, c = plan(pool, manager, ["a", "b", "c"], [1, 1, 1])
    p = manager.current_plan("main")
    assert p.is_decode and p.num_rows == 3 and p.spans == (1, 1, 1)
    assert p.slots.tolist() == c.plan_results[POOL]["main"].slot_indices[:3].tolist()
    assert p.has_state.tolist() == [False, False, False]  # fresh slots read as zeros
    assert len(set(p.slots.tolist())) == 3 and SINK_SLOT not in p.slots.tolist()
    with pytest.raises(KeyError, match="no plan for label"):
        manager.current_plan("other")


def test_step_matches_scan_and_persists_in_the_pool():
    pool, manager = build()
    prm = params()
    rids = ["a", "b"]
    x1, dt1, b1, c1 = inputs(2, gen=torch.Generator().manual_seed(2))
    x2, dt2, b2, c2 = inputs(2, gen=torch.Generator().manual_seed(3))
    rstep, c = plan(pool, manager, rids, [1, 1])
    layer = pool.block("ssm", 0)
    y1 = manager.run(x1, dt1, b1, c1, layer, prm["a_log"], prm["d"], prm["dt_bias"])
    pool.commit(rstep, c)
    rstep, c = plan(pool, manager, rids, [1, 1])
    assert manager.current_plan().has_state.tolist() == [True, True]
    y2 = manager.run(x2, dt2, b2, c2, layer, prm["a_log"], prm["d"], prm["dt_bias"])
    slots = manager.current_plan().slots.tolist()

    a = -torch.exp(prm["a_log"])
    for row in range(2):
        y_ref, s_ref = mamba2_scan(
            torch.stack([x1[row], x2[row]]), torch.stack([dt1[row], dt2[row]]), a,
            torch.stack([b1[row], b2[row]]), torch.stack([c1[row], c2[row]]), None,
            d=prm["d"], dt_bias=prm["dt_bias"],
        )
        torch.testing.assert_close(torch.stack([y1[row], y2[row]]), y_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(layer[slots[row]], s_ref, atol=1e-5, rtol=1e-5)
    # the other layer's block was never touched
    assert pool.block("ssm", 1).abs().sum() == 0


def test_prefill_then_decode_continues_from_the_scattered_state():
    pool, manager = build()
    prm = params()
    seq = 5
    x, dt, b, c = inputs(1, seq=seq, gen=torch.Generator().manual_seed(4))
    xd, dtd, bd, cd = inputs(1, gen=torch.Generator().manual_seed(5))
    layer = pool.block("ssm", 0)
    rstep, ctx_ = plan(pool, manager, ["a"], [seq], walk="prefill_text")
    assert not manager.current_plan().is_decode
    y_pre = manager.run(x, dt, b, c, layer, prm["a_log"], prm["d"], prm["dt_bias"])
    pool.commit(rstep, ctx_)
    rstep, ctx_ = plan(pool, manager, ["a"], [1])
    y_dec = manager.run(xd, dtd, bd, cd, layer, prm["a_log"], prm["d"], prm["dt_bias"])

    a = -torch.exp(prm["a_log"])
    y_ref, s_ref = mamba2_scan(
        torch.cat([x, xd]), torch.cat([dt, dtd]), a, torch.cat([b, bd]), torch.cat([c, cd]), None,
        d=prm["d"], dt_bias=prm["dt_bias"],
    )
    torch.testing.assert_close(torch.cat([y_pre, y_dec]), y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(layer[manager.current_plan().slots[0]], s_ref, atol=1e-5, rtol=1e-5)


def test_padding_rows_leave_live_slots_alone():
    """A row addressing the sink absorbs its write; a NO_SLOT row is skipped
    and produces zeros; live rows are unaffected either way."""
    prm = params()
    x, dt, b, c = inputs(3, gen=torch.Generator().manual_seed(6))
    a = -torch.exp(prm["a_log"])
    state = torch.randn(4, GEOM.num_heads, GEOM.head_dim, GEOM.state_size)
    ref_state = state.clone()
    live_only = mamba2_state_update(
        x[:1].clone(), dt[:1], a, b[:1], c[:1], ref_state, torch.tensor([2], dtype=torch.int32),
        d=prm["d"], dt_bias=prm["dt_bias"],
    )
    out = mamba2_state_update(
        x.clone(), dt, a, b, c, state, torch.tensor([2, SINK_SLOT, PAD_SLOT_ID], dtype=torch.int32),
        d=prm["d"], dt_bias=prm["dt_bias"],
    )
    torch.testing.assert_close(out[0], live_only[0])
    torch.testing.assert_close(state[2], ref_state[2])
    torch.testing.assert_close(state[1], ref_state[1])   # untouched slot
    torch.testing.assert_close(state[3], ref_state[3])
    assert out[2].abs().sum() == 0                          # NO_SLOT row: skipped


def test_dt_clamp_applies_after_softplus():
    prm = params()
    x, dt, b, c = inputs(1, gen=torch.Generator().manual_seed(7))
    a = -torch.exp(prm["a_log"])
    s1 = torch.zeros(1, GEOM.num_heads, GEOM.head_dim, GEOM.state_size)
    s2 = s1.clone()
    slots = torch.tensor([0], dtype=torch.int32)
    y_open = mamba2_state_update_torch(x, dt, a, b, c, s1, slots, d=prm["d"], dt_bias=prm["dt_bias"])
    y_clamped = mamba2_state_update_torch(
        x, dt, a, b, c, s2, slots, d=prm["d"], dt_bias=prm["dt_bias"], time_step_limit=(0.0, 0.01),
    )
    assert not torch.allclose(y_open, y_clamped)
    tiny = torch.nn.functional.softplus(dt.float() + prm["dt_bias"]).clamp(max=0.01)
    exp = torch.zeros_like(s2)
    exp = exp * torch.exp(tiny * a)[..., None, None] + (tiny[..., None] * x.float())[..., None] * \
        b.float().repeat_interleave(GEOM.num_heads // GEOM.n_groups, dim=1)[:, :, None, :]
    torch.testing.assert_close(s2, exp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
@pytest.mark.parametrize("geom,rows,dtype", [
    (GEOM, 5, torch.float32),
    (GEOM, 3, torch.bfloat16),
    # Nemotron-H: 128 heads x 80, state 128, 8 groups (head_dim is not a power of two)
    (Mamba2Geometry(num_heads=128, head_dim=80, state_size=128, n_groups=8, conv_kernel_size=4), 16, torch.bfloat16),
])
def test_triton_step_matches_torch_reference(geom, rows, dtype):
    torch.manual_seed(0)
    dev = "cuda"
    h, p, n, g = geom.num_heads, geom.head_dim, geom.state_size, geom.n_groups
    x = torch.randn(rows, h, p, device=dev, dtype=dtype)
    dt = torch.randn(rows, h, device=dev, dtype=dtype)
    b = torch.randn(rows, g, n, device=dev, dtype=dtype)
    c = torch.randn(rows, g, n, device=dev, dtype=dtype)
    a_log, d, dt_bias = (torch.randn(h, device=dev) for _ in range(3))
    a = -torch.exp(a_log)
    max_slots = rows + 4
    state = torch.randn(max_slots, h, p, n, device=dev)
    ref_state = state.clone()
    # rows in scrambled slots, one on the sink, one skipped
    slots = torch.randperm(max_slots - 1, device=dev)[:rows].to(torch.int32) + 1
    slots[1] = SINK_SLOT
    slots[-1] = PAD_SLOT_ID
    out = mamba2_state_update(x, dt, a, b, c, state, slots, d=d, dt_bias=dt_bias, time_step_limit=(0.0, 5.0))
    ref = mamba2_state_update_torch(x, dt, a, b, c, ref_state, slots, d=d, dt_bias=dt_bias, time_step_limit=(0.0, 5.0))
    tol = dict(atol=2e-2, rtol=2e-2) if dtype is torch.bfloat16 else dict(atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out, ref, **tol)
    live = [i for i in range(rows) if slots[i] not in (SINK_SLOT, PAD_SLOT_ID)]
    torch.testing.assert_close(state[slots[live].long()], ref_state[slots[live].long()], atol=1e-4, rtol=1e-4)
    untouched = [s for s in range(1, max_slots) if s not in slots.tolist()]
    torch.testing.assert_close(state[untouched], ref_state[untouched])
