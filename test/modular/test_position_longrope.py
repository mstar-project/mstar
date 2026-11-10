"""``PositionConfig.longrope``: a short and a long rotary table, chosen per request.

The table is the reference's (Phi-4-mini's ``Phi4MMLongRoPEScaledRotaryEmbedding``):
every wavelength stretched by a per-dimension factor and cos/sin scaled by the
attention scale. A request reaching past ``original_max_position_embeddings`` rotates
with the long half for its whole life; its rows' table ids sit ``max_seq_len`` past
its positions, while the positions the resource publishes stay positions.
"""

import math

import pytest
import torch

from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.position.config import (
    MAX_POSITION_KWARG,
    LongRoPE,
    PositionConfig,
    PositionReqConfig,
    PositionSpec,
    PositionStep,
)
from mstar.engine.resources.position.manager import LongRopeManager, PositionManager, RopeManager
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

MAX_SEQ_LEN = 64
ROTARY_DIM = 16
LONGROPE = LongRoPE(
    short_factor=tuple(1.0 + 0.1 * i for i in range(ROTARY_DIM // 2)),
    long_factor=tuple(2.0 + 3.0 * i for i in range(ROTARY_DIM // 2)),
    original_max_position_embeddings=16,
    max_position_embeddings=512,
)


ROTARY_DIM_CUDA = 32  # FlashInfer's table kernel needs a multiple of 32


def manager(device="cpu", **overrides) -> LongRopeManager:
    config = dict(kv_cache="kv", rotary_dim=ROTARY_DIM, longrope=LONGROPE)
    config.update(overrides)
    return LongRopeManager(PositionConfig(**config), torch.device(device), max_seq_len=MAX_SEQ_LEN, head_dim=32)


def reference_cos_sin(factors, positions, dtype):
    """The reference's LongRoPE forward, line for line, for one factor set."""
    rotary_dim = 2 * len(factors)
    ext = torch.tensor(factors, dtype=torch.float32)
    inv_freq_shape = torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim
    inv_freq = 1.0 / (ext * 10000.0**inv_freq_shape)
    freqs = (inv_freq[None, :, None].float() @ positions[None, None, :].float()).transpose(1, 2)[0]
    scale = LONGROPE.max_position_embeddings / LONGROPE.original_max_position_embeddings
    scaling_factor = math.sqrt(1 + math.log(scale) / math.log(LONGROPE.original_max_position_embeddings))
    return (freqs.cos() * scaling_factor).to(dtype), (freqs.sin() * scaling_factor).to(dtype)


def test_from_hf_and_the_attention_scale():
    hf = {
        "rope_scaling": {"type": "longrope", "short_factor": [1, 1], "long_factor": [2, 3]},
        "original_max_position_embeddings": 4096, "max_position_embeddings": 131072,
    }
    longrope = LongRoPE.from_hf(hf)
    assert longrope.short_factor == (1.0, 1.0) and longrope.long_factor == (2.0, 3.0)
    # Phi-4-mini's: sqrt(1 + ln 32 / ln 4096)
    assert longrope.attention_scale == pytest.approx(1.190238, abs=1e-6)
    assert not longrope.is_long(4096) and longrope.is_long(4097)
    with pytest.raises(ValueError, match="not a LongRoPE config"):
        LongRoPE.from_hf({**hf, "rope_scaling": {"type": "yarn"}})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_table_is_the_reference_short_half_then_long_half(dtype):
    rope = manager(rope_table_dtype=None if dtype == torch.float32 else dtype)
    table = rope._rope_table
    assert table.dtype == torch.float32 and table.shape == (2 * MAX_SEQ_LEN, ROTARY_DIM)
    positions = torch.arange(MAX_SEQ_LEN)
    for half, factors in enumerate((LONGROPE.short_factor, LONGROPE.long_factor)):
        cos, sin = reference_cos_sin(factors, positions, dtype)
        rows = table[half * MAX_SEQ_LEN:(half + 1) * MAX_SEQ_LEN]
        # the reference repeats its angles across both halves of the rotary dims;
        # the kernel's table holds each once
        assert torch.equal(rows[:, : ROTARY_DIM // 2], cos[:, : ROTARY_DIM // 2].float())
        assert torch.equal(rows[:, ROTARY_DIM // 2:], sin[:, : ROTARY_DIM // 2].float())


def plan_step(rope: RopeManager, rows, lease=None, is_preplan=False):
    rids = tuple(dict.fromkeys(rid for rid, _, _ in rows))
    ctx = StepContext(request_ids=rids, graph_walk="decode", slot=0, capture=False, is_preplan=is_preplan)
    ctx.slot_lease = lease
    ctx.plan_results = {"kv": KVPlanOutputs({"main": KVPlanOutput(None, rows=rows)})}
    return rope.plan(PositionStep(segments=()), ctx)


def test_a_long_request_reads_the_long_half_and_keeps_its_positions(monkeypatch):
    rope = manager()
    rope.ingest_request(1, PositionReqConfig(max_position=10))
    rope.ingest_request(2, PositionReqConfig(max_position=17))
    rope.ingest_request(3, None)  # no ceiling given: the short table
    rope._counters[1]["main"] = 5
    rope._counters[2]["main"] = 7

    positions = plan_step(rope, [(1, "main", 2), (2, "main", 3), (3, "main", 1)])

    assert positions["main"].tolist() == [5, 6, 7, 8, 9, 0]
    assert rope.pos_ids("main").tolist() == [5, 6, 7, 8, 9, 0]
    assert rope._current_table_ids["main"].tolist() == [5, 6, 71, 72, 73, 0]

    calls = []
    monkeypatch.setattr(
        torch.ops.mstar, "rope_apply_qk_inplace",
        lambda q, k, pos_ids, cache, *args, **kwargs: calls.append((pos_ids.tolist(), cache)),
    )
    q = torch.ones((6, 1, 32), dtype=torch.bfloat16)
    rope.apply_qk(q, q.clone(), "main")
    assert calls == [([5, 6, 71, 72, 73, 0], rope._rope_table)]


def test_a_prefix_hit_resumes_on_the_requests_table():
    """A request whose prompt the prefix cache served plans its tail from the cached
    position, on its own table."""
    from mstar.engine.resources.base import CachedPrefix

    rope = manager()
    rope.ingest_request(1, PositionReqConfig(max_position=40))
    rope.apply_cached_prefix(1, "LLM", "prefill_text", None, CachedPrefix(label="main", position=32, tokens=32))
    plan_step(rope, [(1, "main", 3)])
    assert rope.pos_ids("main").tolist() == [32, 33, 34]
    assert rope._current_table_ids["main"].tolist() == [MAX_SEQ_LEN + 32, MAX_SEQ_LEN + 33, MAX_SEQ_LEN + 34]


def test_a_recycled_handle_starts_on_the_short_table():
    """Handles are recycled (AGENTS.md invariant 14): the long choice goes with the request."""
    rope = manager()
    rope.ingest_request(1, PositionReqConfig(max_position=100))
    rope.remove_request(1)
    rope.ingest_request(1, None)
    plan_step(rope, [(1, "main", 1)])
    assert rope._current_table_ids["main"].tolist() == [0]

    rope.ingest_request(1, PositionReqConfig(max_position=100))
    rope.ingest_request(1, PositionReqConfig(max_position=3))  # re-opened under a new config
    plan_step(rope, [(1, "main", 1)])
    assert rope._current_table_ids["main"].tolist() == [0]


def test_without_longrope_nothing_changes():
    rope = RopeManager(PositionConfig(kv_cache="kv"), torch.device("cpu"), max_seq_len=MAX_SEQ_LEN, head_dim=32)
    rope.ingest_request(1, None)
    plan_step(rope, [(1, "main", 2)])
    assert rope._rope_table is None and rope.resident_refusal() is None


def test_the_spec_builds_the_manager_its_config_and_its_prefix_variant():
    """The ceiling kwarg decides both the table and the prefix keys, so they agree."""
    from types import SimpleNamespace

    spec = PositionSpec(
        resource_key="pos", nodes={"LLM"},
        config=PositionConfig(kv_cache="kv", rotary_dim=ROTARY_DIM, longrope=LONGROPE),
    )
    assert spec.request_config({}) is None and spec.prefix_variant({}) is None
    assert spec.request_config({MAX_POSITION_KWARG: 16}) == PositionReqConfig(max_position=16)
    assert spec.prefix_variant({MAX_POSITION_KWARG: 16}) is None
    assert spec.prefix_variant({MAX_POSITION_KWARG: 17}) == b"longrope-long"
    plain = PositionSpec(resource_key="pos", nodes={"LLM"}, config=PositionConfig(kv_cache="kv"))
    assert plain.request_config({MAX_POSITION_KWARG: 17}) is None
    assert plain.prefix_variant({MAX_POSITION_KWARG: 17}) is None

    kv = SimpleNamespace(config=SimpleNamespace(max_seq_len=MAX_SEQ_LEN, head_dim=32))
    info = SimpleNamespace(device=torch.device("cpu"), kv_dtype=torch.bfloat16, dependency=lambda key: kv)
    assert type(PositionManager.build(spec, info)) is LongRopeManager
    assert type(PositionManager.build(plain, info)) is RopeManager


def test_a_per_request_ceiling_needs_longrope():
    plain = PositionSpec(resource_key="pos", nodes={"LLM"}, config=PositionConfig(kv_cache="kv"))
    with pytest.raises(ValueError, match="needs PositionConfig.longrope"):
        PositionReqConfig(max_position=10).validate(plain)
    with pytest.raises(ValueError, match="positive int"):
        PositionReqConfig(max_position=0).validate()
    longrope = PositionSpec(resource_key="pos", nodes={"LLM"}, config=PositionConfig(kv_cache="kv", longrope=LONGROPE))
    PositionReqConfig(max_position=10).validate(longrope)


def test_explicit_positions_are_refused():
    """Explicit ids carry no request, so they can't say which table a row reads."""
    rope = manager()
    rope.ingest_request(1, None)
    ctx = StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=False)
    ctx.plan_results = {"kv": KVPlanOutputs({"main": KVPlanOutput(None, rows=[(1, "main", 1)])})}
    with pytest.raises(ValueError, match="explicit pos_ids"):
        rope.plan(PositionStep(segments=(), pos_ids=torch.tensor([3])), ctx)
    q = torch.ones((1, 1, 32), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="explicit pos_ids"):
        rope.apply_qk(q, q.clone(), pos_ids=torch.tensor([3]))


def test_build_refusals():
    with pytest.raises(ValueError, match="needs 4 factors"):
        manager(rotary_dim=8)
    with pytest.raises(ValueError, match="multiple of 32"):
        manager(device="cuda")  # 16 wide
    with pytest.raises(NotImplementedError, match="rope_scale"):
        manager(rope_scale=2.0)
    with pytest.raises(NotImplementedError, match="XPU"):
        manager(device="xpu")
    with pytest.raises(ValueError, match="differ in length"):
        LongRoPE((1.0,), (1.0, 2.0), 16, 64)
    rope = manager()
    # its rows keep their table ids on resident tables (test_resident_planning.py)
    assert rope.resident_refusal() is None
    # its rows are table ids, not positions: a fused kernel must not read it by position
    assert rope.cos_sin_table is None and rope.fusable_rope_theta(32) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging and the lease path need CUDA")
def test_a_preplanned_step_promotes_its_table_ids():
    half = ROTARY_DIM_CUDA // 2
    rope = manager(
        device="cuda", rotary_dim=ROTARY_DIM_CUDA,
        longrope=LongRoPE((1.0,) * half, (2.0,) * half, 16, 512),
    )
    rope.ingest_request(1, PositionReqConfig(max_position=100))
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1))
    plan_step(rope, [(1, "main", 1)], lease=lease, is_preplan=True)
    assert rope._current_table_ids == {}
    plan_step(rope, [(1, "main", 1)], lease=lease)
    assert rope._current_table_ids["main"].tolist() == [MAX_SEQ_LEN]
    assert rope.pos_ids("main").tolist() == [0]
    rope.clear_preplan()
    assert rope._preplan_table_ids == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned staging and the lease path need CUDA")
def test_a_partly_filled_bucket_stages_each_row_at_the_buckets_width():
    """Five live tokens in an eight-token bucket: the rows come back the bucket's width,
    the live ids at the front, and the tail stays in range of the table."""
    half = ROTARY_DIM_CUDA // 2
    rope = manager(device="cuda", rotary_dim=ROTARY_DIM_CUDA, longrope=LongRoPE((1.0,) * half, (2.0,) * half, 16, 512))
    rope.ingest_request(1, PositionReqConfig(max_position=100))
    rope.ingest_request(2, None)
    rope._counters[1]["main"] = 3
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="prefill_text", bs=2, num_tokens=8))
    plan_step(rope, [(1, "main", 3), (2, "main", 2)], lease=lease)
    torch.cuda.synchronize()
    positions, table = rope.pos_ids("main"), rope._current_table_ids["main"]
    assert positions.shape == table.shape == (8,)
    assert positions[:5].tolist() == [3, 4, 5, 0, 1]
    assert table[:5].tolist() == [MAX_SEQ_LEN + 3, MAX_SEQ_LEN + 4, MAX_SEQ_LEN + 5, 0, 1]
    assert int(table.max()) < 2 * MAX_SEQ_LEN


def reference_rotate(x, cos, sin):
    """``apply_rotary_pos_emb`` over the first ``cos.shape[-1]`` dims, as the reference does."""
    rotary_dim = cos.shape[-1]
    rot, keep = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
    return torch.cat((rot * cos + rotated * sin, keep), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashInfer's RoPE kernel")
def test_flashinfer_rotates_partial_dims_as_the_reference():
    """The kernel over the combined table, against the reference's bf16 math, at
    Phi-4-mini's geometry (96 of 128 dims), on a short and a long request in one step."""
    pytest.importorskip("flashinfer")
    half = 48
    longrope = LongRoPE(
        tuple(1.0 + 0.01 * i for i in range(half)), tuple(1.0 + 1.0 * i for i in range(half)), 16, 512,
    )
    rope = LongRopeManager(
        PositionConfig(kv_cache="kv", rotary_dim=2 * half, longrope=longrope, rope_table_dtype=torch.bfloat16),
        torch.device("cuda"), max_seq_len=MAX_SEQ_LEN, head_dim=128,
    )
    rope.ingest_request(1, PositionReqConfig(max_position=8))
    rope.ingest_request(2, PositionReqConfig(max_position=60))
    rope._counters[2]["main"] = 40
    plan_step(rope, [(1, "main", 3), (2, "main", 3)])

    torch.manual_seed(0)
    q = torch.randn(6, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(6, 2, 128, device="cuda", dtype=torch.bfloat16)
    got_q, got_k = rope.apply_qk(q.clone(), k.clone(), "main")

    def want(x, factors, positions):
        cos, sin = (t.repeat(1, 2)[:, None, :].cuda() for t in reference_cos_sin(factors, positions, torch.bfloat16))
        return reference_rotate(x, cos, sin)

    for rows, factors, positions in (
        (slice(0, 3), longrope.short_factor, torch.arange(0, 3)),
        (slice(3, 6), longrope.long_factor, torch.arange(40, 43)),
    ):
        for got, x in ((got_q, q), (got_k, k)):
            assert torch.equal(got[rows, :, 2 * half:], x[rows, :, 2 * half:])
            torch.testing.assert_close(got[rows], want(x[rows], factors, positions), atol=2e-2, rtol=2e-2)
    # the negative control: the long row against the short table's angles is far off
    assert (got_q[3:6] - want(q[3:6], longrope.short_factor, torch.arange(40, 43))).abs().max() > 0.5
