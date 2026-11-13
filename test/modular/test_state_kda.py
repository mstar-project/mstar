"""Kimi delta attention is one update over a Value's ``conv`` window and ``state``.

KDA's kernels read and write the window and the state together, in whatever launches they split them into, so
GLM-5.3's layers handed one resource both blocks. On the contract that is one update call, `kimi_delta`, which records
both fields and runs the kernels on the blocks `KDACallable` passed. A layer's kernel parameters are built from its
weights once, and again for a weights object rebuilt after a move, as GLM-5.3 reset them in `_apply`.
"""

import re
from dataclasses import replace

import pytest
import torch

from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep, LinearAttnVariant
from mstar.engine.resources.linear_attn.kda import KDAManager, KDAParams
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStep
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Causal,
    Entry,
    KimiDelta,
    KimiDeltaWeights,
    KVEntry,
    Read,
    TensorType,
    Value,
    kimi_delta,
)
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle, build_specs

CPU = torch.device("cpu")
HEADS, DIM, KERNEL = 2, 8, 4
P = HEADS * DIM


class _StubKDA:
    """Records the parameters each run gets instead of running the kernels."""

    def __init__(self):
        self.params = []

    def bind_layer(self, *, lower_bound, full_rank_output_gate):
        return

    def run(self, qkv, g, beta, conv_layer, state_layer, params, gate=None, label=None, spec=None):
        self.params.append(params)
        return qkv


def _kv() -> Array:
    return Array("kv", nodes={"LLM"}, layers={0}, entry=KVEntry(heads=1, head_dim=8), max_len=64,
                 reads={"attn": Read(mask=Causal(), op=Attend(q_heads=1))})


def _entry(**changes: TensorType) -> Entry:
    return Entry({
        "conv": TensorType((3 * P, KERNEL - 1), dtype=torch.float32, shard=0),
        "state": TensorType((HEADS, DIM, DIM), dtype=torch.float32, shard=0),
        **changes,
    })


def _kda_value(op: KimiDelta | None = None, entry: Entry | None = None) -> Array:
    return Value("kda_state", nodes={"LLM"}, layers={3, 7}, entry=entry or _entry(),
                 updates={"kda": op or KimiDelta(lower_bound=-5.0)})


def _handle(array: Array, kda) -> tuple:
    specs = build_specs([_kv(), array], kernel_settings={"kda_state": {"max_slots": 4}})
    (config,) = (spec.config for spec in specs if spec.resource_key == "kda_state")
    pool = RecurrentStatePool(CPU, config, kv_dtype=torch.float32)
    return build_handle(array, {"kda_state": pool, "kda": kda}, _resolve([_kv(), array], StateSetup())), pool


def _weights(full_rank: bool = False) -> KimiDeltaWeights:
    return KimiDeltaWeights(
        conv=torch.randn(3 * P, KERNEL), a_log=torch.randn(HEADS), dt_bias=torch.randn(P), f_b=torch.randn(P, DIM),
        g_b=None if full_rank else torch.randn(P, DIM), norm_weight=None if full_rank else torch.rand(DIM) + 0.5,
    )


def _inputs(tokens: int) -> tuple[torch.Tensor, ...]:
    return torch.randn(tokens, 3 * P), torch.randn(tokens, DIM), torch.randn(tokens, HEADS), torch.randn(tokens, DIM)


def _open(h, segments, ctx=None) -> SubmoduleStep:
    step = SubmoduleStep(segments=list(segments), state={h.name: ArrayStep()})
    step.set_ctx(ctx or StepContext(request_ids=[s.request_id for s in segments], graph_walk="decode", slot=0,
                                    capture=False))
    h.begin_step(step)
    return step


@pytest.mark.parametrize("op, entry, refusal", [
    (KimiDelta(lower_bound=-5), _entry(), "lower_bound must be a negative float, got -5"),
    (KimiDelta(lower_bound=0.0), _entry(), "lower_bound must be a negative float, got 0.0"),
    (KimiDelta(lower_bound=-5.0, scale=0.0), _entry(), "scale must be None or a positive float, got 0.0"),
    (KimiDelta(lower_bound=-5.0), _entry(state=TensorType((HEADS, DIM, DIM + 1), dtype=torch.float32)),
     "KimiDelta needs entry fields conv (channels, kernel - 1) and state (h, d, d)"),
    (KimiDelta(lower_bound=-5.0), _entry(conv=TensorType((2 * P, KERNEL - 1), dtype=torch.float32)),
     "its channels must be 3 * 2 * 8, and they are 32"),
], ids=["int-bound", "zero-bound", "zero-scale", "state", "channels"])
def test_kimi_delta_refuses_its_own_bad_fields(op, entry, refusal):
    with pytest.raises(ValueError, match=re.escape(refusal)):
        _kda_value(op, entry)


@pytest.mark.parametrize("entry, refusal", [
    (_entry(extra=TensorType((4,), dtype=torch.float32)), "entry must be exactly conv and state"),
    (_entry(state=TensorType((HEADS, DIM, DIM), dtype=torch.bfloat16, shard=0)),
     "state of dtype torch.bfloat16, and the KDA kernels keep the recurrence in float32"),
], ids=["extra-field", "bf16-state"])
def test_a_kda_value_holds_exactly_its_window_and_state(entry, refusal):
    with pytest.raises(ValueError, match=re.escape(f"Array 'kda_state': no layout yet: {refusal}")):
        build_specs([_kv(), _kda_value(entry=entry)])


def test_one_update_call_records_both_fields():
    h, _ = _handle(_kda_value(), _StubKDA())
    _open(h, [Segment("a", "main", 2)])
    qkv, g, beta, gate = _inputs(2)
    kimi_delta(h, 3, qkv, g, beta, _weights(), gate=gate)
    assert h._updated == {("conv", 3), ("state", 3)}, "one call updates the window and the state"
    with pytest.raises(ValueError, match="a second update of conv on layer 3 in one step"):
        kimi_delta(h, 3, qkv, g, beta, _weights(), gate=gate)


def test_kimi_delta_builds_its_params_once_per_weights_object():
    kda = _StubKDA()
    h, _ = _handle(_kda_value(), kda)
    qkv, g, beta, gate = _inputs(1)
    weights = _weights()
    for each in (weights, weights, replace(weights)):
        step = _open(h, [Segment("a", "main", 1)])
        kimi_delta(h, 7, qkv, g, beta, each, gate=gate)
        h.end_step(step)
    first, again, rebuilt = kda.params
    assert again is first, "the same weights object reuses the layer's params"
    assert rebuilt is not first, "a weights object rebuilt after a move rebuilds them"
    assert (first.num_heads, first.head_dim, first.scale, first.lower_bound) == (HEADS, DIM, DIM ** -0.5, -5.0)


@pytest.mark.parametrize("full_rank, weights_full_rank, with_gate, wrong", [
    (False, False, False, "['gate'] None"),
    (False, True, True, "['g_b', 'norm_weight'] None"),
    (True, False, False, "['g_b', 'norm_weight'] set"),
    (True, True, True, "['gate'] set"),
], ids=["low-rank-no-gate", "low-rank-no-weights", "full-rank-weights", "full-rank-gate"])
def test_kimi_delta_refuses_weights_its_gate_flag_disagrees_with(full_rank, weights_full_rank, with_gate, wrong):
    h, _ = _handle(_kda_value(KimiDelta(lower_bound=-5.0, full_rank_gate=full_rank)), _StubKDA())
    _open(h, [Segment("a", "main", 1)])
    qkv, g, beta, gate = _inputs(1)
    with pytest.raises(ValueError, match=re.escape(f"full_rank_gate={full_rank} takes")) as refused:
        kimi_delta(h, 3, qkv, g, beta, _weights(weights_full_rank), gate=gate if with_gate else None)
    assert str(refused.value).endswith(f"and got {wrong}"), "the refusal names each disagreeing argument"


def _kda_manager() -> KDAManager:
    """On the CPU, so on the torch kernels."""
    geometry = DeltaNetGeometry(num_k_heads=HEADS, num_v_heads=HEADS, head_k_dim=DIM, head_v_dim=DIM,
                                conv_kernel_size=KERNEL)
    config = LinearAttnConfig(recurrent_state="kda_state", variant=LinearAttnVariant.KDA)
    return KDAManager(config, geometry, num_layers=2, state_dtype=torch.float32, device=CPU, conv_dtype=torch.float32)


def _plan(pool, kda, spans, walk) -> tuple[list[Segment], StepContext, RecurrentStep]:
    segments = [Segment(rid, "main", span) for rid, span in zip("ab", spans, strict=False)]
    ctx = StepContext(request_ids=[segment.request_id for segment in segments], graph_walk=walk, slot=0, capture=False)
    pool_step = RecurrentStep(segments=tuple(segments))
    assert pool.admit(pool_step, ctx).ok
    ctx.plan_results["kda_state"] = pool.plan(pool_step, ctx)
    kda.plan(LinearAttnStep(segments=tuple(segments)), ctx)
    return segments, ctx, pool_step


def test_kimi_delta_runs_the_kda_kernels_on_the_value_s_blocks():
    torch.manual_seed(0)
    array, weights = _kda_value(), _weights()
    kda, oracle = _kda_manager(), _kda_manager()
    h, pool = _handle(array, kda)
    _, oracle_pool = _handle(array, oracle)
    params = KDAParams(
        conv_weight=weights.conv, A_log=weights.a_log, dt_bias=weights.dt_bias, lower_bound=-5.0, num_heads=HEADS,
        head_dim=DIM, scale=DIM ** -0.5, f_b=weights.f_b, g_b=weights.g_b, norm_weight=weights.norm_weight,
    )
    for each in (pool, oracle_pool):
        for rid in "ab":
            each.ingest_request(rid)
    for spans, walk in (((3, 2), "prefill"), ((1, 1), "decode")):
        qkv, g, beta, gate = _inputs(sum(spans))
        segments, ctx, pool_step = _plan(pool, kda, spans, walk)
        step = _open(h, segments, ctx)
        outs = [kimi_delta(h, layer, qkv, g, beta, weights, gate=gate) for layer in (3, 7)]
        h.check_step(step)
        h.end_step(step)
        pool.commit(pool_step, ctx)
        _, oracle_ctx, oracle_step = _plan(oracle_pool, oracle, spans, walk)
        expected = [
            oracle.run(qkv, g, beta, oracle_pool.block("conv", i), oracle_pool.block("state", i), params, gate=gate,
                       label="main")
            for i in (0, 1)
        ]
        oracle_pool.commit(oracle_step, oracle_ctx)
        assert all(torch.equal(out, ref) for out, ref in zip(outs, expected, strict=True)), f"{walk} outputs differ"
        for field in ("conv", "state"):
            assert all(torch.equal(pool.block(field, i), oracle_pool.block(field, i)) for i in (0, 1)), (
                f"{walk} left {field} unlike KDAManager.run on the same blocks"
            )


def test_a_kda_value_stores_its_state_k_last():
    h, pool = _handle(_kda_value(), _StubKDA())
    pool.ingest_request("a")
    segments, ctx, _ = _plan(pool, _kda_manager(), (1,), "decode")
    _open(h, segments[:1], ctx)
    state = torch.arange(HEADS * DIM * DIM, dtype=torch.float32).view(1, HEADS, DIM, DIM)
    h.write(3, state=state)
    slot = int(ctx.plan_results["kda_state"]["main"].slot_indices[0])
    assert torch.equal(pool.block("state", 0)[slot], state[0].transpose(-1, -2)), "the kernels read state K-last"
