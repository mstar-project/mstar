"""Speculative decoding: a GDN verify step driven through the state contract's handle, as a model on the
contract runs it (`calls.causal_conv`, `calls.gated_delta_net`), on model layers 1 and 3, which the pool
holds at indices 0 and 1. The rerun corrects the views those layers ran on, with no layer cursor set."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import test_gdn_verify as T
import torch

from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Causal,
    CausalConv,
    Entry,
    GatedDeltaNet,
    KVEntry,
    Read,
    TensorType,
    Value,
)
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle
from mstar.engine.state.calls import causal_conv, gated_delta_net

MODEL_LAYERS = (1, 3)


@pytest.fixture(autouse=True)
def _kernels(monkeypatch):
    import gdn_cpu

    gdn_cpu.install(monkeypatch)


def _handle(pool, manager):
    g = T.GEOMETRY
    entry = Entry({
        "state": TensorType((g.num_v_heads, g.head_k_dim, g.head_v_dim), dtype=torch.float32),
        "conv": TensorType((g.conv_dim, g.conv_kernel_size - 1), dtype=None),
    })
    array = Value(T.POOL, nodes={"LLM"}, layers=frozenset(MODEL_LAYERS), entry=entry,
                  updates={"conv": CausalConv(), T.LIN: GatedDeltaNet(qk_l2norm=True)})
    holder = Array("kv", nodes={"LLM"}, layers=frozenset(MODEL_LAYERS), entry=KVEntry(heads=1, head_dim=8),
                   max_len=1, reads={"attn": Read(mask=Causal(), op=Attend(q_heads=1))})
    resolved = _resolve([holder, array], StateSetup())
    return build_handle(array, {T.POOL: pool, T.LIN: manager}, resolved)


def _hstep(h, pool, manager, rows, walk, inputs):
    rids = list(rows)
    ctx = StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False)
    segments = tuple(Segment(rid, "main", span) for rid, span in rows.items())
    rstep = RecurrentStep(segments=segments)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results[T.POOL] = pool.plan(rstep, ctx)
    manager.plan(LinearAttnStep(segments=segments), ctx)
    step = SubmoduleStep(segments=list(segments), state={T.POOL: ArrayStep()})
    step.set_ctx(ctx)
    h.begin_step(step)
    for i, (x, a, b) in enumerate(inputs):
        w = T.WEIGHTS[i]
        qkv = causal_conv(h, MODEL_LAYERS[i], x, weight=w.conv_w, bias=w.conv_b)
        q, k, v = T._split(qkv)
        gated_delta_net(h, MODEL_LAYERS[i], q, k, v, a, b, w.a_log, w.dt_bias)
    h.end_step(step)
    pool.commit(rstep, ctx)


@pytest.mark.parametrize("accepted", range(T.K + 1))
def test_a_verify_step_through_the_handle_settles_each_layer_at_the_kept_tokens(accepted):
    """The conv windows and states of both layers end as the prompt and the kept tokens leave them."""
    pool, manager = T._build()
    h = _handle(pool, manager)
    T._add(pool, 1)
    prompt = T._inputs(5, seed=1)
    _hstep(h, pool, manager, {1: 5}, T.PREFILL, prompt)
    block = T._inputs(T.K + 1, seed=2)
    _hstep(h, pool, manager, {1: T.K + 1}, T.VERIFY, block)
    manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor([accepted])))
    T._assert_rows(pool, {1: T._cat(prompt, T._head(block, accepted + 1))})
