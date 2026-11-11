"""Speculative decoding on GDN: a verify step's forward runs all ``k + 1``
tokens of a row for their outputs and leaves its recurrent state and conv
window as they were; `GDNManager.correct_verify_state` reruns the kept ones from the
slot with the rejected tokens' gates closed and rebuilds the window from it and
the kept inputs. The state and window at rest must equal a sequential float32 reference
that only ever saw the kept tokens, for every accepted count, across steps, in
a batch of rows, and in a slot another request had before.

On the CPU through `gdn_cpu`'s stand-ins for the kernels.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.linear_attn.base import LinearAttnManager
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.step import Segment, StepContext

POOL, LIN = "gdn_state", "linear_attn"
GEOMETRY = DeltaNetGeometry(num_k_heads=1, num_v_heads=2, head_k_dim=8, head_v_dim=8, conv_kernel_size=4)
LAYERS = 2
K = 3
VERIFY, PREFILL = "decode", "prefill"
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _kernels(monkeypatch):
    import gdn_cpu

    gdn_cpu.install(monkeypatch)


def _build(verifying: bool = True):
    pool_spec = RecurrentStateSpec(POOL, {"LLM"}, RecurrentStateConfig(
        num_layers=LAYERS, blocks=GEOMETRY.to_blocks(state_dtype=torch.float32, conv_dtype=torch.float32),
        max_slots=8,
    ))
    pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(device=CPU))
    manager = LinearAttnManager.build(
        LinearAttnSpec(LIN, {"LLM"}, LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN)),
        EngineResourceInfo(device=CPU, dependencies={POOL: pool_spec}),
    )
    manager.bind_dependencies({POOL: pool})
    if verifying:
        pool.bind_verify_walk(VERIFY, K + 1, 0)
        manager.bind_verify_walk(VERIFY, K + 1, 0)
    return pool, manager


class Weights:
    """One layer's conv and gate parameters, as GatedDeltaNet holds them."""

    def __init__(self, seed: int):
        gen = torch.Generator().manual_seed(seed)
        dim, hv = GEOMETRY.conv_dim, GEOMETRY.num_v_heads
        self.conv_w = torch.randn(dim, GEOMETRY.conv_kernel_size, generator=gen) * 0.5
        self.conv_b = torch.randn(dim, generator=gen) * 0.1
        self.a_log = torch.randn(hv, generator=gen) * 0.3
        self.dt_bias = torch.randn(hv, generator=gen) * 0.3


WEIGHTS = [Weights(10 + layer) for layer in range(LAYERS)]


def _inputs(tokens: int, seed: int):
    """Per layer: the conv's input ``x`` and the raw gate projections ``a``, ``b``."""
    gen = torch.Generator().manual_seed(seed)
    hv = GEOMETRY.num_v_heads
    return [
        (torch.randn(tokens, GEOMETRY.conv_dim, generator=gen), torch.randn(tokens, hv, generator=gen),
         torch.randn(tokens, hv, generator=gen))
        for _ in range(LAYERS)
    ]


def _split(qkv: torch.Tensor):
    g = GEOMETRY
    kd = g.num_k_heads * g.head_k_dim
    q, k, v = torch.split(qkv, [kd, kd, g.num_v_heads * g.head_v_dim], dim=-1)
    n = qkv.shape[0]
    return (q.reshape(n, g.num_k_heads, g.head_k_dim), k.reshape(n, g.num_k_heads, g.head_k_dim),
            v.reshape(n, g.num_v_heads, g.head_v_dim))


def _step(pool, manager, rows: dict[int, int], walk: str, inputs) -> None:
    """One step over ``rows`` (rid -> span), the layers driven as GatedDeltaNet drives them."""
    rids = list(rows)
    ctx = StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False)
    segments = tuple(Segment(rid, "main", span) for rid, span in rows.items())
    rstep = RecurrentStep(segments=segments)
    assert pool.admit(rstep, ctx).ok
    ctx.plan_results[POOL] = pool.plan(rstep, ctx)
    manager.plan(LinearAttnStep(segments=segments), ctx)
    for layer, (x, a, b) in enumerate(inputs):
        manager.set_default_layer_idx(layer)
        w = WEIGHTS[layer]
        qkv = manager.run_conv(x, pool.block("conv", layer), w.conv_w, w.conv_b)
        q, k, v = _split(qkv)
        manager.run(q, k, v, a, b, pool.block("state", layer), w.a_log, w.dt_bias)
    pool.commit(rstep, ctx)


def _reference(layer: int, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """The state and conv window after ``x``'s tokens alone, from zeros, in float32."""
    import gdn_cpu

    w = WEIGHTS[layer]
    width = GEOMETRY.conv_kernel_size - 1
    seq = torch.cat([torch.zeros(width, x.shape[1]), x])
    conv = F.silu(sum(w.conv_w[:, j] * seq[j: j + x.shape[0]] for j in range(width + 1)) + w.conv_b)
    q, k, v = _split(conv)
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    alpha = torch.exp(-torch.exp(w.a_log) * F.softplus(a + w.dt_bias))
    beta = torch.sigmoid(b)
    state = torch.zeros(GEOMETRY.num_v_heads, GEOMETRY.head_v_dim, GEOMETRY.head_k_dim)
    _, final = gdn_cpu.delta_rule(q, k, v, alpha, beta, state, GEOMETRY.head_k_dim ** -0.5)
    return final, seq[-width:].T


def _slot(pool, rid: int) -> int:
    return pool._slots[rid]["main"].index


def _assert_rows(pool, kept_inputs: dict[int, list]) -> None:
    for rid, per_layer in kept_inputs.items():
        for layer, (x, a, b) in enumerate(per_layer):
            state, window = _reference(layer, x, a, b)
            slot = _slot(pool, rid)
            torch.testing.assert_close(pool.block("state", layer)[slot], state, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(pool.block("conv", layer)[slot], window, atol=1e-6, rtol=1e-6)


def _cat(*parts):
    return [tuple(torch.cat(t) for t in zip(*layers, strict=True)) for layers in zip(*parts, strict=True)]


def _head(inputs, n: int):
    return [(x[:n], a[:n], b[:n]) for x, a, b in inputs]


def _add(pool, rid: int):
    pool.ingest_request(rid)
    if pool.ask(rid, "LLM", PREFILL) is not None:
        pool.reserve(rid, "LLM", PREFILL, None)


@pytest.mark.parametrize("accepted", range(K + 1))
def test_the_state_at_rest_is_the_kept_tokens_alone(accepted):
    pool, manager = _build()
    _add(pool, 1)
    prompt = _inputs(5, seed=1)
    _step(pool, manager, {1: 5}, PREFILL, prompt)
    block = _inputs(K + 1, seed=2)
    _step(pool, manager, {1: K + 1}, VERIFY, block)
    manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor([accepted])))
    _assert_rows(pool, {1: _cat(prompt, _head(block, accepted + 1))})


def test_without_the_settle_the_state_holds_none_of_the_block():
    """Negative control: the verify step's forward leaves the state and the conv
    window as they were; only the settle writes them."""
    pool, manager = _build()
    _add(pool, 1)
    prompt, block = _inputs(5, seed=1), _inputs(K + 1, seed=2)
    _step(pool, manager, {1: 5}, PREFILL, prompt)
    slot = _slot(pool, 1)
    before = [(pool.block("state", layer)[slot].clone(), pool.block("conv", layer)[slot].clone())
              for layer in range(LAYERS)]
    _step(pool, manager, {1: K + 1}, VERIFY, block)
    for layer in range(LAYERS):
        assert torch.equal(pool.block("state", layer)[slot], before[layer][0])
        assert torch.equal(pool.block("conv", layer)[slot], before[layer][1])
    with pytest.raises(AssertionError):
        _assert_rows(pool, {1: _cat(prompt, _head(block, 1))})


def test_consecutive_steps_and_a_batch_of_rows_settle_each_row():
    """The next verify step starts from the settled state: three rows, two steps,
    a different accepted count on each row and step."""
    pool, manager = _build()
    rids = [1, 2, 3]
    prompts = {rid: _inputs(3 + rid, seed=rid) for rid in rids}
    for rid in rids:
        _add(pool, rid)
        _step(pool, manager, {rid: 3 + rid}, PREFILL, prompts[rid])
    kept = {rid: prompts[rid] for rid in rids}
    for step, accepted in enumerate(([0, 3, 1], [2, 0, 3])):
        blocks = {rid: _inputs(K + 1, seed=100 * step + rid) for rid in rids}
        packed = _cat(*blocks.values())
        _step(pool, manager, dict.fromkeys(rids, K + 1), VERIFY, packed)
        manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor(accepted)))
        for rid, n in zip(rids, accepted, strict=True):
            kept[rid] = _cat(kept[rid], _head(blocks[rid], n + 1))
    _assert_rows(pool, kept)


def test_a_slot_another_request_held_settles_from_its_own_window():
    pool, manager = _build()
    _add(pool, 1)
    _step(pool, manager, {1: 4}, PREFILL, _inputs(4, seed=1))
    _step(pool, manager, {1: K + 1}, VERIFY, _inputs(K + 1, seed=2))
    manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor([K])))
    slot = _slot(pool, 1)
    pool.remove_request(1)
    _add(pool, 2)
    prompt, block = _inputs(6, seed=3), _inputs(K + 1, seed=4)
    _step(pool, manager, {2: 6}, PREFILL, prompt)
    assert _slot(pool, 2) == slot
    _step(pool, manager, {2: K + 1}, VERIFY, block)
    manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor([1])))
    _assert_rows(pool, {2: _cat(prompt, _head(block, 2))})


def test_nothing_accepted_keeps_the_last_token_and_its_window():
    """The forced-0 path: only the row's last token (the anchor) stays, and the
    conv window is recomputed from the slot's and that one input."""
    pool, manager = _build()
    _add(pool, 1)
    prompt, block = _inputs(2, seed=5), _inputs(K + 1, seed=6)
    _step(pool, manager, {1: 2}, PREFILL, prompt)
    _step(pool, manager, {1: K + 1}, VERIFY, block)
    manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor([0])))
    # the window is rebuilt as [p0, p1, x0]: the slot window's last two columns and the one kept input
    _assert_rows(pool, {1: _cat(prompt, _head(block, 1))})


def test_a_prompt_step_records_nothing_and_settles_nothing():
    pool, manager = _build()
    _add(pool, 1)
    _step(pool, manager, {1: 5}, PREFILL, _inputs(5, seed=1))
    assert manager._kept == {} and manager._verify_block is None
    before = pool.block("state", 0).clone()
    manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor([0])))
    assert torch.equal(pool.block("state", 0), before)


def test_the_settle_kernel_matches_its_reference_on_the_cpu_path():
    """`settle_conv` off CUDA is its torch reference: a padding row (the null
    slot) is skipped, a real row's window is rebuilt from its kept inputs."""
    from mstar.engine.resources.linear_attn.settle_conv import settle_conv

    gen = torch.Generator().manual_seed(7)
    dim, width, block = 6, 3, K + 1
    conv = torch.randn(4, dim, width, generator=gen)
    x = torch.randn(3 * block, dim, generator=gen)
    slots, accepted = torch.tensor([2, 0, 3]), torch.tensor([1, 3, K + 5])
    expect = conv.clone()
    for row, (slot, a) in enumerate(zip(slots.tolist(), accepted.tolist(), strict=True)):
        if slot == 0:
            continue
        kept = min(a, block - 1) + 1
        seq = torch.cat([conv[slot], x[row * block: row * block + kept].T], dim=1)
        expect[slot] = seq[:, -width:]
    settle_conv(conv, slots, accepted, x, block, null_slot_id=0)
    assert torch.equal(conv, expect)
    # two live rows on one slot would race in the kernel's in-place read and write
    with pytest.raises(AssertionError, match="share a slot"):
        settle_conv(conv, torch.tensor([2, 0, 2]), accepted, x, block, null_slot_id=0)


def test_a_pool_whose_state_nothing_settles_is_refused():
    """A pool with no linear-attention resource rerunning it would keep rejected
    tokens folded in: startup refuses it from the specs."""
    from mstar.engine.resources.draft.startup import uncorrected_problems

    pool_spec = RecurrentStateSpec(POOL, {"LLM"}, RecurrentStateConfig(
        num_layers=LAYERS, blocks=GEOMETRY.to_blocks(), max_slots=8,
    ))
    rerun = LinearAttnSpec(LIN, {"LLM"}, LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN))
    decl = SimpleNamespace(node="LLM")
    assert uncorrected_problems(decl, {POOL: pool_spec, LIN: rerun}) == []
    (problem,) = uncorrected_problems(decl, {POOL: pool_spec})
    assert "no linear-attention resource reruns its state" in problem


def test_a_verifying_node_keeps_its_recurrent_state_resident():
    """A resident verify step gathers each row's slot at span k + 1 and replays
    the conv settle and the rerun its capture recorded
    (`test_gdn_verify_resident.py`), so neither resource refuses the tables."""
    pool, manager = _build()
    assert pool.resident_refusal() is None and manager.resident_refusal() is None


def test_settling_without_the_conv_input_raises():
    pool, manager = _build()
    _add(pool, 1)
    _step(pool, manager, {1: 4}, PREFILL, _inputs(4, seed=1))
    _step(pool, manager, {1: K + 1}, VERIFY, _inputs(K + 1, seed=2))
    for rerun in manager._kept.values():
        rerun.x = None
    with pytest.raises(RuntimeError, match="run_conv"):
        manager.correct_verify_state(SimpleNamespace(accepted=torch.tensor([K])))


def test_the_pool_takes_back_its_count_and_keeps_no_decode_snapshot_of_a_verify_step():
    pool, manager = _build()
    _add(pool, 1)
    _step(pool, manager, {1: 5}, PREFILL, _inputs(5, seed=1))
    _step(pool, manager, {1: K + 1}, VERIFY, _inputs(K + 1, seed=2))
    assert pool._slots[1]["main"].tokens == 5 + K + 1
    pool.take_back([1], [K - 1], "main")
    assert pool._slots[1]["main"].tokens == 5 + 2
    assert VERIFY in pool._verify_walks


def test_unsupported_recurrent_setups_are_refused():
    from mstar.engine.resources.spec import ResourceCapabilities, ResourceCapabilitiesCtx

    ctx = ResourceCapabilitiesCtx("LLM", {})
    grants = ResourceCapabilities.granting_all()
    blocks = GEOMETRY.to_blocks()
    ok = RecurrentStateSpec(POOL, {"LLM"}, RecurrentStateConfig(1, blocks))
    assert ok.resource_capabilities(ctx) == grants
    sinkless = RecurrentStateConfig(1, blocks, disable_sink_slot=True)
    assert "sink" in RecurrentStateSpec(POOL, {"LLM"}, sinkless).resource_capabilities(ctx).verify_refusal
    two = RecurrentStateConfig(2, blocks, labels=("main", "aux"))
    assert "one label" in RecurrentStateSpec(POOL, {"LLM"}, two).resource_capabilities(ctx).verify_refusal
    for variant in (LinearAttnVariant.KDA, LinearAttnVariant.MAMBA2):
        spec = LinearAttnSpec(LIN, {"LLM"}, LinearAttnConfig(recurrent_state=POOL, variant=variant))
        assert variant.value in spec.resource_capabilities(ctx).verify_refusal
    gdn = LinearAttnSpec(LIN, {"LLM"}, LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.GDN))
    assert gdn.resource_capabilities(ctx) == grants


def test_a_verifying_node_needs_a_float32_state():
    """The multi-token verify kernel reads a float32 pool only: the resource
    refuses a bf16 state at load, before the mode is agreed."""
    pool_spec = RecurrentStateSpec(POOL, {"LLM"}, RecurrentStateConfig(
        num_layers=LAYERS, blocks=GEOMETRY.to_blocks(state_dtype=torch.bfloat16, conv_dtype=torch.float32),
        max_slots=8,
    ))
    from mstar.engine.resources.linear_attn.gdn import GDNManager

    pool = RecurrentStatePool.build(pool_spec, EngineResourceInfo(device=CPU))
    # the bf16 pool's own decode checks need FlashInfer's bf16 backend: only the verify check runs here
    manager = GDNManager.__new__(GDNManager)
    manager._pool, manager.state_dtype, manager._device, manager._verify_walks = pool, torch.bfloat16, CPU, {}
    caps = manager.capabilities()
    assert "float32" in caps.verify_refusal and caps.async_verify_refusal is None
    manager.state_dtype = torch.float32
    assert manager.capabilities().verify_refusal is None
    manager._pool = None
    assert "needs its recurrent state pool" in manager.capabilities().verify_refusal


class _Chains:
    """A KV cache whose chains key nothing yet: a decode snapshot stays unfiled."""

    def __init__(self, page: int):
        self.config = SimpleNamespace(tokens_per_page=page)

    def keyed_walks(self, label):
        return frozenset({PREFILL})

    def chain_key(self, rid, label, pages):
        return None

    def chain_keys(self, rid, label, pages):
        return []

    def bind_snapshots(self, pool):
        del pool


@pytest.mark.parametrize("verifying", [True, False])
def test_a_verify_step_ending_on_a_page_keeps_no_decode_snapshot(verifying):
    """A step shorter than a page that lands its count on a page multiple saves
    a decode snapshot, except a verify step's: its count runs past the tokens it
    keeps until the take-back. The plain walk saves one (the control)."""
    page = 2 * (K + 1)
    config = RecurrentStateConfig(
        num_layers=LAYERS, blocks=GEOMETRY.to_blocks(state_dtype=torch.float32, conv_dtype=torch.float32),
        max_slots=4, prefix_slots=2, kv_cache="kv",
    )
    pool = RecurrentStatePool(device=CPU, config=config, name=POOL)
    pool.bind_kv(_Chains(page))
    if verifying:
        pool.bind_verify_walk(VERIFY, K + 1, 0)
    _add(pool, 1)
    pool._keyed.add((1, "main"))
    free = list(pool._snap_free)
    for walk, span in ((PREFILL, K + 1), (VERIFY, K + 1)):
        ctx = StepContext(request_ids=[1], graph_walk=walk, slot=0, capture=False)
        rstep = RecurrentStep(segments=(Segment(1, "main", span),))
        assert pool.admit(rstep, ctx).ok
        pool.plan(rstep, ctx)
        pool.commit(rstep, ctx)
    assert pool._slots[1]["main"].tokens == page
    if verifying:
        assert pool._decode == {} and pool._snap_free == free
    else:
        assert (1, "main") in pool._decode and len(pool._snap_free) == len(free) - 1
