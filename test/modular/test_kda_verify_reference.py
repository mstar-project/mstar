"""The checkpoint recurrence of a speculative verify step on the torch reference kernels
(``TorchKDAKernels.run_verify``): after a prefill, three verify blocks with different accepted
counts must produce the outputs the dense reference produces over the accepted sequence, and the
pool must hold the state and window after the last accepted prefix."""
import torch

from mstar.engine.resources import (
    DeltaNetGeometry,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
    Segment,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.kda_kernels import KDAParams, SpecBlocks
from mstar.model.kimi_k3.components.kda import TorchKDAKernels
from mstar.model.kimi_k3.reference.kda import from_v_first, kda_gate, kda_recurrent, l2norm, short_conv

POOL, ATTN = "kda_state", "kda_attn"
H, D, W, K = 2, 8, 4, 3
K1 = K + 1
P = 3 * H * D
DEV = torch.device("cpu")


def build():
    geometry = DeltaNetGeometry(num_k_heads=H, num_v_heads=H, head_k_dim=D, head_v_dim=D, conv_kernel_size=W)
    blocks = geometry.to_blocks(state_dtype=torch.float32, conv_dtype=torch.float32, speculative_tokens=K)
    assert DeltaNetGeometry.speculative_tokens_of(blocks) == K and DeltaNetGeometry.from_blocks(blocks) == geometry
    specs = [
        RecurrentStateSpec(POOL, {"LLM"}, RecurrentStateConfig(num_layers=1, blocks=blocks, max_slots=3)),
        LinearAttnSpec(ATTN, {"LLM"}, LinearAttnConfig(
            recurrent_state=POOL, variant=LinearAttnVariant.KDA, gate_lower_bound=-5.0)),
    ]
    info = EngineResourceInfo(device=DEV, dependencies={POOL: specs[0]})
    pool = build_resource(specs[0], info)
    attn = build_resource(specs[1], info)
    assert attn.speculative_tokens == K
    attn.set_kernels(TorchKDAKernels())
    return pool, attn, StepRunner({POOL: pool, ATTN: attn})


def step(rid, span):
    s = SubmoduleStep(segments=[Segment(rid, "main", span)], steps={POOL: RecurrentStep(), ATTN: LinearAttnStep()})
    ctx = StepContext(request_ids=[rid], graph_walk="decode" if span == K1 else "prefill", slot=0, capture=False)
    s.set_ctx(ctx)
    return s


def inputs(n, gen):
    return (torch.randn(n, P, generator=gen), torch.randn(n, H, D, generator=gen), torch.randn(n, H, generator=gen))


def dense(params, qkv, g_raw, beta_raw):
    """The reference over one whole sequence from zeros: outputs, final state, final window."""
    y, cache = short_conv(qkv, params.conv_weight, None)
    q, k, v = y.split([H * D, H * D, H * D], dim=-1)
    g_log = kda_gate(g_raw, params.A_log, params.dt_bias, params.lower_bound)
    o, state = kda_recurrent(l2norm(q.reshape(-1, H, D)), l2norm(k.reshape(-1, H, D)), v.reshape(-1, H, D),
                             g_log, torch.sigmoid(beta_raw.float()), None, params.scale)
    return o, state, cache[:, 1:]


def test_verify_blocks_match_the_dense_reference_over_the_accepted_sequence():
    gen = torch.Generator().manual_seed(0)
    params = KDAParams(conv_weight=torch.randn(P, W, generator=gen) * 0.3, A_log=torch.randn(H, generator=gen),
                       dt_bias=torch.randn(H * D, generator=gen) * 0.1, lower_bound=-5.0, num_heads=H, head_dim=D,
                       scale=D ** -0.5)
    pool, attn, runner = build()
    runner.ingest_request("a")
    conv, state = pool.block("conv", 0), pool.block("state", 0)
    spec = SpecBlocks(pool.block("spec_prefix", 0), pool.block("spec_g", 0), pool.block("spec_beta", 0),
                      pool.block("spec_len", 0))

    # prefill of 5 tokens through the paged reference path
    acc_qkv, acc_g, acc_beta = inputs(5, gen)
    s = step("a", 5)
    assert runner.admit(s).ok
    runner.plan(s)
    plan = attn.current_plan()
    assert not plan.is_verify
    attn.kernels.run_paged(acc_qkv, acc_g, acc_beta, plan, conv, state, params)
    runner.commit(s)
    slot = plan.slot_ids_cpu[0]
    assert int(spec.length[slot, 0]) == 0  # no pending prefix after a prefill

    for accepted in (2, 0, K):
        b_qkv, b_g, b_beta = inputs(K1, gen)
        s = step("a", K1)
        assert runner.admit(s).ok
        runner.plan(s)
        plan = attn.current_plan()
        assert plan.is_verify and plan.slot_ids_cpu == [slot]
        o = attn.run(b_qkv, b_g, b_beta, conv, state, params, spec=spec)
        # the checkpoint the step left: state and window after every token accepted so far
        _, want_state, want_conv = dense(params, acc_qkv, acc_g, acc_beta)
        assert torch.allclose(from_v_first(state[slot]), want_state, atol=1e-5, rtol=1e-5)
        assert torch.allclose(conv[slot], want_conv, atol=1e-6)
        # the block's outputs: what the dense reference gives at those positions after the accepted sequence
        want_o, _, _ = dense(params, torch.cat([acc_qkv, b_qkv]), torch.cat([acc_g, b_g]),
                             torch.cat([acc_beta, b_beta]))
        assert torch.allclose(o.float(), want_o[-K1:], atol=1e-5, rtol=1e-5)
        # the target accepted `accepted` drafts: bonus + accepted tokens join the sequence
        attn.set_prefix_len(spec.length, torch.tensor([accepted], dtype=torch.int32))
        assert int(spec.length[slot, 0]) == accepted + 1
        runner.commit(s)
        acc_qkv = torch.cat([acc_qkv, b_qkv[: accepted + 1]])
        acc_g = torch.cat([acc_g, b_g[: accepted + 1]])
        acc_beta = torch.cat([acc_beta, b_beta[: accepted + 1]])

    # one more block: its outputs must again follow the accepted sequence (the last block was fully accepted)
    b_qkv, b_g, b_beta = inputs(K1, gen)
    s = step("a", K1)
    assert runner.admit(s).ok
    runner.plan(s)
    o = attn.run(b_qkv, b_g, b_beta, conv, state, params, spec=spec)
    want_o, want_state, _ = dense(params, torch.cat([acc_qkv, b_qkv]), torch.cat([acc_g, b_g]),
                                  torch.cat([acc_beta, b_beta]))
    assert torch.allclose(o.float(), want_o[-K1:], atol=1e-5, rtol=1e-5)
    _, ckpt, _ = dense(params, acc_qkv, acc_g, acc_beta)
    assert torch.allclose(from_v_first(state[slot]), ckpt, atol=1e-5, rtol=1e-5)


def spec_step(rid, span):
    """A verify block declared as one, whatever its length (down to a single token: no drafts)."""
    s = SubmoduleStep(segments=[Segment(rid, "main", span)],
                      steps={POOL: RecurrentStep(), ATTN: LinearAttnStep(speculative=True)})
    ctx = StepContext(request_ids=[rid], graph_walk="decode", slot=0, capture=False)
    s.set_ctx(ctx)
    return s


def test_shorter_and_empty_blocks_follow_the_pool_slots():
    """A block length that changes between steps: a full block, a 2-token block, a 1-token block (a plain
    decode token that still consumes the pending prefix, and is committed in the step since the bonus
    needs no verdict: nothing pends after it), a full block again. Every block's outputs must follow the
    dense reference over the accepted sequence, and the checkpoint must track it."""
    gen = torch.Generator().manual_seed(1)
    params = KDAParams(conv_weight=torch.randn(P, W, generator=gen) * 0.3, A_log=torch.randn(H, generator=gen),
                       dt_bias=torch.randn(H * D, generator=gen) * 0.1, lower_bound=-5.0, num_heads=H, head_dim=D,
                       scale=D ** -0.5)
    pool, attn, runner = build()
    runner.ingest_request("a")
    conv, state = pool.block("conv", 0), pool.block("state", 0)
    spec = SpecBlocks(pool.block("spec_prefix", 0), pool.block("spec_g", 0), pool.block("spec_beta", 0),
                      pool.block("spec_len", 0))
    acc_qkv, acc_g, acc_beta = inputs(5, gen)
    s = step("a", 5)
    assert runner.admit(s).ok
    runner.plan(s)
    attn.kernels.run_paged(acc_qkv, acc_g, acc_beta, attn.current_plan(), conv, state, params)
    runner.commit(s)
    slot = attn.current_plan().slot_ids_cpu[0]
    for span, accepted in ((K1, 3), (2, 1), (1, 0), (K1, 0), (2, 0), (1, 0)):
        b_qkv, b_g, b_beta = inputs(span, gen)
        s = spec_step("a", span)
        assert runner.admit(s).ok
        runner.plan(s)
        plan = attn.current_plan()
        assert plan.is_verify and plan.k1 == span and plan.kmax1 == K1 and not plan.is_decode
        o = attn.run(b_qkv, b_g, b_beta, conv, state, params, spec=spec)
        # the checkpoint after the step: the accepted sequence, plus the block itself when it is one token
        committed = (acc_qkv, acc_g, acc_beta) if span > 1 else (
            torch.cat([acc_qkv, b_qkv]), torch.cat([acc_g, b_g]), torch.cat([acc_beta, b_beta]))
        _, want_state, want_conv = dense(params, *committed)
        assert torch.allclose(from_v_first(state[slot]), want_state, atol=1e-5, rtol=1e-5)
        assert torch.allclose(conv[slot], want_conv, atol=1e-6)
        want_o, _, _ = dense(params, torch.cat([acc_qkv, b_qkv]), torch.cat([acc_g, b_g]),
                             torch.cat([acc_beta, b_beta]))
        assert o.shape == (span, H, D) and torch.allclose(o.float(), want_o[-span:], atol=1e-5, rtol=1e-5)
        attn.set_prefix_len(spec.length, torch.tensor([accepted], dtype=torch.int32))
        assert int(spec.length[slot, 0]) == (accepted + 1 if span > 1 else 0)
        runner.commit(s)
        acc_qkv = torch.cat([acc_qkv, b_qkv[: accepted + 1]])
        acc_g = torch.cat([acc_g, b_g[: accepted + 1]])
        acc_beta = torch.cat([acc_beta, b_beta[: accepted + 1]])
