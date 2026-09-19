"""GPU: the fla verify path (``FLAKDAKernels.run_verify``, one launch of the checkpoint recurrent
kernel over prefix + block) against the torch reference on the same pool, and the kernel alone
against fla's fused recurrent on plain rows."""
import pytest
import torch

from mstar.engine.resources import (
    DeltaNetGeometry, LinearAttnConfig, LinearAttnSpec, LinearAttnStep, LinearAttnVariant,
    RecurrentStateConfig, RecurrentStateSpec, RecurrentStep, Segment, StepContext, StepRunner, SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.kda_kernels import FLAKDAKernels, KDAParams, SpecBlocks
from mstar.model.kimi_k3.components.kda import TorchKDAKernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
POOL, ATTN = "kda_state", "kda_attn"
H, D, W, K = 4, 128, 4, 7
K1 = K + 1
P = 3 * H * D
DEV = torch.device("cuda")


def build(kernels):
    geometry = DeltaNetGeometry(num_k_heads=H, num_v_heads=H, head_k_dim=D, head_v_dim=D, conv_kernel_size=W)
    blocks = geometry.to_blocks(state_dtype=torch.float32, conv_dtype=torch.bfloat16, speculative_tokens=K)
    specs = [
        RecurrentStateSpec(POOL, {"LLM"}, RecurrentStateConfig(num_layers=1, blocks=blocks, max_slots=4)),
        LinearAttnSpec(ATTN, {"LLM"}, LinearAttnConfig(recurrent_state=POOL, variant=LinearAttnVariant.KDA,
                                                        gate_lower_bound=-5.0)),
    ]
    info = EngineResourceInfo(device=DEV, dependencies={POOL: specs[0]})
    pool, attn = build_resource(specs[0], info), build_resource(specs[1], info)
    attn.set_kernels(kernels)
    return pool, attn, StepRunner({POOL: pool, ATTN: attn})


def step(rids, span):
    s = SubmoduleStep(segments=[Segment(r, "main", span) for r in rids],
                      steps={POOL: RecurrentStep(), ATTN: LinearAttnStep()})
    ctx = StepContext(request_ids=list(rids), graph_walk="decode", slot=0, capture=False)
    s.set_ctx(ctx)
    return s


def spec_of(pool):
    return SpecBlocks(pool.block("spec_prefix", 0), pool.block("spec_g", 0), pool.block("spec_beta", 0),
                      pool.block("spec_len", 0))


def test_fla_verify_matches_the_torch_reference():
    gen = torch.Generator(device=DEV).manual_seed(0)
    params = KDAParams(conv_weight=(torch.randn(P, W, device=DEV, generator=gen) * 0.3).to(torch.bfloat16),
                       A_log=torch.randn(H, device=DEV, generator=gen), dt_bias=torch.randn(H * D, device=DEV, generator=gen) * 0.1,
                       lower_bound=-5.0, num_heads=H, head_dim=D, scale=D ** -0.5)
    rids = ["a", "b"]
    trees = {}
    for name, kernels in (("torch", TorchKDAKernels()), ("fla", FLAKDAKernels())):
        torch.manual_seed(1)
        pool, attn, runner = build(kernels)
        for r in rids:
            runner.ingest_request(r)
        trees[name] = (pool, attn, runner)

    def inputs(n):
        g = torch.Generator(device=DEV).manual_seed(n)
        return (torch.randn(n, P, device=DEV, generator=g).to(torch.bfloat16),
                torch.randn(n, H, D, device=DEV, generator=g).to(torch.bfloat16),
                torch.randn(n, H, device=DEV, generator=g).to(torch.bfloat16))

    # prefill of 6 tokens per request on both trees (same inputs)
    pre = inputs(12)
    for pool, attn, runner in trees.values():
        s = step(rids, 6)
        assert runner.admit(s).ok
        runner.plan(s)
        attn.run(*pre, pool.block("conv", 0), pool.block("state", 0), params)
        runner.commit(s)
    def sync_fla_to_torch():
        # the prefills ran on different conv kernels (bf16 roundings differ by an ulp here and
        # there), so seed the fla tree with the torch tree's blocks before each verify step:
        # what is compared is the verify path alone
        src, dst = trees["torch"][0], trees["fla"][0]
        for name in ("conv", "state", "spec_prefix", "spec_g", "spec_beta", "spec_len"):
            dst.block(name, 0).copy_(src.block(name, 0))

    # verify blocks with different acceptance per request and per step
    for accepted in ((3, 0), (7, 2), (0, 5)):
        sync_fla_to_torch()
        blk = inputs(2 * K1)
        outs, states, wins = {}, {}, {}
        for name, (pool, attn, runner) in trees.items():
            s = step(rids, K1)
            assert runner.admit(s).ok
            runner.plan(s)
            plan = attn.current_plan()
            assert plan.is_verify
            outs[name] = attn.run(*blk, pool.block("conv", 0), pool.block("state", 0), params, spec=spec_of(pool)).float()
            attn.set_prefix_len(spec_of(pool).length, torch.tensor(accepted, dtype=torch.int32, device=DEV))
            runner.commit(s)
            slots = plan.slot_ids_cpu
            states[name] = pool.block("state", 0)[slots].clone()
            wins[name] = pool.block("conv", 0)[slots].float().clone()
        assert torch.allclose(outs["fla"], outs["torch"], atol=3e-2, rtol=3e-2), (outs["fla"] - outs["torch"]).abs().max()
        assert torch.allclose(states["fla"], states["torch"], atol=1e-3, rtol=1e-3), (states["fla"] - states["torch"]).abs().max()
        assert torch.equal(wins["fla"], wins["torch"])


def test_checkpoint_kernel_matches_fla_on_plain_rows():
    from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

    from mstar.engine.resources.linear_attn.kda_spec_recurrent import kda_recurrent_checkpoint

    torch.manual_seed(0)
    n, t = 3, 2 * K1
    q, k, v, g = (torch.randn(n * t, H, D, device=DEV).to(torch.bfloat16) for _ in range(4))
    beta = torch.randn(n * t, H, device=DEV).to(torch.bfloat16)
    A_log = torch.randn(H, device=DEV)
    dt_bias = torch.randn(H * D, device=DEV) * 0.1
    S0 = torch.randn(n, H, D, D, device=DEV)
    cu = torch.arange(n + 1, device=DEV, dtype=torch.int32) * t
    # fla over every row, final states returned
    o_ref, S_ref = fused_recurrent_kda_fwd(
        q=q.view(1, n * t, H, D), k=k.view(1, n * t, H, D), v=v.view(1, n * t, H, D), g=g.view(1, n * t, H, D),
        beta=beta.view(1, n * t, H), A_log=A_log, dt_bias=dt_bias, initial_state=S0.clone(), scale=D ** -0.5,
        output_final_state=True, state_v_first=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=-5.0)
    # ours: slots in a pool of 5 (rows use slots 4, 1, 3), checkpoint after the last token for rows 0 and 2, never for row 1
    pool = torch.zeros(5, H, D, D, device=DEV)
    slots = torch.tensor([4, 1, 3], device=DEV, dtype=torch.int32)
    pool[slots.long()] = S0
    ckpt = torch.tensor([t - 1, -1, t - 1], device=DEV, dtype=torch.int32)
    o = kda_recurrent_checkpoint(q, k, v, g, beta, A_log, dt_bias, pool, slots, ckpt, cu, D ** -0.5, -5.0)
    assert torch.allclose(o.float(), o_ref.view(n * t, H, D).float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(pool[4], S_ref[0], atol=1e-4, rtol=1e-4) and torch.allclose(pool[3], S_ref[2], atol=1e-4, rtol=1e-4)
    assert torch.equal(pool[1], S0[1])  # row 1 stored nothing
    # a mid-row checkpoint equals the state after that many tokens
    ckpt2 = torch.tensor([4, 4, 4], device=DEV, dtype=torch.int32)
    pool[slots.long()] = S0
    kda_recurrent_checkpoint(q, k, v, g, beta, A_log, dt_bias, pool, slots, ckpt2, cu, D ** -0.5, -5.0)
    cu5 = torch.arange(n + 1, device=DEV, dtype=torch.int32) * 5
    take = torch.cat([torch.arange(i * t, i * t + 5, device=DEV) for i in range(n)])
    _, S5 = fused_recurrent_kda_fwd(
        q=q[take].view(1, -1, H, D), k=k[take].view(1, -1, H, D), v=v[take].view(1, -1, H, D), g=g[take].view(1, -1, H, D),
        beta=beta[take].view(1, -1, H), A_log=A_log, dt_bias=dt_bias, initial_state=S0.clone(), scale=D ** -0.5,
        output_final_state=True, state_v_first=True, cu_seqlens=cu5, use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=-5.0)
    assert torch.allclose(pool[slots.long()], S5, atol=1e-4, rtol=1e-4)


def test_checkpoint_kernel_block_only_output_matches_the_full_one():
    """``out_skip``: the same recurrence stores only each row's tokens past the prefix part, into
    ``[N * k1, H, D]``, equal to slicing the full output."""
    from mstar.engine.resources.linear_attn.kda_spec_recurrent import kda_recurrent_checkpoint

    torch.manual_seed(3)
    n, kp, k1 = 3, K1, 5
    t = n * (kp + k1)
    q, k, v, g = (torch.randn(t, H, D, device=DEV).to(torch.bfloat16) for _ in range(4))
    beta = torch.randn(t, H, device=DEV).to(torch.bfloat16)
    A_log = torch.randn(H, device=DEV)
    dt_bias = torch.randn(H * D, device=DEV) * 0.1
    S0 = torch.randn(n + 2, H, D, D, device=DEV)
    slots = torch.tensor([3, 0, 4], device=DEV, dtype=torch.int32)
    ckpt = torch.tensor([2, -1, kp - 1], device=DEV, dtype=torch.int32)
    cu = torch.arange(n + 1, device=DEV, dtype=torch.int32) * (kp + k1)
    pool_a, pool_b = S0.clone(), S0.clone()
    full = kda_recurrent_checkpoint(q, k, v, g, beta, A_log, dt_bias, pool_a, slots, ckpt, cu, D ** -0.5, -5.0)
    block = kda_recurrent_checkpoint(q, k, v, g, beta, A_log, dt_bias, pool_b, slots, ckpt, cu, D ** -0.5, -5.0, out_skip=kp)
    assert block.shape == (n * k1, H, D)
    assert torch.equal(block, full.view(n, kp + k1, H, D)[:, kp:].reshape(n * k1, H, D))
    assert torch.equal(pool_a, pool_b)
