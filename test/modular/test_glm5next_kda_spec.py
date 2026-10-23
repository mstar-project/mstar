"""KDA verify steps: the torch reference (``TorchKDAKernels.run_verify``) against sequential
decode over only the accepted tokens, and the fused kernel (``kda_triton.kda_verify``) against
the reference on CUDA."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mstar.engine.resources.linear_attn.kda import KDAPlan, SpecBlocks
from mstar.engine.resources.linear_attn.kda_triton import TritonKDAKernels
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool
from mstar.model.glm5_next import fused_decode
from mstar.model.glm5_next.components.attention import Glm5NextKdaAttention
from mstar.model.glm5_next.kda import Glm5NextKdaConfig, TorchKDAKernels


def _layer(cfg, device, dtype, seed=0):
    torch.manual_seed(seed)
    layer = Glm5NextKdaAttention(cfg, dtype=dtype).to(device)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif name.endswith(("A_log", "dt_bias")):
                p.normal_(0.0, 0.5)
            elif "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.02 if p.shape[-1] > 256 else 0.1)
    layer.process_weights_after_loading(device)
    return layer.requires_grad_(False)


def _pool(layer, slots, k, device, conv_dtype):
    """One layer's pool with speculative blocks, its state and conv tails random."""
    geometry = DeltaNetGeometry(
        num_k_heads=layer.num_heads, num_v_heads=layer.num_heads, head_k_dim=layer.head_dim,
        head_v_dim=layer.head_dim, conv_kernel_size=4,
    )
    pool = RecurrentStatePool(torch.device(device), RecurrentStateConfig(
        num_layers=1, max_slots=slots,
        blocks=geometry.to_blocks(conv_dtype=conv_dtype, speculative_tokens=k),
    ))
    pool.block("state", 0).copy_(torch.randn(pool.block("state", 0).shape) * 0.1)
    pool.block("conv", 0).copy_(torch.randn(pool.block("conv", 0).shape) * 0.5)
    pool.block("spec_side").random_(0, 2)  # the side a slot starts on does not matter
    return pool


def _clone(pool):
    other = RecurrentStatePool(pool._device, pool.config)
    for name, tensor in pool._blocks.items():
        other._blocks[name].copy_(tensor)
    return other


def _plan(slots, block):
    n = slots.numel()
    return KDAPlan(slot_ids=slots.to(torch.int32), has_state=torch.ones(n, dtype=torch.bool),
                   spans=(block,) * n, num_tokens=n * block, is_decode=False, is_verify=True,
                   block=block)


def _verify(bundle, layer, x, pool, plan):
    proj = F.linear(x, layer._in_proj_weight)
    p, h, d = layer.qkv_dim, layer.num_heads, layer.head_dim
    return bundle.run_verify(
        proj[:, :3 * p], proj[:, 3 * p + h:3 * p + h + d], proj[:, 3 * p:3 * p + h], plan,
        pool.block("conv", 0), pool.block("state", 0), SpecBlocks.of(pool, 0), layer.params(),
        gate=proj[:, 3 * p + h + d:3 * p + h + 2 * d],
    )


def _after_verdict(pool, slots, accepted):
    """``KDAManager.set_prefix_len``: the accepted tokens and the token after them."""
    spec = SpecBlocks.of(pool, 0)
    spec.prefix_len[slots, 0] = (accepted + 1).to(torch.int32)
    spec.side[slots, 0] = 1 - spec.side[slots, 0]


@torch.no_grad()
@pytest.mark.parametrize("block", [1, 2, 4])
def test_reference_matches_sequential_decode_over_accepted_tokens(block):
    """Each block's outputs equal sequential decode from the state after every accepted token
    so far; the pool's state and the cached conv window lag one step behind it."""
    layer = _layer(Glm5NextKdaConfig.reduced(), "cpu", torch.float64)
    pool = _pool(layer, 6, 3, "cpu", torch.float64)
    spec = SpecBlocks.of(pool, 0)
    slots = torch.tensor([4, 1, 2])
    true_rec = [pool.block("state", 0)[s].transpose(-1, -2).clone()[None] for s in slots.tolist()]
    true_conv = [pool.block("conv", 0)[s].clone()[None] for s in slots.tolist()]
    gen = torch.Generator().manual_seed(7)
    for step in range(5):
        x = torch.randn(len(slots) * block, layer.hidden_size, generator=gen, dtype=torch.float64)
        out = layer.o_proj(_verify(TorchKDAKernels(), layer, x, pool, _plan(slots, block)))
        accepted = torch.randint(0, block, (len(slots),), generator=gen)
        for i, s in enumerate(slots.tolist()):
            # the checkpoint is the state before this block, and the window this step cached
            # (on the side it wrote) the conv tail before it
            torch.testing.assert_close(pool.block("state", 0)[s], true_rec[i][0].transpose(-1, -2),
                                       rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(spec.conv[s, 1 - int(spec.side[s])], true_conv[i][0])
            r, c = true_rec[i].clone(), true_conv[i].clone()
            for j in range(block):
                want = layer.decode_step(x[i * block + j].view(1, 1, -1), r, c).view(-1)
                # the layer's math is fp32 inside: equal to a few fp32 ulps
                torch.testing.assert_close(out[i * block + j], want, rtol=1e-5, atol=1e-6,
                                           msg=f"step {step} row {i} token {j}")
                if j == int(accepted[i]):
                    true_rec[i], true_conv[i] = r.clone(), c.clone()
        _after_verdict(pool, slots, accepted)


@torch.no_grad()
def test_padding_rows_on_the_sink_clamp_a_garbage_count():
    layer = _layer(Glm5NextKdaConfig.reduced(), "cpu", torch.float64, seed=2)
    pool = _pool(layer, 4, 1, "cpu", torch.float64)
    SpecBlocks.of(pool, 0).prefix_len[SINK_SLOT] = 1_000_000
    x = torch.randn(2 * 2, layer.hidden_size, dtype=torch.float64)
    out = _verify(TorchKDAKernels(), layer, x, pool, _plan(torch.tensor([3, SINK_SLOT]), 2))
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not (torch.cuda.is_available() and fused_decode._HAS_TRITON),
                    reason="the fused verify kernel needs CUDA + triton")
@torch.no_grad()
@pytest.mark.parametrize("block", [1, 2, 4])
@pytest.mark.parametrize("batch", [1, 7, 64])
def test_kernel_matches_reference(block, batch):
    """One TP8 rank's KDA layer (8 heads of 128) over four verify steps with random verdicts."""
    cfg = Glm5NextKdaConfig(hidden_size=4096, linear_num_heads=8, linear_head_dim=128)
    layer = _layer(cfg, "cuda", torch.bfloat16)
    ker = _pool(layer, 70, 3, "cuda", torch.bfloat16)
    ref = _clone(ker)
    slots = torch.randperm(69, device="cuda")[:batch] + 1
    gen = torch.Generator(device="cuda").manual_seed(3)
    for step in range(4):
        x = torch.randn(batch * block, cfg.hidden_size, device="cuda", generator=gen)
        x = x.to(torch.bfloat16)
        out = _verify(TritonKDAKernels(), layer, x, ker, _plan(slots, block))
        out_ref = _verify(TorchKDAKernels(), layer, x, ref, _plan(slots, block))
        torch.testing.assert_close(out, out_ref, rtol=2e-2, atol=2e-2, msg=f"step {step}")
        # the kernel sums the state in 32-column tiles, torch whole: fp32 order alone moves
        # single elements by ~1e-3; the cached inputs and windows are exact
        for name, tol in (("state", 5e-3), ("conv", 0), ("spec_prefix", 0), ("spec_g", 0),
                          ("spec_beta", 0), ("spec_conv", 0)):
            torch.testing.assert_close(ker._blocks[name], ref._blocks[name], rtol=tol, atol=tol,
                                       msg=f"{name} step {step}")
        accepted = torch.randint(0, block, (batch,), device="cuda", generator=gen)
        _after_verdict(ker, slots, accepted)
        _after_verdict(ref, slots, accepted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
@torch.no_grad()
def test_reference_replays_under_a_cuda_graph():
    """The reference reads nothing on the host, so a captured decode can replay it with the
    next step's slots, counts and inputs."""
    cfg = Glm5NextKdaConfig(hidden_size=256, linear_num_heads=2, linear_head_dim=128)
    layer = _layer(cfg, "cuda", torch.bfloat16)
    graph_pool = _pool(layer, 8, 3, "cuda", torch.bfloat16)
    eager_pool = _clone(graph_pool)
    block, slots = 4, torch.tensor([3, 5], device="cuda")
    x = torch.zeros(2 * block, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    plan = _plan(slots, block)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):  # warm up off the default stream, as capture needs
        _verify(TorchKDAKernels(), layer, x, _clone(graph_pool), plan)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _verify(TorchKDAKernels(), layer, x, graph_pool, plan)
    gen = torch.Generator(device="cuda").manual_seed(4)
    for step, new_slots in enumerate(([3, 5], [5, 1], [1, 3])):
        # capture ran nothing: the pools are as they were until the first replay
        plan.slot_ids.copy_(torch.tensor(new_slots, dtype=torch.int32))
        x.copy_(torch.randn(x.shape, device="cuda", generator=gen))
        graph.replay()
        want = _verify(TorchKDAKernels(), layer, x, eager_pool, _plan(plan.slot_ids.long(), block))
        torch.testing.assert_close(out, want, rtol=0, atol=0, msg=f"step {step}")
        accepted = torch.randint(0, block, (2,), device="cuda", generator=gen)
        _after_verdict(graph_pool, plan.slot_ids.long(), accepted)
        _after_verdict(eager_pool, plan.slot_ids.long(), accepted)
