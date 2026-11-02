"""``causal_conv1d_decode`` against ``causal_conv1d_update``, its general kernel: output and state bit-equal at decode's
shapes, null slots and strided rows included, eager and from a captured replay."""

import pytest
import torch

from mstar.platforms import current_platform
from mstar.utils.causal_conv1d import causal_conv1d_decode, causal_conv1d_update

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="needs an AMD GPU")

DEV = "cuda"
DT = torch.bfloat16


def _case(rows, dim, seed=0, gap=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    wide = torch.randn(rows, dim + gap, device=DEV, generator=g).to(DT)
    state = torch.randn(rows + 3, dim, 3, device=DEV, generator=g).to(DT)
    weight = (torch.randn(dim, 4, device=DEV, generator=g) * 0.5).to(DT)
    bias = (torch.randn(dim, device=DEV, generator=g) * 0.1).to(DT)
    slots = torch.randperm(rows + 2, device=DEV, generator=g).to(torch.int32) + 1
    slots = slots[:rows]
    return wide, state, weight, bias, slots


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 32, 48, 64, 128])
@pytest.mark.parametrize("dim", [1536, 3072, 6144, 12288])
def test_bit_equal(rows, dim):
    wide, state, weight, bias, slots = _case(rows, dim)
    x1, s1 = wide.clone(), state.clone()
    x2, s2 = wide.clone(), state.clone()
    causal_conv1d_update(x1, s1, weight, bias=bias, activation="silu", conv_state_indices=slots, null_block_id=0)
    causal_conv1d_decode(x2, s2, weight, bias=bias, activation="silu", conv_state_indices=slots, null_block_id=0)
    torch.cuda.synchronize()
    assert torch.equal(x1, x2) and torch.equal(s1, s2)


@pytest.mark.parametrize("gap", [0, 2080])
def test_null_slots_and_strided_rows(gap):
    rows, dim = 16, 6144
    wide, state, weight, bias, slots = _case(rows, dim, gap=gap)
    slots[[3, 15]] = 0
    x1, s1 = wide.clone(), state.clone()
    x2, s2 = wide.clone(), state.clone()
    common = dict(bias=bias, activation="silu", conv_state_indices=slots, null_block_id=0)
    causal_conv1d_update(x1[:, :dim], s1, weight, **common)
    out = causal_conv1d_decode(x2[:, :dim], s2, weight, **common)
    torch.cuda.synchronize()
    assert torch.equal(x1, x2) and torch.equal(s1, s2)
    # in place, and a null row keeps its input and its state
    assert out.data_ptr() == x2.data_ptr()
    assert torch.equal(x2[3, :dim], wide[3, :dim])
    assert torch.equal(s2[0], state[0])


def test_no_bias_no_activation():
    wide, state, weight, _, slots = _case(4, 3072)
    x1, s1 = wide.clone(), state.clone()
    x2, s2 = wide.clone(), state.clone()
    causal_conv1d_update(x1, s1, weight, conv_state_indices=slots, null_block_id=0)
    causal_conv1d_decode(x2, s2, weight, conv_state_indices=slots, null_block_id=0)
    torch.cuda.synchronize()
    assert torch.equal(x1, x2) and torch.equal(s1, s2)


def test_other_operands_take_the_general_kernel():
    wide, state, weight, bias, slots = _case(4, 3072)
    state5 = torch.randn(7, 3072, 4, device=DEV).to(DT)
    weight5 = (torch.randn(3072, 5, device=DEV) * 0.5).to(DT)
    x1, s1 = wide.clone(), state5.clone()
    x2, s2 = wide.clone(), state5.clone()
    causal_conv1d_update(x1, s1, weight5, bias=bias, activation="silu", conv_state_indices=slots, null_block_id=0)
    causal_conv1d_decode(x2, s2, weight5, bias=bias, activation="silu", conv_state_indices=slots, null_block_id=0)
    torch.cuda.synchronize()
    assert torch.equal(x1, x2) and torch.equal(s1, s2)


def test_captured_replay_matches_eager_steps():
    rows, dim = 64, 6144
    source, state, weight, bias, slots = _case(rows, dim)
    x1, s1 = source.clone(), state.clone()
    x2, s2 = source.clone(), state.clone()
    buf = source.clone()

    def step():
        buf.copy_(source)
        common = dict(bias=bias, activation="silu", conv_state_indices=slots, null_block_id=0)
        return causal_conv1d_decode(buf, s2, weight, **common)

    step()
    s2.copy_(state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    s2.copy_(state)
    torch.cuda.synchronize()
    for _ in range(3):
        x1.copy_(source)
        causal_conv1d_update(x1, s1, weight, bias=bias, activation="silu", conv_state_indices=slots, null_block_id=0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(x1, buf) and torch.equal(s1, s2)
