import torch

from mstar.distributed.communication import CommGroup
from mstar.model.kimi_k2_7.components.language_model import build_moe_block
from mstar.model.kimi_k2_7.config import KimiK2Config
from mstar.utils import fused_moe


def _random_block(seed: int, comm_group: CommGroup | None = None):
    torch.manual_seed(seed)
    cfg = KimiK2Config.reduced()
    block = build_moe_block(cfg, comm_group=comm_group)
    for p in block.parameters():
        p.data.normal_(std=0.02)
    return cfg, block


def test_moe_prefill_slice_matches_unsliced():
    cfg, block = _random_block(0)
    slice_size = 16
    T = 3 * slice_size + 5
    h = torch.randn(T, cfg.hidden_size) * 0.1

    block.moe_prefill_slice = slice_size
    sliced = block(h)

    block.moe_prefill_slice = T + 1
    unsliced = block(h)

    torch.testing.assert_close(sliced, unsliced, rtol=1e-5, atol=1e-5)


def test_moe_small_prefill_skips_slicing():
    cfg, block = _random_block(1)
    block.moe_prefill_slice = 16
    h = torch.randn(5, cfg.hidden_size) * 0.1

    got = block(h)

    flat = h.view(-1, cfg.hidden_size)
    topk_weights, topk_ids = block.gate(flat)
    expected = (
        block._route(flat, topk_weights, topk_ids) + block.shared_expert(flat)
    ).view(h.shape)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


class _CountingGroup(CommGroup):
    """A CommGroup stand-in whose all_reduce is an identity that counts calls."""

    def __init__(self, world_size: int) -> None:
        super().__init__(
            my_global_rank=0, my_group_rank=0, group_members=list(range(world_size))
        )
        self.all_reduce_calls = 0

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        self.all_reduce_calls += 1
        return input_


def _fake_fused_experts(
    hidden_states, w1, w2, topk_weights, topk_ids,
    activation="silu", reduce_results=True, quant=None,
):
    """Triton-free stand-in for the CPU-only test host: treats each expert as
    the identity, weighted by topk_weights, so the (tokens, top_k, hidden)
    shape contract of the real kernel still holds."""
    weighted = hidden_states.unsqueeze(1) * topk_weights.unsqueeze(-1)
    if reduce_results:
        return weighted.sum(dim=1)
    return weighted.contiguous()


def _fake_moe_sum_reduce(input_, output, routed_scaling_factor=1.0):
    output.copy_(input_.sum(dim=1) * routed_scaling_factor)


def test_moe_tp_returns_unreduced_partial(monkeypatch):
    monkeypatch.setattr(fused_moe, "fused_experts", _fake_fused_experts)
    monkeypatch.setattr(fused_moe, "moe_sum_reduce_triton", _fake_moe_sum_reduce)

    group = _CountingGroup(world_size=2)
    cfg, block = _random_block(2, comm_group=group)
    block.moe_prefill_slice = 16

    for num_tokens in (5, 16, 40):
        group.all_reduce_calls = 0
        h = torch.randn(num_tokens, cfg.hidden_size) * 0.1
        out = block(h)
        assert out.shape == h.shape
        # The block returns its rank-local partial; KimiDecoderLayer owns
        # the all-reduce now.
        assert group.all_reduce_calls == 0


def test_moe_tp1_matches_pre_change_reference():
    group = _CountingGroup(world_size=1)
    cfg, block = _random_block(3, comm_group=group)
    h = torch.randn(6, cfg.hidden_size) * 0.1

    flat = h.view(-1, cfg.hidden_size)
    topk_weights, topk_ids = block.gate(flat)
    expected = (
        block._route(flat, topk_weights, topk_ids) + block.shared_expert(flat)
    ).view(h.shape)

    got = block(h)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert group.all_reduce_calls == 0
