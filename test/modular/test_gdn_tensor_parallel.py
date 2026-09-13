"""Sharding the delta-net mixer across ranks.

Weight loading is where tensor parallelism goes wrong here, and it goes wrong
silently: every shape still checks out, the model still runs, and the output is
merely wrong. So these rebuild each rank's slice back into the whole and
compare against the single-rank load.

The one that matters is ``[q|k|v]``. It is a single checkpoint tensor over
``conv_dim``, and a flat ``chunk(conv_dim, tp)`` — the obvious thing — hands
rank 0 the whole of q plus half of k. Both the conv weight and ``in_proj_qkv``
carry that layout, so both get checked.

No collectives are involved: nothing calls ``forward``, so this runs on CPU in
CI rather than needing a real process group.
"""

from __future__ import annotations

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear_attn import ParallelGatedDeltaNet
from mstar.model.components.linear_attn import GDNProjLayout

HIDDEN = 256
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_K = HEAD_V = 16
CONV_WIDTH = 4

KEY_DIM = NUM_K_HEADS * HEAD_K
VALUE_DIM = NUM_V_HEADS * HEAD_V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
QKV_BLOCKS = [KEY_DIM, KEY_DIM, VALUE_DIM]


def build(tp: int, rank: int) -> ParallelGatedDeltaNet:
    return ParallelGatedDeltaNet(
        hidden_size=HIDDEN,
        num_k_heads=NUM_K_HEADS,
        num_v_heads=NUM_V_HEADS,
        head_k_dim=HEAD_K,
        head_v_dim=HEAD_V,
        conv_kernel_size=CONV_WIDTH,
        layout=GDNProjLayout.SPLIT,
        comm_group=CommGroup(rank, rank, list(range(tp))),
    )


def reference_weights() -> dict[str, torch.Tensor]:
    """Stand-in for the checkpoint: distinct values so a misplaced slice shows."""
    torch.manual_seed(0)
    return {
        "in_proj_qkv.weight": torch.randn(CONV_DIM, HIDDEN),
        "in_proj_z.weight": torch.randn(VALUE_DIM, HIDDEN),
        "in_proj_a.weight": torch.randn(NUM_V_HEADS, HIDDEN),
        "in_proj_b.weight": torch.randn(NUM_V_HEADS, HIDDEN),
        "conv1d.weight": torch.randn(CONV_DIM, 1, CONV_WIDTH),
        "A_log": torch.randn(NUM_V_HEADS),
        "dt_bias": torch.randn(NUM_V_HEADS),
        "norm.weight": torch.randn(HEAD_V),
        "out_proj.weight": torch.randn(HIDDEN, VALUE_DIM),
    }


def load(module: ParallelGatedDeltaNet, weights: dict[str, torch.Tensor]) -> None:
    params = dict(module.named_parameters())
    for name, tensor in weights.items():
        param = params[name]
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            param.data.copy_(tensor)
        else:
            loader(param, tensor)


def rejoin_blocks(shards: list[torch.Tensor], blocks: list[int]) -> torch.Tensor:
    """Undo `_shard_blocks`: each rank holds a slice of every block."""
    tp = len(shards)
    out, offset = [], 0
    for size in blocks:
        per = size // tp
        out += [s.narrow(0, offset // tp, per) for s in shards]
        offset += size
    return torch.cat(out, dim=0)


@pytest.mark.parametrize("tp", [2, 4])
def test_qkv_and_conv_shard_per_block(tp):
    """The trap: q, k and v are each divided, not the flat concatenation."""
    ref = reference_weights()
    ranks = [build(tp, r) for r in range(tp)]
    for module in ranks:
        load(module, ref)

    rejoined = rejoin_blocks(
        [m.in_proj_qkv.weight.data for m in ranks], QKV_BLOCKS,
    )
    torch.testing.assert_close(rejoined, ref["in_proj_qkv.weight"])

    conv = rejoin_blocks(
        [m.conv1d.weight.data.view(-1, CONV_WIDTH) for m in ranks], QKV_BLOCKS,
    )
    torch.testing.assert_close(conv, ref["conv1d.weight"].view(-1, CONV_WIDTH))

    # and the naive split really would have been wrong
    naive = torch.cat([m.in_proj_qkv.weight.data for m in ranks], dim=0)
    assert not torch.equal(naive, ref["in_proj_qkv.weight"])


@pytest.mark.parametrize("tp", [2, 4])
def test_every_parameter_reassembles(tp):
    ref = reference_weights()
    ranks = [build(tp, r) for r in range(tp)]
    for module in ranks:
        load(module, ref)

    for name, dim in [
        ("in_proj_z.weight", 0), ("in_proj_a.weight", 0),
        ("in_proj_b.weight", 0), ("A_log", 0), ("dt_bias", 0),
        ("out_proj.weight", 1),
    ]:
        got = torch.cat(
            [dict(m.named_parameters())[name].data for m in ranks], dim=dim,
        )
        torch.testing.assert_close(got, ref[name].to(got.dtype), msg=name)

    # a head *dim*, not a head count — every rank holds it whole
    for module in ranks:
        torch.testing.assert_close(
            module.norm.weight.data, ref["norm.weight"].to(module.norm.weight.dtype),
        )


@pytest.mark.parametrize("tp", [2, 4])
def test_each_rank_holds_its_own_k_heads_value_heads(tp):
    """A rank's v-heads must be the ones its k-heads own.

    The delta rule pairs each k-head with `v_per_k` v-heads. If q/k sharded by
    k-head and v by v-head disagreed, every rank would mix one head's keys with
    another's values — which no shape check would catch.
    """
    ref = reference_weights()
    ranks = [build(tp, r) for r in range(tp)]
    for module in ranks:
        load(module, ref)

    v_per_k = NUM_V_HEADS // NUM_K_HEADS
    for rank, module in enumerate(ranks):
        k_local = NUM_K_HEADS // tp
        v_local = NUM_V_HEADS // tp
        assert module.num_k_heads == k_local
        assert module.num_v_heads == v_local
        # the v-heads this rank's k-heads own
        first_k = rank * k_local
        want = ref["A_log"].narrow(0, first_k * v_per_k, k_local * v_per_k)
        torch.testing.assert_close(module.A_log.data, want)


def tiny_config():
    """A Qwen3.5 shaped like the real thing but small enough for CPU."""
    from mstar.model.qwen3_5.config import (
        FULL_ATTENTION,
        LINEAR_ATTENTION,
        Qwen3_5Config,
    )

    return Qwen3_5Config(
        num_hidden_layers=4,
        hidden_size=128,
        intermediate_size=256,
        layer_types=[LINEAR_ATTENTION] * 3 + [FULL_ATTENTION],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        vocab_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        attn_output_gate=True,
        tie_word_embeddings=True,
    )


def rejoin(
    name: str, modules: dict, shards: list[torch.Tensor],
) -> torch.Tensor:
    """Undo the shard, by how that parameter was divided.

    Dispatches on the module that *owns the layout*, which for the conv is the
    mixer rather than the `Conv1d` the weight hangs off.
    """
    from mstar.model.components.distributed import RowParallelLinear

    owner = modules[name.rsplit(".", 1)[0]]
    if type(owner).__name__ == "_FusedBlockColumnParallelLinear":
        return rejoin_blocks(shards, owner.output_sizes)
    if name.endswith("conv1d.weight"):
        mixer = modules[name[: -len(".conv1d.weight")]]
        width = shards[0].shape[-1]
        return rejoin_blocks(
            [s.view(-1, width) for s in shards], mixer._qkv_blocks,
        ).view(-1, 1, width)
    if isinstance(owner, RowParallelLinear) and name.endswith("weight"):
        return torch.cat(shards, dim=1)          # sharded on the input dim
    if shards[0].shape == shards[1].shape and torch.equal(shards[0], shards[1]):
        return shards[0]                          # replicated (norms, biases)
    return torch.cat(shards, dim=0)               # column / head sharded


def test_whole_stack_reassembles_across_ranks():
    """Every parameter of a TP=2 pair rebuilds the TP=1 model.

    Broader than the GDN checks above: this also covers attention's
    double-width `q_proj`, the KV projections, the MLP and the vocab-parallel
    embedding — anywhere a wrong axis or a wrong offset would survive every
    shape assertion and only show up as bad output.
    """
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM

    config = tiny_config()
    torch.manual_seed(0)
    ref = Qwen3_5ForCausalLM(config, CommGroup(0, 0, [0]))
    for param in ref.parameters():
        param.data.normal_()
    checkpoint = {n: p.data.clone() for n, p in ref.named_parameters()}

    ranks = [Qwen3_5ForCausalLM(config, CommGroup(r, r, [0, 1])) for r in (0, 1)]
    for module in ranks:
        load(module, checkpoint)

    modules = dict(ranks[0].named_modules())
    per_rank = [dict(m.named_parameters()) for m in ranks]
    checked, replicated = 0, 0
    for name, want in checkpoint.items():
        shards = [p[name].data for p in per_rank]
        got = rejoin(name, modules, shards)
        torch.testing.assert_close(
            got, want, msg=lambda s, n=name: f"{n}: {s}",
        )
        if got is shards[0]:
            replicated += want.numel()
        checked += 1
    assert checked > 20, f"only checked {checked} parameters"

    # Everything halves except the norms, which are head *dims* and layer
    # norms — replicated by design, and the only thing a rank holds whole.
    total = sum(p.numel() for p in ref.parameters())
    assert sum(p.numel() for p in ranks[0].parameters()) == (
        (total - replicated) // 2 + replicated
    )
    assert 0 < replicated < total // 100, (
        f"{replicated} replicated of {total} — norms only, so this should be tiny"
    )


def test_fused_layout_refuses_tp_rather_than_sharding_it_wrong():
    with pytest.raises(NotImplementedError, match="head-interleaved"):
        ParallelGatedDeltaNet(
            hidden_size=HIDDEN, num_k_heads=NUM_K_HEADS, num_v_heads=NUM_V_HEADS,
            head_k_dim=HEAD_K, head_v_dim=HEAD_V, conv_kernel_size=CONV_WIDTH,
            layout=GDNProjLayout.FUSED,
            comm_group=CommGroup(0, 0, [0, 1]),
        )
