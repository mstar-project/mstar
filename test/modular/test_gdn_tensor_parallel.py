"""Sharding the delta-net mixer across ranks.

Weight loading is where tensor parallelism goes wrong here, and it goes wrong
silently: every shape still checks out, the model still runs, and the output is
merely wrong. So these rebuild each rank's slice back into the whole and
compare against the single-rank load.

The one that matters is ``[q|k|v]``. It is a single checkpoint tensor over
``conv_dim``, and a flat ``chunk(conv_dim, tp)`` — the obvious thing — hands
rank 0 the whole of q plus half of k. Both the conv weight and the ``[q|k|v]``
part of the fused projection carry that layout, so both get checked.

The four input projections live in one ``in_proj_fused`` parameter (one GEMM
at decode), so the checks pull each block back out by offset.

No collectives are involved: nothing calls ``forward``, so this runs on CPU in
CI rather than needing a real process group.
"""

from __future__ import annotations

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear_attn import ParallelGatedDeltaNet
from mstar.model.components.linear_attn import GDNProjLayout
from mstar.model.loader.base import load_weights_into
from mstar.model.qwen3_5.weight_loader import _STACKED_PARAMS

HIDDEN = 256
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_K = HEAD_V = 16
CONV_WIDTH = 4

KEY_DIM = NUM_K_HEADS * HEAD_K
VALUE_DIM = NUM_V_HEADS * HEAD_V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
QKV_BLOCKS = [KEY_DIM, KEY_DIM, VALUE_DIM]
# the fused projection's blocks, in order: [q|k|v|z|a|pad|b]. The pad keeps
# `b` 16-byte aligned per rank, so its width depends on tp — read the real
# widths off the module rather than recomputing them here.
BLOCK_INDEX = {"q": 0, "k": 1, "v": 2, "z": 3, "a": 4, "pad": 5, "b": 6}


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


def load(module, weights: dict[str, torch.Tensor], source=None) -> None:
    """Load through the real routing, so the stacked rules are under test too.

    The rules match on ``.in_proj_*``, so names need a module prefix — hence
    the holder, which stands in for the decoder layer.

    A name that is already a *fused* parameter's own arrives whole rather than
    per shard; ``test_whole_stack_reassembles_across_ranks`` produces those,
    because its checkpoint is a built model. Splitting them per block here
    keeps the production loader strict about needing a shard id. ``source`` is
    the module they came from — needed because the alignment pad is sized per
    rank, so the incoming blocks are not the destination's widths.
    """
    holder = torch.nn.Module()
    holder.mix = module
    params = dict(holder.named_parameters())
    modules = dict(holder.named_modules())
    src_modules = dict(source.named_modules()) if source is not None else {}

    by_rule, loaded = [], set()
    for name, tensor in weights.items():
        key = f"mix.{name}"
        owner_name = key.rsplit(".", 1)[0]
        owner = modules.get(owner_name)
        sizes = getattr(owner, "output_sizes", None)
        if sizes is None or not key.endswith(".weight"):
            by_rule.append((key, tensor))
            continue
        src = src_modules.get(name.rsplit(".", 1)[0])
        offset = 0
        for block, size in enumerate(getattr(src, "output_sizes", sizes)):
            # the alignment pad is the one block whose width differs between
            # degrees, and it holds no weight — everything else loads
            if size and size == sizes[block]:
                owner.weight_loader(
                    params[key], tensor.narrow(0, offset, size), block,
                )
            offset += size
        loaded.add(key)

    loaded |= load_weights_into(holder, by_rule, stacked_params=_STACKED_PARAMS)
    assert loaded == set(params), set(params) - loaded


def fused_block(module: ParallelGatedDeltaNet, name: str, tp: int) -> torch.Tensor:
    """This rank's slice of one block of the fused input projection."""
    per = [size // tp for size in module.in_proj_fused.output_sizes]
    index = BLOCK_INDEX[name]
    return module.in_proj_fused.weight.data.narrow(
        0, sum(per[:index]), per[index],
    )


def drop_pad(tensor: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """Everything but the alignment block, so tp=1 and tp=2 compare.

    The pad's width is set by the *local* head count, so it differs between
    degrees — it is layout, not weight, and nothing ever reads it.
    """
    keep, offset = [], 0
    for index, size in enumerate(sizes):
        if index != BLOCK_INDEX["pad"]:
            keep.append(tensor.narrow(0, offset, size))
        offset += size
    return torch.cat(keep, dim=0)


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

    qkv_per_rank = [
        torch.cat([fused_block(m, n, tp) for n in ("q", "k", "v")], dim=0)
        for m in ranks
    ]
    rejoined = rejoin_blocks(qkv_per_rank, QKV_BLOCKS)
    torch.testing.assert_close(rejoined, ref["in_proj_qkv.weight"])

    conv = rejoin_blocks(
        [m.conv1d.weight.data.view(-1, CONV_WIDTH) for m in ranks], QKV_BLOCKS,
    )
    torch.testing.assert_close(conv, ref["conv1d.weight"].view(-1, CONV_WIDTH))

    # and the naive split really would have been wrong
    naive = torch.cat(qkv_per_rank, dim=0)
    assert not torch.equal(naive, ref["in_proj_qkv.weight"])


@pytest.mark.parametrize("tp", [2, 4])
def test_every_parameter_reassembles(tp):
    ref = reference_weights()
    ranks = [build(tp, r) for r in range(tp)]
    for module in ranks:
        load(module, ref)

    for block in ("z", "a", "b"):
        got = torch.cat([fused_block(m, block, tp) for m in ranks], dim=0)
        name = f"in_proj_{block}.weight"
        torch.testing.assert_close(got, ref[name].to(got.dtype), msg=name)

    for name, dim in [("A_log", 0), ("dt_bias", 0), ("out_proj.weight", 1)]:
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
    from mstar.model.components.distributed import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

    owner = modules[name.rsplit(".", 1)[0]]
    # covers the delta net's [q|k|v|z|a|b] and the MLP's [gate|up] alike:
    # every block is divided separately, so a flat cat would be wrong
    if isinstance(owner, MergedColumnParallelLinear) and name.endswith("weight"):
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
        load(module, checkpoint, source=ref)

    modules = dict(ranks[0].named_modules())
    ref_modules = dict(ref.named_modules())
    per_rank = [dict(m.named_parameters()) for m in ranks]
    checked, replicated = 0, 0
    for name, want in checkpoint.items():
        shards = [p[name].data for p in per_rank]
        got = rejoin(name, modules, shards)
        owner = modules[name.rsplit(".", 1)[0]]
        if type(owner).__name__ == "_FusedBlockColumnParallelLinear":
            # tp=1 and tp=2 pad `b` differently; compare the weights, not it
            got = drop_pad(got, owner.output_sizes)
            want = drop_pad(
                want, ref_modules[name.rsplit(".", 1)[0]].output_sizes,
            )
        torch.testing.assert_close(
            got, want, msg=lambda s, n=name: f"{n}: {s}",
        )
        if got is shards[0]:
            replicated += want.numel()
        checked += 1
    assert checked > 20, f"only checked {checked} parameters"

    # Everything halves except the norms, which are head *dims* and layer
    # norms — replicated by design, and the only thing a rank holds whole.
    # The alignment pad is storage a rank carries and the reference does not.
    total = sum(p.numel() for p in ref.parameters())
    pad = sum(
        module.output_sizes[BLOCK_INDEX["pad"]] // 2 * module.weight.shape[1]
        for module in ranks[0].modules()
        if type(module).__name__ == "_FusedBlockColumnParallelLinear"
    )
    assert sum(p.numel() for p in ranks[0].parameters()) == (
        (total - replicated) // 2 + replicated + pad
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


def test_untied_lm_head_is_remapped_from_the_checkpoint_root():
    """9B and 27B untie the head; the tied sizes carry no tensor for it.

    An untied `lm_head.weight` sits at the checkpoint's *root*, outside
    `model.language_model.`, so a remapper that only passes that prefix drops
    it and the load fails with one unfilled parameter.
    """
    from mstar.model.qwen3_5.weight_loader import qwen3_5_name_remapper

    assert qwen3_5_name_remapper("lm_head.weight") == "lm_head.weight"
    assert (
        qwen3_5_name_remapper("model.language_model.layers.0.linear_attn.A_log")
        == "model.layers.0.self_attn.A_log"
    )
    # the vision tower and the MTP head load separately, or not at all
    assert qwen3_5_name_remapper("model.visual.blocks.0.attn.qkv.weight") is None
    assert qwen3_5_name_remapper("mtp.layers.0.self_attn.q_proj.weight") is None


@pytest.mark.parametrize("tied", [True, False])
def test_head_is_expected_only_when_untied(tied):
    """Tied, `named_parameters` dedupes it away; untied, it must be loaded."""
    from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM

    config = tiny_config()
    config.tie_word_embeddings = tied
    model = Qwen3_5ForCausalLM(config, CommGroup(0, 0, [0]))
    names = {n for n, _ in model.named_parameters()}
    assert ("lm_head.weight" in names) is not tied
    if tied:
        assert model.lm_head.weight is model.model.embed_tokens.weight
