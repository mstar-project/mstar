"""``QKVParallelLinear`` and ``MergedColumnParallelLinear`` load a checkpoint that stores
the projection already fused (Phi-3 and Phi-4's ``qkv_proj`` and ``gate_up_proj``), and
a LoRA adapter on the fused module packs every slice, as PEFT trains one on it."""

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.linear import MergedColumnParallelLinear, QKVParallelLinear


def test_a_fused_qkv_loads_as_its_three_shards():
    torch.manual_seed(0)
    make = lambda: QKVParallelLinear(  # noqa: E731
        comm_group=CommGroup.trivial(), hidden_size=16, head_size=4, total_num_heads=3, total_num_kv_heads=1,
        bias=False,
    )
    fused, split = make(), make()
    weight = torch.randn(3 * 4 + 2 * 4, 16)
    fused.weight_loader(fused.weight, weight)
    q, k, v = weight.split([12, 4, 4])
    for shard, part in zip("qkv", (q, k, v), strict=True):
        split.weight_loader(split.weight, part, shard)
    assert torch.equal(fused.weight, split.weight)
    assert torch.equal(fused.weight, weight)


def test_a_fused_gate_up_loads_as_its_shards():
    torch.manual_seed(0)
    make = lambda: MergedColumnParallelLinear(  # noqa: E731
        comm_group=CommGroup.trivial(), input_size=8, output_sizes=[6, 6], bias=False,
    )
    fused, split = make(), make()
    weight = torch.randn(12, 8)
    fused.weight_loader(fused.weight, weight)
    for shard, part in enumerate(weight.split([6, 6])):
        split.weight_loader(split.weight, part, shard)
    assert torch.equal(fused.weight, split.weight)


@pytest.mark.parametrize("module", ["qkv", "gate_up"])
def test_an_adapter_on_the_fused_module_packs_every_slice(module):
    group = CommGroup.trivial()
    if module == "qkv":
        target = QKVParallelLinear(
            comm_group=group, hidden_size=16, head_size=4, total_num_heads=3, total_num_kv_heads=1, bias=False,
        ).lora_target()
        assert target.slice_packs == ((None, ("q", "k", "v")),)
    else:
        target = MergedColumnParallelLinear(
            comm_group=group, input_size=8, output_sizes=[6, 6], bias=False,
        ).lora_target()
        assert target.slice_packs == ((None, (0, 1)),)
