"""Small CPU checks for Command A+ normalization and routing contracts."""

import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.command_a_plus.components.language_model import (
    CommandAPlusLayerNorm,
    CommandAPlusMoeBlock,
    CommandAPlusRouter,
)
from mstar.model.command_a_plus.config import CommandAPlusConfig


class LayerNormTests(unittest.TestCase):
    def test_matches_pytorch_reference_and_preserves_dtype(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer = CommandAPlusLayerNorm(4, eps=1e-5).to(dtype=dtype)
                with torch.no_grad():
                    layer.weight.copy_(torch.tensor([0.5, 2.0, -1.0, 1.5]))
                x = torch.tensor(
                    [[[1, 2, 4, 8], [-3, 0, 2, 5]],
                     [[100, 101, 102, 103], [0.01, 0.02, 0.03, 0.04]]],
                    dtype=dtype,
                )
                expected = F.layer_norm(
                    x.float(), (4,), layer.weight.float(), eps=layer.eps,
                ).to(dtype)
                actual = layer(x)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(set(dict(layer.named_parameters())), {"weight"})

    def test_constant_input_is_finite_zero(self):
        layer = CommandAPlusLayerNorm(4, eps=1e-5)
        actual = layer(torch.full((2, 4), 7.0))
        torch.testing.assert_close(actual, torch.zeros_like(actual))


class RouterTests(unittest.TestCase):
    def make_router(self, dtype):
        router = CommandAPlusRouter(2, 3, 2).to(dtype=dtype)
        with torch.no_grad():
            router.weight.copy_(torch.tensor([[1, 0], [2, -1], [-1, 3]]))
        return router

    def test_known_expert_choices_and_normalized_sigmoid_scores(self):
        # The two tokens give logits [1, 2, -1] and [0, -1, 3].
        # Scalar reference values distinguish sigmoid normalization from softmax.
        def sigmoid(value):
            return 1.0 / (1.0 + math.exp(-value))

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                router = self.make_router(dtype)
                weights, indices, state = router(torch.eye(2, dtype=dtype))
                expected = torch.tensor([
                    [sigmoid(2), sigmoid(1)], [sigmoid(3), sigmoid(0)],
                ])
                expected /= expected.sum(dim=-1, keepdim=True)
                tolerance = 0.005 if dtype == torch.bfloat16 else 1e-6
                torch.testing.assert_close(
                    weights.float(), expected, atol=tolerance, rtol=tolerance,
                )
                torch.testing.assert_close(indices, torch.tensor([[1, 0], [2, 0]]))
                self.assertEqual(weights.dtype, dtype)
                self.assertIsNone(state)
                torch.testing.assert_close(
                    weights.float().sum(-1), torch.ones(2),
                    atol=tolerance, rtol=tolerance,
                )

    def test_router_state_is_unused(self):
        router = self.make_router(torch.float32)
        x = torch.eye(2)
        expected = router(x)
        actual = router(x, router_states=torch.ones(2, 3))
        torch.testing.assert_close(actual, expected)

    def test_existing_moe_accepts_router_contract(self):
        from mstar.model.components.moe import ParallelSparseMoeBlock

        block = ParallelSparseMoeBlock(
            hidden_size=2, num_experts=3, num_experts_per_tok=2,
            moe_intermediate_size=1, router=self.make_router(torch.float32),
        )
        with torch.no_grad():
            # All experts use gate=x[0], up=x[1]; only output scales differ.
            block.experts.gate_up_proj.copy_(torch.eye(2).expand(3, -1, -1))
            block.experts.down_proj.copy_(torch.tensor([
                [[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]],
            ]))
        x = torch.tensor([[[1.0, 1.0]]])  # logits=[1, 1, 2]
        # Avoid a top-k tie by changing expert zero's first coefficient.
        with torch.no_grad():
            block.gate.weight[0, 0] = -1.0  # logits=[-1, 1, 2]
        s1, s2 = 1 / (1 + math.exp(-1)), 1 / (1 + math.exp(-2))
        expected = F.silu(x[..., :1]) * (
            s1 * torch.tensor([3.0, 4.0]) + s2 * torch.tensor([5.0, 6.0])
        ) / (s1 + s2)
        actual, state = block(x, return_router_states=True)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(state)


class MoeBlockTests(unittest.TestCase):
    def setUp(self):
        self.config = CommandAPlusConfig.from_json(
            Path(__file__).parent / "fixtures" / "tiny_config.json",
        ).text_config

    def make_block(self, dtype=torch.float32):
        block = CommandAPlusMoeBlock(self.config).to(dtype=dtype)
        generator = torch.Generator().manual_seed(42)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.15)
        return block

    def reference(self, block, x):
        # Evaluate every expert densely, then mask out unselected experts.
        # This bypasses the block's router, sparse dispatcher and shared module.
        flat = x.float().reshape(-1, self.config.hidden_size)
        logits = F.linear(flat, block.gate.weight.float()).to(x.dtype).float()
        selected = logits.argsort(dim=-1, descending=True)[:, :self.config.num_experts_per_tok]
        scores = logits.sigmoid()
        mask = torch.zeros_like(scores).scatter_(1, selected, 1)
        scores = scores * mask
        scores = scores / scores.sum(-1, keepdim=True)
        experts = []
        for index in range(self.config.num_experts):
            gate, up = F.linear(
                flat, block.experts.gate_up_proj[index].float(),
            ).chunk(2, dim=-1)
            experts.append(F.linear(
                F.silu(gate) * up, block.experts.down_proj[index].float(),
            ))
        routed = (torch.stack(experts, dim=1) * scores.unsqueeze(-1)).sum(1)
        gate, up = F.linear(
            flat, block.shared_experts.gate_up_proj.weight.float(),
        ).chunk(2, dim=-1)
        shared = F.linear(
            F.silu(gate) * up, block.shared_experts.down_proj.weight.float(),
        )
        return ((routed + shared) / 2).reshape(x.shape)

    def test_matches_dense_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ((5, 32), (2, 3, 32)):
                with self.subTest(dtype=dtype, shape=shape):
                    block = self.make_block(dtype)
                    x = torch.randn(shape, generator=torch.Generator().manual_seed(17)).to(dtype)
                    actual = block(x)
                    expected = self.reference(block, x)
                    self.assertEqual(actual.shape, x.shape)
                    self.assertEqual(actual.dtype, dtype)
                    tolerance = 0.005 if dtype == torch.bfloat16 else 1e-6
                    torch.testing.assert_close(
                        actual.float(), expected, atol=tolerance, rtol=tolerance,
                    )

    def test_each_branch_contributes_half_even_when_other_branch_is_zero(self):
        for zero_branch in ("routed", "shared"):
            with self.subTest(zero_branch=zero_branch):
                block = self.make_block()
                with torch.no_grad():
                    if zero_branch == "routed":
                        block.experts.down_proj.zero_()
                    else:
                        block.shared_experts.down_proj.weight.zero_()
                x = torch.ones(3, self.config.hidden_size)
                actual, state = block(
                    x, router_states=torch.ones(3, 4), return_router_states=True,
                )
                torch.testing.assert_close(actual, self.reference(block, x))
                self.assertGreater(actual.abs().sum().item(), 0)
                self.assertIsNone(state)

    def test_tp8_shapes_parameter_paths_and_allocation(self):
        # Construction only: no process group or multi-GPU job is needed.
        for rank in (0, 7):
            with self.subTest(rank=rank):
                group = CommGroup(rank, rank, list(range(8)))
                with torch.device("meta"):
                    block = CommandAPlusMoeBlock(self.config, comm_group=group)
                expected_shapes = {
                    "gate.weight": (4, 32),
                    "experts.gate_up_proj": (4, 4, 32),
                    "experts.down_proj": (4, 32, 2),
                    "shared_experts.gate_up_proj.weight": (8, 32),
                    "shared_experts.down_proj.weight": (32, 4),
                }
                self.assertEqual(
                    {name: tuple(p.shape) for name, p in block.named_parameters()},
                    expected_shapes,
                )
                self.assertIs(block.comm_group, group)
                self.assertIs(block.shared_experts.gate_up_proj.comm_group, group)
                self.assertIs(block.shared_experts.down_proj.comm_group, group)
                block.to_empty(device="cpu")
                for name, parameter in block.named_parameters():
                    self.assertEqual(parameter.device.type, "cpu")
                    if name != "gate.weight":
                        self.assertTrue(callable(parameter.weight_loader))


if __name__ == "__main__":
    unittest.main()
