"""Small CPU checks for Command A+ normalization and routing."""

import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

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



if __name__ == "__main__":
    unittest.main()
