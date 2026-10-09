"""CUDA checks for the fused Command A+ kernels and the fused decoder path.

Each kernel is compared against the eager op chain it replaces; the fused
backbone is compared against the unfused one and the dense FP32 reference.
"""

import copy
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from mstar.model.command_a_plus.components.language_model import (
    CommandAPlusLanguageModel,
    CommandAPlusLayerNorm,
    CommandAPlusRouter,
)
from mstar.model.command_a_plus.config import CommandAPlusConfig
from test.command_a_plus.test_backbone import bind_test_resources, dense_reference

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "fused kernels need CUDA")
class FusedKernelTests(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator(device="cuda").manual_seed(5)

    def randn(self, *shape, scale=1.0):
        return torch.randn(*shape, generator=self.generator, device="cuda") * scale

    def test_add_layernorm_matches_eager(self):
        from mstar.model.command_a_plus.kernels import add_layernorm

        norm = CommandAPlusLayerNorm(4096, eps=1e-5).cuda().to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(1 + self.randn(4096, scale=0.1))
        for tokens in (1, 3, 64):
            x = self.randn(tokens, 4096, scale=3).to(torch.bfloat16)
            d = self.randn(tokens, 4096, scale=3).to(torch.bfloat16)
            h, y = add_layernorm(x, d, norm.weight, norm.eps)
            torch.testing.assert_close(h, x + d, atol=0, rtol=0)
            torch.testing.assert_close(y, norm(x + d), atol=2e-2, rtol=1e-2)
            same, y0 = add_layernorm(x, None, norm.weight, norm.eps)
            self.assertIs(same, x)
            torch.testing.assert_close(y0, norm(x), atol=2e-2, rtol=1e-2)

    def test_sigmoid_topk_matches_router(self):
        from mstar.model.command_a_plus.kernels import sigmoid_topk

        router = CommandAPlusRouter(4096, 128, 8).cuda().to(torch.bfloat16)
        with torch.no_grad():
            router.weight.copy_(self.randn(128, 4096, scale=0.02))
        x = self.randn(37, 4096).to(torch.bfloat16)
        expected_w, expected_ids, _ = router(x)
        # A strided view, as produced by slicing the fused projection.
        padded = torch.zeros(37, 200, dtype=torch.bfloat16, device="cuda")
        padded[:, 40:168] = F.linear(x, router.weight)
        weights, ids = sigmoid_topk(padded[:, 40:168], 8, torch.bfloat16)
        # bf16 logits can tie, and torch.topk orders ties arbitrarily; compare
        # the selected (expert, weight) sets.
        ours, order = ids.long().sort(dim=-1)
        theirs, their_order = expected_ids.sort(dim=-1)
        torch.testing.assert_close(ours, theirs, atol=0, rtol=0)
        torch.testing.assert_close(weights.gather(1, order), expected_w.gather(1, their_order),
                                   atol=0, rtol=0)

    def test_sigmoid_topk_ties_pick_lowest_index(self):
        from mstar.model.command_a_plus.kernels import sigmoid_topk

        logits = torch.zeros(2, 128, dtype=torch.bfloat16, device="cuda")
        logits[1, 100] = 1
        _, ids = sigmoid_topk(logits, 8, torch.bfloat16)
        self.assertEqual(ids[0].tolist(), list(range(8)))
        self.assertEqual(ids[1].tolist(), [100] + list(range(7)))

    def test_silu_mul_matches_eager(self):
        from mstar.model.command_a_plus.kernels import silu_mul_into

        gate_up = self.randn(5, 3000, scale=2).to(torch.bfloat16)[:, 200:]  # strided
        out = torch.zeros(5, 2800, dtype=torch.bfloat16, device="cuda")
        silu_mul_into(gate_up, out[:, 1400:], scale=0.5)
        gate, up = gate_up.chunk(2, dim=-1)
        torch.testing.assert_close(out[:, 1400:], (F.silu(gate) * up) * 0.5, atol=0, rtol=0)
        self.assertTrue((out[:, :1400] == 0).all())

    def test_kv_scatter_matches_index_put(self):
        from mstar.engine.resources.kv.cache import _kv_scatter_nhd_eager

        cache = torch.zeros(3, 10, 2, 16, 2, 128, dtype=torch.bfloat16, device="cuda")
        qkv = self.randn(5, 6 * 128).to(torch.bfloat16)  # k, v as strided views
        k = qkv[:, 2 * 128:4 * 128].view(5, 2, 128)
        v = qkv[:, 4 * 128:].view(5, 2, 128)
        page = torch.tensor([3, 3, 7, 0, 9], device="cuda")
        slot = torch.tensor([0, 1, 15, 4, 2], device="cuda")
        _kv_scatter_nhd_eager(cache, 1, k, v, page, slot)
        expected = torch.zeros_like(cache)
        expected[1][page, 0, slot] = k
        expected[1][page, 1, slot] = v
        torch.testing.assert_close(cache, expected, atol=0, rtol=0)

    def test_route_align_matches_topk_and_moe_align(self):
        from mstar.model.command_a_plus.kernels import route_align, sigmoid_topk
        from mstar.utils.fused_moe.align import moe_align_block_size

        for tokens, block_m in ((1, 16), (3, 16), (8, 16), (16, 32)):
            logits = self.randn(tokens, 128).to(torch.bfloat16)
            logits[0, 5] = logits[0, 9] = logits[0].max() + 1  # tie: lowest index first
            weights, ids, (sorted_ids, expert_ids, post_pad) = route_align(logits, 8, block_m, torch.bfloat16)
            ref_weights, ref_ids = sigmoid_topk(logits, 8, torch.bfloat16)
            self.assertTrue(torch.equal(ids, ref_ids))
            self.assertTrue(torch.equal(weights, ref_weights))
            ref_sorted, ref_experts, ref_post = moe_align_block_size(ids, block_m, 128)
            n = int(post_pad)
            self.assertEqual(n, int(ref_post))
            self.assertTrue(torch.equal(expert_ids[:n // block_m], ref_experts[:n // block_m]))
            # Slot order within an expert is unspecified; compare each block as a set.
            for b in range(n // block_m):
                block = slice(b * block_m, (b + 1) * block_m)
                self.assertEqual(sorted(sorted_ids[block].tolist()), sorted(ref_sorted[block].tolist()))

    def test_splitk_linear_then_combine_matches_fp32(self):
        from mstar.model.command_a_plus.kernels import moe_combine, splitk_linear, splitk_supported

        for tokens in (1, 5, 16):
            x = self.randn(tokens, 2048).to(torch.bfloat16)
            w = self.randn(384, 2048, scale=0.02).to(torch.bfloat16)
            cache3 = self.randn(tokens, 3, 384).to(torch.bfloat16)
            expected = x.float() @ w.float().T
            for xs in ((x,), (x[:, :1024].contiguous(), x[:, 1024:])):
                self.assertTrue(splitk_supported(xs, w))
                partials = splitk_linear(xs, w)
                torch.testing.assert_close(partials.sum(0), expected, atol=1e-4, rtol=1e-4)
            out = moe_combine(partials, cache3, scale=0.5)
            torch.testing.assert_close(out, (expected + 0.5 * cache3.float().sum(1)).to(torch.bfloat16))
        self.assertFalse(splitk_supported((self.randn(17, 2048),), w))
        self.assertFalse(splitk_supported((x[:, :1000], x[:, 1000:]), w))

    def test_moe_combine_matches_fp32_sum(self):
        from mstar.model.command_a_plus.kernels import moe_combine

        base = self.randn(4, 4096).to(torch.bfloat16)
        cache = self.randn(4, 8, 4096).to(torch.bfloat16)
        expected = (base.float() + 0.5 * cache.float().sum(1)).to(torch.bfloat16)
        torch.testing.assert_close(moe_combine(base, cache, 0.5), expected, atol=0, rtol=0)


@unittest.skipUnless(CUDA, "fused kernels need CUDA")
class FusedBackboneTests(unittest.TestCase):
    def setUp(self):
        config = CommandAPlusConfig.from_json(
            Path(__file__).parent / "fixtures" / "tiny_config.json",
        ).text_config
        # Route to every expert: with top-k < E, a bf16 near-tie in the router
        # can select a different expert in either path and swamp the comparison.
        self.config = replace(config, num_experts_per_tok=config.num_experts)

    def make_model(self):
        model = CommandAPlusLanguageModel(self.config)
        generator = torch.Generator().manual_seed(81)
        with torch.no_grad():
            for name, p in model.named_parameters():
                values = torch.randn(p.shape, generator=generator) * 0.1
                if "norm.weight" in name:
                    values += 1
                p.copy_(values)
        return model

    def run_steps(self, model, token_ids):
        _, pos, _, _ = bind_test_resources(model, self.config)
        outputs = []
        with torch.inference_mode():
            for start, stop in [(0, 5), (5, 6), (6, 7), (7, 8), (8, 9)]:
                pos.positions["main"] = torch.arange(start, stop)
                outputs.append(model(model.embed_tokens(token_ids[start:stop]), label="main"))
        return torch.cat(outputs)

    def test_fused_matches_unfused_and_dense_reference(self):
        reference = self.make_model()
        token_ids = torch.tensor([2, 17, 8, 91, 6, 11, 57, 3, 22])
        with torch.inference_mode():
            expected = dense_reference(reference, token_ids, self.config)
        with torch.device("cuda"):
            unfused = copy.deepcopy(reference).cuda().to(torch.bfloat16)
            fused = copy.deepcopy(unfused)
            fused.fuse_for_inference()
            ids = token_ids.cuda()
            unfused_out = self.run_steps(unfused, ids)
            fused_out = self.run_steps(fused, ids)
        # Both are bf16 against an fp32 reference: bound the worst element
        # loosely, and require the fused path to be no less accurate on average.
        fused_error = fused_out.float().cpu() - expected
        unfused_error = unfused_out.float().cpu() - expected
        self.assertLess(fused_error.abs().max().item(), 0.06)
        self.assertLess(fused_error.square().mean().sqrt().item(),
                        1.25 * unfused_error.square().mean().sqrt().item())

    def test_fused_weights_are_views_of_concatenated_storage(self):
        model = self.make_model().cuda().to(torch.bfloat16)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        model.fuse_for_inference()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, before[name], atol=0, rtol=0)
        layer = model.layers[0]
        attn, moe = layer.self_attn, layer.mlp
        start = layer.input_weight.data_ptr()
        self.assertEqual(attn.qkv_proj.weight.data_ptr(), start)
        self.assertEqual(moe.gate.weight.untyped_storage().data_ptr(),
                         layer.input_weight.untyped_storage().data_ptr())
        self.assertEqual(moe.shared_experts.down_proj.weight.untyped_storage().data_ptr(),
                         layer.output_weight.untyped_storage().data_ptr())


if __name__ == "__main__":
    unittest.main()
