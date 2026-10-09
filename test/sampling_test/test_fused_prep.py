import unittest

import torch

from mstar.engine.resources.sampler.utils import _PREP_CHUNK, fused_temperature_softmax


def reference(logits, temperature, penalty=None, seen=None):
    vals = logits.float()
    if penalty is not None:
        p = penalty[:, None]
        vals = torch.where(seen, torch.where(vals > 0, vals / p, vals * p), vals)
    greedy = temperature == 0
    temp = torch.where(greedy, torch.ones_like(temperature), temperature)
    probs = torch.softmax(vals / temp[:, None], dim=-1)
    one_hot = torch.zeros_like(probs).scatter_(1, vals.argmax(-1, keepdim=True), 1.0)
    return torch.where(greedy[:, None], one_hot, probs)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class FusedPrepTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_reference_with_penalty_and_greedy_rows(self):
        V = 5 * _PREP_CHUNK + 123
        logits = torch.randn(4, V, device="cuda") * 4
        temperature = torch.tensor([0.0, 0.7, 1.0, 0.0], device="cuda")
        penalty = torch.tensor([1.3, 1.0, 1.2, 2.0], device="cuda")
        seen = torch.rand(4, V, device="cuda") < 0.1
        for args in ((), (penalty, seen)):
            probs = fused_temperature_softmax(logits, temperature, *args, include_greedy=True)
            torch.testing.assert_close(probs, reference(logits, temperature, *args), atol=1e-6, rtol=1e-4)

    def test_greedy_tie_picks_lowest_index_across_chunks(self):
        V = 4 * _PREP_CHUNK
        logits = torch.zeros(1, V, device="cuda")
        logits[0, [3 * _PREP_CHUNK + 1, _PREP_CHUNK + 7]] = 5.0
        probs = fused_temperature_softmax(logits, torch.zeros(1, device="cuda"), include_greedy=True)
        self.assertEqual(probs.argmax().item(), _PREP_CHUNK + 7)
        self.assertEqual(probs.sum().item(), 1.0)

    def test_fully_masked_chunk_and_bf16_logits(self):
        V = 3 * _PREP_CHUNK
        logits = torch.randn(2, V, device="cuda").to(torch.bfloat16)
        logits[:, 2 * _PREP_CHUNK:] = float("-inf")
        temperature = torch.tensor([0.9, 0.0], device="cuda")
        probs = fused_temperature_softmax(logits, temperature, include_greedy=True)
        self.assertFalse(probs.isnan().any())
        torch.testing.assert_close(probs, reference(logits, temperature), atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
