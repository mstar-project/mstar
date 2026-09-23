"""Greedy verification of a speculative block (``verify_greedy`` / ``SamplerResource.sample_verify``)."""
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.engine.resources.sampler.utils import verify_greedy

N, K, V = 3, 4, 10


def planted():
    """Logits whose argmax at (request i, position j) is a known token."""
    want = torch.tensor([[(i * 7 + j * 3) % V for j in range(K + 1)] for i in range(N)])
    logits = torch.zeros(N, K + 1, V).scatter_(2, want.unsqueeze(-1), 5.0) + torch.rand(N, K + 1, V)
    return want, logits.view(N * (K + 1), V)


def test_verify_greedy_counts_leading_matches_only():
    want, logits = planted()
    drafts = want[:, :K].clone()
    drafts[0, 0] = (drafts[0, 0] + 1) % V  # first draft wrong: nothing accepted
    drafts[1, 2] = (drafts[1, 2] + 1) % V  # third draft wrong: two accepted, the matching fourth does not count
    tokens, accepted = verify_greedy(logits, drafts)
    assert torch.equal(tokens, want)
    assert accepted.tolist() == [0, 2, K] and accepted.dtype == torch.int32
    # a request emits tokens[:accepted + 1]; the last of them is its new bonus token
    assert tokens[1, : accepted[1] + 1].tolist() == want[1, :3].tolist()
    assert tokens[0, : accepted[0] + 1].tolist() == want[0, :1].tolist()


def test_sampler_resource_sample_verify_on_cpu():
    res = build_resource(SamplerSpec("sampler", {"LLM"}, vocab_size=V, enable_repetion_penalty=False),
                         EngineResourceInfo(device=torch.device("cpu")))
    want, logits = planted()
    tokens, accepted = res.sample_verify(["a", "b", "c"], logits, want[:, :K])
    assert torch.equal(tokens, want) and accepted.tolist() == [K, K, K]
