"""Greedy decode must return the same token in eager and in a captured graph.

The two paths used to encode ``temperature == 0`` differently. Eager forwards the
real temperature and sets ``any_greedy``, so the fused prep kernel emits a
one-hot at the argmax and the draw is RNG-independent. The graph path rewrote
greedy rows to ``(temperature=1, top_k=1)`` and left ``include_greedy`` off, so
the token came out of FlashInfer's top-k sampler against a Philox stream whose
offset advances once per decode step.

That only diverges when the top of the distribution is tied — but the LM head
emits bf16, where the spacing at logit magnitude ~10 is 0.0625, so any two
candidates closer than that land on exactly the same value. A tie then resolves
differently at every RNG offset, and since the offset is per request and
advances per step, two runs of the same prompt pick different tokens and the
continuations diverge. Outputs stay fluent, which is what made it look like a
numerics bug rather than a sampling one.

    python -m test.sampling_test.sampler_determinism

Exits nonzero if any sweep is not offset-invariant, or if eager and graph
disagree.
"""

import argparse

import torch

from mstar.engine.resources.sampler.utils import (
    SamplerBuffers,
    SamplingConfig,
    sample_tokens,
)

VOCAB = 262144


def _logits(batch: int, vocab: int, seed: int, ties: int) -> torch.Tensor:
    """Random bf16-quantized logits whose top ``ties`` entries are identical."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = (
        torch.randn(batch, vocab, generator=gen, device="cuda", dtype=torch.float32)
        .mul_(2.0)
        .to(torch.bfloat16)
        .to(torch.float32)
    )
    if ties:
        peak = logits.max().item() + 4.0
        for row in range(batch):
            idx = torch.randperm(vocab, generator=gen, device="cuda")[:ties]
            logits[row, idx] = peak
    return logits


def _col(value, batch: int, dtype=torch.float32) -> torch.Tensor:
    return torch.full((batch,), value, device="cuda", dtype=dtype)


def eager_greedy(logits: torch.Tensor, offset: int) -> list[int]:
    batch = logits.shape[0]
    return sample_tokens(
        logits=logits,
        temperature=_col(0.0, batch),
        top_k=_col(0, batch, torch.int32),
        top_p=_col(1.0, batch),
        repetition_penalty=_col(1.0, batch),
        seen_token_mask=None,
        any_greedy=True,
        any_top_k_zero=True,
        all_top_k_zero=True,
        seed=_col(0, batch, torch.long),
        rand_offset=_col(offset, batch, torch.long),
    ).tolist()


class CapturedGreedy:
    """The real graph path: register greedy requests, capture, then replay.

    ``logits`` is a static buffer the caller overwrites between replays, exactly
    as ``CudaGraphRunner`` stages a step's inputs into the slot it captured.
    """

    def __init__(self, batch: int, vocab: int):
        self.rids = [f"r{i}" for i in range(batch)]
        self.buffers = SamplerBuffers.allocate(
            max_batch_size=batch, device=torch.device("cuda"),
        )
        for rid in self.rids:
            config = SamplingConfig(temperature=0.0, top_k=0, top_p=1.0)
            config.set_seed(0)
            self.buffers.register_request(rid, config)
        self.buffers.gather_static(self.rids, batch, 0)
        self.buffers.gather_dynamic(self.rids, batch, 0)
        self.sampler = self.buffers.sampler_for(batch, 0)

        self.logits = torch.zeros(batch, vocab, device="cuda")
        for _ in range(2):  # triton autotune + flashinfer warmup, pre-capture
            self.sampler.sample(self.rids, self.logits)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=torch.cuda.graph_pool_handle()):
            self.tokens = self.sampler.sample(self.rids, self.logits)
        torch.cuda.synchronize()

    def __call__(self, logits: torch.Tensor, offset: int) -> list[int]:
        del offset  # the graph advances its own offset buffer per replay
        self.logits.copy_(logits)
        self.graph.replay()
        torch.cuda.synchronize()
        return self.tokens.tolist()


def sweep(name: str, fn, logits: torch.Tensor, steps: int) -> list[int]:
    """Run ``fn`` once per decode step and report whether the token ever moves."""
    seen = {tuple(fn(logits, step)) for step in range(steps)}
    status = "ok" if len(seen) == 1 else "NONDETERMINISTIC"
    print(f"  {name:<16} {len(seen)} distinct output(s) over {steps} steps  [{status}]")
    return sorted(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--ties", type=int, default=2, help="0 for separated logits")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    captured = CapturedGreedy(args.batch, args.vocab)

    failures = 0
    for trial in range(args.trials):
        logits = _logits(args.batch, args.vocab, trial, args.ties)
        top2 = logits.topk(2, dim=-1).values
        print(f"\ntrial {trial}: top-2 gap {(top2[:, 0] - top2[:, 1]).min().item():.4g}")

        eager = sweep("eager", eager_greedy, logits, args.steps)
        graph = sweep("graph replay", captured, logits, args.steps)

        if len(eager) != 1 or len(graph) != 1:
            failures += 1
        elif eager != graph:
            failures += 1
            print(f"  MISMATCH eager={eager[0]} graph={graph[0]}")

    print(f"\n{'FAIL' if failures else 'PASS'}: {failures}/{args.trials} trials bad")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
