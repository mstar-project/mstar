"""The eager and graph sampling paths draw one RNG stream per request.

A request samples eagerly in its prefill (the Talker's variable-length step)
and in graph for decode. Each path keeps a per-request RNG offset; if the
graph path starts its own count at 0, the first decode draw repeats the
prefill's offset and a seeded request produces a different token sequence
depending on which path a deployment ended up on.
"""

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import Sampler, SamplerBuffers, SamplingConfig
from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the graph sampler needs flashinfer on CUDA"
)

V = 512
SEED = 7
RID = "r"


def _flashinfer_or_skip():
    try:
        import flashinfer  # noqa: F401
    except Exception:
        pytest.skip("flashinfer not installed; sampler path unavailable")


def _logits(n, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    return [torch.randn(1, V, generator=g).to(device) for _ in range(n)]


def _config():
    cfg = SamplingConfig(temperature=0.9, top_k=50, top_p=1.0, vocab_size=V)
    cfg.set_seed(SEED)
    return cfg


def _eager_tokens(logits, device):
    sampler = Sampler(device=device)
    sampler.add_request(RID)
    sampler._sampling_config[RID] = _config()
    return [int(sampler.sample([RID], x).item()) for x in logits]


@requires_cuda
def test_graph_steps_continue_the_eager_offsets():
    _flashinfer_or_skip()
    device = torch.device("cuda")
    logits = _logits(6, device)
    expected = _eager_tokens(logits, device)

    sampler = Sampler(device=device)
    sampler.add_request(RID)
    sampler._sampling_config[RID] = _config()
    bufs = SamplerBuffers.allocate(
        max_batch_size=1, device=device, request_offsets=sampler._step_offset,
    )
    bufs.register_request(RID, sampler._sampling_config[RID])

    # the prefill draws eagerly, then decode draws in graph
    tokens = [int(sampler.sample([RID], logits[0]).item())]
    for x in logits[1:]:
        bufs.gather_static([RID], 1, 0)
        bufs.gather_dynamic([RID], 1, 0, gather_seen_tokens=False)
        cg = bufs.sampler_for(1, 0)
        tokens.append(int(cg.sample([RID], x).item()))
        bufs.scatter_offset(0)
        sampler._step_offset[RID] += 1  # what the resource's commit does

    assert tokens == expected


def _resource(device):
    resource = SamplerResource(
        vocab_size=V, enable_repetion_penalty=False, device=device,
    )
    resource.build_cuda_graph_buffers(
        slots=[SimpleNamespace(slot=0)], max_bs=1, max_seq_len=0,
    )
    resource.ingest_request(RID)
    resource._sampler._sampling_config[RID] = _config()
    resource._cg_buffers.update_request_config(RID, _config())
    return resource


def _ctx(bucket, capture=False):
    lease = SlotLease(slot=0, bucket=bucket) if bucket is not None else None
    return StepContext(
        request_ids=[RID], graph_walk="decode", slot=0, capture=capture,
        slot_lease=lease,
    )


@requires_cuda
def test_the_resource_keeps_one_offset_across_both_paths():
    """Eager prefill, graph decode, an eager step (a batch the graphs do not
    cover), graph decode again: one stream, the same as all-eager sampling."""
    _flashinfer_or_skip()
    device = torch.device("cuda")
    logits = _logits(8, device)
    expected = _eager_tokens(logits, device)

    resource = _resource(device)
    bucket = BucketKey(graph_walk="decode", bs=1, num_tokens=1, cg_key_info=None)
    step = SamplerStep(apply_penalty=False)

    # capture learns how many draws one replay makes
    resource.plan(step, _ctx(bucket, capture=True))
    resource.sample([RID], logits[0])
    assert resource._samples_per_replay[bucket] == 1
    resource._sampler._step_offset[RID] = 0  # the capture's dummy draw is not the request's

    tokens = []
    paths = ["eager", "graph", "graph", "graph", "eager", "graph", "graph", "graph"]
    for x, path in zip(logits, paths, strict=True):
        resource.plan(step, _ctx(bucket if path == "graph" else None))
        tokens.append(int(resource.sample([RID], x).item()))
        resource.commit(step, _ctx(bucket if path == "graph" else None))

    assert tokens == expected
    assert resource._sampler._step_offset[RID] == len(logits)
