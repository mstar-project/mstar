"""GLM-5.3's KDA layers through the KDA resource on the recurrent pool, on CPU.

The resource runs the layer's torch reference (``TorchKDAKernels``) against the
pool's slots; the oracle is the same layer's ``prefill`` / ``decode_step`` with
an explicit per-request state, as a request would see it alone. A slot is handed
out as zeros, so the oracle starts from zero state rather than none: the same
math, the same chunking.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.linear_attn.kda import SpecBlocks
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.glm5_next.components.attention import Glm5NextKdaAttention
from mstar.model.glm5_next.config import KDA, KDA_STATE, Glm5NextModelConfig
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel
from mstar.model.glm5_next.kda import TorchKDAKernels

LAYER = 1  # the second KDA layer's blocks: a layer index that is not 0


def _layer(seed: int) -> Glm5NextKdaAttention:
    torch.manual_seed(seed)
    layer = Glm5NextKdaAttention(Glm5NextModelConfig.reduced(), dtype=torch.float64)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if "conv1d" in name:
                p.normal_(0.0, 0.3)
            elif name.endswith(("A_log", "dt_bias")):
                p.normal_(0.0, 0.5)
            elif "norm" in name:
                p.normal_(1.0, 0.1)
            else:
                p.normal_(0.0, 0.1)
    layer.process_weights_after_loading("cpu")
    layer.requires_grad_(False)
    return layer


def _resources(max_requests: int = 3, mtp: int = 0):
    model = Glm5NextModel(
        "x", config_variant="reduced", kda_conv_dtype=torch.float64,
        kda_max_requests=max_requests, mtp_num_draft_tokens=mtp,
    )
    specs = {s.resource_key: s for s in model.get_node_resources()}
    cpu = torch.device("cpu")
    pool = build_resource(specs[KDA_STATE], EngineResourceInfo(device=cpu))
    kda = build_resource(specs[KDA], EngineResourceInfo(
        device=cpu, dependencies={KDA_STATE: specs[KDA_STATE]},
    ))
    kda.set_kernels(TorchKDAKernels())
    return pool, kda


class _Stepper:
    """Admit, plan and commit the pool and the KDA resource, the runner's order."""

    def __init__(self, pool, kda):
        self.pool, self.kda = pool, kda

    def open(self, spans: dict[str, int], walk: str, speculative: bool = False):
        segments = tuple(Segment(rid, "main", n) for rid, n in spans.items())
        for rid in spans:
            self.pool.ingest_request(rid)
        ctx = StepContext(request_ids=tuple(spans), graph_walk=walk, slot=0, capture=False)
        pool_step = RecurrentStep(segments=segments)
        assert self.pool.admit(pool_step, ctx).ok
        ctx.plan_results[KDA_STATE] = self.pool.plan(pool_step, ctx)
        self.kda.plan(LinearAttnStep(segments=segments, speculative=speculative), ctx)
        return pool_step, ctx

    def close(self, pool_step, ctx):
        self.pool.commit(pool_step, ctx)


@torch.no_grad()
@pytest.mark.parametrize("lengths", [(5, 9), (1, 70)])
def test_prefill_then_decode_matches_each_request_alone(lengths):
    layer = _layer(0)
    pool, kda = _resources()
    layer.bind_resources({KDA_STATE: pool, KDA: kda})
    steps = _Stepper(pool, kda)
    hidden = layer.hidden_size
    torch.manual_seed(1)
    rids = [f"r{i}" for i in range(len(lengths))]
    prompts = {rid: torch.randn(n, hidden, dtype=torch.float64) for rid, n in zip(rids, lengths, strict=True)}
    tokens = [{rid: torch.randn(1, hidden, dtype=torch.float64) for rid in rids} for _ in range(3)]

    step = steps.open({rid: x.shape[0] for rid, x in prompts.items()}, "prefill")
    got_prefill = layer.forward_paged(torch.cat(list(prompts.values())), LAYER)
    steps.close(*step)
    got_decode = []
    for tok in tokens:
        step = steps.open(dict.fromkeys(rids, 1), "decode")
        assert kda.current_plan().is_decode
        got_decode.append(layer.forward_paged(torch.cat([tok[r] for r in rids]), LAYER))
        steps.close(*step)

    start = 0
    for i, rid in enumerate(rids):
        want, rec, conv = layer.prefill(prompts[rid].unsqueeze(0), *layer.init_state(1))
        n = prompts[rid].shape[0]
        torch.testing.assert_close(got_prefill[start:start + n], want[0], atol=1e-10, rtol=1e-8)
        start += n
        for t, tok in enumerate(tokens):
            want = layer.decode_step(tok[rid].unsqueeze(0), rec, conv)
            torch.testing.assert_close(got_decode[t][i], want[0, 0], atol=1e-10, rtol=1e-8)
        slot = pool._slots[rid]["main"].index
        # the pool holds the state V-first; the layer's own math is K-first
        torch.testing.assert_close(pool.block("state", LAYER)[slot], rec[0].transpose(-1, -2))
        torch.testing.assert_close(pool.block("conv", LAYER)[slot], conv[0])
    # only these layers' blocks moved
    assert pool.block("state", 0).abs().sum() == 0


@torch.no_grad()
def test_verify_blocks_decode_only_the_accepted_tokens():
    """MTP's decode steps: after a prefill, verify blocks of three tokens whose verdicts keep
    1, 0 and all 2 drafts. Each block's outputs are sequential decode from the state after
    every token accepted so far."""
    layer = _layer(4)
    pool, kda = _resources(mtp=2)
    layer.bind_resources({KDA_STATE: pool, KDA: kda})
    steps = _Stepper(pool, kda)
    torch.manual_seed(5)
    prompt = torch.randn(6, layer.hidden_size, dtype=torch.float64)
    step = steps.open({"r": 6}, "prefill")
    layer.forward_paged(prompt, LAYER)
    steps.close(*step)
    _, rec, conv = layer.prefill(prompt.unsqueeze(0), *layer.init_state(1))
    for accepted in (1, 0, 2):
        block = torch.randn(3, layer.hidden_size, dtype=torch.float64)
        step = steps.open({"r": 3}, "decode", speculative=True)
        assert kda.current_plan().is_verify
        got = layer.forward_paged(block, LAYER)
        kda.set_prefix_len(SpecBlocks.of(pool, LAYER), torch.tensor([accepted]))
        steps.close(*step)
        r, c = rec.clone(), conv.clone()
        for j in range(3):
            want = layer.decode_step(block[j].view(1, 1, -1), r, c)
            # the layer's math is fp32 inside: equal to a few fp32 ulps
            torch.testing.assert_close(got[j], want[0, 0], rtol=1e-5, atol=1e-6)
            if j == accepted:
                rec, conv = r.clone(), c.clone()


@torch.no_grad()
def test_a_resumed_prefill_continues_from_the_slot():
    """A second chunk for a request reads the state its first one left."""
    layer = _layer(2)
    pool, kda = _resources()
    layer.bind_resources({KDA_STATE: pool, KDA: kda})
    steps = _Stepper(pool, kda)
    torch.manual_seed(3)
    x = torch.randn(40, layer.hidden_size, dtype=torch.float64)
    outs = []
    for chunk in (x[:25], x[25:]):
        step = steps.open({"r": chunk.shape[0]}, "prefill")
        outs.append(layer.forward_paged(chunk, LAYER))
        steps.close(*step)
    first, rec, conv = layer.prefill(x[:25].unsqueeze(0), *layer.init_state(1))
    second, _, _ = layer.prefill(x[25:].unsqueeze(0), rec, conv)
    torch.testing.assert_close(torch.cat(outs), torch.cat([first[0], second[0]]), atol=1e-10, rtol=1e-8)


def test_bind_refuses_a_conv_state_in_another_dtype():
    layer = _layer(0)
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=torch.bfloat16)
    specs = {s.resource_key: s for s in model.get_node_resources()}
    pool = build_resource(specs[KDA_STATE], EngineResourceInfo(device=torch.device("cpu")))
    with pytest.raises(RuntimeError, match="kda_conv_dtype"):
        layer.bind_resources({KDA_STATE: pool, KDA: None})
