"""The shared KDA layer on CPU: through a KDA Value's handle it computes what its dense reference
does, for both output gates, and TP ranks loaded from one checkpoint, each through its own
sharded pool, sum to the full layer.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    Causal,
    Entry,
    KimiDelta,
    KVEntry,
    Read,
    TensorType,
    Value,
    settle,
)
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle, build_specs
from mstar.model.components.distributed import ParallelKimiDeltaAttention
from mstar.model.components.kda import KDAConfig, KimiDeltaAttention

POOL, KDA, LAYER = "kda_state", "kda", 1
CPU = torch.device("cpu")


def _config(full_rank: bool, heads: int = 2) -> KDAConfig:
    return KDAConfig(hidden_size=48, num_heads=heads, head_dim=16,
                     full_rank_output_gate=full_rank)


def _randomize(layer: torch.nn.Module) -> None:
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


def _arrays(cfg: KDAConfig, speculative_tokens: int) -> list[Array]:
    h, d = cfg.num_heads, cfg.head_dim
    kv = Array("kv", nodes={"llm"}, layers={0}, entry=KVEntry(heads=1, head_dim=8), max_len=64,
               reads={"attn": Read(mask=Causal(), op=Attend(q_heads=1))})
    value = Value(
        POOL, nodes={"llm"}, layers={LAYER}, verify_span=speculative_tokens + 1,
        entry=Entry({
            "conv": TensorType((3 * h * d, cfg.conv_kernel_size - 1), dtype=torch.float64, shard=0),
            "state": TensorType((h, d, d), dtype=torch.float32, shard=0),
        }),
        updates={KDA: KimiDelta(lower_bound=cfg.gate_lower_bound, full_rank_gate=cfg.full_rank_output_gate)},
    )
    return [kv, value]


def _resources(cfg: KDAConfig, tp: int = 1, speculative_tokens: int = 0):
    """The layer's Value handle, pool and KDA resource, sharded over ``tp`` ranks as the engine builds them."""
    arrays = _arrays(cfg, speculative_tokens)
    specs = {spec.resource_key: spec for spec in build_specs(arrays, kernel_settings={POOL: {"max_slots": 4}})}
    group = None
    if tp > 1:
        group = SimpleNamespace(
            world_size=tp, tp_group=SimpleNamespace(world_size=tp), sp_group=SimpleNamespace(world_size=1),
        )
    pool = build_resource(specs[POOL], EngineResourceInfo(device=CPU, joint_comm_group=group))
    kda = build_resource(specs[KDA], EngineResourceInfo(
        device=CPU, joint_comm_group=group, dependencies={POOL: specs[POOL]},
    ))
    return build_handle(arrays[1], {POOL: pool, KDA: kda}, _resolve(arrays, StateSetup())), pool, kda


def _step(h, pool, kda, spans: dict[str, int], walk: str, verify: bool):
    segments = tuple(Segment(rid, "main", n) for rid, n in spans.items())
    for rid in spans:
        if rid not in pool._slots:  # once per request, as the worker does (#725)
            pool.ingest_request(rid)
    ctx = StepContext(request_ids=tuple(spans), graph_walk=walk, slot=0, capture=False)
    pool_step = RecurrentStep(segments=segments)
    assert pool.admit(pool_step, ctx).ok
    ctx.plan_results[POOL] = pool.plan(pool_step, ctx)
    kda.plan(LinearAttnStep(segments=segments, speculative=verify), ctx)
    step = SubmoduleStep(segments=list(segments), state={POOL: ArrayStep(verify=verify)})
    step.set_ctx(ctx)
    h.begin_step(step)
    return step, pool_step, ctx


def _serve(layer, resources, steps) -> list[torch.Tensor]:
    """``steps`` through the layer and its Value: (spans, walk, input, accepted drafts for a
    verify step or None)."""
    h, pool, kda = resources
    layer.bind_state({POOL: h})
    outs = []
    for spans, walk, x, accepted in steps:
        verify = accepted is not None
        step, pool_step, ctx = _step(h, pool, kda, spans, walk, verify)
        outs.append(layer(x, decode=walk == "decode"))
        if verify:
            settle(h, accepted + 1)
            kda.commit(LinearAttnStep(segments=pool_step.segments, speculative=True), ctx)
        h.check_step(step)
        h.end_step(step)
        pool.commit(pool_step, ctx)
    return outs


@torch.no_grad()
@pytest.mark.parametrize("full_rank", [False, True], ids=["low_rank_gate", "full_rank_gate"])
def test_the_handle_path_matches_the_dense_reference(full_rank):
    torch.manual_seed(0)
    cfg = _config(full_rank)
    layer = ParallelKimiDeltaAttention(cfg, dtype=torch.float64, layer=LAYER)
    _randomize(layer)
    layer.process_weights_after_loading("cpu")
    lengths = {"a": 5, "b": 70}
    prompts = {rid: torch.randn(n, cfg.hidden_size, dtype=torch.float64)
               for rid, n in lengths.items()}
    tokens = [{rid: torch.randn(1, cfg.hidden_size, dtype=torch.float64) for rid in lengths}
              for _ in range(3)]
    steps = [(lengths, "prefill", torch.cat(list(prompts.values())), None)]
    steps += [(dict.fromkeys(lengths, 1), "decode", torch.cat([tok[r] for r in lengths]), None) for tok in tokens]
    got_prefill, *got_decode = _serve(layer, _resources(cfg), steps)

    start = 0
    for i, rid in enumerate(lengths):
        want, rec, conv = layer.prefill(prompts[rid].unsqueeze(0), *layer.init_state(1))
        n = lengths[rid]
        torch.testing.assert_close(got_prefill[start:start + n], want[0], atol=1e-10, rtol=1e-8)
        start += n
        for t, tok in enumerate(tokens):
            want = layer.decode_step(tok[rid].unsqueeze(0), rec, conv)
            torch.testing.assert_close(got_decode[t][i], want[0, 0], atol=1e-10, rtol=1e-8)


class _FakeGroup:
    """A rank of a TP group whose all-reduce the test does by hand."""

    def __init__(self, rank: int, world_size: int):
        self.rank, self.world_size = rank, world_size

    def all_reduce(self, x):
        return x

    def all_reduce_buffer(self, shape, dtype, device):
        return torch.empty(shape, dtype=dtype, device=device)


def _checkpoint(full: KimiDeltaAttention) -> dict[str, tuple[torch.Tensor, str | None]]:
    """The full layer's tensors under the module's names; the conv as the checkpoints store
    it, three ``[P, 1, K]`` taps with shard ids."""
    ckpt = {name: (p.detach().clone(), None) for name, p in full.named_parameters()
            if name != "conv1d.weight"}
    taps = full.conv1d.weight.detach().split(full.qkv_dim)
    return {**ckpt, **{f"conv1d.weight/{s}": (t.clone(), s) for s, t in zip("qkv", taps, strict=True)}}



@torch.no_grad()
@pytest.mark.parametrize("full_rank", [False, True], ids=["low_rank_gate", "full_rank_gate"])
def test_tensor_parallel_ranks_sum_to_the_full_layer(full_rank):
    """Each rank serves its heads from its own sharded pool, speculative blocks included, and
    the ranks' o_proj partials sum to the unsharded layer's output at every step."""
    torch.manual_seed(1)
    cfg, k, tp = _config(full_rank, heads=4), 2, 2
    full = ParallelKimiDeltaAttention(cfg, dtype=torch.float64, layer=LAYER)
    _randomize(full)
    ckpt = _checkpoint(full)
    full.process_weights_after_loading("cpu")
    x = lambda n: torch.randn(n, cfg.hidden_size, dtype=torch.float64)  # noqa: E731
    steps = [({"a": 5, "b": 70}, "prefill", x(75), None), ({"a": 1, "b": 1}, "decode", x(2), None)]
    steps += [({"a": k + 1, "b": k + 1}, "decode", x(2 * (k + 1)), torch.tensor(accepted))
              for accepted in ([1, 0], [2, 2], [0, 1])]
    want = _serve(full, _resources(cfg, speculative_tokens=k), steps)
    total = [torch.zeros_like(w) for w in want]
    for rank in range(tp):
        layer = ParallelKimiDeltaAttention(cfg, comm_group=_FakeGroup(rank, tp), dtype=torch.float64, layer=LAYER)
        params = dict(layer.named_parameters())
        for key, (tensor, shard) in ckpt.items():
            param = params[key.split("/")[0]]
            loader = getattr(param, "weight_loader", None)
            if loader is None:  # the head dim's weights: replicated
                param.copy_(tensor)
            else:
                loader(param, tensor, shard)
        layer.process_weights_after_loading("cpu")
        for out, partial in zip(total, _serve(layer, _resources(cfg, tp, k), steps), strict=True):
            out += partial
    for i, (got, w) in enumerate(zip(total, want, strict=True)):
        torch.testing.assert_close(got, w, atol=1e-10, rtol=1e-8, msg=lambda m, i=i: f"step {i}: {m}")
