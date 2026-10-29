"""The shared KDA layer (``components/kda.py``, ``components/distributed/kda.py``) on CPU.

Through the engine's KDA resource with its torch bundle, a prefill and decodes equal the same
layer's dense reference on caller-held state (as GLM-5.3's resource test checks for its model),
for both output gates: the low-rank ``g_a``/``g_b`` pair (GLM-5.3, Kimi-Linear) and Kimi K3's
full-rank ``g_proj``, which the layer normalizes itself. Under tensor parallelism, ranks built
from one checkpoint sum to the unsharded layer, whole or pre-sliced tensors alike.
"""
from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.linear_attn.kda_torch import TorchKDAKernels
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.components.distributed import ParallelKimiDeltaAttention
from mstar.model.components.kda import KDAConfig, KimiDeltaAttention

POOL, KDA = "kda_state", "kda"
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


def _resources(cfg: KDAConfig):
    geometry = DeltaNetGeometry(
        num_k_heads=cfg.num_heads, num_v_heads=cfg.num_heads, head_k_dim=cfg.head_dim,
        head_v_dim=cfg.head_dim, conv_kernel_size=cfg.conv_kernel_size,
    )
    pool_spec = RecurrentStateSpec(POOL, {"llm"}, RecurrentStateConfig(
        num_layers=2, blocks=geometry.to_blocks(conv_dtype=torch.float64), max_slots=4,
    ))
    kda_spec = LinearAttnSpec(KDA, {"llm"}, LinearAttnConfig(
        recurrent_state=POOL, variant=LinearAttnVariant.KDA, backend=LinearAttnBackend.TRITON,
        gate_lower_bound=cfg.gate_lower_bound,
    ))
    pool = build_resource(pool_spec, EngineResourceInfo(device=CPU))
    kda = build_resource(kda_spec, EngineResourceInfo(device=CPU, dependencies={POOL: pool_spec}))
    kda.set_kernels(TorchKDAKernels())
    return pool, kda


def _step(pool, kda, spans: dict[str, int], walk: str):
    segments = tuple(Segment(rid, "main", n) for rid, n in spans.items())
    for rid in spans:
        pool.ingest_request(rid)
    ctx = StepContext(request_ids=tuple(spans), graph_walk=walk, slot=0, capture=False)
    pool_step = RecurrentStep(segments=segments)
    assert pool.admit(pool_step, ctx).ok
    ctx.plan_results[POOL] = pool.plan(pool_step, ctx)
    kda.plan(LinearAttnStep(segments=segments), ctx)
    return pool_step, ctx


@torch.no_grad()
@pytest.mark.parametrize("full_rank", [False, True], ids=["low_rank_gate", "full_rank_gate"])
def test_the_resource_path_matches_the_dense_reference(full_rank):
    torch.manual_seed(0)
    cfg = _config(full_rank)
    layer = ParallelKimiDeltaAttention(cfg, dtype=torch.float64)
    _randomize(layer)
    layer.process_weights_after_loading("cpu")
    assert hasattr(layer, "g_proj") == full_rank and hasattr(layer, "g_a_proj") != full_rank
    pool, kda = _resources(cfg)
    layer.bind_resources({POOL: pool, KDA: kda})
    lengths = {"a": 5, "b": 70}
    prompts = {rid: torch.randn(n, cfg.hidden_size, dtype=torch.float64)
               for rid, n in lengths.items()}
    tokens = [{rid: torch.randn(1, cfg.hidden_size, dtype=torch.float64) for rid in lengths}
              for _ in range(3)]

    step = _step(pool, kda, lengths, "prefill")
    got_prefill = layer(torch.cat(list(prompts.values())), 1)
    pool.commit(*step)
    got_decode = []
    for tok in tokens:
        step = _step(pool, kda, dict.fromkeys(lengths, 1), "decode")
        got_decode.append(layer(torch.cat([tok[r] for r in lengths]), 1))
        pool.commit(*step)

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
    torch.manual_seed(1)
    cfg = _config(full_rank, heads=4)
    full = KimiDeltaAttention(cfg, dtype=torch.float64)
    _randomize(full)
    ckpt = _checkpoint(full)
    x = torch.randn(1, 9, cfg.hidden_size, dtype=torch.float64)
    want, _, _ = full.prefill(x, *full.init_state(1))
    tp, total = 2, torch.zeros_like(want)
    for rank in range(tp):
        layer = ParallelKimiDeltaAttention(cfg, comm_group=_FakeGroup(rank, tp), dtype=torch.float64)
        params = dict(layer.named_parameters())
        for key, (tensor, shard) in ckpt.items():
            param = params[key.split("/")[0]]
            loader = getattr(param, "weight_loader", None)
            if loader is None:  # the head dim's norm weight: replicated
                param.copy_(tensor)
            else:
                loader(param, tensor, shard)
        out, _, _ = layer.prefill(x, *layer.init_state(1))
        total += out
    torch.testing.assert_close(total, want, atol=1e-10, rtol=1e-8)
