"""GLM-5.3's KDA layers on the KDA Value it declares, on CPU, with the torch reference kernels
(``TorchKDAKernels``)."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.config import LinearAttnStep
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle
from mstar.model.base import serving_resources
from mstar.model.components.distributed import ParallelKimiDeltaAttention
from mstar.model.glm5_next.config import KDA, KDA_STATE, Glm5NextModelConfig
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel

# the second KDA layer, whose blocks are not the pool's first
LAYER = Glm5NextModelConfig.reduced().kda_layer_indices[1]


def _layer(seed: int) -> ParallelKimiDeltaAttention:
    torch.manual_seed(seed)
    layer = ParallelKimiDeltaAttention(
        Glm5NextModelConfig.reduced().kda_config(), dtype=torch.float64, layer=LAYER,
    )
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



def _value(conv_dtype: torch.dtype = torch.float64):
    """The model's KDA Value handle, its pool and its KDA resource."""
    model = Glm5NextModel("x", config_variant="reduced", kda_conv_dtype=conv_dtype, kda_max_requests=3)
    arrays = [declared for declared in model.declare_state() if isinstance(declared, Array)]
    (value,) = (array for array in arrays if array.name == KDA_STATE)
    specs = {s.resource_key: s for s in serving_resources(model, {})}
    cpu = torch.device("cpu")
    pool = build_resource(specs[KDA_STATE], EngineResourceInfo(device=cpu))
    kda = build_resource(specs[KDA], EngineResourceInfo(
        device=cpu, dependencies={KDA_STATE: specs[KDA_STATE]},
    ))
    return build_handle(value, {KDA_STATE: pool, KDA: kda}, _resolve(arrays, StateSetup())), pool, kda


class _Stepper:
    """Admit, plan and commit the pool and the KDA resource around the Value's step, the runner's order."""

    def __init__(self, h, pool, kda):
        self.h, self.pool, self.kda = h, pool, kda

    def run(self, layer: ParallelKimiDeltaAttention, spans: dict[str, int], walk: str, x: torch.Tensor):
        segments = tuple(Segment(rid, "main", n) for rid, n in spans.items())
        for rid in spans:
            if rid not in self.pool._slots:  # once per request, as the worker does (#725)
                self.pool.ingest_request(rid)
        ctx = StepContext(request_ids=tuple(spans), graph_walk=walk, slot=0, capture=False)
        pool_step = RecurrentStep(segments=segments)
        assert self.pool.admit(pool_step, ctx).ok
        ctx.plan_results[KDA_STATE] = self.pool.plan(pool_step, ctx)
        self.kda.plan(LinearAttnStep(segments=segments), ctx)
        step = SubmoduleStep(segments=list(segments), state={KDA_STATE: ArrayStep()})
        step.set_ctx(ctx)
        self.h.begin_step(step)
        out = layer(x, decode=walk == "decode")
        self.h.end_step(step)
        self.pool.commit(pool_step, ctx)
        return out


@torch.no_grad()
def test_a_resumed_prefill_continues_from_the_slot():
    """A second prefill step for a request reads the state its first one left in the slot."""
    layer = _layer(2)
    h, pool, kda = _value()
    layer.bind_state({KDA_STATE: h})
    steps = _Stepper(h, pool, kda)
    torch.manual_seed(3)
    x = torch.randn(40, layer.hidden_size, dtype=torch.float64)
    outs = [steps.run(layer, {"r": chunk.shape[0]}, "prefill", chunk) for chunk in (x[:25], x[25:])]
    # the slot is handed out as zeros, so the oracle starts from a zero state
    first, rec, conv = layer.prefill(x[:25].unsqueeze(0), *layer.init_state(1))
    second, _, _ = layer.prefill(x[25:].unsqueeze(0), rec, conv)
    torch.testing.assert_close(torch.cat(outs), torch.cat([first[0], second[0]]), atol=1e-10, rtol=1e-8)


def test_bind_refuses_a_conv_state_in_another_dtype():
    layer = _layer(0)
    h, _, _ = _value(conv_dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="KDA conv state is torch.bfloat16"):
        layer.bind_state({KDA_STATE: h})
