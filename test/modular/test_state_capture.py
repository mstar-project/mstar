"""A resource says whether it can capture its own step, and the runner drops a config it can't.

GLM-5.3 used to leave out the prefill configs the torch KDA kernels can't replay, and every config on the MLA torch
fallback, by asking each resource itself. A replay of either would read the capture's spans or pages, so the
resources answer for their own step now and no model asks.
"""

from types import SimpleNamespace

import torch

from mstar.engine.accelerator_graph_runner import blocked_capture_resources
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager
from mstar.engine.resources.linear_attn.config import LinearAttnConfig, LinearAttnStep, LinearAttnVariant
from mstar.engine.resources.linear_attn.kda import KDAManager
from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.engine.resources.step import Segment

CPU = torch.device("cpu")


def _kda() -> KDAManager:
    """On the CPU, so on the torch kernels, whose prefill slices the spans on the host."""
    geometry = DeltaNetGeometry(num_k_heads=2, num_v_heads=2, head_k_dim=8, head_v_dim=8, conv_kernel_size=4)
    config = LinearAttnConfig(recurrent_state="kda_state", variant=LinearAttnVariant.KDA)
    return KDAManager(config, geometry, num_layers=1, state_dtype=torch.float32, device=CPU, speculative_tokens=3)


def _rows(*spans: int) -> tuple[Segment, ...]:
    return tuple(Segment(f"r{i}", "main", span) for i, span in enumerate(spans))


def _blocked(key: str, resource, resource_step) -> list[str]:
    return blocked_capture_resources(SimpleNamespace(steps={key: resource_step}), {key: resource})


def test_the_runner_drops_a_prefill_capture_the_kda_kernels_can_t_replay():
    kda = _kda()
    assert _blocked("kda", kda, LinearAttnStep(segments=_rows(4, 2))) == ["kda"]
    assert _blocked("kda", kda, LinearAttnStep(segments=_rows(1, 1))) == [], "a decode step runs no prefill kernel"
    verify = LinearAttnStep(segments=_rows(4, 4), speculative=True)
    assert _blocked("kda", kda, verify) == [], "a verify step runs the verify kernels, which replay"


def test_the_runner_drops_every_capture_on_the_mla_fallback():
    mla = object.__new__(FlashInferMLAManager)
    mla._device = CPU
    mla._kv_config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
    assert not mla.uses_kernel
    decode = AttentionStep(segments=_rows(1, 1))
    assert _blocked("attn", mla, decode) == ["attn"], "the fallback plans from host lists, which a replay never reads"
