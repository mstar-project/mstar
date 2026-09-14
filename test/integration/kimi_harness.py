"""Shared setup for the Kimi integration tests, on the v1 resource framework.

These tests used to build a ``BatchedCacheManager`` by hand and call
``plan_attention`` / ``run_attention`` on it. Resources are now declared by the
model and driven by steps, so the shared work is: build the resource set a spec
list names, and drive one step through admit -> plan -> forward -> commit.

Everything here is GPU-only by the nature of what it sets up; a caller adds its
own ``pytestmark`` skip.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    KVConfig,
    KVLayout,
    KVSpec,
    KVStep,
    PositionConfig,
    PositionSpec,
    PositionStep,
    SamplerStep,
    Segment,
    StepContext,
)
from mstar.engine.resources.base import EngineResourceInfo, Resource, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.model.kimi_k2_7.config import ATTN, KV_CACHE, ROPE, SAMPLER

DEVICE = torch.device("cuda")

# resource key -> the step class that drives it
_STEP_FOR_KEY = {
    KV_CACHE: KVStep,
    ATTN: AttentionStep,
    ROPE: PositionStep,
    SAMPLER: SamplerStep,
}


def transfer_info(entity_id: str) -> TransferEngineInfo:
    return TransferEngineInfo(
        my_entity_id=entity_id,
        my_session_id="kimi_session",
        transfer_engine=LocalTransferEngine("localhost"),
    )


def build_resources(
    specs: list,
    *,
    device: torch.device = DEVICE,
    kv_dtype: torch.dtype = torch.bfloat16,
    entity_id: str = "kimi_integration_test",
) -> dict[str, Resource]:
    """One resource per spec, with each spec's declared dependencies resolved."""
    by_key = resolve_spec_dependencies(specs)
    resources: dict[str, Resource] = {}
    for spec in specs:
        info = EngineResourceInfo(
            device=device,
            kv_dtype=kv_dtype,
            transfer_engine_info=transfer_info(entity_id),
            dependencies={key: by_key[key] for key in spec.depends_on()},
        )
        resources[spec.resource_key] = build_resource(spec, info)
    return resources


def paged_specs(
    *,
    num_layers: int = 2,
    num_kv_heads: int,
    head_dim: int,
    num_qo_heads: int | None = None,
    page_size: int = 128,
    max_num_pages: int = 8,
) -> list:
    """An NHD cache under the paged backend — what naive MLA attends over."""
    return [
        KVSpec(
            resource_key=KV_CACHE,
            nodes={"LLM"},
            config=KVConfig(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seq_len=page_size * max_num_pages,
                max_num_pages=max_num_pages,
                page_size=page_size,
                num_qo_heads=num_qo_heads or num_kv_heads,
            ),
        ),
        AttentionSpec(
            resource_key=ATTN, nodes={"LLM"},
            config=AttentionConfig(kv_cache=KV_CACHE),
        ),
    ]


def latent_specs(
    *,
    latent_width: int,
    softmax_scale: float,
    mla_ckv_dim: int | None,
    num_qo_heads: int = 1,
    num_layers: int = 2,
    page_size: int = 4,
    max_num_pages: int = 64,
) -> list:
    """An MLA-layout cache under the absorbed backend.

    ``mla_ckv_dim=None`` is how a caller forces the SDPA fallback on a box whose
    dims the kernel would otherwise serve.
    """
    return [
        KVSpec(
            resource_key=KV_CACHE,
            nodes={"LLM"},
            config=KVConfig(
                num_layers=num_layers,
                num_kv_heads=1,
                head_dim=latent_width,
                max_seq_len=page_size * max_num_pages,
                max_num_pages=max_num_pages,
                page_size=page_size,
                num_qo_heads=num_qo_heads,
                layout=KVLayout.MLA,
            ),
        ),
        AttentionSpec(
            resource_key=ATTN, nodes={"LLM"},
            config=AttentionConfig(
                kv_cache=KV_CACHE,
                backend=AttnBackend.MLA,
                softmax_scale=softmax_scale,
                mla_ckv_dim=mla_ckv_dim,
            ),
        ),
    ]


def rope_spec(cfg) -> PositionSpec:
    """Position counters for a Kimi cache; the yarn rotary itself is the
    layer's, so only the counters matter here."""
    return PositionSpec(
        resource_key=ROPE,
        nodes={"LLM"},
        config=PositionConfig(
            kv_cache=KV_CACHE,
            rotary_dim=cfg.qk_rope_head_dim,
            rope_theta=cfg.rope_theta,
        ),
    )


def ingest(resources: dict[str, Resource], *rids: str) -> None:
    for rid in rids:
        for resource in resources.values():
            resource.ingest_request(rid, None)


def remove(resources: dict[str, Resource], *rids: str) -> None:
    for rid in rids:
        for resource in resources.values():
            resource.remove_request(rid)


def cleanup(resources: dict[str, Resource]) -> None:
    for resource in resources.values():
        resource.cleanup()


@contextmanager
def step(
    resources: dict[str, Resource],
    spans: dict[str, int],
    *,
    graph_walk: str = "prefill",
    causal: bool = True,
    commit: bool = True,
    label: str = "main",
):
    """Admit and plan one step over ``{rid: span}``, then commit on exit.

    The steps are declared here rather than by a submodule so a resource-level
    test can drive the resources it built directly. Ordering follows each
    resource's declared dependencies, which for Kimi means KV first.
    """
    rids = list(spans)
    segments = tuple(Segment(rid, label, span) for rid, span in spans.items())
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk=graph_walk, slot=0, capture=False,
    )
    steps = {}
    for key in resources:
        cls = _STEP_FOR_KEY.get(key)
        if cls is KVStep:
            steps[key] = cls(segments=segments, commit=commit)
        elif cls is AttentionStep:
            steps[key] = cls(segments=segments, causal=causal)
        elif cls is not None:
            steps[key] = cls(segments=segments)

    for key in _ordered(resources, steps):
        outcome = resources[key].admit(steps[key], ctx)
        assert outcome.ok, f"{key} refused the step: {outcome.reason}"
    for key in _ordered(resources, steps):
        result = resources[key].plan(steps[key], ctx)
        if result is not None:
            ctx.plan_results[key] = result

    yield ctx

    for key in _ordered(resources, steps):
        resources[key].commit(steps[key], ctx)


def _ordered(resources: dict[str, Resource], steps: dict) -> list[str]:
    """Step keys with each resource after the ones it depends on."""
    from mstar.engine.resources import topo_sort

    return [key for key in topo_sort(resources) if key in steps]


# ── submodule level ─────────────────────────────────────────────────────

def load_submodule(
    model,
    *,
    dtype: torch.dtype = torch.bfloat16,
    max_num_pages: int = 8,
    page_size: int = 128,
):
    """The model's LLM submodule, with its declared resources built and bound.

    The cache is shrunk from what the model declares: a reduced config still
    asks for 2048 pages, which is gigabytes of device memory for a test.
    """
    specs = model.get_node_resources()
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config.max_num_pages = max_num_pages
            spec.config.page_size = page_size
            spec.config.max_seq_len = min(
                spec.config.max_seq_len, max_num_pages * page_size
            )
    resources = build_resources(specs, kv_dtype=dtype)
    submodule = model.get_submodule("LLM", device="cuda", autocast_dtype=dtype)
    submodule.bind_node_resources(resources)
    return submodule, resources


def open_request(model, resources, rid: str, *, greedy: bool = False) -> None:
    """Ingest ``rid`` with the model's own per-request configs."""
    overrides = model.get_request_resource_configs({})
    if greedy and SAMPLER in overrides:
        sampling = overrides[SAMPLER]
        # the kernel emits a one-hot at the argmax when temperature is 0
        sampling.temperature = 0.0
        sampling.top_k, sampling.top_p = 0, 1.0
        sampling.repetition_penalty = 1.0
    for key, resource in resources.items():
        resource.ingest_request(rid, overrides.get(key))


@contextmanager
def submodule_step(submodule, resources, graph_walk: str, tokens: dict[str, torch.Tensor]):
    """Drive one step the way the engine does, off the submodule's own
    declaration, and yield ``(engine_inputs, static_inputs)``."""
    from mstar.engine.cuda_graph_runner import dummy_metadata
    from mstar.model.submodule_base import ModelInputsFromEngine

    rids = list(tokens)
    inputs = [
        submodule.prepare_inputs(
            graph_walk=graph_walk, fwd_info=None,
            inputs={"text_inputs": [tokens[rid]]},
        )
        for rid in rids
    ]
    declared = submodule.declare_step(
        graph_walk=graph_walk, request_ids=rids, inputs=inputs,
    )
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk=graph_walk, slot=0, capture=False,
    )
    declared.set_ctx(ctx)

    from mstar.engine.resources import StepRunner

    runner = StepRunner(resources)
    outcome = runner.admit(declared)
    assert outcome.ok, f"admit refused the step: {outcome.reason}"
    runner.plan(declared)

    engine_inputs = ModelInputsFromEngine(
        request_ids=rids,
        per_request_info=dummy_metadata(rids, graph_walk),
        resources=resources,
    )
    engine_inputs.step = declared
    static_inputs = submodule.preprocess(
        graph_walk=graph_walk, engine_inputs=engine_inputs, inputs=inputs,
    )
    yield engine_inputs, static_inputs
    runner.commit(declared)


def forward_step(submodule, resources, graph_walk: str, tokens: dict[str, torch.Tensor]):
    """One step's ``forward_batched`` output, by request id."""
    with submodule_step(submodule, resources, graph_walk, tokens) as (ei, static):
        with torch.no_grad():
            out = submodule.forward_batched(
                graph_walk=graph_walk, engine_inputs=ei, **static
            )
    torch.cuda.synchronize()
    return out


def logits_step(submodule, resources, graph_walk: str, tokens: dict[str, torch.Tensor]):
    """The same forward, stopping at the logits.

    ``forward_batched`` returns sampled tokens, so a test comparing against a
    reference forward takes the logits here instead.
    """
    attn = resources[ATTN]
    with submodule_step(submodule, resources, graph_walk, tokens) as (_ei, static):
        with torch.no_grad():
            hidden = submodule.language_model.model(
                static["input_ids"], label="main"
            )
            if graph_walk == "prefill":
                hidden = attn.select_last_hidden(hidden, label="main")
            logits = submodule.lm_head(hidden)
    torch.cuda.synchronize()
    return logits
