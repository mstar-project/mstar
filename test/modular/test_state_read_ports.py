"""Pi0.5, V-JEPA 2 and Canary plan on the contract what they planned by hand, field for field.

Their outputs compare byte for byte, so a mask that builds another flag, or a context planned under another label, moves
them with no error. The hand-written specs and steps below are the reference, copied from the models as they stood
before the port, so the adapter can't pass by agreeing with itself. Each reference states where it differs from what
those models wrote, and why.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from test_state_ports import (
    _assert_equal_per_key,
    _assert_steps_equal,
    _next_step,
    _published,
    _QwenAttn,
    _QwenKV,
    _StubResource,
)
from torch import nn

from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig, Span
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import (
    StateSetup,
    _resolve,
    build_handle,
    kv_layouts,
    node_handles,
    node_resources,
    sampling_configs,
    state_steps,
)
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import LabelLayout, serving_resources
from mstar.model.canary.canary_model import CanaryModel
from mstar.model.canary.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    SAMPLER,
    CanaryModelConfig,
)
from mstar.model.canary.decoder import CanaryDecoderModel
from mstar.model.canary.submodules import CanaryDecoderSubmodule
from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.components.request_kwargs import sampler_capabilities, sampling_config
from mstar.model.pi05.components.action_expert import Pi05ActionExpert
from mstar.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mstar.model.pi05.config import CHUNK_WALK, LLM_ATTN, LLM_KV, LLM_NODES, Pi05Config
from mstar.model.pi05.pi05_model import Pi05Model
from mstar.model.pi05.submodules import Pi05ActionSubmodule, Pi05LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs
from mstar.model.utils import kv_pages
from mstar.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC
from mstar.model.vjepa2.config import ATTN as VJEPA2_ATTN
from mstar.model.vjepa2.config import KV_CACHE as VJEPA2_KV
from mstar.model.vjepa2.config import VJepa2ACPredictorConfig
from mstar.model.vjepa2.submodules import BLOCK_LOOP_REGION, VJepa2ACRolloutPredictorSubmodule
from mstar.model.vjepa2.vjepa2_model import VJepa2ACModel, VJepa2Model

ROPE = "rope"
# Pools a deployment could set, each key one a spec's YAML takes.
SYNTHETIC_POOL = {"page_size": 64, "max_num_pages": 33, "cpu_offload_pages": 8}


def _pi05(**overrides) -> Pi05Model:
    return Pi05Model("lerobot/pi05_base", skip_weight_loading=True, **overrides)


def _ac() -> VJepa2ACModel:
    return VJepa2ACModel("unused", skip_weight_loading=True)


def _masked() -> VJepa2Model:
    return VJepa2Model("unused", skip_weight_loading=True, predictor_kind="masked")


def _canary() -> CanaryModel:
    model = object.__new__(CanaryModel)
    model.config = CanaryModelConfig(encoder=FastConformerConfig(), special_ids=defaultdict(int))
    return model


def _pi05_reference(config: Pi05Config) -> list:
    return [
        KVSpec(
            resource_key=LLM_KV, nodes=set(LLM_NODES),
            config=PagedKVConfig(
                num_layers=config.num_layers,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                max_seq_len=config.max_position_embeddings,
                num_qo_heads=config.num_qo_heads,
            ),
        ),
        AttentionSpec(resource_key=LLM_ATTN, nodes=set(LLM_NODES), config=AttentionConfig(kv_cache=LLM_KV)),
        # The model wrote only kv_cache. The three factors are what its layer passed on every call, which picks the
        # Llama 3.1 kernel, and on the contract a layer passes nothing per call.
        PositionSpec(
            resource_key=ROPE, nodes=set(LLM_NODES),
            config=PositionConfig(kv_cache=LLM_KV, low_freq_factor=1.0, high_freq_factor=1.0, old_context_len=8192),
        ),
    ]


def _ac_reference(model: VJepa2ACModel) -> list:
    ac = model.config.ac_predictor
    return [
        KVSpec(
            resource_key=VJEPA2_KV, nodes={"rollout_predictor"},
            config=PagedKVConfig(
                num_layers=ac.depth, num_kv_heads=ac.num_heads, head_dim=ac.predictor_embed_dim // ac.num_heads,
                max_seq_len=16384, num_qo_heads=ac.num_heads,
            ),
        ),
        AttentionSpec(
            resource_key=VJEPA2_ATTN, nodes={"rollout_predictor"}, config=AttentionConfig(kv_cache=VJEPA2_KV),
        ),
    ]


def _canary_reference(model: CanaryModel) -> list:
    config = model.config

    def pool(max_seq_len: int) -> PagedKVConfig:
        # The model set page_size=128, the engine's default page: see the test on its page and arena.
        return PagedKVConfig(
            num_layers=config.num_layers, num_kv_heads=config.num_heads, head_dim=config.head_dim,
            max_seq_len=max_seq_len, num_qo_heads=config.num_heads,
            max_num_pages=kv_pages(max_seq_len, 128, model.MAX_CONCURRENT_REQUESTS),
        )

    return [
        KVSpec(resource_key=KV_CACHE, nodes={"decoder"}, config=pool(config.max_position_embeddings)),
        AttentionSpec(resource_key=ATTN, nodes={"decoder"}, config=AttentionConfig(kv_cache=KV_CACHE)),
        KVSpec(resource_key=CROSS_KV_CACHE, nodes={"decoder"}, config=pool(model._max_encoder_frames())),
        CrossAttentionSpec(
            resource_key=CROSS_ATTN, nodes={"decoder"},
            config=CrossAttentionConfig(kv_cache=CROSS_KV_CACHE, query_kv_cache=KV_CACHE, context_label=CONTEXT_LABEL),
        ),
        # Keyed `positions` by the model. `position_key` names an array's counter, and gives `rope` to this one.
        PositionSpec(resource_key=ROPE, nodes={"decoder"}, config=PositionConfig(kv_cache=KV_CACHE)),
        SamplerSpec(
            resource_key=SAMPLER, nodes={"decoder"}, vocab_size=config.vocab_size,
            **sampler_capabilities(model.sampling_defaults),
        ),
    ]


def test_the_published_configs_are_found():
    assert all(_published(model) for model in ("Pi05Model", "VJepa2Model", "VJepa2ACModel", "CanaryModel")), (
        "an empty list would skip their tests"
    )


@pytest.mark.parametrize("model, reference", [
    (_pi05, lambda m: _pi05_reference(m.config)),
    (_ac, _ac_reference),
    (_masked, lambda m: []),
    (_canary, _canary_reference),
], ids=["pi05", "vjepa2_ac", "vjepa2_masked", "canary"])
def test_specs_equal_the_hand_written_ones(model, reference):
    built = model()
    _assert_equal_per_key(node_resources(built), reference(built))


@pytest.mark.parametrize("path", _published("Pi05Model"), ids=lambda path: path.stem)
def test_pi05_specs_equal_the_hand_written_ones_on_every_published_config(path):
    deployment = yaml.safe_load(path.read_text())
    model = _pi05()
    reference = _pi05_reference(model.config)
    apply_yaml_overrides(reference, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), reference)


@pytest.mark.parametrize("path", _published("VJepa2ACModel"), ids=lambda path: path.stem)
def test_vjepa2_ac_specs_equal_the_hand_written_ones_on_every_published_config(path):
    deployment = yaml.safe_load(path.read_text())
    model = _ac()
    reference = _ac_reference(model)
    apply_yaml_overrides(reference, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), reference)


@pytest.mark.parametrize("path", _published("VJepa2Model"), ids=lambda path: path.stem)
def test_the_masked_vjepa2_declares_nothing_on_every_published_config(path):
    assert serving_resources(_masked(), yaml.safe_load(path.read_text())) == []


@pytest.mark.parametrize("path", _published("CanaryModel"), ids=lambda path: path.stem)
def test_canary_specs_equal_the_hand_written_ones_on_every_published_config(path):
    deployment = yaml.safe_load(path.read_text())
    model = _canary()
    reference = _canary_reference(model)
    apply_yaml_overrides(reference, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), reference)


@pytest.mark.parametrize("model, reference, pools", [
    (_pi05, lambda m: _pi05_reference(m.config), (LLM_KV,)),
    (_ac, _ac_reference, (VJEPA2_KV,)),
    (_canary, _canary_reference, (KV_CACHE, CROSS_KV_CACHE)),
], ids=["pi05", "vjepa2_ac", "canary"])
def test_specs_equal_the_hand_written_ones_after_a_deployments_overrides(model, reference, pools):
    built = model()
    deployment = {"resources": {pool: dict(SYNTHETIC_POOL) for pool in pools}}
    expected = reference(built)
    apply_yaml_overrides(expected, deployment)
    specs = serving_resources(built, deployment)
    _assert_equal_per_key(specs, expected)
    by_key = {spec.resource_key: spec for spec in specs}
    assert all(by_key[pool].config.page_size == 64 for pool in pools), "the YAML should still reach the pool"


def test_canarys_pools_leave_their_page_size_unset_with_the_page_and_arena_it_set():
    """With no window a page of None and of 128 are one page, so dropping the argument moves nothing."""
    model = _canary()
    for spec in node_resources(model):
        if isinstance(spec, KVSpec):
            written = replace(spec.config, page_size=128)
            assert spec.config.tokens_per_page == written.tokens_per_page == 128
            assert spec.config.num_units == written.num_units, f"{spec.resource_key} has another arena"


@pytest.mark.parametrize("model", [_pi05, _ac, _masked, _canary], ids=["pi05", "vjepa2_ac", "vjepa2_masked", "canary"])
def test_a_declaration_has_no_load_problems(model):
    assert state_declaration_problems(model()) == []


def _handles(model, node: str) -> dict:
    return node_handles(model, node, {spec.resource_key: _StubResource() for spec in node_resources(model)})


def _segments(spans) -> list[Segment]:
    return [Segment(request_id=rid, label="main", span=span) for rid, span in zip("abcd", spans, strict=False)]


@pytest.mark.parametrize("node, spans, keep, advance", [
    ("LLM", (910, 654), True, None),
    ("action_expert", (50, 50), False, (0, 0)),
])
def test_pi05_steps_equal_the_hand_written_ones(node, spans, keep, advance):
    model = _pi05()
    submodule = (
        Pi05LLMSubmodule(None, None, model.config) if node == "LLM"
        else Pi05ActionSubmodule(None, None, None, None, model.config)
    )
    inputs = [ARNodeInputs(input_seq_len=span) for span in spans]
    step = state_steps(submodule.declare_step(CHUNK_WALK, ["a", "b"], inputs), _handles(model, node))
    _assert_steps_equal(step, SubmoduleStep(segments=_segments(spans), steps={
        LLM_KV: KVStep(commit=keep),
        LLM_ATTN: AttentionStep(causal=False),
        ROPE: PositionStep(advance=advance),
    }))


def _ac_frame(model) -> int:
    return 2 + model.config.grid_size ** 2


def _ac_reference_step(rows: int, frame: int) -> SubmoduleStep:
    return SubmoduleStep(segments=_segments((frame,) * rows), steps={
        VJEPA2_KV: KVStep(),
        VJEPA2_ATTN: AttentionStep(causal=False),
    })


@pytest.mark.parametrize("walk", [VJepa2Model.PREFILL_VIDEO_ROLLOUT, VJepa2Model.PREFILL_VIDEO_ROLLOUT_STREAMING])
def test_vjepa2_acs_eager_rollout_step_equals_the_hand_written_one(walk):
    model = _ac()
    frame = _ac_frame(model)
    node = VJepa2ACRolloutPredictorSubmodule(None, model.config)
    inputs = [ARNodeInputs(input_seq_len=frame) for _ in "abc"]
    step = state_steps(node.declare_step(walk, list("abc"), inputs), _handles(model, "rollout_predictor"))
    _assert_steps_equal(step, _ac_reference_step(3, frame))


def test_vjepa2_acs_region_step_equals_the_hand_written_one():
    model = _ac()
    frame = _ac_frame(model)
    node = VJepa2ACRolloutPredictorSubmodule(None, model.config)
    region = node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.bfloat16)[BLOCK_LOOP_REGION]
    step = state_steps(region.declare_step(list("abcd"), [frame] * 4), _handles(model, "rollout_predictor"))
    _assert_steps_equal(step, _ac_reference_step(4, frame))


def test_vjepa2_acs_step_declares_nothing_when_the_region_holds_its_lease():
    model = _ac()
    node = VJepa2ACRolloutPredictorSubmodule(None, model.config)
    inputs = [ARNodeInputs(input_seq_len=_ac_frame(model))]
    walk = VJepa2Model.PREFILL_VIDEO_ROLLOUT
    leased = node.declare_step(walk, ["a"], inputs, piecewise_leases={BLOCK_LOOP_REGION: 1})
    assert state_steps(leased, _handles(model, "rollout_predictor")) is None


@pytest.mark.parametrize("walk, span, clips", [("prefill", 10, (13, 7)), ("decode", 1, (0, 0))])
def test_canary_steps_equal_the_hand_written_ones(walk, span, clips):
    model = _canary()
    inputs = [
        ARNodeInputs(
            input_ids=torch.zeros(span, dtype=torch.long), input_seq_len=span,
            tensor_inputs={"encoder_states": torch.zeros(clip, 4)} if clip else {},
        )
        for clip in clips
    ]
    node = CanaryDecoderSubmodule(None, model.config)
    step = state_steps(node.declare_step(walk, ["a", "b"], inputs), _handles(model, "decoder"))
    context = tuple(
        Segment(request_id=rid, label=CONTEXT_LABEL, span=clip) for rid, clip in zip("ab", clips, strict=True)
    )
    prompts = {rid: inp.input_ids for rid, inp in zip("ab", inputs, strict=True)} if walk == "prefill" else {}
    _assert_steps_equal(step, SubmoduleStep(segments=_segments((span, span)), steps={
        KV_CACHE: KVStep(),
        ATTN: AttentionStep(causal=True),
        CROSS_KV_CACHE: KVStep(segments=context),
        CROSS_ATTN: AttentionStep(segments=context, causal=False),
        SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prompts),
        ROPE: PositionStep(),
    }))


@pytest.mark.parametrize("cameras, horizon", [(1, 50), (2, 50), (3, 50), (2, 15)])
def test_pi05_layout_converts_to_the_hand_written_one(cameras, horizon):
    model = _pi05(action_horizon=horizon)
    model.tokenizer = SimpleNamespace(encode_prompt=lambda prompt: torch.tensor([1, 2, 3]))
    out = model.process_prompt(
        "pick", ["image", "text"], ["action"], tensors={"image_inputs": [torch.zeros(3, 4, 4)] * cameras},
    )
    rows = cameras * 256
    assert kv_layouts(model.declare_state(), out.metadata["state_layout"], {}) == {LLM_KV: {"main": LabelLayout(
        spans=(Span("rows", CHUNK_WALK, rows, rows, "img_emb"), Span("ids", CHUNK_WALK, 3, 3, "text_inputs")),
        scratch=horizon,
    )}}


@pytest.mark.parametrize("extrinsics, cond", [(False, 2), (True, 3)])
def test_vjepa2_rollout_layout_converts_to_the_hand_written_one(extrinsics, cond):
    model = _ac()
    model.config.ac_predictor.use_extrinsics = extrinsics
    kwargs = {"actions": [[0.1] * 7] * 5, "states": [[0.2] * 7] * 5}
    if extrinsics:
        kwargs["extrinsics"] = [[0.3] * 6] * 5
    out = model.process_prompt(None, ["video"], ["video"], rollout_horizon=5, **kwargs)
    frame = cond + model.config.grid_size ** 2
    assert kv_layouts(model.declare_state(), out.metadata["state_layout"], {}) == {
        VJEPA2_KV: {"main": LabelLayout(growth=((model.ROLLOUT_LOOP_NAME, frame),))},
    }


def test_vjepa2_layouts_off_the_rollout_convert_to_the_hand_written_ones():
    model = _ac()
    kwargs = {"actions": [[0.1] * 7] * 5, "states": [[0.2] * 7] * 5}
    once = model.process_prompt(None, ["video"], ["video"], **kwargs).metadata["state_layout"]
    assert kv_layouts(model.declare_state(), once, {}) == {VJEPA2_KV: {"main": LabelLayout()}}
    assert _masked().process_prompt(None, ["video"], ["video"]).metadata["state_layout"] == {}


def test_canary_layout_converts_to_the_hand_written_one():
    model = _canary()
    out = model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [torch.zeros(16_000)]})
    frames = model.config.encoder.output_lengths(100)
    assert kv_layouts(model.declare_state(), out.metadata["state_layout"], {}) == {
        KV_CACHE: {"main": LabelLayout(
            spans=(Span("rows", "prefill", 10, 10, "text_inputs"),), growth=(("decode_loop", 1),),
        )},
        CROSS_KV_CACHE: {CONTEXT_LABEL: LabelLayout(
            spans=(Span("rows", "prefill", frames, frames, "encoder_states"),),
        )},
    }


@pytest.mark.parametrize("kwargs", [{}, {"temperature": 0.5, "top_p": 0.9, "ignore_eos": True}])
def test_canarys_sample_settings_convert_to_the_hand_written_ones(kwargs):
    model = _canary()
    settings = model.get_request_resource_configs([], [], kwargs)
    expected = sampling_config(model.sampling_defaults, kwargs)
    assert sampling_configs(settings, model.declare_state())[SAMPLER] == expected


def test_the_rollout_node_opts_out_of_torch_compile():
    """Compiling it would trace the eager predictor through a graph break per state call, for nothing but the glue
    around `_rollout_step`."""
    assert VJepa2ACRolloutPredictorSubmodule.disable_torch_compile, "the engine skips only a node that opts out"


def test_the_cached_path_refuses_a_node_that_declares_no_array():
    """One predictor serves three nodes, and only one of them binds a handle."""
    config = VJepa2ACPredictorConfig(
        embed_dim=32, predictor_embed_dim=32, depth=1, num_heads=4, num_frames=4, tubelet_size=2, img_size=(32, 32),
        patch_size=16,
    )
    attention = VisionTransformerPredictorAC(config).predictor_blocks[0].attn
    with pytest.raises(ValueError, match="cached=True needs the array 'kv'"):
        attention.forward_cached(torch.zeros(1, 6, 32), None, 2)


# A compiled capture must not guard on a layer: a frame per layer hits the recompile limit, then runs uncompiled.


TINY_PI05 = Pi05Config(
    num_layers=2, num_qo_heads=4, num_kv_heads=1, head_dim=8, hidden_size=32, pali_intermediate_size=64,
    action_hidden_size=16, action_intermediate_size=32,
)


class _Identity(nn.Identity):
    """Off CUDA FlashInfer has no RMSNorm, and Triton no adaRMS, so a norm only returns what the caller reads."""

    def __init__(self, gated: bool = False):
        super().__init__()
        self.gated = gated

    def forward(self, x, cond=None):
        return (x, None) if self.gated else x


class _Positions:
    """Rotates nothing, for a stack that calls `rope`."""

    def apply_qk(self, q, k, label=None, rope_dtype=None):
        return q, k


def _pi05_expert(expert_class, num_layers: int = TINY_PI05.num_layers):
    config = replace(TINY_PI05, num_layers=num_layers)
    array = _pi05(**vars(config)).declare_state()[0]
    with torch.no_grad():
        expert = expert_class(config)
    gated = expert_class is Pi05ActionExpert
    expert.norm = _Identity(gated)
    for layer in expert.layers:
        layer.input_layernorm, layer.post_attention_layernorm = _Identity(gated), _Identity(gated)
    generator = torch.Generator().manual_seed(0)
    for parameter in expert.parameters():
        parameter.data.normal_(0, 0.02, generator=generator)
    handle = build_handle(
        array, {LLM_KV: _QwenKV(array), LLM_ATTN: _QwenAttn(), ROPE: _Positions()}, _resolve([array], StateSetup()),
    )
    expert.eval()
    for module in expert.modules():
        if hasattr(module, "bind_state"):
            module.bind_state({LLM_KV: handle})
    _next_step(handle)
    return expert, handle


def _pi05_call(expert) -> tuple:
    if isinstance(expert, Pi05PaliGemmaExpert):
        return (torch.randn(3, TINY_PI05.hidden_size),)
    # the norms are identities, so a modulation's values go unread
    h = TINY_PI05.action_hidden_size
    return torch.randn(3, h), [torch.zeros(2, 3 * h)] * len(expert.layers), torch.zeros(1, 3 * h)


def _canary_stack(num_layers: int = 2):
    config = CanaryModelConfig(
        encoder=FastConformerConfig(), hidden_size=32, num_layers=num_layers, num_heads=4, intermediate_size=64,
        vocab_size=50, max_position_embeddings=64, eos_token_id=3,
    )
    model = _canary()
    model.config = config
    declared = {d.name: d for d in model.declare_state()}
    self_array, cross_array = declared[KV_CACHE], declared[CROSS_KV_CACHE]
    decoder = CanaryDecoderModel(config).eval()
    generator = torch.Generator().manual_seed(0)
    for parameter in decoder.parameters():
        parameter.data.normal_(0, 0.02, generator=generator)
    handles = node_handles(SimpleNamespace(declare_state=lambda: [self_array, cross_array]), "decoder", {
        KV_CACHE: _QwenKV(self_array), ATTN: _QwenAttn(), ROPE: None,
        CROSS_KV_CACHE: _QwenKV(cross_array), CROSS_ATTN: _QwenAttn(),
    })
    kv, context = handles[KV_CACHE], handles[CROSS_KV_CACHE]
    for module in decoder.modules():
        if hasattr(module, "bind_state"):
            module.bind_state({KV_CACHE: kv, CROSS_KV_CACHE: context})
    _next_step(kv, context)
    return decoder, kv, context


def _compiled_run(module, args, *handles, **kwargs) -> tuple[int, list[str]]:
    """Run ``module`` compiled as a capture is, then again under the capture's stance, which raises on any new compile.
    Returns how many frames it compiled and its failed guards."""
    torch._dynamo.reset()
    frames, guards = [], []

    def backend(graph_module, inputs):
        frames.append(graph_module)
        return graph_module.forward

    optimize = torch._dynamo.optimize(backend, dynamic=False, guard_fail_fn=lambda f: guards.append(str(f.reason)))
    compiled = optimize(module)
    compiled(*args, **kwargs)
    first_call = len(frames)
    _next_step(*handles)
    with torch.compiler.set_stance("fail_on_recompile"):
        compiled(*args, **kwargs)
    assert len(frames) == first_call, "a capture's stance raises on any new compile"
    return first_call, guards


@torch.no_grad()
@pytest.mark.parametrize("expert_class", [Pi05PaliGemmaExpert, Pi05ActionExpert])
def test_a_pi05_expert_compiled_as_a_capture_compiles_no_frame_per_layer(expert_class):
    expert, handle = _pi05_expert(expert_class)
    _, guards = _compiled_run(expert, _pi05_call(expert), handle)
    assert not [guard for guard in guards if "layer" in guard], "one frame should serve every layer"


@torch.no_grad()
@pytest.mark.parametrize("expert_class", [Pi05PaliGemmaExpert, Pi05ActionExpert])
def test_a_longer_pi05_expert_compiles_no_more_frames(expert_class):
    frames = []
    for num_layers in (2, 4):
        expert, handle = _pi05_expert(expert_class, num_layers)
        frames.append(_compiled_run(expert, _pi05_call(expert), handle)[0])
    assert frames[0] == frames[1] > 1


@torch.no_grad()
def test_pi05s_action_expert_runs_two_euler_steps_in_one_step():
    """The action step runs every Euler step, so each layer adds its scratch tokens again after its last read."""
    expert, handle = _pi05_expert(Pi05ActionExpert)
    step = SubmoduleStep(segments=[Segment(1, "main", 3)], state={LLM_KV: ArrayStep(keep=False)})
    step.set_ctx(StepContext((1,), CHUNK_WALK, 0, False))
    handle.end_step(step)
    handle.begin_step(step)
    for _ in range(2):
        expert(*_pi05_call(expert))
    handle.check_step(step)


@torch.no_grad()
def test_canarys_decoder_compiled_as_a_capture_compiles_no_frame_per_layer():
    """`disable_torch_compile` stops only the eager forward's compile, so the decode capture does compile."""
    decoder, kv, context = _canary_stack()
    embeds = torch.randn(3, 32)
    _, guards = _compiled_run(decoder, (embeds,), kv, context, label="main", position_ids=torch.arange(3))
    assert not [guard for guard in guards if "layer" in guard], "one frame should serve every layer"


@torch.no_grad()
def test_a_longer_canary_decoder_compiles_no_more_frames():
    frames = []
    for num_layers in (2, 4):
        decoder, kv, context = _canary_stack(num_layers)
        run = _compiled_run(decoder, (torch.randn(3, 32),), kv, context, label="main", position_ids=torch.arange(3))
        frames.append(run[0])
    assert frames[0] == frames[1] > 1


@torch.no_grad()
@pytest.mark.parametrize("expert_class", [Pi05PaliGemmaExpert, Pi05ActionExpert])
def test_each_pi05_layer_reads_the_array_by_its_id_in_the_stack(expert_class):
    """Both experts use one array's layer ids, so the action expert's layer i reads what PaliGemma's layer i kept."""
    expert, handle = _pi05_expert(expert_class)
    seen = []
    handle._record_add = lambda layer, keep: seen.append(("add", layer))
    handle._record_read = lambda layer: seen.append(("read", layer))
    expert(*_pi05_call(expert))
    assert seen == [(call, layer) for layer in range(TINY_PI05.num_layers) for call in ("add", "read")]


@torch.no_grad()
def test_each_canary_layer_names_itself_on_both_arrays():
    decoder, kv, context = _canary_stack()
    seen = []
    kv._record_read = lambda layer: seen.append(("self", layer))
    context._record_read = lambda layer: seen.append(("cross", layer))
    decoder(torch.randn(3, 32), label="main", position_ids=torch.arange(3))
    assert seen == [(array, layer) for layer in range(2) for array in ("self", "cross")]


class _Recording:
    """Stands in for any resource on Canary's decoder node, and logs each call that touches state."""

    requires_kv_write = True

    def __init__(self, key: str, log: list):
        self.key, self.log = key, log

    def layer_view(self, i):
        return i

    def opens_label(self, rid, label):
        return True

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.log.append(("write", self.key, layer_idx, label, k.shape[0]))

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        self.log.append(("read", self.key, kv_cache_layer, label))
        return q

    def pos_ids(self, label):
        self.log.append(("positions", label))
        return torch.arange(64)

    def select_last_hidden(self, x, label="main"):
        self.log.append(("last_token", self.key, label))
        return x[-2:]

    def sample(self, rows, logits):
        self.log.append(("sample", tuple(rows)))
        return logits.argmax(-1)


def _canary_node():
    config = CanaryModelConfig(
        encoder=FastConformerConfig(), hidden_size=32, num_layers=2, num_heads=4, intermediate_size=64, vocab_size=50,
        max_position_embeddings=64, eos_token_id=3,
    )
    model = _canary()
    model.config = config
    log: list = []
    resources = {spec.resource_key: _Recording(spec.resource_key, log) for spec in node_resources(model)}
    handles = node_handles(model, "decoder", resources)
    node = CanaryDecoderSubmodule(CanaryDecoderModel(config).eval(), config)
    for parameter in node.parameters():
        parameter.data.normal_(0, 0.02)
    node.bind_node_state(handles)
    return node, handles, log


def _canary_step(node, handles, walk: str, inputs: list[ARNodeInputs]) -> torch.Tensor:
    step = state_steps(node.declare_step(walk, ["a", "b"], inputs), handles)
    step.set_ctx(StepContext(request_ids=("a", "b"), graph_walk=walk, slot=0, capture=False))
    step.begin_calls()
    preprocessed = node.preprocess(walk, None, inputs)
    output = node.forward_batched(walk, None, **preprocessed)
    step.check_calls()
    step.end_calls()
    return output.row_outputs["new_token"]


def _canary_layers() -> list[tuple]:
    return [
        call for layer in range(2)
        for call in (
            ("write", KV_CACHE, layer, "main", 2), ("read", ATTN, layer, "main"), ("read", CROSS_ATTN, layer, "main"),
        )
    ]


@torch.no_grad()
def test_canarys_prefill_adds_the_context_before_any_layer_reads_it():
    """A forward that skipped the add would read pages nothing wrote, with no error from any kernel."""
    node, handles, log = _canary_node()
    inputs = [
        ARNodeInputs(input_ids=torch.tensor([1]), input_seq_len=1, tensor_inputs={"encoder_states": torch.randn(n, 32)})
        for n in (13, 7)
    ]
    assert _canary_step(node, handles, "prefill", inputs).shape == (2,)
    assert log == [
        ("positions", "main"),
        ("write", CROSS_KV_CACHE, 0, CONTEXT_LABEL, 20), ("write", CROSS_KV_CACHE, 1, CONTEXT_LABEL, 20),
        *_canary_layers(),
        ("last_token", ATTN, "main"),
        ("sample", ("a", "b")),
    ], "a prefill adds each layer's context, runs the layers, then samples each row's last token"


@torch.no_grad()
def test_canarys_decode_reads_the_kept_context_and_adds_none():
    node, handles, log = _canary_node()
    inputs = [ARNodeInputs(input_ids=torch.tensor([1]), input_seq_len=1) for _ in "ab"]
    assert _canary_step(node, handles, "decode", inputs).shape == (2,)
    assert log == [("positions", "main"), *_canary_layers(), ("sample", ("a", "b"))]
