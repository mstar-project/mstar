"""The four DiTs plan the same attention on the contract that they planned by hand.

These models compare their output bytes exactly, so a plan that attends other tokens, or one more resource, moves
their output with no error. The hand-written specs and steps below are the reference, copied from the models as they
stood before the port, so the adapter can't pass by agreeing with itself.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
)
from mstar.engine.resources.lora.config import LoraStep
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Block, Read
from mstar.engine.state.adapters import StateSetup, build_handle, node_handles, node_resources, state_setup, state_steps
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.components.diffusion.attention import dit_entry
from mstar.model.flux2_klein.components.transformer import Flux2JointAttention
from mstar.model.flux2_klein.config import DENOISE_LOOP, JOINT_TOKENS, Flux2KleinConfig
from mstar.model.flux2_klein.config import DIT_ATTN as KLEIN_ATTN
from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
from mstar.model.flux2_klein.submodules import KleinDenoiseSubmodule, KleinShape
from mstar.model.ltx2_5.components.layers import GatedAttention
from mstar.model.ltx2_5.config import (
    AUDIO,
    AUDIO_FOR_VIDEO,
    TEXT_FOR_AUDIO,
    TEXT_FOR_VIDEO,
    VIDEO,
    VIDEO_FOR_AUDIO,
    LTX25Config,
)
from mstar.model.ltx2_5.ltx2_5_model import LTX25Model
from mstar.model.ltx2_5.submodules import LTXDenoiseSubmodule, LTXShape
from mstar.model.submodule_base import NodeInputs, engine_step, lora_rows
from mstar.model.wan22.components.dit import Wan22DiTAttention
from mstar.model.wan22.config import SELF_ATTN, Wan22Config
from mstar.model.wan22.submodules import Wan22DitSubmodule, WanStepKey
from mstar.model.wan22.wan22_model import Wan22Model
from mstar.model.z_image.components.transformer import ZImageAttention
from mstar.model.z_image.config import DIT_ATTN as Z_ATTN
from mstar.model.z_image.config import ZImageConfig
from mstar.model.z_image.submodules import ZImageDenoiseSubmodule, ZShape
from mstar.model.z_image.z_image_model import ZImageModel


def _wan() -> Wan22Model:
    model = Wan22Model.__new__(Wan22Model)
    model.config = Wan22Config()
    return model


def _klein(**kwargs) -> Flux2KleinModel:
    model = Flux2KleinModel(model_path_hf="test/flux2_klein", skip_weight_loading=True, **kwargs)
    model.set_config(Flux2KleinConfig())
    return model


def _z_config() -> ZImageConfig:
    return ZImageConfig()


def _z(**kwargs) -> ZImageModel:
    model = ZImageModel(model_path_hf="test/z_image", skip_weight_loading=True, **kwargs)
    model.set_config(_z_config())
    return model


def _ltx(**kwargs) -> LTX25Model:
    model = LTX25Model(skip_weight_loading=True, **kwargs)
    model.set_config(LTX25Config())
    return model


def _on_sdpa(model) -> StateSetup:
    """SDPA on every resource ``model`` declares, which on a DiT is each attention's."""
    on_sdpa = {spec.resource_key: {"backend": "sdpa"} for spec in node_resources(model)}
    setup, problems = state_setup({"resources": on_sdpa}, model.declare_state())
    assert not problems, problems
    return setup


def _wan_reference() -> RaggedAttentionSpec:
    c = Wan22Config()
    return RaggedAttentionSpec(
        resource_key=SELF_ATTN, nodes={"dit"},
        config=RaggedAttentionConfig(
            num_qo_heads=c.num_attention_heads, num_kv_heads=c.num_attention_heads, head_dim=c.attention_head_dim,
            max_segments_per_request=2, compile_op=True, dtype=torch.bfloat16,
        ),
    )


def _klein_reference() -> RaggedAttentionSpec:
    t = Flux2KleinConfig().transformer
    return RaggedAttentionSpec(
        resource_key=KLEIN_ATTN, nodes={"dit"},
        config=RaggedAttentionConfig(
            num_qo_heads=t.num_attention_heads, num_kv_heads=t.num_attention_heads, head_dim=t.attention_head_dim,
            max_segments_per_request=1, dtype=torch.bfloat16,
        ),
    )


def _z_reference() -> RaggedAttentionSpec:
    t = _z_config().transformer
    return RaggedAttentionSpec(
        resource_key=Z_ATTN, nodes={"dit"},
        config=RaggedAttentionConfig(
            num_qo_heads=t.n_heads, num_kv_heads=t.n_heads, head_dim=t.head_dim, max_segments_per_request=1,
            dtype=torch.bfloat16,
        ),
    )


MODELS = {"wan22": (_wan, _wan_reference), "klein": (_klein, _klein_reference), "z_image": (_z, _z_reference)}


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_one(name):
    build, reference = MODELS[name]
    assert node_resources(build()) == [reference()]


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_one_after_yaml_overrides(name):
    build, reference = MODELS[name]
    overrides = {"resources": {reference().resource_key: {"max_segments_per_request": 7, "flashinfer_backend": "fa2"}}}
    specs, expected = node_resources(build()), [reference()]
    apply_yaml_overrides(specs, overrides)
    apply_yaml_overrides(expected, overrides)
    assert specs == expected


@pytest.mark.parametrize("name", MODELS)
def test_sdpa_builds_no_spec_as_the_hand_written_model_did(name):
    model = MODELS[name][0]()
    assert node_resources(model, setup=_on_sdpa(model)) == []


@pytest.mark.parametrize("name", MODELS)
def test_a_port_writes_no_spec_by_hand_and_has_no_layout_to_convert(name):
    model = MODELS[name][0]()
    assert model.get_node_resources() == [] and model.kv_labels() == {}, "a DiT keeps nothing between steps"


def test_ltx_writes_no_spec_by_hand_and_has_no_layout_to_convert():
    assert _ltx().get_node_resources() == [] and _ltx().kv_labels() == {}


@pytest.mark.parametrize("name", ["klein", "z_image"])
def test_declarations_have_no_problems(name):
    assert state_declaration_problems(MODELS[name][0]()) == []


@pytest.mark.parametrize("build", [_klein, _z])
def test_the_old_knob_is_refused(build):
    with pytest.raises(ValueError, match="attention_backend is a resource's setting"):
        build(attention_backend="sdpa")


def test_the_old_knob_is_refused_on_wan22():
    with pytest.raises(ValueError, match="attention_backend is a resource's setting"):
        Wan22Model("test/wan22", attention_backend="sdpa")


class _StubRagged:
    """Stands in for the built resource a handle binds to."""

    def run(self, q, k=None, v=None, label=None):
        return q


def _inputs(key):
    return NodeInputs(tensor_inputs={}, resource_step_info=key)


def _expanded(model, sub, keys, rids):
    specs = node_resources(model)
    handles = node_handles(model, "dit", {spec.resource_key: _StubRagged() for spec in specs})
    declared = sub.declare_step("gen", rids, [_inputs(key) for key in keys])
    return state_steps(declared, handles)


def _by_label(step) -> dict[str, tuple[Segment, ...]]:
    """A ragged resource groups a step's segments by label before it reads them, and a folded step lists its labels in
    the order the arrays add them, which isn't always the order the model listed them in by hand."""
    labels: dict[str, list[Segment]] = {}
    for segment in step.segments:
        labels.setdefault(segment.label, []).append(segment)
    return {label: tuple(segments) for label, segments in labels.items()}


def _assert_plans_equal(step: SubmoduleStep, reference: SubmoduleStep, key: str) -> None:
    assert step.steps.keys() == reference.steps.keys() and step.cg_key_info == reference.cg_key_info
    assert step.segments == reference.segments, "a LoRA pool's rows are the step's segments"
    assert isinstance(step.steps[key], AttentionStep) and step.steps[key].causal is False
    assert _by_label(step.steps[key]) == _by_label(reference.steps[key])


def _klein_key() -> KleinShape:
    return KleinShape(grid=(4, 4), text_len=8)


@pytest.mark.parametrize("rids", [[0], [0, 1], [0, 1, -1, -2]])
def test_klein_steps_equal_the_hand_written_one(rids):
    """Padding rows carry the bucket key too, so a capture's plan and its replays agree."""
    model = _klein()
    key = _klein_key()
    sub = KleinDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
    segments = [Segment(request_id=rid, label="main", span=key.total_tokens) for rid in rids]
    reference = SubmoduleStep(segments=segments, steps={KLEIN_ATTN: AttentionStep(causal=False)}, cg_key_info=key)
    _assert_plans_equal(_expanded(model, sub, [key] * len(rids), rids), reference, KLEIN_ATTN)


@pytest.mark.parametrize("rids", [[0], [0, 1]])
def test_z_image_steps_equal_the_hand_written_one(rids):
    model = _z()
    key = ZShape(grid=(4, 4), cap_len=32)
    sub = ZImageDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
    segments = [
        Segment(request_id=rid, label=label, span=span) for rid in rids
        for label, span in (("image", key.image_tokens_padded), ("caption", 32), ("main", key.total_tokens))
    ]
    reference = SubmoduleStep(segments=segments, steps={Z_ATTN: AttentionStep(causal=False)}, cg_key_info=key)
    _assert_plans_equal(_expanded(model, sub, [key] * len(rids), rids), reference, Z_ATTN)


@pytest.mark.parametrize("guidance, branches", [(5.0, 2), (1.0, 1)])
def test_wan22_steps_equal_the_hand_written_one(guidance, branches):
    model = _wan()
    key = WanStepKey(frames=3, height=8, width=8, guidance=guidance, i2v=False)
    sub = Wan22DitSubmodule(None, model.config)
    tokens = sub.num_tokens(key)
    segments = [Segment(request_id=0, label="main", span=tokens)] * branches
    reference = SubmoduleStep(segments=segments, steps={SELF_ATTN: AttentionStep(causal=False)}, cg_key_info=key)
    _assert_plans_equal(_expanded(model, sub, [key], [0]), reference, SELF_ATTN)


def test_a_step_on_sdpa_is_none_as_the_hand_written_model_returned():
    model = _klein()
    sub = KleinDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
    handles = node_handles(model, "dit", {}, setup=_on_sdpa(model))
    assert state_steps(sub.declare_step("gen", [0], [_inputs(_klein_key())]), handles) is None


def test_a_lora_pool_gets_its_rows_from_the_steps_segments_as_before():
    model = _klein()
    sub = KleinDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
    key = _klein_key()
    declared = lora_rows(sub.declare_step("gen", [0, 1], [_inputs(key)] * 2), ("lora",))
    assert isinstance(declared.steps["lora"], LoraStep)
    assert list(declared.steps["lora"].segments) == [Segment(rid, "main", key.total_tokens) for rid in (0, 1)]


def test_engine_step_runs_the_expansion_after_the_lora_rows():
    model = _klein()
    sub = SimpleNamespace(
        declare_step=KleinDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
        .declare_step,
        lora_keys=("lora",),
        state_handles=node_handles(model, "dit", {KLEIN_ATTN: _StubRagged()}),
    )
    step = engine_step(sub, "gen", [0], [_inputs(_klein_key())])
    assert isinstance(step.steps["lora"], LoraStep) and isinstance(step.steps[KLEIN_ATTN], AttentionStep)


# One compiled frame per attention, not one per layer


class _BreakingRead:
    """A ragged kernel with no ``compile_op``: its run breaks the compiled graph, and each row attends to itself."""

    requires_kv_write = False

    def run(self, q, k, v, label=None, return_lse=False):
        return self._run(q)

    @torch.compiler.disable
    def _run(self, q):
        return q.clone()


def _wan_attention(layer):
    return Wan22DiTAttention(8, 2, 1e-6, array="tokens", layer=layer), lambda x: ()


def _ltx_attention(layer):
    return GatedAttention(8, 2, 4, CommGroup.trivial(), array="tokens", layer=layer), lambda x: ()


def _z_attention(layer):
    config = replace(_z_config().transformer, dim=8, n_heads=2, qk_norm=False)
    return ZImageAttention(config, "tokens", layer), lambda x: (torch.ones(x.shape[1], 2, 2),)


def _klein_attention(layer):
    rope = (torch.ones(8, 4), torch.zeros(8, 4))
    return Flux2JointAttention(8, 2, 4, 1e-6, layer), lambda x: (x, rope)


@torch.no_grad()
@pytest.mark.parametrize("attention", [_wan_attention, _ltx_attention, _z_attention, _klein_attention])
def test_a_read_that_breaks_the_graph_recompiles_no_frame_per_layer(attention):
    """A DiT compiles with `dynamic=False`, where an int on the module specializes a frame per layer: with a read that
    breaks the graph, six attentions over 48 layers pass the recompile limit, and the frames then run uncompiled."""
    torch._dynamo.reset()
    array = Array(
        JOINT_TOKENS if attention is _klein_attention else "tokens", nodes={"dit"}, layers=frozenset({0, 1}),
        entry=dit_entry(2, 4), labels={"main"}, max_len=0,
        reads={"attn": Read(mask=Block(), op=Attend(q_heads=2))},
    )
    handle = build_handle(array, {"attn": _BreakingRead()})
    built = [attention(layer) for layer in (0, 1)]
    for module, _ in built:
        module.bind_state({array.name: handle})
    x = torch.ones(1, 4, 8)
    step = SubmoduleStep(state={array.name: ArrayStep(segments=(Segment(0, "main", 8),))})
    step.set_ctx(StepContext(request_ids=[0], graph_walk="gen", slot=0, capture=False))
    step.set_state_handles((handle,))
    step.begin_calls()
    guards = []
    optimize = torch._dynamo.optimize(
        lambda graph_module, inputs: graph_module.forward, dynamic=False,
        guard_fail_fn=lambda failure: guards.append(str(failure.reason)),
    )
    for module, extra in built:
        optimize(module)(x, *extra(x))
    step.end_calls()
    assert not [guard for guard in guards if "layer" in guard], "one frame should serve every layer"


# LTX-2.5


def _ltx_reference() -> list:
    t = LTX25Config().transformer
    return [
        spec_class(
            resource_key=key, nodes={"dit"},
            config=RaggedAttentionConfig(
                num_qo_heads=heads, num_kv_heads=heads, head_dim=head_dim, max_segments_per_request=1,
                dtype=torch.bfloat16, shard_over_sp=key == "dit_attn_video", workspace_mb=128,
            ),
        )
        for spec_class, key, heads, head_dim in (
            (RaggedAttentionSpec, "dit_attn_video", t.num_attention_heads, t.attention_head_dim),
            (RaggedAttentionSpec, "dit_attn_audio", t.audio_num_attention_heads, t.audio_attention_head_dim),
            (RaggedCrossAttentionSpec, "dit_xattn_video", t.num_attention_heads, t.attention_head_dim),
            (RaggedCrossAttentionSpec, "dit_xattn_audio", t.audio_num_attention_heads, t.audio_attention_head_dim),
        )
    ]


def test_ltx_specs_equal_the_hand_written_four():
    assert node_resources(_ltx()) == _ltx_reference(), "six arrays fold into the four hand-written resources"


def test_ltx_specs_equal_the_hand_written_four_after_yaml_overrides():
    overrides = {"resources": {key: {"max_segments_per_request": 3} for key in ("dit_attn_video", "dit_xattn_audio")}}
    specs, expected = node_resources(_ltx()), _ltx_reference()
    apply_yaml_overrides(specs, overrides)
    apply_yaml_overrides(expected, overrides)
    assert specs == expected


def test_ltx_on_sdpa_builds_no_spec_and_its_declarations_have_no_problems():
    assert node_resources(_ltx(), setup=_on_sdpa(_ltx())) == []
    assert state_declaration_problems(_ltx()) == []


def _ltx_reference_step(rids, shape: LTXShape, shard: int) -> dict:
    """The steps ``LTXDenoiseSubmodule.declare_step`` wrote on rank ``shard``'s share of the video tokens."""
    video = tuple(Segment(rid, "video", shape.video_tokens) for rid in rids)
    video_shard = tuple(Segment(rid, "video", shard) for rid in rids)
    audio = tuple(Segment(rid, "audio", shape.audio_frames) for rid in rids)
    text = tuple(Segment(rid, "text", shape.text_len) for rid in rids)
    return {
        "dit_attn_video": AttentionStep(segments=video, causal=False),
        "dit_attn_audio": AttentionStep(segments=audio, causal=False),
        "dit_xattn_video": RaggedCrossAttentionStep(segments=video_shard + text, pairs=(("video", "text"),)),
        "dit_xattn_audio": RaggedCrossAttentionStep(
            segments=video_shard + audio + text, pairs=(("audio", "text"), ("video", "audio"), ("audio", "video")),
        ),
    }


@pytest.mark.parametrize("rids", [[11], [11, 12]])
@pytest.mark.parametrize("world_size, rank, shard", [(1, 0, 25), (2, 0, 13), (2, 1, 12)])
def test_ltx_steps_equal_the_hand_written_ones_per_rank(rids, world_size, rank, shard):
    shape = LTXShape(frames=1, height=5, width=5, audio_frames=5, fps=24.0, text_len=16)
    model = _ltx()
    specs = node_resources(model)
    handles = node_handles(
        model, "dit", {spec.resource_key: _StubRagged() for spec in specs},
        sp_group=SimpleNamespace(world_size=world_size, rank=rank),
    )
    sub = LTXDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP)
    declared = sub.declare_step("generate_av", rids, [_inputs(shape)] * len(rids))
    step = state_steps(declared, handles)
    reference = _ltx_reference_step(rids, shape, shard)
    assert step.steps.keys() == reference.keys() and step.cg_key_info == shape
    for key, expected in reference.items():
        assert _by_label(step.steps[key]) == _by_label(expected), key
        if isinstance(expected, RaggedCrossAttentionStep):
            assert step.steps[key].pairs == expected.pairs, "plans run in the order of the hand-written pairs"
        else:
            assert step.steps[key].causal is False


@pytest.mark.parametrize("array, keys, queries, label", [
    (VIDEO, True, True, "video"),
    (AUDIO, False, False, "audio"),
    (TEXT_FOR_VIDEO, False, True, "video<-text"),
    (TEXT_FOR_AUDIO, False, False, "audio<-text"),
    (AUDIO_FOR_VIDEO, False, True, "video<-audio"),
    (VIDEO_FOR_AUDIO, True, False, "audio<-video"),
])
def test_ltx_each_attention_gets_its_exchange_at_sp2_and_its_plan_label(array, keys, queries, label):
    model = _ltx()
    resources = {spec.resource_key: _StubRagged() for spec in node_resources(model)}
    handle = node_handles(model, "dit", resources, sp_group=SimpleNamespace(world_size=2, rank=0))[array]
    assert (handle._keys_sharded, handle._queries_sharded) == (keys, queries)
    assert handle._plan_label == label


def test_ltx_no_attention_exchanges_at_sp1():
    model = _ltx()
    resources = {spec.resource_key: _StubRagged() for spec in node_resources(model)}
    handles = node_handles(model, "dit", resources, sp_group=SimpleNamespace(world_size=1, rank=0))
    assert not any(h._keys_sharded or h._queries_sharded for h in handles.values())


def test_ltx_on_sdpa_at_sp1_declares_no_step_as_main_did():
    model = _ltx()
    sub = LTXDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP)
    shape = LTXShape(frames=1, height=5, width=5, audio_frames=5, fps=24.0, text_len=16)
    handles = node_handles(model, "dit", {}, setup=_on_sdpa(model))
    assert state_steps(sub.declare_step("generate_av", [0], [_inputs(shape)]), handles) is None
