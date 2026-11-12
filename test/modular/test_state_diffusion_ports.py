"""The five DiTs plan the same attention on the contract that they planned by hand.

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
import yaml
from torch import nn

from mstar.distributed.base import node_group_axis_sizes
from mstar.distributed.communication import CommGroup, sp_seq_split
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.attn.ragged.config import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
    RaggedCrossAttentionStep,
)
from mstar.engine.resources.sequence_shard import SeqShardSpec, SequenceShard, sequence_parallel_refusals
from mstar.engine.resources.spec import apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.step import ResourceStep, Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Block, Read
from mstar.engine.state.adapters import (
    StateSetup,
    _resolve,
    build_handle,
    node_handles,
    node_resources,
    state_setup,
    state_steps,
)
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import serving_resources
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
from mstar.model.minimax_h3.components.layout import H3Layout
from mstar.model.minimax_h3.components.transformer import H3Attention
from mstar.model.minimax_h3.config import DIT_ATTN as H3_ATTN
from mstar.model.minimax_h3.config import DIT_SHARD, REFINER_ATTN, VAE_SHARD, H3Config, H3TransformerConfig
from mstar.model.minimax_h3.minimax_h3_model import H3Model
from mstar.model.minimax_h3.submodules import H3DenoiseSubmodule, H3TextEncoderSubmodule
from mstar.model.submodule_base import NodeInputs, engine_step, with_segment_steps
from mstar.model.wan22.components.dit import Wan22DiTAttention
from mstar.model.wan22.config import DIT_SHARD as WAN_SHARD
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


def _h3(**kwargs) -> H3Model:
    model = H3Model(model_path_hf="test/minimax_h3", skip_weight_loading=True, **kwargs)
    model.set_config(H3Config())
    return model


def _on_sdpa(model) -> StateSetup:
    """SDPA on every attention resource ``model`` declares (its sequence shards aren't)."""
    on_sdpa = {
        spec.resource_key: {"backend": "sdpa"} for spec in node_resources(model) if not isinstance(spec, SeqShardSpec)
    }
    setup, problems = state_setup({"resources": on_sdpa}, model.declare_state())
    assert not problems, problems
    return setup


def _wan_reference() -> list:
    """#1138's hand-written Wan2.2 specs: the self-attention over the one label its shard splits, and the shard."""
    c = Wan22Config()
    return [
        RaggedAttentionSpec(
            resource_key=SELF_ATTN, nodes={"dit"}, labels={"main"},
            config=RaggedAttentionConfig(
                num_qo_heads=c.num_attention_heads, num_kv_heads=c.num_attention_heads,
                head_dim=c.attention_head_dim, max_segments_per_request=2, compile_op=True, dtype=torch.bfloat16,
            ),
        ),
        SeqShardSpec(resource_key=WAN_SHARD, nodes={"dit"}, labels=("main",), axis=1),
    ]


def _klein_reference() -> list:
    t = Flux2KleinConfig().transformer
    return [RaggedAttentionSpec(
        resource_key=KLEIN_ATTN, nodes={"dit"}, labels={"main"},
        config=RaggedAttentionConfig(
            num_qo_heads=t.num_attention_heads, num_kv_heads=t.num_attention_heads, head_dim=t.attention_head_dim,
            max_segments_per_request=1, dtype=torch.bfloat16,
        ),
    )]


def _z_reference() -> list:
    t = _z_config().transformer
    return [RaggedAttentionSpec(
        resource_key=Z_ATTN, nodes={"dit"}, labels={"main", "image", "caption"},
        config=RaggedAttentionConfig(
            num_qo_heads=t.n_heads, num_kv_heads=t.n_heads, head_dim=t.head_dim, max_segments_per_request=1,
            dtype=torch.bfloat16,
        ),
    )]


MODELS = {"wan22": (_wan, _wan_reference), "klein": (_klein, _klein_reference), "z_image": (_z, _z_reference)}


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_one(name):
    build, reference = MODELS[name]
    assert node_resources(build()) == reference()


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_one_after_yaml_overrides(name):
    build, reference = MODELS[name]
    overrides = {"resources": {reference()[0].resource_key: {"max_segments_per_request": 7,
                                                              "flashinfer_backend": "fa2"}}}
    specs, expected = node_resources(build()), reference()
    apply_yaml_overrides(specs, overrides)
    apply_yaml_overrides(expected, overrides)
    assert specs == expected


@pytest.mark.parametrize("name", MODELS)
def test_sdpa_builds_no_spec_as_the_hand_written_model_did(name):
    model = MODELS[name][0]()
    assert [spec for spec in node_resources(model, setup=_on_sdpa(model)) if not isinstance(spec, SeqShardSpec)] == []


@pytest.mark.parametrize("name", MODELS)
def test_a_port_writes_no_spec_by_hand_and_has_no_layout_to_convert(name):
    model = MODELS[name][0]()
    assert model.get_node_resources() == [] and model.kv_labels() == {}, (
        "a DiT keeps nothing between steps"
    )


def test_ltx_writes_no_spec_by_hand_and_has_no_layout_to_convert():
    assert _ltx().get_node_resources() == [] and _ltx().kv_labels() == {}


@pytest.mark.parametrize("name", ["klein", "z_image"])
def test_declarations_have_no_problems(name):
    assert state_declaration_problems(MODELS[name][0]()) == []


@pytest.mark.parametrize("build", [_klein, _z, _h3])
def test_the_old_knob_is_refused(build):
    with pytest.raises(ValueError, match="attention_backend is a resource's setting"):
        build(attention_backend="sdpa")


def test_h3_refuses_a_model_kwarg_it_doesnt_take():
    with pytest.raises(TypeError, match="unexpected model_kwargs \\['row_buckets'\\]"):
        _h3(row_buckets=256)


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
    assert step.steps[key].fixed_layout == reference.steps[key].fixed_layout, "a captured bucket replays its layout"
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
    reference = SubmoduleStep(
        segments=segments, steps={KLEIN_ATTN: AttentionStep(causal=False, fixed_layout=True)}, cg_key_info=key,
    )
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
    reference = SubmoduleStep(
        segments=segments, steps={Z_ATTN: AttentionStep(causal=False, fixed_layout=True)}, cg_key_info=key,
    )
    _assert_plans_equal(_expanded(model, sub, [key] * len(rids), rids), reference, Z_ATTN)


@pytest.mark.parametrize("guidance, branches", [(5.0, 2), (1.0, 1)])
def test_wan22_steps_equal_the_hand_written_one(guidance, branches):
    model = _wan()
    key = WanStepKey(frames=3, height=8, width=8, guidance=guidance, i2v=False)
    sub = Wan22DitSubmodule(None, model.config)
    tokens = sub.num_tokens(key)
    segments = [Segment(request_id=0, label="main", span=tokens)] * branches
    reference = SubmoduleStep(
        segments=segments, steps={SELF_ATTN: AttentionStep(causal=False, fixed_layout=True)}, cg_key_info=key,
    )
    _assert_plans_equal(_expanded(model, sub, [key], [0]), reference, SELF_ATTN)



# Wan2.2's parallel configs, whose ranks main's model code split


WAN_PARALLEL_CONFIGS = ["configs/wan22_cfg_parallel_2gpu.yaml", "configs/wan22_cfg2_sp2_4gpu.yaml"]


def _serving(path: str) -> tuple[dict, dict[str, int]]:
    """The serving config at ``path``, and the size of its dit's node group on each mesh axis."""
    with open(path) as f:
        config = yaml.safe_load(f)
    (group,) = [group for group in config["node_groups"] if "dit" in group["node_names"]]
    return config, node_group_axis_sizes(group)


@pytest.mark.parametrize("path", WAN_PARALLEL_CONFIGS)
def test_wan22_parallel_configs_serve_the_hand_written_spec(path):
    config, axes = _serving(path)
    model = _wan()
    specs, expected = serving_resources(model, config), _wan_reference()
    resolve_spec_dependencies(expected)
    assert specs == expected
    assert sequence_parallel_refusals(resolve_spec_dependencies(specs).values(), "dit", axes["tp"], axes["sp"]) == []


@pytest.mark.parametrize("guidance", [5.0, 1.0])
@pytest.mark.parametrize("path", WAN_PARALLEL_CONFIGS)
def test_wan22_parallel_configs_plan_each_ranks_step_as_main_did(path, guidance):
    """A CFG rank plans the guidance branches it forwards, and an SP rank the whole sequence, which its resource
    exchanges."""
    _, axes = _serving(path)
    model = _wan()
    key = WanStepKey(frames=3, height=8, width=8, guidance=guidance, i2v=False)
    branches = 2 // axes["cfg"] if guidance > 1.0 else 1
    for cfg_rank in range(axes["cfg"]):
        sub = Wan22DitSubmodule(None, model.config, cfg_group=SimpleNamespace(world_size=axes["cfg"], rank=cfg_rank))
        segments = [Segment(request_id=0, label="main", span=sub.num_tokens(key))] * branches
        reference = SubmoduleStep(
            segments=segments, steps={SELF_ATTN: AttentionStep(causal=False, fixed_layout=True)}, cg_key_info=key,
        )
        handles = node_handles(model, "dit", {SELF_ATTN: _StubRagged()})
        step = state_steps(sub.declare_step("gen", [0], [_inputs(key)]), handles)
        _assert_plans_equal(step, reference, SELF_ATTN)


@pytest.mark.parametrize("rids", [[0], [0, 1]])
def test_wan22_guided_rows_split_one_rows_tokens_over_sp2(rids):
    """Each guidance branch is a row of its own, so the shard splits one row's tokens however many rows there are."""
    key = WanStepKey(frames=3, height=8, width=8, guidance=5.0, i2v=False)
    sub = Wan22DitSubmodule(None, _wan().config)
    plan = _shard_plan(SequenceShard(("main",), 1, None, 0, SimpleNamespace(world_size=2, rank=0)),
                       sub.declare_step("gen", rids, [_inputs(key)] * len(rids)), rids)
    split = plan.ranges["main"]
    assert split.counts == tuple(sp_seq_split(sub.num_tokens(key), 2)) and split.rows == 2 * len(rids), (
        "a rank's share of one row, in every guidance branch's row"
    )


def _shard_plan(shard: SequenceShard, step: SubmoduleStep, rids):
    """``shard``'s plan of the node's ``step``, whose segments are the shard's step."""
    ctx = StepContext(request_ids=rids, graph_walk="gen", slot=0, capture=False)
    return shard.plan(ResourceStep(segments=tuple(step.segments)), ctx)


def test_a_step_on_sdpa_is_none_as_the_hand_written_model_returned():
    model = _klein()
    sub = KleinDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
    handles = node_handles(model, "dit", {}, setup=_on_sdpa(model))
    assert state_steps(sub.declare_step("gen", [0], [_inputs(_klein_key())]), handles) is None


def test_a_lora_pool_gets_its_rows_from_the_steps_segments_as_before():
    model = _klein()
    sub = KleinDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
    key = _klein_key()
    declared = with_segment_steps(sub.declare_step("gen", [0, 1], [_inputs(key)] * 2), ("lora",))
    segments = tuple(Segment(rid, "main", key.total_tokens) for rid in (0, 1))
    assert declared.steps["lora"] == ResourceStep(segments=segments)


def test_engine_step_runs_the_expansion_after_the_lora_pools_step():
    model = _klein()
    sub = SimpleNamespace(
        declare_step=KleinDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP, compile_transformer=False)
        .declare_step,
        segment_keys=("lora",),
        state_handles=node_handles(model, "dit", {KLEIN_ATTN: _StubRagged()}),
    )
    step = engine_step(sub, "gen", [0], [_inputs(_klein_key())])
    assert type(step.steps["lora"]) is ResourceStep and isinstance(step.steps[KLEIN_ATTN], AttentionStep)


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


def _h3_attention(layer):
    config = replace(H3TransformerConfig(), hidden_size=8, num_attention_heads=2, attention_head_dim=4)
    return H3Attention(config, CommGroup.trivial(), "tokens", layer), lambda x: ()


@torch.no_grad()
@pytest.mark.parametrize("attention", [_wan_attention, _ltx_attention, _z_attention, _klein_attention, _h3_attention])
def test_a_read_that_breaks_the_graph_recompiles_no_frame_per_layer(attention):
    """A DiT compiles with `dynamic=False`, where an int on the module specializes a frame per layer: with a read that
    breaks the graph, six attentions over 48 layers pass the recompile limit, and the frames then run uncompiled."""
    torch._dynamo.reset()
    array = Array(
        JOINT_TOKENS if attention is _klein_attention else "tokens", nodes={"dit"}, layers=frozenset({0, 1}),
        entry=dit_entry(2, 4), labels={"main"}, max_len=0,
        reads={"attn": Read(mask=Block(), op=Attend(q_heads=2))},
    )
    handle = build_handle(array, {"attn": _BreakingRead()}, _resolve([array], StateSetup()))
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
    """#1138's hand-written LTX-2.5 specs: four attentions, the self ones over their labels, and the video shard."""
    t = LTX25Config().transformer
    return [
        spec_class(
            resource_key=key, nodes={"dit"}, **labels,
            config=RaggedAttentionConfig(
                num_qo_heads=heads, num_kv_heads=heads, head_dim=head_dim, max_segments_per_request=1,
                dtype=torch.bfloat16, workspace_mb=128,
            ),
        )
        for spec_class, key, heads, head_dim, labels in (
            (RaggedAttentionSpec, "dit_attn_video", t.num_attention_heads, t.attention_head_dim, {"labels": {"video"}}),
            (RaggedAttentionSpec, "dit_attn_audio", t.audio_num_attention_heads, t.audio_attention_head_dim,
             {"labels": {"audio"}}),
            (RaggedCrossAttentionSpec, "dit_xattn_video", t.num_attention_heads, t.attention_head_dim, {}),
            (RaggedCrossAttentionSpec, "dit_xattn_audio", t.audio_num_attention_heads, t.audio_attention_head_dim, {}),
        )
    ] + [SeqShardSpec(resource_key="dit_video_shard", nodes={"dit"}, labels=("video",), axis=1)]


def test_ltx_specs_equal_the_hand_written_four():
    assert node_resources(_ltx()) == _ltx_reference(), "six arrays fold into the four hand-written resources"


def test_ltx_splits_its_video_tokens_and_never_its_audio():
    """main's LTX-2.5 split the video tokens alone: the shard lists them, so the video self-attention runs Ulysses and
    the audio one keeps every head on each rank."""
    specs = resolve_spec_dependencies(node_resources(_ltx()))
    assert (specs["dit_attn_video"].sp_exchange, specs["dit_attn_audio"].sp_exchange) == ("ulysses", "none")


def test_ltx_specs_equal_the_hand_written_four_after_yaml_overrides():
    overrides = {"resources": {key: {"max_segments_per_request": 3} for key in ("dit_attn_video", "dit_xattn_audio")}}
    specs, expected = node_resources(_ltx()), _ltx_reference()
    apply_yaml_overrides(specs, overrides)
    apply_yaml_overrides(expected, overrides)
    assert specs == expected


def test_ltx_on_sdpa_builds_no_spec_and_its_declarations_have_no_problems():
    assert [spec for spec in node_resources(_ltx(), setup=_on_sdpa(_ltx())) if not isinstance(spec, SeqShardSpec)] == []
    assert state_declaration_problems(_ltx()) == []


def _ltx_reference_step(rids, shape: LTXShape) -> dict:
    """The steps ``LTXDenoiseSubmodule.declare_step`` wrote, every rank alike: a cross resource narrows a side its
    shard splits itself."""
    video = tuple(Segment(rid, "video", shape.video_tokens) for rid in rids)
    audio = tuple(Segment(rid, "audio", shape.audio_frames) for rid in rids)
    text = tuple(Segment(rid, "text", shape.text_len) for rid in rids)
    return {
        "dit_attn_video": AttentionStep(segments=video, causal=False, fixed_layout=True),
        "dit_attn_audio": AttentionStep(segments=audio, causal=False, fixed_layout=True),
        "dit_xattn_video": RaggedCrossAttentionStep(segments=video + text, pairs=(("video", "text"),)),
        "dit_xattn_audio": RaggedCrossAttentionStep(
            segments=video + audio + text, pairs=(("audio", "text"), ("video", "audio"), ("audio", "video")),
        ),
    }


@pytest.mark.parametrize("rids", [[11], [11, 12]])
def test_ltx_steps_equal_the_hand_written_ones(rids):
    shape = LTXShape(frames=1, height=5, width=5, audio_frames=5, fps=24.0, text_len=16)
    model = _ltx()
    specs = node_resources(model)
    handles = node_handles(model, "dit", {spec.resource_key: _StubRagged() for spec in specs})
    sub = LTXDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP)
    declared = sub.declare_step("generate_av", rids, [_inputs(shape)] * len(rids))
    step = state_steps(declared, handles)
    reference = _ltx_reference_step(rids, shape)
    assert step.steps.keys() == reference.keys() and step.cg_key_info == shape
    for key, expected in reference.items():
        assert _by_label(step.steps[key]) == _by_label(expected), key
        if isinstance(expected, RaggedCrossAttentionStep):
            assert step.steps[key].pairs == expected.pairs, "plans run in the order of the hand-written pairs"
        else:
            assert step.steps[key].causal is False and step.steps[key].fixed_layout, key


@pytest.mark.parametrize("array, label", [
    (VIDEO, "video"), (AUDIO, "audio"), (TEXT_FOR_VIDEO, "video<-text"), (TEXT_FOR_AUDIO, "audio<-text"),
    (AUDIO_FOR_VIDEO, "video<-audio"), (VIDEO_FOR_AUDIO, "audio<-video"),
])
def test_ltx_each_attention_gets_its_plan_label(array, label):
    model = _ltx()
    resources = {spec.resource_key: _StubRagged() for spec in node_resources(model)}
    assert node_handles(model, "dit", resources)[array]._plan_label == label


def test_ltx_node_step_holds_each_label_once():
    """Two arrays attend the video tokens (video and video-for-audio): the node's step, which its shard plans from,
    holds them once, or the shard would see each row twice and size every rank's share for twice the rows."""
    shape = LTXShape(frames=1, height=5, width=5, audio_frames=5, fps=24.0, text_len=16)
    sub = LTXDenoiseSubmodule(None, _ltx().config, loop_name=DENOISE_LOOP)
    rids = [11, 12]
    step = sub.declare_step("generate_av", rids, [_inputs(shape)] * len(rids))
    assert sorted((s.request_id, s.label) for s in step.segments) == sorted(
        (rid, label) for rid in rids for label in ("video", "audio", "text")
    )
    plan = _shard_plan(SequenceShard(("video",), 1, None, 0, SimpleNamespace(world_size=2, rank=0)), step, rids)
    assert plan.ranges["video"].rows == len(rids)


def test_ltx_on_sdpa_at_sp1_declares_no_step_as_main_did():
    model = _ltx()
    sub = LTXDenoiseSubmodule(None, model.config, loop_name=DENOISE_LOOP)
    shape = LTXShape(frames=1, height=5, width=5, audio_frames=5, fps=24.0, text_len=16)
    handles = node_handles(model, "dit", {}, setup=_on_sdpa(model))
    assert state_steps(sub.declare_step("generate_av", [0], [_inputs(shape)]), handles) is None


# MiniMax-H3


def _h3_reference() -> list:
    """`H3Model.get_node_resources` before the port (#1138's): the DiT's attention over its rows and their padding, the
    refiner's over the prompt, and the DiT's and the video decoder's sequence shards."""
    t = H3Config().transformer
    return [
        RaggedAttentionSpec(
            resource_key=key, nodes={node}, labels={label},
            config=RaggedAttentionConfig(
                num_qo_heads=t.num_attention_heads, num_kv_heads=t.num_attention_heads, head_dim=t.attention_head_dim,
                max_segments_per_request=segments, dtype=torch.bfloat16, workspace_mb=128,
            ),
        )
        for key, node, label, segments in ((H3_ATTN, "dit", "main", 2), (REFINER_ATTN, "text_encoder", "text", 1))
    ] + [
        SeqShardSpec(resource_key=DIT_SHARD, nodes={"dit"}, labels=("main",), axis=0),
        SeqShardSpec(resource_key=VAE_SHARD, nodes={"vae_decoder"}, labels=("clips",), axis=1, exchanges=False),
    ]


H3_CONFIGS = ["configs/minimax_h3.yaml", "configs/minimax_h3_2gpu.yaml", "configs/minimax_h3_sp2.yaml",
              "configs/minimax_h3_sp8.yaml"]


def test_h3_specs_equal_the_hand_written_ones():
    assert node_resources(_h3()) == _h3_reference()


def test_h3_specs_equal_the_hand_written_two_after_yaml_overrides():
    overrides = {"resources": {key: {"max_segments_per_request": 7, "flashinfer_backend": "fa2"}
                               for key in (H3_ATTN, REFINER_ATTN)}}
    specs, expected = node_resources(_h3()), _h3_reference()
    apply_yaml_overrides(specs, overrides)
    apply_yaml_overrides(expected, overrides)
    assert specs == expected


def test_h3_on_sdpa_builds_no_spec_and_its_declarations_have_no_problems():
    assert [spec for spec in node_resources(_h3(), setup=_on_sdpa(_h3())) if not isinstance(spec, SeqShardSpec)] == []
    assert state_declaration_problems(_h3()) == []


def _h3_layout() -> H3Layout:
    """37 rows, which no SP size past 1 divides."""
    return H3Layout(text_len=5, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4, anchors=("first",))


def _h3_dit(world: int) -> H3DenoiseSubmodule:
    """The DiT's submodule bound to a sequence shard over ``world`` SP ranks, whose size pads its rows."""
    sub = H3DenoiseSubmodule(SimpleNamespace(), H3Config(), loop_name=DENOISE_LOOP)
    sub.bind_node_resources({DIT_SHARD: SequenceShard(("main",), 0, None, 0, SimpleNamespace(world_size=world))})
    return sub


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_h3_steps_equal_the_hand_written_one_on_every_rank(world):
    """Every rank plans the whole row, padding a segment of its own, since the exchange gives each rank all of it.
    The rows pad to a multiple of the shard's SP ranks."""
    model, layout, sub = _h3(), _h3_layout(), _h3_dit(world)
    rows, padded = layout.total_rows, sub.num_tokens(layout)
    assert (padded > rows) == (world > 1)
    segments = [Segment(request_id=0, label="main", span=span) for span in (rows, padded - rows) if span]
    reference = SubmoduleStep(
        segments=segments, steps={H3_ATTN: AttentionStep(causal=False, fixed_layout=True)}, cg_key_info=layout,
    )
    handles = node_handles(model, "dit", {H3_ATTN: _StubRagged()})
    _assert_plans_equal(state_steps(sub.declare_step("gen", [0], [_inputs(layout)]), handles), reference, H3_ATTN)


def test_h3_refiner_step_equals_the_hand_written_one():
    handles = node_handles(_h3(), "text_encoder", {REFINER_ATTN: _StubRagged()})
    sub = H3TextEncoderSubmodule(nn.Identity(), nn.Identity())
    declared = sub.declare_step("encode_text", [0], [NodeInputs(tensor_inputs={}, input_seq_len=7)])
    reference = SubmoduleStep(segments=[Segment(request_id=0, label="text", span=7)],
                              steps={REFINER_ATTN: AttentionStep(causal=False)})
    _assert_plans_equal(state_steps(declared, handles), reference, REFINER_ATTN)


def test_h3s_shard_splits_the_whole_padded_row_at_sp2():
    """Each rank holds a share of the request's rows and their padding as one sequence, not of each segment."""
    layout, sub = _h3_layout(), _h3_dit(2)
    plan = _shard_plan(SequenceShard(("main",), 0, None, 0, SimpleNamespace(world_size=2, rank=0)),
                       sub.declare_step("gen", [0], [_inputs(layout)]), [0])
    padded = sub.num_tokens(layout)
    assert padded > layout.total_rows and plan.ranges["main"].counts == tuple(sp_seq_split(padded, 2))


@torch.no_grad()
def test_h3s_split_guards_on_no_layer_and_no_layout():
    """H3 compiles its blocks with `dynamic=False`, and the split runs outside them, as H3's own did, so neither a
    layer nor a step's spans recompile them."""
    torch._dynamo.reset()
    array = Array(
        "tokens", nodes={"dit"}, layers=frozenset({0, 1}), entry=dit_entry(2, 4), max_len=0,
        reads={"attn": Read(mask=Block(), op=Attend(q_heads=2), packed=False, segmented=True)},
    )
    setup, problems = state_setup({"resources": {"attn": {"backend": "sdpa"}}}, [array])
    assert not problems, problems
    handle = build_handle(array, {}, _resolve([array], setup), setup)
    built = [_h3_attention(layer)[0] for layer in (0, 1)]
    for module in built:
        module.bind_state({array.name: handle})
    guards = []
    optimize = torch._dynamo.optimize(
        lambda graph_module, inputs: graph_module.forward, dynamic=False,
        guard_fail_fn=lambda failure: guards.append(str(failure.reason)),
    )
    for spans in ((6, 2), (5, 3)):
        step = SubmoduleStep(state={array.name: ArrayStep(segments=tuple(Segment(0, "main", n) for n in spans))})
        step.set_ctx(StepContext(request_ids=[0], graph_walk="gen", slot=0, capture=False))
        step.set_state_handles((handle,))
        step.begin_calls()
        for module in built:
            optimize(module)(torch.ones(1, 8, 8))
        step.end_calls()
    assert not [guard for guard in guards if "layer" in guard or "lengths" in guard], guards
