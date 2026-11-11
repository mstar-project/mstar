"""Bagel's specs, steps and layouts on the contract equal the ones it wrote by hand, and its guided text is one forward.

The hand-written specs, steps and layouts below are the reference, copied from the model as it stood before the port, so
the adapter can't pass by agreeing with itself. The port changes four things, each stated where the reference makes it:
a guided ``prefill_text``, ``mixed_text`` or ``decode`` step plans ``main`` and ``cfg_img`` as one plan,
``main+cfg_img``, where Bagel planned each label on its own; a guided denoise step's plan is named
``main+cfg_text[+cfg_img]``, where Bagel named it ``_cfg_batched``; off CFG parallel the pool names no leader, where
Bagel named ``LLM``; and the guidance-on decode capture holds two plan tokens a row. Two edits change nothing a resource
reads: the step names its segments, so its sampler step carries them too, and a step that forks no row names None for
its fork rows, where Bagel named an empty set.

One plan over both labels means one forward over twice the rows, ``main``'s then ``cfg_img``'s, and the sample taken
from ``main``'s. If the rows didn't line up with the plan, ``cfg_img`` would attend ``main``'s pages, which shows up as
images that drift off the prompt rather than as anything that raises, so a float64 run checks it against a forward per
label.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
import yaml
from test_bagel_cfg_parallel import _image_kwargs
from test_multimodal_ordering import _StubTokenizer
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _published

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.conductor.conductor import Conductor
from mstar.conductor.request_info import FINAL_PREFILL
from mstar.engine.accelerator_graph_config import PiecewisePackedConfig
from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig, Span
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import check_capture, kv_layouts, node_handles, node_resources, state_steps
from mstar.engine.state.analysis import declaration_digests, digest_mismatch, state_declaration_problems
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.components.language_model import BagelForCausalLM, BagelLanguageModel
from mstar.model.bagel.components.vit_encoder import BagelVisionModel
from mstar.model.bagel.config import VIT_ATTN, BagelAutoEncoderConfig, BagelModelConfig, BagelViTConfig
from mstar.model.bagel.submodules import (
    _LOGITS,
    IMAGE_SENTINELS,
    VIT_BLOCK_LOOP,
    LLMSubmodule,
    ViTEncoderSubmodule,
    vae_latents,
    vit_patches,
)
from mstar.model.base import LabelLayout, keyed_labels, serving_resources
from mstar.model.multimodal import PromptPart, span_name
from mstar.model.submodule_base import ARNodeInputs, engine_step

CONFIGS = {path.stem: path for path in _published("BagelModel")}
CFG_PARALLEL = CONFIGS["bagel_cfg_parallel"]
LLM_NODES = {"LLM", "LLM_cfg_text", "LLM_cfg_img"}


def _bagel(path: Path | None = None, config: BagelModelConfig | None = None) -> BagelModel:
    """Bagel with no checkpoint, placed by ``path``'s node groups."""
    model = BagelModel.__new__(BagelModel)
    model.config = config or BagelModelConfig(vae_config=BagelAutoEncoderConfig(), vit_config=BagelViTConfig())
    model._has_cfg_parallel = model._has_llm_disaggregation = model._image_gen_remote_handoff = False
    model.tokenizer = _StubTokenizer()
    model.boi_token_id = _StubTokenizer.SPECIALS["<|vision_start|>"]
    model.eoi_token_id = _StubTokenizer.SPECIALS["<|vision_end|>"]
    if path is not None:
        model.get_worker_graphs(str(path))
    return model


def _reference(model: BagelModel) -> list:
    c, vit = model.config, model.config.vit_config
    nodes = set(LLM_NODES if model._has_cfg_parallel else {"LLM"})
    return [
        RaggedAttentionSpec(resource_key="vit_attn", nodes={"vit_encoder"}, config=RaggedAttentionConfig(
            num_qo_heads=vit.num_attention_heads, num_kv_heads=vit.num_attention_heads,
            head_dim=vit.hidden_size // vit.num_attention_heads, max_segments_per_request=1,
        )),
        KVSpec(
            resource_key="kv", nodes=nodes, config=PagedKVConfig(
                num_layers=c.num_hidden_layers, num_kv_heads=c.num_key_value_heads,
                head_dim=c.hidden_size // c.num_attention_heads, max_seq_len=c.max_position_embeddings,
                num_qo_heads=c.num_attention_heads,
            ),
            # Bagel named LLM on every config. Off CFG parallel the pool is on LLM alone, which admits either way.
            leader="LLM" if model._has_cfg_parallel else None,
        ),
        AttentionSpec(resource_key="attn", nodes=nodes, config=AttentionConfig(kv_cache="kv")),
        PositionSpec(resource_key="rope", nodes=nodes, config=PositionConfig(kv_cache="kv", rope_theta=c.rope_theta)),
        SamplerSpec(resource_key="sampler", nodes=nodes, vocab_size=c.vocab_size),
    ]


def test_the_published_configs_are_found():
    assert len(CONFIGS) == 8, "a config missing here would skip its tests"


@pytest.mark.parametrize("path", CONFIGS.values(), ids=CONFIGS)
def test_specs_equal_the_hand_written_ones_on_every_config(path):
    model = _bagel(path)
    _assert_equal_per_key(node_resources(model), _reference(model))


@pytest.mark.parametrize("path", CONFIGS.values(), ids=CONFIGS)
def test_specs_equal_the_hand_written_ones_after_the_configs_overrides(path):
    deployment = yaml.safe_load(path.read_text())
    model = _bagel(path)
    reference = _reference(model)
    apply_yaml_overrides(reference, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), reference)


SYNTHETIC = {"resources": {
    "kv": {"max_num_pages": 777, "page_size": 32}, "attn": {"decode_tensor_cores": False},
    "vit_attn": {"max_segments_per_step": 9}, "sampler": {"enable_logprobs": True},
}}


@pytest.mark.parametrize("path", CONFIGS.values(), ids=CONFIGS)
def test_specs_equal_the_hand_written_ones_under_a_synthetic_resources_block(path):
    model = _bagel(path)
    reference = _reference(model)
    apply_yaml_overrides(reference, SYNTHETIC)
    _assert_equal_per_key(serving_resources(model, {**yaml.safe_load(path.read_text()), **SYNTHETIC}), reference)


def test_the_port_writes_no_spec_by_hand():
    assert _bagel().get_node_resources() == []


@pytest.mark.parametrize("path", CONFIGS.values(), ids=CONFIGS)
def test_the_declarations_have_no_problems_on_every_config(path):
    assert state_declaration_problems(_bagel(path)) == []


# The steps


class _StubResource:
    """Stands in for any built resource a handle reads at bind."""

    def layer_view(self, i):
        return torch.empty(0)

    def opens_label(self, rid, label):
        return True

    def run(self, q, k=None, v=None, label=None):
        return q


def _handles(model: BagelModel, node: str) -> dict:
    return node_handles(model, node, {spec.resource_key: _StubResource() for spec in node_resources(model)})


def _llm(model: BagelModel | None = None, node: str = "LLM") -> LLMSubmodule:
    model = model or _bagel()
    llm = LLMSubmodule(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=None), lm_head=None), llm2vae=None,
        vae2llm=None, time_embedder=None, latent_pos_embed=None, config=model.config, node_name=node,
    )
    llm.bind_node_state(_handles(model, node))
    return llm


def _segments(labels, spans: dict[int, int]) -> tuple[Segment, ...]:
    """Label-major, as Bagel laid them out."""
    return tuple(Segment(rid, label, span) for label in labels for rid, span in spans.items())


def _info(rows: dict[int, tuple[bool, bool]]) -> dict:
    """``rows``: rid -> whether its request is guided, and whether this step samples its prefill's token."""
    return {
        rid: SimpleNamespace(step_metadata={"requires_cfg": guided, FINAL_PREFILL: samples})
        for rid, (guided, samples) in rows.items()
    }


def _prompt(n: int, guided: bool, start: int = 0, walk: str | None = None) -> ARNodeInputs:
    return ARNodeInputs(
        input_seq_len=n, input_ids=torch.arange(start, start + n), resource_step_info=guided, chunk_start=start,
        graph_walk=walk,
    )


def _image(n: int, guided: bool, base: int = 10) -> ARNodeInputs:
    """An image block's or a denoise step's row, at positions ``base``, ``base`` + 1 and ``base`` + 2 on main, cfg_text
    and cfg_img."""
    labels = ("main", "cfg_text", "cfg_img")
    ids = {label: torch.full((n,), base + i, dtype=torch.int32) for i, label in enumerate(labels)}
    return ARNodeInputs(input_seq_len=n, resource_step_info=guided, custom_pos_ids=ids)


def _plan(step: SubmoduleStep) -> str:
    kv = step.steps["kv"]
    return next(iter(kv.combined_labels.values()), kv.segments[0].label)


def _assert_equal(step: SubmoduleStep, reference: SubmoduleStep) -> None:
    """Per key, as `_assert_steps_equal`, with each step's position ids as the position manager reads them for its
    plan: a bare tensor on a step that plans one label, else the entry under the plan's name."""
    ids, want = (RopeManager._explicit_pos_ids(None, s.steps["rope"], _plan(s), 1) for s in (step, reference))
    assert (ids is None and want is None) or torch.equal(ids, want), "the step's tokens would sit at other positions"
    _assert_steps_equal(*(
        replace(s, steps={**s.steps, "rope": replace(s.steps["rope"], pos_ids=None)}) for s in (step, reference)
    ))


# guided text steps plan both labels as one, where Bagel planned "main" and "cfg_img" each on its own
COMBINED = {("main", "cfg_img"): "main+cfg_img"}


@pytest.mark.parametrize("guided", [False, True], ids=["unguided", "guided"])
def test_a_prefill_text_step_equals_the_hand_written_one(guided):
    rows = {1: _prompt(4, guided), 2: _prompt(3, guided)}
    step = engine_step(_llm(), graph_walk="prefill_text", request_ids=[1, 2], inputs=list(rows.values()),
                       per_request_info=_info({1: (guided, True), 2: (guided, False)}))
    labels = ("main", "cfg_img") if guided else ("main",)
    reference = SubmoduleStep(segments=_segments(labels, {1: 4, 2: 3}), cg_key_info=guided, steps={
        "kv": KVStep(
            pre_forks=(("main", "cfg_text"),) if guided else (), fork_rids=frozenset({1, 2}) if guided else None,
            combined_labels=COMBINED if guided else {},
        ),
        "attn": AttentionStep(causal=True),
        "rope": PositionStep(),
        "sampler": SamplerStep(prefill_tracked_tokens={1: torch.arange(4), 2: torch.arange(3)},
                               kept_rids=frozenset({1})),
    })
    _assert_equal(step, reference)


def test_a_guided_batch_holds_its_unguided_rows_place_on_cfg_img():
    rows = [_prompt(4, True), _prompt(3, False)]
    step = engine_step(_llm(), graph_walk="prefill_text", request_ids=[1, 2], inputs=rows,
                       per_request_info=_info({1: (True, False), 2: (False, False)}))
    reference = SubmoduleStep(segments=_segments(("main", "cfg_img"), {1: 4, 2: 3}), cg_key_info=True, steps={
        "kv": KVStep(pre_forks=(("main", "cfg_text"),), fork_rids=frozenset({1}), sink=frozenset({(2, "cfg_img")}),
                     combined_labels=COMBINED),
        "attn": AttentionStep(causal=True),
        "rope": PositionStep(),
        "sampler": SamplerStep(prefill_tracked_tokens={1: torch.arange(4), 2: torch.arange(3)},
                               kept_rids=frozenset()),
    })
    _assert_equal(step, reference)


def test_a_guided_prompt_forks_on_its_first_chunk_and_not_on_its_second():
    llm, info = _llm(), _info({1: (True, False)})
    steps = [
        engine_step(llm, graph_walk="prefill_text", request_ids=[1], inputs=[_prompt(4, True, start)],
                    per_request_info=info)
        for start in (0, 4)
    ]

    def reference(start, kv):
        return SubmoduleStep(segments=_segments(("main", "cfg_img"), {1: 4}), cg_key_info=True, steps={
            "kv": kv, "attn": AttentionStep(causal=True), "rope": PositionStep(),
            "sampler": SamplerStep(prefill_tracked_tokens={1: torch.arange(start, start + 4)}, kept_rids=frozenset()),
        })

    _assert_equal(steps[0], reference(0, KVStep(
        pre_forks=(("main", "cfg_text"),), fork_rids=frozenset({1}), combined_labels=COMBINED,
    )))
    # Bagel named an empty set of fork rows here, which only a fork reads, and this step has none
    _assert_equal(steps[1], reference(4, KVStep(combined_labels=COMBINED)))


@pytest.mark.parametrize("guided", [False, True], ids=["unguided", "guided"])
def test_a_mixed_text_step_equals_the_hand_written_one(guided):
    rows = [_prompt(1, guided, walk="decode"), _prompt(3, guided, walk="prefill_text")]
    step = engine_step(_llm(), graph_walk="mixed_text", request_ids=[1, 2], inputs=rows,
                       per_request_info=_info({1: (guided, False), 2: (guided, True)}))
    labels = ("main", "cfg_img") if guided else ("main",)
    reference = SubmoduleStep(segments=_segments(labels, {1: 1, 2: 3}), cg_key_info=guided, steps={
        # the decode row keeps its own cfg_text: only the prompt forks
        "kv": KVStep(
            pre_forks=(("main", "cfg_text"),) if guided else (), fork_rids=frozenset({2}) if guided else None,
            combined_labels=COMBINED if guided else {},
        ),
        "attn": AttentionStep(causal=True),
        "rope": PositionStep(),
        "sampler": SamplerStep(prefill_tracked_tokens={2: torch.arange(3)}, kept_rids=frozenset({1, 2})),
    })
    _assert_equal(step, reference)


@pytest.mark.parametrize("guided", [False, True], ids=["unguided", "guided"])
def test_a_decode_step_equals_the_hand_written_one(guided):
    rows = [_prompt(1, guided), _prompt(1, False)]
    step = engine_step(_llm(), graph_walk="decode", request_ids=[1, 2], inputs=rows,
                       per_request_info=_info({1: (guided, False), 2: (False, False)}))
    labels = ("main", "cfg_img") if guided else ("main",)
    reference = SubmoduleStep(segments=_segments(labels, {1: 1, 2: 1}), cg_key_info=guided, steps={
        "kv": KVStep(sink=frozenset({(2, "cfg_img")}) if guided else frozenset(),
                     combined_labels=COMBINED if guided else {}),
        "attn": AttentionStep(causal=True),
        "rope": PositionStep(),
        "sampler": SamplerStep(),
    })
    _assert_equal(step, reference)


@pytest.mark.parametrize("walk", ["prefill_vit", "prefill_vae"])
@pytest.mark.parametrize("guided", [False, True], ids=["unguided", "guided"])
def test_an_image_prefill_step_equals_the_hand_written_one(walk, guided):
    rows = [_image(6, guided), _image(5, guided, 20)]
    step = engine_step(_llm(), graph_walk=walk, request_ids=[1, 2], inputs=rows,
                       per_request_info=_info({1: (guided, True), 2: (guided, False)}))
    steps = {
        # cfg_text tracks the context with the image, so it forks at commit
        "kv": KVStep(post_forks=(("main", "cfg_text"),), fork_rids=frozenset({1, 2})) if guided else KVStep(),
        "attn": AttentionStep(causal=False),
        "rope": PositionStep(pos_ids={"main": torch.cat([row.custom_pos_ids["main"] for row in rows])}, advance=(1, 1)),
    }
    if walk == "prefill_vit":
        steps["sampler"] = SamplerStep(kept_rids=frozenset({1}))
    _assert_equal(step, SubmoduleStep(segments=_segments(("main",), {1: 6, 2: 5}), cg_key_info=guided, steps=steps))


@pytest.mark.parametrize("walk, labels, plan", [
    ("image_gen", ("main",), None),
    ("image_gen_head", ("main",), None),
    # Bagel named both plans "_cfg_batched"
    ("image_gen_guided2", ("main", "cfg_text"), "main+cfg_text"),
    ("image_gen_guided3", ("main", "cfg_text", "cfg_img"), "main+cfg_text+cfg_img"),
])
def test_a_denoise_step_equals_the_hand_written_one(walk, labels, plan):
    guided = plan is not None or walk == "image_gen_head"
    step = engine_step(_llm(), graph_walk=walk, request_ids=[1], inputs=[_image(6, guided)])
    ids = torch.cat([torch.full((6,), 10 + i, dtype=torch.int32) for i in range(len(labels))])
    reference = SubmoduleStep(segments=_segments(labels, {1: 6}), cg_key_info=guided, steps={
        "kv": KVStep(commit=False, combined_labels={labels: plan} if plan else {}),
        "attn": AttentionStep(causal=False),
        "rope": PositionStep(pos_ids={plan or "main": ids}, advance=(0,) * len(labels)),
    })
    _assert_equal(step, reference)


@pytest.mark.parametrize("walk, node, label", [
    ("image_gen_cfg2", "LLM", "main"),
    ("image_gen_cfg2", "LLM_cfg_text", "cfg_text"),
    ("image_gen_cfg3", "LLM", "main"),
    ("image_gen_cfg3", "LLM_cfg_text", "cfg_text"),
    ("image_gen_cfg3", "LLM_cfg_img", "cfg_img"),
])
def test_a_cfg_walk_step_on_each_branch_node_equals_the_hand_written_one(walk, node, label):
    step = engine_step(_llm(_bagel(CFG_PARALLEL), node), graph_walk=walk, request_ids=[1], inputs=[_image(6, True)])
    position = 10 + ("main", "cfg_text", "cfg_img").index(label)
    reference = SubmoduleStep(segments=_segments((label,), {1: 6}), cg_key_info=True, steps={
        "kv": KVStep(commit=False),
        "attn": AttentionStep(causal=False),
        "rope": PositionStep(pos_ids={label: torch.full((6,), position, dtype=torch.int32)}, advance=(0,)),
    })
    _assert_equal(step, reference)


TINY_VIT = BagelViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                          patch_size=14)


def test_the_vit_region_step_equals_the_hand_written_one(monkeypatch):
    monkeypatch.setenv("MSTAR_VIT_ACCELERATOR_GRAPH", "1")
    model = _bagel(config=replace(_bagel().config, vit_config=TINY_VIT))
    vit = ViTEncoderSubmodule(BagelVisionModel(TINY_VIT), None, None, 14, 70, accelerator_graph=True)
    vit.bind_node_state(_handles(model, "vit_encoder"))
    config = vit.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[VIT_BLOCK_LOOP]
    runner = object.__new__(PiecewiseAcceleratorGraphRunner)
    runner._config, runner._segment_keys, runner._state_handles = config, (), vit.state_handles
    reference = PiecewisePackedConfig(
        capture_fn=config.capture_fn, make_static_inputs=config.make_static_inputs, attn_key=VIT_ATTN,
        total_tokens=config.total_tokens, capture_batch_sizes=config.capture_batch_sizes,
    ).region_step([7, 8], [3, 5])
    _assert_steps_equal(runner._region_step([7, 8], [3, 5]), reference)


def _captures(llm: LLMSubmodule) -> dict:
    """The LLM's captures, by walk and guidance."""
    configs = llm.get_accelerator_graph_configs(torch.device("cpu"))
    return {(config.capture_graph_walk, config.additional_key_info): config for config in configs}


def test_the_guidance_on_decode_capture_holds_two_plan_tokens_a_row():
    captures = _captures(_llm())
    assert captures["decode", False].get_total_tokens(4) == [4]
    # Bagel planned each label's tokens apart, so its capture held one
    assert captures["decode", True].get_total_tokens(4) == [8], "a guided bucket would hold half the plan's tokens"


def test_each_capture_replays_walks_that_read_the_cache_through_one_mask():
    llm = _llm()
    for config in _captures(llm).values():
        check_capture(llm.state_handles, {config.capture_graph_walk, *config.replay_graph_walks})


# The layouts and request configs


def _laid_out(model: BagelModel, prompt, input_modalities, output_modalities, images=(), **kwargs) -> dict:
    """``process_prompt``'s layouts, as the preprocess worker converts them."""
    out = model.process_prompt(prompt, input_modalities, output_modalities, tensors={"image_inputs": list(images)},
                               **kwargs)
    layouts = kv_layouts(model.declare_state(), out.metadata["state_layout"], out.metadata.get("keyed_spans", {}))
    return layouts["kv"], out.new_input_tensors


def _ids(n: int, k: int = 0) -> Span:
    return Span("ids", "prefill_text", n, n, span_name("text_inputs", k))


def test_a_guided_generation_lays_out_what_bagel_did():
    layouts, tensors = _laid_out(_bagel(), "A cat", ["text"], ["image"])
    text = _ids(len(_StubTokenizer().encode("<|im_start|>A cat<|im_end|><|im_start|>")))
    scratch = (1024 // 16) ** 2 + IMAGE_SENTINELS
    assert layouts == {
        "main": LabelLayout(spans=(text,), scratch=scratch, keyed_spans=1),
        "cfg_text": LabelLayout(scratch=scratch),
        # the default image scale leaves cfg_img out of the denoise
        "cfg_img": LabelLayout(spans=(text,)),
    }, "the request would reserve and key other pages than Bagel's"


def test_an_unguided_generation_lays_out_what_bagel_did():
    layouts, _ = _laid_out(_bagel(), "A cat", ["text"], ["image"], cfg_text_scale=1.0)
    text = _ids(len(_StubTokenizer().encode("<|im_start|>A cat<|im_end|><|im_start|>")))
    assert layouts == {
        "main": LabelLayout(spans=(text,), scratch=(1024 // 16) ** 2 + IMAGE_SENTINELS, keyed_spans=1),
        "cfg_text": LabelLayout(),
        "cfg_img": LabelLayout(),
    }, "the request would reserve and key other pages than Bagel's"


def test_an_understood_image_lays_out_what_bagel_did():
    model = _bagel()
    layouts, tensors = _laid_out(model, "What is it?", ["image", "text"], ["text"], [torch.rand(3, 480, 640)])
    before, after = (tensors[span_name("text_inputs", k)][0].numel() for k in (0, 1))
    c = model.config
    patches = vit_patches(480, 640, "default", c.vit_config.patch_size, c.vit_max_num_patch_per_side)
    image = Span("digest", "prefill_vit", patches + IMAGE_SENTINELS, 1, ("image", 0), ("default",))
    assert layouts == {
        "main": LabelLayout(spans=(_ids(before), image, _ids(after, 1)), growth=(("decode_loop", 1),)),
    }, "the request would reserve and key other pages than Bagel's"


def test_an_edit_lays_out_what_bagel_did():
    model = _bagel()
    layouts, tensors = _laid_out(model, "Make it night", ["image", "text"], ["image"], [torch.rand(3, 480, 640)])
    height, width = tensors[span_name("image_inputs", 0)][0].shape[-2:]
    c = model.config
    patches = vit_patches(height, width, "default", c.vit_config.patch_size, c.vit_max_num_patch_per_side)
    latents = vae_latents(height, width, "default", c.latent_downsample, c.max_latent_size)
    item = (("image", 0), ("default",))
    vae = Span("digest", "prefill_vae", latents + IMAGE_SENTINELS, 1, *item)
    vit = Span("digest", "prefill_vit", patches + IMAGE_SENTINELS, 1, *item)
    text = _ids(tensors[span_name("text_inputs", 0)][0].numel())
    scratch = (height // 16) * (width // 16) + IMAGE_SENTINELS
    assert layouts == {
        # the input is resized before its walks see it, so its file's digest keys nothing
        "main": LabelLayout(spans=(vae, vit, text), scratch=scratch, keyed_spans=0),
        "cfg_text": LabelLayout(spans=(vae, vit), scratch=scratch),
        "cfg_img": LabelLayout(spans=(text,), scratch=scratch),
    }, "the request would reserve and key other pages than Bagel's"


def test_a_request_opens_with_bagels_per_node_labels_and_its_sampling_settings():
    model = _bagel(CFG_PARALLEL)
    conductor = SimpleNamespace(
        model=model, _declarations=model.declare_state, _node_resources=lambda: node_resources(model),
        _kv_labels={key: tuple(labels) for key, labels in model.kv_labels().items()},
    )
    kwargs = _image_kwargs(model, {"temperature": 0.2, "top_k": None})
    configs = Conductor._get_resource_configs(conductor, ["text"], ["image"], kwargs)
    c = model.config
    assert configs["sampler"] == SamplingReqConfig(
        temperature=0.2, top_k=c.top_k, top_p=c.top_p, ignore_eos=c.ignore_eos, repetition_penalty=c.repetition_penalty,
    )
    kv = configs["kv"]
    assert kv.publish_labels_per_node_walk == {
        ("LLM", walk): ["cfg_text"] for walk in ("prefill_text", "prefill_vit", "prefill_vae")
    }
    assert kv.needed_labels_per_node_walk[("LLM_cfg_text", "image_gen_cfg2")] == ["cfg_text"]
    assert kv.needed_labels_per_node_walk[("LLM", "decode")] == ["main", "cfg_img"]
    assert kv.needed_labels is None, (
        "a flat needed_labels would open the request's labels on every node and walk, not per node and walk"
    )


# A prefix hit can't end inside a step that reads its keyed walk as a whole


def _worker(model: BagelModel) -> PreprocessWorkerThread:
    worker = object.__new__(PreprocessWorkerThread)
    worker.model, worker._keyed_labels = model, keyed_labels(model)
    return worker


def _two_images() -> tuple[BagelModel, dict]:
    """An understood prompt of two images between texts, each image's digest span its own step."""
    model = _bagel()
    parts = [
        PromptPart(modality="text", text="what is"), PromptPart(modality="image", index=0),
        PromptPart(modality="text", text="and"), PromptPart(modality="image", index=1),
        PromptPart(modality="text", text="this"),
    ]
    images = [torch.rand(3, 480, 640), torch.rand(3, 320, 320)]
    layouts, _ = _laid_out(
        model, "what is\nand\nthis", [part.modality for part in parts], ["text"], images, prompt_parts=parts,
    )
    return model, layouts


def test_one_image_a_step_passes_the_whole_step_check():
    model, layouts = _two_images()
    _worker(model)._check_whole_steps({"kv": layouts})


def test_a_second_image_sharing_its_step_is_refused_per_request():
    model, layouts = _two_images()
    spans = list(layouts["main"].spans)
    spans[3] = spans[3]._replace(opens_step=False)
    shared = {"kv": {"main": layouts["main"]._replace(spans=tuple(spans))}}
    with pytest.raises(RuntimeError, match=r"Array 'kv'\.reads\['attn_image'\] reads walk 'prefill_vit' as a whole, "
                                           r"and span 3 of 'main' is a digest span that shares its step"):
        _worker(model)._check_whole_steps(shared)


def test_an_ids_span_keyed_on_an_image_walk_is_refused_per_request():
    """The worker keys an ids span by its own source, not the label's keyed tensor, so the load check alone can't see
    one on a walk the LLM reads through ``img_emb``."""
    model, layouts = _two_images()
    spans = list(layouts["main"].spans)
    spans[1] = Span("ids", "prefill_vit", spans[1].length, 1, "caption_ids")
    with pytest.raises(RuntimeError, match=r"reads walk 'prefill_vit' as a whole, and span 1 of 'main' is an ids "
                                           r"span"):
        _worker(model)._check_whole_steps({"kv": {"main": layouts["main"]._replace(spans=tuple(spans))}})


# Declarations that follow the deployment


def test_bagels_declarations_follow_its_placement_and_a_mismatch_names_them():
    model = _bagel()
    before = declaration_digests(model.declare_state())
    model.get_worker_graphs(str(CFG_PARALLEL))
    after = declaration_digests(model.declare_state())
    message = digest_mismatch(after, before, here="conductor", there="API server")
    assert {name for name in after if after[name] != before[name]} == {"kv", "sampler"}
    assert "kv, sampler" in message, "a process that missed the placement would serve other nodes with no error"


# One forward over main's rows and cfg_img's


TINY = BagelModelConfig(
    vae_config=BagelAutoEncoderConfig(), vit_config=BagelViTConfig(), hidden_size=32, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=64, num_hidden_layers=2, vocab_size=50, max_position_embeddings=128,
)


class _Plan:
    """The step's segments and placeholders, as the KV and attention resources hold the plan."""

    segments: tuple[Segment, ...] = ()
    sink: frozenset = frozenset()

    def labelled(self, label: str) -> list[Segment]:
        return [s for s in self.segments if s.label in label.split("+")]


class _Pages:
    """Stands in for the KV resource: each (row, label, layer) stream's k and v, in the order they were added."""

    requires_kv_write = True

    def __init__(self, plan: _Plan):
        self.plan, self.held = plan, {}

    def layer_view(self, i):
        return i

    def opens_label(self, rid, label):
        return True

    def write_kv(self, k, v, layer_idx=None, label=None):
        at = 0
        for s in self.plan.labelled(label):
            if (s.request_id, s.label) not in self.plan.sink:
                added = (k[at:at + s.span], v[at:at + s.span])
                self.held.setdefault((s.request_id, s.label, layer_idx), []).append(added)
            at += s.span

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        """Causal attention of each segment over its stream; a placeholder reads nothing."""
        out, at = [], 0
        for s in self.plan.labelled(label):
            qs, at = q[at:at + s.span], at + s.span
            if (s.request_id, s.label) in self.plan.sink:
                out.append(torch.zeros_like(qs))
                continue
            added = self.held[(s.request_id, s.label, layer_idx)]
            keys, values = (torch.cat(field) for field in zip(*added, strict=True))
            keys, values = (x.repeat_interleave(qs.shape[1] // x.shape[1], 1) for x in (keys, values))
            n = keys.shape[0]
            scores = torch.einsum("qhd,khd->hqk", qs, keys) / math.sqrt(qs.shape[-1])
            scores = scores.masked_fill(torch.arange(n)[None, :] > torch.arange(n - s.span, n)[:, None], float("-inf"))
            out.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), values))
        return torch.cat(out)

    def select_last_hidden(self, x, label=None):
        return x[torch.tensor([s.span for s in self.plan.labelled(label)]).cumsum(0) - 1]


class _Positions:
    """Stands in for the position resource: rotate-half RoPE at each stream's own position."""

    def __init__(self, plan: _Plan):
        self.plan, self.at = plan, {}

    def apply_qk(self, q, k, label=None, rope_dtype=None):
        pos = torch.cat([
            torch.arange(self.at.get(s[:2], 0), self.at.get(s[:2], 0) + s.span) for s in self.plan.labelled(label)
        ]).to(q.dtype)
        half = q.shape[-1] // 2
        angles = torch.outer(pos, TINY.rope_theta ** (-torch.arange(half, dtype=q.dtype) / half)).repeat(1, 2)
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]

        def rotated(x):
            return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin

        return rotated(q), rotated(k)

    def commit(self):
        for s in self.plan.segments:
            if s[:2] not in self.plan.sink:
                self.at[s[:2]] = self.at.get(s[:2], 0) + s.span


class _Sampler:
    """Takes the most likely token."""

    def sample(self, rows, logits):
        return logits.argmax(-1)


@pytest.fixture
def tiny_lm():
    lm = BagelForCausalLM(TINY).double()
    generator = torch.Generator().manual_seed(0)
    for parameter in lm.parameters():  # the model's own parameters start uninitialized
        parameter.data.normal_(0, 0.3, generator=generator)
    return lm


class _Node:
    """The LLM node over the stand-ins: its handles, and one step at a time through them."""

    def __init__(self, lm: BagelForCausalLM):
        model = _bagel(config=TINY)
        self.plan = _Plan()
        self.pages, self.positions = _Pages(self.plan), _Positions(self.plan)
        resources = {spec.resource_key: None for spec in node_resources(model)}
        resources.update(kv=self.pages, attn=self.pages, rope=self.positions, sampler=_Sampler())
        self.handles = node_handles(model, "LLM", resources)
        self.llm = LLMSubmodule(lm, None, None, None, None, TINY)
        self.llm.bind_node_state(self.handles)

    def run(self, step: SubmoduleStep, walk: str, forward):
        ctx = StepContext(request_ids=tuple(dict.fromkeys(s.request_id for s in step.segments)), graph_walk=walk,
                          slot=0, capture=False)
        ctx.set_padded_rids(ctx.request_ids)
        step.set_ctx(ctx)
        self.plan.segments, self.plan.sink = tuple(step.steps["kv"].segments), step.steps["kv"].sink
        step.begin_calls()
        out = forward()
        step.check_calls()
        step.end_calls()
        self.positions.commit()
        return out

    def label_step(self, label: str, spans: dict[int, int], guided: dict[int, bool], walk: str) -> SubmoduleStep:
        """One label's own step, as Bagel planned each label of a guided step."""
        sink = frozenset((rid, label) for rid in spans if label != "main" and not guided[rid])
        step = SubmoduleStep(segments=_segments((label,), spans), state={"kv": ArrayStep(placeholders=sink)})
        return state_steps(step, self.handles, graph_walk=walk)


@torch.no_grad()
def test_one_guided_forward_equals_a_forward_per_label(tiny_lm):
    """A guided prefill_text then decode over a guided row, an unguided one and another guided one, against Bagel's
    algorithm: ``main``'s forward, then ``cfg_img``'s, each over its own plan."""
    spans, guided = {1: 5, 2: 3, 3: 4}, {1: True, 2: False, 3: True}
    generator = torch.Generator().manual_seed(1)
    prompt = torch.randint(0, TINY.vocab_size, (sum(spans.values()),), generator=generator)
    tokens = torch.randint(0, TINY.vocab_size, (len(spans),), generator=generator)
    info = {rid: SimpleNamespace(step_metadata={"requires_cfg": g}) for rid, g in guided.items()}

    port = _Node(tiny_lm)
    rows = [ARNodeInputs(input_seq_len=n, input_ids=ids, resource_step_info=guided[rid])
            for (rid, n), ids in zip(spans.items(), prompt.split(list(spans.values())), strict=True)]
    step = engine_step(port.llm, graph_walk="prefill_text", request_ids=list(spans), inputs=rows, per_request_info=info)
    prefill = port.run(
        step, "prefill_text", lambda: port.llm._forward_prefill_text(prompt, requires_cfg=True)[_LOGITS][0],
    )
    rows = [ARNodeInputs(input_seq_len=1, input_ids=tokens[i:i + 1], resource_step_info=guided[rid])
            for i, rid in enumerate(spans)]
    step = engine_step(port.llm, graph_walk="decode", request_ids=list(spans), inputs=rows, per_request_info=info)

    def decode():
        logits = port.llm._forward_decode_batched(tokens, requires_cfg=True)[_LOGITS][0]
        port.llm._sample_tokens(logits)
        return logits

    decoded = port.run(step, "decode", decode)

    bagel = _Node(tiny_lm)
    embedded, ends = tiny_lm.model.embed_tokens(prompt), torch.tensor(list(spans.values())).cumsum(0) - 1
    hidden = bagel.run(bagel.label_step("main", spans, guided, "prefill_text"), "prefill_text",
                       lambda: tiny_lm(embedded))
    bagel.run(bagel.label_step("cfg_img", spans, guided, "prefill_text"), "prefill_text", lambda: tiny_lm(embedded))
    embedded, ones = tiny_lm.model.embed_tokens(tokens), dict.fromkeys(spans, 1)
    decode_hidden = bagel.run(bagel.label_step("main", ones, guided, "decode"), "decode", lambda: tiny_lm(embedded))
    bagel.run(bagel.label_step("cfg_img", ones, guided, "decode"), "decode", lambda: tiny_lm(embedded))

    torch.testing.assert_close(prefill, tiny_lm.lm_head(hidden[ends]), rtol=0, atol=1e-12)
    torch.testing.assert_close(decoded, tiny_lm.lm_head(decode_hidden), rtol=0, atol=1e-12)
    assert port.pages.held.keys() == bagel.pages.held.keys()
    for stream, added in port.pages.held.items():
        torch.testing.assert_close(added, bagel.pages.held[stream], rtol=0, atol=1e-12,
                                   msg=f"{stream}'s pages would hold other entries than its own label's forward wrote")


# A compiled capture must not guard on a layer: a frame per layer hits the recompile limit, then runs uncompiled.


class _Attend:
    """Returns q, so the stack runs with no kernel."""

    requires_kv_write = True

    def layer_view(self, i):
        return torch.empty(0)

    def opens_label(self, rid, label):
        return True

    def write_kv(self, k, v, layer_idx=None, label=None):
        pass

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        return q.clone()


class _NoRotation:
    """Rotates nothing, for a stack that calls `rope`."""

    def apply_qk(self, q, k, label=None, rope_dtype=None):
        return q, k


def _compiled_frames(num_layers: int) -> tuple[int, list[str]]:
    """The frames Bagel's language model compiles into as a capture does, and its failed guards, over two steps; the
    second runs under the capture's stance, which raises on any new compile."""
    config = replace(TINY, num_hidden_layers=num_layers)
    model = _bagel(config=config)
    resources = {spec.resource_key: None for spec in node_resources(model)}
    resources.update(kv=_Attend(), attn=_Attend(), rope=_NoRotation())
    handle = node_handles(model, "LLM", resources)["kv"]
    stack = BagelLanguageModel(config).eval()
    generator = torch.Generator().manual_seed(0)
    for parameter in stack.parameters():
        parameter.data.normal_(0, 0.02, generator=generator)
    for module in stack.modules():
        if hasattr(module, "bind_state"):
            module.bind_state({"kv": handle})

    def next_step():
        step = SubmoduleStep(segments=[Segment(1, "main", 3)], state={"kv": ArrayStep()})
        step.set_ctx(StepContext((1,), "decode", 0, False))
        handle.end_step(step)
        handle.begin_step(step)

    torch._dynamo.reset()
    frames, guards = [], []

    def backend(graph_module, inputs):
        frames.append(graph_module)
        return graph_module.forward

    optimize = torch._dynamo.optimize(backend, dynamic=False, guard_fail_fn=lambda f: guards.append(str(f.reason)))
    compiled = optimize(stack)
    hidden = torch.randn(3, config.hidden_size)
    next_step()
    compiled(hidden, mode="und")
    first = len(frames)
    next_step()
    with torch.compiler.set_stance("fail_on_recompile"):
        compiled(hidden, mode="und")
    return first, guards


@torch.no_grad()
def test_bagels_language_model_compiles_no_frame_per_layer():
    frames = [_compiled_frames(num_layers) for num_layers in (2, 4)]
    assert not [guard for _, guards in frames for guard in guards if "layer" in guard]
    assert frames[0][0] == frames[1][0] > 1, "one frame should serve every layer"
