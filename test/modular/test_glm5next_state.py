"""GLM-5.3-Flash builds the specs, steps and layouts it wrote by hand, apart from three named differences.

The hand-written specs and steps below are the reference, copied from the model as it stood before the port, so the
adapter can't pass by agreeing with itself. The differences: the KDA pool names the latent pool as its ``kv_cache``,
which does nothing at ``prefix_slots`` 0; MTP's draft seed is its own Value's pool, where the KDA pool held it as two
blocks; and on ``reduced`` the latent pool keeps ``64 * (k + 1)`` rows where the hand-written one kept
``512 * (k + 1)``, since the declaration holds the context to ``min(index_topk, max_seq_len)``.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
import yaml
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _published, _StubResource

from mstar.engine.node_dtype import resolve_model_dtype
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep, AttnBackend
from mstar.engine.resources.kv.config import KVLayout, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.linear_attn.config import (
    LinearAttnBackend,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
)
from mstar.engine.resources.recurrent.config import (
    DeltaNetGeometry,
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state import Span
from mstar.engine.state.adapters import StateSetup, kv_layouts, node_handles, node_resources
from mstar.engine.state.analysis import state_declaration_problems
from mstar.model.base import LabelLayout, enabled_resources, parse_autocast_dtype, serving_resources
from mstar.model.glm5_next.components.causal_lm import Glm5NextForCausalLM
from mstar.model.glm5_next.config import ATTN, KDA, KDA_STATE, KV_CACHE, MTP_SEED, PROMPT_LEN, SAMPLER
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel
from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule
from mstar.model.submodule_base import ARNodeInputs, engine_step

_MODELS = pytest.mark.parametrize("variant, k", [(None, 0), (None, 3), ("reduced", 0), ("reduced", 3)])


def _model(variant: str | None = None, k: int = 0, **kwargs) -> Glm5NextModel:
    return Glm5NextModel("x", mtp_num_draft_tokens=k, **({"config_variant": variant} if variant else {}), **kwargs)


def _hand_written(model: Glm5NextModel) -> list:
    """``get_node_resources`` as the model wrote it, with ``kv_rows`` and ``mla_cache_kpe`` written out."""
    cfg, k = model.config, model.config.mtp_num_draft_tokens
    kda_blocks = DeltaNetGeometry(
        num_k_heads=cfg.linear_num_heads, num_v_heads=cfg.linear_num_heads, head_k_dim=cfg.linear_head_dim,
        head_v_dim=cfg.linear_head_dim, conv_kernel_size=cfg.linear_conv_kernel_size,
    ).to_blocks(conv_dtype=model.kda_conv_dtype, speculative_tokens=k)
    if k:
        kda_blocks["mtp_hidden"] = RecurrentBlockConfig(
            shape=(cfg.hidden_size,), dtype=model.kda_conv_dtype, per_layer=False,
        )
        kda_blocks["mtp_draft"] = RecurrentBlockConfig(shape=(1,), dtype=torch.long, per_layer=False)
    kpe = 64 if k else 0
    return [
        KVSpec(resource_key=KV_CACHE, nodes={"LLM"}, config=PagedKVConfig(
            num_layers=len(cfg.full_attn_layer_indices) + (1 if k else 0), num_kv_heads=1,
            head_dim=cfg.kv_lora_rank + kpe, max_seq_len=cfg.max_seq_len * (k + 1),
            num_qo_heads=cfg.num_attention_heads, layout=KVLayout.MLA, kv_lora_rank=cfg.kv_lora_rank,
            qk_rope_head_dim=kpe,
        )),
        AttentionSpec(resource_key=ATTN, nodes={"LLM"}, config=AttentionConfig(
            kv_cache=KV_CACHE, backend=AttnBackend.FLASHINFER_MLA, sm_scale=cfg.qk_head_dim ** -0.5,
        )),
        SamplerSpec(
            resource_key=SAMPLER, nodes={"LLM"}, vocab_size=cfg.vocab_size, enable_repetition_penalty=True,
            enable_min_p=bool(model.sampling_defaults.get("min_p")),
        ),
        RecurrentStateSpec(resource_key=KDA_STATE, nodes={"LLM"}, config=RecurrentStateConfig(
            num_layers=len(cfg.kda_layer_indices), blocks=kda_blocks, max_slots=model.kda_max_requests + 1,
        )),
        LinearAttnSpec(resource_key=KDA, nodes={"LLM"}, config=LinearAttnConfig(
            recurrent_state=KDA_STATE, variant=LinearAttnVariant.KDA, backend=LinearAttnBackend.TRITON,
        )),
    ]


def _reference(model: Glm5NextModel) -> list:
    """The hand-written specs with the three named differences."""
    cfg, k = model.config, model.config.mtp_num_draft_tokens
    specs = _hand_written(model)
    by_key = {spec.resource_key: spec for spec in specs}
    by_key[KV_CACHE].config.max_seq_len = cfg.context_limit * (k + 1)
    kda = by_key[KDA_STATE].config
    kda.kv_cache = KV_CACHE
    if k:
        del kda.blocks["mtp_hidden"], kda.blocks["mtp_draft"]
        specs.append(RecurrentStateSpec(resource_key=MTP_SEED, nodes={"LLM"}, config=RecurrentStateConfig(
            num_layers=1, max_slots_of=KDA_STATE, blocks={
                "hidden": RecurrentBlockConfig(shape=(cfg.hidden_size,), dtype=model.kda_conv_dtype),
                "draft": RecurrentBlockConfig(shape=(), dtype=torch.long),
            },
        )))
    return specs


@_MODELS
def test_specs_equal_the_hand_written_ones_apart_from_the_named_differences(variant, k):
    model = _model(variant, k)
    _assert_equal_per_key(enabled_resources(model, {}, StateSetup()), _reference(model))


def test_the_published_configs_are_the_two_shipped_ones():
    assert {path.stem for path in _published("Glm5NextModel")} == {"glm5_next_tp8", "glm5_next_tp8_mtp"}, (
        "a config here that the port never compares would change with no test noticing"
    )


@pytest.mark.parametrize("path", _published("Glm5NextModel"), ids=lambda path: path.stem)
def test_specs_equal_the_hand_written_ones_on_every_published_config(path):
    deployment = yaml.safe_load(path.read_text())
    model = Glm5NextModel("x", **deployment["model_kwargs"])
    expected = _reference(model)
    apply_yaml_overrides(expected, deployment)
    kv_dtype = parse_autocast_dtype(resolve_model_dtype(model, deployment))
    _assert_equal_per_key(serving_resources(model, deployment), expected, kv_dtype)


@_MODELS
def test_declarations_have_no_problems(variant, k):
    model = _model(variant, k)
    assert state_declaration_problems(model) == []


@pytest.mark.parametrize("k", [0, 3])
def test_the_layout_converts_to_the_hand_written_one(k):
    model = _model(k=k, tokenizer_mode="byte")
    metadata = model.process_prompt("hello", ["text"], ["text"]).metadata
    assert "kv_layout" not in metadata, "a ported model returns its layout under state_layout"
    converted = kv_layouts(model.declare_state(), metadata["state_layout"], {})
    assert converted == {KV_CACHE: {"main": LabelLayout(
        spans=(Span("ids", "prefill", 5, 5, "text_inputs"),), growth=(("decode_loop", k + 1),),
    )}}


class _Stub(_StubResource):
    """Answers what the layers and the submodule ask at bind: the KDA resource takes a layer's gates, and a pool
    counts its free slots."""

    num_free_slots = 4

    def bind_layer(self, **gates):
        pass


@pytest.mark.parametrize("k", [0, 3])
def test_the_prompt_length_rides_the_decode_step_s_metadata(k):
    """``process_prompt`` hands the walk plan the prompt's length, which the decode steps' stop checks and guard read
    from their metadata, and the data worker drops a client's value under that key."""
    model = _model(k=k, tokenizer_mode="byte")
    metadata = model.process_prompt("hello", ["text"], ["text"]).metadata
    prefill, decode = model.walk_plan("LLM", ["text"], ["text"], {}, {PROMPT_LEN: metadata[PROMPT_LEN]})
    assert PROMPT_LEN not in prefill.step_metadata and decode.step_metadata[PROMPT_LEN] == 5
    assert PROMPT_LEN in model.INTERNAL_KWARGS, "a client could set the length every stop check reads"


def _node(model: Glm5NextModel) -> Glm5NextLLMSubmodule:
    node = Glm5NextLLMSubmodule(Glm5NextForCausalLM(model.config), model.config)
    resources = {spec.resource_key: _Stub() for spec in node_resources(model)}
    node.bind_node_state(node_handles(model, "LLM", resources))
    return node


@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("walk", ["prefill", "decode"])
def test_steps_equal_the_hand_written_ones_apart_from_the_seed(walk, k):
    model = _model("reduced", k, kda_conv_dtype=torch.float32)
    spans = (7, 3) if walk == "prefill" else (k + 1, k + 1)
    rows = [ARNodeInputs(input_ids=torch.arange(span), input_seq_len=span) for span in spans]
    prompt = {rid: row.input_ids for rid, row in zip([1, 2], rows, strict=True)} if walk == "prefill" else {}
    steps = {
        KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True),
        SAMPLER: SamplerStep(apply_penalty=True, prefill_tracked_tokens=prompt),
        KDA_STATE: RecurrentStep(), KDA: LinearAttnStep(),
    }
    if k and walk == "decode":
        # the verify and the drafts are argmaxes in the forward
        del steps[SAMPLER]
        steps[KDA] = LinearAttnStep(speculative=True)
    if k:
        steps[MTP_SEED] = RecurrentStep()
    segments = [Segment(request_id=rid, label="main", span=span) for rid, span in zip([1, 2], spans, strict=True)]
    step = engine_step(_node(model), walk, [1, 2], rows)
    _assert_steps_equal(step, SubmoduleStep(segments=segments, steps=steps))


@pytest.mark.parametrize("k", [0, 3])
def test_only_a_decode_step_is_input_free(k):
    """Under async scheduling the pre-plan declares and admits each decode step over the capture's template rows. The
    GPU thread runs that step when the model says the step reads nothing else, rather than declaring and admitting it
    again before the launch."""
    node = _node(_model("reduced", k, kda_conv_dtype=torch.float32))
    template = ARNodeInputs(input_ids=torch.zeros(1, dtype=torch.long), input_seq_len=k + 1)
    row = ARNodeInputs(input_ids=torch.tensor([7]), input_seq_len=k + 1)
    declared, real = (node.declare_step("decode", ["a", "b"], [one, one]) for one in (template, row))
    assert (declared.segments, declared.state) == (real.segments, real.state)
    assert node.step_is_input_free("decode") and not node.step_is_input_free("prefill"), (
        "the GPU thread would declare and admit every decode step again before its launch"
    )
