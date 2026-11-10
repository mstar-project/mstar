"""Canary-Qwen, Qwen3-ASR, Parakeet, Fish S2-Pro and Qwen3-TTS build the specs, steps and layouts they wrote by hand.

Nothing in the engine changes when a model moves to declarations, because it sees the same objects. The hand-written
specs and steps below are the reference, copied from the models as they stood before the port, so the adapter can't pass
by agreeing with itself. Canary-Qwen's ``page_size`` is the one named difference, and its reference says why.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "test/modular")

import pytest
import torch
import yaml
from test_startup_graph_validation import _FACTORIES, _bare
from test_state_ports import _assert_equal_per_key, _assert_steps_equal, _bind, _published

from mstar.engine.node_dtype import resolve_model_dtype
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec, AttentionStep
from mstar.engine.resources.attn.ragged.config import RaggedAttentionConfig, RaggedAttentionSpec
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.position.config import PositionConfig, PositionSpec, PositionStep
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.sampler.config import SamplerSpec, SamplerStep, SamplingReqConfig
from mstar.engine.resources.spec import apply_yaml_overrides
from mstar.engine.resources.step import Segment, SubmoduleStep
from mstar.engine.state import Span
from mstar.engine.state.adapters import kv_layouts, node_resources, sampling_configs, state_steps
from mstar.engine.state.analysis import declaration_digest, state_declaration_problems
from mstar.model.base import LabelLayout, parse_autocast_dtype, serving_resources
from mstar.model.canary_qwen.canary_qwen_model import CanaryQwenModel
from mstar.model.canary_qwen.submodules import CanaryQwenLLMSubmodule
from mstar.model.components.aut_submodule import AUT_ENCODE
from mstar.model.components.fastconformer import FastConformerConfig
from mstar.model.components.transducer import TransducerConfig
from mstar.model.fish_s2_pro.components.dac import FishDACDecoder
from mstar.model.fish_s2_pro.components.incremental_dac import FRAMES, IncrementalDACDecoder, _state_width
from mstar.model.fish_s2_pro.fish_s2_pro_model import FishS2ProModel
from mstar.model.fish_s2_pro.submodules import ARSubmodule
from mstar.model.fish_s2_pro.submodules import CodecSubmodule as FishCodecSubmodule
from mstar.model.parakeet.config import ParakeetModelConfig
from mstar.model.parakeet.parakeet_model import ParakeetModel
from mstar.model.parakeet.submodules import TransducerSubmodule
from mstar.model.qwen3_asr.config import AUT_ATTN
from mstar.model.qwen3_asr.qwen3_asr_model import Qwen3ASRModel
from mstar.model.qwen3_asr.submodules import Qwen3ASREncoderSubmodule, Qwen3ASRLLMSubmodule
from mstar.model.qwen3_tts.components.incremental_codec import _DILATIONS, _KERNEL, _PRE_CONV_KERNEL
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.model.qwen3_tts.submodules import DEPTH_LOOP_REGION, TalkerSubmodule
from mstar.model.qwen3_tts.submodules import CodecSubmodule as TTSCodecSubmodule
from mstar.model.registry import MODEL_REGISTRY
from mstar.model.submodule_base import ARNodeInputs, NodeInputs, NodeSubmodule, engine_step
from mstar.validation import DeploymentError, validate_deployment


def _canary(tmp_path=None, monkeypatch=None) -> CanaryQwenModel:
    return _FACTORIES["CanaryQwenModel"]("canary_qwen", {}, tmp_path, monkeypatch)


def _asr() -> Qwen3ASRModel:
    return _FACTORIES["Qwen3ASRModel"]("qwen3_asr", {}, None, None)


def _parakeet(name: str = "parakeet") -> ParakeetModel:
    transducer = TransducerConfig(num_prompts=128 if name == "nemotron_asr" else 0)
    return _bare(ParakeetModel, config=ParakeetModelConfig(name, FastConformerConfig(), transducer))


def _fish() -> FishS2ProModel:
    return _FACTORIES["FishS2ProModel"]("fish_s2_pro", {}, None, None)


def _tts(name: str = "qwen3_tts", codec_dtype: str = "float32") -> Qwen3TTSModel:
    return _FACTORIES["Qwen3TTSModel"](name, {"codec_dtype": codec_dtype}, None, None)


def _node(cls, **attrs) -> NodeSubmodule:
    """``cls`` with what its step declarations read, and nothing that needs weights."""
    node = cls.__new__(cls)
    NodeSubmodule.__init__(node)
    for name, value in attrs.items():
        setattr(node, name, value)
    return node


def _canary_reference(model, page_size=None) -> list:
    llm = model.config.llm
    return [
        KVSpec(
            resource_key="kv_cache", nodes={"llm"},
            config=PagedKVConfig(
                num_layers=llm.num_hidden_layers, num_kv_heads=llm.num_key_value_heads, head_dim=llm.head_dim,
                max_seq_len=model.MAX_SEQ_LEN, num_qo_heads=llm.num_attention_heads, page_size=page_size,
                max_num_pages=1025,
            ),
        ),
        AttentionSpec(resource_key="attn", nodes={"llm"}, config=AttentionConfig(kv_cache="kv_cache")),
        PositionSpec(
            resource_key="rope", nodes={"llm"}, config=PositionConfig(kv_cache="kv_cache", rope_theta=llm.rope_theta),
        ),
        SamplerSpec(resource_key="sampler", nodes={"llm"}, vocab_size=llm.vocab_size, enable_repetion_penalty=False),
    ]


def _asr_reference(model) -> list:
    text, audio = model.config.text, model.config.audio
    return [
        RaggedAttentionSpec(
            resource_key="aut_attn", nodes={"audio_encoder"},
            config=RaggedAttentionConfig(
                num_qo_heads=audio.encoder_attention_heads, num_kv_heads=audio.encoder_attention_heads,
                head_dim=audio.head_dim, max_segments_per_request=model.max_windows_per_request,
            ),
        ),
        KVSpec(
            resource_key="kv_cache", nodes={"LLM"},
            config=PagedKVConfig(
                num_layers=text.num_hidden_layers, num_kv_heads=text.num_key_value_heads, head_dim=text.head_dim,
                max_seq_len=text.max_position_embeddings, num_qo_heads=text.num_attention_heads,
                max_num_pages=model.KV_PAGES,
            ),
        ),
        AttentionSpec(
            resource_key="attn", nodes={"LLM"}, config=AttentionConfig(kv_cache="kv_cache", decode_tensor_cores=False),
        ),
        PositionSpec(
            resource_key="rope", nodes={"LLM"}, config=PositionConfig(kv_cache="kv_cache", rope_theta=text.rope_theta),
        ),
        SamplerSpec(resource_key="sampler", nodes={"LLM"}, vocab_size=text.vocab_size, enable_repetion_penalty=False),
    ]


def _parakeet_reference(model) -> list:
    t = model.config.transducer
    layers = (t.num_layers, t.hidden_size)
    return [RecurrentStateSpec(
        resource_key="transducer_state", nodes={"transducer"},
        config=RecurrentStateConfig(
            num_layers=1,
            blocks={
                "h": RecurrentBlockConfig(layers, torch.float32),
                "c": RecurrentBlockConfig(layers, torch.float32),
                "dec_out": RecurrentBlockConfig((t.hidden_size,), torch.float32),
                "cursor": RecurrentBlockConfig((), torch.int64),
                "symbols": RecurrentBlockConfig((), torch.int64),
            },
            max_slots=model.MAX_CONCURRENT_REQUESTS + 1,
        ),
    )]


def _fish_codec_blocks(cfg) -> dict[str, RecurrentBlockConfig]:
    with torch.device("meta"):
        inc = IncrementalDACDecoder(FishDACDecoder(cfg))
    blocks = {FRAMES: RecurrentBlockConfig(shape=(1,), dtype=torch.long)}
    for layer in range(len(inc.decoder.quantizer.post_module.layers)):
        for kv in "kv":
            blocks[f"{kv}{layer}"] = RecurrentBlockConfig((inc.heads, inc.cache, inc.head_dim), torch.float32)
    for name, module in inc.convs:
        blocks[name] = RecurrentBlockConfig((_state_width(module), module.conv.in_channels), torch.float32)
    return blocks


def _fish_reference(model) -> list:
    slow = model.config.slow_ar
    return [
        KVSpec(
            resource_key="slow_ar_kv", nodes={"AR"},
            config=PagedKVConfig(
                num_layers=slow.n_layer, num_kv_heads=slow.n_local_heads, head_dim=slow.head_dim,
                max_seq_len=slow.max_seq_len, num_qo_heads=slow.n_head,
            ),
        ),
        AttentionSpec(resource_key="slow_ar_attn", nodes={"AR"}, config=AttentionConfig(kv_cache="slow_ar_kv")),
        PositionSpec(
            resource_key="slow_ar_pos", nodes={"AR"},
            config=PositionConfig(
                kv_cache="slow_ar_kv", interleave=True, rope_theta=slow.rope_base, rotary_dim=slow.head_dim,
                rope_table_dtype=torch.bfloat16,
            ),
        ),
        SamplerSpec(
            resource_key="semantic_sampler", nodes={"AR"}, vocab_size=model.config.semantic_head_size,
            enable_repetion_penalty=True,
        ),
        RecurrentStateSpec(
            resource_key="codec_state", nodes={"Codec"},
            config=RecurrentStateConfig(
                num_layers=1, blocks=_fish_codec_blocks(model.config.codec), max_slots=model.MAX_CODEC_STREAMS + 1,
            ),
        ),
    ]


def _tts_codec_blocks(config, dtype: torch.dtype) -> dict[str, RecurrentBlockConfig]:
    cache = config.sliding_window - 1
    blocks = {"frames": RecurrentBlockConfig(shape=(1,), dtype=torch.long)}
    for layer in range(config.num_hidden_layers):
        for kv in "kv":
            blocks[f"{kv}{layer}"] = RecurrentBlockConfig(
                shape=(config.num_attention_heads, cache, config.head_dim), dtype=dtype,
            )

    def conv(name, channels, width):
        shape = (channels, width) if name == "pre_conv" else (width, channels)
        blocks[name] = RecurrentBlockConfig(shape=shape, dtype=dtype)

    conv("pre_conv", config.codebook_dim, _PRE_CONV_KERNEL - 1)
    for i in range(len(config.upsampling_ratios)):
        conv(f"up{i}.dwconv", config.latent_dim, _KERNEL - 1)
    conv("dec0", config.latent_dim, _KERNEL - 1)
    for b in range(1, len(config.upsample_rates) + 1):
        conv(f"dec{b}.upsample", config.decoder_dim // 2 ** (b - 1), 1)
        for r, dilation in enumerate(_DILATIONS):
            conv(f"dec{b}.res{r}", config.decoder_dim // 2 ** b, (_KERNEL - 1) * dilation)
    conv("final", config.decoder_dim // 2 ** len(config.upsample_rates), _KERNEL - 1)
    return blocks


def _tts_reference(model) -> list:
    talker = model.config.talker
    return [
        KVSpec(
            resource_key="talker_kv", nodes={"Talker"},
            config=PagedKVConfig(
                num_layers=talker.num_hidden_layers, num_kv_heads=talker.num_key_value_heads, head_dim=talker.head_dim,
                max_seq_len=talker.max_position_embeddings, num_qo_heads=talker.num_attention_heads,
            ),
        ),
        AttentionSpec(
            resource_key="talker_attn", nodes={"Talker"},
            config=AttentionConfig(kv_cache="talker_kv", decode_tensor_cores=False),
        ),
        PositionSpec(
            resource_key="talker_pos", nodes={"Talker"},
            config=PositionConfig(kv_cache="talker_kv", rope_theta=talker.rope_theta),
        ),
        SamplerSpec(
            resource_key="talker_sampler", nodes={"Talker"}, vocab_size=talker.vocab_size,
            enable_repetion_penalty=True,
        ),
        SamplerSpec(
            resource_key="code_predictor", nodes={"Talker"}, vocab_size=talker.code_predictor.vocab_size,
            enable_repetion_penalty=False,
        ),
        RecurrentStateSpec(
            resource_key="codec_state", nodes={"Codec"},
            config=RecurrentStateConfig(
                num_layers=1, blocks=_tts_codec_blocks(model.config.codec, model.codec_dtype),
                max_slots=model.MAX_CODEC_STREAMS + 1,
            ),
        ),
    ]


MODELS = {
    "canary_qwen": (_canary, _canary_reference),
    "qwen3_asr": (_asr, _asr_reference),
    "parakeet": (_parakeet, _parakeet_reference),
    "nemotron_asr": (lambda: _parakeet("nemotron_asr"), _parakeet_reference),
    "fish_s2_pro": (_fish, _fish_reference),
    "qwen3_tts": (_tts, _tts_reference),
    "qwen3_tts_1p7b": (lambda: _tts("qwen3_tts_1p7b"), _tts_reference),
    "qwen3_tts_voicedesign": (lambda: _tts("qwen3_tts_voicedesign"), _tts_reference),
    "qwen3_tts_base": (lambda: _tts("qwen3_tts_base"), _tts_reference),
    "qwen3_tts_bf16_codec": (lambda: _tts(codec_dtype="bfloat16"), _tts_reference),
}
# What a deployment with no `resources:` block of its own could set.
OVERRIDES = {
    "canary_qwen": {"kv_cache": {"max_num_pages": 512}, "attn": {"flashinfer_backend": "fa2"}},
    "qwen3_asr": {"kv_cache": {"max_num_pages": 2048}, "attn": {"decode_tensor_cores": True}},
    "parakeet": {"transducer_state": {"max_slots": 33}},
    "nemotron_asr": {"transducer_state": {"max_slots": 33}},
    "fish_s2_pro": {"slow_ar_attn": {"flashinfer_backend": "fa2"}, "codec_state": {"max_slots": 33}},
    "qwen3_tts": {"talker_kv": {"max_num_pages": 1024}, "codec_state": {"max_slots": 512}},
}
OVERRIDES |= dict.fromkeys(("qwen3_tts_1p7b", "qwen3_tts_voicedesign", "qwen3_tts_base", "qwen3_tts_bf16_codec"),
                           OVERRIDES["qwen3_tts"])


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_ones(name):
    build, reference = MODELS[name]
    model = build()
    _assert_equal_per_key(node_resources(model), reference(model))


@pytest.mark.parametrize("name", MODELS)
def test_specs_equal_the_hand_written_ones_after_a_deployments_overrides(name):
    build, reference = MODELS[name]
    model = build()
    deployment = {"resources": OVERRIDES[name]}
    expected = reference(model)
    apply_yaml_overrides(expected, deployment)
    _assert_equal_per_key(serving_resources(model, deployment), expected)


def _published_cases() -> list:
    classes = ("CanaryQwenModel", "Qwen3ASRModel", "ParakeetModel", "FishS2ProModel", "Qwen3TTSModel")
    return [path for model_class in classes for path in _published(model_class)]


def test_every_published_config_of_the_five_is_covered():
    stems = {path.stem for path in _published_cases()}
    assert stems == {
        "canary_qwen", "qwen3_asr", "qwen3_asr_split", "qwen3_asr_realtime", "qwen3_asr_realtime_split", "parakeet",
        "nemotron_asr", "fish_s2_pro", "qwen3tts", "qwen3tts_split", "qwen3tts_1p7b", "qwen3tts_1p7b_split",
        "qwen3tts_base", "qwen3tts_voicedesign",
    }, "a config here that the port never compares would change with no test noticing"


@pytest.mark.parametrize("path", _published_cases(), ids=lambda path: path.stem)
def test_specs_equal_the_hand_written_ones_on_every_published_config(path, tmp_path, monkeypatch):
    deployment = yaml.safe_load(path.read_text())
    name = deployment["model"]
    model_class = MODEL_REGISTRY[name][1]
    model = _FACTORIES[model_class](name, deployment.get("model_kwargs") or {}, tmp_path, monkeypatch)
    expected = {
        "CanaryQwenModel": _canary_reference, "Qwen3ASRModel": _asr_reference, "ParakeetModel": _parakeet_reference,
        "FishS2ProModel": _fish_reference, "Qwen3TTSModel": _tts_reference,
    }[model_class](model)
    apply_yaml_overrides(expected, deployment)
    kv_dtype = parse_autocast_dtype(resolve_model_dtype(model, deployment))
    _assert_equal_per_key(serving_resources(model, deployment), expected, kv_dtype)


def test_canary_qwen_differs_only_in_a_page_size_that_means_the_same_pages():
    """The hand-written spec set 128 where the declaration leaves ``page_size`` None, whose default is 128. With no
    window the two give the same pages, arena and bytes, and `configs/canary_qwen.yaml` sets only ``max_num_pages``."""
    model = _canary()
    hand_written = _canary_reference(model, page_size=128)[0].config
    declared = next(spec for spec in node_resources(model) if spec.resource_key == "kv_cache").config
    assert declared.page_size is None and hand_written.page_size == 128
    assert (declared.tokens_per_page, declared.num_units) == (hand_written.tokens_per_page, hand_written.num_units)
    assert declared.groups == hand_written.groups, "the arena's layout follows from the groups"


def test_sdpa_on_the_aut_read_is_refused_before_any_worker_boots():
    """The AuT packs its windows, and SDPA can't tell one window's queries from the next. The bindings refuse it where
    they resolve, so the deployment fails validation with its other problems, and not at each worker's bind."""
    path = Path(__file__).parents[2] / "configs" / "qwen3_asr.yaml"
    deployment = {**yaml.safe_load(path.read_text()), "resources": {AUT_ATTN: {"backend": "sdpa"}}}
    with pytest.raises(DeploymentError) as refused:
        validate_deployment(_asr(), deployment, str(path))
    (problem,) = refused.value.problems
    assert f"reads[{AUT_ATTN!r}] runs SDPA, which can't tell one segment of packed queries" in problem, (
        "validation would pass, and each worker would fail on its first packed window"
    )


def test_the_codec_values_allocate_in_each_codec_dtype():
    for dtype, name in ((torch.float32, "float32"), (torch.bfloat16, "bfloat16")):
        specs = {spec.resource_key: spec for spec in node_resources(_tts(codec_dtype=name))}
        blocks = specs["codec_state"].config.blocks
        assert blocks["k0"].dtype is dtype and blocks["frames"].dtype is torch.long


@pytest.mark.parametrize("name", MODELS)
def test_declarations_have_no_problems_and_one_digest(name):
    build, _ = MODELS[name]
    model = build()
    assert state_declaration_problems(model) == []
    assert declaration_digest(model.declare_state()) == declaration_digest(build().declare_state())


# Steps


def _rows(*spans) -> list[ARNodeInputs]:
    return [ARNodeInputs(input_seq_len=span) for span in spans]


def _reference_step(steps: dict, rids, spans, cg_key_info=None) -> SubmoduleStep:
    segments = [Segment(request_id=rid, label="main", span=span) for rid, span in zip(rids, spans, strict=True)]
    return SubmoduleStep(segments=segments, steps=steps, cg_key_info=cg_key_info)


@pytest.mark.parametrize("walk, spans", [("prefill", (7, 3)), ("decode", (1, 1))])
def test_canary_qwen_steps_equal_the_hand_written_ones(walk, spans):
    node = _bind(_node(CanaryQwenLLMSubmodule), _canary(), "llm")
    step = engine_step(node, walk, [1, 2], _rows(*spans))
    _assert_steps_equal(step, _reference_step({
        "kv_cache": KVStep(), "attn": AttentionStep(causal=True), "sampler": SamplerStep(apply_penalty=False),
        "rope": PositionStep(),
    }, [1, 2], spans))


@pytest.mark.parametrize("walk, spans", [("prefill", (7, 3)), ("decode", (1, 1))])
def test_qwen3_asr_llm_steps_equal_the_hand_written_ones(walk, spans):
    node = _bind(_node(Qwen3ASRLLMSubmodule), _asr(), "LLM")
    step = engine_step(node, walk, [1, 2], _rows(*spans))
    _assert_steps_equal(step, _reference_step({
        "kv_cache": KVStep(), "attn": AttentionStep(causal=True), "sampler": SamplerStep(apply_penalty=False),
        "rope": PositionStep(),
    }, [1, 2], spans))


def _asr_encoder(model) -> Qwen3ASREncoderSubmodule:
    encoder = type("E", (), {"config": model.config.audio})()
    return _bind(_node(Qwen3ASREncoderSubmodule, encoder=encoder, array="aut_tokens"), model, "audio_encoder")


def test_qwen3_asr_encoder_step_equals_the_hand_written_one():
    model = _asr()
    audio = model.config.audio
    tokens = (audio.window_tokens + 5, 9)
    step = engine_step(_asr_encoder(model), "prefill", [1, 2], [NodeInputs(input_seq_len=n) for n in tokens])
    windows = [(rid, w) for rid, n in zip([1, 2], tokens, strict=True) for w in audio.window_lengths(n)]
    assert len(windows) == 3, "the first clip should span two windows, and the step lays out each"
    reference = _reference_step(
        {"aut_attn": AttentionStep(causal=False)}, [rid for rid, _ in windows], [w for _, w in windows],
    )
    assert step.state and step.segments == reference.segments
    attention = step.steps["aut_attn"]
    assert attention.causal is False and tuple(attention.segments) == tuple(reference.segments)


def test_qwen3_asr_encoder_leased_declares_none_as_it_did():
    model = _asr()
    step = engine_step(
        _asr_encoder(model), "prefill", [1], [NodeInputs(input_seq_len=9)], piecewise_leases={AUT_ENCODE: object()},
    )
    assert step is None, "leased, the captured stack plans its own attention per replay"


@pytest.mark.parametrize("seq_lens", [[26, 26, 5], [64, 0, 0, 0]], ids=["capture", "padded replay"])
def test_qwen3_asr_region_step_equals_the_hand_written_one(seq_lens):
    node = _asr_encoder(_asr())
    config = node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[AUT_ENCODE]
    step = state_steps(config.declare_step([7], seq_lens), node.state_handles)
    segments = [Segment(request_id=7, label="main", span=n) for n in seq_lens]
    assert step.segments == segments, "every window names the first row, and zero-length rows pad to the bucket"
    assert step.steps["aut_attn"].causal is False and tuple(step.steps["aut_attn"].segments) == tuple(segments)


@pytest.mark.parametrize("name", ["parakeet", "nemotron_asr"])
@pytest.mark.parametrize("bucket", [512, 2048, 8192, None], ids=["512", "2048", "8192", "eager"])
def test_parakeet_iteration_step_equals_the_hand_written_one(name, bucket):
    build, _ = MODELS[name]
    node = _bind(_node(TransducerSubmodule, _start={}), build(), "transducer")
    inputs = [NodeInputs(input_seq_len=1, resource_step_info=bucket) for _ in range(2)]
    step = engine_step(node, "transcribe", [1, 2], inputs)
    _assert_steps_equal(step, _reference_step(
        {"transducer_state": RecurrentStep()}, [1, 2], (1, 1), cg_key_info=bucket,
    ))


def test_parakeet_mixed_buckets_declare_no_key_as_they_did():
    node = _bind(_node(TransducerSubmodule, _start={}), _parakeet(), "transducer")
    inputs = [NodeInputs(input_seq_len=1, resource_step_info=bucket) for bucket in (512, 2048)]
    assert engine_step(node, "transcribe", [1, 2], inputs).cg_key_info is None


@pytest.mark.parametrize("walk, spans", [("ar_prefill", (9, 4)), ("ar_prefill_clone", (30, 4)), ("ar_decode", (1, 1))])
def test_fish_ar_steps_equal_the_hand_written_ones(walk, spans):
    node = _bind(_node(ARSubmodule), _fish(), "AR")
    step = engine_step(node, walk, [1, 2], _rows(*spans))
    _assert_steps_equal(step, _reference_step({
        "slow_ar_kv": KVStep(), "slow_ar_attn": AttentionStep(causal=True), "slow_ar_pos": PositionStep(),
        "semantic_sampler": SamplerStep(apply_penalty=True),
    }, [1, 2], spans))


def test_fish_codec_step_equals_the_hand_written_one():
    node = _bind(_node(FishCodecSubmodule, windows=[4, 8, 16, 32]), _fish(), "Codec")
    inputs = [ARNodeInputs(tensor_inputs={"codec_tokens": torch.zeros(frames, 10)}) for frames in (5, 2)]
    step = engine_step(node, "codec_chunk", [1, 2], inputs)
    _assert_steps_equal(step, _reference_step({"codec_state": RecurrentStep()}, [1, 2], (1, 1), cg_key_info=8))


def _talker_steps() -> dict:
    return {
        "talker_kv": KVStep(), "talker_attn": AttentionStep(causal=True), "talker_pos": PositionStep(),
        "talker_sampler": SamplerStep(apply_penalty=True),
    }


def _talker(model: Qwen3TTSModel) -> TalkerSubmodule:
    return _bind(_node(TalkerSubmodule), model, "Talker")


@pytest.mark.parametrize("walk, spans", [
    ("talker_prefill", (9, 4)), ("talker_prefill_clone", (30, 4)), ("talker_decode", (1, 1)), ("talker_mixed", (9, 1)),
])
def test_qwen3_tts_talker_owns_the_depth_sampler_without_the_regions_lease(walk, spans):
    step = engine_step(_talker(_tts()), walk, [1, 2], _rows(*spans))
    _assert_steps_equal(step, _reference_step({
        **_talker_steps(), "code_predictor": SamplerStep(apply_penalty=False),
    }, [1, 2], spans))


@pytest.mark.parametrize("walk, spans", [
    ("talker_prefill", (9, 4)), ("talker_prefill_clone", (30, 4)), ("talker_decode", (1, 1)), ("talker_mixed", (9, 1)),
])
def test_qwen3_tts_talker_leaves_the_depth_sampler_to_a_leased_region(walk, spans):
    node = _talker(_tts())
    step = engine_step(node, walk, [1, 2], _rows(*spans), piecewise_leases={DEPTH_LOOP_REGION: object()})
    _assert_steps_equal(step, _reference_step(_talker_steps(), [1, 2], spans))


def test_qwen3_tts_depth_regions_step_declares_the_sampler_through_state():
    node = _talker(_tts())
    node.talker_config = _tts().config.talker
    node.model = type("M", (), {"model": type("L", (), {"codec_embedding": type("W", (), {"weight": torch.zeros(1)})})})
    region = node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[DEPTH_LOOP_REGION]
    declared = region.declare_step([1, 2], [1, 1])
    assert declared.state.keys() == {"code_predictor"}, "a region opens only the handles its own step lists"
    step = state_steps(declared, node.state_handles)
    _assert_steps_equal(step, _reference_step({"code_predictor": SamplerStep(apply_penalty=False)}, [1, 2], (1, 1)))
    assert step.state_handles == (node.state_handles["code_predictor"],)


def test_qwen3_tts_codec_step_equals_the_hand_written_one():
    node = _bind(_node(TTSCodecSubmodule), _tts(), "Codec")
    inputs = [ARNodeInputs(kwargs={"bucket": bucket}) for bucket in (3, 8)]
    step = engine_step(node, "codec_chunk", [1, 2], inputs)
    _assert_steps_equal(step, _reference_step({"codec_state": RecurrentStep()}, [1, 2], (1, 1), cg_key_info=8))


# Layouts and request settings


def _assert_layout_converts(model, metadata: dict, array: str, span: Span, loop: str) -> None:
    assert "kv_layout" not in metadata, "a ported model returns its layout under state_layout"
    converted = kv_layouts(model.declare_state(), metadata["state_layout"], {})
    assert converted == {array: {"main": LabelLayout(spans=(span,), growth=((loop, 1),))}}


def test_canary_qwen_layout_converts_to_the_hand_written_one():
    from test_canary_qwen_model import _model_with_word_tokenizer

    model = _model_with_word_tokenizer()
    model.config = _canary().config
    model.audio_token_id = 3
    model.prompt_ids = lambda frames, instruction: list(range(frames + 4))
    out = model.process_prompt(None, ["audio", "text"], ["text"], tensors={"audio_inputs": [torch.zeros(16000)]})
    n = out.new_input_tensors["text_inputs"][0].shape[0]
    span = Span("rows", "prefill", n, n, "text_inputs")
    _assert_layout_converts(model, out.metadata, "kv_cache", span, "decode_loop")


def test_qwen3_asr_layout_converts_to_the_hand_written_one():
    from test_qwen3_asr_model import _make_model

    model = _make_model()
    out = model.process_prompt(None, ["audio", "text"], ["text"], tensors={"audio_inputs": [torch.zeros(16000)]})
    n = out.new_input_tensors["text_inputs"][0].shape[0]
    span = Span("rows", "prefill", n, n, "text_inputs")
    _assert_layout_converts(model, out.metadata, "kv_cache", span, "decode_loop")


@pytest.mark.parametrize("clone", [False, True])
def test_fish_layout_converts_to_the_hand_written_one(clone):
    from test_fish_s2_pro_model import tiny_model

    model = tiny_model()
    tensors = {"audio_inputs": [torch.zeros(20 * model.config.codec.hop_length + 1)]} if clone else None
    out = model.process_prompt("hi", ["text", "audio"] if clone else ["text"], ["audio"], tensors, ref_text="r")
    n = out.new_input_tensors["prompt_ids"][0].shape[0] + (out.metadata["prompt_layout"][1] if clone else 0)
    walk = "ar_prefill_clone" if clone else "ar_prefill"
    span = Span("rows" if clone else "ids", walk, n, n, "prompt_ids")
    _assert_layout_converts(model, out.metadata, "slow_ar_kv", span, "ar_decode_loop")


def test_qwen3_tts_layout_converts_to_the_hand_written_one():
    from test_qwen3_tts_model import _variant_model

    from mstar.model.qwen3_tts.config import LANGUAGE_ID_KEY, PROMPT_LAYOUT_KEY, SPEAKER_ID_KEY
    from mstar.model.qwen3_tts.submodules import prefill_rows

    model = _variant_model("custom_voice")
    out = model.process_prompt("hello big world", ["text"], ["audio"], language="English")
    n = prefill_rows(out.metadata[PROMPT_LAYOUT_KEY], out.metadata[LANGUAGE_ID_KEY], out.metadata[SPEAKER_ID_KEY] >= 0)
    span = Span("rows", "talker_prefill", n, n, "text_inputs")
    _assert_layout_converts(model, out.metadata, "talker_kv", span, "talker_decode_loop")


def test_canary_qwen_request_settings_equal_the_hand_written_config():
    model = _canary()
    kwargs = {"temperature": 0.5, "top_p": 0.9, "ignore_eos": True}
    configs = sampling_configs(model.get_request_resource_configs({}, kwargs), model.declare_state())
    assert configs == {"sampler": SamplingReqConfig(temperature=0.5, top_p=0.9, ignore_eos=True)}
    configs = sampling_configs(model.get_request_resource_configs({}, {}), model.declare_state())
    assert configs == {"sampler": SamplingReqConfig(temperature=0.0, top_p=1.0, ignore_eos=False)}, "NeMo is greedy"


def test_qwen3_asr_request_settings_equal_the_hand_written_config():
    model = _asr()
    configs = sampling_configs(model.get_request_resource_configs({}, {}), model.declare_state())
    assert configs == {"sampler": SamplingReqConfig(temperature=0.0, top_p=1.0, ignore_eos=False)}


def test_fish_request_settings_equal_the_hand_written_config():
    model = _fish()
    generation = model.config.generation
    configs = sampling_configs(model.get_request_resource_configs({}, {"ignore_eos": True}), model.declare_state())
    assert configs == {"semantic_sampler": SamplingReqConfig(
        temperature=generation.temperature, top_k=generation.top_k, top_p=generation.top_p,
        repetition_penalty=generation.repetition_penalty, penalize_prompt=False, ignore_eos=True,
    )}


def test_qwen3_tts_request_settings_equal_the_hand_written_config():
    model = _tts()
    g = model.config.generation
    configs = sampling_configs(model.get_request_resource_configs({}, {}), model.declare_state())
    assert configs == {
        "talker_sampler": SamplingReqConfig(
            temperature=g.temperature if g.do_sample else 0.0, top_k=g.top_k, top_p=g.top_p,
            repetition_penalty=g.repetition_penalty, ignore_eos=False,
        ),
        "code_predictor": SamplingReqConfig(
            temperature=g.subtalker_temperature if g.subtalker_dosample else 0.0, top_k=g.subtalker_top_k,
            top_p=g.subtalker_top_p,
        ),
    }


def test_a_depth_sampler_declares_the_draws_of_the_groups_past_the_first():
    declared = {d.name: d for d in _tts().declare_state()}
    assert declared["code_predictor"].draws == 15, "a step with fewer would leave the depth loop's stream short"
    assert declared["talker_sampler"].draws == 1
