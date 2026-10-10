"""Per-model halves of the generation-input contract:
the keys each model reads, its output limit, its checkpoint defaults, and the
inputs it refuses. Models with a general test module keep theirs there."""

from __future__ import annotations

import glob
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub.errors import EntryNotFoundError

import mstar.model.bagel.bagel_model as bagel_mod
from mstar.conductor.conductor import Conductor
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.model.bagel.bagel_model import BagelModel, check_request_kwargs
from mstar.model.base import MAX_OUTPUT_TOKENS
from mstar.model.higgs_audio.config import SAMPLER as HIGGS_SAMPLER
from mstar.model.higgs_audio.config import HiggsAudioModelConfig
from mstar.model.higgs_audio.higgs_audio_model import DEFAULT_SAMPLING as HIGGS_DEFAULT_SAMPLING
from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel
from mstar.model.higgs_audio.higgs_audio_model import sampling_defaults as higgs_sampling_defaults
from mstar.model.qwen3_5.config import SAMPLER as QWEN3_5_SAMPLER
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_5.qwen3_5_model import sampling_defaults as qwen3_5_sampling_defaults
from mstar.model.qwen3_omni.config import CODE_PRED_SAMPLER, TALKER_SAMPLER, THINKER_SAMPLER
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.whisper.config import SAMPLER as WHISPER_SAMPLER
from mstar.model.whisper.config import WhisperModelConfig
from mstar.model.whisper.whisper_model import DEFAULT_SAMPLING as WHISPER_DEFAULT_SAMPLING
from mstar.model.whisper.whisper_model import WhisperModel
from mstar.model.whisper.whisper_model import sampling_defaults as whisper_sampling_defaults

# ── bagel ───────────────────────────────────────────────────────────────────

BAGEL_CONFIG_DEFAULTS = {
    "temperature": 0.6, "top_k": 0, "top_p": 1.0,
    "repetition_penalty": 1.05, "ignore_eos": False,
}


def _bagel_model(**sampling) -> BagelModel:
    model = BagelModel.__new__(BagelModel)
    model.config = SimpleNamespace(vocab_size=32, num_timesteps=50, **BAGEL_CONFIG_DEFAULTS)
    model.sampling_defaults = {**BAGEL_CONFIG_DEFAULTS, **sampling}
    model._has_cfg_parallel = False
    model._image_gen_remote_handoff = False
    return model


def test_bagel_request_kwargs_cover_what_bagel_and_its_adapter_read():
    keys = _bagel_model().request_kwargs()
    assert {
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty",
        "ignore_eos", "penalize_prompt", "max_output_tokens",
        "cfg_text_scale", "cfg_img_scale", "cfg_interval", "cfg_renorm_type",
        "cfg_renorm_min", "think_mode", "width", "height", "image_preprocess",
    } <= keys
    assert "do_sample" not in keys and "size" not in keys


def test_bagel_limit_is_the_decode_loop_bound():
    model = _bagel_model()
    assert model.get_max_output_tokens_limit() == MAX_OUTPUT_TOKENS
    loop = model.get_graph_walk_graphs()["decode"]
    assert loop.max_iters == model.get_max_output_tokens_limit()


@pytest.mark.parametrize("kwargs", [
    {"cfg_renorm_type": "bogus"},
    {"cfg_text_scale": "4"},
    {"cfg_img_scale": True},
    {"cfg_img_scale": float("nan")},
    {"cfg_interval": [0.6, 0.4]},
    {"cfg_interval": [0.0, 1.5]},
    {"cfg_interval": 0.4},
    {"cfg_renorm_min": 2.0},
    {"think_mode": "false"},
    {"think_mode": 1},
    {"image_preprocess": "hf"},
    {"width": 1000},
    {"width": 2048},
    {"height": 0},
    {"height": 512.0},
    {"width": "512"},
])
def test_bagel_out_of_range_knobs_are_a_400(kwargs):
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        check_request_kwargs(kwargs)


def test_bagel_valid_knobs_pass():
    check_request_kwargs({
        "cfg_text_scale": 4, "cfg_img_scale": 1.0, "cfg_interval": (0.4, 1.0),
        "cfg_renorm_type": "text_channel", "cfg_renorm_min": 0.0,
        "think_mode": True, "image_preprocess": "vllm", "width": 512, "height": 1024,
    })
    check_request_kwargs(None)


def test_bagel_process_prompt_rejects_before_tokenizing():
    with pytest.raises(ValueError, match="cfg_renorm_type"):
        _bagel_model().process_prompt(None, ["text"], ["image"], cfg_renorm_type="x")


def test_bagel_sampling_forwards_min_p_and_penalize_prompt():
    sampling = _bagel_model().get_sampling_config(
        "LLM", {"min_p": 0.1, "penalize_prompt": False, "temperature": 0.0},
    )
    assert sampling.min_p == 0.1
    assert sampling.penalize_prompt is False
    assert sampling.temperature == 0.0
    assert sampling.repetition_penalty == 1.05


def test_bagel_unset_knobs_take_the_checkpoint_generation_config(tmp_path, monkeypatch):
    (tmp_path / "generation_config.json").write_text(json.dumps({
        "do_sample": True, "repetition_penalty": 1.05,
        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    }))
    monkeypatch.setattr(
        bagel_mod, "hf_hub_download",
        lambda **kw: str(tmp_path / kw["filename"]),
    )
    model = _bagel_model()
    model.sampling_defaults = model._load_sampling_defaults("repo", None)
    sampling = model.get_sampling_config("LLM", {"top_k": 5})
    assert (sampling.temperature, sampling.top_p, sampling.top_k) == (0.7, 0.8, 5)
    assert sampling.repetition_penalty == 1.05
    assert sampling.min_p == 0.0


def test_bagel_no_generation_config_keeps_config_defaults(monkeypatch):
    def missing(**kw):
        raise EntryNotFoundError("no file")

    monkeypatch.setattr(bagel_mod, "hf_hub_download", missing)
    model = _bagel_model()
    assert model._load_sampling_defaults("repo", None) == BAGEL_CONFIG_DEFAULTS


def _bagel_stop(walk: str, loop_count: int, max_tokens: int, token: int = 5) -> set[str]:
    import torch

    from mstar.model.bagel.submodules import LLMSubmodule
    from mstar.model.higgs_audio.config import SAMPLER

    llm = LLMSubmodule.__new__(LLMSubmodule)
    llm.eos_token_id = 9
    info = SimpleNamespace(
        graph_walk=walk, max_tokens=max_tokens,
        dynamic_loop_iter_counts={"decode_loop": loop_count},
        resource_configs={SAMPLER: SimpleNamespace(ignore_eos=True)},
    )
    return llm.check_stop("r", info, {"new_token": [torch.tensor([token])]})


def test_bagel_decode_stops_after_exactly_max_output_tokens():
    # the prefill emits token 1; decode step k (0-based count) emits token k + 2
    assert _bagel_stop("prefill_text", 0, 1) == set()
    assert _bagel_stop("decode", 0, 2) == {"decode_loop"}
    assert _bagel_stop("decode", 0, 3) == set()
    assert _bagel_stop("decode", 14, 16) == {"decode_loop"}



# ── higgs_audio ─────────────────────────────────────────────────────────────

def _higgs_model(defaults=None) -> HiggsAudioModel:
    model = object.__new__(HiggsAudioModel)
    model.config = HiggsAudioModelConfig()
    model.sampling_defaults = dict(HIGGS_DEFAULT_SAMPLING if defaults is None else defaults)
    return model


def _higgs_sampler(model, **kw) -> SamplingReqConfig:
    return model.get_request_resource_configs({}, kw)[HIGGS_SAMPLER]


def _higgs_spec(model):
    return next(s for s in model.get_node_resources() if s.resource_key == HIGGS_SAMPLER)


def test_higgs_request_kwargs_are_the_keys_it_reads():
    keys = _higgs_model().request_kwargs()
    for read in ("enable_thinking", "temperature", "top_p", "top_k", "min_p",
                 "repetition_penalty", "penalize_prompt", "ignore_eos", "max_output_tokens"):
        assert read in keys
    # sent by the transcription adapter but never read
    assert "language" not in keys and "timestamps" not in keys


def test_higgs_limit_is_the_decode_loop_bound():
    model = _higgs_model()
    assert model.get_max_output_tokens_limit() == 1024
    assert model.get_graph_walk_graphs()["decode"].max_iters == 1024
    c = Conductor.__new__(Conductor)
    c.model = model
    assert c._max_output_tokens({}) == 1024
    assert c._max_output_tokens({"max_output_tokens": 7}) == 7
    with pytest.raises(ValueError, match="at most 1024"):
        c._max_output_tokens({"max_output_tokens": 1025})


def test_higgs_default_is_greedy_and_client_knobs_are_forwarded():
    model = _higgs_model()
    cfg = _higgs_sampler(model)
    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (0.0, 1.0, 0)
    cfg = _higgs_sampler(model, temperature=0.3, top_k=8, ignore_eos=True)
    assert (cfg.temperature, cfg.top_k, cfg.ignore_eos) == (0.3, 8, True)


@pytest.mark.parametrize("kw", [{"repetition_penalty": 0.9}, {"min_p": 0.05}])
def test_higgs_unsupported_knobs_are_refused_at_admission(kw):
    model = _higgs_model()
    with pytest.raises(ValueError, match="does not support"):
        _higgs_sampler(model, **kw).validate(_higgs_spec(model))


def test_higgs_checkpoint_generation_config_sets_the_defaults(tmp_path):
    # the shape bosonai/higgs-audio-v3-stt ships: token ids only, no sampling knobs
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"_from_model_config": True, "bos_token_id": 151643, "eos_token_id": 151643}
    ))
    assert higgs_sampling_defaults(tmp_path) == HIGGS_DEFAULT_SAMPLING
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": False, "top_p": 0.5, "min_p": 0.1}))
    defaults = higgs_sampling_defaults(tmp_path)
    assert defaults == {"temperature": 0.0, "top_p": 0.5, "min_p": 0.1}
    model = _higgs_model(defaults)
    _higgs_sampler(model).validate(_higgs_spec(model))


@pytest.mark.parametrize("bad", ["false", 0, 1])
def test_higgs_enable_thinking_must_be_a_bool(bad):
    with pytest.raises(ValueError, match="enable_thinking must be a boolean"):
        _higgs_model().process_prompt(None, ["audio"], ["text"], tensors={}, enable_thinking=bad)



# ── omnivoice ───────────────────────────────────────────────────────────────

sys.path.insert(0, ".")

from mstar.model.omnivoice.config import OmniVoiceConfig  # noqa: E402
from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel  # noqa: E402


def _omnivoice_model() -> OmniVoiceModel:
    model = object.__new__(OmniVoiceModel)
    model.config = OmniVoiceConfig()
    model._submodule_cache = {}
    model._ensure_data_worker_assets = lambda: None
    return model


def _omnivoice_initial_args(model, **kwargs):
    signals = {"prefix_ids": ["p"], "prefix_audio_mask": ["m"], "target_len": ["t"]}
    return model.get_initial_forward_pass_args("default", ["text"], ["audio"], signals, kwargs)


def test_omnivoice_request_kwargs_include_the_adapter_alias_and_every_knob():
    declared = _omnivoice_model().request_kwargs()
    # the speech adapter maps voice -> instruct; speed is forwarded
    assert {"instruct", "speed", "language", "ref_text", "duration", "denoise"} <= declared
    assert {"num_step", "guidance_scale", "t_shift", "class_temperature", "position_temperature"} <= declared
    assert {"postprocess_output", "pad_duration", "fade_duration", "layer_penalty_factor"} <= declared
    assert "voice" not in declared and "temperature" not in declared
    assert _omnivoice_model().get_max_output_tokens_limit() is None


def test_omnivoice_defaults_when_absent():
    meta = _omnivoice_initial_args(_omnivoice_model()).step_metadata
    gen = OmniVoiceConfig().generation
    assert meta["num_step"] == gen.num_step and meta["guidance_scale"] == gen.guidance_scale
    assert meta["class_temperature"] == 0.0 and meta["postprocess_output"] is True
    assert meta["pad_duration"] == 0.1 and meta["fade_duration"] == 0.1


def test_omnivoice_valid_knobs_pass_through():
    meta = _omnivoice_initial_args(
        _omnivoice_model(), num_step=64, guidance_scale=0, class_temperature=0, t_shift=0.5, postprocess_output=False
    ).step_metadata
    assert meta["num_step"] == 64 and meta["guidance_scale"] == 0.0 and meta["t_shift"] == 0.5
    assert meta["postprocess_output"] is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_step": 0}, r"num_step must be an integer in \[1, 64\]"),
        ({"num_step": 65}, r"num_step must be an integer in \[1, 64\]"),
        ({"num_step": 16.0}, "num_step must be an integer"),
        ({"num_step": "16"}, "num_step must be an integer"),
        ({"guidance_scale": "2"}, "guidance_scale must be a finite number"),
        ({"guidance_scale": float("nan")}, "guidance_scale must be a finite number"),
        ({"t_shift": 0}, "t_shift must be > 0"),
        ({"class_temperature": -1.0}, "class_temperature must be >= 0"),
        ({"position_temperature": True}, "position_temperature must be a finite number"),
        ({"postprocess_output": "false"}, "postprocess_output must be true or false"),
        ({"postprocess_output": 0}, "postprocess_output must be true or false"),
        ({"pad_duration": -0.1}, "pad_duration must be >= 0"),
    ],
)
def test_omnivoice_bad_step_knobs_are_400s_at_both_seams(kwargs, match):
    model = _omnivoice_model()
    with pytest.raises(ValueError, match=match):
        _omnivoice_initial_args(model, **kwargs)
    with pytest.raises(ValueError, match=match):
        model.process_prompt("Hello", ["text"], ["audio"], **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"speed": 0}, "speed must be > 0"),
        ({"speed": -1.0}, "speed must be > 0"),
        ({"speed": "1.2"}, "speed must be a finite number"),
        ({"duration": 0}, "duration must be > 0"),
        ({"duration": -2.0}, "duration must be > 0"),
        ({"duration": 0.001}, "shorter than one codec frame"),
        ({"denoise": "false"}, "denoise must be true or false"),
        ({"instruct": ""}, "instruct must be a non-empty string"),
        ({"language": 3}, "language must be a non-empty string"),
        ({"ref_text": ""}, "ref_text must be a non-empty string"),
    ],
)
def test_omnivoice_bad_prompt_knobs_are_400s(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _omnivoice_model().process_prompt("Hello", ["text"], ["audio"], **kwargs)



# ── qwen3_5 ─────────────────────────────────────────────────────────────────

QWEN3_5_27B = {
    "bos_token_id": 248044, "do_sample": True, "eos_token_id": [248046, 248044],
    "pad_token_id": 248044, "temperature": 0.6, "top_k": 20, "top_p": 0.95,
}
CACHED_0_8B = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots"


def _qwen3_5_model(defaults=None) -> Qwen3_5DenseModel:
    model = object.__new__(Qwen3_5DenseModel)
    model.config = types.SimpleNamespace(max_position_embeddings=262144, vocab_size=248320)
    model.sampling_defaults = dict(defaults or {})
    return model


def _qwen3_5_sampler(model, **kw) -> SamplingReqConfig:
    return model.get_request_resource_configs({}, kw)[QWEN3_5_SAMPLER]


def test_qwen3_5_request_kwargs_are_the_keys_it_reads():
    keys = _qwen3_5_model().request_kwargs()
    for read in ("enable_thinking", "temperature", "top_p", "top_k", "min_p",
                 "repetition_penalty", "penalize_prompt", "ignore_eos", "max_output_tokens"):
        assert read in keys


def test_qwen3_5_limit_is_the_decode_loop_bound():
    model = _qwen3_5_model()
    assert model.get_max_output_tokens_limit() == 262144
    c = Conductor.__new__(Conductor)
    c.model = model
    assert c._max_output_tokens({}) == 2048
    assert c._max_output_tokens({"max_output_tokens": 262144}) == 262144
    with pytest.raises(ValueError, match="at most 262144"):
        c._max_output_tokens({"max_output_tokens": 262145})


def test_qwen3_5_checkpoint_defaults_fill_unset_knobs(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(QWEN3_5_27B))
    defaults = qwen3_5_sampling_defaults(tmp_path)
    assert defaults == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    model = _qwen3_5_model(defaults)
    cfg = _qwen3_5_sampler(model)
    assert (cfg.temperature, cfg.top_k, cfg.top_p, cfg.repetition_penalty) == (0.6, 20, 0.95, 1)
    cfg = _qwen3_5_sampler(model, top_k=0, temperature=0.0, repetition_penalty=1.1, penalize_prompt=False)
    assert (cfg.temperature, cfg.top_k, cfg.top_p) == (0.0, 0, 0.95)
    assert (cfg.repetition_penalty, cfg.penalize_prompt) == (1.1, False)


def test_qwen3_5_without_a_generation_config_the_engine_defaults_apply(tmp_path):
    assert qwen3_5_sampling_defaults(tmp_path) == {}
    cfg = _qwen3_5_sampler(_qwen3_5_model())
    assert (cfg.temperature, cfg.top_k, cfg.top_p) == (0.6, 0, 1)


def test_qwen3_5_min_p_is_forwarded_so_the_node_refuses_it():
    spec = SamplerSpec(resource_key=QWEN3_5_SAMPLER, nodes={"LLM"}, vocab_size=10, enable_min_p=False)
    with pytest.raises(ValueError, match="min_p"):
        _qwen3_5_sampler(_qwen3_5_model(), min_p=0.1).validate(spec)
    _qwen3_5_sampler(_qwen3_5_model(), min_p=0.0).validate(spec)


@pytest.mark.parametrize("bad", ["false", 0, "yes"])
def test_qwen3_5_enable_thinking_must_be_a_bool(bad):
    with pytest.raises(ValueError, match="enable_thinking must be a boolean"):
        _qwen3_5_model().process_prompt("hi", ["text"], ["text"], enable_thinking=bad)


@pytest.mark.skipif(not CACHED_0_8B.is_dir(), reason="Qwen3.5-0.8B metadata not cached")
def test_qwen3_5_real_0_8b_has_no_generation_config_and_no_min_p_capability():
    model = Qwen3_5DenseModel(str(next(CACHED_0_8B.iterdir())))
    assert model.sampling_defaults == {}
    spec = next(s for s in model.get_node_resources() if s.resource_key == QWEN3_5_SAMPLER)
    assert spec.enable_repetion_penalty and not spec.enable_min_p



# ── qwen3_omni ──────────────────────────────────────────────────────────────

OMNI_GENERATION_CONFIG = {
    "talker_max_new_tokens": 4096,
    "talker_repetition_penalty": 1.05,
    "talker_temperature": 0.9,
    "talker_top_k": 50,
    "talker_top_p": 1.0,
}
OMNI_CACHED = glob.glob(
    "/shared/home/*/mminf_cache/qwen3omni/models--Qwen--Qwen3-Omni-30B-A3B-Instruct"
    "/snapshots/*/generation_config.json"
)


def _omni_model(local_dir=None) -> Qwen3OmniModel:
    model = Qwen3OmniModel.__new__(Qwen3OmniModel)
    model.local_dir = str(local_dir) if local_dir is not None else None
    model.config = SimpleNamespace(
        talker=SimpleNamespace(speaker_id={"chelsie": 2301, "ethan": 2302, "aiden": 2303}),
    )
    return model


@pytest.fixture
def omni_released(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(OMNI_GENERATION_CONFIG))
    return _omni_model(tmp_path)


def _omni_talker_args(model, model_kwargs, output_modalities=("text", "audio")):
    return model.get_initial_forward_pass_args(
        "Talker", ["text"], list(output_modalities),
        {"text_inputs": ["t0"]}, model_kwargs,
    )


@pytest.mark.skipif(not OMNI_CACHED, reason="Qwen3-Omni checkpoint not cached")
def test_qwen3_omni_fixture_matches_the_cached_checkpoint():
    with open(OMNI_CACHED[0]) as f:
        assert json.load(f) == OMNI_GENERATION_CONFIG


def test_qwen3_omni_defaults_come_from_the_checkpoint(omni_released):
    configs = omni_released.get_request_resource_configs({}, {})
    talker = configs[TALKER_SAMPLER]
    assert (talker.temperature, talker.top_k, talker.top_p, talker.repetition_penalty) == (
        0.9, 50, 1.0, 1.05,
    )
    # the checkpoint names no Thinker or CodePredictor knob; the fallback applies
    thinker = configs[THINKER_SAMPLER]
    assert (thinker.temperature, thinker.top_p, thinker.top_k) == (0.7, 0.9, 0)
    code_pred = configs[CODE_PRED_SAMPLER]
    assert (code_pred.temperature, code_pred.top_k, code_pred.top_p) == (1.0, 50, 0.8)
    assert _omni_talker_args(omni_released, {}).full_metadata.kwargs["talker_max_tokens"] == 4096


def test_qwen3_omni_no_checkpoint_falls_back():
    model = _omni_model()
    assert model.get_max_output_tokens() == MAX_OUTPUT_TOKENS
    assert _omni_talker_args(model, {}).full_metadata.kwargs["talker_max_tokens"] == MAX_OUTPUT_TOKENS


def test_qwen3_omni_plain_keys_target_the_thinker_and_prefixed_keys_win(omni_released):
    configs = omni_released.get_request_resource_configs({}, {
        "temperature": 0.2, "top_p": 0.5, "top_k": 7, "repetition_penalty": 1.1,
        "penalize_prompt": False, "ignore_eos": True,
    })
    thinker = configs[THINKER_SAMPLER]
    assert (thinker.temperature, thinker.top_p, thinker.top_k, thinker.repetition_penalty) == (
        0.2, 0.5, 7, 1.1,
    )
    assert thinker.penalize_prompt is False and thinker.ignore_eos is True
    # plain keys do not reach the other stages
    assert configs[TALKER_SAMPLER].temperature == 0.9
    assert configs[CODE_PRED_SAMPLER].temperature == 1.0

    configs = omni_released.get_request_resource_configs({}, {
        "temperature": 0.2, "thinker_temperature": 0.4,
        "talker_temperature": 0.3, "code_predictor_top_k": 5,
    })
    assert configs[THINKER_SAMPLER].temperature == 0.4
    assert configs[TALKER_SAMPLER].temperature == 0.3
    assert configs[CODE_PRED_SAMPLER].top_k == 5


def test_qwen3_omni_code_predictor_gets_no_penalty_and_refuses_one(omni_released):
    spec = SimpleNamespace(enable_repetion_penalty=False, enable_min_p=False)
    config = omni_released.get_request_resource_configs({}, {})[CODE_PRED_SAMPLER]
    assert config.repetition_penalty == 1
    config.validate(spec)
    asked = omni_released.get_request_resource_configs({}, {"code_predictor_repetition_penalty": 1.2})
    with pytest.raises(ValueError, match="repetition_penalty"):
        asked[CODE_PRED_SAMPLER].validate(spec)


def test_qwen3_omni_limits_match_the_decode_loops(omni_released):
    walks = omni_released.get_graph_walk_graphs()
    assert walks["thinker_decode"].max_iters == omni_released.get_max_output_tokens_limit() == MAX_OUTPUT_TOKENS
    assert walks["talker_decode"].max_iters == 4096


@pytest.mark.parametrize("value", [0, 4097, 1.5, True, "100"])
def test_qwen3_omni_talker_max_output_tokens_out_of_range_is_a_400(omni_released, value):
    with pytest.raises(ValueError, match="talker_max_output_tokens"):
        _omni_talker_args(omni_released, {"talker_max_output_tokens": value})


def test_qwen3_omni_talker_max_output_tokens_is_honored(omni_released):
    args = _omni_talker_args(omni_released, {"talker_max_output_tokens": 4096})
    assert args.full_metadata.kwargs["talker_max_tokens"] == 4096
    assert args.step_metadata["talker_max_tokens"] == 4096


def test_qwen3_omni_voice_default_and_case(omni_released):
    assert _omni_talker_args(omni_released, {}).step_metadata["voice"] == "Ethan"
    assert _omni_talker_args(omni_released, {"voice": "chelsie"}).step_metadata["voice"] == "chelsie"
    assert _omni_talker_args(omni_released, {"voice": "Aiden"}).full_metadata.kwargs["voice"] == "Aiden"


@pytest.mark.parametrize("voice", ["", "alloy", 3])
def test_qwen3_omni_bad_voice_is_a_400(omni_released, voice):
    with pytest.raises(ValueError, match="voice"):
        _omni_talker_args(omni_released, {"voice": voice})
    # an explicit voice is checked on text-only output too
    with pytest.raises(ValueError, match="voice"):
        _omni_talker_args(omni_released, {"voice": voice}, output_modalities=("text",))


def test_qwen3_omni_request_kwargs_spot_checks(omni_released):
    declared = omni_released.request_kwargs()
    for key in (
        "temperature", "top_p", "top_k", "min_p", "repetition_penalty", "penalize_prompt",
        "thinker_temperature", "thinker_top_p", "talker_temperature", "talker_top_p",
        "talker_top_k", "talker_repetition_penalty", "code_predictor_top_k",
        "code_predictor_temperature", "talker_max_output_tokens", "voice", "ignore_eos",
        "max_output_tokens", "seed",
    ):
        assert key in declared, key
    assert "do_sample" not in declared
    assert "subtalker_temperature" not in declared



# ── vjepa2 ──────────────────────────────────────────────────────────────────

sys.path.insert(0, ".")

from mstar.model.vjepa2.vjepa2_model import VJepa2Model  # noqa: E402


@pytest.fixture(scope="module")
def vjepa2_masked():
    return VJepa2Model(
        model_path_hf="facebook/vjepa2-vitl-fpc64-256", skip_weight_loading=True, predictor_kind="vjepa2_masked"
    )


@pytest.fixture(scope="module")
def vjepa2_ac_model():
    return VJepa2Model(model_path_hf="facebook/vjepa2-ac-vitg", skip_weight_loading=True, predictor_kind="ac")


def _vjepa2_video(frames: int) -> dict:
    return {"video_inputs": [torch.rand(frames, 3, 32, 40)]}


def test_vjepa2_request_kwargs_and_no_token_limit(vjepa2_masked):
    declared = vjepa2_masked.request_kwargs()
    assert {"num_frames", "rollout_horizon", "stream_rollout", "skip_predictor", "mpc"} <= declared
    assert {"actions", "states", "extrinsics", "goal_hidden", "goal_hidden_fill"} <= declared
    assert {"context_mask", "target_mask"} <= declared
    assert vjepa2_masked.get_max_output_tokens_limit() is None


@pytest.mark.parametrize("horizon", [-1, 17, 2.0, "4", True])
def test_vjepa2_bad_rollout_horizon_is_a_400(vjepa2_masked, horizon):
    limit = vjepa2_masked.config.max_rollout_horizon
    with pytest.raises(ValueError, match=rf"rollout_horizon must be an integer in \[0, {limit}\]"):
        vjepa2_masked._initial_walk({"rollout_horizon": horizon})
    with pytest.raises(ValueError, match="rollout_horizon"):
        vjepa2_masked.process_prompt(None, ["video"], ["video"], tensors=_vjepa2_video(4), rollout_horizon=horizon)


def test_vjepa2_rollout_horizon_range_is_honored_not_clamped(vjepa2_masked):
    assert vjepa2_masked._initial_walk({"rollout_horizon": 0}) == VJepa2Model.PREFILL_VIDEO
    assert vjepa2_masked._initial_walk({"rollout_horizon": 1}) == VJepa2Model.PREFILL_VIDEO
    limit = vjepa2_masked.config.max_rollout_horizon
    args = vjepa2_masked.get_initial_forward_pass_args(
        "default", ["video"], ["video"], {}, {"rollout_horizon": limit}
    )
    assert args.full_metadata.graph_walk == VJepa2Model.PREFILL_VIDEO_ROLLOUT
    assert args.step_metadata["rollout_horizon"] == limit


@pytest.mark.parametrize("flag", ["stream_rollout", "skip_predictor", "mpc"])
@pytest.mark.parametrize("value", ["false", 0, 1])
def test_vjepa2_flags_must_be_bools(vjepa2_masked, flag, value):
    with pytest.raises(ValueError, match=f"{flag} must be true or false"):
        vjepa2_masked._initial_walk({flag: value, "rollout_horizon": 4})
    with pytest.raises(ValueError, match=f"{flag} must be true or false"):
        vjepa2_masked.process_prompt(None, ["video"], ["video"], tensors=_vjepa2_video(4), **{flag: value})


@pytest.mark.parametrize("num_frames", [0, -2, 9, 4.0, "4"])
def test_vjepa2_num_frames_outside_the_decoded_clip_is_a_400(vjepa2_masked, num_frames):
    with pytest.raises(ValueError, match=r"num_frames must be an integer in \[1, 8\]"):
        vjepa2_masked.process_prompt(None, ["video"], ["video"], tensors=_vjepa2_video(8), num_frames=num_frames)


def test_vjepa2_num_frames_default_and_explicit(vjepa2_masked):
    crop = vjepa2_masked.config.crop_size
    out = vjepa2_masked.process_prompt(None, ["video"], ["video"], tensors=_vjepa2_video(8))
    assert tuple(out["video_frames"][0].shape) == (8, 3, crop, crop)
    out = vjepa2_masked.process_prompt(None, ["video"], ["video"], tensors=_vjepa2_video(8), num_frames=4)
    assert out["video_frames"][0].shape[0] == 4


def test_vjepa2_ac_inputs_must_be_numeric(vjepa2_ac_model):
    good = {"actions": [[0.0] * 7] * 4, "states": [[0.0] * 7] * 4}
    with pytest.raises(ValueError, match="actions must be a numeric array"):
        vjepa2_ac_model.process_prompt(None, ["video"], ["video"], **{**good, "actions": [["a"] * 7] * 4})
    with pytest.raises(ValueError, match="states must be a numeric array"):
        vjepa2_ac_model.process_prompt(None, ["video"], ["video"], **{**good, "states": [[0.0] * 7, [0.0] * 3]})
    with pytest.raises(ValueError, match="goal_hidden_fill must be a number"):
        vjepa2_ac_model.process_prompt(None, ["video"], ["video"], mpc=True, goal_hidden_fill="1", **good)
    # the trajectory check uses the validated horizon: 5 needs 1 + 5 - 1 = 5 steps
    with pytest.raises(ValueError, match="trajectory length >= .* = 5; got 4"):
        vjepa2_ac_model.process_prompt(None, ["video"], ["video"], rollout_horizon=5, **good)
    out = vjepa2_ac_model.process_prompt(None, ["video"], ["video"], rollout_horizon=4, **good)
    assert out["actions"][0].shape == (4, 7)



# ── whisper ─────────────────────────────────────────────────────────────────

WHISPER_CACHED = Path("/shared/home/s129652-bf47ee/mminf_cache/hf/hub/models--openai--whisper-large-v3/snapshots")


def _whisper_model(defaults=None) -> WhisperModel:
    model = object.__new__(WhisperModel)
    model.config = WhisperModelConfig(
        lang_to_id={"<|en|>": 50259, "<|de|>": 50261},
        task_to_id={"transcribe": 50360, "translate": 50359},
    )
    model.sampling_defaults = dict(WHISPER_DEFAULT_SAMPLING if defaults is None else defaults)
    return model


def _whisper_sampler(model, **kw) -> SamplingReqConfig:
    return model.get_request_resource_configs({}, kw)[WHISPER_SAMPLER]


def _whisper_spec(model):
    return next(s for s in model.get_node_resources() if s.resource_key == WHISPER_SAMPLER)


def _whisper_conductor(model):
    c = Conductor.__new__(Conductor)
    c.model = model
    return c


def test_whisper_request_kwargs_are_the_keys_it_reads():
    keys = _whisper_model().request_kwargs()
    for read in ("language", "task", "temperature", "top_p", "top_k", "min_p",
                 "repetition_penalty", "penalize_prompt", "ignore_eos", "max_output_tokens"):
        assert read in keys
    # sent by the transcription adapter but never read
    assert "initial_prompt" not in keys and "timestamps" not in keys


def test_whisper_limit_is_the_decode_loop_bound():
    model = _whisper_model()
    assert model.get_max_output_tokens_limit() == 444
    assert model.get_max_output_tokens() == 444
    assert model.get_graph_walk_graphs()["decode"].max_iters == 444


def test_whisper_over_limit_is_refused_not_clamped():
    c = _whisper_conductor(_whisper_model())
    assert c._max_output_tokens({"max_output_tokens": 444}) == 444
    assert c._max_output_tokens({"max_output_tokens": 10}) == 10
    with pytest.raises(ValueError, match="at most 444"):
        c._max_output_tokens({"max_output_tokens": 445})


def test_whisper_default_is_greedy_and_client_knobs_are_forwarded():
    model = _whisper_model()
    cfg = _whisper_sampler(model)
    assert (cfg.temperature, cfg.top_p, cfg.top_k) == (0.0, 1.0, 0)
    cfg = _whisper_sampler(model, temperature=0.4, top_p=0.9, top_k=5, penalize_prompt=False)
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.penalize_prompt) == (0.4, 0.9, 5, False)


@pytest.mark.parametrize("kw", [{"repetition_penalty": 1.2}, {"min_p": 0.1}])
def test_whisper_unsupported_knobs_are_refused_at_admission(kw):
    model = _whisper_model()
    with pytest.raises(ValueError, match="does not support"):
        _whisper_sampler(model, **kw).validate(_whisper_spec(model))
    _whisper_sampler(model).validate(_whisper_spec(model))


def test_whisper_checkpoint_generation_config_sets_the_defaults(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 0.7, "top_k": 3, "repetition_penalty": 1.1, "max_length": 448}
    ))
    defaults = whisper_sampling_defaults(tmp_path)
    assert defaults == {"temperature": 0.7, "top_p": 1.0, "top_k": 3, "repetition_penalty": 1.1}
    # a checkpoint penalty turns the capability on, so its own default is not refused
    model = _whisper_model(defaults)
    _whisper_sampler(model).validate(_whisper_spec(model))


@pytest.mark.skipif(not WHISPER_CACHED.is_dir(), reason="whisper-large-v3 not cached")
def test_whisper_real_checkpoint_declares_no_sampling_so_greedy_stays():
    snapshot = next(WHISPER_CACHED.iterdir())
    assert whisper_sampling_defaults(snapshot) == WHISPER_DEFAULT_SAMPLING


@pytest.mark.parametrize("kw", [{"language": 5}, {"task": ["transcribe"]}, {"language": ""}, {"language": "xx"}])
def test_whisper_bad_language_or_task_is_a_value_error(kw):
    model = _whisper_model()
    if isinstance(kw.get("language"), str):
        # a string reaches the token lookup, which needs the audio checks passed
        with pytest.raises(ValueError, match="language"):
            model.config.decoder_prompt_ids(language=kw["language"])
        return
    with pytest.raises(ValueError, match="must be a string"):
        model.process_prompt(None, ["audio"], ["text"], tensors={}, **kw)
