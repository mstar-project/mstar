"""CPU tests for the VoxCPM2 port: tiny random weights, no checkpoint, no GPU.

Covers the contract (config, graph, partitions, prompt modes, validation),
the Talker's batching and stop rule through dense fake resources, and the
#429 invariant for the VAE stream: windows from the real ``StreamBuffer`` and
chunk policy stitch to exactly a whole decode, in length and in samples.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import importlib.util
import io
import json
import math
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.api_server.openai import adapters
from mstar.api_server.openai.protocol import SpeechRequest
from mstar.engine.accelerator_graph_config import PiecewiseBatchedConfig, PiecewiseCallInputs, PiecewiseCaptureShape
from mstar.engine.engine import Engine
from mstar.engine.resources.step import StepContext
from mstar.engine.state import host
from mstar.engine.state.adapters import StateSetup, kv_layouts, node_resources
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.model.submodule_base import ARNodeInputs, InputMetadata, engine_step
from mstar.model.voxcpm2.components.audio_vae import AudioVAE, CausalResidualUnit, decoder_receptive_field_frames
from mstar.model.voxcpm2.components.minicpm import RopeTable, longrope_cos_sin
from mstar.model.voxcpm2.components.talker import VoxCPM2Generator
from mstar.model.voxcpm2.config import (
    AUDIO_START_TOKEN,
    BASE_KV,
    CFG_KEY,
    MIN_LEN_KEY,
    MODE_REF_CONTINUATION,
    MODE_REFERENCE,
    MODE_ZERO_SHOT,
    PATCH_BUDGET_KEY,
    PROMPT_MODE_KEY,
    REF_AUDIO_END_TOKEN,
    REF_AUDIO_START_TOKEN,
    REF_DIGEST_KEY,
    TALKER_DECODE_METADATA,
    TALKER_STEP,
    VoxCPM2Config,
)
from mstar.model.voxcpm2.submodules import (
    PREFILL_TAIL,
    TALKER_LOOP,
    VAE_WINDOW,
    RefEncoderSubmodule,
    TalkerSubmodule,
    VAEDecoderSubmodule,
    build_prompt,
)
from mstar.model.voxcpm2.voxcpm2_model import (
    STREAM_EDGE,
    VoxCPM2Model,
    decoder_receptive_field_patches,
)
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "voxcpm2.yaml"
HELPERS = Path(__file__).resolve().parents[1] / "voxcpm2"


def _helper(module: str):
    """A helper module from ``test/voxcpm2``, loaded by path: ``import test``
    can find the standard library's ``test`` package instead of this tree."""
    spec = importlib.util.spec_from_file_location(f"voxcpm2_{module}", HELPERS / f"{module}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


bind_dense_resources = _helper("fake_resources").bind_dense_resources
stream_through = _helper("streaming").stream_through

FACTORS = [1.0 + 0.25 * i for i in range(8)]
FP64_ROUNDING = 1e-13  # float64 streams differ from a whole decode by up to about 6e-15 on some CPUs, 0 on others
TINY = {
    "architecture": "voxcpm2",
    "lm_config": {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_size": 64,
        "intermediate_size": 96,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "kv_channels": 16,
        "rope_scaling": {
            "type": "longrope",
            "long_factor": FACTORS,
            "short_factor": FACTORS,
            "original_max_position_embeddings": 512,
        },
        "vocab_size": 128,
        "use_mup": False,
        "scale_emb": 12,
        "dim_model_base": 256,
        "scale_depth": 1.4,
    },
    "patch_size": 4,
    "feat_dim": 8,
    "scalar_quantization_latent_dim": 16,
    "scalar_quantization_scale": 9,
    "residual_lm_num_layers": 1,
    "residual_lm_no_rope": True,
    "encoder_config": {"hidden_dim": 32, "ffn_dim": 48, "num_heads": 4, "num_layers": 1, "kv_channels": 16},
    "dit_config": {"hidden_dim": 32, "ffn_dim": 48, "num_heads": 4, "num_layers": 1, "kv_channels": 16},
    # The published geometry (REAL_VAE) upsamples 1920x; a 64x decoder keeps the
    # CPU stream tests fast. Both are checked for their receptive field.
    "audio_vae_config": {
        "encoder_dim": 8,
        "encoder_rates": [2, 5, 8, 8],
        "latent_dim": 8,
        "decoder_dim": 64,
        "decoder_rates": [2, 2, 2, 2, 2, 2],
        "sr_bin_boundaries": [20000, 30000, 40000],
        "sample_rate": 16000,
        "out_sample_rate": 48000,
    },
    "max_length": 512,
}


REAL_VAE = {
    "encoder_dim": 128,
    "encoder_rates": [2, 5, 8, 8],
    "latent_dim": 64,
    "decoder_dim": 2048,
    "decoder_rates": [8, 6, 5, 2, 2, 2],
    "sr_bin_boundaries": [20000, 30000, 40000],
    "sample_rate": 16000,
    "out_sample_rate": 48000,
}


class _Tokenizer:
    """Whitespace words and CJK runs; '你好' is one vocabulary token, as in the checkpoint."""

    vocab = {
        "\u2581hello": 10,
        "\u2581world": 11,
        "\u2581你好": 12,
        "你好": 16,
        "你": 13,
        "好": 14,
        "\u2581(warm)hello": 15,
    }

    def tokenize(self, text):
        return ["\u2581" + word for word in text.split()]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, 3) for token in tokens]

    def get_vocab(self):
        return dict(self.vocab)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    import transformers

    (tmp_path / "config.json").write_text(json.dumps(TINY))
    monkeypatch.setattr(transformers.LlamaTokenizerFast, "from_pretrained", classmethod(lambda cls, path: _Tokenizer()))
    return tmp_path


@pytest.fixture
def model(checkpoint) -> VoxCPM2Model:
    return VoxCPM2Model(model_path_hf=str(checkpoint))


def _random_generator(config: VoxCPM2Config, seed: int = 0) -> VoxCPM2Generator:
    torch.manual_seed(seed)
    generator = VoxCPM2Generator(config)
    with torch.no_grad():
        for param in generator.parameters():
            param.normal_(0, 0.2)
    generator.materialize_buffers()
    return generator.eval()


def _random_vae(config: VoxCPM2Config, seed: int = 0, residual_gain: float = 0.1) -> AudioVAE:
    """Random weights scaled like a trained decoder's (unit-gain convs, Snake
    alphas near 1, quiet residual branches), so the output stays off the final
    tanh's rails and sample equality is not met trivially by saturation. At
    ``residual_gain`` 0.1 the farthest frames' influence is below float64
    rounding; the receptive-field bound needs full-strength branches."""
    torch.manual_seed(seed)
    vae = AudioVAE(config.audio_vae)
    with torch.no_grad():
        for name, param in vae.named_parameters():
            if name.endswith("alpha"):
                param.uniform_(0.5, 1.5)
            elif "scale_embed" in name:
                param.uniform_(0.8, 1.2)
            elif param.dim() == 3:
                param.normal_(0, 1 / math.sqrt(param.shape[1] * param.shape[2]))
            else:
                param.normal_(0, 0.05)
        for module in vae.modules():
            if isinstance(module, CausalResidualUnit):
                module.block[3].weight.mul_(residual_gain)
    return vae.eval()


def _assert_unsaturated(audio: torch.Tensor) -> None:
    assert (audio.abs() > 0.99).double().mean() < 0.02 and audio.std() > 0.1


# -- contract --------------------------------------------------------------------


def test_config_reads_checkpoint_json_and_rejects_other_architectures():
    config = VoxCPM2Config.from_dict(TINY)
    assert (config.lm.head_dim, config.residual.num_hidden_layers, config.residual.no_rope) == (16, 1, True)
    assert config.encoder.num_key_value_heads == 2 and config.dit.vocab_size == 0
    assert config.lm.rope_factors == tuple(FACTORS) and config.lm.rope_attention_scale == 1.0
    assert config.patch_samples == 4 * 64 and config.ref_patch_samples == 4 * 640
    real = VoxCPM2Config.from_dict({**TINY, "audio_vae_config": REAL_VAE})
    assert real.patch_samples == 7680 and real.output_sample_rate == 48000
    # the reference caps a request at target tokens * 6 + 10 patches
    assert config.max_patches(5) == 40 and config.max_patches(5, max_len=12) == 12
    with pytest.raises(ValueError, match="architecture"):
        VoxCPM2Config.from_dict({**TINY, "architecture": "voxcpm"})
    with pytest.raises(ValueError, match="muP"):
        VoxCPM2Config.from_dict({**TINY, "lm_config": {**TINY["lm_config"], "use_mup": True}})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_tables_are_rounded_through_the_stack_dtype(dtype):
    """The reference's ``model.to(bf16)`` casts its LongRoPE tables too."""
    config = VoxCPM2Config.from_dict(TINY).lm
    with torch.device("meta"):
        table = RopeTable(config, 32)
    table = table.to(dtype).to_empty(device="cpu")
    table.materialize()
    cos, sin = longrope_cos_sin(config, 32)
    assert table.cos.dtype == table.sin.dtype == torch.float32
    assert torch.equal(table.cos, cos.to(dtype).float()) and torch.equal(table.sin, sin.to(dtype).float())
    assert torch.equal(table.cos, cos) == (dtype == torch.float32)
    table = table.to(torch.bfloat16)
    assert table.cos.dtype == torch.float32 and torch.equal(table.cos, cos.to(dtype).float())


def test_registry_yaml_and_adapter_agree(model):
    assert get_model_class("voxcpm2") is VoxCPM2Model
    assert HF_MODELS["voxcpm2"]["model_path_hf"] == "openbmb/VoxCPM2"
    deployment = yaml.safe_load(CONFIG_PATH.read_text())
    assert deployment["model"] == "voxcpm2"
    placed = {node for group in deployment["node_groups"] for node in group["node_names"]}
    assert placed == set(model.nodes) == {"Talker", "RefEncoder", "VAEDecoder"}
    walks = {walk for group in deployment["node_groups"] for walk in group["graph_walks"]}
    assert walks == set(model.get_graph_walk_graphs())
    assert isinstance(adapters.get_adapter("voxcpm2"), adapters.VoxCPM2Adapter)


def test_stream_context_covers_the_vae_receptive_field(model, checkpoint):
    real = VoxCPM2Config.from_dict({**TINY, "audio_vae_config": REAL_VAE})
    with torch.device("meta"):
        assert decoder_receptive_field_frames(AudioVAE(real.audio_vae).decoder) == 20
    assert decoder_receptive_field_patches(real) == 5
    context = decoder_receptive_field_patches(model.config)
    assert model.stream_left_context == context
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    assert isinstance(policy, ScheduledLeftContextChunkPolicy)
    assert policy._left_context == context and policy._schedule == (1, 2, 4) and policy._chunk == 8
    with pytest.raises(ValueError, match="receptive field"):
        VoxCPM2Model(model_path_hf=str(checkpoint), stream_left_context_patches=context - 1)
    with pytest.raises(ValueError, match="inference_timesteps"):
        VoxCPM2Model(model_path_hf=str(checkpoint), inference_timesteps=1)


def test_receptive_field_bound_is_exact():
    """Perturbing latent frame k changes the audio of frames k .. k + rf and
    nothing later: the derived context is enough, and none of it is spare."""
    config = VoxCPM2Config.from_dict(TINY)
    vae = _random_vae(config, residual_gain=1.0).double()
    rf = decoder_receptive_field_frames(vae.decoder)
    frames = 5 + rf + 6
    z = torch.randn(1, config.feat_dim, frames, dtype=torch.float64)
    bumped = z.clone()
    bumped[..., 5] += 1.0
    with torch.no_grad():
        delta = (vae.decode(bumped) - vae.decode(z)).abs().view(frames, -1).amax(-1)
    changed = (delta > 0).nonzero().flatten().tolist()
    assert changed == list(range(5, 5 + rf + 1))


def _prompt(model, *args, **kwargs) -> dict:
    """``process_prompt``'s tensors; its metadata is checked separately."""
    return model.process_prompt(*args, **kwargs).new_input_tensors


def _request(model, text: str, clip: torch.Tensor | None = None, **kwargs) -> tuple[dict, dict]:
    """A request's edge tensors and the Talker partition's ``step_metadata``,
    as the data worker and the conductor build them."""
    modalities = ["text", "audio"] if clip is not None else ["text"]
    tensors = {"audio_inputs": [clip]} if clip is not None else None
    prompt = model.process_prompt(text, modalities, ["audio"], tensors=tensors, **kwargs)
    edges, metadata = prompt.new_input_tensors, prompt.metadata
    if clip is not None:
        edges = {**edges, "audio_inputs": [clip]}
    return edges, dict(model.walk_plan("Talker", modalities, ["audio"], {}, metadata)[0].step_metadata)


def _info(rid: str, seed: int = 0, step_metadata: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(rid_handle=rid, request_id=rid, random_seed=seed, step_metadata=step_metadata or {})


def _decode_info(rid: str, seed: int, prefill_metadata: dict) -> SimpleNamespace:
    """A decode step's info, with only the keys the walk plan's decode step carries."""
    return _info(rid, seed, {key: prefill_metadata[key] for key in TALKER_DECODE_METADATA})


def _layout(metadata: dict) -> list:
    return [metadata[PROMPT_MODE_KEY], metadata[PATCH_BUDGET_KEY], metadata[MIN_LEN_KEY]]


def test_process_prompt_builds_each_reference_mode(model):
    prompt = model.process_prompt("hello  world\n", ["text"], ["audio"], seed=3, cfg_value=1.5)
    zero, zero_meta = prompt.new_input_tensors, prompt.metadata
    assert zero["text_inputs"][0].tolist() == [10, 11, AUDIO_START_TOKEN]
    assert _layout(zero_meta) == [MODE_ZERO_SHOT, 2 * 6 + 10, 2] and zero_meta[REF_DIGEST_KEY] is None
    assert zero_meta[CFG_KEY] == 1.5
    assert set(zero) == {"text_inputs"}

    design = _prompt(model, "hello", ["text"], ["audio"], instructions="warm")
    assert design["text_inputs"][0].tolist() == [15, AUDIO_START_TOKEN]  # "(warm)hello"

    cjk = _prompt(model, "你好", ["text"], ["audio"])
    assert cjk["text_inputs"][0].tolist() == [13, 14, AUDIO_START_TOKEN]  # split like the reference

    clip = torch.zeros(16000)  # 1 s: 7 patches of 2560 samples
    ref_meta = model.process_prompt("hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}).metadata
    assert ref_meta[PROMPT_MODE_KEY] == MODE_REFERENCE
    cont = model.process_prompt(
        "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, ref_text="world "
    )
    tensors, metadata = cont.new_input_tensors, cont.metadata
    assert tensors["text_inputs"][0].tolist() == [11, 10, AUDIO_START_TOKEN]  # transcript first
    # transcript and target are tokenized as one string, with no separator added, as the reference does
    joined = model.process_prompt(
        "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, ref_text="world"
    ).new_input_tensors
    assert joined["text_inputs"][0].tolist() == [3, AUDIO_START_TOKEN]  # "worldhello"
    # max patches follow the target text only
    digest = hashlib.blake2b(clip.numpy().tobytes(), digest_size=16).hexdigest()
    assert {k: v for k, v in metadata.items() if k != "state_layout"} == {
        "stream_lead_items": {STREAM_EDGE: 3},
        PATCH_BUDGET_KEY: 16,
        PROMPT_MODE_KEY: MODE_REF_CONTINUATION,
        MIN_LEN_KEY: 2,
        REF_DIGEST_KEY: digest,
        CFG_KEY: model.config.cfg_value,
    }
    # every key the conductor reads back is one the data worker strips from the client
    for meta in (zero_meta, ref_meta, metadata):
        assert set(meta) - {"state_layout", "stream_lead_items"} <= set(model.INTERNAL_KWARGS)
    # only a continuation's stream leads with prompt patches
    assert "stream_lead_items" not in zero_meta and "stream_lead_items" not in ref_meta


def test_process_prompt_collapses_whitespace_like_the_reference(model, monkeypatch):
    """``VoxCPM.generate`` turns each run of whitespace in the target text into
    one space before tokenizing; the transcript of a continuation is kept as sent."""
    seen = []
    tokenize = model.tokenize
    monkeypatch.setattr(model, "tokenize", lambda text: seen.append(text) or tokenize(text))
    model.process_prompt("hello  \t world\n\nhello ", ["text"], ["audio"], instruct="warm   and calm")
    assert seen == ["(warm and calm)hello world hello "]
    seen.clear()
    clip = torch.zeros(16000)
    model.process_prompt(
        "hello  world", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, ref_text="a  b"
    )
    assert seen == ["hello world", "a  bhello world"]


@pytest.mark.parametrize(
    "kwargs, cap",
    [
        ({}, 4096),
        ({"max_len": 9}, 9),
        ({"max_output_tokens": 3, "max_len": 9}, 3),
        ({"max_new_tokens": 2, "max_output_tokens": 3}, 2),
    ],
)
def test_patch_cap_reads_the_standard_and_model_knobs_in_one_order(model, kwargs, cap):
    """``max_new_tokens``, then the standard ``max_output_tokens``, then ``max_len``,
    the same in the conductor's cap and in the prompt's patch budget."""
    assert model.get_max_output_tokens(**kwargs) == cap
    metadata = model.process_prompt("hello world", ["text"], ["audio"], **kwargs).metadata
    assert metadata[PATCH_BUDGET_KEY] == model.config.max_patches(2, cap)


@pytest.mark.parametrize("budget, done", [(1, True), (2, False)])
def test_a_one_patch_request_ends_after_prefill(model, budget, done):
    """Prefill has no loop for the Talker's stop to end, so a request whose
    budget the prefill patch spends goes straight to done instead of a decode
    step that would only veto itself."""
    kwargs = {PATCH_BUDGET_KEY: budget, PROMPT_MODE_KEY: MODE_ZERO_SHOT, MIN_LEN_KEY: 2, CFG_KEY: 2.0}
    plan = model.walk_plan("Talker", ["text"], ["audio"], {}, kwargs)
    assert [step.walk for step in plan] == ["talker_prefill"] + ([] if done else ["talker_decode"])
    # the decode loop reads its patch budget, min_len and cfg_value off step metadata
    assert done or dict(plan[1].step_metadata) == {PATCH_BUDGET_KEY: budget, MIN_LEN_KEY: 2, CFG_KEY: 2.0}


@pytest.mark.parametrize("budget", [2, 5])
def test_talker_caps_decode_where_its_stop_check_ends_it(model, budget):
    """A one-patch budget ends after prefill (above); past that, decode runs one
    iteration fewer than the budget, since the prefill's patch counts."""
    talker, _, _ = _talker(model)
    edges, meta = _request(model, "hello world", max_new_tokens=budget)
    infos = {"talker_prefill": _info("a", 5, meta), "talker_decode": _decode_info("a", 5, meta)}
    patch = torch.zeros(model.config.patch_size, model.config.feat_dim)
    walk, inputs, iterations = "talker_prefill", edges, 0
    while True:
        prepared = talker.prepare_inputs(walk, infos[walk], inputs)
        # the stop head never fires, so only the budget ends decode
        out = {"latent_patch": [patch], "stop_flag": [torch.tensor([0, prepared.kwargs["step"]])]}
        _commit(talker, walk, ["a"], [prepared])
        inputs = {"latent_patch": out["latent_patch"]}
        stop = talker.check_stop("a", infos[walk], out)
        if walk == "talker_decode":
            iterations += 1
            if stop:
                break
        walk = "talker_decode"

    caps = model.get_loop_caps(**meta)
    assert caps == {TALKER_LOOP: iterations}, "the cap and the stop check end decode on different iterations"
    assert model.get_loop_caps(max_new_tokens=budget) == caps


@pytest.mark.parametrize("ref_text", [None, "", "world "], ids=["zero-shot", "reference", "continuation"])
def test_process_prompt_lays_out_the_rows_the_talker_prefill_writes(model, ref_text):
    from mstar.engine.resources import KVReqConfig
    from mstar.engine.resources.kv.config import refuse_undeclared_labels
    from mstar.validation.resources import _undeclared_pools, _unknown_prefix_walks

    specs = node_resources(model)
    assert list(_unknown_prefix_walks(model)) == []
    assert list(_undeclared_pools(specs, model)) == []

    clip = None if ref_text is None else torch.randn(20000, generator=torch.Generator().manual_seed(1)) * 0.1
    modalities = ["text"] if clip is None else ["text", "audio"]
    tensors = None if clip is None else {"audio_inputs": [clip]}
    kwargs = {} if clip is None else {"ref_text": ref_text}
    out = model.process_prompt("hello world", modalities, ["audio"], tensors=tensors, **kwargs)
    meta = out.metadata
    layout = kv_layouts(model.declare_state(), meta["state_layout"], {})
    assert set(layout) == {"base_kv", "res_kv"} and layout["base_kv"] == layout["res_kv"]
    (span,) = layout["base_kv"]["main"].spans
    walk = "talker_prefill" if clip is None else "talker_prefill_clone"
    assert span.walk == walk

    talker, _, _ = _talker(model)
    edges = dict(out.new_input_tensors)
    if clip is not None:
        info = _info("r", 0, {k: meta[k] for k in (PROMPT_MODE_KEY, REF_DIGEST_KEY)})
        ref = _ref_encoder(model)
        prepared = ref.prepare_inputs(walk, info, {"audio_inputs": [clip]})
        with torch.inference_mode():
            edges.update(ref.forward(walk, None, **prepared.tensor_inputs, **prepared.kwargs))
    step = dict(model.walk_plan("Talker", modalities, ["audio"], {}, meta)[0].step_metadata)
    prefill = talker.prepare_inputs(walk, _info("r", 0, step), edges)
    assert span.length == prefill.input_seq_len, "the layout and the prefill disagree on its rows"

    configs = model.get_request_resource_configs([], [], meta)
    for key in ("base_kv", "res_kv"):
        configs.setdefault(key, KVReqConfig(needed_labels=["main"]))
    refuse_undeclared_labels("VoxCPM2Model", {k: tuple(v) for k, v in model.kv_labels().items()}, layout, configs)
    caps = model.get_loop_caps(**meta)
    for spec in specs:
        if spec.resource_key in layout:
            kv = configs[spec.resource_key]
            kv.apply_conductor_config(kv_layout={"main": layout[spec.resource_key]["main"].row()}, loop_caps=caps)
            kv.validate(spec)
            assert kv.kv_layout["main"].positions(caps) == span.length + meta[PATCH_BUDGET_KEY] - 1


@pytest.mark.parametrize(
    "kwargs, inputs, message",
    [
        ({"ref_text": "x"}, None, "ref_text needs"),
        ({"cfg_value": -1}, None, "between 0 and 10"),
        ({"cfg_value": 10.5}, None, "between 0 and 10"),
        ({"cfg_value": 1e39}, None, "between 0 and 10"),
        ({"cfg_value": float("nan")}, None, "finite"),
        ({"inference_timesteps": 4}, None, "deployment setting"),
        ({"max_new_tokens": 0}, None, "max_new_tokens"),
        ({}, [torch.zeros(1000)], "between"),
        ({}, [torch.zeros(16000), torch.zeros(16000)], "exactly one"),
    ],
)
def test_process_prompt_rejects_bad_requests(model, kwargs, inputs, message):
    modalities = ["text", "audio"] if inputs else ["text"]
    tensors = {"audio_inputs": inputs} if inputs else None
    with pytest.raises(ValueError, match=message):
        model.process_prompt("hello", modalities, ["audio"], tensors=tensors, **kwargs)


@pytest.mark.parametrize("text", ["\U0001f600\U0001f600", "\U0001f389 \U0001f389", "...", "?!"])
def test_process_prompt_refuses_a_text_with_nothing_to_speak(model, text):
    for kwargs in ({}, {"instruct": "a calm voice"}):
        with pytest.raises(ValueError, match="nothing to speak"):
            model.process_prompt(text, ["text"], ["audio"], **kwargs)


@pytest.mark.parametrize("text", ["\U0001f600 hi", "42", "\u4f60\u597d"])
def test_process_prompt_speaks_any_letter_or_digit(model, text):
    model.process_prompt(text, ["text"], ["audio"])


def test_process_prompt_rejects_prompts_past_the_context(model):
    long_text = " ".join(["hello"] * 80)  # 80 tokens + 490 patches > 512 positions
    with pytest.raises(ValueError, match="context"):
        model.process_prompt(long_text, ["text"], ["audio"])


def test_process_prompt_bounds_the_text_length(checkpoint):
    model = VoxCPM2Model(model_path_hf=str(checkpoint), max_text_tokens=4)
    model.process_prompt("hello world hello world", ["text"], ["audio"])
    with pytest.raises(ValueError, match="5 tokens.*at most 4"):
        model.process_prompt("hello world hello world hello", ["text"], ["audio"])
    assert VoxCPM2Model(model_path_hf=str(checkpoint)).max_text_tokens == int(VoxCPM2Config().max_len / 3) == 1365


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "8"])
def test_max_text_tokens_must_be_a_positive_int(checkpoint, bad):
    with pytest.raises(ValueError, match="max_text_tokens"):
        VoxCPM2Model(model_path_hf=str(checkpoint), max_text_tokens=bad)


def test_shipped_cap_fits_both_caches_at_the_full_context(model):
    """Each LM cache holds the shipped ``max_concurrent_requests`` at the full
    context; one more request, or a smaller cache, is refused at startup."""
    deployment = yaml.safe_load(CONFIG_PATH.read_text())
    cap = deployment["max_concurrent_requests"]
    assert cap == VoxCPM2Model.DEFAULT_MAX_CONCURRENT
    model.validate_config_yaml(deployment, "voxcpm2.yaml")
    per_request = -(-model.config.max_length // 128)
    for spec in node_resources(model):
        if hasattr(spec.config, "max_num_pages"):
            assert spec.config.max_num_pages == 1 + cap * per_request
    with pytest.raises(ValueError, match=rf"{cap + 1} concurrent requests .*resources\.base_kv"):
        model.validate_config_yaml({**deployment, "max_concurrent_requests": cap + 1}, "voxcpm2.yaml")
    small = {**deployment, "resources": {"res_kv": {"max_num_pages": cap * per_request}}}
    with pytest.raises(ValueError, match=r"resources\.res_kv"):
        model.validate_config_yaml(small, "voxcpm2.yaml")
    large = {"max_concurrent_requests": 2 * cap, "resources": {
        key: {"max_num_pages": 1 + 2 * cap * per_request} for key in ("base_kv", "res_kv")
    }}
    model.validate_config_yaml(large, "voxcpm2.yaml")


@pytest.mark.parametrize("cap", [0, -2, 1.5, True, "8"])
def test_max_concurrent_requests_must_be_a_positive_int(model, cap):
    with pytest.raises(ValueError, match="positive int"):
        model.validate_config_yaml({"max_concurrent_requests": cap}, "voxcpm2.yaml")


def test_no_request_cap_is_a_warning(model, caplog):
    with caplog.at_level("WARNING"):
        model.validate_config_yaml({}, "voxcpm2.yaml")
    assert "sets no max_concurrent_requests" in caplog.text


def test_build_prompt_matches_the_reference_layout():
    text = torch.tensor([7, 8, AUDIO_START_TOKEN])
    ref = torch.randn(2, 3, 4, 8)
    tokens, text_mask, audio, pos = build_prompt(text, MODE_REF_CONTINUATION, ref)
    assert tokens.tolist() == [REF_AUDIO_START_TOKEN, 0, 0, 0, REF_AUDIO_END_TOKEN, 7, 8, AUDIO_START_TOKEN, 0, 0, 0]
    assert pos.tolist() == [1, 2, 3, 8, 9, 10] and (~text_mask).nonzero().flatten().tolist() == pos.tolist()
    assert torch.equal(audio, torch.cat([ref[0], ref[1]]))
    tokens, _, audio, pos = build_prompt(text, MODE_REFERENCE, ref[:1])
    assert tokens.tolist() == [REF_AUDIO_START_TOKEN, 0, 0, 0, REF_AUDIO_END_TOKEN, 7, 8, AUDIO_START_TOKEN]
    assert torch.equal(audio, ref[0])


def test_walk_plan_clones_through_the_ref_encoder(model):
    metadata = model.process_prompt(
        "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [torch.zeros(16000)]}
    ).metadata
    talker = model.walk_plan("Talker", ["text", "audio"], ["audio"], {}, {**metadata, "unrelated": 1})
    assert [step.walk for step in talker] == ["talker_prefill_clone", "talker_decode"]
    # the prompt facts ride step metadata to the Talker partition's nodes
    keys = (PATCH_BUDGET_KEY, PROMPT_MODE_KEY, MIN_LEN_KEY, REF_DIGEST_KEY, CFG_KEY)
    decode = (PATCH_BUDGET_KEY, MIN_LEN_KEY, CFG_KEY)
    assert [dict(step.step_metadata) for step in talker] == [
        {key: metadata[key] for key in keys}, {key: metadata[key] for key in decode},
    ]
    zero_shot = model.process_prompt("hello", ["text"], ["audio"]).metadata
    plain = model.walk_plan("Talker", ["text"], ["audio"], {}, zero_shot)
    assert [step.walk for step in plain] == ["talker_prefill", "talker_decode"]
    assert [step.walk for step in model.walk_plan("VAEDecoder", ["text"], ["audio"], {}, zero_shot)] == ["vae_chunk"]


@pytest.mark.parametrize("ref_text", ["", "world"])
def test_the_talker_walk_follows_the_prompt_mode_not_input_modalities(model, ref_text):
    """``/generate`` takes ``input_modalities`` from the client: a clip sent with
    ``["text"]`` builds a cloning prompt, so the conductor must run the
    RefEncoder walk for it too, or the Talker has no reference latents."""
    metadata = model.process_prompt(
        "hello", ["text"], ["audio"], tensors={"audio_inputs": [torch.zeros(16000)]}, ref_text=ref_text
    ).metadata
    assert metadata[PROMPT_MODE_KEY] != MODE_ZERO_SHOT
    prefill = model.walk_plan("Talker", ["text"], ["audio"], {}, metadata)[0]
    assert prefill.walk == "talker_prefill_clone"
    assert prefill.step_metadata[PROMPT_MODE_KEY] == metadata[PROMPT_MODE_KEY]


def test_the_talker_walk_needs_the_prompt_mode(model):
    """Only process_prompt sets the mode (INTERNAL_KWARGS drops a client's), so
    a request without one never reached it; refuse it rather than guess."""
    with pytest.raises(ValueError, match=PROMPT_MODE_KEY):
        model.walk_plan("Talker", ["text", "audio"], ["audio"], {}, {})




def test_load_audio_resamples_to_16k_and_bounds_duration(model, tmp_path):
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    path = tmp_path / "clip.wav"
    tone = 0.1 * torch.sin(torch.arange(44100) * 0.05)
    soundfile.write(path, tone.numpy(), 44100)
    loaded = model.load_audio(str(path), "cpu")
    assert loaded.metadata["sample_rate"] == 16000 and loaded.data.shape[0] == 16000
    soundfile.write(path, tone[:4410].numpy(), 44100)
    with pytest.raises(ValueError, match="between"):
        model.load_audio(str(path), "cpu")


def _pipe_written_flac(path: Path, seconds: float, rate: int) -> Path:
    """A FLAC that libsndfile wrote to a pipe, as ``ffmpeg ... -f flac -`` or
    ``sox ... -t flac -`` would: it cannot seek back to fill in STREAMINFO, so
    the header gives no sample count (libsndfile reports 2**63 - 1 frames)."""
    import os
    import threading

    soundfile = pytest.importorskip("soundfile")
    tone = (0.25 * torch.sin(torch.arange(int(seconds * rate)) * 0.05)).numpy()
    read_fd, write_fd = os.pipe()
    received = []

    def drain():
        with os.fdopen(read_fd, "rb") as pipe:
            received.append(pipe.read())

    reader = threading.Thread(target=drain)
    reader.start()
    with soundfile.SoundFile(
        write_fd, mode="w", samplerate=rate, channels=1, format="FLAC", subtype="PCM_16", closefd=True
    ) as sound:
        sound.write(tone)
    reader.join()
    path.write_bytes(received[0])
    assert soundfile.info(str(path)).frames == 2**63 - 1
    return path


# libsndfile cannot decode the last frame of a FLAC with no sample count; its encoder writes 4096 samples a frame
_LAST_FLAC_FRAME = 4096


def test_load_audio_reads_a_flac_written_to_a_pipe(model, tmp_path):
    """Its header names no length, so the length is counted as it decodes
    rather than read as 2**63 - 1 frames (a 400 "clip is 5.8e14 s")."""
    path = _pipe_written_flac(tmp_path / "piped.flac", 3.0, 16000)
    loaded = model.load_audio(str(path), "cpu")
    assert loaded.metadata["sample_rate"] == 16000
    assert 3 * 16000 - _LAST_FLAC_FRAME <= loaded.data.shape[0] <= 3 * 16000


def test_load_audio_resamples_a_flac_written_to_a_pipe(model, tmp_path):
    pytest.importorskip("soxr")
    path = _pipe_written_flac(tmp_path / "piped.flac", 3.0, 48000)
    samples = model.load_audio(str(path), "cpu").data.shape[0]
    assert 3 * 16000 - _LAST_FLAC_FRAME / 3 - 1 <= samples <= 3 * 16000


@pytest.mark.parametrize("seconds, shown", [(0.3, r"0\.\d\d"), (75.0, r"60\.\d\d")])
def test_load_audio_bounds_a_flac_written_to_a_pipe_at_its_counted_length(model, tmp_path, monkeypatch, seconds, shown):
    """Too short or too long is refused at the length counted; a long one is
    decoded only just past the 60 s limit."""
    from mstar.model import media_limits

    path = _pipe_written_flac(tmp_path / "piped.flac", seconds, 16000)
    blocks = []
    count = media_limits._sound_file_blocks

    def counting(*args):
        for block in count(*args):
            blocks.append(len(block))
            yield block

    monkeypatch.setattr(media_limits, "_sound_file_blocks", counting)
    with pytest.raises(ValueError, match=rf"reference clip is {shown} s; it must be between"):
        model.load_audio(str(path), "cpu")
    assert sum(blocks) <= 61 * 16000


@pytest.mark.parametrize("level", [0.0, 5e-4])
def test_load_audio_refuses_a_silent_clip(model, tmp_path, level):
    """Digital silence, or a constant offset with less than -60 dBFS of swing,
    clones no voice: a 400 instead of seconds of near-silent audio."""
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "silent.wav"
    soundfile.write(path, (0.25 + level * torch.sin(torch.arange(32000) * 0.05)).numpy(), 16000)
    with pytest.raises(ValueError, match="silent"):
        model.load_audio(str(path), "cpu")


@pytest.mark.parametrize("payload", [b"just some text, not audio\n", b"RIFF\x24\x00\x00\x00WAVEfmt "])
def test_load_audio_refuses_non_audio_without_naming_the_path(model, tmp_path, payload):
    """An upload libsndfile cannot open is the client's error, and the message
    the client gets does not show where the server stored it."""
    pytest.importorskip("soundfile")
    path = tmp_path / "uploads" / "f3a1c0de.wav"
    path.parent.mkdir()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="could not decode the reference audio") as raised:
        model.load_audio(str(path), "cpu")
    assert str(tmp_path) not in str(raised.value) and path.name not in str(raised.value)


# -- Talker -----------------------------------------------------------------------


def _talker(model, timesteps: int = 4):
    generator = _random_generator(model.config)
    talker = TalkerSubmodule(generator, model.config, timesteps)
    plan, reset = bind_dense_resources(talker)
    return talker, plan, reset


def _commit(talker, walk: str, rids: list, prepared: list) -> None:
    """What exec does to the Talker's host fields: declare the step over the prepared rows and commit it. The
    fields open as a request's handle is minted, which these tests do on its first commit."""
    fields = talker.progress.host_fields
    for rid in rids:
        if not fields.is_open(rid):
            fields.open(rid)
    step = talker.declare_step(walk, rids, prepared)
    step.set_ctx(StepContext(tuple(rids), walk, 0, False))
    talker.progress.begin_step(step)
    talker.progress.commit_step(step)
    talker.progress.end_step(step)


def _run(talker, plan, requests: dict, steps: int, trace: dict | None = None, runners: dict | None = None):
    """Prefill every request together, then ``steps - 1`` batched decode steps,
    each committed as exec does. ``trace`` collects each decode step's index and
    noise per request."""
    rids = list(requests)
    infos = {rid: _info(rid, seed, meta) for rid, ((_, meta), seed) in requests.items()}
    decode_infos = {rid: _decode_info(rid, seed, meta) for rid, ((_, meta), seed) in requests.items()}
    engine = SimpleNamespace(request_ids=rids, piecewise_runners=runners or {}, captured=False)
    prepared = [talker.prepare_inputs("talker_prefill", infos[rid], requests[rid][0][0]) for rid in rids]
    plan([p.input_seq_len for p in prepared])
    out = talker.forward_batched("talker_prefill", engine, **talker.preprocess("talker_prefill", engine, prepared))
    patches = {rid: [out[rid]["latent_patch"][0]] for rid in rids}
    _commit(talker, "talker_prefill", rids, prepared)
    for rid, p in zip(rids, prepared, strict=True):
        talker.postprocess(rid, infos[rid], out[rid], inputs=p)
    for _ in range(steps - 1):
        prepared = [
            talker.prepare_inputs("talker_decode", decode_infos[rid], {"latent_patch": [patches[rid][-1]]})
            for rid in rids
        ]
        plan([1] * len(rids))
        out = talker.forward_batched("talker_decode", engine, **talker.preprocess("talker_decode", engine, prepared))
        _commit(talker, "talker_decode", rids, prepared)
        for rid, p in zip(rids, prepared, strict=True):
            talker.postprocess(rid, infos[rid], out[rid], inputs=p)
            patches[rid].append(out[rid]["latent_patch"][0])
            if trace is not None:
                trace.setdefault(rid, []).append((p.kwargs["step"], p.tensor_inputs["noise"]))
    return {rid: torch.stack(p) for rid, p in patches.items()}


def test_talker_batches_like_single_requests_and_follows_the_seed(model):
    talker, plan, reset = _talker(model)
    a = _request(model, "hello world")
    b = _request(model, "hello hello hello", cfg_value=3.0)
    with torch.inference_mode():
        together = _run(talker, plan, {"a": (a, 5), "b": (b, 9)}, steps=4)
        reset()
        alone_a = _run(talker, plan, {"a": (a, 5)}, steps=4)
        reset()
        alone_b = _run(talker, plan, {"b": (b, 9)}, steps=4)
        reset()
        reseeded = _run(talker, plan, {"a": (a, 6)}, steps=4)
    torch.testing.assert_close(together["a"], alone_a["a"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(together["b"], alone_b["b"], rtol=1e-4, atol=1e-5)
    assert together["a"].shape == (4, 4, 8)
    assert not torch.allclose(reseeded["a"], alone_a["a"])


class _ReplayRunner:
    """CPU stand-in for a captured piecewise region: the real rows go into the
    region's own static buffers at the padded batch size and its capture
    function runs on them, as a replay of the graph would."""

    def __init__(self, region: PiecewiseBatchedConfig):
        self.region = region
        self.calls: list[tuple[int, int]] = []

    def can_run(self, batch_size: int) -> bool:
        return batch_size <= max(self.region.capture_batch_sizes)

    def run(self, static_inputs: dict, real_bs: int) -> dict:
        bs = min(s for s in self.region.capture_batch_sizes if s >= real_bs)
        shape = PiecewiseCaptureShape(bs=bs, seq_lens=[self.region.seq_len] * bs, total_tokens=self.region.seq_len * bs)
        buffers = self.region.make_static_inputs(shape)
        assert set(static_inputs) == set(buffers)
        for name, value in static_inputs.items():
            assert value.shape[1:] == buffers[name].shape[1:] and value.dtype == buffers[name].dtype, name
            buffers[name][:real_bs].copy_(value)
        self.calls.append((bs, real_bs))
        out = self.region.capture_fn(PiecewiseCallInputs(static_inputs=buffers, engine_inputs=None))
        return {name: tensor[:real_bs] for name, tensor in out.items()}


def test_prefill_tail_region_replays_the_eager_tail(model):
    """Three prefills padded to the 4-row bucket give the eager prefill's
    patches, and the decode steps after them carry on the same."""
    talker, plan, reset = _talker(model)
    ((label, region),) = talker.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.bfloat16).items()
    assert label == PREFILL_TAIL and region.seq_len == 1
    assert region.capture_batch_sizes == talker.DECODE_CAPTURE_BATCH_SIZES
    runner = _ReplayRunner(region)
    requests = {
        "a": (_request(model, "hello world"), 5),
        "b": (_request(model, "hello hello hello", cfg_value=3.0), 9),
        "c": (_request(model, "hi"), 2),
    }
    with torch.inference_mode():
        eager = _run(talker, plan, requests, steps=3)
        reset()
        replayed = _run(talker, plan, requests, steps=3, runners={PREFILL_TAIL: runner})
    assert runner.calls == [(4, 3)]
    for rid in requests:
        torch.testing.assert_close(replayed[rid], eager[rid], rtol=1e-4, atol=1e-5)


def test_talker_decode_steps_advance_the_patch_index_and_the_noise(model):
    """Each decode step takes the next patch index and the next draw of the
    request's seeded generator (prefill takes draw 0)."""
    talker, plan, _ = _talker(model)
    request = _request(model, "hello world")
    trace: dict = {}
    with torch.inference_mode():
        _run(talker, plan, {"a": (request, 5)}, steps=4, trace=trace)
    expected = torch.Generator().manual_seed(5)
    draws = [torch.randn((1, 8, 4), generator=expected)[0] for _ in range(4)]
    assert [index for index, _ in trace["a"]] == [1, 2, 3]
    for (_, noise), draw in zip(trace["a"], draws[1:], strict=True):
        torch.testing.assert_close(noise, draw)


def test_talker_clone_prefill_streams_the_prompt_tail_as_context(model):
    talker, plan, _ = _talker(model)
    request, meta = _request(model, "hello", torch.zeros(16000), ref_text="world ")
    ref = torch.randn(2, 7, 4, 8)
    info = _info("c", 1, meta)
    prepared = talker.prepare_inputs("talker_prefill_clone", info, {**request, "ref_patches": [ref]})
    assert prepared.input_seq_len == 1 + 7 + 1 + 3 + 7
    torch.testing.assert_close(prepared.tensor_inputs["cond_patch"], ref[1][-1])
    plan([prepared.input_seq_len])
    engine = SimpleNamespace(request_ids=["c"], piecewise_runners={}, captured=False)
    with torch.inference_mode():
        out = talker.forward_batched(
            "talker_prefill_clone", engine, **talker.preprocess("talker_prefill_clone", engine, [prepared])
        )["c"]
    # the forward streams the patch alone, which a captured prefill can; postprocess leads it
    assert len(out[STREAM_EDGE]) == 1
    talker.postprocess("c", info, out, inputs=prepared)
    stream = out[STREAM_EDGE]
    assert len(stream) == 4 and all(torch.equal(s, r) for s, r in zip(stream[:3], ref[1][-3:], strict=True))
    assert torch.equal(stream[3], out["latent_patch"][0])


def test_talker_stops_like_the_reference_loop(model):
    """``i > min_len and stop`` on the step that produced patch i (kept), or the patch budget."""
    talker, _, _ = _talker(model)
    request, meta = _request(model, "hello", max_new_tokens=6)
    info = _info("r", 0, meta)
    talker.prepare_inputs("talker_prefill", info, request)

    def step(flag: int, index: int) -> set[str]:
        return talker.check_stop("r", info, {"stop_flag": [torch.tensor([flag, index])]})

    assert [step(1, i) for i in range(3)] == [set(), set(), set()]  # i = 0, 1, 2: not past min_len 2
    assert step(1, 3) == {TALKER_LOOP}
    assert [step(0, i) for i in range(6)][-2:] == [set(), {TALKER_LOOP}]  # budget of 6 patches


def test_talker_vetoes_steps_past_the_patch_budget(model):
    talker, _, _ = _talker(model)
    request, meta = _request(model, "hello", max_new_tokens=2)
    info = _info("r", 0, meta)
    _commit(talker, "talker_prefill", ["r"], [talker.prepare_inputs("talker_prefill", info, request)])
    patch = torch.zeros(4, 8)
    first = talker.prepare_inputs("talker_decode", _decode_info("r", 0, meta), {"latent_patch": [patch]})
    assert first is not None and first.kwargs["step"] == 1
    _commit(talker, "talker_decode", ["r"], [first])
    assert talker.prepare_inputs("talker_decode", _decode_info("r", 0, meta), {"latent_patch": [patch]}) is None


def test_talker_reprepares_a_refused_step_with_the_same_index_and_noise(model):
    """A step whose admit was refused is prepared again: nothing ran, so it is
    the same patch index and the same noise draw, and the budget is not spent."""
    talker, _, _ = _talker(model)
    request, meta = _request(model, "hello", max_new_tokens=2)
    info = _info("r", 0, meta)
    _commit(talker, "talker_prefill", ["r"], [talker.prepare_inputs("talker_prefill", info, request)])
    patch = torch.zeros(4, 8)
    refused = talker.prepare_inputs("talker_decode", _decode_info("r", 0, meta), {"latent_patch": [patch]})
    retry = talker.prepare_inputs("talker_decode", _decode_info("r", 0, meta), {"latent_patch": [patch]})
    assert retry is not None and retry.kwargs["step"] == 1
    assert torch.equal(retry.tensor_inputs["noise"], refused.tensor_inputs["noise"])


def _committed(talker, rid) -> dict:
    return dict(host(talker.progress, rid, "main", 0))


def test_talker_host_writes_rely_on_exec_declaring_again_after_a_pre_plan(model):
    """A pre-plan declares over the bucket's template rows, which carry no draw and so no write. Were exec to run that
    step instead of declaring its own, ``generated`` and ``rng`` would never move and every decode step would draw the
    prefill's next noise again."""
    talker, _, _ = _talker(model)
    [graph] = [
        g for g in talker.get_accelerator_graph_configs(torch.device("cpu")) if g.capture_graph_walk == "talker_decode"
    ]
    request, meta = _request(model, "hello")
    info = _info("r", 0, meta)
    _commit(talker, "talker_prefill", ["r"], [talker.prepare_inputs("talker_prefill", info, request)])
    before = _committed(talker, "r")

    rows = graph.get_node_inputs(2, 0)
    runner = SimpleNamespace(declare_inputs_for=lambda lease: rows, step_ids=lambda lease, rids: [*rids, "pad"])
    context = StepContext(("r",), "talker_decode", 0, False)
    batch = SimpleNamespace(node_name="Talker", inputs=None, request_ids=["r"],
                            per_request_info={"r": _decode_info("r", 0, meta)}, step=None,
                            step_context=context)
    engine = Engine.__new__(Engine)
    # a refused pre-admit ends the pre-plan with everything this test reads already decided
    reused = []
    engine._runner = SimpleNamespace(pre_admit=lambda step: reused.append(batch.step) or SimpleNamespace(ok=False))
    engine.reset_pre_plan_for_batch = lambda batch: None
    # no runners, so that `step_is_input_free` alone decides
    mgmt = SimpleNamespace(submodule=talker, piecewise_runners={})
    assert engine._pre_plan_for_batch(batch, mgmt, runner, object()) is False
    assert reused == [None], "exec reuses a pre-planned step only for a walk with no input in its declaration"
    assert _committed(talker, "r") == before, "a pre-plan writes nothing"

    prepared = talker.prepare_inputs("talker_decode", _decode_info("r", 0, meta), {"latent_patch": [torch.zeros(4, 8)]})
    step = engine_step(talker, graph_walk="talker_decode", request_ids=["r", "pad"], inputs=[prepared, rows[1]])
    context.is_preplan = False
    step.set_ctx(context)
    step.begin_calls()
    step.commit_calls()
    step.end_calls()
    assert _committed(talker, "r") == {"generated": before["generated"] + 1, "rng": prepared.kwargs["rng"]}
    assert prepared.kwargs["rng"] != before["rng"], "the draw moved the generator"


def test_talker_declaring_a_step_twice_advances_it_once(model):
    talker, _, _ = _talker(model)
    request, meta = _request(model, "hello")
    info = _info("r", 0, meta)
    prepared = talker.prepare_inputs("talker_prefill", info, request)
    talker.declare_step("talker_prefill", ["r"], [prepared])
    _commit(talker, "talker_prefill", ["r"], [prepared])
    assert _committed(talker, "r")["generated"] == 1, "only a commit applies a declared write"


@pytest.mark.parametrize("walk", ["talker_prefill", "talker_decode"])
def test_talker_capture_rows_carry_cfg_and_no_rng_and_write_nothing(model, walk):
    talker, _, _ = _talker(model)
    graphs = {graph.capture_graph_walk: graph for graph in talker.get_accelerator_graph_configs(torch.device("cpu"))}
    rows = graphs[walk].get_node_inputs(2, 128)
    assert all("rng" not in row.kwargs for row in rows), "a capture's rows have no open fields to write"
    step = talker.declare_step(walk, [91, 92], rows)
    assert TALKER_STEP not in step.state
    packed = talker.preprocess(walk, SimpleNamespace(request_ids=[91, 92]), rows)
    assert packed["cfg"].tolist() == [model.config.cfg_value] * 2


def test_talker_padded_replay_writes_the_real_rows_and_stages_each_rows_cfg(model):
    """``preprocess`` sees padding rows as copies of the template, and ``declare_step`` tells them from real rows by
    ``rng``, so one real row in a padded bucket advances and the padding rows touch no store."""
    talker, _, _ = _talker(model)
    [graph] = [
        g for g in talker.get_accelerator_graph_configs(torch.device("cpu")) if g.capture_graph_walk == "talker_decode"
    ]
    request, meta = _request(model, "hello", cfg_value=1.5)
    info = _info("r", 0, meta)
    _commit(talker, "talker_prefill", ["r"], [talker.prepare_inputs("talker_prefill", info, request)])
    prepared = talker.prepare_inputs("talker_decode", _decode_info("r", 0, meta), {"latent_patch": [torch.zeros(4, 8)]})
    rows, rids = [prepared, *graph.get_node_inputs(3, 0)], ["r", 91, 92, 93]

    packed = talker.preprocess("talker_decode", SimpleNamespace(request_ids=rids), rows)
    assert packed["cfg"].tolist() == [1.5, *[model.config.cfg_value] * 3]
    assert packed["step_index"].tolist() == [1, 1, 1, 1]

    step = engine_step(talker, graph_walk="talker_decode", request_ids=rids, inputs=rows)
    assert set(step.state[TALKER_STEP].host) == {("r", "main", 0)}
    context = StepContext(("r",), "talker_decode", 0, False, slot_lease=object())
    context.set_padded_rids(rids)
    assert context.replays
    step.set_ctx(context)
    step.begin_calls()
    step.commit_calls()
    step.end_calls()
    assert _committed(talker, "r")["generated"] == 2
    assert set(talker.progress.host_fields._fields) == {"r"}, "a padding row opens no fields"


def test_talker_fields_go_with_the_request_when_the_engine_removes_it(model):
    talker, _, _ = _talker(model)
    request, meta = _request(model, "hello")
    info = _info("r", 0, meta)
    _commit(talker, "talker_prefill", ["r"], [talker.prepare_inputs("talker_prefill", info, request)])
    engine = Engine.__new__(Engine)
    engine._loopback_steps, engine._resources, engine._submodules = {}, {}, {}
    engine._runner = SimpleNamespace(remove_request=lambda rid: None)
    engine._chunking = SimpleNamespace(forget_request=lambda rid: None)
    engine._state_setup = StateSetup(host_fields=talker.progress.host_fields)
    engine.remove_request("r")
    assert talker.progress.host_fields._fields == {}, "a recycled handle would read this request's progress"
    with pytest.raises(ValueError, match="no host fields"):
        talker.prepare_inputs("talker_decode", _decode_info("r", 0, meta), {"latent_patch": [torch.zeros(4, 8)]})


@pytest.mark.parametrize("no_rope", [True, False])
def test_the_residual_stack_rotates_at_the_base_arrays_positions_or_not_at_all(model, no_rope):
    """The released checkpoint sets ``residual_lm_no_rope``, and the flag decides whether its layers read positions."""
    config = replace(model.config, residual=replace(model.config.residual, no_rope=no_rope))
    talker = TalkerSubmodule(_random_generator(config), config, 4)
    bind_dense_resources(talker)
    layer = talker.model.residual_lm.layers[0].self_attn
    base = talker.state_handles[BASE_KV]
    assert (layer.position_source, layer._rope is None) == ((None, True) if no_rope else (base, False))


def test_talker_noise_comes_in_order_from_the_seed_through_host_bytes(model):
    talker, _, _ = _talker(model)
    first, after_first = talker._draw(b"", 4)
    second, _ = talker._draw(after_first, 4)
    again, _ = talker._draw(after_first, 4)
    assert torch.equal(again, second), "a re-prepared step draws from the same committed state"
    expected = torch.Generator().manual_seed(4)
    torch.testing.assert_close(first, torch.randn((1, 8, 4), generator=expected)[0])
    torch.testing.assert_close(second, torch.randn((1, 8, 4), generator=expected)[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_talker_noise_through_host_bytes_equals_one_kept_cuda_generator(model):
    """A CUDA generator's state is its seed and Philox offset, so bytes carry it across steps, with a refused step in
    between, as one kept generator would have drawn."""
    talker = TalkerSubmodule(_random_generator(model.config).cuda(), model.config, 4)
    kept = torch.Generator(device="cuda").manual_seed(4)
    state = b""
    for step in range(4):
        noise, after = talker._draw(state, 4)
        if step == 2:
            again, _ = talker._draw(state, 4)
            assert torch.equal(again, noise), "a refused step draws the same noise from the same committed state"
        expected = torch.randn((1, 8, 4), generator=kept, device="cuda", dtype=talker._dtype())[0]
        assert torch.equal(noise, expected), f"draw {step}"
        state = after


def test_talker_declares_the_same_span_in_both_caches(model):
    talker, _, _ = _talker(model)
    step = engine_step(talker, "talker_decode", ["a", "b"], [ARNodeInputs(input_seq_len=1)] * 2)
    assert set(step.steps) == {"base_kv", "base_attn", "base_pos", "res_kv", "res_attn"}
    assert [(s.request_id, s.span) for s in step.segments] == [("a", 1), ("b", 1)]
    (graph,) = [g for g in talker.get_accelerator_graph_configs(torch.device("cpu"))
                if g.capture_graph_walk == "talker_decode"]
    assert graph.capture_graph_walk == "talker_decode"


def _ref_encoder(model) -> RefEncoderSubmodule:
    return RefEncoderSubmodule(_random_vae(model.config), model.config)


def _clip_hash(clip: torch.Tensor) -> str:
    """blake2b of the clip's float32 samples, taken here without ``reference_digest``."""
    return hashlib.blake2b(clip.to(torch.float32).reshape(-1).cpu().numpy().tobytes(), digest_size=16).hexdigest()


@pytest.mark.parametrize("ref_text", ["", "world "])
def test_ref_encoder_keys_its_memo_by_clip_content_and_mode_from_step_metadata(model, ref_text):
    """The digest ``process_prompt`` takes on the host is the hash of the
    clip the RefEncoder receives, so a later request with the clip hits the memo."""
    clip = torch.randn(20000, generator=torch.Generator().manual_seed(1)) * 0.1
    mode = MODE_REF_CONTINUATION if ref_text else MODE_REFERENCE
    edges, meta = _request(model, "hello", clip, ref_text=ref_text)
    delivered = {"audio_inputs": [clip.clone()]}  # the transport hands over a copy of the bytes
    prepared = _ref_encoder(model).prepare_inputs("talker_prefill_clone", _info("r", 0, meta), delivered)
    assert prepared.kwargs == {"mode": mode, "key": (_clip_hash(clip), mode)}
    assert torch.equal(prepared.tensor_inputs["clip"], clip)
    # the same voice in the other mode is another encoding; another clip is another key
    other_mode = _request(model, "hello", clip, ref_text="" if ref_text else "world ")[1]
    assert other_mode[REF_DIGEST_KEY] == meta[REF_DIGEST_KEY] and other_mode[PROMPT_MODE_KEY] != mode
    assert _request(model, "hello", clip * 0.5, ref_text=ref_text)[1][REF_DIGEST_KEY] != meta[REF_DIGEST_KEY]
    # an encode is memoised under that key and hit by the next request with the clip
    encoder = _ref_encoder(model)
    with torch.inference_mode():
        first = encoder.forward("talker_prefill_clone", None, **prepared.tensor_inputs, **prepared.kwargs)
        again = encoder.forward("talker_prefill_clone", None, **prepared.tensor_inputs, **prepared.kwargs)
    assert list(encoder._cache) == [(_clip_hash(clip), mode)]
    assert again["ref_patches"][0] is first["ref_patches"][0]


def test_talker_packs_a_mixed_batch_like_per_request_prompts(model):
    """Zero-shot, voice design, reference and continuation prefilled together,
    then a decode step at different patch counts: every packed tensor equals
    the concatenation of each request's ``build_prompt`` layout (mode, patch
    budget and min_len from step metadata), in values, dtype and shape."""
    talker, plan, _ = _talker(model)
    clip = torch.randn(18000, generator=torch.Generator().manual_seed(2)) * 0.1
    requests = {
        "zero": _request(model, "hello world"),
        "design": _request(model, "hello hello", instructions="warm", cfg_value=3.0, min_len=1),
        "ref": _request(model, "hello", clip, cfg_value=1.5),
        "cont": _request(model, "world hello", clip, ref_text="hello ", max_new_tokens=7),
    }
    expected_layouts = {
        "zero": [MODE_ZERO_SHOT, 2 * 6 + 10, 2],
        "design": [MODE_ZERO_SHOT, 2 * 6 + 10, 1],  # "(warm)hello hello": the description joins the first word
        "ref": [MODE_REFERENCE, 1 * 6 + 10, 2],
        "cont": [MODE_REF_CONTINUATION, 7, 2],
    }
    assert {rid: _layout(meta) for rid, (_, meta) in requests.items()} == expected_layouts
    n_ref = math.ceil(clip.numel() / model.config.ref_patch_samples)
    refs = {rid: torch.randn(2 if rid == "cont" else 1, n_ref, 4, 8) for rid in ("ref", "cont")}
    rids = list(requests)
    infos = {rid: _info(rid, i, meta) for i, (rid, (_, meta)) in enumerate(requests.items())}
    prepared = [
        talker.prepare_inputs(
            "talker_prefill_clone" if rid in refs else "talker_prefill",
            infos[rid],
            {**edges, **({"ref_patches": [refs[rid]]} if rid in refs else {})},
        )
        for rid, (edges, _) in requests.items()
    ]
    engine = SimpleNamespace(request_ids=rids, piecewise_runners={}, captured=False)
    packed = talker.preprocess("talker_prefill_clone", engine, prepared)

    layouts = [
        build_prompt(edges["text_inputs"][0], expected_layouts[rid][0], refs.get(rid))
        for rid, (edges, _) in requests.items()
    ]
    seq_lens = [tokens.shape[0] for tokens, *_ in layouts]
    offsets = torch.tensor([0, *seq_lens[:-1]]).cumsum(0)
    expected = {
        "token_ids": torch.cat([tokens for tokens, *_ in layouts]),
        "text_mask": torch.cat([mask for _, mask, *_ in layouts]),
        "audio_patches": torch.cat([_per_token(tokens, audio, pos) for tokens, _, audio, pos in layouts]),
        "last_token_indices": torch.tensor(seq_lens).cumsum(0) - 1,
        "step_index": torch.zeros(len(rids), dtype=torch.long),
        "cfg": torch.tensor([2.0, 3.0, 1.5, 2.0]),
    }
    for name, value in expected.items():
        got = packed[name]
        assert got.dtype == value.dtype and got.shape == value.shape and torch.equal(got, value), name
    assert [p.kwargs.get("prompt_patches") is not None for p in prepared] == [False, False, False, True]

    plan(seq_lens)
    with torch.inference_mode():
        out = talker.forward_batched("talker_prefill_clone", engine, **packed)
    patches = {rid: out[rid]["latent_patch"][0] for rid in rids}
    _commit(talker, "talker_prefill_clone", rids, prepared)
    design = ARNodeInputs(input_seq_len=1, kwargs={"step": 1, "rng": prepared[1].kwargs["rng"]})
    _commit(talker, "talker_decode", ["design"], [design])  # one row a patch ahead
    decode_infos = {rid: _decode_info(rid, i, meta) for i, (rid, (_, meta)) in enumerate(requests.items())}
    prepared = [
        talker.prepare_inputs("talker_decode", decode_infos[rid], {"latent_patch": [patches[rid]]}) for rid in rids
    ]
    packed = talker.preprocess("talker_decode", engine, prepared)
    assert list(packed) == ["noise", "cfg", "step_index", "prev_patch"]
    assert packed["step_index"].dtype == torch.long and packed["step_index"].tolist() == [1, 2, 1, 1]
    assert torch.equal(packed["prev_patch"], torch.stack([patches[rid] for rid in rids]))


def _per_token(tokens, audio, pos):
    """``build_prompt``'s audio patches at their positions, zeros at the text."""
    out = torch.zeros(tokens.shape[0], *((audio.shape[1:]) if audio is not None else (4, 8)))
    if audio is not None:
        out = out.to(audio.dtype).index_copy(0, pos, audio)
    return out


def test_capture_dummy_inputs_keep_the_static_keys_dtypes_and_shapes(model):
    """The decode graph's static inputs: what preprocess builds from the capture template at every bucket."""
    talker, _, _ = _talker(model)
    (graph,) = [g for g in talker.get_accelerator_graph_configs(torch.device("cpu"))
                if g.capture_graph_walk == "talker_decode"]
    cfg, dtype = model.config, talker._dtype()
    for bs in graph.capture_batch_sizes:
        packed = talker.preprocess(
            "talker_decode", SimpleNamespace(request_ids=list(range(bs))), graph.get_node_inputs(bs, bs)
        )
        expected = {
            "noise": torch.zeros(bs, cfg.feat_dim, cfg.patch_size, dtype=dtype),
            "cfg": torch.full((bs,), cfg.cfg_value, dtype=torch.float32),
            "step_index": torch.ones(bs, dtype=torch.long),
            "prev_patch": torch.zeros(bs, cfg.patch_size, cfg.feat_dim, dtype=dtype),
        }
        assert list(packed) == list(expected)
        for name, value in expected.items():
            assert packed[name].dtype == value.dtype and torch.equal(packed[name], value), (bs, name)


def test_compile_talker_switch_reaches_the_decode_graph(checkpoint, monkeypatch):
    from mstar.model.voxcpm2 import voxcpm2_model

    (checkpoint / "model.safetensors").touch()
    monkeypatch.setattr(voxcpm2_model, "load_generator", lambda config, *args: _random_generator(config))
    for kwargs, compiled in (({}, True), ({"compile_talker": True}, True), ({"compile_talker": False}, False)):
        model = VoxCPM2Model(model_path_hf=str(checkpoint), **kwargs)
        graphs = {g.capture_graph_walk: g for g in model.get_submodule("Talker").get_accelerator_graph_configs("cpu")}
        assert model.compile_talker is compiled and graphs["talker_decode"].compile is compiled
        assert graphs["talker_prefill"].compile is False
    with pytest.raises(ValueError, match="compile_talker"):
        VoxCPM2Model(model_path_hf=str(checkpoint), compile_talker="false")


def _ladder(lengths):
    """A packed capture's token ladder, as a list or (after #1184) {min batch size: ladder}."""
    return sorted({n for rung in lengths.values() for n in rung}) if isinstance(lengths, dict) else list(lengths)


def test_the_prefill_captures_the_talkers_ladder_for_both_prefill_walks(model):
    talker = TalkerSubmodule(_random_generator(model.config), model.config, 4)
    graphs = {g.capture_graph_walk: g for g in talker.get_accelerator_graph_configs(torch.device("cpu"))}
    prefill = graphs["talker_prefill"]
    assert _ladder(prefill.capture_token_lengths) == TalkerSubmodule.PREFILL_TOKEN_BUCKETS
    assert prefill.capture_batch_sizes == TalkerSubmodule.PREFILL_BATCH_SIZES
    assert max(prefill.capture_batch_sizes) <= TalkerSubmodule.MAX_BATCH_SIZE
    assert set(prefill.replay_graph_walks) == {"talker_prefill", "talker_prefill_clone"}


def test_prefill_capture_rows_pack_into_bucket_shaped_inputs(model):
    """The capture's rows, and a replay's zero-length padding rows, pack per token or per row."""
    generator = _random_generator(model.config)
    talker = TalkerSubmodule(generator, model.config, 4)
    talker.PREFILL_TOKEN_BUCKETS, talker.PREFILL_BATCH_SIZES = [64], [4]
    graphs = {g.capture_graph_walk: g for g in talker.get_accelerator_graph_configs(torch.device("cpu"))}
    prefill = graphs["talker_prefill"]
    rows = prefill.get_node_inputs(4, 64)
    engine = SimpleNamespace(request_ids=list(range(4)), captured=True)
    packed = talker.preprocess("talker_prefill", engine, rows)
    assert set(packed) == set(talker._PREFILL_STATIC_KEYS)
    for key in ("token_ids", "text_mask", "audio_patches"):
        assert packed[key].shape[0] == 64, key
    for key in ("last_token_indices", "cond_patch", "noise", "cfg", "step_index"):
        assert packed[key].shape[0] == 4, key
    # one real row of 10 tokens and three padding rows: they all point at a real token
    padded = [talker._prefill_capture_row(10, torch.device("cpu")), *prefill.get_node_inputs(3, 0)]
    assert talker.preprocess("talker_prefill", engine, padded)["last_token_indices"].tolist() == [9, 9, 9, 9]


def test_the_captured_branch_hands_the_prefill_the_same_inputs(model):
    """The forward's captured branch (the whole prefill in one call) and its eager one
    (the LMs, then the tail) are the same math, so this checks the plumbing into
    each, not a capture: nothing here is captured, replayed or padded."""
    talker, plan, reset = _talker(model)
    requests = {
        "a": (_request(model, "hello there", None), 3),
        "b": (_request(model, "hi", torch.randn(16000), ref_text="yo "), 4),
    }
    refs = {"b": torch.randn(2, 7, 4, 8)}
    outs = []
    for captured in (False, True):
        reset()
        infos = {rid: _info(rid, seed, meta) for rid, ((_, meta), seed) in requests.items()}
        prepared = [
            talker.prepare_inputs(
                "talker_prefill_clone", infos[rid], {**edges, **({"ref_patches": [refs[rid]]} if rid in refs else {})}
            )
            for rid, ((edges, _), _) in requests.items()
        ]
        engine = SimpleNamespace(request_ids=list(requests), piecewise_runners={}, captured=captured)
        plan([p.input_seq_len for p in prepared])
        with torch.inference_mode():
            out = talker.forward_batched("talker_prefill_clone", engine, **talker.preprocess("x", engine, prepared))
        outs.append(out)
    for rid in requests:
        torch.testing.assert_close(outs[1][rid]["latent_patch"][0], outs[0][rid]["latent_patch"][0])
        assert torch.equal(outs[1][rid]["stop_flag"][0], outs[0][rid]["stop_flag"][0])


# -- VAE stream -------------------------------------------------------------------


@pytest.mark.parametrize("num_patches", [1, 2, 3, 4, 7, 8, 15, 16, 24, 31])
def test_streamed_audio_equals_a_whole_decode(model, num_patches):
    config = model.config
    vae_node = VAEDecoderSubmodule(_random_vae(config).double(), config)
    torch.manual_seed(num_patches)
    patches = torch.randn(num_patches, config.patch_size, config.feat_dim, dtype=torch.float64)
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    with torch.no_grad():
        chunks = stream_through(vae_node, patches, policy)
        whole = vae_node.vae.decode(patches.reshape(-1, config.feat_dim).t().unsqueeze(0))[0]
    streamed = torch.cat(chunks)
    _assert_unsaturated(whole)
    assert streamed.numel() == whole.numel() == num_patches * config.patch_samples
    # float64: the windows reproduce it up to rounding (bit equality depends on the CPU's kernels)
    assert (streamed - whole).abs().max() <= FP64_ROUNDING
    # the first audio leaves after one patch
    assert chunks[0].numel() == config.patch_samples


def test_stream_context_below_the_receptive_field_changes_the_audio(model):
    config = model.config
    vae_node = VAEDecoderSubmodule(_random_vae(config, residual_gain=1.0).double(), config)
    torch.manual_seed(0)
    # long enough that later windows start past patch 0 and the short context bites
    patches = torch.randn(model.stream_left_context + 10, config.patch_size, config.feat_dim, dtype=torch.float64)
    short = ScheduledLeftContextChunkPolicy(schedule=(1, 2, 4), chunk=8, left_context=model.stream_left_context - 1)
    with torch.no_grad():
        streamed = torch.cat(stream_through(vae_node, patches, short))
        whole = vae_node.vae.decode(patches.reshape(-1, config.feat_dim).t().unsqueeze(0))[0]
    assert streamed.numel() == whole.numel()
    # float64 windows with the full context match to rounding, so a larger difference is the missing context
    assert (streamed - whole).abs().max() > 100 * FP64_ROUNDING  # about 8e-11 with these weights


def test_streamed_continuation_skips_the_prompt_lead(model):
    """A continuation leads its stream with 3 prompt patches: their audio is
    decoded as context, as the reference decodes it, and never emitted."""
    config = model.config
    vae_node = VAEDecoderSubmodule(_random_vae(config).double(), config)
    lead = torch.randn(3, 4, 8, dtype=torch.float64)
    patches = torch.randn(11, 4, 8, dtype=torch.float64)
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    with torch.no_grad():
        streamed = torch.cat(stream_through(vae_node, patches, policy, lead=lead))
        whole = vae_node.vae.decode(torch.cat([lead, patches]).reshape(-1, 8).t().unsqueeze(0))[0]
    _assert_unsaturated(whole)
    assert (streamed - whole[3 * config.patch_samples :]).abs().max() <= FP64_ROUNDING


def test_vae_batches_windows_of_different_widths(model):
    config = model.config
    vae_node = VAEDecoderSubmodule(_random_vae(config).double(), config)
    info = SimpleNamespace(rid_handle="x", request_id="x")
    windows = {"a": (torch.randn(1, 4, 8, dtype=torch.float64), 0), "b": (torch.randn(6, 4, 8, dtype=torch.float64), 2)}
    prepared = {
        rid: vae_node.prepare_inputs(
            "vae_chunk",
            info,
            {STREAM_EDGE: [w]},
            input_metadata=InputMetadata(stream_chunks={STREAM_EDGE: SimpleNamespace(context_items=ctx)}),
        )
        for rid, (w, ctx) in windows.items()
    }
    engine = SimpleNamespace(request_ids=list(windows), piecewise_runners={})
    with torch.no_grad():
        out = vae_node.forward_batched(
            "vae_chunk", engine, **vae_node.preprocess("vae_chunk", engine, list(prepared.values()))
        )
        for rid, (w, ctx) in windows.items():
            vae_node.postprocess(rid, None, out[rid], inputs=prepared[rid])
            alone = vae_node.vae.decode(w.reshape(-1, 8).t().unsqueeze(0))[0][ctx * config.patch_samples :]
            _assert_unsaturated(alone)
            assert (out[rid]["audio_chunk"][0] - alone).abs().max() <= FP64_ROUNDING


def _window_widths(buffer: StreamBuffer) -> list[int]:
    """Patches in each window ready on ``buffer``, popped."""
    widths = []
    while buffer.has_chunk_ready():
        widths.append(buffer.pop_chunk().data["data"].reshape(-1, 4, 8).shape[0])
    return widths


def test_stream_window_widths_bound_every_window(model):
    widths = model.stream_window_widths()
    for lead in (0, 2, model.stream_left_context):
        policy = model.get_partition_topology().connections[0].chunk_policy_factory()
        buffer = StreamBuffer(request_id="r", edge_name=STREAM_EDGE, from_partition="Talker", policy=policy)
        if lead:
            buffer.prime_context(lead)
        popped: list[int] = []
        for i in range(lead + 37):
            buffer.pre_read_register(f"t{i}")
            buffer.put(f"t{i}", torch.zeros(4, 8))
            popped += _window_widths(buffer)
        buffer.signal_done()
        popped += _window_widths(buffer)
        assert max(popped) <= max(widths), (lead, popped)
        if not lead:
            assert set(popped[:-1]) <= set(widths), popped


def test_vae_windows_replay_at_their_bucket_and_equal_a_whole_decode(model):
    """Every window of a stream goes through the region of the next width up,
    padded at the end; the streamed audio still equals one whole decode."""
    config = model.config
    widths = model.stream_window_widths()
    vae_node = VAEDecoderSubmodule(_random_vae(config).double(), config, window_buckets=widths)
    regions = vae_node.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)
    assert list(regions) == [VAE_WINDOW.format(width) for width in widths]
    runners = {label: _ReplayRunner(region) for label, region in regions.items()}
    patches = torch.randn(45, 4, 8, dtype=torch.float64)
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    with torch.no_grad():
        streamed = torch.cat(stream_through(vae_node, patches, policy, runners=runners))
        whole = vae_node.vae.decode(patches.reshape(-1, 8).t().unsqueeze(0))[0]
    _assert_unsaturated(whole)
    assert (streamed - whole).abs().max() <= FP64_ROUNDING
    assert all(calls == [(1, 1)] * len(calls) for calls in (r.calls for r in runners.values()))
    assert all(runner.calls for runner in runners.values())


def test_graph_hooks_are_the_names_the_engine_calls(model):
    """The engine looks up only the accelerator graph hook names, so each
    node defines its own override (a misnamed one inherits the base default,
    which captures nothing) and declares its regions when called as the engine
    calls it."""
    config = model.config
    widths = model.stream_window_widths()
    talker, _, _ = _talker(model)
    vae_node = VAEDecoderSubmodule(_random_vae(config), config, window_buckets=widths)
    for hook in (
        "get_accelerator_graph_configs",
        "can_use_accelerator_graphs",
        "get_piecewise_accelerator_graph_configs",
    ):
        assert hook in TalkerSubmodule.__dict__, hook
    assert "get_piecewise_accelerator_graph_configs" in VAEDecoderSubmodule.__dict__
    device = torch.device("cpu")
    assert list(talker.get_piecewise_accelerator_graph_configs(device, torch.float32, 1)) == [PREFILL_TAIL]
    regions = vae_node.get_piecewise_accelerator_graph_configs(device, torch.float32, 1)
    assert list(regions) == [VAE_WINDOW.format(width) for width in widths]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures CUDA graphs")
def test_captured_regions_replay_the_eager_result_on_cuda(model):
    """The engine's piecewise runner captures both regions; a replay at a
    captured shape gives the eager result bit for bit, and a padded one (three
    rows in the 4-row bucket, a window padded to the next width) stays within
    rounding of the eager result at the real shape."""
    from unittest import mock

    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner

    device, config = torch.device("cuda", torch.cuda.current_device()), model.config
    gen = torch.Generator(device=device).manual_seed(0)

    def capture(label, region):
        runner = PiecewiseAcceleratorGraphRunner(
            label=label,
            config=region,
            resources={},
            step_runner=mock.Mock(),
            device=device,
            autocast_dtype=None,
            num_slots=1,
        )
        runner.warmup_and_capture(None)  # no plan: every bucket
        assert runner.any_graphs and not runner.dropped_shapes
        return runner

    with torch.inference_mode():
        talker = TalkerSubmodule(_random_generator(config).to(device), config, 4)
        ((label, region),) = talker.get_piecewise_accelerator_graph_configs(device, torch.float32).items()
        runner = capture(label, region)
        for rows in (1, 3):
            inputs = {
                "lm_hidden": torch.randn(rows, config.lm.hidden_size, generator=gen, device=device),
                "res_hidden": torch.randn(rows, config.residual.hidden_size, generator=gen, device=device),
                "cond_patch": torch.randn(rows, config.patch_size, config.feat_dim, generator=gen, device=device),
                "noise": torch.randn(rows, config.feat_dim, config.patch_size, generator=gen, device=device),
                "cfg": torch.linspace(1.5, 3.0, rows, device=device),
            }
            eager = talker.model.first_patch(*inputs.values(), talker.t_span())
            out = runner.run(static_inputs=inputs, real_bs=rows)
            for got, want in zip((out["patch"], out["stop_logits"]), eager, strict=True):
                if rows == 1:
                    assert torch.equal(got, want)
                else:
                    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

        widths = model.stream_window_widths()
        vae_node = VAEDecoderSubmodule(_random_vae(config).to(device), config, window_buckets=widths)
        runners = {
            label: capture(label, region)
            for label, region in vae_node.get_piecewise_accelerator_graph_configs(device, torch.float32).items()
        }
        for width in widths:
            latents = torch.randn(2, config.feat_dim, width * config.patch_size, generator=gen, device=device)
            assert torch.equal(vae_node._decode(latents, runners), vae_node.vae.decode(latents, vae_node._sr_index))
            short = latents[..., : -config.patch_size]
            if short.shape[-1]:
                eager = vae_node.vae.decode(short, vae_node._sr_index)
                padded = vae_node._decode(short, runners)
                torch.testing.assert_close(padded[:, : eager.shape[-1]], eager, rtol=1e-4, atol=1e-5)


def test_vae_window_wider_than_every_bucket_or_batch_decodes_eagerly(model):
    config = model.config
    vae_node = VAEDecoderSubmodule(_random_vae(config).double(), config, window_buckets=(1, 3))
    runners = {
        label: _ReplayRunner(region)
        for label, region in vae_node.get_piecewise_accelerator_graph_configs(
            torch.device("cpu"), torch.float32
        ).items()
    }
    rows = max(VAEDecoderSubmodule.CAPTURE_BATCH_SIZES)
    latents = torch.randn(rows + 1, 8, 4 * 5, dtype=torch.float64)
    with torch.no_grad():
        out = vae_node._decode(latents[:2], runners)
        torch.testing.assert_close(out, vae_node.vae.decode(latents[:2]), rtol=0, atol=0)
        many = vae_node._decode(latents[..., :8], runners)
        torch.testing.assert_close(many, vae_node.vae.decode(latents[..., :8]), rtol=0, atol=0)
        assert not any(runner.calls for runner in runners.values())
        two = vae_node._decode(latents[:2, :, :8], runners)
    assert runners[VAE_WINDOW.format(3)].calls == [(2, 2)]
    assert two.shape[-1] == 3 * config.patch_samples


# -- adapter -------------------------------------------------------------------


def test_adapter_maps_voice_design_and_cloning(tmp_path, monkeypatch):
    # the decode check itself runs on real clips in test_media_paths.py
    checked = []
    monkeypatch.setattr(adapters.media_io, "check_ref_audio", lambda path, decoder: checked.append(decoder))
    adapter = adapters.VoxCPM2Adapter()
    req = SpeechRequest(
        model="voxcpm2", input="hello", voice="alloy", speed=1.2, seed=7, instructions="a calm voice", cfg_value=2.5
    )
    sa = adapter.speech_to_request(req, tmp_path)
    assert sa.input_modalities == ["text"] and sa.output_modalities == ["audio"]
    assert sa.model_kwargs == {"instruct": "a calm voice", "seed": 7, "cfg_value": 2.5}

    data_url = "data:audio/wav;base64," + base64.b64encode(_wav_bytes()).decode()
    clone = adapter.speech_to_request(
        SpeechRequest(model="voxcpm2", input="hi", ref_audio=data_url, ref_text="words"),
        tmp_path,
    )
    assert clone.input_modalities == ["text", "audio"] and "ref_audio" not in clone.model_kwargs
    assert Path(clone.file_paths["audio"][0]).is_file() and clone.model_kwargs["ref_text"] == "words"
    assert checked == ["soundfile"]


def _wav_bytes() -> bytes:
    """One second of 16 kHz silence, written with the standard library."""
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(32000))
    return buf.getvalue()


@pytest.fixture
def media_dirs(tmp_path):
    """An upload dir holding a stored clip, and a server-only clip beside it (#411)."""
    uploads, secret = tmp_path / "uploads", tmp_path / "secret"
    uploads.mkdir()
    secret.mkdir()
    (uploads / "stored.wav").write_bytes(_wav_bytes())
    (secret / "voice.wav").write_bytes(_wav_bytes())
    (uploads / "link.wav").symlink_to(secret / "voice.wav")
    return uploads, secret


@pytest.mark.parametrize(
    "ref",
    [
        "{secret}/voice.wav",
        "{uploads}/stored.wav",
        "{uploads}/link.wav",
        "{uploads}/../secret/voice.wav",
        "secret/voice.wav",
        "file://{secret}/voice.wav",
        "{secret}/missing.wav",
        "/etc/passwd",
        "//etc/passwd",
        "secret",
        "ftp://example.com/voice.wav",
    ],
)
def test_adapter_refuses_every_server_path(media_dirs, ref):
    """#411: no plain path, not even a stored upload, and every refusal reads the same."""
    uploads, secret = media_dirs
    before = sorted(uploads.iterdir())
    with pytest.raises(ValueError, match="^server file paths are refused"):
        adapters.VoxCPM2Adapter().speech_to_request(
            SpeechRequest(model="voxcpm2", input="hi", ref_audio=ref.format(uploads=uploads, secret=secret)),
            uploads,
        )
    assert sorted(uploads.iterdir()) == before


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("http://example.test/clip.wav", 302, "redirect to file://", None, None),
        http.client.IncompleteRead(b"RIFF", 996),
    ],
)
def test_adapter_reports_a_failed_fetch_as_a_bad_request(tmp_path, monkeypatch, error):
    """urllib refuses a redirect to ``file://`` with an HTTPError, and a body cut
    short raises IncompleteRead; like any failed fetch they are the request's
    fault, not a server error."""

    def refuse(*args, **kwargs):
        raise error

    monkeypatch.setattr(adapters.media_io, "resolve_media_ref", refuse)
    request = SpeechRequest(model="voxcpm2", input="hi", ref_audio="http://example.test/clip.wav")
    with pytest.raises(ValueError, match="could not fetch ref_audio"):
        adapters.VoxCPM2Adapter().speech_to_request(request, tmp_path)


def _stored_reference(uploads: Path, ref: str) -> Path:
    request = SpeechRequest(model="voxcpm2", input="hi", ref_audio=ref)
    args = adapters.VoxCPM2Adapter().speech_to_request(request, uploads)
    (path,) = args.file_paths["audio"]
    assert Path(path).parent == uploads and Path(path).read_bytes() == _wav_bytes()
    return Path(path)


def test_adapter_takes_bare_base64_that_soundfile_opens(media_dirs):
    pytest.importorskip("soundfile")
    _stored_reference(media_dirs[0], base64.b64encode(_wav_bytes()).decode())


def test_adapter_takes_data_urls_and_gated_urls(media_dirs, monkeypatch):
    from mstar.api_server import media_io

    uploads, _ = media_dirs
    # the decode check itself runs on real clips in test_media_paths.py
    checked = []
    monkeypatch.setattr(media_io, "check_ref_audio", lambda path, decoder: checked.append(decoder))
    adapter = adapters.VoxCPM2Adapter()
    _stored_reference(uploads, "data:audio/wav;base64," + base64.b64encode(_wav_bytes()).decode())
    url = SpeechRequest(model="voxcpm2", input="hi", ref_audio="https://example.com/voice.wav")
    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError, match="Remote media fetch is disabled"):
        adapter.speech_to_request(url, uploads)
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    monkeypatch.setattr(media_io, "save_remote_url", lambda ref, upload_dir: ("audio", str(upload_dir / "fetched.wav")))
    assert adapter.speech_to_request(url, uploads).file_paths == {"audio": [str(uploads / "fetched.wav")]}
    assert checked == ["soundfile", "soundfile"]
    with pytest.raises(ValueError, match="base64 string"):
        adapter.speech_to_request(SpeechRequest(model="voxcpm2", input="hi", ref_audio=["x"]), uploads)


# -- request contract -----------------------------------------------------------


def _loop_conductor(model):
    """A conductor with only what ``_loop_caps`` reads: the model and its loops' ``max_iters``."""
    from mstar.conductor.conductor import Conductor

    c = Conductor.__new__(Conductor)
    c.model = model
    c._loop_max_iters = {
        name: loop.max_iters
        for walk in model.get_graph_walk_graphs().values()
        for name, loop in walk.get_loops().items()
    }
    return c


def test_request_kwargs_are_the_keys_the_model_and_its_adapter_read(model, tmp_path):
    assert model.request_kwargs() == {
        "instruct", "instructions", "ref_text", "cfg_value", "min_len", "inference_timesteps",
        "max_new_tokens", "max_output_tokens", "max_len",
    }
    req = SpeechRequest(model="voxcpm2", input="hi", voice="alloy", speed=1.2, seed=7, instructions="calm",
                        cfg_value=2.5, min_len=2, max_new_tokens=9)
    sa = adapters.VoxCPM2Adapter().speech_to_request(req, tmp_path)
    # seed is the conductor's; the adapter drops voice, speech_request forwards a sent speed to be
    # logged as ignored, and the model reads neither
    assert set(sa.model_kwargs) - {"seed"} <= model.request_kwargs()
    assert not {"voice", "speed"} & model.request_kwargs()


@pytest.mark.parametrize("key", ["max_new_tokens", "max_output_tokens", "max_len", "min_len", "inference_timesteps"])
@pytest.mark.parametrize("bad", [2.7, "20", True])
def test_integer_knobs_are_strict(model, key, bad):
    """``int()`` used to truncate 2.7 to 2 and read "20" as 20."""
    with pytest.raises(ValueError, match=f"{key} must be an integer"):
        model.process_prompt("hello", ["text"], ["audio"], **{key: bad})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"min_len": -1}, "min_len"),
        ({"cfg_value": "2"}, "cfg_value must be a number"),
        ({"cfg_value": True}, "cfg_value must be a number"),
        ({"cfg_value": 10**400}, "finite"),
        ({"max_len": 10**400}, "max_new_tokens <="),
        ({"instruct": 5}, "instruct must be a string"),
        ({"instructions": ["calm"]}, "instructions must be a string"),
        ({"ref_text": 5}, "ref_text must be a string"),
    ],
)
def test_wrong_types_and_ranges_are_value_errors(model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        model.process_prompt("hello", ["text"], ["audio"], **kwargs)


def test_whole_floats_pass_for_integer_knobs(model):
    metadata = model.process_prompt(
        "hello world", ["text"], ["audio"], max_new_tokens=9.0, min_len=2.0,
        inference_timesteps=float(model.inference_timesteps), cfg_value=2,
    ).metadata
    assert metadata[PATCH_BUDGET_KEY] == model.config.max_patches(2, 9) and metadata[MIN_LEN_KEY] == 2
    assert type(metadata[MIN_LEN_KEY]) is int
    assert model.get_loop_caps(max_new_tokens=9.0) == {TALKER_LOOP: 8}


def test_a_patch_budget_past_the_decode_loop_is_refused_not_clamped(model):
    """The loop is built from the default request's cap (one under ``max_len``,
    since the prefill emits the first patch), so one patch past ``max_len`` is
    refused by the conductor as well as by ``process_prompt``."""
    limit = model.config.max_len
    c = _loop_conductor(model)
    assert c._loop_max_iters == {TALKER_LOOP: limit - 1}
    assert c._loop_caps({"max_new_tokens": limit}) == {TALKER_LOOP: limit - 1}
    with pytest.raises(ValueError, match="iterations"):
        c._loop_caps({"max_new_tokens": limit + 1})
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.process_prompt("hello", ["text"], ["audio"], max_new_tokens=limit + 1)


def test_compile_vae_switch_reaches_the_decoder_region(model, checkpoint):
    config = model.config
    for compiled, regions in ((True, ["vae.decoder"]), (False, [])):
        node = VAEDecoderSubmodule(_random_vae(config), config, compile_decoder=compiled)
        assert list(node.compile_regions()) == regions
    (latents, rate), = VAEDecoderSubmodule(_random_vae(config), config).compile_regions()["vae.decoder"]
    assert latents.shape == (2, config.feat_dim, 2 * config.patch_size) and rate.shape == (2,)
    assert VoxCPM2Model(model_path_hf=str(checkpoint)).compile_vae is True
    assert VoxCPM2Model(model_path_hf=str(checkpoint), compile_vae=False).compile_vae is False
    with pytest.raises(ValueError, match="compile_vae"):
        VoxCPM2Model(model_path_hf=str(checkpoint), compile_vae="no")


def test_the_talker_serves_a_ready_prompt_first(model):
    """Each prefill walk outranks decode, a clone's reference encoding with it:
    listing a walk covers every node on it, as the worker resolves the entries."""
    from mstar.worker.services import walk_priorities_of

    priorities = walk_priorities_of(model, model.get_graph_walk_graphs())
    assert {pair: entry.priority for pair, entry in priorities.items()} == {
        ("RefEncoder", "talker_prefill_clone"): 1,
        ("Talker", "talker_prefill"): 1,
        ("Talker", "talker_prefill_clone"): 1,
    }
    assert ("Talker", "talker_decode") not in priorities


# -- the local DiT's solve ----------------------------------------------------------------


def _stepwise_solve(estimator, noise, mu, cond, cfg_value, t_span, mean_mode=False):
    """``solve_cfm`` as the reference writes it: every step re-embeds its time
    and re-projects the condition through ``estimator.forward``."""
    rows, x = noise.shape[0], noise
    t, dt = t_span[0], t_span[0] - t_span[1]
    zero_init_steps = max(1, int(len(t_span) * 0.04))
    mu_in, cond_in = torch.cat([mu, torch.zeros_like(mu)]), torch.cat([cond, cond])
    scale = cfg_value.to(x.dtype).view(rows, 1, 1)
    for step in range(1, len(t_span)):
        if step <= zero_init_steps:
            dphi_dt = torch.zeros_like(x)
        else:
            t_in = t.expand(2 * rows)
            dt_in = dt.expand(2 * rows) if mean_mode else torch.zeros_like(t_in)
            velocity = estimator(torch.cat([x, x]), mu_in, t_in, cond_in, dt_in)
            pos, neg = velocity[:rows], velocity[rows:]
            dot = torch.sum(pos.reshape(rows, -1) * neg.reshape(rows, -1), dim=1, keepdim=True)
            st_star = (dot / (torch.sum(neg.reshape(rows, -1) ** 2, dim=1, keepdim=True) + 1e-8)).view(rows, 1, 1)
            dphi_dt = neg * st_star + scale * (pos - neg * st_star)
        x = x - dt * dphi_dt
        t = t - dt
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]
    return x


@pytest.mark.parametrize("mean_mode", [False, True])
def test_the_solve_computes_its_per_solve_tokens_once_and_matches_the_stepwise_solve(mean_mode):
    from mstar.model.voxcpm2.components.local import cfm_t_span, solve_cfm

    config = VoxCPM2Config.from_dict(TINY)
    estimator = _random_generator(config).feat_decoder.estimator
    gen = torch.Generator().manual_seed(1)
    rows, hidden = 3, config.dit.hidden_size
    noise = torch.randn(rows, config.feat_dim, config.patch_size, generator=gen)
    mu = torch.randn(rows, 2 * hidden, generator=gen)
    cond = torch.randn(rows, config.feat_dim, config.patch_size, generator=gen)
    cfg_value = torch.tensor([1.5, 2.0, 3.0])
    t_span = cfm_t_span(10, "cpu", torch.float32)
    with torch.no_grad():
        got = solve_cfm(estimator, noise, mu, cond, cfg_value, t_span, mean_mode=mean_mode)
        want = _stepwise_solve(estimator, noise, mu, cond, cfg_value, t_span, mean_mode=mean_mode)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel runs on CUDA")
@pytest.mark.parametrize("seq", [5, 11])
@pytest.mark.parametrize("rows", [2, 33])
def test_fused_rope_attention_matches_the_eager_path_at_the_checkpoints_head_size(rows, seq):
    """The local stacks' shape (16 query and 2 KV heads of 128, rows of the
    encoder's 5 or the DiT's 11 tokens): rounding apart, the eager result."""
    from mstar.model.components.rope_attention import rope_attention, rope_attention_reference

    heads, kv_heads, head_dim = 16, 2, 128
    gen = torch.Generator(device="cuda").manual_seed(0)
    # any tables: the kernel reads both halves of each, as the eager rotation does
    cos, sin = (torch.rand(seq, head_dim, device="cuda", generator=gen) * 2 - 1 for _ in range(2))
    qkv = torch.randn(rows * seq, (heads + 2 * kv_heads) * head_dim, device="cuda", generator=gen).to(torch.bfloat16)
    got = rope_attention(qkv, cos, sin, rows, seq, heads, kv_heads, head_dim)
    want = rope_attention_reference(qkv, cos, sin, rows, seq, heads, kv_heads, head_dim)
    assert got.shape == want.shape == (rows * seq, heads * head_dim)
    # a table the kernel can't read with cos's strides takes the eager path
    wide = torch.rand(seq, 2 * head_dim, device="cuda", generator=gen) * 2 - 1
    sliced = wide[:, :head_dim]
    assert torch.equal(rope_attention(qkv, cos, sliced, rows, seq, heads, kv_heads, head_dim),
                       rope_attention_reference(qkv, cos, sliced, rows, seq, heads, kv_heads, head_dim))
    err = (got.float() - want.float()).abs()
    # the softmax weights round to bfloat16 before the value product, as SDPA's do,
    # in another order: a few bfloat16 steps at most, far less on average
    assert err.max() < 0.07 and err.mean() < 0.003


def _local_stack(seed=0):
    """A local stack the fused paths take: head halves of 32, half precision."""
    from mstar.model.voxcpm2.components.minicpm import MiniCPMLocalTransformer
    from mstar.model.voxcpm2.config import MiniCPM4Config

    config = MiniCPM4Config(hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=64, vocab_size=0)
    torch.manual_seed(seed)
    stack = MiniCPMLocalTransformer(config, max_len=11)
    with torch.no_grad():
        for name, param in stack.named_parameters():
            param.copy_(1 + 0.1 * torch.randn_like(param) if "norm" in name else 0.05 * torch.randn_like(param))
    stack.materialize_buffers()
    return stack.eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused paths run on CUDA")
def test_the_local_stack_compiles_whole_and_its_fused_bf16_path_matches_fp32():
    """No graph break between layers (attention, add+norm, SiLU are each one op
    or kernel Dynamo sees whole), and the half-precision fused path within
    rounding of the float32 portable one on the same weights."""
    stack = _local_stack()
    x = torch.randn(6, 11, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = stack(x)  # CPU, float32: the reference rotation, SDPA, unfused add+norm and SiLU
        fused = stack.to("cuda", torch.bfloat16)
        got = fused(x.to("cuda", torch.bfloat16))
        compiled = torch.compile(fused, fullgraph=True)(x.to("cuda", torch.bfloat16))
    err = (got.float().cpu() - want).norm() / want.norm()
    assert err < 0.02, err
    torch.testing.assert_close(compiled, got, rtol=0.02, atol=0.02)
