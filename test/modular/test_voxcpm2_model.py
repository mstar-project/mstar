"""CPU tests for the VoxCPM2 port: tiny random weights, no checkpoint, no GPU.

Covers the contract (config, graph, partitions, prompt modes, validation),
the Talker's batching and stop rule through dense fake resources, and the
#429 invariant for the VAE stream: windows from the real ``StreamBuffer`` and
chunk policy stitch to exactly a whole decode, in length and in samples.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.api_server.openai import adapters
from mstar.api_server.openai.protocol import SpeechRequest
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.model.submodule_base import InputMetadata
from mstar.model.voxcpm2.components.audio_vae import AudioVAE, CausalResidualUnit, decoder_receptive_field_frames
from mstar.model.voxcpm2.components.minicpm import RopeTable, longrope_cos_sin
from mstar.model.voxcpm2.components.talker import VoxCPM2Generator
from mstar.model.voxcpm2.config import (
    AUDIO_START_TOKEN,
    MODE_REF_CONTINUATION,
    MODE_REFERENCE,
    MODE_ZERO_SHOT,
    REF_AUDIO_END_TOKEN,
    REF_AUDIO_START_TOKEN,
    VoxCPM2Config,
)
from mstar.model.voxcpm2.submodules import TALKER_LOOP, TalkerSubmodule, VAEDecoderSubmodule, build_prompt
from mstar.model.voxcpm2.voxcpm2_model import (
    PATCH_BUDGET_KEY,
    STREAM_EDGE,
    STREAM_LEAD_KEY,
    VoxCPM2Model,
    decoder_receptive_field_patches,
)
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy

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


def test_process_prompt_builds_each_reference_mode(model):
    zero = _prompt(model, "hello  world\n", ["text"], ["audio"], seed=3, cfg_value=1.5)
    assert zero["text_inputs"][0].tolist() == [10, 11, AUDIO_START_TOKEN]
    assert zero["prompt_layout"][0].tolist() == [MODE_ZERO_SHOT, 2 * 6 + 10, 2]
    assert zero["gen_params"][0].tolist() == [1.5]

    design = _prompt(model, "hello", ["text"], ["audio"], instructions="warm")
    assert design["text_inputs"][0].tolist() == [15, AUDIO_START_TOKEN]  # "(warm)hello"

    cjk = _prompt(model, "你好", ["text"], ["audio"])
    assert cjk["text_inputs"][0].tolist() == [13, 14, AUDIO_START_TOKEN]  # split like the reference

    clip = torch.zeros(16000)  # 1 s: 7 patches of 2560 samples
    ref = _prompt(model, "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]})
    assert ref["prompt_layout"][0][0] == MODE_REFERENCE
    cont = model.process_prompt(
        "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, ref_text="world "
    )
    tensors, metadata = cont
    assert tensors["text_inputs"][0].tolist() == [11, 10, AUDIO_START_TOKEN]  # transcript first
    # transcript and target are tokenized as one string, with no separator added, as the reference does
    joined, _ = model.process_prompt(
        "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, ref_text="world"
    )
    assert joined["text_inputs"][0].tolist() == [3, AUDIO_START_TOKEN]  # "worldhello"
    # max patches follow the target text only
    assert tensors["prompt_layout"][0].tolist() == [MODE_REF_CONTINUATION, 16, 2]
    assert metadata == {STREAM_LEAD_KEY: 3, PATCH_BUDGET_KEY: 16}


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
    tensors, metadata = model.process_prompt("hello world", ["text"], ["audio"], **kwargs)
    budget = model.config.max_patches(2, cap)
    assert tensors["prompt_layout"][0][1] == budget and metadata[PATCH_BUDGET_KEY] == budget


@pytest.mark.parametrize("budget, done", [(1, True), (2, False)])
def test_a_one_patch_request_ends_after_prefill(model, budget, done):
    """Prefill has no loop for the Talker's stop to end, so a request whose
    budget the prefill patch spends goes straight to done instead of a decode
    step that would only veto itself."""
    args = model.get_initial_forward_pass_args(
        "Talker", ["text"], ["audio"], {}, model_kwargs={PATCH_BUDGET_KEY: budget}
    )
    nxt = model.get_partition_forward_pass_args("Talker", args.full_metadata, {"latent_patch": []})
    assert nxt.request_done is done
    assert nxt.full_metadata.graph_walk == ("talker_prefill" if done else "talker_decode")


@pytest.mark.parametrize(
    "kwargs, inputs, message",
    [
        ({"ref_text": "x"}, None, "ref_text needs"),
        ({"cfg_value": -1}, None, "non-negative"),
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


def test_process_prompt_rejects_prompts_past_the_context(model):
    long_text = " ".join(["hello"] * 80)  # 80 tokens + 490 patches > 512 positions
    with pytest.raises(ValueError, match="context"):
        model.process_prompt(long_text, ["text"], ["audio"])


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


def test_initial_args_route_reference_audio_and_stream_lead(model):
    signals = {name: [f"ptr-{name}"] for name in ("text_inputs", "prompt_layout", "gen_params", "audio_inputs")}
    talker = model.get_initial_forward_pass_args("Talker", ["text", "audio"], ["audio"], signals)
    assert talker.full_metadata.graph_walk == "talker_prefill_clone"
    routes = sorted((edge.next_node, edge.name) for edge in talker.inputs)
    assert ("RefEncoder", "audio_inputs") in routes and ("RefEncoder", "prompt_layout") in routes
    assert ("Talker", "gen_params") in routes
    vae = model.get_initial_forward_pass_args(
        "VAEDecoder", ["text", "audio"], ["audio"], {}, model_kwargs={STREAM_LEAD_KEY: 3}
    )
    assert vae.stream_lead_items == {STREAM_EDGE: 3}
    plain = model.get_initial_forward_pass_args("Talker", ["text"], ["audio"], signals)
    assert plain.full_metadata.graph_walk == "talker_prefill"
    assert {edge.next_node for edge in plain.inputs} == {"Talker"}


def test_postprocess_encodes_pcm16(model):
    pcm = model.postprocess(torch.tensor([0.0, 0.5, -2.0]), "audio")
    assert torch.frombuffer(bytearray(pcm), dtype=torch.int16).tolist() == [0, 16383, -32767]
    assert model.get_output_sample_rate() == 48000


def test_load_audio_resamples_to_16k_and_bounds_duration(model, tmp_path):
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    path = tmp_path / "clip.wav"
    soundfile.write(path, torch.zeros(44100).numpy(), 44100)
    loaded = model.load_audio(str(path), "cpu")
    assert loaded.metadata["sample_rate"] == 16000 and loaded.data.shape[0] == 16000
    soundfile.write(path, torch.zeros(4410).numpy(), 44100)
    with pytest.raises(ValueError, match="between"):
        model.load_audio(str(path), "cpu")


# -- Talker -----------------------------------------------------------------------


def _talker(model, timesteps: int = 4):
    generator = _random_generator(model.config)
    talker = TalkerSubmodule(generator, model.config, timesteps)
    plan, reset = bind_dense_resources(generator)
    return talker, plan, reset


def _run(talker, plan, requests: dict, steps: int):
    """Prefill every request together, then ``steps - 1`` batched decode steps."""
    rids = list(requests)
    infos = {
        rid: SimpleNamespace(rid_handle=rid, request_id=rid, random_seed=seed) for rid, (_, seed) in requests.items()
    }
    engine = SimpleNamespace(request_ids=rids)
    prepared = [talker.prepare_inputs("talker_prefill", infos[rid], requests[rid][0]) for rid in rids]
    plan([p.input_seq_len for p in prepared])
    out = talker.forward_batched("talker_prefill", engine, **talker.preprocess("talker_prefill", engine, prepared))
    patches = {rid: [out[rid]["latent_patch"][0]] for rid in rids}
    for _ in range(steps - 1):
        prepared = [
            talker.prepare_inputs("talker_decode", infos[rid], {"latent_patch": [patches[rid][-1]]}) for rid in rids
        ]
        plan([1] * len(rids))
        out = talker.forward_batched("talker_decode", engine, **talker.preprocess("talker_decode", engine, prepared))
        for rid in rids:
            patches[rid].append(out[rid]["latent_patch"][0])
    return {rid: torch.stack(p) for rid, p in patches.items()}


def test_talker_batches_like_single_requests_and_follows_the_seed(model):
    talker, plan, reset = _talker(model)
    a = _prompt(model, "hello world", ["text"], ["audio"])
    b = _prompt(model, "hello hello hello", ["text"], ["audio"], cfg_value=3.0)
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


def test_talker_clone_prefill_streams_the_prompt_tail_as_context(model):
    talker, plan, _ = _talker(model)
    clip = torch.zeros(16000)
    request, _ = model.process_prompt(
        "hello", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, ref_text="world "
    )
    ref = torch.randn(2, 7, 4, 8)
    info = SimpleNamespace(rid_handle="c", request_id="c", random_seed=1)
    prepared = talker.prepare_inputs("talker_prefill_clone", info, {**request, "ref_patches": [ref]})
    assert prepared.input_seq_len == 1 + 7 + 1 + 3 + 7
    torch.testing.assert_close(prepared.tensor_inputs["cond_patch"], ref[1][-1])
    plan([prepared.input_seq_len])
    engine = SimpleNamespace(request_ids=["c"])
    with torch.inference_mode():
        out = talker.forward_batched(
            "talker_prefill_clone", engine, **talker.preprocess("talker_prefill_clone", engine, [prepared])
        )["c"]
    stream = out[STREAM_EDGE]
    assert len(stream) == 4 and all(torch.equal(s, r) for s, r in zip(stream[:3], ref[1][-3:], strict=True))
    assert torch.equal(stream[3], out["latent_patch"][0])


def test_talker_stops_like_the_reference_loop(model):
    """``i > min_len and stop`` on the step that produced patch i (kept), or the patch budget."""
    talker, _, _ = _talker(model)
    request = _prompt(model, "hello", ["text"], ["audio"], max_new_tokens=6)
    talker.prepare_inputs("talker_prefill", SimpleNamespace(rid_handle="r", request_id="r", random_seed=0), request)

    def step(flag: int, index: int) -> set[str]:
        return talker.check_stop("r", None, {"stop_flag": [torch.tensor([flag, index])]})

    assert [step(1, i) for i in range(3)] == [set(), set(), set()]  # i = 0, 1, 2: not past min_len 2
    assert step(1, 3) == {TALKER_LOOP}
    assert [step(0, i) for i in range(6)][-2:] == [set(), {TALKER_LOOP}]  # budget of 6 patches


def test_talker_stop_judges_each_step_by_its_own_patch_index(model):
    """A speculative step postprocessed before the previous step's stop check
    moves ``generated`` on; the check still counts the patches of its own step."""
    talker, _, _ = _talker(model)
    request = _prompt(model, "hello", ["text"], ["audio"], max_new_tokens=6)
    talker.prepare_inputs("talker_prefill", SimpleNamespace(rid_handle="r", request_id="r", random_seed=0), request)
    for _ in range(6):
        talker.postprocess("r", None, {"stop_flag": [torch.tensor([0, 0])]})
    assert talker.check_stop("r", None, {"stop_flag": [torch.tensor([0, 4])]}) == set()  # patch 5 of 6
    assert talker.check_stop("r", None, {"stop_flag": [torch.tensor([1, 2])]}) == set()  # i = 2: not past min_len


def test_talker_vetoes_steps_past_the_patch_budget(model):
    talker, _, _ = _talker(model)
    request = _prompt(model, "hello", ["text"], ["audio"], max_new_tokens=2)
    info = SimpleNamespace(rid_handle="r", request_id="r", random_seed=0)
    talker.prepare_inputs("talker_prefill", info, request)
    talker.postprocess("r", None, {"stop_flag": [torch.tensor([0, 0])]})
    patch = torch.zeros(4, 8)
    first = talker.prepare_inputs("talker_decode", info, {"latent_patch": [patch]})
    assert first is not None and first.tensor_inputs["step_index"].tolist() == [1]
    talker.postprocess("r", None, {"stop_flag": [torch.tensor([0, 1])]})
    assert talker.prepare_inputs("talker_decode", info, {"latent_patch": [patch]}) is None


def test_talker_reprepares_a_refused_step_with_the_same_index_and_noise(model):
    """A step whose admit was refused is prepared again: nothing ran, so it is
    the same patch index and the same noise draw, and the budget is not spent."""
    talker, _, _ = _talker(model)
    request = _prompt(model, "hello", ["text"], ["audio"], max_new_tokens=2)
    info = SimpleNamespace(rid_handle="r", request_id="r", random_seed=0)
    talker.prepare_inputs("talker_prefill", info, request)
    talker.postprocess("r", None, {"stop_flag": [torch.tensor([0, 0])]})
    patch = torch.zeros(4, 8)
    refused = talker.prepare_inputs("talker_decode", info, {"latent_patch": [patch]})
    retry = talker.prepare_inputs("talker_decode", info, {"latent_patch": [patch]})
    assert retry is not None and retry.tensor_inputs["step_index"].tolist() == [1]
    assert torch.equal(retry.tensor_inputs["noise"], refused.tensor_inputs["noise"])


def test_talker_noise_is_drawn_once_per_step(model):
    talker, _, _ = _talker(model)
    state = talker.request_state("n")
    state.add_all(generator=torch.Generator().manual_seed(4), noise={}, drawn=0)
    first, second = talker._noise(state, 0), talker._noise(state, 1)
    assert talker._noise(state, 1) is second  # a re-prepared step reuses its draw
    expected = torch.Generator().manual_seed(4)
    torch.testing.assert_close(first, torch.randn((1, 8, 4), generator=expected)[0])
    torch.testing.assert_close(second, torch.randn((1, 8, 4), generator=expected)[0])


def test_talker_declares_the_same_span_in_both_caches(model):
    talker, _, _ = _talker(model)
    step = talker.declare_step("talker_decode", ["a", "b"], [SimpleNamespace(input_seq_len=1)] * 2)
    assert set(step.steps) == {"base_kv", "base_attn", "base_pos", "res_kv", "res_attn"}
    assert [(s.request_id, s.span) for s in step.segments] == [("a", 1), ("b", 1)]
    (graph,) = talker.get_cuda_graph_configs(torch.device("cpu"))
    assert graph.capture_graph_walk == "talker_decode" and set(graph.single_request_inputs.tensor_inputs) == {
        "prev_patch",
        "noise",
        "cfg",
        "step_index",
    }


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
    engine = SimpleNamespace(request_ids=list(windows))
    with torch.no_grad():
        out = vae_node.forward_batched(
            "vae_chunk", engine, **vae_node.preprocess("vae_chunk", engine, list(prepared.values()))
        )
        for rid, (w, ctx) in windows.items():
            vae_node.postprocess(rid, None, out[rid], inputs=prepared[rid])
            alone = vae_node.vae.decode(w.reshape(-1, 8).t().unsqueeze(0))[0][ctx * config.patch_samples :]
            _assert_unsaturated(alone)
            assert (out[rid]["audio_chunk"][0] - alone).abs().max() <= FP64_ROUNDING


# -- adapter -------------------------------------------------------------------


def test_adapter_maps_voice_design_and_cloning(tmp_path):
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


def test_adapter_reports_a_failed_fetch_as_a_bad_request(tmp_path, monkeypatch):
    """urllib refuses a redirect to ``file://`` with an HTTPError; like any
    failed fetch it is the request's fault, not a server error."""
    import urllib.error

    def refuse(*args, **kwargs):
        raise urllib.error.HTTPError("http://example.test/clip.wav", 302, "redirect to file://", None, None)

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
    adapter = adapters.VoxCPM2Adapter()
    _stored_reference(uploads, "data:audio/wav;base64," + base64.b64encode(_wav_bytes()).decode())
    url = SpeechRequest(model="voxcpm2", input="hi", ref_audio="https://example.com/voice.wav")
    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError, match="Remote media fetch is disabled"):
        adapter.speech_to_request(url, uploads)
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    monkeypatch.setattr(media_io, "save_remote_url", lambda ref, upload_dir: ("audio", str(upload_dir / "fetched.wav")))
    assert adapter.speech_to_request(url, uploads).file_paths == {"audio": [str(uploads / "fetched.wav")]}
    with pytest.raises(ValueError, match="base64 string"):
        adapter.speech_to_request(SpeechRequest(model="voxcpm2", input="hi", ref_audio=["x"]), uploads)
