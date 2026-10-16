"""CPU tests for the CosyVoice3 port: tiny random weights, no checkpoint, no GPU.

Covers the contract (config, registry, graph, partitions, prompt modes,
validation, the #411 media rules), the LM's prefill/decode, batching and stop
rules through the fake paged resources, repetition-aware sampling, the flow's
chunk mask and hop plan, and the #429 invariant end to end: tokens pushed
through the edge's real ``StreamBuffer`` and chunk policy, the flow node and
the HiFT node, come out with exactly the length and (to rounding) the samples
of one whole streaming-mode decode.
"""

from __future__ import annotations

import base64
import http.client
import importlib.util
import io
import math
import urllib.error
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.api_server.openai import adapters
from mstar.api_server.openai.protocol import SpeechRequest
from mstar.model.cosyvoice3.components.dit import chunk_attention_mask
from mstar.model.cosyvoice3.components.flow import CosyVoice3Flow, FlowRow, ReferenceVoice
from mstar.model.cosyvoice3.components.hift import CausalHiFTGenerator, StreamState, stream_vocode
from mstar.model.cosyvoice3.components.lm import CosyVoice3LM
from mstar.model.cosyvoice3.components.sampling import hashed_uniform, ras_sample
from mstar.model.cosyvoice3.config import (
    END_OF_PROMPT,
    END_OF_PROMPT_ID,
    LM_ATTN,
    LM_KV,
    LM_POS,
    SYSTEM_PROMPT,
    CausalHiFTConfig,
    CosyVoice3Config,
    DiTConfig,
    FlowConfig,
    LMConfig,
)
from mstar.model.cosyvoice3.cosyvoice3_model import LM_PARTITION, TOKEN2WAV_PARTITION, CosyVoice3Model
from mstar.model.cosyvoice3.submodules import (
    AUDIO_CHUNK,
    DECODE_LOOP,
    MEL,
    MEL_FINAL,
    PREV_TOKEN,
    PROMPT_TOKENS,
    SPEECH_TOKENS,
    STEP_INDEX,
    TEXT_INPUTS,
    TOKEN_LIMITS,
    FlowSubmodule,
    HiFTSubmodule,
    LMSubmodule,
    filter_tokens,
)
from mstar.model.registry import HF_MODELS, get_model_class
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "cosyvoice3.yaml"
SR = 24_000


def _fake_resources():
    """Chatterbox's CPU stand-ins for the paged KV, attention and position
    resources (plain RoPE when given no Llama-3 factors, as Qwen2 uses)."""
    path = Path(__file__).resolve().parents[1] / "chatterbox" / "fake_resources.py"
    spec = importlib.util.spec_from_file_location("cosyvoice3_test_fake_resources", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FakeT3Resources()


class _Tokenizer:
    """One id per whitespace word; ``<|endofprompt|>`` is its trained id."""

    def __call__(self, text: str) -> torch.Tensor:
        ids: list[int] = []
        for i, part in enumerate(text.split(END_OF_PROMPT)):
            if i:
                ids.append(END_OF_PROMPT_ID)
            ids += [1000 + sum(map(ord, word)) % 1000 for word in part.split()]
        return torch.tensor(ids, dtype=torch.long)


@pytest.fixture
def model(tmp_path, monkeypatch) -> CosyVoice3Model:
    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: _Tokenizer())
    return CosyVoice3Model(model_path_hf=str(tmp_path))


def _clip(seconds: float = 1.0, amplitude: float = 0.3) -> torch.Tensor:
    t = torch.arange(int(seconds * SR)) / SR
    return amplitude * torch.sin(2 * math.pi * 220 * t)


def _tiny_config(**overrides) -> CosyVoice3Config:
    lm = LMConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        text_vocab_size=64,
        speech_token_size=40,
        num_control_tokens=10,
    )
    flow = FlowConfig(
        vocab_size=40,
        pre_lookahead_channels=32,
        spk_embed_dim=16,
        dit=DiTConfig(dim=32, depth=2, heads=2, dim_head=16, conv_pos_kernel=5, conv_pos_groups=4, time_freq_dim=16),
        noise_frames=2000,
    )
    hift = CausalHiFTConfig(base_channels=16, f0_cond_channels=16, noise_seconds=12.0)
    cfg = CosyVoice3Config(lm=lm, flow=flow, hift=hift, silent_tokens=(1, 2), max_speech_tokens=512)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _random_flow(cfg: CosyVoice3Config, seed: int = 0) -> CosyVoice3Flow:
    torch.manual_seed(seed)
    flow = CosyVoice3Flow(cfg.flow)
    with torch.no_grad():
        for param in flow.parameters():
            param.normal_(0, 0.2)
    return flow.double().eval()


def _random_hift(cfg: CosyVoice3Config, seed: int = 0) -> CausalHiFTGenerator:
    """Random HiFT whose F0 head predicts voiced frames (100 to 300 Hz), so the
    harmonic source and its carried phase are exercised."""
    torch.manual_seed(seed)
    hift = CausalHiFTGenerator(cfg.hift)
    with torch.no_grad():
        for name, param in hift.named_parameters():
            if name.endswith("alpha"):
                param.uniform_(0.5, 1.5)
            elif param.dim() == 3:
                param.normal_(0, 1 / math.sqrt(param.shape[1] * param.shape[2]))
            else:
                param.normal_(0, 0.05)
        hift.f0_predictor.classifier.bias.fill_(200.0)
        hift.f0_predictor.classifier.weight.mul_(20.0)
    return hift.double().eval()


def _voice(cfg: CosyVoice3Config, prompt_tokens: int, seed: int = 0) -> ReferenceVoice:
    g = torch.Generator().manual_seed(seed)
    return ReferenceVoice(
        prompt_tokens=torch.randint(0, cfg.flow.vocab_size, (prompt_tokens,), generator=g),
        prompt_feat=torch.randn(2 * prompt_tokens, cfg.flow.output_size, generator=g, dtype=torch.float64),
        embedding=torch.randn(cfg.flow.spk_embed_dim, generator=g, dtype=torch.float64),
    )


# -- contract ------------------------------------------------------------------------


def test_config_matches_the_checkpoint_geometry():
    cfg = CosyVoice3Config()
    assert (cfg.lm.sos_token, cfg.lm.eos_token, cfg.lm.task_id_token, cfg.lm.speech_vocab_size) == (
        6561,
        6562,
        6563,
        6761,
    )
    assert cfg.flow.chunk_tokens == 25 and cfg.tokenizer.n_layer == 12 and cfg.mel.fmax is None
    assert cfg.hift.stream_holdback_frames == 8 and cfg.hift.upsample_factor == 480
    s = cfg.sampling
    assert (s.top_p, s.top_k, s.ras_window, s.ras_tau) == (0.8, 25, 10, 0.1)


def test_registry_yaml_and_adapter_agree(model):
    assert get_model_class("cosyvoice3") is CosyVoice3Model
    assert HF_MODELS["cosyvoice3"]["model_path_hf"] == "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    deployment = yaml.safe_load(CONFIG_PATH.read_text())
    assert deployment["model"] == "cosyvoice3"
    placed = {node for group in deployment["node_groups"] for node in group["node_names"]}
    assert placed == {"speaker_encoder", "llm", "flow", "hift"}
    walks = {walk for group in deployment["node_groups"] for walk in group["graph_walks"]}
    assert walks == set(model.get_graph_walk_graphs())
    assert deployment["max_seq_len"] == model._max_seq_len()
    model.validate_config_yaml(deployment, str(CONFIG_PATH))
    with pytest.raises(ValueError, match="pages"):
        model.validate_config_yaml({**deployment, "max_concurrent_requests": 64}, str(CONFIG_PATH))
    assert isinstance(adapters.get_adapter("cosyvoice3"), adapters.CosyVoice3Adapter)


def test_graph_walks_and_partitions(model):
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {"prefill", "prefill_voice", "decode", "token2wav"}
    assert [n.name for n in walks["prefill_voice"].sections] == ["speaker_encoder", "llm"]
    assert [n.name for n in walks["token2wav"].sections] == ["flow", "hift"]
    assert walks["decode"].max_iters == model.config.max_speech_tokens
    partitions = {p.name: p for p in model.get_partitions()}
    assert partitions[LM_PARTITION].graph_walks == {"prefill", "prefill_voice", "decode"}
    assert partitions[TOKEN2WAV_PARTITION].producer_partitions == [LM_PARTITION]
    (connection,) = model.get_partition_topology().connections
    assert connection.edge_name == SPEECH_TOKENS
    policy = connection.chunk_policy_factory()
    assert isinstance(policy, ScheduledLeftContextChunkPolicy)
    # every window carries the whole token history
    assert policy._left_context > model.config.max_speech_tokens and policy._chunk == model.config.stream_poll_tokens


def test_process_prompt_modes_and_token_limits(model):
    clip = _clip()
    zero = model.process_prompt(
        "one two three", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, ref_text="ref words"
    )
    tok = _Tokenizer()
    expected = torch.cat([tok(f"{SYSTEM_PROMPT}{END_OF_PROMPT}ref words"), tok("one two three")])
    assert torch.equal(zero[TEXT_INPUTS][0], expected)
    assert zero[TOKEN_LIMITS][0].tolist() == [6, 60]  # 2x and 20x the text's 3 tokens
    assert zero["voice_key"][0].numel() == 1 and zero["ref_audio"][0].shape == clip.shape

    cross = model.process_prompt("one two", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]})
    assert torch.equal(cross[TEXT_INPUTS][0], torch.cat([tok(f"{SYSTEM_PROMPT}{END_OF_PROMPT}"), tok("one two")]))
    assert model.resolve_mode({}, uploaded=True) == ("cross_lingual", None)

    instruct = model.process_prompt(
        "one", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, instruct="speak slowly"
    )
    assert torch.equal(instruct[TEXT_INPUTS][0][:-1], tok(f"{SYSTEM_PROMPT} speak slowly{END_OF_PROMPT}"))

    capped = model.process_prompt(
        "one two three", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, max_new_tokens=10
    )
    assert capped[TOKEN_LIMITS][0].tolist() == [6, 10]
    forced = model.process_prompt(
        "one", ["text", "audio"], ["audio"], tensors={"audio_inputs": [clip]}, max_new_tokens=100, ignore_eos=True
    )
    assert forced[TOKEN_LIMITS][0].tolist() == [2, 100]


@pytest.mark.parametrize(
    "kwargs, inputs, message",
    [
        ({}, None, "needs a reference voice"),
        ({}, [_clip(), _clip()], "one reference clip"),
        ({}, [torch.zeros(SR)], "silent"),
        ({}, [_clip(0.3)], "at least 0.5 s"),
        ({}, [_clip(31.0)], "at most 30"),
        ({}, [torch.full((SR,), float("nan"))], "NaN"),
        ({"top_k": 0}, [_clip()], "top_k"),
        ({"temperature": -1}, [_clip()], "temperature"),
        ({"ras_window": 11}, [_clip()], "ras_window"),
        ({"max_new_tokens": 0}, [_clip()], "max_new_tokens"),
        ({"ref_text": 3}, [_clip()], "ref_text must be a string"),
        ({"ref_audio": "/etc/passwd"}, [_clip()], "upload the clip"),
    ],
)
def test_process_prompt_rejects_bad_requests(model, kwargs, inputs, message):
    modalities = ["text", "audio"] if inputs else ["text"]
    tensors = {"audio_inputs": inputs} if inputs else None
    with pytest.raises(ValueError, match=message):
        model.process_prompt("hello world", modalities, ["audio"], tensors=tensors, **kwargs)


def test_process_prompt_enforces_the_text_limit(model):
    with pytest.raises(ValueError, match="limit is 512"):
        model.process_prompt(" ".join(["w"] * 600), ["text", "audio"], ["audio"], tensors={"audio_inputs": [_clip()]})


def test_preset_voices_resolve_inside_voices_dir_only(tmp_path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")
    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: _Tokenizer())
    voices = tmp_path / "voices"
    voices.mkdir()
    soundfile.write(voices / "anna.wav", _clip().numpy(), SR)
    (voices / "anna.txt").write_text("hello there\n")
    soundfile.write(tmp_path / "outside.wav", _clip().numpy(), SR)
    model = CosyVoice3Model(model_path_hf=str(tmp_path), voices_dir=str(voices))
    assert model.get_voices() == ["anna"]
    assert model.resolve_mode({"voice": "anna"}, uploaded=False) == ("zero_shot", "hello there")
    out = model.process_prompt("one two", ["text"], ["audio"], voice="anna")
    assert out["ref_audio"][0].numel() == SR
    for name in ("../outside", "/etc/passwd", ".hidden", "a\\b"):
        with pytest.raises(ValueError, match="Invalid voice name"):
            model.process_prompt("one", ["text"], ["audio"], voice=name)
    with pytest.raises(ValueError, match="presets"):
        model.process_prompt("one", ["text"], ["audio"], voice="bob")
    with pytest.raises(ValueError, match="presets"):  # past NAME_MAX: the filesystem refuses the lookup
        model.process_prompt("one", ["text"], ["audio"], voice="a" * 300)


def test_initial_and_partition_args_route_the_walks(model):
    signals = {name: [f"ptr-{name}"] for name in (TEXT_INPUTS, TOKEN_LIMITS, "ref_audio", "voice_key")}
    zero = model.get_initial_forward_pass_args(
        LM_PARTITION, ["text", "audio"], ["audio"], signals, model_kwargs={"ref_text": "words"}
    )
    assert zero.full_metadata.graph_walk == "prefill_voice"
    routes = sorted((edge.next_node, edge.name) for edge in zero.inputs)
    assert ("speaker_encoder", "ref_audio") in routes and ("llm", TEXT_INPUTS) in routes
    # the clip stays persisted for the flow node
    assert set(zero.unpersist_tensors) == {f"ptr-{TEXT_INPUTS}", f"ptr-{TOKEN_LIMITS}"}
    cross = model.get_initial_forward_pass_args(LM_PARTITION, ["text", "audio"], ["audio"], signals)
    assert cross.full_metadata.graph_walk == "prefill"
    assert {edge.next_node for edge in cross.inputs} == {"llm"}

    decode = model.get_partition_forward_pass_args(LM_PARTITION, zero.full_metadata, {SPEECH_TOKENS: ["ptr-token"]})
    assert zero.full_metadata.graph_walk == "decode" and decode.inputs[0].name == PREV_TOKEN
    assert decode.unpersist_tensors == ["ptr-token"]
    done = model.get_partition_forward_pass_args(LM_PARTITION, zero.full_metadata, {})
    assert done.request_done

    t2w = model.get_initial_forward_pass_args(TOKEN2WAV_PARTITION, ["text", "audio"], ["audio"], signals)
    assert t2w.full_metadata.graph_walk == "token2wav"
    assert [edge.tensor_info for edge in t2w.inputs] == [["ptr-ref_audio"], ["ptr-voice_key"]]
    later = model.get_partition_forward_pass_args(TOKEN2WAV_PARTITION, t2w.full_metadata, signals)
    assert all(edge.tensor_info == [] for edge in later.inputs)


def test_postprocess_encodes_pcm16(model):
    pcm = model.postprocess(torch.tensor([0.0, 0.5, -2.0]), "audio")
    assert torch.frombuffer(bytearray(pcm), dtype=torch.int16).tolist() == [0, 16383, -32767]
    assert model.get_output_sample_rate() == SR


# -- adapter (#411 media rules) -----------------------------------------------------------


def _wav_bytes() -> bytes:
    soundfile = pytest.importorskip("soundfile")
    buf = io.BytesIO()
    soundfile.write(buf, _clip().numpy(), SR, format="WAV")
    return buf.getvalue()


def _speech(**extra) -> SpeechRequest:
    return SpeechRequest(model="cosyvoice3", input="hello", **extra)


def test_adapter_maps_voice_instructions_and_sampling(tmp_path):
    sa = adapters.CosyVoice3Adapter().speech_to_request(
        _speech(voice="anna", instructions="slowly", temperature=0.7, seed=5, top_k=10), tmp_path
    )
    assert sa.input_modalities == ["text"] and sa.file_paths is None
    assert sa.model_kwargs == {"voice": "anna", "instruct": "slowly", "temperature": 0.7, "seed": 5, "top_k": 10}


def test_adapter_accepts_a_data_url_and_base64_audio(tmp_path):
    adapter = adapters.CosyVoice3Adapter()
    raw = _wav_bytes()
    url = adapter.speech_to_request(
        _speech(ref_audio="data:audio/wav;base64," + base64.b64encode(raw).decode(), ref_text="w"), tmp_path
    )
    assert url.input_modalities == ["text", "audio"] and "ref_audio" not in url.model_kwargs
    assert Path(url.file_paths["audio"][0]).read_bytes() == raw and url.model_kwargs["ref_text"] == "w"
    bare = adapter.speech_to_request(_speech(ref_audio=base64.b64encode(raw).decode()), tmp_path)
    assert Path(bare.file_paths["audio"][0]).read_bytes() == raw


@pytest.mark.parametrize(
    "ref", ["/etc/passwd", "//etc/passwd", "///x", "secret", "relative/clip.wav", "file:///etc/passwd", "../clip.wav"]
)
def test_adapter_refuses_server_paths_and_leaves_nothing_behind(tmp_path, ref):
    upload = tmp_path / "uploads"
    with pytest.raises(ValueError, match="server file paths are refused"):
        adapters.CosyVoice3Adapter().speech_to_request(_speech(ref_audio=ref), upload)
    assert not upload.exists() or not any(upload.iterdir())


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
    with pytest.raises(ValueError, match="could not fetch ref_audio"):
        adapters.CosyVoice3Adapter().speech_to_request(_speech(ref_audio="http://example.test/clip.wav"), tmp_path)


def test_adapter_refuses_a_file_under_the_upload_dir(tmp_path):
    stored = tmp_path / "earlier_upload.wav"
    stored.write_bytes(b"RIFF\x24\x00\x00\x00WAVE")
    with pytest.raises(ValueError, match="server file paths are refused"):
        adapters.CosyVoice3Adapter().speech_to_request(_speech(ref_audio=str(stored)), tmp_path)
    with pytest.raises(ValueError, match="data URL or base64"):
        adapters.CosyVoice3Adapter().speech_to_request(_speech(ref_audio=["x"]), tmp_path)


def test_adapter_fetches_a_url_only_when_the_server_allows(tmp_path, monkeypatch):
    from mstar.api_server import media_io

    fetched = []
    monkeypatch.setattr(media_io, "save_remote_url", lambda url, upload_dir: fetched.append(url) or ("audio", "p"))
    monkeypatch.delenv("MSTAR_ALLOW_REMOTE", raising=False)
    with pytest.raises(ValueError, match="Remote media fetch is disabled"):
        adapters.CosyVoice3Adapter().speech_to_request(_speech(ref_audio="https://example.com/a.wav"), tmp_path)
    monkeypatch.setenv("MSTAR_ALLOW_REMOTE", "1")
    sa = adapters.CosyVoice3Adapter().speech_to_request(_speech(ref_audio="https://example.com/a.wav"), tmp_path)
    assert fetched == ["https://example.com/a.wav"] and sa.file_paths == {"audio": ["p"]}


# -- repetition-aware sampling -------------------------------------------------------------


def _ras(logits, seeds, history=None, temperature=1.0, top_p=0.8, top_k=25, threshold=1.0, blocked=None):
    b, v = logits.shape
    return ras_sample(
        logits,
        seed=seeds,
        step=torch.zeros(b, dtype=torch.long),
        history=history if history is not None else torch.full((b, 10), -1),
        temperature=torch.full((b,), temperature),
        top_p=torch.full((b,), top_p),
        top_k=torch.full((b,), top_k, dtype=torch.long),
        repeat_threshold=torch.full((b,), threshold),
        blocked=blocked if blocked is not None else torch.zeros(b, v, dtype=torch.bool),
    )


def test_hashed_uniforms_are_keyed_and_in_range():
    seed, step = torch.tensor([3, 3, 4]), torch.tensor([7, 8, 7])
    u = hashed_uniform(seed, step, 0, 1000)
    assert ((u > 0) & (u < 1)).all() and torch.equal(u, hashed_uniform(seed, step, 0, 1000))
    assert not torch.equal(u[0], u[1]) and not torch.equal(u[0], u[2])
    assert abs(float(u.mean()) - 0.5) < 0.02


def test_ras_nucleus_follows_the_reference_distribution():
    """The reference keeps ids while the mass before them is under top_p (and
    fewer than top_k), then draws in proportion to their probabilities."""
    n = 20000
    logits = torch.log(torch.tensor([0.4, 0.25, 0.2, 0.1, 0.05]))[None].expand(n, -1).contiguous()
    tokens = _ras(logits, torch.arange(n), threshold=0.0)
    freq = torch.bincount(tokens, minlength=5).double() / n
    expected = torch.tensor([0.4, 0.25, 0.2, 0.0, 0.0]) / 0.85  # 0.4 + 0.25 < 0.8 -> id 2 is still in
    assert (freq - expected).abs().max() < 0.015
    assert torch.equal(_ras(logits[:4], torch.arange(4), top_k=1), torch.zeros(4, dtype=torch.long))
    assert torch.equal(_ras(logits[:4], torch.arange(4), temperature=0.0), torch.zeros(4, dtype=torch.long))


def test_ras_redraws_a_repeated_token_from_the_rest_of_the_distribution():
    n = 20000
    probs = torch.tensor([0.9, 0.06, 0.04])
    logits = torch.log(probs)[None].expand(n, -1).contiguous()
    history = torch.full((n, 10), -1)
    history[:, -1] = 0  # one repeat reaches window * tau = 1
    tokens = _ras(logits, torch.arange(n), history=history)
    first = _ras(logits, torch.arange(n), threshold=0.0)
    redrawn = first == 0
    assert (tokens[redrawn] != 0).all() and torch.equal(tokens[~redrawn], first[~redrawn])
    freq = torch.bincount(tokens[redrawn], minlength=3).double() / int(redrawn.sum())
    assert (freq - torch.tensor([0.0, 0.6, 0.4])).abs().max() < 0.02
    # a token not in the history is never redrawn
    history[:, -1] = 2
    assert torch.equal(_ras(logits, torch.arange(n), history=history)[redrawn], first[redrawn])


def test_ras_respects_blocked_ids_and_is_batch_invariant():
    torch.manual_seed(0)
    logits = torch.randn(6, 50)
    blocked = torch.zeros(6, 50, dtype=torch.bool)
    blocked[:, logits[0].argmax()] = True
    seeds = torch.tensor([1, 2, 3, 4, 5, 6])
    together = _ras(logits, seeds, blocked=blocked, top_k=50, top_p=1.0)
    assert not blocked.gather(1, together[:, None]).any()
    alone = torch.cat(
        [_ras(logits[i : i + 1], seeds[i : i + 1], blocked=blocked[i : i + 1], top_k=50, top_p=1.0) for i in range(6)]
    )
    assert torch.equal(together, alone)


# -- LM -------------------------------------------------------------------------------


def _lm(cfg: CosyVoice3Config, seed: int = 0):
    torch.manual_seed(seed)
    lm = CosyVoice3LM(cfg.lm).eval()
    with torch.no_grad():
        # the layers allocate without initialising (the checkpoint fills them)
        for name, param in lm.named_parameters():
            param.fill_(1.0) if "norm" in name else param.normal_(0, 0.1)
    res = _fake_resources()
    res.bind(lm, LM_ATTN, LM_KV, LM_POS)
    return LMSubmodule(lm, cfg), res


def _info(rid: int, seed: int, **knobs):
    meta = {
        "temperature": 1.0,
        "top_p": 0.8,
        "top_k": 25,
        "ras_window": 10,
        "ras_tau": 0.1,
        "ignore_eos": False,
        "is_prefill": True,
        **knobs,
    }
    return SimpleNamespace(rid_handle=rid, request_id=rid, random_seed=seed, step_metadata=meta)


def _lm_run(sub, res, requests: dict, steps: int, **knobs) -> dict[int, list[int]]:
    """Prefill every request together, then batched decode steps until one stops."""
    rids = list(requests)
    infos = {rid: _info(rid, requests[rid][1], **knobs) for rid in rids}
    engine = SimpleNamespace(request_ids=rids, resources={LM_ATTN: res.kv_attn})
    tokens: dict[int, list[int]] = {rid: [] for rid in rids}
    live = list(rids)
    walk, inputs = "prefill", {rid: requests[rid][0] for rid in rids}
    for _ in range(steps):
        prepared = [sub.prepare_inputs(walk, infos[rid], inputs[rid]) for rid in live]
        res.plan([p.input_seq_len for p in prepared])
        engine.request_ids = live
        out = sub.forward_batched(walk, engine, **sub.preprocess(walk, engine, prepared))
        stopped = []
        for rid, p in zip(live, prepared, strict=True):
            sub.postprocess(rid, infos[rid], out[rid], inputs=p)
            tokens[rid].append(int(out[rid][SPEECH_TOKENS][0]))
            inputs[rid] = {PREV_TOKEN: out[rid][PREV_TOKEN]}
            if sub.check_stop(rid, infos[rid], out[rid]):
                stopped.append(rid)
        walk = "decode"
        if stopped:
            break
    return tokens


def _lm_request(text_len: int = 5, prompt: int = 4, limits=(0, 30)):
    g = torch.Generator().manual_seed(text_len)
    req = {TEXT_INPUTS: [torch.randint(0, 64, (text_len,), generator=g)], TOKEN_LIMITS: [torch.tensor(list(limits))]}
    if prompt:
        req[PROMPT_TOKENS] = [torch.randint(0, 40, (prompt,), generator=g)]
    return req


def test_lm_head_stays_float32_under_the_engines_autocast():
    cfg = _tiny_config()
    lm = CosyVoice3LM(cfg.lm).to(torch.bfloat16)
    lm.keep_head_in_float32()
    with torch.no_grad():
        lm.llm_decoder.weight.normal_(0, 0.1)
        hidden = torch.randn(3, cfg.lm.hidden_size, dtype=torch.bfloat16)
        plain = lm.logits(hidden)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            cast = lm.logits(hidden)
    assert cast.dtype == torch.float32 and torch.equal(cast, plain)


def test_lm_node_loads_the_checkpoint_head_without_rounding(model, monkeypatch):
    """The head is float32 before the checkpoint streams in: a bfloat16 head
    converted afterwards would keep only the bfloat16-rounded values."""
    model.config = _tiny_config()
    source = CosyVoice3LM(model.config.lm)
    with torch.no_grad():
        source.llm_decoder.weight.normal_(0, 0.1).add_(1.001)
    lm = model.config.lm
    q, kv = lm.num_attention_heads * lm.head_dim, lm.num_key_value_heads * lm.head_dim
    checkpoint = {}
    for name, value in source.state_dict().items():  # unfused, as llm.pt stores them
        if ".qkv_proj." in name:
            for part, chunk in zip(("q", "k", "v"), value.split([q, kv, kv]), strict=True):
                checkpoint[name.replace(".qkv_proj.", f".{part}_proj.")] = chunk
        elif ".gate_up_proj." in name:
            for part, chunk in zip(("gate", "up"), value.chunk(2), strict=True):
                checkpoint[name.replace(".gate_up_proj.", f".{part}_proj.")] = chunk
        else:
            checkpoint[name] = value
    monkeypatch.setattr(model, "_state_dict", lambda name: checkpoint.items())
    node = model._create_lm_submodule("cpu")
    head = node.model.llm_decoder.weight
    assert head.dtype == torch.float32 and torch.equal(head, source.llm_decoder.weight)


def test_lm_prefill_layout_matches_the_reference():
    cfg = _tiny_config()
    lm = CosyVoice3LM(cfg.lm)
    text, prompt = torch.tensor([3, 4, 5]), torch.tensor([7, 8])
    embeds = lm.build_prefill_embeds(text, prompt)
    sp = lm.speech_embedding.weight
    expected = torch.cat([sp[40:41], lm.embed_tokens(text), sp[42:43], sp[[7, 8]]])
    assert torch.equal(embeds, expected)


def test_lm_batches_like_single_requests_and_follows_the_seed():
    cfg = _tiny_config()
    sub, res = _lm(cfg)
    a, b = _lm_request(5, 4), _lm_request(7, 0)
    with torch.no_grad():
        together = _lm_run(sub, res, {0: (a, 11), 1: (b, 12)}, steps=6, ignore_eos=True)
        res.reset()
        alone_a = _lm_run(sub, res, {0: (a, 11)}, steps=6, ignore_eos=True)
        res.reset()
        alone_b = _lm_run(sub, res, {1: (b, 12)}, steps=6, ignore_eos=True)
        res.reset()
        reseeded = _lm_run(sub, res, {0: (a, 99)}, steps=6, ignore_eos=True)
    assert len(together[0]) == 6 and together[0] == alone_a[0] and together[1] == alone_b[1]
    assert len(set(together[0])) > 1 and reseeded[0] != alone_a[0]


def _fixed_logits(sub, favourite: int, second: int):
    """Make the head prefer ``favourite``, then ``second``, whatever the hidden state."""
    vocab = sub.lm.speech_vocab_size

    def logits(hidden):
        out = torch.full((hidden.shape[0], vocab), -20.0)
        out[:, favourite], out[:, second] = 10.0, 5.0
        return out

    sub.model.logits = logits


def test_lm_min_len_masks_only_sos_like_the_reference():
    """The reference's minimum length masks id ``speech_token_size`` (its sos),
    so a favoured sos is held off for min_len steps and then ends the decode."""
    cfg = _tiny_config()
    sub, res = _lm(cfg)
    _fixed_logits(sub, favourite=cfg.lm.sos_token, second=5)
    with torch.no_grad():
        tokens = _lm_run(sub, res, {0: (_lm_request(limits=(3, 30)), 0)}, steps=40, ras_window=0)[0]
    assert tokens == [5, 5, 5, cfg.lm.sos_token]
    # eos is not masked during min_len: it ends the decode on step one
    res.reset()
    _fixed_logits(sub, favourite=cfg.lm.eos_token, second=5)
    with torch.no_grad():
        tokens = _lm_run(sub, res, {1: (_lm_request(limits=(3, 30)), 0)}, steps=40, ras_window=0)[1]
    assert tokens == [cfg.lm.eos_token]


def test_lm_stops_on_any_control_id_or_the_length_cap():
    cfg = _tiny_config()
    sub, res = _lm(cfg)
    _fixed_logits(sub, favourite=cfg.lm.speech_token_size + 7, second=5)
    with torch.no_grad():
        assert _lm_run(sub, res, {0: (_lm_request(limits=(0, 30)), 0)}, steps=40, ras_window=0)[0] == [47]
        res.reset()
        # ignore_eos: control ids cannot be drawn and the cap ends the decode
        tokens = _lm_run(sub, res, {1: (_lm_request(limits=(0, 6)), 0)}, steps=40, ras_window=0, ignore_eos=True)
    assert tokens[1] == [5] * 6
    res.reset()
    _fixed_logits(sub, favourite=5, second=6)
    with torch.no_grad():
        assert _lm_run(sub, res, {2: (_lm_request(limits=(0, 4)), 0)}, steps=40, ras_window=0)[2] == [5] * 4


def _lm_prefill_then_decode(sub, res, limits, **knobs):
    """One request: prefill, postprocess, then one decode step on the prefill token."""
    info = _info(0, 0, ras_window=0, **knobs)
    engine = SimpleNamespace(request_ids=[0], resources={LM_ATTN: res.kv_attn})
    tokens, inputs = [], _lm_request(limits=limits)
    for walk in ("prefill", "decode"):
        prepared = sub.prepare_inputs(walk, info, inputs)
        res.plan([prepared.input_seq_len])
        out = sub.forward_batched(walk, engine, **sub.preprocess(walk, engine, [prepared]))[0]
        sub.postprocess(0, info, out, inputs=prepared)
        tokens.append(int(out[SPEECH_TOKENS][0]))
        inputs = {PREV_TOKEN: out[PREV_TOKEN]}
    return tokens, sub.check_stop(0, info, out)


def test_lm_a_request_ended_by_its_prefill_token_adds_no_speech():
    """The prefill walk has no decode loop to stop, so the decode step after a
    prefill stop token (or a spent budget) runs, samples EOS, which the flow
    drops, and ends the loop."""
    cfg = _tiny_config()
    sub, res = _lm(cfg)
    with torch.no_grad():
        _fixed_logits(sub, favourite=cfg.lm.speech_token_size + 7, second=5)
        assert _lm_prefill_then_decode(sub, res, (0, 30)) == ([47, cfg.lm.eos_token], {DECODE_LOOP})
        for limits, knobs in (((0, 1), {}), ((0, 1), {"ignore_eos": True})):
            res.reset()
            _fixed_logits(sub, favourite=5, second=6)
            assert _lm_prefill_then_decode(sub, res, limits, **knobs) == ([5, cfg.lm.eos_token], {DECODE_LOOP})
        res.reset()
        assert _lm_prefill_then_decode(sub, res, (0, 30)) == ([5, 5], set())
    assert filter_tokens(torch.tensor([5, cfg.lm.eos_token]), cfg).tolist() == [5]


def test_lm_stop_judges_each_step_by_its_own_index():
    """A speculative next step postprocessed before this step's stop check has
    moved ``generated`` on; the cap still counts this step's tokens."""
    cfg = _tiny_config()
    sub, _ = _lm(cfg)
    info = _info(4, 0)
    sub.prepare_inputs("prefill", info, _lm_request(limits=(0, 4)))
    for _ in range(4):
        sub.postprocess(4, info, {SPEECH_TOKENS: [torch.tensor([5])]})
    out = {SPEECH_TOKENS: [torch.tensor([5])], STEP_INDEX: [torch.tensor([2])]}
    assert sub.check_stop(4, info, out) == set()  # token 3 of 4
    out[STEP_INDEX] = [torch.tensor([3])]
    assert sub.check_stop(4, info, out) == {DECODE_LOOP}


def test_lm_ras_breaks_a_repeating_favourite():
    """With the redraw on, a token already in the window is redrawn from the
    rest of the distribution: the favourite cannot repeat back to back."""
    cfg = _tiny_config()
    sub, res = _lm(cfg)
    _fixed_logits(sub, favourite=5, second=6)
    with torch.no_grad():
        tokens = _lm_run(sub, res, {0: (_lm_request(limits=(0, 12)), 3)}, steps=40, ignore_eos=True)[0]
    assert len(tokens) == 12 and tokens[0] == 5
    assert all(b != 5 for a, b in pairwise(tokens) if a == 5)


def test_lm_ras_history_is_committed_once_per_step():
    cfg = _tiny_config()
    sub, _ = _lm(cfg)
    info = _info(3, 0)
    sub.prepare_inputs("prefill", info, _lm_request())
    state = sub.request_state(3)
    again = sub.prepare_inputs("decode", info, {PREV_TOKEN: [torch.tensor([9])]})
    sub.prepare_inputs("decode", info, {PREV_TOKEN: [torch.tensor([9])]})  # re-driven step
    assert state["history"].tolist() == [-1] * 10  # not committed until postprocess
    sub.postprocess(3, info, {SPEECH_TOKENS: [torch.tensor([4])]}, inputs=again)
    assert state["history"].tolist() == [-1] * 9 + [9] and state["generated"] == 1


def test_lm_declares_one_kv_segment_per_request_and_a_decode_capture():
    cfg = _tiny_config()
    sub, _ = _lm(cfg)
    step = sub.declare_step("decode", [0, 1], [SimpleNamespace(input_seq_len=1)] * 2)
    assert set(step.steps) == {LM_KV, LM_ATTN, LM_POS}
    assert [(s.request_id, s.span) for s in step.segments] == [(0, 1), (1, 1)]
    (graph,) = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert graph.capture_graph_walk == "decode" and graph.capture_batch_sizes == [1, 2, 4, 8, 16, 32]


# -- flow -----------------------------------------------------------------------------


def test_filter_tokens_drops_control_ids_and_long_silent_runs():
    cfg = _tiny_config()
    raw = torch.tensor([5, 1, 2, 1, 2, 1, 2, 1, 6, 45, 1, 7])
    # silent run 1 2 1 2 1 2 1: the 6th and 7th are cut; 45 is a control id
    assert filter_tokens(raw, cfg).tolist() == [5, 1, 2, 1, 2, 1, 6, 1, 7]


def test_chunk_mask_is_block_causal_from_frame_zero_and_masks_padding():
    valid = torch.tensor([7, 5])
    mask = chunk_attention_mask(valid, 7, 3, torch.tensor([True, False]))
    assert mask.shape == (2, 1, 7, 7)
    stream = mask[0, 0]
    for q in range(7):
        end = min((q // 3 + 1) * 3, 7)
        assert stream[q].tolist() == [k < end for k in range(7)]
    assert mask[1, 0].tolist() == [[k < 5 for k in range(7)]] * 7


def test_flow_hop_plan_follows_the_reference_ramp():
    cfg = _tiny_config()
    sub = FlowSubmodule(_random_flow(cfg), None, cfg)
    voice = _voice(cfg, prompt_tokens=10)
    # first hop 25 tokens, stretched to align prompt + hop on a 25-token chunk: 40; then 50, 100, 100
    assert [sub.hop_size(k, 10) for k in range(4)] == [40, 50, 100, 100]
    state = sub.request_state(0)
    tokens = torch.zeros(300, dtype=torch.long)
    assert sub.plan(state, voice, tokens[:42], final=False) is None  # 40 + 3 look-ahead not there yet
    assert sub.plan(state, voice, tokens[:43], final=False).settle == 40
    state.add("settled", 40)
    assert sub.plan(state, voice, tokens[:200], final=False).settle == 190  # 90 and 190 in one solve
    state.add("settled", 190)
    final = sub.plan(state, voice, tokens[:200], final=True)
    assert final.final and final.settle == 200


def test_chunked_flow_solves_equal_one_whole_streaming_solve():
    cfg = _tiny_config()
    flow = _random_flow(cfg)
    voice = _voice(cfg, prompt_tokens=10)
    tokens = torch.randint(0, 40, (97,), generator=torch.Generator().manual_seed(1))
    whole = flow.solve_rows([FlowRow(voice, tokens, 97, final=True, streaming=True)], 4)[0]
    offline = flow.solve_rows([FlowRow(voice, tokens, 97, final=True, streaming=False)], 4)[0]
    assert whole.shape == (80, 194) and not torch.allclose(whole, offline)
    for settle in (15, 40, 65, 90):
        part = flow.solve_rows([FlowRow(voice, tokens, settle, final=False, streaming=True)], 4)[0]
        torch.testing.assert_close(part, whole[:, : 2 * settle], rtol=0, atol=1e-10)


def test_flow_batches_rows_of_different_lengths_like_single_rows():
    cfg = _tiny_config()
    flow = _random_flow(cfg)
    rows = [
        FlowRow(_voice(cfg, 10, seed=1), torch.randint(0, 40, (60,)), 40, final=False, streaming=True),
        FlowRow(_voice(cfg, 23, seed=2), torch.randint(0, 40, (31,)), 31, final=True, streaming=True),
        FlowRow(_voice(cfg, 5, seed=3), torch.randint(0, 40, (12,)), 12, final=True, streaming=False),
    ]
    together = flow.solve_rows(rows, 3)
    for row, mel in zip(rows, together, strict=True):
        torch.testing.assert_close(mel, flow.solve_rows([row], 3)[0], rtol=0, atol=1e-10)


def test_flow_node_solves_without_tf32_and_restores_the_engine_setting(monkeypatch):
    cfg = _tiny_config()
    flow = _random_flow(cfg)
    node = FlowSubmodule(flow, None, cfg)
    seen = []
    solve = flow.solve_rows

    def spy(rows, steps):
        seen.append((torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32))
        if len(seen) == 2:
            raise RuntimeError("boom")
        return solve(rows, steps)

    monkeypatch.setattr(flow, "solve_rows", spy)
    voice = _voice(cfg, prompt_tokens=10)
    prior = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    try:
        node._run([0], [torch.randint(0, 40, (12,))], [voice], [True], [2])
        with pytest.raises(RuntimeError, match="boom"):
            node._run([1], [torch.randint(0, 40, (12,))], [voice], [True], [2])
        assert seen == [("highest", False)] * 2
        assert (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32) == ("high", True)
    finally:
        torch.set_float32_matmul_precision(prior[0])
        torch.backends.cudnn.allow_tf32 = prior[1]


@pytest.mark.parametrize(
    ("kwargs", "dtype"),
    [
        ({}, torch.bfloat16),
        ({"flow_estimator_dtype": "float32"}, torch.float32),
        ({"flow_estimator_dtype": "fp32"}, torch.float32),
    ],
)
def test_flow_estimator_is_bfloat16_unless_the_deployment_asks_for_float32(tmp_path, monkeypatch, kwargs, dtype):
    monkeypatch.setattr(CosyVoice3Model, "_build_text_tokenizer", lambda self: _Tokenizer())
    model = CosyVoice3Model(model_path_hf=str(tmp_path), **kwargs)
    model.config.flow = _tiny_config().flow
    checkpoint = CosyVoice3Flow(model.config.flow).state_dict()
    monkeypatch.setattr(model, "_state_dict", lambda name: checkpoint.items())
    monkeypatch.setattr(model, "_reference_encoder", lambda device: None)
    flow = model._create_flow_submodule("cpu").flow
    assert {p.dtype for p in flow.decoder.estimator.parameters()} == {dtype}
    assert {p.dtype for p in flow.input_embedding.parameters()} == {torch.float32}
    with pytest.raises(ValueError, match="Unknown dtype"):
        CosyVoice3Model(model_path_hf=str(tmp_path), flow_estimator_dtype="int8")


# -- HiFT -----------------------------------------------------------------------------


def test_hift_nonfinal_call_holds_back_eight_frames():
    cfg = _tiny_config()
    hift = _random_hift(cfg)
    mel = torch.randn(1, 80, 40, dtype=torch.float64)
    whole, _ = hift.vocode(mel, finalize=True)
    part, _ = hift.vocode(mel[:, :, :30], finalize=False)
    assert whole.shape[-1] == 40 * 480 and part.shape[-1] == (30 - 8) * 480
    torch.testing.assert_close(part[0], whole[0, : part.shape[-1]], rtol=0, atol=1e-10)


@pytest.mark.parametrize("chunk", [3, 17, 80])
@pytest.mark.parametrize("context", [0, 32])
def test_hift_stream_stitches_to_the_whole_decode(chunk, context):
    cfg = _tiny_config()
    hift = _random_hift(cfg)
    total = 80
    mel = torch.randn(80, total, dtype=torch.float64, generator=torch.Generator().manual_seed(chunk))
    with torch.no_grad():
        whole, _ = hift.vocode(mel[None], finalize=True)
        state, pieces, pos = StreamState(mel=mel.new_zeros(80, 0)), [], 0
        while pos < total:
            nxt = min(pos + chunk, total)
            pieces.append(stream_vocode(hift, state, mel[:, pos:nxt], nxt == total, context))
            pos = nxt
            # every non-final call emits exactly the samples no later frame can change
            assert state.emitted == (nxt * 480 if nxt == total else max(nxt - 8, 0) * 480)
    streamed = torch.cat(pieces)
    assert streamed.numel() == whole.shape[-1] == total * 480
    torch.testing.assert_close(streamed, whole[0], rtol=0, atol=1e-9)
    if context:
        assert state.mel.shape[-1] <= context + 8 + chunk + 8  # bounded history


def test_hift_node_final_flush_without_new_frames_releases_the_tail():
    cfg = _tiny_config()
    node = HiFTSubmodule(_random_hift(cfg), cfg)
    mel = torch.randn(80, 30, dtype=torch.float64)
    info = SimpleNamespace(rid_handle=0, request_id=0)
    outs = []
    for frames, final in ((mel, False), (mel[:, :0], True)):
        p = node.prepare_inputs("token2wav", info, {MEL: [frames], MEL_FINAL: [torch.tensor([final])]})
        outs.append(node.forward("token2wav", None, **node.preprocess("token2wav", None, [p]))[AUDIO_CHUNK][0])
    assert [o.numel() for o in outs] == [22 * 480, 8 * 480] and outs[0].dtype == torch.int16


# -- end to end: the #429 invariant ---------------------------------------------------------


def _stream_token2wav(cfg, flow_node, hift_node, policy, tokens: list[int]) -> list[torch.Tensor]:
    """Feed tokens one at a time through a real StreamBuffer and the two
    Token2Wav nodes, as the worker does; return the emitted audio chunks."""
    buffer = StreamBuffer(request_id=0, edge_name=SPEECH_TOKENS, from_partition=LM_PARTITION, policy=policy)
    info = SimpleNamespace(rid_handle=0, request_id=0, step_metadata={"n_cfm_timesteps": 3})
    chunks: list[torch.Tensor] = []

    def drain():
        while buffer.has_chunk_ready():
            chunk = buffer.pop_chunk()
            p = flow_node.prepare_inputs(
                "token2wav", info, {SPEECH_TOKENS: [chunk.data["data"]]}, is_final_stream_chunk=chunk.is_final
            )
            mel = flow_node.forward("token2wav", None, **flow_node.preprocess("token2wav", None, [p]))
            h = hift_node.prepare_inputs("token2wav", info, mel)
            chunks.append(
                hift_node.forward("token2wav", None, **hift_node.preprocess("token2wav", None, [h]))[AUDIO_CHUNK][0]
            )

    for i, token in enumerate(tokens):
        buffer.pre_read_register(f"t{i}")
        buffer.put(f"t{i}", torch.tensor([token]))
        drain()
    buffer.signal_done()
    drain()
    return chunks


@pytest.mark.parametrize("num_tokens, prompt_tokens", [(1, 10), (20, 10), (43, 10), (97, 10), (180, 23), (260, 25)])
def test_streamed_audio_equals_the_whole_streaming_decode(model, num_tokens, prompt_tokens):
    cfg = _tiny_config()
    model.config = cfg
    flow, hift = _random_flow(cfg), _random_hift(cfg)
    flow_node, hift_node = FlowSubmodule(flow, None, cfg), HiFTSubmodule(hift, cfg)
    voice = _voice(cfg, prompt_tokens)
    flow_node.request_state(0).add("voice", voice)
    g = torch.Generator().manual_seed(num_tokens)
    tokens = torch.randint(3, 40, (num_tokens,), generator=g).tolist()
    tokens[num_tokens // 2 : num_tokens // 2 + 8] = [1] * min(8, num_tokens - num_tokens // 2)  # a silent run
    tokens.append(cfg.lm.eos_token)
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    with torch.no_grad():
        chunks = _stream_token2wav(cfg, flow_node, hift_node, policy, tokens)
        kept = filter_tokens(torch.tensor(tokens), cfg)
        mel = flow.solve_rows([FlowRow(voice, kept, kept.numel(), final=True, streaming=True)], 3)[0]
        whole, _ = hift.vocode(mel[None], finalize=True)
    whole = (whole[0].clamp(-1, 1) * 32767).to(torch.int16)
    streamed = torch.cat(chunks)
    assert streamed.numel() == whole.numel() == 2 * 480 * kept.numel()
    assert (streamed.int() - whole.int()).abs().max() <= 1
    if num_tokens >= 97:
        # audio leaves before the stream ends
        assert sum(c.numel() for c in chunks[:-1]) > 0


def test_offline_deployment_decodes_once_with_full_attention(model):
    cfg = _tiny_config(stream_hop_tokens=0)
    model.config = cfg
    flow, hift = _random_flow(cfg), _random_hift(cfg)
    flow_node, hift_node = FlowSubmodule(flow, None, cfg), HiFTSubmodule(hift, cfg)
    voice = _voice(cfg, 10)
    flow_node.request_state(0).add("voice", voice)
    tokens = torch.randint(3, 40, (60,), generator=torch.Generator().manual_seed(0)).tolist() + [cfg.lm.eos_token]
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    with torch.no_grad():
        chunks = _stream_token2wav(cfg, flow_node, hift_node, policy, tokens)
        mel = flow.solve_rows([FlowRow(voice, torch.tensor(tokens[:-1]), 60, final=True, streaming=False)], 3)[0]
        whole, _ = hift.vocode(mel[None], finalize=True)
    assert [c.numel() for c in chunks if c.numel()] == [60 * 960]
    whole = (whole[0].clamp(-1, 1) * 32767).to(torch.int16)
    assert (torch.cat(chunks).int() - whole.int()).abs().max() <= 1
