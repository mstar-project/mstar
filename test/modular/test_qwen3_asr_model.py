"""CPU tests for the Qwen3-ASR integration (1.7B and the 0.6B realtime checkpoint).

Config reading from the checkpoint JSON, the reference SDK's prompt layout
(placeholder count, forced-language prefix), audio-token arithmetic, the
encoder submodule's window declaration and packing, the LLM submodule's
embedding splice and stop rules, the request state machine, and the
registry/CLI/benchmark wiring.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, ".")

from mstar.engine.resources import KVSpec, RaggedAttentionSpec  # noqa: E402
from mstar.model.components.audio_features import LogMelSpectrogram  # noqa: E402
from mstar.model.components.aut_encoder import AuTEncoder, AuTEncoderConfig  # noqa: E402
from mstar.model.components.qwen3_lm import Qwen3LMConfig  # noqa: E402
from mstar.model.qwen3_asr.config import (  # noqa: E402
    ASR_TEXT_TAG,
    AUT_ATTN,
    DECODE_WALK,
    ENCODER_NODE,
    KV_CACHE,
    LLM_NODE,
    PREFILL_WALK,
    Qwen3ASRModelConfig,
)
from mstar.model.qwen3_asr.qwen3_asr_model import Qwen3ASRModel  # noqa: E402
from mstar.model.qwen3_asr.submodules import Qwen3ASREncoderSubmodule, Qwen3ASRLLMSubmodule  # noqa: E402
from mstar.model.registry import HF_MODELS, get_model_class  # noqa: E402
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

IM_START, IM_END, EOT = 151644, 151645, 151643
AUDIO_START, AUDIO_END, AUDIO_PAD, ASR_TEXT = 151669, 151670, 151676, 151704


class _FakeTokenizer:
    """Whole-word tokenizer with the checkpoint's special ids."""

    SPECIALS = {
        "<|im_start|>": IM_START, "<|im_end|>": IM_END, "<|endoftext|>": EOT,
        "<|audio_start|>": AUDIO_START, "<|audio_end|>": AUDIO_END, "<|audio_pad|>": AUDIO_PAD,
        "<asr_text>": ASR_TEXT,
    }

    def __init__(self):
        self.all_special_ids = sorted(v for k, v in self.SPECIALS.items() if k != "<asr_text>")
        self._words: dict[str, int] = {}
        self._ids: dict[int, str] = {v: k for k, v in self.SPECIALS.items()}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        import re

        ids = []
        for piece in re.split(r"(<\|[a-z_]+\|>|<asr_text>)", text):
            if not piece:
                continue
            if piece in self.SPECIALS:
                ids.append(self.SPECIALS[piece])
                continue
            for word in re.findall(r"\S+|\s+", piece):
                if word not in self._words:
                    self._words[word] = 1000 + len(self._words)
                    self._ids[self._words[word]] = word.replace(" ", "Ġ").replace("\n", "Ċ")
                ids.append(self._words[word])
        return ids

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._ids[ids]
        return [self._ids[i] for i in ids]


def _tiny_config() -> Qwen3ASRModelConfig:
    return Qwen3ASRModelConfig(
        audio=AuTEncoderConfig(
            d_model=64, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128, num_mel_bins=16,
            n_window=50, n_window_infer=200, downsample_hidden_size=8, output_dim=32,
        ),
        text=Qwen3LMConfig(
            hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            intermediate_size=48, vocab_size=151936,
        ),
        support_languages=["Chinese", "English", "German"],
        num_mel_bins=16,
    )


def _make_model(config: Qwen3ASRModelConfig | None = None) -> Qwen3ASRModel:
    model = object.__new__(Qwen3ASRModel)
    model.config = config or _tiny_config()
    model.tokenizer = _FakeTokenizer()
    model.log_mel = LogMelSpectrogram(
        num_mel_bins=model.config.num_mel_bins, sampling_rate=model.config.sampling_rate,
        n_fft=model.config.n_fft, hop_length=model.config.hop_length,
    )
    from mstar.model.utils import ByteLevelDetokenizer

    model._detokenizer = ByteLevelDetokenizer(model.tokenizer)
    model._submodule_cache = {}
    return model


def _engine_inputs(rids):
    return ModelInputsFromEngine(request_ids=list(rids), per_request_info={})


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def _snapshot(repo: str):
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, allow_patterns=["*.json"], local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


@pytest.mark.parametrize("repo,d_model,enc_layers,hidden,ffn", [
    ("Qwen/Qwen3-ASR-1.7B", 1024, 24, 2048, 6144),
    ("Qwen/Qwen3-ASR-0.6B", 896, 18, 1024, 3072),
])
def test_config_reads_both_checkpoints(repo, d_model, enc_layers, hidden, ffn):
    snap = _snapshot(repo)
    if snap is None:
        pytest.skip(f"{repo} not in the local HF cache")
    cfg = Qwen3ASRModelConfig.from_pretrained(snap)
    assert cfg.audio.d_model == d_model and cfg.audio.encoder_layers == enc_layers
    assert cfg.audio.output_dim == hidden == cfg.text.hidden_size
    assert cfg.text.intermediate_size == ffn and cfg.text.num_hidden_layers == 28
    assert cfg.text.num_key_value_heads == 8 and cfg.text.head_dim == 128
    assert cfg.text.rope_theta == 1_000_000 and cfg.text.tie_word_embeddings is True
    assert (cfg.audio_start_token_id, cfg.audio_end_token_id, cfg.audio_token_id) == (AUDIO_START, AUDIO_END, AUDIO_PAD)
    assert "English" in cfg.support_languages and len(cfg.support_languages) == 30
    assert cfg.num_mel_bins == 128 and cfg.stop_token_ids == {IM_END, EOT}
    assert cfg.audio.window_tokens == 104 and cfg.num_audio_tokens(16_000 * 10) == 130


def test_language_names():
    cfg = _tiny_config()
    assert cfg.language_name(None) is None and cfg.language_name("") is None
    assert cfg.language_name("en") == "English" and cfg.language_name("english") == "English"
    assert cfg.language_name("de") == "German" and cfg.language_name("zh") == "Chinese"
    with pytest.raises(ValueError, match="Unsupported"):
        cfg.language_name("fr")  # not in this tiny checkpoint's list


# --------------------------------------------------------------------------
# prompt + features
# --------------------------------------------------------------------------


def test_prompt_layout_matches_reference_template():
    model = _make_model()
    text = model.prompt_text(3, context="", language=None)
    assert text == (
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert model.prompt_text(1, context="ctx", language="English").endswith(
        f"<|im_start|>assistant\nlanguage English{ASR_TEXT_TAG}"
    )
    ids = model.prompt_ids(5, context="", language=None)
    assert ids.count(AUDIO_PAD) == 5
    start, end = ids.index(AUDIO_START), ids.index(AUDIO_END)
    assert end - start == 6 and ids[start + 1:end] == [AUDIO_PAD] * 5
    assert ids[0] == IM_START and ids[-1] not in (ASR_TEXT,)
    forced = model.prompt_ids(2, context="", language="German")
    assert forced[-1] == ASR_TEXT and forced.count(AUDIO_PAD) == 2
    # streaming continues from the stable part of the previous hypothesis
    continued = model.prompt_ids(2, context="", language="German", assistant_prefix=" so far")
    assert continued[:len(forced)] == forced and continued[len(forced):] == model.tokenizer.encode(" so far")


def test_process_prompt_builds_features_and_prompt():
    model = _make_model()
    wave = torch.randn(16_000 * 3 + 40)
    out = model.process_prompt(None, ["audio", "text"], ["text"], {"audio_inputs": [wave]}, language="en")
    assert torch.equal(out["audio"][0], wave)  # samples as they are; the encoder makes the mel
    n_tokens = model.config.audio.tokens_for_frames(wave.numel() // 160)  # one frame per hop, unpadded
    ids = out["text_inputs"][0].tolist()
    assert ids.count(AUDIO_PAD) == n_tokens and ids[-1] == ASR_TEXT

    free = model.process_prompt("hot words", ["audio", "text"], ["text"], {"audio_inputs": [wave]})
    assert free["text_inputs"][0].tolist()[-1] != ASR_TEXT  # model reports the language itself
    assert model.tokenizer.encode("hot words")[0] in free["text_inputs"][0].tolist()

    prefixed = model.process_prompt(
        None, ["audio"], ["text"], {"audio_inputs": [wave]}, assistant_prefix="language English<asr_text> hi",
    )
    assert prefixed["text_inputs"][0].tolist()[-3:] == model.tokenizer.encode("<asr_text> hi")[-3:]

    short = model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [torch.randn(1600)]})
    assert short["audio"][0].numel() == 8000  # padded to 0.5 s
    with pytest.raises(ValueError, match="at most"):
        model.process_prompt(None, ["audio"], ["text"],
                             {"audio_inputs": [torch.zeros(model.config.max_audio_samples + 16_000)]})
    with pytest.raises(ValueError, match="exactly one"):
        model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": []})


def test_postprocess_keeps_asr_text_tag_and_drops_specials():
    model = _make_model()
    ids = model.tokenizer.encode("language English<asr_text> hello world") + [IM_END]
    text = model.postprocess(torch.tensor(ids), "text").decode("utf-8")
    assert text == "language English<asr_text> hello world"


# --------------------------------------------------------------------------
# submodules
# --------------------------------------------------------------------------


def _encoder_submodule(cfg):
    encoder = AuTEncoder(cfg.audio, attn_key=AUT_ATTN)
    torch.manual_seed(0)
    state = {k: torch.randn_like(v) * 0.05 for k, v in encoder.state_dict().items()}
    with torch.no_grad():
        for name, param in encoder.named_parameters():
            param.copy_(state[name])
    encoder.reset_buffers()
    # attention unbound -> SDPA per window (the CPU path)
    for layer in encoder.layers:
        layer.self_attn.ragged = None
    return Qwen3ASREncoderSubmodule(encoder, cfg).eval()


def test_encoder_submodule_declares_windows_and_packs_requests():
    cfg = _tiny_config()
    sub = _encoder_submodule(cfg)
    fwd = type("F", (), {"request_id": "a"})()
    # samples in (one mel frame per hop), features made on the fly
    rows = [sub.prepare_inputs(PREFILL_WALK, fwd, {"audio": [torch.randn(n * cfg.hop_length)]})
            for n in (230, 150, 305)]
    assert [r.tensor_inputs["audio_features"].shape for r in rows] == [(cfg.num_mel_bins, n) for n in (230, 150, 305)]
    assert [r.input_seq_len for r in rows] == [cfg.audio.tokens_for_frames(n) for n in (230, 150, 305)]
    step = sub.declare_step(PREFILL_WALK, ["a", "b", "c"], rows)
    assert step.steps[AUT_ATTN].causal is False
    expected = [(rid, w) for rid, r in zip("abc", rows, strict=True) for w in cfg.audio.window_lengths(r.input_seq_len)]
    assert [(s.request_id, s.span) for s in step.segments] == expected
    assert all(s.label == "main" for s in step.segments)

    batch = sub.preprocess(PREFILL_WALK, _engine_inputs("abc"), rows)
    assert batch["audio_features"].shape == (3, cfg.num_mel_bins, 305) and batch["feature_lens"] == [230, 150, 305]
    with torch.no_grad():
        out = sub.forward_batched(PREFILL_WALK, _engine_inputs("abc"), **batch)
        single = sub.forward(PREFILL_WALK, _engine_inputs("b"),
                             audio_features=rows[1].tensor_inputs["audio_features"], feature_lens=[150])
    assert [out[r]["audio_embeds"][0].shape[0] for r in "abc"] == [r.input_seq_len for r in rows]
    assert out["a"]["audio_embeds"][0].shape[1] == cfg.audio.output_dim
    assert torch.allclose(out["b"]["audio_embeds"][0], single["audio_embeds"][0], atol=1e-5)
    assert sub.max_batch_size(PREFILL_WALK) == sub.MAX_BATCH_SIZE and sub.can_batch(None, [])


def _llm_submodule(cfg):
    sub = object.__new__(Qwen3ASRLLMSubmodule)
    torch.nn.Module.__init__(sub)
    sub.request_states, sub.node_resources = {}, {}
    sub.config = cfg
    embed = torch.nn.Embedding(cfg.text.vocab_size, cfg.text.hidden_size)
    sub.model = type("M", (), {"embed": staticmethod(embed.forward), "embed_tokens": embed})()
    sub.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
    return sub


def test_llm_prefill_splices_audio_embeddings_and_decode_carries_ids():
    cfg = _tiny_config()
    sub = _llm_submodule(cfg)
    ids = torch.tensor([IM_START, 1001, AUDIO_START, AUDIO_PAD, AUDIO_PAD, AUDIO_PAD, AUDIO_END, 1002])
    audio = torch.arange(3 * cfg.text.hidden_size, dtype=torch.float32).view(3, -1)
    fwd = type("F", (), {"request_id": "r"})()
    row = sub.prepare_inputs(PREFILL_WALK, fwd, {"text_inputs": [ids], "audio_embeds": [audio]})
    assert row.input_seq_len == 8 and row.input_embeds.shape == (8, cfg.text.hidden_size)
    assert torch.equal(row.input_embeds[3:6], audio)
    assert torch.equal(row.input_embeds[0], sub.model.embed_tokens(ids[:1])[0])
    with pytest.raises(ValueError, match="placeholders"):
        sub.prepare_inputs(PREFILL_WALK, fwd, {"text_inputs": [ids], "audio_embeds": [audio[:2]]})
    batch = sub.preprocess(PREFILL_WALK, _engine_inputs("r"), [row])
    assert set(batch) == {"input_embeds"}

    decode_row = sub.prepare_inputs(DECODE_WALK, fwd, {"text_inputs": [torch.tensor([42])]})
    assert decode_row.input_seq_len == 1 and decode_row.input_embeds is None
    assert sub.preprocess(DECODE_WALK, _engine_inputs("r"), [decode_row])["input_ids"].tolist() == [42]
    step = sub.declare_step(PREFILL_WALK, ["r"], [row])
    assert [(s.label, s.span) for s in step.segments] == [("main", 8)] and KV_CACHE in step.steps
    (decode, prefill) = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert decode.capture_batch_sizes == sub.DECODE_CAPTURE_BATCH_SIZES
    assert prefill.capture_token_lengths == sub.PREFILL_TOKEN_BUCKETS
    assert prefill.make_node_input(7).input_embeds.shape == (7, cfg.text.hidden_size)


def test_llm_check_stop():
    cfg = _tiny_config()
    sub = _llm_submodule(cfg)

    class _Cfg:
        ignore_eos = False

    def info(walk, decoded, max_tokens=4096):
        return type("I", (), {"graph_walk": walk, "max_tokens": max_tokens,
                              "resource_configs": {"sampler": _Cfg()},
                              "dynamic_loop_iter_counts": {"decode_loop": decoded}})()

    word, im_end, eot = ({"new_token": [torch.tensor([t])]} for t in (1234, IM_END, EOT))
    assert sub.check_stop("r", info(PREFILL_WALK, 0), im_end) == set()
    assert sub.check_stop("r", info(DECODE_WALK, 0), im_end) == {"decode_loop"}
    assert sub.check_stop("r", info(DECODE_WALK, 0), eot) == {"decode_loop"}
    assert sub.check_stop("r", info(DECODE_WALK, 0), word) == set()
    assert sub.check_stop("r", info(DECODE_WALK, 9, max_tokens=10), word) == {"decode_loop"}


# --------------------------------------------------------------------------
# model: graph, resources, state machine
# --------------------------------------------------------------------------


def test_graph_resources_and_state_machine():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {PREFILL_WALK, DECODE_WALK} and model.nodes == [LLM_NODE, ENCODER_NODE]
    specs = model.get_node_resources()
    ragged = next(s for s in specs if isinstance(s, RaggedAttentionSpec))
    assert ragged.nodes == {ENCODER_NODE} and ragged.config.num_qo_heads == 4 and ragged.config.head_dim == 16
    assert ragged.config.max_segments_per_request == model.MAX_WINDOWS_PER_REQUEST
    kv = next(s for s in specs if isinstance(s, KVSpec))
    assert kv.nodes == {LLM_NODE} and kv.config.num_kv_heads == 2 and kv.config.max_num_pages == model.KV_PAGES
    assert model.get_max_output_tokens() == 4096 and model.get_max_output_tokens(max_output_tokens=8) == 8

    args = model.get_initial_forward_pass_args(
        "default", ["audio", "text"], ["text"], {"audio": [object()], "text_inputs": [object()]},
    )
    assert args.full_metadata.graph_walk == PREFILL_WALK
    assert [(e.next_node, e.name) for e in args.inputs] == [(ENCODER_NODE, "audio"), (LLM_NODE, "text_inputs")]
    nxt = model.get_partition_forward_pass_args("default", args.full_metadata, {"new_token": [object()]})
    assert nxt.full_metadata.graph_walk == DECODE_WALK and not nxt.full_metadata.is_prefill
    assert model.get_partition_forward_pass_args("default", nxt.full_metadata, {}).request_done


def test_llm_weight_remap():
    remap = Qwen3ASRModel._llm_remap
    assert remap("thinker.model.layers.3.self_attn.q_proj.weight") == "layers.3.self_attn.q_proj.weight"
    assert remap("thinker.model.embed_tokens.weight") == "embed_tokens.weight"
    assert remap("thinker.lm_head.weight") == "lm_head.weight"
    assert remap("thinker.audio_tower.conv2d1.weight") is None


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,repo", [
    ("qwen3_asr", "Qwen/Qwen3-ASR-1.7B"),
    ("qwen3_asr_realtime", "Qwen/Qwen3-ASR-0.6B"),
])
def test_registry_cli_config_and_benchmark_entries_agree(key, repo):
    from benchmark.base import ModelType, RequestType
    from mstar.cli.main import DEFAULT_CONFIGS, _next_steps

    assert get_model_class(key) is Qwen3ASRModel
    assert HF_MODELS[key] == {"model_path_hf": repo}
    assert DEFAULT_CONFIGS[key] == f"{key}.yaml"
    serving = yaml.safe_load((REPO / "configs" / f"{key}.yaml").read_text())
    assert serving["model"] == key and serving["node_groups"][0]["node_names"] == [ENCODER_NODE, LLM_NODE]
    assert set(serving["resources"]) == {KV_CACHE}
    bench = ModelType(key).inst()
    assert bench.get_hf_url() == repo and bench.get_supported_modalities() == {RequestType.A2T}
    assert "transcribe(" in _next_steps(key, "0.0.0.0", 8000)


# --------------------------------------------------------------------------
# OpenAI adapter
# --------------------------------------------------------------------------


def test_openai_adapter_parses_language_line_and_continues_hypotheses():
    from mstar.api_server.openai import adapters
    from mstar.api_server.openai.protocol import TranscriptionRequest

    for key in ("qwen3_asr", "qwen3_asr_realtime", "whisper_large_v3_turbo"):
        assert adapters.get_adapter(key).supports_transcriptions
    ad = adapters.get_adapter("qwen3_asr")
    assert ad.supports_realtime_transcription and ad.max_audio_seconds == 1200.0

    req = TranscriptionRequest(language="en", prompt="names: Ada")
    sa = ad.transcription_to_request(req, "/tmp/a.wav")
    assert sa.model_kwargs == {"language": "en", "initial_prompt": "names: Ada", "temperature": 0.0}
    step = ad.realtime_step_request(req, "/tmp/a.wav", "language English<asr_text> so far")
    assert step.model_kwargs["assistant_prefix"] == "language English<asr_text> so far"

    free = TranscriptionRequest()
    t = ad.parse_transcript("language English<asr_text> Hello there.", free)
    assert (t.text, t.language) == ("Hello there.", "en")  # ISO code, like Whisper's language token
    assert ad.parse_transcript("language Cantonese<asr_text> x", free).language == "yue"
    assert ad.parse_transcript("language Klingon<asr_text> x", free).language == "Klingon"
    assert ad.parse_transcript("language None<asr_text>", free).language is None
    assert ad.parse_transcript(" plain words ", req).text == "plain words"
    assert ad.parse_transcript(" plain words ", req).language == "en"
    assert ad.stream_delta("language English<asr_text>") == "" and ad.stream_delta(" Hello") == " Hello"
