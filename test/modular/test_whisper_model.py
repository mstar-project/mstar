"""CPU tests for the Whisper ASR integration (turbo and large-v3 from one class).

Real-weight parity lives in ``test/asr``; here everything runs on random
weights or on config JSON alone: the log-mel front end and the native
encoder are pinned to their HF counterparts on tiny shapes, the prompt
builder and request state machine are exercised for both the forced-language
and language-detection paths, and the registry/CLI/benchmark wiring is
checked for consistency.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, ".")

from mstar.engine.resources import KVSpec  # noqa: E402
from mstar.model.components.audio_features import (  # noqa: E402
    LogMelSpectrogram,
    load_audio_file,
    sinusoid_positions,
)
from mstar.model.registry import HF_MODELS, get_model_class  # noqa: E402
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs  # noqa: E402
from mstar.model.whisper.components.encoder import WhisperEncoderModel  # noqa: E402
from mstar.model.whisper.config import (  # noqa: E402
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    DECODE_WALK,
    DECODER_NODE,
    DETECT_LANGUAGE_WALK,
    ENCODER_NODE,
    KV_CACHE,
    PREFILL_PROMPT_WALK,
    PREFILL_WALK,
    WhisperModelConfig,
)
from mstar.model.whisper.submodules import WhisperDecoderSubmodule, WhisperEncoderSubmodule  # noqa: E402
from mstar.model.whisper.whisper_model import WhisperDetokenizer, WhisperModel  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# large-v3 token ids, as in the checkpoints' generation_config.json
SOT, EOT, EN, DE, TRANSCRIBE, TRANSLATE, PREV, NOTS = 50258, 50257, 50259, 50261, 50360, 50359, 50362, 50364


def _tiny_config(**overrides) -> WhisperModelConfig:
    values = dict(
        d_model=64, decoder_layers=2, decoder_attention_heads=4, decoder_ffn_dim=128,
        encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128,
        num_mel_bins=16, vocab_size=51866, max_target_positions=448, max_source_positions=50,
        lang_to_id={"<|en|>": EN, "<|de|>": DE}, task_to_id={"transcribe": TRANSCRIBE, "translate": TRANSLATE},
        suppress_tokens=[1, 2], begin_suppress_tokens=[220, EOT],
    )
    values.update(overrides)
    return WhisperModelConfig(**values)


class _FakeTokenizer:
    """Just enough of the Whisper tokenizer for prompt and detokenizer tests."""

    def __init__(self, config: WhisperModelConfig):
        specials = {SOT, EOT, EN, DE, TRANSCRIBE, TRANSLATE, PREV, NOTS}
        self.all_special_ids = sorted(specials)
        self._names = {SOT: "<|startoftranscript|>", EOT: "<|endoftext|>", EN: "<|en|>", DE: "<|de|>",
                       TRANSCRIBE: "<|transcribe|>", TRANSLATE: "<|translate|>",
                       PREV: "<|startofprev|>", NOTS: "<|notimestamps|>"}
        self.config = config

    def convert_ids_to_tokens(self, ids):
        single = isinstance(ids, int)
        ids = [ids] if single else ids
        out = []
        for i in ids:
            if i in self._names:
                out.append(self._names[i])
            elif self.config.is_timestamp(i):
                out.append(f"<|{self.config.timestamp_seconds(i):.2f}|>")
            else:
                # byte-level BPE renders a leading space as 'Ġ'
                out.append("Ġ" + chr(ord("a") + i % 26))
        return out[0] if single else out

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1000 + (ord(c) % 50) for c in text]


def _make_model(config: WhisperModelConfig | None = None) -> WhisperModel:
    model = object.__new__(WhisperModel)
    model.config = config or _tiny_config()
    model.tokenizer = _FakeTokenizer(model.config)
    model.log_mel = LogMelSpectrogram(
        num_mel_bins=model.config.num_mel_bins, sampling_rate=model.config.sampling_rate,
        n_fft=model.config.n_fft, hop_length=model.config.hop_length, chunk_length=model.config.chunk_length,
    )
    model._detokenizer = WhisperDetokenizer(model.tokenizer, model.config)
    model._submodule_cache = {}
    return model


# --------------------------------------------------------------------------
# config + prompts
# --------------------------------------------------------------------------


def _snapshot(repo: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo, allow_patterns=["*.json", "*.txt"], local_files_only=True))
    except Exception:  # noqa: BLE001 — no cache, skip
        return None


@pytest.mark.parametrize("repo,decoder_layers,alignment_heads", [
    ("openai/whisper-large-v3-turbo", 4, 6),
    ("openai/whisper-large-v3", 32, 10),
])
def test_config_reads_turbo_and_v3_checkpoints(repo, decoder_layers, alignment_heads):
    snap = _snapshot(repo)
    if snap is None:
        pytest.skip(f"{repo} not in the local HF cache")
    cfg = WhisperModelConfig.from_pretrained(snap)
    assert cfg.decoder_layers == decoder_layers and cfg.encoder_layers == 32
    assert cfg.d_model == 1280 and cfg.encoder_attention_heads == 20 and cfg.encoder_ffn_dim == 5120
    assert cfg.num_mel_bins == 128 and cfg.num_frames == 3000 and cfg.n_samples == 480_000
    assert len(cfg.alignment_heads) == alignment_heads
    assert cfg.prev_sot_token_id == PREV and cfg.timestamp_begin == NOTS + 1
    assert cfg.language_token("en") == EN and cfg.language_of(DE) == "de"
    assert len(cfg.language_token_ids) == 100
    assert cfg.decoder_prompt_ids("en") == [SOT, EN, TRANSCRIBE, NOTS]


def test_prompt_ids_cover_forced_detected_timestamped_and_prev():
    cfg = _tiny_config()
    assert cfg.decoder_prompt_ids("en") == [SOT, EN, TRANSCRIBE, NOTS]
    assert cfg.decoder_prompt_ids("de", task="translate", timestamps=True) == [SOT, DE, TRANSLATE]
    # detection stops at <|startoftranscript|>; the tail follows the sampled language
    assert cfg.decoder_prompt_ids(None) == [SOT]
    assert cfg.prompt_tail_ids() == [TRANSCRIBE, NOTS]
    assert cfg.prompt_tail_ids(timestamps=True) == [TRANSCRIBE]
    prev = list(range(1000, 1300))
    ids = cfg.decoder_prompt_ids("en", prev_tokens=prev)
    assert ids[0] == PREV and ids[-4:] == [SOT, EN, TRANSCRIBE, NOTS]
    assert ids[1:-4] == prev[-cfg.max_prev_tokens:] and cfg.max_prev_tokens == 223
    with pytest.raises(ValueError, match="Unknown Whisper language"):
        cfg.decoder_prompt_ids("xx")
    with pytest.raises(ValueError, match="Unknown Whisper task"):
        cfg.decoder_prompt_ids("en", task="sing")


def test_timestamp_tokens():
    cfg = _tiny_config()
    assert cfg.is_timestamp(cfg.timestamp_begin) and not cfg.is_timestamp(NOTS)
    assert cfg.timestamp_seconds(cfg.timestamp_begin + 150) == pytest.approx(3.0)


# --------------------------------------------------------------------------
# log-mel front end
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_mel_bins", [80, 128])
def test_log_mel_matches_hf_feature_extractor(num_mel_bins):
    transformers = pytest.importorskip("transformers")
    fe = transformers.WhisperFeatureExtractor(feature_size=num_mel_bins)
    mel = LogMelSpectrogram(num_mel_bins=num_mel_bins)
    torch.manual_seed(0)
    wave = torch.randn(int(16_000 * 3.37)) * 0.1
    hf = torch.from_numpy(fe(wave.numpy(), sampling_rate=16_000, return_tensors="np")["input_features"][0])
    ours = mel(mel.pad_or_trim(wave))
    assert ours.shape == hf.shape == (num_mel_bins, 3000)
    assert torch.allclose(ours, hf, atol=2e-4), (ours - hf).abs().max()


def test_log_mel_frame_count_and_trim():
    mel = LogMelSpectrogram()
    wave = torch.zeros(16_000 * 3 + 37)
    assert mel(wave).shape == (128, mel.num_frames(wave.numel()))
    assert mel.num_frames(wave.numel()) == wave.numel() // 160
    assert mel.pad_or_trim(torch.ones(16_000 * 40)).shape[-1] == 480_000
    assert mel.pad_or_trim(torch.ones(16_000)).shape[-1] == 480_000
    batched = mel(torch.stack([mel.pad_or_trim(torch.randn(16_000)), mel.pad_or_trim(torch.randn(32_000))]))
    assert batched.shape == (2, 128, 3000)


def test_load_audio_file_decodes_and_resamples(tmp_path):
    sf = pytest.importorskip("soundfile")
    import numpy as np

    t = np.arange(8000) / 8000.0
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t)], axis=1).astype(np.float32)
    sf.write(tmp_path / "a.flac", stereo, 8000)
    wave = load_audio_file(str(tmp_path / "a.flac"), 16_000)
    assert wave.dtype == torch.float32 and wave.dim() == 1
    assert abs(wave.numel() - 16_000) <= 16  # 1 s, resampled 8 -> 16 kHz, mono
    model = _make_model()
    loaded = model.load_audio(str(tmp_path / "a.flac"), "cpu")
    assert loaded.metadata["sample_rate"] == 16_000 and loaded.data.shape == wave.shape


def test_sinusoid_positions_match_hf():
    modeling = pytest.importorskip("transformers.models.whisper.modeling_whisper")
    assert torch.allclose(sinusoid_positions(50, 64), modeling.sinusoids(50, 64), atol=1e-6)


# --------------------------------------------------------------------------
# native encoder vs HF
# --------------------------------------------------------------------------


def _hf_encoder(cfg: WhisperModelConfig):
    transformers = pytest.importorskip("transformers")
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    hf_cfg = transformers.WhisperConfig(
        d_model=cfg.d_model, encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads, encoder_ffn_dim=cfg.encoder_ffn_dim,
        decoder_layers=cfg.decoder_layers, decoder_attention_heads=cfg.decoder_attention_heads,
        decoder_ffn_dim=cfg.decoder_ffn_dim, num_mel_bins=cfg.num_mel_bins,
        max_source_positions=cfg.max_source_positions, vocab_size=cfg.vocab_size,
    )
    torch.manual_seed(0)
    return WhisperEncoder._from_config(hf_cfg, attn_implementation="eager").eval()


def test_native_encoder_matches_hf_and_loads_completely():
    cfg = _tiny_config()
    hf = _hf_encoder(cfg)
    ours = WhisperEncoderModel(cfg)
    loaded = ours.load_weights(list(hf.state_dict().items()))
    assert set(loaded) == set(dict(ours.named_parameters()))
    # the fused K-bias slice the checkpoint doesn't carry is zero
    d = cfg.d_model
    assert torch.equal(ours.layers[0].self_attn.qkv_proj.bias[d:2 * d], torch.zeros(d))
    torch.manual_seed(1)
    feats = torch.randn(3, cfg.num_mel_bins, cfg.max_source_positions * 2)
    with torch.no_grad():
        expected = hf(feats).last_hidden_state
        actual = ours(feats)
    assert actual.shape == (3, cfg.max_source_positions, cfg.d_model)
    assert torch.allclose(actual, expected, atol=1e-4), (actual - expected).abs().max()


def test_native_encoder_rejects_wrong_window_and_missing_weights():
    cfg = _tiny_config()
    ours = WhisperEncoderModel(cfg)
    with pytest.raises(ValueError, match="mel frames"):
        ours(torch.zeros(1, cfg.num_mel_bins, 10))
    hf = _hf_encoder(cfg)
    partial = [(k, v) for k, v in hf.state_dict().items() if "layers.1" not in k]
    with pytest.raises(RuntimeError, match="unloaded"):
        WhisperEncoderModel(cfg).load_weights(partial)


# --------------------------------------------------------------------------
# encoder submodule: batching + capture config
# --------------------------------------------------------------------------


def _engine_inputs(rids):
    return ModelInputsFromEngine(request_ids=list(rids), per_request_info={})


def test_encoder_submodule_batches_and_declares_one_capture_per_batch_size():
    cfg = _tiny_config()
    encoder = WhisperEncoderModel(cfg)
    encoder.load_weights(list(_hf_encoder(cfg).state_dict().items()))
    sub = WhisperEncoderSubmodule(encoder, cfg).eval()
    (capture,) = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert capture.capture_graph_walk == PREFILL_WALK
    assert set(capture.replay_graph_walks) == {PREFILL_WALK, DETECT_LANGUAGE_WALK}
    assert capture.single_request_inputs.tensor_inputs["audio_features"].shape == (cfg.num_mel_bins, cfg.num_frames)
    assert capture.capture_batch_sizes == sub.ENCODER_CAPTURE_BATCH_SIZES
    assert capture.get_total_tokens(8) == [8]
    assert sub.declare_step(PREFILL_WALK, ["r0"], []) is None  # no resources
    assert sub.can_batch(None, [])

    frames = cfg.max_source_positions * 2
    rows = [sub.prepare_inputs(PREFILL_WALK, None, {"audio_features": [torch.randn(cfg.num_mel_bins, frames)]})
            for _ in range(3)]
    assert all(r.input_seq_len == 1 for r in rows)
    batch = sub.preprocess(PREFILL_WALK, _engine_inputs(["a", "b", "c"]), rows)
    assert batch["audio_features"].shape == (3, cfg.num_mel_bins, frames)
    with torch.no_grad():
        out = sub.forward_batched(PREFILL_WALK, _engine_inputs(["a", "b", "c"]), **batch)
        single = sub.forward(PREFILL_WALK, _engine_inputs(["a"]), audio_features=batch["audio_features"][:1])
    assert set(out) == {"a", "b", "c"}
    assert out["b"]["encoder_states"][0].shape == (cfg.max_source_positions, cfg.d_model)
    assert torch.allclose(out["a"]["encoder_states"][0], single["encoder_states"][0], atol=1e-5)


# --------------------------------------------------------------------------
# decoder submodule: step declaration
# --------------------------------------------------------------------------


def _decoder_submodule(cfg: WhisperModelConfig) -> WhisperDecoderSubmodule:
    sub = object.__new__(WhisperDecoderSubmodule)
    torch.nn.Module.__init__(sub)
    sub.request_states = {}
    sub.node_resources = {}
    sub.config = cfg
    sub._suppress_ids = sub._begin_suppress_ids = sub._language_mask = None
    return sub


def test_decoder_declares_context_span_only_on_encoder_walks():
    cfg = _tiny_config()
    sub = _decoder_submodule(cfg)
    prefill_row = ARNodeInputs(
        input_seq_len=4, input_ids=torch.tensor([SOT, EN, TRANSCRIBE, NOTS]),
        tensor_inputs={"encoder_states": torch.zeros(cfg.max_source_positions, cfg.d_model)},
    )
    decode_row = ARNodeInputs(input_seq_len=1, input_ids=torch.tensor([7]))
    step = sub.declare_step(PREFILL_WALK, ["p", "d"], [prefill_row, decode_row])
    assert [(s.label, s.span) for s in step.segments] == [("main", 4), ("main", 1)]
    ctx = step.steps[CROSS_KV_CACHE].segments
    assert [(s.request_id, s.label, s.span) for s in ctx] == [
        ("p", CONTEXT_LABEL, cfg.max_source_positions), ("d", CONTEXT_LABEL, 0),
    ]
    assert step.steps[CROSS_ATTN].causal is False and step.steps[KV_CACHE] is not None


def test_decoder_timestamp_rules_ride_along_as_a_staged_row():
    cfg = _tiny_config()
    sub = _decoder_submodule(cfg)
    sub.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
    from mstar.model.whisper.components.timestamps import FIRST_TOKEN, TimestampRules, inactive_state

    sub.timestamp_rules = TimestampRules(cfg)
    fwd = type("F", (), {"request_id": "ts"})()
    ctx = {"encoder_states": [torch.zeros(cfg.max_source_positions, cfg.d_model)]}
    # a prompt without <|notimestamps|> turns the rules on; the first token must be a timestamp
    row = sub.prepare_inputs(PREFILL_WALK, fwd, {"text_inputs": [torch.tensor([SOT, EN, TRANSCRIBE])], **ctx})
    assert sub.request_state("ts")["timestamps"] is True
    assert row.tensor_inputs["ts_rules"].tolist() == [1, FIRST_TOKEN, cfg.timestamp_begin - 1, cfg.timestamp_begin + 51]
    # a plain prompt: inactive row
    plain = type("F", (), {"request_id": "plain"})()
    row2 = sub.prepare_inputs(PREFILL_WALK, plain, {"text_inputs": [torch.tensor([SOT, EN, TRANSCRIBE, NOTS])], **ctx})
    assert row2.tensor_inputs["ts_rules"].tolist() == inactive_state()
    batch = sub.preprocess(PREFILL_WALK, _engine_inputs(["ts", "plain"]), [row, row2])
    assert batch["ts_rules"].shape == (2, 4)

    # a decode step takes the state routed with its token, never a host history
    routed = torch.tensor([1, 2, cfg.timestamp_begin + 2, cfg.vocab_size])
    decode_row = sub.prepare_inputs(
        DECODE_WALK, fwd, {"text_inputs": [torch.tensor([cfg.timestamp_begin + 2])], "ts_rules": [routed]},
    )
    assert torch.equal(decode_row.tensor_inputs["ts_rules"], routed)
    bare = sub.prepare_inputs(DECODE_WALK, fwd, {"text_inputs": [torch.tensor([7])]})
    assert bare.tensor_inputs["ts_rules"].tolist() == inactive_state()
    # postprocess keeps the state under its own name for the loop-back edge
    outputs = {"new_token": [torch.tensor([5])], "ts_rules": [routed]}
    sub.postprocess("ts", None, outputs)
    assert outputs["text_inputs"] is outputs["new_token"] and outputs["ts_rules"] == [routed]
    # the capture template carries the same row so the graph's input set is fixed
    (decode_cfg,) = sub.get_cuda_graph_configs(torch.device("cpu"))
    assert decode_cfg.single_request_inputs.tensor_inputs["ts_rules"].tolist() == inactive_state()
    # language detection never applies the rules
    detect = sub.prepare_inputs(DETECT_LANGUAGE_WALK, type("F", (), {"request_id": "d"})(),
                                {"text_inputs": [torch.tensor([SOT])], **ctx})
    assert detect.tensor_inputs["ts_rules"].tolist() == inactive_state()


def test_decoder_prefill_prompt_appends_the_tail_to_the_detected_language():
    cfg = _tiny_config()
    sub = _decoder_submodule(cfg)
    fwd = type("F", (), {"request_id": "r0"})()
    # prepare_inputs reads the module device; give it a parameter to live on
    sub.register_parameter("anchor", torch.nn.Parameter(torch.zeros(1)))
    row = sub.prepare_inputs(
        PREFILL_PROMPT_WALK, fwd,
        {"text_inputs": [torch.tensor([DE])], "prompt_tail": [torch.tensor([TRANSCRIBE, NOTS])]},
    )
    assert row.input_seq_len == 3
    batch = sub.preprocess(PREFILL_PROMPT_WALK, _engine_inputs(["r0"]), [row])
    assert batch["input_ids"].tolist() == [DE, TRANSCRIBE, NOTS] and "encoder_states" not in batch
    assert sub.request_state("r0")["prompt_len"] == 3

    detect_row = sub.prepare_inputs(
        DETECT_LANGUAGE_WALK, fwd,
        {"text_inputs": [torch.tensor([SOT])],
         "encoder_states": [torch.zeros(cfg.max_source_positions, cfg.d_model)]},
    )
    assert detect_row.input_seq_len == 1 and sub._context_span(detect_row) == cfg.max_source_positions
    assert sub.max_batch_size(DECODE_WALK) is None
    assert sub.max_batch_size(PREFILL_WALK) == sub.MAX_PREFILL_BATCH_SIZE


def test_decoder_logit_rules():
    cfg = _tiny_config()
    sub = _decoder_submodule(cfg)
    logits = torch.zeros(2, cfg.vocab_size)
    restricted = sub._restrict_to_languages(logits.clone())
    finite = torch.isfinite(restricted[0]).nonzero().flatten().tolist()
    assert finite == sorted([EN, DE])
    suppressed = sub._apply_suppress(logits.clone(), is_first_token=True)
    assert torch.isinf(suppressed[0, [1, 2, 220, EOT]]).all()
    later = sub._apply_suppress(logits.clone(), is_first_token=False)
    assert torch.isinf(later[0, [1, 2]]).all() and torch.isfinite(later[0, EOT])


def test_decoder_check_stop_honors_eos_max_tokens_and_position_table():
    cfg = _tiny_config()
    sub = _decoder_submodule(cfg)

    class _Cfg:
        ignore_eos = False

    def info(walk, decoded, max_tokens=444):
        return type("I", (), {
            "graph_walk": walk, "max_tokens": max_tokens,
            "resource_configs": {"sampler": _Cfg()},
            "dynamic_loop_iter_counts": {"decode_loop": decoded},
        })()

    sub.request_state("r").add("prompt_len", 4)
    eos = {"new_token": [torch.tensor([EOT])]}
    word = {"new_token": [torch.tensor([1234])]}
    assert sub.check_stop("r", info(PREFILL_WALK, 0), eos) == set()
    assert sub.check_stop("r", info(DECODE_WALK, 0), eos) == {"decode_loop"}
    assert sub.check_stop("r", info(DECODE_WALK, 0), word) == set()
    assert sub.check_stop("r", info(DECODE_WALK, 3, max_tokens=4), word) == {"decode_loop"}
    # 4 prompt + 1 first token + 443 decoded fill the 448-slot table
    assert sub.check_stop("r", info(DECODE_WALK, 441), word) == set()
    assert sub.check_stop("r", info(DECODE_WALK, 442), word) == {"decode_loop"}


# --------------------------------------------------------------------------
# model: graph, state machine, prompt processing, output rendering
# --------------------------------------------------------------------------


def test_graph_walks_and_resources():
    model = _make_model()
    walks = model.get_graph_walk_graphs()
    assert set(walks) == {PREFILL_WALK, DETECT_LANGUAGE_WALK, PREFILL_PROMPT_WALK, DECODE_WALK}
    loop_node = walks[DECODE_WALK].section
    assert set(loop_node.input_names) == {"text_inputs", "ts_rules"}
    assert {e.name for e in loop_node.outputs if e.next_node == DECODER_NODE} == {"text_inputs", "ts_rules"}
    assert model.nodes == [ENCODER_NODE, DECODER_NODE]
    specs = model.get_node_resources()
    kv = {s.resource_key: s for s in specs if isinstance(s, KVSpec)}
    assert set(kv) == {KV_CACHE, CROSS_KV_CACHE}
    assert all(s.nodes == {DECODER_NODE} for s in specs)
    per_req_ctx = -(-model.config.max_source_positions // 128)
    assert kv[CROSS_KV_CACHE].config.max_num_pages == per_req_ctx * model.MAX_CONCURRENT_REQUESTS
    assert kv[KV_CACHE].config.max_num_pages == 4 * model.MAX_CONCURRENT_REQUESTS + 2 * model.MAX_CONCURRENT_REQUESTS
    assert model.get_max_output_tokens() == 444
    assert model.get_max_output_tokens(max_output_tokens=32) == 32


def _signals(names):
    return {n: [object()] for n in names}


def test_state_machine_forced_language():
    model = _make_model()
    args = model.get_initial_forward_pass_args(
        "default", ["audio", "text"], ["text"], _signals(["audio_features", "text_inputs"]),
    )
    assert args.full_metadata.graph_walk == PREFILL_WALK and args.full_metadata.is_prefill
    assert [(e.next_node, e.name) for e in args.inputs] == [
        (ENCODER_NODE, "audio_features"), (DECODER_NODE, "text_inputs"),
    ]
    assert len(args.unpersist_tensors) == 2

    rules = object()
    nxt = model.get_partition_forward_pass_args(
        "default", args.full_metadata, {"new_token": [object()], "ts_rules": [rules]},
    )
    assert nxt.full_metadata.graph_walk == DECODE_WALK and not nxt.full_metadata.is_prefill
    assert [(e.next_node, e.name) for e in nxt.inputs] == [(DECODER_NODE, "text_inputs"), (DECODER_NODE, "ts_rules")]
    assert nxt.inputs[1].tensor_info == [rules] and len(nxt.unpersist_tensors) == 2
    done = model.get_partition_forward_pass_args("default", nxt.full_metadata, {})
    assert done.request_done


def test_state_machine_language_detection():
    model = _make_model()
    tail = object()
    args = model.get_initial_forward_pass_args(
        "default", ["audio", "text"], ["text"],
        {"audio_features": [object()], "text_inputs": [object()], "prompt_tail": [tail]},
    )
    assert args.full_metadata.graph_walk == DETECT_LANGUAGE_WALK
    # the tail is held for the second prefill, not dropped after the first
    assert tail not in args.unpersist_tensors and len(args.unpersist_tensors) == 2

    lang, stale_rules = object(), object()
    second = model.get_partition_forward_pass_args(
        "default", args.full_metadata, {"new_token": [lang], "ts_rules": [stale_rules]},
    )
    assert second.full_metadata.graph_walk == PREFILL_PROMPT_WALK and second.full_metadata.is_prefill
    assert [(e.next_node, e.name, e.tensor_info) for e in second.inputs] == [
        (DECODER_NODE, "text_inputs", [lang]), (DECODER_NODE, "prompt_tail", [tail]),
    ]
    # the detection step's (inactive) rule state is dropped, not routed
    assert set(map(id, second.unpersist_tensors)) == {id(lang), id(tail), id(stale_rules)}
    assert second.step_metadata == {"is_prefill": True}

    third = model.get_partition_forward_pass_args("default", second.full_metadata, {"new_token": [object()]})
    assert third.full_metadata.graph_walk == DECODE_WALK and not third.full_metadata.is_prefill
    assert model.get_partition_forward_pass_args("default", third.full_metadata, {}).request_done


def test_process_prompt_builds_window_and_prompts():
    model = _make_model()
    wave = torch.randn(16_000 * 2)
    out = model.process_prompt(None, ["audio", "text"], ["text"], {"audio_inputs": [wave]}, language="en")
    assert out["audio_features"][0].shape == (model.config.num_mel_bins, 3000)
    assert out["text_inputs"][0].tolist() == [SOT, EN, TRANSCRIBE, NOTS]
    assert "prompt_tail" not in out

    detect = model.process_prompt(None, ["audio", "text"], ["text"], {"audio_inputs": [wave]}, timestamps=True)
    assert detect["text_inputs"][0].tolist() == [SOT]
    assert detect["prompt_tail"][0].tolist() == [TRANSCRIBE]

    prev = model.process_prompt(
        None, ["audio", "text"], ["text"], {"audio_inputs": [wave]}, language="de", initial_prompt="Hallo Welt",
    )
    ids = prev["text_inputs"][0].tolist()
    assert ids[0] == PREV and ids[-4:] == [SOT, DE, TRANSCRIBE, NOTS]
    assert ids[1:-4] == model.tokenizer.encode(" Hallo Welt")

    long = torch.randn(16_000 * 45)  # beyond one window: trimmed, not rejected
    assert model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [long]}, language="en")[
        "audio_features"][0].shape[-1] == 3000
    with pytest.raises(ValueError, match="exactly one audio"):
        model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": []}, language="en")
    with pytest.raises(ValueError, match="empty audio"):
        model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [torch.zeros(0)]}, language="en")


def test_postprocess_renders_language_and_timestamps_only():
    model = _make_model()
    cfg = model.config
    ids = [SOT, EN, TRANSCRIBE, NOTS, cfg.timestamp_begin, 1000, 1001, cfg.timestamp_begin + 75, EOT]
    text = model.postprocess(torch.tensor(ids), "text").decode("utf-8")
    assert text.startswith("<|en|><|0.00|>") and text.endswith("<|1.50|>")
    assert "<|startoftranscript|>" not in text and "<|notimestamps|>" not in text and "endoftext" not in text
    assert " " in text  # the byte-level 'Ġ' became a space
    with pytest.raises(ValueError):
        model.postprocess(torch.tensor([1]), "audio")


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,repo,yaml_name", [
    ("whisper_large", "openai/whisper-large-v3", "whisper_large.yaml"),
    ("whisper_large_v3_turbo", "openai/whisper-large-v3-turbo", "whisper_large_v3_turbo.yaml"),
])
def test_registry_cli_config_and_benchmark_entries_agree(key, repo, yaml_name):
    from benchmark.base import ModelType, RequestType
    from mstar.cli.main import DEFAULT_CONFIGS, _next_steps

    assert get_model_class(key) is WhisperModel
    assert HF_MODELS[key] == {"model_path_hf": repo}
    assert DEFAULT_CONFIGS[key] == yaml_name
    serving = yaml.safe_load((REPO / "configs" / yaml_name).read_text())
    assert serving["model"] == key
    assert serving["node_groups"][0]["node_names"] == [ENCODER_NODE, DECODER_NODE]
    assert set(serving["resources"]) <= {KV_CACHE, CROSS_KV_CACHE}
    bench = ModelType(key).inst()
    assert bench.get_hf_url() == repo and bench.get_supported_modalities() == {RequestType.A2T}
    assert "transcribe(" in _next_steps(key, "0.0.0.0", 8000)
    assert "audio.transcriptions" in _next_steps(key, "0.0.0.0", 8000)


def test_node_inputs_clone_keeps_encoder_template_shape():
    template = NodeInputs(tensor_inputs={"audio_features": torch.zeros(4, 6)}, input_seq_len=1)
    clone = template.clone()
    assert clone.tensor_inputs["audio_features"].shape == (4, 6) and clone.input_seq_len == 1
    assert clone.tensor_inputs["audio_features"] is not template.tensor_inputs["audio_features"]
