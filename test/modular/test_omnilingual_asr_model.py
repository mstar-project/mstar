"""Omnilingual ASR CTC on CPU, with tiny random weights and no downloads.

Covers the waveform preprocessing (resampling, downmix, normalization, the
40 s cap), the tokenizer and CTC decoding, the checkpoint name mapping and
weight-norm fold, the encoder's padding masks, the model contract (graph,
process_prompt, postprocess), the submodule's batched forward, and the
transcription route end to end through the adapter with the model in-process.
"""

import io
import math
import struct
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio
import yaml

from mstar.model.omnilingual_asr import audio
from mstar.model.omnilingual_asr.components.wav2vec2 import Wav2Vec2CTC, attention_mask
from mstar.model.omnilingual_asr.config import (
    CDN_CHECKPOINTS,
    ENCODER_NODE,
    HUB_CHECKPOINTS,
    MAX_AUDIO_SECONDS,
    TOKEN_IDS,
    TRANSCRIBE_WALK,
    WAVEFORM,
    OmniASRCTCConfig,
    ctc_config,
)
from mstar.model.omnilingual_asr.tokenizer import (
    UNK_SURFACE,
    CharTokenizer,
    ctc_collapse,
    ctc_greedy_ids,
    read_sentencepiece_pieces,
)
from mstar.model.omnilingual_asr.weight_loader import convert_state_dict, load_into, read_checkpoint

REPO = Path(__file__).resolve().parents[2]
KEYS = ["omnilingual_asr_ctc_300m", "omnilingual_asr_ctc_1b", "omnilingual_asr_ctc_3b", "omnilingual_asr_ctc_7b"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _tiny_config(vocab_size: int = 12) -> OmniASRCTCConfig:
    # hop 20 samples; 3 convs; a 4-group positional conv of kernel 8
    return OmniASRCTCConfig(
        model_dim=32,
        num_layers=2,
        ffn_inner_dim=64,
        vocab_size=vocab_size,
        num_heads=4,
        feature_dim=16,
        feature_extractor_layer_descs=[(16, 10, 5), (16, 3, 2), (16, 2, 2)],
        pos_conv_kernel_size=8,
        num_pos_conv_groups=4,
    )


def _random_model(cfg: OmniASRCTCConfig, seed: int = 0) -> Wav2Vec2CTC:
    torch.manual_seed(seed)
    model = Wav2Vec2CTC(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.2)
    return model


def _fairseq2_state(model: Wav2Vec2CTC) -> dict[str, torch.Tensor]:
    """``model``'s parameters under the checkpoint's names: the positional conv
    split into weight_g / weight_v as fairseq2's weight_norm(dim=2) stores it."""
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    w = state.pop("encoder_frontend.pos_encoder.conv.weight")
    g = torch.linalg.vector_norm(w, dim=(0, 1), keepdim=True)  # (1, 1, K)
    state["encoder_frontend.pos_encoder.conv.weight_g"] = g * 1.5
    state["encoder_frontend.pos_encoder.conv.weight_v"] = w
    return state


def _spm_bytes(pieces: list[tuple[str, int]]) -> bytes:
    """A serialized SentencePiece ModelProto holding only ``pieces``."""

    def varint(n):
        out = bytearray()
        while True:
            b = n & 0x7F
            n >>= 7
            out.append(b | (0x80 if n else 0))
            if not n:
                return bytes(out)

    def field(num, wire, payload):
        return varint(num << 3 | wire) + payload

    body = b""
    for piece, ptype in pieces:
        raw = piece.encode("utf-8")
        msg = field(1, 2, varint(len(raw)) + raw) + field(2, 5, struct.pack("<f", 0.0)) + field(3, 0, varint(ptype))
        body += field(1, 2, varint(len(msg)) + msg)
    # an unrelated trainer_spec (field 2), as real files carry, to be skipped
    return body + field(2, 2, varint(3) + b"\x08\x01\x10")


def _vocab(size: int) -> list[tuple[str, int]]:
    pieces = [("<s>", 3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2), (" ", 1)]
    pieces += [(chr(0x61 + i) if i < 26 else chr(0x4E00 + i), 1) for i in range(size - len(pieces))]
    return pieces


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    samples = np.atleast_2d(samples.T).T  # (time, channels)
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------
# config, registry, deployment files
# --------------------------------------------------------------------------


def test_sizes_match_the_reference_archs():
    assert (ctc_config("300m", "v2").model_dim, ctc_config("300m", "v2").num_layers) == (1024, 24)
    assert (ctc_config("1b", "v2").model_dim, ctc_config("1b", "v2").num_layers) == (1280, 48)
    assert (ctc_config("3b", "v1").model_dim, ctc_config("3b", "v1").num_layers) == (2048, 60)
    assert ctc_config("7b", "v2").num_layers == 128
    assert ctc_config("300m", "v1").vocab_size == 9812 and ctc_config("300m", "v2").vocab_size == 10288
    cfg = ctc_config("1b", "v2")
    assert cfg.hop_samples == 320 and cfg.model_dim // cfg.num_heads == 80
    with pytest.raises(ValueError, match="size"):
        ctc_config("2b", "v2")
    with pytest.raises(ValueError, match="version"):
        ctc_config("300m", "v3")


def test_every_size_is_registered_served_and_configured():
    from mstar.api_server.openai.adapters import OmnilingualASRAdapter, get_adapter
    from mstar.cli.main import DEFAULT_CONFIGS, _next_steps
    from mstar.model.registry import HF_MODELS, MODEL_REGISTRY, get_model_class, model_init_kwargs

    for key in KEYS:
        size = key.rsplit("_", 1)[1]
        assert MODEL_REGISTRY[key][1] == "OmnilingualASRModel"
        assert model_init_kwargs(key)["size"] == size and HF_MODELS[key]["model_path_hf"] is None
        assert get_model_class(key).__name__ == "OmnilingualASRModel"
        assert isinstance(get_adapter(key), OmnilingualASRAdapter)
        cfg = yaml.safe_load((REPO / "configs" / DEFAULT_CONFIGS[key]).read_text())
        assert cfg["model"] == key
        assert cfg["node_groups"] == [{"node_names": [ENCODER_NODE], "ranks": [0], "graph_walks": [TRANSCRIBE_WALK]}]
        assert size in CDN_CHECKPOINTS and size in HUB_CHECKPOINTS
        help_text = _next_steps(key, "localhost", 8000)
        assert "audio.transcriptions.create" in help_text and 'audio="speech.wav"' in help_text


def test_pinned_artifacts_are_well_formed():
    for art in CDN_CHECKPOINTS.values():
        assert art.url.startswith("https://dl.fbaipublicfiles.com/mms/") and art.url.endswith(art.filename)
        assert len(art.sha256) == 64
    for hub in HUB_CHECKPOINTS.values():
        assert hub.repo_id.startswith("facebook/omniASR-CTC-") and len(hub.revision) == 40
        assert len(hub.sha256) == 64


def test_checkpoints_are_verified_cached_or_downloaded(tmp_path, monkeypatch):
    import hashlib

    import mstar.model.omnilingual_asr.weight_loader as wl
    from mstar.model.omnilingual_asr.config import Artifact

    good = b"checkpoint bytes"
    art = Artifact("https://example.invalid/x.pt", hashlib.sha256(good).hexdigest(), "x.pt")
    # a download goes through torch.hub with the pinned hash, then is verified
    calls = []

    def fake_download(url, dst, hash_prefix=None, progress=True):
        calls.append(hash_prefix)
        Path(dst).write_bytes(good)

    monkeypatch.setattr(wl.torch.hub, "download_url_to_file", fake_download)
    path = wl.fetch_cdn(art, tmp_path)
    assert path.read_bytes() == good and calls == [art.sha256]
    # a cached hit is not downloaded again, and a verified file is not hashed again
    hashed = []
    real = wl.sha256_of
    monkeypatch.setattr(wl, "sha256_of", lambda p: hashed.append(p) or real(p))
    assert wl.fetch_cdn(art, tmp_path) == path and calls == [art.sha256] and hashed == []
    # a file replaced in the cache is hashed again and refused
    path.write_bytes(b"tampered checkpoint bytes")
    with pytest.raises(ValueError, match="not the pinned"):
        wl.fetch_cdn(art, tmp_path)
    # a cached file that was never verified is checked too
    other = Artifact("https://example.invalid/y.pt", art.sha256, "y.pt")
    (tmp_path / "y.pt").write_bytes(b"not the pinned file")
    with pytest.raises(ValueError, match="not the pinned"):
        wl.fetch_cdn(other, tmp_path)
    # Hub files are verified against the pinned LFS hash
    hub_file = tmp_path / "hub.pt"
    hub_file.write_bytes(b"something else")
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: str(hub_file)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(ValueError, match="not the pinned"):
        wl.fetch_hub("facebook/omniASR-CTC-300M", "x.pt", "rev", art.sha256, tmp_path)
    hub_file.write_bytes(good)
    assert wl.fetch_hub("facebook/omniASR-CTC-300M", "x.pt", "rev", art.sha256, tmp_path) == hub_file.resolve()


def test_a_cache_without_room_for_the_marker_still_loads(tmp_path, monkeypatch, caplog):
    """A read-only cache (a mounted volume) cannot take the verified-hash
    marker: the file is still verified, and hashed again on the next load."""
    import hashlib

    import mstar.model.omnilingual_asr.weight_loader as wl

    good = b"checkpoint bytes"
    ckpt = tmp_path / "x.pt"
    ckpt.write_bytes(good)
    (tmp_path / "verified").write_bytes(b"")  # the marker directory cannot be made
    hashed = []
    real = wl.sha256_of
    monkeypatch.setattr(wl, "sha256_of", lambda p: hashed.append(p) or real(p))
    digest = hashlib.sha256(good).hexdigest()
    wl.verify_sha256(ckpt, digest, tmp_path)
    wl.verify_sha256(ckpt, digest, tmp_path)
    assert len(hashed) == 2 and "cannot record the verified hash" in caplog.text
    with pytest.raises(ValueError, match="not the pinned"):
        wl.verify_sha256(ckpt, "0" * 64, tmp_path)


# --------------------------------------------------------------------------
# waveform preprocessing
# --------------------------------------------------------------------------


def test_downmix_resamples_each_channel_then_averages_in_order():
    torch.manual_seed(0)
    stereo = torch.randn(4410, 2)
    out = audio.to_16k_mono(stereo, 44100)
    per_channel = torchaudio.functional.resample(stereo.T.contiguous(), 44100, 16000)
    expected = per_channel[0].clone()
    expected += per_channel[1]
    expected /= 2
    assert torch.equal(out, expected)
    mono = torch.randn(1600)
    assert torch.equal(audio.to_16k_mono(mono, 16000), mono)


def test_normalize_is_per_utterance_layer_norm():
    x = torch.randn(16000) * 3 + 2
    y = audio.normalize(x)
    assert y.dtype == torch.float32
    assert abs(float(y.mean())) < 1e-5 and abs(float(y.var(unbiased=False)) - 1) < 1e-3
    assert torch.equal(y, torch.nn.functional.layer_norm(x, x.shape))


def test_length_policy_rejects_long_and_tiny_clips_without_truncating():
    ok = audio.preprocess(torch.randn(int(MAX_AUDIO_SECONDS * 16000)))
    assert ok.shape == (640000,)
    with pytest.raises(ValueError, match="at most 40 s"):
        audio.preprocess(torch.randn(int(MAX_AUDIO_SECONDS * 16000) + 1))
    with pytest.raises(ValueError, match="at least 400"):
        audio.preprocess(torch.randn(399))
    with pytest.raises(ValueError, match="1-D"):
        audio.preprocess(torch.randn(2, 1600))


def test_wav_upload_decodes_at_its_rate_then_resamples(tmp_path, monkeypatch):
    # force the standard-library path (no soundfile / torchcodec)
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    rng = np.random.default_rng(0)
    left, right = rng.uniform(-0.5, 0.5, 4410), rng.uniform(-0.5, 0.5, 4410)
    path = tmp_path / "stereo.wav"
    _write_wav(path, np.stack([left, right], axis=1), 44100)
    decoded, rate = audio.decode_file(str(path))
    assert rate == 44100 and decoded.shape == (4410, 2)
    pcm = torch.from_numpy(np.round(np.stack([left, right], 1) * 32767).astype(np.float32) / 32768.0)
    assert torch.equal(decoded, pcm)
    assert torch.equal(audio.load_waveform(str(path)), audio.to_16k_mono(pcm, 44100))


@pytest.mark.parametrize(
    "rate", [8000, 11025, 16000, 22050, 24000, 32000, 44056, 44100, 47952, 48000, 48048, 88200, 96000, 176400, 192000]
)
def test_common_and_pull_down_sample_rates_resample(rate):
    samples = torch.zeros(rate // 100)
    assert audio.to_16k_mono(samples, rate).shape == (160,)


def test_a_header_rate_coprime_to_16k_is_refused_before_resampling(tmp_path, monkeypatch):
    """The header's rate is the client's: resampling from 24007 Hz builds a
    kernel of about 16000 x 24007 floats (6.6 GB peak) for a 48 KB file."""
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)

    def no_resample(*args, **kwargs):
        raise AssertionError("resampled from an unsupported rate")

    monkeypatch.setattr(torchaudio.functional, "resample", no_resample)
    # 655,325 Hz is under 2^24 taps only when the filter padding is left out
    for rate in (24007, 100003, 655325, 0, 10**9):
        path = tmp_path / f"r{rate}.wav"
        path.write_bytes(_riff(1, 1, rate, 16, b"\x00\x00" * 24000))
        with pytest.raises(ValueError, match="unsupported sample rate"):
            audio.load_waveform(str(path))


def _riff(fmt_tag: int, channels: int, rate: int, bits: int, payload: bytes, extensible: bool = False) -> bytes:
    block = channels * bits // 8
    if extensible:
        fmt = struct.pack("<HHIIHH", 0xFFFE, channels, rate, rate * block, block, bits)
        fmt += struct.pack("<HHI", 22, bits, 0) + struct.pack("<H", fmt_tag) + b"\x00" * 14
    else:
        fmt = struct.pack("<HHIIHH", fmt_tag, channels, rate, rate * block, block, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_wav_reader_takes_float_and_24_bit_files(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    x = np.array([0.0, 0.5, -0.25, 1.0, -1.0], dtype=np.float32)
    path = tmp_path / "f32.wav"
    path.write_bytes(_riff(3, 1, 16000, 32, x.astype("<f4").tobytes()))
    decoded, rate = audio.decode_file(str(path))
    assert rate == 16000 and torch.equal(decoded, torch.from_numpy(x))
    ints = np.array([0, 4194304, -2097152, 8388607, -8388608], dtype=np.int32)  # 24-bit
    raw = b"".join(int(v).to_bytes(4, "little", signed=True)[:3] for v in ints)
    path.write_bytes(_riff(1, 1, 8000, 24, raw, extensible=True))
    decoded, rate = audio.decode_file(str(path))
    assert rate == 8000 and torch.equal(decoded, torch.from_numpy(ints.astype(np.float32) / 8388608.0))
    path.write_bytes(b"not a wav file at all")
    with pytest.raises(ValueError, match="cannot decode"):
        audio.decode_file(str(path))


def test_malformed_wav_headers_are_value_errors_without_the_path(tmp_path, monkeypatch):
    """A ValueError is a 400 at the data worker; anything else (the old
    ZeroDivisionError on 0 channels) is a 500. The message reaches the client,
    so it must not name the server's upload path."""
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    cases = {
        "zero_channels": _riff(1, 0, 16000, 16, b"\x00\x00" * 800),
        "four_bit": _riff(1, 1, 16000, 4, b"\x00" * 800),
        "short_fmt": b"RIFF" + struct.pack("<I", 24) + b"WAVEfmt " + struct.pack("<I", 4) + b"\x01\x00\x01\x00"
        + b"data" + struct.pack("<I", 0),
    }
    for name, data in cases.items():
        path = tmp_path / f"{name}.wav"
        path.write_bytes(data)
        with pytest.raises(ValueError, match="cannot decode") as err:
            audio.load_waveform(str(path))
        assert str(tmp_path) not in str(err.value), err.value


def test_a_decoder_quoting_the_path_with_repr_does_not_leak_it(tmp_path, monkeypatch):
    """soundfile formats the path with repr(), so a tab or backslash in the
    upload name reaches the message escaped, not as the plain path."""
    def _read(path, **_):
        raise RuntimeError(f"Error opening {path!r}: Format not recognised.")

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(read=_read, SoundFile=_read))
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    for name in ("tab\there.mp3", "back\\slash.mp3", "caf\u00e9\nnew.mp3"):
        path = tmp_path / name
        path.write_bytes(b"not audio at all")
        with pytest.raises(ValueError, match="cannot decode") as err:
            audio.load_waveform(str(path))
        message = str(err.value)
        assert str(tmp_path) not in message, message
        assert repr(str(tmp_path))[1:-1] not in message, message
        assert "<upload>" in message


# --------------------------------------------------------------------------
# tokenizer and CTC decoding
# --------------------------------------------------------------------------


def test_sentencepiece_reader_and_reference_decode_rules(tmp_path):
    pieces = [("<s>", 3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2), (" ", 1), ("a", 1), ("b", 1), ("é", 1)]
    path = tmp_path / "tok.model"
    path.write_bytes(_spm_bytes(pieces))
    assert read_sentencepiece_pieces(path.read_bytes()) == pieces
    tok = CharTokenizer.from_file(path)
    assert tok.vocab_size == 8
    # controls render as nothing, <unk> as " ⁇ ", spaces kept as is (fairseq2's decoder)
    assert tok.decode([4, 5, 4]) == " a "
    assert tok.decode([0, 5, 1, 6, 2]) == "ab"
    assert tok.decode([5, 3, 7]) == "a" + UNK_SURFACE + "é"
    with pytest.raises(ValueError, match="unsupported"):
        CharTokenizer([("\u2581a", 1)])


def test_ctc_collapses_repeats_before_dropping_the_blank():
    blank, a, b = 0, 5, 6
    tok = CharTokenizer([("<s>", 3), ("<pad>", 3), ("</s>", 3), ("<unk>", 2), (" ", 1), ("a", 1), ("b", 1)])
    ids = torch.tensor([blank, a, a, blank, a, b, b, blank, blank])
    collapsed = ctc_collapse(ids)
    assert collapsed.tolist() == [blank, a, blank, a, b, blank]
    assert tok.decode(collapsed.tolist()) == "aab"
    # repeated letters with no blank between collapse to one
    assert tok.decode(ctc_collapse(torch.tensor([a, a, a])).tolist()) == "a"
    assert ctc_collapse(torch.tensor([], dtype=torch.long)).numel() == 0
    logits = torch.full((5, 7), -1.0)
    for t, i in enumerate([a, a, b, blank, b]):
        logits[t, i] = 1.0
    # frames past num_frames (padding) are ignored
    assert ctc_greedy_ids(logits, 3).tolist() == [a, b]


# --------------------------------------------------------------------------
# weights and the encoder
# --------------------------------------------------------------------------


def test_checkpoint_names_load_and_weight_norm_folds(tmp_path):
    cfg = _tiny_config()
    src = _random_model(cfg)
    state = _fairseq2_state(src)
    path = tmp_path / "ckpt.pt"
    torch.save({"model": state, "fs2": True}, path)
    loaded = read_checkpoint(path)
    with torch.device("meta"):
        model = Wav2Vec2CTC(cfg)
    load_into(model, convert_state_dict(loaded, torch.float32))
    # the fold is exactly what torch's weight_norm parametrization computes
    ref = torch.nn.Conv1d(32, 32, 8, padding=4, groups=4)
    torch.nn.utils.parametrizations.weight_norm(ref, dim=2)
    with torch.no_grad():
        ref.parametrizations.weight.original0.copy_(state["encoder_frontend.pos_encoder.conv.weight_g"])
        ref.parametrizations.weight.original1.copy_(state["encoder_frontend.pos_encoder.conv.weight_v"])
    assert torch.allclose(model.encoder_frontend.pos_encoder.conv.weight, ref.weight, atol=0, rtol=1e-6)
    x = torch.randn(1, 3000)
    assert torch.isfinite(model(x)[0]).all()
    bf16 = convert_state_dict(loaded, torch.bfloat16)
    assert all(t.dtype == torch.bfloat16 for t in bf16.values())


def test_load_rejects_mismatched_checkpoints():
    cfg = _tiny_config()
    state = convert_state_dict(_fairseq2_state(_random_model(cfg)), torch.float32)
    with torch.device("meta"):
        model = Wav2Vec2CTC(cfg)
    missing = dict(state)
    missing.pop("final_proj.bias")
    with pytest.raises(ValueError, match="missing"):
        load_into(model, missing)
    with pytest.raises(ValueError, match="unexpected"):
        load_into(model, {**state, "encoder.extra.weight": torch.zeros(1)})
    wrong = dict(state)
    wrong["final_proj.weight"] = torch.zeros(13, 32)
    with pytest.raises(ValueError, match="shape"):
        load_into(model, wrong)


def test_frame_lengths_follow_the_conv_arithmetic():
    model = _random_model(_tiny_config())
    for n in (50, 51, 69, 70, 1000, 1234):
        logits, frames = model(torch.randn(1, n))
        assert frames == [logits.shape[1]]
    with torch.device("meta"):
        real = Wav2Vec2CTC(ctc_config("300m", "v2"))
    # 1 s -> 49 frames, 40 s -> 1999; 400 samples is the shortest clip with a frame
    assert real.frame_lengths([16000, 640000, 400, 399]) == [49, 1999, 1, 0]


def test_padded_batch_matches_single_runs_and_needs_the_mask():
    model = _random_model(_tiny_config())
    torch.manual_seed(1)
    waves = [torch.randn(n) for n in (1200, 700, 333, 1200)]
    lens = [w.shape[0] for w in waves]
    batch = torch.zeros(4, 1200)
    for i, w in enumerate(waves):
        batch[i, : w.shape[0]] = w
    with torch.no_grad():
        logits, frames = model(batch, lens)
        unmasked, _ = model(batch, None)
        for i, w in enumerate(waves):
            single, (n,) = model(w[None])
            assert frames[i] == n
            assert torch.allclose(logits[i, :n], single[0], atol=1e-5)
            assert torch.equal(logits[i, :n].argmax(-1), single[0].argmax(-1))
    # without the lengths, the short rows see their padding
    assert not torch.allclose(unmasked[2, : frames[2]], logits[2, : frames[2]], atol=1e-3)


def test_attention_mask_keeps_padded_rows_finite():
    frame_mask = torch.tensor([[True, True, False]])
    m = attention_mask(frame_mask)[0, 0]
    assert m.tolist() == [[True, True, False], [True, True, False], [True, True, True]]


def test_attention_never_runs_on_cudnn(monkeypatch):
    """cuDNN attention builds a plan for every new sequence length (about
    10x slower per forward when lengths vary, as uploads do), so the encoder
    keeps SDPA to the flash, memory-efficient and math kernels."""
    import mstar.model.omnilingual_asr.components.wav2vec2 as w2v

    seen = []
    real = w2v.F.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append((torch.backends.cuda.cudnn_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(),
                     torch.backends.cuda.mem_efficient_sdp_enabled()))
        return real(*args, **kwargs)

    monkeypatch.setattr(w2v.F, "scaled_dot_product_attention", spy)
    model = _random_model(_tiny_config())
    with torch.no_grad():
        model(torch.randn(2, 400), [400, 300])
        model(torch.randn(1, 333))
    assert seen and all(s == (False, True, True) for s in seen), seen


# --------------------------------------------------------------------------
# the model contract and the node
# --------------------------------------------------------------------------


@pytest.fixture
def tiny_model(tmp_path):
    """An OmnilingualASRModel over a tiny random network with the real v2
    vocabulary size, loaded from a fairseq2-named checkpoint on disk."""
    from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel

    vocab = 10288
    tok_path = tmp_path / "tok.model"
    tok_path.write_bytes(_spm_bytes(_vocab(vocab)))
    cfg = _tiny_config(vocab)
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": _fairseq2_state(_random_model(cfg)), "fs2": True}, ckpt)
    model = OmnilingualASRModel(
        size="300m",
        checkpoint_path=str(ckpt),
        tokenizer_path=str(tok_path),
        dtype="float32",
        max_batch_size=4,
        max_batch_seconds=80,
    )
    model.config = cfg
    return model


def test_constructor_validates_options(tmp_path):
    from mstar.model.omnilingual_asr.omnilingual_asr_model import OmnilingualASRModel

    tok = tmp_path / "tok.model"
    tok.write_bytes(_spm_bytes(_vocab(10288)))
    with pytest.raises(ValueError, match="Unknown Omnilingual ASR option"):
        OmnilingualASRModel(tokenizer_path=str(tok), sise="1b")
    with pytest.raises(ValueError, match="size"):
        OmnilingualASRModel(tokenizer_path=str(tok), size="2b")
    with pytest.raises(ValueError, match="weights_version"):
        OmnilingualASRModel(tokenizer_path=str(tok), weights_version="v3")
    with pytest.raises(ValueError, match="dtype"):
        OmnilingualASRModel(tokenizer_path=str(tok), dtype="int8")
    # the v2 tokenizer does not belong to v1 weights
    with pytest.raises(ValueError, match="does not belong"):
        OmnilingualASRModel(tokenizer_path=str(tok), weights_version="v1")


def test_model_hooks_are_not_shadowed(tiny_model):
    # the engine calls these on every model; an attribute of the same name breaks worker start-up
    from mstar.model.base import Model

    for name, member in vars(Model).items():
        if callable(member) and not name.startswith("__"):
            assert callable(getattr(tiny_model, name)), name
    assert tiny_model.checkpoint_path() is None


def test_graph_is_one_stateless_node(tiny_model):
    assert tiny_model.get_node_resources() == []
    walks = tiny_model.get_graph_walk_graphs()
    assert list(walks) == [TRANSCRIBE_WALK]
    node = walks[TRANSCRIBE_WALK]
    assert node.name == ENCODER_NODE and node.input_names == {WAVEFORM}
    assert [(e.name, e.output_modality) for e in node.outputs] == [(TOKEN_IDS, "text")]
    fwd = tiny_model.get_initial_forward_pass_args("default", ["audio"], ["text"], {WAVEFORM: []})
    assert fwd.full_metadata.graph_walk == TRANSCRIBE_WALK and not fwd.request_done
    assert [e.name for e in fwd.inputs] == [WAVEFORM]
    done = tiny_model.get_partition_forward_pass_args("default", fwd.full_metadata, {})
    assert done.request_done
    assert tiny_model.get_autocast_dtype() == torch.float32


def test_process_prompt_normalizes_and_validates(tiny_model):
    wave_in = torch.randn(16000) * 0.1 + 0.05
    out = tiny_model.process_prompt("ignored", ["audio"], ["text"], {"audio_inputs": [wave_in]}, language="en_US")
    assert torch.equal(out[WAVEFORM][0], audio.normalize(wave_in))
    with pytest.raises(ValueError, match="exactly one audio"):
        tiny_model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": []})
    with pytest.raises(ValueError, match="exactly one audio"):
        tiny_model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [wave_in, wave_in]})
    with pytest.raises(ValueError, match="at most 40 s"):
        tiny_model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [torch.zeros(41 * 16000)]})
    with pytest.raises(ValueError, match="produces text"):
        tiny_model.process_prompt(None, ["audio"], ["audio"], {"audio_inputs": [wave_in]})


def test_postprocess_drops_blank_and_specials(tiny_model):
    ids = torch.tensor([0, 5, 1, 4, 6, 2, 3])
    pieces = _vocab(10288)
    expected = pieces[5][0] + " " + pieces[6][0] + UNK_SURFACE
    assert tiny_model.postprocess(ids, "text") == expected.encode("utf-8")
    # the node emits one id per frame; repeats collapse here, a blank splits a real double
    frames = torch.tensor([5, 5, 0, 0, 5, 4, 4, 6, 6, 6])
    double = pieces[5][0] * 2 + " " + pieces[6][0]
    assert tiny_model.postprocess(frames, "text") == double.encode("utf-8")
    with pytest.raises(ValueError, match="modality"):
        tiny_model.postprocess(ids, "audio")


def _engine_inputs(rids):
    return types.SimpleNamespace(request_ids=list(rids))


def _run_node(sub, waveforms: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    rids = [f"r{i}" for i in range(len(waveforms))]
    inputs = [
        sub.prepare_inputs(TRANSCRIBE_WALK, types.SimpleNamespace(request_id=r), {WAVEFORM: [w]})
        for r, w in zip(rids, waveforms, strict=True)
    ]
    pre = sub.preprocess(TRANSCRIBE_WALK, _engine_inputs(rids), inputs)
    with torch.no_grad():
        if len(inputs) == 1:
            return {rids[0]: sub.forward(TRANSCRIBE_WALK, _engine_inputs(rids), **pre)[TOKEN_IDS][0]}
        out = sub.forward_batched(TRANSCRIBE_WALK, _engine_inputs(rids), **pre)
    return {r: o[TOKEN_IDS][0] for r, o in out.items()}


def test_node_batches_and_matches_single_requests(tiny_model):
    sub = tiny_model.get_submodule(ENCODER_NODE, "cpu")
    assert sub is tiny_model.get_submodule(ENCODER_NODE, "cpu")
    assert sub.disable_autocast and sub.disable_torch_compile
    torch.manual_seed(3)
    waves = [audio.preprocess(torch.randn(n)) for n in (16000, 9000, 4000)]
    batched = _run_node(sub, waves)
    for i, w in enumerate(waves):
        single = _run_node(sub, [w])["r0"]
        assert torch.equal(batched[f"r{i}"], single)
        # one id per valid frame, blanks included: never an empty emit
        assert single.numel() == sub.model.frame_lengths([w.shape[0]])[0] >= 1
    # any set batches; the step splits it, shortest first, to fit 80 padded seconds
    nodes = [types.SimpleNamespace(input_seq_len=n) for n in (16000, 16000)]
    assert sub.can_batch(None, nodes) and not sub.can_batch(None, nodes[:1])
    assert sub.sub_batches([640000, 16000, 640000, 16000, 640000]) == [[1, 3], [0, 2], [4]]
    assert sub.sub_batches([16000] * 4) == [[0, 1, 2, 3]]
    # a split step gives the same ids as each request alone (a 0.5 s budget: 8000 samples)
    from mstar.model.omnilingual_asr.submodules import OmniASRCTCSubmodule

    small = OmniASRCTCSubmodule(sub.model, max_batch_size=4, max_batch_seconds=0.5)
    lengths = (6000, 1000, 5000, 2000)
    assert small.sub_batches(list(lengths)) == [[1, 3], [2], [0]]
    short_waves = [audio.preprocess(torch.randn(n)) for n in lengths]
    split = _run_node(small, short_waves)
    for i, w in enumerate(short_waves):
        assert torch.equal(split[f"r{i}"], _run_node(sub, [w])["r0"])
    assert sub.max_batch_size(TRANSCRIBE_WALK) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_node_step_never_waits_on_the_gpu(tiny_model):
    """A padded step cut into sub-batches launches without a host sync."""
    sub = tiny_model.get_submodule(ENCODER_NODE, "cuda")
    small = type(sub)(sub.model, max_batch_size=4, max_batch_seconds=0.5)
    torch.manual_seed(5)
    waves = [audio.preprocess(torch.randn(n)).cuda() for n in (6000, 1000, 5000, 2000)]
    rids = [f"r{i}" for i in range(len(waves))]
    inputs = [
        small.prepare_inputs(TRANSCRIBE_WALK, types.SimpleNamespace(request_id=r), {WAVEFORM: [w]})
        for r, w in zip(rids, waves, strict=True)
    ]
    pre = small.preprocess(TRANSCRIBE_WALK, _engine_inputs(rids), inputs)
    with torch.no_grad():
        warm = small.forward_batched(TRANSCRIBE_WALK, _engine_inputs(rids), **pre)  # warms the pinned pool
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = small.forward_batched(TRANSCRIBE_WALK, _engine_inputs(rids), **pre)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    for r, w in zip(rids, waves, strict=True):
        ids = out[r][TOKEN_IDS][0]
        assert ids.is_cuda and ids.numel() == small.model.frame_lengths([w.shape[0]])[0]
        assert torch.equal(ids, warm[r][TOKEN_IDS][0])


def test_load_audio_returns_16k_mono(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    path = tmp_path / "a.wav"
    _write_wav(path, np.random.default_rng(1).uniform(-0.3, 0.3, (22050, 2)), 22050)
    out = tiny_model.load_audio(str(path), "cpu")
    assert out.data.shape == (16000,) and out.metadata == {"sample_rate": 16000, "num_channels": 1}


# --------------------------------------------------------------------------
# adapter and the transcription route, with the model in-process
# --------------------------------------------------------------------------


def test_adapter_maps_the_request():
    pytest.importorskip("pydantic")
    from mstar.api_server.openai.adapters import OmnilingualASRAdapter
    from mstar.api_server.openai.protocol import TranscriptionRequest

    ad = OmnilingualASRAdapter()
    assert ad.supports_transcriptions and not ad.supports_realtime_transcription
    assert ad.max_audio_seconds == MAX_AUDIO_SECONDS and ad.long_form == "parallel"
    assert ad.window_seconds == 16.0 and ad.whole_seconds == 20.0 and ad.balanced_windows
    assert (ad.window_search_seconds, ad.window_frame_seconds) == (5.0, 0.1)
    assert ad.compression_ratio_threshold is None and not ad.seeks_by_timestamps
    req = TranscriptionRequest(model=KEYS[0], language="sw", prompt="habari", temperature=0.0)
    sa = ad.transcription_to_request(req, "/tmp/a.wav")
    assert sa.file_paths == {"audio": ["/tmp/a.wav"]} and sa.text == ""
    assert sa.input_modalities == ["audio"] and sa.output_modalities == ["text"]
    assert ad.parse_transcript(" habari yako ", req).text == "habari yako"


class _InProcessAPI:
    """The APIServer surface the transcription route uses, running the model
    in-process: load_audio -> process_prompt -> the node -> postprocess."""

    def __init__(self, model, upload_dir):
        self.model = model
        self.upload_dir = Path(upload_dir)
        self.sub = model.get_submodule(ENCODER_NODE, "cpu")
        self.results: dict[str, list] = {}
        self.submits: list = []

    def submit_request(self, **kw):
        self.submits.append(kw)
        path = kw["file_paths"]["audio"][0]
        loaded = self.model.load_audio(path, "cpu").data
        tensors = self.model.process_prompt(
            kw["text"],
            kw["input_modalities"],
            kw["output_modalities"],
            {"audio_inputs": [loaded]},
            **kw["model_kwargs"],
        )
        ids = _run_node(self.sub, tensors[WAVEFORM])["r0"]
        text = self.model.postprocess(ids, "text")
        self.results[kw["request_id"]] = [types.SimpleNamespace(modality="text", data=text, metadata={})]
        return kw["request_id"]

    async def collect_results(self, request_id, raw_request=None):
        return self.results.pop(request_id)

    async def iter_result_chunks(self, request_id):
        for chunk in self.results.pop(request_id):
            yield chunk

    def release_request(self, request_id):
        self.results.pop(request_id, None)


@pytest.fixture
def route(tiny_model, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import mstar.api_server

    api = _InProcessAPI(tiny_model, tmp_path / "uploads")
    api.model_name = KEYS[0]
    fake_ep = types.ModuleType("mstar.api_server.entrypoint")
    fake_ep.api_server = api
    monkeypatch.setitem(sys.modules, "mstar.api_server.entrypoint", fake_ep)
    monkeypatch.setattr(mstar.api_server, "entrypoint", fake_ep, raising=False)
    monkeypatch.setitem(sys.modules, "soundfile", None)
    monkeypatch.setitem(sys.modules, "torchcodec", None)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)
    from mstar.api_server.openai.router import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), api


def _wav_upload(seconds: float, rate: int = 16000, seed: int = 0) -> bytes:
    path = io.BytesIO()
    samples = np.random.default_rng(seed).uniform(-0.3, 0.3, int(seconds * rate))
    pcm = np.round(samples * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return path.getvalue()


def test_transcription_route_end_to_end(route, tiny_model, tmp_path):
    client, api = route
    upload = _wav_upload(2.5)
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": KEYS[0], "language": "yo", "response_format": "json"},
        files={"file": ("clip.wav", upload, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    # the same text as running the pieces by hand on the same file
    path = tmp_path / "same.wav"
    path.write_bytes(upload)
    wave_in = audio.preprocess(audio.load_waveform(str(path)))
    expected = tiny_model.postprocess(_run_node(api.sub, [wave_in])["r0"], "text").decode()
    assert r.json() == {"text": expected.strip()}
    assert len(api.submits) == 1 and api.submits[0]["input_modalities"] == ["audio"]
    # no timestamps: verbose_json has no segments, srt one cue spanning the clip
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": KEYS[0], "response_format": "verbose_json"},
        files={"file": ("clip.wav", upload, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["text"] == expected.strip() and r.json()["segments"] == []
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": KEYS[0], "response_format": "srt"},
        files={"file": ("clip.wav", upload, "audio/wav")},
    )
    assert r.status_code == 200 and r.text.count(" --> ") == 1, r.text


def test_transcription_route_rejects_an_over_cap_clip_it_cannot_window(route):
    # without soundfile/torchcodec the route cannot read the duration to window
    # the upload, so the model sees all 41 s and refuses it rather than cutting it
    client, api = route
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": KEYS[0]},
        files={"file": ("long.wav", _wav_upload(41), "audio/wav")},
    )
    assert r.status_code == 400 and "at most 40 s" in r.text, (r.status_code, r.text)


def test_long_uploads_are_cut_into_balanced_windows(tmp_path):
    """Windows of at most 15 s, of about equal length before each cut moves
    back (at most 2 s) to a pause: none shorter than half a window less 2 s."""
    from mstar.api_server import media_io

    rng = np.random.default_rng(0)
    speech = rng.uniform(-0.5, 0.5, 75 * 16000).astype(np.float32)
    speech[int(13.6 * 16000) : int(13.8 * 16000)] = 0.0  # a pause just before the first boundary (15 s)
    pieces = media_io.split_windows(speech, 15.0, 16000, search_seconds=2.0, balanced=True)
    assert all(5.5 * 16000 <= len(p) <= 15 * 16000 for p in pieces), [len(p) for p in pieces]
    assert 13.6 * 16000 <= len(pieces[0]) <= 13.8 * 16000
    assert np.array_equal(np.concatenate(pieces), speech)
    # 40.01 s ending in silence: the old fixed 40 s cut left a 10 ms window the model refuses
    tail = np.concatenate([rng.uniform(-0.5, 0.5, 38 * 16000), np.zeros(round(2.01 * 16000))]).astype(np.float32)
    assert len(media_io.split_windows(tail, 40.0, 16000, search_seconds=2.0)[-1]) == 160
    for window in (15.0, 40.0):
        pieces = media_io.split_windows(tail, window, 16000, search_seconds=2.0, balanced=True)
        assert len(pieces) >= math.ceil(40.01 / window)
        assert (window / 2 - 2) * 16000 <= min(map(len, pieces)) <= max(map(len, pieces)) <= window * 16000
        for p in pieces:
            audio.preprocess(torch.from_numpy(p))


class _WaveSoundfile(types.ModuleType):
    """``soundfile`` for WAV files over the standard library, so the route can
    read durations and decode windowed uploads where soundfile isn't installed."""

    def __init__(self):
        super().__init__("soundfile")

    @staticmethod
    def _open(path):
        with wave.open(str(path)) as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").reshape(-1, wf.getnchannels())
            return pcm.astype(np.float32) / 32768.0, wf.getframerate()

    def info(self, path):
        with wave.open(str(path)) as wf:
            return types.SimpleNamespace(frames=wf.getnframes(), samplerate=wf.getframerate())

    def read(self, path, dtype="float32", always_2d=False):
        data, rate = self._open(path)
        return (data if always_2d or data.shape[1] > 1 else data[:, 0]), rate

    def SoundFile(self, path):
        return _WaveSoundFile(*self._open(path))


class _WaveSoundFile:
    """A ``_WaveSoundfile`` file opened for block reads."""

    def __init__(self, data, rate):
        self._data, self.samplerate, self._pos = data, rate, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frames=-1, dtype="float32", always_2d=False):
        end = len(self._data) if frames < 0 else min(self._pos + frames, len(self._data))
        block, self._pos = self._data[self._pos : end], end
        return block if always_2d or block.shape[1] > 1 else block[:, 0]


@pytest.fixture
def windowing_route(route, monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", _WaveSoundfile())
    return route


def _wav_bytes(samples: np.ndarray, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.round(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def test_a_long_upload_ending_in_silence_is_transcribed(windowing_route):
    client, api = windowing_route
    rng = np.random.default_rng(3)
    samples = np.concatenate([rng.uniform(-0.3, 0.3, 38 * 16000), np.zeros(round(2.01 * 16000))])
    for fields, windows in (({}, 3), ({"window_seconds": "40"}, 2)):
        api.submits.clear()
        r = client.post(
            "/v1/audio/transcriptions",
            data={"model": KEYS[0], **fields},
            files={"file": ("long.wav", _wav_bytes(samples), "audio/wav")},
        )
        assert r.status_code == 200, r.text
        assert len(api.submits) >= windows
        assert all("window_seconds" not in s["model_kwargs"] for s in api.submits)


def test_uploads_up_to_20_s_are_heard_whole_and_longer_ones_in_16_s_windows(windowing_route):
    client, api = windowing_route
    rng = np.random.default_rng(5)
    for seconds, fields, windows in ((20.0, {}, 1), (21.0, {}, 2), (39.0, {}, 3), (18.0, {"window_seconds": "10"}, 2)):
        # every frame equally loud, so each cut stays on its nominal boundary
        samples = 0.3 * rng.choice([-1.0, 1.0], int(seconds * 16000))
        api.submits.clear()
        r = client.post(
            "/v1/audio/transcriptions",
            data={"model": KEYS[0], **fields},
            files={"file": ("a.wav", _wav_bytes(samples), "audio/wav")},
        )
        assert r.status_code == 200, r.text
        assert len(api.submits) == windows, (seconds, fields, len(api.submits))


def test_the_planner_takes_the_adapters_window_policy(tmp_path):
    from mstar.api_server.openai.adapters import OmnilingualASRAdapter
    from mstar.api_server.openai.serving_transcriptions import planner_for

    path = tmp_path / "a.wav"
    path.write_bytes(_wav_bytes(np.zeros(16000)))
    planner = planner_for(OmnilingualASRAdapter(), str(path), tmp_path)
    assert (planner.max_seconds, planner.whole_seconds, planner.balanced) == (16.0, 20.0, True)
    assert (planner.search_seconds, planner.frame_seconds) == (5.0, 0.1)
    # a short window a request asks for searches at most a third of its length
    planner = planner_for(OmnilingualASRAdapter(), str(path), tmp_path, 6.0)
    assert (planner.max_seconds, planner.whole_seconds, planner.search_seconds) == (6.0, 6.0, 2.0)


def test_window_seconds_is_bounded_by_the_clip_cap(windowing_route):
    client, api = windowing_route
    for bad in ("4", "41", "nan", "soon"):
        r = client.post(
            "/v1/audio/transcriptions",
            data={"model": KEYS[0], "window_seconds": bad},
            files={"file": ("a.wav", _wav_bytes(np.zeros(16000)), "audio/wav")},
        )
        assert r.status_code == 400 and "window_seconds must be between 5 and 40" in r.text, (bad, r.text)
    assert api.submits == []


def test_a_windowed_upload_at_a_crafted_rate_is_refused_before_resampling(windowing_route, monkeypatch):
    """Uploads over the window length are decoded by the route, not the model:
    a 41 s WAV at 100,003 Hz used to build a 1.6G-float resampling kernel there
    (seconds of a blocked event loop) before any check ran."""
    client, api = windowing_route

    def no_resample(*args, **kwargs):
        raise AssertionError("resampled from an unsupported rate")

    monkeypatch.setattr(torchaudio.functional, "resample", no_resample)
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": KEYS[0]},
        files={"file": ("odd.wav", _wav_bytes(np.zeros(41 * 100003), 100003), "audio/wav")},
    )
    assert r.status_code == 400 and "unsupported sample rate 100003 Hz" in r.text, r.text
    assert api.submits == []
    assert list(api.upload_dir.iterdir()) == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("error", [RuntimeError, ImportError])
def test_a_windowed_upload_that_fails_to_decode_is_removed(windowing_route, monkeypatch, stream, error):
    # the header is readable, the body is not: the decode raises something other than ValueError
    from mstar.api_server import media_io

    def broken(path, sample_rate, max_seconds=None):
        raise error("corrupt stream")

    monkeypatch.setattr(media_io, "decode_audio", broken)
    client, api = windowing_route
    r = client.post(
        "/v1/audio/transcriptions",
        data={"model": KEYS[0], "stream": "true" if stream else "false"},
        files={"file": ("corrupt.wav", _wav_bytes(np.zeros(21 * 16000)), "audio/wav")},
    )
    assert r.status_code == 500, r.text
    assert api.submits == []
    assert list(api.upload_dir.iterdir()) == []
