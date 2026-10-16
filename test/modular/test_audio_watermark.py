"""Opt-in AudioSeal watermark (``mstar.model.audio_watermark``), CPU only.

Config and dependency checks need nothing. The streaming checks build the
streaming generator from its bundled card with random weights, so they need
the ``audioseal`` package but no download; detection needs the real weights
and runs only when they are already in AudioSeal's cache.
"""

from __future__ import annotations

import sys
from hashlib import sha1
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.model.audio_watermark import (
    DEFAULT_DETECTOR,
    DEFAULT_GENERATOR,
    AudioSealWatermarker,
    AudioWatermarkConfig,
    WatermarkStream,
    import_audioseal,
)
from mstar.model.submodule_base import NodeSubmodule

# -- config -------------------------------------------------------------------


def test_watermark_is_off_by_default():
    assert AudioWatermarkConfig.parse(None) == AudioWatermarkConfig()
    assert AudioWatermarkConfig().enabled is False
    assert AudioWatermarkConfig.parse({}).enabled is False


def test_watermark_config_reads_message_forms():
    as_int = AudioWatermarkConfig.parse({"enabled": True, "message": 0xBEEF, "alpha": 0.5})
    as_bits = AudioWatermarkConfig.parse({"enabled": True, "message": "1011111011101111"})
    assert as_int.message == as_bits.message == 0xBEEF
    assert as_int.alpha == 0.5 and as_bits.alpha == 1.0
    assert as_bits.message_bits() == [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("value", [
    True,
    {"enabled": "yes"},
    {"enabled": True, "message": 1 << 16},
    {"enabled": True, "message": -1},
    {"enabled": True, "message": "1011"},
    {"enabled": True, "message": "101111101110111x"},
    {"enabled": True, "message": True},
    {"enabled": True, "alpha": 0},
    {"enabled": True, "alpha": 5.0},
    {"enabled": True, "alpha": True},
    {"enabled": True, "generator": ""},
    {"enabled": True, "strength": 1.0},
])
def test_watermark_config_rejects_bad_values(value):
    with pytest.raises(ValueError, match="audio_watermark"):
        AudioWatermarkConfig.parse(value)


def test_missing_audioseal_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "audioseal", None)
    with pytest.raises(ImportError, match=r"mstar-ai\[audioseal\]"):
        import_audioseal()
    with pytest.raises(ImportError, match="audio_watermark.enabled"):
        AudioSealWatermarker.build(AudioWatermarkConfig(enabled=True), "cpu")


def test_qwen3_tts_checks_the_watermark_before_loading_anything(monkeypatch):
    pytest.importorskip("transformers")
    from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel

    monkeypatch.setitem(sys.modules, "audioseal", None)
    with pytest.raises(ImportError, match=r"mstar-ai\[audioseal\]"):
        Qwen3TTSModel("does/not-exist", audio_watermark={"enabled": True})
    with pytest.raises(ValueError, match="audio_watermark.message"):
        Qwen3TTSModel("does/not-exist", audio_watermark={"enabled": True, "message": "nope"})


# -- engine hook --------------------------------------------------------------


class _Tail(NodeSubmodule):
    def __init__(self, fail: bool = False):
        super().__init__()
        self.fail = fail
        self.calls = []

    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def postprocess(self, request_id, request_info, outputs, inputs=None, **kwargs):
        if request_id == "bad":
            raise ValueError("bad request")

    def postprocess_batch_tail(self, request_ids, outputs, inputs):
        self.calls.append((list(request_ids), list(inputs)))
        if self.fail:
            raise RuntimeError("tail failed")


def _batch(rids):
    failures = {}
    return SimpleNamespace(
        node_name="Codec",
        request_ids=rids,
        inputs=[f"in-{rid}" for rid in rids],
        per_request_info_wrapped={rid: None for rid in rids},
        failed_requests=failures,
        register_failure=lambda rid, error: failures.__setitem__(rid, str(error)),
        step_context=SimpleNamespace(graph_walk="codec_chunk"),
    )


def test_batch_tail_runs_once_on_the_requests_postprocess_kept():
    tail = _Tail()
    engine = SimpleNamespace(_submodules={"Codec": SimpleNamespace(submodule=tail)})
    batch = _batch(["a", "bad", "c"])
    Engine._postprocess_batch(engine, batch, {"a": {}, "bad": {}, "c": {}})
    assert tail.calls == [(["a", "c"], ["in-a", "in-c"])]
    assert set(batch.failed_requests) == {"bad"}


def test_batch_tail_failure_fails_its_requests_only():
    tail = _Tail(fail=True)
    engine = SimpleNamespace(_submodules={"Codec": SimpleNamespace(submodule=tail)})
    batch = _batch(["a", "bad", "c"])
    Engine._postprocess_batch(engine, batch, {})
    assert set(batch.failed_requests) == {"a", "bad", "c"}
    assert "tail failed" in batch.failed_requests["a"]


# -- streaming (random weights) ----------------------------------------------


def _random_generator(seed: int = 0):
    pytest.importorskip("audioseal")
    from audioseal.builder import AudioSealWMConfig, create_generator
    from audioseal.loader import AudioSeal, load_local_model_config
    from omegaconf import OmegaConf

    card = OmegaConf.to_container(load_local_model_config(DEFAULT_GENERATOR))
    card.pop("checkpoint")
    torch.manual_seed(seed)
    generator = create_generator(AudioSeal.parse_config(card, AudioSealWMConfig, nbits=16))
    return generator.eval().requires_grad_(False)


def _watermarker(**config):
    generator = _random_generator()
    _, no_compile = import_audioseal()
    return AudioSealWatermarker(generator, AudioWatermarkConfig(enabled=True, **config), no_compile)


def _library_offline(wmk: AudioSealWatermarker, audio: torch.Tensor) -> torch.Tensor:
    """AudioSeal's own non-streaming watermark of the whole utterance."""
    with torch.no_grad(), wmk._no_compile():
        message = torch.tensor([wmk.config.message_bits()])
        wm = wmk.generator.get_watermark(audio.view(1, 1, -1), message=message).view(-1)
    return audio + wmk.config.alpha * wm


def _speechlike(samples: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(samples) / 16000
    tone = sum(torch.sin(2 * torch.pi * (130 + seed) * k * t) / k for k in range(1, 6))
    return 0.2 * tone * (0.6 + 0.4 * torch.sin(2 * torch.pi * 4 * t)) + 0.01 * torch.randn(samples, generator=gen)


def _stream(wmk, audio, sizes):
    stream, out, start = WatermarkStream(), [], 0
    for i, size in enumerate(sizes):
        out.append(wmk.apply([audio[start:start + size]], [stream], [i == len(sizes) - 1])[0])
        start += size
    return out


@pytest.mark.parametrize("sizes", [
    [640, 1280, 320, 1920],          # whole frames: nothing held back
    [500, 1000, 37, 1500, 1123],     # ragged: partial frames wait for the next chunk
])
def test_streamed_watermark_equals_the_offline_watermark(sizes):
    wmk = _watermarker(message=0xBEEF)
    audio = _speechlike(sum(sizes), seed=1)
    pieces = _stream(wmk, audio, sizes)
    streamed = torch.cat(pieces)
    assert streamed.shape == audio.shape
    reference = _library_offline(wmk, audio)
    scale = (reference - audio).abs().max()
    assert scale > 1e-4   # the random generator does mark the audio
    torch.testing.assert_close(streamed, reference, atol=1e-5 * max(1.0, scale.item()), rtol=0)
    torch.testing.assert_close(wmk.offline(audio), reference, atol=1e-6, rtol=0)
    # every chunk but the last emits only whole frames
    assert all(piece.shape[0] % wmk.hop == 0 for piece in pieces[:-1])


def test_an_utterance_ending_mid_frame_differs_from_the_library_in_its_last_frame_only():
    wmk = _watermarker(message=0xBEEF)
    sizes = [1280, 20, 1260, 1440]   # 4,000 samples: 12.5 frames
    audio = _speechlike(sum(sizes), seed=1)
    streamed = torch.cat(_stream(wmk, audio, sizes))
    assert streamed.shape == audio.shape
    torch.testing.assert_close(streamed, wmk.offline(audio), atol=1e-6, rtol=0)
    whole = audio.shape[0] - audio.shape[0] % wmk.hop
    torch.testing.assert_close(streamed[:whole], _library_offline(wmk, audio)[:whole], atol=1e-6, rtol=0)


def test_library_streaming_context_does_not_carry_history():
    """Why the module keeps the state itself (upstream issue #105)."""
    wmk = _watermarker()
    audio = _speechlike(2560, seed=2)
    message = torch.tensor([wmk.config.message_bits()])
    with torch.no_grad(), wmk._no_compile(), wmk.generator.streaming(1):
        parts = [wmk.generator(audio[i:i + 640].view(1, 1, -1), message=message) for i in range(0, 2560, 640)]
    naive = torch.cat([p.view(-1) for p in parts])
    assert (naive - _library_offline(wmk, audio)).abs().max() > 1e-4


def test_streams_batch_without_mixing():
    wmk = _watermarker(message=0x1234)
    audios = [_speechlike(2240, seed=3), _speechlike(2240, seed=4), _speechlike(1600, seed=5)]
    streams = [WatermarkStream() for _ in audios]
    out = [[] for _ in audios]
    # (stream, start, end, final) per step; stream 2 starts at step 1 with a
    # chunk as long as the others', so a new stream shares a batch with running ones
    steps = [
        [(0, 0, 640, False), (1, 0, 640, False)],
        [(0, 640, 960, False), (1, 640, 960, False), (2, 0, 320, False)],
        [(0, 960, 2240, True), (1, 960, 2240, True), (2, 320, 1600, True)],
    ]
    for step in steps:
        chunks = [audios[k][lo:hi] for k, lo, hi, _ in step]
        marked = wmk.apply(chunks, [streams[k] for k, *_ in step], [final for *_, final in step])
        for (k, *_), piece in zip(step, marked, strict=True):
            out[k].append(piece)
    for k, audio in enumerate(audios):
        torch.testing.assert_close(torch.cat(out[k]), _library_offline(wmk, audio), atol=1e-5, rtol=0)


def test_streams_leave_a_batch_with_contiguous_caches():
    """A stream that ran in a batch can run alone next: cuDNN's LSTM (CUDA
    only) refuses the non-contiguous hx a batch-dim slice would be."""
    wmk = _watermarker()
    streams = [WatermarkStream() for _ in range(3)]
    wmk.apply([_speechlike(640, seed=s) for s in range(3)], streams)
    for stream in streams:
        assert all(cache.is_contiguous() for cache in stream.caches)


def test_pcm16_round_trip_matches_offline_within_one_step():
    wmk = _watermarker()
    audio = _speechlike(1920, seed=6)
    pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16)
    stream = WatermarkStream()
    marked = torch.cat([
        wmk.apply_pcm16([pcm[:640]], [stream])[0],
        wmk.apply_pcm16([pcm[640:]], [stream], [True])[0],
    ])
    expected = (_library_offline(wmk, pcm.float() / 32767).clamp(-1, 1) * 32767).to(torch.int16)
    assert marked.dtype == torch.int16
    assert (marked.int() - expected.int()).abs().max() <= 1


def test_watermark_runs_without_tf32_and_restores_the_flags(monkeypatch):
    """TF32 rounds a chunk and the whole utterance apart on GPU; the engine
    turns TF32 matmuls on, so the generator must turn both kinds off."""
    wmk = _watermarker()
    seen = []
    forward = wmk.generator.encoder.forward

    def encoder(x):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return forward(x)

    monkeypatch.setattr(wmk.generator.encoder, "forward", encoder)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    wmk.apply([torch.zeros(320)], [WatermarkStream()])
    assert seen == [(False, False)]
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32


def test_stream_refuses_chunks_after_its_final_one():
    wmk = _watermarker()
    stream = WatermarkStream()
    wmk.apply([torch.zeros(320)], [stream], [True])
    with pytest.raises(RuntimeError, match="after its final"):
        wmk.apply([torch.zeros(320)], [stream])


def test_non_causal_generator_is_refused():
    pytest.importorskip("audioseal")
    from audioseal.builder import AudioSealWMConfig, create_generator
    from audioseal.loader import AudioSeal, load_local_model_config
    from omegaconf import OmegaConf

    card = OmegaConf.to_container(load_local_model_config("audioseal_wm_16bits"))
    card.pop("checkpoint")
    generator = create_generator(AudioSeal.parse_config(card, AudioSealWMConfig, nbits=16))
    _, no_compile = import_audioseal()
    with pytest.raises(ValueError, match="causal"):
        AudioSealWatermarker(generator, AudioWatermarkConfig(enabled=True), no_compile)


# -- detection (real weights, only when cached) -------------------------------


def _weights_cached(card: str) -> bool:
    try:
        from audioseal.loader import _get_cache_dir, load_local_model_config
    except ImportError:
        return False
    url = load_local_model_config(card).checkpoint
    name = sha1(urlparse(url).path.encode()).hexdigest()[:24]
    return (_get_cache_dir(["AUDIOSEAL_CACHE_DIR", "XDG_CACHE_HOME"]) / name).is_file()


@pytest.mark.skipif(
    not (_weights_cached(DEFAULT_GENERATOR) and _weights_cached(DEFAULT_DETECTOR)),
    reason="AudioSeal streaming weights are not in the local cache",
)
def test_detector_finds_the_streamed_mark_and_its_message():
    from mstar.model.audio_watermark import AudioSealDetector

    wmk = AudioSealWatermarker.build(AudioWatermarkConfig(enabled=True, message=0xBEEF), "cpu")
    detector = AudioSealDetector.build("cpu")
    audio = _speechlike(16000, seed=7)
    streamed = torch.cat(_stream(wmk, audio, [1920, 3840, 1920, 8320]))
    score, message = detector.detect(streamed)
    assert score > 0.9 and message == 0xBEEF
    assert detector.detect(audio)[0] < 0.1
