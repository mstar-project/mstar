"""Opt-in AudioSeal watermark (``mstar.model.audio_watermark``), CPU only.

Config and dependency checks need nothing. The streaming checks build the
streaming generator from its bundled card with random weights, so they need
the ``audioseal`` package but no download; detection needs the real weights
and runs only when they are already in AudioSeal's cache.
"""

from __future__ import annotations

import logging
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
    _watermark_precision,
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

    def batch_postprocess(self, outputs, request_infos, inputs, piecewise_runners=None):
        self.calls.append((list(request_infos), list(inputs), piecewise_runners))
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
    runners = {"watermark_1920": object()}
    engine = SimpleNamespace(_submodules={"Codec": SimpleNamespace(submodule=tail, piecewise_runners=runners)})
    batch = _batch(["a", "bad", "c"])
    Engine._postprocess_batch(engine, batch, {"a": {}, "bad": {}, "c": {}})
    # the node's captured regions come along, as the forward gets them
    assert tail.calls == [(["a", "c"], ["in-a", "in-c"], runners)]
    assert set(batch.failed_requests) == {"bad"}


def test_batch_tail_failure_fails_its_requests_only():
    tail = _Tail(fail=True)
    engine = SimpleNamespace(_submodules={"Codec": SimpleNamespace(submodule=tail, piecewise_runners={})})
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


def test_streams_leave_a_batch_with_one_flat_cache_row_each():
    """A stream's state is one row of its batch's caches, so streams regroup
    freely; the LSTM gets a contiguous ``[layers, batch, hidden]`` hx (cuDNN,
    CUDA only, refuses a non-contiguous one)."""
    wmk = _watermarker()
    streams = [WatermarkStream() for _ in range(3)]
    wmk.apply([_speechlike(640, seed=s) for s in range(3)], streams)
    for stream in streams:
        assert stream.caches.shape == (wmk.cache_size,)
    lstm = [cache for slot, cache in enumerate(wmk._unpack(torch.stack([s.caches for s in streams])))
            if slot in wmk._layout.lstm_slots]
    assert lstm and all(cache.is_contiguous() and cache.shape[1] == 3 for cache in lstm)
    alone = wmk.apply([_speechlike(320, seed=9)], [streams[1]])[0]
    assert alone.shape == (320,)


def _moshi_streaming(wmk, audio, sizes):
    """AudioSeal's own streaming layers as they would run if the state
    reached them (the propagation bug, upstream #105, set by hand)."""
    from audioseal.libs.moshi.modules.streaming import StreamingModule

    gen = wmk.generator
    modules = [m for m in gen.modules() if isinstance(m, StreamingModule)]
    for module in modules:
        module._streaming_state = module._init_streaming_state(audio.shape[0])
        module._streaming_state.init_stream()
    try:
        out, start = [], 0
        with torch.no_grad(), wmk._no_compile():
            for size in sizes:
                x = audio[:, start:start + size]
                hidden = gen.encoder(x.unsqueeze(1)) + wmk._message_embedding
                out.append(x + wmk.config.alpha * gen.decoder(hidden).squeeze(1))
                start += size
    finally:
        for module in modules:
            module._streaming_state = module._empty_state()
    return torch.cat(out, dim=-1)


def test_the_stream_equals_audioseals_streaming_layers_bit_for_bit():
    """The generator run as a function of its caches does what AudioSeal's
    streaming layers do, op for op: the same floats, not just close ones."""
    wmk = _watermarker(message=0xBEEF)
    audios = torch.stack([_speechlike(5760, seed=s) for s in (10, 11, 12)])
    sizes = [640, 1920, 320, 2880]
    streams = [WatermarkStream() for _ in range(3)]
    pieces, start = [], 0
    for size in sizes:
        chunks = list(audios[:, start:start + size])
        pieces.append(torch.stack(wmk.apply(chunks, streams)))
        start += size
    assert torch.equal(torch.cat(pieces, dim=-1), _moshi_streaming(wmk, audios, sizes))


class _ReplayingRunner:
    """``PiecewiseAcceleratorGraphRunner.run``'s contract without CUDA: the real
    inputs go into the head of buffers padded to the capture batch (the tail
    zeroed), the captured callable runs on the whole buffers, and the real
    rows come back."""

    def __init__(self, config, batch_sizes):
        from mstar.engine.accelerator_graph_config import PiecewiseCallInputs, PiecewiseCaptureShape

        self.config, self.batch_sizes, self.calls = config, batch_sizes, []
        self._shape, self._call = PiecewiseCaptureShape, PiecewiseCallInputs

    def can_run(self, batch_size):
        return batch_size <= max(self.batch_sizes)

    def run(self, static_inputs, real_bs):
        from mstar.engine.accelerator_graph_runner import PiecewiseOutput

        bs = min(b for b in self.batch_sizes if b >= real_bs)
        buffers = self.config.make_static_inputs(self._shape(bs=bs, seq_lens=[self.config.seq_len] * bs,
                                                             total_tokens=self.config.seq_len * bs))
        for name, value in static_inputs.items():
            buffers[name][:real_bs].copy_(value)
            buffers[name][real_bs:].zero_()
        self.calls.append((real_bs, bs))
        return PiecewiseOutput(self.config.capture_fn(self._call(static_inputs=buffers, engine_inputs=None)), real_bs)


def test_replayed_pcm16_calls_equal_the_eager_ones_and_finals_pad_into_a_capture():
    """Codec-shaped streams through ``apply_pcm16`` with the captured lengths
    replayed (batches padded up to a capture bucket) give the eager path's
    samples, and a short final chunk shares the call of a longer length."""
    frame = 320 * 6
    lengths = [frame, 3 * frame]
    replay, eager = _watermarker(message=0xBEEF), _watermarker(message=0xBEEF)
    configs = replay.piecewise_configs(lengths, [1, 2, 4])
    assert sorted(configs) == [replay.capture_label(n) for n in lengths]
    runners = {label: _ReplayingRunner(config, [1, 2, 4]) for label, config in configs.items()}
    pcm = [(_speechlike(6 * frame, seed=s).clamp(-1, 1) * 32767).to(torch.int16) for s in (20, 21, 22)]
    # (stream, start, end, final): streams at different points share steps; in
    # the last, two 2-frame finals pad into the call of a 3-frame chunk
    steps = [
        [(0, 0, frame, False), (1, 0, frame, False), (2, 0, frame, False)],
        [(0, frame, 4 * frame, False), (1, frame, 4 * frame, False), (2, frame, 2 * frame, False)],
        [(0, 4 * frame, 6 * frame, True), (1, 4 * frame, 6 * frame, True), (2, 2 * frame, 5 * frame, False)],
    ]
    replay_streams, eager_streams = [WatermarkStream() for _ in pcm], [WatermarkStream() for _ in pcm]
    for step in steps:
        chunks = [pcm[k][lo:hi] for k, lo, hi, _ in step]
        finals = [final for *_, final in step]
        ks = [k for k, *_ in step]
        got = replay.apply_pcm16(chunks, [replay_streams[k] for k in ks], finals, runners=runners)
        want = eager.apply_pcm16(chunks, [eager_streams[k] for k in ks], finals)
        for g, w in zip(got, want, strict=True):
            assert g.dtype == torch.int16 and g.shape == w.shape
            assert (g.int() - w.int()).abs().max() <= 1
    one, three = (runners[replay.capture_label(n)].calls for n in lengths)
    assert one == [(3, 4), (1, 1)] and three == [(2, 2), (3, 4)]
    assert all(s.finished and s.caches is None for s in replay_streams[:2])
    assert (eager_streams[2].caches - replay_streams[2].caches).abs().max() < 1e-5


def test_calls_that_miss_the_captures_say_so_once(caplog):
    """With captures in place, a chunk length with no capture, a batch over the
    largest captured one and a partial frame each run eagerly and log it once."""
    frame = 320 * 6
    wmk = _watermarker()
    configs = wmk.piecewise_configs([frame, 3 * frame], [1, 2])
    runners = {label: _ReplayingRunner(config, [1, 2]) for label, config in configs.items()}
    pcm = (_speechlike(4 * frame, seed=40).clamp(-1, 1) * 32767).to(torch.int16)
    streams = [WatermarkStream() for _ in range(5)]
    with caplog.at_level(logging.WARNING, logger="mstar.model.audio_watermark"):
        for start in (0, 2 * frame):
            wmk.apply_pcm16([pcm[start:start + 2 * frame]], streams[:1], runners=runners)
            wmk.apply_pcm16([pcm[start:start + frame]] * 3, streams[1:4], runners=runners)
        wmk.apply_pcm16([pcm[:frame + 100]], streams[4:], runners=runners)
        wmk.apply_pcm16([pcm[frame + 100:2 * frame + 100]], streams[4:], runners=runners)
    messages = [r.getMessage() for r in caplog.records if "runs eagerly" in r.getMessage()]
    assert len(messages) == 3, messages
    assert f"a chunk of {2 * frame} samples" in messages[0]
    assert f"3 chunks of {frame} samples" in messages[1]
    assert "partial frame" in messages[2]
    assert all(not r.calls for r in runners.values())
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mstar.model.audio_watermark"):
        wmk.apply_pcm16([pcm[:frame]], [WatermarkStream()], runners=runners)
        _watermarker().apply_pcm16([pcm[:2 * frame]], [WatermarkStream()])
    assert not [r for r in caplog.records if "runs eagerly" in r.getMessage()]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs CUDA")
def test_engine_captures_replay_the_eager_watermark_bit_for_bit():
    """The engine's piecewise runner captures ``_mark_pcm16`` per length and
    batch bucket. At a bucket's batch size a replay gives the eager call's
    samples and caches exactly, step after step; a batch padded up to its
    bucket, or a final chunk padded to a captured length, convolves another
    shape, which cuDNN may run with another TF32 algorithm: within 2 steps of
    PCM16 on Hopper, up to 14 on an A100, so other GPUs get 1e-3 of full scale.
    Detection is unchanged either way."""
    from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
    from mstar.engine.accelerator_graph_runner import PiecewiseAcceleratorGraphRunner
    from mstar.platforms import current_platform

    padded_bound = 2 if current_platform.has_cuda_capability(9, 0) else 33

    class NoResources:
        def build_accelerator_graph_buffers(self, *args, **kwargs):
            pass

        def ingest_capture_row(self, rid):
            pass

    frame = 320 * 6
    wmk = AudioSealWatermarker(_random_generator().cuda(), AudioWatermarkConfig(enabled=True, message=0xBEEF),
                               import_audioseal()[1])
    configs = wmk.piecewise_configs([frame, 3 * frame], [1, 2, 4])
    pool = get_accelerator_graph_backend(wmk.device).graph_pool_handle()
    runners = {}
    for label, config in configs.items():
        runner = PiecewiseAcceleratorGraphRunner(
            label=label, config=config, resources={}, step_runner=NoResources(), device=wmk.device,
            autocast_dtype=None, num_slots=1, memory_pool=pool,
        )
        runner.warmup_and_capture()
        assert runner.any_graphs and not runner.dropped_shapes
        runners[label] = runner
    pcm = [(_speechlike(7 * frame, seed=s).clamp(-1, 1) * 32767).to(torch.int16).cuda() for s in range(30, 34)]

    def steps(streams_used, exact):
        replayed, eager = [WatermarkStream() for _ in streams_used], [WatermarkStream() for _ in streams_used]
        for lo, hi in [(0, frame), (frame, 4 * frame), (4 * frame, 7 * frame)]:
            chunks = [x[lo:hi] for x in streams_used]
            got = wmk.apply_pcm16(chunks, replayed, runners=runners)
            want = wmk.apply_pcm16(chunks, eager)
            for g, w in zip(got, want, strict=True):
                assert torch.equal(g, w) if exact else (g.int() - w.int()).abs().max() <= padded_bound
            for r, e in zip(replayed, eager, strict=True):
                assert torch.equal(r.caches, e.caches) if exact else torch.allclose(r.caches, e.caches, atol=1e-2)
        return replayed, eager

    replayed, eager = steps(pcm, exact=True)        # 4 rows: the bs=4 capture as is
    steps(pcm[:3], exact=False)                     # 3 rows, padded to 4
    # two streams then finish with 2 frames, padded into the 3-frame capture
    tail = [x[:2 * frame] for x in pcm[:2]]
    got = wmk.apply_pcm16(tail, replayed[:2], [True, True], runners=runners)
    want = wmk.apply_pcm16(tail, eager[:2], [True, True])
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape and (g.int() - w.int()).abs().max() <= padded_bound


def test_capture_lengths_must_be_whole_frames():
    wmk = _watermarker()
    with pytest.raises(ValueError, match="whole"):
        wmk.piecewise_configs([wmk.hop + 1], [1])


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


def test_watermark_runs_fp32_matmuls_and_tf32_cudnn_and_restores_the_flags(monkeypatch, torch_flag):
    """cuDNN on H100 picks a far slower conv with TF32 off; matmuls, which the
    engine turns to TF32, stay fp32."""
    wmk = _watermarker()
    seen = []
    first_conv = wmk._encoder[0].conv
    forward = first_conv.forward

    def conv(x):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return forward(x)

    monkeypatch.setattr(first_conv, "forward", conv)
    torch_flag(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    wmk.apply([torch.zeros(320)], [WatermarkStream()])
    assert seen == [(False, True)]
    assert torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32


def test_precision_getters_answer_while_the_watermark_runs(monkeypatch):
    """Flipping ``cuda.matmul.allow_tf32`` alone made
    ``get_float32_matmul_precision()`` raise (a mix of the legacy and new
    APIs) for every thread until the window closed."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with _watermark_precision():
            assert torch.get_float32_matmul_precision() == "highest"
            assert not torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "high"
        assert torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(previous)


def test_message_embedding_is_built_once_and_matches_the_processor(monkeypatch):
    """AudioSeal's MsgProcessor copies a host ``arange`` to the device on
    every call; the watermarker adds the same embedding, built at init."""
    wmk = _watermarker(message=0xBEEF)
    hidden = torch.randn(3, wmk._message_embedding.shape[1], 7)
    message = torch.tensor([wmk.config.message_bits()] * 3)
    with torch.no_grad():
        expected = wmk.generator.msg_processor(hidden, message)
    assert torch.equal(hidden + wmk._message_embedding, expected)

    def refuse(*args, **kwargs):
        raise AssertionError("MsgProcessor ran during a watermark call")

    audio = _speechlike(1920, seed=8)
    reference = _library_offline(wmk, audio)
    monkeypatch.setattr(wmk.generator.msg_processor, "forward", refuse)
    torch.testing.assert_close(wmk.offline(audio), reference, atol=1e-6, rtol=0)


class _FakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.lengths = []

    def detect_watermark(self, x):
        self.lengths.append(x.shape[-1])
        return torch.ones(1), torch.zeros(1, 16, dtype=torch.int)


@pytest.mark.parametrize(("rate", "expected"), [(16_000, 24_000), (24_000, 24_000), (48_000, 24_000)])
def test_detect_resamples_to_the_rate_the_mark_was_added_at(rate, expected):
    from contextlib import nullcontext

    from mstar.model.audio_watermark import MARK_SAMPLE_RATE, AudioSealDetector

    fake = _FakeDetector()
    AudioSealDetector(fake, nullcontext).detect(torch.zeros(rate), rate)
    assert MARK_SAMPLE_RATE == 24_000 and fake.lengths == [expected]


@pytest.mark.parametrize(
    ("decoded", "argv", "verdict"),
    [
        (0xBEEF, ["--message", "0xBEEF"], "watermarked"),
        (0xBEEF, ["--message", "1011111011101111"], "watermarked"),
        (0x1234, ["--message", "48879"], "not watermarked"),
        (0x1234, [], "watermarked"),
    ],
)
def test_detect_cli_counts_a_file_as_marked_only_with_the_deployments_message(
    tmp_path, monkeypatch, capsys, decoded, argv, verdict
):
    """An unmarked file re-encoded as MP3 can score above the threshold (50 of
    167 served Qwen3-TTS files did) but decodes to some other message."""
    sf = pytest.importorskip("soundfile")
    from mstar.model import audio_watermark

    class _Detector:
        def detect(self, audio, sample_rate):
            return 0.9, decoded

    monkeypatch.setattr(audio_watermark.AudioSealDetector, "build", classmethod(lambda cls, device: _Detector()))
    path = tmp_path / "speech.wav"
    sf.write(path, torch.zeros(24_000).numpy(), 24_000)
    audio_watermark._main(["detect", *argv, str(path)])
    assert capsys.readouterr().out.startswith(f"{path}: {verdict} (score 0.900")


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
    import torchaudio

    for rate in (16_000, 48_000):
        copy = torchaudio.functional.resample(streamed, 24_000, rate)
        score, message = detector.detect(copy, rate)
        assert score > 0.9 and message == 0xBEEF, rate


def test_stream_check_batches_at_most_the_codecs_rows(monkeypatch, tmp_path):
    """33 streams in step: the check splits each step at the codec's 32 rows, as
    serving does, so ``--captured`` replays every group instead of running it eagerly."""
    import numpy as np

    check = pytest.importorskip("benchmark.audio_watermark_check")
    rows = []

    class _Marker:
        def apply_pcm16(self, chunks, streams, finals, **kwargs):
            rows.append(len(chunks))
            return chunks

    monkeypatch.setattr(check.AudioSealWatermarker, "build", staticmethod(lambda config, device: _Marker()))
    monkeypatch.setattr(check, "_wavs", lambda d: {f"f{i:02d}": np.zeros(240000, dtype=np.int16) for i in range(33)})
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=lambda *a, **k: None))
    args = SimpleNamespace(device="cpu", captured=False, wavs="in", out=str(tmp_path), compare=None)
    assert check.stream(args)["files"] == 33
    assert max(rows) == max(check.CAPTURE_BATCH_SIZES) and 1 in rows
