"""Qwen3-Omni Code2Wav streaming: the stitched chunks are one continuous decode.

Each test drives Talker codec frames through the edge's
``ScheduledLeftContextChunkPolicy`` and a real ``StreamBuffer``, then through
``Code2WavSubmodule``'s ``cg_key_info`` / ``prepare_inputs`` / ``preprocess``
/ ``forward_batched`` / ``postprocess``, and compares the concatenated audio
with one decode of the whole stream. Schedule ``()`` is a plain 25-frame
first chunk; the default ramps up from 4 frames.

The decoder's ``ConvTranspose1d(2r, stride r, padding r)`` blocks look one
frame ahead, so W frames decode to ``W * 1920 - 555`` samples, not
``W * 1920``. The trim has to account for that or 555 samples go missing at
every chunk boundary.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from mstar.model.qwen3_omni.components import code2wav as code2wav_module
from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
from mstar.model.qwen3_omni.config import Code2WavConfig, Qwen3OmniModelConfig
from mstar.model.qwen3_omni.submodules import Code2WavSubmodule
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine
from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer, StreamChunkInfo

UPSAMPLE = 1920
CONTEXT_ONLY = Code2WavSubmodule._CONTEXT_ONLY_KEY
SCHEDULES = [(), Code2WavConfig().codec_chunk_schedule]


def _config(schedule: tuple[int, ...]) -> Qwen3OmniModelConfig:
    return Qwen3OmniModelConfig(code2wav=Code2WavConfig(codec_chunk_schedule=schedule))


def _chunk_sizes(cfg: Code2WavConfig):
    """New frames per chunk, in stream order."""
    yield from cfg.codec_chunk_schedule
    while True:
        yield cfg.codec_chunk_frames


def _decodes(cfg: Code2WavConfig, num_frames: int) -> int:
    """Windows that carry new frames: chunks until the stream is covered (the last may be short)."""
    calls = delivered = 0
    for size in _chunk_sizes(cfg):
        if delivered >= num_frames:
            return calls
        calls += 1
        delivered += size


def _ends_on_boundary(cfg: Code2WavConfig, num_frames: int) -> bool:
    """Whether the stream ends with a full chunk, so a flush of context alone follows."""
    delivered = 0
    for size in _chunk_sizes(cfg):
        if delivered >= num_frames:
            return delivered == num_frames
        delivered += size


def _to_pcm(wav: torch.Tensor) -> torch.Tensor:
    """The submodule's float -> int16 conversion."""
    return (wav.clamp(-1, 1) * 32767).to(torch.int16)


def _stream_batched(
    submodule: Code2WavSubmodule, frames: dict[str, torch.Tensor], delays: dict[str, int] | None = None,
) -> tuple[dict, list, list]:
    """Stream each request's ``frames`` ``[T, 16]`` through its own buffer
    with the edge's policy. Every step batches the requests that still have a
    chunk, the way the scheduler batches Code2Wav; a request with a delay
    joins that many steps late. Returns, per request, its
    ``(chunk_info, audio, decoded)`` per step (``decoded``: whether the step
    ran the vocoder), and each step's ``cg_key_info`` and row count."""
    cfg = submodule.config.code2wav
    buffers = {}
    for rid, request_frames in frames.items():
        buffers[rid] = StreamBuffer(
            request_id=rid, edge_name="codec_tokens", from_partition="Talker",
            policy=ScheduledLeftContextChunkPolicy(
                schedule=cfg.codec_chunk_schedule, chunk=cfg.codec_chunk_frames,
                left_context=cfg.codec_left_context_frames,
            ),
        )
        for i, frame in enumerate(request_frames):
            buffers[rid].pre_read_register(f"t{i}")
            buffers[rid].put(f"t{i}", frame)
        buffers[rid].signal_done()

    fwd_infos = {rid: SimpleNamespace(request_id=rid, rid_handle=rid) for rid in frames}
    steps = {rid: [] for rid in frames}
    keys, sizes, step = [], [], 0
    delays = delays or {}
    while any(buffer.has_chunk_ready() for buffer in buffers.values()):
        rids = [rid for rid, buffer in buffers.items() if step >= delays.get(rid, 0) and buffer.has_chunk_ready()]
        step += 1
        if not rids:
            continue
        chunks = {rid: buffers[rid].pop_chunk() for rid in rids}
        meta = {rid: InputMetadata(stream_chunks={"codec_tokens": chunk.info}) for rid, chunk in chunks.items()}
        # The engine replays the capture of the window the key names, and runs
        # the eager forward for the context-only key.
        key = submodule.cg_key_info(
            "code2wav_chunk", {rid: fwd_infos[rid] for rid in rids}, per_request_input_metadata=meta,
        )
        keys.append(key)
        sizes.append(len(rids))
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids, per_request_info={}, per_request_input_metadata=meta, captured=key != CONTEXT_ONLY,
        )
        items = [
            submodule.prepare_inputs(
                "code2wav_chunk", fwd_infos[rid], {"codec_tokens": [chunks[rid].data["data"]]},
                input_metadata=meta[rid],
            )
            for rid in rids
        ]
        assert submodule.can_batch(None, items)
        packed = submodule.preprocess("code2wav_chunk", engine_inputs, items)
        decoded = packed["codec_tokens"].shape[-1] > 0
        if key != CONTEXT_ONLY:
            # the replayed capture's input shape
            assert key in submodule.windows and packed["codec_tokens"].shape[-1] == key
        with torch.no_grad():
            outputs = submodule.forward_batched("code2wav_chunk", engine_inputs, **packed)
        for rid, item in zip(rids, items, strict=True):
            submodule.postprocess(rid, fwd_infos[rid], outputs[rid], inputs=item)
            steps[rid].append((chunks[rid].info, outputs[rid]["audio_chunk"][0], decoded))
            if chunks[rid].is_final:
                submodule.cleanup_request(rid)
    return steps, keys, sizes


def _stream(submodule: Code2WavSubmodule, frames: torch.Tensor) -> list[tuple]:
    """``_stream_batched`` for a single request."""
    return _stream_batched(submodule, {"r": frames})[0]["r"]


def _expected_samples_per_step(steps: list[tuple], tail: int) -> list[int]:
    """The first window emits its frames' audio less the tail; every later
    window emits exactly its new frames' audio, which includes the tail the
    window before it could not decode."""
    expected = []
    for info, _, _ in steps:
        new = info.num_items - info.context_items
        expected.append(max(new * UPSAMPLE - tail, 0) if info.context_items == 0 else new * UPSAMPLE)
    return expected


# ---------------------------------------------------------------------------
# A fake decoder with the real output length: exact trim arithmetic
# ---------------------------------------------------------------------------

class _LookaheadDecoder(torch.nn.Module):
    """W frames in, ``W * 1920 - 555`` samples out, like the real decoder.

    Sample ``s`` belongs to frame ``s // 1920``, and the last 555 samples of a
    frame also read the next frame. Every value is keyed to its frame's code
    (``_codes`` makes it the absolute frame index + 1, with 0 left for padding)
    and to its offset in the frame, so a dropped, repeated or pad-derived
    sample changes the stream.
    """

    TAIL = 555

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, codes: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        width = codes.shape[-1]
        sample = torch.arange(width * UPSAMPLE - self.TAIL)
        frame, offset = sample // UPSAMPLE, sample % UPSAMPLE
        here = codes[:, 0, frame]
        ahead = torch.where(offset >= UPSAMPLE - self.TAIL, codes[:, 0, (frame + 1).clamp(max=width - 1)], 0)
        value = (here * 1931 + offset * 7 + ahead * 613) % 65521 - 32760
        return (value.float() / 32768).unsqueeze(1)


def _codes(num_frames: int) -> torch.Tensor:
    return (torch.arange(num_frames) + 1)[:, None].expand(num_frames, 16).contiguous()


def test_windows_follow_the_schedule():
    assert Code2WavConfig(codec_chunk_schedule=()).codec_windows() == [25, 50]
    assert Code2WavConfig(codec_chunk_schedule=(4, 8, 16)).codec_windows() == [4, 12, 28, 50]
    assert Code2WavConfig(codec_chunk_schedule=(1, 3, 8, 16)).codec_windows() == [1, 4, 12, 28, 50]
    assert Code2WavConfig(codec_chunk_schedule=(4,)).codec_windows() == [4, 29, 50]


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("num_frames", [1, 4, 5, 11, 12, 24, 25, 26, 28, 49, 50, 51, 53, 75, 76, 120, 125])
def test_streamed_chunks_stitch_into_the_full_decode(schedule, num_frames):
    """First chunk, ramp, steady chunks and the terminal flush, for streams
    shorter than a chunk, ending on a chunk boundary, and one frame past it."""
    decoder = _LookaheadDecoder()
    config = _config(schedule)
    submodule = Code2WavSubmodule(decoder, config)
    codes = _codes(num_frames)

    steps = _stream(submodule, codes)
    assert decoder.calls == _decodes(config.code2wav, num_frames)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert full.numel() == num_frames * UPSAMPLE - decoder.TAIL
    assert [audio.numel() for _, audio, _ in steps] == _expected_samples_per_step(steps, decoder.TAIL)
    streamed = torch.cat([audio for _, audio, _ in steps])
    assert streamed.shape == full.shape
    assert torch.equal(streamed, full)


def test_first_audio_comes_after_the_first_scheduled_chunk():
    """With the ramp, the first window holds 4 frames and emits their audio."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, _config((4, 8, 16)))

    steps = _stream(submodule, _codes(120))

    first_info, first_audio, _ = steps[0]
    assert (first_info.context_items, first_info.num_items) == (0, 4)
    assert first_audio.numel() == 4 * UPSAMPLE - decoder.TAIL
    assert [(info.context_items, info.num_items) for info, _, _ in steps[:5]] == [
        (0, 4), (4, 12), (12, 28), (25, 50), (25, 50),
    ]


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("num_frames", [1, 4, 11, 12, 24, 25, 26, 28, 49, 50, 51, 53, 125])
def test_a_trailing_codec_eos_frame_adds_no_audio_and_no_decode(schedule, num_frames):
    """Served streams end with the Talker's codec_eos frame. A reply that
    ends on a chunk boundary then ends with a flush of context frames and the
    EOS frame, which emits nothing and decodes nothing; one frame short of a
    boundary, the EOS frame is the last new item of a window and then a
    context item of the flush."""
    decoder = _LookaheadDecoder()
    config = _config(schedule)
    submodule = Code2WavSubmodule(decoder, config)
    codes = _codes(num_frames)
    eos = torch.full((1, 16), config.talker.codec_eos_token_id)

    steps = _stream(submodule, torch.cat([codes, eos]))
    assert decoder.calls == _decodes(config.code2wav, num_frames)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert torch.equal(torch.cat([audio for _, audio, _ in steps]), full)
    flush_info, flush_audio, flush_decoded = steps[-1]
    assert flush_info.is_final
    if _ends_on_boundary(config.code2wav, num_frames):
        assert flush_info.num_items - flush_info.context_items == 1
        assert not flush_decoded and flush_audio.numel() == 0


def _boundaries(cfg: Code2WavConfig, count: int) -> list[int]:
    """The first ``count`` stream lengths that end on a chunk boundary."""
    out, delivered, sizes = [], 0, _chunk_sizes(cfg)
    while len(out) < count:
        delivered += next(sizes)
        out.append(delivered)
    return out


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("boundary", range(5))
def test_terminal_flush_without_new_frames_skips_the_decode(schedule, boundary):
    """A stream that ends on a chunk boundary ends with a flush of the
    retained context frames. It emits nothing, so its batch is keyed off the
    captures and runs no decode."""
    config = _config(schedule)
    num_frames = _boundaries(config.code2wav, boundary + 1)[-1]
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, config)

    steps = _stream(submodule, _codes(num_frames))

    flush_info, flush_audio, flush_decoded = steps[-1]
    retained = min(config.code2wav.codec_left_context_frames, num_frames)
    assert flush_info.is_final and flush_info.context_items == flush_info.num_items == retained
    assert not flush_decoded and flush_audio.numel() == 0
    assert decoder.calls == _decodes(config.code2wav, num_frames)
    assert all(decoded for _, _, decoded in steps[:-1])


def test_a_batch_with_any_new_frames_keeps_the_capture():
    """Only a batch made entirely of flushes that may be context-only (a final
    chunk with at most one new item, the codec_eos frame) leaves the capture."""
    submodule = Code2WavSubmodule(_LookaheadDecoder(), Qwen3OmniModelConfig())
    flush = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(50, 25, 25, True)})
    flush_eos = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(50, 25, 26, True)})
    steady = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(25, 25, 50, False)})
    one_new = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(50, 25, 26, False)})
    infos = {"a": None, "b": None}

    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": flush}) == CONTEXT_ONLY
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": flush_eos}) == CONTEXT_ONLY
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": one_new}) == 28
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": steady}) == 50
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush}) == 50
    assert submodule.cg_key_info("code2wav_chunk", infos, None) == 50


def test_small_windows_pad_to_the_full_window_in_large_batches():
    """Small windows are captured for batches of up to SMALL_WINDOW_MAX_BS
    only; the capture configs agree with the keys."""
    submodule = Code2WavSubmodule(_LookaheadDecoder(), _config((4, 8, 16)))
    first = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(0, 0, 4, False)})
    small = submodule.SMALL_WINDOW_MAX_BS
    assert submodule.cg_key_info("code2wav_chunk", dict.fromkeys(range(small)), dict.fromkeys(range(small), first)) == 4
    big = dict.fromkeys(range(small + 1))
    assert submodule.cg_key_info("code2wav_chunk", big, dict.fromkeys(big, first)) == 50

    configs = {cfg.additional_key_info: cfg for cfg in submodule.get_accelerator_graph_configs("cpu")}
    assert sorted(configs) == submodule.windows == [4, 12, 28, 50]
    assert configs[50].capture_batch_sizes == [1, 2, 4, 8, 16, 32] and configs[50].caps_eager_batch_size
    for window in (4, 12, 28):
        assert max(configs[window].capture_batch_sizes) == small and not configs[window].caps_eager_batch_size


def test_mixed_batch_of_a_terminal_flush_and_a_live_window():
    """Request a (50 frames) reaches its context-only flush in the same step
    as request b (76 frames, distinct codes) decodes a full window; b ends a
    step later. The mixed step keeps the capture and decodes both rows; a's
    row emits nothing and both streams stitch into their own full decodes."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, _config(()))
    codes = {"a": _codes(50), "b": _codes(76) + 200}

    steps, keys, _ = _stream_batched(submodule, codes)

    assert [len(steps["a"]), len(steps["b"])] == [3, 4]
    flush_info, flush_audio, flush_decoded = steps["a"][2]
    live_info, live_audio, live_decoded = steps["b"][2]
    assert flush_info.is_final and flush_info.context_items == flush_info.num_items
    assert not live_info.is_final and live_info.num_items - live_info.context_items == 25
    # The mixed step replays the capture; b's flush of one new frame runs
    # eager and still decodes.
    assert keys == [25, 50, 50, CONTEXT_ONLY]
    assert flush_decoded and live_decoded and decoder.calls == 4
    assert flush_audio.numel() == 0 and live_audio.numel() == 25 * UPSAMPLE
    for rid, request_codes in codes.items():
        full = _to_pcm(decoder(request_codes.t()[None], torch.arange(request_codes.shape[0])[None]))[0, 0]
        streamed = torch.cat([audio for _, audio, _ in steps[rid]])
        assert torch.equal(streamed, full), rid


def test_the_vocoder_searches_conv_algorithms_only_in_its_own_calls(monkeypatch):
    """Benchmark mode is on while the vocoder runs and back to what it was
    after, so the worker's other nodes (the encoders) never search."""
    seen = []

    class Spy(_LookaheadDecoder):
        def forward(self, codes, position_ids):
            seen.append(torch.backends.cudnn.benchmark)
            return super().forward(codes, position_ids)

    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    _stream(Code2WavSubmodule(Spy(), Qwen3OmniModelConfig()), _codes(30))

    assert seen and all(seen)
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize("num_requests", [2, 5])
def test_ramp_windows_batch_with_steady_ones(num_requests):
    """Requests at different points of the ramp share steps: each step pads to
    its widest window (the full window past SMALL_WINDOW_MAX_BS rows), and
    every stream still stitches into its full decode."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, _config((4, 8, 16)))
    codes = {f"r{i}": _codes(30 + 17 * i) + 300 * i for i in range(num_requests)}
    delays = {f"r{i}": 2 * i for i in range(num_requests)}

    steps, keys, sizes = _stream_batched(submodule, codes, delays)

    assert all(key in submodule.windows or key == CONTEXT_ONLY for key in keys)
    assert any(size > 1 and key not in (50, CONTEXT_ONLY) for key, size in zip(keys, sizes, strict=True))
    for key, size in zip(keys, sizes, strict=True):
        if size > submodule.SMALL_WINDOW_MAX_BS:
            assert key in (50, CONTEXT_ONLY)
    for rid, request_codes in codes.items():
        full = _to_pcm(decoder(request_codes.t()[None], torch.arange(request_codes.shape[0])[None]))[0, 0]
        streamed = torch.cat([audio for _, audio, _ in steps[rid]])
        assert torch.equal(streamed, full), rid


# ---------------------------------------------------------------------------
# The real Code2Wav architecture, small and with random weights
# ---------------------------------------------------------------------------

def _rms_norm(x, weight, eps=1e-6, out=None):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def _apply_rope_pos_ids(q, k, position_ids, rope_theta=10000.0):
    half = q.shape[-1] // 2
    inv_freq = rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
    angle = position_ids.float()[:, None, None] * inv_freq
    cos, sin = angle.cos(), angle.sin()
    for t in (q, k):
        x1, x2 = t[..., :half].float(), t[..., half:].float()
        t.copy_(torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1))
    return q, k


def _sliding_window_attn(q, k, v, window, scale=None):
    pos = torch.arange(q.shape[1])
    allowed = (pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < window)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed, scale=scale,
    )
    return out.transpose(1, 2)


def _tiny_code2wav_config() -> Code2WavConfig:
    """Narrow and shallow, with the real upsampling: the same transposed convs,
    so the same tail. Two layers of 4-frame attention plus the causal convs see
    about 15 frames back, within the 25 frames of left context, so a streamed
    window decodes its new frames the way the full stream does."""
    return Code2WavConfig(
        codebook_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=64, decoder_dim=96, sliding_window=4,
    )


def _decoder_tail(config: Code2WavConfig) -> int:
    """Samples a decode comes up short: each decoder block's transposed conv
    drops ``rate`` samples, scaled by the upsampling after it."""
    rates = config.upsample_rates
    return sum(rate * math.prod(rates[i + 1:]) for i, rate in enumerate(rates))


@pytest.fixture(scope="module")
def tiny_code2wav():
    with pytest.MonkeyPatch.context() as patch:
        # Triton kernels on GPU; small torch equivalents on CPU.
        patch.setattr(code2wav_module, "rms_norm", _rms_norm)
        patch.setattr(code2wav_module, "apply_rope_pos_ids", _apply_rope_pos_ids)
        patch.setattr(code2wav_module, "sliding_window_attn", _sliding_window_attn)
        torch.manual_seed(0)
        config = _tiny_code2wav_config()
        model = Qwen3OmniMoeCode2Wav(config).eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith(("alpha", "beta")):
                    param.normal_(std=0.3)
        # consolidate() compiles the forward; eager is enough here
        patch.setattr(torch, "compile", lambda fn, **kwargs: fn)
        model.consolidate()
        yield model, config


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("num_frames", [3, 24, 25, 26, 50, 51, 120])
def test_real_decoder_streams_the_full_decode(tiny_code2wav, schedule, num_frames):
    model, config = tiny_code2wav
    tail = _decoder_tail(config)
    assert tail == 8 * 60 + 5 * 12 + 4 * 3 + 3 == 555
    config = dataclasses.replace(config, codec_chunk_schedule=schedule)
    submodule = Code2WavSubmodule(model, Qwen3OmniModelConfig(code2wav=config))
    generator = torch.Generator().manual_seed(num_frames)
    codes = torch.randint(0, config.codebook_size, (num_frames, config.num_quantizers), generator=generator)

    steps = _stream(submodule, codes)
    with torch.no_grad():
        full = _to_pcm(model(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert full.numel() == num_frames * UPSAMPLE - tail
    assert full.float().std() > 300 and (full.abs() < 32767).float().mean() > 0.9   # a signal, not silence or clipping
    streamed = torch.cat([audio for _, audio, _ in steps])
    # Only float rounding differs (convs over a 50-frame window vs over the
    # whole stream). A dropped, repeated or pad-derived sample misaligns
    # everything after it by hundreds of samples.
    common = min(streamed.numel(), full.numel())
    assert (streamed[:common].int() - full[:common].int()).abs().max().item() <= 2
    assert streamed.numel() == full.numel()
    assert [audio.numel() for _, audio, _ in steps] == _expected_samples_per_step(steps, tail)
