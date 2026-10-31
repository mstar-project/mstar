"""Qwen3-Omni Code2Wav streaming: the stitched chunks are one continuous decode.

Each test drives Talker codec frames through the edge's real
``LeftContextChunkPolicy`` and ``StreamBuffer``, then through
``Code2WavSubmodule``'s ``cg_key_info`` / ``prepare_inputs`` / ``preprocess``
/ ``forward_batched`` / ``postprocess``, and compares the concatenated audio
with one decode of the whole stream.

The decoder's ``ConvTranspose1d(2r, stride r, padding r)`` blocks look one
frame ahead, so W frames decode to ``W * 1920 - 555`` samples, not
``W * 1920``. The trim has to account for that or 555 samples go missing at
every chunk boundary.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml

from mstar.model.qwen3_omni.components import code2wav as code2wav_module
from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
from mstar.model.qwen3_omni.config import Code2WavConfig, Qwen3OmniModelConfig
from mstar.model.qwen3_omni.submodules import Code2WavSubmodule, TalkerSubmodule
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine, StopCheck
from mstar.streaming.chunk_policy import LeftContextChunkPolicy, ScheduledLeftContextChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer, StreamChunkInfo

UPSAMPLE = 1920


def _to_pcm(wav: torch.Tensor) -> torch.Tensor:
    """The submodule's float -> int16 conversion."""
    return (wav.clamp(-1, 1) * 32767).to(torch.int16)


def _scheduled_policy(cfg: Code2WavConfig) -> ScheduledLeftContextChunkPolicy:
    """The edge's policy, as ``Qwen3OmniModel`` builds it."""
    return ScheduledLeftContextChunkPolicy(
        schedule=cfg.codec_chunk_schedule, chunk=cfg.codec_chunk_frames, left_context=cfg.codec_left_context_frames,
    )


def _talker_streamed(config: Qwen3OmniModelConfig, frames: torch.Tensor) -> tuple[torch.Tensor, list]:
    """The frames the Talker puts on the Code2Wav stream, out of the frames it
    samples (``[T, 16]``), and its stop check's answer per frame: the
    codec_eos frame is withheld."""
    talker = TalkerSubmodule.__new__(TalkerSubmodule)
    talker.config = config
    info = SimpleNamespace(step_metadata={}, max_tokens=frames.shape[0] + 1, dynamic_loop_iter_counts={})
    kept, results = [], []
    for frame in frames:
        result = talker.check_stop("r", info, {"layer0_codes": [frame[:1]], "codec_tokens": [frame]})
        results.append(result)
        if not (isinstance(result, StopCheck) and "codec_tokens" in result.withheld):
            kept.append(frame)
    return (torch.stack(kept) if kept else frames[:0]), results


def _stream_batched(
    submodule: Code2WavSubmodule, frames: dict[str, torch.Tensor], policy=None, starts: dict[str, int] | None = None,
) -> tuple[dict, list]:
    """Stream each request's ``frames`` ``[T, 16]`` through its own buffer
    with ``policy(cfg)`` (default: 25-frame first chunks, as before the ramp).
    Every step batches the requests that still have a chunk, the way the
    scheduler batches Code2Wav; a request joins at step ``starts[rid]``
    (default 0). Returns, per request, its ``(chunk_info, audio, decoded)``
    per step (``decoded``: whether the step ran the vocoder), and each step's
    ``cg_key_info``."""
    starts = starts or {}
    cfg = submodule.config.code2wav
    if policy is None:
        def policy(cfg):
            return LeftContextChunkPolicy(chunk=cfg.codec_chunk_frames, left_context=cfg.codec_left_context_frames)
    buffers = {}
    for rid, request_frames in frames.items():
        buffers[rid] = StreamBuffer(
            request_id=rid, edge_name="codec_tokens", from_partition="Talker", policy=policy(cfg),
        )
        for i, frame in enumerate(request_frames):
            buffers[rid].pre_read_register(f"t{i}")
            buffers[rid].put(f"t{i}", frame)
        buffers[rid].signal_done(len(request_frames))

    fwd_infos = {rid: SimpleNamespace(request_id=rid, rid_handle=rid) for rid in frames}
    steps = {rid: [] for rid in frames}
    keys = []
    step = -1
    while any(buffer.has_chunk_ready() for buffer in buffers.values()):
        step += 1
        rids = [rid for rid, buffer in buffers.items() if step >= starts.get(rid, 0) and buffer.has_chunk_ready()]
        if not rids:
            continue
        chunks = {rid: buffers[rid].pop_chunk() for rid in rids}
        meta = {rid: InputMetadata(stream_chunks={"codec_tokens": chunk.info}) for rid, chunk in chunks.items()}
        # The engine replays the captured 50-frame graph when the batch's key
        # names a capture (None), and runs the eager forward otherwise.
        key = submodule.cg_key_info(
            "code2wav_chunk", {rid: fwd_infos[rid] for rid in rids}, per_request_input_metadata=meta,
        )
        keys.append(key)
        engine_inputs = ModelInputsFromEngine(
            request_ids=rids, per_request_info={}, per_request_input_metadata=meta, captured=key is None,
        )
        # an empty chunk reaches the node as an edge with no tensors
        items = [
            submodule.prepare_inputs(
                "code2wav_chunk", fwd_infos[rid],
                {"codec_tokens": [] if chunks[rid].data["data"] is None else [chunks[rid].data["data"]]},
                input_metadata=meta[rid],
            )
            for rid in rids
        ]
        assert submodule.can_batch(None, items)
        packed = submodule.preprocess("code2wav_chunk", engine_inputs, items)
        decoded = packed["codec_tokens"].shape[-1] > 0
        with torch.no_grad():
            outputs = submodule.forward_batched("code2wav_chunk", engine_inputs, **packed)
        for rid, item in zip(rids, items, strict=True):
            submodule.postprocess(rid, fwd_infos[rid], outputs[rid], inputs=item)
            steps[rid].append((chunks[rid].info, outputs[rid]["audio_chunk"][0], decoded))
            if chunks[rid].is_final:
                submodule.cleanup_request(rid)
    return steps, keys


def _stream(submodule: Code2WavSubmodule, frames: torch.Tensor, policy=None) -> list[tuple]:
    """``_stream_batched`` for a single request."""
    return _stream_batched(submodule, {"r": frames}, policy)[0]["r"]


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
        self.anchor = torch.nn.Parameter(torch.zeros(()))   # the submodule's device
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


@pytest.mark.parametrize("num_frames", [1, 24, 25, 26, 49, 50, 51, 75, 76, 120, 125])
def test_streamed_chunks_stitch_into_the_full_decode(num_frames):
    """First chunk, steady chunks and the terminal flush, for streams shorter
    than a chunk, a multiple of 25 frames, and 25k + 1 frames."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, Qwen3OmniModelConfig())
    codes = _codes(num_frames)

    steps = _stream(submodule, codes)
    assert decoder.calls == math.ceil(num_frames / 25)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert full.numel() == num_frames * UPSAMPLE - decoder.TAIL
    assert [audio.numel() for _, audio, _ in steps] == _expected_samples_per_step(steps, decoder.TAIL)
    streamed = torch.cat([audio for _, audio, _ in steps])
    assert streamed.shape == full.shape
    assert torch.equal(streamed, full)


@pytest.mark.parametrize("num_frames", [1, 24, 25, 26, 49, 50, 51, 125])
def test_the_talker_withholds_codec_eos_from_the_stream(num_frames):
    """Served replies end with the Talker's codec_eos frame, which its stop
    check withholds, so every stream item is a frame. A reply of k * 25 frames
    ends with a flush of 25 context frames that emits nothing and decodes
    nothing; any other reply's final chunk carries its last frames and emits
    their audio."""
    decoder = _LookaheadDecoder()
    config = Qwen3OmniModelConfig()
    submodule = Code2WavSubmodule(decoder, config)
    codes = _codes(num_frames)
    eos = torch.full((1, 16), config.talker.codec_eos_token_id)

    streamed_frames, results = _talker_streamed(config, torch.cat([codes, eos]))
    assert results[-1] == StopCheck({"talker_decode_loop"}, ("codec_tokens",))
    assert results[:-1] == [set()] * num_frames
    assert torch.equal(streamed_frames, codes)

    steps = _stream(submodule, streamed_frames)
    assert decoder.calls == math.ceil(num_frames / 25)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert torch.equal(torch.cat([audio for _, audio, _ in steps]), full)
    final_info, final_audio, final_decoded = steps[-1]
    assert final_info.is_final
    if num_frames % 25 == 0:
        assert final_info.num_items == final_info.context_items
        assert not final_decoded and final_audio.numel() == 0
    else:
        assert final_info.num_items > final_info.context_items
        assert final_decoded and final_audio.numel() == _expected_samples_per_step(steps, decoder.TAIL)[-1] > 0


def test_a_stream_of_only_codec_eos_ends_with_an_empty_chunk():
    """The Talker's first frame was codec_eos: the stream gets no item, and its
    final chunk reaches Code2Wav with no tensor, decodes nothing and emits no audio."""
    decoder = _LookaheadDecoder()
    config = Qwen3OmniModelConfig()
    submodule = Code2WavSubmodule(decoder, config)
    eos = torch.full((1, 16), config.talker.codec_eos_token_id)

    streamed_frames, _ = _talker_streamed(config, eos)
    steps, keys = _stream_batched(submodule, {"r": streamed_frames})

    ((info, audio, decoded),) = steps["r"]
    assert info == StreamChunkInfo(0, 0, 0, True)
    assert keys == [Code2WavSubmodule._CONTEXT_ONLY_KEY]
    assert not decoded and audio.numel() == 0 and decoder.calls == 0


@pytest.mark.parametrize("num_frames", [25, 50, 125])
def test_terminal_flush_without_new_frames_skips_the_decode(num_frames):
    """A stream of k * 25 frames ends with a flush of the 25 retained context
    frames. It emits nothing, so its batch is keyed off the captures and runs
    no decode."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, Qwen3OmniModelConfig())

    steps = _stream(submodule, _codes(num_frames))

    flush_info, flush_audio, flush_decoded = steps[-1]
    assert flush_info.is_final and flush_info.context_items == flush_info.num_items == 25
    assert not flush_decoded and flush_audio.numel() == 0
    assert decoder.calls == num_frames // 25
    assert all(decoded for _, _, decoded in steps[:-1])


def test_a_batch_with_any_new_frames_keeps_the_capture():
    """Only a batch made entirely of context-only flushes leaves the capture.
    Every item is a frame, so a final chunk with one new item has audio."""
    submodule = Code2WavSubmodule(_LookaheadDecoder(), Qwen3OmniModelConfig())
    flush = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(50, 25, 25, True)})
    final_one_new = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(50, 25, 26, True)})
    steady = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(25, 25, 50, False)})
    one_new = InputMetadata(stream_chunks={"codec_tokens": StreamChunkInfo(50, 25, 26, False)})
    infos = {"a": None, "b": None}

    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": flush}) is not None
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": final_one_new}) is None
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": one_new}) is None
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush, "b": steady}) is None
    assert submodule.cg_key_info("code2wav_chunk", infos, {"a": flush}) is None
    assert submodule.cg_key_info("code2wav_chunk", infos, None) is None


def test_mixed_batch_of_a_terminal_flush_and_a_live_window():
    """Request a (50 frames) reaches its context-only flush in the same step
    as request b (76 frames, distinct codes) decodes a full window; b ends a
    step later. The mixed step keeps the capture and decodes both rows; a's
    row emits nothing and both streams stitch into their own full decodes."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, Qwen3OmniModelConfig())
    codes = {"a": _codes(50), "b": _codes(76) + 200}

    steps, keys = _stream_batched(submodule, codes)

    assert [len(steps["a"]), len(steps["b"])] == [3, 4]
    flush_info, flush_audio, flush_decoded = steps["a"][2]
    live_info, live_audio, live_decoded = steps["b"][2]
    assert flush_info.is_final and flush_info.context_items == flush_info.num_items
    assert not live_info.is_final and live_info.num_items - live_info.context_items == 25
    # The mixed step replays the capture, and so does b's flush of one new frame.
    assert keys == [None, None, None, None]
    assert flush_decoded and live_decoded and decoder.calls == 4
    assert flush_audio.numel() == 0 and live_audio.numel() == 25 * UPSAMPLE
    for rid, request_codes in codes.items():
        full = _to_pcm(decoder(request_codes.t()[None], torch.arange(request_codes.shape[0])[None]))[0, 0]
        streamed = torch.cat([audio for _, audio, _ in steps[rid]])
        assert torch.equal(streamed, full), rid
    # Request handles are reused: a finished handle starts a new stream from scratch.
    again = _stream_batched(submodule, {"a": codes["b"]})[0]["a"]
    assert torch.equal(torch.cat([audio for _, audio, _ in again]), torch.cat([audio for _, audio, _ in steps["b"]]))


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


@pytest.mark.parametrize("num_frames", [24, 25, 26, 50, 51, 120])
def test_real_decoder_streams_the_full_decode(tiny_code2wav, num_frames):
    model, config = tiny_code2wav
    tail = _decoder_tail(config)
    assert tail == 8 * 60 + 5 * 12 + 4 * 3 + 3 == 555
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


# ---------------------------------------------------------------------------
# The first-chunk ramp (the edge's ScheduledLeftContextChunkPolicy)
# ---------------------------------------------------------------------------

def _config(schedule=(4, 8, 13), chunk=25, left_context=25, **code2wav) -> Qwen3OmniModelConfig:
    return Qwen3OmniModelConfig(code2wav=Code2WavConfig(
        codec_chunk_schedule=schedule, codec_chunk_frames=chunk, codec_left_context_frames=left_context, **code2wav,
    ))


def _windows(policy, num_frames: int) -> list[tuple[int, int]]:
    """``(context_items, num_items)`` of every window a stream of
    ``num_frames`` frames pops, the terminal flush included."""
    buffer = StreamBuffer(request_id="r", edge_name="codec_tokens", from_partition="Talker", policy=policy)
    for i in range(num_frames):
        buffer.pre_read_register(f"t{i}")
        buffer.put(f"t{i}", torch.zeros(16))
    buffer.signal_done()
    windows = []
    while buffer.has_chunk_ready():
        chunk = buffer.pop_chunk()
        windows.append((chunk.context_items, chunk.num_items))
    return windows


RAMPS = [
    pytest.param((12, 13), 25, 25, id="default-12-13-25"),
    pytest.param((4, 8, 13), 25, 25, id="4-8-13-25"),
    pytest.param((4, 10), 25, 25, id="4-10-25"),
    pytest.param((5, 10), 25, 25, id="5-10-25"),
    pytest.param((1, 2, 4, 8, 16), 25, 25, id="doubling-ramp"),
    pytest.param((4,), 25, 25, id="4-then-25"),
    pytest.param((8, 10, 13, 16, 20), 25, 25, id="early-audio-8-10-13-16-20"),
    pytest.param((4, 5, 6, 8, 10, 13, 16, 20), 25, 25, id="ramp4g-4-5-6-8-10-13-16-20"),
    # a ramp chunk wider than the steady chunk widens the padded window
    pytest.param((30, 10), 25, 25, id="30-10-25"),
    pytest.param((4, 10), 25, 50, id="4-10-25-context-50"),
    # context wider than the chunk: windows 10, 20, 30, then 35
    pytest.param((), 10, 25, id="10-frame-chunks"),
]


@pytest.mark.parametrize("schedule, chunk, left_context", RAMPS)
@pytest.mark.parametrize("num_frames", [1, 3, 4, 5, 13, 14, 15, 38, 39, 40, 64, 65, 120, 200])
def test_ramped_stream_stitches_into_the_full_decode(schedule, chunk, left_context, num_frames):
    """Streams shorter than the first chunk, on and around every ramp
    boundary, and long enough to reach steady chunks. Every window pads to
    the schedule's widest one."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, _config(schedule, chunk, left_context))
    codes = _codes(num_frames)

    steps = _stream(submodule, codes, _scheduled_policy)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert submodule.full_seqlen == left_context + max((chunk, *schedule))
    assert all(info.num_items <= submodule.full_seqlen for info, _, _ in steps)
    assert steps[0][0].num_items == min((*schedule, chunk)[0], num_frames)   # first audio after the first chunk
    assert [audio.numel() for _, audio, _ in steps] == _expected_samples_per_step(steps, decoder.TAIL)
    streamed = torch.cat([audio for _, audio, _ in steps])
    assert streamed.shape == full.shape
    assert torch.equal(streamed, full)


def test_a_batch_mixes_windows_from_different_ramp_stages():
    """Requests that join a step apart share batches of 4-, 12-, 25- and
    50-frame windows with different contexts. They all pad to one shape and
    replay the capture, and each stitches into its own full decode."""
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, _config())
    codes = {"a": _codes(90), "b": _codes(70) + 200, "c": _codes(3) + 400}

    steps, keys = _stream_batched(submodule, codes, _scheduled_policy, starts={"a": 0, "b": 1, "c": 2})

    assert keys == [None] * len(keys)
    # step 2 batches a's third window (12 + 13), b's second (4 + 8) and c's whole stream
    assert [steps[rid][2 - start][0][1:3] for rid, start in (("a", 0), ("b", 1), ("c", 2))] == [
        (12, 25), (4, 12), (0, 3),
    ]
    for rid, request_codes in codes.items():
        full = _to_pcm(decoder(request_codes.t()[None], torch.arange(request_codes.shape[0])[None]))[0, 0]
        assert torch.equal(torch.cat([audio for _, audio, _ in steps[rid]]), full), rid


@pytest.mark.parametrize(
    "schedule",
    [(12, 13), (8, 10, 13, 16, 20), (4, 5, 6, 8, 10, 13, 16, 20)],
    ids=["default", "early-audio", "ramp4g"],
)
@pytest.mark.parametrize("num_frames", [3, 4, 12, 25, 26, 50, 120])
def test_real_decoder_streams_the_ramp_like_the_full_decode(tiny_code2wav, num_frames, schedule):
    """A ramp's windows carry every frame decoded so far as context (none for
    the first chunk, then all of them until they reach 25), so they see the
    full decode's history, and the steady windows' 25 frames cover this
    decoder's ~15-frame reach. The same holds for the early-audio ramp, whose
    first two windows sum to 18 frames."""
    model, config = tiny_code2wav
    tail = _decoder_tail(config)
    assert config.codec_chunk_schedule == (12, 13)
    config = replace(config, codec_chunk_schedule=schedule)
    submodule = Code2WavSubmodule(model, Qwen3OmniModelConfig(code2wav=config))
    generator = torch.Generator().manual_seed(1000 + num_frames)
    codes = torch.randint(0, config.codebook_size, (num_frames, config.num_quantizers), generator=generator)

    steps = _stream(submodule, codes, _scheduled_policy)
    with torch.no_grad():
        full = _to_pcm(model(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    streamed = torch.cat([audio for _, audio, _ in steps])
    assert streamed.numel() == full.numel() == num_frames * UPSAMPLE - tail
    assert (streamed.int() - full.int()).abs().max().item() <= 2
    assert [audio.numel() for _, audio, _ in steps] == _expected_samples_per_step(steps, tail)


@pytest.mark.parametrize("num_frames", [1, 11, 12, 13, 24, 25, 26, 49, 50, 51])
def test_the_talker_withholds_codec_eos_on_the_default_ramp(num_frames):
    """With the default ramp, a reply that ends on a chunk boundary (12, 25,
    50 frames) ends with a flush of context frames that decodes nothing; one
    frame short, where codec_eos used to be the last new item of a window,
    that window is the final chunk. Either way the stream is the full decode
    of the real frames."""
    decoder = _LookaheadDecoder()
    config = Qwen3OmniModelConfig()
    submodule = Code2WavSubmodule(decoder, config)
    codes = _codes(num_frames)
    eos = torch.full((1, 16), config.talker.codec_eos_token_id)

    streamed_frames, _ = _talker_streamed(config, torch.cat([codes, eos]))
    steps = _stream(submodule, streamed_frames, _scheduled_policy)
    # one decode per window that starts before the last real frame
    assert decoder.calls == sum(1 for start in (0, 12, 25, 50) if start < num_frames)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert torch.equal(torch.cat([audio for _, audio, _ in steps]), full)
    final_info, final_audio, final_decoded = steps[-1]
    assert final_info.is_final
    if num_frames in (12, 25, 50):
        assert not final_decoded and final_audio.numel() == 0
    else:
        assert final_decoded and final_audio.numel() > 0


def test_default_config_ramps_12_13_then_25():
    """The checkpoint carries no streaming fields, so the dataclass defaults
    are what serves: windows of 12 and 12 + 13 frames, then 25 + 25, all
    padded to one 50-frame capture."""
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    cfg = Code2WavConfig.from_dict({"codebook_size": 2048})
    assert (cfg.codec_chunk_schedule, cfg.codec_chunk_frames, cfg.codec_left_context_frames) == ((12, 13), 25, 25)
    assert Qwen3OmniModelConfig().code2wav == cfg
    policy = Qwen3OmniModel._code2wav_chunk_policy(SimpleNamespace(config=Qwen3OmniModelConfig()))
    assert isinstance(policy, ScheduledLeftContextChunkPolicy)
    windows = _windows(policy, 120)
    assert windows == [(0, 12), (12, 25), (25, 50), (25, 50), (25, 50), (25, 45)]
    # from frame 25 on, the windows of 25-frame first chunks
    assert windows[2:] == _windows(LeftContextChunkPolicy(chunk=25, left_context=25), 120)[1:]

    submodule = Code2WavSubmodule(_LookaheadDecoder(), Qwen3OmniModelConfig())
    capture = submodule.get_accelerator_graph_configs("cpu")[0].single_request_inputs.tensor_inputs["codec_tokens"]
    assert submodule.full_seqlen == cfg.codec_max_window == 50 and capture.shape == (16, 50)


@pytest.mark.parametrize("num_frames", list(range(1, 131)))
def test_an_empty_schedule_pops_the_windows_of_25_frame_first_chunks(num_frames):
    """``codec_chunk_schedule=()`` brings back the edge's previous policy
    window for window, terminal flush included."""
    scheduled = _scheduled_policy(Code2WavConfig(codec_chunk_schedule=()))
    assert _windows(scheduled, num_frames) == _windows(LeftContextChunkPolicy(chunk=25, left_context=25), num_frames)


def test_yaml_model_kwargs_set_the_streaming_knobs():
    from mstar.model.qwen3_omni.qwen3_omni_model import apply_code2wav_serving_overrides

    cfg = Code2WavConfig.from_dict({"codec_chunk_schedule": [5, 10]})
    assert cfg.codec_chunk_schedule == (5, 10)
    kwargs = {"codec_chunk_schedule": [], "codec_left_context_frames": 50, "codec_chunk_frames": None, "other": 1}
    apply_code2wav_serving_overrides(cfg, kwargs)
    assert (cfg.codec_chunk_schedule, cfg.codec_chunk_frames, cfg.codec_left_context_frames) == ((), 25, 50)
    assert kwargs == {"codec_chunk_frames": None, "other": 1}
    assert cfg.codec_max_window == 75


EARLY_AUDIO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qwen3omni_colocated_early_audio.yaml"


def test_the_early_audio_config_ramps_in_small_steps_inside_the_default_capture():
    """The AMD deployment's knobs: first audio after 8 frames, no chunk over
    1.3x the one before, and every window still padded to the default's 50
    frames, so the vocoder needs no new capture."""
    from mstar.model.qwen3_omni.qwen3_omni_model import apply_code2wav_serving_overrides

    kwargs = dict(yaml.safe_load(EARLY_AUDIO_CONFIG.read_text())["model_kwargs"])
    cfg = Code2WavConfig()
    apply_code2wav_serving_overrides(cfg, kwargs)

    sizes = (*cfg.codec_chunk_schedule, cfg.codec_chunk_frames)
    assert sizes == (8, 10, 13, 16, 20, 25) and not kwargs
    assert max(b / a for a, b in zip(sizes, sizes[1:], strict=False)) <= 1.3
    assert cfg.codec_max_window == Code2WavConfig().codec_max_window == 50
    assert _windows(_scheduled_policy(cfg), 120) == [
        (0, 8), (8, 18), (18, 31), (25, 41), (25, 45), (25, 50), (25, 50), (25, 28),
    ]


@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("boundary", [8, 18, 31, 47, 67, 92])
def test_the_early_audio_ramp_streams_the_full_decode_around_each_boundary(boundary, offset):
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, _config((8, 10, 13, 16, 20)))
    codes = _codes(boundary + offset)

    steps = _stream(submodule, codes, _scheduled_policy)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(boundary + offset)[None]))[0, 0]

    assert torch.equal(torch.cat([audio for _, audio, _ in steps]), full)


RAMP4G_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "qwen3omni_2gpu_ramp4g.yaml"


def test_the_gentle_ramp4_config_needs_no_more_of_the_talker_than_the_early_audio_one():
    """First audio after 4 frames, while each chunk still arrives in time at
    about 17 frames/s per request: the first chunk plays 555 samples short."""
    from mstar.model.qwen3_omni.qwen3_omni_model import apply_code2wav_serving_overrides

    kwargs = dict(yaml.safe_load(RAMP4G_CONFIG.read_text())["model_kwargs"])
    cfg = Code2WavConfig()
    apply_code2wav_serving_overrides(cfg, kwargs)

    sizes = (*cfg.codec_chunk_schedule, cfg.codec_chunk_frames)
    assert sizes[0] == 4 and not kwargs
    assert cfg.codec_max_window == Code2WavConfig().codec_max_window == 50
    plays_s = [(sizes[0] * UPSAMPLE - cfg.codec_lookahead_samples) / 24000]
    plays_s += [n * UPSAMPLE / 24000 for n in sizes[1:-1]]
    assert max(n / s for n, s in zip(sizes[1:], plays_s, strict=True)) < 17


@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("boundary", [4, 9, 15, 23, 33, 46, 62, 82, 107])
def test_the_gentle_ramp4_schedule_streams_the_full_decode_around_each_boundary(boundary, offset):
    decoder = _LookaheadDecoder()
    submodule = Code2WavSubmodule(decoder, _config((4, 5, 6, 8, 10, 13, 16, 20)))
    codes = _codes(boundary + offset)

    steps = _stream(submodule, codes, _scheduled_policy)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(boundary + offset)[None]))[0, 0]

    assert torch.equal(torch.cat([audio for _, audio, _ in steps]), full)


def test_one_context_frame_covers_the_lookahead_samples():
    """555 lookahead samples fit in one 1920-sample frame. A decoder without
    the 2x2 upsample stack (480 samples a frame) would need two."""
    cfg = Code2WavConfig()
    assert (cfg.codec_lookahead_samples, cfg.codec_min_left_context_frames) == (555, 1)
    assert Code2WavConfig(upsampling_ratios=(), codec_left_context_frames=2).codec_min_left_context_frames == 2
    with pytest.raises(ValueError, match="codec_left_context_frames must be an integer >= 2"):
        Code2WavConfig(upsampling_ratios=(), codec_left_context_frames=1)


def test_zero_context_frames_would_drop_the_lookahead_at_every_boundary():
    """Why the minimum is 1: forced past validation, a context of 0 loses 555
    samples at each of the (4, 8, 13) ramp's three boundaries in 50 frames."""
    cfg = _config()
    cfg.code2wav.codec_left_context_frames = 0
    decoder = _LookaheadDecoder()
    steps = _stream(Code2WavSubmodule(decoder, cfg), _codes(50), _scheduled_policy)
    assert sum(audio.numel() for _, audio, _ in steps) == 50 * UPSAMPLE - decoder.TAIL - 3 * decoder.TAIL == 93780


BAD_KNOBS = [
    pytest.param("codec_left_context_frames", 0, id="context-0"),
    pytest.param("codec_left_context_frames", -1, id="context-negative"),
    pytest.param("codec_left_context_frames", 2.5, id="context-float"),
    pytest.param("codec_left_context_frames", True, id="context-bool"),
    pytest.param("codec_chunk_frames", 0, id="chunk-0"),
    pytest.param("codec_chunk_frames", -25, id="chunk-negative"),
    pytest.param("codec_chunk_frames", 12.5, id="chunk-float"),
    pytest.param("codec_chunk_schedule", [0], id="schedule-0"),
    pytest.param("codec_chunk_schedule", [4, 0, 13], id="schedule-inner-0"),
    pytest.param("codec_chunk_schedule", [4, -8], id="schedule-negative"),
    pytest.param("codec_chunk_schedule", [4.5], id="schedule-float"),
    pytest.param("codec_chunk_schedule", 4, id="schedule-scalar"),
    pytest.param("codec_chunk_schedule", "4,8", id="schedule-string"),
]


@pytest.mark.parametrize("key, value", BAD_KNOBS)
def test_a_bad_streaming_knob_is_rejected_naming_its_key(key, value):
    from mstar.model.qwen3_omni.qwen3_omni_model import apply_code2wav_serving_overrides

    with pytest.raises(ValueError, match=key):
        apply_code2wav_serving_overrides(Code2WavConfig(), {key: value})
    with pytest.raises(ValueError, match=key):
        Code2WavConfig(**{key: value})


@pytest.mark.parametrize("left_context", [1, 2, 25, 50])
@pytest.mark.parametrize("chunk", [1, 25])
@pytest.mark.parametrize("schedule", [(), (1,), (12, 13), (4, 8, 13), (4, 10), (30,), (2, 60)])
@pytest.mark.parametrize("num_frames", [1, 4, 26, 77])
def test_every_accepted_knob_combination_streams_the_full_decode(schedule, chunk, left_context, num_frames):
    """Schedules that sum to less than, exactly or more than the context, and
    contexts from the minimum up: every window's audio lands where the full
    decode puts it, no sample lost or repeated."""
    from mstar.model.qwen3_omni.qwen3_omni_model import apply_code2wav_serving_overrides

    code2wav = Code2WavConfig()
    apply_code2wav_serving_overrides(code2wav, {
        "codec_chunk_schedule": list(schedule), "codec_chunk_frames": chunk, "codec_left_context_frames": left_context,
    })
    decoder = _LookaheadDecoder()
    codes = _codes(num_frames)

    steps = _stream(Code2WavSubmodule(decoder, Qwen3OmniModelConfig(code2wav=code2wav)), codes, _scheduled_policy)
    full = _to_pcm(decoder(codes.t()[None], torch.arange(num_frames)[None]))[0, 0]

    assert [audio.numel() for _, audio, _ in steps] == _expected_samples_per_step(steps, decoder.TAIL)
    assert torch.equal(torch.cat([audio for _, audio, _ in steps]), full)


@pytest.fixture(scope="module")
def long_reach_code2wav():
    """The tiny decoder with a 72-frame attention window at full layer scale:
    it reaches past 25 frames of left context, so the steady windows' audio
    drifts from the full decode at their boundaries, as the real one does."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(code2wav_module, "rms_norm", _rms_norm)
        patch.setattr(code2wav_module, "apply_rope_pos_ids", _apply_rope_pos_ids)
        patch.setattr(code2wav_module, "sliding_window_attn", _sliding_window_attn)
        torch.manual_seed(0)
        config = Code2WavConfig(
            codebook_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            intermediate_size=64, decoder_dim=96, sliding_window=72, layer_scale_initial_scale=1.0,
        )
        model = Qwen3OmniMoeCode2Wav(config).eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith(("alpha", "beta")):
                    param.normal_(std=0.3)
        patch.setattr(torch, "compile", lambda fn, **kwargs: fn)
        model.consolidate()
        yield model, config


def _boundary_snrs(model, config: Code2WavConfig, schedule, seeds=range(4), num_frames=200):
    """SNR within 20 ms of every boundary, against the full decode, split by
    whether the window after it had full left context (steady) or less (ramp)."""
    submodule = Code2WavSubmodule(model, Qwen3OmniModelConfig(code2wav=replace(config, codec_chunk_schedule=schedule)))
    steady, ramp = [], []
    for seed in seeds:
        codes = torch.randint(0, config.codebook_size, (num_frames, 16), generator=torch.Generator().manual_seed(seed))
        steps = _stream(submodule, codes, _scheduled_policy)
        with torch.no_grad():
            full = _to_pcm(model(codes.t()[None], torch.arange(num_frames)[None]))[0, 0].double()
        audio = torch.cat([audio for _, audio, _ in steps]).double()
        signal = full.pow(2).mean().item()
        position = 0
        for info, chunk_audio, _ in steps:
            if info.context_items and position + 480 <= full.numel():
                span = slice(position - 480, position + 480)
                error = (audio[span] - full[span]).pow(2).mean().item()
                snr = 10 * math.log10(signal / max(error, 1e-12))
                (steady if info.context_items == config.codec_left_context_frames else ramp).append(snr)
            position += chunk_audio.numel()
    return steady, ramp


def test_the_default_ramp_plays_the_audio_of_25_frame_first_chunks(long_reach_code2wav):
    """The ramp sums to the left context, so its windows see the whole stream
    so far and every later window is the one 25-frame first chunks pop: the
    audio is the same, on a decoder whose boundaries do lose history."""
    model, config = long_reach_code2wav
    ramp = Code2WavSubmodule(model, Qwen3OmniModelConfig(code2wav=config))
    first_chunk_25 = Code2WavSubmodule(model, Qwen3OmniModelConfig(code2wav=replace(config, codec_chunk_schedule=())))
    for num_frames in (3, 12, 25, 40, 50, 120):
        codes = torch.randint(0, 64, (num_frames, 16), generator=torch.Generator().manual_seed(num_frames))
        streamed = [
            torch.cat([audio for _, audio, _ in _stream(sub, codes, _scheduled_policy)])
            for sub in (ramp, first_chunk_25)
        ]
        assert (streamed[0].int() - streamed[1].int()).abs().max().item() <= 1, num_frames


@pytest.mark.parametrize("schedule", [(4, 10), (8, 10, 13, 16, 20)], ids=["4-10", "early-audio"])
def test_ramp_boundaries_are_no_worse_than_25_frame_chunks(long_reach_code2wav, schedule):
    """A ramp that does not sum to the context, (4, 10) or the early-audio
    (8, 10, 13, 16, 20), shifts the boundaries off the 25-frame grid. At
    steady state both schedules still start each window on 25 frames of
    history, so their boundaries match (33 dB median here); the ramp's own
    boundaries (two per stream, before its windows reach 25 frames of
    context) see the whole stream and match the full decode."""
    model, config = long_reach_code2wav
    base_steady, base_ramp = _boundary_snrs(model, config, ())
    steady, ramp = _boundary_snrs(model, config, schedule)

    assert not base_ramp and len(base_steady) >= 20 and len(steady) >= 20
    assert 25 < statistics.median(base_steady) < 45   # the truncated history shows
    assert statistics.median(steady) >= statistics.median(base_steady) - 1
    assert min(steady) >= min(base_steady) - 2
    assert len(ramp) == 8 and min(ramp) > 80
