"""Chunked S3Gen streaming on real weights (CPU).

Streams a real token sequence through the S3Gen node the way the stream
buffer delivers it, each chunk with every token before it and the stream's
state in its Value between chunks, and checks the result against the
whole-utterance decode of the same tokens with the same noise field: same
length, no discontinuity at chunk boundaries, and a bounded deviation (the
streamed mel is re-estimated with growing context, so it is close but not
identical). The offline policy stays bit-exact with the reference path.
"""

from __future__ import annotations

import os

import pytest
import torch

from mstar.model.chatterbox.components.s3gen import ReferenceConditioning, S3Gen
from mstar.model.chatterbox.config import ChatterboxConfig
from mstar.model.chatterbox.loader import iter_weights, resolve_snapshot
from mstar.model.chatterbox.submodules import S3GenSubmodule

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# a submodule of the reference: mstar/model/chatterbox also answers to `chatterbox` once pytest
# puts mstar/model on sys.path for the model-local tests
pytest.importorskip("chatterbox.models")
from fake_resources import FakeS3GenStream  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    try:
        snapshot = resolve_snapshot("ResembleAI/chatterbox")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"checkpoint not available: {exc}")
    config = ChatterboxConfig.chatterbox()
    conds = torch.load(f"{snapshot}/conds.pt", map_location="cpu", weights_only=True)["gen"]
    s3gen = S3Gen(config.s3gen)
    s3gen.load_weights(iter_weights(f"{snapshot}/s3gen.safetensors"))
    s3gen.eval()
    ref = ReferenceConditioning(
        prompt_tokens=conds["prompt_token"], prompt_feat=conds["prompt_feat"], embedding=conds["embedding"],
    )
    sub = S3GenSubmodule(s3gen, s3_tokenizer=None, config=config, builtin_voice=ref, watermarker=None)
    tokens = conds["prompt_token"][0, :60].clone()
    return sub, ref, tokens


def _stream(sub, tokens, chunk_sizes, seed=11, n_timesteps=6):
    stream = FakeS3GenStream(sub)
    pieces, boundaries, pos = [], [], 0
    offset = 0
    for i, size in enumerate(chunk_sizes):
        offset += size
        final = i == len(chunk_sizes) - 1
        pcm = stream.step({0: stream.prepare(0, tokens[:offset], final, seed, n_timesteps)})[0]
        if pcm.numel():
            pieces.append(pcm.float() / 32767)
            pos += pcm.numel()
            boundaries.append(pos)
    held = stream.held(0)
    assert held["done"]
    return torch.cat(pieces), boundaries[:-1], held


def _noise_field(sub, ref, seed):
    """A stream's noise field: the first draw of a generator seeded with its seed."""
    return sub._noise_field(ref.num_prompt_tokens * sub.frames_per_token, torch.Generator().manual_seed(seed))


def test_streaming_matches_offline_length_and_stays_continuous(setup):
    sub, ref, tokens = setup
    streamed, boundaries, held = _stream(sub, tokens, [15, 25, 20])
    assert streamed.numel() == tokens.numel() * 960
    assert held["token_offset"] == tokens.numel()

    # whole-utterance decode with the same noise field and steps
    lens = torch.tensor([tokens.numel()])
    with torch.no_grad():
        mel = sub.s3gen.tokens_to_mel(tokens[None], lens, ref, n_timesteps=6, noise=_noise_field(sub, ref, 11))
        offline = sub.s3gen.mel_to_wav(mel, generator=torch.Generator().manual_seed(11))[0]
    assert offline.numel() == streamed.numel()

    # The vocoder draws a random harmonic phase per call, so the two
    # waveforms are not sample-aligned; compare their log-mel spectrograms
    # (what the flow decoder actually produced) and their level.
    with torch.no_grad():
        mel_stream = sub.s3gen.mel_extractor(streamed[None])[0]
        mel_offline = sub.s3gen.mel_extractor(offline[None])[0]
    mel_err = (mel_stream - mel_offline).abs()
    mel_corr = torch.corrcoef(torch.stack([mel_stream.flatten(), mel_offline.flatten()]))[0, 1]
    rms_ratio = streamed.pow(2).mean().sqrt() / offline.pow(2).mean().sqrt()
    print(f"[s3gen streaming] log-mel mean|d|={mel_err.mean():.3f} p99|d|={mel_err.flatten().quantile(0.99):.3f} "
          f"corr={mel_corr:.4f} rms ratio={rms_ratio:.3f}")
    assert torch.isfinite(streamed).all()
    assert mel_corr > 0.97 and mel_err.mean() < 0.5
    assert abs(rms_ratio - 1) < 0.25

    # no click at a chunk boundary: the sample-to-sample jump there is within
    # the jumps seen everywhere else in the signal
    jumps = (streamed[1:] - streamed[:-1]).abs()
    typical = jumps.quantile(0.999)
    for b in boundaries:
        assert jumps[b - 3:b + 3].max() <= max(2 * typical, 0.05), (b, jumps[b - 3:b + 3].max(), typical)


def test_single_final_chunk_is_the_offline_path(setup):
    sub, ref, tokens = setup
    short = tokens[:20]
    stream = FakeS3GenStream(sub)
    pcm = stream.step({0: stream.prepare(0, short, True, seed=3, n_timesteps=4)})[0]
    # the reference's order: the flow's noise, then the vocoder's, from one generator seeded with the seed
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        mel = sub.s3gen.tokens_to_mel(short[None], torch.tensor([20]), ref, n_timesteps=4, generator=gen)
        offline = (sub.s3gen.mel_to_wav(mel, generator=gen)[0].clamp(-1, 1) * 32767).to(torch.int16)
    assert pcm.numel() == offline.numel() == 20 * 960
    # the node pads the solve to its frame bucket, which moves the valid frames by float noise only
    assert (pcm.float() - offline.float()).abs().max() <= 1.0


def test_streaming_lookahead_and_tail_accounting(setup):
    sub, ref, tokens = setup
    stream = FakeS3GenStream(sub)
    with torch.no_grad():
        first = stream.step({0: stream.prepare(0, tokens[:15], False, seed=5, n_timesteps=2)})[0]
        assert first.numel() == 12 * 960 - sub.cache_samples
        assert stream.held(0)["token_offset"] == 12
        # a chunk with no new token adds no usable token: nothing to do yet
        none = stream.step({0: stream.prepare(0, tokens[:15], False, seed=5, n_timesteps=2)})[0]
        assert none.numel() == 0 and stream.held(0)["token_offset"] == 12
        last = stream.step({0: stream.prepare(0, tokens[:30], True, seed=5, n_timesteps=2)})[0]
        assert last.numel() == 18 * 960 + sub.cache_samples
    assert first.numel() + last.numel() == 30 * 960 and stream.held(0)["done"]


def test_batched_streams_match_their_single_runs(setup):
    """Two streams with different chunkings advanced in one step (one padded
    flow solve per step) emit, step by step, the audio each gets in a step of
    its own."""
    sub, ref, tokens = setup
    schedules = {"a": [18, 24, 18], "b": [15, 15, 15, 15]}
    steps = max(len(s) for s in schedules.values())

    def window(rid, step):
        """Every token delivered by ``step``, and whether that chunk ends the stream."""
        sizes = schedules[rid]
        return tokens[: sum(sizes[: step + 1])], step == len(sizes) - 1

    alone = FakeS3GenStream(sub)
    single = {rid: [] for rid in schedules}
    for rid, sizes in schedules.items():
        for step in range(len(sizes)):
            prepared = alone.prepare(rid, *window(rid, step), seed=21, n_timesteps=4)
            single[rid].append(alone.step({rid: prepared})[rid])

    # the batched pass runs over a stream Value of its own, so it starts from fresh state
    together = FakeS3GenStream(sub)
    batched = {rid: [] for rid in schedules}
    for step in range(steps):
        live = [rid for rid in schedules if step < len(schedules[rid])]
        prepared = {rid: together.prepare(rid, *window(rid, step), seed=21, n_timesteps=4) for rid in live}
        for rid, pcm in together.step(prepared).items():
            batched[rid].append(pcm)

    for rid in schedules:
        for step, (mine, alone_pcm) in enumerate(zip(batched[rid], single[rid], strict=True)):
            assert mine.shape == alone_pcm.shape, (rid, step)
            diff = (mine.float() - alone_pcm.float()).abs().max().item() if mine.numel() else 0.0
            print(f"[{rid} step {step}] batched vs single: {diff:.0f} LSB over {mine.numel()} samples")
            # float32 reassociation in the padded batch, amplified by the vocoder;
            # measured 1-6 LSB on CPU
            assert diff <= 16.0, (rid, step, diff)


def test_context_window_keeps_streaming_close_to_offline(setup):
    """A bounded left-context window (``stream_context_tokens``) makes each
    chunk's flow solve cost constant; the streamed mel must stay close to the
    whole-utterance decode, and the chunks must still join without clicks."""
    sub, ref, tokens = setup
    saved = sub.context_tokens
    sub.context_tokens = 20
    try:
        streamed, boundaries, held = _stream(sub, tokens, [15, 15, 15, 15], seed=13)
    finally:
        sub.context_tokens = saved
    assert streamed.numel() == tokens.numel() * 960 and held["done"]

    lens = torch.tensor([tokens.numel()])
    with torch.no_grad():
        mel = sub.s3gen.tokens_to_mel(tokens[None], lens, ref, n_timesteps=6, noise=_noise_field(sub, ref, 13))
        offline = sub.s3gen.mel_to_wav(mel, generator=torch.Generator().manual_seed(13))[0]
        mel_stream = sub.s3gen.mel_extractor(streamed[None])[0]
        mel_offline = sub.s3gen.mel_extractor(offline[None])[0]
    mel_err = (mel_stream - mel_offline).abs()
    mel_corr = torch.corrcoef(torch.stack([mel_stream.flatten(), mel_offline.flatten()]))[0, 1]
    print(f"[s3gen streaming, context 20] log-mel mean|d|={mel_err.mean():.3f} corr={mel_corr:.4f}")
    assert torch.isfinite(streamed).all()
    assert mel_corr > 0.95 and mel_err.mean() < 0.6
    jumps = (streamed[1:] - streamed[:-1]).abs()
    typical = jumps.quantile(0.999)
    for b in boundaries:
        assert jumps[b - 3:b + 3].max() <= max(2 * typical, 0.05), (b, jumps[b - 3:b + 3].max(), typical)
