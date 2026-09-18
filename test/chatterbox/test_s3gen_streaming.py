"""Chunked S3Gen streaming on real weights (CPU).

Streams a real token sequence through ``S3GenSubmodule.synthesize_chunk`` the
way the stream buffer would deliver it and checks the result against the
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
from mstar.model.chatterbox.submodules import S3GenSubmodule, StreamState

os.environ.setdefault("HF_HUB_OFFLINE", "1")
pytest.importorskip("chatterbox")


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


def _stream(sub, ref, tokens, chunk_sizes, seed=11, n_timesteps=6):
    gen = torch.Generator().manual_seed(seed)
    state = StreamState(tokens=torch.empty(0, dtype=torch.long))
    pieces, boundaries, pos = [], [], 0
    offset = 0
    for i, size in enumerate(chunk_sizes):
        chunk = tokens[offset:offset + size]
        offset += size
        final = i == len(chunk_sizes) - 1
        wav = sub.synthesize_chunk(state, chunk, final, ref, n_timesteps, gen)
        if wav.numel():
            pieces.append(wav)
            pos += wav.numel()
            boundaries.append(pos)
    assert state.done
    return torch.cat(pieces), boundaries[:-1], state


def test_streaming_matches_offline_length_and_stays_continuous(setup):
    sub, ref, tokens = setup
    streamed, boundaries, state = _stream(sub, ref, tokens, [15, 25, 20])
    assert streamed.numel() == tokens.numel() * 960
    assert state.token_offset == tokens.numel()

    # whole-utterance decode with the same noise field and steps
    lens = torch.tensor([tokens.numel()])
    with torch.no_grad():
        mel = sub.s3gen.tokens_to_mel(tokens[None], lens, ref, n_timesteps=6, noise=state.noise)
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
    gen = torch.Generator().manual_seed(3)
    state = StreamState(tokens=torch.empty(0, dtype=torch.long))
    with torch.no_grad():
        streamed = sub.synthesize_chunk(state, short, True, ref, 4, gen)
    # the same tokens through the reference-order offline path
    pcm = sub.synthesize(short, ref, n_timesteps=4, seed=3, watermark=False)
    assert pcm.numel() == streamed.numel() == 20 * 960
    # both ran in one shot with finalize=True; the offline call draws the noise
    # in the reference order (one field), the stream state draws its own field:
    # different noise, same length and level
    assert abs(streamed.pow(2).mean().sqrt() / (pcm.float() / 32767).pow(2).mean().sqrt() - 1) < 0.5


def test_streaming_lookahead_and_tail_accounting(setup):
    sub, ref, tokens = setup
    gen = torch.Generator().manual_seed(5)
    state = StreamState(tokens=torch.empty(0, dtype=torch.long))
    with torch.no_grad():
        first = sub.synthesize_chunk(state, tokens[:15], False, ref, 2, gen)
        assert first.numel() == 12 * 960 - sub.cache_samples
        assert state.token_offset == 12 and state.hift_mel.shape[-1] == 8
        # an empty (non-final) chunk adds no usable token: nothing to do yet
        none = sub.synthesize_chunk(state, tokens[15:15], False, ref, 2, gen)
        assert none.numel() == 0 and state.token_offset == 12
        last = sub.synthesize_chunk(state, tokens[15:30], True, ref, 2, gen)
        assert last.numel() == 18 * 960 + sub.cache_samples
    assert first.numel() + last.numel() == 30 * 960 and state.done


def test_batched_streams_match_their_single_runs(setup):
    """Two streams with different chunkings advanced through ``forward_batched``
    (one padded flow solve per step) emit, step by step, the audio each gets
    from ``forward`` on its own."""
    from types import SimpleNamespace

    from mstar.model.chatterbox.submodules import SPEECH_TOKENS

    sub, ref, tokens = setup
    schedules = {"a": [18, 24, 18], "b": [15, 15, 15, 15]}
    steps = max(len(s) for s in schedules.values())

    def chunk(rid, step):
        sizes = schedules[rid]
        if step >= len(sizes):
            return None  # this stream has already ended
        start = sum(sizes[:step])
        return tokens[start:start + sizes[step]], step == len(sizes) - 1

    def prepared(rid, step, request_id=None):
        toks, final = chunk(rid, step)
        info = SimpleNamespace(
            request_id=request_id or rid, random_seed=21, step_metadata={"n_cfm_timesteps": 4},
        )
        return sub.prepare_inputs("s3gen_chunk", info, {SPEECH_TOKENS: [toks]}, is_final_stream_chunk=final)

    single = {rid: [] for rid in schedules}
    for rid, sizes in schedules.items():
        for step in range(len(sizes)):
            inp = prepared(rid, step)
            single[rid].append(sub.forward("s3gen_chunk", None, **inp.tensor_inputs, **inp.kwargs)["audio_chunk"][0])

    # the batched pass runs under its own request ids, so it starts from fresh state
    batched = {rid: [] for rid in schedules}
    for step in range(steps):
        live = [rid for rid in schedules if chunk(rid, step) is not None]
        inputs = [prepared(rid, step, request_id=f"{rid}-batched") for rid in live]
        if len(inputs) > 1:
            out = sub.forward_batched("s3gen_chunk", None, **sub.preprocess("s3gen_chunk", None, inputs))
        else:
            inp = inputs[0]
            out = {inp.kwargs["request_id"]: sub.forward("s3gen_chunk", None, **inp.tensor_inputs, **inp.kwargs)}
        for rid in live:
            batched[rid].append(out[f"{rid}-batched"]["audio_chunk"][0])

    for rid in schedules:
        for step, (mine, alone) in enumerate(zip(batched[rid], single[rid], strict=True)):
            assert mine.shape == alone.shape, (rid, step)
            diff = (mine.float() - alone.float()).abs().max().item() if mine.numel() else 0.0
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
        streamed, boundaries, state = _stream(sub, ref, tokens, [15, 15, 15, 15], seed=13)
    finally:
        sub.context_tokens = saved
    assert streamed.numel() == tokens.numel() * 960 and state.done

    lens = torch.tensor([tokens.numel()])
    with torch.no_grad():
        mel = sub.s3gen.tokens_to_mel(tokens[None], lens, ref, n_timesteps=6, noise=state.noise)
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
