"""CPU parity of the M* S3Gen components against the reference package.

Loads the real ``s3gen.safetensors`` / ``s3gen_meanflow.safetensors`` from the
HF cache and compares, on the built-in voice, the flow encoder output, the
flow-matched mel, the vocoded waveform and the reference embedding with
``chatterbox.models.s3gen.S3Gen``. Skipped without the package or the weights.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("chatterbox")
from safetensors.torch import load_file  # noqa: E402

from mstar.model.chatterbox.components.s3gen import ReferenceConditioning, S3Gen  # noqa: E402
from mstar.model.chatterbox.config import S3GenConfig  # noqa: E402
from mstar.model.chatterbox.loader import iter_weights, resolve_snapshot  # noqa: E402

NUM_TOKENS = 40


def _snapshot(repo: str) -> Path | None:
    try:
        return Path(resolve_snapshot(repo))
    except Exception:
        return None


VARIANTS = {
    "standard": ("ResembleAI/chatterbox", "s3gen.safetensors", False, 10),
    "meanflow": ("ResembleAI/chatterbox-turbo", "s3gen_meanflow.safetensors", True, 2),
}


def _load_pair(variant: str):
    repo, filename, meanflow, n_steps = VARIANTS[variant]
    snap = _snapshot(repo)
    if snap is None or not (snap / filename).is_file():
        pytest.skip(f"{repo}/{filename} is not in the HF cache")
    from chatterbox.models.s3gen import S3Gen as RefS3Gen

    ref = RefS3Gen(meanflow=meanflow)
    ref.load_state_dict(load_file(str(snap / filename)), strict=False)
    ref.eval()

    config = S3GenConfig.meanflow_distilled() if meanflow else S3GenConfig.standard()
    mine = S3Gen(config)
    mine.load_weights(iter_weights(snap / filename))
    mine.eval()

    conds = torch.load(snap / "conds.pt", map_location="cpu", weights_only=True)["gen"]
    ref_dict = {
        "prompt_token": conds["prompt_token"],
        "prompt_token_len": conds["prompt_token_len"],
        "prompt_feat": conds["prompt_feat"],
        "prompt_feat_len": conds["prompt_feat_len"],
        "embedding": conds["embedding"],
    }
    mine_ref = ReferenceConditioning(
        prompt_tokens=conds["prompt_token"], prompt_feat=conds["prompt_feat"], embedding=conds["embedding"],
    )
    return ref, mine, ref_dict, mine_ref, n_steps


@pytest.fixture(scope="module", params=list(VARIANTS))
def pair(request):
    return request.param, _load_pair(request.param)


def _tokens(mine_ref: ReferenceConditioning) -> torch.Tensor:
    # a slice of the built-in voice's own prompt tokens: valid codes, realistic statistics
    return mine_ref.prompt_tokens[:, 20:20 + NUM_TOKENS].clone()


def _snr_db(ref: torch.Tensor, other: torch.Tensor) -> float:
    err = (ref - other).pow(2).sum()
    return float(10 * torch.log10(ref.pow(2).sum() / err.clamp_min(1e-20)))


def test_flow_encoder_mu_matches(pair):
    variant, (ref, mine, ref_dict, mine_ref, _) = pair
    tokens = _tokens(mine_ref)
    full = torch.cat([ref_dict["prompt_token"], tokens], dim=1)
    lens = torch.tensor([full.shape[1]])
    with torch.no_grad():
        mask = torch.ones(1, full.shape[1], 1)
        h = ref.flow.input_embedding(full) * mask
        h, h_masks = ref.flow.encoder(h, lens)
        ref_mu = ref.flow.encoder_proj(h).transpose(1, 2)
        my_mu, my_masks = mine.flow_encoder(full, lens)
    assert my_mu.shape == ref_mu.shape == (1, 80, 2 * full.shape[1])
    assert torch.equal(my_masks, h_masks)
    diff = (my_mu - ref_mu).abs().max().item()
    print(f"[{variant}] flow encoder mu max abs diff {diff:.3e}")
    assert diff < 1e-4


def test_tokens_to_mel_matches(pair):
    variant, (ref, mine, ref_dict, mine_ref, n_steps) = pair
    tokens = _tokens(mine_ref)
    torch.manual_seed(1234)
    with torch.no_grad():
        ref_mel = ref.flow_inference(tokens, ref_dict=dict(ref_dict), n_cfm_timesteps=n_steps, finalize=True)
    torch.manual_seed(1234)
    my_mel = mine.tokens_to_mel(tokens, torch.tensor([NUM_TOKENS]), mine_ref, n_timesteps=n_steps)
    assert my_mel.shape == ref_mel.shape == (1, 80, 2 * NUM_TOKENS)
    diff = (my_mel - ref_mel).abs().max().item()
    print(f"[{variant}] mel max abs diff {diff:.3e} (ref mel range {ref_mel.min():.2f}..{ref_mel.max():.2f})")
    assert diff < 1e-3


def test_mel_to_wav_matches(pair):
    variant, (ref, mine, ref_dict, mine_ref, n_steps) = pair
    tokens = _tokens(mine_ref)
    torch.manual_seed(7)
    with torch.no_grad():
        mel = ref.flow_inference(tokens, ref_dict=dict(ref_dict), n_cfm_timesteps=n_steps, finalize=True)
    torch.manual_seed(99)
    with torch.no_grad():
        ref_wav, _ = ref.hift_inference(mel, None)
    torch.manual_seed(99)
    my_wav = mine.mel_to_wav(mel, fade_in=False)
    assert my_wav.shape == ref_wav.shape == (1, 2 * NUM_TOKENS * 480)
    diff = (my_wav - ref_wav).abs().max().item()
    snr = _snr_db(ref_wav, my_wav)
    print(f"[{variant}] wav max abs diff {diff:.3e}, SNR {snr:.1f} dB")
    assert diff < 1e-3
    assert snr > 60
    assert torch.allclose(mine.trim_fade, ref.trim_fade)


def test_end_to_end_inference_matches(pair):
    variant, (ref, mine, ref_dict, mine_ref, n_steps) = pair
    tokens = _tokens(mine_ref)
    torch.manual_seed(2024)
    with torch.no_grad():
        ref_wav, _ = ref.inference(speech_tokens=tokens, ref_dict=dict(ref_dict), n_cfm_timesteps=n_steps)
    torch.manual_seed(2024)
    mel = mine.tokens_to_mel(tokens, torch.tensor([NUM_TOKENS]), mine_ref, n_timesteps=n_steps)
    my_wav = mine.mel_to_wav(mel)
    diff = (my_wav - ref_wav).abs().max().item()
    print(f"[{variant}] end-to-end wav max abs diff {diff:.3e}, SNR {_snr_db(ref_wav, my_wav):.1f} dB")
    assert diff < 1e-3


def test_batched_tokens_to_mel_matches_single(pair):
    """Two requests of different lengths in one padded batch reproduce the
    single-request mels (same noise per request)."""
    variant, (ref, mine, ref_dict, mine_ref, n_steps) = pair
    tokens = _tokens(mine_ref)
    a, b = tokens, tokens[:, :NUM_TOKENS - 7]
    prompt = mine_ref.num_prompt_tokens
    gen = torch.Generator().manual_seed(5)
    noise_a = torch.randn(1, 80, 2 * (prompt + a.shape[1]), generator=gen)
    noise_b = torch.randn(1, 80, 2 * (prompt + b.shape[1]), generator=gen)
    mel_a = mine.tokens_to_mel(a, torch.tensor([a.shape[1]]), mine_ref, n_timesteps=n_steps, noise=noise_a)
    mel_b = mine.tokens_to_mel(b, torch.tensor([b.shape[1]]), mine_ref, n_timesteps=n_steps, noise=noise_b)

    padded = torch.zeros(2, a.shape[1], dtype=torch.long)
    padded[0] = a[0]
    padded[1, : b.shape[1]] = b[0]
    noise = torch.zeros(2, 80, 2 * (prompt + a.shape[1]))
    noise[0] = noise_a[0]
    noise[1, :, : noise_b.shape[-1]] = noise_b[0]
    mel = mine.tokens_to_mel(padded, torch.tensor([a.shape[1], b.shape[1]]), mine_ref, n_timesteps=n_steps, noise=noise)
    diff_a = (mel[0] - mel_a[0]).abs().max().item()
    diff_b = (mel[1, :, : 2 * b.shape[1]] - mel_b[0]).abs().max().item()
    print(f"[{variant}] batched vs single: {diff_a:.3e} / {diff_b:.3e}")
    assert diff_a < 1e-3 and diff_b < 1e-3


def test_tokens_to_mel_rows_matches_single_per_row(pair):
    """Rows with their own reference (different prompt lengths), look-ahead
    state and noise, solved as one batch, reproduce each row's single-request
    mel: a final row, a streaming row with the look-ahead cut, and a row with
    a shorter prompt."""
    from mstar.model.chatterbox.components.s3gen import FlowRow, ReferenceConditioning

    variant, (ref, mine, ref_dict, mine_ref, n_steps) = pair
    tokens = _tokens(mine_ref)[0]
    short_ref = ReferenceConditioning(
        prompt_tokens=mine_ref.prompt_tokens[:, :-11],
        prompt_feat=mine_ref.prompt_feat[:, :-22],
        embedding=mine_ref.embedding,
    )
    gen = torch.Generator().manual_seed(9)
    rows = [
        (tokens, mine_ref, True),
        (tokens[: NUM_TOKENS - 5], mine_ref, False),
        (tokens[: NUM_TOKENS - 9], short_ref, True),
    ]
    flow_rows, singles = [], []
    for toks, r, final in rows:
        noise = torch.randn(1, 80, 2 * (r.num_prompt_tokens + toks.numel()), generator=gen)
        flow_rows.append(FlowRow(tokens=toks, ref=r, finalize=final, noise=noise))
        singles.append(mine.tokens_to_mel(
            toks[None], torch.tensor([toks.numel()]), r, n_timesteps=n_steps, noise=noise, finalize=final,
        ))

    mels = mine.tokens_to_mel_rows(flow_rows, n_timesteps=n_steps)

    for (toks, r, final), mel, single in zip(rows, mels, singles, strict=True):
        usable = toks.numel() - (0 if final else 3)
        assert mel.shape == (1, 80, 2 * usable) == single.shape
        diff = (mel - single).abs().max().item()
        print(f"[{variant}] rows vs single (final={final}, prompt={r.num_prompt_tokens}): {diff:.3e}")
        assert diff < 1e-3


def test_embed_reference_matches(pair):
    variant, (ref, mine, _, _, _) = pair
    sr = 24000
    t = torch.arange(3 * sr) / sr
    gen = torch.Generator().manual_seed(11)
    wav24 = 0.3 * torch.sin(2 * math.pi * 220 * t) + 0.15 * torch.sin(2 * math.pi * 660 * t + 0.3)
    wav24 = (wav24 * (0.6 + 0.4 * torch.sin(2 * math.pi * 3 * t)) + 0.02 * torch.randn(len(t), generator=gen)).float()
    with torch.no_grad():
        ref_dict = ref.embed_ref(wav24, sr)
        import torchaudio

        wav16 = torchaudio.functional.resample(wav24[None], sr, 16000)
        tokens16, _ = ref.tokenizer(wav16)
        mine_ref = mine.embed_reference(wav24[None], wav16, tokens16)
    assert torch.equal(mine_ref.prompt_tokens, ref_dict["prompt_token"])
    assert mine_ref.prompt_feat.shape == ref_dict["prompt_feat"].shape
    mel_diff = (mine_ref.prompt_feat - ref_dict["prompt_feat"]).abs().max().item()
    xvec_diff = (mine_ref.embedding - ref_dict["embedding"]).abs().max().item()
    print(f"[{variant}] embed_reference: mel diff {mel_diff:.3e}, x-vector diff {xvec_diff:.3e}, "
          f"tokens {mine_ref.num_prompt_tokens}, mel frames {mine_ref.prompt_feat.shape[1]}")
    assert mel_diff < 1e-4
    assert xvec_diff < 1e-4
