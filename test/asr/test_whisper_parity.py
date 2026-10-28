"""Real-weight parity for Whisper (GPU): native encoder vs HF, served greedy text vs HF generate.

Skipped without CUDA, the checkpoint in the local HF cache, or the benchmark
set (``python -m benchmark.asr_data``, under ``MSTAR_ASR_DATA``). Compares
against ``transformers``' ``WhisperForConditionalGeneration`` on real audio:

  * encoder: ``max |mstar - hf|`` over the 1500x1280 window, both in the
    serving dtype;
  * end to end: ``mstar serve`` with the shipped config against HF
    ``generate``, greedy with the language forced, on single-window files;
    the texts must match (see ``_served.assert_greedy_parity``). Needs the
    GPU to itself (the server takes most of it);
  * word timestamps: the decoder's alignment heads against HF's token
    timestamps.

    MSTAR_WHISPER_REPO=openai/whisper-large-v3-turbo pytest test/asr/test_whisper_parity.py -q
"""

import gc
import os

import _served
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = os.environ.get("MSTAR_WHISPER_REPO", "openai/whisper-large-v3-turbo")
KEY = "whisper_large_v3_turbo" if REPO.endswith("turbo") else "whisper_large"


def _snapshot():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(REPO, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{REPO} is not in the local HF cache")


def _models():
    from transformers import WhisperForConditionalGeneration

    from mstar.model.registry import get_model_class

    local_dir = _snapshot()
    model = get_model_class(KEY)(model_path_hf=local_dir)
    hf = WhisperForConditionalGeneration.from_pretrained(local_dir, torch_dtype=torch.bfloat16).cuda().eval()
    return model, hf


@pytest.fixture
def models():
    # per test, so nothing a test put on the GPU is left when the served
    # comparison starts a server
    yield _models()
    gc.collect()
    torch.cuda.empty_cache()


def test_encoder_matches_hf_on_real_audio(models):
    model, hf = models
    enc_sub = model.get_submodule("audio_encoder", device="cuda", autocast_dtype=torch.bfloat16)
    worst = 0.0
    for path in _served.utterances(4):
        wave = model.load_audio(str(path), "cpu").data
        prompt = model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [wave]}, language="en")
        samples = prompt.new_input_tensors["audio"][0]
        with torch.no_grad():
            # the submodule's GPU log-mel against the same transform on the CPU
            gpu_feats = enc_sub.log_mel(enc_sub.log_mel.pad_or_trim(samples.cuda()))
            cpu_feats = model.log_mel(model.log_mel.pad_or_trim(samples))
            assert torch.allclose(gpu_feats.cpu(), cpu_feats, atol=1e-3), (gpu_feats.cpu() - cpu_feats).abs().max()
        x = gpu_feats.to(torch.bfloat16).unsqueeze(0)
        with torch.no_grad():
            ours = enc_sub.encoder(x).float()
            theirs = hf.model.encoder(x).last_hidden_state.float()
        worst = max(worst, (ours - theirs).abs().max().item())
    # bf16 encoders agree to rounding, anything structural is orders larger
    assert worst < 0.1, worst


def test_word_timestamps_match_hf_token_timestamps(models):
    """Word starts against HF's DTW token timestamps (same alignment heads,
    same median filter), on real utterances: most words within 0.1 s."""
    model, hf = models
    from transformers import WhisperProcessor

    from mstar.model.whisper.components import alignment

    processor = WhisperProcessor.from_pretrained(_snapshot())
    dec_sub = model.get_submodule("decoder", device="cuda", autocast_dtype=torch.bfloat16)
    enc_sub = model.get_submodule("audio_encoder", device="cuda", autocast_dtype=torch.bfloat16)
    generation_config = hf.generation_config
    within, total = 0, 0
    for path in _served.utterances(4):
        wave = model.load_audio(str(path), "cpu").data
        feats = processor(wave.numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
        with torch.no_grad():
            out = hf.generate(
                feats.cuda().to(torch.bfloat16), language="en", task="transcribe",
                return_token_timestamps=True, return_timestamps=False, num_frames=wave.numel() // 160,
            )
        out = dict(out) if not isinstance(out, dict) else out  # a ModelOutput or a plain dict, by transformers version
        tokens = out["sequences"][0].tolist()
        hf_times = out["token_timestamps"][0].tolist()  # one per sequence position
        if tokens[0] == model.config.decoder_start_token_id:  # older versions keep the forced prompt
            tokens, hf_times = tokens[4:], hf_times[4:]
        keep = [i for i, t in enumerate(tokens) if t < model.config.eos_token_id]
        text = [tokens[i] for i in keep]
        hf_times = [hf_times[i] for i in keep]
        # ours: the same teacher-forced sequence through the served decoder weights
        encoder_states = enc_sub.encoder(
            enc_sub.log_mel(enc_sub.log_mel.pad_or_trim(wave.cuda())).to(torch.bfloat16).unsqueeze(0)
        )[0]
        seq = torch.tensor(tokens[:4] + text + [model.config.eos_token_id], device="cuda")
        heads = [(int(layer), int(head)) for layer, head in model.config.alignment_heads]
        weights = dec_sub.decoder.cross_attention_weights(seq, encoder_states, heads)
        matrix = alignment.alignment_matrix(weights[:, 3:-1], wave.numel() // 160)
        starts = alignment.token_start_frames(matrix) / alignment.FRAMES_PER_SECOND
        # HF reports each token's *end*, and a token starts where the previous one ends
        for i in range(1, len(text)):
            total += 1
            within += abs(starts[i] - hf_times[i - 1]) <= 0.1
    assert within / total > 0.9, (within, total)


def test_served_greedy_text_matches_hf_generate():
    """The served model against HF ``generate``, greedy and bf16, on the
    benchmark's first files that fit one 30 s window (so both sides see the
    same audio: HF's short-form generate truncates at 30 s)."""
    from transformers import WhisperProcessor

    model, hf = _models()
    processor = WhisperProcessor.from_pretrained(_snapshot())
    paths = _served.utterances(8, max_seconds=29.5)
    reference = {}
    for path in paths:
        wave = model.load_audio(str(path), "cpu").data
        feats = processor(wave.numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
        prompt = model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [wave]}, language="en")
        ours = prompt.new_input_tensors
        assert torch.allclose(feats[0], model.log_mel(model.log_mel.pad_or_trim(ours["audio"][0])), atol=2e-4)
        with torch.no_grad():
            out = hf.generate(
                feats.cuda().to(torch.bfloat16), language="en", task="transcribe",
                do_sample=False, num_beams=1, max_new_tokens=440,
            )
        reference[path.name] = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    del hf, out
    gc.collect()
    torch.cuda.empty_cache()

    with _served.serving(KEY) as base:
        served = {path.name: _served.transcribe(base, path, "en").strip() for path in paths}
    _served.assert_greedy_parity(reference, served)
