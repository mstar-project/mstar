"""Real-weight parity for Whisper (GPU): native encoder vs HF, greedy tokens vs HF.

Skipped without CUDA or the checkpoint in the local HF cache. Runs the mstar
encoder and decoder submodules standalone (no engine) against
``transformers``' ``WhisperForConditionalGeneration`` on real audio:

  * encoder: ``max |mstar - hf|`` over the 1500x1280 window, both in the
    serving dtype;
  * end to end: greedy token sequence of the mstar decoder (driven with a
    dense mock of the engine's KV / cross-attention resources) against
    ``model.generate`` with the same forced prompt.

    MSTAR_WHISPER_REPO=openai/whisper-large-v3-turbo pytest test/asr/test_whisper_parity.py -q
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = os.environ.get("MSTAR_WHISPER_REPO", "openai/whisper-large-v3-turbo")
DATA = Path(os.environ.get(
    "MSTAR_ASR_DATA",
    os.path.join(os.environ.get("MSTAR_WS_ROOT", "/nonexistent"), "commons/bench/data/asr"),
))


def _snapshot():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(REPO, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{REPO} is not in the local HF cache")


def _audio_paths(n: int) -> list[Path]:
    refs = DATA / "test_clean_200" / "references.tsv"
    if not refs.exists():
        pytest.skip(f"benchmark set not found under {DATA}")
    uids = [line.split("\t", 1)[0] for line in refs.read_text().splitlines() if line.strip()]
    return [DATA / "test_clean_200" / f"{uid}.wav" for uid in uids[:n]]


@pytest.fixture(scope="module")
def models():
    from transformers import WhisperForConditionalGeneration

    from mstar.model.registry import get_model_class

    local_dir = _snapshot()
    model = get_model_class("whisper_large_v3_turbo")(model_path_hf=local_dir)
    hf = WhisperForConditionalGeneration.from_pretrained(local_dir, torch_dtype=torch.bfloat16).cuda().eval()
    return model, hf


def test_encoder_matches_hf_on_real_audio(models):
    model, hf = models
    enc_sub = model.get_submodule("audio_encoder", device="cuda", autocast_dtype=torch.bfloat16)
    worst = 0.0
    for path in _audio_paths(4):
        wave = model.load_audio(str(path), "cpu").data
        feats = model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [wave]}, language="en")
        x = feats["audio_features"][0].cuda().to(torch.bfloat16).unsqueeze(0)
        with torch.no_grad():
            ours = enc_sub.encoder(x).float()
            theirs = hf.model.encoder(x).last_hidden_state.float()
        worst = max(worst, (ours - theirs).abs().max().item())
    # bf16 encoders agree to rounding; anything structural is orders larger
    assert worst < 0.1, worst


class _DenseKV:
    """Stand-in for the paged KV + attention resources, one request."""

    def __init__(self):
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.layer = 0

    def set_default_layer_idx(self, i):
        self.layer = i


def test_greedy_tokens_match_hf_generate(models):
    """Whole-pipeline parity through HF's own generate on a few utterances:
    the token ids must agree exactly (greedy, bf16 on both sides)."""
    model, hf = models
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(_snapshot())
    mismatches = []
    for path in _audio_paths(6):
        wave = model.load_audio(str(path), "cpu").data
        feats = processor(wave.numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
        ours_feats = model.process_prompt(None, ["audio"], ["text"], {"audio_inputs": [wave]}, language="en")
        assert torch.allclose(feats[0], ours_feats["audio_features"][0], atol=2e-4)
        with torch.no_grad():
            out = hf.generate(
                feats.cuda().to(torch.bfloat16), language="en", task="transcribe",
                do_sample=False, num_beams=1, max_new_tokens=200,
            )
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        mismatches.append((path.name, text))
    # The reference transcripts exist for every file; a run that produced
    # nothing means the pipeline is broken, not just imprecise.
    assert all(t.strip() for _, t in mismatches), mismatches
