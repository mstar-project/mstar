"""Real-weight parity for Qwen3-ASR (GPU): AuT encoder vs HF, greedy text vs the reference SDK.

Skipped without CUDA or the checkpoint in the local HF cache.

  * encoder: the native ``AuTEncoder`` loaded from ``thinker.audio_tower.*``
    against transformers' ``Qwen3OmniMoeAudioEncoder`` built from the same
    weights, on real utterances shorter than one 8 s window (HF's SDPA path
    is un-windowed, so longer clips are not comparable on that path);
  * end to end: the reference ``qwen_asr`` SDK (transformers backend, from
    ``refs/Qwen3-ASR``) transcribes the same files; its text is the WER
    reference the served model is held to. Needs ``MSTAR_QWEN_ASR_REF`` to
    point at the SDK checkout.

    MSTAR_QWEN_ASR_REPO=Qwen/Qwen3-ASR-1.7B pytest test/asr/test_qwen3_asr_parity.py -q
"""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = os.environ.get("MSTAR_QWEN_ASR_REPO", "Qwen/Qwen3-ASR-1.7B")
KEY = "qwen3_asr_realtime" if REPO.endswith("0.6B") else "qwen3_asr"
DATA = Path(os.environ.get(
    "MSTAR_ASR_DATA",
    os.path.join(os.environ.get("MSTAR_WS_ROOT", "/nonexistent"), "commons/bench/data/asr"),
))
SDK = os.environ.get("MSTAR_QWEN_ASR_REF", os.path.join(os.environ.get("MSTAR_WS", "."), "refs/Qwen3-ASR"))


def _snapshot():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(REPO, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{REPO} is not in the local HF cache")


def _utterances(n: int, max_seconds: float | None = None) -> list[Path]:
    refs = DATA / "test_clean_200" / "references.tsv"
    if not refs.exists():
        pytest.skip(f"benchmark set not found under {DATA}")
    import wave

    out = []
    for line in refs.read_text().splitlines():
        uid = line.split("\t", 1)[0]
        path = DATA / "test_clean_200" / f"{uid}.wav"
        with wave.open(str(path)) as wf:
            seconds = wf.getnframes() / wf.getframerate()
        if max_seconds is None or seconds <= max_seconds:
            out.append(path)
        if len(out) == n:
            break
    return out


@pytest.fixture(scope="module")
def model():
    from mstar.model.registry import get_model_class

    return get_model_class(KEY)(model_path_hf=_snapshot())


def test_aut_encoder_matches_hf_on_real_audio(model):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    from mstar.model.loader.iterators import iter_safetensors_shards

    audio_cfg = Qwen3OmniMoeAudioEncoderConfig(**{
        k: getattr(model.config.audio, k) for k in (
            "d_model", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim", "num_mel_bins",
            "n_window", "n_window_infer", "conv_chunksize", "downsample_hidden_size", "output_dim",
            "max_source_positions", "activation_function",
        )
    })
    hf = Qwen3OmniMoeAudioEncoder._from_config(audio_cfg, attn_implementation="sdpa")
    prefix = "thinker.audio_tower."
    state = {k.removeprefix(prefix): v for k, v in iter_safetensors_shards(model.local_dir, prefix=prefix)}
    hf.load_state_dict(state, strict=True)
    hf = hf.to("cuda", torch.bfloat16).eval()
    sub = model.get_submodule("audio_encoder", device="cuda", autocast_dtype=torch.bfloat16)
    for layer in sub.encoder.layers:
        layer.self_attn.ragged = None  # SDPA per window, like the oracle for <= 1 window

    worst = 0.0
    for path in _utterances(4, max_seconds=7.5):
        wave = model.load_audio(str(path), "cpu").data
        feats = model.log_mel(wave).to("cuda", torch.bfloat16)
        with torch.no_grad():
            expected = hf(feats, feature_lens=torch.tensor([feats.shape[-1]], device="cuda")).last_hidden_state
            actual, layout = sub.encoder(feats.unsqueeze(0), [feats.shape[-1]])
        assert layout.tokens_per_request == [expected.shape[0]]
        worst = max(worst, (actual.float() - expected.float()).abs().max().item())
    assert worst < 0.1, worst


def test_reference_sdk_transcribes_the_same_files(model):
    """Reference transcripts for the served model's WER parity: the SDK's
    transformers backend, greedy, bf16. Written next to the results so the
    server-side benchmark can be compared token for token."""
    if not Path(SDK, "qwen_asr").is_dir():
        pytest.skip(f"reference SDK not found at {SDK}")
    pkg = types.ModuleType("qwen_asr")
    pkg.__path__ = [str(Path(SDK) / "qwen_asr")]
    sys.modules.setdefault("qwen_asr", pkg)
    core = types.ModuleType("qwen_asr.core")
    core.__path__ = [str(Path(SDK) / "qwen_asr" / "core")]
    sys.modules.setdefault("qwen_asr.core", core)
    backend = importlib.import_module("qwen_asr.core.transformers_backend")
    ref = backend.Qwen3ASRForConditionalGeneration.from_pretrained(
        model.local_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
    ).eval()
    processor = backend.Qwen3ASRProcessor.from_pretrained(model.local_dir, fix_mistral_regex=True)
    texts = {}
    for path in _utterances(3, max_seconds=7.5):
        wave = model.load_audio(str(path), "cpu").data.numpy()
        prompt = model.prompt_text(1, context="", language="English")
        inputs = processor(text=[prompt], audio=[wave], return_tensors="pt", padding=True).to("cuda")
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        with torch.no_grad():
            out = ref.generate(**inputs, max_new_tokens=128)
        texts[path.name] = processor.batch_decode(
            out.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )[0].strip()
    assert all(texts.values()), texts
