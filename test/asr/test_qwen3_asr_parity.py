"""Real-weight parity for Qwen3-ASR (GPU): AuT encoder vs HF, served greedy text vs the reference SDK.

Skipped without CUDA, the checkpoint in the local HF cache, or the benchmark
set (``python -m benchmark.asr_data``, under ``MSTAR_ASR_DATA``).

  * encoder: the native ``AuTEncoder`` loaded from ``thinker.audio_tower.*``
    against transformers' ``Qwen3OmniMoeAudioEncoder`` built from the same
    weights, on real utterances shorter than one 8 s window (HF's SDPA path
    is un-windowed, so longer clips are not comparable on that path);
  * end to end: the reference ``qwen_asr`` SDK (transformers backend, run by
    ``benchmark.asr_reference`` in a subprocess) and ``mstar serve`` with the
    shipped config transcribe the same files, greedy with the language forced;
    the texts must match (see ``_served.assert_greedy_parity``). Needs
    ``MSTAR_QWEN_ASR_REF`` pointing at the SDK checkout and the GPU to itself
    (the server takes most of it). The SDK's model code needs transformers
    4.57, so under transformers 5 ``MSTAR_QWEN_ASR_REF_SITE`` names a directory
    holding it, which goes first on the subprocess's path:

    pip install --target /opt/qwen-asr-site --no-deps transformers==4.57.3 \\
        "huggingface-hub>=0.34,<1.0" "tokenizers>=0.22,<=0.23"
    MSTAR_QWEN_ASR_REF_SITE=/opt/qwen-asr-site pytest test/asr/test_qwen3_asr_parity.py -q
"""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import _served
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

REPO = os.environ.get("MSTAR_QWEN_ASR_REPO", "Qwen/Qwen3-ASR-1.7B")
KEY = "qwen3_asr_realtime" if REPO.endswith("0.6B") else "qwen3_asr"
SDK = os.environ.get("MSTAR_QWEN_ASR_REF", "refs/Qwen3-ASR")
SDK_SITE = os.environ.get("MSTAR_QWEN_ASR_REF_SITE")
ROOT = Path(__file__).resolve().parents[2]


def _snapshot():
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(REPO, local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip(f"{REPO} is not in the local HF cache")


@pytest.fixture
def model():
    # per test, so the submodules a test put on the GPU are gone before the
    # served comparison starts a server
    from mstar.model.registry import get_model_class

    yield get_model_class(KEY)(model_path_hf=_snapshot())
    gc.collect()
    torch.cuda.empty_cache()


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
    for path in _served.utterances(4, max_seconds=7.5):
        wave = model.load_audio(str(path), "cpu").data
        feats = model.log_mel(wave).to("cuda", torch.bfloat16)
        with torch.no_grad():
            expected = hf(feats, feature_lens=torch.tensor([feats.shape[-1]], device="cuda")).last_hidden_state
            actual, layout = sub.encoder(feats.unsqueeze(0), [feats.shape[-1]])
        assert layout.tokens_per_request == [expected.shape[0]]
        worst = max(worst, (actual.float() - expected.float()).abs().max().item())
    assert worst < 0.1, worst


def test_served_greedy_text_matches_the_sdk(tmp_path):
    """The served model against the reference SDK (transformers backend, the
    benchmark's WER reference), greedy and bf16, on the benchmark's first files."""
    if not Path(SDK, "qwen_asr").is_dir():
        pytest.skip(f"reference SDK not found at {SDK}")
    import transformers

    if SDK_SITE is None and int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip("the SDK needs transformers 4.57: set MSTAR_QWEN_ASR_REF_SITE (see the module docstring)")
    from mstar.model.qwen3_asr.repetition import fix_repetitions

    def norm(text: str) -> str:
        # what the served route applies to a finished transcript (Qwen3ASRAdapter.final_text)
        return fix_repetitions(text.strip()).strip()

    paths = _served.utterances(8)
    out = tmp_path / "reference.json"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [SDK_SITE, str(ROOT)])))
    # batch 1 as served; the language is forced the way the served route maps language="en"
    subprocess.run([
        sys.executable, "-m", "benchmark.asr_reference", "--model", _snapshot(), "--sdk", str(Path(SDK).resolve()),
        "--data-dir", str(_served.DATA.resolve()), "--limit", str(len(paths)), "--batch-size", "1",
        "--language", "en", "--output-json", str(out),
    ], cwd=ROOT, env=env, check=True)
    hyps = json.loads(out.read_text())["hypotheses"]
    reference = {path.name: norm(hyps[path.stem]) for path in paths}

    with _served.serving(KEY) as base:
        served = {path.name: norm(_served.transcribe(base, path, "en")) for path in paths}
    _served.assert_greedy_parity(reference, served)
