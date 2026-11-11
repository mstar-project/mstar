"""GPU parity of Phi-4-multimodal's audio encoder against the reference.

Recordings: ``PHI4MM_ORACLE`` names the reference recordings' directory.
``sdpa/<case>.pt`` (``record.py``) holds the full model's run: processor
outputs, the conformer's output and the projected rows. ``audio/conformer.pt``
(``record_audio.py``) holds the conformer and both projections run alone, as
served (bf16, manual attention), through the reference's own SDPA route
(``use_pt_scaled_dot_product_attention``) and in fp32, on the clips alone and
padded in pairs.

The bound is the reference's spread: the largest relative error among its
bf16 manual, bf16 SDPA and fp32 runs. The port must sit within it in bf16,
match the fp32 reference to fp32 rounding, give the same rows batched as
alone, and diverge well past it with the T5 bias dropped, the other
projection, or the long clip's last window masked. Run with ``-s`` for the
table. Skipped without CUDA, the cached checkpoint or the recordings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

from mstar.model.phi4mm.audio import (
    AUDIO_EMBEDS,
    AUDIO_FEATURES,
    VISION_AUDIO_WALK,
    Phi4MMAudioEncoderSubmodule,
    audio_intake,
)
from mstar.model.phi4mm.components.audio import SPEECH, VISION, Phi4MMAudioConfig, Phi4MMAudioTower, audio_batch
from mstar.model.phi4mm.tests.test_audio_intake import _info, _snapshot
from mstar.model.submodule_base import ModelInputsFromEngine

ORACLE = Path(os.environ["PHI4MM_ORACLE"]) if os.environ.get("PHI4MM_ORACLE") else None
RECORDED = ORACLE / "audio" / "conformer.pt" if ORACLE else None
SNAPSHOT = _snapshot() if torch.cuda.is_available() else None

pytestmark = pytest.mark.skipif(
    SNAPSHOT is None or RECORDED is None or not RECORDED.exists()
    or not (SNAPSHOT / "model.safetensors.index.json").exists(),
    reason="needs CUDA, the cached checkpoint and PHI4MM_ORACLE with audio/conformer.pt",
)

SINGLES = ("short", "long", "cut")
KEYS = ("encoded", SPEECH, VISION)
LABELS = ("bf16_manual", "bf16_sdpa", "fp32_manual")


class Stats(NamedTuple):
    max_abs: float
    rel: float
    cos: float

    def __str__(self) -> str:
        return f"max {self.max_abs:7.4f}  rel {self.rel:.2e}  cos {self.cos:.6f}"


def stats(got: torch.Tensor, want: torch.Tensor) -> Stats:
    got, want = got.double().flatten(), want.double().flatten()
    return Stats(
        float((got - want).abs().max()), float((got - want).norm() / want.norm()),
        float(F.cosine_similarity(got, want, dim=0)),
    )


@pytest.fixture(scope="module")
def ref() -> dict:
    return torch.load(RECORDED, weights_only=False)


@pytest.fixture(scope="module")
def towers() -> dict[torch.dtype, Phi4MMAudioTower]:
    import json

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    config = Phi4MMAudioConfig.from_hf(json.loads((SNAPSHOT / "config.json").read_text()))
    return {
        dtype: Phi4MMAudioTower.load(config, SNAPSHOT, device="cuda", dtype=dtype)
        for dtype in (torch.bfloat16, torch.float32)
    }


def _clips(case: dict) -> list[torch.Tensor]:
    mel = case["inputs"]["input_audio_embeds"]
    mask = case["inputs"].get("audio_attention_mask")
    lengths = [mel.shape[1]] if mask is None else mask.sum(1).tolist()
    return [mel[i, :n] for i, n in enumerate(lengths)]


def _run(tower: Phi4MMAudioTower, clips: list[torch.Tensor], **batch_changes) -> dict[str, torch.Tensor]:
    dtype = tower.encoder.rel_bias.weight.dtype
    batch = audio_batch(clips, tower.config, torch.device("cuda"), dtype)._replace(**batch_changes)
    with torch.no_grad():
        rows = tower.encode(batch)
        return {"encoded": rows.cpu(), **{m: tower.project(rows, m).cpu() for m in (SPEECH, VISION)}}


def spread(ref: dict, case: str, key: str) -> float:
    runs = [ref[case][label][key] for label in LABELS]
    return max(stats(a, b).rel for i, a in enumerate(runs) for b in runs[i + 1:])


@pytest.mark.parametrize("case", SINGLES)
def test_bf16_within_reference_spread(towers, ref, case):
    mine = _run(towers[torch.bfloat16], _clips(ref[case]))
    for key in KEYS:
        bound = spread(ref, case, key)
        served, fp32 = stats(mine[key], ref[case]["bf16_manual"][key]), stats(mine[key], ref[case]["fp32_manual"][key])
        print(f"\n{case:5s} {key:8s} vs ref {served} | vs fp32 ref {fp32} | spread rel {bound:.2e}")
        assert served.rel <= 1.25 * bound and fp32.rel <= 1.25 * bound
        assert served.cos > 0.995


@pytest.mark.parametrize("case", SINGLES)
def test_fp32_port_is_exact(towers, ref, case):
    """With both sides in fp32 (TF32 off) only rounding is left: the bf16
    residue above is rounding, not a convention."""
    mine = _run(towers[torch.float32], _clips(ref[case]))
    for key in KEYS:
        s = stats(mine[key], ref[case]["fp32_manual"][key])
        print(f"\n{case:5s} {key:8s} fp32 vs fp32 ref {s}")
        assert s.rel < 1e-4


@pytest.mark.parametrize(
    ("case", "wav", "walk", "mode", "spread_case"),
    [
        ("audio_short", "speech_short", "prefill_audio", SPEECH, "short"),
        ("audio_long", "speech_long", "prefill_audio", SPEECH, "long"),
        ("image_audio", "speech_short", VISION_AUDIO_WALK, VISION, "short"),
    ],
)
def test_served_path_matches_full_model_run(towers, ref, case, wav, walk, mode, spread_case):
    """From the wav, through M*'s decoder, the intake and the node on the
    walk's projection, in the tower's own bf16 as the engine runs it
    (``disable_autocast``), against what the reference model's own forward
    recorded."""
    from mstar.model.components.audio_features import load_speech_file

    record = torch.load(ORACLE / "sdpa" / f"{case}.pt", weights_only=False)
    tower = towers[torch.bfloat16]
    waveform = load_speech_file(str(ORACLE / "media" / f"{wav}.wav"), 16_000, 60.0)
    prepared = audio_intake(waveform, {"sample_rate": 16_000})
    clip = prepared.inputs[AUDIO_FEATURES]
    assert torch.equal(clip, record["inputs"]["input_audio_embeds"][0])
    assert prepared.tokens == record["inputs"]["audio_embed_sizes"].tolist()

    node = Phi4MMAudioEncoderSubmodule(tower)
    info = _info("r", walk, [clip.shape[0]])
    inputs = [node.prepare_inputs(walk, info, {AUDIO_FEATURES: [clip]})]
    engine_inputs = ModelInputsFromEngine(request_ids=["r"], per_request_info={"r": info})
    assert node.disable_autocast
    with torch.no_grad():
        embeds = node.forward_batched(walk, engine_inputs, **node.preprocess(walk, engine_inputs, inputs))
        encoded = _run(tower, [clip])["encoded"]
    embeds = embeds["r"][AUDIO_EMBEDS][0].cpu()
    for name, got, want, key in [
        ("audio_encoded", encoded, record["audio_encoded"][0][0], "encoded"),
        (f"audio_embeds_{mode}", embeds, record[f"audio_embeds_{mode}"][0][0], mode),
    ]:
        s, bound = stats(got, want), spread(ref, spread_case, key)
        fp32 = stats(got, ref[spread_case]["fp32_manual"][key])
        print(f"\n{case:11s} {name:19s} {s} | vs fp32 ref {fp32} | spread rel {bound:.2e}")
        assert got.shape == want.shape and s.rel <= 1.25 * bound and fp32.rel <= 1.25 * bound


def test_negative_controls_diverge(towers, ref):
    """The gate can fail: the T5 bias dropped, or the long clip's last window
    masked past its frames, moves the rows far past the spread."""
    tower = towers[torch.bfloat16]
    bound = spread(ref, "short", "encoded")
    weight = tower.encoder.rel_bias.weight
    kept = weight.detach().clone()
    with torch.no_grad():
        weight.zero_()
    try:
        no_bias = stats(_run(tower, _clips(ref["short"]))["encoded"], ref["short"]["bf16_manual"]["encoded"])
    finally:
        with torch.no_grad():
            weight.copy_(kept)

    # the reference attends a long clip's last window in full, its zero tail
    # included; masking the tail is a different computation
    long_clip = _clips(ref["long"])
    tokens = tower.config.tokens(long_clip[0].shape[0])
    window = tower.config.window
    keys = [window] * (tower.config.windows(tokens) - 1) + [tokens % window]
    masked = _run(tower, long_clip, key_lengths=torch.tensor(keys, device="cuda"))
    last = slice(tokens - tokens % window, tokens)
    masked_tail = stats(masked["encoded"][last], ref["long"]["bf16_manual"]["encoded"][last])
    long_bound = spread(ref, "long", "encoded")
    print(f"\nno T5 bias         {no_bias} (spread {bound:.2e})"
          f"\nlast window masked {masked_tail} (spread {long_bound:.2e})")
    assert no_bias.rel > 10 * bound
    assert masked_tail.rel > 3 * long_bound


def test_checkpoint_projections_are_identical(towers):
    """Why no parity test can tell the projections apart: this checkpoint's
    ``speech`` and ``vision`` MLPs hold the same weights. The node's choice
    by walk is tested on random weights (``test_audio_intake``)."""
    projection = towers[torch.bfloat16].projection
    for a, b in zip(projection[SPEECH].parameters(), projection[VISION].parameters(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize(("pair", "parts"), [("pair_short", ("short", "cut")), ("pair_long", ("short", "long"))])
def test_rows_independent_of_batching(towers, ref, pair, parts):
    """Two clips padded together give each its rows alone: exactly in fp32,
    within the spread in bf16. The reference's own batch moves the shorter
    clip by more (printed): its padding reaches the subsampling convs, and
    a long clip's tail is masked only when batched."""
    clips = _clips(ref[pair])
    for dtype in (torch.float32, torch.bfloat16):
        batched = _run(towers[dtype], clips)["encoded"]
        alone = torch.cat([_run(towers[dtype], [c])["encoded"] for c in clips])
        s = stats(batched, alone)
        bound = max(spread(ref, p, "encoded") for p in parts)
        print(f"\n{pair} {str(dtype):14s} batched vs alone {s} (spread {bound:.2e})")
        assert s.rel < (1e-4 if dtype == torch.float32 else bound)
    for label in LABELS:
        theirs = stats(ref[pair][label]["encoded"], torch.cat([ref[p][label]["encoded"] for p in parts]))
        print(f"{pair} reference {label:11s} batched vs alone {theirs}")
