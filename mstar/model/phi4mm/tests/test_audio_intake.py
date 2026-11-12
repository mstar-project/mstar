"""CPU tests of Phi-4-multimodal's audio intake and window layout, and of the
audio node on a tiny random tower: frame and token counts, refusals, the
window plan, row independence under padding, and the node's per-request
output and projection choice.

With ``PHI4MM_ORACLE`` set to the reference recordings (``record.py``'s
directory, whose ``media/`` holds the wavs), the intake is also checked bit
for bit against the reference processor's features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.multimodal import MEDIA_LAYOUT
from mstar.model.phi4mm.audio import (
    AUDIO_EMBEDS,
    AUDIO_FEATURES,
    VISION_AUDIO_WALK,
    Phi4MMAudioEncoderSubmodule,
    audio_intake,
    log_mel,
)
from mstar.model.phi4mm.components.audio import (
    SPEECH,
    VISION,
    Phi4MMAudioConfig,
    Phi4MMAudioTower,
    audio_batch,
    plan_windows,
)
from mstar.model.phi4mm.phi4mm_model import Phi4MMModel
from mstar.model.phi4mm.tests._paths import ORACLE as _ORACLE
from mstar.model.phi4mm.tests._paths import snapshot
from mstar.model.submodule_base import ModelInputsFromEngine

ORACLE = None if _ORACLE is None else str(_ORACLE)


def _tone(samples: int, rate: int) -> torch.Tensor:
    t = torch.arange(samples, dtype=torch.float64) / rate
    return (0.3 * torch.sin(2 * math.pi * 440 * t)).to(torch.float32)


@pytest.mark.parametrize(
    ("samples", "rate", "frames"),
    [
        (400, 16_000, 1),  # one window, the shortest clip
        (559, 16_000, 1),  # a tail short of a hop is dropped
        (560, 16_000, 2),
        (16_000 * 14 + 1234, 16_000, (16_000 * 14 + 1234 - 400) // 160 + 1),
        (200, 8_000, 1),
        (8_000 * 3 + 37, 8_000, (8_000 * 3 + 37 - 200) // 80 + 1),
    ],
)
def test_frame_and_token_counts(samples: int, rate: int, frames: int):
    prepared = audio_intake(_tone(samples, rate), {"sample_rate": rate})
    # the model's intake, called as the scaffold calls it (with the request's kwargs)
    clip, metadata = _tone(samples, rate), {"sample_rate": rate}
    via_model = Phi4MMModel._audio_intake(SimpleNamespace(audio_config=None), clip, metadata, {})
    assert via_model.tokens == prepared.tokens
    mel = prepared.inputs[AUDIO_FEATURES]
    assert mel.shape == (frames, 80) and mel.dtype == torch.float32
    assert prepared.tokens == [-(-frames // 8)]
    assert prepared.grids == [[frames]]
    # the log is clipped at 1 before it is taken
    assert float(mel.min()) >= 0.0


def test_too_short_refused():
    with pytest.raises(ValueError, match="at least 25 ms"):
        audio_intake(torch.zeros(399), {"sample_rate": 16_000})


def test_other_rates_and_channels_refused():
    with pytest.raises(ValueError, match="16 or 8 kHz"):
        audio_intake(_tone(44_100, 44_100), {"sample_rate": 44_100})
    with pytest.raises(ValueError, match="mono"):
        audio_intake(torch.zeros(2, 16_000), {"sample_rate": 16_000})


def test_narrowband_fills_the_upper_bins():
    """An 8 kHz clip's spectrum sits in the low half of the 16 kHz filters, so
    the filters above 4 kHz see nothing and clip to log(1) = 0."""
    mel = log_mel(_tone(8_000, 8_000).numpy(), 8_000)
    top = mel[:, -10:]
    assert np.all(top == 0.0)
    assert mel[:, :40].max() > 1.0


def test_plan_windows_one_window():
    plan = plan_windows([5, 3], window=8)
    assert (plan.width, plan.per_clip) == (5, 1)
    assert plan.source == [0, 1]
    assert plan.keys == [5, 3]
    assert plan.rows == [0, 1, 2, 3, 4, 5, 6, 7]


def test_plan_windows_unfolds_long_clips():
    # a 19-frame clip in windows of 8: two whole windows and a 3-frame one,
    # all attending in full as the reference's unfold does; the short clip
    # beside it is masked past its own frames
    plan = plan_windows([19, 5], window=8)
    assert (plan.width, plan.per_clip) == (8, 3)
    assert plan.source == [0, 1, 2, 3]
    assert plan.keys == [8, 8, 8, 5]
    assert plan.rows == [*range(19), *range(24, 29)]


def test_plan_windows_at_a_bucket_width():
    # a capture bucket's width: every window masked past its frames, the rows
    # strided by the bucket
    plan = plan_windows([5, 3], window=8, width=8)
    assert (plan.width, plan.per_clip, plan.keys) == (8, 1, [5, 3])
    assert plan.rows == [0, 1, 2, 3, 4, 8, 9, 10]
    with pytest.raises(ValueError, match="don't run 4 wide"):
        plan_windows([5], window=8, width=4)
    with pytest.raises(ValueError, match="don't run 6 wide"):
        plan_windows([19], window=8, width=6)


def test_bucketed_batch_matches_its_own_shape(tiny_tower):
    """A batch padded to a mel bucket and a window bucket gives the rows its
    own shape gives (the captured path's batch, run eagerly)."""
    clips = _clips(0.3, 0.45)
    with torch.no_grad():
        own = tiny_tower.encode(audio_batch(clips, tiny_tower.config, torch.device("cpu"), torch.float32))
        bucketed = audio_batch(clips, tiny_tower.config, torch.device("cpu"), torch.float32, mel_frames=64, width=8)
        assert bucketed.key_lengths is not None and bucketed.mel.shape[1] == 64
        torch.testing.assert_close(tiny_tower.encode(bucketed), own, rtol=1e-5, atol=1e-5)


def _tiny_config() -> Phi4MMAudioConfig:
    return Phi4MMAudioConfig(
        num_mel_bins=80, hidden_size=32, num_heads=4, intermediate_size=48, num_layers=2,
        subsampling_channels=8, max_distance=8, window=8, llm_hidden_size=24,
    )


@pytest.fixture(scope="module")
def tiny_tower() -> Phi4MMAudioTower:
    torch.manual_seed(0)
    tower = Phi4MMAudioTower(_tiny_config()).eval()
    with torch.no_grad():
        tower.encoder.global_mean.normal_()
        tower.encoder.global_invstd.uniform_(0.5, 1.5)
        for p in tower.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return tower


def _clips(*seconds: float) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [
        audio_intake(_tone(n, 16_000) + 0.05 * torch.randn(n), {"sample_rate": 16_000})
        .inputs[AUDIO_FEATURES]
        for n in (int(s * 16_000) for s in seconds)
    ]


def _encode(tower: Phi4MMAudioTower, clips: list[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        return tower.encode(audio_batch(clips, tower.config, torch.device("cpu"), torch.float32))


def test_rows_independent_of_padding(tiny_tower):
    """Clips of one, several and part windows give the same rows batched and
    padded together as alone: padding is zeroed before every subsampling conv
    and masked in attention."""
    clips = _clips(0.3, 1.5, 0.61)  # 4, 19 and 8 tokens with windows of 8
    alone = [_encode(tiny_tower, [c]) for c in clips]
    assert [a.shape[0] for a in alone] == [tiny_tower.config.tokens(c.shape[0]) for c in clips]
    batched = _encode(tiny_tower, clips)
    torch.testing.assert_close(batched, torch.cat(alone), rtol=1e-5, atol=1e-5)


def test_unzeroed_padding_reaches_the_short_clip(tiny_tower):
    """The negative control of the test above: CMVN turns zero padding
    nonzero, and the subsampling convs see both sides, so without the
    zeroing a padded clip's last frames change."""
    short, long_ = _clips(0.3, 1.5)
    enc = tiny_tower.encoder
    lengths = torch.tensor([short.shape[0], long_.shape[0]])
    mel = torch.nn.utils.rnn.pad_sequence([short, long_], batch_first=True)
    with torch.no_grad():
        zeroed = enc.subsample(mel, lengths)[0]
        alone = enc.subsample(short[None], lengths[:1])[0]
        unzeroed = enc.subsampling((mel - enc.global_mean) * enc.global_invstd, None)[0]
    n = alone.shape[0]
    torch.testing.assert_close(zeroed[:n], alone, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(unzeroed[n - 1], alone[n - 1], atol=1e-3)
    # frames far from the end are out of the convs' reach
    torch.testing.assert_close(unzeroed[: n - 2], alone[: n - 2], rtol=1e-5, atol=1e-5)


def _info(rid: str, walk: str, frames: list[int]) -> CurrentForwardPassInfo:
    tokens = [-(-f // 8) for f in frames]
    offsets = [3 + sum(tokens[:i]) + 2 * i for i in range(len(frames))]
    layout = {
        "offset": offsets, "tokens": tokens, "output": [AUDIO_EMBEDS] * len(frames), "grid": [[f] for f in frames],
    }
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=walk, fwd_index=0, random_seed=0, max_tokens=0,
        step_metadata={MEDIA_LAYOUT: layout},
    )


@pytest.mark.parametrize(("walk", "mode"), [("prefill_audio", SPEECH), (VISION_AUDIO_WALK, VISION)])
def test_node_rows_per_request_in_prompt_order(tiny_tower, walk, mode):
    node = Phi4MMAudioEncoderSubmodule(tiny_tower)
    a, b1, b2 = _clips(0.4, 1.5, 0.25)
    infos = {"a": _info("a", walk, [a.shape[0]]), "b": _info("b", walk, [b1.shape[0], b2.shape[0]])}
    inputs = [
        node.prepare_inputs(walk, infos["a"], {AUDIO_FEATURES: [a]}),
        node.prepare_inputs(walk, infos["b"], {AUDIO_FEATURES: [b1, b2]}),
    ]
    assert node.can_batch(None, inputs)
    engine_inputs = ModelInputsFromEngine(request_ids=["a", "b"], per_request_info=infos)
    with torch.no_grad():
        out = node.forward_batched(walk, engine_inputs, **node.preprocess(walk, engine_inputs, inputs))
        want_a = tiny_tower.project(_encode(tiny_tower, [a]), mode)
        want_b = tiny_tower.project(torch.cat([_encode(tiny_tower, [b1]), _encode(tiny_tower, [b2])]), mode)
    assert [inp.input_seq_len for inp in inputs] == [want_a.shape[0], want_b.shape[0]]
    torch.testing.assert_close(out["a"][AUDIO_EMBEDS][0], want_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out["b"][AUDIO_EMBEDS][0], want_b, rtol=1e-5, atol=1e-5)


def test_node_refuses_a_layout_that_disagrees():
    node = Phi4MMAudioEncoderSubmodule(Phi4MMAudioTower(_tiny_config()))
    (clip,) = _clips(0.4)
    with pytest.raises(ValueError, match="frames for a layout"):
        node.prepare_inputs("prefill_audio", _info("a", "prefill_audio", [clip.shape[0] + 1]), {AUDIO_FEATURES: [clip]})


def _snapshot() -> Path | None:
    return snapshot()


def test_config_from_checkpoint():
    snapshot = _snapshot()
    if snapshot is None or not (snapshot / "config.json").exists():
        pytest.skip("Phi-4-multimodal's config.json is not cached")
    hf = json.loads((snapshot / "config.json").read_text())
    config = Phi4MMAudioConfig.from_hf(hf)
    assert config == Phi4MMAudioConfig()
    hf["audio_processor"]["config"]["chunk_size"] = 18
    with pytest.raises(ValueError, match="chunk_size"):
        Phi4MMAudioConfig.from_hf(hf)


@pytest.mark.skipif(not ORACLE, reason="set PHI4MM_ORACLE to the reference recordings")
@pytest.mark.parametrize(("case", "wav"), [("audio_short", "speech_short"), ("audio_long", "speech_long")])
def test_intake_matches_reference_exactly(case: str, wav: str):
    """Through M*'s own decoder: the reference processor's features, bit for
    bit, and its token count."""
    from mstar.model.components.audio_features import load_speech_file

    record = torch.load(Path(ORACLE) / "sdpa" / f"{case}.pt", weights_only=False)
    waveform = load_speech_file(str(Path(ORACLE) / "media" / f"{wav}.wav"), 16_000, 60.0)
    prepared = audio_intake(waveform, {"sample_rate": 16_000, "num_channels": 1})
    assert torch.equal(prepared.inputs[AUDIO_FEATURES], record["inputs"]["input_audio_embeds"][0])
    assert prepared.tokens == record["inputs"]["audio_embed_sizes"].tolist()
