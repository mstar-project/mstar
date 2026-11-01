"""Qwen3-Omni's encoders declare their attention as ragged resources and run
natively against them.

CPU, tiny random weights. A stand-in resource attends within the segments the
node's step declared (what the planned kernel does), and transformers'
``Qwen3OmniMoe{Audio,Vision}Encoder`` on SDPA are the oracle: the same
windows, frames and deepstack outputs, with no flash_attn anywhere.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # noqa: E402
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeVisionEncoder,
)

from mstar.engine.resources import RaggedAttentionSpec  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo, build_resource  # noqa: E402
from mstar.model.components.aut_encoder import AuTEncoder, AuTEncoderConfig  # noqa: E402
from mstar.model.components.aut_submodule import AuTEncoderSubmodule  # noqa: E402
from mstar.model.loader import load_hf_weights  # noqa: E402
from mstar.model.qwen3_omni.components.vision import Qwen3OmniVisionTower  # noqa: E402
from mstar.model.qwen3_omni.config import (  # noqa: E402
    AUDIO_ATTN,
    VISION_ATTN,
    Qwen3OmniModelConfig,
    VisionEncoderConfig,
)
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel  # noqa: E402
from mstar.model.qwen3_omni.submodules import AudioEncoderSubmodule, VisionEncoderSubmodule  # noqa: E402
from mstar.model.utils import check_loaded  # noqa: E402

AUDIO = dict(
    d_model=64, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128,
    num_mel_bins=16, n_window=50, n_window_infer=200, conv_chunksize=3,
    downsample_hidden_size=8, output_dim=48, max_source_positions=1500,
)
VISION = dict(
    depth=3, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=4,
    temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=24,
    num_position_embeddings=16, deepstack_visual_indexes=[0, 2],
)


class StandInRagged:
    """The ragged resource as a node sees it: the step's segments in, and
    ``run`` attending within each of them, as the planned kernel does."""

    max_segments_per_request = 64

    def __init__(self):
        self.spans: list[int] = []
        self.calls = 0

    def plan(self, step):
        self.spans = [s.span for s in step.segments]

    def run(self, q, k, v):
        self.calls += 1
        assert sum(self.spans) == q.shape[0], "the forward's rows are not the planned segments"
        out, start = torch.empty_like(q), 0
        for n in self.spans:
            o = F.scaled_dot_product_attention(*(t[start:start + n].transpose(0, 1) for t in (q, k, v)))
            out[start:start + n] = o.transpose(0, 1)
            start += n
        return out


def _engine_inputs(rids):
    # no capture runners and a declared step: the eager path
    return SimpleNamespace(request_ids=list(rids), piecewise_runners={}, step=object())


def _fwd(rid="r"):
    return SimpleNamespace(request_id=rid)


@pytest.fixture(scope="module")
def audio_pair():
    torch.manual_seed(0)
    hf = Qwen3OmniMoeAudioEncoder._from_config(
        Qwen3OmniMoeAudioEncoderConfig(**AUDIO), attn_implementation="sdpa",
    ).eval()
    encoder = AuTEncoder(AuTEncoderConfig.from_hf(AUDIO), attn_key=AUDIO_ATTN)
    encoder.load_weights(list(hf.state_dict().items()))
    sub = AudioEncoderSubmodule(encoder.eval(), Qwen3OmniModelConfig(audio_encoder=encoder.config))
    ragged = StandInRagged()
    sub.bind_node_resources({AUDIO_ATTN: ragged})
    return hf, sub, ragged


@pytest.fixture(scope="module")
def vision_pair():
    torch.manual_seed(0)
    hf = Qwen3OmniMoeVisionEncoder._from_config(
        Qwen3OmniMoeVisionEncoderConfig(**VISION), attn_implementation="sdpa",
    ).eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    config = VisionEncoderConfig.from_dict(VISION)
    tower = Qwen3OmniVisionTower(config, attn_key=VISION_ATTN)
    check_loaded(tower, load_hf_weights(tower, hf.state_dict().items()), "tower")
    sub = VisionEncoderSubmodule(tower.eval(), Qwen3OmniModelConfig(vision=config))
    ragged = StandInRagged()
    sub.bind_node_resources({VISION_ATTN: ragged})
    return hf, sub, ragged


def test_encoders_declare_ragged_attention():
    """Each encoder node gets one ragged resource with its tower's heads, and
    the Thinker and Talker keep their paged attention."""
    model = object.__new__(Qwen3OmniModel)
    model.config = Qwen3OmniModelConfig(
        audio_encoder=AuTEncoderConfig.from_hf(AUDIO), vision=VisionEncoderConfig.from_dict(VISION),
    )
    specs = {s.resource_key: s for s in model.get_node_resources()}

    audio, vision = specs[AUDIO_ATTN], specs[VISION_ATTN]
    assert isinstance(audio, RaggedAttentionSpec) and isinstance(vision, RaggedAttentionSpec)
    assert audio.nodes == {"audio_encoder"} and vision.nodes == {"vision_encoder"}
    assert (audio.config.num_qo_heads, audio.config.num_kv_heads, audio.config.head_dim) == (4, 4, 16)
    assert (vision.config.num_qo_heads, vision.config.num_kv_heads, vision.config.head_dim) == (4, 4, 8)
    # a capture replay can always be handed the windows of what fits its largest bucket
    window = AuTEncoderConfig.from_hf(AUDIO).window_tokens
    biggest = AuTEncoderSubmodule.ENCODE_TOKEN_BUCKETS[-1]
    assert audio.config.max_segments_per_request * window >= biggest
    others = {k for k, s in specs.items() if s.nodes & {"audio_encoder", "vision_encoder"}}
    assert others == {AUDIO_ATTN, VISION_ATTN}


def test_ragged_spec_builds_its_backend_without_flash_attn(monkeypatch):
    """The declaration resolves to the platform's ragged backend (FlashInfer by
    default); the backend module is stubbed, so nothing here needs a GPU."""
    built = {}

    class FakeManager:
        def __init__(self, device, dtype, config, key):
            built.update(device=device, dtype=dtype, config=config)

    fake = SimpleNamespace(FlashInferRaggedManager=FakeManager)
    monkeypatch.setitem(sys.modules, "mstar.engine.resources.attn.ragged.flashinfer", fake)
    model = object.__new__(Qwen3OmniModel)
    model.config = Qwen3OmniModelConfig(vision=VisionEncoderConfig.from_dict(VISION))
    spec = next(s for s in model.get_node_resources() if s.resource_key == VISION_ATTN)

    res = build_resource(spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.bfloat16))

    assert isinstance(res, FakeManager)
    assert built["dtype"] == torch.bfloat16 and built["config"].head_dim == 8
    assert "flash_attn" not in sys.modules


def test_audio_windows_match_hf(audio_pair):
    """Three clips packed in one forward, every window a declared segment:
    each clip's embeddings match HF's encoder on that clip alone."""
    hf, sub, ragged = audio_pair
    # under one window, about one, and several (a window is 200 frames here)
    feats = [torch.randn(AUDIO["num_mel_bins"], n) for n in (150, 230, 905)]
    rows = [sub.prepare_inputs("prefill_audio", _fwd(), {"audio_features": [f]}) for f in feats]
    step = sub.declare_step("prefill_audio", ["a", "b", "c"], rows)
    window = sub.encoder.config
    assert step.steps[AUDIO_ATTN].causal is False
    assert [(s.request_id, s.span) for s in step.segments] == [
        (rid, w) for rid, r in zip("abc", rows, strict=True) for w in window.window_lengths(r.input_seq_len)
    ]
    ragged.plan(step)
    with torch.no_grad():
        out = sub.forward_batched("prefill_audio", _engine_inputs("abc"), **sub.preprocess("prefill_audio", None, rows))
        for rid, f in zip("abc", feats, strict=True):
            want = hf(f, feature_lens=torch.tensor([f.shape[1]])).last_hidden_state
            got = out[rid]["audio_embeds"][0]
            assert got.shape == want.shape
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    assert ragged.calls == AUDIO["encoder_layers"]


@pytest.mark.parametrize(
    "grid", [[[1, 4, 6]], [[3, 4, 4]], [[1, 2, 2], [1, 4, 4]]], ids=["image", "video", "two_images"],
)
def test_vision_frames_and_deepstack_match_hf(vision_pair, grid):
    """One segment per frame, and the merged embeddings and every deepstack
    feature match HF's encoder."""
    hf, sub, ragged = vision_pair
    grid_thw = torch.tensor(grid)
    patch_numel = 3 * VISION["temporal_patch_size"] * VISION["patch_size"] ** 2
    pixels = torch.randn(int(grid_thw.prod(-1).sum()), patch_numel)
    row = sub.prepare_inputs("prefill_vision", _fwd(), {"pixel_values": [pixels], "image_grid_thw": [grid_thw]})
    assert row.input_seq_len == pixels.shape[0]
    step = sub.declare_step("prefill_vision", ["v"], [row])
    assert [s.span for s in step.segments] == [h * w for t, h, w in grid for _ in range(t)]
    assert step.steps[VISION_ATTN].causal is False
    ragged.plan(step)
    with torch.no_grad():
        out = sub.forward("prefill_vision", _engine_inputs("v"), **sub.preprocess("prefill_vision", None, [row]))
        want = hf(pixels, grid_thw=grid_thw)
    torch.testing.assert_close(out["vision_embeds"][0], want.pooler_output, atol=1e-4, rtol=1e-4)
    assert len(out["deepstack"]) == len(VISION["deepstack_visual_indexes"])
    for got, ref in zip(out["deepstack"], want.deepstack_features, strict=True):
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_vision_capture_refuses_more_frames_than_a_replay_plans(vision_pair):
    """A request with more frames than the resource's per-request ceiling
    takes no lease, so its block loop runs eagerly."""
    _, sub, ragged = vision_pair
    rows = [SimpleNamespace(tensor_inputs={"seq_lengths": (4,) * n}) for n in (1, ragged.max_segments_per_request + 1)]
    assert sub._fits_capture(rows[:1])
    assert not sub._fits_capture(rows[1:])


def test_rotary_table_stays_fp32_through_a_cast():
    tower = Qwen3OmniVisionTower(VisionEncoderConfig.from_dict(VISION), attn_key=VISION_ATTN)
    want = tower.rotary_pos_emb.inv_freq.clone()
    tower = tower.to(torch.bfloat16)
    assert tower.patch_embed.proj.weight.dtype == torch.bfloat16
    assert tower.rotary_pos_emb.inv_freq.dtype == torch.float32
    assert torch.equal(tower.rotary_pos_emb.inv_freq, want)


def test_tower_parameters_are_the_checkpoints():
    """Names line up with ``thinker.visual.*``, so the checkpoint loads by name."""
    hf = Qwen3OmniMoeVisionEncoder._from_config(Qwen3OmniMoeVisionEncoderConfig(**VISION), attn_implementation="sdpa")
    tower = Qwen3OmniVisionTower(VisionEncoderConfig.from_dict(VISION), attn_key=VISION_ATTN)
    assert dict(tower.named_parameters()).keys() == dict(hf.named_parameters()).keys()
