"""End-to-end numerical parity between mminf's V-JEPA 2 port and the
HuggingFace Transformers reference on real ``facebook/vjepa2-vitl-fpc64-256``
weights.

Skipped automatically when:
  * CUDA is not available, or
  * ``transformers`` is not installed, or
  * ``facebook/vjepa2-vitl-fpc64-256`` isn't in the local HF cache
    (we avoid triggering a ~1.2 GB download from CI).

Local run::

    pip install "transformers>=4.47"
    huggingface-cli download facebook/vjepa2-vitl-fpc64-256
    pytest test/integration/test_vjepa2.py -v -s

What it does
------------
1. Loads HF ``VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256")``.
2. Instantiates our components (``VJEPA2Encoder`` / ``VJEPA2Predictor``) with
   matching config + copies the HF weights in (keys map 1:1 — this is the
   whole point of our ``encoder.*`` / ``predictor.*`` layout).
3. Builds a deterministic video (random [T, C, H, W] in [0, 1]) and runs
   both pipelines in fp32 on the same device.
4. Asserts max abs diff < 1e-3 on encoder output and < 1e-3 on predictor
   output with the default full-context / full-target masks.

Also covers ``skip_predictor`` mode and the AC-variant's ``to_empty``
survival in a lightweight sanity run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC  # noqa: E402
from mminf.model.vjepa2.components.predictor import VJEPA2Predictor  # noqa: E402
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder  # noqa: E402
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config  # noqa: E402

HF_REPO = "facebook/vjepa2-vitl-fpc64-256"

# Default to the coriander shared cache convention
# (/m-coriander/coriander/$USER/mminf_cache/vjepa2), matching the launch
# scripts under test/<model>/.  ``setdefault`` means a shell-level
# ``HF_HUB_CACHE`` still wins if set, keeping the test portable.
if "USER" in os.environ:
    os.environ.setdefault(
        "HF_HUB_CACHE",
        f"/m-coriander/coriander/{os.environ['USER']}/mminf_cache/vjepa2",
    )


def _hf_cache_has_vjepa2() -> bool:
    """Check that the HF cache holds ``HF_REPO`` so we don't trigger a
    multi-hundred-MB download from CI."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_slug = HF_REPO.replace("/", "--")
    candidate = cache_root / f"models--{repo_slug}"
    if candidate.exists():
        return True
    alt_root = os.environ.get("HF_HUB_CACHE")
    return bool(alt_root) and (Path(alt_root) / f"models--{repo_slug}").exists()


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        pytest.importorskip("transformers", reason="transformers not installed") is None,
        reason="transformers not installed",
    ),
    pytest.mark.skipif(
        not _hf_cache_has_vjepa2(),
        reason=f"{HF_REPO} not in local HF cache; run `huggingface-cli download {HF_REPO}`",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures: HF reference model + our ports loaded with matched weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def hf_model(device):
    """Load the HF VJEPA2Model in eager attention mode (no SDPA fallback)
    so its numerics line up bit-for-bit with our eager port.
    """
    from transformers import VJEPA2Model

    model = VJEPA2Model.from_pretrained(HF_REPO, attn_implementation="eager")
    return model.eval().to(device=device, dtype=torch.float32)


@pytest.fixture(scope="module")
def our_config(hf_model) -> VJepa2Config:
    hf_cfg = hf_model.config
    return VJepa2Config(
        patch_size=hf_cfg.patch_size,
        crop_size=hf_cfg.crop_size,
        frames_per_clip=hf_cfg.frames_per_clip,
        tubelet_size=hf_cfg.tubelet_size,
        hidden_size=hf_cfg.hidden_size,
        in_chans=hf_cfg.in_chans,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        mlp_ratio=hf_cfg.mlp_ratio,
        layer_norm_eps=hf_cfg.layer_norm_eps,
        qkv_bias=hf_cfg.qkv_bias,
        hidden_act=hf_cfg.hidden_act,
        pred_hidden_size=hf_cfg.pred_hidden_size,
        pred_num_attention_heads=hf_cfg.pred_num_attention_heads,
        pred_num_hidden_layers=hf_cfg.pred_num_hidden_layers,
        pred_num_mask_tokens=hf_cfg.pred_num_mask_tokens,
        pred_mlp_ratio=hf_cfg.pred_mlp_ratio,
    )


@pytest.fixture(scope="module")
def our_encoder(hf_model, our_config, device):
    encoder = VJEPA2Encoder(our_config).to(device=device, dtype=torch.float32).eval()
    encoder.load_state_dict(hf_model.encoder.state_dict(), strict=True)
    return encoder


@pytest.fixture(scope="module")
def our_predictor(hf_model, our_config, device):
    predictor = VJEPA2Predictor(our_config).to(device=device, dtype=torch.float32).eval()
    predictor.load_state_dict(hf_model.predictor.state_dict(), strict=True)
    return predictor


@pytest.fixture(scope="module")
def sample_video(our_config, device):
    """A short 16-frame clip of synthetic video data (random [0, 1] floats)
    resized to the model's crop_size.  Kept small so the test runs quickly
    even on modest GPUs."""
    torch.manual_seed(0)
    t = 16
    return torch.rand(
        1,
        t,
        our_config.in_chans,
        our_config.crop_size,
        our_config.crop_size,
        device=device,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encoder_parity(hf_model, our_encoder, sample_video):
    """Our VJEPA2Encoder output matches HF's encoder output exactly (eager
    attention, fp32)."""
    with torch.no_grad():
        hf_out = hf_model.encoder(pixel_values_videos=sample_video).last_hidden_state
        ours_out = our_encoder(sample_video)
    assert ours_out.shape == hf_out.shape
    diff = (ours_out - hf_out).abs().max().item()
    assert diff < 1e-3, f"encoder max abs diff = {diff}"


def test_full_model_parity_default_masks(hf_model, our_encoder, our_predictor, sample_video):
    """Encoder + predictor pipeline matches HF VJEPA2Model.forward() with
    default (full-context, full-target) masks."""
    with torch.no_grad():
        hf_out = hf_model(pixel_values_videos=sample_video)
        hf_predicted = hf_out.predictor_output.last_hidden_state

        our_enc = our_encoder(sample_video)
        b, n, _ = our_enc.shape
        mask_all = torch.arange(n, device=sample_video.device).unsqueeze(0).repeat(b, 1)
        our_predicted = our_predictor(our_enc, [mask_all], [mask_all])

    assert our_predicted.shape == hf_predicted.shape
    diff = (our_predicted - hf_predicted).abs().max().item()
    assert diff < 1e-3, f"predictor max abs diff = {diff}"


def test_full_model_parity_partial_mask(hf_model, our_encoder, our_predictor, sample_video):
    """Repeat the parity check with a partial mask (first half context,
    second half target), exercising the sort/unsort path."""
    from transformers.models.vjepa2.modeling_vjepa2 import apply_masks

    device_ = sample_video.device
    window = 256
    with torch.no_grad():
        our_enc = our_encoder(sample_video)
        b, n, _ = our_enc.shape
        ids = torch.arange(n, device=device_).unsqueeze(0)
        ctx = [ids[:, :window].repeat(b, 1)]
        tgt = [ids[:, window : 2 * window].repeat(b, 1)]

        hf_predicted = hf_model.predictor(
            encoder_hidden_states=our_enc,
            context_mask=ctx,
            target_mask=tgt,
        ).last_hidden_state
        our_predicted = our_predictor(our_enc, ctx, tgt)

        # Sanity: also confirm apply_masks agrees
        assert apply_masks(our_enc, ctx).shape == (b, window, our_enc.size(-1))

    assert our_predicted.shape == hf_predicted.shape
    diff = (our_predicted - hf_predicted).abs().max().item()
    assert diff < 1e-3, f"predictor (partial-mask) max abs diff = {diff}"


def test_encoder_only_mode(hf_model, our_encoder, sample_video):
    """HF's ``skip_predictor=True`` fast path returns the encoder output
    directly; our encoder submodule must emit the same tensor."""
    with torch.no_grad():
        hf_out = hf_model.get_vision_features(sample_video)
        ours_out = our_encoder(sample_video)
    assert ours_out.shape == hf_out.shape
    diff = (ours_out - hf_out).abs().max().item()
    assert diff < 1e-3, f"encoder-only max abs diff = {diff}"


def test_ac_predictor_instantiation_and_forward(device):
    """Lightweight AC sanity: instantiate, ``to_empty``, run a forward.
    Validates the buffer-less attn_mask fix under the production
    meta → to_empty pattern.  Uses random weights — this is not a parity
    check against upstream; it's a structural smoke test.
    """
    cfg = VJepa2ACPredictorConfig(
        img_size=(64, 64),
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=128,
        predictor_embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        action_embed_dim=7,
        use_extrinsics=False,
    )
    with torch.device("meta"):
        predictor = VisionTransformerPredictorAC(cfg)
    predictor = predictor.to_empty(device=device)
    with torch.no_grad():
        for p in predictor.parameters():
            torch.nn.init.normal_(p, std=0.02)
    predictor.eval()

    b = 1
    t = cfg.num_frames // cfg.tubelet_size
    grid = cfg.img_size[0] // cfg.patch_size
    n_ctxt = t * grid * grid
    x = torch.randn(b, n_ctxt, cfg.embed_dim, device=device)
    a = torch.randn(b, t, cfg.action_embed_dim, device=device)
    s = torch.randn(b, t, cfg.action_embed_dim, device=device)
    with torch.no_grad():
        out = predictor(x, a, s)
    assert out.shape == (b, n_ctxt, cfg.embed_dim)
    assert torch.isfinite(out).all()


def test_registry_has_both_variants():
    """Pre-flight: both "vjepa2" and "vjepa2_ac" appear in MODEL_REGISTRY so
    ``mminf-serve --config configs/vjepa2_ac.yaml`` can find the class."""
    from mminf.model.registry import HF_MODELS, MODEL_REGISTRY

    assert "vjepa2" in MODEL_REGISTRY
    assert "vjepa2_ac" in MODEL_REGISTRY
    # Masked variants resolve through HuggingFace — owner prefix present.
    assert HF_MODELS["vjepa2"]["model_path_hf"].startswith("facebook/vjepa2-")
    # AC loads from the public S3 mirror (dl.fbaipublicfiles.com) rather
    # than HF; registry.py keeps the ``model_path_hf`` entry as a bare
    # backbone identifier (``vjepa2-ac-vitg``), so assert on the
    # ``vjepa2-ac-`` substring instead of the (nonexistent) ``facebook/``
    # prefix.  See registry.py's comment above the entry.
    assert "vjepa2-ac-" in HF_MODELS["vjepa2_ac"]["model_path_hf"]
