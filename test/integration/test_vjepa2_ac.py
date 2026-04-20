"""Numerical parity between mminf's V-JEPA 2-AC port and the upstream
``vjepa2/src/hub/backbones.py::_make_vjepa2_ac_model`` on real
``facebook/vjepa2-ac-vitg`` weights.

Skipped automatically when:
  * CUDA is not available, or
  * The upstream ``vjepa2`` repo isn't importable (``src.models.*``
    needs to be on ``PYTHONPATH``), or
  * The upstream ``.pt`` hasn't been cached at
    ``{cache_dir}/vjepa2-ac-vitg.pt``.

Local run::

    git clone https://github.com/facebookresearch/vjepa2 /m-coriander/coriander/$USER/vjepa2
    export PYTHONPATH=/m-coriander/coriander/$USER/vjepa2:$PYTHONPATH
    # Pre-download (one time, ~11.7 GB, direct from S3 — no HF auth):
    python -c "import torch.hub; torch.hub.download_url_to_file( \
        'https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt', \
        '/m-coriander/coriander/$USER/mminf_cache/vjepa2/vjepa2-ac-vitg.pt', \
        progress=True)"
    pytest test/integration/test_vjepa2_ac.py -v -s

What it does
------------
1. Loads the upstream checkpoint via our
   :func:`load_vjepa2_ac_upstream_weights` into mminf ``VJEPA2Encoder`` +
   ``VisionTransformerPredictorAC`` instances (with HF-keyed encoder and
   upstream-keyed AC predictor layouts).
2. Instantiates the upstream reference model via
   ``vjepa2.src.hub.backbones._make_vjepa2_ac_model(pretrained=False)`` and
   loads the **same** ``.pt`` into it (via upstream's own ``_clean_backbone_key``).
3. Builds a deterministic video clip + a deterministic 32-step 7-DOF
   action/state sequence in fp32.
4. Runs both pipelines on the same device; asserts encoder and predictor
   outputs agree to ``atol=1e-3`` (fp32 bit-parity, tolerating
   accumulation order differences between eager attention and
   SDPA-style kernels).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mminf.model.vjepa2.components.ac_predictor import VisionTransformerPredictorAC  # noqa: E402
from mminf.model.vjepa2.components.vit_encoder import VJEPA2Encoder  # noqa: E402
from mminf.model.vjepa2.config import VJepa2ACPredictorConfig, VJepa2Config  # noqa: E402
from mminf.model.vjepa2.weight_loader import (  # noqa: E402
    VJEPA2_AC_VITG_S3_URL,
    download_vjepa2_ac_upstream_pt,
    load_vjepa2_ac_upstream_weights,
)


def _default_cache_dir() -> Path:
    if "USER" in os.environ:
        return Path(f"/m-coriander/coriander/{os.environ['USER']}/mminf_cache/vjepa2")
    return Path.home() / ".cache" / "mminf_vjepa2"


def _upstream_pt_path() -> Path | None:
    """Resolve the upstream ``.pt`` without triggering a download.

    Returns ``None`` if the file hasn't been fetched (so the tests skip
    gracefully rather than blocking on a ~12 GB download).
    """
    candidates: list[Path] = []
    env_cache = os.environ.get("MMINF_VJEPA2_CACHE")
    if env_cache:
        candidates.append(Path(env_cache))
    candidates.append(_default_cache_dir())
    for root in candidates:
        pt = root / "vjepa2-ac-vitg.pt"
        if pt.is_file() and pt.stat().st_size > 0:
            return pt
    return None


def _upstream_importable() -> bool:
    try:
        import importlib

        importlib.import_module("src.models.ac_predictor")
        importlib.import_module("src.models.vision_transformer")
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        not _upstream_importable(),
        reason=(
            "upstream vjepa2 repo not on PYTHONPATH; clone "
            "https://github.com/facebookresearch/vjepa2 and add it to PYTHONPATH"
        ),
    ),
    pytest.mark.skipif(
        _upstream_pt_path() is None,
        reason=(
            f"vjepa2-ac-vitg.pt not found in cache; pre-download from "
            f"{VJEPA2_AC_VITG_S3_URL} or let `test/vjepa2/launch_server_vjepa2_ac.sh` "
            "trigger the download on first model init"
        ),
    ),
]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def ac_config() -> VJepa2Config:
    # Match facebook/vjepa2-ac-vitg: ViT-g @ 256, 64 frames, tubelet_size=2.
    # Encoder: embed_dim=1408, heads=22, layers=40 (upstream vit_giant_xformers).
    cfg = VJepa2Config(
        patch_size=16,
        crop_size=256,
        frames_per_clip=64,
        tubelet_size=2,
        hidden_size=1408,
        in_chans=3,
        num_attention_heads=22,
        num_hidden_layers=40,
        mlp_ratio=48 / 11,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        predictor_kind="ac",
    )
    cfg.ac_predictor = VJepa2ACPredictorConfig(
        img_size=(256, 256),
        patch_size=16,
        num_frames=64,
        tubelet_size=2,
        embed_dim=cfg.hidden_size,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
    )
    return cfg


@pytest.fixture(scope="module")
def pt_path() -> Path:
    # Prefer what's already cached; if not present the module marker above
    # would already have skipped these tests, so this fallback is effectively
    # unreachable in normal runs — but we keep it so the test still works
    # when re-enabled via ``MMINF_VJEPA2_CACHE`` + pre-downloaded weights.
    found = _upstream_pt_path()
    if found is not None:
        return found
    return download_vjepa2_ac_upstream_pt(cache_dir=str(_default_cache_dir()))


@pytest.fixture(scope="module")
def our_modules(ac_config: VJepa2Config, pt_path: Path, device: torch.device):
    encoder = VJEPA2Encoder(ac_config)
    predictor = VisionTransformerPredictorAC(ac_config.ac_predictor)
    # Modules initialize on CPU by default; move to device and then load.
    encoder.to(device)
    predictor.to(device)
    load_vjepa2_ac_upstream_weights(
        pt_path=pt_path,
        encoder_module=encoder,
        predictor_module=predictor,
        device=str(device),
        hidden_size=ac_config.hidden_size,
    )
    encoder.eval()
    predictor.eval()
    return encoder, predictor


@pytest.fixture(scope="module")
def upstream_modules(ac_config: VJepa2Config, pt_path: Path, device: torch.device):
    """Build the upstream reference model via ``_make_vjepa2_ac_model`` and
    load the same ``.pt`` via upstream's own ``_clean_backbone_key`` path.

    ``pretrained=False`` so the call doesn't try to reach the upstream
    CDN (which expects ``VJEPA_BASE_URL=http://localhost:8300`` for tests
    per ``src/hub/backbones.py``).  We apply weights ourselves to keep the
    test self-contained.
    """
    from src.hub.backbones import _clean_backbone_key, _make_vjepa2_ac_model

    encoder, predictor = _make_vjepa2_ac_model(
        model_name="vit_ac_giant",
        img_size=256,
        patch_size=16,
        tubelet_size=2,
        num_frames=64,
        pretrained=False,
    )
    blob = torch.load(pt_path, map_location="cpu", weights_only=True)
    encoder_sd = _clean_backbone_key(dict(blob["encoder"]))
    predictor_sd = _clean_backbone_key(dict(blob["predictor"]))
    encoder.load_state_dict(encoder_sd, strict=False)
    predictor.load_state_dict(predictor_sd, strict=True)
    encoder.to(device)
    predictor.to(device)
    encoder.eval()
    predictor.eval()
    return encoder, predictor


@pytest.fixture(scope="module")
def synthetic_inputs(ac_config: VJepa2Config, device: torch.device):
    """Deterministic inputs usable for both pipelines.

    Shapes match the real AC data path:

      * ``video``: ``[1, T, C, H, W]``  — already normalized (pretend
        VJEPA2VideoProcessor has run).
      * ``actions``: ``[1, T_action, action_embed_dim]``
      * ``states``:  ``[1, T_action, action_embed_dim]``

    where ``T_action = num_frames // tubelet_size``.  Upstream ViT-g-AC
    expects tubelet-aligned action sequences.
    """
    torch.manual_seed(0xAC01)
    cfg = ac_config
    T = cfg.frames_per_clip
    C = cfg.in_chans
    H = W = cfg.crop_size
    video = torch.randn(1, T, C, H, W, device=device, dtype=torch.float32)
    T_a = T // cfg.tubelet_size
    A = cfg.ac_predictor.action_embed_dim
    actions = torch.linspace(-0.5, 0.5, T_a, device=device).view(1, T_a, 1).expand(1, T_a, A).contiguous()
    states = torch.linspace(0.1, 0.9, T_a, device=device).view(1, T_a, 1).expand(1, T_a, A).contiguous()
    return video, actions, states


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@torch.no_grad()
def test_ac_encoder_parity(our_modules, upstream_modules, synthetic_inputs):
    our_enc, _ = our_modules
    up_enc, _ = upstream_modules
    video, _actions, _states = synthetic_inputs

    # Our encoder expects [B, T, C, H, W]; upstream ``VisionTransformer``
    # operates on [B, C, T, H, W] (see vjepa2/src/models/vision_transformer.py).
    our_out = our_enc(video)
    up_out = up_enc(video.permute(0, 2, 1, 3, 4))
    assert our_out.shape == up_out.shape, (our_out.shape, up_out.shape)
    max_abs = (our_out - up_out).abs().max().item()
    print(f"[vjepa2-ac] encoder max_abs_diff = {max_abs:.6g}")
    assert max_abs < 1e-3, f"encoder parity broken: max_abs={max_abs}"


@torch.no_grad()
def test_ac_predictor_parity(our_modules, upstream_modules, synthetic_inputs):
    _, our_pred = our_modules
    up_enc, up_pred = upstream_modules
    video, actions, states = synthetic_inputs

    # Upstream feeds its own encoder output into its own predictor, so run
    # it first to get a reference ``x``.  Parity of the two encoders is
    # confirmed by the previous test, so using the upstream x for both
    # predictors isolates predictor-side numerics.
    x = up_enc(video.permute(0, 2, 1, 3, 4))

    up_out = up_pred(x, actions, states)
    our_out = our_pred(x, actions, states)
    assert our_out.shape == up_out.shape, (our_out.shape, up_out.shape)
    max_abs = (our_out - up_out).abs().max().item()
    print(f"[vjepa2-ac] predictor max_abs_diff = {max_abs:.6g}")
    assert max_abs < 1e-3, f"predictor parity broken: max_abs={max_abs}"
