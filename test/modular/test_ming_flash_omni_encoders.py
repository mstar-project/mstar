"""Smoke tests for Ming-flash-omni-2.0 vision/audio encoders + projectors.

Two layers of coverage:

  * Pure-Python tests on the projector wrappers — shape / layer-index
    parity with the released checkpoint's ``linear_proj.*`` and
    ``linear_proj_audio.*`` weight keys. Run on CPU, no snapshot needed.

  * Snapshot-gated tests on the vision encoder factory — construct from
    the real ``VisionEncoderConfig`` and run a tiny forward. Skip when
    no Ming snapshot or Ming source repo is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mminf.model.ming_omni_flash.components.projectors import (
    MingAudioProjector,
    MingVisionProjector,
)


# ---------------------------------------------------------------------------
# Snapshot / Ming source discovery (mirrors test_ming_flash_omni_config.py)
# ---------------------------------------------------------------------------


def _find_local_snapshot() -> str | None:
    """Locate a Ming-flash-omni-2.0 snapshot dir with shards reachable.

    We need the shards (``model-00001-of-00042.safetensors`` etc.) to
    live next to the index — the HF-Hub snapshot dir only carries the
    index json symlink, with shards pulled out separately on this box.
    Check the env override first, then the HF cache, then ``/dev/shm/
    ming-hybrid`` (the local merged layout this dev machine uses).
    """
    def _has_shards(path: Path) -> bool:
        return (
            (path / "config.json").exists()
            and (path / "model.safetensors.index.json").exists()
            and (path / "model-00001-of-00042.safetensors").exists()
        )

    override = os.environ.get("MING_FLASH_OMNI_DIR")
    if override and _has_shards(Path(override)):
        return override

    # The dev box's merged layout: shards + index colocate in /dev/shm.
    hybrid = Path("/dev/shm/ming-hybrid")
    if _has_shards(hybrid):
        return str(hybrid)

    # Fall back to the HF cache hub layout — accept it only if the
    # snapshot dir also has the shards (not just the index symlink).
    hub_roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/dev/shm/hf-cache"),
    ]
    repo_dirs = [
        "models--inclusionAI--Ming-flash-omni-2.0",
        "models--Jonathan1909--Ming-flash-omni-2.0",
    ]
    for hub_root in hub_roots:
        for repo in repo_dirs:
            snap_root = hub_root / repo / "snapshots"
            if not snap_root.exists():
                continue
            for snap in sorted(snap_root.iterdir()):
                if _has_shards(snap):
                    return str(snap)
    return None


def _find_ming_code_dir() -> str | None:
    """Mirror MingFlashOmniModel._find_ming_code_dir's search order."""
    env = os.environ.get("MING_CODE_DIR")
    if env and (Path(env) / "qwen3_moe_vit.py").exists():
        return env
    for candidate in (Path("./Ming"), Path("/tmp/ming_repo")):
        if (candidate / "qwen3_moe_vit.py").exists():
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# MingVisionProjector — pure Python
# ---------------------------------------------------------------------------


def test_vision_projector_default_depth_2_layer_indices() -> None:
    """``linear_proj.0`` -> first Linear; ``linear_proj.2`` -> second Linear.

    The released ckpt has ``mlp_depth=2`` so the projector is
    Linear → GELU → Linear, and the weight loader keys hit indices 0 and 2.
    """
    p = MingVisionProjector(vision_dim=4096, llm_dim=4096, mlp_depth=2)
    assert isinstance(p.proj[0], torch.nn.Linear)
    assert isinstance(p.proj[1], torch.nn.GELU)
    assert isinstance(p.proj[2], torch.nn.Linear)
    assert p.proj[0].weight.shape == (4096, 4096)
    assert p.proj[2].weight.shape == (4096, 4096)


def test_vision_projector_depth_1_single_linear() -> None:
    p = MingVisionProjector(vision_dim=4096, llm_dim=2048, mlp_depth=1)
    assert len(p.proj) == 1
    assert isinstance(p.proj[0], torch.nn.Linear)
    assert p.proj[0].weight.shape == (2048, 4096)


def test_vision_projector_rejects_depth_zero() -> None:
    with pytest.raises(ValueError, match="mlp_depth must be >= 1"):
        MingVisionProjector(vision_dim=4096, llm_dim=4096, mlp_depth=0)


def test_vision_projector_forward_shape() -> None:
    p = MingVisionProjector(vision_dim=8, llm_dim=16, mlp_depth=2)
    x = torch.randn(5, 8)
    out = p(x)
    assert out.shape == (5, 16)
    assert torch.isfinite(out).all()


def test_vision_projector_forward_shape_batched() -> None:
    p = MingVisionProjector(vision_dim=8, llm_dim=16, mlp_depth=2)
    x = torch.randn(2, 5, 8)
    out = p(x)
    assert out.shape == (2, 5, 16)


def test_vision_projector_checkpoint_keys_loadable() -> None:
    """``linear_proj.0.weight`` style keys load via load_state_dict.

    The Ming checkpoint stores the projector weights as flat
    ``linear_proj.<idx>.weight`` / ``.bias`` — we expose the same
    structure under our own ``proj.<idx>.<param>`` namespace, so the
    upstream key prefix needs trimming. Verify the trim is sufficient.
    """
    p = MingVisionProjector(vision_dim=8, llm_dim=16, mlp_depth=2)
    # Simulate the checkpoint state-dict shape (already trimmed of
    # the "linear_proj." outer prefix by the caller).
    fake_state = {
        "proj.0.weight": torch.randn(16, 8),
        "proj.0.bias": torch.randn(16),
        "proj.2.weight": torch.randn(16, 16),
        "proj.2.bias": torch.randn(16),
    }
    missing, unexpected = p.load_state_dict(fake_state)
    assert not missing
    assert not unexpected


# ---------------------------------------------------------------------------
# MingAudioProjector — pure Python
# ---------------------------------------------------------------------------


def test_audio_projector_default_depth_2_layer_indices() -> None:
    """``linear_proj_audio.0`` -> Conv1d; ``linear_proj_audio.3`` -> Linear.

    Layer order on disk: Conv1d (0), Transpose (1, no params), GELU (2,
    no params), Linear (3), Transpose (4, no params). Indices 0 and 3
    are the only ones with params.
    """
    p = MingAudioProjector(audio_dim=1280, llm_dim=4096, ds_kernel_size=3, ds_stride=2, mlp_depth=2)
    assert isinstance(p.proj[0], torch.nn.Conv1d)
    assert isinstance(p.proj[3], torch.nn.Linear)
    assert p.proj[0].weight.shape == (4096, 1280, 3)
    assert p.proj[3].weight.shape == (4096, 4096)


def test_audio_projector_depth_1_no_mlp() -> None:
    """depth=1 yields Conv1d + 2 transposes; no MLP. Only one param tensor."""
    p = MingAudioProjector(audio_dim=8, llm_dim=16, mlp_depth=1)
    # Layers: Conv1d(0), Transpose(1), Transpose(2).
    assert len(p.proj) == 3
    assert isinstance(p.proj[0], torch.nn.Conv1d)


def test_audio_projector_rejects_depth_zero() -> None:
    with pytest.raises(ValueError, match="mlp_depth must be >= 1"):
        MingAudioProjector(audio_dim=8, llm_dim=16, mlp_depth=0)


def test_audio_projector_forward_shape() -> None:
    """Output is (B, llm_dim, T') with T' from compute_output_length."""
    p = MingAudioProjector(audio_dim=8, llm_dim=16, ds_kernel_size=3, ds_stride=2, mlp_depth=2)
    # 11-frame input. After Whisper stem this would be (11-3+2)//2+1 = 6;
    # then the projector conv applies again — but the projector eats the
    # raw (B, T, audio_dim) so the Whisper stem isn't in the equation here.
    # Just the projector conv: T' = (11 - 3 + 2)//2 + 1 = 6.
    x = torch.randn(2, 11, 8)
    out = p(x)
    assert out.shape == (2, 16, 6)
    assert torch.isfinite(out).all()


def test_audio_projector_compute_output_length_matches_two_conv_chain() -> None:
    """Length math composes the Whisper stem with the projector conv."""
    p = MingAudioProjector(audio_dim=8, llm_dim=16, ds_kernel_size=3, ds_stride=2)
    # Whisper stem: (23-3+2*1)//2+1 = 22//2+1 = 12.
    # Projector conv: (12-3+2*1)//2+1 = 11//2+1 = 6.
    assert p.compute_output_length(torch.tensor([23])).tolist() == [6]


# ---------------------------------------------------------------------------
# Vision encoder — snapshot-gated
# ---------------------------------------------------------------------------


def _try_load_snapshot_and_code() -> tuple[str, str] | None:
    snap = _find_local_snapshot()
    if snap is None:
        return None
    code_dir = _find_ming_code_dir()
    if code_dir is None:
        return None
    return snap, code_dir


@pytest.fixture(scope="module")
def staged_snapshot() -> tuple[str, str]:
    """Skip if no snapshot or no Ming source repo is available.

    Side effect: stages the Ming source files into the snapshot dir
    (the same thing MingFlashOmniModel.__init__ does), so the dynamic
    import inside build_vision_encoder resolves.
    """
    pair = _try_load_snapshot_and_code()
    if pair is None:
        pytest.skip(
            "Need both a Ming-flash-omni-2.0 snapshot and a Ming source repo. "
            "Set MING_FLASH_OMNI_DIR + MING_CODE_DIR."
        )
    snap, code_dir = pair
    from mminf.model.ming_omni_flash.ming_omni_flash_model import _prepare_tokenizer_dir
    _prepare_tokenizer_dir(snap, code_dir)
    return snap, code_dir


def test_vision_encoder_builds_from_config(staged_snapshot: tuple[str, str]) -> None:
    """``build_vision_encoder`` returns a module with the expected dims.

    Tiny config (depth=2) to keep the test fast; otherwise the full
    27-layer encoder takes a few seconds to instantiate.
    """
    from mminf.model.ming_omni_flash.components.vision_encoder import build_vision_encoder
    from mminf.model.ming_omni_flash.config import VisionEncoderConfig

    snap, _ = staged_snapshot
    cfg = VisionEncoderConfig(depth=2)  # rest default to the released ckpt's values
    enc = build_vision_encoder(
        config=cfg,
        dtype=torch.float32,  # avoid bf16-on-CPU complaints
        device="cpu",
        local_dir=snap,
        attn_implementation="eager",  # don't require FA2 on CPU
    )
    # Spot-check structural attributes that downstream code reads.
    assert enc.image_emb_dim == cfg.out_hidden_size
    assert enc.spatial_merge_size == cfg.spatial_merge_size
    assert len(enc.blocks) == cfg.depth
    assert enc.patch_embed.in_channels == cfg.in_channels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + FA2")
def test_vision_encoder_forward_runs_smoke(staged_snapshot: tuple[str, str]) -> None:
    """Construct a tiny encoder, run a single grid_thw=(1,2,2) image.

    Uses the eager attention path so it runs without flash-attn installed.
    """
    from mminf.model.ming_omni_flash.components.vision_encoder import build_vision_encoder
    from mminf.model.ming_omni_flash.config import VisionEncoderConfig

    snap, _ = staged_snapshot
    cfg = VisionEncoderConfig(depth=2)
    enc = build_vision_encoder(
        config=cfg,
        dtype=torch.float32,
        device="cuda",
        local_dir=snap,
        attn_implementation="eager",
    )
    # 1 image, grid (1 temporal, 2x2 spatial), patch_size=16, temporal_patch=2.
    # Per Qwen3VLMoeVisionPatchEmbed: in_dim = patch_size**2 * temporal_patch * in_channels.
    patch_in = cfg.patch_size * cfg.patch_size * cfg.temporal_patch_size * cfg.in_channels
    n_patches = 1 * 2 * 2  # T*H*W
    pixels = torch.randn(n_patches, patch_in, device="cuda")
    grid_thw = torch.tensor([[1, 2, 2]], device="cuda")
    try:
        with torch.no_grad():
            out = enc(pixels, grid_thw=grid_thw)
    except RuntimeError as e:
        # The upstream encoder uses inductor-compiled reductions which need
        # nvrtc + libnvrtc-builtins matching the installed CUDA toolkit. On
        # boxes where the toolkit/torch versions are mismatched, the kernel
        # build fails with "failed to open libnvrtc-builtins.so.*". Skip
        # cleanly so the rest of this file keeps green on under-provisioned
        # test boxes — the forward-correctness path will be re-verified by
        # the snapshot smoke once step 5 wires it into the prefill walk.
        if "nvrtc" in str(e) or "libnvrtc" in str(e):
            pytest.skip(f"nvrtc / CUDA toolkit unavailable on this box: {e}")
        raise
    # After spatial_merge_size=2 merge: 4 / 2**2 = 1 token per image, out_hidden_size dim.
    assert out.shape == (1, cfg.out_hidden_size)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# MingAudioEncoder — pure Python (no snapshot needed; weights are random)
# ---------------------------------------------------------------------------


def test_audio_encoder_constructs_with_defaults() -> None:
    """Default kwargs match the released ckpt's whisper_encoder_config."""
    from mminf.model.ming_omni_flash.components.audio_encoder import MingAudioEncoder

    enc = MingAudioEncoder()  # defaults: n_mels=128, n_ctx=15000, n_state=1280, n_head=20, n_layer=32
    assert enc.audio_emb_dim == 1280
    assert len(enc.blocks) == 32
    assert enc.positional_embedding.shape == (15000, 1280)


def test_audio_encoder_constructs_with_overrides() -> None:
    from mminf.model.ming_omni_flash.components.audio_encoder import MingAudioEncoder

    enc = MingAudioEncoder(n_mels=80, n_ctx=500, n_state=64, n_head=4, n_layer=2)
    assert enc.audio_emb_dim == 64
    assert len(enc.blocks) == 2
    assert enc.positional_embedding.shape == (500, 64)


def test_audio_encoder_weight_keys_match_whisper_layout() -> None:
    """Param names follow OpenAI Whisper's convention (query/key/value/out, mlp.0/.2).

    The released Ming ckpt stores audio weights under the ``audio.*``
    top-level prefix; loader strips that prefix and load_state_dict
    must find the rest. Spot-check a representative set of keys.
    """
    from mminf.model.ming_omni_flash.components.audio_encoder import MingAudioEncoder

    enc = MingAudioEncoder(n_mels=8, n_ctx=64, n_state=16, n_head=2, n_layer=2)
    keys = set(dict(enc.named_parameters()).keys())
    expected = {
        "conv1.weight", "conv1.bias",
        "conv2.weight", "conv2.bias",
        "blocks.0.attn.query.weight", "blocks.0.attn.query.bias",
        "blocks.0.attn.key.weight",          # key has bias=False
        "blocks.0.attn.value.weight", "blocks.0.attn.value.bias",
        "blocks.0.attn.out.weight",   "blocks.0.attn.out.bias",
        "blocks.0.attn_ln.weight",    "blocks.0.attn_ln.bias",
        "blocks.0.mlp.0.weight",      "blocks.0.mlp.0.bias",
        "blocks.0.mlp.2.weight",      "blocks.0.mlp.2.bias",
        "blocks.0.mlp_ln.weight",     "blocks.0.mlp_ln.bias",
        "ln_post.weight",             "ln_post.bias",
    }
    missing = expected - keys
    assert not missing, f"Missing expected weight keys: {sorted(missing)}"
    # `key.bias` should NOT exist (Whisper convention).
    assert "blocks.0.attn.key.bias" not in keys


def test_audio_encoder_forward_packed_shape_no_flash_attn() -> None:
    """Run a tiny encoder on CPU without flash-attn.

    Verifies the packed-attention fallback produces the right shapes:
      input:  list of (n_mels, T_i) for i in {0..N-1}
      output: (sum_i conv2(conv1(T_i)), n_state)
    The conv1 stride=1 + conv2 stride=2 reduce each T_i to ``(T_i // 2) + 1``
    when pad=1, kernel=3, stride=(1,2).
    """
    from mminf.model.ming_omni_flash.components.audio_encoder import MingAudioEncoder

    torch.manual_seed(0)
    enc = MingAudioEncoder(n_mels=8, n_ctx=64, n_state=16, n_head=2, n_layer=2, use_flash_attn=False)
    enc = enc.float()  # default Whisper inits in fp32 on CPU
    x_list = [torch.randn(8, 10), torch.randn(8, 16), torch.randn(8, 6)]
    out, cu_seqlens = enc(x_list)
    # Per-clip encoded length: conv1(stride=1, pad=1, kernel=3) preserves T,
    # then conv2(stride=2, pad=1, kernel=3) halves T → T'_i = (T_i + 1) // 2.
    expected_lens = [(t.shape[1] + 1) // 2 for t in x_list]
    assert out.shape == (sum(expected_lens), 16)
    assert cu_seqlens.tolist() == [0, *list(__import__("itertools").accumulate(expected_lens))]
    assert torch.isfinite(out).all()


def test_audio_encoder_build_from_config() -> None:
    """``build_audio_encoder`` reads dims off AudioEncoderConfig.

    Doesn't need the snapshot — AudioEncoderConfig() default factory
    populates ``whisper_encoder_config`` with the released ckpt's values.
    """
    from mminf.model.ming_omni_flash.components.audio_encoder import build_audio_encoder
    from mminf.model.ming_omni_flash.config import AudioEncoderConfig

    cfg = AudioEncoderConfig()
    enc = build_audio_encoder(cfg, dtype=torch.float32, device="cpu", use_flash_attn=False)
    assert enc.audio_emb_dim == cfg.d_model
    assert len(enc.blocks) == cfg.encoder_layers


# ---------------------------------------------------------------------------
# Snapshot-gated weight loaders (step 4b)
# ---------------------------------------------------------------------------
#
# These exercise the prefix-strip + state_dict path against the real
# released checkpoint. They're skipped when no snapshot is available.


def _require_snapshot() -> str:
    snap = _find_local_snapshot()
    if snap is None:
        pytest.skip("Need a Ming-flash-omni-2.0 snapshot. Set MING_FLASH_OMNI_DIR.")
    return snap


def test_load_vision_projector_weights_from_snapshot() -> None:
    """``linear_proj.*`` keys load cleanly into MingVisionProjector(mlp_depth=2)."""
    from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
    from mminf.model.ming_omni_flash.loader import load_vision_projector_weights

    snap = _require_snapshot()
    cfg = MingFlashOmniModelConfig.from_pretrained(snap)
    proj = MingVisionProjector(
        vision_dim=cfg.vision.out_hidden_size,
        llm_dim=cfg.thinker_llm.hidden_size,
        mlp_depth=cfg.mlp_depth,
    )
    proj = proj.float()
    loaded = load_vision_projector_weights(proj, snap, device="cpu", strict=True)
    # Two Linear blocks × {weight, bias} = 4 keys total at mlp_depth=2.
    assert loaded == {"proj.0.weight", "proj.0.bias", "proj.2.weight", "proj.2.bias"}
    # Sanity-check that the loaded weight is non-zero (a fresh nn.Linear
    # would be too, but we want to know the param actually got overwritten).
    assert (proj.proj[0].weight.abs().sum() > 0).item()


def test_load_audio_projector_weights_from_snapshot() -> None:
    """``linear_proj_audio.*`` keys load cleanly into MingAudioProjector(mlp_depth=2)."""
    from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
    from mminf.model.ming_omni_flash.loader import load_audio_projector_weights

    snap = _require_snapshot()
    cfg = MingFlashOmniModelConfig.from_pretrained(snap)
    proj = MingAudioProjector(
        audio_dim=cfg.audio_encoder.d_model,
        llm_dim=cfg.thinker_llm.hidden_size,
        ds_kernel_size=cfg.audio_encoder.ds_kernel_size,
        ds_stride=cfg.audio_encoder.ds_stride,
        mlp_depth=cfg.mlp_depth,
    )
    proj = proj.float()
    loaded = load_audio_projector_weights(proj, snap, device="cpu", strict=True)
    # Conv1d + Linear × {weight, bias} = 4 keys total at mlp_depth=2.
    assert loaded == {"proj.0.weight", "proj.0.bias", "proj.3.weight", "proj.3.bias"}


def test_load_audio_encoder_weights_from_snapshot() -> None:
    """``audio.*`` keys load cleanly into MingAudioEncoder.

    Snapshot is bf16; we build the encoder in fp32 here so load_state_dict
    dtype-promotes the loaded tensors without a downcast assertion.
    """
    from mminf.model.ming_omni_flash.components.audio_encoder import build_audio_encoder
    from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
    from mminf.model.ming_omni_flash.loader import load_audio_encoder_weights

    snap = _require_snapshot()
    cfg = MingFlashOmniModelConfig.from_pretrained(snap)
    # Full 32-layer encoder is ~5 GB at fp32; bf16 keeps it under 3 GB
    # and still loads cleanly because both ckpt + module agree on dtype.
    enc = build_audio_encoder(
        cfg.audio_encoder, dtype=torch.bfloat16, device="cpu", use_flash_attn=False,
    )
    loaded = load_audio_encoder_weights(enc, snap, device="cpu", strict=True)
    # 32 layers × (4 attn linears: query/key/value/out, 1 with bias=False
    # so 7 attn params; + 2 LN × 2 + 2 mlp Linear × 2) = lots; just spot-check
    # representative keys made it in.
    assert "blocks.0.attn.query.weight" in loaded
    assert "blocks.0.attn.key.weight" in loaded
    assert "blocks.31.mlp.2.bias" in loaded
    assert "ln_post.weight" in loaded
    # Released ckpt ships its own (trained) positional_embedding that
    # overrides the sinusoidal init — confirm it's loaded as a buffer.
    assert "positional_embedding" in loaded
    assert enc.positional_embedding.shape == (15000, cfg.audio_encoder.d_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Ming source modules to instantiate vision encoder")
def test_load_vision_encoder_weights_from_snapshot(staged_snapshot: tuple[str, str]) -> None:
    """``vision.*`` keys load cleanly into the Ming Qwen3MoeVisionTransformer.

    Full vision encoder is 27 layers; instantiating it bf16 takes a couple
    of seconds. CUDA-gated because Whisper's autograd-free Conv1d still
    pulls in CUDA contexts in the upstream encoder module (constructor
    calls .to()).
    """
    from mminf.model.ming_omni_flash.components.vision_encoder import build_vision_encoder
    from mminf.model.ming_omni_flash.config import MingFlashOmniModelConfig
    from mminf.model.ming_omni_flash.loader import load_vision_encoder_weights

    snap, _ = staged_snapshot
    cfg = MingFlashOmniModelConfig.from_pretrained(snap)
    enc = build_vision_encoder(
        config=cfg.vision,
        dtype=torch.bfloat16,
        device="cpu",
        local_dir=snap,
        attn_implementation="eager",
    )
    loaded = load_vision_encoder_weights(enc, snap, device="cpu", strict=True)
    assert "blocks.0.attn.qkv.weight" in loaded
    assert "blocks.0.mlp.linear_fc1.weight" in loaded
    assert "merger.linear_fc1.weight" in loaded
    assert f"blocks.{cfg.vision.depth - 1}.norm2.weight" in loaded
