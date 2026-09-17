"""CPU checks for the Cosmos3-Edge backbone family and reasoner plumbing.

Everything here runs without a GPU. The tiny-config checks need no weights;
the checkpoint-structure checks need the Cosmos3-Edge snapshot (its JSON
configs and safetensors headers only) and skip when ``COSMOS3_EDGE_DIR`` is
unset and the shared HF cache has no copy.

Run: python -m pytest mstar/model/cosmos3/tests/test_edge.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from mstar.model.cosmos3.components.reasoner import (
    MediaGrid,
    mrope_position_ids,
    patchify,
    preprocess_image,
    preprocess_video,
    sample_frame_indices,
    smart_resize,
)
from mstar.model.cosmos3.components.transformer import (
    Cosmos3OmniTransformer,
    NemotronRMSNorm,
    RMSNorm,
    norm_class_for,
)
from mstar.model.cosmos3.config import (
    Cosmos3Config,
    Cosmos3MediaProcessorConfig,
    Cosmos3ReasonerConfig,
    Cosmos3VisionEncoderConfig,
)


def _edge_dir() -> Path | None:
    env = os.environ.get("COSMOS3_EDGE_DIR")
    if env:
        return Path(env)
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", ""), "hub")
    snaps = sorted(glob.glob(os.path.join(cache, "models--nvidia--Cosmos3-Edge", "snapshots", "*")))
    for snap in snaps:
        if os.path.exists(os.path.join(snap, "transformer", "config.json")):
            return Path(snap)
    return None


EDGE_DIR = _edge_dir()
needs_edge = pytest.mark.skipif(EDGE_DIR is None, reason="set COSMOS3_EDGE_DIR to a Cosmos3-Edge dir")


def _tiny_edge_config(**overrides) -> Cosmos3Config:
    """A CPU-cheap config with Edge's backbone family: relu2 MLPs, Nemotron
    norms, no text QK-norm, a k_norm_und_for_gen, and a reasoner."""
    cfg = Cosmos3Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        vocab_size=100,
        rope_axes_dim=(4, 2, 2),
        rope_theta=1e8,
        rms_norm_eps=1e-5,
        latent_channel=8,
        latent_patch_size=2,
        patch_latent_dim=32,
        sound_gen=False,
        action_gen=False,
        hidden_act="relu2",
        qk_norm_for_text=False,
        use_und_k_norm_for_gen=True,
        backbone_type="cosmos3_edge_nemotron_dense",
        use_native_flow_schedule=True,
        reasoner=Cosmos3ReasonerConfig(
            vision=Cosmos3VisionEncoderConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                              num_attention_heads=2, num_patches=16),
            projector_input_hidden_size=32, projector_hidden_size=48, projector_out_hidden_size=64,
        ),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _init_all(module: torch.nn.Module, seed: int = 0) -> None:
    """The parallel linears allocate uninitialized storage; give every
    parameter deterministic values so a CPU forward is finite."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("weight") and p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))


# ---------------------------------------------------------------------------
# Backbone family
# ---------------------------------------------------------------------------


def test_nemotron_norm_ordering_and_selection() -> None:
    torch.manual_seed(0)
    x = torch.randn(5, 64, dtype=torch.bfloat16) * 3
    norm = NemotronRMSNorm(64, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(64) + 0.5)
        norm.weight.data = norm.weight.data.to(torch.bfloat16)
    xf = x.float()
    ref = (norm.weight.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5))).to(torch.bfloat16)
    assert torch.equal(norm(x), ref)
    assert norm_class_for(_tiny_edge_config()) is NemotronRMSNorm
    assert norm_class_for(Cosmos3Config()) is RMSNorm


def test_edge_family_modules_and_state_dict() -> None:
    cfg = _tiny_edge_config()
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)
    layer = model.layers[0]
    # No text QK-norm params, one k_norm_und_for_gen, relu2 (up/down only) MLPs, an lm_head.
    keys = set(model.state_dict())
    assert "layers.0.self_attn.norm_q.weight" not in keys and "layers.0.self_attn.norm_k.weight" not in keys
    assert "layers.0.self_attn.k_norm_und_for_gen.weight" in keys
    assert "layers.0.mlp.up_proj.weight" in keys and "layers.0.mlp.gate_proj.weight" not in keys
    assert "layers.0.mlp_moe_gen.down_proj.weight" in keys
    assert "lm_head.weight" in keys
    assert isinstance(layer.input_layernorm, NemotronRMSNorm)
    assert isinstance(layer.self_attn.norm_added_q, NemotronRMSNorm)
    # Nano keeps its family.
    with torch.device("meta"):
        nano = Cosmos3OmniTransformer(Cosmos3Config(num_hidden_layers=1))
    nkeys = set(nano.state_dict())
    assert "layers.0.self_attn.norm_q.weight" in nkeys and "layers.0.mlp.gate_proj.weight" in nkeys
    assert "lm_head.weight" not in nkeys and "layers.0.self_attn.k_norm_und_for_gen.weight" not in nkeys


def test_relu2_mlp_forward_matches_reference() -> None:
    cfg = _tiny_edge_config()
    model = Cosmos3OmniTransformer(cfg)
    _init_all(model)
    mlp = model.layers[0].mlp
    x = torch.randn(3, cfg.hidden_size)
    ref = F.linear(torch.square(F.relu(F.linear(x, mlp.up_proj.weight))), mlp.down_proj.weight)
    assert torch.allclose(mlp(x), ref, atol=1e-6)


def test_fused_gen_attends_to_normed_und_k() -> None:
    """The fused reference pass: GEN attends to k_norm_und_for_gen(K_und)
    while UND self-attention keeps the raw K — recomputed here by hand."""
    cfg = _tiny_edge_config()
    model = Cosmos3OmniTransformer(cfg).eval()
    _init_all(model)
    attn = model.layers[0].self_attn
    und = torch.randn(5, cfg.hidden_size)
    gen = torch.randn(7, cfg.hidden_size)
    cos_u, sin_u = torch.randn(5, cfg.head_dim), torch.randn(5, cfg.head_dim)
    cos_g, sin_g = torch.randn(7, cfg.head_dim), torch.randn(7, cfg.head_dim)
    out_u, out_g = attn(und, gen, (cos_u, sin_u, cos_g, sin_g))

    H, Hkv, D = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    q_u = attn._apply_rope(attn.to_q(und).view(-1, H, D), cos_u, sin_u)
    k_raw = attn.to_k(und).view(-1, Hkv, D)
    k_u = attn._apply_rope(k_raw, cos_u, sin_u)
    k_u_gen = attn._apply_rope(attn.k_norm_und_for_gen(k_raw), cos_u, sin_u)
    v_u = attn.to_v(und).view(-1, Hkv, D)
    q_g = attn._apply_rope(attn.norm_added_q(attn.add_q_proj(gen).view(-1, H, D)), cos_g, sin_g)
    k_g = attn._apply_rope(attn.norm_added_k(attn.add_k_proj(gen).view(-1, Hkv, D)), cos_g, sin_g)
    v_g = attn.add_v_proj(gen).view(-1, Hkv, D)
    ref_u = attn.to_out(attn._attend(q_u, k_u, v_u, is_causal=True))
    ref_g = attn.to_add_out(attn._attend(q_g, torch.cat([k_u_gen, k_g]), torch.cat([v_u, v_g]), is_causal=False))
    assert torch.allclose(out_u, ref_u, atol=1e-5)
    assert torch.allclose(out_g, ref_g, atol=1e-5)
    # ...and with the raw K it would differ: the norm is not a no-op.
    wrong_g = attn.to_add_out(attn._attend(q_g, torch.cat([k_u, k_g]), torch.cat([v_u, v_g]), is_causal=False))
    assert not torch.allclose(out_g, wrong_g, atol=1e-3)


class _OverwriteKV:
    """A CPU stand-in for the kv + attn resources over one request: the
    UND prefill writes K/V, attends with the K it passed, and the GEN-facing
    re-write replaces what the cache holds. Denoise steps then read the
    committed prefix."""

    requires_kv_write = True

    def __init__(self):
        self._label = "main"
        self._layer = 0
        self.committed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.pending: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.causal = True

    @property
    def default_label(self):
        return self._label

    def set_default_label(self, label):
        self._label = label

    def set_default_layer_idx(self, i):
        self._layer = i

    def layer_view(self, layer_idx=None):
        return self._layer if layer_idx is None else layer_idx

    def write_kv(self, k, v, layer_idx=None, label=None):
        self.pending[self._layer if layer_idx is None else layer_idx] = (k, v)

    def commit(self):
        self.committed.update(self.pending)
        self.pending = {}

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        prefix = self.committed.get(kv_cache_layer)
        if prefix is not None:
            k = torch.cat([prefix[0], k])
            v = torch.cat([prefix[1], v])
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2),
            is_causal=self.causal, enable_gqa=True,
        )
        return out.transpose(1, 2).squeeze(0)


def test_cached_prefill_writes_gen_facing_k() -> None:
    """Cache-once path == fused reference on Edge: the prefill must leave the
    normed (GEN-facing) K in the cache while attending with the raw K."""
    from mstar.model.cosmos3.components.packing import build_static_inputs

    cfg = _tiny_edge_config()
    model = Cosmos3OmniTransformer(cfg).eval()
    _init_all(model)
    res = _OverwriteKV()
    for child in model.modules():
        bind = getattr(child, "bind_resources", None)
        if bind is not None:
            bind({"kv": res, "attn": res})

    ids = [3, 5, 7, 11, 13]
    latent = torch.randn(1, cfg.latent_channel, 1, 4, 4)
    static = build_static_inputs(ids, tuple(latent.shape), cfg, 4, 24.0, "cpu")
    fields = ("input_ids", "text_indexes", "position_ids", "und_len", "sequence_length", "vision_token_shapes",
              "vision_sequence_indexes", "vision_mse_loss_indexes", "vision_noisy_frame_indexes")
    ts = torch.full((static["num_noisy_vision_tokens"],), 500.0)
    with torch.no_grad():
        fused, _ = model(vision_tokens=[latent], vision_timesteps=ts, **{k: static[k] for k in fields})
        res.causal = True
        model.prefill_und(static["input_ids"], static["text_mrope_ids"], "main")
        res.commit()
        res.causal = False
        cached = model.denoise_step(
            latent, ts, static["vision_mrope_ids"], static["vision_token_shapes"],
            static["vision_noisy_frame_indexes"], static["vision_mse_loss_indexes"] - static["und_len"],
            "main", res,
        )
    assert torch.allclose(fused[0], cached, atol=1e-4), (fused[0] - cached).abs().max()
    # The committed prefix K is the normed one, not the raw K.
    attn0 = model.layers[0].self_attn
    und_norm = model.layers[0].input_layernorm(model.embed_tokens(static["input_ids"]))
    cos, sin = model._rotary(static["text_mrope_ids"], und_norm.device, und_norm.dtype)
    k_raw = attn0.to_k(und_norm).view(-1, attn0.num_key_value_heads, attn0.head_dim)
    expected = attn0._apply_rope(attn0.k_norm_und_for_gen(k_raw), cos, sin)
    assert torch.allclose(res.committed[0][0], expected, atol=1e-5)


def test_text_forward_keeps_raw_k() -> None:
    """The reasoner path caches the raw (RoPE'd) K, like any causal LM."""
    cfg = _tiny_edge_config()
    model = Cosmos3OmniTransformer(cfg).eval()
    _init_all(model)
    res = _OverwriteKV()
    for child in model.modules():
        bind = getattr(child, "bind_resources", None)
        if bind is not None:
            bind({"kv": res, "attn": res})
    ids = torch.tensor([1, 2, 3, 4])
    pos = torch.arange(4).view(1, -1).expand(3, -1)
    with torch.no_grad():
        embeds = model.embed_tokens(ids)
        hidden = model.text_forward(embeds, pos, "main")
        res.commit()
        logits = model.lm_head(hidden)
    assert hidden.shape == (4, cfg.hidden_size) and logits.shape == (4, cfg.vocab_size)
    attn0 = model.layers[0].self_attn
    und_norm = model.layers[0].input_layernorm(embeds)
    cos, sin = model._rotary(pos, embeds.device, embeds.dtype)
    k_raw = attn0._apply_rope(attn0.to_k(und_norm).view(-1, attn0.num_key_value_heads, attn0.head_dim), cos, sin)
    assert torch.allclose(res.committed[0][0], k_raw, atol=1e-6)


def test_conditioning_frame_recipes() -> None:
    from mstar.model.cosmos3.components.conditioning import prepare_conditioning_frames

    torch.manual_seed(0)
    img = torch.rand(3, 90, 160)  # 16:9 source
    # Stretch: plain resize to a square target, aspect not preserved.
    out = prepare_conditioning_frames(img, 64, 64, "stretch")
    assert out.shape == (1, 3, 1, 64, 64) and out.min() >= -1 and out.max() <= 1
    ref = torch.nn.functional.interpolate(img.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
    assert torch.allclose(out[:, :, 0], ref * 2 - 1, atol=1e-6)
    # Aspect crop: cover-scale (90x160 -> 64x114), center crop to 64x64,
    # values quantized to 8-bit steps.
    out = prepare_conditioning_frames(img, 64, 64, "aspect_crop")
    assert out.shape == (1, 3, 1, 64, 64)
    steps = (out + 1.0) * 127.5
    assert torch.allclose(steps, steps.round(), atol=1e-4)
    full = torch.nn.functional.interpolate(img.unsqueeze(0) * 255, size=(64, 114), mode="bilinear",
                                           align_corners=False, antialias=True)
    crop = full[:, :, :, 25:89].round().clamp(0, 255) / 127.5 - 1
    assert torch.allclose(out[:, :, 0], crop, atol=1e-6)
    # Same-aspect targets crop nothing; 8-bit inputs and video stacks work too.
    vid = (torch.rand(4, 3, 45, 80) * 255).to(torch.uint8)
    out = prepare_conditioning_frames(vid, 90, 160, "aspect_crop")
    assert out.shape == (1, 3, 4, 90, 160)
    with pytest.raises(ValueError, match="conditioning_resize"):
        prepare_conditioning_frames(img, 64, 64, "pad")


def test_native_flow_sigmas() -> None:
    from mstar.model.cosmos3.submodules import native_flow_sigmas

    sig = native_flow_sigmas(4, 1000)
    assert len(sig) == 4 and abs(sig[0] - 0.999) < 1e-9 and sig[-1] > 0
    assert all(a > b for a, b in zip(sig, sig[1:], strict=False))


# ---------------------------------------------------------------------------
# Reasoner prompt plumbing
# ---------------------------------------------------------------------------

_PROC = Cosmos3MediaProcessorConfig()


def test_smart_resize_bounds_and_factor() -> None:
    h, w = smart_resize(1, 341, 512, 1, 32, _PROC.min_pixels, _PROC.max_pixels)
    assert (h, w) == (352, 512) and h % 32 == 0 and w % 32 == 0
    # Too small: scaled up to min_pixels; too large: scaled down to max_pixels.
    h, w = smart_resize(1, 64, 64, 1, 32, 65536, 16777216)
    assert h * w >= 65536
    h, w = smart_resize(1, 8000, 8000, 1, 32, 65536, 16777216)
    assert h * w <= 16777216
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize(1, 10, 3000, 1, 32, 65536, 16777216)


def test_patchify_block_major_order() -> None:
    # 2 x 2 merge blocks of 2 x 2 patches over a 2-channel 8x8 frame whose
    # pixel value encodes its (row, col): the first block holds the four
    # top-left patches in row-major block order.
    p, m = 2, 2
    frame = torch.zeros(1, 2, 8, 8)
    frame[0, 0] = torch.arange(8).view(8, 1).expand(8, 8)  # row
    frame[0, 1] = torch.arange(8).view(1, 8).expand(8, 8)  # col
    patches = patchify(frame, p, m)
    assert patches.shape == (16, p * p * 2)
    # patch k inside a patch: values ordered (ph, pw, C)
    first = patches[0].view(p, p, 2)
    assert first[..., 0].tolist() == [[0, 0], [1, 1]] and first[..., 1].tolist() == [[0, 1], [0, 1]]
    second = patches[1].view(p, p, 2)  # next patch in the same block: to the right
    assert second[..., 1].tolist() == [[2, 3], [2, 3]] and second[..., 0].tolist() == [[0, 0], [1, 1]]
    third = patches[2].view(p, p, 2)  # below the first
    assert third[..., 0].tolist() == [[2, 2], [3, 3]]
    fifth = patches[4].view(p, p, 2)  # first patch of the next block (to the right)
    assert fifth[..., 1].tolist() == [[4, 5], [4, 5]] and fifth[..., 0].tolist() == [[0, 0], [1, 1]]


def test_preprocess_image_and_video_shapes() -> None:
    img = torch.rand(3, 341, 512)
    pv, grid = preprocess_image(img, _PROC)
    assert grid == MediaGrid(1, 22, 32) and pv.shape == (22 * 32, 3 * 16 * 16)
    assert grid.tokens(2) == 176
    # 8-bit and [0, 1] float inputs give the same patches.
    pv8, _ = preprocess_image((img * 255).round().to(torch.uint8), _PROC)
    assert torch.equal(pv, pv8)
    vcfg = Cosmos3MediaProcessorConfig(min_pixels=4096, max_pixels=25165824)
    video = torch.rand(48, 3, 90, 160)
    pvv, vgrid = preprocess_video(video, vcfg, source_fps=24.0)
    assert vgrid.t == 4 and len(vgrid.timestamps) == 4 and vgrid.timestamps[0] == 0.0
    assert pvv.shape == (vgrid.t * vgrid.h * vgrid.w, 768)


def test_sample_frame_indices() -> None:
    cfg = Cosmos3MediaProcessorConfig(fps=2.0, min_frames=4, max_frames=768)
    # 48 frames at 24 fps -> 2 s -> 4 frames at 2 fps, linspace-rounded.
    assert sample_frame_indices(48, 24.0, cfg) == [0, 16, 31, 47]
    # Short clips clamp up to min_frames, never past the clip.
    assert sample_frame_indices(3, 24.0, cfg) == [0, 1, 2]
    assert sample_frame_indices(100, 10.0, cfg, num_frames=5) == [0, 25, 50, 74, 99]
    with pytest.raises(ValueError):
        sample_frame_indices(10, 24.0, cfg, num_frames=2, fps=1.0)


def test_mrope_positions_image_then_text() -> None:
    cfg = Cosmos3ReasonerConfig()
    # [text x3][image 2x4 merged grid = 8 pads][text x2]
    ids = torch.tensor([5, 6, 7] + [cfg.image_token_id] * 8 + [8, 9])
    grid = MediaGrid(1, 4, 8)  # merged 2 x 4
    pos, nxt = mrope_position_ids(ids, cfg, [grid], [])
    assert pos[:, :3].tolist() == [[0, 1, 2]] * 3
    assert pos[0, 3:11].tolist() == [3] * 8                      # temporal: the cursor
    assert pos[1, 3:11].tolist() == [3, 3, 3, 3, 4, 4, 4, 4]     # height
    assert pos[2, 3:11].tolist() == [3, 4, 5, 6, 3, 4, 5, 6]     # width
    # The cursor advanced by max(2, 4) = 4: text resumes at 7.
    assert pos[:, 11:].tolist() == [[7, 8]] * 3
    assert nxt == 9
    with pytest.raises(ValueError, match="does not match"):
        mrope_position_ids(ids, cfg, [MediaGrid(1, 4, 4)], [])


def test_mrope_positions_video_frames() -> None:
    cfg = Cosmos3ReasonerConfig()
    grid = MediaGrid(2, 2, 2, timestamps=(0.0, 0.5))  # 1 merged token per frame
    ids = torch.tensor([1, cfg.video_token_id, 2, cfg.video_token_id, 3])
    pos, nxt = mrope_position_ids(ids, cfg, [], [grid])
    assert pos[:, 1].tolist() == [1, 1, 1] and pos[:, 2].tolist() == [2, 2, 2]
    assert pos[:, 3].tolist() == [3, 3, 3] and pos[:, 4].tolist() == [4, 4, 4]
    assert nxt == 5


# ---------------------------------------------------------------------------
# Checkpoint structure (needs the snapshot's JSON + safetensors headers)
# ---------------------------------------------------------------------------


@needs_edge
def test_edge_config_roundtrip() -> None:
    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    assert cfg.num_hidden_layers == 28 and cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 16 and cfg.num_key_value_heads == 8 and cfg.head_dim == 128
    assert cfg.intermediate_size == 9216 and cfg.vocab_size == 131072
    assert cfg.hidden_act == "relu2" and cfg.backbone_type == "cosmos3_edge_nemotron_dense"
    assert cfg.qk_norm_for_text is False and cfg.use_und_k_norm_for_gen is True
    assert cfg.nemotron_norm and not cfg.gated_mlp
    assert cfg.rope_theta == 1e8 and cfg.rms_norm_eps == 1e-5
    assert tuple(cfg.rope_axes_dim) == (24, 20, 20)
    assert cfg.sound_gen is False and cfg.action_gen is True and cfg.max_action_dim == 64
    assert cfg.use_native_flow_schedule is True
    r = cfg.reasoner
    assert r is not None and cfg.serves_reasoner
    assert r.vision.num_hidden_layers == 27 and r.vision.hidden_size == 1152 and r.vision.num_patches == 256
    assert r.projector_hidden_size == 11520 and r.projector_out_hidden_size == 2048
    assert (r.image_token_id, r.video_token_id, r.vision_start_token_id, r.vision_end_token_id) == (19, 18, 20, 21)
    assert r.eos_token_id == 11 and r.max_position_embeddings == 131072
    assert r.image_processor.min_pixels == 65536 and r.image_processor.max_pixels == 16777216
    assert r.video_processor.min_pixels == 4096 and r.video_processor.max_pixels == 25165824


@needs_edge
def test_edge_transformer_key_and_shape_coverage() -> None:
    from mstar.model.cosmos3.loader import (
        cosmos3_name_remapper,
        read_transformer_weight_keys,
        read_transformer_weight_shapes,
    )

    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    with torch.device("meta"):
        model = Cosmos3OmniTransformer(cfg)
    model_keys = set(model.state_dict())
    index_keys = read_transformer_weight_keys(EDGE_DIR)
    # A reasoner backbone keeps lm_head; every key maps one-to-one.
    mapped = {cosmos3_name_remapper(k, with_lm_head=True) for k in index_keys}
    assert mapped == model_keys, (sorted(model_keys - mapped)[:5], sorted(mapped - model_keys)[:5])
    assert len(index_keys) == 549
    try:
        shapes = read_transformer_weight_shapes(EDGE_DIR)
    except Exception as exc:  # noqa: BLE001 — LFS pointers / missing shards
        pytest.skip(f"transformer shards unreadable: {exc}")
    mismatched = {k: (tuple(v.shape), shapes.get(k)) for k, v in model.state_dict().items()
                  if tuple(v.shape) != shapes.get(k)}
    assert not mismatched, mismatched


@needs_edge
def test_edge_vision_encoder_key_and_shape_coverage() -> None:
    from mstar.model.cosmos3.components.vision import Cosmos3VisionModel
    from mstar.model.cosmos3.loader import read_vision_encoder_weight_shapes

    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    with torch.device("meta"):
        model = Cosmos3VisionModel(cfg.reasoner)
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    try:
        ckpt = read_vision_encoder_weight_shapes(EDGE_DIR)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"vision encoder shard unreadable: {exc}")
    missing = sorted(set(model_shapes) - set(ckpt))[:5]
    unexpected = sorted(set(ckpt) - set(model_shapes))[:5]
    assert set(model_shapes) == set(ckpt), (missing, unexpected)
    assert {k for k, s in model_shapes.items() if s != ckpt[k]} == set()
    assert len(ckpt) == 443


@needs_edge
def test_edge_prompt_rendering_matches_reference_layout() -> None:
    """The rendered reasoner prompt: chat template + one pad per merged block,
    the position ids advancing by the merged grid's longer side."""
    from transformers import AutoTokenizer

    from mstar.model.cosmos3.components.reasoner import expand_placeholders, render_chat
    from mstar.model.multimodal import PromptPart

    cfg = Cosmos3Config.from_pretrained(EDGE_DIR)
    tok = AutoTokenizer.from_pretrained(str(EDGE_DIR))
    grid = MediaGrid(1, 22, 32)
    parts = [PromptPart("image", None, 0), PromptPart("text", "Describe the scene.")]
    text = render_chat(tok, parts, cfg.reasoner, enable_thinking=False)
    assert text.endswith("<|im_start|>assistant\n<think></think>")
    expanded = expand_placeholders(text, tok, cfg.reasoner, [grid], [])
    ids = torch.tensor(tok(expanded, add_special_tokens=False)["input_ids"])
    assert int((ids == cfg.reasoner.image_token_id).sum()) == 176
    pos, nxt = mrope_position_ids(ids, cfg.reasoner, [grid], [])
    start = int((ids == cfg.reasoner.vision_start_token_id).nonzero()[0]) + 1
    assert pos[0, start:start + 176].unique().tolist() == [start]
    assert pos[1, start:start + 176].max().item() == start + 10 and pos[2, start:start + 176].max().item() == start + 15
    assert nxt == int(pos.max()) + 1
    # Thinking on by default: the generation prompt opens a <think> block.
    assert render_chat(tok, parts, cfg.reasoner).endswith("<think>\n")


@needs_edge
def test_edge_reasoner_video_prompt() -> None:
    """A video attachment renders one timestamped span per sampled frame, the
    packed patches cover every frame, and the positions advance per frame."""
    from transformers import AutoTokenizer

    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.multimodal import PromptPart

    model = Cosmos3Model(model_path_hf=str(EDGE_DIR), skip_weight_loading=True)
    model.tokenizer = AutoTokenizer.from_pretrained(str(EDGE_DIR))
    r = model.config.reasoner
    video = torch.rand(30, 3, 96, 160)  # 30 frames at 10 fps -> 3 s -> 6 frames at 2 fps
    out = model.process_prompt(
        None, ["video", "text"], ["text"], tensors={"video_inputs": [video]},
        prompt_parts=[PromptPart("video", None, 0), PromptPart("text", "What happens?")],
        input_metadata={"video_inputs": [{"average_fps": 10.0, "num_frames": 30}]},
        enable_thinking=False,
    )
    ids = out["text_inputs"][0]
    grid = out["vision_grid_thw"][0]
    assert grid.shape == (1, 3) and int(grid[0, 0]) == 6
    per_frame = int(grid[0, 1] * grid[0, 2]) // 4
    assert int((ids == r.video_token_id).sum()) == 6 * per_frame
    assert int((ids == r.vision_start_token_id).sum()) == 6
    assert out["pixel_values"][0].shape[0] == int(grid[0].prod())
    pos = out["position_ids"][0]
    starts = (ids == r.vision_start_token_id).nonzero().flatten().tolist()
    # Frame k's tokens sit at one temporal position; successive frames are
    # separated by the timestamp text plus the merged grid's longer side.
    temporal = [int(pos[0, s + 1]) for s in starts]
    assert temporal == sorted(temporal) and len(set(temporal)) == 6
    rendered = model.tokenizer.decode(ids[: starts[1]])
    assert "<0.0 seconds>" in rendered
    # Explicit frame count / sampling rate overrides.
    out2 = model.process_prompt(
        None, ["video", "text"], ["text"], tensors={"video_inputs": [video]},
        prompt_parts=[PromptPart("video", None, 0), PromptPart("text", "What happens?")],
        input_metadata={"video_inputs": [{"average_fps": 10.0}]}, video_num_frames=4,
    )
    assert int(out2["vision_grid_thw"][0][0, 0]) == 4
