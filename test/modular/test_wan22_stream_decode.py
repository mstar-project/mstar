"""CPU checks for Wan2.2's ``stream_decode``: the vae_decoder as a loop over pieces of the
clip, each piece's frames emitted as a ``video_part`` of one mp4. The pieces in order are
the whole-clip decode's frames bit for bit, on every decode path, and two requests'
streams interleaved on one node don't mix. Tiny random ``AutoencoderKLWan``; no weights,
no GPU."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.components.video_encode import encode_mp4  # noqa: E402
from mstar.model.wan22 import submodules as wan22_submodules  # noqa: E402
from mstar.model.wan22.config import WAN22_VARIANT_T2V_A14B, WAN22_VARIANT_TI2V_5B, Wan22Config  # noqa: E402
from mstar.model.wan22.submodules import DECODE_LOOP, DENOISE_LOOP_NAME, Wan22VaeDecoderSubmodule  # noqa: E402
from mstar.model.wan22.wan22_model import VIDEO_PART, Wan22Model  # noqa: E402

Z_DIM = 8
LISTED = ("96x64",)  # the 4x6 latent grid below decodes to 96x64
LISTED_A14B = ("48x32",)  # and to 48x32 on A14B's 8x VAE


def _tiny_vae(**layout):
    vae = AutoencoderKLWan(
        base_dim=8, z_dim=Z_DIM, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
        latents_mean=[0.1 * i for i in range(Z_DIM)], latents_std=[0.5 + 0.1 * i for i in range(Z_DIM)], **layout,
    ).eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return vae


@pytest.fixture(scope="module")
def vae():
    return _tiny_vae(
        decoder_base_dim=16, is_residual=True, in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
    )


@pytest.fixture(scope="module")
def vae_a14b():
    """The Wan2.1 VAE, which A14B decodes with: plain up blocks, no patch, 8x spatial."""
    return _tiny_vae()


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    # The tiling gate reads free VRAM; on CPU there is none to read.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2**40, 2**40))


def _latents(frames: int = 7, seed: int = 1) -> torch.Tensor:
    return torch.randn(1, Z_DIM, frames, 4, 6, generator=torch.Generator().manual_seed(seed))


def _decoder(monkeypatch, vae, *, per_part: int, fast: bool = True, tiling: str = "untiled",
             dtype=torch.float32, sizes=LISTED, variant: str = WAN22_VARIANT_TI2V_5B):
    monkeypatch.setattr(wan22_submodules, "wan_vae_decode_dtype", lambda: dtype)
    cfg = Wan22Config.for_variant(
        variant, vae_decode_tiling=tiling, vae_decode_fast=fast, vae_decode_fast_sizes=sizes,
        stream_decode=True, stream_decode_latents=per_part,
    )
    return Wan22VaeDecoderSubmodule(copy.deepcopy(vae), cfg)


def _whole(sub, latents):
    return sub.forward("video_gen", SimpleNamespace(request_ids=["whole"]), latents)["video_output"][0]


def _piece(sub, latents, part: int, request_id: str = "r"):
    out = sub.forward("video_gen", SimpleNamespace(request_ids=[request_id]), latents, part=part)
    return out["video_output"][0]


@pytest.mark.parametrize("per_part", [1, 3, 4, 7])
@pytest.mark.parametrize(("fast", "tiling", "dtype"), [
    (True, "untiled", torch.float32),   # the shared decoder, compiled: decoded piece by piece
    (True, "untiled", torch.bfloat16),
    (False, "untiled", torch.float32),  # stock vae.decode: decoded whole, cut
    (True, "tiled", torch.float32),     # tiled_decode: decoded whole, cut
])
@pytest.mark.parametrize("variant", [WAN22_VARIANT_TI2V_5B, WAN22_VARIANT_T2V_A14B])
def test_the_pieces_in_order_are_the_whole_decodes_frames(
    monkeypatch, request, variant, per_part, fast, tiling, dtype,
):
    a14b = variant == WAN22_VARIANT_T2V_A14B
    sub = _decoder(
        monkeypatch, request.getfixturevalue("vae_a14b" if a14b else "vae"), per_part=per_part, fast=fast,
        tiling=tiling, dtype=dtype, sizes=LISTED_A14B if a14b else LISTED, variant=variant,
    )
    z = _latents()
    want = _whole(sub, z)
    assert want.shape[1:3] == ((32, 48) if a14b else (64, 96))
    parts = sub.config.decode_parts(want.shape[0])
    assert parts == -(-z.shape[2] // per_part)
    pieces = [_piece(sub, z, 0)]
    if parts > 1:
        # only the fast path decodes piece by piece (the post-quantized clip waits in the request's state)
        assert ("stream_x" in sub.request_states["r"]) == (fast and tiling == "untiled")
    pieces += [_piece(sub, z, part) for part in range(1, parts)]
    # the first latent decodes to one frame, every later one to four
    assert [p.shape[0] for p in pieces] == [
        4 * (min(z.shape[2], (i + 1) * per_part) - i * per_part) - (3 if i == 0 else 0) for i in range(parts)
    ]
    assert torch.equal(torch.cat(pieces), want)
    assert not sub.request_states["r"].tensors and not sub.request_states["r"].kwargs  # let go at the last piece


def test_interleaved_streams_keep_their_own_caches(monkeypatch, vae):
    """Two requests' pieces alternate on one node (the worker runs others' steps between a
    request's pieces); each request gets its own clip's frames."""
    sub = _decoder(monkeypatch, vae, per_part=2)
    za, zb = _latents(seed=1), _latents(seed=2)
    want_a, want_b = _whole(sub, za), _whole(sub, zb)
    parts = sub.config.decode_parts(want_a.shape[0])
    a, b = [], []
    for part in range(parts):
        a.append(_piece(sub, za, part, "a"))
        b.append(_piece(sub, zb, part, "b"))
    assert torch.equal(torch.cat(a), want_a) and torch.equal(torch.cat(b), want_b)


def test_a_piece_out_of_order_is_refused(monkeypatch, vae):
    sub = _decoder(monkeypatch, vae, per_part=2)
    z = _latents()
    with pytest.raises(ValueError, match="out of order"):
        _piece(sub, z, 1)
    _piece(sub, z, 0)
    with pytest.raises(ValueError, match="out of order"):
        _piece(sub, z, 0)
    with pytest.raises(ValueError, match="out of order"):
        _piece(sub, z, 4)  # past the clip's 7 latents


def test_the_pieces_encode_to_the_whole_clips_mp4(monkeypatch, vae):
    """What the data worker's Mp4Stream writes from the pieces is what postprocess writes
    from the whole clip (same encoding, the writer's frames in order)."""
    from mstar.model.components.video_encode import Mp4Stream

    model = Wan22Model(model_path_hf="unused", skip_weight_loading=True, stream_decode=True)
    sub = _decoder(monkeypatch, vae, per_part=3)
    z = _latents()
    whole = model.postprocess(_whole(sub, z), "video", {})
    pieces = [_piece(sub, z, part).numpy() for part in range(3)]
    stream = Mp4Stream(model.mp4_encoding({}), len(pieces))
    for index in (2, 0, 1):  # in any order, as the transport delivers them
        stream.add_part(index, pieces[index])
    assert stream.future.result(timeout=60) == whole


def test_stream_decode_makes_the_decoder_a_loop_over_the_clips_pieces():
    model = Wan22Model(model_path_hf="unused", skip_weight_loading=True, stream_decode=True)
    for walk in (Wan22Model.VIDEO_GEN_WALK, Wan22Model.VIDEO_GEN_I2V_WALK):
        loops = model.get_graph_walk_graphs()[walk].get_loops()
        assert set(loops) == {DENOISE_LOOP_NAME, DECODE_LOOP}
        assert [edge.output_modality for edge in loops[DECODE_LOOP].section.outputs] == [VIDEO_PART]
    # 81 frames are 21 latents: 7 pieces of 3
    assert model.get_loop_caps() == {DENOISE_LOOP_NAME: 50, DECODE_LOOP: 7}
    assert model.get_loop_caps(num_frames=17, num_inference_steps=4) == {DENOISE_LOOP_NAME: 4, DECODE_LOOP: 2}
    plain = Wan22Model(model_path_hf="unused", skip_weight_loading=True)
    assert plain.get_loop_caps() == {DENOISE_LOOP_NAME: 50}
    assert set(plain.get_graph_walk_graphs()[Wan22Model.VIDEO_GEN_WALK].get_loops()) == {DENOISE_LOOP_NAME}


def test_wan_declares_its_mp4_encoding_and_postprocess_writes_it():
    model = Wan22Model(model_path_hf="unused", skip_weight_loading=True)
    encoding = model.mp4_encoding({"fps": 16})
    assert encoding.fps == 16.0 and dict(encoding.options) == {"crf": "18", "preset": "ultrafast", "threads": "0"}
    assert model.mp4_encoding({}).fps == model.config.video_fps
    frames = np.random.default_rng(0).integers(0, 256, (5, 64, 96, 3), dtype=np.uint8)
    assert model.postprocess(torch.from_numpy(frames), "video", {"fps": 16}) == encode_mp4(
        frames, 16.0, options=encoding.options,
    )
    with pytest.raises(ValueError, match="fps"):
        model.mp4_encoding({"fps": 0})


@pytest.mark.parametrize(("key", "value"), [
    ("stream_decode", "true"), ("stream_decode", 1), ("stream_decode_latents", 0), ("stream_decode_latents", True),
    ("stream_decode_latents", 2.0),
])
def test_stream_decode_settings_are_checked(key, value):
    with pytest.raises(ValueError, match=key):
        Wan22Config(**{key: value})
    with pytest.raises(ValueError, match=key):
        Wan22Model(model_path_hf="unused", skip_weight_loading=True, **{key: value})
