"""CPU checks for the streaming VAE decoder's warm path (Cosmos3VAEDecoderARSubmodule
with ``windowed_decode_warm_cache``) on a tiny randomly initialized Wan VAE:
the per-window frames equal one decode of the whole stream, the re-decode
covers a warm stream's first window (fresh or resumed), and the cache rides
the decode loop's loop-back edge. No GPU, no weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.components.wan_vae import is_channels_last  # noqa: E402
from mstar.model.cosmos3 import constants as C  # noqa: E402
from mstar.model.cosmos3 import submodules as cosmos3_submodules  # noqa: E402
from mstar.model.cosmos3.config import Cosmos3Config  # noqa: E402
from mstar.model.cosmos3.submodules import (  # noqa: E402
    DECODE_CACHE,
    DECODE_FRAMES,
    DECODE_SENT,
    DECODE_TAIL,
    Cosmos3VAEDecoderARSubmodule,
    Cosmos3VAEDecoderSubmodule,
)

# 8-latent windows re-pinning 2 latents, as the served default (29 / 8 frames).
WINDOW, OVERLAP = 8, 2


def _tiny_vae() -> AutoencoderKLWan:
    vae = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=16, z_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1, is_residual=True,
        in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
        latents_mean=[0.1 * i for i in range(8)], latents_std=[0.5 + 0.1 * i for i in range(8)],
    ).eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return vae


@pytest.fixture(scope="module")
def vae():
    return _tiny_vae()


def _decoder(monkeypatch, vae, warm=True, ctx=3, channels_last=False):
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    cfg = Cosmos3Config()
    cfg.windowed_decode_context_latents = ctx
    cfg.windowed_decode_warm_cache = warm
    cfg.vae_decode_channels_last = channels_last
    sub = Cosmos3VAEDecoderARSubmodule(vae, cfg)
    sub._decode_dtype_cached = torch.float32
    return sub


def _stream(units: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, 8, units, 4, 6, generator=gen)


def _windows(stream: torch.Tensor, start: int = 0):
    """The chained windows over ``stream[start:]``: each re-pins the last
    OVERLAP latents of the one before."""
    out, a = [], start
    while True:
        out.append(stream[:, :, a : a + WINDOW])
        if a + WINDOW >= stream.shape[2]:
            return out
        a += WINDOW - OVERLAP


def _run(sub, rid, windows, frames, **md):
    """Each window's emitted ``[T, C, H, W]`` frames, as ``[1, C, T, H, W]`` pixels
    to compare with ``_decode_pixels``. The decode loop's edges carry from one
    call to the next for ``rid``, as the loop carries them from window to window."""
    md = {"num_windows": len(windows), "overlap_latent_units": OVERLAP, "num_frames": frames,
          "stream_video": True, **md}
    infos = SimpleNamespace(request_ids=[rid], per_request_info={rid: SimpleNamespace(step_metadata=md)})
    loop = sub.__dict__.setdefault("_loops", {}).setdefault(rid, {"window": 0, "edges": {}})
    emitted = []
    for w in windows:
        out = sub.forward(C.VIDEO_DECODE_AR_WALK, infos, w, window=loop["window"], edges=loop["edges"])
        loop["window"] += 1
        loop["edges"] = {name: out[name] for name in (DECODE_TAIL, DECODE_SENT, DECODE_FRAMES, DECODE_CACHE)
                         if name in out}
        emitted.append(out["video_output"][0])
    assert all(f.is_contiguous() for f in emitted)
    return [f.transpose(0, 1)[None] for f in emitted]


def _u8_close(a: torch.Tensor, b: torch.Tensor, tol: int = 1) -> None:
    """One-call-per-window vs per-frame ``vae.decode`` is the same math to
    float rounding; after uint8 that is at most one level."""
    assert a.shape == b.shape
    assert (a.int() - b.int()).abs().max() <= tol


def test_warm_windows_are_the_whole_stream_decode(monkeypatch, vae) -> None:
    """Each window decodes only its new latents from the stream's cache; the
    chunks concatenate to one decode of the whole stream, exactly, which the
    re-decode path only approximates."""
    sub = _decoder(monkeypatch, vae)
    stream = _stream(20)  # 3 windows: [0, 8), [6, 14), [12, 20)
    chunks = _run(sub, "r", _windows(stream), 77)
    whole = sub._decode_pixels(stream)
    assert [c.shape[2] for c in chunks] == [29, 24, 24]
    # One latent per decoder call (the default) is bit-exact against ``vae.decode``.
    assert torch.equal(torch.cat(chunks, dim=2), whole)
    one_call = _decoder(monkeypatch, vae)
    one_call.config.windowed_decode_latents_per_call = 0
    _u8_close(torch.cat(_run(one_call, "r1", _windows(stream), 77), dim=2), whole)
    redecode = torch.cat(_run(_decoder(monkeypatch, vae, warm=False), "r", _windows(stream), 77), dim=2)
    assert not torch.equal(redecode, whole)
    # Context covering the whole stream makes the re-decode path exact too.
    exact = torch.cat(_run(_decoder(monkeypatch, vae, warm=False, ctx=20), "r", _windows(stream), 77), dim=2)
    assert torch.equal(exact, whole)


def test_the_cache_rides_the_loop_back_edge(monkeypatch, vae) -> None:
    """A warm window's decoder cache leaves on the loop's edge as tensors and
    comes back a ``FeatCache`` holding the same slots."""
    sub = _decoder(monkeypatch, vae)
    _run(sub, "r", _windows(_stream(14))[:1], 53, num_windows=2)
    edge = sub._loops["r"]["edges"][DECODE_CACHE]
    assert edge and all(torch.is_tensor(t) for t in edge)
    cache = wan_vae.decoder_cache(vae, edge)
    assert isinstance(cache, wan_vae.FeatCache) and any(torch.is_tensor(t) for t in cache)
    again = wan_vae.to_edge(cache, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(again, edge, strict=True)), "a slot changed on its way round"


def test_a_failed_window_decodes_again_from_its_edges(monkeypatch, vae) -> None:
    """A decode that raises mid-window writes its cache's slots into a copy of
    the edge's list, so a retry of the window from the same edges continues
    the stream exactly."""
    stream = _stream(20)
    windows = _windows(stream)
    sub = _decoder(monkeypatch, vae)
    _run(sub, "r", windows[:1], 77, num_windows=3)
    decoder, calls = sub._stream.step._decoder, []

    def fail_third(x, **kw):
        calls.append(x.shape[2])
        if len(calls) == 3:
            raise RuntimeError("injected")
        return decoder(x, **kw)

    sub._stream.step._decoder = fail_third
    with pytest.raises(RuntimeError, match="injected"):
        _run(sub, "r", windows[1:2], 77, num_windows=3)
    assert sub._loops["r"]["window"] == 1
    sub._stream.step._decoder = decoder
    retried = _run(sub, "r", windows[1:2], 77, num_windows=3)
    third = _run(sub, "r", windows[2:], 77, num_windows=3)
    whole = sub._decode_pixels(stream)
    assert torch.equal(retried[0], whole[:, :, 29:53]) and torch.equal(third[0], whole[:, :, 53:])


def test_resumed_session_warms_up_from_the_stored_latents(monkeypatch, vae) -> None:
    """A resumed session's first window decodes behind the session's stored
    latent context (the store never holds a decoder cache), exactly as the
    re-decode path does; its later windows run warm."""
    stream = _stream(26)
    first = _windows(stream)[:2]  # request 1: units [0, 14)
    # Request 2 resumes at unit 14 with its first window re-pinning 12..14.
    resumed = _windows(stream, start=12)
    outs = {}
    for warm in (True, False):
        sub = _decoder(monkeypatch, vae, warm=warm)
        _run(sub, "r1", first, 53, session_id="s")
        stored = sub._sessions.get("s")
        assert torch.is_tensor(stored) and stored.shape[2] == 3
        outs[warm] = _run(sub, "r2", resumed, 48, session_id="s", resume_latent_units=OVERLAP)
    assert [c.shape[2] for c in outs[True]] == [24, 24]
    _u8_close(outs[True][0], outs[False][0])
    # The second window continues the cache: the stream from the stored
    # context onwards, decoded in one go.
    whole = sub._decode_pixels(stream[:, :, 11:])
    _u8_close(outs[True][1], whole[:, :, -24:])


def test_concurrent_streams_keep_their_own_cache(monkeypatch, vae) -> None:
    sub = _decoder(monkeypatch, vae)
    streams = {"a": _stream(20, seed=2), "b": _stream(20, seed=3)}
    windows = {r: _windows(s) for r, s in streams.items()}
    chunks = {"a": [], "b": []}
    for i in range(3):
        for r in ("a", "b"):
            chunks[r] += _run(sub, r, windows[r][i : i + 1], 77, num_windows=3)
    for r, s in streams.items():
        _u8_close(torch.cat(chunks[r], dim=2), sub._decode_pixels(s))


def test_channels_last_decoder(monkeypatch) -> None:
    """The channels-last decode (the default) keeps the warm stream exact
    against its own whole-stream decode, and matches the default layout to
    8-bit rounding."""
    stream = _stream(20)
    sub = _decoder(monkeypatch, _tiny_vae(), channels_last=True)
    assert is_channels_last(sub.vae)
    assert sub._vae_input(stream).is_contiguous(memory_format=torch.channels_last_3d)
    video = torch.cat(_run(sub, "r", _windows(stream), 77), dim=2)
    _u8_close(video, sub._decode_pixels(stream))
    plain = _decoder(monkeypatch, _tiny_vae())._decode_pixels(stream)
    assert (video.int() - plain.int()).abs().max() <= 1


def _record_calls(sub) -> list:
    calls = []
    decoder = sub._stream.step._decoder
    sub._stream.step._decoder = lambda x, **kw: calls.append(x.shape[2]) or decoder(x, **kw)
    return calls


def test_window_in_one_decoder_call(monkeypatch, vae) -> None:
    """``windowed_decode_latents_per_call=0`` (opt-in) decodes a steady
    window's new latents in one decoder call, same frames to 8-bit rounding;
    the stream's first window runs one latent per call. The default ``=1`` is
    the stock per-frame decode."""
    stream = _stream(20)
    sub = _decoder(monkeypatch, vae)
    sub.config.windowed_decode_latents_per_call = 0
    calls = _record_calls(sub)
    video = torch.cat(_run(sub, "r", _windows(stream), 77), dim=2)
    assert calls == [1] * WINDOW + [6, 6]
    assert (video.int() - sub._decode_pixels(stream).int()).abs().max() <= 1

    per_frame = _decoder(monkeypatch, vae)
    assert per_frame.config.windowed_decode_latents_per_call == 1
    frame_calls = _record_calls(per_frame)
    per_frame_video = torch.cat(_run(per_frame, "r2", _windows(stream), 77), dim=2)
    assert frame_calls == [1] + [1] * (7 + 6 + 6)
    assert torch.equal(per_frame_video, per_frame._decode_pixels(stream))
    _u8_close(per_frame_video, video)


def test_resumed_first_window_is_not_grouped(monkeypatch, vae) -> None:
    """At ``=0`` a resumed session's first window re-decodes its stored
    context ahead of the new latents one latent per call, so it peaks no
    higher than the default; the windows after it are grouped."""
    stream = _stream(26)
    sub = _decoder(monkeypatch, vae)
    sub.config.windowed_decode_latents_per_call = 0
    _run(sub, "r1", _windows(stream)[:2], 53, session_id="s")
    calls = _record_calls(sub)
    _run(sub, "r2", _windows(stream, start=12), 48, session_id="s", resume_latent_units=OVERLAP)
    # 3 stored context latents + 6 new, then one 6-latent call.
    assert calls == [1] * 9 + [6]


def test_warm_cache_switch_off_keeps_no_cache(monkeypatch, vae) -> None:
    sub = _decoder(monkeypatch, vae, warm=False)
    _run(sub, "r", _windows(_stream(14))[:1], 53, num_windows=2)
    assert DECODE_CACHE not in sub._loops["r"]["edges"] and sub._stream is None


def _record_cudnn_benchmark(monkeypatch) -> list:
    import torch.nn.functional as F

    seen = []
    conv3d = F.conv3d

    def conv(*args, **kw):
        seen.append(torch.backends.cudnn.benchmark)
        return conv3d(*args, **kw)

    monkeypatch.setattr(F, "conv3d", conv)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    return seen


def test_whole_clip_decoder_keeps_its_default_layout_and_cudnn_flag(monkeypatch) -> None:
    """The streaming switches leave the whole-clip decoder alone: its VAE stays
    in the default layout and cuDNN's flag untouched, even with the streaming
    node's channels-last and cudnn.benchmark both on."""
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    cfg = Cosmos3Config()
    cfg.vae_decode_cudnn_benchmark = True
    cfg.vae_decode_fast = False  # the stock whole-clip path; the fast one is tested below
    assert cfg.vae_decode_channels_last
    vae = _tiny_vae()
    sub = Cosmos3VAEDecoderSubmodule(vae, cfg)
    sub._decode_dtype_cached = torch.float32
    seen = _record_cudnn_benchmark(monkeypatch)
    stream = _stream(5)
    video = sub._decode_pixels(stream)
    assert not is_channels_last(vae)
    assert seen and not any(seen)
    with torch.no_grad():
        ref = vae.decode(sub._vae_input(stream)).sample
    assert torch.equal(video, (ref.float() / 2 + 0.5).clamp(0, 1).mul(255).round().to(torch.uint8))


def test_fast_whole_clip_decode_matches_stock_decode(monkeypatch) -> None:
    """vae_decode_fast decodes a whole clip through the shared decoder,
    channels-last, on the node's own VAE: the frames of the stock decode."""
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    ref_vae = _tiny_vae()
    stream = _stream(5)
    stock = Cosmos3VAEDecoderSubmodule(ref_vae, Cosmos3Config(vae_decode_fast=False))
    stock._decode_dtype_cached = torch.float32
    want = stock._decode_pixels(stream)
    fast_vae = _tiny_vae()
    fast_vae.load_state_dict(ref_vae.state_dict())
    cfg = Cosmos3Config()
    assert cfg.vae_decode_fast
    monkeypatch.setattr(cosmos3_submodules, "wan_vae_decode_dtype", lambda: torch.float32)
    fast = Cosmos3VAEDecoderSubmodule(fast_vae, cfg, fast=cfg.vae_decode_fast)
    assert is_channels_last(fast_vae)
    got = fast._decode_pixels(stream)
    assert got.shape == want.shape
    assert (got.int() - want.int()).abs().max() <= 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("warm", [True, False])
def test_streaming_cudnn_benchmark_is_opt_in_and_scoped(monkeypatch, vae, enabled, warm) -> None:
    """Off by default; when on, it covers every conv of both streaming paths
    (warm and stateless) and nothing after them."""
    sub = _decoder(monkeypatch, vae, warm=warm)
    assert not sub.config.vae_decode_cudnn_benchmark
    sub.config.vae_decode_cudnn_benchmark = enabled
    seen = _record_cudnn_benchmark(monkeypatch)
    _run(sub, "r", _windows(_stream(14)), 53)
    assert seen and all(f is enabled for f in seen)
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.parametrize("key, value", [
    ("windowed_decode_latents_per_call", -1),
    ("windowed_decode_latents_per_call", 9),  # the default window holds 8 latents
    ("windowed_decode_latents_per_call", True),
    ("windowed_decode_warm_cache", "false"),
    ("vae_decode_channels_last", 1),
    ("vae_decode_cudnn_benchmark", None),
])
def test_streaming_decode_settings_are_checked(key, value) -> None:
    cfg = Cosmos3Config()
    setattr(cfg, key, value)
    with pytest.raises(ValueError, match=key):
        cfg.check_streaming_decode()


def test_streaming_decode_settings_accept_the_documented_range() -> None:
    cfg = Cosmos3Config()
    cfg.check_streaming_decode()
    for per_call in (0, 1, 8):
        cfg.windowed_decode_latents_per_call = per_call
        cfg.check_streaming_decode()


def test_model_load_checks_the_streaming_decode_overrides() -> None:
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

    with pytest.raises(ValueError, match="windowed_decode_latents_per_call"):
        Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, windowed_decode_latents_per_call=-2)
