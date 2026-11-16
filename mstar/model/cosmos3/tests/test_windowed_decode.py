"""CPU checks for the streaming VAE decoder's warm path (Cosmos3VAEDecoderARSubmodule
with ``windowed_decode_warm_cache``) on a tiny randomly initialized Wan VAE:
the per-window frames equal one decode of the whole stream, the re-decode
covers a warm stream's first window (fresh or resumed), and the cache goes
from window to window through its sized Value. No GPU, no weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402

from mstar.engine.resources.sized.store import SizedValueStore  # noqa: E402
from mstar.engine.resources.step import StepContext  # noqa: E402
from mstar.engine.state import Layout, Value  # noqa: E402
from mstar.engine.state.adapters import build_specs, node_handles, state_sizes, state_steps  # noqa: E402
from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.components.wan_vae import is_channels_last  # noqa: E402
from mstar.model.cosmos3 import constants as C  # noqa: E402
from mstar.model.cosmos3 import submodules as cosmos3_submodules  # noqa: E402
from mstar.model.cosmos3.config import Cosmos3Config, Cosmos3VAEConfig  # noqa: E402
from mstar.model.cosmos3.submodules import (  # noqa: E402
    DECODE_CACHE,
    DECODE_FRAMES,
    DECODE_LABEL,
    DECODE_SENT,
    DECODE_TAIL,
    DECODER,
    Cosmos3VAEDecoderARSubmodule,
    Cosmos3VAEDecoderSubmodule,
    decode_cache_entry,
)

# 8-latent windows re-pinning 2 latents, as the served default (29 / 8 frames).
WINDOW, OVERLAP = 8, 2
# the latent size of `_stream`, and the decoder's node
LAT = (4, 6)
NODE = "vae_decoder_ar"


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
    cfg.vae = Cosmos3VAEConfig(z_dim=8, decoder_base_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1)
    sub = Cosmos3VAEDecoderARSubmodule(vae, cfg)
    sub._decode_dtype_cached = torch.float32
    if warm:
        sub.caches = _Caches(cfg)
        sub.bind_node_state(sub.caches.handles)
    return sub


class _Caches:
    """``DECODE_CACHE`` as the decoder's node holds it, on the CPU: its store, and the node's handle over it."""

    def __init__(self, cfg: Cosmos3Config):
        self.declared = [Value(
            DECODE_CACHE, nodes={NODE}, layers=frozenset({0}), entry=decode_cache_entry(cfg, torch.float32),
            largest={"lat_h": 1, "lat_w": cfg.windowed_decode_max_latent_area},
        )]
        (self.spec,) = build_specs(self.declared)
        self.store = SizedValueStore(self.spec.config, torch.device("cpu"), name=DECODE_CACHE)
        model = SimpleNamespace(declare_state=lambda: self.declared, get_combined_graph_walks=lambda: {})
        self.handles = node_handles(model, NODE, {DECODE_CACHE: self.store})

    def open(self, rid) -> None:
        """Take ``rid`` in at `_stream`'s latent size, as intake and the engine's admission queue do."""
        layout = {DECODE_CACHE: {DECODE_LABEL: Layout(sizes=dict(zip(("lat_h", "lat_w"), LAT, strict=True)))}}
        sizes = state_sizes(self.declared, layout)
        self.store.ingest_request(rid, overrides=self.spec.request_config({"state_sizes": sizes}))
        assert self.store.reserve(rid, NODE, C.VIDEO_DECODE_AR_WALK, None) is None

    def held(self, rid) -> dict[str, torch.Tensor]:
        return {name: tensor for (_, _, name), tensor in self.store._requests[rid].tensors.items()}


def _forward(sub, rid, infos, latents, window, edges) -> dict:
    """One window through the submodule's forward, stepped as the engine steps it where it declares a step."""
    declared = sub.declare_step(C.VIDEO_DECODE_AR_WALK, [rid], [None])
    if declared is None:
        return sub.forward(C.VIDEO_DECODE_AR_WALK, infos, latents, window=window, edges=edges)
    if rid not in sub.caches.store._requests:
        sub.caches.open(rid)
    step = state_steps(declared, sub.caches.handles, graph_walk=C.VIDEO_DECODE_AR_WALK)
    step.set_ctx(StepContext(request_ids=(rid,), graph_walk=C.VIDEO_DECODE_AR_WALK, slot=0, capture=False))
    step.begin_calls()
    out = sub.forward(C.VIDEO_DECODE_AR_WALK, infos, latents, window=window, edges=edges)
    step.check_calls()
    step.commit_calls()
    step.end_calls()
    return out


def _stream(units: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, 8, units, *LAT, generator=gen)


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
        out = _forward(sub, rid, infos, w, loop["window"], loop["edges"])
        loop["window"] += 1
        loop["edges"] = {name: out[name] for name in (DECODE_TAIL, DECODE_SENT, DECODE_FRAMES)}
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


def test_the_cache_goes_from_window_to_window_through_its_value(monkeypatch, vae) -> None:
    """A warm window leaves its decoder cache in ``DECODE_CACHE`` and on no output, which an edge would copy to the
    host every window. The next window decodes from the tensors it left, and the last one gives their bytes back."""
    sub = _decoder(monkeypatch, vae)
    windows = _windows(_stream(20))
    _run(sub, "r", windows[:1], 77, num_windows=3)
    left = sub.caches.held("r")
    assert left and DECODE_CACHE not in sub._loops["r"]["edges"]
    seen = []
    decode = sub._decode_stream
    sub._decode_stream = lambda new, cache, clip_start: seen.append(list(cache)) or decode(new, cache, clip_start)
    _run(sub, "r", windows[1:2], 77, num_windows=3)
    slots = {f"{DECODER}_{i:02d}": slot for i, slot in enumerate(seen[0]) if torch.is_tensor(slot)}
    assert slots.keys() == left.keys() and all(slots[name] is left[name] for name in left), "a slot was copied"
    held = sub.caches.store._used
    _run(sub, "r", windows[2:], 77, num_windows=3)
    assert held and not sub.caches.store._used and not sub.caches.held("r"), "the ended stream holds its bytes"


def test_a_window_past_the_first_refuses_an_empty_cache(monkeypatch, vae) -> None:
    """A window whose forward failed after its read leaves nothing behind, which is what a stream's first finds."""
    sub = _decoder(monkeypatch, vae)
    windows = _windows(_stream(20))
    _run(sub, "r", windows[:1], 77, num_windows=3)
    sub.caches.store._requests["r"].tensors.clear()
    with pytest.raises(RuntimeError, match="window 1 found no decoder cache in 'decode_cache'"):
        _run(sub, "r", windows[1:2], 77, num_windows=3)


@pytest.mark.parametrize("lat_hw", [(2, 3), (3, 5)])
def test_the_declared_cache_is_the_vae_s_at_any_latent_size(vae, lat_hw) -> None:
    """Every process declares the cache from the VAE's config, without loading it. Against a residual Wan VAE at
    random weights: each slot a stream holds between windows has its field's shape, and owns its bytes."""
    cfg = Cosmos3Config()
    cfg.vae = Cosmos3VAEConfig(z_dim=8, decoder_base_dim=16, dim_mult=[1, 2, 2, 2], num_res_blocks=1)
    fields = decode_cache_entry(cfg, torch.float32).tensors
    cache = wan_vae.new_feat_cache(vae)
    with torch.inference_mode():
        wan_vae.FastDecoder(vae, channels_last=False, compile=False).decode_stream(
            torch.zeros(1, 8, 2, *lat_hw), cache, clip_start=True,
        )
    wan_vae.check_stream_cache(cache, fields, DECODER, lat_hw)
    held = [slot for slot in cache if torch.is_tensor(slot)]
    assert len(held) == len(fields) and all(
        slot.untyped_storage().nbytes() == slot.numel() * slot.element_size() for slot in held
    ), "a slot that is a view of a larger buffer would be refused at its write"


def test_the_cosmos3_vae_s_cache_takes_913600_bytes_a_latent_pixel_in_bf16() -> None:
    """1.33 GiB at 832x480, which the default budget holds twice: one stream at 1280x720."""
    cfg = Cosmos3Config()
    value = Value(
        DECODE_CACHE, nodes={NODE}, layers=frozenset({0}), entry=decode_cache_entry(cfg, torch.bfloat16),
        largest={"lat_h": 1, "lat_w": cfg.windowed_decode_max_latent_area},
    )
    (spec,) = build_specs([value])
    assert spec.config.bytes_at({"lat_h": 30, "lat_w": 52}) == 913600 * 30 * 52
    assert spec.config.budget == 913600 * 45 * 80


# Request 2 of a session whose request 1 covered units [0, 14): it resumes at
# unit 14, its first window re-pinning 12..14 behind the session's last 3
# latents, which the DiT sends ahead of the window's.
_RESUME = {"resume_latent_units": OVERLAP, "window_latent_units": WINDOW, "total_latent_units": 14}


def _resumed(stream: torch.Tensor) -> list:
    windows = _windows(stream, start=12)
    return [torch.cat([stream[:, :, 11:14], windows[0]], dim=2), *windows[1:]]


def test_a_resumed_session_warms_up_from_the_context_the_dit_sends(monkeypatch, vae) -> None:
    """A resumed session's first window decodes behind the session's decode
    context, which arrives ahead of the window's latents (no decoder cache
    outlives a request), exactly as the re-decode path does; its later
    windows run warm."""
    stream = _stream(26)
    first = _windows(stream)[:2]  # request 1: units [0, 14)
    outs = {}
    for warm in (True, False):
        sub = _decoder(monkeypatch, vae, warm=warm)
        _run(sub, "r1", first, 53)
        outs[warm] = _run(sub, "r2", _resumed(stream), 48, **_RESUME)
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
    """At ``=0`` a resumed session's first window re-decodes its decode
    context ahead of the new latents one latent per call, so it peaks no
    higher than the default; the windows after it are grouped."""
    stream = _stream(26)
    sub = _decoder(monkeypatch, vae)
    sub.config.windowed_decode_latents_per_call = 0
    _run(sub, "r1", _windows(stream)[:2], 53)
    calls = _record_calls(sub)
    _run(sub, "r2", _resumed(stream), 48, **_RESUME)
    # 3 context latents + 6 new, then one 6-latent call.
    assert calls == [1] * 9 + [6]


def test_warm_cache_switch_off_keeps_no_cache(monkeypatch, vae) -> None:
    sub = _decoder(monkeypatch, vae, warm=False)
    _run(sub, "r", _windows(_stream(14))[:1], 53, num_windows=2)
    assert sub.declare_step(C.VIDEO_DECODE_AR_WALK, ["r"], [None]) is None and sub._stream is None


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


def test_a_config_that_still_caps_the_warm_streams_is_refused_at_load() -> None:
    """The cap is gone, so a deployment that set it would run uncapped behind a warning."""
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model

    with pytest.raises(ValueError, match="windowed_decode_max_warm_streams is no longer read"):
        Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, windowed_decode_max_warm_streams=2)
