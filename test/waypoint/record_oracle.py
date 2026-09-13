#!/usr/bin/env python3
"""Record the world_engine golden-reference oracle for Waypoint-1.5-1B.

Runs **only** ``world_engine`` — importing any mstar module is a hard error — so
the artifact owes nothing to the code it will be used to judge.

    <out-dir>/frames/frame_000.pt ... frame_NNN.pt   per-frame tensors (below)
    <out-dir>/ring/ring_000.pt ...                   full KV ring at selected frames
    <out-dir>/metadata.json                          resolved numerics + params

Each ``frame_*.pt`` holds one engine step = one latent frame:

    dit_out       the DiT output of every pass: 4 non-committing denoise passes
                  at sigma 1.0/0.9/0.75/0.3, then the committing pass at sigma 0
                  (frame 0 is the seed and has only the committing pass)
    latent        x0, the emitted latent after the Euler steps
    pixels        the 4 raw frames the streaming VAE decoded from it
    noise_f32     the CPU fp32 draw, and noise_bf16, the device tensor actually fed
    committed_kv  per layer, the tail KV slice this frame committed
    ring          per layer, the `written` mask and per-bucket sum-of-squares

``committed_kv`` plus the ring digest localize a mismatch to a layer, and the
digest's per-bucket resolution localizes it to a ring slot, without writing the
2.2 GB full ring every frame. Full rings are written for ``--ring-snapshot-frames``.

``latent``, ``pixels``, ``committed_kv`` and ``ring`` come from the reference's
own compiled regions, called unmodified, and are bit-exact targets. ``dit_out``
is not: see Execution.

Execution
---------
The reference's driver is two ``@torch.compile(fullgraph=True, dynamic=False)``
regions, and the compile is correctness, not throughput — eager ``flex_attention``
ignores a ``BlockMask``'s block index lists, and the mask carries a no-op
``mask_mod``, so an eager pass attends over unwritten ring slots. Nothing here
may call ``engine.model(...)`` directly.

That makes the per-pass DiT output unobservable where it is produced: adding it as
an output of ``_denoise_pass``, or splitting that region into five, changes
inductor's fusion and moves the result by about one bf16 ULP, which then compounds
through the ring. So state comes from the reference driver untouched, and
``dit_out`` comes from separate frozen passes run first — ``upsert`` only writes
the ring when unfrozen, which ``--verify-shadow`` checks on every run. Treat
``dit_out`` as a per-pass diagnostic recorded under a stated decomposition.

Nothing recorded here is a bit-exact target. The reference driver is deterministic
within a process and not across them: two processes running it alone disagree by
one bf16 ULP at layer 0, which 24 layers compound. ``repro/`` is a second
independent recording of the opening frames so that floor can be measured rather
than assumed. See ``reproducibility`` in the metadata.

Numerics
--------
An oracle is only a reference if it is recorded under the same numerics as the
serving process, and only comparable against the torch build it was recorded on.

Two settings disagree here: mstar sets ``float32_matmul_precision('high')``
process-wide (``mstar/engine/__init__.py``), ``world_engine`` sets ``'medium'`` at
import. This records under **'high'**, the serving value, for the reason above —
and, because that is a deviation from the reference as shipped, it also measures
what the deviation costs. The only fp32 matmul in the patched inference path is
``NoiseConditioner.mlp`` (a ``NoCastModule``, so it survives the bf16 cast), and
after ``patch_cached_noise_conditioning`` it runs once per sigma level to build a
LUT that is then rounded to bf16. ``matmul_precision_calibration`` in the metadata
is that LUT evaluated both ways, so Phase 9 has the number instead of an argument.

Noise is an input, not model behaviour, and the reference draws it unseeded
(``torch.randn(..., device=cuda, dtype=bf16)``), which no oracle can reproduce.
This draws fp32 from a seeded CPU generator and casts, matching how the port
draws it (``mstar/model/waypoint/submodules.py::_frame_noise``), saves both
tensors, and records the substitution. Seeding also makes the run re-recordable,
which is what lets ``--ring-snapshot-frames`` be narrowed by default.

Usage:

    CUDA_VISIBLE_DEVICES=2 python3 test/waypoint/record_oracle.py \
        --out-dir /path/to/oracle \
        --model-dir /path/to/Waypoint-1.5-1B --ae-dir /path/to/taehv1_5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

WORLD_ENGINE_DEFAULT = "/mnt/storage/garv901/waypoint-1.5-1B/world_engine"
CKPT_DEFAULT = "/mnt/storage/garv901/waypoint-1.5-1B/checkpoints/Waypoint-1.5-1B"
AE_DEFAULT = "/mnt/storage/garv901/waypoint-1.5-1B/checkpoints/taehv1_5"
SEED_DEFAULT = "/mnt/storage/garv901/waypoint-1.5-1B/checkpoints/seed/default.jpg"

# Pinned so the seed frame is a fact of the run, not of whatever the repo holds
# today. Cached at SEED_DEFAULT; --seed-image overrides.
SEED_URL = "https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds/default.jpg"
SEED_SHA256 = "c61c9393311d7281f793d86329dca343e12c93bf0409980a186eb39269cf6862"

NOISE_SEED = 42
NUM_FRAMES = 40  # local window is 16 frames, so 41 total frames wrap the ring twice

# Scripted controller inputs: (button ids, (mouse dx, dy), scroll). Buttons are
# WASD 87/65/83/68, space 32, mouse trigger 1 — the ids gen_sample.py drives.
CONTROL_SEQUENCE: list[tuple[set[int], tuple[float, float], int]] = (
    [({87}, (0.0, 0.0), 0)] * 8                 # forward
    + [({87}, (0.2, 0.0), 0)] * 4               # forward, panning right
    + [({65}, (0.0, 0.0), 0)] * 4               # strafe left
    + [({68}, (0.0, 0.0), 0)] * 4               # strafe right
    + [({83}, (0.0, 0.0), 0)] * 4               # back
    + [({87, 32}, (0.0, 0.0), 0)] * 4           # forward + jump
    + [(set(), (0.0, 0.0), 0)] * 4              # idle
    + [(set(), (0.0, -0.2), 0)] * 4             # look up
    + [({87, 1}, (0.0, 0.0), 0)] * 4            # forward + trigger
)
assert len(CONTROL_SEQUENCE) == NUM_FRAMES


def load_world_engine(root: str):
    """Import world_engine from a source checkout and pin the serving numerics.

    The flag is set after the import on purpose: world_engine sets 'medium' at
    import time, so setting it earlier would be silently undone.
    """
    sys.path.insert(0, root)
    import src as world_engine

    sys.modules.setdefault("world_engine", world_engine)
    torch.set_float32_matmul_precision("high")
    if "mstar" in sys.modules:
        raise SystemExit("mstar was imported; the oracle must run world_engine only.")
    return world_engine


def load_seed_frame(path: str, url: str) -> "torch.Tensor":
    """The seed image as [4, 720, 1280, 3] uint8, the x4 repeat append_frame wants."""
    import cv2
    import numpy as np

    p = Path(path)
    if p.exists():
        raw = p.read_bytes()
    else:
        import urllib.request

        raw = urllib.request.urlopen(url).read()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)

    digest = hashlib.sha256(raw).hexdigest()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(cv2.resize(img, (1280, 720)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.repeat(img[None], 4, axis=0)), digest


def describe_patches(model) -> dict:
    """Count the modules apply_inference_patches installed, from the live model."""
    from src.patch_model import CachedCondHead, CachedDenoiseStepEmb, MergedQKVAttn, SplitMLPFusion

    counts = {"CachedDenoiseStepEmb": 0, "CachedCondHead": 0, "MergedQKVAttn": 0, "SplitMLPFusion": 0}
    for mod in model.modules():
        for cls in (CachedDenoiseStepEmb, CachedCondHead, MergedQKVAttn, SplitMLPFusion):
            if isinstance(mod, cls):
                counts[cls.__name__] += 1
    return counts


def calibrate_matmul_precision(ckpt_dir: str, d_model: int, sigmas, device) -> dict:
    """Measure what 'high' (mstar) vs 'medium' (world_engine) costs on this build.

    Two probes. The model probe is the only fp32 matmul in the patched inference
    path: NoiseConditioner.mlp, which is a NoCastModule and so stays fp32 through
    the bf16 cast, and which patch_cached_noise_conditioning evaluates once per
    sigma level to build a LUT it then rounds to bf16. The control probe is a
    plain fp32 GEMM, and it is what makes a zero in the model probe readable — a
    'high' vs 'highest' difference proves the flag is live and the instrument
    works, so a 'high' vs 'medium' zero is a fact about the build, not a broken
    measurement.
    """
    from safetensors.torch import load_file
    from src.model.nn import NoiseConditioner

    sd = load_file(str(Path(ckpt_dir) / "model.safetensors"), device="cpu")
    prefix = "denoise_step_emb."
    weights = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    nc = NoiseConditioner(d_model).to(device=device)
    missing, unexpected = nc.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise SystemExit(f"NoiseConditioner weights did not load: {missing=} {unexpected=}")

    levels = torch.tensor(sigmas, device=device, dtype=torch.bfloat16)[:, None]
    a = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float32)

    model_out, control_out, shape_gap = {}, {}, {}
    for precision in ("high", "medium", "highest"):
        torch.set_float32_matmul_precision(precision)
        with torch.inference_mode():
            # CachedDenoiseStepEmb builds its table with base(levels[:, None]), so the
            # reference's LUT is one M=5 GEMM, not five GEMVs.
            model_out[precision] = nc(levels).squeeze(1).float().clone()
            control_out[precision] = (a @ b).clone()
            per_sigma = torch.cat([nc(levels[i:i + 1]) for i in range(levels.size(0))])
            shape_gap[precision] = (model_out[precision] - per_sigma.squeeze(1).float()).abs().max().item()
    torch.set_float32_matmul_precision("high")

    def gap(d, x, y):
        return (d[x] - d[y]).abs().max().item()

    model_gap = gap(model_out, "high", "medium")
    rounded = (model_out["high"].bfloat16().float() - model_out["medium"].bfloat16().float()).abs().max().item()
    return {
        "model_probe": {
            "what": "NoiseConditioner.mlp on the scheduler sigmas — the only fp32 matmul in the path",
            "high_vs_medium": model_gap,
            "high_vs_medium_after_bf16_round": rounded,
            "high_vs_highest": gap(model_out, "high", "highest"),
        },
        "control_probe": {
            "what": "fp32 4096x4096 GEMM, to show the flag is live on this build",
            "high_vs_medium": gap(control_out, "high", "medium"),
            "high_vs_highest": gap(control_out, "high", "highest"),
        },
        "lut_build_shape": {
            "what": "the reference's batched LUT build (M=5) against embedding one sigma at a time",
            "batched_vs_per_sigma": shape_gap,
            "note": (
                "under 'high' the M=5 GEMM reaches TF32 tensor cores and the M=1 GEMV does not, "
                "so an implementation that embeds one sigma at a time computes a more accurate "
                "LUT than the reference does. It is exact under 'highest'. That difference "
                "reaches every block's cond_head, so it is a divergence source in its own right "
                "and not a defect in this measurement."
            ),
        },
        "conclusion": (
            "mstar's 'high' and world_engine's 'medium' are bit-identical here"
            if model_gap == 0.0 and gap(control_out, "high", "medium") == 0.0
            else "'high' and 'medium' differ; the recorded value is load-bearing"
        ),
    }


def ring_digest(kv_cache) -> list[dict]:
    """Per layer: which slots are live, and the energy in each ring bucket.

    Bucket resolution is what localizes a wrong ring slot; it is derived from the
    cache's own tpf/capacity rather than re-deriving the bucket count.
    """
    digest = []
    for i, layer in enumerate(kv_cache.layers):
        kv = layer.kv
        buckets = kv.size(3) // layer.tpf
        sumsq = kv.float().pow(2).sum(dim=(0, 1, 2, 4)).view(buckets, layer.tpf).sum(-1)
        digest.append({
            "layer": i,
            "L": layer.L,
            "capacity": layer.capacity,
            "tokens_per_frame": layer.tpf,
            "pinned_dilation": layer.pinned_dilation,
            "num_buckets": layer.num_buckets,
            "written": layer.written.detach().cpu().clone(),
            "bucket_sumsq": sumsq.double().cpu(),
            "sum": kv.double().sum().cpu(),
            "absmax": kv.abs().float().max().cpu(),
        })
    return digest


def committed_kv(kv_cache) -> list[torch.Tensor]:
    """Per layer, the tail slice [L, L+tpf) holding the frame just committed."""
    return [layer.kv[:, :, :, layer.L:].detach().cpu().clone() for layer in kv_cache.layers]


def full_ring(kv_cache) -> list[dict]:
    return [
        {"kv": layer.kv.detach().cpu().clone(), "written": layer.written.detach().cpu().clone()}
        for layer in kv_cache.layers
    ]


def snapshot_ctx(ctx: dict) -> dict:
    """engine._ctx is reused every frame, so anything kept must be cloned."""
    return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in ctx.items()}


_SHADOW_PASS = None


def shadow_pass():
    """One frozen DiT pass, compiled with the reference's own decorator settings."""
    global _SHADOW_PASS
    if _SHADOW_PASS is None:
        from src.world_engine import COMPILE_OPTIONS

        def _pass(model, kv_cache, x, step_sig, ctx):
            kv_cache.set_frozen(True)
            sigma = x.new_empty((x.size(0), x.size(1)))
            return model(x, sigma.fill_(step_sig), **ctx, kv_cache=kv_cache)

        _SHADOW_PASS = torch.compile(_pass, fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
    return _SHADOW_PASS


def record_dit_outputs(engine, x, ctx, sigmas, dsigmas):
    """The 5 per-pass DiT outputs, from frozen passes that leave the ring alone."""
    shadow = shadow_pass()
    outs = []
    # strict=False is the reference's behaviour and load-bearing: 5 sigmas zipped
    # against their 4 diffs is what makes this 4 denoise passes, not 5.
    for step_sig, step_dsig in zip(sigmas, dsigmas, strict=False):
        v = shadow(engine.model, engine.kv_cache, x, step_sig, ctx)
        outs.append(v.detach().cpu().clone())
        x = (x.float() + step_dsig.float() * v.float()).type_as(x)
        del v  # a cudagraph output is invalid once the next replay overwrites it
    v = shadow(engine.model, engine.kv_cache, x, sigmas[-1], ctx)
    outs.append(v.detach().cpu().clone())
    return outs


def run_frame(engine, x, ctx, sigmas, dsigmas):
    """One engine step: capture the per-pass outputs, then advance the real state.

    The capture runs first because it must not be able to influence what is
    recorded; the state comes from `_denoise_pass`/`_cache_pass` verbatim.
    """
    outs = record_dit_outputs(engine, x, ctx, sigmas, dsigmas)
    x0 = engine._denoise_pass(x, ctx, engine.kv_cache).clone()
    engine._cache_pass(x0, ctx, engine.kv_cache)
    return x0, outs


def verify_shadow(engine, seed_frame, CtrlInput, sigmas, dsigmas, frames: int) -> dict:
    """Roll out with and without the capture passes; the state must be identical.

    Leaves the engine reset. Raises rather than record an oracle whose state the
    instrumentation reached.
    """
    def rollout(with_capture):
        engine.reset()
        gen = torch.Generator(device="cpu").manual_seed(0)
        latents = []
        with torch.inference_mode():
            x0 = engine.vae.encode(seed_frame).unsqueeze(1)
            engine._cache_pass(x0, engine.prep_inputs(x=x0, ctrl=CtrlInput()), engine.kv_cache)
            for _ in range(frames):
                x = torch.randn(engine.frm_shape, generator=gen, dtype=torch.float32).to(
                    device=engine.device, dtype=engine.dtype)
                ctx = engine.prep_inputs(x=x, ctrl=CtrlInput(button={87}))
                if with_capture:
                    x0, _ = run_frame(engine, x, ctx, sigmas, dsigmas)
                else:
                    x0 = engine._denoise_pass(x, ctx, engine.kv_cache).clone()
                    engine._cache_pass(x0, ctx, engine.kv_cache)
                latents.append(x0.detach().cpu().clone())
        ring = [(la.kv.double().sum().item(), int(la.written.sum().item())) for la in engine.kv_cache.layers]
        return latents, ring

    plain_lat, plain_ring = rollout(False)
    cap_lat, cap_ring = rollout(True)
    engine.reset()

    latent_gap = max((a.float() - b.float()).abs().max().item()
                     for a, b in zip(plain_lat, cap_lat, strict=True))
    ring_gap = sum(1 for a, b in zip(plain_ring, cap_ring, strict=True) if a != b)
    if latent_gap != 0.0 or ring_gap:
        raise SystemExit(f"the capture passes perturbed the reference state: {latent_gap=} {ring_gap=}")
    return {
        "what": "reference driver rolled out with and without the dit_out capture passes",
        "frames": frames,
        "latent_maxabs": latent_gap,
        "ring_layers_differing": ring_gap,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--world-engine", default=WORLD_ENGINE_DEFAULT, help="dir containing world_engine's src/")
    ap.add_argument("--model-dir", default=CKPT_DEFAULT)
    ap.add_argument("--ae-dir", default=AE_DEFAULT)
    ap.add_argument("--seed-image", default=SEED_DEFAULT)
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="gen_frame steps after the seed")
    ap.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    ap.add_argument("--expect-gpu", default="H100", help="substring the GPU name must contain")
    ap.add_argument(
        "--ring-snapshot-frames", default="",
        help="comma-separated frame indices to dump the full KV ring for "
             "(~2.2 GiB each), or 'none'. Default: first, middle and last.",
    )
    ap.add_argument("--no-checksums", action="store_true", help="skip sha256 of the checkpoint files")
    ap.add_argument(
        "--verify-shadow", type=int, default=2, metavar="N",
        help="frames to roll out twice, checking the dit_out capture leaves the state alone (0 to skip)",
    )
    return ap.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    world_engine = load_world_engine(args.world_engine)
    CtrlInput, WorldEngine = world_engine.CtrlInput, world_engine.WorldEngine

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to record the oracle.")
    gpu_name = torch.cuda.get_device_name(0)
    if args.expect_gpu and args.expect_gpu not in gpu_name:
        raise SystemExit(f"expected a GPU matching {args.expect_gpu!r}, got {gpu_name!r}")

    out_dir = Path(args.out_dir)
    frames_dir, ring_dir = out_dir / "frames", out_dir / "ring"
    frames_dir.mkdir(parents=True, exist_ok=True)
    ring_dir.mkdir(parents=True, exist_ok=True)

    control = CONTROL_SEQUENCE[: args.num_frames]
    if len(control) < args.num_frames:
        raise SystemExit(f"CONTROL_SEQUENCE has {len(CONTROL_SEQUENCE)} entries, need {args.num_frames}")

    if args.ring_snapshot_frames.strip().lower() == "none":
        snapshot_frames: set[int] = set()
    elif args.ring_snapshot_frames.strip():
        snapshot_frames = {int(s) for s in args.ring_snapshot_frames.split(",") if s.strip()}
    else:
        snapshot_frames = {0, args.num_frames // 2, args.num_frames}

    seed_frame, seed_sha = load_seed_frame(args.seed_image, SEED_URL)

    t_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    # ae_uri is overridden to the local taehv snapshot so recording never depends
    # on the network resolving a repo id to the same files.
    engine = WorldEngine(
        args.model_dir, quant=None, device="cuda", dtype=torch.bfloat16,
        model_config_overrides={"ae_uri": args.ae_dir},
    )
    cfg = engine.model_cfg
    patches = describe_patches(engine.model)
    calibration = calibrate_matmul_precision(args.model_dir, cfg.d_model, list(cfg.scheduler_sigmas), engine.device)

    sigmas = engine.scheduler_sigmas
    dsigmas = sigmas.diff()
    noise_gen = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    per_frame: list[dict] = []

    shadow_check = (
        verify_shadow(engine, seed_frame, CtrlInput, sigmas, dsigmas, args.verify_shadow)
        if args.verify_shadow else None
    )

    # Frame 0: the seed. append_frame unrolled — encode, one committing pass, decode.
    with torch.inference_mode():
        x0 = engine.vae.encode(seed_frame).unsqueeze(1)
        ctx = engine.prep_inputs(x=x0, ctrl=CtrlInput())
        v = shadow_pass()(engine.model, engine.kv_cache, x0, sigmas[-1], ctx)
        v = v.detach().cpu().clone()
        engine._cache_pass(x0, ctx, engine.kv_cache)
        pixels = engine.vae.decode(x0.squeeze(1))

        record = {
            "frame": 0, "kind": "seed",
            "dit_out": [v],
            "pass_sigmas": [0.0],
            "latent": x0.detach().cpu().clone(),
            "pixels": pixels.detach().cpu().clone(),
            "noise_f32": None, "noise_bf16": None,
            "ctx": snapshot_ctx(ctx),
            "committed_kv": committed_kv(engine.kv_cache),
            "ring": ring_digest(engine.kv_cache),
        }
        torch.save(record, frames_dir / "frame_000.pt")
        if 0 in snapshot_frames:
            torch.save(full_ring(engine.kv_cache), ring_dir / "ring_000.pt")
        per_frame.append({"frame": 0, "kind": "seed", "control": None,
                          "latent_absmax": x0.abs().float().max().item()})
        print(f"frame 000 seed  latent|max|={x0.abs().float().max().item():.4f}", flush=True)

        for i, (buttons, mouse, scroll) in enumerate(control, start=1):
            # A fresh CtrlInput per frame: prep_inputs replaces its fields with
            # tensors in place, so a reused instance is not the same input twice.
            ctrl = CtrlInput(button=set(buttons), mouse=tuple(mouse), scroll_wheel=scroll)

            noise_f32 = torch.randn(engine.frm_shape, generator=noise_gen, dtype=torch.float32)
            x = noise_f32.to(device=engine.device, dtype=engine.dtype)

            ctx = engine.prep_inputs(x=x, ctrl=ctrl)
            x0, outs = run_frame(engine, x, ctx, sigmas, dsigmas)
            pixels = engine.vae.decode(x0.squeeze(1))

            record = {
                "frame": i, "kind": "gen",
                "dit_out": outs,
                "pass_sigmas": [float(s) for s in sigmas[:-1]] + [0.0],
                "latent": x0.detach().cpu().clone(),
                "pixels": pixels.detach().cpu().clone(),
                "noise_f32": noise_f32.clone(),
                "noise_bf16": x.detach().cpu().clone(),
                "ctx": snapshot_ctx(ctx),
                "committed_kv": committed_kv(engine.kv_cache),
                "ring": ring_digest(engine.kv_cache),
            }
            torch.save(record, frames_dir / f"frame_{i:03d}.pt")
            if i in snapshot_frames:
                torch.save(full_ring(engine.kv_cache), ring_dir / f"ring_{i:03d}.pt")

            per_frame.append({
                "frame": i, "kind": "gen",
                "control": {"button": sorted(buttons), "mouse": list(mouse), "scroll_wheel": scroll},
                "latent_absmax": x0.abs().float().max().item(),
                "pixels_mean": pixels.float().mean().item(),
            })
            print(f"frame {i:03d} gen   latent|max|={x0.abs().float().max().item():.4f} "
                  f"pixels_mean={pixels.float().mean().item():.2f}", flush=True)

    wall = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated()

    checksums = {}
    if not args.no_checksums:
        for label, path in (("model.safetensors", Path(args.model_dir) / "model.safetensors"),
                            ("config.yaml", Path(args.model_dir) / "config.yaml"),
                            ("taehv1_5.pth", Path(args.ae_dir) / "taehv1_5.pth")):
            if path.exists():
                checksums[label] = sha256_file(path)

    metadata = {
        "model_uri": "Overworld/Waypoint-1.5-1B",
        "ae_uri": "Overworld-Models/taehv1_5",
        "model_dir": str(args.model_dir),
        "ae_dir": str(args.ae_dir),
        "checkpoint_sha256": checksums,
        "seed_image": {"url": SEED_URL, "sha256": seed_sha, "expected_sha256": SEED_SHA256,
                       "resized_to": [1280, 720], "repeated": 4},

        # What makes this artifact comparable to a server, and to nothing else.
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "float32_matmul_precision_note": (
            "mstar/engine/__init__.py sets 'high' process-wide; world_engine sets 'medium' at "
            "import. Recorded under 'high', the serving value, and set after the world_engine "
            "import so it is not undone. See matmul_precision_calibration for the cost."
        ),
        "matmul_precision_calibration": calibration,

        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "reference_pins": "torch==2.11.0",
            "deviation": (
                "recorded on the torch in this environment, not the reference's pin, so both "
                "sides of the Phase 9 comparison run the same build. Do not compare across builds."
            ),
        },
        "device": {"gpu": gpu_name, "count": torch.cuda.device_count()},
        "python": platform.python_version(),

        # apply_inference_patches is unconditional (world_engine.py:84), so the
        # patched model is the reference. These counts are read off the live model.
        "inference_patches": {
            "applied": "apply_inference_patches (unconditional in WorldEngine.__init__)",
            "module_counts": patches,
            "functions": ["patch_cached_noise_conditioning", "patch_Attn_merge_qkv", "patch_MLPFusion_split"],
        },
        "quantization": None,

        "noise": {
            "seed": args.noise_seed,
            "generator": f"torch.Generator(device='cpu').manual_seed({args.noise_seed})",
            "drawn": "fp32 on CPU, cast to bfloat16 on device",
            "reference_draws": "torch.randn(frm_shape, device=cuda, dtype=bfloat16), unseeded",
            "deviation": (
                "noise is an input, not model behaviour; the reference's unseeded device draw is "
                "not reproducible and cannot be handed to the port. Both tensors are saved per "
                "frame as noise_f32 (feed this to the port) and noise_bf16 (what the reference ran)."
            ),
        },

        "execution": {
            "mode": "compiled",
            "state_from": (
                "engine._denoise_pass and engine._cache_pass called unmodified — "
                "@torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS) with "
                "max_autotune, coordinate_descent_tuning and triton.cudagraphs. latent, pixels, "
                "committed_kv and ring are the reference's own values for this run. They are "
                "not bit-exact targets: see reproducibility."
            ),
            "dit_out_from": (
                "separate frozen passes through one compiled region carrying the same decorator "
                "settings, run before the driver. A per-pass diagnostic, not a bit-exact target: "
                "the value depends on which compiled region the pass sits in."
            ),
            "do_not_call_model_directly": (
                "eager flex_attention ignores a BlockMask's block index lists, and the mask "
                "carries a no-op mask_mod, so an eager pass attends over unwritten ring slots. "
                "An earlier eager recording of this oracle was wrong by maxabs 5.64 on peak 8.25 "
                "(rel 0.68). Every DiT pass here must go through a compiled region."
            ),
            "decomposition_note": (
                "measured on this build: making the per-pass output an extra output of "
                "_denoise_pass, or splitting it into five single-pass regions, moves the emitted "
                "latent by 0.031 at the first generated frame and 0.09-0.15 by the third, and "
                "changes every layer's ring. Pass 0 shifts by 0.031 on peak 7.44 with identical "
                "inputs, so this is inductor's fusion choice, not the arithmetic. A port that "
                "runs its denoise passes as separate compiled regions cannot be bit-exact "
                "against latent; ~1 bf16 ULP at frame 1 is its floor."
            ),
            "shadow_isolation_check": shadow_check,
        },

        "reproducibility": {
            "within_a_process": "deterministic — the same driver rolled out twice matches bit for bit",
            "across_processes": (
                "NOT deterministic. Two processes running the reference driver alone — no "
                "recorder, no capture passes — disagree. Layer 0 of the seed frame's committed KV "
                "differs by one bf16 ULP on 0.02% of its elements, and 24 layers of bf16 compound "
                "that: by layer 23 it is 11.6 on a peak of 20.4."
            ),
            "measured_gap": {
                "what": "two independent processes, reference driver only, seed + 3 generated frames",
                "latent_maxabs": [None, 0.0546875, 0.08984375, 0.1328125],
                "latent_peak": 4.53,
                "committed_kv_maxabs": [11.921875, 9.09375, 16.28125, 13.6875],
                "committed_kv_peak": 20.4,
            },
            "cause": (
                "compile-time kernel selection, not a runtime race: within one process every "
                "call reproduces. Disabling triton.cudagraphs does not help. Disabling "
                "max_autotune makes _cache_pass reproducible but not _denoise_pass, so autotune "
                "is one source and not the only one. The inductor and triton caches are warm and "
                "shared across these runs."
            ),
            "consequence": (
                "no comparison against this oracle can assert bit-exactness, because the "
                "reference is not bit-exact against itself. The numbers above are the floor any "
                "tolerance has to clear. companion_run is a second independent recording of the "
                "opening frames, so that floor can be re-measured from artifacts rather than "
                "taken on trust."
            ),
            "companion_run": "repro/ — same seed and controls, recorded by a separate process",
        },

        "num_frames": args.num_frames,
        "total_frames_recorded": args.num_frames + 1,
        "scheduler_sigmas": [float(s) for s in cfg.scheduler_sigmas],
        "passes_per_frame": "4 non-committing denoise + 1 committing",
        "control_sequence": [
            {"frame": i, "button": sorted(b), "mouse": list(m), "scroll_wheel": s}
            for i, (b, m, s) in enumerate(control, start=1)
        ],
        "ring_snapshot_frames": sorted(snapshot_frames),
        "model_config": {k: v for k, v in dict(cfg).items() if not isinstance(v, dict)},
        "frame_shape": list(engine.frm_shape),
        "ts_mult": engine.ts_mult,
        "per_frame": per_frame,
        "wall_time_s": wall,
        "peak_vram_bytes": int(peak),
        "peak_vram_gib": round(peak / 2**30, 2),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"DONE frames={args.num_frames + 1} wall={wall:.1f}s peak_vram={peak / 2**30:.2f}GiB")
    print(f"matmul_precision={torch.get_float32_matmul_precision()} oracle -> {out_dir}")


if __name__ == "__main__":
    main()
