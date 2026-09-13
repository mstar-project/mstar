"""L4: the whole pipeline, pixels out, against the reference.

Seed clip -> TAEHV encode -> ``append_frame`` -> decode, then N x (noise -> the
4+1 driver -> decode). The reference is driven live in-process under
``reference_compat=True`` and the bar is bit-exact pixels on every emitted
frame. ``test_waypoint_reference_equivalence.py`` builds both sides and this
file imports its fixtures' internals rather than restating them, so a failure
here cannot be a harness difference between L3 and L4.

The stored oracle's ``pixels`` are not a bit-exact target -- two reference
processes disagree by up to 106/255 by frame 6, measured from ``repro/``. Two
things it is still authoritative for, both gated below at zero tolerance:

  * **Frame 0's pixels**, which the two recordings agree on exactly. Priming is
    a pure AE round trip -- ``append_frame`` decodes ``vae.encode(img)`` and the
    DiT only writes the ring -- so no compiled DiT kernel reaches those bytes.
  * **Frame ordering.** Across the 28 raw frames of the ``repro/`` pair, every
    frame's nearest neighbour in the other recording is itself, by a factor of
    at least 2.06 in L2. A streaming-decoder shift is therefore separable from
    the reference's own run-to-run noise without picking a tolerance.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from test_waypoint_reference_equivalence import (
    CHECKPOINT,
    DEVICE,
    DTYPE,
    ORACLE,
    _admit,
    _build_port,
    _commit,
    _ctx,
    _deviation,
    _frame,
    _import_reference,
    _new_request,
    _port_frame,
    _reference_frame,
    _reference_importable,
    _reset,
)

from mstar.model.waypoint.components.taehv import ChunkedStreamingTAEHV, load_taehv
from mstar.model.waypoint.config import waypoint_1_5_1b_720p

AE_CHECKPOINT = Path(os.environ.get("WAYPOINT_AE_CHECKPOINT", CHECKPOINT.parent / "taehv1_5"))
# The oracle's seed image, cached by test/waypoint/record_oracle.py. The digest
# is what makes the oracle comparisons below mean anything: a different image is
# a different world, and the port would then be compared against a recording of
# something else.
SEED_IMAGE = Path(os.environ.get("WAYPOINT_SEED_IMAGE", CHECKPOINT.parent / "seed/default.jpg"))
SEED_SHA256 = "c61c9393311d7281f793d86329dca343e12c93bf0409980a186eb39269cf6862"

REPRO = ORACLE / "repro"

# Frames in the pixel rollout, priming included. 21 wraps the 16-frame local
# ring once, which is where a ring-addressing fault would first reach pixels.
L4_FRAMES = int(os.environ.get("WAYPOINT_L4_FRAMES", "21"))


def _repro_frames() -> int:
    return len(sorted((REPRO / "frames").glob("frame_*.pt")))


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not _reference_importable(), reason="reference source missing"),
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors").exists(), reason="checkpoint missing"),
    pytest.mark.skipif(not (ORACLE / "frames").is_dir(), reason=f"oracle not at {ORACLE}"),
    pytest.mark.skipif(not AE_CHECKPOINT.exists(), reason=f"TAEHV not at {AE_CHECKPOINT}"),
    pytest.mark.skipif(not SEED_IMAGE.exists(), reason=f"seed image not at {SEED_IMAGE}"),
]


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference():
    """The served reference, minus the fp32-island capture L0 needs.
    ``float32_matmul_precision`` is what the oracle recorded under and what the
    compat port's sigma LUT depends on; see the L1-L3 file's fixture."""
    from mstar.engine.resources.attn.flex import flex_attention_masked

    WorldModel, StaticKVCache, patch_model = _import_reference()
    torch.set_float32_matmul_precision("high")

    cfg = WorldModel.load_config(str(CHECKPOINT))
    model = WorldModel.from_pretrained(
        str(CHECKPOINT), cfg=cfg, device=DEVICE, dtype=DTYPE
    ).eval()
    patch_model.apply_inference_patches(model)
    patch_model.flex_attention = flex_attention_masked
    kv = StaticKVCache(cfg, batch_size=1, dtype=DTYPE).to(device=DEVICE)
    return {"cfg": cfg, "model": model, "kv": kv}


@pytest.fixture(scope="module")
def port(reference):
    """The compat port. Depends on ``reference`` for the ordering, not the
    object: the sigma LUT is a fp32 GEMM and that fixture sets the precision."""
    return _build_port(
        replace(
            waypoint_1_5_1b_720p(),
            reference_compat=True,
            compile_dit=False,
        )
    )


@pytest.fixture(scope="module")
def taehv_weights():
    """One TAEHV, shared by both sides' sessions. ``StreamingTAEHV`` keeps every
    piece of stream state on itself and never writes back to the weights, so
    sharing is what isolates the two wrappers as the thing under test."""
    return load_taehv(str(AE_CHECKPOINT)).to(device=DEVICE, dtype=DTYPE)


@pytest.fixture(scope="module")
def oracle():
    return {"frames": sorted((ORACLE / "frames").glob("frame_*.pt"))}


@pytest.fixture(scope="module")
def seed_clip():
    """The oracle's seed frame as ``[4, 720, 1280, 3]`` uint8, decoded and
    resized in ``record_oracle.load_seed_frame``'s order."""
    import cv2
    import numpy as np

    raw = SEED_IMAGE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(cv2.resize(img, (1280, 720)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.repeat(img[None], 4, axis=0)), digest


# ---------------------------------------------------------------------------
# Driving both sides
# ---------------------------------------------------------------------------


def _reference_session(weights, cfg):
    from src.ae import ChunkedStreamingTAEHV as ReferenceSession

    ph, pw = cfg.patch
    return ReferenceSession(
        weights,
        auto_aspect_ratio=cfg.auto_aspect_ratio,
        device=DEVICE,
        dtype=DTYPE,
        height=cfg.height * ph,
        width=cfg.width * pw,
    )


def _port_session(weights, config):
    return ChunkedStreamingTAEHV(
        weights,
        auto_aspect_ratio=config.auto_aspect_ratio,
        device=DEVICE,
        dtype=DTYPE,
        height=config.latent_height,
        width=config.latent_width,
    )


def _reference_prime(reference, session, clip, ctx):
    """``WorldEngine.append_frame`` unrolled: encode, one committing pass at
    sigma 0, decode."""
    with torch.inference_mode():
        x0 = session.encode(clip).unsqueeze(1)
        reference["kv"].set_frozen(False)
        reference["model"](x0, x0.new_zeros((1, 1)), **ctx, kv_cache=reference["kv"])
        return x0, session.decode(x0.squeeze(1))


def _port_prime(port, session, clip, ctx):
    """The prime walk's three nodes, in the order the graph runs them."""
    with torch.inference_mode():
        # WaypointVaeEncoderSubmodule.prepare_inputs: cast, then divide.
        latent = session.encode(clip.to(device=DEVICE, dtype=DTYPE).div(255)).unsqueeze(1)
        _admit(port, 0)
        x0 = port["dit"].append_frame(
            latent,
            torch.tensor(0, dtype=torch.int64, device=DEVICE),
            mouse=ctx["mouse"],
            button=ctx["button"],
            scroll=ctx["scroll"],
        )
        _commit(port, 0)
        return x0, session.decode(x0.squeeze(1))


def _pixel_gap(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.to(torch.int16) - right.to(torch.int16)).abs().max().item())


@pytest.fixture(scope="module")
def rollout(port, reference, taehv_weights, oracle, seed_clip):
    """One shared pixel rollout: both sides stepped in lockstep on the oracle's
    noise and controls, from an empty ring and a fresh decoder on each side.
    Pixels are kept only as far as ``repro/`` reaches; a raw frame is 2.8 MB."""
    clip, digest = seed_clip
    sessions = (
        _reference_session(taehv_weights, reference["cfg"]),
        _port_session(taehv_weights, port["config"]),
    )
    sigmas = torch.tensor(
        list(reference["cfg"].scheduler_sigmas), dtype=DTYPE, device=DEVICE
    )
    keep = min(_repro_frames(), L4_FRAMES)

    _new_request(port)
    _reset(port, reference)
    out = {"seed_sha256": digest, "latent_gap": [], "pixel_gap": [], "port_pixels": []}
    with torch.inference_mode():
        for index in range(L4_FRAMES):
            record = _frame(oracle, index)
            ctx = _ctx(record)
            if record["kind"] == "seed":
                left, left_pixels = _reference_prime(reference, sessions[0], clip, ctx)
                right, right_pixels = _port_prime(port, sessions[1], clip, ctx)
            else:
                noise = record["noise_bf16"].to(DEVICE)
                left, _ = _reference_frame(reference, noise, ctx, sigmas)
                left_pixels = sessions[0].decode(left.squeeze(1))
                right, _ = _port_frame(port, noise, ctx, index)
                right_pixels = sessions[1].decode(right.squeeze(1))

            out["latent_gap"].append(_deviation(left, right)[0])
            out["pixel_gap"].append(_pixel_gap(left_pixels, right_pixels))
            if index < keep:
                out["port_pixels"].append(right_pixels.cpu().clone())
            out.setdefault("emitted", []).append(
                (tuple(right_pixels.shape), right_pixels.dtype)
            )
    return out


# ---------------------------------------------------------------------------
# L4 -- pixels against the live reference
# ---------------------------------------------------------------------------


def test_the_decoded_pixels_are_bit_exact_under_reference_compat(rollout):
    """The headline claim, on every emitted frame rather than the last: the
    streaming decoder's memory advances per call, so a duplicate, gap or reorder
    shifts the stream from that point on and the final frame cannot see it.

    A divergent latent means the DiT; an equal latent with divergent pixels
    means this file's own seam, the AE.
    """
    assert len(rollout["pixel_gap"]) == L4_FRAMES, "the rollout stopped early"
    for index, (latent_gap, pixel_gap) in enumerate(
        zip(rollout["latent_gap"], rollout["pixel_gap"], strict=True)
    ):
        print(f"frame {index:3d} latent maxabs={latent_gap:.4e} pixels maxabs={pixel_gap}")

    divergent = [
        (index, latent_gap, pixel_gap)
        for index, (latent_gap, pixel_gap) in enumerate(
            zip(rollout["latent_gap"], rollout["pixel_gap"], strict=True)
        )
        if pixel_gap != 0
    ]
    assert not divergent, (
        f"pixels differ from the reference at {len(divergent)} frame(s); first is "
        f"frame={divergent[0][0]} latent maxabs={divergent[0][1]:.4e} "
        f"pixels maxabs={divergent[0][2]}/255"
    )


def test_the_seed_clip_encodes_identically_on_both_sides(rollout):
    """``append_frame`` returns its input unchanged, so frame 0's latent gap is
    the encoder's alone -- the one place in the pipeline the DiT cannot reach."""
    assert rollout["latent_gap"][0] == 0.0, (
        f"the two TAEHV sessions encoded the seed clip differently: maxabs="
        f"{rollout['latent_gap'][0]:.4e}"
    )


def test_every_emit_carries_one_latent_frame_of_raw_rgb(rollout, port):
    """The payload the client is handed: ``[temporal_compression, H, W, 3]``
    uint8 per engine step, priming included."""
    config = port["config"]
    expected = (config.temporal_compression, 720, 1280, 3)
    assert rollout["emitted"] == [(expected, torch.uint8)] * L4_FRAMES


# ---------------------------------------------------------------------------
# L4 -- the stored oracle, where it is authoritative
# ---------------------------------------------------------------------------


def _oracle_pixels(root: Path, count: int) -> torch.Tensor:
    """``[4 * count, H, W, 3]`` -- one recording's raw frames, in emission
    order."""
    return torch.cat(
        [
            torch.load(root / f"frame_{i:03d}.pt", map_location="cpu", weights_only=False)[
                "pixels"
            ]
            for i in range(count)
        ]
    )


def _distances(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """``[i, j] = ||left[i] - right[j]||_2``, by explicit difference.

    Not ``torch.cdist``: it expands to ``|a|^2 + |b|^2 - 2a.b``, and under the
    ``float32_matmul_precision("high")`` this file runs at, the TF32 dot product
    cancels away every digit separating two near-identical 720p frames.
    """
    return torch.stack([(right - row).flatten(1).norm(dim=1) for row in left])


def test_the_primed_frame_matches_the_oracle_exactly(rollout, oracle):
    """The one frame the oracle *is* a bit-exact target for.

    Priming decodes ``vae.encode(seed)`` and the DiT's committing pass only
    writes the ring, so no compiled DiT kernel reaches these bytes and the two
    recordings agree on them exactly. A zero-tolerance gate on the seed clip,
    the encoder, ``frames_to_trim`` priming, both resizes and the uint8
    quantization at once.
    """
    assert rollout["seed_sha256"] == SEED_SHA256, (
        f"{SEED_IMAGE} is not the image the oracle was recorded from "
        f"({rollout['seed_sha256']} != {SEED_SHA256})"
    )
    recorded = _frame(oracle, 0)["pixels"]
    floor = _pixel_gap(recorded, _oracle_pixels(REPRO / "frames", 1))
    gap = _pixel_gap(rollout["port_pixels"][0], recorded)
    print(f"prime pixels: port vs oracle maxabs={gap}, oracle vs repro maxabs={floor}")
    assert floor == 0, (
        "the two reference recordings disagree on the primed frame, so it is no "
        f"longer a bit-exact target: maxabs={floor}/255"
    )
    assert gap == 0, f"primed pixels differ from the oracle by maxabs={gap}/255"


def test_the_emitted_stream_stays_aligned_with_the_oracle(rollout, oracle):
    """A decoder desync is a *shift*, and a shift is visible without a tolerance:
    every raw frame's nearest neighbour in the recording must be itself. Over
    the 28 raw frames of the ``repro/`` pair that holds with the closest wrong
    frame at least 2.06x further away.

    The magnitudes printed here are reported, not gated: the port runs eager and
    the oracle ran compiled under ``max_autotune``, and nothing in the artifacts
    bounds that.
    """
    count = min(_repro_frames(), L4_FRAMES)
    recorded = _oracle_pixels(ORACLE / "frames", count).to(DEVICE, torch.float32)
    port_stream = torch.cat(rollout["port_pixels"]).to(DEVICE, torch.float32)
    repro = _oracle_pixels(REPRO / "frames", count).to(DEVICE, torch.float32)

    n = recorded.shape[0]
    assert port_stream.shape[0] == n, f"{port_stream.shape[0]} raw frames against {n}"
    distance = _distances(port_stream, recorded)
    floor = _distances(repro, recorded).diagonal()
    del port_stream, repro

    diagonal = distance.diagonal()
    # inf on the diagonal, not `eye * inf`: that leaves 0 * inf = nan everywhere
    # else, and every `<=` against a nan is False -- the assertion below would
    # hold whatever the pixels did.
    mask = torch.eye(n, dtype=torch.bool, device=DEVICE)
    off = distance.masked_fill(mask, torch.inf).min(dim=1)
    assert torch.isfinite(off.values).all(), "the off-diagonal search produced non-finite distances"
    for i in range(n):
        print(
            f"raw frame {i:3d} L2 to oracle={diagonal[i]:9.1f} "
            f"(repro floor {floor[i]:9.1f})  nearest other="
            f"{off.values[i]:9.1f} at {off.indices[i].item()}"
        )
    shifted = [i for i in range(n) if off.values[i] <= diagonal[i]]
    assert not shifted, (
        f"raw frames closer to a different oracle frame than to their own: "
        f"{[(i, off.indices[i].item()) for i in shifted]}; the streaming decoder "
        "is out of step with the world"
    )
