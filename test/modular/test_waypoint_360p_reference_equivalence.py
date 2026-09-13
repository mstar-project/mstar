"""Live, zero-tolerance 360p Waypoint parity against ``world_engine``.

The existing stored oracle was recorded from the distinct 720p checkpoint and
cannot be resized into a numerical target for the 360p model. This gate instead
drives both implementations in one process with the canonical seed, controller
script, and seeded CPU noise recipe used by the oracle recorder. Both DiTs run
eager and the reference's eager Flex call is rebound to the port's masked Flex
kernel, exactly as in the 720p localization harness. Consequently this checks
model arithmetic, state progression, and the functional AE boundary; it is not
a comparison between separately compiled serving processes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_waypoint_reference_equivalence import (
    DEVICE,
    DTYPE,
    REFERENCE_SRC,
    _admit,
    _assert_ring,
    _build_port,
    _commit,
    _deviation,
    _divergent_stages,
    _import_reference,
    _island_tables,
    _new_request,
    _port_forward,
    _port_frame,
    _reference_forward,
    _reference_frame,
    _reference_importable,
    _reset,
    _ring_deviation,
    _stage_capture,
    _stage_modules,
)

from mstar.model.waypoint.components.taehv import (
    decode_latent,
    encode_seed_clip,
    initial_decoder_histories,
    load_taehv,
)
from mstar.model.waypoint.config import waypoint_1_5_1b_360p

_ROOT = Path("/mnt/storage/garv901/waypoint-1.5-1B")
CHECKPOINT = Path(
    os.environ.get(
        "WAYPOINT_360P_CHECKPOINT",
        "/tmp/waypoint-hf-cache/models--Overworld--Waypoint-1.5-1B-360P/"
        "snapshots/35acd20e649fe79c1c1002df456696408547202d",
    )
)
AE_CHECKPOINT = Path(os.environ.get("WAYPOINT_AE_CHECKPOINT", _ROOT / "checkpoints/taehv1_5"))
SEED_IMAGE = Path(os.environ.get("WAYPOINT_SEED_IMAGE", _ROOT / "checkpoints/seed/default.jpg"))
REPORT_PATH = Path(os.environ.get("WAYPOINT_360P_PARITY_REPORT", "/tmp/waypoint-360p-reference-parity.json"))

CHECKPOINT_REVISION = "35acd20e649fe79c1c1002df456696408547202d"
CHECKPOINT_SHA256 = "a2cdccb5eb074afc48a1c99b0868cebef38cf944d4b366aed2a866f50c8101d9"
SEED_SHA256 = "c61c9393311d7281f793d86329dca343e12c93bf0409980a186eb39269cf6862"
NOISE_SEED = 42
ROLLOUT_FRAMES = int(os.environ.get("WAYPOINT_360P_PARITY_FRAMES", "41"))

# Same 40 post-prime actions as ``test/waypoint/record_oracle.py``. Inputs are
# resolution-independent; only the seeded noise tensor takes the 360p shape.
CONTROL_SEQUENCE: list[tuple[set[int], tuple[float, float], int]] = (
    [({87}, (0.0, 0.0), 0)] * 8
    + [({87}, (0.2, 0.0), 0)] * 4
    + [({65}, (0.0, 0.0), 0)] * 4
    + [({68}, (0.0, 0.0), 0)] * 4
    + [({83}, (0.0, 0.0), 0)] * 4
    + [({87, 32}, (0.0, 0.0), 0)] * 4
    + [(set(), (0.0, 0.0), 0)] * 4
    + [(set(), (0.0, -0.2), 0)] * 4
    + [({87, 1}, (0.0, 0.0), 0)] * 4
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not _reference_importable(), reason=f"reference not at {REFERENCE_SRC}"),
    pytest.mark.skipif(
        not (CHECKPOINT / "model.safetensors").is_file(),
        reason=f"360p checkpoint missing at {CHECKPOINT}",
    ),
    pytest.mark.skipif(not AE_CHECKPOINT.exists(), reason=f"TAEHV missing at {AE_CHECKPOINT}"),
    pytest.mark.skipif(not SEED_IMAGE.is_file(), reason=f"seed missing at {SEED_IMAGE}"),
    pytest.mark.skipif(importlib.util.find_spec("taehv") is None, reason="taehv is not installed"),
]


def _context(
    config,
    frame: int,
    buttons: set[int] | None = None,
    mouse: tuple[float, float] = (0.0, 0.0),
    scroll: int = 0,
) -> dict[str, torch.Tensor]:
    button = torch.zeros((1, 1, config.n_buttons), dtype=DTYPE, device=DEVICE)
    if buttons:
        button[..., sorted(buttons)] = 1
    return {
        "button": button,
        "mouse": torch.tensor([[mouse]], dtype=DTYPE, device=DEVICE),
        "scroll": torch.tensor([[[scroll]]], dtype=DTYPE, device=DEVICE),
        "frame_timestamp": torch.tensor([[frame]], dtype=torch.int64, device=DEVICE),
        "frame_idx": torch.tensor([[frame]], dtype=torch.int64, device=DEVICE),
    }


def _seed_clip() -> tuple[torch.Tensor, str]:
    import cv2
    import numpy as np

    raw = SEED_IMAGE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.repeat(image[None], 4, axis=0)), digest


def _load_reference() -> dict:
    WorldModel, StaticKVCache, patch_model = _import_reference()
    torch.set_float32_matmul_precision("high")
    cfg = WorldModel.load_config(str(CHECKPOINT))
    assert (cfg.tokens_per_frame, cfg.height, cfg.width) == (128, 8, 16)
    model = WorldModel.from_pretrained(str(CHECKPOINT), cfg=cfg, device=DEVICE, dtype=DTYPE).eval()
    islands = {
        "freq": model.denoise_step_emb.freq.clone(),
        "xy": model.transformer.rope_angles.xy.clone(),
        "inv_t": model.transformer.rope_angles.inv_t.clone(),
    }
    bare_conditioner = model.denoise_step_emb
    patch_model.apply_inference_patches(model)
    patch_model.flex_attention = __import__(
        "mstar.engine.resources.attn.flex", fromlist=["flex_attention_masked"]
    ).flex_attention_masked
    cache = StaticKVCache(cfg, batch_size=1, dtype=DTYPE).to(device=DEVICE)
    return {
        "cfg": cfg,
        "model": model,
        "kv": cache,
        "islands": islands,
        "bare_conditioner": bare_conditioner,
    }


def _reference_decoder_histories(session) -> tuple[torch.Tensor, ...]:
    return tuple(value for value in session.streaming_ae_model.decoder_memory if torch.is_tensor(value))


def _assert_histories_equal(actual, expected, frame: int) -> None:
    assert len(actual) == len(expected) == 9
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(left, right), (
            f"frame {frame} decoder history {index} differs: maxabs={_deviation(left, right)[0]:.4e}"
        )


@torch.inference_mode()
def test_360p_reference_compat_is_bit_exact_end_to_end():
    """41 live frames: tables through pixels, including two local-ring wraps."""
    assert ROLLOUT_FRAMES > 32, "the live gate must cross two 16-frame local windows"
    assert ROLLOUT_FRAMES <= len(CONTROL_SEQUENCE) + 1

    reference = _load_reference()
    config = replace(waypoint_1_5_1b_360p(), reference_compat=True, compile_dit=False)
    port = _build_port(config, CHECKPOINT)

    table_names = []
    for name, (expected, actual) in _island_tables(port, reference).items():
        gap, relative = _deviation(expected, actual)
        print(f"table {name:>24} maxabs={gap:.4e} rel={relative:.4e}")
        assert gap == 0.0, f"360p {name} maxabs={gap:.4e} rel={relative:.4e}"
        table_names.append(name)

    for value in reference["cfg"].scheduler_sigmas:
        sigma = torch.tensor([[value]], dtype=DTYPE, device=DEVICE)
        expected = reference["model"].denoise_step_emb(sigma)
        actual = port["dit"].denoise_step_emb(sigma)
        gap, relative = _deviation(expected, actual)
        print(f"conditioner sigma={value:<7.4f} maxabs={gap:.4e} rel={relative:.4e}")
        assert gap == 0.0, f"360p conditioner sigma={value} maxabs={gap:.4e} rel={relative:.4e}"

    clip, seed_digest = _seed_clip()
    assert seed_digest == SEED_SHA256
    ae = load_taehv(str(AE_CHECKPOINT)).to(device=DEVICE, dtype=DTYPE)
    from src.ae import ChunkedStreamingTAEHV as ReferenceTAEHV

    reference_ae = ReferenceTAEHV(
        ae,
        auto_aspect_ratio=True,
        device=DEVICE,
        dtype=DTYPE,
        height=config.latent_height,
        width=config.latent_width,
    )
    reference_seed = reference_ae.encode(clip)
    functional_seed = encode_seed_clip(
        ae,
        clip.to(device=DEVICE, dtype=DTYPE).div(255),
        output_size=(256, 512),
    )
    assert torch.equal(reference_seed, functional_seed), (
        f"360p functional encoder differs from upstream: maxabs={_deviation(reference_seed, functional_seed)[0]:.4e}"
    )

    idle = _context(config, 0)
    stage_names = [name for name, _ in _stage_modules(port["dit"], is_port=True)]
    assert len(stage_names) == 30
    for sigma_value in (1.0, 0.0):
        commit = sigma_value == 0.0
        _new_request(port)
        _reset(port, reference)
        reference_stages: dict[str, torch.Tensor] = {}
        with _stage_capture(_stage_modules(reference["model"], is_port=False), reference_stages):
            expected = _reference_forward(reference, reference_seed.unsqueeze(1), sigma_value, idle, commit=commit)
        port_stages: dict[str, torch.Tensor] = {}
        _admit(port, 0)
        with _stage_capture(_stage_modules(port["dit"], is_port=True), port_stages):
            actual = _port_forward(port, functional_seed.unsqueeze(1), sigma_value, idle, 0, commit=commit)
        _commit(port, 0)
        divergent = _divergent_stages(reference_stages, port_stages, stage_names)
        gap, relative = _deviation(expected, actual)
        print(f"stages sigma={sigma_value}: divergent={len(divergent)}/30 output maxabs={gap:.4e} rel={relative:.4e}")
        assert not divergent, f"360p sigma={sigma_value} first divergence: {divergent[0]}"
        assert gap == 0.0
        _assert_ring(_ring_deviation(reference, port), exact=True)

    _new_request(port)
    _reset(port, reference)
    histories = initial_decoder_histories(ae, functional_seed)
    sigmas = torch.tensor(list(reference["cfg"].scheduler_sigmas), dtype=DTYPE, device=DEVICE)
    noise_generator = torch.Generator(device="cpu").manual_seed(NOISE_SEED)
    frame_shape = (
        1,
        1,
        config.channels,
        config.latent_height,
        config.latent_width,
    )

    max_gaps = {
        "tables": 0.0,
        "conditioner": 0.0,
        "stages": 0.0,
        "passes": 0.0,
        "latents": 0.0,
        "ring_kv": 0.0,
        "encoder": 0.0,
        "decoder_histories": 0.0,
        "pixels": 0,
    }
    for frame in range(ROLLOUT_FRAMES):
        if frame == 0:
            ctx = idle
            expected = reference_seed.unsqueeze(1)
            actual = functional_seed.unsqueeze(1)
            expected_velocity = _reference_forward(reference, expected, 0.0, ctx, commit=True)
            _admit(port, 0)
            actual_velocity = _port_forward(port, actual, 0.0, ctx, 0, commit=True)
            _commit(port, 0)
            pass_outputs = [(expected_velocity, actual_velocity)]
        else:
            buttons, mouse, scroll = CONTROL_SEQUENCE[frame - 1]
            ctx = _context(config, frame, buttons, mouse, scroll)
            noise = torch.randn(frame_shape, generator=noise_generator, dtype=torch.float32)
            noise = noise.to(device=DEVICE, dtype=DTYPE)
            expected, reference_passes = _reference_frame(reference, noise, ctx, sigmas)
            actual, port_passes = _port_frame(port, noise, ctx, frame)
            assert len(reference_passes) == len(port_passes) == 5
            pass_outputs = list(zip(reference_passes, port_passes, strict=True))

        for pass_index, (left, right) in enumerate(pass_outputs):
            pass_gap, pass_relative = _deviation(left, right)
            assert pass_gap == 0.0, (
                f"360p frame {frame} pass {pass_index}: maxabs={pass_gap:.4e} rel={pass_relative:.4e}"
            )
        latent_gap, latent_relative = _deviation(expected, actual)
        assert latent_gap == 0.0, f"360p frame {frame} latent maxabs={latent_gap:.4e} rel={latent_relative:.4e}"
        ring = _ring_deviation(reference, port)
        ring_worst = _assert_ring(ring, exact=True)

        reference_pixels = reference_ae.decode(expected.squeeze(1))
        functional_pixels, histories = decode_latent(
            ae,
            actual.squeeze(1),
            histories,
            output_size=(360, 640),
            initialize=frame == 0,
        )
        pixel_gap = int((reference_pixels.to(torch.int16) - functional_pixels.to(torch.int16)).abs().max().item())
        assert pixel_gap == 0, f"360p frame {frame} pixels maxabs={pixel_gap}/255"
        _assert_histories_equal(histories, _reference_decoder_histories(reference_ae), frame)
        print(
            f"frame {frame:3d} passes={len(pass_outputs)} latent maxabs={latent_gap:.4e} "
            f"ring maxabs={ring_worst[1]:.4e} pixels maxabs={pixel_gap}"
        )

    report = {
        "status": "passed",
        "variant": "waypoint-1.5-1b-360p",
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "reference_source": str(REFERENCE_SRC.resolve()),
        "device": torch.cuda.get_device_name(DEVICE),
        "device_index_visible": DEVICE.index,
        "dtype": str(DTYPE),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "reference_compat": True,
        "rollout_latent_frames": ROLLOUT_FRAMES,
        "generated_latent_frames": ROLLOUT_FRAMES - 1,
        "denoise_and_commit_passes": 1 + 5 * (ROLLOUT_FRAMES - 1),
        "decoded_rgb_frames_including_internal_prime": 4 * ROLLOUT_FRAMES,
        "stage_names": stage_names,
        "stage_probes": {"sigmas": [1.0, 0.0], "stages_per_probe": len(stage_names)},
        "derived_tables": table_names,
        "scheduler_sigmas": list(config.scheduler_sigmas),
        "noise": {"source": "seeded CPU fp32 then cast to CUDA BF16", "seed": NOISE_SEED},
        "controls": "canonical 40-action record_oracle.py sequence",
        "seed_sha256": seed_digest,
        "maximum_absolute_differences": max_gaps,
        "scope": {
            "reference": "live same-process world_engine",
            "dit_execution": "eager on both sides",
            "attention": "shared mstar flex_attention_masked kernel",
            "stored_oracle": "not used; existing artifacts belong to the 720p checkpoint",
            "taehv": "upstream streaming state versus functional explicit nine-history path",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"360p parity report: {REPORT_PATH}")
