"""Real-weight CPU parity for Waypoint's functional TAEHV graph boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from mstar.model.waypoint.components.taehv import (
    decode_latent,
    encode_seed_clip,
    initial_decoder_histories,
)

taehv = pytest.importorskip("taehv")


_ROOT = Path(__file__).resolve().parents[4]
_AE_SOURCE = Path(
    os.environ.get(
        "WAYPOINT_AE_CHECKPOINT",
        _ROOT / "checkpoints" / "taehv1_5" / "taehv1_5.pth",
    )
)
_AE_CHECKPOINT = (
    _AE_SOURCE / "taehv1_5.pth" if _AE_SOURCE.is_dir() else _AE_SOURCE
)

pytestmark = pytest.mark.skipif(
    not _AE_CHECKPOINT.is_file(),
    reason=f"real TAEHV checkpoint not found at {_AE_CHECKPOINT}",
)


def _reference_decode(stream, latent: torch.Tensor) -> tuple[torch.Tensor, ...]:
    first = stream.decode(latent[:, None])
    assert first is not None
    return (first, *stream.flush_decoder())


def _rgb24(frames: tuple[torch.Tensor, ...]) -> torch.Tensor:
    decoded = torch.cat(frames, dim=1)
    return (
        (decoded.clamp(0, 1) * 255)
        .round()
        .to(torch.uint8)
        .squeeze(0)
        .permute(0, 2, 3, 1)[..., :3]
        .contiguous()
    )


@torch.inference_mode()
def test_functional_taehv_matches_upstream_init_and_steady_state():
    """Exercise actual block types while keeping the spatial grid inexpensive."""
    ae = taehv.TAEHV(str(_AE_CHECKPOINT)).eval().float()
    generator = torch.Generator(device="cpu").manual_seed(123)
    seed = torch.rand((4, 32, 32, 3), generator=generator)

    encoded = encode_seed_clip(ae, seed, output_size=(32, 32))
    encoder_reference = taehv.StreamingTAEHV(ae)
    expected_encoded = encoder_reference.encode(
        seed[None].permute(0, 1, 4, 2, 3).contiguous()
    )
    assert expected_encoded is not None
    assert torch.equal(encoded, expected_encoded.squeeze(1))

    histories = initial_decoder_histories(ae, encoded)
    initialized_frames, initialized_histories = decode_latent(
        ae, encoded, histories, output_size=(32, 32), initialize=True
    )

    decoder_reference = taehv.StreamingTAEHV(ae)
    for _ in range(ae.frames_to_trim):
        decoder_reference.decode(encoded[:, None])
        decoder_reference.flush_decoder()
    expected_initialized = _rgb24(_reference_decode(decoder_reference, encoded))
    reference_histories = tuple(
        value for value in decoder_reference.decoder_memory if torch.is_tensor(value)
    )
    assert torch.equal(initialized_frames, expected_initialized)
    assert len(reference_histories) == len(initialized_histories) == 9
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            initialized_histories, reference_histories, strict=True
        )
    )

    next_latent = torch.randn(encoded.shape, generator=generator)
    steady_frames, steady_histories = decode_latent(
        ae,
        next_latent,
        initialized_histories,
        output_size=(32, 32),
        initialize=False,
    )
    expected_steady = _rgb24(_reference_decode(decoder_reference, next_latent))
    reference_histories = tuple(
        value for value in decoder_reference.decoder_memory if torch.is_tensor(value)
    )
    assert torch.equal(steady_frames, expected_steady)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(steady_histories, reference_histories, strict=True)
    )
