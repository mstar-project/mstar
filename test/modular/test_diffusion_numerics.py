"""CPU tests pinning the DiT scaffold's small numerics choices to the reference pipelines:
the HF rounding order of the Qwen3 RMSNorm, the image processor's uint8 quantisation order,
and the two pipelines' base sigma grids.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from mstar.model.components.diffusion.flow_match import FlowMatchConfig, FlowMatchSchedule, euler_step  # noqa: E402
from mstar.model.components.diffusion.image_io import encode_image, pixels_to_uint8, uint8_to_png  # noqa: E402
from mstar.model.components.diffusion.qwen3.encoder import Qwen3RMSNorm, qwen3_rotary_tables  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_qwen3_rmsnorm_matches_hf_rounding_order(dtype):
    hf = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
    torch.manual_seed(0)
    dim, eps = 64, 1e-6
    reference = hf.Qwen3RMSNorm(dim, eps=eps)
    native = Qwen3RMSNorm(dim, eps=eps)
    with torch.no_grad():
        reference.weight.copy_(torch.randn(dim) * 0.5 + 1.0)
        native.weight.copy_(reference.weight)
    reference, native = reference.to(dtype), native.to(dtype)
    x = (torch.randn(4, 37, dim) * 3).to(dtype)
    out, expected = native(x), reference(x)
    assert out.dtype == dtype
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_pixels_to_uint8_matches_the_image_processor():
    image_processor = pytest.importorskip("diffusers.image_processor")
    torch.manual_seed(0)
    image = (torch.rand(2, 3, 16, 24) * 2.2 - 1.1).to(torch.bfloat16)  # slightly past [-1, 1], as decoders are
    processor = image_processor.VaeImageProcessor()
    pil = processor.postprocess(image.clone(), output_type="pil", do_denormalize=[True, True])
    expected = torch.from_numpy(np.stack([np.array(im) for im in pil])).permute(0, 3, 1, 2)
    out = pixels_to_uint8(image)
    assert out.dtype == torch.uint8 and out.shape == (2, 3, 16, 24)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_steps", [3, 8, 10, 50])
def test_base_grid_follows_the_pipeline(num_steps):
    # shift 1.0 makes the exponential shift the identity, exposing the base grid
    numpy_grid = FlowMatchSchedule.build(FlowMatchConfig(shift=1.0), num_steps, 1024).sigmas[:-1]
    torch_grid = FlowMatchSchedule.build(FlowMatchConfig(shift=1.0, torch_linspace=True), num_steps, 1024).sigmas[:-1]
    torch.testing.assert_close(
        numpy_grid, torch.from_numpy(np.linspace(1.0, 1 / num_steps, num_steps).astype(np.float32)), rtol=0, atol=0,
    )
    torch.testing.assert_close(torch_grid, torch.linspace(1.0, 1 / num_steps, num_steps), rtol=0, atol=0)
    if num_steps in (3, 10, 50):
        assert not torch.equal(numpy_grid, torch_grid), "these step counts are where the two grids differ"


@pytest.mark.parametrize("seq,head_dim,theta", [(512, 128, 1_000_000.0), (37, 64, 10_000.0)])
def test_qwen3_rotary_tables_match_hf(seq, head_dim, theta):
    hf = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
    config = hf.Qwen3Config(
        hidden_size=head_dim * 2, num_attention_heads=2, num_key_value_heads=2, head_dim=head_dim, rope_theta=theta,
        max_position_embeddings=max(seq, 4096),
    )
    rotary = hf.Qwen3RotaryEmbedding(config)
    positions = torch.arange(seq)
    cos_ref, sin_ref = rotary(torch.zeros(1, seq, head_dim), positions[None])
    cos, sin = qwen3_rotary_tables(positions, head_dim, theta)
    assert cos.dtype == sin.dtype == torch.float32 and cos.shape == (seq, head_dim)
    torch.testing.assert_close(cos, cos_ref[0], rtol=0, atol=0)
    torch.testing.assert_close(sin, sin_ref[0], rtol=0, atol=0)


def test_qwen3_encoder_builds_one_rotary_table_across_prompt_lengths(monkeypatch):
    """The encoder builds its rotary table once per device, at its longest length, and
    slices it: a table per prompt length was a device allocation per new length, never
    freed. A slice computes what a table built at that length does."""
    from mstar.model.components.diffusion.qwen3 import encoder as qwen3_encoder

    torch.manual_seed(0)
    dims = dict(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_heads=2, num_kv_heads=1,
        head_dim=8, rms_norm_eps=1e-6, rope_theta=10_000.0, tap_layers=(1,),
    )
    encoder = qwen3_encoder.Qwen3HiddenStateEncoder(**dims, max_positions=16).eval()
    with torch.no_grad():
        for p in encoder.parameters():  # some layers allocate with torch.empty
            p.normal_(std=0.1)
    builds = []
    build = qwen3_encoder.qwen3_rotary_tables
    monkeypatch.setattr(qwen3_encoder, "qwen3_rotary_tables", lambda *a: builds.append(a) or build(*a))
    prompts = [torch.randint(0, 32, (1, seq)) for seq in (3, 7, 16)]
    outputs = [encoder(ids, torch.ones_like(ids, dtype=torch.bool), padded=False) for ids in prompts]
    assert len(builds) == 1
    for ids, out in zip(prompts, outputs, strict=True):
        exact = qwen3_encoder.Qwen3HiddenStateEncoder(**dims, max_positions=ids.shape[1]).eval()
        exact.load_state_dict(encoder.state_dict())
        want = exact(ids, torch.ones_like(ids, dtype=torch.bool), padded=False)
        torch.testing.assert_close(out, want, rtol=0, atol=0)
    with pytest.raises(ValueError, match="prompt tokens"):
        encoder(torch.zeros(1, 17, dtype=torch.long), torch.ones(1, 17, dtype=torch.bool), padded=False)


def _decode(data: bytes) -> torch.Tensor:
    import io

    from PIL import Image

    return torch.from_numpy(np.array(Image.open(io.BytesIO(data)).convert("RGB"))).permute(2, 0, 1)


@pytest.mark.parametrize("kwargs", [{}, {"compress_level": 0}, {"compress_level": 3, "workers": 1},
                                    {"adaptive_filters": True}, {"workers": 3}])
def test_png_writers_round_trip_exactly(kwargs):
    torch.manual_seed(0)
    image = torch.randint(0, 256, (3, 96, 160), dtype=torch.uint8)
    image[:, :20] = 7  # a flat region the deflate blocks share
    data = uint8_to_png(image, **kwargs)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    torch.testing.assert_close(_decode(data), image, rtol=0, atol=0)
    torch.testing.assert_close(_decode(uint8_to_png(image[None], **kwargs)), image, rtol=0, atol=0)


def test_parallel_deflate_is_one_valid_zlib_stream():
    import zlib

    from mstar.model.components.diffusion.image_io import _zlib_parallel

    payload = bytes(range(256)) * 6000  # 1.5 MB, split into several blocks
    for workers in (1, 2, 5):
        stream = _zlib_parallel(payload, 1, workers, min_block=1 << 16)
        assert zlib.decompress(stream) == payload


def test_encode_image_honours_the_openai_output_knobs():
    torch.manual_seed(1)
    image = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    assert encode_image(image, None)[:8] == b"\x89PNG\r\n\x1a\n"
    torch.testing.assert_close(_decode(encode_image(image, {"png_compress_level": 0})), image, rtol=0, atol=0)
    jpeg = encode_image(image, {"output_format": "jpeg", "output_compression": 90})
    assert jpeg[:3] == b"\xff\xd8\xff" and _decode(jpeg).shape == image.shape
    webp = encode_image(image, {"output_format": "webp"})
    assert webp[:4] == b"RIFF" and _decode(webp).shape == image.shape
    with pytest.raises(ValueError, match="output_format"):
        encode_image(image, {"output_format": "gif"})


@pytest.mark.parametrize("shape", [(3, 40, 8), (3, 16, 4, 6)])
def test_batched_euler_step_broadcasts_per_request_dt_over_any_latent_rank(shape):
    """The loop stacks sigmas as [B, 1, 1]; token layouts [B, L, C] and latent layouts [B, C, H, W]
    must both step each row by its own dt (right-aligned broadcasting would pair B with C)."""
    gen = torch.Generator().manual_seed(0)
    sample = torch.randn(*shape, generator=gen)
    velocity = torch.randn(*shape, generator=gen).to(torch.bfloat16)
    sigma = torch.tensor([1.0, 0.75, 0.5]).view(-1, 1, 1)
    sigma_next = torch.tensor([0.75, 0.5, 0.0]).view(-1, 1, 1)
    out = euler_step(sample, velocity, sigma, sigma_next)
    assert out.shape == sample.shape and out.dtype == velocity.dtype
    for i in range(3):
        row = euler_step(sample[i], velocity[i], sigma[i, 0, 0], sigma_next[i, 0, 0])
        torch.testing.assert_close(out[i], row, rtol=0, atol=0)
