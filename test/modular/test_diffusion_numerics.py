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

from mstar.model.components.diffusion.flow_match import FlowMatchConfig, FlowMatchSchedule  # noqa: E402
from mstar.model.components.diffusion.image_io import encode_image, pixels_to_uint8, uint8_to_png  # noqa: E402
from mstar.model.components.diffusion.text_encoder import Qwen3RMSNorm, qwen3_rotary_tables  # noqa: E402


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
