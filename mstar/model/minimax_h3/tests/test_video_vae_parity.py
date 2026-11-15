"""The native MiniMax-H3 video decoder against diffusers' ``AutoencoderKLMiniMaxH3``, tiny and on CPU.

A random decoder in the original checkpoint's layout (per-head interleaved ``to_qkv``,
``w1 = [gate; value]``, ``x_embedder``) loads through this port's loader, and the same
weights, re-laid-out the way the official diffusers ``vae/`` stores them, into
diffusers' model. The clip walk (chunks of five latent frames overlapping by two, the
temporal cross-fade) and the spatial tiles (blended overlaps) must agree.
"""

from __future__ import annotations

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLMiniMaxH3"):
    pytest.skip("this diffusers has no MiniMax-H3", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from mstar.model.components.diffusion.attention import sdpa_attention  # noqa: E402
from mstar.model.minimax_h3 import weight_loader  # noqa: E402
from mstar.model.minimax_h3.config import H3Config, H3VideoVAEConfig  # noqa: E402

HEADS, HEAD_DIM, LAYERS, LATENT = 2, 32, 2, 4
DIM = HEADS * HEAD_DIM
VAE = H3VideoVAEConfig(latent_channels=LATENT, decoder_layers=LAYERS, decoder_heads=HEADS, decoder_head_dim=HEAD_DIM)


def _random_decoder() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def t(*shape, scale=0.2):
        return torch.randn(*shape, generator=g) * scale

    state = {
        "decoder.x_embedder.weight": t(DIM, LATENT), "decoder.x_embedder.bias": t(DIM),
        "decoder.register_tokens": t(1, 4, DIM), "decoder.mask_token": torch.zeros(1, 1, DIM),
        "decoder.norm_out.weight": 1 + t(DIM), "decoder.norm_out.bias": t(DIM),
        "decoder.proj_out.weight": t(3 * 4 * 16 * 16, DIM), "decoder.proj_out.bias": t(3 * 4 * 16 * 16),
        "post_quant_conv.weight": t(LATENT, LATENT, 1, 1, 1), "post_quant_conv.bias": t(LATENT),
    }
    for i in range(LAYERS):
        p = f"decoder.transformer_blocks.{i}"
        state.update({
            f"{p}.norm1.weight": 1 + t(DIM), f"{p}.norm2.weight": 1 + t(DIM),
            f"{p}.scale1": 1 + t(DIM), f"{p}.scale2": 1 + t(DIM),
            f"{p}.attn.to_qkv.weight": t(3 * DIM, DIM), f"{p}.attn.to_qkv.bias": t(3 * DIM),
            f"{p}.attn.to_out.weight": t(DIM, DIM), f"{p}.attn.to_out.bias": t(DIM),
            f"{p}.ff.w1.weight": t(8 * DIM, DIM), f"{p}.ff.w1.bias": t(8 * DIM),
            f"{p}.ff.w2.weight": t(DIM, 4 * DIM), f"{p}.ff.w2.bias": t(DIM),
        })
    return state


def _to_diffusers(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key == "decoder.mask_token":
            continue
        key = key.replace("decoder.x_embedder.", "decoder.proj_in.")
        if ".attn.to_qkv." in key:
            parts = value.unflatten(0, (HEADS, 3, HEAD_DIM)).unbind(1)
            for name, part in zip(("to_q", "to_k", "to_v"), parts, strict=True):
                out[key.replace("to_qkv", name)] = part.flatten(0, 1)
            continue
        if ".ff.w1." in key:
            gate, value_rows = value.chunk(2, dim=0)
            out[key.replace("ff.w1", "ff.net.0.proj")] = torch.cat([value_rows, gate], dim=0)
            continue
        key = key.replace(".ff.w2.", ".ff.net.2.").replace(".attn.to_out.", ".attn.to_out.0.")
        out[key] = value
    return out


@pytest.fixture(scope="module")
def decoders(tmp_path_factory):
    root = tmp_path_factory.mktemp("h3vae")
    state = _random_decoder()
    (root / "FL2VA" / "video_vae" / "source").mkdir(parents=True)
    save_file(state, str(root / "FL2VA" / "video_vae" / "source" / "model.safetensors"))
    ours = weight_loader.build_video_decoder(H3Config(video_vae=VAE), root, "cpu", dtype=torch.float32)
    reference = diffusers.AutoencoderKLMiniMaxH3(
        latent_channels=LATENT, block_out_channels=(8, 8, 8, 8, 8, 16), norm_num_groups=4,
        decoder_num_layers=LAYERS, decoder_num_attention_heads=HEADS, decoder_attention_head_dim=HEAD_DIM,
    ).eval()
    missing, unexpected = reference.load_state_dict(_to_diffusers(state), strict=False)
    assert not unexpected and all(m.startswith(("encoder.", "quant_conv.")) for m in missing), (missing, unexpected)
    return ours, reference


@pytest.mark.parametrize("shape", [(7, 4, 6), (12, 20, 18)])
def test_decode_matches_diffusers(decoders, shape):
    """One clip, untiled; and three clips (with a padded tail) over 2 x 2 blended tiles."""
    ours, reference = decoders
    frames, height, width = shape
    z = torch.randn(1, LATENT, frames, height, width, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = reference.decode(z, return_dict=False)[0]
        got = ours.decode(z, sdpa_attention)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
