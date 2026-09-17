"""Z-Image native components vs the diffusers / transformers references, on CPU with tiny
random configurations. Bit-exact in fp32 and bf16; never widen a bound to pass.

Covers the transformer (with image-token padding to the 32 multiple, mixed caption lengths
and the learned pad tokens), the loader remap (bijection onto the native parameters), the
caption encoder tap (``hidden_states[-2]`` == layer ``N-1``) and the fp32 Euler update with
the negated velocity."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

pytest.importorskip("diffusers", reason="diffusers is the reference oracle")
transformers = pytest.importorskip("transformers")

from diffusers import ZImageTransformer2DModel  # noqa: E402

from mstar.model.components.diffusion.flow_match import euler_step  # noqa: E402
from mstar.model.flux2_klein.config import Qwen3EncoderConfig  # noqa: E402
from mstar.model.flux2_klein.weight_loader import (  # noqa: E402
    _QKV_RULES_LM,
    load_native,
    make_text_encoder,
    remap_text_encoder_key,
    text_encoder_skip,
)
from mstar.model.z_image.components.transformer import (  # noqa: E402
    ZImageDiT,
    ZImageRoPE,
    patchify_image,
    unpatchify_image,
)
from mstar.model.z_image.config import SEQ_MULTIPLE, ZImageTransformerConfig  # noqa: E402
from mstar.model.z_image.submodules import padded_length  # noqa: E402
from mstar.model.z_image.weight_loader import _STACKED_RULES, remap_transformer_key  # noqa: E402

TINY = dict(
    all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4, dim=48, n_layers=2, n_refiner_layers=1, n_heads=3,
    n_kv_heads=3, norm_eps=1e-5, qk_norm=True, cap_feat_dim=20, rope_theta=256.0, t_scale=1000.0,
    axes_dims=[4, 6, 6], axes_lens=[256, 64, 64],
)


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(0)
    ref = ZImageTransformer2DModel(**TINY).eval()
    with torch.no_grad():
        ref.x_pad_token.normal_()
        ref.cap_pad_token.normal_()
    native = ZImageDiT(ZImageTransformerConfig.from_dict(TINY)).eval()
    load_native(native, iter(ref.state_dict().items()), remap_transformer_key, "tiny z-image",
                stacked_params=_STACKED_RULES)
    return ref, native


def _layout(batch, h, w, cap_lens, dtype, gen):
    """The submodule's token layout for a batch at one (grid, padded caption length)."""
    latents = [torch.randn(4, h * 2, w * 2, generator=gen).to(dtype) for _ in range(batch)]
    caps = [torch.randn(n, 20, generator=gen).to(dtype) for n in cap_lens]
    n_img, cap_len = h * w, padded_length(max(cap_lens))
    n_pad = padded_length(n_img) - n_img
    tokens = torch.stack([patchify_image(lat, 2) for lat in latents])
    tokens = torch.cat([tokens, tokens[:, -1:].expand(-1, n_pad, -1)], dim=1)
    img_pad = torch.zeros(batch, n_img + n_pad, dtype=torch.bool)
    img_pad[:, n_img:] = True
    cap_tokens = torch.zeros(batch, cap_len, 20, dtype=dtype)
    cap_pad = torch.ones(batch, cap_len, dtype=torch.bool)
    for i, c in enumerate(caps):
        cap_tokens[i, : len(c)] = c
        cap_pad[i, : len(c)] = False
    rope = ZImageRoPE(256.0, (4, 6, 6), (256, 64, 64))
    cap_ids = torch.zeros(cap_len, 3, dtype=torch.long)
    cap_ids[:, 0] = torch.arange(1, cap_len + 1)
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    img_ids = torch.stack([torch.full((n_img,), cap_len + 1), hh.flatten(), ww.flatten()], dim=-1)
    img_ids = torch.cat([img_ids, torch.zeros(n_pad, 3, dtype=torch.long)])
    return latents, caps, tokens, img_pad, cap_tokens, cap_pad, rope(img_ids), rope(cap_ids)


def test_remap_is_a_bijection(pair):
    ref, native = pair
    native_params = set(dict(native.named_parameters()))
    fused = {".attention.to_q.": ".attn.qkv.", ".attention.to_k.": ".attn.qkv.", ".attention.to_v.": ".attn.qkv.",
             ".feed_forward.w1.": ".ff.w13.", ".feed_forward.w3.": ".ff.w13."}
    mapped = set()
    for key in ref.state_dict():
        name = remap_transformer_key(key)
        for src, dst in fused.items():
            name = name.replace(src, dst)
        assert name in native_params, f"{key} -> {name} is not a native parameter"
        mapped.add(name)
    assert mapped == native_params
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in native.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grid,cap_lens", [((3, 5), [7, 11]), ((4, 8), [5, 5]), ((2, 16), [33, 40])])
def test_transformer_forward_bit_exact(pair, dtype, grid, cap_lens):
    ref, native = pair
    ref, native = ref.to(dtype), native.to(dtype)
    gen = torch.Generator().manual_seed(3)
    h, w = grid
    latents, caps, tokens, img_pad, cap_tokens, cap_pad, img_freqs, cap_freqs = _layout(2, h, w, cap_lens, dtype, gen)
    t = torch.tensor([0.9, 0.3])
    with torch.no_grad():
        expected = ref([lat.unsqueeze(1) for lat in latents], t, caps, return_dict=False)[0]
        velocity = native(tokens, cap_tokens, cap_pad, img_pad, t, img_freqs, cap_freqs)
    assert velocity.shape == (2, padded_length(h * w), 16)
    for i, exp in enumerate(expected):
        out = unpatchify_image(velocity[i], grid, 2, 4)
        assert torch.equal(exp[:, 0], out), f"row {i}: max abs {(exp[:, 0].float() - out.float()).abs().max().item()}"
    ref.to(torch.float32), native.to(torch.float32)


def test_patchify_roundtrip_matches_reference_layout():
    lat = torch.randn(4, 6, 10)
    tokens = patchify_image(lat, 2)
    ref = lat.view(4, 1, 1, 3, 2, 5, 2).permute(1, 3, 5, 2, 4, 6, 0).reshape(15, 16)  # ZImage _patchify_image
    assert torch.equal(tokens, ref)
    assert torch.equal(unpatchify_image(tokens, (3, 5), 2, 4), lat)
    assert padded_length(15) == SEQ_MULTIPLE and padded_length(32) == 32 and padded_length(33) == 64


def test_caption_encoder_tap_is_the_second_to_last_hidden_state():
    torch.manual_seed(0)
    hf_cfg = transformers.Qwen3Config(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=5, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=1024,
        tie_word_embeddings=True, attn_implementation="sdpa",
    )
    ref = transformers.Qwen3ForCausalLM(hf_cfg).eval()
    cfg = Qwen3EncoderConfig.from_dict({**hf_cfg.to_dict(), "model_type": "qwen3"},
                                       hidden_state_layers=(hf_cfg.num_hidden_layers - 1,), max_sequence_length=12)
    native = make_text_encoder(cfg).eval()
    load_native(native, iter(ref.state_dict().items()), remap_text_encoder_key, "tiny qwen3",
                stacked_params=_QKV_RULES_LM, skip=text_encoder_skip(cfg))
    assert len(native.layers) == hf_cfg.num_hidden_layers - 1
    ids = torch.randint(3, 100, (2, 12), generator=torch.Generator().manual_seed(1))
    mask = torch.ones(2, 12, dtype=torch.long)
    ids[1, 9:] = 0
    mask[1, 9:] = 0
    with torch.no_grad():
        expected = ref(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-2]
        ours = native(ids, mask)
    # the Z-Image pipeline keeps only the real tokens' rows
    assert torch.equal(expected[0], ours[0]) and torch.equal(expected[1, :9], ours[1, :9])
    # real-token states are independent of how much padding follows (what lets rows batch)
    with torch.no_grad():
        shorter = native(ids[1:, :9], mask[1:, :9])
    assert torch.equal(shorter[0], ours[1, :9])


def test_euler_step_with_negated_fp32_velocity_matches_reference_scheduler():
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    from mstar.model.components.diffusion.flow_match import FlowMatchConfig, FlowMatchSchedule

    cfg = {"num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": False}
    steps = 8
    ref = FlowMatchEulerDiscreteScheduler(**cfg)
    ref.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=1.15)  # the pipeline passes an ignored mu
    ref.set_begin_index(0)
    ours = FlowMatchSchedule.build(FlowMatchConfig.from_scheduler_config(cfg), steps, 4096)
    assert torch.equal(ref.sigmas, ours.sigmas) and torch.equal(ref.timesteps, ours.timesteps)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 16, 8, 8, generator=g)
    for k, t in enumerate(ref.timesteps):
        model_out = torch.randn(1, 16, 8, 8, generator=g).to(torch.bfloat16)
        expected = ref.step(-model_out.float(), t, x, return_dict=False)[0]
        got = euler_step(x, -model_out.float(), ours.sigmas[k], ours.sigmas[k + 1])
        assert expected.dtype == got.dtype == torch.float32 and torch.equal(expected, got)
        # the reference conditions the transformer on (1000 - t) / 1000 in fp32
        assert torch.equal((1000.0 - ours.timesteps[k]) / 1000.0, (1000 - t) / 1000)
        x = expected
