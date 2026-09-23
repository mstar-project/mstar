"""FLUX.2 [klein] native components vs the diffusers / transformers references, on CPU
with tiny random configurations.

Every native module here is an exact port, so the bar is BIT-EXACT — in fp32 and in the
serving dtype (bf16) — on random weights loaded through the same remaps the real
checkpoint uses (which also proves each remap is a bijection). Never widen a bound to
pass: a nonzero difference is a port or loader bug.

The real-weight counterpart (per-stage max-abs on the checkpoint and PSNR against a
recorded pipeline run) lives in ``test_flux2_klein_reference_equivalence.py`` and
needs a GPU.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

pytest.importorskip("diffusers", reason="diffusers is the reference oracle")
transformers = pytest.importorskip("transformers")

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel  # noqa: E402
from diffusers.models.embeddings import apply_rotary_emb  # noqa: E402
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed  # noqa: E402
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline  # noqa: E402

from mstar.model.components.diffusion.flow_match import (  # noqa: E402
    FlowMatchConfig,
    FlowMatchSchedule,
    ShiftMode,
    compute_empirical_mu,
    euler_step,
)
from mstar.model.components.diffusion.image_io import (  # noqa: E402
    image_grid_ids,
    pack_latents,
    patchify_latents,
    text_ids,
    unpack_latents,
    unpatchify_latents,
)
from mstar.model.components.diffusion.rope import MultiAxisRoPE, apply_rotary_interleaved  # noqa: E402
from mstar.model.flux2_klein.components.transformer import Flux2DiT  # noqa: E402
from mstar.model.flux2_klein.components.vae import Flux2VAE  # noqa: E402
from mstar.model.flux2_klein.config import Flux2TransformerConfig, Flux2VaeConfig, Qwen3EncoderConfig  # noqa: E402
from mstar.model.flux2_klein.weight_loader import (  # noqa: E402
    _QKV_RULES_DIT,
    _QKV_RULES_LM,
    load_native,
    make_text_encoder,
    remap_text_encoder_key,
    remap_transformer_key,
    remap_vae_key,
    text_encoder_skip,
)

KLEIN_SCHEDULER = dict(
    base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=4096, max_shift=1.15,
    num_train_timesteps=1000, shift=3.0, shift_terminal=None, stochastic_sampling=False,
    time_shift_type="exponential", use_beta_sigmas=False, use_dynamic_shifting=True,
    use_exponential_sigmas=False, use_karras_sigmas=False,
)
ZIMAGE_SCHEDULER = dict(num_train_timesteps=1000, use_dynamic_shifting=False, shift=3.0)

TINY_TRANSFORMER = dict(
    patch_size=1, in_channels=16, num_layers=2, num_single_layers=3, attention_head_dim=8, num_attention_heads=4,
    joint_attention_dim=24, timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(2, 2, 2, 2),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)
TINY_VAE = dict(
    in_channels=3, out_channels=3, block_out_channels=(8, 16, 16), layers_per_block=1, latent_channels=4,
    norm_num_groups=4, down_block_types=("DownEncoderBlock2D",) * 3, up_block_types=("UpDecoderBlock2D",) * 3,
    batch_norm_eps=1e-4, patch_size=(2, 2),
)


# ---------------------------------------------------------------------------
# schedule + Euler step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheduler_cfg,empirical,mode", [
    (KLEIN_SCHEDULER, True, ShiftMode.EMPIRICAL_MU),
    (ZIMAGE_SCHEDULER, False, ShiftMode.LINEAR),
])
@pytest.mark.parametrize("num_steps", [1, 4, 8, 50])
@pytest.mark.parametrize("image_seq_len", [256, 1024, 4096, 6144])
def test_flow_match_schedule_bit_exact(scheduler_cfg, empirical, mode, num_steps, image_seq_len):
    config = FlowMatchConfig.from_scheduler_config(scheduler_cfg, empirical_mu=empirical)
    assert config.shift_mode is mode
    ref = FlowMatchEulerDiscreteScheduler(**scheduler_cfg)
    mu = compute_empirical_mu(image_seq_len, num_steps) if empirical else None
    ref.set_timesteps(sigmas=np.linspace(1.0, 1 / num_steps, num_steps), mu=mu)
    ours = FlowMatchSchedule.build(config, num_steps, image_seq_len)
    assert ours.num_steps == num_steps
    assert torch.equal(ref.sigmas, ours.sigmas)
    assert torch.equal(ref.timesteps, ours.timesteps)


def test_flow_match_schedule_rejects_unsupported_scheduler_options():
    with pytest.raises(NotImplementedError):
        FlowMatchConfig.from_scheduler_config({**ZIMAGE_SCHEDULER, "use_karras_sigmas": True})
    with pytest.raises(NotImplementedError):
        FlowMatchConfig.from_scheduler_config({**ZIMAGE_SCHEDULER, "shift_terminal": 0.1})
    with pytest.raises(ValueError):
        FlowMatchSchedule.build(FlowMatchConfig.from_scheduler_config(ZIMAGE_SCHEDULER), 0, 4096)


def test_euler_step_matches_reference_scalar_and_batched():
    num_steps = 4
    ref = FlowMatchEulerDiscreteScheduler(**KLEIN_SCHEDULER)
    ref.set_timesteps(sigmas=np.linspace(1.0, 1 / num_steps, num_steps), mu=compute_empirical_mu(4096, num_steps))
    ref.set_begin_index(0)
    ours = FlowMatchSchedule.build(FlowMatchConfig.from_scheduler_config(KLEIN_SCHEDULER, True), num_steps, 4096)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 64, 16, generator=g, dtype=torch.bfloat16)
    for k, t in enumerate(ref.timesteps):
        v = torch.randn(2, 64, 16, generator=g, dtype=torch.bfloat16)
        expected = ref.step(v, t, x, return_dict=False)[0]
        assert expected.dtype == torch.bfloat16
        scalar = euler_step(x, v, ours.sigmas[k], ours.sigmas[k + 1])
        batched = euler_step(
            x, v, ours.sigmas[k].expand(2).view(2, 1, 1), ours.sigmas[k + 1].expand(2).view(2, 1, 1),
        )
        assert torch.equal(expected, scalar)
        assert torch.equal(expected, batched)
        x = expected


# ---------------------------------------------------------------------------
# rotary tables + latent layout helpers
# ---------------------------------------------------------------------------


def test_rope_tables_and_application_bit_exact():
    ids = torch.cat([text_ids(37), image_grid_ids(6, 9), image_grid_ids(4, 5, t=10), image_grid_ids(3, 3, t=20)])
    ref_cos, ref_sin = Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32])(ids)
    cos, sin = MultiAxisRoPE(2000, (32, 32, 32, 32))(ids)
    assert cos.dtype == torch.float32 and cos.shape == (ids.shape[0], 128)
    assert torch.equal(ref_cos, cos) and torch.equal(ref_sin, sin)
    x = torch.randn(2, ids.shape[0], 3, 128).to(torch.bfloat16)
    assert torch.equal(apply_rotary_emb(x, (ref_cos, ref_sin), sequence_dim=1), apply_rotary_interleaved(x, cos, sin))


def test_position_ids_match_pipeline_layout():
    t, one = torch.arange(1), torch.arange(1)
    assert torch.equal(torch.cartesian_prod(t, torch.arange(6), torch.arange(9), one), image_grid_ids(6, 9))
    assert torch.equal(torch.cartesian_prod(t, one, one, torch.arange(512)), text_ids(512))
    assert torch.equal(
        torch.cartesian_prod(torch.tensor([20]), torch.arange(4), torch.arange(5), one), image_grid_ids(4, 5, t=20),
    )


def test_latent_pack_unpack_matches_pipeline():
    latents = torch.randn(2, 32, 64, 48)
    patched = patchify_latents(latents)
    assert patched.shape == (2, 128, 32, 24)
    assert torch.equal(Flux2KleinPipeline._patchify_latents(latents), patched)
    assert torch.equal(unpatchify_latents(patched), latents)
    tokens = pack_latents(patched)
    assert torch.equal(Flux2KleinPipeline._pack_latents(patched), tokens)
    ids = image_grid_ids(32, 24).unsqueeze(0).expand(2, -1, -1)
    assert torch.equal(Flux2KleinPipeline._unpack_latents_with_ids(tokens, ids, 32, 24), unpack_latents(tokens, 32, 24))


# ---------------------------------------------------------------------------
# transformer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_transformer_pair():
    torch.manual_seed(0)
    ref = Flux2Transformer2DModel(**TINY_TRANSFORMER).eval()
    native = Flux2DiT(Flux2TransformerConfig.from_dict(TINY_TRANSFORMER)).eval()
    load_native(native, iter(ref.state_dict().items()), remap_transformer_key, "tiny transformer",
                stacked_params=_QKV_RULES_DIT)
    return ref, native


def _transformer_inputs(dtype, with_refs: bool):
    g = torch.Generator().manual_seed(1)
    batch, text_len, h, w = 2, 6, 4, 5
    img = torch.randn(batch, h * w, 16, generator=g).to(dtype)
    txt = torch.randn(batch, text_len, 24, generator=g).to(dtype)
    timestep = torch.tensor([0.75, 0.25]).to(dtype)
    ids = [text_ids(text_len), image_grid_ids(h, w)]
    img_ids = image_grid_ids(h, w)
    if with_refs:
        ref_tokens = torch.randn(batch, 9, 16, generator=g).to(dtype)
        img = torch.cat([img, ref_tokens], dim=1)
        ref_ids = image_grid_ids(3, 3, t=10)
        img_ids = torch.cat([img_ids, ref_ids])
        ids.append(ref_ids)
    rope = MultiAxisRoPE(2000, (2, 2, 2, 2))(torch.cat(ids))
    txt_ids = text_ids(text_len).unsqueeze(0).expand(batch, -1, -1)
    return img, txt, timestep, img_ids.unsqueeze(0).expand(batch, -1, -1), txt_ids, rope


def test_transformer_remap_is_a_bijection(tiny_transformer_pair):
    ref, native = tiny_transformer_pair
    ref_keys = list(ref.state_dict().keys())
    native_params = set(dict(native.named_parameters()).keys())
    fused = {
        ".attn.to_q.": ".attn.img_qkv.", ".attn.to_k.": ".attn.img_qkv.", ".attn.to_v.": ".attn.img_qkv.",
        ".attn.add_q_proj.": ".attn.txt_qkv.", ".attn.add_k_proj.": ".attn.txt_qkv.",
        ".attn.add_v_proj.": ".attn.txt_qkv.",
    }
    mapped = set()
    for key in ref_keys:
        name = remap_transformer_key(key)
        for src, dst in fused.items():
            name = name.replace(src, dst)
        assert name in native_params, f"{key} -> {name} is not a native parameter"
        mapped.add(name)
    assert mapped == native_params
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in native.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_refs", [False, True])
def test_transformer_forward_bit_exact(tiny_transformer_pair, dtype, with_refs):
    ref, native = tiny_transformer_pair
    ref, native = ref.to(dtype), native.to(dtype)
    img, txt, timestep, img_ids, txt_ids, rope = _transformer_inputs(dtype, with_refs)
    with torch.no_grad():
        expected = ref(hidden_states=img, encoder_hidden_states=txt, timestep=timestep, img_ids=img_ids,
                       txt_ids=txt_ids, guidance=None, return_dict=False)[0]
        out = native(img, txt, timestep, rope)
    assert out.dtype == dtype
    assert torch.equal(expected, out), f"max abs {(expected.float() - out.float()).abs().max().item()}"
    ref.to(torch.float32), native.to(torch.float32)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_vae_pair():
    torch.manual_seed(0)
    ref = AutoencoderKLFlux2(**TINY_VAE).eval()
    with torch.no_grad():
        ref.bn.running_mean.normal_()
        ref.bn.running_var.uniform_(0.5, 2.0)
    native = Flux2VAE(Flux2VaeConfig.from_dict(TINY_VAE)).eval()
    load_native(native, iter(ref.state_dict().items()), remap_vae_key, "tiny vae")
    return ref, native


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vae_encode_normalize_decode_bit_exact(tiny_vae_pair, dtype):
    ref, native = tiny_vae_pair
    ref, native = ref.to(dtype), native.to(dtype)
    x = torch.randn(2, 3, 32, 48, generator=torch.Generator().manual_seed(2)).to(dtype)
    with torch.no_grad():
        expected_latent = ref.encode(x).latent_dist.mode()
        latent = native.encode(x)
        assert torch.equal(expected_latent, latent)
        patched = patchify_latents(latent)
        mean = ref.bn.running_mean.view(1, -1, 1, 1).to(dtype)
        std = torch.sqrt(ref.bn.running_var.view(1, -1, 1, 1) + ref.config.batch_norm_eps).to(dtype)
        assert torch.equal((patched - mean) / std, native.normalize_latents(patched))
        assert torch.equal(patched * std + mean, native.denormalize_latents(patched))
        assert torch.equal(ref.decode(latent, return_dict=False)[0], native.decode(latent))
    ref.to(torch.float32), native.to(torch.float32)


# ---------------------------------------------------------------------------
# Qwen3 hidden-state text encoder
# ---------------------------------------------------------------------------


def test_text_encoder_tapped_hidden_states_bit_exact_with_padding():
    torch.manual_seed(0)
    hf_cfg = transformers.Qwen3Config(
        vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=1024,
        tie_word_embeddings=False, attn_implementation="sdpa",
    )
    ref = transformers.Qwen3ForCausalLM(hf_cfg).eval()
    taps = (2, 4, 5)
    text_cfg = Qwen3EncoderConfig.from_dict({**hf_cfg.to_dict(), "model_type": "qwen3"}, hidden_state_layers=taps,
                                            max_sequence_length=12)
    native = make_text_encoder(text_cfg).eval()
    load_native(native, iter(ref.state_dict().items()), remap_text_encoder_key, "tiny qwen3",
                stacked_params=_QKV_RULES_LM, skip=text_encoder_skip(text_cfg))
    # only the layers up to the deepest tap exist (klein: 27 of 36)
    assert len(native.layers) == max(taps) < hf_cfg.num_hidden_layers

    seq = 12
    ids = torch.randint(3, 100, (2, seq), generator=torch.Generator().manual_seed(3))
    mask = torch.ones(2, seq, dtype=torch.long)
    ids[0, 7:] = 0
    mask[0, 7:] = 0  # right padding on row 0
    with torch.no_grad():
        out = ref(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        expected = torch.stack([out.hidden_states[k] for k in taps], dim=1).permute(0, 2, 1, 3).reshape(2, seq, -1)
        ours = native(ids, mask)
    assert ours.shape == (2, seq, len(taps) * hf_cfg.hidden_size)
    assert torch.equal(expected, ours)
    # the reference feeds the pad positions' states to the DiT, so they must match too
    assert expected[0, 7:].abs().max() > 0


def test_vae_decoder_node_stacks_per_request_latents(tiny_vae_pair):
    """The dit emits ``[L, 128]`` per request; the decoder node stacks rows at one grid."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.model.components.diffusion.denoise_loop import LATENTS
    from mstar.model.flux2_klein.config import Flux2KleinConfig
    from mstar.model.flux2_klein.submodules import IMAGE_OUTPUT, KleinVaeDecoderSubmodule

    _, native = tiny_vae_pair
    config = Flux2KleinConfig(vae=Flux2VaeConfig.from_dict(TINY_VAE))
    node = KleinVaeDecoderSubmodule(native.to(torch.float32), config)
    height, width = 4 * config.spatial_alignment, 6 * config.spatial_alignment
    grid = config.latent_grid(height, width)
    assert grid == (4, 6)
    gen = torch.Generator().manual_seed(5)
    rows = []
    for i in range(2):
        tokens = torch.randn(grid[0] * grid[1], config.vae.patched_latent_channels, generator=gen)
        info = CurrentForwardPassInfo(
            request_id=f"r{i}", graph_walk="image_gen", fwd_index=0, random_seed=0, max_tokens=1,
            step_metadata={"height": height, "width": width},
        )
        rows.append(node.prepare_inputs("image_gen", info, {LATENTS: [tokens]}))
    with torch.no_grad():
        batched = node.forward("image_gen", None, **node.preprocess("image_gen", None, rows))[IMAGE_OUTPUT][0]
        singles = [
            node.forward("image_gen", None, **node.preprocess("image_gen", None, [row]))[IMAGE_OUTPUT][0]
            for row in rows
        ]
    assert batched.dtype == torch.uint8 and batched.shape == (2, 3, height, width)
    for i, single in enumerate(singles):
        assert single.shape == (1, 3, height, width)
        assert torch.equal(single[0], batched[i])
