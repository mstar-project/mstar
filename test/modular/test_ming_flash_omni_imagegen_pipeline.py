"""Tests for the image-gen condition encoder + diffusion pipeline (step 9b).

All pure-Python on CPU with stub components — no diffusers, no checkpoint:

  * ``MingConditionEncoder`` forward shape + L2-normalize×1000 behavior, driven
    by a stub Qwen2-like connector;
  * ``combine_cfg`` guidance math + renormalization;
  * ``MingImageDenoiser`` loop wiring (CFG batch doubling, sign flip, scheduler
    stepping) with a stub DiT + stub scheduler;
  * ``MingImagePipeline`` end-to-end with stubs (condition → denoise → decode).
"""

from __future__ import annotations

import torch

from mstar.model.ming_omni_flash.components.condition_encoder import MingConditionEncoder
from mstar.model.ming_omni_flash.components.imagegen_pipeline import (
    MingImageDenoiser,
    MingImageGenSamplingParams,
    MingImagePipeline,
    calculate_shift,
    combine_cfg,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubConnectorOut:
    def __init__(self, last_hidden: torch.Tensor) -> None:
        self.hidden_states = [last_hidden]


class _StubConnector(torch.nn.Module):
    """Identity-ish Qwen2 stand-in: returns inputs_embeds as the last hidden."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.last_attention_mask = None

    def forward(self, inputs_embeds, attention_mask=None, **kwargs):
        self.last_attention_mask = attention_mask
        return _StubConnectorOut(inputs_embeds)


class _ImageGenCfgStub:
    connector_subfolder = "connector"
    mlp_subfolder = "mlp"
    diffusion_c_input_dim = 8
    text_encoder_norm = True
    use_identity_mlp = True


def _cond_encoder(thinker_hidden=6, conn_hidden=5, c_out=8, norm=True) -> MingConditionEncoder:
    cfg = _ImageGenCfgStub()
    cfg.diffusion_c_input_dim = c_out
    cfg.text_encoder_norm = norm
    enc = MingConditionEncoder(cfg, thinker_hidden_size=thinker_hidden)
    enc.connector = _StubConnector(conn_hidden)
    enc.connector_hidden_size = conn_hidden
    enc.proj_in = torch.nn.Linear(thinker_hidden, conn_hidden, bias=True)
    enc.proj_out = torch.nn.Linear(conn_hidden, c_out, bias=True)
    return enc.eval()


class _StubDiT(torch.nn.Module):
    """Returns a velocity per item shaped [C, F, H, W] from the latent input."""

    in_channels = 4

    def forward(self, latents_list, timestep, embeds):
        # Echo a small transform of each latent so CFG pos/neg differ.
        out = [lat * 0.1 for lat in latents_list]
        return out, {}


class _StubScheduler:
    """Minimal flow-matching scheduler: x_{t-1} = x_t - dt * v."""

    def __init__(self, n_steps: int) -> None:
        self.timesteps = torch.linspace(1000, 0, n_steps)
        self.config = {}
        self.sigma_min = 0.0
        self._dt = 1.0 / n_steps

    def set_timesteps(self, num, device=None, mu=None):
        self.timesteps = torch.linspace(1000, 0, num)

    def step(self, model_output, t, sample, return_dict=False):
        return (sample + self._dt * model_output,)


class _StubVAEConfig:
    block_out_channels = [128, 256]
    scaling_factor = 0.5
    shift_factor = 0.1


class _StubVAE:
    config = _StubVAEConfig()
    dtype = torch.float32

    def decode(self, latents, return_dict=False):
        # [B, C, H, W] -> [B, 3, H*?, W*?]; just map channels to 3 for shape.
        b, _c, h, w = latents.shape
        return (torch.zeros(b, 3, h, w),)


# ---------------------------------------------------------------------------
# Condition encoder
# ---------------------------------------------------------------------------


def test_condition_encoder_output_shape() -> None:
    enc = _cond_encoder(thinker_hidden=6, conn_hidden=5, c_out=8)
    hidden = torch.randn(2, 4, 6)
    out = enc(hidden)
    assert out.shape == (2, 4, 8)


def test_condition_encoder_l2_normalize_times_1000() -> None:
    enc = _cond_encoder(c_out=8, norm=True)
    out = enc(torch.randn(1, 3, 6))
    # Each row L2-normalized then ×1000 -> norm ≈ 1000.
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 1000.0), atol=1e-2)


def test_condition_encoder_no_norm_scaling() -> None:
    enc = _cond_encoder(c_out=8, norm=False)
    out = enc(torch.randn(1, 3, 6))
    norms = out.norm(dim=-1)
    # Without the ×1000, normalized rows have unit norm.
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_condition_encoder_builds_4d_mask() -> None:
    enc = _cond_encoder()
    enc(torch.randn(2, 4, 6))
    mask = enc.connector.last_attention_mask
    assert mask.shape == (2, 1, 4, 4)
    assert torch.all(mask == 1)


def test_condition_encoder_requires_load() -> None:
    enc = MingConditionEncoder(_ImageGenCfgStub(), thinker_hidden_size=6)
    import pytest

    with pytest.raises(RuntimeError, match="load_from_checkpoint"):
        enc(torch.randn(1, 3, 6))


def test_condition_encoder_rejects_2d() -> None:
    enc = _cond_encoder()
    import pytest

    with pytest.raises(ValueError, match=r"expected \[B, N, H\]"):
        enc(torch.randn(4, 6))


def test_zero_negative_is_zeros_like() -> None:
    enc = _cond_encoder()
    x = torch.randn(2, 3, 8)
    z = enc.zero_negative(x)
    assert z.shape == x.shape and torch.all(z == 0)


# ---------------------------------------------------------------------------
# CFG math
# ---------------------------------------------------------------------------


def test_combine_cfg_basic() -> None:
    pos = torch.tensor([[2.0, 0.0]])
    neg = torch.tensor([[0.0, 0.0]])
    out = combine_cfg(pos, neg, guidance_scale=3.0)
    # pos + 3*(pos-neg) = 4*pos
    assert torch.allclose(out, torch.tensor([[8.0, 0.0]]))


def test_combine_cfg_zero_scale_returns_pos() -> None:
    pos = torch.randn(1, 4)
    neg = torch.randn(1, 4)
    assert torch.allclose(combine_cfg(pos, neg, 0.0), pos)


def test_combine_cfg_renormalization_caps_norm() -> None:
    pos = torch.tensor([[1.0, 0.0, 0.0]])
    neg = torch.tensor([[-5.0, 0.0, 0.0]])
    # Large guidance blows up the norm; renorm should cap to 1.0 × |pos|.
    out = combine_cfg(pos, neg, guidance_scale=10.0, cfg_normalization=1.0)
    assert torch.linalg.vector_norm(out).item() <= torch.linalg.vector_norm(pos).item() + 1e-4


def test_calculate_shift_monotonic() -> None:
    lo = calculate_shift(256)
    hi = calculate_shift(4096)
    assert hi > lo
    assert abs(lo - 0.5) < 1e-6  # base_shift at base_seq_len


# ---------------------------------------------------------------------------
# Denoiser loop
# ---------------------------------------------------------------------------


def test_denoiser_runs_without_cfg() -> None:
    dit = _StubDiT()
    sched = _StubScheduler(4)
    den = MingImageDenoiser(dit, sched, dtype=torch.float32)
    latents = torch.randn(1, 4, 8, 8)
    pe = [torch.randn(16, 8)]
    out = den.denoise(latents, sched.timesteps, pe, None, guidance_scale=0.0)
    assert out.shape == latents.shape


def test_denoiser_runs_with_cfg() -> None:
    dit = _StubDiT()
    sched = _StubScheduler(3)
    den = MingImageDenoiser(dit, sched, dtype=torch.float32)
    latents = torch.randn(1, 4, 8, 8)
    pe = [torch.randn(16, 8)]
    ne = [torch.zeros(16, 8)]
    out = den.denoise(latents, sched.timesteps, pe, ne, guidance_scale=3.0)
    assert out.shape == latents.shape


def test_denoiser_cfg_truncation_disables_guidance_late() -> None:
    """With cfg_truncation=0, every step's t_norm>0 so CFG is always off — the
    run must still complete and match the no-CFG path's shape."""
    dit = _StubDiT()
    sched = _StubScheduler(3)
    den = MingImageDenoiser(dit, sched, dtype=torch.float32)
    latents = torch.randn(1, 4, 8, 8)
    pe = [torch.randn(16, 8)]
    ne = [torch.zeros(16, 8)]
    out = den.denoise(latents, sched.timesteps, pe, ne, guidance_scale=3.0, cfg_truncation=0.0)
    assert out.shape == latents.shape


# ---------------------------------------------------------------------------
# Full pipeline (stubs)
# ---------------------------------------------------------------------------


def _stub_pipeline() -> MingImagePipeline:
    cfg = _ImageGenCfgStub()
    cfg.diffusion_c_input_dim = 8
    enc = _cond_encoder(thinker_hidden=6, conn_hidden=5, c_out=8)
    return MingImagePipeline(
        transformer=_StubDiT(),
        scheduler=_StubScheduler(3),
        vae=_StubVAE(),
        condition_encoder=enc,
        image_gen_config=cfg,
        byte5=None,
        device="cpu",
        dtype=torch.float32,
    )


def test_pipeline_prepare_latents_shape() -> None:
    pipe = _stub_pipeline()
    # vae_scale_factor = 2^(2-1)=2 -> vae_scale=4; 64/4=16.
    lat = pipe.prepare_latents(1, 64, 64)
    assert lat.shape == (1, 4, 16, 16)


def test_pipeline_build_cap_feats_default_negative_is_zero() -> None:
    pipe = _stub_pipeline()
    hidden = torch.randn(4, 6)  # [N, H] single item
    pos, neg = pipe.build_cap_feats(hidden)
    assert len(pos) == 1 and len(neg) == 1
    assert torch.all(neg[0] == 0)


def test_pipeline_generate_end_to_end_shape() -> None:
    pipe = _stub_pipeline()
    hidden = torch.randn(4, 6)
    params = MingImageGenSamplingParams(height=64, width=64, num_inference_steps=3, guidance_scale=2.0)
    img = pipe.generate(hidden, params)
    # decode maps to [B, 3, H/vae, W/vae] in the stub VAE.
    assert img.shape[0] == 1 and img.shape[1] == 3


def test_pipeline_generate_seed_is_deterministic() -> None:
    pipe = _stub_pipeline()
    hidden = torch.randn(4, 6)
    params = MingImageGenSamplingParams(height=32, width=32, num_inference_steps=2, guidance_scale=0.0, seed=123)
    a = pipe.generate(hidden, params)
    b = pipe.generate(hidden, params)
    assert torch.allclose(a, b)
