"""``exclude_from_compile`` / ``compile_exact_ops``: norms and activations stay on the eager kernels
inside a compiled transformer, the excluded modules compute exactly what they did before, and the
knob reaches both denoise nodes."""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.flux2_klein.components.transformer import Flux2DiT
from mstar.model.flux2_klein.config import Flux2TransformerConfig
from mstar.model.flux2_klein.submodules import EXACT_OP_TYPES, exclude_from_compile
from mstar.model.z_image.components.transformer import ScaledRMSNorm

TINY = Flux2TransformerConfig(
    in_channels=8, out_channels=8, num_layers=1, num_single_layers=2, attention_head_dim=8,
    num_attention_heads=2, joint_attention_dim=16, timestep_guidance_channels=16, axes_dims_rope=(2, 2, 2, 2),
)


def _excluded(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, EXACT_OP_TYPES) or getattr(m, "compile_exact_op", False)]


def test_exclude_marks_every_norm_and_activation_once():
    torch.manual_seed(0)
    dit = Flux2DiT(TINY)
    expected = _excluded(dit)
    # 1 double block: 4 LayerNorms + 4 q/k RMSNorms + FF SiLU x2; 2 single blocks: LayerNorm + 2 RMSNorms + SiLU
    # each; DiT: norm_out, mod_act SiLU, time embedder SiLU
    assert len(expected) == 4 + 4 + 2 + 2 * 4 + 3
    assert exclude_from_compile(dit) == len(expected)
    for m in expected:
        assert getattr(m.forward, "_torchdynamo_disable", False), type(m).__name__
    untouched = [m for m in dit.modules() if m not in expected and not isinstance(m, (Flux2DiT, nn.ModuleList))]
    assert untouched and all(not getattr(m.forward, "_torchdynamo_disable", False) for m in untouched)


def test_excluded_modules_compute_the_same_values():
    torch.manual_seed(0)
    dit = Flux2DiT(TINY)
    x = torch.randn(2, 5, TINY.hidden_size)
    before = {id(m): m(x if isinstance(m, (nn.LayerNorm, nn.SiLU)) else x[..., :TINY.attention_head_dim])
              for m in _excluded(dit)}
    exclude_from_compile(dit)
    for m in _excluded(dit):
        after = m(x if isinstance(m, (nn.LayerNorm, nn.SiLU)) else x[..., :TINY.attention_head_dim])
        assert torch.equal(after, before[id(m)])


def test_vae_norms_and_activations_are_excluded():
    from mstar.model.components.diffusion.autoencoder_kl import ResnetBlock

    block = ResnetBlock(8, 8, groups=4, eps=1e-6)
    assert exclude_from_compile(block) == 3  # two GroupNorms + the shared SiLU module
    x = torch.randn(1, 8, 4, 4)
    assert torch.equal(block(x), block.conv2(block.act(block.norm2(block.conv1(block.act(block.norm1(x)))))) + x)


def test_scaled_rmsnorm_opts_in_by_attribute():
    norm = ScaledRMSNorm(6, 1e-5)
    assert ScaledRMSNorm.compile_exact_op is True
    assert exclude_from_compile(nn.Sequential(norm, nn.Linear(6, 6))) == 1
    assert getattr(norm.forward, "_torchdynamo_disable", False)


def test_knob_reaches_both_models():
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
    from mstar.model.z_image.z_image_model import ZImageModel

    klein = Flux2KleinModel(model_path_hf="x", compile=True, compile_exact_ops=True, cuda_graph=False)
    z = ZImageModel(model_path_hf="x", compile=True, compile_exact_ops=True, cuda_graph=False)
    assert klein.compile_exact_ops is True and z.compile_exact_ops is True
    assert Flux2KleinModel(model_path_hf="x").compile_exact_ops is False
    assert ZImageModel(model_path_hf="x").compile_exact_ops is False
