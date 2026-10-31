"""``exclude_from_compile`` / ``compile_exact_ops``: norms and activations stay on the eager result
inside a compiled transformer, the kept modules compute exactly what they did before, and the
knob reaches both denoise nodes."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mstar.model.components.diffusion.compile_utils import (
    EXACT_OP_TYPES,
    exact_op_types,
    exclude_from_compile,
    kept_exact,
)
from mstar.model.components.diffusion.exact_norm import _reproduces_torch, exact_rms_norm
from mstar.model.flux2_klein.components.transformer import Flux2DiT
from mstar.model.flux2_klein.config import Flux2TransformerConfig
from mstar.model.z_image.components.transformer import ScaledRMSNorm
from mstar.platforms import current_platform

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
        assert kept_exact(m), type(m).__name__
    untouched = [m for m in dit.modules() if m not in expected and not isinstance(m, (Flux2DiT, nn.ModuleList))]
    assert untouched and not any(kept_exact(m) for m in untouched)


def test_a_kept_torch_rms_norm_compiles_without_a_graph_break():
    """klein's q/k norms under [norms]: torch's RMSNorm runs an op the compiled graph keeps whole."""
    model = nn.Sequential(nn.Linear(128, 128), nn.RMSNorm(128, eps=1e-6), nn.Linear(128, 128))
    assert exclude_from_compile(model, exact_op_types(["norms"])) == 1
    x = torch.randn(2, 5, 128)
    with torch.no_grad():
        assert torch.equal(torch.compile(model, backend="eager", fullgraph=True)(x), model(x))


def test_blocks_sharing_a_compiled_frame_compile_it_once():
    """klein's blocks run one compiled block frame, each block with its own kept norms: a kept
    module's forward is its class's, so the frame's guards hold for every block. A forward set per
    instance failed them once per block, until dynamo's recompile limit sent the rest eager."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(128, 128)
            self.norm = nn.LayerNorm(128)  # a graph break, as klein's modulation norms are
            self.q_norm = nn.RMSNorm(128, eps=1e-6)
            self.k_norm = nn.RMSNorm(128, eps=1e-6)

        def forward(self, x):
            h = self.norm(self.proj(x))
            return self.q_norm(h) + self.k_norm(h)

    torch.manual_seed(0)
    blocks = [Block() for _ in range(3)]
    for block in blocks:
        assert exclude_from_compile(block, exact_op_types(["norms"])) == 3
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    x = torch.randn(2, 5, 128)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(Block.forward, backend=backend, dynamic=False)
        with torch.no_grad():
            assert torch.equal(compiled(blocks[0], x), blocks[0](x))
            first = len(graphs)
            for block in blocks[1:]:
                assert torch.equal(compiled(block, x), block(x))
    finally:
        torch._dynamo.reset()
    assert first and len(graphs) == first


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


def test_folded_upsample_equals_nearest_upsample_then_conv():
    from mstar.model.components.diffusion.autoencoder_kl import UpStage

    torch.manual_seed(0)
    stage = UpStage(8, 8, num_layers=1, groups=4, eps=1e-6, upsample=True).double()
    x = torch.randn(2, 8, 5, 7, dtype=torch.float64)
    expected = stage(x)  # eager: nearest x2 then the 3x3 conv
    stage.fold_upsample()
    folded = torch.compile(stage, backend="eager", fullgraph=True)(x)
    assert folded.shape == expected.shape == (2, 8, 10, 14)
    torch.testing.assert_close(folded, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape, layout", [
    ((2, 128, 256, 256), "nchw"),
    ((2, 128, 256, 256), "channels_last"),
    ((2, 128, 256, 256), "channel_slice"),  # not dense: F.group_norm, not the kernel
    ((1, 32, 2048, 2048), "nchw"),  # 65,536 pixel tiles, more than a grid's y axis holds
])
def test_split_group_norm_silu_matches_fp32_group_norm(shape, layout):
    from mstar.model.components.diffusion.group_norm import group_norm_silu

    torch.manual_seed(0)
    x = (torch.randn(*shape, device="cuda") * 3 + 1).to(torch.bfloat16)
    if layout == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
    elif layout == "channel_slice":
        x = torch.cat([x, x], dim=1)[:, : shape[1]]
    w, b = (torch.randn(shape[1], device="cuda").to(torch.bfloat16) for _ in range(2))
    ref = torch.nn.functional.silu(torch.nn.functional.group_norm(x.float(), 32, w.float(), b.float(), 1e-6))
    got = group_norm_silu(x, w, b, 32, 1e-6, True)
    assert layout == "channel_slice" or got.stride() == x.stride()
    torch.testing.assert_close(got.float(), ref, rtol=2 ** -7, atol=2 ** -6)  # bf16 output rounding


def test_norm_act_fuses_only_silu(monkeypatch):
    """The fused kernel applies SiLU or nothing; any other activation, or a module kept exact,
    runs as its module."""
    from mstar.model.components.diffusion import group_norm

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(group_norm, "_fits", lambda *a: True)
    monkeypatch.setattr(group_norm, "group_norm_silu", lambda x, w, b, g, eps, silu: torch.full_like(x, float(silu)))
    norm, act, x = nn.GroupNorm(2, 4), nn.SiLU(), torch.randn(1, 4, 3, 3)
    assert torch.equal(group_norm.norm_act(norm, x, act), torch.ones_like(x))
    torch.testing.assert_close(group_norm.norm_act(norm, x, nn.GELU()), nn.functional.gelu(norm(x)))
    exclude_from_compile(nn.Sequential(norm, act))
    torch.testing.assert_close(group_norm.norm_act(norm, x, act), nn.functional.silu(norm(x)))


def test_scaled_rmsnorm_opts_in_by_attribute():
    norm = ScaledRMSNorm(6, 1e-5)
    assert ScaledRMSNorm.compile_exact_op is True
    assert exclude_from_compile(nn.Sequential(norm, nn.Linear(6, 6))) == 1
    assert getattr(norm.forward, "_torchdynamo_disable", False)


def test_exact_op_classes_can_be_chosen():
    assert exact_op_types(True) == EXACT_OP_TYPES and exact_op_types(False) is None and exact_op_types([]) is None
    assert exact_op_types(["activations"]) == (nn.SiLU,)
    assert set(exact_op_types(["norms"])) == {nn.LayerNorm, nn.RMSNorm, nn.GroupNorm}
    with pytest.raises(ValueError, match="unknown op class"):
        exact_op_types(["gelu"])
    dit = Flux2DiT(TINY)
    silus = sum(isinstance(m, nn.SiLU) for m in dit.modules())
    assert exclude_from_compile(dit, exact_op_types(["activations"])) == silus
    norm_only = Flux2DiT(TINY)
    assert exclude_from_compile(norm_only, exact_op_types(["norms"])) == len(_excluded(norm_only)) - silus
    # a marked norm (ScaledRMSNorm) counts as a norm, not as an activation
    assert exclude_from_compile(nn.Sequential(ScaledRMSNorm(4, 1e-5)), exact_op_types(["activations"])) == 0
    assert exclude_from_compile(nn.Sequential(ScaledRMSNorm(4, 1e-5)), exact_op_types(["norms"])) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("scale", [0.01, 1.0, 30.0])
def test_exact_rms_norm_matches_torch_bit_for_bit(scale):
    torch.manual_seed(0)
    # a klein q view: a strided chunk of the fused qkv projection, [B, L, H, 128]
    qkv = (torch.randn(2, 777, 3 * 6 * 128, device="cuda") * scale).to(torch.bfloat16)
    q = qkv.chunk(3, dim=-1)[1].unflatten(-1, (-1, 128))
    w = (1 + 0.3 * torch.randn(128, device="cuda")).to(torch.bfloat16)
    # on NVIDIA the replica must reproduce torch, else exact_rms_norm runs torch's kernel itself;
    # elsewhere it always does
    if current_platform.is_cuda():
        assert _reproduces_torch(q.device, q.dtype)
    for x in (q, q.contiguous(), q[0], q.reshape(-1, 128)[:999]):
        assert torch.equal(exact_rms_norm(x, w, 1e-6), torch.nn.functional.rms_norm(x, (128,), w, 1e-6))
    assert torch.equal(exact_rms_norm(q, w, None), torch.nn.functional.rms_norm(q, (128,), w, None))


def test_knob_reaches_both_models():
    from mstar.model.flux2_klein.flux2_klein_model import Flux2KleinModel
    from mstar.model.z_image.z_image_model import ZImageModel

    klein = Flux2KleinModel(model_path_hf="x", compile=True, compile_exact_ops=True, accelerator_graph=False)
    z = ZImageModel(model_path_hf="x", compile=True, compile_exact_ops=True, accelerator_graph=False)
    assert klein.compile_exact_ops is True and z.compile_exact_ops is True
    assert Flux2KleinModel(model_path_hf="x", compile_exact_ops=["norms"]).compile_exact_ops == ["norms"]
    assert Flux2KleinModel(model_path_hf="x").compile_exact_ops is False
    assert ZImageModel(model_path_hf="x").compile_exact_ops is False
