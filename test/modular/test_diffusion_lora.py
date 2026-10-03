"""Static LoRA merge on CPU with a tiny FLUX.2 klein transformer.

A random adapter in diffusers naming (and the same adapter in the BFL layout with fused qkv
matrices) is folded into the native transformer; the reference is the diffusers module with
the identical ``W += scale * alpha / r * B @ A`` surgery applied to its own weights, so the
forward passes must agree bit for bit. Also checks alpha handling, prefixes, the
Kohya-style ``lora_down`` / ``lora_up`` names and the error paths."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

pytest.importorskip("diffusers")
from diffusers import Flux2Transformer2DModel  # noqa: E402

from mstar.model.components.diffusion.image_io import image_grid_ids, text_ids  # noqa: E402
from mstar.model.components.diffusion.lora import LoraSpec, merge_lora, normalize_lora_state_dict  # noqa: E402
from mstar.model.components.diffusion.rope import MultiAxisRoPE  # noqa: E402
from mstar.model.flux2_klein.components.transformer import Flux2DiT  # noqa: E402
from mstar.model.flux2_klein.config import Flux2TransformerConfig  # noqa: E402
from mstar.model.flux2_klein.weight_loader import (  # noqa: E402
    _QKV_RULES_DIT,
    convert_bfl_lora_keys,
    load_native,
    remap_transformer_key,
)

TINY = dict(
    patch_size=1, in_channels=16, num_layers=2, num_single_layers=2, attention_head_dim=8, num_attention_heads=4,
    joint_attention_dim=24, timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(2, 2, 2, 2),
    rope_theta=2000, eps=1e-6, guidance_embeds=False,
)
RANK = 4
# diffusers module paths the adapter touches (a fused-shard target, plain linears, the single-block fused GEMM)
MODULES = [
    "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_v", "transformer_blocks.0.attn.add_k_proj",
    "transformer_blocks.1.attn.to_out.0", "transformer_blocks.1.ff.linear_in",
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj", "single_transformer_blocks.1.attn.to_out",
    "x_embedder", "proj_out",
]


def _pair():
    torch.manual_seed(0)
    ref = Flux2Transformer2DModel(**TINY).eval()
    native = Flux2DiT(Flux2TransformerConfig.from_dict(TINY)).eval()
    load_native(native, iter(ref.state_dict().items()), remap_transformer_key, "tiny", stacked_params=_QKV_RULES_DIT)
    return ref, native


def _random_adapter(ref, alpha=None):
    g = torch.Generator().manual_seed(7)
    sd = {}
    weights = dict(ref.named_parameters())
    for module in MODULES:
        w = weights[module + ".weight"]
        sd[f"transformer.{module}.lora_A.weight"] = torch.randn(RANK, w.shape[1], generator=g) * 0.1
        sd[f"transformer.{module}.lora_B.weight"] = torch.randn(w.shape[0], RANK, generator=g) * 0.1
        if alpha is not None:
            sd[f"transformer.{module}.alpha"] = torch.tensor(float(alpha))
    return sd


def _surgery(ref, sd, scale, alpha):
    """Reference merge on the diffusers module's own weights."""
    with torch.no_grad():
        for module in MODULES:
            a, b = sd[f"transformer.{module}.lora_A.weight"], sd[f"transformer.{module}.lora_B.weight"]
            p = dict(ref.named_parameters())[module + ".weight"]
            factor = scale * (alpha if alpha is not None else RANK) / RANK
            p.copy_((p.float() + (b.float() @ a.float()) * factor).to(p.dtype))


def _inputs():
    g = torch.Generator().manual_seed(3)
    img = torch.randn(1, 20, 16, generator=g)
    txt = torch.randn(1, 6, 24, generator=g)
    ids_img, ids_txt = image_grid_ids(4, 5), text_ids(6)
    rope = MultiAxisRoPE(2000, (2, 2, 2, 2))(torch.cat([ids_txt, ids_img]))
    return img, txt, torch.tensor([0.5]), ids_img[None], ids_txt[None], rope


@pytest.mark.parametrize("scale,alpha", [(1.0, None), (0.7, 2.0)])
def test_merge_matches_reference_weight_surgery(scale, alpha):
    ref, native = _pair()
    sd = _random_adapter(ref, alpha=alpha)
    adapter = normalize_lora_state_dict(sd)
    assert set(adapter.layers) == set(MODULES)
    touched = merge_lora(native, adapter, remap_transformer_key, _QKV_RULES_DIT, scale=scale)
    assert len(touched) == len(MODULES) and "double_blocks.0.attn.img_qkv.weight" in touched
    _surgery(ref, sd, scale, alpha)
    img, txt, t, ids_img, ids_txt, rope = _inputs()
    with torch.no_grad():
        expected = ref(hidden_states=img, encoder_hidden_states=txt, timestep=t, img_ids=ids_img, txt_ids=ids_txt,
                       return_dict=False)[0]
        out = native(img, txt, t, rope)
    assert torch.equal(expected, out)


def test_bfl_layout_and_kohya_names_are_equivalent():
    """A BFL-layout file (fused qkv with one shared A, native block names, lora_down/up, the
    diffusion_model prefix) merges exactly like its diffusers-format conversion."""
    ref, native = _pair()
    weights = dict(ref.named_parameters())
    g = torch.Generator().manual_seed(11)

    def ab(module, rank=RANK):
        w = weights[module + ".weight"]
        return torch.randn(rank, w.shape[1], generator=g) * 0.1, torch.randn(w.shape[0], rank, generator=g) * 0.1

    a_qkv, b_qkv = ab("transformer_blocks.0.attn.to_q")           # shared A; B for q only ...
    b_qkv = torch.cat([b_qkv, torch.zeros_like(b_qkv), ab("transformer_blocks.0.attn.to_v")[1]])  # ... k zero, v random
    a_proj, b_proj = ab("transformer_blocks.1.attn.to_out.0")
    a_mlp, b_mlp = ab("transformer_blocks.1.ff.linear_in")
    a_l1, b_l1 = ab("single_transformer_blocks.0.attn.to_qkv_mlp_proj")
    a_l2, b_l2 = ab("single_transformer_blocks.1.attn.to_out")
    a_in, b_in = ab("x_embedder")
    a_out, b_out = ab("proj_out")
    bfl = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": a_qkv,
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight": b_qkv,
        "diffusion_model.double_blocks.1.img_attn.proj.lora_A.weight": a_proj,
        "diffusion_model.double_blocks.1.img_attn.proj.lora_B.weight": b_proj,
        "diffusion_model.double_blocks.1.img_mlp.0.lora_A.weight": a_mlp,
        "diffusion_model.double_blocks.1.img_mlp.0.lora_B.weight": b_mlp,
        "diffusion_model.single_blocks.0.linear1.lora_A.weight": a_l1,
        "diffusion_model.single_blocks.0.linear1.lora_B.weight": b_l1,
        "diffusion_model.single_blocks.1.linear2.lora_A.weight": a_l2,
        "diffusion_model.single_blocks.1.linear2.lora_B.weight": b_l2,
        "diffusion_model.img_in.lora_A.weight": a_in,
        "diffusion_model.img_in.lora_B.weight": b_in,
        "diffusion_model.final_layer.linear.lora_A.weight": a_out,
        "diffusion_model.final_layer.linear.lora_B.weight": b_out,
    }
    adapter = normalize_lora_state_dict(bfl, convert_keys=convert_bfl_lora_keys)
    expected_modules = {
        "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v",
        "transformer_blocks.1.attn.to_out.0", "transformer_blocks.1.ff.linear_in",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj", "single_transformer_blocks.1.attn.to_out",
        "x_embedder", "proj_out",
    }
    assert set(adapter.layers) == expected_modules
    assert torch.equal(adapter.layers["transformer_blocks.0.attn.to_k"][0], a_qkv)
    assert torch.equal(adapter.layers["transformer_blocks.0.attn.to_v"][1], b_qkv.chunk(3)[2])
    merge_lora(native, adapter, remap_transformer_key, _QKV_RULES_DIT)
    # reference: the converted (diffusers-naming) adapter applied to the diffusers module's weights
    with torch.no_grad():
        for module, (a, b, alpha) in adapter.layers.items():
            p = weights[module + ".weight"]
            p.copy_((p.float() + (b.float() @ a.float()) * (alpha / RANK)).to(p.dtype))
    img, txt, t, ids_img, ids_txt, rope = _inputs()
    with torch.no_grad():
        expected = ref(hidden_states=img, encoder_hidden_states=txt, timestep=t, img_ids=ids_img, txt_ids=ids_txt,
                       return_dict=False)[0]
        out = native(img, txt, t, rope)
    assert torch.equal(expected, out)


def test_error_paths_and_spec_parsing():
    ref, native = _pair()
    with pytest.raises(ValueError, match="missing lora_B"):
        normalize_lora_state_dict({"transformer.x_embedder.lora_A.weight": torch.zeros(2, 16)})
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_lora_state_dict({"transformer.x_embedder.weight": torch.zeros(2, 16)})
    adapter = normalize_lora_state_dict({
        "transformer.nowhere.lora_A.weight": torch.zeros(2, 16), "transformer.nowhere.lora_B.weight": torch.zeros(8, 2),
    })
    with pytest.raises(KeyError):
        merge_lora(native, adapter, remap_transformer_key, _QKV_RULES_DIT)
    assert LoraSpec.parse("a.safetensors") == LoraSpec("a.safetensors", 1.0)
    assert LoraSpec.parse({"path": "b.safetensors", "scale": 0.5}) == LoraSpec("b.safetensors", 0.5)
