"""``compile_vae_encode_size``: the vae_encoder node compiles the one-frame
encode at the deployment's observation size, on channels-last encoder weights,
and runs every other encode eager; the DROID deployments set the RoboLab size
and warm it at startup. CPU, no weights (the compile itself is stubbed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip("diffusers")
from diffusers import AutoencoderKLWan  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d  # noqa: E402

from mstar.model.components import wan_vae  # noqa: E402
from mstar.model.cosmos3.config import Cosmos3Config  # noqa: E402
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model  # noqa: E402
from mstar.model.cosmos3.submodules import Cosmos3VAEEncoderSubmodule  # noqa: E402
from mstar.model.cosmos3.tests.test_edge import _tiny_edge_config  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[4] / "configs"


def _tiny_wan22_vae(seed: int = 0) -> AutoencoderKLWan:
    # the Wan 2.2 family the Cosmos3 checkpoints ship (residual blocks, 2x2 patch)
    vae = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=16, z_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1,
        is_residual=True, in_channels=12, out_channels=12, patch_size=2, scale_factor_spatial=16,
        latents_mean=[0.0] * 8, latents_std=[1.0] * 8,
    ).eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    return vae


def _encoder_convs(vae):
    # a 1x1x1 kernel's weight is contiguous in both layouts
    return [m for m in vae.encoder.modules() if isinstance(m, torch.nn.Conv3d) and m.weight[0, 0].numel() > 1]


def test_only_the_configured_one_frame_encode_is_compiled(monkeypatch):
    compiled = []

    def fake_compile_encode(vae):
        def encode(x):
            compiled.append(tuple(x.shape))
            return vae.encode(x).latent_dist.mode()
        return encode

    monkeypatch.setattr(wan_vae, "compile_encode", fake_compile_encode)
    stock_vae = _tiny_wan22_vae()
    stock = Cosmos3VAEEncoderSubmodule(vae=stock_vae, config=_tiny_edge_config())
    assert stock._compiled_encode is None
    assert not any(c.weight.is_contiguous(memory_format=torch.channels_last_3d) for c in _encoder_convs(stock_vae))

    vae = _tiny_wan22_vae()
    node = Cosmos3VAEEncoderSubmodule(vae=vae, config=_tiny_edge_config(compile_vae_encode_size=[96, 64]))
    # the node compiles that one encode itself; the engine's whole-forward compile stays off
    assert node.disable_torch_compile and node._compiled_encode is not None
    convs = _encoder_convs(vae)
    assert convs and all(c.weight.is_contiguous(memory_format=torch.channels_last_3d) for c in convs)
    assert not any(type(m) is WanCausalConv3d for m in vae.encoder.modules())
    # the decode path is the decoder node's, and keeps its layout
    assert any(type(m) is WanCausalConv3d for m in vae.decoder.modules())

    torch.manual_seed(1)
    # [width, height] 96x64 -> pixels [1, 3, 1, 64, 96]; another size and a clip run eager
    for shape, takes_compiled in (((1, 3, 1, 64, 96), True), ((1, 3, 1, 64, 64), False), ((1, 3, 5, 64, 96), False)):
        compiled.clear()
        x = torch.rand(shape) * 2 - 1
        with torch.no_grad():
            got = node.forward("prefill_cond", None, x)["cond_latents"][0]
            want = stock.forward("prefill_cond", None, x)["cond_latents"][0]
        assert compiled == ([shape] if takes_compiled else []), shape
        # channels-last weights compute the stock encode (both are bf16 outputs of fp32 encodes)
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("size", [[736], "736x544", [0, 544], [True, 544], [736.0, 544]])
def test_a_malformed_size_is_refused_at_startup(size):
    with pytest.raises(ValueError, match="compile_vae_encode_size must be"):
        _tiny_edge_config(compile_vae_encode_size=size).check_compile()


def test_a_size_is_accepted():
    _tiny_edge_config(compile_vae_encode_size=[736, 544]).check_compile()
    _tiny_edge_config(compile_vae_encode_size=None).check_compile()


@pytest.mark.parametrize("name", ["cosmos3_droid.yaml", "cosmos3_edge_droid.yaml"])
def test_droid_deployments_compile_the_robolab_observation_and_warm_it(name):
    cfg = yaml.safe_load((CONFIGS / name).read_text())
    mk = cfg["model_kwargs"]
    assert mk["compile_vae_encode_size"] == [736, 544]
    assert set(mk) <= set(Cosmos3Config.__dataclass_fields__)
    # a startup policy request at that size, so no client pays the encode's compile
    assert any(
        w.get("output_modalities") == ["action"] and w["model_kwargs"]["size"] == "736x544" and w.get("files")
        for w in cfg["warmup_requests"]
    )


def test_the_nano_droid_warmup_is_a_policy_request_at_the_compiled_size(tmp_path):
    """Nano's checkpoint.json is empty, so the warmup's chunk, fps and domain
    come from the config's policy_* fields, as a RoboLab request's do."""
    cfg = yaml.safe_load((CONFIGS / "cosmos3_droid.yaml").read_text())
    root = tmp_path / "nano"
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "config.json").write_text(json.dumps({"_class_name": "Cosmos3OmniTransformer"}))
    (root / "checkpoint.json").write_text("{}")
    model = Cosmos3Model(model_path_hf=str(root), skip_weight_loading=True)
    for k in ("policy_action_chunk_size", "policy_conditioning_fps", "policy_domain_name"):
        setattr(model.config, k, cfg["model_kwargs"][k])
    warmup = cfg["warmup_requests"][0]
    p = model._resolve_gen_params(warmup["model_kwargs"], ["image", "text"], warmup["output_modalities"])
    assert p["action_mode"] == "policy" and p["action_state_rows"] == 1
    assert p["action_chunk_size"] == 32 and p["num_frames"] == 33
    width, height = cfg["model_kwargs"]["compile_vae_encode_size"]
    assert (p["width"], p["height"]) == (width, height)
