"""CPU-only verification of Cosmos3 in-place weight reload (``Cosmos3Model.reload_weights``).

Covers:
1. In-place storage (``data_ptr``, ``shape``, ``dtype``) invariance across
   single-file and 2-shard diffusers checkpoints, mixed ``bf16`` + ``fp32``
   ``time_embedder``, and ``reload.py`` precedence / ``_diffusers_version``.
2. Nano (``silu``, ``serves_reasoner=False``, ``DROP_KEYS``) vs Edge
   (``relu2``, ``serves_reasoner=True``, ``lm_head``) families, plus
   ``dit``-only, ``reasoner``-only, shared ``dit`` + ``reasoner`` deduplication,
   and unrelated-node no-op.
3. Startup snapshot resilience (deleted/overwritten startup dir, HF hub
   symlink layout, bare ``transformer/`` dir / file).
4. ``float32`` checkpoint -> ``bfloat16`` storage dtype conversion.
5. TP-sharded (``tp_size=2`` and GQA ``tp_size=4``) slice routing.
6. Negative refusal paths copying nothing and preserving weights + sessions.
7. ``SessionStore`` drop semantics on ``Cosmos3DiTSubmodule`` and
   ``Cosmos3VAEDecoderARSubmodule`` (plus ``vae_decoder_ar``-only rank).
8. ``Engine.reload_weights`` + ``KVManager`` prefix-cache drop + ``WeightOps``
   (``PREPARE`` -> ``APPLY``).
9. CPU eager forward parity (A -> B == fresh B, and B -> A round-trip).
10. ``torch.compile`` (``aot_eager`` and ``inductor``) + static captured-buffer
    replay parity under ``torch.compiler.set_stance("fail_on_recompile")``.
    (CPU static-buffer + ``fail_on_recompile`` replay proxy; real CUDA graph
    capture hardware replay requires a CUDA device.)
11. Auxiliary submodules (``vae_encoder``, ``vae_decoder``, ``vae_decoder_ar``,
    ``dit._window_head`` VAEs, ``vision_encoder`` + ``_position_tables``,
    ``audio_decoder`` ``weight_norm``), plus ``transformer/``-only RL reload.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from safetensors.torch import save_file
from test_weight_reload import _StubTransfer
from test_worker_weight_control import _Deployment

from mstar.distributed.communication import CommGroup
from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as kv_manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.manager import KVManager
from mstar.model.components.wan_vae import is_channels_last
from mstar.model.cosmos3 import loader as cosmos3_loader
from mstar.model.cosmos3.components.sound_tokenizer import Cosmos3SoundTokenizer
from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.components.vision import Cosmos3VisionModel
from mstar.model.cosmos3.config import (
    Cosmos3Config,
    Cosmos3ReasonerConfig,
)
from mstar.model.cosmos3.cosmos3_model import (
    AUDIO_DECODER_NODE,
    DIT_NODE,
    REASONER_NODE,
    VAE_DECODER_AR_NODE,
    VAE_DECODER_NODE,
    VAE_ENCODER_NODE,
    VISION_ENCODER_NODE,
    Cosmos3Model,
)
from mstar.model.cosmos3.sessions import SessionStore
from mstar.model.cosmos3.submodules import (
    CAPTURED_GEN_INPUTS,
    CFG_BATCHED_LABEL,
    COND_LABEL,
    UNCOND_LABEL,
    Cosmos3DiTSubmodule,
)
from mstar.model.cosmos3.tests.test_edge import _init_all, _tiny_edge_config
from mstar.model.cosmos3.tests.test_edge_reasoner_parity import _bind as _bind_reasoner_kv
from mstar.model.cosmos3.tests.test_edge_reasoner_parity import _SdpaKV as _ReasonerSdpaKV
from mstar.model.cosmos3.tests.test_engine_cache import _SdpaResources
from mstar.model.loader.reload import (
    _IGNORED_KEYS,
    check_reload_compatible,
    safetensors_files,
    snapshot_checkpoint,
)
from mstar.model.reload_errors import WeightReloadRefusedError
from mstar.utils.ipc_format import ContinueGeneration, UpdateWeightsFromDisk

# ── Shared builders ───────────────────────────────────────────────────────────


def _tiny_nano_config(**overrides) -> Cosmos3Config:
    cfg = Cosmos3Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        vocab_size=100,
        rope_axes_dim=(4, 2, 2),
        rope_theta=1e8,
        rms_norm_eps=1e-5,
        latent_channel=8,
        latent_patch_size=2,
        patch_latent_dim=32,
        sound_gen=False,
        action_gen=True,
        max_action_dim=12,
        num_embodiment_domains=4,
        hidden_act="silu",
        use_moe=True,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        use_und_k_norm_for_gen=False,
        backbone_type="cosmos3_nano_dense",
        reasoner=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _tiny_vae(seed: int = 0, dtype: torch.dtype = torch.float32) -> AutoencoderKLWan:
    vae = AutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=16,
        z_dim=8,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        is_residual=True,
        in_channels=12,
        out_channels=12,
        patch_size=2,
        scale_factor_spatial=16,
        latents_mean=[0.1 * i for i in range(8)],
        latents_std=[0.5 + 0.1 * i for i in range(8)],
    ).eval()
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in vae.parameters():
            p.copy_((torch.randn(p.shape, generator=gen) * 0.2).to(dtype))
    return vae.to(dtype)


def _tiny_sound_config() -> dict:
    return {
        "sampling_rate": 48000,
        "stereo": True,
        "vocoder_input_dim": 8,
        "dec_dim": 8,
        "dec_strides": [2, 2],
        "hop_size": 4,
        "dec_c_mults": [1, 2],
    }


def _transformer_checkpoint_dict(
    cfg: Cosmos3Config,
    seed: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    include_lm_head: bool = True,
) -> dict[str, torch.Tensor]:
    """Build unfused diffusers-format ``transformer/`` state dict."""
    with torch.device("cpu"):
        tf = Cosmos3OmniTransformer(cfg)
    _init_all(tf, seed=seed)
    gen = torch.Generator().manual_seed(seed + 999)
    raw = tf.state_dict()
    out: dict[str, torch.Tensor] = {}
    q_dim = cfg.num_attention_heads * cfg.head_dim
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    for name, tensor in raw.items():
        if name == "lm_head.weight" and not include_lm_head:
            continue
        if ".to_qkv." in name:
            q_part, k_part, v_part = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
            out[name.replace(".to_qkv.", ".to_q.")] = q_part.clone().to(dtype)
            out[name.replace(".to_qkv.", ".to_k.")] = k_part.clone().to(dtype)
            out[name.replace(".to_qkv.", ".to_v.")] = v_part.clone().to(dtype)
        else:
            out[name] = tensor.clone().to(dtype)
    if include_lm_head and "lm_head.weight" not in out:
        out["lm_head.weight"] = (0.05 * torch.randn(cfg.vocab_size, cfg.hidden_size, generator=gen)).to(dtype)
    return out


def _vision_checkpoint_dict(
    reasoner: Cosmos3ReasonerConfig,
    seed: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    with torch.device("cpu"):
        vis = Cosmos3VisionModel(reasoner)
    _init_all(vis, seed=seed)
    out: dict[str, torch.Tensor] = {}
    dim = reasoner.vision.hidden_size
    for name, tensor in vis.state_dict().items():
        if ".qkv_proj." in name:
            q, k, v = tensor.split([dim, dim, dim], dim=0)
            out["model." + name.replace(".qkv_proj.", ".q_proj.")] = q.clone().to(dtype)
            out["model." + name.replace(".qkv_proj.", ".k_proj.")] = k.clone().to(dtype)
            out["model." + name.replace(".qkv_proj.", ".v_proj.")] = v.clone().to(dtype)
        else:
            out["model." + name] = tensor.clone().to(dtype)
    return out


def _sound_checkpoint_dict(
    sound_cfg: dict, seed: int, *, dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    st = Cosmos3SoundTokenizer(sound_cfg)
    _init_all(st, seed=seed)
    out = {k: v.clone().to(dtype) for k, v in st.state_dict().items()}
    # Include an encoder.* weight as published AVAE checkpoints do.
    out["encoder.conv1.weight_v"] = torch.ones(4, 2, 3, dtype=dtype)
    return out


def _transformer_config_json(cfg: Cosmos3Config, *, diffusers_version: str = "0.39.0") -> dict:
    return {
        "_class_name": "Cosmos3OmniTransformer",
        "_diffusers_version": diffusers_version,
        "max_action_dim": cfg.max_action_dim,
        "action_gen": cfg.action_gen,
        "attention_bias": cfg.attention_bias,
        "backbone_type": cfg.backbone_type,
        "base_fps": cfg.base_fps,
        "enable_fps_modulation": cfg.enable_fps_modulation,
        "head_dim": cfg.head_dim,
        "hidden_act": cfg.hidden_act,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "latent_channel": cfg.latent_channel,
        "latent_patch_size": cfg.latent_patch_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_embodiment_domains": cfg.num_embodiment_domains,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_key_value_heads": cfg.num_key_value_heads,
        "patch_latent_dim": cfg.patch_latent_dim,
        "qk_norm_for_diffusion": cfg.qk_norm_for_diffusion,
        "qk_norm_for_text": cfg.qk_norm_for_text,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_scaling": {
            "mrope_section": list(cfg.rope_axes_dim),
            "mrope_interleaved": cfg.mrope_interleaved,
        },
        "rope_theta": cfg.rope_theta,
        "sound_dim": cfg.sound_dim if cfg.sound_gen else None,
        "sound_gen": cfg.sound_gen,
        "sound_latent_fps": cfg.sound_latent_fps,
        "temporal_compression_factor": cfg.vae.scale_factor_temporal,
        "timestep_scale": cfg.timestep_scale,
        "unified_3d_mrope_reset_spatial_ids": cfg.unified_3d_mrope_reset_spatial_ids,
        "unified_3d_mrope_temporal_modality_margin": cfg.unified_3d_mrope_temporal_modality_margin,
        "use_moe": cfg.use_moe,
        "use_und_k_norm_for_gen": cfg.use_und_k_norm_for_gen,
        "vocab_size": cfg.vocab_size,
    }


def _edge_top_config_json(cfg: Cosmos3Config, *, diffusers_version: str = "0.39.0") -> dict:
    r = cfg.reasoner
    v = r.vision
    return {
        "_diffusers_version": diffusers_version,
        "architectures": ["Cosmos3EdgeForConditionalGeneration"],
        "model_type": "cosmos3_edge",
        "image_token_id": r.image_token_id,
        "video_token_id": r.video_token_id,
        "vision_start_token_id": r.vision_start_token_id,
        "vision_end_token_id": r.vision_end_token_id,
        "projector_config": {
            "input_hidden_size": r.projector_input_hidden_size,
            "merger_intermediate_size": r.projector_hidden_size,
            "out_hidden_size": r.projector_out_hidden_size,
            "spatial_merge_size": v.spatial_merge_size,
            "use_postshuffle_norm": False,
        },
        "text_config": {
            "eos_token_id": r.eos_token_id,
            "hidden_size": cfg.hidden_size,
            "max_position_embeddings": 1024,
        },
        "vision_config": {
            "hidden_size": v.hidden_size,
            "intermediate_size": v.intermediate_size,
            "num_attention_heads": v.num_attention_heads,
            "num_hidden_layers": v.num_hidden_layers,
            "num_patches": v.num_patches,
            "patch_size": v.patch_size,
            "spatial_merge_size": v.spatial_merge_size,
        },
    }


def _save_sharded_diffusion(target_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    keys = sorted(tensors)
    mid = len(keys) // 2
    s1 = {k: tensors[k] for k in keys[:mid]}
    s2 = {k: tensors[k] for k in keys[mid:]}
    f1 = "diffusion_pytorch_model-00001-of-00002.safetensors"
    f2 = "diffusion_pytorch_model-00002-of-00002.safetensors"
    save_file(s1, str(target_dir / f1))
    save_file(s2, str(target_dir / f2))
    weight_map = {k: f1 for k in s1} | {k: f2 for k in s2}
    (target_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )


def _write_cosmos3_checkpoint(
    root: Path,
    cfg: Cosmos3Config,
    seed: int,
    *,
    sharded: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    include_lm_head: bool = True,
    include_vae: bool = False,
    include_vision: bool = False,
    include_sound: bool = False,
    transformer_only: bool = False,
    diffusers_version: str = "0.39.0",
    extra_transformer_config: dict | None = None,
    extra_top_config: dict | None = None,
    extra_modular_index: dict | None = None,
    extra_scheduler_config: dict | None = None,
) -> tuple[str, dict[str, torch.Tensor]]:
    tdir = root / "transformer"
    tdir.mkdir(parents=True, exist_ok=True)
    tcfg = _transformer_config_json(cfg, diffusers_version=diffusers_version)
    if extra_transformer_config:
        tcfg.update(extra_transformer_config)
    (tdir / "config.json").write_text(json.dumps(tcfg))

    tf_tensors = _transformer_checkpoint_dict(cfg, seed, dtype=dtype, include_lm_head=include_lm_head)
    if sharded:
        _save_sharded_diffusion(tdir, tf_tensors)
    else:
        save_file(tf_tensors, str(tdir / "diffusion_pytorch_model.safetensors"))

    if not transformer_only:
        (root / "model_index.json").write_text(
            json.dumps({
                "_class_name": "Cosmos3Pipeline",
                "_diffusers_version": diffusers_version,
                "use_native_flow_schedule": cfg.use_native_flow_schedule,
            })
        )
        UniPCMultistepScheduler().save_pretrained(str(root / "scheduler"))
        if extra_scheduler_config:
            sp = root / "scheduler" / "scheduler_config.json"
            sc = json.loads(sp.read_text())
            sc.update(extra_scheduler_config)
            sp.write_text(json.dumps(sc))

        if extra_modular_index is not None:
            (root / "modular_model_index.json").write_text(json.dumps(extra_modular_index))

        if cfg.reasoner is not None:
            top = _edge_top_config_json(cfg, diffusers_version=diffusers_version)
            if extra_top_config:
                top.update(extra_top_config)
            (root / "config.json").write_text(json.dumps(top))
            if include_vision:
                vdir = root / "vision_encoder"
                vdir.mkdir(exist_ok=True)
                vis_tensors = _vision_checkpoint_dict(cfg.reasoner, seed + 100, dtype=dtype)
                save_file(vis_tensors, str(vdir / "model.safetensors"))
            else:
                (root / "vision_encoder").mkdir(exist_ok=True)

        if include_vae:
            vae = _tiny_vae(seed=seed + 200, dtype=dtype)
            vae.save_pretrained(str(root / "vae"))

        if include_sound:
            sdir = root / "sound_tokenizer"
            sdir.mkdir(exist_ok=True)
            scfg = _tiny_sound_config()
            (sdir / "config.json").write_text(json.dumps(scfg))
            save_file(
                _sound_checkpoint_dict(scfg, seed + 300, dtype=dtype),
                str(sdir / "diffusion_pytorch_model.safetensors"),
            )

    return str(root), tf_tensors


def _dummy_tokenizer():
    return SimpleNamespace(eos_token_id=11, bos_token_id=1, pad_token_id=0)


def _build_cosmos3(
    ckpt_dir: str,
    monkeypatch,
    nodes: tuple[str, ...],
    *,
    skip_weight_loading: bool = False,
    tp_group=None,
    **model_kwargs,
) -> tuple[Cosmos3Model, dict[str, torch.nn.Module]]:
    monkeypatch.setattr(Cosmos3Model, "_load_tokenizer", lambda self: _dummy_tokenizer())
    model = Cosmos3Model(
        model_path_hf=ckpt_dir,
        skip_weight_loading=skip_weight_loading,
        **model_kwargs,
    )
    subs: dict[str, torch.nn.Module] = {}
    for node in nodes:
        sub = model.get_submodule(node, device="cpu", tp_group=tp_group)
        assert sub is not None
        subs[node] = sub
    return model, subs


def _engine(model, submodules: dict[str, torch.nn.Module], kv=None) -> Engine:
    eng = Engine.__new__(Engine)
    eng._model = model
    eng._device = torch.device("cpu")
    eng._submodules = {name: SimpleNamespace(submodule=sub) for name, sub in submodules.items()}
    eng._resources = {"kv": kv} if kv is not None else {}
    eng._resource_nodes = {"kv": set(submodules)} if kv is not None else {}
    return eng


def _storage_snapshot(submodules: dict[str, torch.nn.Module]) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    snap: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    for node, sub in submodules.items():
        for name, param in sub.named_parameters():
            snap[f"{node}.{name}"] = (param.data_ptr(), tuple(param.shape), param.dtype)
        for name, buf in sub.named_buffers():
            snap[f"{node}.__buffer__.{name}"] = (buf.data_ptr(), tuple(buf.shape), buf.dtype)
    return snap


# ── 1. In-place storage invariance, mixed bf16/fp32, reload.py precedence ─────


def test_cosmos3_reload_preserves_pointers_and_mixed_bf16_fp32(tmp_path, monkeypatch):
    cfg = _tiny_edge_config(action_gen=True, max_action_dim=12, num_embodiment_domains=4, compile_denoise=False)
    dir_a, _ = _write_cosmos3_checkpoint(
        tmp_path / "ckpt_a", cfg, seed=1, sharded=True, diffusers_version="0.39.0",
    )
    dir_b, state_b = _write_cosmos3_checkpoint(
        tmp_path / "ckpt_b", cfg, seed=2, sharded=False, diffusers_version="0.40.0",
    )

    model, subs = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)
    tf = subs[DIT_NODE].transformer
    assert subs[REASONER_NODE].model.transformer is tf

    # Startup dtype contract: time_embedder is float32, rest of transformer is bfloat16.
    assert tf.time_embedder.linear_1.weight.dtype == torch.float32
    assert tf.layers[0].self_attn.to_qkv.weight.dtype == torch.bfloat16
    assert tf.lm_head.weight.dtype == torch.bfloat16

    before_storage = _storage_snapshot(subs)
    before_qkv = tf.layers[0].self_attn.to_qkv.weight.clone()
    before_time = tf.time_embedder.linear_1.weight.clone()

    engine = _engine(model, subs)
    engine.check_weight_reload(dir_b)
    assert torch.equal(tf.layers[0].self_attn.to_qkv.weight, before_qkv)
    assert torch.equal(tf.time_embedder.linear_1.weight, before_time)

    engine.reload_weights(dir_b, "v1")
    after_storage = _storage_snapshot(subs)
    assert after_storage == before_storage
    assert tf.time_embedder.linear_1.weight.dtype == torch.float32
    assert tf.layers[0].self_attn.to_qkv.weight.dtype == torch.bfloat16

    expected_qkv = torch.cat([
        state_b["layers.0.self_attn.to_q.weight"],
        state_b["layers.0.self_attn.to_k.weight"],
        state_b["layers.0.self_attn.to_v.weight"],
    ], dim=0)
    assert torch.equal(tf.layers[0].self_attn.to_qkv.weight, expected_qkv)
    assert torch.equal(tf.layers[0].self_attn.add_q_proj.weight, state_b["layers.0.self_attn.add_q_proj.weight"])
    assert torch.equal(tf.layers[0].mlp.up_proj.weight, state_b["layers.0.mlp.up_proj.weight"])
    assert torch.equal(tf.proj_in.weight, state_b["proj_in.weight"])
    assert torch.equal(tf.proj_out.weight, state_b["proj_out.weight"])
    assert torch.equal(tf.action_proj_in.fc.weight, state_b["action_proj_in.fc.weight"])
    assert torch.equal(tf.action_proj_out.fc.weight, state_b["action_proj_out.fc.weight"])
    assert torch.equal(tf.norm.weight, state_b["norm.weight"])
    assert torch.equal(tf.lm_head.weight, state_b["lm_head.weight"])
    assert torch.equal(tf.time_embedder.linear_1.weight, state_b["time_embedder.linear_1.weight"].float())

    # reload.py: model.safetensors takes precedence over diffusion_pytorch_model.safetensors,
    # and _diffusers_version is ignored at any depth in config.json.
    assert "_diffusers_version" in _IGNORED_KEYS["config.json"]
    prec = tmp_path / "precedence"
    prec.mkdir()
    save_file({"a": torch.ones(2, 2)}, str(prec / "model.safetensors"))
    save_file({"b": torch.ones(2, 2)}, str(prec / "diffusion_pytorch_model.safetensors"))
    assert safetensors_files(prec) == [prec / "model.safetensors"]
    snap = snapshot_checkpoint(prec)
    (prec / "config.json").write_text(json.dumps({"nested": {"_diffusers_version": "0.1"}}))
    prec2 = tmp_path / "precedence2"
    prec2.mkdir()
    save_file({"a": torch.zeros(2, 2)}, str(prec2 / "model.safetensors"))
    (prec2 / "config.json").write_text(json.dumps({"nested": {"_diffusers_version": "0.9"}}))
    check_reload_compatible(prec2, snapshot_checkpoint(prec))
    assert "a" in snap.specs


# ── 2. Nano vs Edge families, DROP_KEYS, dit+reasoner deduplication ───────────


def test_cosmos3_reload_nano_swiglu_drops_lm_head_and_edge_relu2_keeps_lm_head(tmp_path, monkeypatch):
    # Nano (silu + MoE + serves_reasoner=False): accepts checkpoints with OR without lm_head.weight.
    nano_cfg = _tiny_nano_config(compile_denoise=False)
    nano_a, _ = _write_cosmos3_checkpoint(tmp_path / "nano_a", nano_cfg, seed=10, include_lm_head=True)
    nano_b, state_b = _write_cosmos3_checkpoint(tmp_path / "nano_b", nano_cfg, seed=11, include_lm_head=True)
    nano_c, state_c = _write_cosmos3_checkpoint(tmp_path / "nano_c", nano_cfg, seed=12, include_lm_head=False)

    nano_model, nano_subs = _build_cosmos3(nano_a, monkeypatch, (DIT_NODE,), compile_denoise=False)
    assert not hasattr(nano_subs[DIT_NODE].transformer, "lm_head")
    nano_ptrs = _storage_snapshot(nano_subs)

    engine = _engine(nano_model, nano_subs)
    engine.reload_weights(nano_b, "v1")
    assert _storage_snapshot(nano_subs) == nano_ptrs
    assert torch.equal(nano_subs[DIT_NODE].transformer.layers[0].mlp.gate_proj.weight,
                       state_b["layers.0.mlp.gate_proj.weight"])

    engine.reload_weights(nano_c, "v2")
    assert _storage_snapshot(nano_subs) == nano_ptrs
    assert torch.equal(nano_subs[DIT_NODE].transformer.layers[0].mlp_moe_gen.down_proj.weight,
                       state_c["layers.0.mlp_moe_gen.down_proj.weight"])

    # Unrelated node is a safe no-op.
    nano_model.reload_weights(nano_b, {"unrelated": torch.nn.Linear(2, 2)})

    # Edge (dit-only, reasoner-only, and shared dit+reasoner loaded only ONCE).
    edge_cfg = _tiny_edge_config(action_gen=False, compile_denoise=False)
    edge_a, _ = _write_cosmos3_checkpoint(tmp_path / "edge_a", edge_cfg, seed=20, include_lm_head=True)
    edge_b, edge_b_state = _write_cosmos3_checkpoint(tmp_path / "edge_b", edge_cfg, seed=21, include_lm_head=True)

    # dit-only on Edge (with serves_reasoner=True)
    m_dit, s_dit = _build_cosmos3(edge_a, monkeypatch, (DIT_NODE,), compile_denoise=False)
    _engine(m_dit, s_dit).reload_weights(edge_b, "v1")
    assert torch.equal(s_dit[DIT_NODE].transformer.lm_head.weight, edge_b_state["lm_head.weight"])

    # reasoner-only on Edge
    m_reas, s_reas = _build_cosmos3(edge_a, monkeypatch, (REASONER_NODE,), compile_denoise=False)
    _engine(m_reas, s_reas).reload_weights(edge_b, "v1")
    assert torch.equal(s_reas[REASONER_NODE].model.transformer.lm_head.weight, edge_b_state["lm_head.weight"])

    # shared dit + reasoner: load_transformer_path called ONCE, not twice.
    m_both, s_both = _build_cosmos3(edge_a, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)
    calls = []
    orig_load = cosmos3_loader.load_transformer_path

    def spy_load(tf, path, device="cpu"):
        calls.append((id(tf), path, device))
        return orig_load(tf, path, device=device)

    monkeypatch.setattr(cosmos3_loader, "load_transformer_path", spy_load)
    _engine(m_both, s_both).reload_weights(edge_b, "v1")
    assert len(calls) == 1


# ── 3. Snapshot resilience, HF symlinks, bare transformer path ────────────────


def test_cosmos3_reload_survives_startup_delete_and_overwrite_and_hub_symlinks(tmp_path, monkeypatch):
    cfg = _tiny_nano_config(compile_denoise=False)
    dir_a, _ = _write_cosmos3_checkpoint(tmp_path / "a", cfg, seed=30)
    dir_b, state_b = _write_cosmos3_checkpoint(tmp_path / "b", cfg, seed=31, diffusers_version="0.41.0")
    model, subs = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE,), compile_denoise=False)
    engine = _engine(model, subs)

    # Overwrite startup dir with incompatible shapes: still refused against in-memory snapshot.
    bad_state = {**state_b, "proj_in.weight": torch.zeros(1, 1, dtype=torch.bfloat16)}
    save_file(bad_state, str(Path(dir_a) / "transformer" / "diffusion_pytorch_model.safetensors"))
    with pytest.raises(WeightReloadRefusedError, match="shape"):
        engine.check_weight_reload(dir_a)

    # Delete startup dir completely: valid reload from B still succeeds.
    shutil.rmtree(dir_a)
    engine.reload_weights(dir_b, "v1")
    assert torch.equal(subs[DIT_NODE].transformer.proj_in.weight, state_b["proj_in.weight"])

    # Bare transformer directory and direct .safetensors file also work.
    dir_c, state_c = _write_cosmos3_checkpoint(tmp_path / "c", cfg, seed=32)
    engine.reload_weights(str(Path(dir_c) / "transformer"), "v2")
    assert torch.equal(subs[DIT_NODE].transformer.proj_in.weight, state_c["proj_in.weight"])
    engine.reload_weights(str(Path(dir_b) / "transformer" / "diffusion_pytorch_model.safetensors"), "v3")
    assert torch.equal(subs[DIT_NODE].transformer.proj_in.weight, state_b["proj_in.weight"])

    # HF hub cache symlink layout (snapshots/<rev>/transformer/*.safetensors -> ../../blobs/*).
    hub_root = tmp_path / "hub"
    blobs = hub_root / "blobs"
    snap_dir = hub_root / "snapshots" / "rev1"
    blobs.mkdir(parents=True)
    (snap_dir / "transformer").mkdir(parents=True)
    blob_file = blobs / "deadbeef"
    shutil.copyfile(Path(dir_c) / "transformer" / "diffusion_pytorch_model.safetensors", blob_file)
    shutil.copyfile(Path(dir_c) / "transformer" / "config.json", snap_dir / "transformer" / "config.json")
    os.symlink(os.path.relpath(blob_file, snap_dir / "transformer"),
               snap_dir / "transformer" / "diffusion_pytorch_model.safetensors")
    engine.reload_weights(str(snap_dir), "v4")
    assert torch.equal(subs[DIT_NODE].transformer.proj_in.weight, state_c["proj_in.weight"])


# ── 4. Dtype conversion (fp32 ckpt -> bf16 storage) ───────────────────────────


def test_cosmos3_reload_converts_fp32_ckpt_into_bf16_storage(tmp_path, monkeypatch):
    cfg = _tiny_edge_config(action_gen=False, compile_denoise=False)
    dir_a, _ = _write_cosmos3_checkpoint(tmp_path / "a", cfg, seed=40, dtype=torch.bfloat16)
    dir_b_fp32, state_b_fp32 = _write_cosmos3_checkpoint(tmp_path / "b_fp32", cfg, seed=41, dtype=torch.float32)

    model, subs = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE,), compile_denoise=False)
    before = _storage_snapshot(subs)
    _engine(model, subs).reload_weights(dir_b_fp32, "v1")
    assert _storage_snapshot(subs) == before

    tf = subs[DIT_NODE].transformer
    assert tf.proj_in.weight.dtype == torch.bfloat16
    assert tf.time_embedder.linear_1.weight.dtype == torch.float32
    assert torch.equal(tf.proj_in.weight, state_b_fp32["proj_in.weight"].to(torch.bfloat16))
    assert torch.equal(tf.time_embedder.linear_1.weight, state_b_fp32["time_embedder.linear_1.weight"])


# ── 5. TP-sharded slices & GQA replication ────────────────────────────────────


@pytest.mark.parametrize("tp_size", [2, 4])
def test_cosmos3_reload_tp_sharded_slices_and_gqa_replication(tmp_path, monkeypatch, tp_size):
    # Test both silu+MoE (gate_proj/up_proj/down_proj) and Edge lm_head across ranks.
    num_heads = 4 if tp_size == 2 else 8
    num_kv_heads = 2 if tp_size == 2 else 4
    cfg = _tiny_edge_config(
        hidden_act="silu",
        use_moe=True,
        action_gen=False,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=16,
        compile_denoise=False,
    )
    dir_a, _ = _write_cosmos3_checkpoint(tmp_path / f"tp{tp_size}_a", cfg, seed=50, sharded=True)
    dir_b, state_b = _write_cosmos3_checkpoint(tmp_path / f"tp{tp_size}_b", cfg, seed=51, sharded=True)

    q_head_dim = cfg.head_dim
    q_dim = cfg.num_attention_heads * q_head_dim
    kv_heads_per_rank = num_kv_heads // tp_size

    for rank in range(tp_size):
        group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(tp_size)))
        model, subs = _build_cosmos3(
            dir_a, monkeypatch, (DIT_NODE, REASONER_NODE), tp_group=group, compile_denoise=False,
        )
        before = _storage_snapshot(subs)
        _engine(model, subs).reload_weights(dir_b, "v1")
        assert _storage_snapshot(subs) == before

        tf = subs[DIT_NODE].transformer
        # 1) QKVParallelLinear
        q_slice = state_b["layers.0.self_attn.to_q.weight"].chunk(tp_size, dim=0)[rank]
        k_slice = state_b["layers.0.self_attn.to_k.weight"].chunk(tp_size, dim=0)[rank]
        v_slice = state_b["layers.0.self_attn.to_v.weight"].chunk(tp_size, dim=0)[rank]
        assert q_slice.shape[0] == q_dim // tp_size
        assert k_slice.shape[0] == kv_heads_per_rank * q_head_dim
        assert torch.equal(tf.layers[0].self_attn.to_qkv.weight, torch.cat([q_slice, k_slice, v_slice], dim=0))

        # 2) ColumnParallelLinear: add_{q,k,v}_proj, mlp.{gate,up}_proj, mlp_moe_gen.{gate,up}_proj, lm_head
        assert torch.equal(
            tf.layers[0].self_attn.add_q_proj.weight,
            state_b["layers.0.self_attn.add_q_proj.weight"].chunk(tp_size, dim=0)[rank],
        )
        assert torch.equal(
            tf.layers[0].self_attn.add_k_proj.weight,
            state_b["layers.0.self_attn.add_k_proj.weight"].chunk(tp_size, dim=0)[rank],
        )
        assert torch.equal(
            tf.layers[0].mlp.gate_proj.weight,
            state_b["layers.0.mlp.gate_proj.weight"].chunk(tp_size, dim=0)[rank],
        )
        assert torch.equal(
            tf.layers[0].mlp.up_proj.weight,
            state_b["layers.0.mlp.up_proj.weight"].chunk(tp_size, dim=0)[rank],
        )
        assert torch.equal(
            tf.layers[0].mlp_moe_gen.up_proj.weight,
            state_b["layers.0.mlp_moe_gen.up_proj.weight"].chunk(tp_size, dim=0)[rank],
        )
        assert torch.equal(tf.lm_head.weight, state_b["lm_head.weight"].chunk(tp_size, dim=0)[rank])

        # 3) RowParallelLinear: to_out, to_add_out, mlp.down_proj, mlp_moe_gen.down_proj
        assert torch.equal(
            tf.layers[0].self_attn.to_out.weight,
            state_b["layers.0.self_attn.to_out.weight"].chunk(tp_size, dim=1)[rank],
        )
        assert torch.equal(
            tf.layers[0].self_attn.to_add_out.weight,
            state_b["layers.0.self_attn.to_add_out.weight"].chunk(tp_size, dim=1)[rank],
        )
        assert torch.equal(
            tf.layers[0].mlp.down_proj.weight,
            state_b["layers.0.mlp.down_proj.weight"].chunk(tp_size, dim=1)[rank],
        )
        assert torch.equal(
            tf.layers[0].mlp_moe_gen.down_proj.weight,
            state_b["layers.0.mlp_moe_gen.down_proj.weight"].chunk(tp_size, dim=1)[rank],
        )

    # Also verify QKVParallelLinear GQA KV-head replication (tp_size=4, num_kv_heads=2 -> kv_head_replicas=2)
    # in-place through load_weights_into.
    if tp_size == 4:
        from mstar.model.components.distributed.linear import QKVParallelLinear
        from mstar.model.cosmos3.loader import TRANSFORMER_STACKED
        from mstar.model.loader import load_weights_into

        class _QKVHolder(torch.nn.Module):
            def __init__(self, grp):
                super().__init__()
                self.attn = torch.nn.Module()
                self.attn.to_qkv = QKVParallelLinear(grp, 64, 16, 4, 2, bias=False)

        q_w = torch.randn(4 * 16, 64, dtype=torch.bfloat16)
        k_w = torch.randn(2 * 16, 64, dtype=torch.bfloat16)
        v_w = torch.randn(2 * 16, 64, dtype=torch.bfloat16)
        for rank in range(4):
            grp = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1, 2, 3])
            holder = _QKVHolder(grp).to(torch.bfloat16)
            ptr = holder.attn.to_qkv.weight.data_ptr()
            load_weights_into(
                holder,
                [("attn.to_q.weight", q_w), ("attn.to_k.weight", k_w), ("attn.to_v.weight", v_w)],
                stacked_params=TRANSFORMER_STACKED,
            )
            assert holder.attn.to_qkv.weight.data_ptr() == ptr
            kv_idx = rank // 2
            want = torch.cat([
                q_w.chunk(4, dim=0)[rank],
                k_w.chunk(2, dim=0)[kv_idx],
                v_w.chunk(2, dim=0)[kv_idx],
            ], dim=0)
            assert torch.equal(holder.attn.to_qkv.weight, want)


# ── 6. Negative error paths copy nothing ─────────────────────────────────────


def test_cosmos3_reload_refuses_incompatible_checkpoints_without_copying(tmp_path, monkeypatch):
    cfg = _tiny_edge_config(action_gen=True, max_action_dim=12, num_embodiment_domains=4, compile_denoise=False)
    dir_a, state_a = _write_cosmos3_checkpoint(tmp_path / "a", cfg, seed=60)
    dir_b, state_b = _write_cosmos3_checkpoint(tmp_path / "b", cfg, seed=61)

    model, subs = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)
    engine = _engine(model, subs)
    tf = subs[DIT_NODE].transformer

    def assert_still_a():
        assert torch.equal(tf.proj_in.weight, state_a["proj_in.weight"])
        assert torch.equal(tf.lm_head.weight, state_a["lm_head.weight"])

    # Missing directory & empty transformer dir
    with pytest.raises(WeightReloadRefusedError, match="no checkpoint"):
        engine.check_weight_reload(str(tmp_path / "does_not_exist"))
    empty_t = tmp_path / "empty_t"
    (empty_t / "transformer").mkdir(parents=True)
    with pytest.raises(WeightReloadRefusedError, match="no model.safetensors"):
        engine.check_weight_reload(str(empty_t))

    # Corrupt / truncated safetensors
    trunc = tmp_path / "trunc"
    (trunc / "transformer").mkdir(parents=True)
    (trunc / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(
        (Path(dir_b) / "transformer" / "diffusion_pytorch_model.safetensors").read_bytes()[:-64]
    )
    with pytest.raises(WeightReloadRefusedError, match="unreadable"):
        engine.check_weight_reload(str(trunc))

    # Index path traversal & non-dict weight_map
    for bad_name, weight_map in (
        ("traversal_rel", {"a": "../escape.safetensors"}),
        ("traversal_abs", {"a": "/tmp/escape.safetensors"}),
        ("non_dict_wm", ["not", "a", "dict"]),
    ):
        bad_idx = tmp_path / bad_name
        (bad_idx / "transformer").mkdir(parents=True)
        (bad_idx / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        with pytest.raises(WeightReloadRefusedError):
            engine.check_weight_reload(str(bad_idx))

    # Missing key & partial to_qkv shard missing to_k & extra key
    missing_k = tmp_path / "missing_key"
    (missing_k / "transformer").mkdir(parents=True)
    save_file(
        {k: v for k, v in state_b.items() if k != "layers.0.self_attn.to_k.weight"},
        str(missing_k / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    with pytest.raises(WeightReloadRefusedError, match="missing"):
        engine.check_weight_reload(str(missing_k))

    extra_k = tmp_path / "extra_key"
    (extra_k / "transformer").mkdir(parents=True)
    save_file(
        {**state_b, "unexpected.weight": torch.zeros(2, 2, dtype=torch.bfloat16)},
        str(extra_k / "transformer" / "diffusion_pytorch_model.safetensors"),
    )
    with pytest.raises(WeightReloadRefusedError, match="does not have"):
        engine.check_weight_reload(str(extra_k))

    # Shape mismatches (to_q, add_q_proj, action_proj_in.fc) & non-float dtype (int32)
    for tag, overrides in (
        ("bad_to_q", {"layers.0.self_attn.to_q.weight": torch.zeros(4, cfg.hidden_size, dtype=torch.bfloat16)}),
        ("bad_add_q", {"layers.0.self_attn.add_q_proj.weight": torch.zeros(4, cfg.hidden_size, dtype=torch.bfloat16)}),
        ("bad_act", {"action_proj_in.fc.weight": torch.zeros(4, cfg.max_action_dim, dtype=torch.bfloat16)}),
        ("bad_int32", {"proj_in.weight": torch.ones_like(state_b["proj_in.weight"], dtype=torch.int32)}),
    ):
        d = tmp_path / tag
        (d / "transformer").mkdir(parents=True)
        save_file({**state_b, **overrides}, str(d / "transformer" / "diffusion_pytorch_model.safetensors"))
        expected_msg = "dtype" if tag == "bad_int32" else "shape"
        with pytest.raises(WeightReloadRefusedError, match=expected_msg):
            engine.check_weight_reload(str(d))

    # Mismatched transformer/config.json, top-level config.json, scheduler_config.json, modular_model_index.json
    bad_tcfg, _ = _write_cosmos3_checkpoint(
        tmp_path / "bad_tcfg", cfg, seed=62, extra_transformer_config={"hidden_size": 128},
    )
    with pytest.raises(WeightReloadRefusedError, match="hidden_size"):
        engine.check_weight_reload(bad_tcfg)

    bad_top, _ = _write_cosmos3_checkpoint(
        tmp_path / "bad_top", cfg, seed=63, extra_top_config={"image_token_id": 999},
    )
    with pytest.raises(WeightReloadRefusedError, match="image_token_id"):
        engine.check_weight_reload(bad_top)

    bad_sched, _ = _write_cosmos3_checkpoint(
        tmp_path / "bad_sched", cfg, seed=64, extra_scheduler_config={"solver_order": 3},
    )
    with pytest.raises(WeightReloadRefusedError, match="scheduler_config.json"):
        engine.check_weight_reload(bad_sched)

    dist_a, _ = _write_cosmos3_checkpoint(
        tmp_path / "dist_a", cfg, seed=67, extra_modular_index={"is_distilled": True, "distilled_sigmas": [1.0, 0.5]},
    )
    dist_b, _ = _write_cosmos3_checkpoint(
        tmp_path / "dist_b", cfg, seed=68, extra_modular_index={"is_distilled": True, "distilled_sigmas": [1.0, 0.25]},
    )
    m_dist, s_dist = _build_cosmos3(dist_a, monkeypatch, (DIT_NODE,), compile_denoise=False)
    with pytest.raises(WeightReloadRefusedError, match="modular_model_index.json"):
        _engine(m_dist, s_dist).check_weight_reload(dist_b)

    # ReloadTarget(subdir=...) resolves subdir automatically via shared Model.reload_weights.
    from mstar.model.base import Model

    class _SubdirMod(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = torch.nn.Parameter(w.clone())

        def load_weights(self, weights):
            for k, v in weights:
                if k == "weight":
                    self.weight.data.copy_(v)
            return {"weight"}

    class _SubdirModel(Model):
        get_graph_walk_graphs = get_submodule = process_prompt = walk_plan = lambda *a, **kw: None

        def __init__(self, ckpt):
            self._register_reload_target("core", checkpoint_dir=ckpt, subdir="weights")

    sd1 = tmp_path / "sd1"
    sd2 = tmp_path / "sd2"
    (sd1 / "weights").mkdir(parents=True)
    (sd2 / "weights").mkdir(parents=True)
    save_file({"weight": torch.ones(2, 2)}, str(sd1 / "weights" / "model.safetensors"))
    save_file({"weight": torch.full((2, 2), 3.0)}, str(sd2 / "weights" / "model.safetensors"))
    sd_mod = _SubdirMod(torch.ones(2, 2))
    ptr = sd_mod.weight.data_ptr()
    _SubdirModel(sd1).reload_weights(str(sd2), {"core": sd_mod})
    assert sd_mod.weight.data_ptr() == ptr and torch.equal(sd_mod.weight, torch.full((2, 2), 3.0))

    # Transformer tokenizer.json mismatch (if checkpoint ships tokenizer.json alongside transformer/)
    tok_a, _ = _write_cosmos3_checkpoint(tmp_path / "tok_a", cfg, seed=65)
    tok_b, _ = _write_cosmos3_checkpoint(tmp_path / "tok_b", cfg, seed=66)
    (Path(tok_a) / "transformer" / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0}}}))
    (Path(tok_b) / "transformer" / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 1}}}))
    m_tok, s_tok = _build_cosmos3(tok_a, monkeypatch, (DIT_NODE,), compile_denoise=False)
    with pytest.raises(WeightReloadRefusedError, match="tokenizer.json"):
        _engine(m_tok, s_tok).check_weight_reload(tok_b)

    # skip_weight_loading=True refuses cleanly
    dummy_m, dummy_s = _build_cosmos3(
        dir_a, monkeypatch, (DIT_NODE,), skip_weight_loading=True, compile_denoise=False,
    )
    with pytest.raises(WeightReloadRefusedError, match="not loaded from a checkpoint"):
        _engine(dummy_m, dummy_s).check_weight_reload(dir_b)

    # Unreadable startup snapshot string refuses cleanly
    unreadable = Cosmos3Model.__new__(Cosmos3Model)
    unreadable.config = cfg
    unreadable.skip_weight_loading = False
    unreadable._transformer_checkpoint = "no checkpoint at /gone"
    with pytest.raises(WeightReloadRefusedError, match="could not be read at startup"):
        _engine(unreadable, subs).check_weight_reload(dir_b)

    assert_still_a()

    # Direct load_transformer_path also catches partial to_qkv shard missing to_k.
    with pytest.raises(KeyError, match="missing shards"):
        cosmos3_loader.load_transformer_path(tf, missing_k / "transformer", device="cpu")


# ── 7. SessionStore drop semantics ────────────────────────────────────────────


def test_cosmos3_reload_session_store_check_only_keeps_and_apply_clears(tmp_path, monkeypatch):
    cfg = _tiny_edge_config(action_gen=False, compile_denoise=False)
    dir_a, _ = _write_cosmos3_checkpoint(tmp_path / "a", cfg, seed=70, include_vae=True)
    dir_b, _ = _write_cosmos3_checkpoint(tmp_path / "b", cfg, seed=71, include_vae=True)

    model, subs = _build_cosmos3(
        dir_a, monkeypatch, (DIT_NODE, VAE_DECODER_AR_NODE),
        compile_denoise=False, enable_windowed_video=True, vae_decode_channels_last=False,
    )
    dit: Cosmos3DiTSubmodule = subs[DIT_NODE]
    ar_dec = subs[VAE_DECODER_AR_NODE]

    # Populate both SessionStores with idle entries + live pinned entries.
    dit._sessions.put("idle_sess", torch.ones(1, 8, 2, 4, 4))
    dit._sessions.begin("pinned_sess", "req_1")
    dit._sessions.put("pinned_sess", torch.full((1, 8, 2, 4, 4), 2.0))
    dit._sessions.begin("pinned_sess", "req_2")

    ar_dec._sessions.put("idle_sess", torch.ones(1, 8, 2, 4, 4))
    ar_dec._sessions.begin("pinned_sess", "req_1")
    ar_dec._sessions.put("pinned_sess", torch.full((1, 8, 2, 4, 4), 3.0))
    ar_dec._sessions.begin("pinned_sess", "req_2")

    engine = _engine(model, subs)

    # 1) check_only=True preserves both stores and pins.
    engine.check_weight_reload(dir_b)
    assert set(dit._sessions.ids()) == {"idle_sess", "pinned_sess"}
    assert dit._sessions.live("pinned_sess") == "req_2"
    assert set(ar_dec._sessions.ids()) == {"idle_sess", "pinned_sess"}
    assert ar_dec._sessions.live("pinned_sess") == "req_2"

    # 2) Refused reload (check_only=False) leaves both stores intact.
    with pytest.raises(WeightReloadRefusedError):
        engine.reload_weights(str(tmp_path / "missing"), "v_bad")
    assert set(dit._sessions.ids()) == {"idle_sess", "pinned_sess"}
    assert set(ar_dec._sessions.ids()) == {"idle_sess", "pinned_sess"}

    # 3) Apply clears both SessionStores and rejects expired IDs afterward.
    engine.reload_weights(dir_b, "v1")
    assert len(dit._sessions) == 0 and dit._sessions.get("idle_sess") is None
    assert len(ar_dec._sessions) == 0 and ar_dec._sessions.get("idle_sess") is None
    assert dit._sessions.live("pinned_sess") is None
    assert ar_dec._sessions.live("pinned_sess") is None

    # DiT resume_session rejects old session id cleanly; new sessions work afterward.
    with pytest.raises(ValueError, match="unknown or expired session"):
        dit._session_tail("idle_sess", 2, 64, 64)
    dit._sessions.begin("fresh", "req_3")
    dit._sessions.put("fresh", torch.zeros(1, 8, 2, 4, 4))
    assert "fresh" in dit._sessions

    # 4) Standalone vae_decoder_ar-only rank:
    ar_model, ar_subs = _build_cosmos3(
        dir_a, monkeypatch, (VAE_DECODER_AR_NODE,),
        compile_denoise=False, enable_windowed_video=True, vae_decode_channels_last=False,
    )
    ar_only = ar_subs[VAE_DECODER_AR_NODE]
    ar_only._sessions.put("s1", torch.ones(1, 8, 2, 4, 4))
    rl_only_dir, _ = _write_cosmos3_checkpoint(tmp_path / "rl_only", cfg, seed=72, transformer_only=True)
    ar_engine = _engine(ar_model, ar_subs)
    ar_engine.check_weight_reload(rl_only_dir)
    assert "s1" in ar_only._sessions
    ar_engine.reload_weights(rl_only_dir, "v2")
    assert len(ar_only._sessions) == 0

    # Refuses missing dir and skip_weight_loading=True on vae_decoder_ar-only rank.
    with pytest.raises(WeightReloadRefusedError, match="no checkpoint"):
        ar_engine.check_weight_reload(str(tmp_path / "nope"))
    ar_dummy, ar_dummy_subs = _build_cosmos3(
        dir_a, monkeypatch, (VAE_DECODER_AR_NODE,),
        skip_weight_loading=True, compile_denoise=False, enable_windowed_video=True,
    )
    with pytest.raises(WeightReloadRefusedError, match="not loaded from a checkpoint"):
        _engine(ar_dummy, ar_dummy_subs).check_weight_reload(dir_b)

    # Direct SessionStore.clear() unit test
    s = SessionStore(4, 10.0, 60.0)
    s.put("a", 1)
    s.begin("b", "r")
    assert s.clear() == 2
    assert len(s) == 0


# ── 8. Engine + KVManager prefix cache + WeightOps integration ────────────────


def test_cosmos3_engine_reload_drops_kv_prefix_cache_and_runs_weight_ops(tmp_path, monkeypatch):
    cfg = _tiny_edge_config(action_gen=False, compile_denoise=False)
    dir_a, state_a = _write_cosmos3_checkpoint(tmp_path / "a", cfg, seed=80, include_vae=True)
    dir_b, state_b = _write_cosmos3_checkpoint(tmp_path / "b", cfg, seed=81, include_vae=True)

    model, subs = _build_cosmos3(
        dir_a, monkeypatch, (DIT_NODE, VAE_DECODER_AR_NODE),
        compile_denoise=False, enable_windowed_video=True, vae_decode_channels_last=False,
    )
    subs[DIT_NODE]._sessions.put("s1", torch.ones(1, 8, 2, 4, 4))
    subs[VAE_DECODER_AR_NODE]._sessions.put("s1", torch.ones(1, 8, 2, 4, 4))

    monkeypatch.setattr(kv_manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=8, page_size=8,
        ),
        name="kv",
        joint_comm_group=None,
        transfer_engine_info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"root")
    pages = kv._arena.acquire(2)
    kv._index.insert_run([fingerprint(b"root", k) for k in (b"k0", b"k1")], pages, None)
    kv._arena.release(pages)
    assert kv._index.num_pages() == 2

    # Mid-copy failure: KV prefix cache and SessionStores are already cleared, weights stay A.
    class _MidCopyFail(Cosmos3Model):
        def _apply_transformer_reload(self, target_path, submodules):
            raise OSError("mid-copy read error")

    failing = _MidCopyFail.__new__(_MidCopyFail)
    failing.config = model.config
    failing.skip_weight_loading = False
    failing._transformer_checkpoint = model._transformer_checkpoint
    failing._vae_checkpoint = model._vae_checkpoint
    failing._top_level_configs = model._top_level_configs

    with pytest.raises(OSError, match="mid-copy read error"):
        _engine(failing, subs, kv=kv).reload_weights(dir_b, "v1")
    assert kv._index.num_pages() == 0
    assert kv._prefix_root == fingerprint(b"root", "weights", "v1")
    assert len(subs[DIT_NODE]._sessions) == 0
    assert len(subs[VAE_DECODER_AR_NODE]._sessions) == 0
    assert torch.equal(subs[DIT_NODE].transformer.proj_in.weight, state_a["proj_in.weight"])

    # Successful Engine.reload_weights + WeightOps across 2 mock workers.
    ranks = []
    for r in range(2):
        group = CommGroup(my_global_rank=r, my_group_rank=r, group_members=[0, 1])
        m_r, s_r = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE,), tp_group=group, compile_denoise=False)
        s_r[DIT_NODE]._sessions.put("s_r", torch.ones(1, 8, 2, 4, 4))
        ranks.append((m_r, s_r, _engine(m_r, s_r)))

    dep = _Deployment([ranks[0][2], ranks[1][2]])
    assert dep.pause().status == 200
    dep.control.update(UpdateWeightsFromDisk(op_id="u1", model_path=dir_b, weight_version="v_wo"))
    dep.run(passes=4)
    assert dep.answer("u1").status == 200
    dep.control.resume(ContinueGeneration(op_id="c1"))
    dep.run()
    assert dep.answer("c1").status == 200
    for _, s_r, _ in ranks:
        assert len(s_r[DIT_NODE]._sessions) == 0
        assert torch.equal(s_r[DIT_NODE].transformer.proj_in.weight, state_b["proj_in.weight"])


# ── 9. CPU Eager Forward Parity & Round-Trip ──────────────────────────────────


@torch.no_grad()
def test_cosmos3_cpu_forward_parity_dit_and_reasoner_after_reload(tmp_path, monkeypatch):
    cfg = _tiny_edge_config(action_gen=True, max_action_dim=12, num_embodiment_domains=4, compile_denoise=False)
    dir_a, _ = _write_cosmos3_checkpoint(tmp_path / "a", cfg, seed=90)
    dir_b, _ = _write_cosmos3_checkpoint(tmp_path / "b", cfg, seed=91)

    model, subs = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)
    fresh_b, fresh_b_subs = _build_cosmos3(dir_b, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)
    fresh_a, fresh_a_subs = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)

    dit: Cosmos3DiTSubmodule = subs[DIT_NODE]
    tf = dit.transformer
    res = _SdpaResources()
    res.bind(tf)
    dit.get_device = lambda: torch.device("cpu")

    fresh_b_dit = fresh_b_subs[DIT_NODE]
    res_b = _SdpaResources()
    res_b.bind(fresh_b_dit.transformer)
    fresh_b_dit.get_device = lambda: torch.device("cpu")

    fresh_a_dit = fresh_a_subs[DIT_NODE]
    res_a = _SdpaResources()
    res_a.bind(fresh_a_dit.transformer)
    fresh_a_dit.get_device = lambda: torch.device("cpu")

    static = dit._build_static(list(range(5)), 64, 64, 1, 24.0, False, "cpu")
    a_static = dit._build_action_static(list(range(5)), 64, 64, 1, 4, "policy", 10.0, 10.0, 1, "cpu")
    prompt_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    prompt_pos = torch.arange(5, dtype=torch.long).unsqueeze(0).expand(3, -1)
    latents = torch.randn(dit._latent_shape(64, 64, 1), generator=torch.Generator().manual_seed(7))
    t_gen = torch.full((static["num_noisy_vision_tokens"],), 500.0)
    action_latents = torch.randn(1, 4, cfg.max_action_dim, generator=torch.Generator().manual_seed(8))
    t_act = torch.full((a_static["num_noisy_action_tokens"],), 500.0)
    domain_id = torch.tensor([2], dtype=torch.long)
    text_ids = torch.tensor([10, 12, 13, 14], dtype=torch.long)
    text_pos = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(3, -1)

    def run_dit(tf_mod, res_obj):
        res_obj.kv.committed.clear()
        res_obj.kv.pending.clear()
        res_obj.plan(COND_LABEL, [(COND_LABEL, 5)], causal=True)
        tf_mod.prefill_und(prompt_ids, prompt_pos, COND_LABEL)
        res_obj.kv.promote()
        res_obj.plan(UNCOND_LABEL, [(UNCOND_LABEL, 5)], causal=True)
        tf_mod.prefill_und(prompt_ids, prompt_pos, UNCOND_LABEL)
        res_obj.kv.promote()
        res_obj.plan(
            CFG_BATCHED_LABEL,
            [(COND_LABEL, static["num_vision_tokens"]), (UNCOND_LABEL, static["num_vision_tokens"])],
            causal=False,
        )
        v_cond, v_uncond = tf_mod.denoise_step_batched_cfg(
            latents, t_gen, static["rotary"], static["rotary"],
            static["vision_token_shapes"], static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"], CFG_BATCHED_LABEL, res_obj.attn,
        )
        res_obj.plan(
            COND_LABEL,
            [(COND_LABEL, a_static["num_vision_tokens"] + a_static["num_action_tokens"])],
            causal=False,
        )
        _, a_out = tf_mod.denoise_step(
            latents, t_gen, a_static["rotary"],
            a_static["vision_token_shapes"], a_static["vision_noisy_frame_indexes"],
            a_static["mse_gen_indexes"], COND_LABEL, res_obj.attn,
            action_latents=action_latents,
            action_token_shapes=a_static["action_token_shapes"],
            action_noisy_frame_indexes=a_static["action_noisy_frame_indexes"],
            action_mse_gen_indexes=a_static["action_mse_gen_indexes"],
            action_timesteps=t_act,
            action_domain_id=domain_id,
        )
        return v_cond, v_uncond, a_out

    def run_reasoner(tf_mod, res_obj):
        rkv = _ReasonerSdpaKV()
        _bind_reasoner_kv(tf_mod, rkv)
        embeds = tf_mod.embed_tokens(text_ids)
        cos_sin = tf_mod.build_cos_sin(text_pos, embeds.dtype)
        hidden = tf_mod.text_layers(embeds, *cos_sin, "main")
        out = tf_mod.lm_head(hidden).float()
        res_obj.bind(tf_mod)
        return out

    dit_a = run_dit(tf, res)
    reas_a = run_reasoner(tf, res)

    _engine(model, subs).reload_weights(dir_b, "v1")
    dit_b = run_dit(tf, res)
    reas_b = run_reasoner(tf, res)

    assert not torch.allclose(dit_a[0], dit_b[0])
    assert not torch.allclose(dit_a[2], dit_b[2])
    assert not torch.allclose(reas_a, reas_b)

    res_b.bind(fresh_b_dit.transformer)
    dit_b_exp = run_dit(fresh_b_dit.transformer, res_b)
    reas_b_exp = run_reasoner(fresh_b_dit.transformer, res_b)
    for got, exp in zip(dit_b, dit_b_exp, strict=True):
        torch.testing.assert_close(got, exp, atol=0.0, rtol=0.0)
    torch.testing.assert_close(reas_b, reas_b_exp, atol=0.0, rtol=0.0)

    # Round-trip B -> A matches original A exactly.
    _engine(model, subs).reload_weights(dir_a, "v2")
    dit_rt = run_dit(tf, res)
    reas_rt = run_reasoner(tf, res)
    res_a.bind(fresh_a_dit.transformer)
    dit_a_exp = run_dit(fresh_a_dit.transformer, res_a)
    reas_a_exp = run_reasoner(fresh_a_dit.transformer, res_a)
    for got, exp in zip(dit_rt, dit_a_exp, strict=True):
        torch.testing.assert_close(got, exp, atol=0.0, rtol=0.0)
    torch.testing.assert_close(reas_rt, reas_a_exp, atol=0.0, rtol=0.0)


# ── 10. torch.compile + Static Captured-Buffer Replay Parity ──────────────────


@torch.no_grad()
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_cosmos3_compiled_and_captured_static_replay_parity_after_reload(tmp_path, monkeypatch, backend):
    """CPU static-buffer + ``torch.compiler.set_stance("fail_on_recompile")``
    replay proxy for CUDA-graph static pointer invariance; actual ``CUDAGraph``
    hardware capture cannot execute on a CPU-only host."""
    torch._dynamo.reset()
    cfg = _tiny_edge_config(action_gen=False, compile_denoise=False)
    dir_a, _ = _write_cosmos3_checkpoint(tmp_path / "a", cfg, seed=100)
    dir_b, state_b = _write_cosmos3_checkpoint(tmp_path / "b", cfg, seed=101)

    model, subs = _build_cosmos3(dir_a, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)
    fresh_b, fresh_b_subs = _build_cosmos3(dir_b, monkeypatch, (DIT_NODE, REASONER_NODE), compile_denoise=False)

    dit = subs[DIT_NODE]
    tf = dit.transformer
    res = _SdpaResources()
    res.bind(tf)
    dit.get_device = lambda: torch.device("cpu")

    # Pre-reload static parameter pointers (proxy for captured graph weight pointers).
    static_refs = (
        tf.layers[0].self_attn.to_qkv.weight,
        tf.layers[0].self_attn.add_q_proj.weight,
        tf.layers[0].mlp.up_proj.weight,
        tf.time_embedder.linear_1.weight,
        tf.proj_out.weight,
        tf.lm_head.weight,
    )
    static_ptrs = tuple(t.data_ptr() for t in static_refs)

    compiled_denoise_step = torch.compile(tf.denoise_step, backend=backend, fullgraph=False, dynamic=False)
    compiled_denoise_cfg = torch.compile(tf.denoise_step_batched_cfg, backend=backend, fullgraph=False, dynamic=False)
    compiled_gen_layers = torch.compile(tf.gen_layers_batched, backend=backend, fullgraph=False, dynamic=None)
    compiled_text_layers = torch.compile(tf.text_layers, backend=backend, fullgraph=False, dynamic=False)

    # Static pre-allocated input buffers matching CAPTURED_GEN_INPUTS.
    static = dit._build_static(list(range(5)), 64, 64, 1, 24.0, False, "cpu")
    assert set(CAPTURED_GEN_INPUTS) == {
        "latents", "vision_timesteps", "cos_cond", "sin_cond", "cos_uncond", "sin_uncond", "noisy_token_mask",
    }
    buf_latents = torch.randn(dit._latent_shape(64, 64, 1), generator=torch.Generator().manual_seed(11))
    buf_timesteps = torch.full((static["num_noisy_vision_tokens"],), 500.0)
    buf_cos_cond = static["rotary"][0].clone()
    buf_sin_cond = static["rotary"][1].clone()
    buf_cos_uncond = static["rotary"][0].clone()
    buf_sin_uncond = static["rotary"][1].clone()
    buf_noisy_mask = torch.ones(static["num_vision_tokens"], dtype=torch.bool)
    buf_text_embeds = torch.randn(4, cfg.hidden_size, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(12))
    text_pos = torch.arange(4, dtype=torch.long).unsqueeze(0).expand(3, -1)
    buf_text_cos, buf_text_sin = tf.build_cos_sin(text_pos, torch.bfloat16)

    prompt_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    prompt_pos = torch.arange(5, dtype=torch.long).unsqueeze(0).expand(3, -1)
    buf_gen_x = torch.randn(
        2 * int(buf_noisy_mask.sum().item()), cfg.hidden_size, dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(13),
    )

    def run_compiled(tf_mod, res_obj, *, use_compiled: bool):
        res_obj.kv.committed.clear()
        res_obj.kv.pending.clear()
        res_obj.plan(COND_LABEL, [(COND_LABEL, 5)], causal=True)
        tf_mod.prefill_und(prompt_ids, prompt_pos, COND_LABEL)
        res_obj.kv.promote()
        res_obj.plan(UNCOND_LABEL, [(UNCOND_LABEL, 5)], causal=True)
        tf_mod.prefill_und(prompt_ids, prompt_pos, UNCOND_LABEL)
        res_obj.kv.promote()

        # 1) Single-branch denoise_step
        res_obj.plan(COND_LABEL, [(COND_LABEL, static["num_vision_tokens"])], causal=False)
        fn_step = compiled_denoise_step if use_compiled else tf_mod.denoise_step
        step_out = fn_step(
            buf_latents, buf_timesteps, (buf_cos_cond, buf_sin_cond),
            static["vision_token_shapes"], static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"], COND_LABEL, res_obj.attn,
        )

        # 2) Batched CFG denoise_step_batched_cfg
        res_obj.plan(
            CFG_BATCHED_LABEL,
            [(COND_LABEL, static["num_vision_tokens"]), (UNCOND_LABEL, static["num_vision_tokens"])],
            causal=False,
        )
        fn_cfg = compiled_denoise_cfg if use_compiled else tf_mod.denoise_step_batched_cfg
        cfg_cond, cfg_uncond = fn_cfg(
            buf_latents, buf_timesteps, (buf_cos_cond, buf_sin_cond), (buf_cos_uncond, buf_sin_uncond),
            static["vision_token_shapes"], static["vision_noisy_frame_indexes"],
            static["mse_gen_indexes"], CFG_BATCHED_LABEL, res_obj.attn,
        )

        # 3) gen_layers_batched with static-buffer mask shape
        tf_mod._gen_attend.bind_step(attn=res_obj.attn, label=CFG_BATCHED_LABEL)
        gen_cos = torch.cat([buf_cos_cond, buf_cos_uncond], dim=0)
        gen_sin = torch.cat([buf_sin_cond, buf_sin_uncond], dim=0)
        fn_layers = compiled_gen_layers if use_compiled else tf_mod.gen_layers_batched
        layers_out = fn_layers(buf_gen_x, gen_cos, gen_sin)

        # 4) Reasoner text_layers + lm_head
        rkv = _ReasonerSdpaKV()
        _bind_reasoner_kv(tf_mod, rkv)
        fn_text = compiled_text_layers if use_compiled else tf_mod.text_layers
        text_hidden = fn_text(buf_text_embeds, buf_text_cos, buf_text_sin, "main")
        logits = tf_mod.lm_head(text_hidden).float()
        res_obj.bind(tf_mod)
        return step_out, cfg_cond, cfg_uncond, layers_out, logits

    out_a = run_compiled(tf, res, use_compiled=True)

    _engine(model, subs).reload_weights(dir_b, "v1")
    assert tuple(t.data_ptr() for t in static_refs) == static_ptrs
    assert torch.equal(static_refs[4], state_b["proj_out.weight"])

    with torch.compiler.set_stance("fail_on_recompile"):
        out_b = run_compiled(tf, res, use_compiled=True)

    for a, b in zip(out_a, out_b, strict=True):
        assert not torch.allclose(a, b)

    res_fb = _SdpaResources()
    res_fb.bind(fresh_b_subs[DIT_NODE].transformer)
    exp_b = run_compiled(fresh_b_subs[DIT_NODE].transformer, res_fb, use_compiled=False)
    tol = 2e-2 if backend == "inductor" else 1e-5
    for got, exp in zip(out_b, exp_b, strict=True):
        torch.testing.assert_close(got, exp, atol=tol, rtol=tol)


# ── 11. Auxiliary submodules (VAE, vision, sound) & RL transformer-only ───────


def test_cosmos3_auxiliary_submodules_reload_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("COSMOS3_COMPILE_VAE", "0")
    cfg = _tiny_edge_config(action_gen=False, sound_gen=True, sound_dim=8, compile_denoise=False)
    dir_a, _ = _write_cosmos3_checkpoint(
        tmp_path / "a", cfg, seed=110, include_vae=True, include_vision=True, include_sound=True,
    )
    dir_b, state_b = _write_cosmos3_checkpoint(
        tmp_path / "b", cfg, seed=111, include_vae=True, include_vision=True, include_sound=True,
    )
    rl_dir, state_rl = _write_cosmos3_checkpoint(
        tmp_path / "rl_tf_only", cfg, seed=112, transformer_only=True,
    )

    all_nodes = (
        DIT_NODE, REASONER_NODE, VISION_ENCODER_NODE,
        VAE_ENCODER_NODE, VAE_DECODER_NODE, VAE_DECODER_AR_NODE, AUDIO_DECODER_NODE,
    )
    model, subs = _build_cosmos3(
        dir_a, monkeypatch, all_nodes,
        compile_denoise=False, enable_windowed_video=True, vae_decode_fast=True,
        vae_decode_channels_last=True, sound_sample_rate=48000,
    )
    fresh_b, fresh_b_subs = _build_cosmos3(
        dir_b, monkeypatch, all_nodes,
        compile_denoise=False, enable_windowed_video=True, vae_decode_fast=True,
        vae_decode_channels_last=True, sound_sample_rate=48000,
    )

    dit = subs[DIT_NODE]
    vis = subs[VISION_ENCODER_NODE]
    vae_enc = subs[VAE_ENCODER_NODE]
    vae_dec = subs[VAE_DECODER_NODE]
    vae_ar = subs[VAE_DECODER_AR_NODE]
    aud = subs[AUDIO_DECODER_NODE]

    # Verify VAE sharing and channels-last decoder-only setup.
    assert dit._window_head is not None
    assert vae_dec.vae is vae_ar.vae is dit._window_head.decoder.vae
    assert vae_enc.vae is dit._window_head.encode_vae
    assert vae_dec.vae.encoder is None
    assert is_channels_last(vae_dec.vae)

    # Populate vision_encoder._position_tables + SessionStores.
    _ = vis._position_table(2, 2)
    assert len(vis._position_tables) == 1 and vis._position_table_bytes > 0
    dit._sessions.put("s_aux", torch.ones(1, 8, 2, 4, 4))
    vae_ar._sessions.put("s_aux", torch.ones(1, 8, 2, 4, 4))

    before = _storage_snapshot(subs)
    engine = _engine(model, subs)

    # 1) check_only=True preserves _position_tables and SessionStores.
    engine.check_weight_reload(dir_b)
    assert len(vis._position_tables) == 1 and vis._position_table_bytes > 0
    assert "s_aux" in dit._sessions and "s_aux" in vae_ar._sessions

    # 2) Incompatible auxiliary components refuse WITHOUT clearing _position_tables / _sessions or copying.
    bad_vis_dir = tmp_path / "bad_vis"
    shutil.copytree(dir_b, bad_vis_dir)
    save_file({"model.unexpected": torch.ones(2, 2)}, str(bad_vis_dir / "vision_encoder" / "model.safetensors"))
    with pytest.raises(WeightReloadRefusedError):
        engine.reload_weights(str(bad_vis_dir), "v_bad_vis")
    assert len(vis._position_tables) == 1 and "s_aux" in dit._sessions and "s_aux" in vae_ar._sessions

    bad_vae_dir = tmp_path / "bad_vae"
    shutil.copytree(dir_b, bad_vae_dir)
    vae_cfg = json.loads((bad_vae_dir / "vae" / "config.json").read_text())
    vae_cfg["z_dim"] = 999
    (bad_vae_dir / "vae" / "config.json").write_text(json.dumps(vae_cfg))
    with pytest.raises(WeightReloadRefusedError, match="z_dim"):
        engine.reload_weights(str(bad_vae_dir), "v_bad_vae")
    assert len(vis._position_tables) == 1 and "s_aux" in dit._sessions and "s_aux" in vae_ar._sessions

    # 3) Full reload updates transformer, vision_encoder, VAEs, sound_tokenizer in place,
    # deduplicating shared VAEs and clearing _position_tables + SessionStores.
    vae_calls = []
    orig_load_vae = cosmos3_loader.load_vae_path

    def spy_vae(vae, path, device="cpu"):
        vae_calls.append(id(vae))
        return orig_load_vae(vae, path, device=device)

    monkeypatch.setattr(cosmos3_loader, "load_vae_path", spy_vae)
    engine.reload_weights(dir_b, "v1")
    assert len(vae_calls) == 2  # once for shared decode VAE, once for shared encode VAE
    assert _storage_snapshot(subs) == before
    assert len(vis._position_tables) == 0 and vis._position_table_bytes == 0
    assert len(dit._sessions) == 0 and len(vae_ar._sessions) == 0
    assert is_channels_last(vae_dec.vae)

    # Vision encoder output & position_table match fresh B.
    with torch.no_grad():
        assert torch.equal(vis.model.position_table(2, 2), fresh_b_subs[VISION_ENCODER_NODE].model.position_table(2, 2))
        patches = torch.randn(4, 3 * 16 * 16, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(1))
        assert torch.equal(
            vis.model(patches, [(1, 2, 2)]),
            fresh_b_subs[VISION_ENCODER_NODE].model(patches, [(1, 2, 2)]),
        )

        # VAE decode output matches fresh B.
        z = torch.randn(1, 8, 1, 4, 4, generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(vae_dec._decode_pixels(z), fresh_b_subs[VAE_DECODER_NODE]._decode_pixels(z))

        # Sound tokenizer (weight_norm weight_g/weight_v) output matches fresh B.
        s_lat = torch.randn(1, 8, 3, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(
            aud.sound_tokenizer.decode(s_lat),
            fresh_b_subs[AUDIO_DECODER_NODE].sound_tokenizer.decode(s_lat),
        )

    # 4) transformer/-only RL checkpoint leaves frozen VAE / vision / sound untouched
    # (and does NOT falsely refuse dit._window_head or clear _position_tables!).
    _ = vis._position_table(2, 2)
    vis_ptr = vis.model.visual.embeddings.position_embedding.weight.clone()
    vae_ptr = vae_dec.vae.decoder.conv_in.weight.clone()
    aud_ptr = aud.sound_tokenizer.decoder.conv1.weight_v.clone()
    dit._sessions.put("s_rl", torch.ones(1, 8, 2, 4, 4))
    vae_ar._sessions.put("s_rl", torch.ones(1, 8, 2, 4, 4))

    engine.reload_weights(rl_dir, "v2")
    assert _storage_snapshot(subs) == before
    assert torch.equal(dit.transformer.proj_in.weight, state_rl["proj_in.weight"])
    assert torch.equal(vis.model.visual.embeddings.position_embedding.weight, vis_ptr)
    assert torch.equal(vae_dec.vae.decoder.conv_in.weight, vae_ptr)
    assert torch.equal(aud.sound_tokenizer.decoder.conv1.weight_v, aud_ptr)
    assert len(vis._position_tables) == 1  # vision_encoder was not reloaded
    assert len(dit._sessions) == 0 and len(vae_ar._sessions) == 0

    # Also verify public load_vae_weights / load_sound_tokenizer_weights helpers directly.
    cosmos3_loader.load_vae_weights(vae_dec.vae, dir_b, device="cpu")
    cosmos3_loader.load_sound_tokenizer_weights(aud.sound_tokenizer, dir_b, device="cpu")
    assert _storage_snapshot(subs) == before
    assert not math.isnan(float(state_b["proj_in.weight"].float().sum()))
