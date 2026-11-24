"""In-place weight reload for Pi0.5 and shared opt-in ``Model.reload_weights``.

Exercises ``Pi05Model`` + ``Engine.reload_weights`` on CPU with tiny explicit
``Pi05Config`` checkpoints:
  - ``data_ptr``, shape, and mixed ``float32`` / ``bfloat16`` dtype preservation
    across all parameters and non-persistent buffers (SigLIP ``position_ids``)
  - config-less ``RLinf-Pi05-LIBERO-SFT``-style reload & pinned-horizon startup
  - fused ``qkv_proj`` / ``gate_up_proj``, ``time_mlp``, tied ``embed_tokens``,
    and unfused SigLIP ``q/k/v_proj``
  - ``_lerobot_flat`` bypass and startup snapshot preservation after delete/overwrite
  - ``PI05Policy`` ``model.``-prefixed interoperability and sharded index loading
  - rank-split partial submodule hosting (``vit_encoder`` vs ``LLM``+``action_expert``)
  - full CPU forward-pass parity (ViT -> PaliGemma -> ActionExpert) & round-trip
  - ``torch.compile`` replay under ``torch.compiler.set_stance("fail_on_recompile")``
  - negative error paths (nothing copied) and shared opt-in ``Model.reload_weights``
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from mstar.engine.engine import Engine
from mstar.engine.node_dtype import resolve_node_dtype
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.keys import fingerprint
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle
from mstar.model.base import Model
from mstar.model.loader.reload import WeightReloadRefusedError, snapshot_checkpoint
from mstar.model.orpheus.orpheus_model import OrpheusModel
from mstar.model.pi05.components.siglip import Pi05SiglipEncoder
from mstar.model.pi05.components.tokenization import Pi05Tokenizer
from mstar.model.pi05.config import CHUNK_WALK, LLM_ATTN, LLM_KV, Pi05Config
from mstar.model.pi05.pi05_model import Pi05Model
from mstar.model.pi05.submodules import (
    Pi05ActionSubmodule,
    Pi05LLMSubmodule,
    Pi05ViTEncoderSubmodule,
)

ROPE = "rope"

TINY_CFG = dict(
    vit_hidden_size=8,
    vit_num_layers=1,
    vit_num_heads=2,
    vit_intermediate_size=16,
    vit_patch_size=14,
    vit_image_size=28,
    tokens_per_image=4,
    num_cameras=1,
    num_layers=1,
    num_qo_heads=2,
    num_kv_heads=1,
    head_dim=8,
    hidden_size=16,
    pali_intermediate_size=32,
    vocab_size=32,
    pad_token_id=0,
    action_hidden_size=8,
    action_intermediate_size=16,
    num_flow_steps=3,
    action_horizon=6,
    action_dim=4,
    state_dim=4,
    max_lang_tokens=16,
    max_position_embeddings=64,
)


def _tiny_config(**overrides) -> Pi05Config:
    return Pi05Config(**{**TINY_CFG, **overrides})


def _hf_config_dict(cfg: Pi05Config, **extra) -> dict:
    return {
        "type": "pi05",
        "chunk_size": cfg.action_horizon,
        "n_action_steps": cfg.action_horizon,
        "action_dim": cfg.action_dim,
        "max_state_dim": cfg.state_dim,
        "num_inference_steps": cfg.num_flow_steps,
        "num_hidden_layers": cfg.num_layers,
        "num_attention_heads": cfg.num_qo_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.pali_intermediate_size,
        "vocab_size": cfg.vocab_size,
        "vit_hidden_size": cfg.vit_hidden_size,
        "vit_num_layers": cfg.vit_num_layers,
        "vit_num_heads": cfg.vit_num_heads,
        "vit_intermediate_size": cfg.vit_intermediate_size,
        "vit_patch_size": cfg.vit_patch_size,
        "vit_image_size": cfg.vit_image_size,
        "tokens_per_image": cfg.tokens_per_image,
        "num_cameras": cfg.num_cameras,
        "action_hidden_size": cfg.action_hidden_size,
        "action_intermediate_size": cfg.action_intermediate_size,
        "dtype": "float32",
        "device": "mps",
        **extra,
    }


def _lerobot_state(
    cfg: Pi05Config,
    seed: int,
    *,
    dtype: torch.dtype = torch.float32,
    mixed_rlinf: bool = False,
    policy_prefixed: bool = False,
) -> dict[str, torch.Tensor]:
    """Generate a complete lerobot Pi0.5 state dict for ``cfg``."""
    g = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float = 0.25, force_fp32: bool = False) -> torch.Tensor:
        t = torch.randn(*shape, generator=g) * scale
        target_dtype = torch.float32 if (mixed_rlinf and force_fp32) else dtype
        return t.to(target_dtype)

    state: dict[str, torch.Tensor] = {}
    pali = "paligemma_with_expert.paligemma.model"
    ge = "paligemma_with_expert.gemma_expert.model"

    # SigLIP + connector (using installed transformers' SiglipVisionModel layout)
    with torch.device("meta"):
        siglip = Pi05SiglipEncoder(cfg)
    tower_prefix = Pi05Model._siglip_tower_prefix(siglip)
    for name, param in siglip.named_parameters():
        keep_fp32 = "embeddings.patch_embedding." in name or "embeddings.position_embedding." in name
        if name.startswith(tower_prefix):
            key = f"{pali}.vision_tower.vision_model.{name.removeprefix(tower_prefix)}"
        else:
            key = f"{pali}.multi_modal_projector.linear.{name.removeprefix('connector.')}"
        state[key] = rand(*param.shape, force_fp32=keep_fp32)

    # PaliGemma lm_head (tied to embed_tokens) + norm + layers
    state["paligemma_with_expert.paligemma.lm_head.weight"] = rand(cfg.vocab_size, cfg.hidden_size)
    state[f"{pali}.language_model.norm.weight"] = rand(cfg.hidden_size, force_fp32=True)
    for i in range(cfg.num_layers):
        p = f"{pali}.language_model.layers.{i}."
        state[p + "self_attn.q_proj.weight"] = rand(cfg.num_qo_heads * cfg.head_dim, cfg.hidden_size)
        state[p + "self_attn.k_proj.weight"] = rand(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
        state[p + "self_attn.v_proj.weight"] = rand(cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
        state[p + "self_attn.o_proj.weight"] = rand(cfg.hidden_size, cfg.num_qo_heads * cfg.head_dim)
        state[p + "mlp.gate_proj.weight"] = rand(cfg.pali_intermediate_size, cfg.hidden_size)
        state[p + "mlp.up_proj.weight"] = rand(cfg.pali_intermediate_size, cfg.hidden_size)
        state[p + "mlp.down_proj.weight"] = rand(cfg.hidden_size, cfg.pali_intermediate_size)
        state[p + "input_layernorm.weight"] = rand(cfg.hidden_size, force_fp32=True)
        state[p + "post_attention_layernorm.weight"] = rand(cfg.hidden_size, force_fp32=True)

    # Action expert + projections + time_mlp
    ah = cfg.action_hidden_size
    state[f"{ge}.norm.dense.weight"] = rand(3 * ah, ah, force_fp32=True)
    state[f"{ge}.norm.dense.bias"] = rand(3 * ah, force_fp32=True)
    for i in range(cfg.num_layers):
        p = f"{ge}.layers.{i}."
        state[p + "self_attn.q_proj.weight"] = rand(cfg.num_qo_heads * cfg.head_dim, ah)
        state[p + "self_attn.k_proj.weight"] = rand(cfg.num_kv_heads * cfg.head_dim, ah)
        state[p + "self_attn.v_proj.weight"] = rand(cfg.num_kv_heads * cfg.head_dim, ah)
        state[p + "self_attn.o_proj.weight"] = rand(ah, cfg.num_qo_heads * cfg.head_dim)
        state[p + "mlp.gate_proj.weight"] = rand(cfg.action_intermediate_size, ah)
        state[p + "mlp.up_proj.weight"] = rand(cfg.action_intermediate_size, ah)
        state[p + "mlp.down_proj.weight"] = rand(ah, cfg.action_intermediate_size)
        state[p + "input_layernorm.dense.weight"] = rand(3 * ah, ah, force_fp32=True)
        state[p + "input_layernorm.dense.bias"] = rand(3 * ah, force_fp32=True)
        state[p + "post_attention_layernorm.dense.weight"] = rand(3 * ah, ah, force_fp32=True)
        state[p + "post_attention_layernorm.dense.bias"] = rand(3 * ah, force_fp32=True)

    state["action_in_proj.weight"] = rand(ah, cfg.action_dim)
    state["action_in_proj.bias"] = rand(ah)
    state["action_out_proj.weight"] = rand(cfg.action_dim, ah)
    state["action_out_proj.bias"] = rand(cfg.action_dim)
    state["time_mlp_in.weight"] = rand(ah, ah, force_fp32=True)
    state["time_mlp_in.bias"] = rand(ah, force_fp32=True)
    state["time_mlp_out.weight"] = rand(ah, ah, force_fp32=True)
    state["time_mlp_out.bias"] = rand(ah, force_fp32=True)

    if policy_prefixed:
        lm_head = state["paligemma_with_expert.paligemma.lm_head.weight"]
        prefixed = {"model." + k: v for k, v in state.items()}
        prefixed[f"model.{pali}.language_model.embed_tokens.weight"] = lm_head.clone()
        return prefixed
    return state


def _write_pi05_checkpoint(
    root: Path,
    state: dict[str, torch.Tensor],
    cfg: Pi05Config | None = None,
    *,
    include_config: bool = True,
    extra_config: dict | None = None,
    sharded: bool = False,
    norm_stats_subpath: str | None = None,
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    if sharded:
        keys = sorted(state)
        shards = {
            "model-00001-of-00002.safetensors": keys[::2],
            "model-00002-of-00002.safetensors": keys[1::2],
        }
        for shard_name, shard_keys in shards.items():
            save_file({k: state[k] for k in shard_keys}, str(root / shard_name))
        weight_map = {k: shard_name for shard_name, shard_keys in shards.items() for k in shard_keys}
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(state, str(root / "model.safetensors"))

    if include_config and cfg is not None:
        (root / "config.json").write_text(json.dumps(_hf_config_dict(cfg, **(extra_config or {}))))

    if norm_stats_subpath is not None:
        stats_file = root / norm_stats_subpath
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.write_text(json.dumps({
            "norm_stats": {
                "state": {"q01": [0.0] * 4, "q99": [1.0] * 4},
                "actions": {"q01": [-1.0] * 4, "q99": [1.0] * 4},
            }
        }))
    return str(root)


class _StubTokenizer(Pi05Tokenizer):
    def __init__(self, cfg: Pi05Config):
        self.config = cfg

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return torch.tensor([1, 2, 3, 4], dtype=torch.long)


def _build_pi05(
    ckpt_dir: str,
    monkeypatch,
    *,
    nodes: tuple[str, ...] = ("vit_encoder", "LLM", "action_expert"),
    cast_mixed_bf16: bool = True,
    **model_kwargs,
) -> tuple[Pi05Model, dict[str, nn.Module]]:
    monkeypatch.setattr(
        Pi05Model,
        "_load_tokenizer",
        lambda self: None if self.skip_weight_loading else _StubTokenizer(self.config),
    )
    model = Pi05Model(ckpt_dir, **model_kwargs)
    submodules: dict[str, nn.Module] = {}
    for node in nodes:
        sub = model.get_submodule(node, device="cpu")
        assert sub is not None
        if cast_mixed_bf16:
            node_dtype = resolve_node_dtype(sub, torch.bfloat16)
            sub = sub.to(device="cpu", dtype=node_dtype)
        sub.eval()
        submodules[node] = sub
    return model, submodules


def _engine(model: Model, submodules: dict[str, nn.Module], kv: KVManager | None = None) -> Engine:
    engine = Engine.__new__(Engine)
    engine._model = model
    engine._device = torch.device("cpu")
    engine._submodules = {
        name: SimpleNamespace(submodule=sub) for name, sub in submodules.items()
    }
    engine._resources = {} if kv is None else {"kv": kv}
    engine._resource_nodes = {} if kv is None else {"kv": frozenset(submodules) & {"LLM", "action_expert"}}
    return engine


def _storage_snapshot(submodules: dict[str, nn.Module]) -> dict[str, tuple[int, tuple[int, ...], torch.dtype]]:
    out = {
        f"{node}.{name}": (p.data_ptr(), tuple(p.shape), p.dtype)
        for node, sub in submodules.items()
        for name, p in sub.named_parameters()
    }
    for node, sub in submodules.items():
        for name, buf in sub.named_buffers():
            out[f"{node}.__buffer__.{name}"] = (buf.data_ptr(), tuple(buf.shape), buf.dtype)
    return out


def _unprefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state or not all(k.startswith("model.") for k in state):
        return state
    out = {k.removeprefix("model."): v for k, v in state.items()}
    out.pop("paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight", None)
    return out


# ── 1. Mixed fp32/bf16 reload, pointer/buffer invariance, config-less RLinf ──


def test_pi05_reload_rlinf_configless_preserves_pointers_buffers_and_mixed_dtypes(tmp_path, monkeypatch):
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=10, dtype=torch.float32)
    state_b = _lerobot_state(cfg, seed=20, dtype=torch.bfloat16, mixed_rlinf=True)

    dir_a = _write_pi05_checkpoint(tmp_path / "pi05_base", state_a, cfg)
    dir_b = _write_pi05_checkpoint(
        tmp_path / "RLinf-Pi05-LIBERO-SFT",
        state_b,
        include_config=False,
        norm_stats_subpath="physical-intelligence/libero/norm_stats.json",
    )

    model, subs = _build_pi05(dir_a, monkeypatch)
    assert model._checkpoint is not None and not isinstance(model._checkpoint, str)

    before = _storage_snapshot(subs)
    assert any(".__buffer__." in k and k.endswith(".embeddings.position_ids") for k in before)

    # Expected mixed precision after EngineManager.build (.to(bf16) with fp32 selectors)
    vit = subs["vit_encoder"]
    assert vit.encoder.tower.embeddings.patch_embedding.weight.dtype == torch.float32
    assert vit.encoder.tower.embeddings.position_embedding.weight.dtype == torch.float32
    assert vit.encoder.connector.weight.dtype == torch.bfloat16
    assert subs["LLM"].paligemma.norm.weight.dtype == torch.float32
    assert subs["LLM"].paligemma.layers[0].self_attn.qkv_proj.weight.dtype == torch.bfloat16
    assert subs["action_expert"].time_mlp.linear_in.weight.dtype == torch.float32
    assert subs["action_expert"].action_in_proj.weight.dtype == torch.bfloat16

    engine = _engine(model, subs)
    engine.check_weight_reload(dir_b)
    # check_only must leave startup weights intact
    assert torch.equal(
        subs["action_expert"].action_in_proj.weight,
        state_a["action_in_proj.weight"].to(torch.bfloat16),
    )

    engine.reload_weights(dir_b, "v1")

    after = _storage_snapshot(subs)
    assert after == before, "reload moved or cast a parameter or buffer"
    assert torch.equal(
        subs["action_expert"].action_in_proj.weight,
        state_b["action_in_proj.weight"].to(torch.bfloat16),
    )
    assert torch.equal(
        subs["action_expert"].time_mlp.linear_in.weight,
        state_b["time_mlp_in.weight"].to(torch.float32),
    )
    assert torch.equal(
        vit.encoder.tower.embeddings.patch_embedding.weight,
        state_b["paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"].to(
            torch.float32
        ),
    )


def test_pi05_configless_local_startup_requires_pinned_action_horizon(tmp_path, monkeypatch):
    cfg = _tiny_config()
    state = _lerobot_state(cfg, seed=1, dtype=torch.bfloat16, mixed_rlinf=True)
    dir_rlinf = _write_pi05_checkpoint(
        tmp_path / "rlinf_local",
        state,
        include_config=False,
        norm_stats_subpath="physical-intelligence/libero/norm_stats.json",
    )
    monkeypatch.setattr(Pi05Model, "_load_tokenizer", lambda self: _StubTokenizer(self.config))

    with pytest.raises(RuntimeError, match="config.json"):
        Pi05Model(dir_rlinf)

    # With action_horizon (and tiny geometry) pinned in YAML overrides, startup succeeds.
    model = Pi05Model(dir_rlinf, **TINY_CFG)
    assert model.config.action_horizon == TINY_CFG["action_horizon"]
    vit = model.get_submodule("vit_encoder", device="cpu")
    assert vit is not None and model._checkpoint is not None


def test_pi05_offline_uncached_hf_repo_refuses_even_with_pinned_action_horizon(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(Pi05Model, "_load_tokenizer", lambda self: _StubTokenizer(self.config))
    empty_cache = tmp_path / "empty_hf_cache"
    empty_cache.mkdir()

    # Unmocked HF_HUB_OFFLINE=1 on an uncached HF repo must fail even with action_horizon pinned.
    with pytest.raises(RuntimeError, match="could not read the config.json"):
        Pi05Model(
            "mstar-nonexistent-org/nonexistent-pi05-repo-12345",
            cache_dir=str(empty_cache),
            action_horizon=15,
        )

    # Explicit LocalEntryNotFoundError when no local snapshot exists also refuses.
    def _raise_local_entry(*args, **kwargs):
        raise LocalEntryNotFoundError("not in local cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _raise_local_entry)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _raise_local_entry)
    with pytest.raises(RuntimeError, match="could not read the config.json"):
        Pi05Model("lerobot/pi05_droid", cache_dir=str(empty_cache), action_horizon=15)
    with pytest.raises(RuntimeError, match="could not read the config.json"):
        Pi05Model("lerobot/pi05_droid", cache_dir=str(empty_cache))


def test_pi05_hf_remote_404_and_cached_configless_snapshot_require_pinned_action_horizon(tmp_path, monkeypatch):
    import huggingface_hub
    from httpx import Request, Response
    from huggingface_hub.errors import RemoteEntryNotFoundError

    monkeypatch.setattr(Pi05Model, "_load_tokenizer", lambda self: _StubTokenizer(self.config))

    # 1. RemoteEntryNotFoundError (Hub 404): fails without action_horizon, succeeds with it.
    def _raise_remote_404(*args, **kwargs):
        resp = Response(404, request=Request("GET", "https://huggingface.co/org/rlinf/resolve/main/config.json"))
        raise RemoteEntryNotFoundError("404 Client Error", response=resp)

    with monkeypatch.context() as mp:
        mp.setattr(huggingface_hub, "hf_hub_download", _raise_remote_404)
        with pytest.raises(RuntimeError, match="could not read the config.json"):
            Pi05Model("org/rlinf-configless", cache_dir=str(tmp_path))

        model_remote = Pi05Model("org/rlinf-configless", cache_dir=str(tmp_path), action_horizon=15, action_dim=7)
        assert (model_remote.config.action_horizon, model_remote.config.action_dim) == (15, 7)

    # 2. Unmocked locally cached HF snapshot genuinely lacking config.json under HF_HUB_OFFLINE=1.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    hf_cache = tmp_path / "hf_cache"
    commit = "a" * 40
    repo_cache = hf_cache / "models--org--rlinf-cached"
    snap_dir = repo_cache / "snapshots" / commit
    refs_dir = repo_cache / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text(commit)
    cfg = _tiny_config()
    state = _lerobot_state(cfg, seed=2, dtype=torch.bfloat16, mixed_rlinf=True)
    _write_pi05_checkpoint(snap_dir, state, include_config=False)

    with pytest.raises(RuntimeError, match="could not read the config.json"):
        Pi05Model("org/rlinf-cached", cache_dir=str(hf_cache))

    model_cached = Pi05Model("org/rlinf-cached", cache_dir=str(hf_cache), **TINY_CFG)
    assert model_cached.config.action_horizon == TINY_CFG["action_horizon"]
    vit = model_cached.get_submodule("vit_encoder", device="cpu")
    assert vit is not None and model_cached._checkpoint is not None


def test_engine_reload_weights_catches_buffer_mutation(tmp_path, monkeypatch):
    cfg = _tiny_config()
    dir_a = _write_pi05_checkpoint(tmp_path / "a", _lerobot_state(cfg, 0), cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", _lerobot_state(cfg, 1), cfg)
    model, subs = _build_pi05(dir_a, monkeypatch, nodes=("vit_encoder",))

    class _RebindingVit(Pi05Model):
        def _apply_reload(self, path, submodules):
            super()._apply_reload(path, submodules)
            emb = submodules["vit_encoder"].encoder.tower.embeddings
            emb.position_ids = emb.position_ids.clone()

    rebinding = _RebindingVit.__new__(_RebindingVit)
    rebinding._checkpoint = model._checkpoint
    rebinding._lerobot_flat = None
    with pytest.raises(RuntimeError, match="__buffer__.*position_ids"):
        _engine(rebinding, subs).reload_weights(dir_b, "v1")


# ── 2. Fused layers, tied embed_tokens, time_mlp, SigLIP unfused ─────────────


def test_pi05_reload_routes_fused_layers_tied_embed_and_siglip_unfused(tmp_path, monkeypatch):
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=3)
    state_b = _lerobot_state(cfg, seed=4)
    dir_a = _write_pi05_checkpoint(tmp_path / "a", state_a, cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", state_b, cfg)

    model, subs = _build_pi05(dir_a, monkeypatch, cast_mixed_bf16=False)
    _engine(model, subs).reload_weights(dir_b, "v1")

    pali = "paligemma_with_expert.paligemma.model.language_model.layers.0."
    ge = "paligemma_with_expert.gemma_expert.model.layers.0."
    assert torch.equal(
        subs["LLM"].paligemma.layers[0].self_attn.qkv_proj.weight,
        torch.cat([
            state_b[pali + "self_attn.q_proj.weight"],
            state_b[pali + "self_attn.k_proj.weight"],
            state_b[pali + "self_attn.v_proj.weight"],
        ]),
    )
    assert torch.equal(
        subs["LLM"].paligemma.layers[0].mlp.gate_up_proj.weight,
        torch.cat([state_b[pali + "mlp.gate_proj.weight"], state_b[pali + "mlp.up_proj.weight"]]),
    )
    assert torch.equal(
        subs["action_expert"].action_expert.layers[0].self_attn.qkv_proj.weight,
        torch.cat([
            state_b[ge + "self_attn.q_proj.weight"],
            state_b[ge + "self_attn.k_proj.weight"],
            state_b[ge + "self_attn.v_proj.weight"],
        ]),
    )
    assert torch.equal(
        subs["action_expert"].action_expert.layers[0].mlp.gate_up_proj.weight,
        torch.cat([state_b[ge + "mlp.gate_proj.weight"], state_b[ge + "mlp.up_proj.weight"]]),
    )
    assert torch.equal(
        subs["LLM"].embed_tokens.weight,
        state_b["paligemma_with_expert.paligemma.lm_head.weight"],
    )
    assert torch.equal(subs["action_expert"].time_mlp.linear_in.weight, state_b["time_mlp_in.weight"])
    assert torch.equal(subs["action_expert"].time_mlp.linear_out.bias, state_b["time_mlp_out.bias"])

    siglip_attn = subs["vit_encoder"].encoder.tower.encoder.layers[0].self_attn
    vit_p = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn."
    assert torch.equal(siglip_attn.q_proj.weight, state_b[vit_p + "q_proj.weight"])
    assert torch.equal(siglip_attn.k_proj.weight, state_b[vit_p + "k_proj.weight"])
    assert torch.equal(siglip_attn.v_proj.weight, state_b[vit_p + "v_proj.weight"])


# ── 3. _lerobot_flat bypass & startup snapshot preservation ──────────────────


def test_pi05_reload_bypasses_lerobot_flat_and_survives_startup_delete_or_overwrite(tmp_path, monkeypatch):
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=5)
    state_b = _lerobot_state(cfg, seed=6)
    dir_start = _write_pi05_checkpoint(tmp_path / "start", state_a, cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", state_b, cfg)

    model, subs = _build_pi05(dir_start, monkeypatch, cast_mixed_bf16=False)
    # Poison _lerobot_flat with stale/sentinel tensors; reload must ignore and clear it.
    model._lerobot_flat = {k: torch.full_like(v, -999.0) for k, v in state_a.items()}

    # Overwriting startup dir in-place with an incompatible shape must be refused
    # against the startup snapshot rather than comparing the overwritten dir to itself.
    bad_state = dict(state_b)
    bad_state["action_in_proj.weight"] = torch.randn(cfg.action_hidden_size + 2, cfg.action_dim)
    save_file(bad_state, str(Path(dir_start) / "model.safetensors"))
    with pytest.raises(WeightReloadRefusedError, match="shape"):
        _engine(model, subs).check_weight_reload(dir_start)

    # Delete startup dir completely; valid reload from dir_b still succeeds and clears _lerobot_flat.
    shutil.rmtree(dir_start)
    _engine(model, subs).reload_weights(dir_b, "v1")
    assert model._lerobot_flat is None
    assert torch.equal(subs["action_expert"].action_in_proj.weight, state_b["action_in_proj.weight"])


# ── 4. PI05Policy "model."-prefixed & sharded checkpoint interoperability ──────


@pytest.mark.parametrize("start_prefixed, reload_prefixed, sharded", [
    (False, True, False),
    (True, False, False),
    (False, True, True),
])
def test_pi05_reload_policy_prefixed_and_sharded_checkpoints(
    tmp_path, monkeypatch, start_prefixed, reload_prefixed, sharded,
):
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=7, policy_prefixed=start_prefixed)
    state_b = _lerobot_state(cfg, seed=8, policy_prefixed=reload_prefixed)
    dir_a = _write_pi05_checkpoint(tmp_path / "a", state_a, cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", state_b, cfg, sharded=sharded)

    model, subs = _build_pi05(dir_a, monkeypatch, cast_mixed_bf16=False)
    before = _storage_snapshot(subs)
    _engine(model, subs).reload_weights(dir_b, "v1")

    assert _storage_snapshot(subs) == before
    raw_b = _unprefix(state_b)
    assert torch.equal(subs["action_expert"].action_out_proj.weight, raw_b["action_out_proj.weight"])
    assert torch.equal(
        subs["LLM"].embed_tokens.weight,
        raw_b["paligemma_with_expert.paligemma.lm_head.weight"],
    )


# ── 5. Rank-split partial submodules ─────────────────────────────────────────


def test_pi05_reload_rank_split_partial_submodules(tmp_path, monkeypatch):
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=11)
    state_b = _lerobot_state(cfg, seed=12)
    dir_a = _write_pi05_checkpoint(tmp_path / "a", state_a, cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", state_b, cfg)

    vit_model, vit_subs = _build_pi05(dir_a, monkeypatch, nodes=("vit_encoder",), cast_mixed_bf16=False)
    vit_before = _storage_snapshot(vit_subs)
    _engine(vit_model, vit_subs).reload_weights(dir_b, "v1")
    assert _storage_snapshot(vit_subs) == vit_before
    assert torch.equal(
        vit_subs["vit_encoder"].encoder.connector.weight,
        state_b["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"],
    )

    llm_model, llm_subs = _build_pi05(dir_a, monkeypatch, nodes=("LLM", "action_expert"), cast_mixed_bf16=False)
    llm_before = _storage_snapshot(llm_subs)
    _engine(llm_model, llm_subs).reload_weights(dir_b, "v1")
    assert _storage_snapshot(llm_subs) == llm_before
    assert torch.equal(llm_subs["action_expert"].action_in_proj.weight, state_b["action_in_proj.weight"])

    # Unrelated node name is a clean no-op.
    llm_model.reload_weights(dir_b, {"other_node": nn.Linear(2, 2)})


# ── 6. Full CPU forward-pass parity (ViT -> PaliGemma -> ActionExpert) ───────


class _CpuSdpaKV:
    """In-memory prefix + scratch KV store for CPU Pi0.5 parity tests."""

    def __init__(self, num_layers: int):
        self.keep_writes = True
        self.kept: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        self.scratch: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    def layer_view(self, i: int):
        return i

    def write_kv(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int = 0, label=None, tokens=None) -> None:
        del label, tokens
        if self.keep_writes:
            self.kept[layer_idx] = (k, v)
        else:
            self.scratch[layer_idx] = (k, v)

    def opens_label(self, rid, label) -> bool:
        return True


class _CpuSdpaAttn:
    requires_kv_write = True

    def __init__(self, kv: _CpuSdpaKV, num_heads: int, num_kv_heads: int, head_dim: int):
        self.kv = kv
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = head_dim ** -0.5

    def run(self, q: torch.Tensor, label=None, kv_cache_layer=None, k=None, v=None, layer_idx: int = 0):
        del label, kv_cache_layer, k, v
        kept_k, kept_v = self.kv.kept[layer_idx]
        if not self.kv.keep_writes and self.kv.scratch[layer_idx] is not None:
            sc_k, sc_v = self.kv.scratch[layer_idx]
            all_k = torch.cat([kept_k, sc_k], dim=0)
            all_v = torch.cat([kept_v, sc_v], dim=0)
        else:
            all_k, all_v = kept_k, kept_v
        if self.num_heads != self.num_kv_heads:
            rep = self.num_heads // self.num_kv_heads
            all_k = all_k.repeat_interleave(rep, dim=1)
            all_v = all_v.repeat_interleave(rep, dim=1)
        q_t = q.transpose(0, 1).float()
        k_t = all_k.transpose(0, 1).float()
        v_t = all_v.transpose(0, 1).float()
        scores = torch.einsum("hqd,hkd->hqk", q_t, k_t) * self.scale
        out = torch.einsum("hqk,hkd->hqd", scores.softmax(dim=-1), v_t)
        return out.transpose(0, 1).contiguous().to(q.dtype)


class _CpuRope:
    def apply_qk(self, q: torch.Tensor, k: torch.Tensor, label=None, rope_dtype=None):
        del label, rope_dtype
        return q, k


def _bind_cpu_sdpa(model: Pi05Model, subs: dict[str, nn.Module]) -> tuple[_CpuSdpaKV, object]:
    array = model.declare_state()[0]
    kv = _CpuSdpaKV(model.config.num_layers)
    attn = _CpuSdpaAttn(kv, model.config.num_qo_heads, model.config.num_kv_heads, model.config.head_dim)
    handle = build_handle(
        array, {LLM_KV: kv, LLM_ATTN: attn, ROPE: _CpuRope()}, _resolve([array], StateSetup()),
    )
    for node in ("LLM", "action_expert"):
        for mod in subs[node].modules():
            if hasattr(mod, "bind_state"):
                mod.bind_state({LLM_KV: handle})
    return kv, handle


def _run_pi05_cpu_pipeline(
    subs: dict[str, nn.Module],
    kv: _CpuSdpaKV,
    handle,
    image: torch.Tensor,
    text_ids: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    vit: Pi05ViTEncoderSubmodule = subs["vit_encoder"]
    llm: Pi05LLMSubmodule = subs["LLM"]
    act: Pi05ActionSubmodule = subs["action_expert"]

    img_emb = vit.forward(CHUNK_WALK, None, image.unsqueeze(0))["img_emb"][0]

    llm_inp = llm.prepare_inputs(CHUNK_WALK, None, {"img_emb": [img_emb], "text_inputs": [text_ids]})
    llm_step = SubmoduleStep(segments=[Segment(1, "main", llm_inp.input_seq_len)], state={LLM_KV: ArrayStep()})
    llm_step.set_ctx(StepContext((1,), CHUNK_WALK, 0, False))
    kv.keep_writes = True
    handle.begin_step(llm_step)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        llm.forward(CHUNK_WALK, None, llm_inp.input_embeds)
    handle.check_step(llm_step)
    handle.end_step(llm_step)

    act_step = SubmoduleStep(
        segments=[Segment(1, "main", noise.shape[0])], state={LLM_KV: ArrayStep(keep=False)},
    )
    act_step.set_ctx(StepContext((1,), CHUNK_WALK, 0, False))
    kv.keep_writes = False
    handle.begin_step(act_step)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        actions = act._denoise(noise, fraction=act._get_timestep_emb_fraction())
    handle.check_step(act_step)
    handle.end_step(act_step)
    return img_emb, actions


@torch.no_grad()
def test_pi05_cpu_forward_pass_parity_and_round_trip(tmp_path, monkeypatch):
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=13, dtype=torch.bfloat16, mixed_rlinf=False)
    state_b = _lerobot_state(cfg, seed=14, dtype=torch.bfloat16, mixed_rlinf=False)
    dir_a = _write_pi05_checkpoint(tmp_path / "a", state_a, cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", state_b, cfg)

    image = torch.linspace(-1.0, 1.0, cfg.num_cameras * 3 * cfg.vit_image_size * cfg.vit_image_size).reshape(
        cfg.num_cameras, 3, cfg.vit_image_size, cfg.vit_image_size,
    )
    text_ids = torch.tensor([1, 5, 9, 12], dtype=torch.long)
    noise = torch.randn(cfg.action_horizon, cfg.action_dim, generator=torch.Generator().manual_seed(99))

    model_a, subs_a = _build_pi05(dir_a, monkeypatch)
    kv_a, handle_a = _bind_cpu_sdpa(model_a, subs_a)
    emb_a, act_a = _run_pi05_cpu_pipeline(subs_a, kv_a, handle_a, image, text_ids, noise)

    model_fresh_b, subs_fresh_b = _build_pi05(dir_b, monkeypatch)
    kv_fb, handle_fb = _bind_cpu_sdpa(model_fresh_b, subs_fresh_b)
    emb_fb, act_fb = _run_pi05_cpu_pipeline(subs_fresh_b, kv_fb, handle_fb, image, text_ids, noise)

    assert not torch.equal(emb_a, emb_fb)
    assert not torch.equal(act_a, act_fb)

    # Reload A -> B in place
    _engine(model_a, subs_a).reload_weights(dir_b, "v1")
    emb_reloaded_b, act_reloaded_b = _run_pi05_cpu_pipeline(subs_a, kv_a, handle_a, image, text_ids, noise)
    torch.testing.assert_close(emb_reloaded_b, emb_fb, atol=0.0, rtol=0.0)
    torch.testing.assert_close(act_reloaded_b, act_fb, atol=0.0, rtol=0.0)

    # Round-trip B -> A reproduces exact checkpoint A outputs
    _engine(model_a, subs_a).reload_weights(dir_a, "v2")
    emb_rt_a, act_rt_a = _run_pi05_cpu_pipeline(subs_a, kv_a, handle_a, image, text_ids, noise)
    torch.testing.assert_close(emb_rt_a, emb_a, atol=0.0, rtol=0.0)
    torch.testing.assert_close(act_rt_a, act_a, atol=0.0, rtol=0.0)


# ── 7. torch.compile + static pointer replay parity ──────────────────────────


@torch.no_grad()
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_pi05_compiled_and_static_tensor_replay_parity_after_reload(tmp_path, monkeypatch, backend):
    torch._dynamo.reset()
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=31, dtype=torch.float32)
    state_b = _lerobot_state(cfg, seed=32, dtype=torch.float32)
    dir_a = _write_pi05_checkpoint(tmp_path / "a", state_a, cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", state_b, cfg)

    model, subs = _build_pi05(dir_a, monkeypatch, cast_mixed_bf16=False)
    fresh_b, fresh_b_subs = _build_pi05(dir_b, monkeypatch, cast_mixed_bf16=False)

    vit: Pi05ViTEncoderSubmodule = subs["vit_encoder"]
    act: Pi05ActionSubmodule = subs["action_expert"]
    kv, handle = _bind_cpu_sdpa(model, subs)
    kv_fb, handle_fb = _bind_cpu_sdpa(fresh_b, fresh_b_subs)

    # Hold direct pre-reload tensor references (simulating captured-graph static pointers)
    static_qkv_ref = subs["LLM"].paligemma.layers[0].self_attn.qkv_proj.weight
    static_time_ref = act.time_mlp.linear_in.weight
    static_siglip_ref = vit.encoder.connector.weight
    static_ptrs = (static_qkv_ref.data_ptr(), static_time_ref.data_ptr(), static_siglip_ref.data_ptr())

    compiled_vit = torch.compile(vit.encoder, backend=backend, fullgraph=True)
    compiled_pali = torch.compile(subs["LLM"].paligemma, backend=backend, dynamic=False)
    compiled_denoise = torch.compile(act._denoise, backend=backend, dynamic=False)

    pv = torch.randn(1, 3, cfg.vit_image_size, cfg.vit_image_size, generator=torch.Generator().manual_seed(1))
    prefix_embs = torch.randn(6, cfg.hidden_size, generator=torch.Generator().manual_seed(2))
    noise = torch.randn(cfg.action_horizon, cfg.action_dim, generator=torch.Generator().manual_seed(3))
    fraction = act._get_timestep_emb_fraction()

    def run_compiled(pali_fn, denoise_fn, kv_obj, h_obj):
        v_out = compiled_vit(pv)
        llm_step = SubmoduleStep(segments=[Segment(1, "main", 6)], state={LLM_KV: ArrayStep()})
        llm_step.set_ctx(StepContext((1,), CHUNK_WALK, 0, False))
        kv_obj.keep_writes = True
        h_obj.begin_step(llm_step)
        p_out = pali_fn(query_sequence=prefix_embs)
        h_obj.check_step(llm_step)
        h_obj.end_step(llm_step)

        act_step = SubmoduleStep(
            segments=[Segment(1, "main", cfg.action_horizon)], state={LLM_KV: ArrayStep(keep=False)},
        )
        act_step.set_ctx(StepContext((1,), CHUNK_WALK, 0, False))
        kv_obj.keep_writes = False
        h_obj.begin_step(act_step)
        d_out = denoise_fn(noise, fraction=fraction)
        h_obj.check_step(act_step)
        h_obj.end_step(act_step)
        return v_out, p_out, d_out

    vit_a, pali_a, act_a = run_compiled(compiled_pali, compiled_denoise, kv, handle)

    # Reload ckpt B in place through Engine.reload_weights
    _engine(model, subs).reload_weights(dir_b, "v1")
    assert (static_qkv_ref.data_ptr(), static_time_ref.data_ptr(), static_siglip_ref.data_ptr()) == static_ptrs
    assert torch.equal(
        static_siglip_ref,
        state_b["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"],
    )

    # Replay compiled callables with fail_on_recompile: zero recompilations allowed
    with torch.compiler.set_stance("fail_on_recompile"):
        vit_b, pali_b, act_b = run_compiled(compiled_pali, compiled_denoise, kv, handle)

    assert not torch.allclose(vit_a, vit_b)
    assert not torch.allclose(pali_a, pali_b)
    assert not torch.allclose(act_a, act_b)

    # Compare against fresh eager checkpoint B
    def run_fresh_eager():
        v_out = fresh_b_subs["vit_encoder"].encoder(pv)
        llm_step = SubmoduleStep(segments=[Segment(1, "main", 6)], state={LLM_KV: ArrayStep()})
        llm_step.set_ctx(StepContext((1,), CHUNK_WALK, 0, False))
        kv_fb.keep_writes = True
        handle_fb.begin_step(llm_step)
        p_out = fresh_b_subs["LLM"].paligemma(query_sequence=prefix_embs)
        handle_fb.check_step(llm_step)
        handle_fb.end_step(llm_step)

        act_step = SubmoduleStep(
            segments=[Segment(1, "main", cfg.action_horizon)], state={LLM_KV: ArrayStep(keep=False)},
        )
        act_step.set_ctx(StepContext((1,), CHUNK_WALK, 0, False))
        kv_fb.keep_writes = False
        handle_fb.begin_step(act_step)
        d_out = fresh_b_subs["action_expert"]._denoise(
            noise, fraction=fresh_b_subs["action_expert"]._get_timestep_emb_fraction(),
        )
        handle_fb.check_step(act_step)
        handle_fb.end_step(act_step)
        return v_out, p_out, d_out

    vit_exp, pali_exp, act_exp = run_fresh_eager()
    torch.testing.assert_close(vit_b, vit_exp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(pali_b, pali_exp, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(act_b, act_exp, atol=1e-5, rtol=1e-5)


# ── 8. Negative error paths copy nothing & drop KV prefix cache ──────────────


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass


def test_pi05_reload_negative_error_paths_copy_nothing(tmp_path, monkeypatch):
    cfg = _tiny_config()
    state_a = _lerobot_state(cfg, seed=41)
    state_b = _lerobot_state(cfg, seed=42)
    dir_a = _write_pi05_checkpoint(tmp_path / "a", state_a, cfg)
    dir_b = _write_pi05_checkpoint(tmp_path / "b", state_b, cfg)

    model, subs = _build_pi05(dir_a, monkeypatch, cast_mixed_bf16=False)
    engine = _engine(model, subs)

    def assert_still_a():
        assert torch.equal(subs["action_expert"].action_in_proj.weight, state_a["action_in_proj.weight"])
        assert torch.equal(subs["vit_encoder"].encoder.connector.weight,
                           state_a["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"])

    # Missing dir & missing safetensors
    with pytest.raises(WeightReloadRefusedError, match="no checkpoint"):
        engine.check_weight_reload(str(tmp_path / "missing_dir"))
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(WeightReloadRefusedError, match="no model.safetensors"):
        engine.check_weight_reload(str(empty_dir))

    # Truncated / corrupt safetensors
    trunc = tmp_path / "truncated"
    trunc.mkdir()
    (trunc / "model.safetensors").write_bytes((Path(dir_b) / "model.safetensors").read_bytes()[:-64])
    with pytest.raises(WeightReloadRefusedError, match="unreadable"):
        engine.check_weight_reload(str(trunc))

    # Missing key & extra key
    missing_key = {k: v for k, v in state_b.items() if k != "action_in_proj.weight"}
    with pytest.raises(WeightReloadRefusedError, match="missing"):
        engine.check_weight_reload(_write_pi05_checkpoint(tmp_path / "missing_k", missing_key, cfg))
    extra_key = {**state_b, "extra.tensor": torch.zeros(2, 2)}
    with pytest.raises(WeightReloadRefusedError, match="does not have"):
        engine.check_weight_reload(_write_pi05_checkpoint(tmp_path / "extra_k", extra_key, cfg))

    # Reshaped tensor (q_proj, action_in_proj, patch_embedding)
    bad_q = {
        **state_b,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(
            4, cfg.hidden_size,
        ),
    }
    with pytest.raises(WeightReloadRefusedError, match="shape"):
        engine.check_weight_reload(_write_pi05_checkpoint(tmp_path / "bad_q", bad_q, cfg))
    bad_patch = {
        **state_b,
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight": (
            torch.randn(1, 3, 14, 14)
        ),
    }
    with pytest.raises(WeightReloadRefusedError, match="shape"):
        engine.check_weight_reload(_write_pi05_checkpoint(tmp_path / "bad_patch", bad_patch, cfg))

    # Non-float dtype mismatch
    bad_int = {**state_b, "action_in_proj.bias": torch.ones(cfg.action_hidden_size, dtype=torch.int32)}
    with pytest.raises(WeightReloadRefusedError, match="dtype"):
        engine.check_weight_reload(_write_pi05_checkpoint(tmp_path / "bad_int", bad_int, cfg))

    # Mismatched config.json (chunk_size)
    with pytest.raises(WeightReloadRefusedError, match="chunk_size"):
        engine.check_weight_reload(
            _write_pi05_checkpoint(tmp_path / "bad_cfg", state_b, cfg, extra_config={"chunk_size": 15}),
        )

    # skip_weight_loading=True (no startup snapshot)
    dummy_model, dummy_subs = _build_pi05(
        dir_a, monkeypatch, skip_weight_loading=True, cast_mixed_bf16=False, **TINY_CFG,
    )
    with pytest.raises(WeightReloadRefusedError, match="not loaded from a checkpoint"):
        _engine(dummy_model, dummy_subs).check_weight_reload(dir_b)

    # Unreadable startup snapshot string
    unreadable = Pi05Model.__new__(Pi05Model)
    unreadable._checkpoint = "no checkpoint at /gone"
    with pytest.raises(WeightReloadRefusedError, match="could not be read at startup"):
        _engine(unreadable, subs).check_weight_reload(dir_b)

    assert_still_a()

    # Engine.reload_weights drops KV prefix cache before apply even if apply fails mid-copy
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=8, page_size=8,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(b"root")
    pages = kv._arena.acquire(2)
    kv._index.insert_run([fingerprint(b"root", k) for k in (b"k0", b"k1")], pages, None)
    kv._arena.release(pages)

    class _MidCopyFail(Pi05Model):
        def _apply_reload(self, path, submodules):
            raise OSError("mid-copy read error")

    failing = _MidCopyFail.__new__(_MidCopyFail)
    failing._checkpoint = model._checkpoint
    failing._lerobot_flat = None
    with pytest.raises(OSError, match="mid-copy read error"):
        _engine(failing, subs, kv=kv).reload_weights(dir_b, "v1")
    assert kv._index.num_pages() == 0
    assert kv._prefix_root == fingerprint(b"root", "weights", "v1")
    assert_still_a()


# ── 9. Shared opt-in Model.reload_weights + Orpheus compatibility ────────────


def test_shared_model_reload_weights_registration_and_orpheus_compatibility(tmp_path, monkeypatch):
    # Opt-in Model subclass using _register_reload_target
    ckpt_1 = tmp_path / "m1"
    ckpt_2 = tmp_path / "m2"
    ckpt_1.mkdir()
    ckpt_2.mkdir()
    w1 = torch.full((3, 3), 1.0)
    w2 = torch.full((3, 3), 2.0)
    save_file({"weight": w1}, str(ckpt_1 / "model.safetensors"))
    save_file({"weight": w2}, str(ckpt_2 / "model.safetensors"))

    class _SimpleMod(nn.Module):
        def __init__(self, tensor):
            super().__init__()
            self.weight = nn.Parameter(tensor.clone())

        def load_weights(self, weights):
            for name, t in weights:
                if name == "weight":
                    self.weight.data.copy_(t)
            return {"weight"}

    class _OptInModel(Model):
        get_graph_walk_graphs = get_submodule = process_prompt = walk_plan = lambda *a, **kw: None

        def __init__(self, ckpt):
            self._register_reload_target("core", checkpoint_dir=ckpt)

    opt_in = _OptInModel(str(ckpt_1))
    mod = _SimpleMod(w1)
    ptr = mod.weight.data_ptr()
    opt_in.reload_weights(str(ckpt_2), {"core": mod}, check_only=True)
    assert torch.equal(mod.weight, w1)
    opt_in.reload_weights(str(ckpt_2), {"core": mod})
    assert mod.weight.data_ptr() == ptr
    assert torch.equal(mod.weight, w2)

    # Bad shape refused
    ckpt_bad = tmp_path / "m_bad"
    ckpt_bad.mkdir()
    save_file({"weight": torch.zeros(2, 2)}, str(ckpt_bad / "model.safetensors"))
    with pytest.raises(WeightReloadRefusedError, match="shape"):
        opt_in.reload_weights(str(ckpt_bad), {"core": mod}, check_only=True)

    # Un-enrolled Model subclass still refuses by default
    class _Unenrolled(Model):
        get_graph_walk_graphs = get_submodule = process_prompt = walk_plan = lambda *a, **kw: None

    with pytest.raises(WeightReloadRefusedError, match="_Unenrolled does not support"):
        _Unenrolled.__new__(_Unenrolled).reload_weights(str(ckpt_2), {"core": mod})

    # OrpheusModel and Pi05Model both inherit shared Model.reload_weights
    assert OrpheusModel.reload_weights is Model.reload_weights
    assert Pi05Model.reload_weights is Model.reload_weights
    # _device ignored in config.json for LeRobot checkpoints
    cfg = _tiny_config()
    d1 = _write_pi05_checkpoint(tmp_path / "dev1", _lerobot_state(cfg, 50), cfg, extra_config={"device": "mps"})
    d2 = _write_pi05_checkpoint(tmp_path / "dev2", _lerobot_state(cfg, 51), cfg, extra_config={"device": "cuda"})
    snap = snapshot_checkpoint(d1, normalize_specs=Pi05Model._normalize_lerobot_specs)
    from mstar.model.loader.reload import check_reload_compatible

    check_reload_compatible(d2, snap, normalize_specs=Pi05Model._normalize_lerobot_specs)
    assert not math.isnan(float(mod.weight.detach().sum()))
