#!/usr/bin/env python3
"""Layer-by-layer diff of mminf's Pi0.5 forward path against lerobot.

Unlike ``compare_with_lerobot.py``, this script does NOT go through the HTTP
server. Instead it instantiates ``Pi05Model`` in-process via the same
``load_lerobot_pi05_into_model`` remapper the production server uses, then
walks the forward in stages and prints summary stats for each intermediate
tensor next to the corresponding lerobot tensor:

  Stage 1: Pi05ViTEncoderSubmodule output (per-camera image embeddings)
           vs lerobot ``paligemma_with_expert.embed_image(image)``.
  Stage 2: Pi05LLMSubmodule._preprocess_prefill output (prefix_embs)
           vs lerobot ``embed_prefix(images, masks, tokens, masks)``.
  Stage 3: Action expert first-step velocity
           vs lerobot ``denoise_step`` first iteration.
  Stage 4: Final 50-step action trajectory.

If a stage diverges by more than ``stage_tolerance`` (default 1e-2 max abs
delta), the script flags it and prints norms / diff stats so we can
localize the bug to the specific stage.

This is purely an in-process diagnostic — no server / no FlashInfer paged
cache (we use the same _PrefixCacheCapture / _MockCacheHandle pair from
``test/integration/test_pi05_real_weights.py``). It's the strictest possible
component-level test: if mminf's components match lerobot here but the server
diverges, the bug is in the server-only path (FlashInfer cache, dtype,
data_worker preprocessing). If mminf already diverges here, the bug is in
the components themselves.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Make sure repo root is importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HF_REPO = "lerobot/pi05_base"


def stub_broken_lerobot_subpackages():
    if "lerobot.policies.groot.groot_n1" in sys.modules:
        return
    def _make_stub(name):
        m = types.ModuleType(name)
        m.__path__ = []
        return m
    pkg = _make_stub("lerobot.policies.groot")
    g_n1 = _make_stub("lerobot.policies.groot.groot_n1")
    g_n1.GR00TN15 = type("GR00TN15", (), {})
    cfg = _make_stub("lerobot.policies.groot.configuration_groot")
    cfg.GrootConfig = type("GrootConfig", (), {})
    modg = _make_stub("lerobot.policies.groot.modeling_groot")
    modg.GrootPolicy = type("GrootPolicy", (), {})
    sys.modules["lerobot.policies.groot"] = pkg
    sys.modules["lerobot.policies.groot.groot_n1"] = g_n1
    sys.modules["lerobot.policies.groot.configuration_groot"] = cfg
    sys.modules["lerobot.policies.groot.modeling_groot"] = modg


# ----------------------------------------------------------------------------
# Build deterministic inputs (same as compare_with_lerobot.py).
# ----------------------------------------------------------------------------

def build_inputs(seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    images_pil = []
    image_arrs_minus1_to_plus1 = []
    for cam_idx in range(3):
        arr = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        arr[:, :, cam_idx] = np.clip(arr[:, :, cam_idx].astype(int) + 8, 0, 255).astype(np.uint8)
        images_pil.append(Image.fromarray(arr, mode="RGB"))
        # Reproduce the data_worker -> _preprocess_one path: uint8 -> float [0,1] -> float [-1,1]
        f01 = arr.astype(np.float32) / 255.0
        chw = torch.from_numpy(f01).permute(2, 0, 1).to(device)  # CHW float [0,1]
        image_arrs_minus1_to_plus1.append((chw * 2.0 - 1.0).unsqueeze(0))  # [1, 3, 224, 224]
    return images_pil, image_arrs_minus1_to_plus1


# ----------------------------------------------------------------------------
# lerobot reference: capture intermediates.
# ----------------------------------------------------------------------------

def load_lerobot(device):
    stub_broken_lerobot_subpackages()
    from lerobot.policies.pi05 import PI05Policy
    print("Loading lerobot/pi05_base ...")
    policy = PI05Policy.from_pretrained(HF_REPO).to(device).eval()
    return policy.model, policy.config


# ----------------------------------------------------------------------------
# mminf in-process: load Pi05Model via the same remapper the server uses.
# ----------------------------------------------------------------------------

def load_mminf(ref_model, ref_config, device):
    from safetensors.torch import load_file

    from mminf.model.pi05.config import Pi05Config
    from mminf.model.pi05.pi05_model import Pi05Model
    from mminf.model.pi05.weight_loader import load_lerobot_pi05_into_model

    cache_root = os.environ.get(
        "HF_HUB_CACHE",
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")) + "/hub",
    )
    snap_root = Path(cache_root) / "models--lerobot--pi05_base" / "snapshots"
    if not snap_root.exists():
        snap_root = Path(cache_root).parent / "models--lerobot--pi05_base" / "snapshots"
    safetensors_path = next(snap_root.iterdir()) / "model.safetensors"
    print(f"loading mminf weights from {safetensors_path}")
    state_dict = load_file(str(safetensors_path), device="cpu")

    pali_layer = ref_model.paligemma_with_expert.paligemma.model.language_model.layers[0]
    ge_layer = ref_model.paligemma_with_expert.gemma_expert.model.layers[0]
    mminf_model = Pi05Model(model_path_hf="lerobot/pi05_base", skip_weight_loading=True)
    mminf_model.config = Pi05Config(
        hidden_size=pali_layer.self_attn.config.hidden_size,
        action_hidden_size=ref_model.action_in_proj.out_features,
        num_layers=len(ref_model.paligemma_with_expert.paligemma.model.language_model.layers),
        num_qo_heads=pali_layer.self_attn.config.num_attention_heads,
        num_kv_heads=pali_layer.self_attn.config.num_key_value_heads,
        head_dim=pali_layer.self_attn.head_dim,
        pali_intermediate_size=pali_layer.mlp.gate_proj.out_features,
        action_intermediate_size=ge_layer.mlp.gate_proj.out_features,
        rms_norm_eps=pali_layer.input_layernorm.eps,
        num_flow_steps=ref_config.num_inference_steps,
        action_horizon=ref_config.chunk_size,
        action_dim=ref_config.max_action_dim,
        vit_hidden_size=1152,
        vit_intermediate_size=4304,
        vit_num_layers=27,
        vit_num_heads=16,
        vit_image_size=224,
        vit_patch_size=14,
    )
    missing = load_lerobot_pi05_into_model(
        mminf_model, state_dict, device=str(device), strict=False
    )
    for name, keys in missing.items():
        if keys:
            print(f"  WARNING: bucket {name} has {len(keys)} missing keys: {keys[:3]}")
    return mminf_model


# ----------------------------------------------------------------------------
# Diff helpers.
# ----------------------------------------------------------------------------

def stat(name, ours, ref, tol=1e-3):
    diff = (ours.float() - ref.float()).abs()
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    rel = diff / (ref.float().abs() + 1e-6)
    mean_rel = float(rel.mean())
    ours_norm = float(ours.float().norm())
    ref_norm = float(ref.float().norm())
    flag = "OK   " if max_d <= tol else "FAIL "
    print(
        f"  {flag} {name:40s}  shape={tuple(ours.shape)}  "
        f"max_abs={max_d:.4e}  mean_abs={mean_d:.4e}  mean_rel={mean_rel:.4e}"
    )
    print(
        f"           ours norm={ours_norm:.4e}  ref norm={ref_norm:.4e}"
    )
    return max_d <= tol


# ----------------------------------------------------------------------------
# Main diagnostic.
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--text", default="pick up the block")
    parser.add_argument(
        "--state", default=",".join(["0.0"] * 8), help="Comma-separated robot state"
    )
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device}  dtype={dtype}\n")

    # ----- 1. Load both models -----
    ref_model, ref_config = load_lerobot(device)
    mminf_model = load_mminf(ref_model, ref_config, device)

    # ----- 2. Build inputs -----
    images_pil, mminf_images = build_inputs(args.seed, device)
    img_masks_lerobot = [torch.ones(1, dtype=torch.bool, device=device) for _ in mminf_images]

    # Build tokens via the SAME tokenizer mminf uses (PaliGemma fast).
    from transformers import AutoTokenizer

    from mminf.model.pi05.components.flow_matching import discretize_state
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", use_fast=True)
    state_t = torch.tensor(
        [float(x) for x in args.state.split(",") if x.strip()] + [0.0] * 32,
        dtype=torch.float32,
    )[:32]
    bins = discretize_state(state_t, num_bins=256).tolist()
    state_str = " ".join(str(b) for b in bins)
    cleaned = args.text.strip().replace("_", " ").replace("\n", " ")
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: ".strip().lower()
    ids = tokenizer(full_prompt, add_special_tokens=True).input_ids[:200]
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    masks_lerobot = torch.ones(tokens.shape, dtype=torch.bool, device=device)
    print(f"\nprompt token count = {tokens.shape[1]}")
    print(f"image count = {len(mminf_images)}")
    print(f"prompt = {full_prompt!r}")

    # Build deterministic noise (same shape as compare_with_lerobot.py).
    horizon = ref_config.chunk_size
    action_dim = ref_config.max_action_dim
    g = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(1, horizon, action_dim, device=device, generator=g, dtype=dtype)

    # Cast both models to the test dtype.
    ref_model = ref_model.to(dtype=dtype).eval()
    vit_submodule = mminf_model.get_submodule("vit_encoder", device=str(device))
    llm_submodule = mminf_model.get_submodule("LLM", device=str(device))
    vit_submodule = vit_submodule.to(device, dtype=dtype).eval()
    llm_submodule = llm_submodule.to(device, dtype=dtype).eval()

    print("\n=== Stage 0: weight-equality sanity check ===")
    with torch.no_grad():
        # patch_embedding.weight is the very first SigLIP layer; if this differs,
        # the remapper is wrong and everything downstream is garbage.
        ref_patch_w = (
            ref_model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight
        )
        ours_patch_w = (
            vit_submodule.encoder.vision_model.vision_model.embeddings.patch_embedding.weight
        )
        stat("siglip patch_embedding.weight", ours_patch_w, ref_patch_w, tol=1e-6)

        # connector (multi_modal_projector.linear) — last layer of the vision pipeline
        ref_conn_w = ref_model.paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight
        ours_conn_w = vit_submodule.encoder.connector.weight
        stat("connector.weight", ours_conn_w, ref_conn_w, tol=1e-6)

        # spot-check one mid-layer SigLIP transformer block
        ref_blk = (
            ref_model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers[10]
        )
        ours_blk = (
            vit_submodule.encoder.vision_model.vision_model.encoder.layers[10]
        )
        stat(
            "siglip layer10 q_proj.weight",
            ours_blk.self_attn.q_proj.weight,
            ref_blk.self_attn.q_proj.weight,
            tol=1e-6,
        )
        stat(
            "siglip layer10 layer_norm1.weight",
            ours_blk.layer_norm1.weight,
            ref_blk.layer_norm1.weight,
            tol=1e-6,
        )

    print("\n=== Stage 1: per-camera image embedding (vit_encoder) ===")
    with torch.no_grad():
        # lerobot: per-camera embed_image
        ref_per_cam = []
        for img in mminf_images:
            ref_emb = ref_model.paligemma_with_expert.embed_image(img.to(dtype))
            ref_per_cam.append(ref_emb)  # [1, 256, 2048] each, ALREADY scaled by sqrt(hidden)

        # mminf: run Pi05ViTEncoderSubmodule.preprocess + forward on the SAME [-1,1] inputs
        # (mimicking what data_worker would feed in, but in [-1,1] form to avoid double scaling).
        from mminf.communication.tensors import NameToTensorList
        per_request_inputs: list[NameToTensorList] = [
            {
                "image_inputs": [
                    torch.stack([img.squeeze(0) for img in mminf_images], dim=0)
                ]  # [num_cams=3, 3, 224, 224] in [-1,1]
            }
        ]
        # NOTE: passing already-[-1,1] images so _preprocess_one's auto-detect leaves them alone.
        from mminf.conductor.request_info import CurrentForwardPassInfo
        info = CurrentForwardPassInfo(
            graph_walk="prefill", requires_cfg=False, fwd_index=0, random_seed=0,
            sampling_config={}
        )
        prep = vit_submodule.preprocess(
            graph_walk="prefill",
            per_request_inputs=per_request_inputs,
            request_ids=["r0"],
            per_request_info={"r0": info},
            cache_manager=None,
        )
        mminf_img_features = vit_submodule.forward(
            request_info=info, **prep
        )["img_emb"][0]  # [num_cams * 256, 2048] — UNSCALED (sqrt happens in LLM submodule)

    # Compare per-camera. mminf img_emb is unscaled; lerobot ref_per_cam is scaled
    # by sqrt(hidden) in embed_image.
    image_scale = mminf_model.config.hidden_size**0.5
    mminf_per_cam_scaled = (
        mminf_img_features.reshape(3, 256, -1) * image_scale
    )
    ref_per_cam_stacked = torch.cat([r for r in ref_per_cam], dim=0)  # [3, 256, 2048]
    stat("vit_encoder per-cam scaled", mminf_per_cam_scaled, ref_per_cam_stacked, tol=args.tol)

    print("\n=== Stage 2: prefix embeddings (image+text concatenated, scaled) ===")
    with torch.no_grad():
        # lerobot embed_prefix
        ref_prefix_embs, ref_prefix_pad_masks, _ = ref_model.embed_prefix(
            [i.to(dtype) for i in mminf_images], img_masks_lerobot, tokens, masks_lerobot
        )
        ref_prefix = ref_prefix_embs[0]

        # mminf _preprocess_prefill, with the freshly-encoded img_emb above.
        per_request_inputs_llm: list[NameToTensorList] = [
            {
                "img_emb": [mminf_img_features],
                "text_inputs": [tokens[0]],
            }
        ]
        # We don't actually need a real cache_manager for this stage — just to
        # call the helper that builds prefix_embs. Pi05LLMSubmodule._preprocess_prefill
        # also calls plan_attention/plan_rope which need a real cache manager.
        # Build a dummy that just no-ops the plan_* calls:
        class _NoopCache:
            def plan_attention(self, *a, **k): pass
            def plan_rope(self, *a, **k): pass
        mminf_prefill_in = llm_submodule._preprocess_prefill(
            per_request_inputs=per_request_inputs_llm,
            request_ids=["r0"],
            cache_manager=_NoopCache(),
        )
        mminf_prefix = mminf_prefill_in["prefix_embs"]

    stat("prefix embeds (img+text)", mminf_prefix, ref_prefix, tol=args.tol)

    print("\n=== Stage 3: prefix KV cache after PaliGemma forward ===")
    with torch.no_grad():
        # mminf paligemma forward through a capturing mock cache.
        sys.path.insert(0, str(ROOT / "test" / "integration"))
        from test_pi05_real_weights import _MockCacheHandle, _PrefixCacheCapture

        prefix_len = mminf_prefix.shape[0]
        prefill_handle = _PrefixCacheCapture(
            head_dim=mminf_model.config.head_dim, prefix_len=prefix_len
        )
        llm_submodule.paligemma(
            query_sequence=mminf_prefix.to(dtype),
            cache_handle=prefill_handle,
            write_cache=False,
        )

        # lerobot prefill (replicate sample_actions's first call).
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(
            ref_prefix_pad_masks,
            torch.zeros(ref_prefix_pad_masks.shape, dtype=torch.long, device=device),
        )
        prefix_position_ids = torch.cumsum(ref_prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = ref_model._prepare_attention_masks_4d(prefix_att_2d_masks)
        ref_model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, ref_past_kv = ref_model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[ref_prefix_embs.to(dtype), None],
            use_cache=True,
        )

    # Compare per-layer KV norms.
    print("  layer-by-layer prefix KV cache:")
    n_layers = mminf_model.config.num_layers
    for L in range(n_layers):
        ours_k, ours_v = prefill_handle.captured_kv[L]
        # import ipdb; ipdb.set_trace()
        # ref_k = ref_past_kv[L][0][0].transpose(0, 1)  # [seq, n_kv, head_dim]
        # ref_v = ref_past_kv[L][1][0].transpose(0, 1)
        ref_k = ref_past_kv.layers[L].keys[0].transpose(0, 1)   # [seq, n_kv, head_dim]
        ref_v = ref_past_kv.layers[L].values[0].transpose(0, 1)
        if L < 3 or L >= n_layers - 2:
            stat(f"layer {L:2d} K", ours_k, ref_k, tol=args.tol)
            stat(f"layer {L:2d} V", ours_v, ref_v, tol=args.tol)

    print("\n=== Stage 4: full action trajectory (denoise loop) ===")
    with torch.no_grad():
        # mminf full denoise loop using the captured prefix KV (mock cache).
        suffix_positions = (
            torch.arange(horizon, device=device, dtype=torch.long) + prefix_len
        )
        action_handle = _MockCacheHandle(
            head_dim=mminf_model.config.head_dim, suffix_positions=suffix_positions
        )
        for L in range(n_layers):
            action_handle._prefix_kv[L] = prefill_handle.captured_kv[L]

        from mminf.model.pi05.components.flow_matching import sincos_timestep_embedding
        num_steps = ref_config.num_inference_steps
        dt = -1.0 / num_steps
        x_t = noise.clone().to(dtype)
        time = torch.tensor(1.0, device=device, dtype=dtype)
        for _ in range(num_steps):
            time_emb = sincos_timestep_embedding(
                time.unsqueeze(0),
                dim=mminf_model.config.action_hidden_size,
                min_period=ref_config.min_period,
                max_period=ref_config.max_period,
            ).squeeze(0)
            adarms_cond = llm_submodule.time_mlp(time_emb.to(dtype))
            suffix = llm_submodule.action_in_proj(x_t[0])
            suffix_out = llm_submodule.action_expert(
                query_sequence=suffix, cache_handle=action_handle, adarms_cond=adarms_cond
            )
            v_t = llm_submodule.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t.unsqueeze(0)
            time = time + dt
        ours_actions = x_t

        ref_actions = ref_model.sample_actions(
            images=[i.to(dtype) for i in mminf_images],
            img_masks=img_masks_lerobot,
            tokens=tokens,
            masks=masks_lerobot,
            noise=noise.to(dtype),
            num_steps=num_steps,
        )

    stat("final actions", ours_actions, ref_actions, tol=1e-2)
    print()
    print(f"ours[0, 0, :8]: {ours_actions[0, 0, :8].tolist()}")
    print(f"ref [0, 0, :8]: {ref_actions[0, 0, :8].tolist()}")


if __name__ == "__main__":
    main()
