"""Validate the glm5_next weight loader against a REAL GLM-5.3-Flash checkpoint."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.weight_loader import (
    FP32_PARAM_SUFFIXES,
    resolve_index_names,
)

# safetensors dtype strings -> a stable label we assert against.
_FP8 = "F8_E4M3"
_BF16 = "BF16"
_F32 = "F32"


def read_safetensors_header(path: Path) -> dict[str, tuple[str, list[int]]]:
    """{name: (dtype_str, shape)} from one file's JSON header — no tensor data."""
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        out[name] = (meta["dtype"], list(meta["shape"]))
    return out


def collect_real_headers(ckpt: Path) -> dict[str, tuple[str, list[int]]]:
    real: dict[str, tuple[str, list[int]]] = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        for name, ds in read_safetensors_header(f).items():
            real[name] = ds
    return real


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python env/smoke_glm53_loader.py <checkpoint_dir>")
        return 2
    ckpt = Path(sys.argv[1])
    cfg = Glm5NextModelConfig.from_hf_config(
        json.loads((ckpt / "config.json").read_text()), strict=True
    )
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    index_names = set(index["weight_map"])
    real = collect_real_headers(ckpt)
    real_names = set(real)

    fails: list[str] = []
    warns: list[str] = []

    # (1) file/index drift.
    only_index = index_names - real_names
    only_files = real_names - index_names
    if only_index:
        fails.append(f"{len(only_index)} names in index but not in files, e.g. {sorted(only_index)[:3]}")
    if only_files:
        fails.append(f"{len(only_files)} names in files but not in index, e.g. {sorted(only_files)[:3]}")
    print(f"[1] names: {len(real_names)} real, {len(index_names)} index, "
          f"{'MATCH' if not (only_index or only_files) else 'DRIFT'}")

    # (2) full mapping on the real names (0 unmapped). MTP tensors are loaded
    # only when drafting is enabled; at M1 (MTP off) they are a counted skip.
    load_mtp = cfg.mtp_num_draft_tokens > 0
    res = resolve_index_names(real_names, cfg, load_mtp=load_mtp)
    unmapped = res.get("unmapped", ())
    if unmapped:
        fails.append(f"{len(unmapped)} unmapped real names, e.g. {list(unmapped)[:3]}")
    loaded = len(res.get("loaded", ())) + len(res.get("absorbed", ()))
    skipped = len(res.get("skip_vision", ())) + len(res.get("skip_mtp", ()))
    print(f"[2] mapping: {loaded} loaded/absorbed, {skipped} skipped, "
          f"{len(unmapped)} unmapped")

    # (3) dtype partition. Classify each real name by its dtype-family rule and
    # assert the header dtype matches. fp8 weights must carry a scale sibling.
    def dtype_label(ds: str) -> str:
        u = ds.upper()
        if u.startswith("F8") or "E4M3" in u:
            return _FP8
        if u in ("BF16", "BFLOAT16"):
            return _BF16
        if u in ("F32", "FLOAT32"):
            return _F32
        return u

    scale_suffix = ".weight_scale_inv"
    scale_bases = {n[: -len(scale_suffix)] for n in real_names if n.endswith(scale_suffix)}
    n_fp8 = n_bf16 = n_f32 = 0
    for name, (ds, _shape) in real.items():
        if name.startswith("model.visual."):
            continue
        if name.endswith(scale_suffix):
            continue
        lbl = dtype_label(ds)
        base = name.rsplit(".", 1)[0]
        if lbl == _FP8:
            n_fp8 += 1
            if base not in scale_bases and name not in scale_bases:
                # routed experts are fp8-resident and still pair a scale; any
                # fp8 weight with no scale sibling would silently mis-dequant.
                fails.append(f"fp8 tensor with no weight_scale_inv sibling: {name}")
        elif lbl == _F32:
            n_f32 += 1
        elif lbl == _BF16:
            n_bf16 += 1
        else:
            fails.append(f"unexpected dtype {ds} on {name}")
    print(f"[3] dtypes: {n_fp8} fp8(+scales), {n_bf16} bf16, {n_f32} fp32; "
          f"{len(scale_bases)} scale siblings")

    # (3b) fp32-preservation coverage — the load path narrows every float param
    # to bf16 then re-widens only FP32_PARAM_SUFFIXES. Every fp32 tensor that
    # becomes a param (the 'loaded' stream; absorbed fp8 scales don't) must map
    # to a target ending in one of those suffixes, else it loads bf16-downcast.
    loaded_map: dict[str, str] = {}
    for entry in res.get("loaded", ()):
        raw, _, target = entry.partition(" -> ")
        loaded_map[raw] = target
    miss = []
    n_fp32_params = 0
    for raw, target in loaded_map.items():
        ds = real.get(raw, ("", None))[0]
        if dtype_label(ds) != _F32:
            continue
        n_fp32_params += 1
        if not target.endswith(FP32_PARAM_SUFFIXES):
            miss.append(f"{raw} -> {target}")
    if miss:
        fails.append(
            f"{len(miss)} fp32 checkpoint tensors map to a target NOT in "
            f"FP32_PARAM_SUFFIXES (would load bf16-downcast), e.g. {miss[:4]}"
        )
    print(f"[3b] fp32 preservation: {n_fp32_params} fp32 params, "
          f"{'all covered' if not miss else str(len(miss)) + ' UNCOVERED'}")

    # (4) representative shapes vs config.
    H = cfg.hidden_size
    V = cfg.vocab_size
    kda_qkv = cfg.linear_num_heads * cfg.linear_head_dim
    K = cfg.linear_conv_kernel_size

    def sh(name: str):
        return real.get(name, (None, None))[1]

    def expect(name: str, want: list[int]):
        got = sh(name)
        if got is None:
            warns.append(f"[shape] {name} absent (skipped check)")
        elif got != want:
            fails.append(f"[shape] {name}: got {got}, want {want}")
        else:
            print(f"    ok {name} {got}")

    # find first KDA layer and first MLA layer from the schedule.
    kda_layer = next(i for i, t in enumerate(cfg.layer_types) if t == "linear_attention")
    mla_layer = next(i for i, t in enumerate(cfg.layer_types) if t != "linear_attention")
    pre = "model.language_model.layers"
    print(f"[4] shapes (KDA layer {kda_layer}, MLA layer {mla_layer}):")
    expect("model.language_model.embed_tokens.weight", [V, H])
    expect("lm_head.weight", [V, H])
    for c in ("q_conv1d", "k_conv1d", "v_conv1d"):
        expect(f"{pre}.{kda_layer}.self_attn.{c}.weight", [kda_qkv, 1, K])
    # dt_bias / A_log fp32 restore set — shape sanity (per KDA spec: [qkv], [heads]).
    for leaf, want in (("dt_bias", [kda_qkv]), ("A_log", [cfg.linear_num_heads])):
        got = sh(f"{pre}.{kda_layer}.self_attn.{leaf}")
        if got is None:
            warns.append(f"[shape] KDA {leaf} absent")
        else:
            print(f"    ok {pre}.{kda_layer}.self_attn.{leaf} {got} (want ~{want})")

    ok = not fails
    print()
    for w in warns:
        print(f"WARN {w}")
    for f in fails:
        print(f"FAIL {f}")
    print(f"\nsmoke_glm53_loader: {'PASS' if ok else 'FAIL'} "
          f"({len(real_names)} tensors, {len(fails)} failures, {len(warns)} warnings)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
