"""Per-layer equivalence: the native ``Wan22DiT`` vs the wrapped diffusers
``WanTransformer3DModel``, both loaded in the serving dtypes (bf16 + fp32 islands).

The native module is an exact port, so the bar is BIT-EXACT at every stage: the
weights (which also proves the loader's remap is a bijection), the RoPE tables, the
condition-embedder outputs, and the activations at patchify, at each of the 30 block
boundaries, and at the head. Never widen a bound to pass — a nonzero anywhere is a
port bug, and the printed table names the first divergent stage.

Stage outputs are captured with forward hooks in execution order on BOTH modules, so
a block-indexing mistake shifts every later row rather than passing silently.

Inputs use the batch-2 CFG serving shape with an I2V-style per-token timestep grid
(frame-0 tokens zeroed), which exercises the general path; T2V's uniform grid is the
special case. Skipped without CUDA or the checkpoint in the local HF cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.model.wan22.config import Wan22Config
from mstar.model.wan22.weight_loader import build_wan22_dit, remap_checkpoint_key

MODEL_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
CUDA_AVAILABLE = torch.cuda.is_available()


def _hf_cache_has_checkpoint() -> bool:
    dirname = f"models--{MODEL_REPO.replace('/', '--')}"
    for env in ("HF_HUB_CACHE", "HF_HOME"):
        root = os.environ.get(env)
        if not root:
            continue
        base = Path(root) if env == "HF_HUB_CACHE" else Path(root) / "hub"
        if (base / dirname).exists():
            return True
    return (Path.home() / ".cache" / "huggingface" / "hub" / dirname).exists()


pytest.importorskip("diffusers", reason="diffusers not installed")

pytestmark = [
    pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA"),
    pytest.mark.skipif(
        not _hf_cache_has_checkpoint(),
        reason=f"{MODEL_REPO} not in the local HF cache",
    ),
]

DEVICE = torch.device("cuda")

# Oracle-run latent grid: 480x832x33 frames -> (9, 30, 52), post-patch
# (9, 15, 26) = 3510 tokens.
LATENT_SHAPE = (2, 48, 9, 30, 52)


@pytest.fixture(scope="module")
def wrapped():
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        MODEL_REPO, subfolder="transformer", torch_dtype=torch.bfloat16
    ).eval()
    return model.to(DEVICE)


@pytest.fixture(scope="module")
def native():
    return build_wan22_dit(Wan22Config(), MODEL_REPO, device=DEVICE)


@pytest.fixture(scope="module")
def inputs():
    generator = torch.Generator(device="cpu").manual_seed(1234)
    hidden = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32).to(
        DEVICE, torch.bfloat16
    )
    text = torch.randn((2, 512, 4096), generator=generator, dtype=torch.float32).to(
        DEVICE, torch.bfloat16
    )
    # I2V-style per-token grid: frame-0 tokens zeroed, later frames at t=999.
    mask = torch.ones(1, 1, *LATENT_SHAPE[2:], dtype=torch.float32, device=DEVICE)
    mask[:, :, 0] = 0
    t = torch.tensor(999, dtype=torch.int64, device=DEVICE)
    temp_ts = (mask[0, 0][:, ::2, ::2] * t).flatten()
    timestep = temp_ts.unsqueeze(0).expand(2, -1)
    return hidden, timestep, text


# ---------------------------------------------------------------------------
# 1. Loaded weights: bijective remap, bit-identical tensors, matching islands
# ---------------------------------------------------------------------------


def test_loaded_parameters_bitwise_identical(wrapped, native):
    native_params = dict(native.named_parameters())
    wrapped_params = dict(wrapped.named_parameters())
    # 825 = the checkpoint's parameter-tensor count. A missing or extra tensor is a
    # different model, however well it happens to run.
    assert len(native_params) == len(wrapped_params) == 825

    # Equal counts do not prove a bijection: a remap that collided two keys onto one
    # would still match, leaving a native parameter never compared.
    remapped = {remap_checkpoint_key(name) for name in wrapped_params}
    assert len(remapped) == len(wrapped_params), (
        "remap is not injective — two checkpoint keys collided onto one native key: "
        f"{len(wrapped_params) - len(remapped)} collision(s)"
    )
    assert remapped == set(native_params), (
        "remap is not onto: "
        f"unmapped native params {sorted(set(native_params) - remapped)[:5]}, "
        f"remapped-to-nothing {sorted(remapped - set(native_params))[:5]}"
    )

    mismatched_dtype, mismatched_bits = [], []
    for wrapped_name, wrapped_param in wrapped_params.items():
        native_name = remap_checkpoint_key(wrapped_name)
        native_param = native_params[native_name]
        if native_param.dtype != wrapped_param.dtype:
            mismatched_dtype.append((wrapped_name, wrapped_param.dtype, native_param.dtype))
        elif not torch.equal(native_param, wrapped_param):
            mismatched_bits.append(wrapped_name)

    fp32_islands = sum(p.dtype == torch.float32 for p in native_params.values())
    print(f"parameters: {len(native_params)} compared, fp32 islands: {fp32_islands}")
    assert not mismatched_dtype, f"island/dtype mismatches: {mismatched_dtype[:5]}"
    assert not mismatched_bits, f"bit mismatches: {mismatched_bits[:5]}"
    # The wrapped module's _keep_in_fp32_modules yields exactly these 95:
    # time_embedder (4) + 31 scale_shift_tables + 30x norm2 weight+bias.
    assert fp32_islands == 95


def test_rope_tables_bitwise_identical(wrapped, native):
    native_cos, native_sin = native.rope.tables(DEVICE)
    for name, ours, theirs in (
        ("freqs_cos", native_cos, wrapped.rope.freqs_cos),
        ("freqs_sin", native_sin, wrapped.rope.freqs_sin),
    ):
        assert ours.dtype == theirs.dtype == torch.float32
        assert ours.shape == theirs.shape
        diff = (ours - theirs).abs().max().item()
        print(f"rope {name}: max_abs={diff:.3e}")
        assert torch.equal(ours, theirs), f"rope {name} tables differ (max_abs {diff:.3e})"


# ---------------------------------------------------------------------------
# 2. Per-stage forward: patchify -> condition embedder -> 30 blocks -> head
# ---------------------------------------------------------------------------


def _first_tensor(output):
    """Hook outputs are tensors or tuples (condition embedder, model head)."""
    return output[0] if isinstance(output, tuple) else output


def test_per_layer_outputs_bitwise_identical(wrapped, native, inputs):
    hidden, timestep, text = inputs

    stages: dict[str, list[torch.Tensor]] = {"wrapped": [], "native": []}
    hooks = []

    def capture(side, module):
        def hook(_module, _args, output):
            stages[side].append(_first_tensor(output).detach())
        hooks.append(module.register_forward_hook(hook))

    # Identical capture order on both sides: patchify, embedder temb,
    # blocks 0..29 (execution order), then the forward return as the head.
    for side, model in (("wrapped", wrapped), ("native", native)):
        capture(side, model.patch_embedding)
        capture(side, model.condition_embedder)
        assert len(model.blocks) == 30
        for block in model.blocks:
            capture(side, block)

    try:
        with torch.no_grad():
            wrapped_out = wrapped(
                hidden_states=hidden, timestep=timestep,
                encoder_hidden_states=text, return_dict=False,
            )[0]
            native_out = native(hidden_states=hidden, timestep=timestep, encoder_hidden_states=text)
    finally:
        for hook in hooks:
            hook.remove()

    stages["wrapped"].append(wrapped_out)
    stages["native"].append(native_out)
    labels = ["patchify", "time_embed"] + [f"block_{i:02d}" for i in range(30)] + ["head_out"]
    assert len(stages["wrapped"]) == len(stages["native"]) == len(labels)

    print("\nper-stage max-abs diff (wrapped vs native):")
    first_divergent = None
    for label, ours, theirs in zip(labels, stages["native"], stages["wrapped"], strict=True):
        assert ours.shape == theirs.shape, f"{label}: shape {tuple(ours.shape)} vs {tuple(theirs.shape)}"
        diff = (ours.float() - theirs.float()).abs().max().item()
        marker = ""
        if diff > 0 and first_divergent is None:
            first_divergent = label
            marker = " <- FIRST DIVERGENT"
        print(f"  {label:11s} max_abs={diff:.6e}{marker}")

    assert first_divergent is None, (
        f"native DiT diverges from the wrapped module; first divergent stage: {first_divergent}"
    )


def test_condition_embedder_all_outputs_bitwise_identical(wrapped, native, inputs):
    """The per-stage table sees only the embedder's first output (temb); pin
    the modulation projection and the projected text stream too."""
    _, timestep, text = inputs
    with torch.no_grad():
        ref_temb, ref_proj, ref_text, _ = wrapped.condition_embedder(
            timestep.flatten(), text, None, timestep_seq_len=timestep.shape[1]
        )
        ref_proj = ref_proj.unflatten(2, (6, -1))
        temb, proj, text_out = native.condition_embedder(timestep, text)
    for name, ours, theirs in (
        ("temb", temb, ref_temb),
        ("timestep_proj", proj, ref_proj),
        ("text_proj", text_out, ref_text),
    ):
        diff = (ours.float() - theirs.float()).abs().max().item()
        print(f"condition_embedder {name}: max_abs={diff:.3e}")
        assert torch.equal(ours, theirs), f"condition_embedder {name} diverges (max_abs {diff:.3e})"
