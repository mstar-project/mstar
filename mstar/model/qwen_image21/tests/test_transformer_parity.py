"""The native Qwen-Image-2.1 DiT against diffusers' ``QwenImage21Transformer2DModel``, tiny and on CPU.

One random checkpoint goes both ways: into the reference as is, into the port through its loader.
The reference runs the whole block-causal sequence (with and without its KV cache); the port runs
the prefix one span at a time over a paged-KV stand-in, text spans causal and image spans
bidirectional, then the target alone. Their prefix K/V and target velocities must agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Transformer2DModel"):
    pytest.skip("this diffusers has no Qwen-Image-2.1 (needs >= 0.41)", allow_module_level=True)

from diffusers.models.transformers.transformer_qwenimage21 import (  # noqa: E402
    QwenImage21AttnProcessor,
    QwenImage21FlexAttnProcessor,
    QwenImage21KVCache,
)

from mstar.model.qwen_image21.components.transformer import (  # noqa: E402
    TOKENS_PER_SLOT,
    QwenImage21Transformer,
    image_positions,
    text_positions,
)
from mstar.model.qwen_image21.tests.paged_stub import (  # noqa: E402
    CFG,
    IMAGE_PAD_ID,
    LAYOUT,
    SCALE,
    STEPS,
    TARGET,
    PagedAttendStub,
    Row,
    dit_node,
    dit_request,
    drive,
)
from mstar.model.qwen_image21.weight_loader import load_transformer  # noqa: E402


@pytest.fixture(autouse=True)
def _threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # bit-exact CPU comparisons
    yield
    torch.set_num_threads(threads)


def _models(dtype: torch.dtype, processor=None):
    torch.manual_seed(0)
    ref = diffusers.QwenImage21Transformer2DModel(
        in_channels=CFG.in_channels, out_channels=CFG.out_channels, num_layers=CFG.num_layers,
        attention_head_dim=CFG.attention_head_dim, num_attention_heads=CFG.num_attention_heads,
        context_in_dim=CFG.context_in_dim, mlp_ratio=CFG.mlp_ratio, axes_dims_rope=CFG.axes_dims_rope,
    ).eval()
    with torch.no_grad():
        for p in ref.parameters():  # zero-centred norms and unit RMSNorms included
            # parameters only: from_pretrained(torch_dtype=...) leaves the fp32 timestep frequencies alone
            p.data = (torch.randn_like(p) * 0.2).to(dtype)
    if processor is not None:
        ref.set_attn_processor(processor())
    ours = QwenImage21Transformer(CFG).to(dtype)
    load_transformer(ours, iter(ref.state_dict().items()))
    return ref, ours


def _inputs(dtype: torch.dtype):
    g = torch.Generator().manual_seed(1)
    vl_len = sum(part if isinstance(part, int) else part[0] * part[1] // TOKENS_PER_SLOT for part in LAYOUT)
    vl = torch.randn(1, vl_len, CFG.context_in_dim, generator=g).to(dtype)
    refs = [torch.randn(h * w, CFG.in_channels, generator=g).to(dtype) for h, w in LAYOUT[1::2]]
    target = torch.randn(1, TARGET[0] * TARGET[1], CFG.in_channels, generator=g).to(dtype)
    return vl, refs, target


def _reference(ref, vl, refs, target, timesteps, cache: bool):
    """The reference's velocities at each timestep, and its prefix K/V (from the extract step)."""
    mask = []
    for part in LAYOUT:
        mask += [False] * part if isinstance(part, int) else [True] * (part[0] * part[1] // TOKENS_PER_SLOT)
    mask += [True] * (TARGET[0] * TARGET[1] // TOKENS_PER_SLOT)
    shapes = [[*((1, h, w) for h, w in LAYOUT[1::2]), (1, *TARGET)]]
    kv = QwenImage21KVCache(CFG.num_layers) if cache else None
    out = []
    for i, t in enumerate(timesteps):
        hidden = torch.cat([*refs, target[0]])[None]
        velocity = ref(
            hidden_states=hidden, encoder_hidden_states=vl, timestep=(t.to(vl.dtype) / 1000)[None],
            img_shapes=shapes, img_mask=torch.tensor([mask]), kv_cache=kv,
            kv_cache_mode=None if not cache else ("extract" if i == 0 else "cached"), return_dict=False,
        )[0]
        out.append(velocity[:, -target.shape[1]:])
    return out, kv


def _ours(model, vl, refs, target, timesteps):
    attend = PagedAttendStub()
    stream = ("r", "main")
    vl_row, position, image = 0, 0, 0
    for part in LAYOUT:
        if isinstance(part, int):
            hidden = model.txt_in(vl[0, vl_row:vl_row + part])
            positions = text_positions(position, part)
            vl_row, position, causal = vl_row + part, position + part, True
        else:
            h, w = part
            hidden = model.img_in(refs[image])
            positions = image_positions(position, h, w)
            vl_row, position, image, causal = vl_row + h * w // TOKENS_PER_SLOT, position + max(h, w), image + 1, False
        model.prefix(hidden, positions, attend.step([Row(stream, hidden.shape[0])], causal=causal, commit=True))
    prefix_kv = {layer: kv for layer, kv in attend.cache[stream].items()}
    positions = image_positions(position, *TARGET)[None]
    out = []
    for t in timesteps:
        attend.step([Row(stream, positions.shape[1])], causal=False, commit=False)
        out.append(model.target(target, t[None], positions, attend))
    return out, prefix_kv


def _close(got: torch.Tensor, expected: torch.Tensor, bound: float) -> None:
    """Max-abs error relative to the reference's largest magnitude."""
    err = ((got.float() - expected.float()).abs().max() / expected.float().abs().max()).item()
    assert err <= bound, f"relative error {err:.3g} > {bound:g}"


# The per-segment SDPA reference runs the same ops in the same dtype: bf16 matches it to the bit.
# The flex reference's one-pass mask rounds differently; its own cached and uncached runs differ by
# about 2.3% in bf16.
@pytest.mark.parametrize("dtype,processor,bound", [
    (torch.float32, QwenImage21AttnProcessor, 1e-5),
    (torch.float32, QwenImage21FlexAttnProcessor, 1e-5),
    (torch.bfloat16, QwenImage21AttnProcessor, 0.0),
    (torch.bfloat16, QwenImage21FlexAttnProcessor, 5e-2),
])
@torch.no_grad()
def test_span_prefix_and_target_match_the_block_causal_reference(dtype, processor, bound):
    ref, ours = _models(dtype, processor)
    vl, refs, target = _inputs(dtype)
    timesteps = torch.tensor([1000.0, 613.5, 87.25])
    expected, ref_kv = _reference(ref, vl, refs, target, timesteps, cache=True)
    dense, _ = _reference(ref, vl, refs, target, timesteps, cache=False)
    got, our_kv = _ours(ours, vl, refs, target, timesteps)
    for layer in range(CFG.num_layers):
        k, v = ref_kv.get_layer(layer).get()
        _close(our_kv[layer][0], k[0], bound)
        _close(our_kv[layer][1], v[0], bound)
    for ours_t, cached_t, dense_t in zip(got, expected, dense, strict=True):
        _close(ours_t, cached_t, bound)
        _close(ours_t, dense_t, bound)


# ----------------------------------------------------------------------
# the dit node: its walks, declared steps and CFG, against the pipeline's loop
# ----------------------------------------------------------------------

@torch.no_grad()
def test_dit_node_matches_the_pipeline_with_cfg():
    """The model's plan for a 2-reference edit under CFG, run through the dit node (prefix walks,
    the fork, the two-label tail and the two-label denoise), equals the pipeline's loop: its cached
    prefix per branch, ``neg + s * (pos - neg)``, the scheduler's step. (Cache hits and batching
    against this run: ``test_model.py``.)"""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from mstar.model.components.diffusion.flow_match import calculate_shift
    from mstar.model.qwen_image21.submodules import DitKey

    dtype = torch.bfloat16
    ref, ours = _models(dtype, QwenImage21AttnProcessor)
    pos, neg, pos_vl, neg_vl, refs = dit_request(dtype)
    sub, attend, plan, embeds, latents_in = dit_node(ours, dtype)
    got = drive(sub, attend, "a", plan, embeds, latents_in)

    # the pipeline: every branch prefills its own cache on step 0
    scheduler = FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192, max_shift=0.9, shift=1.0,
        shift_terminal=0.02, use_dynamic_shifting=True,
    )
    n = TARGET[0] * TARGET[1]
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / STEPS, STEPS), mu=calculate_shift(n, 256, 8192, 0.5, 0.9))
    scheduler.set_begin_index(0)
    generator = torch.Generator().manual_seed(5)
    latents = sub.seed_latents(None, DitKey(*TARGET), generator)[None]
    caches = {name: QwenImage21KVCache(CFG.num_layers) for name in ("pos", "neg")}
    shapes = [[*((1, h, w) for h, w in LAYOUT[1::2]), (1, *TARGET)]]
    for i, t in enumerate(scheduler.timesteps):
        preds = {}
        for name, ids, vl in (("pos", pos, pos_vl), ("neg", neg, neg_vl)):
            mask = torch.cat([ids == IMAGE_PAD_ID, torch.ones(n // 4, dtype=torch.bool)])[None]
            preds[name] = ref(
                hidden_states=torch.cat([*refs, latents[0]])[None], encoder_hidden_states=vl[None],
                timestep=t.expand(1).to(dtype) / 1000, img_shapes=shapes, img_mask=mask, kv_cache=caches[name],
                kv_cache_mode="extract" if i == 0 else "cached", return_dict=False,
            )[0][:, -n:]
        noise = preds["neg"] + SCALE * (preds["pos"] - preds["neg"])
        latents = scheduler.step(noise, t, latents, return_dict=False)[0]
    torch.testing.assert_close(got, latents[0], rtol=0, atol=0)
