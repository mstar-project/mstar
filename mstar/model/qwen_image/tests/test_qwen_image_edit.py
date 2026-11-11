"""Qwen-Image-Edit on CPU: the dit's guided step against diffusers, the VAE nodes against the pipeline's math,
and the model's intake in dummy mode. The prompt construction against the pipeline's processor runs with a
checkpoint's processor files (``QWEN_IMAGE_EDIT_SNAPSHOT``)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

diffusers = pytest.importorskip("diffusers")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.graph.base import TensorPointerInfo  # noqa: E402
from mstar.model.components.diffusion.denoise_loop import LATENTS  # noqa: E402
from mstar.model.qwen_image.components.transformer import (  # noqa: E402
    QwenImageTransformer,
    QwenImageTransformerConfig,
)
from mstar.model.qwen_image.config import DENOISE_LOOP, QwenImageConfig  # noqa: E402
from mstar.model.qwen_image.qwen_image_model import (  # noqa: E402
    ENCODE_EDIT_WALK,
    ENCODE_REFS_WALK,
    IMAGE_EDIT_WALK,
    QwenImageModel,
)
from mstar.model.qwen_image.submodules import (  # noqa: E402
    IMAGE_INPUTS,
    PROMPT_LENGTHS,
    REF_LATENTS,
    TEXT_EMBEDS_NEG,
    TEXT_EMBEDS_POS,
    TEXT_INPUTS,
    VL_GRIDS,
    VL_PIXELS,
    QwenImageDenoiseSubmodule,
    QwenImageVaeDecoderSubmodule,
    QwenImageVaeEncoderSubmodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

TINY = dict(in_channels=64, out_channels=16, num_layers=2, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=24, axes_dims_rope=(4, 6, 6), patch_size=2)
DROP = 64


@pytest.fixture(autouse=True)
def _threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # bit-exact CPU comparisons
    yield
    torch.set_num_threads(threads)


def _transformers(zero_cond_t: bool):
    from diffusers import QwenImageTransformer2DModel

    torch.manual_seed(0)
    reference = QwenImageTransformer2DModel(**TINY, zero_cond_t=zero_cond_t).eval()
    with torch.no_grad():
        for p in reference.parameters():
            p.normal_(std=0.2)
    ours = QwenImageTransformer(QwenImageTransformerConfig(**TINY, zero_cond_t=zero_cond_t)).eval()
    ours.load_state_dict(reference.state_dict())
    return reference, ours


class _Group:
    """One rank of a two-rank cfg group, whose all-gather returns every rank's branches."""

    def __init__(self, rank: int, full: torch.Tensor):
        self.world_size, self.rank, self.full = 2, rank, full

    def all_gather(self, x, dim):
        assert dim == 1 and torch.equal(x, self.full[:, self.rank:self.rank + 1])
        return self.full


def _info(rid, meta):
    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=IMAGE_EDIT_WALK, fwd_index=1, random_seed=0, max_tokens=0,
        rid_handle=hash(rid) % 1000, step_metadata=meta, dynamic_loop_iter_counts={DENOISE_LOOP: 1},
    )


def _guided(node, requests, meta):
    """``guided_output`` for one stacked step over ``requests`` of (latents, refs, pos, neg)."""
    rids = [f"r{i}" for i in range(len(requests))]
    inputs = [
        node.prepare_inputs(IMAGE_EDIT_WALK, _info(rid, meta), {
            LATENTS: [lat], REF_LATENTS: [refs], TEXT_EMBEDS_POS: [pos], TEXT_EMBEDS_NEG: [neg],
        })
        for rid, (lat, refs, pos, neg) in zip(rids, requests, strict=True)
    ]
    engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info={})
    kwargs = node.preprocess(IMAGE_EDIT_WALK, engine_inputs, inputs)
    key = kwargs.pop("step_key")
    latents, timestep = kwargs.pop(LATENTS), kwargs.pop("timestep")
    for name in ("sigma", "sigma_next"):
        kwargs.pop(name)
    with torch.no_grad():
        return key, node.guided_output(engine_inputs, key, latents, timestep, **kwargs)


def _branch(reference, timestep, grid, ref_grids):
    """diffusers' forward for one prompt alone, as the pipeline calls it per branch: the target's prediction."""
    img_shapes = [[(1, *grid), *((1, h, w) for h, w in ref_grids)]]

    def run(latents, refs, embeds):
        out = reference(hidden_states=torch.cat([latents, refs])[None], timestep=timestep / 1000,
                        encoder_hidden_states=embeds[None], img_shapes=img_shapes, return_dict=False)[0]
        return out[0, :latents.shape[0]]

    return run


def _renorm_cfg(cond, uncond, scale):
    """The pipeline's ``true_cfg_scale`` combine."""
    comb = uncond + scale * (cond - uncond)
    return comb * (torch.norm(cond, dim=-1, keepdim=True) / torch.norm(comb, dim=-1, keepdim=True))


def _request(g, ref_grids, lengths):
    return (torch.randn(16, 64, generator=g), torch.randn(sum(h * w for h, w in ref_grids), 64, generator=g),
            torch.randn(lengths[0], 24, generator=g), torch.randn(lengths[1], 24, generator=g))


@pytest.mark.parametrize("zero_cond_t", [True, False])
@pytest.mark.parametrize("ref_grids", [((4, 6),), ((4, 6), (2, 4))], ids=["one-ref", "two-refs"])
def test_a_step_matches_diffusers(zero_cond_t, ref_grids):
    """One request unguided is diffusers' forward bit for bit (SDPA within each segment). Two guided requests in
    one forward, their branches as rows and each negative prompt left-padded, match diffusers' per-branch forwards
    and the pipeline's norm-rescaled CFG to the GEMM noise of batching rows (the GEMMs' shapes change)."""
    reference, ours = _transformers(zero_cond_t)
    config = QwenImageConfig(transformer=ours.config)
    lengths = (9, 5)
    meta = {"height": 64, "width": 64, "num_inference_steps": 4, "ref_grids": ref_grids,
            PROMPT_LENGTHS: [n + DROP for n in lengths]}
    g = torch.Generator().manual_seed(1)
    requests = [_request(g, ref_grids, lengths) for _ in range(2)]

    def node():
        sub = QwenImageDenoiseSubmodule(ours, config, loop_name=DENOISE_LOOP, attn_resource_key=None)
        sub.get_device = lambda: torch.device("cpu")
        return sub

    alone = node()
    key, pred = _guided(alone, requests[:1], {**meta, "guidance_scale": 1.0})
    timestep = alone.request_state(_info("r0", meta).rid_handle)["timesteps"][1:2]
    branch = _branch(reference, timestep, (4, 4), ref_grids)
    with torch.no_grad():
        assert not key.guided and torch.equal(pred[0], branch(*requests[0][:3]))

        key, pred = _guided(node(), requests, {**meta, "guidance_scale": 4.0})
        want = torch.stack([_renorm_cfg(branch(lat, refs, pos), branch(lat, refs, neg), 4.0)
                            for lat, refs, pos, neg in requests])
    assert key.guided and key.branches == (0, 1)
    # batching two rows moved a branch's prediction by up to 4e-6 here (fp32, CPU); the combine weighs the two
    # branches 4 and 3, so up to 7x that, with a 1.5x margin
    torch.testing.assert_close(pred, want, rtol=0, atol=7 * 4e-6 * 1.5)


@pytest.mark.parametrize("rank", [0, 1])
def test_a_cfg_rank_runs_its_branch_unpadded(rank):
    """Under cfg_size 2 each rank forwards its own prompt alone at its own length (the negative unpadded on rank
    1), bit for bit diffusers' forward of that prompt, and every rank combines the gathered pair as the pipeline."""
    reference, ours = _transformers(zero_cond_t=True)
    config = QwenImageConfig(transformer=ours.config)
    ref_grids, lengths = ((4, 6),), (9, 5)
    meta = {"height": 64, "width": 64, "num_inference_steps": 4, "guidance_scale": 4.0, "ref_grids": ref_grids,
            PROMPT_LENGTHS: [n + DROP for n in lengths]}
    latents, refs, pos, neg = request = _request(torch.Generator().manual_seed(2), ref_grids, lengths)
    probe = QwenImageDenoiseSubmodule(ours, config, loop_name=DENOISE_LOOP, attn_resource_key=None)
    probe.get_device = lambda: torch.device("cpu")
    probe.prepare_inputs(IMAGE_EDIT_WALK, _info("r0", meta), {LATENTS: [latents], REF_LATENTS: [refs],
                                                             TEXT_EMBEDS_POS: [pos], TEXT_EMBEDS_NEG: [neg]})
    branch = _branch(reference, probe.request_state(_info("r0", meta).rid_handle)["timesteps"][1:2], (4, 4), ref_grids)
    with torch.no_grad():
        cond, uncond = branch(latents, refs, pos), branch(latents, refs, neg)
    node = QwenImageDenoiseSubmodule(ours, config, loop_name=DENOISE_LOOP, attn_resource_key=None,
                                     cfg_group=_Group(rank, torch.stack([cond, uncond])[None]))
    node.get_device = lambda: torch.device("cpu")
    key, pred = _guided(node, [request], meta)
    assert key.branches == (rank,) and node.attention_segments(key) == (("main", lengths[rank] + 16 + 24),)
    assert torch.equal(pred[0], _renorm_cfg(cond, uncond, 4.0))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"))])
def test_rope_rounds_as_the_reference(device):
    """The rotation is diffusers' complex product bit for bit on each device: CUDA's complex multiply rounds through
    fused multiply-adds, so a real-arithmetic rotation that matched on CPU drifted 0.4% over a GPU step."""
    from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

    from mstar.model.qwen_image.components.transformer import apply_rope, qwen_rope_tables

    img_freqs, txt_freqs = qwen_rope_tables((16, 56, 56), [(16, 24), (8, 8)], [7], 7)
    g = torch.Generator().manual_seed(0)
    for freqs in (img_freqs, txt_freqs[0]):
        x = torch.randn(1, freqs.shape[0], 4, 128, generator=g).to(device, torch.bfloat16)
        freqs = freqs.to(device)
        assert torch.equal(apply_rope(x, freqs), apply_rotary_emb_qwen(x, freqs, use_real=False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels run on CUDA")
@pytest.mark.parametrize("zero_cond_t", [True, False])
def test_the_fused_kernels_round_as_the_eager_ops(zero_cond_t, monkeypatch):
    """On CUDA the q/k norms with the rotation, the LayerNorms with the modulation, and the gated residuals run as
    fused kernels. They round where the eager ops do (the LayerNorm's statistics in torch's own order), so a bf16
    forward is the forward with the eager elementwise ops bit for bit (the MLP's GELU epilogue, the one fusion that
    is not exact, runs in both: the next test's). The copy-free GEMMs are held to bf16 rounding of the plain path.
    The kernels alone: ``test/modular/test_diffusion_dit_kernels.py``."""
    from mstar.model.components.diffusion.attention import joint_attention
    from mstar.model.components.diffusion.qwen import dit_kernels as kernels
    from mstar.model.qwen_image.components import transformer
    from mstar.model.qwen_image.components.transformer import QwenStepLayout, _GeluProj, qwen_rope_tables

    # four 128-wide heads: a 512-wide model, the narrowest the LayerNorm kernel takes
    config = QwenImageTransformerConfig(
        **{**TINY, "attention_head_dim": 128, "num_attention_heads": 4, "axes_dims_rope": (16, 56, 56)},
        zero_cond_t=zero_cond_t)
    torch.manual_seed(0)
    dit = QwenImageTransformer(config).eval()
    with torch.no_grad():
        for p in dit.parameters():
            p.normal_(std=0.2)
    dit = dit.to("cuda", torch.bfloat16)
    # two rows (the guidance branches), the target 4x4 then a 4x6 reference, the second prompt left-padded
    img_freqs, txt_freqs = qwen_rope_tables(config.axes_dims_rope, [(4, 4), (4, 6)], [7, 5], 7)
    layout = QwenStepLayout(img_freqs.cuda(), txt_freqs.cuda(), 16)
    g = torch.Generator().manual_seed(1)
    img = torch.randn(2, 40, 64, generator=g).to("cuda", torch.bfloat16)
    txt = torch.randn(2, 7, 24, generator=g).to("cuda", torch.bfloat16)
    timestep = torch.tensor([0.3, 0.7], device="cuda", dtype=torch.bfloat16)

    def forward():
        with torch.no_grad():
            return dit(img, txt, timestep, layout, joint_attention)

    fused = forward()
    monkeypatch.setattr(kernels, "fits", lambda *tensors: False)
    assert torch.equal(fused, forward())
    # the copy-free GEMMs (v written into the joint tensor, the output projections reading in place) against the
    # plain cat / contiguous path: the same math in a different GEMM shape, so cuBLAS may split K differently
    # and the bits are not promised (bit-exact at the served shape on H200). The GELU epilogue's
    # projections stay as they are.
    gelu_projs = {id(m.proj) for m in dit.modules() if isinstance(m, _GeluProj)}
    monkeypatch.setattr(transformer, "_plain", lambda linear, cls: id(linear) in gelu_projs)
    torch.testing.assert_close(fused, forward(), rtol=3e-2, atol=3e-2)  # a few bf16 steps (2^-7 each)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the GELU epilogue runs on CUDA")
def test_the_mlp_up_projection_takes_its_bias_and_gelu_in_the_gemm():
    """On CUDA the MLP's up-projection is one GEMM with the bias and the tanh-GELU in cuBLASLt's epilogue (as SGLang
    runs it). That rounds once, from the fp32 accumulator, where eager rounds the GEMM's output and then the GELU's:
    not eager's bits, but within one bf16 step of the exact result everywhere and nearer it than eager on average."""
    from mstar.distributed.communication import CommGroup
    from mstar.model.qwen_image.components.transformer import _GeluProj

    torch.manual_seed(0)
    # the served width (3072 -> 12288), two rows of an odd token count
    up = _GeluProj(CommGroup.trivial(), 3072, 4 * 3072)
    with torch.no_grad():
        up.proj.weight.normal_(std=3072 ** -0.5)
        up.proj.bias.normal_(std=0.5)
    up = up.to("cuda", torch.bfloat16)
    x = torch.randn(2, 37, 3072).to("cuda", torch.bfloat16)
    weight, bias = up.proj.weight, up.proj.bias
    with torch.no_grad():
        got = up(x)
        eager = F.gelu(up.proj(x), approximate="tanh")
        exact = F.gelu(torch.addmm(bias.double(), x.flatten(0, 1).double(), weight.double().t()),
                       approximate="tanh").view(got.shape)
    # one bf16 step (2^-7 relative); the absolute floor covers the fp32 accumulation where the GELU is near zero
    torch.testing.assert_close(got.double(), exact, rtol=2 ** -7, atol=1e-5)
    assert (got.double() - exact).abs().mean() < (eager.double() - exact).abs().mean()


def _vaes():
    from diffusers import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

    # the checkpoint's three downsamples (8x), narrow
    cfg = dict(base_dim=4, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1, temperal_downsample=[False, True, True],
               latents_mean=[0.1 * i for i in range(16)], latents_std=[1.0 + 0.05 * i for i in range(16)])
    torch.manual_seed(0)
    qwen = AutoencoderKLQwenImage(**cfg).eval()
    wan = AutoencoderKLWan(**cfg).eval()
    wan.load_state_dict(qwen.state_dict())
    return qwen, wan


def test_the_vae_nodes_match_the_pipelines_math():
    """Encode: the image processor's pixels, to the bit and memory layout (on an H200 a contiguous input moved the
    latents 1e-3, cuDNN picking kernels by layout), then the posterior's mode, (z - mean) / std, packed; decode:
    unpacked, z / (1 / std) + mean, rounded as the pipeline's image processor does. The nodes skip the VAE's
    one-frame feature cache (written, never read): a diffusers change that reads it would break this. The Wan
    class holds the QwenImage checkpoint."""
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline as P
    from PIL import Image

    qwen, wan = _vaes()
    image = torch.randint(0, 256, (3, 32, 48), dtype=torch.uint8)
    pil = Image.fromarray(image.permute(1, 2, 0).numpy())
    pixels = VaeImageProcessor(vae_scale_factor=16).preprocess(pil, height=32, width=48).unsqueeze(2)
    seen = []
    wan.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    encoder = QwenImageVaeEncoderSubmodule(wan)
    encoder.get_device = lambda: torch.device("cpu")
    with torch.no_grad():
        ours = encoder.forward(IMAGE_INPUTS, None, image_0=image)[REF_LATENTS][0]
        z = qwen.encode(pixels).latent_dist.mode()
        mean = torch.tensor(qwen.config.latents_mean).view(1, 16, 1, 1, 1)
        std = torch.tensor(qwen.config.latents_std).view(1, 16, 1, 1, 1)
        want = P._pack_latents((z - mean) / std, 1, 16, z.shape[3], z.shape[4])[0]
    assert torch.equal(seen[0], pixels) and seen[0].stride() == pixels.stride()
    assert torch.equal(ours, want)

    config = QwenImageConfig()
    decoder = QwenImageVaeDecoderSubmodule(wan, config)
    decoder.get_device = lambda: torch.device("cpu")
    with torch.no_grad():
        image_out = decoder.run_batch(want[None], (2, 3))["image_output"]
        latents = P._unpack_latents(want[None], 32, 48, 8) / (1.0 / std) + mean
        decoded = qwen.decode(latents, return_dict=False)[0][:, :, 0]
        want_out = ((decoded * 0.5 + 0.5).clamp(0, 1).float() * 255).round().to(torch.uint8)
    assert torch.equal(image_out, want_out)


def test_the_compiled_vae_regions_serve_other_sizes_without_recompiling():
    """Each region warmed on its ``compile_regions`` examples (the engine compiles it with dynamic=True) must take a
    live request's input as it comes, any size and decode batch: a recompile there stalls that request for minutes."""
    _, wan = _vaes()
    encoder, decoder = QwenImageVaeEncoderSubmodule(wan, compile=True), QwenImageVaeDecoderSubmodule(
        wan, QwenImageConfig(), compile=True)
    for sub in (encoder, decoder):
        sub.get_device = lambda: torch.device("cpu")
        for path, examples in sub.compile_regions().items():
            module = sub.get_submodule(path)
            module.forward = torch.compile(module.forward, dynamic=True, backend="eager")
            with torch.no_grad():
                for args in examples:
                    module.forward(*args)
    try:  # sizes above the warm-up's: on CPU (not CUDA) the conv backend switches below a size, a guard of its own
        with torch._dynamo.config.patch(error_on_recompile=True), torch.no_grad():
            for (h, w), batch in (((464, 336), 1), ((480, 480), 3)):
                encoder.forward(IMAGE_INPUTS, None, image_0=torch.zeros(3, h, w, dtype=torch.uint8))
                grid = (h // 16, w // 16)
                decoder.run_batch(torch.randn(batch, grid[0] * grid[1], 64), grid)
    finally:
        torch._dynamo.reset()


class _FakeProcessor:
    """The checkpoint processor's outputs, sized from the request: one pad per merged image token."""

    def __call__(self, text, images, padding, return_tensors):
        grids = [[1, 2 * (im.height // 28), 2 * (im.width // 28)] for im in images]
        pads = sum(h * w // 4 for _, h, w in grids)
        return {"input_ids": torch.zeros(1, len(text[0]) + pads, dtype=torch.long),
                "pixel_values": torch.zeros(sum(h * w for _, h, w in grids), 1176),
                "image_grid_thw": torch.tensor(grids)}


def _model():
    model = QwenImageModel(model_path_hf="test/qwen-image-edit", skip_weight_loading=True)
    model.set_config(QwenImageConfig())
    model._processor = _FakeProcessor()
    return model


def test_intake_plans_the_edit_and_refuses_what_it_cannot_serve():
    model = _model()
    images = [torch.zeros(3, 300, 400, dtype=torch.uint8), torch.zeros(3, 600, 300, dtype=torch.uint8)]
    out = model.process_prompt("make it blue", ["image", "text"], ["image"], tensors={IMAGE_INPUTS: images})
    tensors, grids = out.new_input_tensors, out.metadata[VL_GRIDS]
    assert len(tensors[TEXT_INPUTS]) == 2 and len(grids) == 2  # guided by default: both prompts
    # each reference at ~1024^2 in its own aspect, the default output the last one's
    assert [tuple(t.shape) for t in tensors[IMAGE_INPUTS]] == [(3, 896, 1184), (3, 1440, 736)]
    assert model.max_input_files() == {"image": 3}

    def info(t):
        return TensorPointerInfo(dims=list(t.shape), dtype=t.dtype, nbytes=0, address=0, stride=[1] * t.ndim, uuid=0,
                                 source_session_id="test:0", source_entity="test")

    signals = {
        TEXT_INPUTS: [info(t) for t in tensors[TEXT_INPUTS]],
        IMAGE_INPUTS: [info(t) for t in tensors[IMAGE_INPUTS]],
        VL_PIXELS: [info(tensors[VL_PIXELS][0])],
    }
    plan = model.walk_plan("default", ["image", "text"], ["image"], signals, out.metadata)
    assert [step.walk for step in plan] == [ENCODE_EDIT_WALK, ENCODE_REFS_WALK, IMAGE_EDIT_WALK]
    meta = plan[0].step_metadata
    assert (meta["height"], meta["width"], meta["num_inference_steps"], meta["guidance_scale"]) == (1440, 736, 40, 4.0)
    assert meta["ref_grids"] == [[56, 74], [90, 46]] and meta[VL_GRIDS] == grids
    assert model.get_loop_caps(num_inference_steps=12) == {DENOISE_LOOP: 12}

    unguided = model.process_prompt("x", ["image", "text"], ["image"], tensors={IMAGE_INPUTS: images[:1]},
                                    guidance_scale=1.0)
    assert len(unguided.new_input_tensors[TEXT_INPUTS]) == 1  # no negative prompt to encode
    with pytest.raises(ValueError, match="send the CFG scale as guidance_scale"):
        model.process_prompt("x", ["image", "text"], ["image"], tensors={IMAGE_INPUTS: images}, true_cfg_scale=4.0)
    with pytest.raises(ValueError, match="edits images"):
        model.process_prompt("x", ["text"], ["image"], tensors={})


@pytest.mark.parametrize("prompt,kwargs,match", [
    ("x", {"num_inference_steps": 1}, r"num_inference_steps must be an integer in \[2, "),
    ("x" * 5000, {}, "rotary positions"),
    ("x", {"height": 16, "width": 262144}, "rotary positions"),  # within max_image_area
], ids=["one-step-shift-terminal", "long-prompt", "long-axis"])
def test_intake_refuses_what_the_dit_would_fail_mid_run(prompt, kwargs, match):
    """A 400 at intake, not a failed request after the encoders: a one-step schedule stretched to shift_terminal,
    and text whose rotary positions run past the table's end."""
    images = [torch.zeros(3, 300, 400, dtype=torch.uint8)]
    with pytest.raises(ValueError, match=match):
        _model().process_prompt(prompt, ["image", "text"], ["image"], tensors={IMAGE_INPUTS: images}, **kwargs)


@pytest.mark.skipif(not os.environ.get("QWEN_IMAGE_EDIT_SNAPSHOT"), reason="needs a checkpoint's processor files")
def test_prompts_match_the_pipelines_processor_call():
    """With the checkpoint's processor: the token ids, patches and grids the pipeline's ``encode_prompt`` makes
    from the same uploads (its resize to each area, the "Picture N:" template)."""
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions
    from PIL import Image
    from transformers import AutoProcessor

    snapshot = Path(os.environ["QWEN_IMAGE_EDIT_SNAPSHOT"])
    model = QwenImageModel(model_path_hf=str(snapshot), skip_weight_loading=True)
    model.set_config(QwenImageConfig())
    model._snapshot = snapshot
    rng = np.random.default_rng(0)
    arrays = [rng.integers(0, 256, (300, 400, 3), dtype=np.uint8), rng.integers(0, 256, (500, 260, 3), dtype=np.uint8)]
    images = [torch.from_numpy(a).permute(2, 0, 1).contiguous() for a in arrays]
    out = model.process_prompt("make the sky purple", ["image", "text"], ["image"], tensors={IMAGE_INPUTS: images})

    processor = AutoProcessor.from_pretrained(str(snapshot / "processor"))
    resizer = VaeImageProcessor(vae_scale_factor=16)
    pils = [Image.fromarray(a) for a in arrays]
    condition = [resizer.resize(p, *reversed(calculate_dimensions(384 * 384, p.width / p.height))) for p in pils]
    prefix = "".join(f"Picture {i}: <|vision_start|><|image_pad|><|vision_end|>" for i in (1, 2))
    template = model.config.recipe.template
    want = [processor(text=[template.format(prefix + text)], images=condition, padding=True, return_tensors="pt")
            for text in ("make the sky purple", " ")]
    tensors = out.new_input_tensors
    assert [ids.tolist() for ids in tensors[TEXT_INPUTS]] == [w["input_ids"][0].tolist() for w in want]
    assert torch.equal(tensors[VL_PIXELS][0], want[0]["pixel_values"])
    assert out.metadata[VL_GRIDS] == want[0]["image_grid_thw"].tolist()
    for tensor, pil in zip(tensors[IMAGE_INPUTS], pils, strict=True):
        w, h = calculate_dimensions(1024 * 1024, pil.width / pil.height)
        expect = (resizer.preprocess(pil, h, w)[0] + 1.0) / 2.0 * 255.0
        assert torch.equal(tensor.float(), expect.round())
