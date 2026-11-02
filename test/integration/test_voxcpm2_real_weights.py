"""Real-weight tests for the VoxCPM2 port against the official model's outputs.

Never downloads: the module skips unless ``openbmb/VoxCPM2`` is already in a
local Hugging Face cache (or ``VOXCPM2_PATH`` names a snapshot). The fixtures
under ``test/voxcpm2/`` were recorded from the official ``VoxCPM2Model``
(``voxcpm`` at cce58a5): the prefill token layout of five requests, and the
fp32 trajectory (per-step latent patches and stop flags, seed 42) of the
voice-design request, on a B200 with TF32 off and the LongRoPE tables in
fp32. The teacher-forced check also needs CUDA.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

HF_REPO = "openbmb/VoxCPM2"
FIXTURES = Path(__file__).resolve().parents[1] / "voxcpm2"
# fp32 on both sides gives a median of 1e-6 and a worst patch of 2e-6 on the
# recording GPU; the margin covers another GPU's rounding amplified by the CFM
MEDIAN_REL_L2 = 1e-5
WORST_REL_L2 = 1e-3


def _helper(module: str):
    """A helper module from ``test/voxcpm2``, loaded by path: ``import test``
    can find the standard library's ``test`` package instead of this tree."""
    spec = importlib.util.spec_from_file_location(f"voxcpm2_{module}", FIXTURES / f"{module}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _find_snapshot() -> Path | None:
    if os.environ.get("VOXCPM2_PATH"):
        return Path(os.environ["VOXCPM2_PATH"])
    roots = [Path(os.environ[k]) / sub for k, sub in (("HF_HUB_CACHE", ""), ("HF_HOME", "hub")) if os.environ.get(k)]
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        snapshots = root / f"models--{HF_REPO.replace('/', '--')}" / "snapshots"
        for snapshot in sorted(snapshots.iterdir()) if snapshots.is_dir() else []:
            if (snapshot / "model.safetensors").is_file() and (snapshot / "audiovae.pth").is_file():
                return snapshot
    return None


SNAPSHOT = _find_snapshot()
pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason=f"{HF_REPO} is not in a local Hugging Face cache")


@pytest.fixture(scope="module")
def model():
    from mstar.model.voxcpm2.voxcpm2_model import VoxCPM2Model

    return VoxCPM2Model(model_path_hf=str(SNAPSHOT))


@pytest.fixture(scope="module")
def prompts() -> dict:
    return _helper("official_prompts").OFFICIAL_PROMPTS


@pytest.fixture(scope="module")
def design() -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(FIXTURES / "official_design_fp32.safetensors"))


def test_prompts_match_the_reference_layout(model, prompts):
    """The real tokenizer (with CJK splitting) and ``build_prompt`` give the
    official prefill tokens and text mask for every prompt mode."""
    from mstar.model.voxcpm2.config import PROMPT_MODE_KEY
    from mstar.model.voxcpm2.submodules import build_prompt

    for name, case in prompts.items():
        kwargs = {"instructions": case["instruct"]} if case["instruct"] else {}
        tensors, modalities = None, ["text"]
        if case["ref_patches"]:
            clip = torch.zeros(case["ref_patches"] * model.config.ref_patch_samples)
            tensors, modalities = {"audio_inputs": [clip]}, ["text", "audio"]
            kwargs["ref_text"] = case["ref_text"]
        out, metadata = model.process_prompt(case["text"], modalities, ["audio"], tensors=tensors, **kwargs)
        mode = metadata[PROMPT_MODE_KEY]
        ref = torch.zeros(2, case["ref_patches"], model.config.patch_size, model.config.feat_dim)
        tokens, text_mask, _, _ = build_prompt(out["text_inputs"][0], mode, ref if case["ref_patches"] else None)
        assert tokens.tolist() == case["tokens"], name
        assert text_mask.int().tolist() == case["text_mask"], name


def _vae_node(model, dtype):
    from mstar.model.voxcpm2.submodules import VAEDecoderSubmodule
    from mstar.model.voxcpm2.voxcpm2_model import load_audio_vae

    return VAEDecoderSubmodule(load_audio_vae(model.config, str(SNAPSHOT), "cpu").to(dtype), model.config)


def test_whitespace_runs_reach_the_tokenizer_as_one_space(model):
    """The checkpoint's tokenizer gives doubled spaces and tabs tokens of their
    own; the prompt drops them, as the reference's ``re.sub`` does."""
    spaced, single = "Hello  there,\tworld.   Bye", "Hello there, world. Bye"
    assert len(model.tokenize(spaced)) > len(model.tokenize(single))
    ids = [model.process_prompt(text, ["text"], ["audio"])[0]["text_inputs"][0].tolist() for text in (spaced, single)]
    assert ids[0] == ids[1]


def test_vae_stream_equals_a_whole_decode(model, design):
    """#429: the chunk-policy stream has the whole decode's length exactly and,
    in float64, its samples up to rounding; one patch less context than the
    receptive field does not. Bit equality depends on the CPU thread count
    (1e-15 at 1 to 4 threads, 0 at 8), the short context is off by about 6e-7."""
    from mstar.streaming.chunk_policy import ScheduledLeftContextChunkPolicy

    stream_through = _helper("streaming").stream_through

    patches = design["patches"][:14].double()
    node = _vae_node(model, torch.float64)
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    short = ScheduledLeftContextChunkPolicy(
        schedule=model.stream_schedule, chunk=model.stream_chunk_patches, left_context=model.stream_left_context - 1
    )
    with torch.no_grad():
        whole = node.vae.decode(patches.reshape(-1, 64).t().unsqueeze(0))[0]
        streamed = torch.cat(stream_through(node, patches, policy))
        cut = torch.cat(stream_through(node, patches, short))
    assert streamed.numel() == cut.numel() == whole.numel() == 14 * model.config.patch_samples
    assert (streamed - whole).abs().max() <= 1e-12
    assert (cut - whole).abs().max() > 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_teacher_forced_fp32_steps_match_the_reference(model, prompts, design, monkeypatch):
    """Fed the official patch k - 1, the fp32 Talker predicts official patch k
    with the same seeded CFM noise, and the stop head fires on the same step."""
    from mstar.model.components import norm
    from mstar.model.voxcpm2.components.local import cfm_t_span
    from mstar.model.voxcpm2.voxcpm2_model import load_generator

    bind_dense_resources = _helper("fake_resources").bind_dense_resources

    # mstar::rms_norm runs fp32 inputs in bf16 on CUDA
    monkeypatch.setattr(
        norm,
        "run_rms_norm",
        lambda x, weight, eps=1e-6, rms_norm_dtype=None: (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
        ),
    )
    case = prompts["design"]
    device = torch.device("cuda")
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        generator = load_generator(model.config, str(SNAPSHOT), device, torch.float32)
        plan, reset = bind_dense_resources(generator)
        forced = design["patches"].to(device)
        noise = torch.Generator(device=device).manual_seed(case["seed"])
        t_span = cfm_t_span(model.inference_timesteps, device, torch.float32)
        cfg = torch.tensor([2.0], device=device)
        tokens = torch.tensor(case["tokens"], device=device)
        empty = torch.zeros(0, 4, 64, device=device)
        preds, stops = [], []
        with torch.inference_mode():
            reset()
            plan([tokens.numel()])
            patch, stop = generator.prefill(
                tokens,
                torch.ones_like(tokens, dtype=torch.bool),
                empty,
                tokens.new_zeros(0),
                torch.tensor([tokens.numel() - 1], device=device),
                torch.zeros(1, 4, 64, device=device),
                torch.randn((1, 64, 4), generator=noise, device=device),
                cfg,
                t_span,
            )
            preds.append(patch[0])
            stops.append(int(stop.argmax(-1)))
            for k in range(1, forced.shape[0]):
                plan([1])
                step_noise = torch.randn((1, 64, 4), generator=noise, device=device)
                patch, stop = generator.decode_step(forced[k - 1 : k], step_noise, cfg, t_span)
                preds.append(patch[0])
                stops.append(int(stop.argmax(-1)))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    preds = torch.stack(preds)
    rel = ((preds - forced).flatten(1).norm(dim=1) / forced.flatten(1).norm(dim=1)).cpu()
    assert stops == design["stops"].tolist()
    assert rel.median() < MEDIAN_REL_L2 and rel.max() < WORST_REL_L2, rel.tolist()
