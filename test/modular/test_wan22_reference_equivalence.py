"""Reference-equivalence tests: mstar's Wan2.2-TI2V-5B vs the diffusers oracle.

The reference is a recorded run of the stock diffusers ``WanPipeline`` (T2V, seed
42, 50 UniPC steps): the latents after every step, the final latents, the exported
mp4, and a metadata.json holding every resolved parameter.

An oracle is only valid for the torch build and numerics flags it was recorded
under, so record it on the machine you are testing. The suite skips without CUDA,
without the checkpoint in the local HF cache, or without ``WAN22_ORACLE_DIR``.

    export WAN22_ORACLE_DIR=/somewhere/with/room/wan22_oracle
    CUDA_VISIBLE_DEVICES=0 /path/to/wan-oracle-venv/bin/python test/wan22/record_oracle.py \
        --out-dir "$WAN22_ORACLE_DIR"
    CUDA_VISIBLE_DEVICES=0 pytest test/modular/test_wan22_reference_equivalence.py -v -s

The initial noise is regenerated here from the seed rather than loaded, so these
tests do not depend on the recorded run's RNG state.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import ModelInputsFromEngine, NodeSubmodule
from mstar.model.wan22.submodules import (
    Wan22DitSubmodule,
    Wan22TextEncoderSubmodule,
)
from mstar.model.wan22.wan22_model import Wan22Model

MODEL_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# No default path on purpose: unset, the suite skips with instructions rather than
# silently comparing against someone else's numerics.
_ORACLE_ENV = os.environ.get("WAN22_ORACLE_DIR", "")
# Path("") is PosixPath("."), so test the env string, not the Path.
ORACLE_DIR = Path(_ORACLE_ENV) if _ORACLE_ENV else None

# A small guidance_scale=1.0 oracle (record_oracle.py --guidance 1.0) for the
# no-CFG batch-1 trajectory; its test skips when this is unset.
_NOCFG_ENV = os.environ.get("WAN22_NOCFG_ORACLE_DIR", "")
NOCFG_ORACLE_DIR = Path(_NOCFG_ENV) if _NOCFG_ENV else None

# ---------------------------------------------------------------------------
# Thresholds. NEVER loosen one to make a run pass — a regression is a finding.
# mstar's only deliberate departure is batching the two CFG forwards into one
# batch-2 call, so each bound below is either 0.0 or prices that batching's noise.
# ---------------------------------------------------------------------------
# Same tokenizer, same UMT5 instance, same padded batch.
TEXT_EMBEDS_MAX_ABS = 0.0
# Identical fp32 ops on identical inputs.
UNIPC_PORT_MAX_ABS = 0.0
# Run CFG the reference's way and the whole path must match exactly at every step.
SEQ_TRAJECTORY_MAX_ABS = 0.0
# Batched-CFG trajectory: batch-2 kernel noise compounds over the 50 steps (1.6e-3
# at step 0, 3.7 by step 49). ~1.2x that; a math bug passes 1e1 within a few steps.
PER_STEP_MAX_ABS = 4.5
FINAL_LATENTS_MAX_ABS = 4.5
# One step, batched vs sequential CFG (3.9e-1 on outputs whose abs max is 8.6): bf16
# kernel noise, amplified up to 9x by the guidance combine. ~1.5x observed.
CFG_BATCH_VS_SEQ_MAX_ABS = 0.6
# Decoded frames vs the oracle's mp4, which is lossy — hence PSNR, not equality.
# Observed ~23.8 dB mean; the floor sits ~1.8 dB under it.
DECODE_MIN_PSNR_DB = 22.0
# Tiled vs untiled decode, on pre-quantization floats. Error sits at tile seams (the
# temporal feature cache restarts per tile) and is structural, not dtype noise. Max
# ~1.5x observed; the mean bound catches a blend regression that stays under the max.
TILED_DECODE_MAX_ABS = 0.35
TILED_DECODE_MEAN_ABS = 3.0e-3
# Compiled DiT region vs the eager reference, over the full 50-step batched-CFG
# trajectory. Only the transformer region compiles (the UniPC solver's CPU-resident
# sigma stays eager); Inductor's fusion reorders fp accumulation, a step-level noise
# that compounds exactly like the batched-CFG kernel noise (1.6e-3 at step 0 -> 2.66
# by step 49 on the RTX 5090). Bound = observed max x ~1.3 headroom. The eager path
# stays the bit-exact reference (SEQ/UNIPC gates); this only prices compile.
COMPILED_VS_EAGER_MAX_ABS = 3.5
# No-CFG (guidance_scale <= 1.0) batch-1 path vs a no-CFG oracle. Same discipline
# as the sequential-CFG trajectory: one batch-1 forward per step, so BIT-EXACT.
NO_CFG_TRAJECTORY_MAX_ABS = 0.0

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
    pytest.mark.skipif(
        ORACLE_DIR is None or not (ORACLE_DIR / "metadata.json").exists(),
        reason=(
            "no diffusers oracle: set WAN22_ORACLE_DIR to a directory recorded by "
            "`python test/wan22/record_oracle.py --out-dir <dir>` (run it in an "
            "oracle venv on the SAME torch build as the server under test)"
            + (f"; no metadata.json at {ORACLE_DIR}" if ORACLE_DIR is not None else "")
        ),
    ),
]

DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_meta() -> dict:
    with open(ORACLE_DIR / "metadata.json") as f:
        return json.load(f)


def _step_metadata(meta: dict) -> dict:
    return {
        "is_prefill": False,
        "num_inference_steps": meta["num_inference_steps"],
        "guidance_scale": meta["guidance_scale"],
        "height": meta["height"],
        "width": meta["width"],
        "num_frames": meta["num_frames"],
    }


def _fwd_info(graph_walk: str, step_metadata: dict, seed: int) -> CurrentForwardPassInfo:
    return CurrentForwardPassInfo(
        request_id="equiv-r0",
        graph_walk=graph_walk,
        requires_cfg=False,
        fwd_index=0,
        random_seed=seed,
        max_tokens=0,
        sampling_config={},
        step_metadata=step_metadata,
    )


def _engine_inputs(fwd_info: CurrentForwardPassInfo) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(
        request_ids=[fwd_info.request_id], per_request_info={fwd_info.request_id: fwd_info}
    )


def _drive(
    submodule: NodeSubmodule,
    graph_walk: str,
    fwd_info: CurrentForwardPassInfo,
    inputs: dict[str, list[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    """One engine-shaped pass: prepare_inputs -> preprocess -> forward
    (StatelessEngine._execute_sequential's exact call sequence)."""
    node_inputs = submodule.prepare_inputs(graph_walk=graph_walk, fwd_info=fwd_info, inputs=inputs)
    engine_inputs = _engine_inputs(fwd_info)
    preprocessed = submodule.preprocess(
        graph_walk=graph_walk, engine_inputs=engine_inputs, inputs=[node_inputs]
    )
    return submodule.forward(graph_walk=graph_walk, engine_inputs=engine_inputs, **preprocessed)


@pytest.fixture(scope="module")
def text_embeds(oracle_meta) -> dict[str, torch.Tensor]:
    """Our text path vs the reference pipeline's, sharing one UMT5 instance.

    Loads the text-encoder-only pipeline, computes the reference embeddings
    via ``_get_t5_prompt_embeds`` (its own ftfy tokenization path), computes
    ours via ``Wan22Model.process_prompt`` + ``Wan22TextEncoderSubmodule``,
    then frees the encoder. Sharing the instance is deliberate: the weight
    bytes are identical ``from_pretrained`` results either way, and the test
    targets the tokenize/pad/mask/zero math, not the loader.
    """
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(
        MODEL_REPO, transformer=None, torch_dtype=torch.bfloat16
    )
    pipe.text_encoder.to(DEVICE)

    with torch.no_grad():
        ref_pos = pipe._get_t5_prompt_embeds(oracle_meta["prompt"], max_sequence_length=512, device=DEVICE)
        ref_neg = pipe._get_t5_prompt_embeds(
            oracle_meta["negative_prompt"], max_sequence_length=512, device=DEVICE
        )

    model = Wan22Model(model_path_hf=MODEL_REPO)
    tokens = model.process_prompt(
        prompt=oracle_meta["prompt"],
        input_modalities=["text"],
        output_modalities=["video"],
        negative_prompt=oracle_meta["negative_prompt"],
    )
    submodule = Wan22TextEncoderSubmodule(pipe.text_encoder, model.config)
    fwd_info = _fwd_info("encode_text", _step_metadata(oracle_meta), seed=oracle_meta["seed"])
    with torch.no_grad():
        outputs = _drive(submodule, "encode_text", fwd_info, {"text_inputs": tokens["text_inputs"]})

    result = {
        "ours_pos": outputs["text_embeds_pos"][0],
        "ours_neg": outputs["text_embeds_neg"][0],
        "ref_pos": ref_pos,
        "ref_neg": ref_neg,
    }
    del submodule, pipe
    torch.cuda.empty_cache()
    return result


# The bit-exact gates are defined against the EAGER transformer, so the shared
# dit_submodule is built with torch.compile OFF. Set WAN22_TEST_COMPILE_DIT=1 to
# rebuild it compiled and re-run the whole suite once — a "nothing downstream
# breaks" check: the priced/decode/structural tests still pass, while the bit-exact
# 0.0 gates (sequential-CFG, no-CFG) then FAIL by exactly the compile fusion delta,
# which is precisely why the default fixture keeps them on the eager path. The
# compiled path's own numeric bound is pinned by
# test_compiled_dit_matches_eager_per_step regardless of this switch.
_TEST_COMPILE_DIT = os.environ.get("WAN22_TEST_COMPILE_DIT", "0").lower() in ("1", "true", "yes", "on")


@pytest.fixture(scope="module")
def dit_submodule() -> Wan22DitSubmodule:
    model = Wan22Model(model_path_hf=MODEL_REPO)
    model.config.compile_dit = _TEST_COMPILE_DIT
    submodule = model.get_submodule("dit", device=str(DEVICE))
    return submodule.to(DEVICE)


class _RecordingTransformer(torch.nn.Module):
    """Wraps the DiT: always keeps the last output; optionally records the
    (hidden_states, timestep) of every call (I2V smoke)."""

    def __init__(self, inner: torch.nn.Module, record_inputs: bool = False):
        super().__init__()
        self.inner = inner
        self.record_inputs = record_inputs
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.last_output: torch.Tensor | None = None

    @property
    def dtype(self):
        return self.inner.dtype

    def forward(self, hidden_states, timestep, **kwargs):
        if self.record_inputs:
            self.calls.append((hidden_states.detach().clone(), timestep.detach().clone()))
        out = self.inner(hidden_states=hidden_states, timestep=timestep, **kwargs)
        self.last_output = out.detach()
        return out


@pytest.fixture(scope="module")
def t2v_trajectory(oracle_meta, text_embeds, dit_submodule) -> dict:
    """Run the full 50-step T2V denoise through the dit submodule, with the
    executable diffusers ``UniPCMultistepScheduler`` stepped in LOCKSTEP on
    the submodule's own recorded noise predictions.

    Iteration 0 sends the loop-back edges empty so ``prepare_inputs`` seeds
    the noise from ``random_seed=42`` — regenerating the oracle's initial
    noise bit-exactly. Records per step:
      * ``diffs``      — max-abs vs the oracle's post-step latents;
      * ``port_diffs`` — max-abs vs the reference scheduler fed the SAME noise_pred,
        which is kernel-noise-free and so isolates the inline UniPC port.
    """
    from diffusers import UniPCMultistepScheduler

    from mstar.model.wan22.components.unipc import make_unipc_tables

    meta = _step_metadata(oracle_meta)
    fwd_info = _fwd_info("video_gen", meta, seed=oracle_meta["seed"])
    persisted = {
        "text_embeds_pos": [text_embeds["ours_pos"]],
        "text_embeds_neg": [text_embeds["ours_neg"]],
    }
    num_steps = oracle_meta["num_inference_steps"]
    guidance = float(meta["guidance_scale"])

    ref_scheduler = UniPCMultistepScheduler.from_config(oracle_meta["scheduler_config"])
    ref_scheduler.set_timesteps(num_steps, device=DEVICE)
    ref_scheduler.set_begin_index(0)
    # The per-request tables must agree with the reference before any step.
    our_sigmas, our_timesteps = make_unipc_tables(num_steps, oracle_meta["scheduler_config"]["flow_shift"])
    assert torch.equal(ref_scheduler.sigmas, our_sigmas)
    assert torch.equal(ref_scheduler.timesteps.cpu(), our_timesteps)

    generator = torch.Generator(device="cpu").manual_seed(oracle_meta["seed"])
    ref_latents = torch.randn(
        (1, 48, 9, meta["height"] // 16, meta["width"] // 16),
        generator=generator, dtype=torch.float32,
    ).to(DEVICE)

    recorder = _RecordingTransformer(dit_submodule.transformer)
    original = dit_submodule.transformer
    dit_submodule.transformer = recorder
    diffs: list[float] = []
    port_diffs: list[float] = []
    latents_per_step: list[torch.Tensor] = []
    carried: dict[str, list[torch.Tensor]] = {}
    latents = None
    try:
        with torch.no_grad():
            for step in range(num_steps):
                outputs = _drive(dit_submodule, "video_gen", fwd_info, {**persisted, **carried})
                carried = {
                    name: outputs[name]
                    for name in ("latents", "time_index", "unipc_model_outputs", "unipc_last_sample")
                }
                latents = outputs["latents"][0]
                latents_per_step.append(latents.detach().cpu().float())
                ref = torch.load(ORACLE_DIR / "latents" / f"step_{step:03d}.pt", map_location="cpu")
                diffs.append((latents.cpu().float() - ref.float()).abs().max().item())

                # Lockstep replay on the submodule's own noise_pred.
                out = recorder.last_output
                noise_pred = out[1:2] + guidance * (out[0:1] - out[1:2])
                ref_latents = ref_scheduler.step(
                    noise_pred, ref_scheduler.timesteps[step], ref_latents, return_dict=False
                )[0]
                port_diffs.append((latents - ref_latents).abs().max().item())
    finally:
        dit_submodule.transformer = original
    return {
        "diffs": diffs,
        "port_diffs": port_diffs,
        "latents_per_step": latents_per_step,
        "final_latents": latents,
        "fwd_info": fwd_info,
    }


# ---------------------------------------------------------------------------
# 1. Text embeddings
# ---------------------------------------------------------------------------


def test_text_embeddings_match_reference(text_embeds):
    for name in ("pos", "neg"):
        ours = text_embeds[f"ours_{name}"]
        ref = text_embeds[f"ref_{name}"]
        assert ours.shape == ref.shape == (1, 512, 4096)
        assert ours.dtype == ref.dtype == torch.bfloat16
        diff = (ours.float() - ref.float()).abs().max().item()
        print(f"text_embeds_{name}: max_abs_diff={diff:.3e}")
        assert diff <= TEXT_EMBEDS_MAX_ABS, f"{name} embeddings diverge: {diff:.3e}"


# ---------------------------------------------------------------------------
# 2. Per-step T2V trajectory
# ---------------------------------------------------------------------------


def test_per_step_trajectory_matches_oracle(t2v_trajectory, oracle_meta):
    diffs = t2v_trajectory["diffs"]
    print("\nper-step max-abs diff vs oracle (step: diff):")
    for step, diff in enumerate(diffs):
        marker = " <- first nonzero" if diff > 0 and all(d == 0 for d in diffs[:step]) else ""
        print(f"  step {step:2d}  t={oracle_meta['per_step'][step]['timestep']:6.1f}  "
              f"max_abs={diff:.6e}{marker}")

    first_divergent = next((i for i, d in enumerate(diffs) if d > 0), None)
    growth = [diffs[i] for i in range(0, len(diffs), 10)]
    print(f"first divergent step: {first_divergent}")
    print(f"growth curve (every 10th step): {[f'{d:.3e}' for d in growth]}")
    print(f"max over trajectory: {max(diffs):.6e} at step {diffs.index(max(diffs))}")

    assert max(diffs) <= PER_STEP_MAX_ABS, (
        f"trajectory diverged: max per-step diff {max(diffs):.3e} at step "
        f"{diffs.index(max(diffs))}, first divergent step {first_divergent}"
    )


# ---------------------------------------------------------------------------
# 2a. Inline UniPC port vs the executable reference scheduler
# ---------------------------------------------------------------------------


def test_unipc_port_matches_reference_scheduler_bitwise(t2v_trajectory):
    """Kernel-noise-free port check: at every one of the 50 steps, the
    submodule's carried latents must be BIT-EXACTLY what the diffusers
    ``UniPCMultistepScheduler`` produces when fed the identical noise_pred
    (recorded from the submodule's own transformer call). Covers the order
    warmup ramp (k=0,1), the k=0 corrector skip, the bh2 corrector solve,
    the sigma-table endpoints (sigmas[0] -= 1e-6, terminal zero sigma), and
    the ``lower_order_final`` order-1 last step — against the reference
    *implementation*, not a re-derivation.
    """
    port_diffs = t2v_trajectory["port_diffs"]
    nonzero = [(i, d) for i, d in enumerate(port_diffs) if d > 0]
    print(f"UniPC port lockstep: max_abs={max(port_diffs):.3e}, "
          f"nonzero steps={nonzero[:5]}{'...' if len(nonzero) > 5 else ''}")
    assert max(port_diffs) <= UNIPC_PORT_MAX_ABS, (
        f"UniPC port diverges from the reference scheduler; first divergent "
        f"step {nonzero[0][0]} (max_abs {nonzero[0][1]:.3e})"
    )


# ---------------------------------------------------------------------------
# 2b. Sequential-CFG trajectory — the implementation-exactness check
# ---------------------------------------------------------------------------


def _sequential_noise_prediction(self, model_input, temp_ts, text_embeds_pos, text_embeds_neg, guidance_scale):
    """Reference CFG semantics: two batch-1 forwards + the same combine."""
    timestep = temp_ts.unsqueeze(0).expand(1, -1)
    cond = self.transformer(
        hidden_states=model_input, timestep=timestep,
        encoder_hidden_states=text_embeds_pos.to(self.transformer.dtype),
    )
    uncond = self.transformer(
        hidden_states=model_input, timestep=timestep,
        encoder_hidden_states=text_embeds_neg.to(self.transformer.dtype),
    )
    return uncond + guidance_scale * (cond - uncond)


def test_sequential_cfg_trajectory_is_bitexact_vs_oracle(oracle_meta, text_embeds, dit_submodule):
    """Drive the engine path with the reference's sequential batch-1 CFG:
    every one of the 50 per-step latents must be BIT-EXACT vs the oracle.
    This proves the whole mstar implementation (tokenize -> encode ->
    per-token timesteps -> transformer -> inline UniPC) identical to the
    stock pipeline, leaving the batch-2 CFG kernel shape as the suite's ONLY
    priced deviation (tests 2/3/4)."""
    meta = _step_metadata(oracle_meta)
    fwd_info = _fwd_info("video_gen", meta, seed=oracle_meta["seed"])
    persisted = {
        "text_embeds_pos": [text_embeds["ours_pos"]],
        "text_embeds_neg": [text_embeds["ours_neg"]],
    }
    num_steps = oracle_meta["num_inference_steps"]

    original = Wan22DitSubmodule._noise_prediction
    Wan22DitSubmodule._noise_prediction = _sequential_noise_prediction
    diffs: list[float] = []
    try:
        carried: dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for step in range(num_steps):
                outputs = _drive(dit_submodule, "video_gen", fwd_info, {**persisted, **carried})
                carried = {
                    name: outputs[name]
                    for name in ("latents", "time_index", "unipc_model_outputs", "unipc_last_sample")
                }
                ref = torch.load(ORACLE_DIR / "latents" / f"step_{step:03d}.pt", map_location="cpu")
                diffs.append((outputs["latents"][0].cpu().float() - ref.float()).abs().max().item())
    finally:
        Wan22DitSubmodule._noise_prediction = original

    nonzero = [(i, d) for i, d in enumerate(diffs) if d > 0]
    print(f"sequential-CFG trajectory vs oracle: max_abs={max(diffs):.3e}, "
          f"first nonzero step={nonzero[0] if nonzero else None}")
    assert max(diffs) <= SEQ_TRAJECTORY_MAX_ABS, (
        f"sequential-CFG trajectory diverged from the oracle: first nonzero "
        f"step {nonzero[0][0]} (max_abs {nonzero[0][1]:.3e}) — implementation "
        "regression, not kernel noise"
    )


# ---------------------------------------------------------------------------
# 3. CFG batched vs sequential
# ---------------------------------------------------------------------------


def test_cfg_batched_matches_sequential(oracle_meta, text_embeds, dit_submodule):
    """Price the one deviation from the reference: our single batch-2 CFG
    forward vs the pipeline's two sequential batch-1 forwards, on the exact
    step-0 inputs (regenerated oracle noise, timestep 999)."""
    meta = _step_metadata(oracle_meta)
    guidance = float(meta["guidance_scale"])
    generator = torch.Generator(device="cpu").manual_seed(oracle_meta["seed"])
    latents = torch.randn(
        (1, 48, 9, meta["height"] // 16, meta["width"] // 16),
        generator=generator, dtype=torch.float32,
    ).to(DEVICE)
    transformer = dit_submodule.transformer
    model_input = latents.to(transformer.dtype)
    t = torch.tensor(999, dtype=torch.int64, device=DEVICE)
    mask = torch.ones(1, 1, *latents.shape[2:], dtype=torch.float32, device=DEVICE)
    temp_ts = (mask[0, 0][:, ::2, ::2] * t).flatten()
    pos, neg = text_embeds["ours_pos"], text_embeds["ours_neg"]

    with torch.no_grad():
        batched_raw = transformer(
            hidden_states=model_input.repeat(2, 1, 1, 1, 1),
            timestep=temp_ts.unsqueeze(0).expand(2, -1),
            encoder_hidden_states=torch.cat([pos, neg], dim=0).to(transformer.dtype),
        )
        batched = dit_submodule._noise_prediction(model_input, temp_ts, pos, neg, guidance)
        timestep = temp_ts.unsqueeze(0).expand(1, -1)
        cond = transformer(
            hidden_states=model_input, timestep=timestep,
            encoder_hidden_states=pos.to(transformer.dtype),
        )
        uncond = transformer(
            hidden_states=model_input, timestep=timestep,
            encoder_hidden_states=neg.to(transformer.dtype),
        )
        sequential = uncond + guidance * (cond - uncond)

    for name, b, s in (("cond", batched_raw[0:1], cond), ("uncond", batched_raw[1:2], uncond)):
        d = (b.float() - s.float()).abs().max().item()
        print(f"CFG per-forward {name}: max_abs_delta={d:.3e} (abs max {s.float().abs().max().item():.3e})")
    delta = (batched.float() - sequential.float()).abs().max().item()
    scale = sequential.float().abs().max().item()
    print(f"CFG batched-vs-sequential: max_abs_delta={delta:.3e} (output abs max {scale:.3e})")
    # Batching semantics are proven exact by the sequential-CFG test, so this is
    # pure kernel noise; the bound fails loudly if batching breaks outright.
    assert delta <= CFG_BATCH_VS_SEQ_MAX_ABS, f"batch-2 CFG diverges from sequential by {delta:.3e}"


# ---------------------------------------------------------------------------
# 5. I2V first-frame injection smoke
# ---------------------------------------------------------------------------


def test_i2v_first_frame_injection_semantics(oracle_meta, text_embeds, dit_submodule):
    """8-step small-resolution I2V run asserting the expand_timesteps
    conditioning contract:
      * the transformer INPUT's frame 0 equals the condition at every step, with
        frame-0 per-token timesteps zeroed;
      * the loop-carried latents stay un-injected mid-loop;
      * the OUTPUT latents get the condition only at the final iteration.
    """
    num_steps, height, width, num_frames = 8, 224, 384, 9
    h_lat, w_lat, t_lat = height // 16, width // 16, (num_frames - 1) // 4 + 1
    meta = {
        "is_prefill": False, "num_inference_steps": num_steps, "guidance_scale": 5.0,
        "height": height, "width": width, "num_frames": num_frames,
    }
    fwd_info = _fwd_info("video_gen_i2v", meta, seed=7)
    condition = torch.randn(1, 48, 1, h_lat, w_lat, dtype=torch.float32, device=DEVICE)
    # Truncated embeddings keep the smoke fast; content is irrelevant here.
    pos = text_embeds["ours_pos"][:, :64].contiguous()
    neg = text_embeds["ours_neg"][:, :64].contiguous()

    recorder = _RecordingTransformer(dit_submodule.transformer, record_inputs=True)
    original = dit_submodule.transformer
    dit_submodule.transformer = recorder
    try:
        persisted = {
            "text_embeds_pos": [pos], "text_embeds_neg": [neg], "image_latent": [condition],
        }
        carried: dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for step in range(num_steps):
                outputs = _drive(dit_submodule, "video_gen_i2v", fwd_info, {**persisted, **carried})
                carried = {
                    name: outputs[name]
                    for name in ("latents", "time_index", "unipc_model_outputs", "unipc_last_sample")
                }
                out_latents = outputs["latents"][0]
                frame0_injected = torch.equal(out_latents[:, :, 0], condition[:, :, 0])
                if step + 1 < num_steps:
                    # Mid-loop the carried latents stay un-injected.
                    assert not frame0_injected, f"step {step}: mid-loop latents were injected"
                else:
                    assert frame0_injected, "final-iteration output latents missing the injection"
    finally:
        dit_submodule.transformer = original

    assert len(recorder.calls) == num_steps
    cond_bf16 = condition.to(recorder.dtype)
    for step, (hidden_states, timestep) in enumerate(recorder.calls):
        # CFG batch-2: frame 0 of BOTH batch elements is the condition.
        assert hidden_states.shape[0] == 2
        for b in range(2):
            assert torch.equal(hidden_states[b : b + 1, :, 0], cond_bf16[:, :, 0]), (
                f"step {step} batch {b}: transformer input frame 0 != condition"
            )
        # Per-token timesteps: frame-0 tokens 0, all later frames uniform t.
        grid = timestep[0].reshape(t_lat, (h_lat + 1) // 2, (w_lat + 1) // 2)
        assert torch.all(grid[0] == 0), f"step {step}: frame-0 timesteps not zeroed"
        assert torch.all(grid[1:] == grid[1, 0, 0]), f"step {step}: non-uniform later timesteps"
        assert grid[1, 0, 0].item() > 0


# ---------------------------------------------------------------------------
# 4. Final latents + decode PSNR. Evicts the DiT so the decode's workspace fits.
# ---------------------------------------------------------------------------


def _decode_oracle_mp4() -> torch.Tensor:
    av = pytest.importorskip("av")
    container = av.open(str(ORACLE_DIR / "wan22_ti2v5b_t2v_seed42.mp4"))
    frames = [
        torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)
    ]
    return torch.stack(frames)  # [F, H, W, 3] uint8


def test_final_latents_and_decode_psnr(t2v_trajectory, oracle_meta, dit_submodule):
    final_latents = t2v_trajectory["final_latents"].clone()
    final_diff = (
        final_latents.cpu().float()
        - torch.load(ORACLE_DIR / "final_latents.pt", map_location="cpu").float()
    ).abs().max().item()
    print(f"final latents: max_abs_diff={final_diff:.6e}")
    assert final_diff <= FINAL_LATENTS_MAX_ABS

    dit_submodule.to("cpu")  # make room for the decode workspace
    torch.cuda.empty_cache()
    model = Wan22Model(model_path_hf=MODEL_REPO)
    decoder = model.get_submodule("vae_decoder").to(DEVICE)
    with torch.no_grad():
        outputs = _drive(
            decoder, "video_gen", t2v_trajectory["fwd_info"], {"latents": [final_latents]}
        )
    video = outputs["video_output"][0]
    assert video.shape == (1, 3, oracle_meta["num_frames"], oracle_meta["height"], oracle_meta["width"])
    # Both are uint8 on the same 255 scale, so the PSNR is apples to apples.
    assert video.dtype == torch.uint8
    ours = video[0].permute(1, 2, 3, 0).cpu()

    oracle_frames = _decode_oracle_mp4()
    assert oracle_frames.shape == ours.shape
    mse = (ours.float() - oracle_frames.float()).pow(2).mean().item()
    psnr = 10 * torch.log10(torch.tensor(255.0**2 / mse)).item()
    per_frame_mse = (ours.float() - oracle_frames.float()).pow(2).mean(dim=(1, 2, 3))
    worst = 10 * torch.log10(255.0**2 / per_frame_mse).min().item()
    print(f"decode PSNR vs oracle mp4: mean={psnr:.2f} dB, worst frame={worst:.2f} dB")
    assert psnr >= DECODE_MIN_PSNR_DB, f"decoded video PSNR {psnr:.2f} dB below threshold"


# ---------------------------------------------------------------------------
# 6. Tiled vs untiled VAE decode — prices the decoder submodule's tiling.
# ---------------------------------------------------------------------------


def test_tiled_decode_matches_untiled(oracle_meta):
    """The decoder submodule decodes through ``tiled_decode`` (256px tiles,
    64px blended overlap) to bound the decode workspace. Tiles restart the
    VAE's temporal feature cache at each tile boundary, so seam-adjacent pixels
    legitimately differ from the untiled path. This prices that.

    The bounds compare PRE-quantization floats, because the mean bound is sub-1/255
    and would be invisible through the submodule's uint8 output. The emission is
    separately pinned to the boundary-quantized tiled decode, which proves the
    quantization happens at the worker boundary and nowhere upstream."""
    latent = torch.load(ORACLE_DIR / "final_latents.pt", map_location="cpu").to(DEVICE)

    model = Wan22Model(model_path_hf=MODEL_REPO)
    decoder = model.get_submodule("vae_decoder").to(DEVICE)
    meta = _step_metadata(oracle_meta)
    fwd_info = _fwd_info("video_gen", meta, seed=oracle_meta["seed"])
    with torch.no_grad():
        # Drive the submodule down each forced tiling path (Feature 1's override),
        # so the emission for each is pinned to that exact decode. The gate would
        # otherwise pick untiled here (DiT not resident -> lots of free VRAM).
        decoder.config.vae_decode_tiling = "tiled"
        emitted_tiled = _drive(decoder, "video_gen", fwd_info, {"latents": [latent]})["video_output"][0]
        decoder.config.vae_decode_tiling = "untiled"
        emitted_untiled = _drive(decoder, "video_gen", fwd_info, {"latents": [latent]})["video_output"][0]

        # Reference decodes both ways, in floats, matching the submodule's dtype.
        vae = decoder.vae
        z = latent.to(device=DEVICE, dtype=decoder._decode_dtype())
        z_dim = model.config.vae_z_dim
        mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(z.device, z.dtype)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(z.device, z.dtype)
        tiled_raw = vae.tiled_decode(z / std + mean, return_dict=False)[0]
        tiled = (tiled_raw / 2 + 0.5).clamp(0, 1).float()
        untiled_raw = vae.decode(z / std + mean, return_dict=False)[0]
        untiled = (untiled_raw / 2 + 0.5).clamp(0, 1).float()

    # Each forced path's emission is exactly that decode, boundary-quantized —
    # proving the gate routes to tiled_decode / decode and quantizes only at the
    # worker boundary, nowhere upstream.
    assert emitted_tiled.dtype == emitted_untiled.dtype == torch.uint8
    assert torch.equal(emitted_tiled, (tiled_raw / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8))
    assert torch.equal(emitted_untiled, (untiled_raw / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8))

    diff = (tiled - untiled).abs()
    print(f"tiled-vs-untiled decode: max_abs={diff.max().item():.3e} "
          f"mean_abs={diff.mean().item():.3e} (values in [0, 1]); "
          f"pixels > 1/255: {(diff > 1 / 255).float().mean().item() * 100:.3f}%")
    assert tiled.shape == untiled.shape
    assert diff.max().item() <= TILED_DECODE_MAX_ABS, (
        f"tiled decode diverges from untiled by {diff.max().item():.3e}"
    )
    assert diff.mean().item() <= TILED_DECODE_MEAN_ABS, (
        f"tiled decode mean error {diff.mean().item():.3e} exceeds bound — "
        "seam blending regressed beyond localized overlap effects"
    )


# ---------------------------------------------------------------------------
# 7. Compiled DiT region vs eager — Feature 2. Proves compile ENGAGED (no silent
#    eager fallback) and bounds the compiled trajectory against eager.
# ---------------------------------------------------------------------------


def test_compiled_dit_matches_eager_per_step(oracle_meta, text_embeds, dit_submodule, t2v_trajectory):
    """torch.compile the inner DiT region only, then run the full batched-CFG
    trajectory compiled and compare per-step against the eager reference
    (``t2v_trajectory``'s recorded latents). Two proofs:

      * compile ENGAGED — a counting Inductor backend must be invoked at least
        once, so a wrong region boundary that silently fell back to eager fails
        here (it would otherwise "pass" at 0.0);
      * the compiled trajectory stays within ``COMPILED_VS_EAGER_MAX_ABS`` of
        eager, a bound derived from the observed per-step max.

    The eager path stays the bit-exact reference (the SEQ/UNIPC gates); this test
    prices compile and nothing else.
    """
    import torch._dynamo
    from torch._inductor.compile_fx import compile_fx

    if _TEST_COMPILE_DIT:
        pytest.skip("dit_submodule is already compiled (WAN22_TEST_COMPILE_DIT); this test compiles it itself")

    # An earlier test (final-latents decode) evicts the shared DiT to CPU to make
    # room for the decode workspace; re-home it before compiling.
    dit_submodule.to(DEVICE)
    eager_latents = t2v_trajectory["latents_per_step"]
    num_steps = oracle_meta["num_inference_steps"]
    assert len(eager_latents) == num_steps
    meta = _step_metadata(oracle_meta)
    fwd_info = _fwd_info("video_gen", meta, seed=oracle_meta["seed"])
    persisted = {
        "text_embeds_pos": [text_embeds["ours_pos"]],
        "text_embeds_neg": [text_embeds["ours_neg"]],
    }

    compile_calls = {"n": 0}

    def _counting_inductor(gm, example_inputs):
        compile_calls["n"] += 1
        return compile_fx(gm, example_inputs)

    transformer = dit_submodule.transformer
    had_instance_forward = "forward" in transformer.__dict__
    original_forward = transformer.forward
    torch._dynamo.reset()
    dit_submodule._compiled_shapes = set()
    dit_submodule._compile_dit = True  # let the one-time compile-pause log fire
    diffs: list[float] = []
    try:
        transformer.forward = torch.compile(
            original_forward, backend=_counting_inductor, fullgraph=False, dynamic=False,
        )
        carried: dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for step in range(num_steps):
                outputs = _drive(dit_submodule, "video_gen", fwd_info, {**persisted, **carried})
                carried = {
                    name: outputs[name]
                    for name in ("latents", "time_index", "unipc_model_outputs", "unipc_last_sample")
                }
                diffs.append(
                    (outputs["latents"][0].cpu().float() - eager_latents[step]).abs().max().item()
                )
    finally:
        if had_instance_forward:
            transformer.forward = original_forward
        else:
            del transformer.forward  # restore class-method lookup (eager)
        dit_submodule._compile_dit = _TEST_COMPILE_DIT
        torch._dynamo.reset()

    print(f"\ncompile engaged: Inductor backend invoked {compile_calls['n']}x")
    print(f"compiled-vs-eager per-step max_abs={max(diffs):.3e} at step {diffs.index(max(diffs))}")
    print(f"growth (every 10th step): {[f'{diffs[i]:.3e}' for i in range(0, len(diffs), 10)]}")
    # A silent eager fallback (wrong region boundary) never reaches Inductor.
    assert compile_calls["n"] >= 1, "torch.compile did not engage — Inductor was never invoked"
    assert max(diffs) <= COMPILED_VS_EAGER_MAX_ABS, (
        f"compiled trajectory diverged from eager by {max(diffs):.3e} at step "
        f"{diffs.index(max(diffs))} — beyond the priced compile fusion noise"
    )


# ---------------------------------------------------------------------------
# 8. No-CFG batch-1 path — Feature 4. guidance_scale <= 1.0 takes the batch-1
#    branch no other test exercises; it is bit-exact vs a no-CFG oracle.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    NOCFG_ORACLE_DIR is None or not (NOCFG_ORACLE_DIR / "metadata.json").exists(),
    reason=(
        "no no-CFG oracle: set WAN22_NOCFG_ORACLE_DIR to a directory recorded by "
        "`python test/wan22/record_oracle.py --guidance 1.0 --num-inference-steps 8 "
        "--height 224 --width 384 --num-frames 9 --out-dir <dir>` (oracle venv, same torch)"
    ),
)
def test_no_cfg_batch1_trajectory_is_bitexact(oracle_meta, text_embeds, dit_submodule):
    """guidance_scale <= 1.0 runs one batch-1 forward per step (no CFG), the branch
    no other test covered. Drive the full no-CFG oracle trajectory at guidance 1.0
    and require BIT-EXACT latents at every step — the same discipline as the
    sequential-CFG gate, since a no-CFG step is one batch-1 forward exactly like the
    reference at guidance 1.0. A recorder PROVES the batch-1 branch is taken (the
    transformer sees batch 1, not the CFG batch-2), so the test cannot pass while
    silently exercising the wrong path.
    """
    with open(NOCFG_ORACLE_DIR / "metadata.json") as f:
        nocfg_meta = json.load(f)
    assert nocfg_meta["guidance_scale"] <= 1.0, (
        f"no-CFG oracle must be recorded with --guidance 1.0; got {nocfg_meta['guidance_scale']}"
    )
    assert nocfg_meta["prompt"] == oracle_meta["prompt"], (
        "no-CFG oracle must use the same prompt as the main oracle so the shared "
        "positive embeds apply (record_oracle.py uses one PROMPT constant)"
    )
    meta = _step_metadata(nocfg_meta)
    assert meta["guidance_scale"] <= 1.0  # the request is genuinely no-CFG
    num_steps = nocfg_meta["num_inference_steps"]
    # An earlier test (final-latents decode) may have evicted the shared DiT to
    # CPU; re-home it so the batch-1 forward runs entirely on device.
    dit_submodule.to(DEVICE)
    fwd_info = _fwd_info("video_gen", meta, seed=nocfg_meta["seed"])
    persisted = {
        "text_embeds_pos": [text_embeds["ours_pos"]],
        "text_embeds_neg": [text_embeds["ours_neg"]],  # unused on the no-CFG path
    }

    recorder = _RecordingTransformer(dit_submodule.transformer, record_inputs=True)
    original = dit_submodule.transformer
    dit_submodule.transformer = recorder
    diffs: list[float] = []
    try:
        carried: dict[str, list[torch.Tensor]] = {}
        with torch.no_grad():
            for step in range(num_steps):
                outputs = _drive(dit_submodule, "video_gen", fwd_info, {**persisted, **carried})
                carried = {
                    name: outputs[name]
                    for name in ("latents", "time_index", "unipc_model_outputs", "unipc_last_sample")
                }
                ref = torch.load(NOCFG_ORACLE_DIR / "latents" / f"step_{step:03d}.pt", map_location="cpu")
                diffs.append((outputs["latents"][0].cpu().float() - ref.float()).abs().max().item())
    finally:
        dit_submodule.transformer = original

    # PROVE the batch-1 branch ran at every step (guidance<=1.0 => no CFG batch-2).
    assert len(recorder.calls) == num_steps
    for step, (hidden_states, _ts) in enumerate(recorder.calls):
        assert hidden_states.shape[0] == 1, (
            f"step {step}: transformer batch {hidden_states.shape[0]} != 1 — the CFG "
            "batch-2 path was taken, so this test would not be exercising no-CFG"
        )

    nonzero = [(i, d) for i, d in enumerate(diffs) if d > 0]
    print(f"no-CFG batch-1 trajectory vs oracle: max_abs={max(diffs):.3e}, "
          f"first nonzero step={nonzero[0] if nonzero else None}")
    assert max(diffs) <= NO_CFG_TRAJECTORY_MAX_ABS, (
        f"no-CFG trajectory diverged from the oracle: first nonzero step "
        f"{nonzero[0][0]} (max_abs {nonzero[0][1]:.3e}) — the batch-1 path is not "
        "bit-exact against the reference"
    )
