"""Cosmos3's served kv rollout reproduces its block-causal reference, window by window.

The reference (``Cosmos3Pipeline.windowed_kv``) denoises each window over explicit per-layer context and commits
and releases the finished window's K/V by hand. The served path runs the DiT's declared steps through the harness
the GPU parity tests drive. Both run in seconds on a tiny config with seeded weights, so the harness can't drift
from the engine unseen: it once opened only the conditional label, and never committed ``PROGRESS``, so every
window would have drawn the first window's noise again.
"""

import torch
from diffusers import UniPCMultistepScheduler

from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
from mstar.model.cosmos3.config import Cosmos3Config
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule
from mstar.model.cosmos3.tests import test_engine_cache as harness
from mstar.model.cosmos3.tests.pipeline import Cosmos3Pipeline
from mstar.model.cosmos3.tests.test_edge import _init_all


def test_the_served_kv_rollout_matches_its_reference_window_by_window():
    config = Cosmos3Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=100, rope_axes_dim=(4, 2, 2), latent_channel=8, latent_patch_size=2,
        patch_latent_dim=32, sound_gen=False, action_gen=False, compile_denoise=False, enable_windowed_video=True,
    )
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, enable_windowed_video=True)
    model.config = config
    transformer = Cosmos3OmniTransformer(config).eval()
    _init_all(transformer)
    # a flow-matching schedule, as every Cosmos3 checkpoint's
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True)
    dit = Cosmos3DiTSubmodule(transformer=transformer, config=config, scheduler=scheduler)
    pipe = Cosmos3Pipeline.__new__(Cosmos3Pipeline)
    pipe.transformer, pipe.scheduler, pipe.config = transformer, scheduler, config
    pipe.device, pipe.dtype, pipe.vae_scale_spatial, pipe.vae_scale_temporal = "cpu", torch.float32, 16, 4
    md = model._resolve_gen_params(
        {
            "window_mode": "kv", "num_frames": harness.W_NUM_FRAMES, "size": "256x256",
            "window_frames": harness.W_WINDOW_FRAMES, "context_frames": harness.W_CONTEXT_FRAMES,
            "num_inference_steps": 2, "guidance_scale": 6.0,
        },
        [], ["video"],
    )
    cond, uncond = list(range(3, 40)), [1, 2, 3, 4, 5]
    with torch.no_grad():
        reference = pipe.windowed_kv(
            cond, uncond, total_units=md["total_latent_units"], window_units=md["window_latent_units"],
            context_units=md["context_latent_units"], height=256, width=256, num_inference_steps=2,
            guidance_scale=6.0, fps=md["fps"], flow_shift=md.get("flow_shift"),
            generator=torch.Generator().manual_seed(harness.SEED),
        )
    resources = harness._SdpaResources()
    resources.kv.ingest_request("r0", harness._windowed_config(model, md, cond, uncond))
    served = harness._run_windowed_kv_served(model, dit, resources.as_dict(), md, cond, uncond, "cpu")
    assert resources.kv.released == {"main": 384, "uncond": 384}
    assert len(served) == len(reference) == 3
    for window, (ours, theirs) in enumerate(zip(served, reference, strict=True)):
        assert torch.equal(ours, theirs.reshape(ours.shape)), f"window {window} left the reference"
