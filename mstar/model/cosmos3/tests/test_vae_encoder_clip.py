"""The vae_encoder node's conditioning clip per request kind.

Policy and forward-dynamics requests condition on latent frame 0 only, and the
Wan VAE is temporally causal (frame 0's latent does not depend on later frames,
measured bit-identical on Edge at 480p), so the node encodes the observation as a
one-frame clip and places it in the full latent shape the denoise loop pins —
not the 33-frame repeat the reference pipelines encode (~10x the encode time).
Inverse dynamics still encodes the whole observed clip.
"""

from __future__ import annotations

import torch

from mstar.model.cosmos3.submodules import Cosmos3VAEEncoderSubmodule
from mstar.model.cosmos3.tests.test_edge import _tiny_edge_config
from mstar.model.submodule_base import CurrentForwardPassInfo, ModelInputsFromEngine


class _Dist:
    def __init__(self, mu):
        self._mu = mu

    def mode(self):
        return self._mu


class _StubVAE(torch.nn.Module):
    """Encodes [1, 3, T, H, W] pixels to [1, C, 1 + (T-1)//4, H/16, W/16]: frame t's latent is a function of frame t."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = type("cfg", (), {"latents_mean": [0.0] * channels, "latents_std": [1.0] * channels})()
        self.channels = channels

    def encode(self, x):
        b, c, t, h, w = x.shape
        lt = 1 if t == 1 else 1 + (t - 1) // 4
        pooled = torch.nn.functional.avg_pool3d(x, kernel_size=(1, 16, 16))  # [1, 3, t, h/16, w/16]
        frames = [pooled[:, :, 0]] + [pooled[:, :, 1 + 4 * i] for i in range(lt - 1)]
        z = torch.stack(frames, dim=2).repeat(1, self.channels // 3 + 1, 1, 1, 1)[:, : self.channels]
        return type("out", (), {"latent_dist": _Dist(z)})()


def _node(cfg):
    return Cosmos3VAEEncoderSubmodule(vae=_StubVAE(cfg.latent_channel), config=cfg)


def _run(enc, md, inputs):
    fwd = CurrentForwardPassInfo(
        request_id="r", graph_walk="prefill_cond", fwd_index=0, random_seed=0, max_tokens=0, step_metadata=md,
    )
    ei = ModelInputsFromEngine(request_ids=["r"], per_request_info={"r": fwd})
    ni = enc.prepare_inputs("prefill_cond", fwd, inputs)
    out = enc.forward("prefill_cond", ei, **enc.preprocess("prefill_cond", ei, [ni]))
    return ni, out["cond_latents"][0]


def test_policy_and_forward_dynamics_encode_one_frame():
    cfg = _tiny_edge_config()
    enc = _node(cfg)
    image = torch.rand(3, 64, 96)
    for mode in ("policy", "forward_dynamics"):
        md = {"height": 32, "width": 48, "num_frames": 33, "action_mode": mode, "action_chunk_size": 32}
        ni, lat = _run(enc, md, {"image_inputs": [image]})
        assert ni.tensor_inputs["vision"].shape[2] == 1, mode
        assert ni.kwargs["condition_indexes"] == (0,)
        assert tuple(lat.shape) == (1, cfg.latent_channel, 9, 2, 3), (mode, tuple(lat.shape))
        # Frame 0 is the encoded observation; the frames the vmask never reads are zero padding.
        assert lat[:, :, 0].abs().sum() > 0
        assert lat[:, :, 1:].abs().sum() == 0


def test_one_frame_encode_matches_frame0_of_the_repeated_clip():
    cfg = _tiny_edge_config()
    enc = _node(cfg)
    image = torch.rand(3, 64, 96)
    md = {"height": 32, "width": 48, "num_frames": 33, "action_mode": "policy", "action_chunk_size": 32}
    _, lat = _run(enc, md, {"image_inputs": [image]})
    # What the reference pipelines encode: the frame repeated over the clip.
    frame = enc._video_processor.preprocess(image, height=32, width=48).unsqueeze(2).float()
    repeated = enc.vae.encode(frame.expand(-1, -1, 33, -1, -1)).latent_dist.mode().to(lat.dtype)
    torch.testing.assert_close(lat[:, :, 0], repeated[:, :, 0])


def test_inverse_dynamics_and_i2v_keep_their_clips():
    cfg = _tiny_edge_config()
    enc = _node(cfg)
    video = torch.rand(9, 3, 64, 96)
    md = {"height": 32, "width": 48, "num_frames": 9, "action_mode": "inverse_dynamics", "action_chunk_size": 8}
    ni, lat = _run(enc, md, {"video_inputs": [video]})
    assert ni.tensor_inputs["vision"].shape[2] == 9
    assert tuple(lat.shape) == (1, cfg.latent_channel, 3, 2, 3)
    # Image-to-video: the single anchor frame, no padding kwargs (the DiT reads it as ``cond_latents``).
    ni, lat = _run(enc, {"height": 32, "width": 48, "num_frames": 33}, {"image_inputs": [torch.rand(3, 64, 96)]})
    assert ni.tensor_inputs["vision"].shape[2] == 1 and "condition_indexes" not in ni.kwargs
    assert tuple(lat.shape) == (1, cfg.latent_channel, 1, 2, 3)
