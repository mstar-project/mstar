#!/usr/bin/env python3
"""Record NVIDIA's GR00T N1.7 on DROID requests: the inputs and outputs GR00T's tests compare to.

Runs NVIDIA's own ``Gr00tPolicy`` (Isaac-GR00T), never any mstar code, and writes one
``req_XX.pt`` per request to ``--out``:

    collated           the processor's model inputs (input_ids, pixel_values, image_grid_thw,
                       state, embodiment_id, ...), as ``model.get_action`` receives them
    noise              the action head's initial noise, as drawn ([1, 40, 132] bf16)
    vlln_out, vl_embeds, state_features
                       the VL bridge's LayerNorm and its output, and the state token
    action_pred        the normalised action chunk ([1, 40, 132])
    decoded_action     NVIDIA's decode, by joint group
    raw_obs            the observation as a client sends it (uint8 frames, state, instruction)
    seed, embodiment_tag, video_keys, state_keys

``record_oracle.py`` then runs NVIDIA's model on these inputs in mstar's environment, per seed.

Requires Isaac-GR00T's own environment (``uv sync`` in its checkout: torch 2.9, transformers
4.57, diffusers 0.35, flash-attn), which conflicts with mstar's:

    cd ~/ref/Isaac-GR00T && .venv/bin/python <mstar>/test/gr00t_n17/record_dumps.py \\
        --out ~/gr00t-runs/dump

The default sampling is the 20 requests the tests expect: ``demo_data/droid_sample``'s
episodes in order, every 15th step from the first with a t-15 frame; the first 3
observations only warm the model, so request 0 is episode 0, step 60. The ~200-request set
is ``--stride 4 --warmup 0 --n 200``. A machine whose FFmpeg cannot decode the dataset's AV1
videos reads observations decoded elsewhere: ``--save-obs FILE`` on one machine, ``--obs-file
FILE`` on the other.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


def observations(dataset_path: str, modality, count: int, stride: int):
    """DROID observations parsed as ``Gr00tPolicy`` takes them: episodes in order, every
    ``stride`` steps from the first that has its history frame."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.utils import parse_observation_gr00t

    tag = EmbodimentTag[EMBODIMENT_TAG]
    obs_modality = {k: v for k, v in modality.items() if k != "action"}
    loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=modality)
    out = []
    for ep in range(len(loader)):
        traj = loader[ep]
        first = max(-min(obs_modality["video"].delta_indices), 0)
        for step in range(first, len(traj), stride):
            dp = extract_step_data(traj, step, obs_modality, tag)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            obs.update({f"video.{k}": np.array(v) for k, v in dp.images.items()})
            for lang_key in loader.modality_configs["language"].modality_keys:
                obs[lang_key] = dp.text
            out.append(parse_observation_gr00t(obs, loader.modality_configs))
            if len(out) == count:
                return out
    return out


class _Help(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """The docstring as written, and each option's default."""


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_cpu(v) for v in x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=_Help)
    ap.add_argument("--model-path", default="nvidia/GR00T-N1.7-3B", help="the checkpoint, a Hub id or a folder")
    ap.add_argument("--dataset-path", default="demo_data/droid_sample", help="DROID, LeRobot format; from the cwd")
    ap.add_argument("--n", type=int, default=20, help="requests to record")
    ap.add_argument("--warmup", type=int, default=3, help="observations run first and not recorded")
    ap.add_argument("--stride", type=int, default=15, help="steps between sampled observations")
    ap.add_argument("--seed", type=int, default=0, help="request i draws its noise under seed + i")
    ap.add_argument("--out", type=Path, help="the folder for req_XX.pt (this or --save-obs)")
    ap.add_argument("--obs-file", type=Path, help="read observations saved by --save-obs")
    ap.add_argument("--save-obs", type=Path, help="only decode the observations and save them")
    args = ap.parse_args()
    if (args.out is None) == (args.save_obs is None):
        ap.error("give exactly one of --out and --save-obs")

    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Gr00tPolicy(embodiment_tag=EMBODIMENT_TAG, model_path=args.model_path, device=device)
    total = args.warmup + args.n
    if args.obs_file:
        loaded = torch.load(args.obs_file, weights_only=False)
        # the warm-up reuses the file's first observations; the recorded ones are the file's, in order
        obs_list = [loaded[i % len(loaded)] for i in range(args.warmup)] + loaded[: args.n]
    else:
        obs_list = observations(args.dataset_path, policy.get_modality_config(), total, args.stride)
    if len(obs_list) < total:
        raise SystemExit(f"the dataset yields {len(obs_list)} observations, not {total}")
    if args.save_obs:
        torch.save(obs_list[args.warmup:], args.save_obs)
        print(f"{args.n} observations -> {args.save_obs}")
        return

    model, head = policy.model, policy.model.action_head
    captured: dict[str, torch.Tensor] = {}
    for name, module in (("vlln_out", head.vlln), ("vl_embeds", head.vl_self_attention),
                         ("state_features", head.state_encoder)):
        module.register_forward_hook(lambda _m, _i, output, name=name: captured.__setitem__(name, output.detach()))
    # The head's initial noise is an unseeded torch.randn inside get_action: record what it drew,
    # since the mstar side cannot reproduce another stack's RNG stream.
    real_randn = torch.randn

    def recording_randn(*a, **k):
        t = real_randn(*a, **k)
        captured["noise"] = t.detach().clone()  # a copy: the loop may update its latents in place
        return t

    def run(obs, rid: int):
        # Gr00tPolicy._get_action, with the collated inputs and the normalised chunk kept
        torch.manual_seed(args.seed + rid)
        states, processed = [], []
        for o in policy._unbatch_observation(obs):
            step = policy._to_vla_step_data(o)
            states.append(step.states)
            processed.append(policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": step}]))
        collated = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)
        torch.randn = recording_randn
        try:
            with torch.inference_mode():
                pred = model.get_action(**collated)
        finally:
            torch.randn = real_randn
        batched_states = {k: np.stack([s[k] for s in states], axis=0)
                          for k in policy.modality_configs["state"].modality_keys}
        action = policy.processor.decode_action(
            pred["action_pred"].float().cpu().numpy(), policy.embodiment_tag, batched_states)
        return collated, pred, action

    for i in range(args.warmup):
        run(obs_list[i], 0)
    args.out.mkdir(parents=True, exist_ok=True)
    for rid in range(args.n):
        obs = obs_list[args.warmup + rid]
        captured.clear()
        collated, pred, action = run(obs, rid)
        torch.save({
            "collated": to_cpu(collated),
            "noise": to_cpu(captured["noise"]),
            "vlln_out": to_cpu(captured["vlln_out"]),
            "vl_embeds": to_cpu(captured["vl_embeds"]),
            "state_features": to_cpu(captured["state_features"]),
            "image_mask": to_cpu(pred["image_mask"]) if "image_mask" in pred else None,
            "backbone_attention_mask": to_cpu(pred.get("backbone_attention_mask")),
            "action_pred": to_cpu(pred["action_pred"]),
            "decoded_action": {k: np.asarray(v) for k, v in action.items()},
            "seed": args.seed + rid,
            "raw_obs": obs,
            "embodiment_tag": EMBODIMENT_TAG,
            "video_keys": list(policy.modality_configs["video"].modality_keys),
            "state_keys": list(policy.modality_configs["state"].modality_keys),
        }, args.out / f"req_{rid:02d}.pt")
        print(f"req {rid:02d} -> {args.out / f'req_{rid:02d}.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
