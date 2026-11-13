"""Capture one simulator observation and its live M* cold DROID reply.

The saved NPZ lets the pinned reference run after the M* server exits, so
the two full models never occupy the H200 together. This is one paired call,
not a simulated rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from praxis import single_arm
from praxis.serve.openpi_sim_client import openpi_observation
from praxis.sim.config import load_config
from praxis.sim.env import SimEnv
from praxis.sim.loop import History
from websockets.sync.client import connect

from mstar.api_server.openpi import pack, unpack
from mstar.model.praxis1.components.pose_frames import pose_to_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    cfg = load_config(args.config, ["task.name=lift_cube", "viewer.enabled=false"])
    env = SimEnv(cfg.env_config())
    history = History()
    history.push(env.reset(0))
    observed = history.obs_dict()
    prompt = env.prompt
    request, pose8_model = openpi_observation(history, prompt, 0)
    hand_pose = np.concatenate((request["observation/cartesian_position"],
                                request["observation/gripper_position"]))
    np.testing.assert_allclose(pose_to_model(hand_pose, "single_arm", ""), pose8_model, atol=1e-6)
    np.testing.assert_allclose(
        single_arm.pose8_to_model(
            np.concatenate((observed["robot_state.ee_pose"][-1], observed["grip"][-1])),
            single_arm.TURNED_FLANGE,
        ), pose8_model, atol=1e-6,
    )
    with connect(
        f"ws://{args.host}:{args.port}/", compression=None, max_size=None,
        open_timeout=args.timeout, close_timeout=5,
    ) as socket:
        meta = unpack(socket.recv(timeout=args.timeout))
        socket.send(pack(request))
        raw = socket.recv(timeout=args.timeout)
    if isinstance(raw, str):
        raise RuntimeError(raw)
    actions = np.asarray(unpack(raw)["actions"])
    assert actions.shape == (21, 10) and actions.dtype == np.float32 and np.isfinite(actions).all()
    np.savez_compressed(
        args.output, exterior=observed["image.exterior"], wrist=observed["image.wrist"],
        ee_pose=observed["robot_state.ee_pose"], grip=observed["grip"],
        descriptor=observed["descriptor"], prompt=prompt, actions=actions,
        pose8_model=pose8_model, meta=json.dumps(meta),
    )
    print(json.dumps({"phase": "sim_first_call", "output": str(args.output),
                      "prompt": prompt, "rows": len(actions), "meta": meta}), flush=True)


if __name__ == "__main__":
    main()
