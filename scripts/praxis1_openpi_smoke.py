"""One real cold DROID request over M*'s openpi WebSocket.

Run only against an isolated diagnostic server whose Praxis serving guard was
temporarily lifted. A finite 21x10 reply proves the full request and graph
path completed; it does not establish policy quality or simulation parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

from mstar.api_server.openpi import pack, unpack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, help="save raw actions for a later reference comparison")
    args = parser.parse_args()
    random = np.random.default_rng(71)
    observation = {
        "observation/exterior_image_1_left": random.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8,
        ),
        "observation/wrist_image_left": random.integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8,
        ),
        "observation/cartesian_position": np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
        "observation/gripper_position": np.array([0.0]),
        "prompt": "pick up the marker",
        "seed": 17,
    }
    started = time.perf_counter()
    with connect(
        f"ws://{args.host}:{args.port}/", compression=None,
        max_size=None, open_timeout=args.timeout, close_timeout=5,
    ) as socket:
        hello = socket.recv(timeout=args.timeout)
        assert isinstance(hello, bytes), f"unexpected handshake: {hello!r}"
        print(json.dumps({"phase": "handshake", "metadata": unpack(hello)}), flush=True)
        socket.send(pack(observation))
        reply = socket.recv(timeout=args.timeout)
    if isinstance(reply, str):
        raise RuntimeError(f"server returned an error: {reply}")
    actions = np.asarray(unpack(reply)["actions"])
    assert actions.shape == (21, 10), f"expected cold DROID rows [21,10], got {actions.shape}"
    assert actions.dtype == np.float32, f"expected float32 rows, got {actions.dtype}"
    assert np.isfinite(actions).all(), "the policy returned a non-finite action"
    assert (0 <= actions[:, 9]).all() and (actions[:, 9] <= 1).all(), "gripper must be clipped to [0,1]"
    if args.output is not None:
        np.save(args.output, actions)
    print(json.dumps({
        "phase": "actions", "shape": list(actions.shape), "dtype": str(actions.dtype),
        "min": float(actions.min()), "max": float(actions.max()),
        "grip_min": float(actions[:, 9].min()), "grip_max": float(actions[:, 9].max()),
        "sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "seconds": round(time.perf_counter() - started, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
