"""Exercise cold, warm, reset and end on a disposable Praxis session server.

The diagnostic server must enable episode arrays and the Positronic readiness
gate only in its temporary source copy. This checks the live wire and graph;
the generated frames are not a robot or simulation quality test.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from websockets.sync.client import connect

from mstar.api_server.openpi import pack, unpack


def _observation(count: int, random: np.random.Generator, *, reset: bool = False) -> dict:
    exterior = random.integers(0, 256, size=(count, 224, 224, 3), dtype=np.uint8)
    wrist = random.integers(0, 256, size=(count, 224, 224, 3), dtype=np.uint8)
    pose = np.tile(np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    return {
        "image.exterior": exterior, "image.wrist": wrist,
        "robot_state.ee_pose": pose, "grip": np.zeros(count, dtype=np.float32),
        "task": "pick up the marker", "reset_context": reset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8126)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    random = np.random.default_rng(73)
    with connect(
        f"ws://{args.host}:{args.port}/api/v1/session", compression=None,
        max_size=None, open_timeout=args.timeout, close_timeout=5,
    ) as socket:
        loading = unpack(socket.recv(timeout=args.timeout))
        ready = unpack(socket.recv(timeout=args.timeout))
        assert loading.get("status") == "loading", loading
        assert ready.get("status") == "ready" and ready.get("protocol_version") == 2, ready
        session_id = ready["session_id"]
        print(json.dumps({"phase": "ready", "meta": ready["meta"]}), flush=True)
        for count, reset, expected in ((1, False, 21), (22, False, 24), (1, True, 21)):
            socket.send(pack({
                "session_id": session_id,
                "observation": _observation(count, random, reset=reset),
            }))
            reply = unpack(socket.recv(timeout=args.timeout))
            if reply.get("status") == "error":
                raise RuntimeError(reply["error"])
            commands = reply.get("result")
            assert isinstance(commands, list) and len(commands) == expected, (
                count, reset, expected, reply,
            )
            assert all(isinstance(row, dict) for row in commands)
            print(json.dumps({
                "phase": "call", "frames": count, "reset": reset,
                "rows": len(commands), "command_keys": sorted(commands[0]),
                "timing": reply.get("timing"),
            }), flush=True)
        socket.send(pack({"session_id": session_id, "end_session": True}))
        assert unpack(socket.recv(timeout=args.timeout)) == {
            "session_id": session_id, "end_session": True,
        }
        print(json.dumps({"phase": "end", "acknowledged": True}), flush=True)


if __name__ == "__main__":
    main()
