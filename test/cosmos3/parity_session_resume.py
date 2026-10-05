#!/usr/bin/env python3
"""Does a resumed windowed rollout still come out the same as on ``main``?

The session port moved cosmos3's world state out of its own ``SessionStore``
into the engine's per-session submodule state. That is meant to change *where*
the state lives, not the pixels, so the proof is a two-turn rollout driven on
both sides and compared frame by frame.

The two sides differ only in how a session is named, which is the point of the
port: ``main`` reads ``session_id`` / ``resume_session`` out of ``model_kwargs``,
this branch takes them as request fields. ``--api`` picks which.

Run one server at a time (cosmos3 is deterministic only at concurrency 1, so
both sides must also be sequential), dump each side's frames, then diff:

    # on main, in a worktree
    CUDA_VISIBLE_DEVICES=7 python -m mstar.api_server.entrypoint \
        --config configs/cosmos3_nano_ar.yaml --port 8150 ...
    python test/cosmos3/parity_session_resume.py --api kwargs --out /tmp/main.npz

    # on this branch
    python test/cosmos3/parity_session_resume.py --api fields --out /tmp/branch.npz

    python test/cosmos3/parity_session_resume.py --compare /tmp/main.npz /tmp/branch.npz

Numerically identical output is the pass. A difference says the port changed the
rollout, and the per-turn/per-frame breakdown says where it started.
"""

import argparse
import base64
import io
import json
import sys

import numpy as np
import requests

# One world, two prompts: turn 2 steers the same rollout somewhere new, which is
# what a resume is for. Seeded, so both sides draw the same noise.
TURN_1 = "a drone flies over a coastal town at dawn"
TURN_2 = "the drone turns inland over the hills"
SEED = 20261004


def _model_kwargs(args, **extra) -> dict:
    return {
        "num_frames": args.frames, "window_mode": args.window_mode,
        "window_frames": args.frames, "height": args.size, "width": args.size,
        "num_steps": args.steps, "seed": SEED, **extra,
    }


def _frames(body: bytes) -> np.ndarray:
    """Every decoded frame of the reply's video chunk, as uint8 RGB."""
    import av
    for line in body.splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        if chunk.get("modality") != "video":
            continue
        with av.open(io.BytesIO(base64.b64decode(chunk["data"]))) as container:
            return np.stack([
                frame.to_ndarray(format="rgb24")
                for frame in container.decode(video=0)
            ])
    raise RuntimeError("no video chunk in the reply")


def _session_id(body: bytes) -> str:
    for line in body.splitlines():
        if line.strip() and json.loads(line).get("modality") == "session":
            return json.loads(line)["metadata"]["session_id"]
    raise RuntimeError("the reply named no session")


def _post(url: str, text: str, mk: dict, fields: dict, timeout: float) -> bytes:
    data = [("text", text), ("output_modalities", "video"),
            ("model_kwargs", json.dumps(mk))]
    data += [(k, v) for k, v in fields.items()]
    r = requests.post(f"{url}/generate", data=data, timeout=timeout)
    r.raise_for_status()
    return r.content


def run(args) -> int:
    """Two turns of one session, returning both turns' frames."""
    # ``main`` has no session fields on the request: the id rides on the kwargs
    # there, which is what this PR moves off.
    kwargs_api = args.api == "kwargs"
    if kwargs_api:
        first = _post(args.url, TURN_1, _model_kwargs(
            args, session_id=args.session_id,
        ), {}, args.timeout)
        session_id = args.session_id
    else:
        first = _post(args.url, TURN_1, _model_kwargs(args),
                      {"start_session": "true"}, args.timeout)
        session_id = _session_id(first)
    print(f"turn 1 done, session {session_id}", flush=True)

    if kwargs_api:
        second = _post(args.url, TURN_2, _model_kwargs(
            args, session_id=session_id, resume_session=True,
        ), {}, args.timeout)
    else:
        second = _post(args.url, TURN_2, _model_kwargs(args), {
            "resume_session": "true", "session_id": session_id,
        }, args.timeout)
    print("turn 2 done", flush=True)

    turns = {"turn1": _frames(first), "turn2": _frames(second)}
    for name, frames in turns.items():
        print(f"  {name}: {frames.shape} {frames.dtype}")
    np.savez_compressed(args.out, **turns)
    print(f"wrote {args.out}")
    return 0


def compare(left: str, right: str) -> int:
    a, b = np.load(left), np.load(right)
    ok = True
    for name in ("turn1", "turn2"):
        x, y = a[name], b[name]
        if x.shape != y.shape:
            print(f"[FAIL] {name}: {x.shape} vs {y.shape}")
            ok = False
            continue
        diff = np.abs(x.astype(np.int16) - y.astype(np.int16))
        per_frame = diff.reshape(len(x), -1).max(axis=1)
        bad = np.nonzero(per_frame)[0]
        if not len(bad):
            print(f"[PASS] {name}: {len(x)} frames identical")
            continue
        ok = False
        print(
            f"[FAIL] {name}: {len(bad)}/{len(x)} frames differ, "
            f"first at {bad[0]}, max |delta| {diff.max()}, "
            f"mean |delta| {diff.mean():.3f}"
        )
    # turn 2 is the one the port could have broken: it reads the session's state
    print("\nturn 2 is the resumed one; turn 1 differing means something other "
          "than the session changed." if not ok else "\nthe resume is unchanged.")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8150")
    ap.add_argument("--api", choices=("fields", "kwargs"), default="fields",
                    help="'kwargs' for main's model_kwargs session knobs")
    ap.add_argument("--session-id", default="parity-world",
                    help="the id to use on the kwargs API, which mints none")
    ap.add_argument("--out", default="frames.npz")
    ap.add_argument("--frames", type=int, default=29)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--window-mode", default="kv", choices=("kv", "chained"))
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"),
                    help="diff two runs' frames instead of generating")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
