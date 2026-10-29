#!/usr/bin/env python3
"""Serve GR00T N1.7 through mstar and check what a client gets back, end to end.

Launches ``mstar/api_server/entrypoint.py`` on ``configs/gr00t_n17.yaml``, sends recorded DROID
observations through ``MStarClient`` exactly as ``mstar serve`` tells a client to (lossless
frames time-major over the views, ``embodiment_tag`` and ``state`` in ``model_kwargs``), and
holds the decoded actions to the parity bar: per request, the worst joint group's mean abs error
against NVIDIA's model in fp32 is at most 1.2x the worst group's bf16 floor (NVIDIA's model in
bf16 against itself in fp32), all through the same decode, **each averaged over the seeds**.

One seed is one sample: the four-step loop turns a floor-sized rounding difference into anything
from 0.3x to 2.2x the floor depending on its direction, and NVIDIA's own autocast route misses a
one-seed bar too. So every observation is served once per seed.

The server draws its own noise from the request's ``seed``, so the recordings must be NVIDIA's
reference run (its ``action_pred`` per request, fp32 and bf16) made with that same seed. Then, on
the same server:

* the same request twice returns identical bytes;
* the streamed action, read off the binary frames, equals the non-streamed one;
* malformed requests are a 400, and the next good request still returns the first answer;
* noise that does not match its oracle misses the bar, or the bar measures nothing.

    python test/gr00t_n17/serve_check.py --dump ~/gr00t-runs/dump \\
        --oracle-fp32 '~/gr00t-runs/oracle_fp32_seed{seed}' --oracle-bf16 '~/gr00t-runs/oracle_bf16_seed{seed}' \\
        --seeds 7 8 9 10 11 --out /tmp/gr00t_serve
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import msgpack
import numpy as np
import requests
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mstar.client import MStarClient  # noqa: E402
from mstar.client.media import BINARY_STREAM_MEDIA_TYPE, iter_binary_frames  # noqa: E402
from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor  # noqa: E402

BAR = 1.2
GR00T_REPO = "nvidia/GR00T-N1.7-3B"


def observation(dump: dict) -> tuple[str, list[tuple[str, bytes]], dict]:
    """One recorded observation as a client sends it: instruction, PNG frames, kwargs."""
    raw = dump["raw_obs"]
    text = raw["language"]["annotation.language.language_instruction"][0][0]
    views = [np.asarray(raw["video"][v])[0] for v in dump["video_keys"]]  # (T, H, W, 3) each
    frames = []
    for t in range(views[0].shape[0]):  # time-major over the views
        for name, view in zip(dump["video_keys"], views, strict=True):
            buf = io.BytesIO()
            Image.fromarray(view[t]).save(buf, format="PNG")
            frames.append((f"{name}_t{t}.png", buf.getvalue()))
    state = {k: np.asarray(v, dtype=np.float32).tolist() for k, v in raw["state"].items()}
    return text, frames, {"embodiment_tag": dump["embodiment_tag"], "state": state}


def decode(payload: bytes) -> dict[str, np.ndarray]:
    return {k: np.frombuffer(g["data"], g["dtype"]).reshape(g["shape"])
            for k, g in msgpack.unpackb(payload).items()}


def send(client: MStarClient, obs, seed: int) -> bytes:
    text, frames, kwargs = obs
    res = client.generate(text=text, images=frames, output_modalities=("action",), seed=seed, **kwargs)
    (entry,) = [r for r in res.raw if r["modality"] == "action"]
    return entry["bytes"]


def send_walk(client: MStarClient, obs, walk: str) -> np.ndarray:
    """A reuse walk's features: one ``tensor`` entry, laid out like an action group."""
    text, frames, kwargs = obs
    res = client.generate(text=text, images=frames, output_modalities=("tensor",), walk=walk, **kwargs)
    (entry,) = [r for r in res.raw if r["modality"] == "tensor"]
    e = msgpack.unpackb(entry["bytes"])
    return np.frombuffer(e["data"], e["dtype"]).reshape(e["shape"])


def send_streamed(url: str, obs, seed: int) -> bytes:
    """The streamed action off the binary frames. ``MStarClient.stream`` cannot carry it:
    its event fallback decodes an action's bytes as UTF-8 (``MStarClient._to_event``)."""
    text, frames, kwargs = obs
    data = {"text": text, "output_modalities": "action", "streaming": "true",
            "model_kwargs": json.dumps({**kwargs, "seed": seed})}
    with requests.post(f"{url}/generate", data=data, files=[("files", f) for f in frames], stream=True,
                       headers={"Accept": BINARY_STREAM_MEDIA_TYPE, "Accept-Encoding": "identity"},
                       timeout=600) as resp:
        resp.raise_for_status()
        assert resp.headers["content-type"].startswith(BINARY_STREAM_MEDIA_TYPE), resp.headers
        chunks = [f["bytes"] for f in iter_binary_frames(resp.raw) if f["modality"] == "action"]
    assert len(chunks) == 1, f"{len(chunks)} action chunks streamed"
    return chunks[0]


def errors(pre: Gr00tN17Preprocessor, got: dict, fp32: dict, bf16: dict, dump: dict) -> tuple[float, float]:
    """The served actions' parity after the decode: the worst joint group's error and the worst
    group's floor, the two recordings through the same ``unapply``."""
    state = dump["raw_obs"]["state"]
    want = pre.unapply(dump["embodiment_tag"], fp32["action_pred"].float().numpy(), state)
    rounded = pre.unapply(dump["embodiment_tag"], bf16["action_pred"].float().numpy(), state)
    assert list(got) == list(want), (list(got), list(want))
    worst = max(float(np.abs(got[g] - want[g]).mean()) for g in want)
    floor = max(float(np.abs(rounded[g] - want[g]).mean()) for g in want)
    return worst, floor


def malformed(url: str, obs, seed: int) -> dict[str, int]:
    """Each request is wrong in one way; each must be refused with a 400."""
    text, frames, kwargs = obs
    jpeg = io.BytesIO()
    Image.open(io.BytesIO(frames[0][1])).save(jpeg, format="JPEG")
    narrow = {**kwargs["state"], "joint_position": [[kwargs["state"]["joint_position"][0][0][:-1]]]}
    cases = {
        "model_kwargs not JSON": (frames, "{not json"),
        "a lossy frame": ([("f.jpg", jpeg.getvalue()), *frames[1:]], json.dumps({**kwargs, "seed": seed})),
        "one frame short": (frames[:-1], json.dumps({**kwargs, "seed": seed})),
        "a state group one value short": (frames, json.dumps({**kwargs, "state": narrow, "seed": seed})),
        # the wrong JSON type, which once reached an AttributeError and a 500
        "state a list, not an object": (frames, json.dumps({**kwargs, "state": [1.0], "seed": seed})),
        "embodiment_tag a number": (frames, json.dumps({**kwargs, "embodiment_tag": 3, "seed": seed})),
    }
    status = {}
    for name, (files, mk) in cases.items():
        r = requests.post(f"{url}/generate", data={"text": text, "output_modalities": "action",
                                                   "streaming": "false", "model_kwargs": mk},
                          files=[("files", f) for f in files], timeout=600)
        status[name] = r.status_code
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--oracle-fp32", required=True, help="recording dir, with {seed} in it")
    ap.add_argument("--oracle-bf16", required=True, help="recording dir, with {seed} in it")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="the seeds the oracles were made on")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18432)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    pre = Gr00tN17Preprocessor.from_pretrained(snapshot_download(GR00T_REPO), max_state_dim=132,
                                               max_action_horizon=40)
    names = sorted(p.name for p in args.dump.glob("req_*.pt"))[: args.n]
    load = lambda d, n: torch.load(d / n, map_location="cpu", weights_only=False)  # noqa: E731
    oracle = lambda which, seed, n: load(Path(which.format(seed=seed)).expanduser(), n)  # noqa: E731

    workdir = Path(tempfile.mkdtemp(prefix="gr00t-serve-"))
    log = open(args.out / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "mstar/api_server/entrypoint.py"),
         "--config", str(REPO / "configs/gr00t_n17.yaml"),
         "--port", str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(workdir / "sock"),
         "--upload-dir", str(workdir / "uploads"), "--tensor-comm-protocol", "SHM", "--log-level", "INFO",
         "--timeout", "600"],
        stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
    )
    url = f"http://127.0.0.1:{args.port}"
    client = MStarClient(url)
    report: dict = {"seeds": args.seeds, "bar": BAR, "ratios": {}, "per_seed": {}}
    failures: list[str] = []
    try:
        deadline = time.time() + 900
        while not client.health():
            if proc.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not come up (exit {proc.poll()}); see {args.out / 'server.log'}")
            time.sleep(2)

        seed, first = args.seeds[0], None
        wrong = max(args.seeds) + 1  # noise no oracle here was made on
        report["other_seed_ratios"] = {}
        for name in names:
            dump = load(args.dump, name)
            obs = observation(dump)
            pairs = []
            for s in args.seeds:
                payload = send(client, obs, s)
                first = first or (name, payload)
                pairs.append(errors(pre, decode(payload), oracle(args.oracle_fp32, s, name),
                                    oracle(args.oracle_bf16, s, name), dump))
            r = sum(e for e, _ in pairs) / sum(f for _, f in pairs)
            report["ratios"][name] = r
            report["per_seed"][name] = [e / f for e, f in pairs]
            print(f"  {name}  worst-group ratio {r:.3f} over {len(pairs)} seeds  (bar {BAR}; "
                  f"per seed {', '.join(f'{e / f:.2f}' for e, f in pairs)})")
            if not r <= BAR:
                failures.append(f"{name}: ratio {r:.3f} past the bar {BAR}")
            # every request, not one: the weakest observed (req_00) is 3.7x, the strongest 44x
            e, f = errors(pre, decode(send(client, obs, wrong)), oracle(args.oracle_fp32, seed, name),
                          oracle(args.oracle_bf16, seed, name), dump)
            report["other_seed_ratios"][name] = e / f
            if not e / f > BAR:
                failures.append(f"{name}: seed {wrong} against seed {seed}'s oracle meets the bar too "
                                f"({e / f:.3f}): it measures nothing")

        name, payload = first
        obs = observation(load(args.dump, name))
        report["repeat_identical"] = send(client, obs, seed) == payload
        report["stream_identical"] = send_streamed(url, obs, seed) == payload
        # each reuse walk returns its features, the same bytes twice
        report["walks"] = {}
        for walk in ("backbone", "state"):
            try:
                got = send_walk(client, obs, walk)
                same = np.array_equal(got, send_walk(client, obs, walk))
            except Exception as exc:  # a refused walk is a finding, not the end of the run
                report["walks"][walk] = f"{type(exc).__name__}: {exc}"[:300]
                failures.append(f"walk {walk}: {report['walks'][walk]}")
                continue
            report["walks"][walk] = {"shape": list(got.shape), "finite": bool(np.isfinite(got).all()),
                                     "repeat_identical": same}
            if not (same and np.isfinite(got).all()):
                failures.append(f"walk {walk}: {report['walks'][walk]}")
        report["malformed_status"] = malformed(url, obs, seed)
        report["serves_after_malformed"] = send(client, obs, seed) == payload
        for key in ("repeat_identical", "stream_identical", "serves_after_malformed"):
            if not report[key]:
                failures.append(key)
        failures += [f"{k}: {v}, not 400" for k, v in report["malformed_status"].items() if v != 400]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    ratios = sorted(report["ratios"].values())
    report["ratio_median"], report["ratio_max"] = (ratios[len(ratios) // 2], ratios[-1]) if ratios else (None, None)
    report["failures"] = failures
    (args.out / "serve.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "ratios"}, indent=2))
    if failures:
        print("served GR00T FAILED:\n  " + "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
