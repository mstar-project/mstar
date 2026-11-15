"""Concurrent LingBot-World rollouts through the served path.

Starts the API server on a LingBot config (``--max-concurrent`` sets its
``max_concurrent_requests``, ``--pool-tokens`` its ``resources: dit_kv: pool_tokens``), streams each distinct
rollout alone, then all the sessions at once (``--sessions`` of the same rollout, or one per ``--session HxW:CHUNKS``,
started ``--stagger`` seconds apart in that order), and checks:

  * every concurrent rollout's frames are byte-identical to its solo run: a
    session's rolling K/V never reach another's
  * when the pool or ``max_concurrent_requests`` has room for fewer rollouts
    than sessions, the extra rollouts wait (their first chunk arrives after an
    earlier rollout's last) and still finish, instead of running the GPU out of
    memory
  * with room for every session, the rollouts' chunks interleave
  * ``--expect-rearranges``: the pool rearranged at least that often (the
    server log's count, at INFO), which sizes like these force: D takes the front of
    the pool, S the room after it, and L fits only once D has ended and S has
    slid left into D's gap, with S's K/V moved under it while it runs

The trajectory is synthetic and the image one of Cosmos3-Edge's assets, as in
``run_served_numerics.py``.

LingBot-World refuses any ``max_concurrent_requests`` but 1 until the rollouts'
text K/V and VAE caches are engine resources (``LingBotWorldModel.validate_config_yaml``),
so the cases below that pass ``--max-concurrent`` above 1 need that check lifted locally.

    python test/lingbot_world/serve_concurrent.py --out /tmp/lingbot/conc1
    python test/lingbot_world/serve_concurrent.py --out /tmp/lingbot/conc2 --max-concurrent 2
    python test/lingbot_world/serve_concurrent.py --out /tmp/lingbot/pool --max-concurrent 2 \
        --pool-tokens 28080 --expect-waits 1
    python test/lingbot_world/serve_concurrent.py --out /tmp/lingbot/rearrange --max-concurrent 3 \
        --session 480x832:4 --session 240x416:24 --session 576x1024:6 --stagger 1 --expect-rearranges 1
    # a 72-token D: S slides across a gap short enough to be staged
    python test/lingbot_world/serve_concurrent.py --out /tmp/lingbot/short_gap --max-concurrent 3 \
        --pool-tokens 46440 --session 32x64:8 --session 240x416:24 --session 576x1024:6 --stagger 0.5 \
        --expect-rearranges 1
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mstar.client import MStarClient, VideoFrameChunk  # noqa: E402

PROMPT = "A quiet suburban street on a clear afternoon, the camera driving slowly forward."
INTRINSICS = [480.0, 480.0, 416.0, 240.0]


def trajectory(num_frames: int, yaw_per_frame: float = 0.002, step: float = 0.02) -> list:
    poses = np.tile(np.eye(4), (num_frames, 1, 1))
    for t in range(num_frames):
        a = yaw_per_frame * t
        poses[t, :3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        poses[t, :3, 3] = [step * t * math.sin(a), 0.0, step * t * math.cos(a)]
    return poses.tolist()


class Session(NamedTuple):
    # the requested output area, None for the model's default
    height: int | None
    width: int | None
    chunks: int


def parse_session(text: str) -> Session:
    size, chunks = text.split(":")
    height, width = size.split("x")
    return Session(int(height), int(width), int(chunks))


def kept_tokens(image: str, session: Session) -> int:
    """The rolling K/V tokens a layer the session's rollout declares, as the model computes them."""
    from PIL import Image

    from mstar.model.lingbot_world.lingbot_world_model import LingBotWorldModel
    from mstar.model.lingbot_world.submodules import rolling_extent

    model = LingBotWorldModel(skip_weight_loading=True)
    cfg = model.config
    width, height = Image.open(image).size
    lat_h, lat_w = model.geometry((height, width), session.height or cfg.default_height,
                                  session.width or cfg.default_width)
    return sum(rolling_extent(cfg, (lat_h * cfg.vae_stride[1], lat_w * cfg.vae_stride[2])))


def rollout(client: MStarClient, image: str, request_id: str, session: Session, t0: float) -> dict:
    chunks, arrivals = [], []
    num_frames = (session.chunks * 4 - 1) * 4 + 1
    for event in client.stream(
        text=PROMPT, images=image, input_modalities=("text", "image"), output_modalities=("video_frame",),
        request_id=request_id, seed=42, num_frames=num_frames, camera_poses=trajectory(num_frames),
        intrinsics=INTRINSICS, height=session.height, width=session.width,
    ):
        if not isinstance(event, VideoFrameChunk):
            raise RuntimeError(f"unexpected stream event {type(event).__name__}")
        chunks.append(event.data)
        arrivals.append(time.perf_counter() - t0)
    return {"chunks": chunks, "arrivals": arrivals}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs/lingbot_world_1_3b.yaml")
    ap.add_argument("--max-concurrent", type=int, help="the YAML's max_concurrent_requests")
    ap.add_argument("--pool-tokens", type=int, help="the YAML's resources: dit_kv: pool_tokens")
    ap.add_argument("--expect-waits", type=int,
                    help="rollouts that must wait for an earlier one; default: sessions - max_concurrent_requests")
    ap.add_argument("--sessions", type=int, default=2, help="concurrent copies of the default rollout")
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--session", type=parse_session, action="append", dest="specs",
                    help="HxW:CHUNKS, once per concurrent rollout (instead of --sessions)")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between the concurrent rollouts' starts")
    ap.add_argument("--expect-rearranges", type=int, default=0)
    ap.add_argument("--image")
    ap.add_argument("--port", type=int, default=18432)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.image is None:
        from huggingface_hub import hf_hub_download

        args.image = hf_hub_download("nvidia/Cosmos3-Edge", "assets/edge_action_id_av_0_output.png")

    config = yaml.safe_load(args.config.read_text())
    if args.max_concurrent is not None:
        config["max_concurrent_requests"] = args.max_concurrent
    if args.pool_tokens is not None:
        config.setdefault("resources", {}).setdefault("dit_kv", {})["pool_tokens"] = args.pool_tokens
    sessions = args.specs or [Session(None, None, args.chunks)] * args.sessions
    concurrent = config.get("max_concurrent_requests", len(sessions))
    expected_waits = args.expect_waits if args.expect_waits is not None else max(0, len(sessions) - concurrent)
    workdir = Path(tempfile.mkdtemp(prefix="lingbot-conc-"))
    config_path = workdir / "lingbot.yaml"
    config_path.write_text(yaml.safe_dump(config))

    log = open(args.out / "server.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(REPO / "mstar/api_server/entrypoint.py"), "--config", str(config_path),
         "--port", str(args.port), "--host", "127.0.0.1", "--socket-path-prefix", str(workdir / "sock"),
         "--upload-dir", str(workdir / "uploads"), "--tensor-comm-protocol", "SHM", "--log-level", "INFO",
         "--timeout", "900"],
        stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
    )
    client = MStarClient(f"http://127.0.0.1:{args.port}", timeout=900)
    report: dict = {"max_concurrent": concurrent, "pool_tokens": (config.get("resources") or {}).get("dit_kv", {}),
                    "sessions": [s._asdict() for s in sessions], "expected_waits": expected_waits,
                    "kept_tokens": [kept_tokens(args.image, s) for s in sessions]}
    try:
        deadline = time.time() + 1200
        while not client.health():
            if proc.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not come up (exit {proc.poll()}); see {args.out / 'server.log'}")
            time.sleep(2)
        solos = {}
        for j, session in enumerate(dict.fromkeys(sessions)):
            solo = solos[session] = rollout(client, args.image, f"solo-{j}", session, time.perf_counter())
            tag = "" if j == 0 else f"_{j}"
            np.save(args.out / f"solo{tag}_frames.npy",
                    np.stack([np.frombuffer(c, dtype=np.uint8) for c in solo["chunks"][1:]]))
            np.save(args.out / f"solo{tag}_first_chunk.npy", np.frombuffer(solo["chunks"][0], dtype=np.uint8))
        log.flush()
        rearranges_alone = (args.out / "server.log").read_text().count("rearranged the pool")
        results: dict[int, dict] = {}
        errors: dict[int, str] = {}
        t0 = time.perf_counter()

        def run(i: int):
            try:
                results[i] = rollout(client, args.image, f"conc-{i}", sessions[i], t0)
            except Exception as e:  # reported below
                errors[i] = f"{type(e).__name__}: {e}"

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(sessions))]
        for i, t in enumerate(threads):
            if i and args.stagger:
                time.sleep(args.stagger)
            t.start()
        for t in threads:
            t.join()
        report["errors"] = errors
        report["chunks_each"] = {i: len(r["chunks"]) for i, r in results.items()}
        report["identical_to_solo"] = {i: r["chunks"] == solos[sessions[i]]["chunks"] for i, r in results.items()}
        report["chunks_equal_to_solo"] = {
            i: [x == y for x, y in zip(r["chunks"], solos[sessions[i]]["chunks"], strict=False)]
            for i, r in results.items()}
        for i, r in results.items():
            frames = np.stack([np.frombuffer(c, dtype=np.uint8) for c in r["chunks"][1:]])
            np.save(args.out / f"session{i}_frames.npy", frames)
        spans = sorted((r["arrivals"][0], r["arrivals"][-1]) for r in results.values())
        report["first_last_chunk_s"] = spans
        # waited: a rollout's first chunk came after an earlier one's last
        report["waited"] = sum(1 for a, b in zip(spans, spans[1:], strict=False) if b[0] > a[1])
        arrivals = solos[sessions[0]]["arrivals"]
        report["solo_chunk_gaps_s"] = [round(b - a, 3) for a, b in zip(arrivals, arrivals[1:], strict=False)]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    report["rearranges"] = (args.out / "server.log").read_text().count("rearranged the pool") - rearranges_alone
    (args.out / "concurrent.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    ok = not report.get("errors") and all(report.get("identical_to_solo", {0: False}).values()) \
        and len(report.get("chunks_each", {})) == len(sessions)
    if ok and (report["waited"] < expected_waits or report["rearranges"] < args.expect_rearranges):
        ok = False
    print("concurrent rollouts", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
