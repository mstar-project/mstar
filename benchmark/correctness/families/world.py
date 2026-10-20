"""Waypoint: 3 seed images, 16 steps each under a fixed action stream.

world_engine isn't installed and the server's per-frame noise can't be fed into it, so this checks
frame counts, flat frames and same-seed repeatability across two server instances.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import Family, failed_items, frame_identity, self_consistency_check
from benchmark.correctness.report import FamilyScore
from benchmark.correctness.sets import SampleSet

NUM_STEPS = 16


def actions(n: int) -> list[dict]:
    """Fixed controls. A slow mouse circle, with button 0 pressed every fourth step."""
    return [{"mouse": [round(12 * math.cos(i / 4), 2), round(12 * math.sin(i / 4), 2)],
             "buttons": [0] if i % 4 == 0 else [], "scroll": 0} for i in range(n)]


class WaypointWorld(Family):
    name = "world-waypoint"
    set_name = "waypoint_v1"
    default_restarts = 2
    self_consistency_only = True
    request_timeout_s = 3600.0

    @property
    def size_key(self) -> str:
        return "seed_360p" if "360" in str(self.cfg.get("model_kwargs", {}).get("variant", "")) else "seed_720p"

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, files=[sset.path(it.files[self.size_key])], input_modalities="image",
                        output_modalities="video_frame",
                        model_kwargs={"num_steps": NUM_STEPS, "actions": actions(NUM_STEPS), "seed": it.params["seed"]})
                for it in sset.items]

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        for r in responses:
            meta = next((m for m in r.metadata if m.get("modality") == "video_frame"), None)
            if r.ok and r.video and meta:
                frames = np.frombuffer(bytes(r.video), dtype=np.uint8).reshape(-1, meta["height"], meta["width"], 3)
                np.save(out_dir / f"{r.item_id}.npy", frames)

    def reference_id(self) -> str:
        return "world-engine-no-served-reference"

    def reference_description(self) -> str:
        return "world_engine isn't installed, and the server's noise can't be fed into it anyway"

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        fails = failed_items(runs[0])
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} rollouts stream frames",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        rows, degenerate, wrong_count, repro, sheet, labels = [], [], [], [], [], []
        for it in sset.items:
            p = runs[0] / "outputs" / f"{it.id}.npy"
            row = {"id": it.id}
            if not p.is_file():
                rows.append(row)
                continue
            f = np.load(p)
            row.update(frames=len(f), dims="x".join(map(str, f.shape[1:])))
            if len(f) != 4 * NUM_STEPS:
                wrong_count.append(it.id)
            stds = f.reshape(len(f), -1).std(axis=1)
            if (stds < 2).any():
                degenerate.append(it.id)
            cont = [M.psnr(f[i], f[i + 1]) for i in range(len(f) - 1)]
            row["consecutive_psnr_median"] = round(statistics.median(cont), 2) if cont else None
            if len(runs) > 1 and (runs[1] / "outputs" / f"{it.id}.npy").is_file():
                g = np.load(runs[1] / "outputs" / f"{it.id}.npy")
                row["frames_identical_r1_r2"] = M.frames_equal(f, g)
                if g.shape == f.shape:
                    ps = M.frame_psnrs(f, g)
                    row["r1_r2_psnr_min"] = min(ps)
                    repro.append(min(ps))
            rows.append(row)
            sheet.append([f[0], f[len(f) // 2], f[-1]])
            labels += [f"{it.id} first", "mid", "last"]
        sc.check("rollouts with 4 frames per step", n - len(fails) - len(wrong_count),
                 "PASS" if not wrong_count else "FAIL", f"{4 * NUM_STEPS} frames each", ", ".join(wrong_count))
        sc.check("rollouts with a flat frame (pixel std < 2)", len(degenerate), "PASS" if not degenerate else "FAIL",
                 "0", ", ".join(degenerate))
        self_consistency_check(sc, repro, "same-seed min frame PSNR, two server instances (dB)", len(runs),
                               lambda v: v >= 40, lambda v: v >= 25,
                               ">=40 PASS (inf = byte-identical), >=25 WARN (provisional)")
        frame_identity(sc, rows, encoded=False)
        if sheet:
            M.contact_sheet(sheet, labels, work / "contact_sheet.png")
            sc.artifacts.append(str(work / "contact_sheet.png"))
        sc.rows = rows
        (work / "rollouts.json").write_text(json.dumps(rows, indent=1, default=str))
        sc.artifacts += [str(r / "outputs") for r in runs]
        return sc
