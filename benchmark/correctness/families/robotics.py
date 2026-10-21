"""Robotics and video-latent models on DROID episodes: droid_v1 (10 episodes of lerobot/droid_100), and
droid_v2 (6 DROID 1.0.1 episodes with joint positions) for the Cosmos3 DROID policies.

Only V-JEPA2 has a reference here (HF VJEPA2Model in fp32). Its embeddings are gated on how far they
sit from fp32 compared with HF's own bf16 when that copy exists, else on cosine >= 0.999. lerobot,
openpi and the upstream vjepa2 repo aren't installed, so the rest are checked for sane output and
same-seed repeatability across two server instances.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import Family, failed_items, self_consistency_check, write_meta
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet


class ArrayOutputFamily(Family):
    """Families whose output is one float32 array per item."""

    set_name = "droid_v1"
    default_restarts = 2
    self_consistency_only = True
    out_modality = "action"
    repro_pass, repro_warn = 0.05, 0.2  # max-abs difference between two server instances, same seed

    def select_items(self, items: list) -> list:
        """--family-opt skip_empty_prompt=1 drops episodes without a task string (two of the ten in droid_v1)."""
        if self.opt("skip_empty_prompt", "0") in ("1", "true", "True"):
            return [it for it in items if it.prompt.strip()]
        return items

    def to_array(self, raw: np.ndarray, resp: Response | None = None) -> np.ndarray:
        return raw

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        for r in responses:
            data = bytes(r.action if self.out_modality == "action" else r.video)
            if r.ok and data:
                np.save(out_dir / f"{r.item_id}.npy", self.to_array(np.frombuffer(data, dtype=np.float32), r))

    def reference_id(self) -> str:
        return f"{self.name}-no-reference"

    def reference_description(self) -> str:
        return "none available in the ROCm env"

    def compare_to_reference(self, out: np.ndarray, ref: np.ndarray) -> dict:
        return M.array_errors(out, ref)

    def reference_check(self, sc: FamilyScore, rows: list[dict], ref_dir: Path | None = None) -> None:
        pass

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        fails = failed_items(runs[0])
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} requests return output",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        ref_out = self.ref_outputs(ref_dir)
        rows, nonfinite, shapes, repro = [], [], set(), []
        for it in sset.items:
            row = {"id": it.id}
            p = runs[0] / "outputs" / f"{it.id}.npy"
            if not p.is_file():
                rows.append(row)
                continue
            out = np.load(p)
            shapes.add(out.shape)
            row.update(dims="x".join(map(str, out.shape)), absmax=round(float(np.nanmax(np.abs(out))), 4),
                       std=round(float(np.nanstd(out)), 4))
            if not np.isfinite(out).all():
                nonfinite.append(it.id)
            if len(runs) > 1 and (runs[1] / "outputs" / f"{it.id}.npy").is_file():
                o2 = np.load(runs[1] / "outputs" / f"{it.id}.npy")
                if o2.shape == out.shape:
                    row["r1_r2_maxabs"] = round(float(np.abs(out - o2).max()), 6)
                    repro.append(row["r1_r2_maxabs"])
            if ref_out and (ref_out / f"{it.id}.npy").is_file():
                ref = np.load(ref_out / f"{it.id}.npy")
                if ref.shape == out.shape:
                    row.update({f"ref_{k}": round(v, 6) for k, v in self.compare_to_reference(out, ref).items()})
            rows.append(row)
        sc.check("outputs with non-finite values", len(nonfinite), "PASS" if not nonfinite else "FAIL", "0",
                 ", ".join(nonfinite))
        sc.check("one output size across items", len(shapes), "PASS" if len(shapes) == 1 else "FAIL", "1",
                 str(sorted(shapes))[:200])
        self_consistency_check(sc, repro, "same-seed max-abs difference, two server instances", len(runs),
                               lambda v: v <= self.repro_pass, lambda v: v <= self.repro_warn,
                               f"<={self.repro_pass} PASS, <={self.repro_warn} WARN (provisional, kernel choice "
                               "can differ per process)")
        if ref_out:
            self.reference_check(sc, rows, ref_dir)
        elif not self.self_consistency_only:
            sc.check("agreement with reference", None, "NO_REFERENCE", self.reference_description())
        sc.rows = rows
        (work / "arrays.json").write_text(json.dumps(rows, indent=1))
        sc.artifacts += [str(work / "arrays.json")] + [str(r / "outputs") for r in runs]
        return sc


def _robot_state(sset: SampleSet, it, width: int = 32) -> list[float]:
    s = np.load(sset.path(it.files["states"]))[it.params["step"]]
    return [float(v) for v in np.pad(s, (0, max(0, width - len(s))))[:width]]


class Pi05Actions(ArrayOutputFamily):
    """pi0.5 base: 3 camera frames + task + state -> [horizon, 32] normalized actions. The
    lerobot reference isn't installed.
    """

    name = "robotics-pi05"
    cams = ("exterior_image_1_left", "exterior_image_2_left", "wrist_image_left")

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, text=it.prompt, files=[sset.path(it.files[c]) for c in self.cams],
                        input_modalities="image,text", output_modalities="action",
                        model_kwargs={"robot_state": _robot_state(sset, it), "seed": it.params["seed"]})
                for it in sset.items]

    def to_array(self, raw, resp=None):
        return raw.reshape(-1, 32)

    def reference_id(self) -> str:
        return "lerobot-pi05-base"

    def reference_description(self) -> str:
        return "lerobot PI05Policy (not installed in the ROCm env)"


DROID_VIEW_TEXT = (
    "This video contains concatenated views from multiple camera perspectives. "
    "The top row is from the wrist-mounted camera. The bottom row contains two horizontally "
    "concatenated third-person perspective views of the scene from opposite sides, with the robot visible."
)


def droid_policy_prompt(task: str, json_prompt: bool, frames: int = 33, fps: float = 15.0,
                        hw: tuple[int, int] = (544, 736)) -> str:
    """The prompt cosmos-framework's action transform builds for the RoboLab server: the task with the
    concat-view, duration and resolution text (viewpoint_utils, duration_fps_text_timestamps,
    resolution_text_info), or the same fields as a JSON caption (json_formatter) for Edge."""
    task = task.strip()
    if not task:
        return ""
    if json_prompt:
        return json.dumps({
            "cinematography": {"framing": DROID_VIEW_TEXT},
            "actions": [{"time": f"0:00-0:{round(frames / fps):02d}",
                         "description": task if task.endswith((".", "!", "?")) else task + "."}],
            "duration": f"{int(frames / fps)}s",
            "fps": float(fps),
            "resolution": {"H": hw[0], "W": hw[1]},
            "aspect_ratio": "4,3",
        })
    text = task + (" " if task.endswith(".") else ". ") + DROID_VIEW_TEXT
    return (f"{text} The video is {float(int(frames / fps)):.1f} seconds long and is of {fps:.0f} FPS. "
            f"This video is of {hw[0]}x{hw[1]} resolution.")


class Cosmos3DroidActions(ArrayOutputFamily):
    """Cosmos3 DROID policies, conditioned the way cosmos-framework's RoboLab server serves them
    (action_policy_server_robolab.py): the 736x544 wrist-over-exteriors composite, the current joint
    positions and 1 - gripper as a clean first action row (33 rows and frames, 15 fps), and the task
    with view, duration and resolution text, a JSON caption for Edge, which also gets the cookbook's
    guidance_interval [960, 1001]. The server drops the state row, so each output is the 32 predicted
    rows, first 8 dims, gripper still in the model's 1 - gripper convention.
    """

    name = "robotics-cosmos3-droid"
    set_name = "droid_v2"
    raw_action_dim, chunk = 8, 32

    @property
    def is_edge(self) -> bool:
        return "edge" in self.model_key

    def _state(self, sset: SampleSet, it) -> np.ndarray:
        return np.load(sset.path(it.files["joint_states"]))[it.params["step"]]

    def requests(self, sset: SampleSet) -> list[Request]:
        reqs = []
        for it in sset.items:
            s = self._state(sset, it)
            kw = {"action_mode": "policy", "domain_name": "droid_lerobot", "raw_action_dim": self.raw_action_dim,
                  "action_chunk_size": self.chunk + 1, "num_frames": self.chunk + 1, "fps": 15.0,
                  "size": "736x544", "action": [[float(v) for v in s[:7]] + [1.0 - float(s[7])]],
                  "seed": it.params["seed"]}
            if self.is_edge:
                kw["guidance_interval"] = [960, 1001]
            reqs.append(Request(it.id, text=droid_policy_prompt(it.prompt, self.is_edge),
                                files=[sset.path(it.files["composite"])], input_modalities="image,text",
                                output_modalities="action", model_kwargs=kw))
        return reqs

    def to_array(self, raw, resp=None):
        return raw.reshape(self.chunk, -1)[:, : self.raw_action_dim]

    def score(self, sset, runs, ref_dir, work, device):
        # No reference, so these are context, not gates: how far the first predicted row is from the
        # joints the request sent, the step-to-step change over the chunk's spread, and the mean error
        # against the joints the robot recorded over the next 32 frames.
        sc = super().score(sset, runs, ref_dir, work, device)
        items = {it.id: it for it in sset.items}
        for row in sc.rows:
            p = runs[0] / "outputs" / f"{row['id']}.npy"
            if not p.is_file():
                continue
            out, it = np.load(p).astype(np.float64), items[row["id"]]
            k, states = it.params["step"], np.load(sset.path(it.files["joint_states"]))
            row.update(droid_chunk_metrics(out, states, k))
        for key in ("row1_vs_state_maxabs", "smooth_ratio", "mae_vs_recorded"):
            vals = [r[key] for r in sc.rows if key in r]
            if vals:
                sc.metrics[key] = M.distribution(vals)
        (work / "arrays.json").write_text(json.dumps(sc.rows, indent=1))
        return sc

    def reference_id(self) -> str:
        return f"cosmos3-droid-{self.model_key}"

    def reference_description(self) -> str:
        return "no Cosmos3 action reference runs on ROCm (cosmos-framework's RoboLab server is CUDA-only)"


def droid_chunk_metrics(out: np.ndarray, states: np.ndarray, k: int) -> dict:
    """A predicted [32, 8] joint chunk (model convention) against the episode's joint states [T, 8]."""
    joints = out[:, :7]
    future = states[k + 1:k + 1 + len(out), :7].astype(np.float64)
    x = out[:, :8]
    smooth = (np.abs(np.diff(x, axis=0)).mean(0) / np.maximum(x.std(0), 1e-6)).mean()
    m = {"row1_vs_state_maxabs": round(float(np.abs(joints[0] - states[k, :7]).max()), 4),
         "smooth_ratio": round(float(smooth), 3)}
    if len(future) == len(out):
        m["mae_vs_recorded"] = round(float(np.abs(joints - future).mean()), 4)
    return m


def _window(arr: np.ndarray, start: int, length: int) -> list[list[float]]:
    w = arr[start:start + length]
    if len(w) < length:
        w = np.concatenate([w, np.repeat(w[-1:], length - len(w), axis=0)])
    return w.astype(float).tolist()


class Vjepa2AcLatents(ArrayOutputFamily):
    """V-JEPA2-AC: 8-frame clip + actions/states -> rollout latents (DROIDDataset layout, H = 4)."""

    name = "robotics-vjepa2-ac"
    out_modality = "video"
    horizon = 4
    repro_pass, repro_warn = 1e-3, 1e-2

    def requests(self, sset: SampleSet) -> list[Request]:
        reqs = []
        for it in sset.items:
            start = max(it.params["step"] - 7, 0)
            n = 32 + self.horizon - 1
            reqs.append(Request(it.id, files=[sset.path(it.files["clip_mp4"])], input_modalities="video",
                                output_modalities="video",
                                model_kwargs={"actions": _window(np.load(sset.path(it.files["actions"])), start, n),
                                              "states": _window(np.load(sset.path(it.files["states"])), start, n),
                                              "rollout_horizon": self.horizon, "seed": it.params["seed"]}))
        return reqs

    def to_array(self, raw, resp=None):
        return raw.reshape(self.horizon, -1, 1408)

    def score(self, sset, runs, ref_dir, work, device):
        # Each step has its own action, state and context, so a step equal to the one before it is a step
        # the response repeated, which agreement between two instances can't show.
        sc = super().score(sset, runs, ref_dir, work, device)
        repeated = []
        for row in sc.rows:
            p = runs[0] / "outputs" / f"{row['id']}.npy"
            if p.is_file():
                out = np.load(p)
                row["step_maxabs_min"] = round(float(min(np.abs(out[k + 1] - out[k]).max()
                                                         for k in range(len(out) - 1))), 6)
                if row["step_maxabs_min"] == 0.0:
                    repeated.append(row["id"])
        sc.check("outputs with a rollout step equal to the step before", len(repeated),
                 "PASS" if not repeated else "FAIL", "0", ", ".join(repeated))
        return sc

    def reference_id(self) -> str:
        return "vjepa2-ac-upstream"

    def reference_description(self) -> str:
        return "upstream vjepa2 AC predictor (benchmark/vjepa2_repo_ac.py), repo not available"


class Vjepa2Embedding(ArrayOutputFamily):
    """V-JEPA2 ViT-L encoder (skip_predictor) vs HF VJEPA2Model in fp32."""

    name = "embedding-vjepa2"
    out_modality = "video"
    default_restarts = 1
    self_consistency_only = False
    hf_id = "facebook/vjepa2-vitl-fpc64-256"
    repro_pass, repro_warn = 1e-3, 1e-2

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, files=[sset.path(it.files["clip_mp4"])], input_modalities="video",
                        output_modalities="video", model_kwargs={"skip_predictor": True}) for it in sset.items]

    def to_array(self, raw, resp=None):
        return raw.reshape(1, -1, 1024)

    @property
    def ref_dtype(self) -> str:
        """fp32 (the gate) or bf16 (--family-opt ref_dtype=bf16, to size the bf16 gap)."""
        return self.opt("ref_dtype", "fp32")

    def reference_id(self) -> str:
        return f"hf-vjepa2-vitl-fpc64-256-encoder-{self.ref_dtype}"

    def reference_description(self) -> str:
        return f"HF transformers VJEPA2Model {self.hf_id} ({self.ref_dtype}, eager) encoder, AutoVideoProcessor"

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        import torch
        from transformers import AutoVideoProcessor, VJEPA2Model

        from benchmark.correctness.families.video import read_video

        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[self.ref_dtype]
        proc = AutoVideoProcessor.from_pretrained(self.hf_id)
        model = VJEPA2Model.from_pretrained(self.hf_id, dtype=dtype).to(device).eval()
        for it in sset.items:
            frames = read_video(sset.path(it.files["clip_mp4"]))
            inputs = proc(videos=torch.from_numpy(frames).permute(0, 3, 1, 2), return_tensors="pt").to(device)
            inputs = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in inputs.items()}
            with torch.inference_mode():
                out = model(**inputs, skip_predictor=True).last_hidden_state
            np.save(ref_dir / "outputs" / f"{it.id}.npy", out.float().cpu().numpy())
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name)

    def compare_to_reference(self, out, ref):
        e = M.array_errors(out, ref)
        e["cosine"] = M.cosine(out, ref)
        return e

    def reference_check(self, sc: FamilyScore, rows: list[dict], ref_dir: Path | None = None) -> None:
        cos = [r["ref_cosine"] for r in rows if "ref_cosine" in r]
        if not cos:
            sc.check("embedding cosine vs reference", None, "ERROR", "no comparable outputs (size mismatch?)")
            return
        worst = min(cos)
        sc.metrics["cosine"] = M.distribution(cos)
        # The bf16 reference against the fp32 one, when both exist: the gap bf16 alone opens.
        spread = {}
        other = ref_dir.parent.parent / "hf-vjepa2-vitl-fpc64-256-encoder-bf16" / ref_dir.name if ref_dir else None
        if self.ref_dtype == "fp32" and other is not None and (other / "meta.json").is_file():
            for r in rows:
                if "ref_cosine" in r and (other / "outputs" / f"{r['id']}.npy").is_file():
                    a = np.load(ref_dir / "outputs" / f"{r['id']}.npy")
                    b = np.load(other / "outputs" / f"{r['id']}.npy")
                    if a.shape == b.shape:
                        spread[r["id"]] = (r["ref_cosine"], M.cosine(b, a))
            sc.metrics["hf_bf16_vs_hf_fp32_cosine"] = M.distribution([c for _, c in spread.values()])
        raw_rule = ">=0.999 PASS (competitor matrix bar), >=0.99 WARN"
        if not spread:
            sc.check("min embedding cosine vs reference", round(worst, 6),
                     band(worst, lambda v: v >= 0.999, lambda v: v >= 0.99), raw_rule)
            return
        # A bf16 serving path can't reach 0.999 against fp32, so compare its distance from fp32 (1 - cosine) with
        # HF bf16's own, at the median and at the worst item. 2x distance is the 3 dB the image/video floors allow.
        served = [1 - c for c, _ in spread.values()]
        hf_bf16 = [1 - c for _, c in spread.values()]
        ratio = max(statistics.median(served) / max(statistics.median(hf_bf16), 1e-12),
                    max(served) / max(*hf_bf16, 1e-12))
        sc.check("embedding distance from fp32 vs HF bf16's (larger of the median and worst-item ratios)",
                 round(ratio, 3), band(ratio, lambda v: v <= 2, lambda v: v <= 4),
                 f"<=2x PASS, <=4x WARN; raw min cosine {worst:.6f} (fixed bar for fp32 serving: {raw_rule})")
