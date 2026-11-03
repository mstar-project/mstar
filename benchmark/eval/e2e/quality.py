"""Perceptual quality of each arm's saved outputs, scored on the GPU host once both arms' servers have stopped.

python -m benchmark.eval.e2e.quality JOB.json      (run by ``arm`` after the last session)

For every case that lists ``quality`` judges, per arm and session, ``<arm>/s<k>/<case>/quality.json`` holds one score
per saved c=1 output: ``mos`` (UTMOS22 of each speech clip), ``speaker_sim`` (WavLM similarity of each clip to its
item's reference clip) and ``clip`` (CLIP score of each image, or of a video's sampled frames, against its prompt).
``checks`` turns the two arms' scores into facts and ``report`` judges them. Scoring after the servers stop keeps the
judges' GPU time and memory out of every timed pass.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from benchmark.eval.audio import pcm16_to_float, read_wav
from benchmark.eval.e2e import checks

SIM_MODEL = "microsoft/wavlm-base-plus-sv"
CLIP_MODEL = "openai/clip-vit-base-patch32"
VIDEO_FRAMES = 4      # evenly spaced frames a video's CLIP score averages


class Judges(Protocol):
    ids: dict[str, str]

    def mos(self, clip: tuple[np.ndarray, int]) -> float | None: ...

    def speaker_sim(self, clip: tuple[np.ndarray, int], ref: tuple[np.ndarray, int]) -> float: ...

    def clip(self, images: list, text: str) -> float: ...


class ModelJudges:
    """#502's UTMOS22 and WavLM judges and a CLIP judge, each loaded on first use and kept for the process."""

    def __init__(self, device: str | None = None):
        self.device, self.ids = device, {}

    def mos(self, clip):
        from benchmark.eval import judges

        j = judges.mos_judge(self.device)
        self.ids["mos"] = j.id
        return j.score(clip)

    def speaker_sim(self, clip, ref):
        from benchmark.eval import judges

        j = judges.sim_judge(SIM_MODEL, self.device)
        self.ids["speaker_sim"] = j.id
        return j.similarity(clip, ref)

    def clip(self, images, text):
        from benchmark.eval import judges

        j = judges.clip_judge(CLIP_MODEL, self.device)
        self.ids["clip"] = j.id
        return j.score(images, text)


def speech(p: Path) -> tuple[np.ndarray, int] | None:
    if p.suffix == ".wav":
        return read_wav(p)
    if p.suffix == ".pcm":
        return pcm16_to_float(p.read_bytes(), 1), checks.PCM_RATE
    return None


def pictures(p: Path) -> list | None:
    from PIL import Image

    if p.suffix == ".png":
        return [Image.open(p).convert("RGB")]
    if p.suffix == ".mp4":
        frames = checks.load(p)
        idx = np.linspace(0, len(frames) - 1, min(VIDEO_FRAMES, len(frames))).round().astype(int)
        return [Image.fromarray(frames[i]) for i in idx]
    return None


def prompt(item_id: str) -> str:
    from benchmark.eval.suites.e2e import GEN_PROMPTS

    return GEN_PROMPTS[int(item_id) % len(GEN_PROMPTS)]   # the e2e suite's prompt for item index int(id)


def score_dir(d: Path, kinds: list[str], judges: Judges, refs: dict[str, str]) -> dict[str, Any]:
    """{kind: {"judge", "scores": {item id: score}}} for one directory of saved outputs; a judge that cannot score
    records its error instead, so one broken judge leaves the others' scores standing."""
    out: dict[str, Any] = {}
    files = sorted(p for p in d.iterdir() if p.is_file())
    for kind in kinds:
        scores: dict[str, float | None] = {}
        try:
            for p in files:
                if kind == "clip":
                    imgs = pictures(p)
                    if imgs:
                        scores[p.stem] = judges.clip(imgs, prompt(p.stem))
                    continue
                clip = speech(p)
                if clip is None or not len(clip[0]):
                    continue
                if kind == "mos":
                    scores[p.stem] = judges.mos(clip)
                elif refs.get(p.stem):
                    scores[p.stem] = judges.speaker_sim(clip, read_wav(refs[p.stem]))
            out[kind] = {"judge": judges.ids.get(kind, kind), "scores": scores}
        except Exception as e:  # noqa: BLE001 - a judge that fails to load or score is reported, not fatal
            out[kind] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    return out


def score_model(out: Path, cases: list[dict], sessions: int, data: Path, datasets: dict[str, str],
                judges: Judges) -> int:
    """quality.json for every case with quality judges, arm and session; returns how many were written."""
    n = 0
    for case in cases:
        kinds = case.get("quality") or []
        if not kinds:
            continue
        refs: dict[str, str] = {}
        if "speaker_sim" in kinds and case.get("data"):
            from benchmark.eval.datasets import load_manifest

            refs = {it.id: it.ref_audio for it in load_manifest(data / datasets[case["data"]]) if it.ref_audio}
        for arm in ("base", "head"):
            for s in range(sessions):
                cd = out / arm / f"s{s}" / case["id"]
                saved = checks.saved_dir(cd, min(case["conc"])) if cd.is_dir() else None
                if saved is None:
                    continue
                res = score_dir(saved, kinds, judges, refs)
                tmp = cd / ".quality.json.tmp"
                tmp.write_text(json.dumps(res))
                os.replace(tmp, cd / "quality.json")
                n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    job = json.loads(Path((argv or sys.argv[1:])[0]).read_text())
    n = score_model(Path(job["out"]), job["model"]["cases"], job["sessions"], Path(job["data"]), job["datasets"],
                    ModelJudges())
    print(f"quality: scored {n} case directories", flush=True)


if __name__ == "__main__":
    main()
