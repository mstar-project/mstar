"""Nemotron VoiceChat duplex on the 20 spoken questions of the omni set.

There's no independent reference (parity_compare.py runs mstar's own offline path), so replies are
checked for non-empty text, sane audio, and Whisper hearing roughly what the text says.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families import asr_judge
from benchmark.correctness.families.base import Family, failed_items, self_consistency_check
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet


class NemotronDuplex(Family):
    name = "duplex"
    set_name = "omni_v1"
    default_concurrency = 2
    default_restarts = 2
    self_consistency_only = True

    @property
    def trailing_s(self) -> float:
        """Silence after each question. The agent answers in the frames after the user stops
        (test/nemotron_duplex/_audio_prep.py pads 6 s too)."""
        return self.opt("trailing_s", 6.0, float)

    def select_items(self, items: list) -> list:
        return [it for it in items if it.params.get("modality") == "audio"]

    def padded_clip(self, sset: SampleSet, it) -> Path:
        """The question with trailing silence, written once next to the sets."""
        out = sset.root.parent.parent / "derived" / f"{sset.name}-trailing{self.trailing_s:g}s" / f"{it.id}.wav"
        if not out.is_file():
            x, sr = M.read_wav(sset.path(it.files["audio"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.wav")
            M.write_wav(tmp, np.concatenate([x, np.zeros(int(sr * self.trailing_s), np.float32)]), sr)
            tmp.rename(out)
        return out

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, files=[self.padded_clip(sset, it)], input_modalities="audio",
                        output_modalities="text,audio", model_kwargs={"temperature": 0.0, "seed": i})
                for i, it in enumerate(sset.items)]

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        for r in responses:
            if r.ok:
                (out_dir / f"{r.item_id}.txt").write_text(r.text, encoding="utf-8")
                if r.audio:
                    M.write_wav(out_dir / f"{r.item_id}.wav", M.pcm16_to_float(bytes(r.audio)),
                                r.audio_sample_rate(22050))

    def reference_id(self) -> str:
        return "nemotron-duplex-no-independent-reference"

    def reference_description(self) -> str:
        return (f"none independent of mstar, parity_compare.py runs mstar's own offline path; questions get "
                f"{self.trailing_s:g} s of trailing silence")

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        items = sset.items
        fails = failed_items(runs[0])
        sc.check("requests ok", len(items) - len(fails), "PASS" if not fails else "FAIL", "all",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        out = runs[0] / "outputs"
        wavs = [out / f"{it.id}.wav" for it in items]
        heard = asr_judge.transcribe(wavs, device)
        norm = M.make_normalizer("whisper")
        rows, bad, wers = [], [], []
        for it, w, h in zip(items, wavs, heard, strict=True):
            t = (out / f"{it.id}.txt").read_text(encoding="utf-8") if (out / f"{it.id}.txt").is_file() else ""
            row = {"id": it.id, "text": t[:160], "heard": h[:160]}
            if w.is_file():
                x, sr = M.read_wav(w)
                if p := M.audio_problems(M.audio_stats(x, sr), max_silence_frac=0.95):
                    bad.append(it.id)
                    row["problems"] = "; ".join(p)
            else:
                bad.append(it.id)
            if norm(t):
                row["wer_vs_own_text"] = round(M.wer([t], [h], norm)["wer"], 3)
                wers.append(row["wer_vs_own_text"])
            rows.append(row)
        empty = [r["id"] for r in rows if not norm(r["text"])]
        sc.check("empty text replies", len(empty), band(len(empty), lambda v: v == 0, lambda v: v <= 2), "0",
                 ", ".join(empty))
        sc.check("replies whose audio fails sanity", len(bad), band(len(bad), lambda v: v == 0, lambda v: v <= 2),
                 "0 PASS, <=2 WARN (up to 95% silence is fine, the agent listens first)", ", ".join(bad))
        if wers:
            med = 100 * statistics.median(wers)
            sc.check("median WER of spoken reply vs its own text (%)", round(med, 2),
                     band(med, lambda v: v <= 15, lambda v: v <= 30), "<=15 PASS, <=30 WARN (provisional)")
        same = []
        if len(runs) > 1:
            for it in items:
                a, b = (out / f"{it.id}.txt"), (runs[1] / "outputs" / f"{it.id}.txt")
                if a.is_file() and b.is_file():
                    same.append(float(norm(a.read_text(encoding="utf-8")) == norm(b.read_text(encoding="utf-8"))))
        frac = [sum(same) / len(same)] if same else []
        self_consistency_check(sc, frac, "greedy text replies identical across two server instances (fraction)",
                               len(runs), lambda v: v >= 0.8, lambda v: v >= 0.5,
                               ">=0.8 PASS, >=0.5 WARN (provisional, batching can flip near ties)")
        sc.rows = rows
        (work / "duplex.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        sc.artifacts += [str(work / "duplex.json"), str(out)]
        return sc
