"""ASR on 100 LibriSpeech test-clean clips. Gate: WER vs the HF reference on the same clips."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families.base import Family, failed_items, write_meta
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet


class AsrFamily(Family):
    name = "asr"
    set_name = "asr_librispeech_v1"
    default_concurrency = 8
    hf_id = ""
    prompt = ""  # the text field sent with the audio
    abs_wer_pass = 5.0  # whisper-large-v3 is ~2% on test-clean

    def model_kwargs(self, i: int) -> dict:
        return {"temperature": 0.0, "max_tokens": 448, "max_output_tokens": 448, "seed": i}

    def requests(self, sset: SampleSet) -> list[Request]:
        return [Request(it.id, text=self.prompt or None, files=[sset.path(it.files["audio"])],
                        input_modalities="audio,text" if self.prompt else "audio", output_modalities="text",
                        model_kwargs=self.model_kwargs(i)) for i, it in enumerate(sset.items)]

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        for r in responses:
            if r.ok:
                (out_dir / f"{r.item_id}.txt").write_text(r.text, encoding="utf-8")

    def reference_transcribe(self, sset: SampleSet, device: str) -> list[str]:
        raise NotImplementedError(f"{self.name}: no reference implementation")

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        for it, text in zip(sset.items, self.reference_transcribe(sset, device), strict=True):
            (ref_dir / "outputs" / f"{it.id}.txt").write_text(text, encoding="utf-8")
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name)

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        norm = M.make_normalizer("whisper")
        truth = [it.expected["transcript"] for it in sset.items]
        ref_out = self.ref_outputs(ref_dir)
        ref = [(ref_out / f"{it.id}.txt").read_text(encoding="utf-8") for it in sset.items] if ref_out else None
        ref_wer = M.wer(truth, ref, norm) if ref else None
        runs_hyp = []
        for run in runs:
            hyps = []
            for it in sset.items:
                p = run / "outputs" / f"{it.id}.txt"
                hyps.append(p.read_text(encoding="utf-8") if p.is_file() else "")
            runs_hyp.append(hyps)
        hyps = runs_hyp[0]
        fails = failed_items(runs[0])
        w = M.wer(truth, hyps, norm)
        sc.metrics["wer_percent"] = [round(100 * M.wer(truth, h, norm)["wer"], 3) for h in runs_hyp]
        sc.metrics["words"] = w["words"]
        n = len(sset.items)
        sc.check("requests ok", n - len(fails), "PASS" if not fails else "FAIL", f"all {n} requests return text",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(fails.items())[:3]))
        empty = [it.id for it, h in zip(sset.items, hyps, strict=True) if not h.strip() and it.id not in fails]
        sc.check("empty transcripts", len(empty), "PASS" if not empty else "FAIL", "0", ", ".join(empty[:8]))
        abs_wer = 100 * w["wer"]
        sc.check("WER vs ground truth (%)", round(abs_wer, 3), band(abs_wer, lambda v: v <= self.abs_wer_pass,
                                                                     lambda v: v <= 2 * self.abs_wer_pass),
                 f"<={self.abs_wer_pass} PASS, <={2 * self.abs_wer_pass} WARN, a sanity bound for test-clean")
        if ref_wer:
            delta = 100 * (w["wer"] - ref_wer["wer"])
            sc.metrics["reference_wer_percent"] = round(100 * ref_wer["wer"], 3)
            agree = sum(norm(a) == norm(b) for a, b in zip(hyps, ref, strict=True))
            sc.metrics["transcripts_identical_to_reference"] = agree
            sc.metrics["wer_vs_reference_transcripts_percent"] = round(100 * M.wer(ref, hyps, norm)["wer"], 3)
            sc.check("WER - reference WER (pt)", round(delta, 3), band(delta, lambda v: v <= 0.5, lambda v: v <= 1.5),
                     "<=0.5 PASS, <=1.5 WARN (provisional, 0.5 pt is about 10 words here)",
                     f"WER {100 * w['wer']:.2f}% vs reference {100 * ref_wer['wer']:.2f}%")
            sc.check("transcripts identical to reference (normalized)", f"{agree}/{n}",
                     band(agree / n, lambda v: v >= 0.85, lambda v: v >= 0.7),
                     ">=85% PASS, >=70% WARN (provisional)")
        else:
            sc.notes.append(f"No reference outputs in {ref_dir}, scored against ground truth only.")
            sc.check("WER - reference WER (pt)", None, "NO_REFERENCE", "gate needs the reference WER on the same clips")
        if len(runs_hyp) > 1:
            same = sum(norm(a) == norm(b) for a, b in zip(runs_hyp[0], runs_hyp[1], strict=True))
            sc.metrics["noise_floor_identical_r1_r2"] = f"{same}/{n}"
        for i, it in enumerate(sset.items):
            sc.rows.append({"id": it.id, "wer": round(w["rows"][i]["wer"], 3),
                            "ref_wer": round(ref_wer["rows"][i]["wer"], 3) if ref_wer else None,
                            "truth": truth[i], "hyp": hyps[i].strip(), "ref": ref[i].strip() if ref else None})
        sc.rows.sort(key=lambda r: -r["wer"])
        (work / "transcripts.json").write_text(json.dumps(sc.rows, indent=1, ensure_ascii=False))
        sc.artifacts.append(str(work / "transcripts.json"))
        return sc
