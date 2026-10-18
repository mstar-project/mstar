"""TTS on 40 Seed-TTS sentences, judged by Whisper.

Sampled audio can't be compared sample by sample. So every clip has to be finite and voiced, WER
has to be within a point of the reference package's WER (the repo's engine-parity bar), and
durations should be close to the reference's. Kokoro also gets a log-mel distance.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

from benchmark.correctness import metrics as M
from benchmark.correctness.client import Request, Response
from benchmark.correctness.families import asr_judge
from benchmark.correctness.families.base import Family, failed_items, write_meta
from benchmark.correctness.report import FamilyScore, band
from benchmark.correctness.sets import SampleSet

PCM_RATE = 24000  # /generate audio is headerless int16; its chunk metadata carries the actual rate


class TtsFamily(Family):
    name = "tts"
    set_name = "tts_seedtts_v1"
    default_concurrency = 4
    deterministic = False  # True: same text -> same audio up to kernel noise (no sampling)
    judge_model = asr_judge.JUDGE_MODEL

    # --- requests -----------------------------------------------------------------------------
    def tts_kwargs(self, i: int, item) -> dict:
        """model_kwargs for one /generate request."""
        return {"seed": i}

    def request_files(self, sset: SampleSet, item) -> list[Path]:
        return []

    def requests(self, sset: SampleSet) -> list[Request]:
        reqs = []
        for i, it in enumerate(sset.items):
            files = self.request_files(sset, it)
            reqs.append(Request(it.id, text=it.prompt, files=files, output_modalities="audio",
                                input_modalities="audio,text" if files else None,
                                model_kwargs=self.tts_kwargs(i, it)))
        return reqs

    def save(self, sset: SampleSet, responses: list[Response], out_dir: Path) -> None:
        for r in responses:
            if r.ok and r.audio:
                M.write_wav(out_dir / f"{r.item_id}.wav", M.pcm16_to_float(bytes(r.audio)),
                            r.audio_sample_rate(PCM_RATE))

    # --- reference ----------------------------------------------------------------------------
    def reference_synthesize(self, sset: SampleSet, device: str):
        """Yield (item, samples, sample_rate) from the reference implementation."""
        raise NotImplementedError(f"{self.name}: no reference implementation")

    def make_reference(self, sset: SampleSet, ref_dir: Path, device: str) -> None:
        for it, samples, sr in self.reference_synthesize(sset, device):
            M.write_wav(ref_dir / "outputs" / f"{it.id}.wav", np.asarray(samples, dtype=np.float32), sr)
            print(f"  ref {it.id}: {len(samples) / sr:.2f}s", flush=True)
        write_meta(ref_dir, family=self.name, reference=self.reference_description(), set=sset.name)

    # --- scoring ------------------------------------------------------------------------------
    def _judge(self, wavs: list[Path], device: str, cache: Path | None) -> list[str]:
        if cache is not None and cache.is_file():
            data = json.loads(cache.read_text())
            if data.get("judge") == self.judge_model and len(data.get("texts", [])) == len(wavs):
                return data["texts"]
        texts = asr_judge.transcribe(wavs, device, self.judge_model)
        if cache is not None:
            cache.write_text(json.dumps({"judge": self.judge_model, "texts": texts}, indent=1, ensure_ascii=False))
        return texts

    def score(self, sset: SampleSet, runs: list[Path], ref_dir: Path | None, work: Path,
              device: str) -> FamilyScore:
        sc = FamilyScore(self.name)
        norm = M.make_normalizer("whisper")
        texts = [it.prompt for it in sset.items]
        ref_out = self.ref_outputs(ref_dir)
        ref_wavs = [ref_out / f"{it.id}.wav" for it in sset.items] if ref_out else None
        ref_hyp = (self._judge(ref_wavs, device, ref_dir / f"judge_{self.judge_model.split('/')[-1]}.json")
                   if ref_wavs and all(p.is_file() for p in ref_wavs) else None)
        ref_wer = M.wer(texts, ref_hyp, norm) if ref_hyp else None
        if ref_hyp is None:
            sc.notes.append(f"No reference outputs in {ref_dir}. WER is against the input text only, and "
                            "there are no duration ratios.")

        per_run = []
        for k, run in enumerate(runs, start=1):
            fails = failed_items(run)
            wavs = [run / "outputs" / f"{it.id}.wav" for it in sset.items]
            hyps = self._judge(wavs, device, work / f"judge_r{k}.json")
            w = M.wer(texts, hyps, norm)
            stats, problems, ratios, mel = [], {}, [], []
            for it, wav in zip(sset.items, wavs, strict=True):
                if not wav.is_file():
                    problems[it.id] = [fails.get(it.id, "missing output")]
                    stats.append(None)
                    ratios.append(None)
                    continue
                x, sr = M.read_wav(wav)
                s = M.audio_stats(x, sr)
                stats.append(s)
                if p := M.audio_problems(s):
                    problems[it.id] = p
                if ref_out and (ref_out / f"{it.id}.wav").is_file():
                    rx, rsr = M.read_wav(ref_out / f"{it.id}.wav")
                    ratios.append(s["duration_s"] / max(len(rx) / rsr, 1e-6))
                    if self.deterministic:
                        mel.append(M.log_mel_distance(M.resample(x, sr, rsr), rx, rsr))
                else:
                    ratios.append(None)
            per_run.append({"wer": w, "hyps": hyps, "stats": stats, "problems": problems, "ratios": ratios,
                            "mel": mel, "failed": fails})

        r1 = per_run[0]
        sc.metrics["wer_percent"] = [round(100 * r["wer"]["wer"], 3) for r in per_run]
        sc.metrics["reference_wer_percent"] = round(100 * ref_wer["wer"], 3) if ref_wer else None
        sc.metrics["judge"] = self.judge_model
        durations = [s["duration_s"] for s in r1["stats"] if s]
        sc.metrics["audio_seconds_total"] = round(sum(durations), 2)
        valid_ratios = [x for x in r1["ratios"] if x is not None]
        sc.metrics["duration_ratio"] = M.distribution(valid_ratios)
        if r1["mel"]:
            sc.metrics["log_mel_distance_db"] = M.distribution(r1["mel"])

        n = len(sset.items)
        nfail = len(r1["failed"])
        sc.check("requests ok", n - nfail, "PASS" if nfail == 0 else "FAIL", f"all {n} requests return audio",
                 "; ".join(f"{k}: {v[:80]}" for k, v in list(r1["failed"].items())[:3]))
        nbad = sum(1 for k in r1["problems"] if k not in r1["failed"])
        sc.check("clips failing sanity", nbad, band(nbad, lambda v: v == 0, lambda v: v <= 2),
                 "0 PASS, <=2 WARN (finite, at least 0.3 s, at most 70% silent and 1% clipped)",
                 "; ".join(f"{k}: {', '.join(v)}" for k, v in list(r1["problems"].items())[:4]))
        wer_pt = 100 * r1["wer"]["wer"]
        if ref_wer:
            delta = wer_pt - 100 * ref_wer["wer"]
            sc.check("WER - reference WER (pt)", round(delta, 2), band(delta, lambda v: v <= 1.0, lambda v: v <= 3.0),
                     "<=1.0 PASS, <=3.0 WARN (provisional, the repo wants engines within 1 point)",
                     f"WER {wer_pt:.2f}% vs reference {100 * ref_wer['wer']:.2f}% ({r1['wer']['words']} words)")
        else:
            sc.check("WER vs input text (%)", round(wer_pt, 2), "NO_REFERENCE",
                     "needs the reference's WER on the same sentences", "absolute WER is reported for context")
        if valid_ratios:
            med = statistics.median(valid_ratios)
            sc.check("median duration ratio vs reference", round(med, 3),
                     band(med, lambda v: 0.85 <= v <= 1.15, lambda v: 0.7 <= v <= 1.3),
                     "0.85-1.15 PASS, 0.7-1.3 WARN (provisional, sampled speech rate varies)")
            out = [it.id for it, r in zip(sset.items, r1["ratios"], strict=True) if r is not None and not
                   0.5 <= r <= 2.0]
            sc.check("clips with duration ratio outside [0.5, 2]", len(out),
                     band(len(out), lambda v: v == 0, lambda v: v <= 2),
                     "0 PASS, <=2 WARN (catches truncated or runaway clips)", ", ".join(out[:8]))
        if r1["mel"]:
            med_mel = statistics.median(r1["mel"])
            sc.check("median log-mel distance to reference (dB)", round(med_mel, 3),
                     band(med_mel, lambda v: v <= 2.0, lambda v: v <= 4.0),
                     "<=2 PASS, <=4 WARN (provisional)")
        if len(per_run) > 1:
            sc.metrics["wer_spread_across_restarts_pt"] = round(max(sc.metrics["wer_percent"]) -
                                                                min(sc.metrics["wer_percent"]), 3)
        for i, it in enumerate(sset.items):
            s = r1["stats"][i]
            sc.rows.append({
                "id": it.id, "wer": round(r1["wer"]["rows"][i]["wer"], 3),
                "ref_wer": round(ref_wer["rows"][i]["wer"], 3) if ref_wer else None,
                "dur_s": round(s["duration_s"], 2) if s else None,
                "dur_ratio": round(r1["ratios"][i], 3) if r1["ratios"] and r1["ratios"][i] else None,
                "silence": round(s["silence_frac"], 2) if s else None,
                "problems": "; ".join(r1["problems"].get(it.id, [])),
                "text": it.prompt, "heard": r1["hyps"][i], "ref_heard": ref_hyp[i] if ref_hyp else None,
            })
        sc.rows.sort(key=lambda r: -(r["wer"] or 0))
        sc.artifacts += [str(r / "outputs") for r in runs]
        return sc
