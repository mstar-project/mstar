#!/usr/bin/env python3
"""Does a spoken answer match its own text?

For a thinker-talker model the prompt is not the reference: the model answers
rather than reads aloud, and its talker samples, so scoring against the prompt
or across two runs measures nothing. Scoring each answer's audio against that
same answer's text does: speech that does not match the text it was generated
alongside means the talker was conditioned on something other than what the
thinker produced -- a truncated prompt, say.

``benchmark.runner --output-dir`` writes ``req_<id>.txt`` and ``req_<id>.wav``
per request, so the pairing is by request id. An answer longer than Whisper's
30 s window is transcribed window by window, and the report counts them.

    python -m benchmark.answer_audio_wer --dirs out/off out/c256 --out wer.json

With two arms, the report adds a two-sided Fisher exact test of their counts of
answers above ``--bad-wer``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.tts_wer import WHISPER_WINDOW_SECONDS, transcribe


def pairs(d: Path) -> list[tuple[str, str, Path]]:
    """``(request id, text, wav)`` for every answer that produced both."""
    out = []
    for txt in sorted(d.glob("req_*.txt")):
        wav = txt.with_suffix(".wav")
        text = txt.read_text(encoding="utf-8").strip()
        if wav.exists() and text:
            out.append((txt.stem.removeprefix("req_"), text, wav))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="one benchmark.runner --output-dir per arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--asr-batch-size", type=int, default=16)
    # #365 read "garbled" off this threshold; kept so the numbers compare
    ap.add_argument("--bad-wer", type=float, default=0.5)
    args = ap.parse_args()

    import jiwer
    import soundfile as sf
    import torch
    from transformers import WhisperTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    normalize = WhisperTokenizer.from_pretrained(args.asr_model).normalize

    report: dict[str, dict] = {}
    for d in args.dirs:
        path = Path(d)
        arm = path.name
        got = pairs(path)
        if not got:
            report[arm] = {"error": f"no req_*.txt / req_*.wav pairs under {path}"}
            print(f"{arm}: no answers found under {path}")
            continue
        hyps = transcribe([wav for _, _, wav in got], args.asr_model, device, args.asr_batch_size, windowed=True)
        rows = []
        for (rid, text, wav), hyp in zip(got, hyps, strict=True):
            ref_n, hyp_n = normalize(text), normalize(hyp)
            # an empty reference makes jiwer divide by zero; count it, don't score it
            wer = None if not ref_n else jiwer.wer(ref_n, hyp_n)
            seconds = sf.info(wav).duration
            rows.append({
                "request": rid, "wer": wer, "seconds": seconds,
                "text": text, "heard": hyp, "wav": str(wav),
            })
        scored = sorted(r["wer"] for r in rows if r["wer"] is not None)
        bad = [r for r in rows if r["wer"] is not None and r["wer"] > args.bad_wer]
        report[arm] = {
            "answers": len(rows),
            "scored": len(scored),
            "wer_mean": sum(scored) / len(scored) if scored else None,
            "wer_median": scored[len(scored) // 2] if scored else None,
            "wer_p90": scored[min(len(scored) - 1, int(0.9 * len(scored)))] if scored else None,
            f"above_{args.bad_wer}": len(bad),
            # transcribed in several windows, where a word split at a boundary may be misheard
            "over_window": sum(r["seconds"] > WHISPER_WINDOW_SECONDS for r in rows),
            "worst": sorted(rows, key=lambda r: -(r["wer"] or 0))[:5],
            "rows": rows,
        }
        print(
            f"{arm}: {len(rows)} answers, mean WER "
            f"{report[arm]['wer_mean']:.3f}, {len(bad)} above {args.bad_wer}"
            if scored else f"{arm}: {len(rows)} answers, none scorable"
        )

    arms = [arm for arm in report if "error" not in report[arm]]
    if len(arms) == 2:
        from scipy.stats import fisher_exact

        table = [
            [report[arm][f"above_{args.bad_wer}"], report[arm]["scored"] - report[arm][f"above_{args.bad_wer}"]]
            for arm in arms
        ]
        p = fisher_exact(table).pvalue
        report["fisher_exact"] = {"arms": arms, "table": table, "p": p}
        print(f"Fisher exact, {arms[0]} vs {arms[1]} above {args.bad_wer}: p = {p:.3g}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
