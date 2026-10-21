"""Recompute text error rates from the transcripts a results file already holds.

Transcribing is the expensive part of a TTS or ASR eval; normalizing is not.
``python -m benchmark.eval rescore`` reapplies a normalizer to the stored
hypotheses (TTS judge transcripts, ASR outputs) against the manifest's
references, so runs scored differently can be compared without a GPU.
"""

from __future__ import annotations

from benchmark.eval import stats
from benchmark.eval.datasets import Item
from benchmark.eval.schema import RunResult
from benchmark.eval.text_metrics import error_rate, get_normalizer

RESCORABLE = ("tts", "asr")


def rescore(result: RunResult, items: list[Item], normalizer: str) -> RunResult:
    if result.suite not in RESCORABLE:
        raise ValueError(f"{result.suite!r} results hold no transcripts to rescore")
    name, normalize = get_normalizer(normalizer)
    by_id = {it.id: it for it in items}
    suite_cfg = result.config.setdefault("suite", {})
    default_lang = suite_cfg.get("judge_language")
    for rep in result.repeats:
        if result.suite == "asr":
            rows = rep.requests  # a failed request scores as an empty transcript, as in the suite
            hyps = [r.get("hyp", "") if r["ok"] else "" for r in rows]
        else:
            rows = [r for r in rep.requests if r["ok"] and "hyp" in r]
            hyps = [r["hyp"] for r in rows]
        if not rows:
            continue
        redacted = [r["id"] for r in rows if "hyp" in (r.get("scrubbed") or [])]
        if redacted:
            raise ValueError(f"{result.label}: transcripts of {redacted[:3]} were redacted when the results were "
                             "written (they held an address or a home path); rescoring them would score the "
                             "redaction, so rerun the suite with the new normalizer instead")
        missing = [r["id"] for r in rows if r["id"] not in by_id]
        if missing:
            raise ValueError(f"manifest lacks items {missing[:3]} from {result.label}")
        refs = [by_id[r["id"]].text for r in rows]
        langs = [by_id[r["id"]].lang or default_lang for r in rows]
        rate, per = error_rate(refs, hyps, normalize, langs)
        for r, counts in zip(rows, per, strict=True):
            r["wer"] = counts.rate
        rep.metrics["wer"] = rate
        if result.suite == "tts":
            rep.metrics["wer_utt_mean"] = stats.mean(c.rate for c in per)
    result.config.setdefault("rescored", []).append({"from": suite_cfg.get("normalizer"), "to": name})
    suite_cfg["normalizer"] = name
    return result.finalize()
