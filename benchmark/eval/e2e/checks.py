"""Base vs head correctness facts for one model (limits are applied by ``report``, not here).

Per case and session: identical-output share and closeness for text, PSNR for images, video and frames, the
largest action difference, audio seconds, WER and transcript identity from the eval suites, and status codes for
request-path cases. Written to ``<model dir>/compare.json``.
"""

from __future__ import annotations

import difflib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.eval.schema import validate

PCM_RATE = 24000
TRUNCATED_RATIO = 0.8   # a head reply that is base's reply cut below this share of its length is truncated


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * float(np.log10(255.0**2 / mse))


def load(p: Path) -> Any:
    if p.suffix == ".png":
        from PIL import Image

        return np.asarray(Image.open(p).convert("RGB"))
    if p.suffix == ".mp4":
        import av

        with av.open(str(p)) as c:
            return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    if p.suffix == ".u8":
        return np.frombuffer(p.read_bytes(), dtype=np.uint8)
    if p.suffix == ".pcm":
        return np.frombuffer(p.read_bytes(), dtype=np.int16)
    if p.suffix == ".f32":
        return np.frombuffer(p.read_bytes(), dtype=np.float32)
    return p.read_text(encoding="utf-8")


def _try_load(p: Path) -> Any:
    try:
        return load(p)
    except Exception:  # noqa: BLE001 - a truncated PNG or MP4 is a finding about that arm, not a crash
        return None


def text_facts(pairs: list[tuple[str, str]]) -> dict:
    """Identity plus how close the texts are: greedy decoding forks at near-ties across server processes, so a
    late fork is not a regression, while an empty, truncated or unrelated reply is. ``text_truncated`` counts head
    replies that are base's reply cut short (a strict prefix under ``TRUNCATED_RATIO`` of its length)."""
    sims, prefix, ratio = [], [], []
    cut = [x != y and x.startswith(y) and len(y) < TRUNCATED_RATIO * len(x) for x, y in pairs]
    for x, y in pairs:
        sims.append(difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() if x or y else 1.0)
        k = next((i for i, (u, v) in enumerate(zip(x, y, strict=False)) if u != v), min(len(x), len(y)))
        prefix.append(k / max(len(x), len(y), 1))
        ratio.append(len(y) / len(x) if x else (1.0 if not y else 0.0))
    return {"text_identical": sum(x == y for x, y in pairs), "text_n": len(pairs),
            "text_sim_median": statistics.median(sims), "text_prefix_median": statistics.median(prefix),
            "text_len_ratio_median": statistics.median(ratio), "text_len_ratio_min": min(ratio),
            "text_truncated": sum(cut)}


def saved(a: Path | None, b: Path | None) -> dict:
    """Facts from two directories of saved outputs, matched by file name."""
    if a is None or b is None or not a.is_dir() or not b.is_dir():
        return {"missing": [arm for arm, x in (("base", a), ("head", b)) if x is None or not x.is_dir()]}
    na, nb = {p.name for p in a.iterdir()}, {p.name for p in b.iterdir()}
    texts, ps, act, bad = [], [], [], []
    for n in sorted(na & nb):
        x, y = _try_load(a / n), _try_load(b / n)
        if y is None and x is not None:   # base's output decodes and head's does not
            bad.append(n)
        if x is None or y is None:
            continue
        if isinstance(x, str):
            texts.append((x, y))
        elif n.endswith(".f32"):
            d = float(np.max(np.abs(x - y))) if x.shape == y.shape else math.inf
            act.append(math.inf if math.isnan(d) else d)   # max() would skip a NaN
        elif not n.endswith(".pcm"):
            ps.append(psnr(x, y))
    # Outputs base returned and head did not (speech without its audio, an empty reply) fail on their own.
    out: dict = {"n_base": len({n.split(".")[0] for n in na}), "n_head": len({n.split(".")[0] for n in nb}),
                 "only_base": sorted(na - nb), "head_undecodable": bad}
    pcm = {arm: sum(f.stat().st_size for f in d.glob("*.pcm")) / 2 / PCM_RATE for arm, d in (("base", a), ("head", b))}
    if texts:
        out.update(text_facts(texts))
    if ps:
        finite = [p for p in ps if not math.isnan(p)]
        out["psnr_min_db"] = min(finite) if len(finite) == len(ps) else None
        out["psnr_median_db"] = float(np.median(finite)) if finite else None
        out["shape_mismatch"] = len(ps) - len(finite)
    if any(n.endswith(".pcm") for n in na | nb):
        out["audio_s_base"], out["audio_s_head"] = round(pcm["base"], 2), round(pcm["head"], 2)
    if act:
        out["action_max_abs_diff"] = max(act)
    return out


def saved_dir(case_dir: Path, conc: int) -> Path | None:
    """Where the harness kept a point's outputs: ``audio/<label>-c<N>/r<repeat>/``."""
    dirs = sorted(case_dir.glob(f"audio/*-c{conc}/r*"))
    return dirs[-1] if dirs else None


def eval_file(d: Path, conc: int) -> dict | None:
    """The point's results, or None when there are none or the file is not a complete result (truncated)."""
    for f in sorted(d.glob(f"*-c{conc}.json")):
        try:
            js = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if validate(js) == [] and js["repeats"] and isinstance(js.get("summary"), dict):
            return js
    return None


def suite_facts(a: Path, b: Path, conc: int) -> dict:
    ja, jb = eval_file(a, conc), eval_file(b, conc)
    if not ja or not jb:
        return {"missing": [arm for arm, j in (("base", ja), ("head", jb)) if not j]}
    out: dict = {}
    for k in ("wer", "sim_mean", "take_turn_rate", "yield_rate", "audio_s_mean"):
        va, vb = (ja["summary"].get(k) or {}).get("mean"), (jb["summary"].get(k) or {}).get("mean")
        if va is not None or vb is not None:
            out[f"{k}_base"], out[f"{k}_head"] = va, vb
    ha = {r["id"]: r.get("hyp") for r in ja["repeats"][0]["requests"] if r.get("ok")}
    hb = {r["id"]: r.get("hyp") for r in jb["repeats"][0]["requests"] if r.get("ok")}
    ids = sorted(set(ha) & set(hb))
    if ids and any(ha[i] is not None for i in ids):
        out.update(text_facts([(ha[i] or "", hb[i] or "") for i in ids]))
    return out


def status_of(v: dict) -> str:
    """A request's effective outcome: its status, plus an in-band error (an NDJSON or SSE error event, a WebSocket
    error message) that rides on an HTTP 200."""
    if "status" in v:
        out = str(v["status"])
    elif "status_line" in v:
        out = v["status_line"].split(" ")[1] if " " in v["status_line"] else v["status_line"]
    elif "close_code" in v:
        out = f"ws{v['close_code']}"
    elif "exception" in v:
        return f"exc:{v['exception']}"
    else:
        out = "ok"
    if v.get("error_status"):
        return f"{out}+error {v['error_status']}"
    if v.get("error") or any(k.endswith("_error") for k in v):
        return f"{out}+error"
    return out


def _rp(d: Path) -> dict | None:
    try:
        js = json.loads((d / "rp.json").read_text())
    except (OSError, ValueError):   # absent, or cut short by a killed client
        return None
    return js if isinstance(js, dict) else None


def rp_facts(a: Path, b: Path) -> dict:
    ja, jb = _rp(a), _rp(b)
    if ja is None or jb is None:
        return {"missing": [arm for arm, j in (("base", ja), ("head", jb)) if j is None]}
    # A cancel case whose request was never cancelled tested nothing: on base for any reason, on head only when its
    # request simply finished first (``_completed``); a head error instead is a changed outcome, judged below.
    for arm, j in (("base", ja), ("head", jb)):
        if j.get("_unexercised") and (arm == "base" or j.get("_completed")):
            return {"unexercised": f"{arm}: {j['_unexercised']}"}
    diffs = []
    keys = sorted(k for k in set(ja) | set(jb) if not k.startswith("_"))
    for k in keys:
        va, vb = ja.get(k), jb.get(k)
        sa = status_of(va) if isinstance(va, dict) else "absent"
        sb = status_of(vb) if isinstance(vb, dict) else "absent"
        flags = [f for f in ("identical", "ok", "sse_done", "events") if isinstance(va, dict) and f in va]
        if sa != sb or any(va.get(f) != (vb or {}).get(f) for f in flags):
            diffs.append({"case": k, "base": sa, "head": sb, **{f"{f}_base": va.get(f) for f in flags},
                          **{f"{f}_head": (vb or {}).get(f) for f in flags}})
    return {"n": len(keys), "diffs": diffs}


def _quality(d: Path) -> dict | None:
    try:
        js = json.loads((d / "quality.json").read_text())
    except (OSError, ValueError):
        return None
    return js if isinstance(js, dict) else None


def quality_facts(a: Path, b: Path, kinds: list[str]) -> dict:
    """Per quality judge: both arms' mean score over the outputs both have a score for (``quality.py``), or why
    there is none."""
    qa, qb = _quality(a), _quality(b)
    out: dict = {}
    for kind in kinds:
        ea, eb = (qa or {}).get(kind) or {}, (qb or {}).get(kind) or {}
        sa, sb = ea.get("scores") or {}, eb.get("scores") or {}
        ids = sorted(i for i in set(sa) & set(sb) if sa[i] is not None and sb[i] is not None)
        f: dict = {"n": len(ids), "judge": ea.get("judge") or eb.get("judge")}
        if ids:
            f["base"], f["head"] = statistics.mean(sa[i] for i in ids), statistics.mean(sb[i] for i in ids)
        else:
            f["error"] = (ea.get("error") or eb.get("error")
                          or ("not scored" if qa is None or qb is None else "no output both arms scored"))
        out[kind] = f
    return out


def compare_model(model_dir: Path, cases: list[dict], sessions: int) -> dict:
    """compare.json for one model's results directory (``<arm>/s<k>/<case>/``)."""
    res: dict = {}
    for case in cases:
        conc = min(case["conc"])
        kinds = case.get("quality") or []
        per_session = []
        for s in range(sessions):
            a, b = (model_dir / arm / f"s{s}" / case["id"] for arm in ("base", "head"))
            try:   # one unreadable case must not take the other cases' facts with it
                if case["driver"] == "client":
                    facts = saved(saved_dir(a, conc), saved_dir(b, conc))
                elif case["driver"] == "suite":
                    facts = suite_facts(a, b, conc)
                else:
                    facts = rp_facts(a, b)
            except Exception as e:  # noqa: BLE001
                facts = {"missing": ["base", "head"], "error": f"{type(e).__name__}: {str(e)[:200]}"}
            if kinds:
                facts["quality"] = quality_facts(a, b, kinds)
            per_session.append(facts)
        if sessions > 1 and (case["driver"] == "client" or kinds):
            # Base against itself across two server processes: the noise floor for the identity and quality checks.
            s0, s1 = (model_dir / "base" / f"s{s}" / case["id"] for s in (0, 1))
            selfb = saved(saved_dir(s0, conc), saved_dir(s1, conc)) if case["driver"] == "client" else {}
            if kinds:
                selfb["quality"] = quality_facts(s0, s1, kinds)
            per_session.append({"self_base": selfb})
        res[case["id"]] = per_session
    tmp = model_dir / ".compare.json.tmp"   # whole or not at all
    tmp.write_text(json.dumps(res, indent=1, default=str))
    os.replace(tmp, model_dir / "compare.json")
    return res
