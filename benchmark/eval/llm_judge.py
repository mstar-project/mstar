"""LLM judges for Full-Duplex-Bench v1.5: behavior labels and semantic A/F/U quality.

A judge is any OpenAI-compatible chat-completions endpoint, named on the command line:

* local open-weights: ``--judge qwen=http://127.0.0.1:30000/v1`` (any
  OpenAI-compatible server, M\\* included, serving for example ``Qwen/Qwen3.8-27B``). The model
  sent is the endpoint's only ``/v1/models`` entry (or ``URL#served-name``), and every response must report it.
  ``--judge-revision qwen=<commit>`` records the checkpoint revision.
* API: ``--judge gpt4o=openai`` is ``gpt-4o-2024-08-06`` (the paper's judge) with the key
  in ``OPENAI_API_KEY`` (``OPENAI_BASE_URL`` overrides the endpoint). Without a key the
  judge is skipped and the results file says so. Keys are read from the environment
  only, and never written to results or logs.

Behavior labels use the official ``eval_behavior`` prompt and seed 1. An unparseable reply
moves to the next seed, at most three attempts (the reference runner's cap; the official
loop has none). A transport error is retried on the same seed, and once those retries are
exhausted the pair is labeled failed on that seed. A reply from another model, a label
outside the four ``C_*`` tags, or (local judges) a non-``stop`` finish is not a valid label;
there is no fallback judge. Local judges decode as the reference judge does (non-thinking,
temperature 0, top_p 1, 512 tokens).

The semantic judge grades each pair on ``interaction_handling``, ``relevance`` and
``grounding`` (accept, fail or unresolved) with the reference judge's frozen prompt and schema,
batches of 8 pairs, thinking on, temperature 0, seed 1. A verdict without an exact
substring quote of the source it names becomes ``unresolved``, never a pass. Before
grading, the six frozen control cases must reproduce all 18 expected statuses; otherwise
the run is not graded (a rubric check, not a measure of judge accuracy).
"""

from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

OPENAI_MODEL = "gpt-4o-2024-08-06"
OPENAI_URL = "https://api.openai.com/v1"
C_LABELS = ("C_RESPOND", "C_RESUME", "C_UNCERTAIN_HANDLING", "C_UNKNOWN")
AXES = ("interaction_handling", "relevance", "grounding")
STATUSES = ("accept", "fail", "unresolved")
SOURCES = ("output_asr_text", "native_output_text", "input_asr_text", "original_task", "event_text",
           "clean_output_asr_text", "none")
BEHAVIOR_SEEDS = (1, 2, 3)
LOCAL_BEHAVIOR_DECODING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
                           "max_tokens": 512}
SEMANTIC_DECODING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 32768, "seed": 1}
SEMANTIC_BATCH = 8
MAX_ATTEMPTS = 3
# Rate limits and server errors are retried at the transport with the same seed and exponential
# backoff, so they never use up the official loop's seed attempts.
RETRY_HTTP = {408, 409, 429, 500, 502, 503, 504}
TRANSPORT_RETRIES = 6
BACKOFF_S = 2.0
BACKOFF_MAX_S = 60.0
RETRY_AFTER_MAX_S = 300.0
ACCEPTABLE_OUTCOMES = ["Apply the supplied rubric."]


class JudgeError(RuntimeError):
    pass


class TransportError(JudgeError):
    """The endpoint stayed unavailable through every transport retry. Not a reply, so the
    official loop must not move on to its next seed."""


def _count(v: Any) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


@dataclass
class Judge:
    name: str
    kind: str  # "local" or "openai"
    base_url: str
    model: str | None = None
    api_key_env: str | None = None
    revision: str | None = None
    root: str | None = None
    skipped: str | None = None
    timeout_s: float = 900.0
    retry_sleep_s: float = 1.0
    post: Callable[[str, dict, dict], dict] | None = field(default=None, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    usage: dict[str, int] = field(default_factory=lambda: {"requests": 0, "retries": 0, "prompt": 0, "completion": 0})
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def available(self) -> bool:
        return self.skipped is None

    def identity(self) -> dict[str, Any]:
        """What decides this judge's labels, for the results file (no keys)."""
        out = {"name": self.name, "kind": self.kind, "model": self.model, "revision": self.revision,
               "base_url": self.base_url, "skipped": self.skipped}
        if self.root and self.root != self.model:
            out["root"] = self.root
        out["behavior"] = {"seeds": list(BEHAVIOR_SEEDS),
                           "decoding": LOCAL_BEHAVIOR_DECODING if self.kind == "local" else "official (model default)",
                           "enable_thinking": False if self.kind == "local" else None}
        return out

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(TRANSPORT_RETRIES + 1):
            wait = min(BACKOFF_MAX_S, BACKOFF_S * 2**attempt) * (0.5 + random.random())
            try:
                resp = (self.post or _post)(f"{self.base_url}/chat/completions", body, self.headers(), self.timeout_s)
            except urllib.error.HTTPError as exc:
                # The status is enough; a body or message could echo the request headers.
                error, retry = f"HTTP {exc.code}", exc.code in RETRY_HTTP
                after = (exc.headers or {}).get("Retry-After")
                if retry and after and after.replace(".", "", 1).isdigit():
                    if float(after) > RETRY_AFTER_MAX_S:
                        raise TransportError(error) from None
                    wait = float(after)  # a minimum, so never shortened
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:  # incl. truncated bodies
                error, retry = type(exc).__name__, True
            except ValueError as exc:  # an unparseable body is the reply's problem, not the transport's
                raise JudgeError(type(exc).__name__) from None
            else:
                if not isinstance(resp, dict):
                    raise JudgeError("reply is not an object")
                u = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
                with self._lock:
                    self.usage["requests"] += 1
                    self.usage["prompt"] += _count(u.get("prompt_tokens"))
                    self.usage["completion"] += _count(u.get("completion_tokens"))
                return resp
            if not retry:
                raise JudgeError(error)
            if attempt == TRANSPORT_RETRIES:
                raise TransportError(error)
            with self._lock:
                self.usage["retries"] += 1
            self.sleep(wait)
        raise AssertionError("unreachable")


def _post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def _get(url: str, headers: dict, timeout_s: float = 30.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout_s) as r:
        return json.loads(r.read())


def parse_judges(specs: list[str], revisions: list[str] | None = None) -> list[Judge]:
    """``NAME=openai`` or ``NAME=URL[#served-model]`` specs, in order."""
    revs = {}
    for kv in revisions or []:
        k, sep, v = kv.partition("=")
        if not sep:
            raise ValueError(f"--judge-revision expects NAME=REV, got {kv!r}")
        revs[k] = v
    judges: list[Judge] = []
    for spec in specs or []:
        name, sep, target = spec.partition("=")
        if not sep or not name or not target:
            raise ValueError(f"--judge expects NAME=openai or NAME=URL, got {spec!r}")
        if not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ValueError(f"judge name {name!r}: letters, digits, '.', '_' or '-'")
        if any(j.name == name for j in judges):
            raise ValueError(f"judge {name!r} given twice")
        if target == "openai":
            j = Judge(name, "openai", os.environ.get("OPENAI_BASE_URL", OPENAI_URL).rstrip("/"), OPENAI_MODEL,
                      api_key_env="OPENAI_API_KEY", revision=OPENAI_MODEL)
            if not os.environ.get("OPENAI_API_KEY"):
                j.skipped = "OPENAI_API_KEY is not set"
        else:
            url, _, served = target.partition("#")
            j = Judge(name, "local", url.rstrip("/"), served or None, api_key_env="JUDGE_API_KEY",
                      revision=revs.get(name))
        judges.append(j)
    unknown = set(revs) - {j.name for j in judges}
    if unknown:
        raise ValueError(f"--judge-revision for unknown judges {sorted(unknown)}")
    return judges


def resolve(judge: Judge) -> Judge:
    """Ask a local endpoint which model it serves (once, before any request is judged)."""
    if judge.kind != "local" or not judge.available:
        return judge
    try:
        models = _get(f"{judge.base_url}/models", judge.headers()).get("data", [])
    except (OSError, ValueError) as exc:
        raise SystemExit(f"judge {judge.name!r}: its /models endpoint is unreachable ({type(exc).__name__})") from None
    ids = {m.get("id"): m.get("root") for m in models}
    if judge.model is None:
        if len(ids) != 1:
            raise SystemExit(f"judge {judge.name!r} serves {sorted(ids)}; name one with URL#model")
        judge.model = next(iter(ids))
    elif judge.model not in ids:
        raise SystemExit(f"judge {judge.name!r}: {judge.model!r} is not served (it serves {sorted(ids)})")
    judge.root = ids[judge.model]
    return judge


# --- behavior labels --------------------------------------------------------------------------


def behavior_body(judge: Judge, behavior, transcripts: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {"model": judge.model, "messages": [
        {"role": "system", "content": behavior.instruction},
        {"role": "user", "content": behavior.user_message(transcripts)},
    ]}
    if judge.kind == "local":
        body |= LOCAL_BEHAVIOR_DECODING | {"chat_template_kwargs": {"enable_thinking": False}}
    return body


def behavior_label(judge: Judge, behavior, transcripts: dict[str, Any], sleep=time.sleep) -> dict[str, Any]:
    """One pair's label: ``{"status", "label", "attempts"}``."""
    body = behavior_body(judge, behavior, transcripts)
    errors = []
    for i, seed in enumerate(BEHAVIOR_SEEDS):
        try:
            resp = judge.chat(body | {"seed": seed})
            choice = resp["choices"][0]
            parsed = behavior.parse_eval(choice["message"]["content"])
        except TransportError as exc:
            return {"status": "failed", "label": None, "seed": seed, "attempts": i + 1, "errors": [*errors, str(exc)]}
        except Exception as exc:  # noqa: BLE001 - the official loop retries on any exception
            errors.append(str(exc) if isinstance(exc, JudgeError) else type(exc).__name__)
            if i + 1 < len(BEHAVIOR_SEEDS):
                sleep(judge.retry_sleep_s)
            continue
        labels = parsed.get("behaviour") if isinstance(parsed, dict) else None
        if resp.get("model") != judge.model:
            status = "model_mismatch"
        elif isinstance(labels, list) and len(labels) == 1 and labels[0] in C_LABELS:
            status = "valid"
        else:
            status = "invalid_label"
        if judge.kind == "local" and status in ("valid", "invalid_label") and choice.get("finish_reason") != "stop":
            status = "invalid_finish"
        return {"status": status, "label": labels[0] if status == "valid" else None, "seed": seed,
                "attempts": i + 1, "errors": errors}
    return {"status": "failed", "label": None, "seed": None, "attempts": len(BEHAVIOR_SEEDS), "errors": errors}


def label_pairs(judge: Judge, behavior, transcripts: list[dict[str, Any] | None], concurrency: int = 8) -> list[dict]:
    """Labels for every pair (``None`` transcripts: the pair is not assessable)."""
    def one(t):
        return {"status": "not_assessable", "label": None} if t is None else behavior_label(judge, behavior, t)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        return list(ex.map(one, transcripts))


# --- semantic A/F/U ---------------------------------------------------------------------------


@dataclass
class SemanticAssets:
    prompt: str
    schema: dict[str, Any]
    controls: list[dict[str, Any]]
    expected: dict[str, dict[str, str]]

    @classmethod
    def load(cls) -> SemanticAssets:
        from benchmark.eval.fdb_reference import semantic_file

        expected = json.loads(semantic_file("expected").read_text())
        return cls(
            prompt=semantic_file("prompt").read_text(encoding="utf-8"),
            schema=json.loads(semantic_file("schema").read_text()),
            controls=[json.loads(ln) for ln in semantic_file("controls").read_text().splitlines() if ln],
            expected={a["sample_id"]: {ax: a["axes"][ax]["status"] for ax in AXES} for a in expected["assessments"]},
        )


def semantic_body(judge: Judge, assets: SemanticAssets, packets: list[dict[str, Any]]) -> dict[str, Any]:
    user = ("Response JSON schema:\n" + json.dumps(assets.schema) + "\n\nCases, in input order:\n"
            + "\n".join(json.dumps(p, ensure_ascii=False) for p in packets))
    body = {"model": judge.model,
            "messages": [{"role": "system", "content": assets.prompt}, {"role": "user", "content": user}],
            **SEMANTIC_DECODING,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "semantic_assessments", "schema": assets.schema}}}
    if judge.kind == "local":
        body["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        body["max_tokens"] = 16384  # the API's output cap for gpt-4o
    return body


def submit(judge: Judge, body: dict[str, Any]) -> dict[str, Any] | None:
    """The assessments object, retrying transport errors, non-stop finishes and unparseable
    JSON; a parseable reply is final, and a reply from another model is rejected."""
    for _ in range(MAX_ATTEMPTS):
        try:
            resp = judge.chat(body)
        except JudgeError:
            continue
        if resp.get("model") != judge.model:
            return None
        try:
            choice = resp["choices"][0]
            finish, text = choice.get("finish_reason"), choice["message"]["content"]
        except (KeyError, IndexError, TypeError, AttributeError):
            continue
        if finish != "stop":
            continue
        try:
            content = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(content, dict) and isinstance(content.get("assessments"), list):
            return content
    return None


def _unresolved(reason: str) -> dict[str, dict[str, Any]]:
    return {ax: {"status": "unresolved", "conversion": reason} for ax in AXES}


def validate_axis(raw: Any, packet: dict[str, Any]) -> dict[str, Any]:
    """An axis verdict as scored: schema-checked, and ``unresolved`` unless its quote is an
    exact substring of the source it names."""
    keys = {"status", "reason", "source", "quote"}
    if (not isinstance(raw, dict) or set(raw) != keys or raw["status"] not in STATUSES or raw["source"] not in SOURCES
            or not isinstance(raw["reason"], str) or len(raw["reason"]) > 350
            or not isinstance(raw["quote"], str) or len(raw["quote"]) > 220):
        return {"status": "unresolved", "conversion": "schema"}
    status, source, quote = raw["status"], raw["source"], raw["quote"]
    if source == "none":
        if status != "unresolved" or quote:
            return {"status": "unresolved", "conversion": "source none is valid only for unresolved with no quote"}
        return {"status": "unresolved"}
    if not quote:
        if status == "unresolved":
            return {"status": "unresolved"}
        return {"status": "unresolved", "conversion": f"{status} verdict has no evidence quote"}
    if quote not in str(packet[source]):
        return {"status": "unresolved", "conversion": f"quote is not an exact substring of {source}"}
    return {"status": status}


def validate_response(content: dict[str, Any] | None, packets: list[dict[str, Any]]) -> dict[str, dict]:
    by_id = {p["sample_id"]: p for p in packets}
    if content is None:
        return {sid: _unresolved("batch response failed") for sid in by_id}
    out = {sid: _unresolved("no assessment returned") for sid in by_id}
    items = content["assessments"]
    ids = [it.get("sample_id") if isinstance(it, dict) else None for it in items]
    for it, sid in zip(items, ids, strict=True):
        if sid not in by_id:
            continue
        if ids.count(sid) > 1:
            out[sid] = _unresolved("duplicate sample_id in response")
        elif not isinstance(it.get("axes"), dict) or set(it) != {"sample_id", "axes"}:
            out[sid] = _unresolved("assessment does not match the schema")
        else:
            out[sid] = {ax: validate_axis(it["axes"][ax], by_id[sid]) if ax in it["axes"]
                        else {"status": "unresolved", "conversion": "axis missing"} for ax in AXES}
    return out


def joint_status(axes: dict[str, str]) -> str:
    """``fail`` if any axis fails, ``accept`` only if all three accept, else ``unresolved``."""
    statuses = set(axes.values())
    if "fail" in statuses:
        return "fail"
    return "accept" if statuses == {"accept"} else "unresolved"


def run_controls(judge: Judge, assets: SemanticAssets) -> dict[str, Any]:
    got = validate_response(submit(judge, semantic_body(judge, assets, assets.controls)), assets.controls)
    mismatches = [f"{sid} {ax}: expected {assets.expected[sid][ax]}, got {got[sid][ax]['status']}"
                  for sid in assets.expected for ax in AXES if got[sid][ax]["status"] != assets.expected[sid][ax]]
    return {"total": len(AXES) * len(assets.expected), "mismatches": mismatches, "passed": not mismatches}


def grade(judge: Judge, assets: SemanticAssets, packets: list[dict[str, Any]], concurrency: int = 8) -> dict[str, dict]:
    """Axis outcomes per packet ``sample_id``, in batches of ``SEMANTIC_BATCH`` pairs."""
    batches = [packets[i:i + SEMANTIC_BATCH] for i in range(0, len(packets), SEMANTIC_BATCH)]

    def one(batch):
        return validate_response(submit(judge, semantic_body(judge, assets, batch)), batch)

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for res in ex.map(one, batches):
            out |= res
    return out
