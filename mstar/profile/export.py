"""Machine-readable export of a :class:`RequestProfile`.

``pretty_print_profile`` renders a human report; this module renders the same
data as one JSON object per request, appended as JSON Lines. The pretty print
is for eyeballs and is not a stable interface — the wan22 benchmark already
regex-parses it, which is the demand signal this module answers.

Consumers (validation harnesses, the simulator's calibration step) read the
JSONL and get exactly what the server measured: the per-request stage
timeline, per-(node, graph_walk) timings including true GPU time, and
per-edge tensor-transfer volumes.

Timing conventions, so a consumer does not have to rediscover them:

* All timestamps are raw ``time.perf_counter()`` seconds. They are only
  comparable within one host (mstar is single-host today). ``timing_rel_ms``
  gives the same checkpoints as milliseconds relative to ``recv_time``,
  which is what a report usually wants.
* ``forward_time`` is a CPU launch/enqueue span, NOT GPU time, and under
  speculative scheduling ``postprocess_time`` overlaps the next step — the
  phases are not additive. ``gpu_time`` is the honest per-step GPU-busy
  measurement (CUDA event pair); prefer it.
* ``prepare/plan/launch/sample`` decompose the engine's CPU work per step.
"""

import dataclasses
import json
import os
import threading
from typing import Any

from mstar.profile.format import RequestProfile

# Bump when the emitted object's shape changes incompatibly.
SCHEMA_VERSION = 1

# One lock per process: _finalize_profile runs on whichever thread popped the
# request, and two concurrent appends would interleave partial lines.
_write_lock = threading.Lock()


def profile_to_dict(prof: RequestProfile) -> dict[str, Any]:
    """Render one request profile as a plain JSON-able dict."""
    timing = dataclasses.asdict(prof.timing)

    # Relative-to-arrival milliseconds: the form nearly every consumer wants,
    # computed here so each of them doesn't re-derive it (and get the None
    # handling subtly wrong).
    recv = prof.timing.recv_time
    timing_rel_ms: dict[str, float] = {}
    if recv is not None:
        for name, value in timing.items():
            if value is not None:
                timing_rel_ms[name.replace("_time", "_ms")] = (value - recv) * 1e3

    return {
        "schema_version": SCHEMA_VERSION,
        "rid": prof.rid,
        "timing": timing,
        "timing_rel_ms": timing_rel_ms,
        "graph_timings": [dataclasses.asdict(g) for g in prof.graph_timings],
        "rx_info": [dataclasses.asdict(r) for r in prof.rx_info],
        "tx_info": [dataclasses.asdict(t) for t in prof.tx_info],
        "inputs": [dataclasses.asdict(i) for i in prof.inputs],
        "outputs": [dataclasses.asdict(o) for o in prof.outputs],
    }


def append_profile_json(prof: RequestProfile, path: str) -> None:
    """Append ``prof`` to ``path`` as one JSON line.

    Creates parent directories on demand. Serialization happens outside the
    lock so a slow encode doesn't stall other completing requests; only the
    append is serialized.
    """
    line = json.dumps(profile_to_dict(prof), separators=(",", ":"))
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _write_lock:
        with open(path, "a") as fh:
            fh.write(line + "\n")


def read_profiles_json(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file written by :func:`append_profile_json`.

    Blank lines are skipped. A truncated final line (the writer was killed
    mid-append) is dropped rather than raising, so a partial capture is still
    usable — the validation harnesses treat profiles as a sample, not a ledger.
    """
    out: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
