"""Example oracle override (a TEMPLATE — copy to <model>.reference.py and fill in).

This is the ONE model-specific Python file a user writes. If present next to
<model>.toml, it replaces the rendered templates/reference.py.tmpl. The generic
checker.py / benchmark.py drive these four hooks, so they never change per model.

- sample_input(walk): the fixed, deterministic probe input for a walk.
- to_request(walk, sample): (method, path, json_body) to call the mstar server.
- from_response(walk, response): parse the candidate output into a comparable
  artifact (token-id list / logits array / audio samples / image|video array).
- oracle(walk, sample): the reference ground truth (same artifact type), used
  only in strict mode. Weights are at /model (docker) or the HF id.

Real <model>.reference.py files are gitignored; only this example is tracked.
"""

from __future__ import annotations

from typing import Any

SERVED = "mymodel"
_MODEL: Any = None


def sample_input(walk: str) -> dict:
    return {"prompt": "a fixed probe prompt", "seed": 0}


def to_request(walk: str, s: dict) -> tuple[str, str, dict]:
    # Drive exact-match walks greedily (temperature 0 + fixed seed).
    return "POST", "/v1/chat/completions", {
        "model": SERVED,
        "messages": [{"role": "user", "content": s["prompt"]}],
        "temperature": 0,
        "seed": s["seed"],
    }


def from_response(walk: str, response) -> Any:
    return response.json()["choices"][0]["message"]["content"]


# --- strict-mode oracle (wire up once you capture/derive a reference) --------
def load() -> None:
    global _MODEL
    if _MODEL is not None:
        return
    # e.g. AutoModelForCausalLM.from_pretrained("/model" or HF id).eval().cuda()
    raise NotImplementedError("reference.load: construct the upstream model")


def oracle(walk: str, s: dict) -> Any:
    load()
    raise NotImplementedError("reference.oracle: return the reference output")
