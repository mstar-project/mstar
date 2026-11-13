"""Live checks on a running Phi-4-multimodal server: greedy parity with the reference
recordings and the serving contract. Exits nonzero on any failure.

Greedy tokens must follow the reference until a step where its top-2 logits are
within ``TIE`` of each other: there the kernels decide, as they do between the
reference's own runs. The reference is ``record.py``'s ``sdpa`` run, and for prompts
with an image its run with the vision tower on FA2's padded-patch semantics, which the
node follows (``sdpa+fa2vision``). The other recorded routes are reported beside it.
The negative control: on four prompts the modality LoRAs serve (``CONTROLS``, both
LoRAs and both audio projections), the output must leave the reference run without
the LoRAs (``sdpa+nolora``) within ``CONTROL_STEPS`` tokens.

The contract: streamed text equals non-streamed; two identical requests return the
same text; ``max_tokens`` of 1 and 2 return exactly that many tokens; a malformed
field is a 400 and the server answers normally afterwards; a stream cancelled
mid-reply leaves it healthy.

usage: PHI4MM_ORACLE=<recordings> python test/phi4mm/served_checks.py [base_url]
"""

from __future__ import annotations

import base64
import glob
import json
import os
import sys

import requests
import torch

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ORACLE = os.environ["PHI4MM_ORACLE"]
MEDIA = os.path.join(ORACLE, "media")
REFS = {"text": ("sdpa", "eager"), "image": ("sdpa_fa2v", "eager_fa2v", "sdpa_fa2v_sdpa")}
# two bf16 steps at the logits' magnitude (~60)
TIE = 0.5
NO_LORA = "sdpa_nolora"
CONTROLS = ("audio_short", "audio_long", "image_audio", "two_images")
CONTROL_STEPS = 4

CASES = {
    "text": [("text", "What is the capital of France? Answer in one sentence.")],
    "text_long": [("text", "Repeat after me: the quick brown fox jumps over the lazy dog. " * 400 + "Now summarize.")],
    "image_bagel": [("image", "bagel.png"), ("text", "Describe this image in two sentences.")],
    "image_food_small": [("image", "food_small.png"), ("text", "What food is this?")],
    "image_food_wide": [("image", "food_wide.png"), ("text", "Describe the image in detail.")],
    "two_images": [("image", "bagel.png"), ("image", "food_small.png"), ("text", "Compare the two images.")],
    "audio_short": [("audio", "speech_short.wav"), ("text", "Transcribe the audio clip into text.")],
    "audio_long": [("audio", "speech_long.wav"), ("text", "Transcribe the audio clip into text.")],
    "image_audio": [("image", "bagel.png"), ("audio", "speech_short.wav"),
                    ("text", "Answer the spoken words, then describe the image.")],
}


def content(parts) -> list[dict]:
    out = []
    for kind, value in parts:
        if kind == "text":
            out.append({"type": "text", "text": value})
            continue
        data = base64.b64encode(open(os.path.join(MEDIA, value), "rb").read()).decode()
        if kind == "image":
            out.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        else:
            out.append({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})
    return out


def chat(parts, stream=False, **extra) -> tuple[int, str, list[int]]:
    body = {"model": "phi4mm", "messages": [{"role": "user", "content": content(parts)}], "temperature": 0,
            "max_tokens": 64, "stream": stream, "prefix_cache": False, "return_token_ids": True, **extra}
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, stream=stream, timeout=600)
    if not r.ok:
        return r.status_code, r.text, []
    if not stream:
        return r.status_code, r.json()["choices"][0]["message"]["content"], []
    text, ids = "", []
    for line in r.iter_lines():
        if line.startswith(b"data: ") and line != b"data: [DONE]":
            chunk = json.loads(line[6:])
            for choice in chunk.get("choices", []):
                text += choice["delta"].get("content") or ""
            ids += chunk.get("token_ids") or []
    return r.status_code, text, ids


def first_diff(a, b) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> int:
    from transformers import GPT2TokenizerFast

    hub = os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"))
    snapshot = glob.glob(f"{hub}/models--microsoft--Phi-4-multimodal-instruct/snapshots/*/")[0]
    tok = GPT2TokenizerFast.from_pretrained(snapshot)
    fails = []
    for name, parts in CASES.items():
        routes = REFS["image" if any(kind == "image" for kind, _ in parts) else "text"]
        records = [torch.load(os.path.join(ORACLE, r, f"{name}.pt"), weights_only=False) for r in routes]
        _, text, _ = chat(parts)
        ids = tok.encode(text)
        ref, margins = records[0]["greedy"].tolist(), records[0]["greedy_margin"].tolist()
        n = first_diff(ids, ref)
        others = [first_diff(ids, r["greedy"].tolist()) for r in records[1:]]
        # the served text drops the stop token the reference's ids end on
        end = len(tok.encode(records[0]["greedy_text"]))
        tie = n < end and margins[n] <= TIE
        print(f"{name}: follows the reference for {n} of {end} tokens"
              + (f" (step {n} margin {margins[n]:.3f}: a tie)" if tie else "") + f"; other routes {others}")
        if n < end and not tie:
            fails.append(f"{name} diverges at step {n}, margin {margins[n]:.3f}")
        if name in CONTROLS:
            base = torch.load(os.path.join(ORACLE, NO_LORA, f"{name}.pt"), weights_only=False)["greedy"].tolist()
            left = first_diff(ids, base)
            print(f"  without the LoRA, the reference leaves ours at step {left}")
            if left > CONTROL_STEPS:
                fails.append(f"{name}: the no-LoRA control follows ours for {left} tokens")
    parts = CASES["image_audio"]
    _, a, _ = chat(parts)
    _, b, _ = chat(parts)
    _, s, _ = chat(parts, stream=True)
    if a != b:
        fails.append("two identical requests differ")
    if s != a:
        fails.append("streamed text differs from non-streamed")
    for n in (1, 2):
        _, t, _ = chat(parts, max_tokens=n)
        if len(tok.encode(t)) != n:
            fails.append(f"max_tokens={n} gave {len(tok.encode(t))} tokens")
    status, _, _ = chat(parts, temperature="hot")
    if status != 400:
        fails.append(f"a malformed field gave {status}")
    body = {"model": "phi4mm", "messages": [{"role": "user", "content": content(CASES["audio_long"])}],
            "max_tokens": 400, "stream": True}
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, stream=True, timeout=600)
    next(r.iter_lines())
    r.close()
    status, after, _ = chat(parts)
    if status != 200 or after != a:
        fails.append("the server changed after a 400 and a cancelled stream")
    print("FAIL" if fails else "OK", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
