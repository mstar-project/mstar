"""SCAFFOLD: end-to-end output parity for M*-new (integrated build: native encoders
+ GPU mel + GPU image preprocess) vs a reference stack (HF transformers and/or
vLLM-Omni), under GREEDY decoding.

STATUS: skipped. This module is structure-only — it runs in the GPU phase once the
unified M* entrypoint (task T3) and a reference server are up. It does NOT launch
servers; it talks to already-running OpenAI-compatible endpoints (M* serves one;
vLLM-Omni / an HF reference serves the other), or compares against pre-captured
reference token sequences.

Why greedy + fixed seed: with ``temperature=0`` decoding is argmax and therefore
deterministic, so two numerically-equivalent stacks must emit the SAME token id
sequence for the same prompt. That makes token-sequence equality (not just a fuzzy
text/cosine match) the right, strict acceptance criterion for the TEXT path. Small
late divergence is tolerated via a longest-common-prefix bar (numerics can flip a
single argmax deep in a long generation); an early divergence is a real parity bug.

For the SPEECH/audio path, exact codec-token equality is the strict form; a looser
fallback compares decoded-audio similarity. Both are scaffolded below and skipped.

GPU-phase TODO (wire these up, then drop the skip):
  1. Bring up M*-new:    ``mstar serve qwen3_omni`` (SHM transport) -> MSTAR_OMNI_BASE_URL.
  2. Bring up reference: vLLM-Omni (text/I2T/S2T) or an HF generate harness ->
     OMNI_REFERENCE_BASE_URL, OR capture reference token ids offline into a JSON
     (OMNI_REFERENCE_TOKENS) keyed by prompt id.
  3. Pin BOTH stacks to the SAME tokenizer snapshot (see
     benchmark/HANDOFF_qwen3_omni_serving.md: tokenizer.json is cross-venv poison).
  4. Set temperature=0, a fixed seed, identical max_tokens, identical prompt &
     chat template, and request token ids (logprobs / echo) from both.
  5. Confirm acceptance bars below against the real measured divergence and tighten.
"""
import json
import os

import pytest

# Whole module is a GPU-phase scaffold; remove this skip once the endpoints exist.
pytestmark = pytest.mark.skip(
    reason="GPU-phase scaffold: needs M* + reference servers / captured tokens (see module TODO)"
)

MSTAR_BASE_URL = os.environ.get("MSTAR_OMNI_BASE_URL", "http://localhost:8000/v1")
REFERENCE_BASE_URL = os.environ.get("OMNI_REFERENCE_BASE_URL")        # vLLM-Omni / HF server
REFERENCE_TOKENS = os.environ.get("OMNI_REFERENCE_TOKENS")           # path to captured token JSON
MODEL = os.environ.get("MSTAR_OMNI_MODEL", "qwen3_omni")
SEED = int(os.environ.get("MSTAR_OMNI_SEED", "1234"))
MAX_TOKENS = int(os.environ.get("MSTAR_OMNI_MAX_TOKENS", "128"))

# Acceptance bars (confirm/tighten against measured divergence in the GPU phase).
MIN_PREFIX_RATIO = 1.0       # text: require an EXACT match by default (greedy => deterministic)
AUDIO_TOKEN_PREFIX_RATIO = 1.0
AUDIO_COS_MIN = 0.99         # fallback if comparing decoded audio instead of codec tokens

# Text prompts exercised on the text path (no audio in -> text out).
TEXT_PROMPTS = {
    "fun_fact": "Give me one fun fact about octopuses.",
    "count": "Count from one to ten.",
    "explain": "In two sentences, explain what a GPU is.",
}


# --------------------------------------------------------------------------- #
# helpers (filled in / exercised during the GPU phase)
# --------------------------------------------------------------------------- #
def _greedy_token_ids(base_url, prompt, *, max_tokens=MAX_TOKENS, seed=SEED):
    """Return the greedy (temperature=0) generated token-id list from an
    OpenAI-compatible endpoint.

    TODO(GPU phase): the exact way to recover token ids depends on the server:
      - completions API with ``logprobs`` -> token strings -> re-encode, OR
      - a server flag that returns raw output token ids, OR
      - prompt_logprobs / echo. Use whichever M* and the reference both expose,
        and make sure BOTH go through the identical tokenizer.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="none")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=seed,
        max_tokens=max_tokens,
        logprobs=True,
    )
    choice = resp.choices[0]
    # Prefer real ids if the server returns them; else fall back to logprob tokens.
    toks = getattr(choice, "token_ids", None)
    if toks is not None:
        return list(toks)
    lp = choice.logprobs.content if choice.logprobs else []
    return [t.token for t in lp]  # token strings; compared structurally if ids absent


def _load_reference_tokens(prompt_id):
    """Reference token ids for ``prompt_id`` from a captured JSON, or None."""
    if not REFERENCE_TOKENS or not os.path.isfile(REFERENCE_TOKENS):
        return None
    with open(REFERENCE_TOKENS) as f:
        return json.load(f).get(prompt_id)


def _longest_common_prefix(a, b):
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _assert_token_parity(got, ref, *, min_ratio, label):
    assert got and ref, f"{label}: empty token sequence (got={len(got)} ref={len(ref)})"
    lcp = _longest_common_prefix(got, ref)
    ratio = lcp / min(len(got), len(ref))
    assert ratio >= min_ratio, (
        f"{label}: greedy token sequences diverge at position {lcp} "
        f"(prefix ratio {ratio:.3f} < {min_ratio}); "
        f"got[:{lcp + 1}]={got[:lcp + 1]} ref[:{lcp + 1}]={ref[:lcp + 1]}")


# --------------------------------------------------------------------------- #
# text path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prompt_id", list(TEXT_PROMPTS), ids=list(TEXT_PROMPTS))
def test_text_greedy_token_parity(prompt_id):
    """M*-new vs reference: identical greedy token ids on the text path."""
    prompt = TEXT_PROMPTS[prompt_id]
    got = _greedy_token_ids(MSTAR_BASE_URL, prompt)

    ref = _load_reference_tokens(prompt_id)
    if ref is None:
        if not REFERENCE_BASE_URL:
            pytest.skip("no reference: set OMNI_REFERENCE_BASE_URL or OMNI_REFERENCE_TOKENS")
        ref = _greedy_token_ids(REFERENCE_BASE_URL, prompt)

    _assert_token_parity(got, ref, min_ratio=MIN_PREFIX_RATIO, label=f"text[{prompt_id}]")


# --------------------------------------------------------------------------- #
# speech path (M*-only generation -> codec tokens / decoded audio)
# --------------------------------------------------------------------------- #
def test_speech_greedy_codec_token_parity():
    """SCAFFOLD: greedy parity for the speech path (codec/talker token ids).

    Strict form: codec-token id sequence equality vs a captured reference.
    Fallback: decoded-audio cosine >= AUDIO_COS_MIN (use only if codec tokens are
    not exposed). vLLM-Omni has no image->speech path, so the reference here is a
    captured M* baseline (pre-integration) or an HF talker reference.
    """
    ref = _load_reference_tokens("speech_baseline")
    if ref is None:
        pytest.skip("no captured speech reference tokens (OMNI_REFERENCE_TOKENS['speech_baseline'])")
    # TODO(GPU phase): request talker/codec token ids from M*-new for the fixed
    # prompt+seed, then:
    #   _assert_token_parity(got_codec, ref, min_ratio=AUDIO_TOKEN_PREFIX_RATIO,
    #                        label="speech.codec")
    # or, if only audio is available, compare waveforms with AUDIO_COS_MIN.
    pytest.skip("speech path generation not wired yet (GPU phase)")
