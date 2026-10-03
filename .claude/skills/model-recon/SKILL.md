---
name: model-recon
description: "Read-only reconnaissance of a model's reference implementation (GitHub repo, vLLM/SGLang model file, paper, checkpoint config) to produce a ramp-up document: architecture diagram with file:line citations, runtime trace, state and shape ledger, serving precedents, an unanswered design worksheet, and a reading order. Use when: the user says /model-recon, shares reference code for a model they intend to port, or asks how a model works before designing an mstar port. Decides nothing."
disable-model-invocation: true
---

# model-recon

Purpose: let the user ramp up on a model from its reference implementation and answer the
port design questions themselves. This skill reports what the code does and where. It does not
design, recommend, or choose.

## Ground rules

1. Read-only. No edits to any repo. No "I recommend", "should", or "better". If the user asks for
   options, give trade-offs sourced from the reference, still without a pick.
2. Every statement is a citation (`path:line`) or is prefixed `inference:`. Unknowns are written as
   `unknown: <what>; look in <file>` rather than guessed.
3. File sweeps run in Sonnet subagents (`model: "sonnet"`, self-contained prompts, one component or
   one phase each). The parent reads every cited line before it goes in the document.
4. Depth default: backbone and inference loop first; encoders, decoders, tokenizers, and training-only
   code are summarized in one line each and traced only on request.
5. Artifact lives outside the target repo by default: `<repo-parent>/_recon/<model>/recon.md`,
   unless the user names a location. Never inside a git-excluded docs folder.

## Inputs

Ask for, or locate from what the user gave: reference repo path or URL; vLLM and/or SGLang model
files if a port exists; paper or README; checkpoint `config.json`. Clone URLs into `_recon/<model>/src/`.
Missing inputs are listed at the top of the artifact, not substituted.

## Phases

Pause for the user after phase 2 and after phase 4. Show the artifact so far and ask which
components to trace deeper; do not ask design questions.

### 1. Inventory
Locate: model class, config class, weight loader and key mapping, preprocessing, inference entry
point (generate / sample / rollout), streaming or output path, serving integration. Output: file map,
one line per file, what it holds, approximate size.

### 2. Static architecture
From config and model class: components, tensor flow with shapes and dtypes, attention kinds and
masks, positional scheme, conditioning inputs, caches and buffers, dtype boundaries. Output: ASCII
block diagram, every box labelled with `path:line`; a parameter table (name, value, source line).

### 3. Runtime trace
Walk one request through the entry point. For each operation record: once-per-request or
per-iteration; inputs and outputs with shapes; what is carried between iterations; how termination is
decided; how output leaves (whole artifact, chunks, tokens). Flag every request-time operation that
touches shapes or the host: resizes, RNG draws, `.item()`/`.cpu()` syncs, dtype casts, padding.

### 4. State and shape ledger
Table: persistent tensor or buffer, lifetime (global / per request / per iteration), shape, which
dims vary at request time and over what range, who allocates, who frees. Followed by a flat list of all
request-time shape variation (input resolution, sequence length, batch, conditioning length).

### 5. Serving precedents (only if a vLLM/SGLang/other serving port exists)
How the port registered the model; cache configuration; attention backend; multimodal processor;
batching and scheduling hooks; any special-case code added for this model. Reported as what they
did, with citations, plus a one-line note of anything they explicitly did not support.

### 6. Design worksheet
One block per decision an mstar port must make. Fields, in this order:
`reference does:` (citations) / `facts that bear on it:` / `unknown:`. No answer field.
Decisions to cover: units of once-per-request vs repeated work; persistent state and who owns it;
output lifecycle and how it streams; request-time shape variation; termination and abort; batching
across requests; numerics that a port could change (resizes, RNG, casts); external dependencies.

### 7. Reading order
Ordered list of functions to read, each with: path:line, one sentence on why it matters for the port,
estimated minutes. First pass targets about one hour total.

## Artifact layout

`recon.md` sections in this order: Inputs and gaps; File map; Architecture diagram and parameters;
Runtime trace; State and shape ledger; Serving precedents; Design worksheet; Reading order.

## Hand-off

The filled worksheet (the user's answers) plus the ledger are the input to `/mstar-model-port`.
