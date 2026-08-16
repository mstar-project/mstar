# Nemotron-Duplex (VoiceChat-11B) — serving tests

Full-duplex speech-to-speech: user speech in → agent **text + agent speech** out,
through the M* engine (`conformer_encoder` → `nano_llm` → `eartts_talker` →
`audio_codec`).

## CPU unit tests (CI, no GPU / weights)

Run from the repo root:

```bash
pytest test/modular/test_nemotron_duplex_model.py \
       test/modular/test_nemotron_duplex_submodules.py \
       test/modular/test_streaming_loop_rearm.py
```

- `test_nemotron_duplex_model.py` — walk graphs, partitions, streaming topology,
  chunk policies, frame-0 seeding, forward-pass-args routing.
- `test_nemotron_duplex_submodules.py` — nano `check_stop` (EOS is not a stop),
  `_fuse_frame` empty-terminal-chunk guard, stateful codec (left-context + emit
  only new frames).
- `test_streaming_loop_rearm.py` — generic: a pure-streaming loop node is
  re-armed each iteration (the talker/codec loops rely on this).

## End-to-end serving (GPU + real weights)

```bash
cp test/nemotron_duplex/.sample.env test/nemotron_duplex/.env   # then edit it
```

Launch the server (run from the repo root):

```bash
bash test/nemotron_duplex/launch_server.sh          # single-GPU  (configs/nemotron_duplex.yaml)
bash test/nemotron_duplex/launch_server.sh disagg   # 3-GPU pipeline (nemotron_duplex_disagg.yaml)
```

Both layouts produce identical text + audio. The first request compiles kernels
(cold, ~90 s); once warm the pipeline runs ~0.1 s/frame.

### One-shot request (`duplex_request.py`)

With no `--audio`, a clean **user-only** input is prepared from the base
VoiceChat-11B demo `turn_taking.wav` — those demo wavs are 2-channel recordings
of the whole conversation (user left, agent right), so we take the user channel,
isolate the first user turn, and append silence for the reply.

```bash
cd test/nemotron_duplex
python duplex_request.py                                 # default user clip -> agent_out.wav
python duplex_request.py --audio /path/to/user.wav --output agent.wav
python duplex_request.py --text-only                     # skip talker/codec (fast)
python duplex_request.py -n 3                             # 3 concurrent (batching)
```

Your own `--audio` should be **user speech only** (mono), with trailing silence —
the model is frame-synchronous and replies in the frames *after* you stop talking.

### Duplex / streaming request (`duplex_stream_request.py`)

Sends the user utterance followed by a window of no-sound frames (the "keep
talking after the user stops" case) and consumes the reply as it streams back,
reporting the turn boundary and how much of the reply was actually voiced:

```bash
cd test/nemotron_duplex
python duplex_stream_request.py                          # 30 s reply window
python duplex_stream_request.py --silence 15 --output agent.wav
python duplex_stream_request.py --audio user.wav --silence 20
```

Behavior to expect / verify:
- The agent stays silent while the user talks and starts replying a few frames
  after; with a longer `--silence` it will emit **more reply text** (it does not
  wait for the user — it fills silence with plausible follow-up turns).
- The reply is voiced as **real, intelligible speech**. On a long-context input
  (a full ~53 s user channel) the agent produces a coherent multi-turn reply with
  ~13 s of voiced audio (ASR-verifiable, e.g. "Hello! How can I help you today? …
  a quick peanut butter cookie recipe … bake at three hundred fifty degrees …").
  Very dense late sentences in a long reply can still thin out (the talker emits
  ~1 code-frame per text token and the nano packs words densely) — a minor
  remaining model-reimplementation tail, not a serving-engine bug. (The transport
  is fixed-buffer input + streamed output; the engine has no incremental-audio-input
  path — that lives in the model's standalone `realtime_api.py` over `DuplexStream`.)

`--text-only` returns the agent text in seconds; full audio is slower.

### Browser UI (`webui.py`)

A tiny local web app to talk to the model from a browser (record from the mic or
upload a wav). It's a small proxy: the page posts audio to it, it resamples to
16 kHz mono, appends the silence reply window, forwards to the model server with
streaming, and streams the agent's text + audio back to play.

```bash
# with the model server already running (e.g. on :8019):
python test/nemotron_duplex/webui.py --mstar-url http://127.0.0.1:8019 --port 8500
# then open http://127.0.0.1:8500/
```

The mic needs a secure context (works on `http://localhost` / `127.0.0.1`; for
remote access put it behind HTTPS). The agent replies with full text and voiced
speech; on very dense late sentences of a long reply the audio can thin out (see
the streaming-request note above).

## Validate against the oracle

`offline_inference` is the verified ground truth (100% text-token match vs the
NeMo reference). The engine's text must match it token-for-token; audio is
seeded-stochastic (compare duration / RMS, not samples). Run the oracle on a GPU
the server is **not** using:

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> python test/nemotron_duplex/oracle_compare.py --audio /path/to/user.wav
```
