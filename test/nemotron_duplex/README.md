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
(cold, ~90 s); the talker is eager MaskGIT (~2 s/frame), so prefer short clips.

Send a request (the base VoiceChat-11B HF repo ships `turn_taking.wav`,
`interruptions.wav`, `tool_call.wav`):

```bash
cd test/nemotron_duplex
python duplex_request.py --audio /path/to/user.wav --output agent.wav
python duplex_request.py --audio /path/to/user.wav --text-only          # skip talker/codec
python duplex_request.py --audio /path/to/user.wav -n 3                  # 3 concurrent (batching)
```

## Validate against the oracle

`offline_inference` is the verified ground truth (100% text-token match vs the
NeMo reference). The engine's text must match it token-for-token; audio is
seeded-stochastic (compare duration / RMS, not samples). Run the oracle on a GPU
the server is **not** using:

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> python test/nemotron_duplex/oracle_compare.py --audio /path/to/user.wav
```
