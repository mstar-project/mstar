# HTTP end-to-end runner

Starts a real M* server for each model profile, sends public HTTP requests
(OpenAI chat, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/generate`,
streamed `/generate`), and checks what comes back. It covers the path component
tests stub out: request parsing, preprocessing, scheduling, tensor transport,
the model, postprocessing and response delivery.

**Needs CUDA GPUs and the model weights.** It is a developer command, not part
of CI; the validators and matrix logic it relies on are unit tested on CPU in
`test/modular/test_e2e_validators.py`.

It checks function, not latency or quality: one request per case, with checks a
broken output fails (an answer, a dominant color, a transcript, a frame count, a
tensor width). It records timings and peak VRAM but asserts on neither. Text
checks are lenient on purpose: the expected word anywhere in the answer passes,
so a hedged or negated answer ("not red") passes too; a decimal ("4.5") does not
pass for "4".

## Running

```bash
python test/e2e/run.py --list            # the coverage matrix for this machine; starts nothing
python test/e2e/run.py --tier smoke      # one fast case per family (not omni), eager servers
python test/e2e/run.py                   # every case whose profile fits this machine
python test/e2e/run.py --models bagel qwen3_5 --offline
```

Each run writes `report.json`, one `server.log` per model and checkout, and
every checked output (images, video, audio, tensors, text) under `--out`
(default: `$TMPDIR/mstar-e2e/<timestamp>-<pid>/`). The report records the
commit and dirty state of each checkout, GPUs and driver, package versions, and
per model its status, config path and hash, `model_kwargs`, startup time, peak
VRAM, and per case the verdict, detail, timing, warnings and a hash of the
decoded output. The exit code is non-zero if any selected profile fails.

## Server lifecycle

Each server runs in its own process group. On exit, failure, Ctrl-C, SIGTERM
or SIGHUP the runner sends SIGINT to the API server, its graceful stop (it
shuts down its conductor, which stops the workers), and SIGKILLs the whole
group if it is not gone within 30 s. A second interrupt during that wait skips
straight to SIGKILL; the runner then stops. A run under `nohup` keeps ignoring
SIGHUP, so `nohup python test/e2e/run.py ... &` survives the terminal closing.
A model interrupted mid-run keeps its completed cases in the report.

If a process of the group survives SIGKILL, the profile fails with its pids
(`server_survivors` in the report) and the server is not restarted.

Ports: by default each server gets a free port, picked just before launch and
before the VRAM sampler starts, and picked again (up to 10 tries) if it reads as
busy. The runner only talks to a server that its own process group is listening
on (checked through `/proc` on Linux). Whether another process holds the port at
launch is decided by test-binding 127.0.0.1 as the server would, so an IPv6-only
listener on the same port does not count and a dual-stack one does. If another
process takes the port while the model loads, the server fails to bind and is
restarted once on a fresh port (`server_restarts` in the report); what `/proc`
shows of other processes never restarts it, since that can be stale by the time
it is read. With `--port`, a port held by another process fails the profile
instead. A server that exits on its own has its exit code recorded as
`server_exit_code` and fails the profile.

A runner killed outright (SIGKILL, the OOM killer) cannot tear down. On Linux
the API server gets SIGKILL when the runner dies (`PR_SET_PDEATHSIG`), and its
conductor and workers exit within seconds once their parent is gone. What can
remain: the scratch directory (`mstar-e2e-*` under `$TMPDIR`, or `/tmp` when
`$TMPDIR` is too long for socket paths), with sockets and uploads, and any
`/dev/shm` segments the conductor did not unlink, which the next server's
startup sweeps. Elsewhere, or if something survived, find the group with
`pgrep -af mstar.api_server.entrypoint` and stop it with `kill -INT -<pgid>`,
then `kill -KILL -<pgid>`.

## The coverage matrix

Each `ModelSpec` in `cases.py` is one deployment profile: a config under
`configs/` and the cases that exercise it. Every profile ends up in the report
as `passed`, `failed`, or `skipped` with the reason:

| Skip reason | Meaning |
|---|---|
| `needs N GPUs; M available` | the config's ranks run on more distinct GPUs (`rank_devices` can share one) than this machine has |
| `needs ~X GB per GPU` | the recorded peak VRAM, plus 10% headroom, exceeds the free memory |
| `missing package(s): ...` | a module the model needs is not importable (see below) |
| `its input needs missing package(s): ...` | the runner cannot build a case's input; per case under `skipped_cases`, and the profile is skipped if no case is left |
| `gated checkpoint: accept the license at ...` | the Hugging Face account cannot download the checkpoint |

If checking a gated checkpoint fails for another reason (offline mode, DNS, a
proxy), the profile runs anyway, so cached weights still load, and the report
carries the warning `could not check access: <repo> (<error>); running anyway`
under the model's `warnings`.

A profile that was selected and then runs out of memory is a **failure**, not a
skip: either the recorded peak is wrong or something regressed. Peak VRAM is
sampled while each server runs and reported, with a warning when it exceeds the
recorded figure by more than 10%.

The report also lists every config under `configs/` with no case (with the GPUs
it needs) and every registered model with no config, so gaps stay visible.

**Generate, then transcribe.** After every model has run, one Whisper server
(`whisper_large`) transcribes each speech output that passed its own checks, and
the transcript must contain at least 75% of the words the speech was asked to
say. That turns "not silent" into "says the right sentence". Skip it with
`--no-transcribe`; if Whisper's own profile cannot run here, the report says why.

**Between servers** the runner waits until GPU memory is back near its level
before the server started, so the next server neither starts short of memory
nor measures its peak against a stale baseline. Memory still held two minutes
after a server exits is reported as a warning.

Tiers: `smoke` runs only cases marked `smoke=True` (one fast case per family;
omni has none, since its only profile is a 30B model) with
`TORCH_COMPILE_DISABLE=1`; `full` runs every case with configs as written,
including any compilation and accelerator graph capture, which can add minutes to a
server's startup.

## Model requirements

- Optional extras: Kokoro needs `misaki`, `spacy` and `en_core_web_sm`;
  Chatterbox needs `pyloudnorm` and `resemble-perth` (multilingual adds
  `pykakasi` and `spacy-pkuseg`); Qwen3-TTS needs `qwen-tts`; Qwen3-Omni needs
  `qwen-omni-utils` and `flash-attn`; OmniVoice needs the git build described in
  `docs/installation.rst`. Install them without changing the installed `torch`,
  `torchcodec` or `transformers` versions. The exception is OmniVoice: its git
  build requires `transformers>=5.3`, so run it from a separate environment.
- Gated checkpoints (Orpheus, FLUX.2-klein 9B) need a logged-in Hugging Face
  account that has accepted the model's license.
- Weights download from the Hub on first use unless `--offline` is given. Some
  configs' warm-up requests also fetch `hf://` assets.
- The speech-recognition and voice-cloning cases use one LibriSpeech utterance
  from the public `hf-internal-testing/librispeech_asr_dummy` dataset, read with
  `pyarrow` (installed with `datasets`) and `soundfile` (the `audio` extra).
  Without them those cases are skipped.
- V-JEPA 2, Whisper and Higgs Audio decode their input on the server with
  `torchcodec`, which needs FFmpeg shared libraries on the loader path (a
  system FFmpeg, or `LD_LIBRARY_PATH`). Without them those cases fail, and so
  does every speech profile's "says the sentence" check, which transcribes with
  Whisper. The server environment is inherited from the runner.

## Comparing two checkouts

```bash
python test/e2e/run.py --compare-source ../mstar-main
```

Runs every selected case against both checkouts and additionally requires each
decoded output to be identical. Exact comparison is only meaningful when the
outputs are deterministic:

- the same GPU model, driver, CUDA, PyTorch and other library versions on both sides;
- fixed seeds (every case sets one) and no sampling where the API allows;
- no compilation or accelerator graph capture whose results vary between launches.
  Several configs enable both (for example `configs/flux2_klein.yaml` and
  `configs/cosmos3_nano.yaml`); pass eager configs with `--config MODEL=PATH`
  and use `--eager` to set `TORCH_COMPILE_DISABLE=1` for every server.

Without `--compare-source`, only the content checks apply, and none of this is
required.

## Adding a model

Add a `ModelSpec` to `MODELS` in `cases.py`:

- `config`: a config under `configs/`. Its `node_groups` decide how many GPUs
  the profile needs.
- One `Case` per input/output path worth covering. `build` returns the JSON body
  (`chat`, `speech`) or the form fields plus an optional `files` tuple
  (`transcribe`, `generate`, `generate_stream`). Generate inputs in code;
  `cases.py` has helpers for a square image, a robotics scene, short MP4 clips
  and the speech sample.
- An `Expect` that a wrong output would fail. `/generate` cases must name the
  output `modality`.
- A `must_differ` pair when an input could be silently ignored (an action, a
  control signal).
- `requires` for optional packages, `gated` for checkpoints behind a license,
  and `smoke=True` on one fast case if the family has none. A `Case` has its
  own `requires` for modules the runner needs to build its input (the
  LibriSpeech cases list `SPEECH_FIXTURE`).
- `peak_vram_gb` from the report of a passing run on the GPU you measured on.

`test_every_model_case_is_well_formed` checks the config exists and the case
names, routes and `must_differ` pairs are consistent.
