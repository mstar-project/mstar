# PersonaPlex validation and benchmarks

Checks M*'s PersonaPlex against NVIDIA's reference implementation and measures
it as served. M* serves PersonaPlex as an offline replay: a prerecorded user
clip in, an agent track of the same length out. Concurrent requests share LM
steps (`model_kwargs.max_batch_size`). Nothing in this directory exercises
live duplex input or streamed audio.

NVIDIA's numbers come from running its reference model code
(`NVIDIA/personaplex` at `3428dfd95309a7f3c84fd93259ded0f810d1ff91`) in the
same process and PyTorch runtime as M*, not from NVIDIA's web server.

## What it checks

`python -m benchmark.personaplex.suite` runs these stages in order, each in its
own subprocess with a timeout, and records every command, log and result:

| Stage | What runs | Passes when |
|---|---|---|
| `parity-<clip>-dense` | M*'s LM with dense attention, step by step against NVIDIA's `LMGen`, greedy, from each side's own Mimi codes; then a negative control, M* rotating RoPE over split halves instead of adjacent pairs | Every text and audio token matches, and the negative control is reported as a mismatch; float audio error is reported |
| `parity-<clip>-resources` | The same, on the served path (paged KV + FlashInfer); then NVIDIA again under PyTorch's math SDPA kernel | As above, without the negative control. Near-tied token choices can flip here; the first divergence is recorded with its logit margin, next to how far NVIDIA moves against itself under the other kernel |
| `batch-resources` | Each clip alone, then 2 and 4 requests batched with staggered starts and two voices, on the served path | Every emitted token matches the request run alone; otherwise the first differing choice is recorded with both runs' margins. LM step time per batch size is reported |
| `judge` | Whisper transcripts of each served-path parity stage's input, NVIDIA and M* audio | Informational: the agents' word error rate, not a quality score |
| `lifecycle-python`, `lifecycle-rust` | A real server on each graph runtime, one request per LM step (`max_batch_size` 1): each clip alone, then 1, 2 and 4 concurrent requests, cancellation mid-request, then malformed, empty, over-length and invalid-option requests, each followed by a normal one | Concurrent and post-cancellation outputs are byte-identical to the same request run alone, bad input gets a 4xx, and no agent frame is left in the worker's tensor store after the decoder runs |
| `soak-rust` | 20 minutes of mixed short/long requests with periodic cancellations, batched | Every response is well formed, the request tensor store drains, and idle allocated GPU memory does not grow; agreement with the isolated control is recorded |
| `warm-rust` | `benchmark.runner` per clip at concurrency 1, 2 and 4, batched: 3 warm-up and 10 measured requests | Every measured request completes; a compile detected during measurement invalidates the cell, which is retried once |

The soak and benchmark run only if `lifecycle-rust` passed. A failing parity
stage does not stop the serving stages.

`python -m benchmark.personaplex.timing` is separate: it times NVIDIA's
`offline.py` per-frame loop (with and without its CUDA graphs) against M*'s LM
node directly, per 80 ms frame, without HTTP.

## Settings

The suite reads a JSON settings file:

| Key | Meaning |
|---|---|
| `expected_head` | The M* commit to test; the checkout must be at it with no tracked changes |
| `source` | A clean checkout of `NVIDIA/personaplex` at the commit above |
| `assets` | The checkpoint snapshot holding `model.safetensors`, the Mimi file and `tokenizer_spm_32k_3.model` |
| `voices` | Extracted voice presets (from the repo's `voices.tgz`); `NATF2.pt` and `NATM1.pt` are used |
| `cache`, `server_config` | The Hugging Face cache and the server YAML the serving stages launch with |
| `existing_clip` | A mono 24 kHz PCM16 clip of exactly 222 frames; `test/qwen3-omni/audio.wav` is one |
| `english_clip` | Optional second clip; without it its two parity stages are reported BLOCKED |
| `extension` | Optional directory holding a built `mstar_rust*.so`; otherwise the installed module is used |
| `reference_site` | Optional directory with the extra packages NVIDIA's code imports (`sphn`, `tqdm`), added after M*'s own torch |
| `asr_model` | Optional local Whisper checkpoint (e.g. `openai/whisper-large-v3-turbo`) for `judge`; without it `judge` is BLOCKED |
| `port` | Server port, default 8000 |

## Reproduce

Needs one CUDA GPU that holds two copies of the 7B LM (parity loads NVIDIA's
and M*'s side by side; an 80 GB H100 is enough), the `personaplex` extra, the
`mstar_rust` extension (`docs/installation.rst`) and access to the gated
checkpoint.

```bash
python -m benchmark.personaplex.suite preflight --settings settings.json --output runs/personaplex
python -m benchmark.personaplex.suite full --settings settings.json --output runs/personaplex
python -m benchmark.personaplex.suite report --output runs/personaplex
```

`preflight` checks identities and input hashes without a GPU. `smoke` is a
quick end-to-end check on a 16-frame clip: served-path parity with both
controls, a batch of 2, the judge, the Rust lifecycle and one benchmark
request. `resume` continues a run in the same output
directory, reusing finished stages if nothing it depends on has changed;
`--retry-failed` reruns failed ones. `--deadline-seconds N` stops launching
stages shortly before N seconds have passed, for runs inside a fixed job
length. A full run took 42 minutes of stage time on one H100.

The benchmark cells can also be run by hand against a running server. The
manifest lists clips as `[{"id": "clip", "path": "clip.wav", "sha256": "..."}]`:

```bash
MSTAR_SHM_ARENA=1 python -m benchmark.runner --url http://127.0.0.1:8000 --model personaplex \
  --inference-system ours --dataset local_audio --audio-manifest inputs.json \
  --request-type audio_to_speech --profiling-type closed_loop \
  --max-concurrency 1 --num-warmup 3 --num-requests 10 --output-dir results/clip-c1
```

## Outputs

`results.json` holds each stage's status: PASS, FAIL, SKIP, BLOCKED or NOT RUN.
`REPORT.rst` summarises them, `numerical.csv` and `benchmarks.csv` tabulate the
parity metrics and benchmark cells, and `listen.html` embeds the input,
NVIDIA's output and M*'s output for listening. The suite exits nonzero unless
every stage passed or was skipped.

`python -m benchmark.personaplex.compare <parity-dir>` recomputes a parity
stage's token differences from its saved arrays and exits 1 if any token
differs.

`python -m benchmark.personaplex.sessions` runs N concurrent streamed requests
against a running server on a clip of user speech followed by silence
(`test/personaplex/duplex_clip.py` makes one). It reports per session:
time to first audio; response onset on the audio timeline (signed, with
speech already running at the user's speech end reported separately); and
playback underruns with a fixed buffer. It exits 1 if any session fails or
returns the wrong length of audio. A discarded warm-up wave runs first. It measures streamed delivery;
`benchmark.runner` remains the throughput harness.

## Limits

- Exact-token parity on the served path is not guaranteed: FlashInfer's
  rounding differs from PyTorch's SDPA, so near-tied choices can flip. The comparison
  measures this; it does not decide whether it is acceptable.
- Audio quality is not scored: `judge` compares what the two agents said, not
  how they sound; listen to `listen.html`.
- A batched step rounds differently from a request run alone, which is why
  only the batch-size-1 lifecycle compares outputs byte for byte.
- Ten requests per benchmark cell make the percentiles descriptive only.
- The soak covers 20 minutes of one request mix; it is not proof against every
  leak.
