# PersonaPlex validation

Checks M*'s PersonaPlex against NVIDIA's reference implementation and as served. M*
serves PersonaPlex as an offline replay: a prerecorded user clip in, an agent track of the
same length out, streamed back in chunks while the model runs. Nothing here exercises live
duplex input (audio fed into a running request).

NVIDIA's numbers come from running its reference model code (`NVIDIA/personaplex` at
`3428dfd95309a7f3c84fd93259ded0f810d1ff91`) in the same process and PyTorch runtime as M*.

Needs one CUDA GPU that holds two copies of the 7B LM for parity (an 80 GB H100 is enough),
the `personaplex` extra, the `mstar_rust` extension (`docs/installation.rst`) and access to
the gated checkpoint. `ASSETS` is the checkpoint snapshot, `VOICES` the extracted voice
presets and `SOURCE` a clean checkout of NVIDIA's code at the commit above. A manifest lists
clips as `[{"id": "clip", "path": "clip.wav", "sha256": "..."}]`.

## Parity against NVIDIA

M*'s LM step by step against NVIDIA's `LMGen`, greedy, with `dense` attention or the served
path (`resources`: paged KV and FlashInfer). `compare` recomputes a run's token differences
from its saved arrays and exits 1 if any token differs. `judge` writes Whisper transcripts of
each side's audio, for information; it is not a quality score.

```bash
python -m benchmark.personaplex.parity --input clip.wav --frames 2999 --source "$SOURCE" \
  --assets "$ASSETS" --voices "$VOICES" --backend resources --output runs/parity
python -m benchmark.personaplex.compare runs/parity
python -m benchmark.personaplex.judge --asr-model "$WHISPER" --output runs/judge runs/parity
```

## Batched against alone

Direct LM calls with paged KV and FlashInfer: each clip alone, then batched with staggered
starts and two voices. It compares emitted tokens, with no HTTP or worker scheduler.

```bash
python -m benchmark.personaplex.batch --inputs clip.wav --assets "$ASSETS" --voices "$VOICES" \
  --output runs/batch
```

## Serving

`serving` starts a real server from a settings file with the keys `server_config`, `cache`
and optionally `port` and `extension` (a directory holding a built `mstar_rust*.so`). It
sends each clip alone, then concurrent requests, a timeout, a disconnect after the first
audio, and malformed inputs (`--mode lifecycle` only), and checks that the server keeps
serving. With `--max-batch-size 1`, concurrent outputs must equal isolated ones byte for byte;
a batched step rounds differently, so otherwise agreement is only recorded. `--mode soak` adds
20 minutes of mixed requests. These checks see what a client sees, so they do not show that a
cancelled request's state was released.

`warm` runs `benchmark.runner` per clip at concurrency 1, 2 and 4: 3 warm-up waves and 10
measured requests, so its percentiles are descriptive only.

```bash
python -m benchmark.personaplex.serving --settings settings.json --manifest inputs.json \
  --max-batch-size 1 --output runs/lifecycle
python -m benchmark.personaplex.warm --settings settings.json --manifest inputs.json --output runs/warm
```

## Limits

- Exact-token parity on the served path is not guaranteed: FlashInfer's rounding differs from
  PyTorch's SDPA, so near-tied choices can flip. The comparison measures this; it does not
  decide whether it is acceptable.
- Audio quality is not scored: `judge` compares what the two agents said, not how they sound.
