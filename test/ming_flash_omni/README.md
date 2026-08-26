# Ming-flash-omni-2.0 bring-up

Serving scripts for the native M\* port of
[`inclusionAI/Ming-flash-omni-2.0`](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0)
(`mstar/model/ming_flash_omni/`). The CPU-side contract (walks, partitions,
weight remap, component math) is covered by `test/modular/test_ming_flash_omni_*.py`;
these scripts are the GPU path.

## Graph

| Node | Engine | What it is |
|---|---|---|
| `Thinker` | `KV_CACHE`, TP | Ling-2.0 sparse MoE LLM (100B total / 6B active) — the understanding core |
| `vision_encoder` | `STATELESS` | Qwen3-MoE ViT + projector |
| `audio_encoder` | `STATELESS` | Whisper-style encoder + projector |
| `Talker` | `STATELESS` | CFM talker; the AudioVAE is wrapped **inside** the submodule, not a separate node |
| `ImageGen` | `STATELESS` | Z-Image DiT + ByT5/T5 condition stack (only when the checkpoint ships the imagegen sub-config) |

The Thinker→Talker bridge passes **detokenized text**, re-tokenized with the
talker's own tokenizer, so the talker is a near-standalone TTS partition fed by a
streaming connection (`MingFlashOmniModel.get_partition_topology`).

## Deploys

* `configs/ming_flash_omni.yaml` — full omni. Thinker TP=8 across 8×H100;
  encoders + talker colocate on rank 0.
* `configs/ming_flash_omni_thinker_only.yaml` — text out, no talker. Still TP=8:
  TP=4 OOMs at ~78.5/80 GB per rank during checkpoint streaming.

A deploy that omits a node simply cannot serve the walks that need it — those
walks are skipped at worker-graph division (`Model.get_worker_graphs`) rather
than failing the launch.

## Run

```bash
# 1. Server (8 GPUs, ~238 GB / 42 shards to stream — first launch is slow).
bash test/ming_flash_omni/launch_server.sh

# 2. Requests (in another shell; HOST/PORT default to 127.0.0.1:8000).
python test/ming_flash_omni/t2t_request.py --text "Who are you?"
python test/ming_flash_omni/a2t_request.py --audio test/qwen3-omni/audio.wav
python test/ming_flash_omni/t2s_request.py --output ming_speech.wav   # needs the Talker
```

Both scripts stream NDJSON from `POST /generate`. Audio comes back as headerless
int16 PCM with `sample_rate` / `num_channels` in each chunk's metadata;
`t2s_request.py` wraps it into a WAV.

## Benchmarking

`benchmark/base.py` registers the model as `ming_flash_omni`, so the standard
harness drives it against either backend:

```bash
python -m benchmark.runner --model ming_flash_omni --inference-system ours ...
```
