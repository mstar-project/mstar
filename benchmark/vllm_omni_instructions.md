# Setup vllm omni
# (check the latest guide here: https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/#prerequisites)

```
uv venv --python 3.12 --seed
source .venv/bin/activate

uv pip install vllm==0.19.0 --torch-backend=auto

git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

### Run vllm omni server
```
export HF_HOME=...
CUDA_VISIBLE_DEVICES=3 vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8000 --stage-configs-path vllm_omni/model_executor/stage_configs/bagel.yaml
```

### for qwen3-omni:
```
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml
```

### for ming-flash-omni-2.0:

The released `inclusionAI/Ming-flash-omni-2.0` ckpt (~238 GB / 42 shards)
does NOT load cleanly into vllm-omni's `MingFlashOmniForConditionalGeneration`
class as-is. Two patches are needed (one-time setup):

1. **Replace metadata files.** vllm-omni's model class uses
   `Qwen2VLImageProcessor` + `MingWhisperFeatureExtractor` (its own
   registered classes), while the inclusionAI snapshot declares the
   `BailingMM2*` processor variants via `auto_map` and `trust_remote_code`.
   Use `Jonathan1909/Ming-flash-omni-2.0`'s `preprocessor_config.json`,
   `config.json` (auto_map stripped), and `tokenizer*.json` instead.

2. **Replace the talker weights.** vllm-omni's `MingFlashOmniTalker` expects
   weights under `audio_vae.*` but the inclusionAI talker safetensors uses
   `audio.*` prefix. Jonathan1909 reshipped the talker with renamed weights
   (~1.5 GB).

Building a hybrid snapshot avoids re-downloading the 200+ GB thinker weights:

```bash
# 1. Make sure the inclusionAI thinker shards are cached
huggingface-cli download inclusionAI/Ming-flash-omni-2.0 \
    --include="model-*.safetensors" --include="model.safetensors.index.json"

# 2. Pull only Jonathan1909's metadata + talker (no thinker weights)
huggingface-cli download Jonathan1909/Ming-flash-omni-2.0 \
    --include="*.json" --include="*.py" --include="*.txt" --include="*.mvn" \
    --include="talker/**" \
    --cache-dir /dev/shm/hf-cache    # or any path with ~3 GB free

# 3. Stitch the two together
INCL=$(huggingface-cli scan-cache | grep inclusionAI/Ming-flash-omni-2.0 \
       | awk '{print $NF}')/snapshots/$(ls ~/.cache/huggingface/hub/models--inclusionAI--Ming-flash-omni-2.0/snapshots | head -1)
JONA=/dev/shm/hf-cache/models--Jonathan1909--Ming-flash-omni-2.0/snapshots/*
HYBRID=/dev/shm/ming-hybrid
mkdir -p $HYBRID
for f in $INCL/model-*.safetensors; do ln -s "$f" "$HYBRID/$(basename $f)"; done
for f in $JONA/*; do
    base=$(basename "$f")
    [ -L "$HYBRID/$base" ] && rm "$HYBRID/$base"
    ln -s "$f" "$HYBRID/$base"
done
```

Then serve and benchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /dev/shm/ming-hybrid \
  --omni --port 8091 --host 0.0.0.0 --trust-remote-code \
  --stage-configs-path /tmp/vllm-omni/vllm_omni/model_executor/stage_configs/ming_flash_omni.yaml

# Wait for "Application startup complete" then:
MODEL=ming_flash_omni INF_SYS=vllm_omni TASK=text_to_text \
  URL=http://0.0.0.0:8091 ./benchmark/run_benchmark.sh
```

NOTE: vllm-omni's `/v1/chat/completions` rejects unknown model ids, so the
client must send `"model": "/dev/shm/ming-hybrid"` (the served path), not
`"inclusionAI/Ming-flash-omni-2.0"`. Easiest is to monkey-patch
`MingFlashOmni.get_hf_url` before calling the benchmark runner:

```python
from benchmark.base import MingFlashOmni
MingFlashOmni.get_hf_url = lambda self: "/dev/shm/ming-hybrid"
```

Or pass `--served-model-name inclusionAI/Ming-flash-omni-2.0` to `vllm serve`
(untested; would also work in principle).

#### Modalities exercised on a local 4×H100 run (2026-06-06)

| Task | Status | Notes |
|---|---|---|
| T2T (text → text) | ✅ | offline B=1: 110 tok/s, closed-loop C=32: **1060 tok/s** (full scaling sweep in [`results/ming_t2t_sweep/SUMMARY.md`](../results/ming_t2t_sweep/SUMMARY.md)) |
| I2T (image → text) | ✅ | TTFT 87 ms, ~100 tok/s on Food101 |
| A2T (audio → text) | ✅ | English transcription + Chinese audio QA both work |
| T2S (text → speech) | ✅ | RTF 0.14, 24 kHz mono PCM via harness; 44.1 kHz via direct OpenAI path |
| V2T (video → text) | ✅ | Local Ming demo mp4s; coherent descriptions (`yoga.mp4` → yoga pose narration, `cup_change.mp4` → "shell game") |
| V2S (video → speech) | ✅ | Local Ming demo mp4s; 2-3 MB WAV/clip @ 44.1 kHz |
| I2S (image → speech) | ✅ | Food101 in, ~7 s/req for ~48 s of audio |
| A2S (audio → speech) | ✅ | Ming sample wavs; 0.5-3 MB WAV/clip @ 44.1 kHz |
| T2I / I2I (image gen) | not wired | requires `ming_flash_omni_image.yaml` + a benchmark wrapper similar to BAGEL's `/v1/images/generations` path |

The V2T/V2S/A2S runs sidestep the bench harness's `UCF101Dataset` and
`LibriSpeechDataset` (both want fresh HF-Hub downloads) by hitting
`/v1/chat/completions` directly with base64-inlined media from local files
(Ming repo's `figures/cases/*.mp4` and `data/wavs/*.wav`).