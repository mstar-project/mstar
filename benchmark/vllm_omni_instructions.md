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
The released checkpoint is `inclusionAI/Ming-flash-omni-2.0` (~238 GB, 42
safetensors shards). Pick a deploy yaml based on what you want to benchmark:

```
# thinker + talker (text + speech out, 4 GPUs + colocated talker on GPU 3)
vllm serve inclusionAI/Ming-flash-omni-2.0 --omni --port 8092 \
  --stage-configs-path vllm_omni/deploy/ming_flash_omni.yaml

# thinker only (text out, 4 GPUs full memory)
vllm serve inclusionAI/Ming-flash-omni-2.0 --omni --port 8092 \
  --stage-configs-path vllm_omni/deploy/ming_flash_omni_thinker_only.yaml

# standalone TTS / talker only (single GPU)
vllm serve inclusionAI/Ming-flash-omni-2.0 --omni --port 8092 \
  --stage-configs-path vllm_omni/deploy/ming_flash_omni_tts.yaml
```

Then run the benchmark against it:

```
MODEL=ming_flash_omni INF_SYS=vllm_omni TASK=text_to_text \
  URL=http://0.0.0.0:8092 ./benchmark/run_benchmark.sh
```

All eight modalities Ming-flash-omni-2.0 exposes through the omni pipeline
are registered on `MingFlashOmni.get_supported_modalities()`
(T2T/I2T/A2T/V2T + T2S/I2S/A2S/V2S). Image-gen tasks (T2I/I2I) require the
`ming_flash_omni_image` deploy yaml and a benchmark wrapper similar to BAGEL's
`/v1/images/generations` path — not wired yet.