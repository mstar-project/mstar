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