# Shared environment for E6E (PD-disaggregation interference) servers + runner.
# Source before launching anything.
export HF_HUB_OFFLINE=1
export HF_HUB_CACHE=/m-coriander/coriander/naomi/mminf_cache/qwen3omni
export HF_HOME=/m-coriander/coriander/atindra/.cache/huggingface
export XDG_CACHE_HOME=/m-coriander/coriander/atindra/.cache
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TMPDIR=/m-coriander/coriander/atindra/.cache/tmp
mkdir -p $TMPDIR
export OPENAI_API_KEY=EMPTY
export PIP_CACHE_DIR=/m-coriander/coriander/atindra/.cache/pip

ENVS=/m-coriander/coriander/atindra/envs
BENCH=/m-coriander/coriander/atindra/mstar_rebuttal/bench
EXP=/m-coriander/coriander/atindra/mstar_rebuttal/experiments/E6E
SEEDTTS=$BENCH/data/seedtts_testset
LIBRI_LONG=$BENCH/data/libri_long
export BENCH_LIBRI_WAV_DIR=$LIBRI_LONG

# Both arms run on the SAME physical GPUs; rank i = i-th listed GPU.
# 2,3,7 chosen 2026-07-27: 0/1 rohan sglang, 4 naomi (small), 5/6 workspace jobs.
E6E_GPUS="${E6E_GPUS:-2,3,7}"
E6E_PORT="${E6E_PORT:-8311}"
E6E_MIX='audio_to_text:1:libri,text_to_speech:1:seed_tts'
E6E_MIX_SEED=0
E6E_WINDOW_S=240

mkdir -p "$EXP" /m-coriander/coriander/atindra/.cache
