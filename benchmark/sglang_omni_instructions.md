### Setup sglang-omni
```
# clone this repository
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# create a virtual environment in docker
uv venv .venv -p 3.11
source .venv/bin/activate

# install
uv pip install -v .

# install for development
uv pip install -v -e .
```
### run the server for qwen3-omni
```
# Qwen3-Omni, speech mode — for section 3 (SeedTTS; multi-GPU)
python -m sglang_omni.cli.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000

# Qwen3-Omni, text-only mode — for sections 4 (MMSU) and 5 (MMMU)
python -m sglang_omni.cli.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only --port 8000
```