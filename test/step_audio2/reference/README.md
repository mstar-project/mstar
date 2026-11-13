# Step-Audio2 reference oracle

The parity harnesses (`parity_encoder.py`, `parity_llm.py`, `parity_token2wav.py`) and the
greedy cases of `serve_check.py` compare M* against tensors these scripts record from the
reference implementation. They run in their own venv, since the reference pins
`transformers==4.49.0`. They don't run in CI.

## Inputs

- The reference repo: `git clone https://github.com/stepfun-ai/Step-Audio2` (recorded at
  `76e272b56c3917a8d7188f18bbb5a65dfc8a0845`). Its `assets/` holds the question clips and the
  two voices.
- The checkpoint: `huggingface-cli download stepfun-ai/Step-Audio-2-mini --local-dir <hf>`.

## Venv

```bash
uv venv --python 3.12 /path/to/venv-oracle
VIRTUAL_ENV=/path/to/venv-oracle uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchaudio==2.11.0 \
  transformers==4.49.0 librosa onnxruntime s3tokenizer diffusers hyperpyyaml einops soundfile scipy numpy
```

torchaudio 2.11 loads and saves through torchcodec; `sf_shim.py` routes both through soundfile
instead, with the same float32 samples. Every script imports it.

## Recording

One GPU (an H200 here), in this order, with `R=<github/Step-Audio2>`, `H=<hf/Step-Audio-2-mini>`
and `O=<oracle>`:

```bash
PY="/path/to/venv-oracle/bin/python -I"
$PY oracle_probe.py   $R $H $O/probe   # greedy replies; probe.json holds the spoken reply's tokens
$PY record_oracle.py  $R $H $O/rec     # bf16 LLM and tower per case, the fp32 token2wav stream
$PY record_fp32.py       $H $O/rec     # the fp32 configuration, on the same inputs
$PY greedy_spread.py     $H $O/rec     # greedy tokens under math SDPA and fp32, with top-2 margins
$PY record_t2w_fp16.py $R $H $O/rec    # the reference's float16 token2wav over the same windows
$PY hift_float.py     $R $H $O/rec     # the vocoder replayed on the recorded mels
HIFT_DETERMINISTIC=1 $PY hift_float.py $R $H $O/rec   # the same, deterministic (bit-exact check)
```

Then point the harnesses at the outputs:

```bash
export STEPAUDIO2_CKPT=$H STEPAUDIO2_ORACLE=$O/rec STEPAUDIO2_ASSETS=$R/assets
python test/step_audio2/parity_llm.py
```

The serving config's `voices_dir` takes the two voice clips (`$R/assets/default_female.wav`
and `default_male.wav`).
