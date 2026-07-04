# Zonos2 — booting & benchmarking runbook

How to bring up the **M\* Zonos2 port** and the **original ZONOS2** side by side,
generate audio, benchmark throughput, and tear everything down cleanly. Written
from the setup used in this repo; see `README.md` for the serving-architecture
overview.

Checkpoint for both: `Zyphra/ZONOS2` (HF; 28 layers, dim 2048, 16 experts). It
auto-downloads to the HF cache on first launch.

---

## 0. One-time prerequisites

1. **DAC vocoder**: `pip install descript-audio-codec` (already installed here).
2. **M\* must be installed editable**, or the server silently runs stale code:
   ```bash
   cd /home/stephenduan/mstar
   pip install -e . --no-deps --no-build-isolation
   ```
   Why: the server launches via `python mstar/api_server/entrypoint.py`, which puts
   the *script's* dir on `sys.path` — a non-editable site-packages copy would win
   over the working tree, so edits (e.g. the MoE weight-loader fix) wouldn't take
   effect. Verify with `python -c "import mstar,os;print(os.path.realpath(mstar.__file__))"`
   → must point at `/home/stephenduan/mstar/mstar`, not `.venv/.../site-packages`.
3. **HF access** to `Zyphra/ZONOS2` (the box already has it cached at
   `~/.cache/huggingface/models--Zyphra--ZONOS2`).
4. Pick **free GPUs** first: `nvidia-smi --query-gpu=index,memory.free --format=csv`.
   Examples below use GPU 0 for M\* and GPU 1 for the original.

---

## 1. Boot the M\* server (GPU 0, port 20002)

```bash
cd /home/stephenduan/mstar
ZONOS2_MODEL_PATH=Zyphra/ZONOS2 DEVICES=0 PORT=20002 WARMUP=1 \
    bash test/zonos2/launch_server_zonos2.sh
```

- `launch_server_zonos2.sh` pre-downloads the DAC weights and fires **one warm-up
  request** once `/health` is up. This is essential: the first request pays a
  one-time `torch.compile` + DAC-load cost that otherwise blows past the server's
  **600 s request timeout** and looks like a hang. With warmup, first real request
  is fast. Disable with `WARMUP=0`.
- Ready when the log shows `loaded 503/503 model tensors` (a `479/503` + "tensors
  not found" warning means the MoE weight fix is missing / stale install) and
  `[warmup] complete`.
- 2-GPU split: `DEVICES=0,1 CONFIG=configs/zonos2.yaml` (LLM on 0, DAC on 1).
- Health check: `curl -sf http://127.0.0.1:20002/health`.

## 2. Boot the original ZONOS2 (GPU 1, port 1919)

```bash
cd /home/stephenduan/ZONOS2
CUDA_VISIBLE_DEVICES=1 uv run python -m zonos2 \
    --model-path Zyphra/ZONOS2 --tts-default-voices-dir ./default_voices/
```

- SGLang-based; captures CUDA graphs for bs=1…256 and grabs most of the GPU
  (~110 GiB KV cache). Ready when the log prints `API server is ready to serve on
  127.0.0.1:1919`.
- Health check: `curl -sf http://127.0.0.1:1919/v1/models`.
- Endpoints: `POST /tts/generate` (streams raw **f32le** 44.1 kHz PCM),
  `POST /v1/audio/speech`, `GET /v1/models`. Same sampling defaults as M\*.

---

## 3. Generate audio

**M\*** (int16 WAV out):
```bash
python test/zonos2/tts_request.py --text "Hello World" --output test/zonos2/hello_mstar.wav
```

**Original** (streams f32le PCM — wrap to WAV yourself):
```bash
python - <<'PY'
import requests, numpy as np, wave
buf=b"".join(c for c in requests.post("http://127.0.0.1:1919/tts/generate",
     json={"text":"Hello World","stream":True}, stream=True).iter_content(8192) if c)
pcm=(np.clip(np.frombuffer(buf,"<f4"),-1,1)*32767).astype("<i2")
with wave.open("test/zonos2/hello_original.wav","wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(pcm.tobytes())
PY
```

## 4. Benchmark throughput

```bash
python test/zonos2/bench_concurrent.py --concurrency 8 --max-tokens 300   # M* (:20002)
python test/zonos2/bench_original.py  --concurrency 8 --max-tokens 300    # original (:1919)
```
Both report TTFT / E2E / throughput (audio-s per wall-s, req/s). Chart of a prior
sweep: `test/zonos2/throughput_chart.html`.

---

## 5. Shut down & free GPUs

Killing the launcher alone leaves orphaned worker processes holding VRAM. Full
teardown:

```bash
pkill -f "launch_server_zonos2.sh"; pkill -f "entrypoint.py --config configs/zonos2"
for pid in $(pgrep -f "uv run python -m zonos2"); do pkill -9 -P $pid; kill -9 $pid; done
# reap orphaned workers still holding GPU memory (ours only — leaves other users' jobs)
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  case "$(ps -o cmd= -p $p)" in *mstar/.venv*|*ZONOS2/.venv*) kill -9 $p;; esac
done
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader   # confirm freed
```

---

## Known issues / status

- **M\* batching**: `Zonos2LLMSubmodule` supports BS>1 (eager path,
  `can_batch=True` + `forward_batched`). Correct, but eager (no CUDA-graph decode)
  → ~88 ms/token and ~3× scaling at BS=8, vs the original's CUDA-graph path.
  CUDA-graph decode capture is the deferred next step to close the gap (needs
  graph-safe multi-codebook sampling — template is `qwen3_omni`, or capture the
  forward only and sample eagerly via a `cuda_graph_runner._sample_and_remap` hook).
- **M\* audio is ~2× hotter than the original** and can peak near clipping
  (~0.98 vs ~0.44). The waveform shape is otherwise clean (no buzz); the DAC
  decode + `shear_up` match the reference exactly. Root cause not yet pinned
  (sampling variance vs. a systematic level difference) — **under investigation**.
  The M\* `StreamingDacDecoder` also omits the reference's overlap-add crossfade
  (`overlap_frames=4`, raised-cosine window) and exact EOS-frame trimming, so
  chunk-boundary smoothing is weaker.
- **Cold start**: first request after boot without warmup can exceed the 600 s
  timeout — always launch with `WARMUP=1` (default).
```
