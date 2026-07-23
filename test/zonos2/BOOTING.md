# Zonos2 — booting runbook

How to bring up the **M\* Zonos2 port**, generate audio, and tear it down cleanly.
Paths are repo-relative — run the commands from the repo root. See `README.md` in
this directory for the serving-architecture overview, and `docs/installation.rst`
for the full install matrix.

Checkpoint: `Zyphra/ZONOS2` (HF; 28 layers, dim 2048, 16 experts). It auto-downloads
to the HF cache on first launch. Override with `ZONOS2_MODEL_PATH=/path/to/ckpt`.

---

## 0. One-time prerequisites

1. **Install M\* with the zonos2 extra** (editable, so the server runs your working
   tree — see the note below):
   ```bash
   uv pip install --torch-backend=auto -e ".[zonos2]"
   ```
2. **DAC vocoder** — installed separately (it pins `protobuf<3.20` transitively, so
   it's kept out of the extras; see `docs/installation.rst`):
   ```bash
   pip install descript-audio-codec
   python -c "import dac; print('dac ok')"
   ```
3. **HF access** to `Zyphra/ZONOS2` (cached under `~/.cache/huggingface` after the
   first download).
4. Pick a **free GPU**: `nvidia-smi --query-gpu=index,memory.free --format=csv`.

> **Why editable?** The launch script runs `python mstar/api_server/entrypoint.py`,
> which puts the script's dir on `sys.path`. A non-editable site-packages copy would
> shadow the working tree, so local edits wouldn't take effect. Verify with
> `python -c "import mstar,os;print(os.path.realpath(mstar.__file__))"` → it must
> point at `<repo>/mstar`, not `.venv/.../site-packages`.

---

## 1. Boot the server

Simplest (colocated on one GPU — LLM prefill+decode and the DAC vocoder share rank 0):

```bash
mstar serve zonos2 --gpus 0                     # default port 8000
```

Or via the launch script, which pre-downloads the DAC weights and fires **one warm-up
request** once `/health` is up:

```bash
ZONOS2_MODEL_PATH=Zyphra/ZONOS2 DEVICES=0 PORT=20002 WARMUP=1 \
    bash test/zonos2/launch_server_zonos2.sh
```

- **Warm-up matters.** The first request pays a one-time `torch.compile` + DAC-load
  cost that can otherwise blow past the server's **600 s request timeout** and look
  like a hang. The launch script warms up automatically (`WARMUP=1`, the default);
  with plain `mstar serve`, send a throwaway request right after the server reports
  ready. Disable with `WARMUP=0`.
- **Ready** when the log shows `loaded 503/503 model tensors` and (with the launch
  script) `[warmup] complete`. A `479/503` + "tensors not found" warning means the
  MoE weight-loader fix is missing / the install is stale.
- **Two-GPU split** (LLM on 0, DAC on 1): `DEVICES=0,1 CONFIG=configs/zonos2.yaml`
  with the launch script, or `mstar serve zonos2 --config configs/zonos2.yaml --gpus 0,1`.
- **Health check:** `curl -sf http://127.0.0.1:20002/health` (or `:8000`).

## 2. Generate audio

```bash
python test/zonos2/tts_request.py --text "Hello there." --output test/zonos2/hello.wav
```

Or via the SDK:

```python
from mstar import MStarClient
MStarClient("http://localhost:20002").tts("Hello there.").to_wav("out.wav")
```

Zonos2 is served on the native `/generate` endpoint and the SDK; there is no OpenAI
`/v1/audio/speech` adapter yet.

## 3. Batching smoke test

`test/zonos2/test_batching.py` exercises the BS>1 eager path (multiple concurrent
requests through one server). Run it against a booted server to sanity-check batched
decode.

---

## 4. Shut down & free GPUs

Killing the launcher alone can leave orphaned worker processes holding VRAM. The
launch script reaps its own orphans on the next start; for a full manual teardown:

```bash
pkill -f "launch_server_zonos2.sh"; pkill -f "entrypoint.py --config configs/zonos2"
# reap orphaned workers still holding GPU memory (ours only — leaves other users' jobs)
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  case "$(ps -o cmd= -p $p)" in *mstar/.venv*) kill -9 $p;; esac
done
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader   # confirm freed
```

---

## Known issues / status

- **Eager decode (no CUDA-graph capture on the batched path).** `Zonos2LLMSubmodule`
  supports BS>1 via `can_batch=True` + `forward_batched`, but the batched path runs
  eager. Single-request decode is CUDA-graph captured; closing the batched gap is the
  deferred next step.
- **Audio level.** M\* output can run hotter than the reference and peak near
  clipping. The waveform shape is otherwise clean (DAC decode + `shear_up` match the
  reference); root cause not yet pinned. The `StreamingDacDecoder` also omits the
  reference's overlap-add crossfade and exact EOS-frame trimming, so chunk-boundary
  smoothing is weaker (the trailing ~`n_codebooks-1` frames, ~90 ms, aren't emitted).
- **Real-text concurrency hang** under investigation — long, real-world prompts run
  concurrently can stall the stream. Use short prompts / low concurrency for now.
- **Cold start.** First request after boot without warmup can exceed the 600 s
  timeout — always warm up (see §1).
- **Speaker/voice cloning** is not modeled; those checkpoint tensors, if present, are
  ignored (`client.tts(...)` takes no `voice` argument for Zonos2).
