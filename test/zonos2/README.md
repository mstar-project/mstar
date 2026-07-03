# Zonos2 TTS — local serving

Multi-codebook TTS: an autoregressive LLM (`LLM` partition) streams audio-code
frames to the DAC vocoder (`DAC` partition), which emits 44.1 kHz PCM.

## Prerequisites
1. **DAC vocoder**: `pip install descript-audio-codec`
2. **A checkpoint** in the reference layout — a directory with `params.json`
   (config) + `model.pth` (a torch `state_dict`; `model.pt` or
   `consolidated/consolidated.pth` also work), or an HF repo id. Point at it with:
   ```bash
   export ZONOS2_MODEL_PATH=/path/to/zonos2_checkpoint
   ```
   Without it the server still starts, but the LLM runs with random weights
   (noise audio) — useful only for smoke-testing the plumbing.

## Launch
```bash
# simplest (colocated on one GPU)
mstar serve zonos2 --gpus 0

# or this script (colocated by default; set CONFIG/DEVICES for 2-GPU)
ZONOS2_MODEL_PATH=/path/to/ckpt DEVICES=0 PORT=20002 \
    bash test/zonos2/launch_server_zonos2.sh

# two GPUs: LLM on 0, DAC on 1
ZONOS2_MODEL_PATH=/path/to/ckpt DEVICES=0,1 CONFIG=configs/zonos2.yaml \
    bash test/zonos2/launch_server_zonos2.sh
```

## Request
```bash
python test/zonos2/tts_request.py --text "Hello there." --output out.wav
```
Or via the SDK:
```python
from mstar import MStarClient
MStarClient("http://localhost:20002").tts("Hello there.").to_wav("out.wav")
```
OpenAI-compatible `POST /v1/audio/speech` also works.

## Configs
- `configs/zonos2_colocated.yaml` — 1 GPU (LLM + DAC on rank 0); the
  `mstar serve zonos2` default.
- `configs/zonos2.yaml` — 2 GPUs (LLM on rank 0, DAC on rank 1).

Per-request/model overrides go in a `model_kwargs:` block in the YAML (forwarded
to `Zonos2Model.__init__`, e.g. to override config fields).

## Notes / current limitations
- The config is read from the checkpoint's `params.json`
  (`mstar/model/zonos2/weight_loader.py`); dims must match the weights.
- DAC decoding is incremental without overlap-add crossfade or exact eos-frame
  trimming, so the trailing ~`n_codebooks-1` frames (~90 ms) aren't emitted.
- Sampling params are model-level (`TTSSamplingParams` defaults); the LLM runs
  eager (no CUDA-graph capture) and single-request (`can_batch=False`).
- Speaker/voice-cloning conditioning is not modeled (those checkpoint tensors,
  if present, are ignored).
