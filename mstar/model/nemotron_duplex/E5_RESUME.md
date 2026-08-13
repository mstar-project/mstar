# Nemotron-Duplex — engine integration resume notes (E5)

Migration/handoff doc for continuing on another server (e.g. **ptc**). The branch
`keisuke/model-support-nemotron-duplex` (PR #215) is the source of truth — everything
below is committed through `bf3fe346`.

## Status at a glance

| Milestone | State |
|---|---|
| E1 batched Mamba (`Mamba2Mixer` + `MambaStateAccessor`) | done, verified 8.5e-4 vs standalone |
| E1 nano node batching (`can_batch`/`forward_batched`/`seq_lens`) | done |
| E2 `conformer_encoder.forward`, `audio_codec.forward` | done, bit-exact vs standalone |
| E3 `eartts_talker.forward` (internal MoG/CFG, per-request state) | done, bit-exact codes |
| E4 walks + partitions + topology + fusion + function aux | done; `get_worker_graphs` derives all 5; 8 CI tests in `test/modular/test_nemotron_duplex_model.py` |
| E4 multi-partition forward-pass-args driver | done; routing + prefill→decode verified |
| E5 live serving (conductor + worker + GPU) | **in progress — hangs; see below** |

**Standalone oracle (use to validate engine output):** `offline_inference` (text 100%
token-match vs the NeMo reference) and `DuplexStream` / `create_stream`. Audio is
stochastic (noise+Gumbel) but seeded (`eartts.inference_seed`).

## What the live E5 run proved

Launched the full stack via `mstar-serve` with the model on cuda:0:
- All four engines load (`stateless.audio_codec`, `kv_cache` = nano+talker,
  `stateless.enc_dec` = conformer_encoder).
- All five walks are served: `encode`, `prefill_text`, `decode`, `talker_decode`,
  `codec_chunk`. CUDA-graph warmup runs the forwards (nano/talker eager by design,
  conformer torch.compiled) — no `NotImplementedError`, no crash.
- A request with audio input is accepted, audio ingested + routed, pipeline starts.

Three real integration bugs were found + fixed live (commit `bf3fe346`):
1. `load_audio` — decode uploads via `soundfile` (base uses torchcodec, native libs
   unavailable here).
2. `process_prompt` — map the data worker's `audio_inputs` key → `audio_features`.
3. `conformer_encoder.forward` — emit under `audio_frame` (the LLM decode edge name);
   was `audio_embeds`, so the LLM decode loop never received frames.

## The open bug (resume here)

An audio request **hangs** (no crash, no error) after `Request … submitted`. The stall
is in the **frame-synchronous Encoder→LLM streaming loop** — the deepest runtime piece.
Note this is an unusual shape: no reference model streams an encoder *frame-by-frame*
into an LLM *decode* loop (qwen3_omni folds its encoder into a `prefill_audio`
`Sequential`, then the LLM decodes self-fed).

`NDTRACE` logging is committed (WIP) to localize it — `grep NDTRACE <server-log>`:
- `conformer_encoder.forward` (encoder ran, T frames)
- `nano.prepare_inputs walk=… keys=…`
- `get_initial_forward_pass_args` / `get_partition_forward_pass_args` (partition scheduling)

Decision tree from the traces:
- **No `encoder.forward` trace** → the Encoder partition isn't being scheduled →
  look at partition scheduling / `get_partitions` producer wiring.
- **Encoder fires, `nano.prepare_inputs walk=decode` never fires** → the `audio_frame`
  stream isn't delivering to the LLM decode loop → the cross-partition streaming
  (`StreamingGraphEdge` + `FixedChunkPolicy(chunk=1)` on the Encoder→LLM connection),
  or a single-worker scheduling deadlock (Encoder must produce before LLM consumes).
- **`nano.prepare_inputs` fires N times then stops** → the decode loop doesn't
  terminate/emit → `_stream_exhausted` completion in `get_partition_forward_pass_args`
  (verify the `StreamingConnectionState` fields `producer_done`/`consumed_count`/
  `token_count` are the right names/semantics).

**Likely fix direction** (if the stream doesn't deliver): reshape the Encoder→LLM
coupling to a supported pattern — e.g. make the encoder the first node of a
`prefill_audio` `Sequential` in the LLM partition that emits the whole `audio_frame`
stream, rather than a standalone producer partition. Compare to
`mstar/model/qwen3_omni` `get_graph_walk_graphs` / `get_partitions` /
`get_partition_topology` and its conductor `_get_*_forward` state machines.

Strip the `NDTRACE` lines once E5 is green.

## Repro on the new server

Env: mstar serving deps + the NeMo-reference deps (the `nemoref` conda env used on the
dev box), plus the HF weights cached: `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B` and the
`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark` tokenizer repo.

Serve (single GPU). Two gotchas that cost time here:
- `PYTHONPATH` must point at this checkout so `import mstar` resolves this branch (the
  installed `mstar-serve` otherwise imports the main checkout, which lacks the
  `nemotron_duplex` registration).
- `--tensor-comm-protocol SHM` (single node; RDMA/InfiniBand mlx5 devices are inactive
  here, so the default RDMA Mooncake registration fails).

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD PYTHONUNBUFFERED=1 \
  mstar-serve --config configs/nemotron_duplex.yaml \
  --host 127.0.0.1 --port 8019 --tensor-comm-protocol SHM \
  --socket-path-prefix /tmp/mstar_nd/ --upload-dir /tmp/mstar_nd_up/
```

Request:
```python
from mstar.client.client import MStarClient
c = MStarClient("http://127.0.0.1:8019", timeout=150)
r = c.generate(audio="user_16k_mono.wav", input_modalities=("audio",),
               output_modalities=("text", "audio"), temperature=0.0)
print(r.text, len(r.audio or b""))
```

Validate engine output per-request against `offline_inference` on the same wav (the
verified oracle). Then confirm true batching with ≥2 concurrent requests.
