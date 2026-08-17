# Nemotron-Duplex — M* serving-engine integration plan

Goal: production serving via the M* engine (walk graphs + wired audio submodule
forwards) with **true batched inference** of concurrent real-time requests.

Oracle for every phase: the verified standalone paths (`offline_inference`,
`DuplexStream`) — engine output must match them per-request.

## Node → engine mapping (template: `mstar/model/qwen3_omni`)

| Node | Engine | Analog | Role |
|---|---|---|---|
| `conformer_encoder` | STATELESS (`enc_dec`) | audio_encoder | audio chunk → `combined_embeds` (+ RNN-T `transcript`) |
| `nano_llm` | KV_CACHE (+ Mamba pool) | Thinker | frame loop: fuse(audio, prev_text, prev_func) → text/func token |
| `eartts_talker` | KV_CACHE | Talker | text token → 31 RVQ codes (MoG+MaskGIT+CFG, internal sampling) |
| `audio_codec` | STATELESS (`audio_codec`) | Code2Wav | codes → 22.05 kHz PCM chunk |

Key structural difference vs qwen3_omni: the nano decode loop is **frame-driven**
(each step consumes a new `audio_embeds` frame streamed from the encoder, fused
with its own previous token) rather than purely self-fed. So the nano decode node
takes two inputs: streamed `audio_embeds` + self-feedback `prev_text`/`prev_func`.

## Phases (each committed + verified against the oracle)

- **E1 — Batched Mamba decode (this is the core of "true batch inference").**
  Make `Mamba2Mixer.forward` + `MambaStateAccessor` batch-aware: stack each
  request's conv/ssm state → one fused SSD scan/step → unstack. `NemotronHLLM`
  batched forward segments the packed `(L,H)` by per-request seq_lens. Nano
  submodule: `can_batch=True`, `forward_batched`. Attention already batches via the
  paged-KV pool; only conv/ssm state needs batching. Eager first (state lives in
  `PerRequestState`); CUDA-graph capture (fixed-buffer Option-B pool) is a later
  perf layer. Verify: batched engine nano decode == per-request standalone tokens.
- **E2 — Stateless submodule forwards.** `ConformerEncoderSubmodule.forward`
  (audio→combined_embeds, + RNN-T transcript), `AudioCodecDecoderSubmodule.forward`
  (codes→PCM, `audio_codec` flavor, left-context trim). Each verified vs standalone.
- **E3 — Talker submodule forward (AR).** `EarTTSTalkerSubmodule` per-frame:
  text token → MoG/MaskGIT/CFG internal sampling → 31 codes. Verified vs standalone.
- **E4 — Walks + partitions + topology.** `prefill_audio` (Sequential enc→nano),
  frame-driven `decode` Loop (streams text→talker), `talker_decode` Loop
  (streams codes→codec), `codec_chunk`. Partitions ENC/LLM, Talker, Codec with
  StreamingGraphEdge + chunk policies. Emit text + audio (+ transcript) modalities.
- **E5 — End-to-end batched serving check.** N concurrent audio requests through
  the engine → per-request outputs match standalone; confirm one fused batched
  step advances all requests.
- **E6 (later perf) — CUDA-graph capture** via a fixed-buffer conv/ssm pool
  (mirror `kv_store` paged pool + `cuda_graph_runner` dummy-rid state swap).

## Batched Mamba decode — mechanism (E1)

Decode step: `L == B` (1 token/request). Reshape packed `(B,H)` → per-request.
- conv: stack `prev_conv (B, conv_dim, k-1)` + new col → depthwise conv over the
  last-k window per request → new `(B, conv_dim, k-1)`.
- ssm: stack `h0 (B, nheads, head_dim, d_state)`; batched single SSD step.
- Fresh requests (no state yet) seed zeros.
Prefill (varlen, per request at turn start) handled by segmenting on seq_lens.
`MambaStateAccessor` gains `request_ids` + `seq_lens`; read/write iterate the batch.
