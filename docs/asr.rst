Speech recognition: Whisper and Qwen3-ASR
==========================================

M* serves two ASR families on one encoder-decoder scaffold: OpenAI Whisper
(``whisper_large_v3_turbo``, ``whisper_large``) and Qwen3-ASR (``qwen3_asr``
for the 1.7B checkpoint, ``qwen3_asr_realtime`` for the 0.6B one trained for
streaming). Both are reached through the OpenAI endpoints described in
:doc:`clients`: ``POST /v1/audio/transcriptions`` and, for Qwen3-ASR, the
``/v1/realtime`` transcription WebSocket.

.. code-block:: bash

   mstar-serve --config configs/whisper_large_v3_turbo.yaml --port 8000
   curl http://localhost:8000/v1/audio/transcriptions \
        -F model=whisper_large_v3_turbo -F file=@speech.wav \
        -F response_format=verbose_json -F 'timestamp_granularities[]=word'

Whisper
-------

**Graph.** Two nodes. ``audio_encoder`` takes one window of samples
(at most 30 s) per request, pads it, turns the batch into log-mel features
with one STFT on the GPU and runs the convolutional-transformer encoder; the
forward is captured once per batch size (1 to 32) and replayed. ``decoder``
is the autoregressive text decoder on paged KV: the encoder output is
projected once per request into a separate paged cross-attention cache
(written at prefill, read by every later step), so decode steps of many
requests batch together and are captured per batch size (1 to 64).

**Walks.**

``prefill``
    encoder, then the forced prompt ``[<|startofprev|> context]
    <|startoftranscript|><|lang|><|task|>[<|notimestamps|>]`` through the
    decoder, and samples the first transcript token.
``detect_language``
    the same with a prompt that stops at ``<|startoftranscript|>`` and
    sampling restricted to the language tokens; ``prefill_prompt`` then
    appends ``<|task|>[<|notimestamps|>]`` after the detected token.
``decode``
    a dynamic loop, one token per step, until ``<|endoftext|>`` or the
    448-position table is full. The Whisper timestamp rules (first token a
    timestamp, timestamps in monotonic pairs, timestamp-vs-text
    log-probability test) are applied inside the captured forward from a
    four-integer state that travels with the token, so the worker's
    speculative next-step launch never sees a stale history.
``align``
    only when word timestamps were asked for: one teacher-forced pass over
    ``<|startoftranscript|><|lang|><|task|><|notimestamps|> transcript
    <|endoftext|>`` on the served decoder weights, outside the caches. On the
    GPU, the alignment heads' cross-attention is standardized per head,
    median-filtered and averaged, and the walk emits that matrix with the
    token ids as one float32 tensor. The API server's ``postprocess`` runs the
    dynamic time warping (compiled by numba, part of the ``asr`` extra, when
    the API server loads the model) and groups tokens into words, rendered in
    Whisper's own timestamp vocabulary after ``<|startoflm|>``, which the
    serving layer lifts into ``words``.

**Request options** (``model_kwargs`` / the OpenAI fields): ``language``
(ISO code; omit to detect), ``task`` (``transcribe`` or ``translate``),
``initial_prompt`` (the OpenAI ``prompt``: prior text carried over as
``<|startofprev|>`` context), ``timestamps`` (``segment`` or ``word``, set
from ``response_format`` / ``timestamp_granularities``), ``temperature``.

**Long-form audio.** Whisper hears 30 s; the transcription route serves a
longer upload as consecutive windows (openai-whisper's loop): each window is
conditioned on the transcript so far, decoded with timestamps so the next
window starts where the last closed segment ended, and re-decoded (first
without the prompt, then at rising temperatures) when its text is a
repetition loop. ``long_form="parallel"`` submits fixed windows at once,
cut at the quietest point before each boundary. See :doc:`clients`.

Qwen3-ASR
---------

**Graph.** ``audio_encoder`` is the AuT encoder: three strided convolutions
(13 tokens per 100 mel frames), transformer layers whose attention is
confined to 8 s windows, run as one packed forward per batch with ragged
attention (one segment per window of every request), and a projection into
the LLM's hidden size. ``LLM`` is a dense Qwen3 decoder: the ChatML prompt
with ``<|audio_pad|>`` placeholders is embedded, the encoder output is
scattered over the placeholders, and prefill (captured on packed token
buckets) and decode (captured per batch size) run on paged KV. The model
writes ``language {Name}<asr_text>{text}``; the adapter reports the
language as an ISO code and the text alone.

A request hears up to 20 minutes; longer uploads are cut into 20 minute
windows. The ``context`` (the OpenAI ``prompt``) is the system-turn hint
the reference SDK also takes; ``language`` forces the language;
``assistant_prefix`` prefills the assistant turn, which is what streaming
builds on.

**Realtime.** ``/v1/realtime?intent=transcription`` (see :doc:`clients`)
re-transcribes the audio heard so far every ``chunk_seconds`` as one engine
request whose assistant turn starts with the previous hypothesis minus its
last ``unfixed_tokens`` tokens, the SDK's streaming algorithm: only the tail
is ever revised, partial results are pushed as
``conversation.item.input_audio_transcription.delta`` events plus
``mstar.transcription.partial`` (the stable hypothesis with the seconds of
audio it covers), and ``input_audio_buffer.commit`` yields the final
transcript.

Serving notes
-------------

The ASR models cap the API server's torch intra-op pool at 4 threads
(``PREPROCESS_TORCH_THREADS``; ``preprocess_torch_threads:`` in the config
overrides it). That process only decodes uploads and stages tensors, and a
pool of one thread per core spins after each of those small ops and starves
the event loop and transport threads. On a 192-core H200 host the cap matched
``OMP_WAIT_POLICY=PASSIVE`` on RTFx and latency at concurrency 1 to 64 for both
models, where the default pool lost 4-9 % (up to 29 % with the server pinned
to 16 cores), so no launch setting is needed.

Serve with the shared-memory arena, ``MSTAR_SHM_ARENA=1`` (needs the ``rust/``
extension; see :doc:`environment_variables`). The default SHM transport creates
and removes one file per tensor, and a transcription moves its audio, encoder
states and output tokens as many small tensors; the arena places them in
preallocated segments instead. Qwen3-ASR 1.7B in one run: RTFx 81, 218, 350 and
412 at concurrency 1, 8, 32 and 64 on the file transport, 87, 252, 494 and 558
on the arena, with the same 1.41 % WER.

Every ASR config sets ``max_concurrent_requests``. Whisper's decoder caches
hold 64 requests (12 cross-attention pages and 4 self-attention pages each),
and the model refuses at startup a cap they cannot hold. Requests past the cap
wait in the conductor. Qwen3-ASR's requests vary far more in size: a 20 minute
file holds about 122 pages of 128 tokens for its prompt and up to 32 more for
its transcript, so the shipped config's 2048 pages (``kv_cache.max_num_pages``)
hold about 13 of them at full size at once. More concurrent long files than
the pool holds offload to the host pool and, once that is full too, wait on
each other until they time out: 16 concurrent single-window 20 minute files
stalled on 2048 pages, and peaked at 2352 pages with 4096 in the pool.

``qwen3_asr.yaml`` and ``qwen3_asr_split.yaml`` therefore cap Qwen3-ASR at 12
concurrent requests, which any mix of files up to 20 minutes fits. The cap costs short-utterance
throughput: on LibriSpeech test-clean (one H200, Qwen3-ASR 1.7B) RTFx was
about 420 at 8 concurrent requests, 900 at 32 and 1280 at 64. A deployment
that only serves short audio can raise ``max_concurrent_requests`` (64 fits
the default pool); one that serves many long files at once can raise
``kv_cache.max_num_pages`` with the cap. Admission that sizes the cap by each
request's pages instead is planned.

Both decode loops stream their transcript token by token. A decode step
keeps its rows' tokens on the device for the next step (the sampler's device
loop-back, ``device_loopback_signals``), and the worker sends the batch's
tokens to the client in one frame per step from the stop check's host copy
(``inline_client_signals``), rather than one tensor message per request. A
streaming request therefore receives one delta per token, and tokens reach
the API server while the window decodes: a 20 minute Qwen3-ASR window is
about 3,450 tokens, which, sent as one tensor apiece when the loop ended,
could miss the API server's 15 s delivery deadline under load. Each chunk is
one byte-level BPE token, which can end inside a UTF-8 character; the server
decodes the joined bytes.

The ``*_split`` configs (``whisper_large_split.yaml``,
``whisper_large_v3_turbo_split.yaml``, ``qwen3_asr_split.yaml``,
``qwen3_asr_realtime_split.yaml``) put the encoder and the decoder in two
worker processes on one GPU (``rank_devices``), so a decode step never queues
behind an encoder batch on the same stream. They set ``mps: true``, so both
workers run as clients of one CUDA MPS daemon and their kernels overlap rather
than time-slice (see :doc:`serving`).

Benchmarks and parity
---------------------

``benchmark/asr_bench.py`` drives ``/v1/audio/transcriptions`` on the
LibriSpeech test-clean set the protocol names (200 utterances at
concurrency 1, 8 and 32, RTFx and p50/p95 latency, WER with the Whisper
normalizer, and the 10-minute long-form file); ``benchmark/asr_realtime_bench.py``
plays utterances into ``/v1/realtime`` in real time and measures
partial-result latency; ``benchmark/asr_reference.py`` produces the HF /
reference-SDK transcripts the WER parity is held to. The GPU parity tests
under ``test/asr`` pin the encoders to the HF modules on real audio, the
Whisper greedy tokens to HF ``generate``, and the word timestamps to HF's
token timestamps.

.. code-block:: bash

   python -m benchmark.asr_bench --url http://localhost:8000 --model whisper_large_v3_turbo \
       --system "M*" --concurrency 1 8 32 --long-form --output-json results.json
   python -m benchmark.asr_realtime_bench --url ws://localhost:8003 --model qwen3_asr_realtime \
       --system "M*" --sessions 1 8 --output-json realtime.json
