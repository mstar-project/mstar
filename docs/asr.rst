Speech recognition: Whisper and Qwen3-ASR
==========================================

M* serves two ASR families on one encoder-decoder scaffold: OpenAI Whisper
(``whisper_large_v3_turbo``, ``whisper_large``) and Qwen3-ASR (``qwen3_asr``
for the 1.7B checkpoint, ``qwen3_asr_realtime`` for the 0.6B one trained for
streaming). Both are reached through the OpenAI surfaces described in
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
    decoder; samples the first transcript token.
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
    <|endoftext|>`` on the served decoder weights, outside the caches. The
    alignment heads' cross-attention becomes word start/end times (per-head
    standardization, median filter, dynamic time warping), emitted in
    Whisper's own timestamp vocabulary after ``<|startoflm|>`` so the same
    detokenizer renders it and the serving layer lifts it into ``words``.

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

A request hears up to 20 minutes, so uploads are never windowed. The
``context`` (the OpenAI ``prompt``) is the system-turn hint the reference
SDK also takes; ``language`` forces the language; ``assistant_prefix``
prefills the assistant turn, which is what streaming builds on.

**Realtime.** ``/v1/realtime?intent=transcription`` (see :doc:`clients`)
re-transcribes the audio heard so far every ``chunk_seconds`` as one engine
request whose assistant turn starts with the previous hypothesis minus its
last ``unfixed_tokens`` tokens, the SDK's streaming algorithm: only the tail
is ever revised, partial results are pushed as
``conversation.item.input_audio_transcription.delta`` events plus
``mstar.transcription.partial`` (the stable hypothesis with the seconds of
audio it covers), and ``input_audio_buffer.commit`` yields the final
transcript.

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
