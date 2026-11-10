Nemotron VoiceChat
==================

Full-duplex speech-to-speech (NemotronLabs VoiceChat-11B): Fast-Conformer
encoder + Nemotron-H hybrid Mamba-2/attention backbone + EarTTS talker +
RVQ codec; user speech in, agent text and 22.05 kHz speech out.

Registry key: ``nemotron_duplex`` *(Beta)*. Checkpoint: ``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B``.

Deployment configs
------------------

The CLI maps the model keys to these deployment configs. See :doc:`/installation`
for dependencies; model-specific requirements and limits follow on this page:

.. list-table::
   :header-rows: 1

   * - CLI model key
     - Config in the repository
   * - ``nemotron_duplex``
     - ``configs/nemotron_duplex.yaml``

From the repository root, launch a deployment with:

.. code-block:: bash

   mstar serve nemotron_duplex

For client requests and response formats, see :doc:`/clients`.

Offline checkpoint preparation
------------------------------

The nano text tokenizer is read from the ``nano/`` folder of
``pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark`` (the base checkpoint
ships only the RNN-T tokenizer); prefetch both repositories on machines
without egress.

Deployment and sampling defaults
--------------------------------

Default deployment: ``configs/nemotron_duplex.yaml`` (all three nodes on one
GPU); ``configs/nemotron_duplex_disagg.yaml`` puts the encoder+LLM and the
talker (with its codec) on two ranks. Launch with ``--tensor-comm-protocol SHM`` on a
single node. Text samples at ``temperature: 0.7, top_p: 0.9`` by default;
send ``temperature: 0`` for greedy. The talker's mixture-of-Gaussians sampling
is stochastic by design and seeded per request.

Send audio and receive replies
------------------------------

Inputs are user-only 16 kHz mono audio plus an optional system prompt (the request's text, primed before the first frame exactly as the reference does): one clip uploaded with the request, a live call over ``/generate/ws`` (an opening with ``end_input: false``, then ``append`` messages until ``end_input``; see :doc:`/clients`), or an OpenAI Realtime conversation over ``/v1/realtime``, which takes PCM16 at one of the eight rates :doc:`/clients` lists (8 to 48 kHz) or G.711, resamples it to 16 kHz, appends it as it comes, and sends each agent reply as one response (BOS / EOS on the agent text channel mark the reply, ``metadata.turn`` on ``/generate``).

The voice is ``Aria``. Output is clocked by the input, one agent frame per 80 ms of user audio, so a ``/generate/ws`` client keeps streaming (silence included) for as long as it wants the agent to talk; ``/v1/realtime`` pads a pause in the client's audio with silence, up to 30 s. The model answers in the frames *after* the user stops, so append a few seconds of trailing silence to a clip. The checkpoint's demo recordings are two-channel conversations (user left, agent right); feed the left channel only.

Session and input limits
------------------------

Limits: a call is capped at ``max_session_s`` (160 s) of audio. An upload, or any one file, past it gets a 400, as does one of more than two channels, and so does a live call whose appends pass it: ``audio_in`` declares the session's samples as its ``max_items``, which the client stream counts before the encoder sees them (raise ``max_session_s``, not the stream's limit, for longer calls). A call that ends with no more than 16 ms of audio, which the encoder cannot featurize, fails with a 500 (the worker reports it without a client status). ``/generate/ws`` appends must be 16 kHz audio (they are neither resampled nor rescaled one by one, which would put a seam at every append); ``/v1/realtime`` resamples for the client.

On both, ``audio_in`` declares the encoder's 80 ms frame (``ClientAudio``), and the data worker sends a live call's audio in whole frames, a step each, the rest waiting for the next append or the call's end.

Background-speech limitation
----------------------------

Known limitation of the checkpoint: it often answers speech that is not
addressed to it, such as a side conversation or background speech, where it
should keep its turn (about half of such pairs on Full-Duplex-Bench v1.5).
NVIDIA's NeMo reference inference (``examples/speechlm2/offline_voicechat_infer.py``
on the ``nemotron-labs-voicechat`` branch of NVIDIA-NeMo/Speech) answers the
same side speech on the same inputs, and the model card's known limitations
say the model is not suitable for noisy environments, especially where
background speech may occur. Nothing in the serving path tells speakers
apart. Real interruptions and backchannels are handled well.



Session capacity
----------------

Every per-session resource is sized for 32 sessions: a slot in each of
``mamba_state``, ``encoder_state``, ``encoder_front`` and ``codec_state`` (``max_slots``, one
more than the sessions, for the sink) and in ``encoder_kv`` (``max_slots``,
no sink), and its pages in ``nano_kv`` and
``talker_kv`` (``max_num_pages``; both sized from ``max_session_s``, and the
nano's from ``max_prompt_tokens`` too). The shipped configs set a
conservative ``max_concurrent_requests: 16``; raising it to 32 needs
no other change, and to serve more, raise the cap and all six with it. The server
refuses to start with a cap one of them cannot hold, since a session past
any of them would wait on admit, and its frames with it, until another ends.
The nano, the talker and the encoder each run at most 32 sessions
a step, their widest CUDA graph; more are split over several steps rather
than run eagerly.

Validate a deployment
---------------------

Checks: ``test/nemotron_duplex/duplex_request.py`` prepares a clip from the
bundled ``turn_taking.wav`` and sends it; ``parity_compare.py`` runs the
standalone reference path on the same clip and checks the served text
against it token for token (bf16, the serving precision, must match save the
last frame; the fp32 oracle can differ where the agent's turn-taking decision
is a knife-edge, which its printed logit margins show).
``live_ws_request.py`` streams a clip in 80 ms appends and reports per-frame
lag; with ``--compare-one-shot`` it checks that a live call sending the whole
clip as one append matches the upload save the last frame, and reports where
the paced call's text first differs from it.

Measured live-call capacity
---------------------------

Measured on one H100 80GB (``benchmark/nemotron_duplex/sessions.py`` on the 106-frame demo clip, three repeats): 24-25 ms per 80 ms tick with one session, 61-64 ms with 32 concurrent sessions, 76-85 ms with 40, 87-92 ms with 48 and 95-97 ms with 64 (2026-10-08). 32 is the most that keep up with real time, hence the pools' size; every session received all its frames. Those are one-shot replays (the whole clip uploaded, so the server runs as fast as it can). Live calls paced in real time over ``/generate/ws`` (``test/nemotron_duplex/live_ws_request.py``, 80 ms appends, 2026-10-17, one H100): input-to-token lag p50 48 ms with one call and about 100-120 ms with 8, which keep up; from 24 calls up the server fell behind real time, so lag grew over the call.

Capturing the encoder's live step (below) took its host time at 32 calls from 23.6 to 8.4 ms and cut live lag 2-6x at 16-32 calls (paced runs interleaved against the uncaptured encoder, on the SHM arena); 32 calls staggered over 2 s all finish, heard latency p50 2.1-2.3 s (from the user stopping to agent audio playable without gaps). With the codec in the talker's step, on one H200: 16 live calls keep real time whether they start together or 2 s apart; 32 all finish but sit at the edge of real time (one run kept up with 32 started together, another did not; later runs kept up at 32), so the shipped cap is a conservative 16.

Served audio is checked for intelligibility by transcribing it (``test/nemotron_duplex/asr_check.py``, Whisper large-v3-turbo): MaskGIT sampling is knife-edge, so attention-backend numerics change the waveform but not the words.

Reference serving comparison
----------------------------

Against vLLM-Omni 0.28's native duplex serving of the same checkpoint (three
eager fp32 engines, one session per GPU by its deploy config, its realtime
client): on the demo clip both systems' response latency is the model's
turn-taking decision (first agent text about 1.8 s after the user stops;
first voiced audio 1.6 s here vs 2.0 s in stream time there, 2.3-2.6 s wall
there), while M* serves 32 concurrent sessions with margin on one H100.

Audio flow through the graph
----------------------------

Three nodes in three async partitions, joined by streaming edges, with the user's audio a client stream (``audio_in``) into the first: ``conformer_encoder`` (16 kHz audio → one LLM-space embedding per 80 ms frame, streamed: each chunk runs the layers over its new frames only, against the call's cached last 70 frames, so a step costs the same however long the call is; it emits the frames no later audio can change, the last ones at the end of input. It is the shared FastConformer, Canary's encoder, as a stream node (``FastConformerStreamSubmodule``), with the featurizer of the VoiceChat reference, which predates NeMo 2.5: reflected clip edges and ``samples // 160 + 1`` mel frames.

The checkpoint's RNN-T transcription head is not loaded) → ``nano_llm`` (frame-synchronous decode loop: each step fuses one audio frame with the previous agent-text and tool-call tokens, emits one agent-text token) → ``eartts_talker`` (one text token → 31 RVQ codes, decoded by the RVQ codec in the same step to that frame's 80 ms of PCM, which goes straight to the client: no codec step of its own, so a frame's audio leaves with the step that made its codes. The decoder is causal, so each call keeps its stream state in a recurrent-state pool, ``codec_state``: each ConvNeXt block's last 6 inputs and the inverse STFT's 12-sample overlap, ~95 KiB a slot in bf16.

A step decodes only the call's new frame against that state, and a fresh, zeroed slot is the decoder's causal padding, so a call's PCM equals the full decode of its frames from its first frame; at 32 sessions the decode is ~100 GMAC per 80 ms tick, inside the talker's CUDA graph). The decode loops are stream-terminated: they end when the upstream stream closes, not on EOS, which is an ordinary per-frame token in duplex speech.

Language-model resources
------------------------

``nano_llm`` declares a paged KV cache over its four attention layers with
the attention plan on it (``resources: nano_kv`` / ``nano_attn``; no
positional encoding, so no position resource), a recurrent-state pool holding
the 27 Mamba-2 layers' conv window and SSM state, one slot per session
(``mamba_state``: 32 sessions by default, ~137 MB per slot in fp32, tunable
with ``max_slots``), the Mamba-2 resource planned on it
(``mamba``) and the agent-text sampler (``nano_sampler``). Its decode step is
captured as a CUDA graph for batch sizes 1 to 32.

Talker state and sampling
-------------------------

``eartts_talker`` advances every live session in one backbone pass per frame through the engine's paged KV (``talker_kv`` / ``talker_attn`` / ``talker_pos``): each session owns two streams, the conditional and the unconditional CFG context, planned together under one combined label, so a step is one packed batch of two tokens per session (a session's first step prefills its 38-token speaker/text warm-up instead). The pool's head dim is 128: FlashInfer computes 64/128/256 exactly and returns wrong values for the talker's 72, so q/k/v are zero-padded (exact) and RoPE is applied in torch from the positions the position resource planned.

Sampling noise is drawn per row from the request's seeded generator before the step, so a session's speech does not depend on which other sessions share it. The per-frame step is captured as a CUDA graph for 1 to 32 sessions; a session's first step (its 38-token speaker warm-up) runs eager. The talker KV pool is 1025 pages of 128 positions (28 GiB): 32 sessions x 2 streams x 16 pages plus a sink page, enough for ``max_session_s`` (160 s) per session, with no sliding-window eviction yet; longer audio is refused (see Session and input limits).

Encoder numerical differences
-----------------------------

Encoder numerics, measured on CPU at the checkpoint's dimensions with random weights on the duplex encoder the shared FastConformer replaced, and not since: an upload was one encoder pass, bit for bit that encoder's offline pass in fp32 and under bf16 autocast, save the last frame when the end of input reaches the encoder after the audio (about 1e-6 in fp32). 80 ms appends put the frames within about 2e-6 of offline in fp32 and about 0.7% under bf16 autocast. The shared encoder's stream attention adds the relative-position term before scaling and its offline attention after, so an upload is no longer expected to match the offline pass bit for bit.

From those figures, live and one-shot text should agree token for token in fp32 (inferred, not measured on served text); under the served bf16 a different chunking can flip a knife-edge frame such as the agent's turn start. Under bf16 autocast the stream node keeps the blocks' residual in fp32, as the replaced encoder did from its second layer on; the offline FastConformer encoders keep theirs in bf16. Against an fp32 run on 13 clips (one H200, real checkpoint, 2026-10-29), live frames have a mean per-frame relative error of 1.35%, level with the replaced encoder's 1.39%, where a bf16 residual gave 1.63%; uploads and offline passes are as close as the replaced encoder's too.

Streaming encoder state
-----------------------

The encoder keeps each call's state in a slot per session of three resources: its attention keys and values in a bounded KV (``encoder_kv``), which keeps a call's last 70 frames per layer and runs the attention, each new frame over the 70 before it, with the relative-position term as a band of distances; its conv inputs (``encoder_state``); and its newest audio and mel frames (``encoder_front``), the last two recurrent-state pools. Every live step, one frame (1280 samples) whatever the client's append size, runs whole as a CUDA graph; other steps take the front end eagerly, then the layer stack as a graph when each got at most one frame (the call's end), or eagerly (a one-shot upload, an opening clip, the call's last append: a chunk over a second is not batched with other calls, so the engine runs each call of that step on its own).
