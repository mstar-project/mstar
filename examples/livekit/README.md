# LiveKit voice agent on M\*

A [LiveKit Agents](https://github.com/livekit/agents) voice agent whose speech-to-text, LLM and text-to-speech all run on M\*, through LiveKit's stock OpenAI plugins. Nothing LiveKit-specific runs inside M\*: each plugin's `base_url` points at an M\* server.

```
browser / phone ── LiveKit server ── agent.py (livekit-agents)
                                       ├─ STT  openai.STT  ── mstar whisper_large  /v1/audio/transcriptions
                                       ├─ LLM  openai.LLM  ── mstar qwen3_5_4b     /v1/chat/completions
                                       └─ TTS  openai.TTS  ── mstar kokoro         /v1/audio/speech
```

M\* serves one model per process, so the pipeline is three M\* servers. Turn detection runs in the agent (Silero VAD).

## Run it

1. **M\* servers**, one per model. They fit on one GPU together: with the default configs on a B200 they take about 68 GB (Whisper 13 GB, Qwen3.5 32 GB, Kokoro 24 GB). Qwen3-TTS takes 46 GB in Kokoro's place.

   ```bash
   mstar serve whisper_large --port 8001 --host localhost
   mstar serve qwen3_5_4b --config configs/qwen3_5_4b.yaml --port 8002 --host localhost
   mstar serve kokoro --port 8003 --host localhost      # or: mstar serve qwen3_tts --port 8003 ...
   ```

   Servers on one host need their own `--socket-path-prefix`. `mstar serve` and `mstar/api_server/entrypoint.py` derive it from `--port`, so distinct ports are enough.

2. **A LiveKit server.** For local development, the [release binary](https://github.com/livekit/livekit/releases) in dev mode (API key `devkey`, secret `secret`):

   ```bash
   livekit-server --dev      # serves ws://localhost:7880
   ```

3. **The agent**, in its own environment:

   ```bash
   pip install "livekit-agents[openai,silero]~=1.8.7"
   export LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret
   python examples/livekit/agent.py start
   ```

   On a host whose CPU is busy with other work, the worker reports itself as full and the LiveKit server stops dispatching rooms to it. Set `AGENT_MAX_JOBS=16` (or your target) to cap by conversations instead.

   The defaults match the ports above. `MSTAR_STT_URL`, `MSTAR_LLM_URL`, `MSTAR_TTS_URL`, `MSTAR_*_MODEL` and `MSTAR_TTS_VOICE` override them. For Qwen3-TTS set `MSTAR_TTS_MODEL=qwen3_tts MSTAR_TTS_VOICE=ryan`.

4. **Talk to it** from any LiveKit client, for example the [Agents Playground](https://agents-playground.livekit.io) pointed at your server, or with the scripted participant below.

## Settings that matter

- `turn_detection="vad"`. LiveKit 1.8's default turn detector is a LiveKit Cloud model. Without Cloud credentials every turn waits out the 2.5 s maximum endpointing delay.
- `use_realtime=False` on the STT. M\*'s Whisper serves the batch endpoint. The plugin's realtime transcription speaks a session protocol that M\* does not serve.
- `extra_body={"enable_thinking": False}` on the LLM. Qwen3.5's chat template opens a `<think>` block by default; the plugin strips it, but nothing is spoken until it closes.
- `response_format="pcm"` on the TTS. The plugin reads bare PCM as 24 kHz mono, which is what Kokoro and Qwen3-TTS produce. For a model with another output rate use `wav`, which the plugin decodes and resamples.
- The TTS plugin asks for streaming with OpenAI's `stream_format`, so the first audio of each sentence leaves M\* as soon as it is synthesized.
- A TTS failure after a sentence's audio has started arrives as an in-band SSE `error` event, which the plugin ignores: the speech just stops early and the agent logs nothing. The M\* TTS server's log has the reason.
- Barge-in. Self-hosted, LiveKit interrupts on VAD and needs 0.5 s of user speech first. In the runs below the agent went quiet 0.7 to 0.8 s (p50) after the user started talking over it, and on 4 of the 16 turns that talked over it the agent was still speaking when the user finished. `TurnHandlingOptions(interruption={"min_duration": ...})` tunes this. M\* is not on that path: the agent stops its own playout.
- LiveKit's [audio turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/) (waits through mid-sentence pauses) and [adaptive interruption handling](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/) (tells a backchannel such as "uh-huh" from a barge-in) are set on the agent's `AgentSession`, outside M\*, and LiveKit documents both for an STT, LLM and TTS pipeline like this one. Their full models run on LiveKit Inference with LiveKit Cloud credentials (self-hosted use counts toward a monthly allowance); the turn detector's `v1-mini` runs on CPU in the agent. This example does not enable them, and the runs below were measured without them.

## Scripted participant

`participant.py` joins a room as a user, plays WAV utterances into its microphone in real time, and times the agent. It reads the agent's `lk.agent.state` attribute and both sides' transcripts.

Any short spoken questions work as utterances (16-bit PCM WAV, any rate). To synthesize some with the Kokoro server, in a voice unlike the agent's:

```bash
mkdir -p utts && i=0
for q in "What is the tallest mountain in the world?" "How far away is the moon?" \
         "Tell me a fun fact about octopuses." "Why is the sky blue?"; do
  i=$((i + 1))
  curl -s localhost:8003/v1/audio/speech -H 'Content-Type: application/json' -o "utts/$i.wav" \
    -d "{\"model\": \"kokoro\", \"input\": \"$q\", \"voice\": \"am_michael\", \"response_format\": \"wav\"}"
done
```

```bash
python examples/livekit/participant.py --utterances 'utts/*.wav' --conversations 1 --turns 6
python examples/livekit/participant.py --utterances 'utts/*.wav' --conversations 16 --scenario mix
```

Scenarios: `turns` (question and answer), `bargein` (talk over the reply), `cancel` (interrupt several times in a row, then let it answer), `disconnect` (leave mid-reply), and `mix` (all four, round robin). It prints p50 and p90 turn latency: from the end of the user's speech to the agent's first audible audio, which is what a caller hears.

Measured on one B200 with the three M\* servers, the LiveKit server, the agent and the participant all on it, on this branch. The user audio is synthesized speech, played in real time. `mix` runs `turns`, `bargein`, `cancel` and `disconnect` conversations side by side.

| TTS | Run | Conversations completed | Turn latency p50 / p90 | After endpointing p50 / p90 |
|---|---|---|---|---|
| Kokoro | 6 turns, twice | 2 / 2 | 1.35 / 1.39 s | 0.50 / 0.51 s |
| Kokoro | `mix`, 8 at once | 7 / 8 | 1.35 / 1.41 s | 0.49 / 0.55 s |
| Kokoro | `mix`, 16 at once | 14 / 16 | 1.37 / 1.48 s | 0.47 / 0.53 s |
| Qwen3-TTS | 5 turns | 1 / 1 | 1.48 / 1.76 s | 0.62 / 0.87 s |
| Qwen3-TTS | `mix`, 8 at once | 7 / 8 | 1.49 / 1.79 s | 0.60 / 0.91 s |

A barge-in counts only if the agent leaves `speaking` and its audio stops before the user finishes. The four incomplete conversations are barge-ins with the shortest utterance (1.4 to 1.5 s), where the agent was still `speaking` when the user finished: in two LiveKit stopped it 0.3 s later, and the other two ended with it still speaking. Across the stress runs every M\* request was torn down and none failed, and GPU memory stayed flat.

"After endpointing" starts when the agent decides the user is done. About 0.68 s of each turn passes before that decision: the VAD's silence window, then Whisper's transcript, which LiveKit waits for. The VAD and `endpointing` settings in `agent.py` tune it. M\*'s own calls in a Kokoro turn are short: Whisper takes 0.10 s on 2.6 s of speech, Qwen3.5's first token takes 24 ms, and each sentence's first audio takes 36 ms (37 ms with Qwen3-TTS). These are p50 values from LiveKit's metrics. The rest is the LLM writing its first sentence, plus LiveKit's audio path.

## Known limits

- **No tool calling:** a chat with `tools` (any `@function_tool`) gets a 400.
- **System messages only at the start (Qwen3.5):** Qwen3.5's chat template refuses a system message after the first turn, and a chat with no user turn, so `session.generate_reply(instructions=...)` gets a 400 both as a greeting before the user speaks and mid-conversation. Greet with `session.say(...)`, or pass `user_input=` instead.
- **No speech-to-speech:** `openai.realtime.RealtimeModel` needs an OpenAI Realtime session on `/v1/realtime`, which M\* does not serve yet.
