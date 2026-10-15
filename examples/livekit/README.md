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

   Servers on one host need their own `--socket-path-prefix`. `mstar serve` derives it from `--port`. If you launch `mstar/api_server/entrypoint.py` directly, pass a distinct prefix to each server.

2. **A LiveKit server.** For local development, the [release binary](https://github.com/livekit/livekit/releases) in dev mode (API key `devkey`, secret `secret`):

   ```bash
   livekit-server --dev      # serves ws://localhost:7880
   ```

3. **The agent**, in its own environment:

   ```bash
   pip install "livekit-agents[openai,silero]~=1.8"
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
- Barge-in. Self-hosted, LiveKit interrupts on VAD and needs 0.5 s of user speech first. In the runs below the agent went quiet 0.4 to 1.4 s (p50) after the user started talking over it. `TurnHandlingOptions(interruption={"min_duration": ...})` tunes this. M\* is not on that path: the agent stops its own playout.

## Scripted participant

`participant.py` joins a room as a user, plays WAV utterances into its microphone in real time, and times the agent. It reads the agent's `lk.agent.state` attribute and both sides' transcripts.

```bash
python examples/livekit/participant.py --utterances 'utts/*.wav' --conversations 1 --turns 6
python examples/livekit/participant.py --utterances 'utts/*.wav' --conversations 16 --scenario mix
```

Scenarios: `turns` (question and answer), `bargein` (talk over the reply), `cancel` (interrupt several times in a row, then let it answer), `disconnect` (leave mid-reply), and `mix` (all four, round robin). It prints p50 and p90 turn latency: from the end of the user's speech to the agent's first audible audio, which is what a caller hears.

Measured on one B200 with the three M\* servers, the LiveKit server, the agent and the participant all on it. The user audio is synthesized speech, played in real time. `mix` runs `turns`, `bargein`, `cancel` and `disconnect` conversations side by side.

| TTS | Run | Conversations completed | Turn latency p50 / p90 | After endpointing p50 / p90 |
|---|---|---|---|---|
| Kokoro | 6 turns, twice | 2 / 2 | 1.37 / 1.42 s, 1.32 / 1.32 s | 0.51 / 0.52 s, 0.47 / 0.50 s |
| Kokoro | `mix`, 8 at once | 8 / 8 | 1.36 / 1.41 s | 0.50 / 0.53 s |
| Kokoro | `mix`, 16 at once | 16 / 16 | 1.43 / 1.51 s | 0.49 / 0.54 s |
| Qwen3-TTS | 5 turns | 1 / 1 | 1.51 / 1.63 s | 0.63 / 0.75 s |
| Qwen3-TTS | `mix`, 8 at once | 8 / 8 | 1.47 / 1.55 s | 0.60 / 0.68 s |

"After endpointing" starts when the agent decides the user is done. About 0.68 s of each turn passes before that decision: the VAD's silence window, then Whisper's transcript, which LiveKit waits for. The VAD and `endpointing` settings in `agent.py` tune it. M\*'s own calls in a Kokoro turn are short: Whisper takes 0.10 s on 2.6 s of speech, Qwen3.5's first token takes 23 ms, and each sentence's first audio takes 36 ms (43 ms with Qwen3-TTS). These are p50 values from LiveKit's metrics. The rest is the LLM writing its first sentence, plus LiveKit's audio path.

## Known limits

- **No tool calling:** `@function_tool` tools are accepted and ignored.
- **No speech-to-speech:** `openai.realtime.RealtimeModel` needs an OpenAI Realtime session on `/v1/realtime`, which M\* does not serve yet.
