"""Speech-to-text via the official OpenAI SDK, pointed at an mstar server.

    pip install openai
    mstar serve whisper_large_v3_turbo      # or qwen3_asr
"""

import sys

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
path = sys.argv[1] if len(sys.argv) > 1 else "speech.wav"

# plain text
with open(path, "rb") as f:
    result = client.audio.transcriptions.create(model="whisper_large_v3_turbo", file=f, language="en")
print(result.text)

# segments with timestamps (the server asks the model for them)
with open(path, "rb") as f:
    verbose = client.audio.transcriptions.create(
        model="whisper_large_v3_turbo", file=f, response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
for seg in verbose.segments or []:
    print(f"[{seg.start:7.2f} - {seg.end:7.2f}] {seg.text}")

# streaming: text deltas as the decoder produces them
with open(path, "rb") as f:
    for event in client.audio.transcriptions.create(model="whisper_large_v3_turbo", file=f, stream=True):
        if event.type == "transcript.text.delta":
            print(event.delta, end="", flush=True)
print()
