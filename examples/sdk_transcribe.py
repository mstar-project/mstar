"""Speech-to-text via the mstar Python SDK.

    mstar serve whisper_large_v3_turbo      # or qwen3_asr
"""

import sys

from mstar import MStarClient

client = MStarClient("http://localhost:8000")
path = sys.argv[1] if len(sys.argv) > 1 else "speech.wav"

print(client.transcribe(path, language="en"))
# let the model detect the language, and hand it prior context
print(client.transcribe(path, initial_prompt="Names: Ada Lovelace, Grace Hopper."))
