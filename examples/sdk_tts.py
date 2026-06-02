"""Text-to-speech via the Python SDK.

Start a server first, e.g.:  mminf serve orpheus
"""

from mminf import MMInfClient

client = MMInfClient("http://localhost:8000")

audio = client.tts("Hello from mminf!", voice="tara")
audio.to_wav("out.wav")
print(f"wrote out.wav — {len(audio)} samples @ {audio.sample_rate} Hz")
