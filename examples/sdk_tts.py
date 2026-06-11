"""Text-to-speech via the Python SDK.

Start a server first, e.g.:  mstar serve orpheus
"""

from mstar import MStarClient

client = MStarClient("http://localhost:8000")

audio = client.tts("Hello from M star!", voice="tara")
audio.to_wav("out.wav")
print(f"wrote out.wav — {len(audio)} samples @ {audio.sample_rate} Hz")
