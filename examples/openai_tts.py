"""Text-to-speech via the official OpenAI SDK, pointed at an mstar server.

    pip install openai
    mstar serve orpheus
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with client.audio.speech.with_streaming_response.create(
    model="orpheus",
    input="Hello there, this is M star speaking.",
    voice="tara",
    response_format="wav",
) as resp:
    resp.stream_to_file("out.wav")
print("wrote out.wav")
