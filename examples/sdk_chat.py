"""Chat with a running mstar server via the Python SDK.

Start a server first, e.g.:  mstar serve bagel
"""

from mstar import MStarClient
from mstar.client import TextChunk

client = MStarClient("http://localhost:8000")

# Non-streaming
print(client.chat("What is the capital of France?").text)

# Streaming (yields TextChunk / ImageChunk / AudioChunk)
for event in client.chat("Tell me a short story.", stream=True):
    if isinstance(event, TextChunk):
        print(event.text, end="", flush=True)
print()
