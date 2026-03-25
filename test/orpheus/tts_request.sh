#!/bin/bash

# Quick curl test for Orpheus TTS.
# Streams NDJSON with base64-encoded PCM audio chunks.

URL="${1:-http://127.0.0.1:8000/generate}"

curl -v -X POST "$URL" \
  -F 'text=Hello, how are you doing today?' \
  -F 'output_modalities=audio' \
  -F 'model_kwargs={"voice": "tara"}'
