#!/bin/bash

# Quick curl test for Orpheus TTS.
# Streams NDJSON with base64-encoded PCM audio chunks.

curl -v -X POST http://0.0.0.0:8000/generate \
  -F 'text=Hello, how are you doing today?' \
  -F 'output_modalities=audio' \
  -F 'model_kwargs={"voice": "tara"}'
