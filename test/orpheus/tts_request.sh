#!/bin/bash

# Quick curl test for Orpheus TTS.
# Returns one final PCM audio payload after decode completes.

URL="${1:-http://127.0.0.1:20001/generate}"

curl -v -X POST "$URL" \
  -F 'text=Hello, how are you doing today?' \
  -F 'output_modalities=audio' \
  -F 'model_kwargs={"voice": "tara"}'
