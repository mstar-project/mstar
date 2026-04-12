#!/bin/bash

# Quick curl test for Orpheus TTS streaming audio generation.
# Saves the raw NDJSON response and extracts audio to a WAV file.

URL="${1:-http://127.0.0.1:20001/generate}"
OUTPUT="${2:-output.wav}"

TMPFILE=$(mktemp /tmp/orpheus_response.XXXXXX)
trap "rm -f $TMPFILE" EXIT

curl -s -X POST "$URL" \
  -F 'text=And now for something completely different.' \
  -F 'output_modalities=audio' \
  -F 'model_kwargs={"voice": "tara", "max_output_tokens": 1000}' \
  -o "$TMPFILE"

python3 -c "
import base64, json, struct, sys

pcm = b''
count = 0
for line in open('$TMPFILE'):
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        continue
    if msg.get('modality') != 'audio':
        continue
    data = msg.get('data', '')
    if not data:
        continue
    pcm += base64.b64decode(data)
    count += 1

if not pcm:
    print('No audio data received.')
    sys.exit(1)

sr, ch, sw = 24000, 1, 2
with open('$OUTPUT', 'wb') as f:
    f.write(b'RIFF')
    f.write(struct.pack('<I', 36 + len(pcm)))
    f.write(b'WAVE')
    f.write(b'fmt ')
    f.write(struct.pack('<IHHIIHH', 16, 1, ch, sr, sr * ch * sw, ch * sw, sw * 8))
    f.write(b'data')
    f.write(struct.pack('<I', len(pcm)))
    f.write(pcm)

print(f'Received {count} chunks, {len(pcm)} bytes, {len(pcm) / (sr * sw):.2f}s of audio')
print(f'Saved to $OUTPUT')
"
