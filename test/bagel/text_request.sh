#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

curl -v -X POST http://${HOST}:${PORT}/generate \
  -F 'text=Hello, how are you?'
