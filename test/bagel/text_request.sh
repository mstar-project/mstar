#!/bin/bash

curl -v -X POST http://0.0.0.0:8000/generate \
  -F 'text=Hello, how are you?'