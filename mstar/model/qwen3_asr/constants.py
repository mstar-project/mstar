"""Qwen3-ASR's output format, shared by the model and the API server's adapter
(which imports this without loading the model's modules)."""

# Qwen3-ASR writes ``language {Name}<asr_text>{text}`` (or the text alone when
# the language is forced)
ASR_TEXT_TAG = "<asr_text>"
