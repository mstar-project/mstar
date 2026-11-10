"""Step-Audio2's settings for the shared token2wav (``mstar/model/components/token2wav``).

The network is the same as MiniCPM-o's; the reference streams it differently
(``Token2WavStreaming``): the voice's caches are set up with the prompt's own first 3 codes
as lookahead (``token2wav.py:106-108``), a reply's windows start on its own codes with no
lead silence and no zeros before the first window's samples (``examples-vllm-stream.py``,
``token2wav.py:143-162``), and only the DiT's cache is cut (``token2wav.py:137-141``), so the
conformer's grow with the reply.

The conformer's caches are sized to ``CONFORMER_REPLY_TOKENS`` codes of reply. The reference
cannot go much further either: its relative position table has 5000 entries a side
(``EspnetRelPositionalEncoding``, ``max_len=5000``, never extended on the chunked path), and
the 50 Hz conformer reads ``2 (P + written) + 56`` of them, so a voice of ``P`` tokens and a
reply of ``written`` codes must stay under ~2470 tokens together. 2048 codes (82 s of speech)
leaves room for voices up to 16 s.
"""
from __future__ import annotations

from mstar.model.components.token2wav.model import Token2WavStreaming

CONFORMER_REPLY_TOKENS = 2048

T2W_STREAMING = Token2WavStreaming(
    prompt_lookahead=True,
    lead_silence=0,
    first_window_zeros=False,
    conformer_reply_tokens=CONFORMER_REPLY_TOKENS,
)
