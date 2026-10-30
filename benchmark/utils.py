import struct

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes

# Re-tokenizing a stream's joined text drifts from what was generated: byte-level tokens come back as replacement
# characters and adjacent tokens can merge. On streams known to carry one token per chunk that was -7 to +4 tokens in
# 257 over the MI325X window cells. A chunk holds at least one token, so the chunk count is a floor, and only a count
# clearly above it means multi-token chunks.
_RETOKENIZE_DRIFT = 0.05
_RETOKENIZE_DRIFT_MIN = 2


def clearly_multi_token(tokens: int, chunks: int) -> bool:
    return tokens > chunks + max(_RETOKENIZE_DRIFT_MIN, _RETOKENIZE_DRIFT * chunks)


def _write_wav(pcm_data: bytes, path: str):
    """Write raw int16 PCM bytes to a WAV file."""
    data_size = len(pcm_data)
    byte_rate = SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH
    block_align = NUM_CHANNELS * SAMPLE_WIDTH

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))       # PCM format
        f.write(struct.pack("<H", NUM_CHANNELS))
        f.write(struct.pack("<I", SAMPLE_RATE))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", SAMPLE_WIDTH * 8))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm_data)
