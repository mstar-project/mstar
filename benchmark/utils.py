import struct


SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes


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