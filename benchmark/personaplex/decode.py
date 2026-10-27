"""The PersonaPlex decoder node driven outside a server, chunk by chunk, as the codec partition drives it."""

from types import SimpleNamespace

import torch

from mstar.model.personaplex import config as cfg


def schedule_sizes(total: int) -> list[int]:
    """The chunk sizes the stream releases for ``total`` frames (the last one short), plus an empty final
    chunk when ``total`` ends on a chunk boundary."""
    from mstar.model.personaplex.submodules import MimiDecoderSubmodule

    sizes, done = [], 0
    while done < total:
        full = MimiDecoderSubmodule.block_frames(done)
        sizes.append(min(full, total - done))
        done += sizes[-1]
    if not sizes or sizes[-1] == full:  # a full last chunk leaves before the stream ends
        sizes.append(0)
    return sizes


def stream_decode(decoder, frames, info, rid="decode", sizes=None):
    """Feed ``frames`` (each ``[1, 8, 1]``) through the decoder node in chunks of ``sizes`` (default: the
    stream's schedule); the last chunk is the final one. Returns the joined PCM."""
    sizes = schedule_sizes(len(frames)) if sizes is None else sizes
    assert sum(sizes) == len(frames), (sizes, len(frames))
    engine, pieces, start = SimpleNamespace(request_ids=[rid]), [], 0
    for i, size in enumerate(sizes):
        group = frames[start : start + size]
        start += size
        inputs = {"agent_frame": [torch.stack(group) if len(group) > 1 else group[0]]} if group else {}
        node = decoder.prepare_inputs(cfg.CODEC_WALK, info, inputs, is_final_stream_chunk=i == len(sizes) - 1)
        if node is None:
            continue
        staged = decoder.preprocess(cfg.CODEC_WALK, engine, [node])
        pieces.append(decoder.forward(cfg.CODEC_WALK, engine, **staged)["audio_output"][0])
    decoder.cleanup_request(rid)
    return torch.cat(pieces, -1)
