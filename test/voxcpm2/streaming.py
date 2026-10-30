"""Drive latent patches through the Talker -> VAEDecoder edge as the runtime does.

Patches go through a real ``StreamBuffer`` with the edge's chunk policy; each
ready chunk is decoded by ``VAEDecoderSubmodule`` (prepare, preprocess,
forward, postprocess) and the emitted audio chunks are returned in order.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mstar.model.submodule_base import InputMetadata
from mstar.model.voxcpm2.voxcpm2_model import STREAM_EDGE
from mstar.streaming.stream_buffer import StreamBuffer


def stream_through(vae_node, patches: torch.Tensor, policy, lead: torch.Tensor | None = None) -> list[torch.Tensor]:
    buffer = StreamBuffer(request_id="r", edge_name=STREAM_EDGE, from_partition="Talker", policy=policy)
    items = list(lead) if lead is not None else []
    if lead is not None:
        buffer.prime_context(len(lead))
    items += list(patches)
    info = SimpleNamespace(rid_handle="r", request_id="r")
    engine = SimpleNamespace(request_ids=["r"])
    chunks = []

    def drain():
        while buffer.has_chunk_ready():
            chunk = buffer.pop_chunk()
            prepared = vae_node.prepare_inputs(
                "vae_chunk",
                info,
                {STREAM_EDGE: [chunk.data["data"]]},
                input_metadata=InputMetadata(stream_chunks={STREAM_EDGE: chunk.info}),
            )
            out = vae_node.forward_batched("vae_chunk", engine, **vae_node.preprocess("vae_chunk", engine, [prepared]))
            vae_node.postprocess("r", None, out["r"], inputs=prepared)
            chunks.append(out["r"]["audio_chunk"][0])

    for i, item in enumerate(items):
        buffer.pre_read_register(f"t{i}")
        buffer.put(f"t{i}", item)
        drain()
    buffer.signal_done()
    drain()
    return chunks
