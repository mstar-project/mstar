"""Which stream the model forward runs on; should not run on the default
stream because under the TCP and RDMA transports, a rank waiting for a
collective on the default stream blocks its peers from reading out of its
tensor store, which can cause deadlock with TP.

The cost is that the default stream stops being the thing to order against.
Anything outside the GPU thread that has to see the forward's writes, or be
seen by them, names this stream instead: the H2D of a step's inputs, a KV
reload, an event recorded to fence a peer's read.
"""

import torch

# Set once, by the worker that owns the device. Module-level because the code
# that has to order against the forward is spread across the engine, the KV
# resources and the transports, and none of them holds a worker reference.
_compute_stream: "torch.cuda.Stream | None" = None


def set_compute_stream(stream: "torch.cuda.Stream | None") -> None:
    """Declare the stream this process submits its forward passes on."""
    global _compute_stream
    _compute_stream = stream


def compute_stream(device=None) -> "torch.cuda.Stream":
    """The stream to order against.

    Falls back to the calling thread's current stream when no forward stream
    has been declared — which keeps every non-worker process (the API server,
    unit tests, CPU-only runs) on exactly its old behaviour.
    """
    if _compute_stream is not None:
        return _compute_stream
    return torch.cuda.current_stream(device)
