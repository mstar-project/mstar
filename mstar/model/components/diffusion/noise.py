"""Staging a request's seeded noise to the device without stalling the caller.

A request's initial noise is drawn on a CPU generator (``randn_tensor`` parity with
the reference pipelines fixes that), so it has to cross to the device before the
first denoise step. The obvious ``tensor.to(device)`` copies from *pageable* host
memory, and such a copy is synchronous with respect to the host: the calling thread
blocks until it retires. On the GPU thread -- where ``prepare_inputs`` runs -- that
is a pipeline bubble, not just bandwidth: the pre-plan cannot run ahead and the copy
cannot overlap the previous step's postprocess.

``NoiseStager`` removes the stall. It keeps a small ring of pinned host buffers;
``to_device`` memcpys the tensor into the next free one and issues a single
``non_blocking`` H2D from it, so the caller continues while the copy is in flight.
The copy is stream-ordered, so kernels issued afterwards on the same stream see the
finished tensor without any explicit wait.

Reusing a pinned buffer whose copy has not retired would let the source change
underneath an in-flight DMA -- silent, data-dependent corruption of exactly the kind
that only shows up under concurrency. Each buffer therefore records an event after
its copy and is not rewritten until that event has completed.

The staging is deliberately the *last* step: a model's ``seed_loop_back`` may
post-process the draw (klein packs ``[C, h, w]`` to tokens), and doing that on the
host keeps the device tensor bit-identical to what the pageable path produced.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

import torch

from mstar.utils.device import DeviceCapability


class NoiseStager:
    """A ring of pinned host buffers for one dtype.

    Not model-specific: anything that has a host tensor and needs it on the device
    without a host stall can use it. One instance per (dtype) per submodule; it is
    thread-safe.
    """

    def __init__(self, dtype: torch.dtype, depth: int = 4, numel: int = 0):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.dtype = dtype
        self._depth = depth
        self._numel = 0
        self._bufs: list[torch.Tensor] = []
        self._events: list[torch.Event | None] = [None] * depth
        self._next = 0
        self._lock = threading.Lock()
        # Without CUDA there is nothing to pin and nothing to overlap; the staging
        # degrades to a plain copy so callers need no branch of their own.
        self._pinned = torch.accelerator.is_available()
        if numel:
            self._grow(numel)

    # ------------------------------------------------------------------ buffers
    def _grow(self, numel: int) -> None:
        # Nothing may still be copying out of the buffers being replaced.
        for event in self._events:
            if event is not None:
                event.synchronize()
        self._numel = max(numel, 2 * self._numel, 1)
        self._bufs = [
            torch.empty(self._numel, dtype=self.dtype, pin_memory=self._pinned)
            for _ in range(self._depth)
        ]
        self._events = [None] * self._depth
        self._next = 0

    # ---------------------------------------------------------------- staging
    def _take(self, numel: int, device: torch.device):
        """Next ring slot whose previous copy has retired. Caller holds the lock."""
        index = self._next
        self._next = (index + 1) % self._depth
        event = self._events[index]
        if event is not None:
            # The previous copy out of this buffer must have retired before the
            # write below changes what it is reading.
            event.synchronize()
        return index, event, self._bufs[index][:numel]

    def _record(self, index: int, event, device: torch.device) -> None:
        if event is None:
            event = self._events[index] = torch.Event(device=device)
        event.record(torch.accelerator.current_stream(device))

    def to_device(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """``tensor`` on ``device``, copied from pinned memory without blocking.

        The returned tensor is only valid on the stream the copy was issued on,
        which is the current stream of ``device``. Reading it from another stream
        needs that stream to wait on this one, as for any async copy.
        """
        device = torch.device(device)
        if not DeviceCapability.of(device).is_accelerator or not self._pinned:
            return tensor.to(device)
        if tensor.dtype != self.dtype:
            raise ValueError(f"stager holds {self.dtype}, got {tensor.dtype}")

        source = tensor if tensor.is_contiguous() else tensor.contiguous()
        numel = source.numel()
        out = torch.empty(source.shape, dtype=self.dtype, device=device)
        if numel == 0:
            return out

        with self._lock:
            if numel > self._numel:
                self._grow(numel)
            index, event, staging = self._take(numel, device)
            # One memcpy on the calling thread: copy_ would split it over torch's
            # intra-op pool, where a descheduled thread holds up the GPU thread.
            staging.view(torch.uint8).numpy()[:] = source.view(-1).view(torch.uint8).numpy()
            out.view(-1).copy_(staging, non_blocking=True)
            self._record(index, event, device)
        return out

    def stage_all(
        self, tensors: Mapping[str, torch.Tensor], device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """:meth:`to_device` over a mapping, for a node's whole seed set."""
        return {name: self.to_device(tensor, device) for name, tensor in tensors.items()}
