"""CUDA-graph replay of S3Gen's stages: the flow solve, the flow encoder and the vocoder.

The flow-matching estimator is 56 small transformer blocks of width 256 over
a few hundred mel frames. One Euler step launches several hundred kernels of
a few microseconds each, so a solve (10 steps, guidance doubling the rows)
is bound by kernel launches rather than by arithmetic, whatever the batch;
the conformer encoder in front of it and the HiFT vocoder behind it have the
same profile at chunk sizes. Capturing each stage as one CUDA graph per
input shape and replaying it removes that overhead.

``ShapeGraphs`` holds the mechanics: static input buffers per shape, capture
on first use (or ahead of time), least-recently-used eviction past
``max_graphs``, one memory pool shared by every graph (they never run
concurrently and every static input and output stays allocated, so no replay
can write into another graph's memory), and an eager fallback off the GPU
or after a failed capture. The subclasses decide the shape key: the solve
pads request rows to a captured row count, the encoder pads rows and tokens
to a bucket, the vocoder runs one request at its exact length.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

from mstar.engine.cuda_graph_runner import capture_into_graph

logger = logging.getLogger(__name__)

Tensors = dict[str, torch.Tensor]
Output = torch.Tensor | tuple[torch.Tensor, ...]


@dataclass
class _Entry:
    graph: object  # torch.cuda.CUDAGraph (or a stand-in with ``replay``)
    inputs: Tensors  # static buffers the graph reads
    output: Output  # static tensors the graph writes


def _clone(output: Output) -> Output:
    if isinstance(output, tuple):
        return tuple(t.clone() for t in output)
    return output.clone()


class ShapeGraphs:
    """Replays ``fn(tensors, extra)`` from a captured CUDA graph per input shape.

    ``fn`` takes a dict of same-device tensors plus a tuple of hashable
    extras (step counts, flags) and returns a tensor or a tuple of tensors; it
    must not draw random numbers or synchronise. Subclasses may pad the
    tensors to a captured shape (``pad``) and cut the result back (``trim``).
    """

    def __init__(self, fn: Callable[[Tensors, tuple], Output], *, max_graphs: int = 64, capture_after: int = 1):
        self._fn = fn
        self.max_graphs = int(max_graphs)
        # a shape is captured on its ``capture_after``-th use: 1 captures at
        # once, 2 leaves shapes that never recur (an utterance's last chunk
        # has its own length) to the eager path instead of paying a capture
        # for a graph nobody replays
        self.capture_after = max(1, int(capture_after))
        self._seen: dict[tuple, int] = {}
        self._graphs: OrderedDict[tuple, _Entry] = OrderedDict()
        self._pool = None
        self.captures = 0
        self.replays = 0
        self.disabled = False

    def __len__(self) -> int:
        return len(self._graphs)

    def keys(self) -> list[tuple]:
        return list(self._graphs)

    # -- shape policy (subclasses) ------------------------------------------

    def pad(self, tensors: Tensors, extra: tuple) -> Tensors | None:
        """The tensors brought to a captured shape, or None to run eagerly."""
        return tensors

    def trim(self, output: Output, tensors: Tensors) -> Output:
        """The result cut back to the caller's shape."""
        return output

    @staticmethod
    def key_of(tensors: Tensors, extra: tuple) -> tuple:
        return (tuple((name, tuple(t.shape)) for name, t in tensors.items()), extra)

    # -- replay ------------------------------------------------------------

    @torch.no_grad()
    def run(self, tensors: Tensors, extra: tuple = ()) -> Output:
        if self.disabled or not self._capturable(tensors):
            return self._fn(tensors, extra)
        padded = self.pad(tensors, extra)
        if padded is None:
            return self._fn(tensors, extra)
        key = self.key_of(padded, extra)
        entry = self._graphs.get(key)
        if entry is None:
            if self.capture_after > 1:
                seen = self._seen.get(key, 0) + 1
                self._seen[key] = seen
                if seen < self.capture_after:
                    return self._fn(tensors, extra)
                del self._seen[key]
            try:
                entry = self._capture(key, padded, extra)
            except Exception:
                logger.warning("%s: capturing shape %s failed; running eagerly from now on",
                               type(self).__name__, key, exc_info=True)
                self.disabled = True
                return self._fn(tensors, extra)
        else:
            self._graphs.move_to_end(key)
            self._load(entry, padded)
        entry.graph.replay()
        self.replays += 1
        return self.trim(_clone(entry.output), tensors)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _capturable(tensors: Tensors) -> bool:
        return next(iter(tensors.values())).is_cuda

    @staticmethod
    def _load(entry: _Entry, tensors: Tensors) -> None:
        for name, value in tensors.items():
            entry.inputs[name].copy_(value)

    def _capture(self, key: tuple, tensors: Tensors, extra: tuple) -> _Entry:
        while len(self._graphs) >= self.max_graphs:
            self._graphs.popitem(last=False)
        static = {name: value.clone() for name, value in tensors.items()}

        def run() -> Output:
            return self._fn(static, extra)

        graph, output = self._record(run, next(iter(static.values())).device)
        entry = _Entry(graph=graph, inputs=static, output=output)
        self._graphs[key] = entry
        self.captures += 1
        return entry

    def _record(self, run: Callable[[], Output], device: torch.device):
        """Warm ``run`` up on a side stream (cuBLAS/cuDNN pick their kernels;
        twice before the very first capture), then capture it into the shared
        pool with the engine's capture helper, which undoes a failed capture's
        allocator and stream state."""
        with torch.cuda.device(device):
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2 if self.captures == 0 else 1):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize(device)
            return capture_into_graph(run, self._pool, device, autocast_dtype=None)


def _rows_for(rows: Sequence[int], batch: int) -> int | None:
    for r in rows:
        if r >= batch:
            return r
    return None


def _pad_rows(tensors: Tensors, rows: int) -> Tensors:
    """Zero rows appended so every tensor has ``rows`` on its first dim."""
    out = {}
    for name, value in tensors.items():
        if value.shape[0] == rows:
            out[name] = value
        else:
            padded = value.new_zeros((rows, *value.shape[1:]))
            padded[: value.shape[0]] = value
            out[name] = padded
    return out


class SolveGraphs(ShapeGraphs):
    """One graph per ``(rows, frames, steps)`` for the whole flow solve.

    ``solve`` takes ``(mu, mask, spks, cond, noise, n_timesteps)`` (batch first,
    mel frames last) and returns the solved mel. Frames are already padded to
    a bucket by ``S3Gen.tokens_to_mel_rows``; request rows are padded here to
    the next captured row count (zero rows behind a zero mask, cut off the
    result). A batch beyond the largest row count runs eagerly.
    """

    INPUTS = ("mu", "mask", "spks", "cond", "noise")

    def __init__(self, solve: Callable[..., torch.Tensor], *, rows: Sequence[int] = (1, 2, 4, 8), max_graphs: int = 64):
        if not rows or min(rows) < 1:
            raise ValueError("rows must be positive row counts")
        self.rows = tuple(sorted(set(int(r) for r in rows)))

        def fn(tensors: Tensors, extra: tuple) -> torch.Tensor:
            return solve(*(tensors[name] for name in self.INPUTS), extra[0])

        super().__init__(fn, max_graphs=max_graphs)

    def rows_for(self, batch: int) -> int | None:
        """The captured row count a ``batch``-row solve is padded to, or None."""
        return _rows_for(self.rows, batch)

    @staticmethod
    def key_of(tensors: Tensors, extra: tuple) -> tuple:
        """``(rows, frames, steps)``: the channel dims are fixed by the model."""
        return (tensors["mu"].shape[0], tensors["mu"].shape[-1], extra[0])

    def pad(self, tensors: Tensors, extra: tuple) -> Tensors | None:
        rows = self.rows_for(tensors["mu"].shape[0])
        return None if rows is None else _pad_rows(tensors, rows)

    def trim(self, output: Output, tensors: Tensors) -> Output:
        return output[: tensors["mu"].shape[0]]

    def __call__(
        self, mu: torch.Tensor, mask: torch.Tensor, spks: torch.Tensor, cond: torch.Tensor,
        noise: torch.Tensor, n_timesteps: int,
    ) -> torch.Tensor:
        tensors = dict(zip(self.INPUTS, (mu, mask, spks, cond, noise), strict=True))
        return self.run(tensors, (int(n_timesteps),))

    @torch.no_grad()
    def warmup(self, frames: Iterable[int], n_timesteps: int, example: Tensors) -> int:
        """Capture every ``(rows, frames)`` shape ahead of time from ``example``
        inputs (one row each, frames ≥ every requested length; the head is used).
        Returns how many graphs were captured. Inference only, like ``run``:
        with autograd on, the warm-up solves would keep every activation of
        the 560 estimator blocks alive (tens of GB for a few rows)."""
        before = self.captures
        if not self._capturable(example):
            return 0
        for f in sorted(set(int(x) for x in frames)):
            for rows in self.rows:
                if self.disabled:
                    break
                one = {name: (t[:1] if name == "spks" else t[:1, :, :f]) for name, t in example.items()}
                padded = _pad_rows(one, rows)
                key = self.key_of(padded, (int(n_timesteps),))
                if key in self._graphs:
                    continue
                try:
                    self._capture(key, padded, (int(n_timesteps),))
                except Exception:
                    logger.warning("SolveGraphs: capturing shape %s failed; graphs stay off", key, exc_info=True)
                    self.disabled = True
        return self.captures - before


class EncoderGraphs(ShapeGraphs):
    """One graph per ``(rows, tokens)`` bucket for the flow token encoder.

    Rows are padded to a captured row count and the token axis to a multiple
    of ``token_bucket`` (the frame bucket over the token-to-mel ratio, so the
    frames the encoder returns are already bucketed). Padding rows carry
    length 0 and padding tokens fall behind each row's length; the encoder
    zeroes and masks them, so valid rows and positions are unaffected.
    """

    def __init__(
        self, encoder: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]], *,
        rows: Sequence[int] = (1, 2, 4, 8), token_bucket: int = 32, max_graphs: int = 64,
    ):
        if token_bucket < 1:
            raise ValueError("token_bucket must be at least 1")
        self.rows = tuple(sorted(set(int(r) for r in rows)))
        self.token_bucket = int(token_bucket)

        def fn(tensors: Tensors, extra: tuple) -> tuple[torch.Tensor, torch.Tensor]:
            return tuple(encoder(tensors["tokens"], tensors["lens"]))

        super().__init__(fn, max_graphs=max_graphs)

    def pad(self, tensors: Tensors, extra: tuple) -> Tensors | None:
        tokens, lens = tensors["tokens"], tensors["lens"]
        rows = _rows_for(self.rows, tokens.shape[0])
        if rows is None:
            return None
        length = -(-tokens.shape[1] // self.token_bucket) * self.token_bucket
        padded_tokens = tokens.new_zeros((rows, length))
        padded_tokens[: tokens.shape[0], : tokens.shape[1]] = tokens
        padded_lens = lens.new_zeros((rows,))
        padded_lens[: lens.shape[0]] = lens
        return {"tokens": padded_tokens, "lens": padded_lens}

    def trim(self, output: Output, tensors: Tensors) -> Output:
        batch = tensors["tokens"].shape[0]
        return tuple(t[:batch] for t in output)

    def __call__(self, tokens: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.run({"tokens": tokens, "lens": lens})


class VocoderGraphs(ShapeGraphs):
    """One graph per exact ``(frames, with cache)`` for the HiFT vocoder up to
    its output spectrum (``HiFTGenerator.spectrum``); the inverse STFT runs
    eagerly behind the replay because ``torch.istft`` synchronises.

    The vocoder runs one request at a time behind its own excitation cache,
    so nothing is padded: the mel's frame count and whether a cache is
    present make the key. The excitation noise is drawn outside (see
    ``HiFTGenerator.draw_noise``) and comes in as a tensor input.
    """

    def __init__(
        self, spectrum: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]], *, max_graphs: int = 128,
        capture_after: int = 2,
    ):
        def fn(tensors: Tensors, extra: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return tuple(spectrum(tensors["mel"], tensors["phase"], tensors["harmonic"], tensors.get("cache")))

        # the ramp's chunk lengths recur across requests and get captured on
        # their second use; a final chunk's length is usually unique and stays eager
        super().__init__(fn, max_graphs=max_graphs, capture_after=capture_after)

    def __call__(
        self, mel: torch.Tensor, phase: torch.Tensor, harmonic: torch.Tensor, cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = {"mel": mel, "phase": phase, "harmonic": harmonic}
        if cache is not None and cache.shape[-1] > 0:
            tensors["cache"] = cache
        return self.run(tensors)
