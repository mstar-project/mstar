"""The streaming FastConformer encoder node: a stream of 16 kHz audio in, one
encoder frame out per 80 ms, for a cache-aware causal checkpoint with no right
context (Nemotron VoiceChat's ``[70, 0]``).

It runs the same featurizer and encoder as the whole-clip node, a chunk at a
time, and matches them over the whole stream up to float rounding: about 1e-6
in fp32, about 1% relative in bf16, where each chunking takes different
kernel shapes.

- Mel: a frame is computed once its window is complete, from the
  preemphasized signal (``NemoLogMel.windows``).
- Subsampling: the strided convs are recomputed over a window that starts 16
  mel frames before the first new output frame. An output frame reads 15 mel
  frames, so the first two outputs of the window are discarded and the rest are
  exact; the window starts on a multiple of 8 so the strides line up with the
  offline ones.
- Blocks: each new frame attends the ``left_context`` frames before it and
  itself in the bounded KV, and each depthwise conv continues from its cached
  last inputs (``FastConformerEncoder.blocks_stream``).

A stream's state is bounded however long it runs: its host counters
(``StreamState``, in its per-request state), its newest audio and mel frames
(``front_state_blocks``), the conv tails (``encoder_state_blocks``) and the
bounded KV (``encoder_kv_config``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.communication.tensors import NameToTensorList
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
from mstar.engine.resources import Segment, SubmoduleStep
from mstar.engine.resources.kv.bounded import BoundedKVConfig, BoundedKVStep, SinkWindow, StreamPosition
from mstar.engine.resources.recurrent import RecurrentStep
from mstar.engine.resources.recurrent.config import RecurrentBlockConfig
from mstar.model.components.audio_features import NemoLogMel
from mstar.model.components.fastconformer import FastConformerConfig, FastConformerEncoder
from mstar.model.submodule_base import BatchedModelOutput, NodeInputs, NodeSubmodule
from mstar.utils.h2d import to_device_nonblocking


class StreamState:
    """One stream's place in its audio: host counters, which drive every slice
    so a step needs no sync. The audio and mel frames a later step still reads
    live in the front pool (``front_state_blocks``)."""

    __slots__ = ("samples", "mels", "position")

    def __init__(self):
        self.samples = 0        # samples of the stream so far
        self.mels = 0           # mel frames computed
        # its place in the bounded KV, which a step's declarations share
        self.position = StreamPosition(source_len=0, written=0)

    @property
    def frames(self) -> int:
        """Encoder frames emitted."""
        return self.position.written

    @frames.setter
    def frames(self, value: int):
        self.position = StreamPosition(source_len=0, written=value)


def front_state_blocks(config: FastConformerConfig, hop_length: int) -> dict[str, RecurrentBlockConfig]:
    """A stream's front-end state, all a later step reads: its newest
    preemphasized samples (``sig``; every sample until its first mel frame,
    then the window of the next one), its newest mel frames (``mel``; the
    subsampling window of the next encoder frame) and its last raw sample, for
    the preemphasis. Right-aligned: the newest sample / frame is last."""
    return {
        "sig": RecurrentBlockConfig(shape=(hop_length * config.subsampling_factor,), dtype=torch.float32),
        "mel": RecurrentBlockConfig(shape=(2 * config.subsampling_factor, config.num_mel_bins), dtype=torch.float32),
        "last": RecurrentBlockConfig(shape=(1,), dtype=torch.float32),
    }


def encoder_state_blocks(
    config: FastConformerConfig, dtype: torch.dtype | None = None,
) -> dict[str, RecurrentBlockConfig]:
    """A stream's conv state, per layer: the last ``kernel - 1`` inputs of its
    depthwise conv. By default in the engine's compute dtype, which the layers
    compute it in: stored in another, a stream's frames drift from the offline ones."""
    return {"conv": RecurrentBlockConfig(shape=(config.hidden_size, config.conv_kernel_size - 1), dtype=dtype)}


def encoder_kv_config(config: FastConformerConfig, max_slots: int) -> BoundedKVConfig:
    """A stream's attention keys and values, per layer: each new frame attends
    the ``left_context`` frames before it and itself, so the KV keeps that many.
    In the engine's compute dtype, as ``encoder_state_blocks``."""
    return BoundedKVConfig(
        num_layers=config.num_layers, num_heads=config.num_heads, head_dim=config.head_dim, rows_per_request=1,
        max_source_len=0, retention=SinkWindow(0, config.left_context),
        query_window=config.left_context, max_slots=max_slots, dtype=None,
    )


class FastConformerStreamSubmodule(NodeSubmodule):
    """A causal FastConformer over a stream: each step takes one chunk of a
    stream's audio (``input_edge``) and emits the encoder frames no later audio
    can change, through ``projection``, one stream item per frame on
    ``output_edge``; the end of input emits the last ones. A frame per item
    lets a consumer's ``FixedChunkPolicy(1)`` pace a frame-synchronous loop.

    Live streams' chunks run in one batched pass, except that a step holding a
    chunk of more than a second (an upload, a large append) is not batched
    (``can_batch``): the engine then runs each stream of it on its own. The
    encoder is bound to the bounded KV and the conv pool; ``front_key`` names
    the pool of each stream's audio and mel tails.

    A stream one frame of audio further on the frame grid (every live step of
    a client appending 80 ms) runs whole as a CUDA graph (``STEP_REGION``). The
    others take the front end eagerly, then the blocks as a graph
    (``LAYERS_REGION``) when each got at most one frame (a live step, the
    stream's end), else eagerly (a one-shot upload, a large append). In a step
    whose other streams take the eager blocks, or when the whole-step region
    has no graph for its streams, those streams run eagerly too.
    """

    # The captured regions are not compiled (compile=False); the eager front end
    # takes variable-length chunks.
    disable_torch_compile = True

    STEP_REGION = "conformer_step"
    LAYERS_REGION = "conformer_layers"
    STATE_KEY = "encoder_stream"         # a stream's StreamState
    # Longest chunk batched with other streams': a pass pads every stream to its
    # longest chunk. One second is a judgement, not a measured crossover.
    BATCHED_CHUNK_S = 1.0

    def __init__(
        self,
        encoder: FastConformerEncoder,
        log_mel: NemoLogMel,
        projection: torch.nn.Module,
        *,
        input_edge: str,
        output_edge: str,
        front_key: str,
        batch_sizes: list[int],
    ):
        super().__init__()
        c = encoder.config
        frame_samples = log_mel.hop_length * c.subsampling_factor
        unsupported = {
            name: value for name, value, ok in (
                ("causal", c.causal, c.causal), ("lookahead", c.lookahead, c.lookahead == 0),
                ("left_context", c.left_context, c.left_context >= 0),
                ("normalize", log_mel.normalize, not log_mel.normalize),
                ("reflect_edges", log_mel.reflect_edges, log_mel.reflect_edges),
                ("preemphasis", log_mel.preemphasis, bool(log_mel.preemphasis)),
                ("n_fft", log_mel.n_fft, log_mel.n_fft <= frame_samples),
                ("kv_key", encoder.kv_key, encoder.kv_key is not None),
                ("conv_key", encoder.conv_key, encoder.conv_key is not None),
                ("batch_sizes", batch_sizes, bool(batch_sizes)),
            ) if not ok
        }
        if unsupported:
            # [L, R] streaming takes one chunk per step and a per-step window; a
            # per-clip normalization can't be streamed; the front end keeps one
            # encoder frame of samples, which must hold the next mel window
            raise ValueError(f"the streaming FastConformer takes a causal [L, 0] encoder bound to its "
                             f"KV and conv pool, NeMo's pre-2.5 unnormalized featurizer with a preemphasis "
                             f"and n_fft <= {frame_samples}, and capture batch sizes; got {unsupported}")
        self.encoder = encoder
        self.log_mel = log_mel
        self.projection = projection
        self.input_edge, self.output_edge, self.front_key = input_edge, output_edge, front_key
        self.front = None   # the front pool, bound as the encoder's layers bind theirs
        self.capture_batch_sizes = sorted(batch_sizes)
        self.frame_samples = frame_samples
        # the whole-step region's slices take a featurizer whose half window
        # spans more than one hop and at most two (n_fft 512, hop 160); any
        # other runs every step through the front end and the blocks
        self.has_steady_step = -(-(log_mel.n_fft // 2) // log_mel.hop_length) == 2

    # -- frame counts ---------------------------------------------------------

    def _final_frames(self, mels: int, samples: int, final: bool) -> int:
        """Encoder frames that no later audio can change, given at least one mel
        frame: frame ``j`` reads mel frames up to ``8 * j``. At the end, every
        frame offline would produce."""
        if final:
            return int(self.encoder.config.output_lengths(self.log_mel.num_frames(samples)))
        return (mels - 1) // self.encoder.config.subsampling_factor + 1

    def new_frames(self, st: StreamState, samples: int, final: bool) -> int:
        """The encoder frames a step makes final for a stream at ``st`` that
        then has ``samples`` of audio, counted on the host before the step runs,
        so its declaration can name them and the front end make them. Also
        without a new mel frame: the end of input can make its last frame final."""
        mels = max(st.mels, self.log_mel.complete_frames(samples, final))
        if not mels:
            return 0
        return max(0, self._final_frames(mels, samples, final) - st.frames)

    def is_steady(self, st: StreamState, chunk: torch.Tensor | None, final: bool) -> bool:
        """Whether the whole-step region serves this stream's step: one frame of
        audio onto a stream already two frames in, on the frame grid."""
        fs = self.frame_samples
        return (self.has_steady_step and not final and chunk is not None and chunk.numel() == fs
                and st.samples >= 2 * fs and st.samples % fs == 0)

    def bind_node_resources(self, resources: dict) -> None:
        super().bind_node_resources(resources)
        self.front = resources[self.front_key]

    def encode_clip(self, wav: torch.Tensor) -> torch.Tensor:
        """A whole clip through the featurizer, encoder and projection, as
        offline: ``(frames, projection out)``."""
        states, _ = self.encoder(self.log_mel(wav)[None].to(self.projection.weight.dtype))
        return self.projection(states)[0]

    # -- engine contract ------------------------------------------------------

    def prepare_inputs(
        self, graph_walk, fwd_info, inputs, is_final_stream_chunk=False, **kwargs,
    ) -> NodeInputs:
        chunk = inputs.get(self.input_edge)
        wav = chunk[0].reshape(-1) if chunk else None
        # the client stream releases one item per step; only its final flush is empty
        assert wav is not None or is_final_stream_chunk, "an empty chunk before the end of the stream"
        # Read, not counted here: admit can refuse the step and run this again.
        # The previous step's forward has returned, so the state is current.
        st = self.request_state(fwd_info.rid_handle).get(self.STATE_KEY)
        samples = (st.samples if st is not None else 0) + (wav.numel() if wav is not None else 0)
        shortest, rate = self.log_mel.min_samples, self.log_mel.sampling_rate
        if is_final_stream_chunk and samples < shortest:
            # too short to featurize (offline can't either): fail rather than reply to nothing
            raise ValueError(
                f"the call ended with {1000 * samples / rate:g} ms of audio; "
                f"it needs at least {1000 * shortest / rate:g} ms"
            )
        final = bool(is_final_stream_chunk)
        at = st or StreamState()
        return NodeInputs(
            tensor_inputs={} if wav is None else {"chunk": wav},
            kwargs={
                "final": final,
                # whether the whole-step region can serve the stream: decided
                # here, once, for the declaration and the forward to share
                "steady": st is not None and self.is_steady(st, wav, final),
                # the stream's position in the KV and the frames this step adds
                "position": at.position,
                "n": self.new_frames(at, samples, final),
            },
            input_seq_len=0 if wav is None else wav.numel(),   # samples, for can_batch
        )

    def preprocess(self, graph_walk, engine_inputs, inputs):
        return {
            "chunk": [inp.tensor_inputs.get("chunk") for inp in inputs],
            "final": [inp.kwargs.get("final", False) for inp in inputs],
            "steady": [inp.kwargs.get("steady", False) for inp in inputs],
            "n": [inp.kwargs.get("n", 0) for inp in inputs],
        }

    def can_batch(self, batch, model_inputs) -> bool:
        # A pass pads every stream to its longest chunk, so a chunk of more than
        # BATCHED_CHUNK_S keeps its step from being batched, and the engine runs
        # each stream's step on its own (declare, admit, plan, forward, commit).
        # Such chunks are uploads and large appends, so it is a rare step.
        longest = int(self.BATCHED_CHUNK_S * self.log_mel.sampling_rate)
        return len(model_inputs) == 1 or all(inp.input_seq_len <= longest for inp in model_inputs)

    def max_batch_size(self, graph_walk: str) -> int:
        # The engine caps a batch at its whole-forward graphs only; past the
        # widest region capture every stream of the step would run eagerly.
        return self.capture_batch_sizes[-1]

    def _declare(self, request_ids, keys, kv: BoundedKVStep) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
            steps={**{key: RecurrentStep() for key in keys}, self.encoder.kv_key: kv},
        )

    @staticmethod
    def _kv_step(request_ids, spans: list[int], positions: dict) -> BoundedKVStep:
        """Each stream's new frames (``spans``) at its position in the stream."""
        return BoundedKVStep(
            segments=tuple(Segment(request_id=rid, label="main", span=n)
                           for rid, n in zip(request_ids, spans, strict=True)),
            positions=positions,
        )

    def _region_kv_step(self, request_ids) -> BoundedKVStep:
        """A region's streams each add one frame. A capture's or a replay's
        padding row has no stream state, so it gets no position: the resource
        plans it as padding."""
        positions = {}
        for rid in request_ids:
            state = self.request_states.get(rid)
            st = None if state is None else state.get(self.STATE_KEY)
            if st is not None:
                positions[rid] = st.position
        return self._kv_step(request_ids, [1] * len(request_ids), positions)

    def declare_step(self, graph_walk, request_ids, inputs, slot_lease=None, piecewise_leases=None,
                     **kwargs) -> SubmoduleStep | None:
        del graph_walk, slot_lease, kwargs
        # A live step where every stream is one frame on replays STEP_REGION
        # over the whole batch, at the bucket its lease names, and the region's
        # own step owns every resource. Each such stream already holds its slots,
        # from a first step declared below, so that step's admit reserves nothing.
        if self.STEP_REGION in (piecewise_leases or {}) and all(inp.kwargs["steady"] for inp in inputs):
            return None
        # Otherwise this step reserves and plans every stream's slots and new
        # frames, and the eager path reads that plan. A region the forward still
        # replays for some streams re-plans their rows into its own buffers, after
        # the eager path has read this plan. It can't own those rows instead: its
        # lease names the whole batch's bucket, not the sub-batch's, which may
        # have no graph, and its admit can't refuse a stream that has no slot yet.
        spans = [inp.kwargs["n"] for inp in inputs]
        positions = {rid: inp.kwargs["position"] for rid, inp in zip(request_ids, inputs, strict=True)}
        return self._declare(request_ids, (self.front_key, self.encoder.conv_key),
                             self._kv_step(request_ids, spans, positions))

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        """A live step, one frame of audio per stream, whole (``STEP_REGION``);
        and the blocks alone for any step with at most one frame per stream
        (``LAYERS_REGION``)."""
        del tp_world_size
        hidden = self.encoder.config.hidden_size

        def step_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {"chunk": torch.zeros(shape.bs, self.frame_samples, dtype=torch.float32, device=device)}

        def layers_inputs(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
            return {
                "x": torch.zeros(shape.bs, 1, hidden, dtype=autocast_dtype, device=device),
                "n": torch.zeros(shape.bs, dtype=torch.long, device=device),
            }

        def region(fn, inputs, keys, lease) -> PiecewiseBatchedConfig:
            return PiecewiseBatchedConfig(
                capture_fn=fn, make_static_inputs=inputs,
                declare_step=lambda request_ids, seq_lens: self._declare(
                    request_ids, keys, self._region_kv_step(request_ids)),
                seq_len=1, capture_batch_sizes=self.capture_batch_sizes, compile=False, lease_before_step=lease,
            )

        conv_key = self.encoder.conv_key
        regions = {
            self.STEP_REGION: region(self._step_region, step_inputs, (self.front_key, conv_key), True),
            self.LAYERS_REGION: region(self._layers_region, layers_inputs, (conv_key,), False),
        }
        if not self.has_steady_step:
            del regions[self.STEP_REGION]
        return regions

    def _step_region(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        return {"y": self._steady_step(call.static_inputs["chunk"])}

    def _layers_region(self, call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
        s = call.static_inputs
        return {"y": self._blocks(s["x"], s["n"])}

    def _blocks(self, x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder.blocks_stream(x, n))

    # -- the front end --------------------------------------------------------

    def _steady_step(self, chunk: torch.Tensor) -> torch.Tensor:
        """One frame for each stream, from ``chunk (B, frame_samples)``: the new
        mel frames, the subsampling window and the blocks, all on the device.
        It is captured, so its STFT is a matmul: a graph must not hold a cuFFT
        plan, which the plan cache can destroy (see ``dft``).

        A stream ``S = (k - 1) fs`` samples in has ``8(k - 1) - 1`` mel frames
        and ``k - 1`` encoder frames. This frame of audio adds mel frames
        ``8k - 9 .. 8k - 2`` and encoder frame ``k - 1``, whose subsampling
        window is the 15 mel frames before them plus them (the third output)."""
        fs, f = self.frame_samples, self.encoder.config.subsampling_factor
        hop, pad = self.log_mel.hop_length, self.log_mel.n_fft // 2
        last, sig = self.front.gather("last"), self.front.gather("sig")
        mel_tail = self.front.gather("mel")
        x = chunk.to(torch.float32)
        y = x - self.log_mel.preemphasis * torch.cat([last, x[:, :-1]], dim=1)
        win = torch.cat([sig, y], dim=1)                                    # samples S - fs .. S + fs
        # mel frames f(k-1)-1 .. fk-2 span samples [S - hop - pad, S + fs - 2 hop + pad)
        mel_new = self.log_mel.windows(win[:, fs - hop - pad: 2 * fs - 2 * hop + pad], by_matmul=True)
        window = torch.cat([mel_tail[:, 1:], mel_new], dim=1)               # (B, 23, n_mels)
        new = self.encoder.subsample(window)[:, 2:3]
        self.front.scatter_("sig", y)
        self.front.scatter_("last", x[:, -1:])
        self.front.scatter_("mel", torch.cat([mel_tail, mel_new], dim=1)[:, -2 * f:])
        return self._blocks(new, chunk.new_ones(chunk.shape[0], dtype=torch.long))

    def _front(self, sts: list[StreamState], chunks: list[torch.Tensor | None], final: list[bool],
               n: list[int]) -> torch.Tensor | None:
        """Append each stream's chunk; the ``n`` block inputs it made final
        (``new_frames``, counted before the step), ``(B, max(n), hidden)``
        zero-padded, or None when no stream has one. A stream with no chunk and
        not at its end is left as it is."""
        f = self.encoder.config.subsampling_factor
        keep_sig, keep_mel = self.frame_samples, 2 * f
        busy = [i for i, (ch, fin) in enumerate(zip(chunks, final, strict=True))
                if fin or (ch is not None and ch.numel())]
        if not busy:
            return None
        last, sig = self.front.gather("last"), self.front.gather("sig")
        mel_tail = self.front.gather("mel")
        new_sig, new_last, new_mel = list(sig), list(last), list(mel_tail)
        rows, lo_hi, mel_rows = [], [], []
        for i in busy:
            st, chunk = sts[i], chunks[i]
            held = min(st.samples, keep_sig)
            win, base = sig[i, keep_sig - held:], st.samples - held          # win[0] is sample ``base``
            if chunk is not None and chunk.numel():
                xr = chunk.to(torch.float32).reshape(-1)
                # ``last`` reads 0 in a fresh slot: the featurizer's preemphasis keeps the first sample
                y = xr - self.log_mel.preemphasis * torch.cat([last[i], xr[:-1]])
                win = torch.cat([win, y])
                new_last[i] = xr[-1:]
                st.samples += xr.numel()
            new_sig[i] = F.pad(win[-keep_sig:], (max(0, keep_sig - win.numel()), 0))
            lo, hi = st.mels, self.log_mel.complete_frames(st.samples, final[i])
            if hi <= lo:
                continue
            rows.append(self.log_mel.stream_span(win, base, lo, hi, final[i]))
            lo_hi.append((i, lo, hi))
        added = {}
        if rows:
            width = max(r.numel() for r in rows)
            mel = self.log_mel.windows(torch.stack([F.pad(r, (0, width - r.numel())) for r in rows]))
            added = {i: mel[row, : hi - lo] for row, (i, lo, hi) in enumerate(lo_hi)}
        for i in busy:
            st = sts[i]
            held = min(st.mels, keep_mel)
            mels, first = mel_tail[i, keep_mel - held:], st.mels - held      # mels[0] is mel frame ``first``
            if i in added:
                mels = torch.cat([mels, added[i]])
                st.mels += added[i].shape[0]
                new_mel[i] = F.pad(mels[-keep_mel:], (0, 0, max(0, keep_mel - mels.shape[0]), 0))
            if n[i]:
                start = max(0, f * st.frames - 2 * f)     # the subsampling window's first mel frame
                mel_rows.append((i, mels[start - first:], (st.frames - start // f, n[i])))
        self.front.scatter_("sig", torch.stack(new_sig))
        self.front.scatter_("last", torch.stack(new_last))
        self.front.scatter_("mel", torch.stack(new_mel))
        if not mel_rows:
            return None
        lens = [w.shape[0] for _, w, _ in mel_rows]
        windows = torch.stack([F.pad(w, (0, 0, 0, max(lens) - w.shape[0])) for _, w, _ in mel_rows])
        # a strided conv reads a frame past a row's end, which alone is the conv's
        # zero padding: the subsampling masks each row past its own length
        out = self.encoder.subsample(windows, to_device_nonblocking(torch.tensor(lens), windows.device))
        x = out.new_zeros(len(sts), max(n), out.shape[-1])
        for row, (i, _, (at, k)) in enumerate(mel_rows):
            x[i, :k] = out[row, at: at + k]
        return x

    # -- the step ---------------------------------------------------------------

    def _stream_state(self, rid) -> StreamState:
        state = self.request_state(rid)
        st = state.get(self.STATE_KEY)
        if st is None:
            st = StreamState()
            state.add(self.STATE_KEY, st)
        return st

    @staticmethod
    def _to_device(chunks: list, device: torch.device) -> list:
        """The step's chunks on the device in one copy: a copy per stream, from
        pageable memory, would each wait for its transfer."""
        host = [i for i, c in enumerate(chunks) if c is not None and c.device != device]
        moved = list(chunks)
        if host:
            flat = torch.cat([chunks[i].reshape(-1) for i in host])
            parts = to_device_nonblocking(flat, device).split([chunks[i].numel() for i in host])
            for i, part in zip(host, parts, strict=True):
                moved[i] = part
        return moved

    def forward_batched(self, graph_walk, engine_inputs, chunk=None, final=None, steady=None, n=None, **kwargs):
        rids = list(engine_inputs.request_ids)
        device = self.projection.weight.device
        states = [self._stream_state(rid) for rid in rids]
        chunks = self._to_device(chunk, device)
        runners = engine_inputs.piecewise_runners or {}
        frames: list[torch.Tensor | None] = [None] * len(rids)

        # streams one frame of audio further on the frame grid: the whole step as one graph, run last
        step_runner, layers_runner = runners.get(self.STEP_REGION), runners.get(self.LAYERS_REGION)

        def layers_graph(rows) -> bool:
            """Whether the blocks region takes these streams' new frames: at most one each."""
            new = [i for i in rows if n[i]]
            return (bool(new) and max(n[i] for i in new) == 1 and layers_runner is not None
                    and layers_runner.can_run(len(new)))

        if engine_inputs.step is None:
            # declare_step left every resource to the region, which its lease says can replay the batch
            assert all(steady), "an encoder step declared nothing but is not all steady"
            steady = list(range(len(rids)))
        else:
            # This step's KV plan writes every stream's new frames, so the region (which
            # writes its streams' under its own plan) may serve the steady streams only when
            # that plan goes unused: the other streams' frames, if any, all go through the
            # blocks region. Otherwise, or with no graph for them, they run eagerly too.
            rest = [i for i, s in enumerate(steady) if not s]
            steady = [i for i, s in enumerate(steady) if s]
            if steady and not (step_runner is not None and step_runner.can_run(len(steady))
                               and (not any(n[i] for i in rest) or layers_graph(rest))):
                steady = []
        steady_chunks = [chunks[i] for i in steady]
        eager_n = list(n)
        for i in steady:
            chunks[i], eager_n[i] = None, 0     # the eager pass leaves these streams as they are

        # The rest: the same math eagerly, reading this step's plans where it
        # uses them, before any region plans its own; its blocks as a graph when
        # no stream got more than a frame. A chunk of more than a second comes in
        # a step of its own (see can_batch).
        x = None
        if len(steady) < len(rids):
            x = self._front(states, chunks, final, eager_n)
        if x is not None:
            new = [i for i, k in enumerate(eager_n) if k]
            if layers_graph(new):
                # only the streams with a frame: the region plans its own rows
                y = layers_runner.run(static_inputs={"x": torch.stack([x[i] for i in new]),
                                                     "n": torch.ones(len(new), dtype=torch.long, device=device)},
                                      request_ids=[rids[i] for i in new])["y"]
                y = dict(zip(new, y, strict=True))
            else:
                y = self._blocks(x, to_device_nonblocking(torch.tensor(eager_n), device))
            for i in new:
                states[i].frames += eager_n[i]
                frames[i] = y[i][: eager_n[i]]

        if steady:
            y = step_runner.run(static_inputs={"chunk": torch.stack(steady_chunks)},
                                request_ids=[rids[i] for i in steady])["y"]
            for row, i in enumerate(steady):
                st = states[i]
                st.samples += self.frame_samples
                st.mels += self.encoder.config.subsampling_factor
                st.frames += 1
                frames[i] = y[row]
        # One stream item per frame: a single (T, H) tensor would be ONE item,
        # which the chunk policy could not slice. No check_stop, so nothing to
        # copy to the host for one.
        return BatchedModelOutput(
            per_rid_outputs={rid: {self.output_edge: [] if f is None else list(f)}
                             for rid, f in zip(rids, frames, strict=True)},
            check_stop_buffers={},
        )

    def forward(self, graph_walk, engine_inputs, **kwargs) -> NameToTensorList:
        # one stream's step: preprocess's lists of one row
        rid = engine_inputs.request_ids[0]
        return self.forward_batched(graph_walk, engine_inputs, **kwargs).per_rid_outputs[rid]
