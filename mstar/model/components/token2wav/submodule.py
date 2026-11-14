"""Token2wav as a graph node: streamed speech-code windows -> 24 kHz audio."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    PiecewiseAcceleratorGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
)
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import Segment, SubmoduleStep
from mstar.engine.state import ArrayStep, host
from mstar.model.chatterbox.components.s3gen_hift import HiFTNoise
from mstar.model.components.token2wav.flow import MEL_BINS
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    HOP,
    LAST_WINDOW_TOKENS,
    SILENCE_CODE,
    WINDOW,
    VoiceBank,
    WindowCaches,
    window_frames,
)
from mstar.model.submodule_base import (
    EMPTY_INPUT_METADATA,
    InputMetadata,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.utils.h2d import to_device_nonblocking


class Token2WavSubmodule(NodeSubmodule):
    """Speech codes -> 24 kHz audio, one stream window per step: upstream's
    ``Token2wav.stream``, fed the way the model's reference feeds it: 28-code windows
    advancing by 25, a last flush, and for MiniCPM-o (``streaming_generate``) three
    silence codes first. The model's ``Token2WavStreaming`` (``model.streaming``) says
    which, and which caches its reference cuts.

    For MiniCPM-o the windows are the ``LeftContextChunkPolicy(25, 3)`` stream from the
    TTS; the first window gets the silence prepended here. For Step-Audio2 they are the
    ``SlidingWindowChunkPolicy(28, 25)`` stream of its LLM's speech ids, which carries no
    stop code (the LLM withholds every other id). A request's attention caches
    are the ``keys.kv`` arrays (``token2wav_caches``), each row's stream starting
    from its voice's prepared caches, which the node hands each array once. Its
    conv caches, HiFT's held-back frames and its held-back samples are plain
    Values (``token2wav_state``), each listed by the region that writes it, and
    whether its last window ran is a host field.
    Its stream position is the chunk's offset in the stream (``prepare_inputs``), and
    its voice the walk's step metadata.

    Float32, but for the DiT's GEMMs, convolutions, attention and caches, which are
    ``DIT_COMPUTE_DTYPE``. Each window runs in two stages (``forward_batched``), each a piecewise
    capture bucketed by batch size: the flow (``t2w/<voice>/flow28``), for windows
    at any position, and HiFT, separately for a request's first window, whose
    input is shorter. A last window has one of a few lengths, each a one-row
    capture; other shapes run eagerly. The inverse STFT and cross-fade stay eager, in
    a region of their own, since they write the held-back samples.
    The tests' one-request driver over the same modules is bit-exact against the
    reference at full fp32; this path attends the DiT's cache in another order and
    runs on fused kernels, with the DiT in float16.
    """

    WINDOW_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32]
    # a request ends once, so last windows of one length rarely meet in a pass
    LAST_CAPTURE_BATCH_SIZES = [1]

    disable_torch_compile = True
    disable_autocast = True
    # the caches as the model's reference retains them (``Token2WavStreaming``); MiniCPM-o's
    # until ``__init__`` reads the model's
    families = CACHE_FAMILIES

    def __init__(
        self, model, keys: Token2WavKeys, voices: dict, eos_code: int | None, dit_dtype: torch.dtype,
        code_offset: int = 0,
    ):
        """``keys``: the node's input signal and resource keys. ``voices``: the model's
        bundled ones, by name. ``eos_code``: the stream's stop code, which ends the speech,
        or None for a stream that carries none. ``dit_dtype``: the dtype of the DiT's
        caches, as the model declares them, which the DiT computes in. ``code_offset``:
        what the stream's items are past the codes (Step-Audio2 streams its LLM's speech
        ids, ``151696 + code``); subtracted on the device."""
        super().__init__()
        self.model = model
        self.keys = keys
        self.eos_code = eos_code
        self.code_offset = code_offset
        self.bank = VoiceBank.build(voices, dit_dtype=dit_dtype)
        # the model's reference's streaming knobs (``Token2WavStreaming``)
        self.families = model.streaming.cache_families()
        # in the stream's ids, as the codes they lead
        self._lead_silence = torch.full(
            (model.streaming.lead_silence,), SILENCE_CODE + code_offset, dtype=torch.int32)
        # Scratch: each row of this step's forward with codes, (voice, codes written before its window), set around
        # the regions it runs, whose steps name its voice's source (``_cache_state``) and whose rows it voices.
        self._rows: dict[int, tuple[int, int]] = {}

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.flow_state = handles[self.keys.flow]
        self.hift_state = handles[self.keys.hift]
        self.tail_state = handles[self.keys.tail]
        self.request = handles[self.keys.request]
        # each cache family's source is its voices' own caches, which its array now holds
        for name, family in self.families.items():
            source_lens = [family.rate * tokens for tokens in self.bank.prompt_tokens]
            handles[self.keys.kv[name]].set_sources(self.bank.source_fields(name, family.head_dim), source_lens)
        self._window_caches = WindowCaches.bind(handles, self.keys.kv, self.families, {
            name: handles[self.keys.kv[name]].array.source.max_len for name in self.families
        })
        self.bank = self.bank._replace(sources={})

    def compile_regions(self) -> dict[str, list[tuple]]:
        # HiFT up to its spectrum, compiled once with dynamic shapes (every region's
        # rows and frames): Inductor fuses its elementwise source and snake
        # activations, about half its GPU time. It compiles in the captures' warm-up.
        return {"model.hift.spectrum": []}

    def _voice_ids(self, rids: list[int], device: torch.device) -> torch.Tensor:
        # pinned and queued: a pageable copy would wait out the running step
        return to_device_nonblocking([self._rows[r][0] for r in rids], device, torch.long)

    def _cache_state(
        self, request_ids: list[int], num_tokens: list[int] | None, lasts: list[bool],
    ) -> dict[str, ArrayStep]:
        """Each cache's step: every row's source, its voice's caches, and what its window adds (``num_tokens`` None:
        nothing, a step that only takes each row's slot). A capture's dummy row names the first voice."""
        sources = {rid: self._rows[rid][0] if rid in self._rows else 0 for rid in request_ids}
        return {
            self.keys.kv[name]: ArrayStep(
                segments=tuple(
                    Segment(request_id=rid, label="main", span=0 if num_tokens is None else family.span(n, last))
                    for rid, n, last in zip(request_ids, num_tokens or [0] * len(request_ids), lasts, strict=True)
                ),
                sources=sources,
            )
            for name, family in self.families.items()
        }

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        is_final_stream_chunk: bool = False,
        input_metadata: InputMetadata = EMPTY_INPUT_METADATA,
        **kwargs: Any,
    ) -> NodeInputs | None:
        if not inputs.get(self.keys.codes):
            # The TTS never ran: the spoken reply ended at the LLM's prefill, and its
            # stream's one chunk is final and empty. It still runs, voicing nothing,
            # as the empty step below does.
            return NodeInputs(tensor_inputs={"codes": torch.zeros(1, 0, dtype=torch.int32)}, kwargs={"last": True})
        codes = inputs[self.keys.codes][0].reshape(-1)
        if self.eos_code is None:
            # Nothing to filter, and the window's length (which decides its capture and
            # its cache writes) is its shape: no host copy here. A by-value window is
            # already on the host; one holding a stored item stays on the device.
            codes = codes.to(torch.int32)
            last = is_final_stream_chunk
        else:
            # The TTS's stop code is streamed before its stop is seen. It is not speech,
            # and it ends the speech: the window holding it is the last, as upstream's
            # stream ends there. A window whose codes all came inline is already a CPU
            # tensor, so this reads no device memory; one holding a stored item (a row
            # that ran no forward on its step) is on the device, and this is a host sync.
            # Filtering the stop code on the device would need check_stop to filter
            # outputs (an engine change), and would leave the length on the device.
            codes = codes.to("cpu", torch.int32)
            last = is_final_stream_chunk or bool((codes == self.eos_code).any())
            codes = codes[codes != self.eos_code]
        if host(self.request, fwd_info.rid_handle, "main", 0)["ended"]:
            # Past the last window: the stream's final chunk after the window that held
            # the stop code, which only repeats that window's tail. It still runs,
            # voicing nothing (``forward_batched``): a skipped step never completes the
            # node, which then refuses the stream's end. Told by the last window having
            # run, not by the chunk being all look-back: a stream that ends at a full
            # window without the stop code has a last window that is all look-back.
            return NodeInputs(tensor_inputs={"codes": codes[None, :0]}, kwargs={"last": True})
        chunk = input_metadata.stream_chunks[self.keys.codes]
        first = chunk.start_offset == 0
        if first and self._lead_silence.numel():
            codes = torch.cat([self._lead_silence.to(codes.device), codes])
        # Window k starts 25k codes into the (silence-led, for MiniCPM-o) stream, its
        # look-back included, which is what its caches hold before it.
        written = 0 if first else chunk.start_offset + self._lead_silence.numel()
        return NodeInputs(tensor_inputs={"codes": codes[None]}, kwargs={
            "last": last, "first": first, "written": written,
            "voice": self.bank.index(fwd_info.step_metadata[self.keys.voice]),
            # HiFT's excitation for this window, from the request's seed (``Token2Wav.window_noise``)
            "noise": (fwd_info.random_seed, written // HOP),
        })

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep:
        # This step takes each row's slot of the caches, adding nothing (span 0), which has
        # to happen before the forward; the flow region that adds to them, and each Value's
        # region, take theirs inside it, and a Value has one whenever the caches did
        # (``check_token2wav_slots``). This step's plan goes unused.
        # TODO: an admit-only step would save that plan's host work.
        # A row with no codes voices nothing (``forward_batched``), so it declares
        # nothing: a reply that never reached the TTS takes no slot.
        rows = [(rid, inp) for rid, inp in zip(request_ids, inputs, strict=True) if inp.tensor_inputs["codes"].shape[1]]
        ended = {(rid, "main", 0): {"ended": True} for rid, inp in rows if inp.kwargs["last"]}
        self._rows = {rid: (inp.kwargs["voice"], inp.kwargs["written"]) for rid, inp in rows}
        try:
            state = self._cache_state([rid for rid, _ in rows], None, [False] * len(rows)) if rows else {}
        finally:
            self._rows = {}
        if ended:
            state[self.keys.request] = ArrayStep(host=ended)
        return SubmoduleStep(segments=[Segment(request_id=rid, label="main", span=1) for rid, _ in rows], state=state)

    # Each window's flow and HiFT run through a piecewise region: replayed when a graph
    # fits, else eagerly through the same step, so the resources always hold the plan
    # of exactly the rows being run. Shapes that are never captured (a one-window
    # reply's HiFT) have eager-only regions.

    @staticmethod
    def flow_region(num_tokens: int, last: bool) -> str:
        return f"t2w/flow{'_last' if last else ''}{num_tokens}"

    @staticmethod
    def hift_region(first: bool, frames: int, last: bool) -> str:
        return f"t2w/hift/{'first' if first else 'next'}{'_last' if last else ''}{frames}"

    @staticmethod
    def eager_region(stage: str, first: bool, last: bool) -> str:
        return f"t2w/{stage}_eager/{'first' if first else 'next'}{'_last' if last else ''}"

    @staticmethod
    def finish_region(first: bool, last: bool) -> str:
        return f"t2w/finish/{'first' if first else 'next'}{'_last' if last else ''}"

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        def static_tokens(num_tokens: int):
            def make(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
                return {
                    "tokens": torch.zeros(shape.bs, num_tokens, dtype=torch.int32, device=device),
                    # each row's voice, an index into ``bank``
                    "voices": torch.zeros(shape.bs, dtype=torch.long, device=device),
                }
            return make

        def static_hift(first: bool, frames: int):
            def make(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
                noise = self.model.window_noise(frames, first, [(0, 0)] * shape.bs)
                return {"mel": torch.zeros(shape.bs, MEL_BINS, frames, device=device),
                        "phase": noise.phase, "harmonic": noise.harmonic}
            return make

        def declare_flow(last: bool):
            def declare_step(request_ids: list[int], seq_lens: list[int]) -> SubmoduleStep:
                return SubmoduleStep(
                    segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                    state={
                        self.keys.flow: ArrayStep(),
                        **self._cache_state(request_ids, list(seq_lens), [last] * len(request_ids)),
                    },
                )
            return declare_step

        def declare_value(name: str):
            def declare_step(request_ids: list[int], seq_lens: list[int]) -> SubmoduleStep:
                return SubmoduleStep(
                    segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                    state={name: ArrayStep()},
                )
            return declare_step

        def flow(last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                mel, conv = self.model.window_flow(
                    self.flow_state.read(0).fields, self.bank, call.static_inputs["voices"],
                    call.static_inputs["tokens"], self._window_caches, last,
                )
                self.flow_state.write(0, **conv)
                return {"mel": mel}
            return run

        def hift(first: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                inputs = call.static_inputs
                spectrum, held = self.model.window_vocode(
                    self.hift_state.read(0).fields, inputs["mel"], first,
                    HiFTNoise(phase=inputs["phase"], harmonic=inputs["harmonic"]),
                )
                self.hift_state.write(0, **held)
                return spectrum
            return run

        def finish(first: bool, last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                inputs = call.static_inputs
                speech, held = self.model.window_finish(
                    self.tail_state.read(0).fields, inputs["magnitude"], inputs["phase"], first, last,
                )
                self.tail_state.write(0, **held)
                return {"speech": speech}
            return run

        def region(run, make_static_inputs, declare_step, seq_len, batch_sizes):
            return PiecewiseBatchedConfig(
                capture_fn=run, make_static_inputs=make_static_inputs, declare_step=declare_step,
                seq_len=seq_len, capture_batch_sizes=batch_sizes, eager_fallback=True,
            )

        # Full windows at any position share one graph per batch size. A last window has
        # one of HOP lengths and rarely meets another of its length, so each length is
        # a one-row graph: 50 of this node's 65 graphs, most of its ~25 s of capture
        # and ~1 GiB. Padding last windows to one length would batch them and cut that
        # (the flow's time barely depends on length), but needs the padded tokens
        # masked out of attention; HiFT's convs are not causal, so its graphs stay per
        # length.
        #
        # Each row's voice is an input (an index into ``bank``), so one graph serves every
        # voice and rows of different voices share a batch.
        full = window_frames(WINDOW, False)
        configs = {
            self.flow_region(WINDOW, False): region(
                flow(False), static_tokens(WINDOW), declare_flow(False), WINDOW, self.WINDOW_CAPTURE_BATCH_SIZES),
        }
        for first in (True, False):
            configs[self.hift_region(first, full, False)] = region(
                hift(first), static_hift(first, full), declare_value(self.keys.hift), full,
                self.WINDOW_CAPTURE_BATCH_SIZES)
        for n in LAST_WINDOW_TOKENS:
            frames = window_frames(n, True)
            configs[self.flow_region(n, True)] = region(
                flow(True), static_tokens(n), declare_flow(True), n, self.LAST_CAPTURE_BATCH_SIZES)
            configs[self.hift_region(False, frames, True)] = region(
                hift(False), static_hift(False, frames), declare_value(self.keys.hift), frames,
                self.LAST_CAPTURE_BATCH_SIZES)
        for last in (False, True):
            configs[self.eager_region("flow", False, last)] = region(
                flow(last), static_tokens(WINDOW), declare_flow(last), WINDOW, [])
            for first in (False, True):
                configs[self.eager_region("hift", first, last)] = region(
                    hift(first), static_hift(first, full), declare_value(self.keys.hift), full, [])
                # The inverse STFT syncs, so this one is never captured. A region of its own all the same: it writes
                # the held-back samples, under a step of its rows.
                configs[self.finish_region(first, last)] = region(
                    finish(first, last), lambda shape: {}, declare_value(self.keys.tail), full, [])
        return configs

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        rows = []
        for inp in inputs:
            codes = inp.tensor_inputs["codes"].to(device)
            if self.code_offset:  # a kernel per row only for a stream that needs it
                codes = codes - self.code_offset
            rows.append((codes, inp.kwargs))
        return {"rows": rows}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows: list | None = None, **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Each window in two stages: the flow, batched over every row of one window length
        whatever its voice and position, then HiFT, batched over rows that are all or none
        their request's first window."""
        out, flows, kwargs = {}, {}, {}
        for rid, (codes, row) in zip(engine_inputs.request_ids, rows, strict=True):
            if not codes.shape[1]:
                # past its last window (``prepare_inputs``)
                out[rid] = {}
                continue
            kwargs[rid] = row
            flows.setdefault((codes.shape[1], row["last"]), []).append((rid, codes))
        self._rows = {rid: (row["voice"], row["written"]) for rid, row in kwargs.items()}
        try:
            out.update(self._windows(engine_inputs.piecewise_runners, flows, kwargs))
        finally:
            self._rows = {}
        return out

    def _windows(self, runners, flows: dict, kwargs: dict) -> dict[int, NameToTensorList]:
        """``flows``' windows, grouped by (length, last), through the flow then HiFT; ``kwargs`` are each row's."""
        out, mels, hifts = {}, {}, {}
        for (n, last), group in flows.items():
            runner = runners.get(self.flow_region(n, last)) or runners[self.eager_region("flow", False, last)]
            for chunk in self._chunks(runner, group, last):
                rids = [rid for rid, _ in chunk]
                tokens = torch.cat([codes for _, codes in chunk])
                mel = self._run_flow(runner, rids, tokens)
                for i, rid in enumerate(rids):
                    mels[rid] = mel[i]
                    hifts.setdefault((kwargs[rid]["first"], mel.shape[-1], last), []).append(rid)

        for (is_first, frames, last), rids in hifts.items():
            runner = runners.get(self.hift_region(is_first, frames, last)) or runners[
                self.eager_region("hift", is_first, last)]
            for chunk in self._chunks(runner, rids, last):
                mel = torch.stack([mels[rid] for rid in chunk])
                noise = [kwargs[rid]["noise"] for rid in chunk]
                wav = self._run_hift(runners, runner, chunk, mel, noise, is_first, last)
                for i, rid in enumerate(chunk):
                    out[rid] = {self.keys.audio: [wav[i]]}
        return out

    def _chunks(self, runner, group: list, last: bool) -> list[list]:
        """``group`` in batches its region can replay (all of it when it runs eagerly)."""
        if not runner.any_graphs:
            return [group]
        cap = (self.LAST_CAPTURE_BATCH_SIZES if last else self.WINDOW_CAPTURE_BATCH_SIZES)[-1]
        return [group[i:i + cap] for i in range(0, len(group), cap)]

    @torch.compiler.disable
    def _run_flow(self, runner, rids: list[str], tokens: torch.Tensor) -> torch.Tensor:
        """``window_flow`` for these rows (``[B, 80, frames]``)."""
        voices = self._voice_ids(rids, tokens.device)
        out = runner.run(
            static_inputs={"tokens": tokens, "voices": voices}, request_ids=rids,
            seq_lens=[tokens.shape[1]] * len(rids), real_bs=len(rids),
        )
        # copied out: the next replay of this region reuses the buffer
        return out.get_view("mel").clone()

    @torch.compiler.disable
    def _run_hift(
        self,
        runners,
        runner,
        rids: list[int],
        mel: torch.Tensor,
        noise: list[tuple[int, int]],
        first: bool,
        last: bool,
    ) -> torch.Tensor:
        """``window_vocode`` and then (eagerly: the inverse STFT syncs) ``window_finish``: the rows'
        samples. ``noise`` is each row's ``(seed, window)``."""
        drawn = self.model.window_noise(mel.shape[-1], first, noise)
        frames = mel.shape[-1]
        out = runner.run(
            static_inputs={"mel": mel, "phase": drawn.phase, "harmonic": drawn.harmonic},
            request_ids=rids, seq_lens=[frames] * len(rids), real_bs=len(rids),
        )
        n = len(rids)
        finished = runners[self.finish_region(first, last)].run(
            static_inputs={"magnitude": out.get_view("magnitude")[:n], "phase": out.get_view("phase")[:n]},
            request_ids=rids, seq_lens=[frames] * n, real_bs=n,
        )
        return finished.get_view("speech")
