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
from mstar.engine.state import ArrayStep, Segment, SubmoduleStep, host
from mstar.model.chatterbox.components.s3gen_hift import HiFTNoise
from mstar.model.components.token2wav.flow import MEL_BINS, FlowKV, HandleLayerKV
from mstar.model.components.token2wav.keys import Token2WavKeys
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    DIT_DEPTH,
    ENC_BLOCKS,
    ENC_UP_BLOCKS,
    HOP,
    LAST_WINDOW_TOKENS,
    LEAD_SILENCE,
    N_TIMESTEPS,
    SILENCE_CODE,
    WINDOW,
    VoiceBank,
    voice_start,
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
    ``Token2wav.stream``, fed the way its ``streaming_generate`` feeds it (three
    silence codes first, 28-code windows advancing by 25, a last flush).

    The windows are the ``LeftContextChunkPolicy(25, 3)`` stream from the TTS;
    the first window gets the silence prepended here. A request's attention caches
    are the ``keys.kv`` arrays, whose rows begin from their voice's prepared caches;
    its conv caches and HiFT tails are its slots of the ``keys.state`` Values, one per
    region that writes them, and its voice and last window are host fields of
    ``keys.request``.

    Float32, but for the DiT's GEMMs, convolutions, attention and caches, which are
    ``DIT_COMPUTE_DTYPE``. Each window runs in three stages (``forward_batched``), each a
    piecewise region: the flow (``t2w/flow28``), for windows at any position, captured by
    batch size; HiFT, separately for a request's first window, whose input is shorter;
    and, eagerly since the inverse STFT syncs, the cross-fade. A last window has one of
    a few lengths, each a one-row capture; other shapes run eagerly. The tests'
    one-request driver over the same modules is bit-exact against the reference at full
    fp32; this path attends the DiT's cache in another order and runs on fused kernels,
    with the DiT in float16.
    """

    WINDOW_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32]
    # a request ends once, so last windows of one length rarely meet in a pass
    LAST_CAPTURE_BATCH_SIZES = [1]

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, model, keys: Token2WavKeys, voices: dict, eos_code: int, dit_dtype: torch.dtype):
        """``keys``: the node's input signal and state keys. ``voices``: the model's
        bundled ones, by name. ``dit_dtype``: the dtype of the DiT's caches, as the model
        declares them, which the DiT computes in."""
        super().__init__()
        self.model = model
        self.keys = keys
        self.eos_code = eos_code
        self.bank = VoiceBank.build(voices, dit_dtype=dit_dtype)
        # each family's voice caches in stream order, which `bind_state` hands to its array and drops
        self._sources: dict[str, list[torch.Tensor]] | None = {
            name: [family.source(voices[voice]) for voice in self.bank.names]
            for name, family in CACHE_FAMILIES.items()
        }
        self._lead_silence = torch.full((LEAD_SILENCE,), SILENCE_CODE, dtype=torch.int32)

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.slots = {part: handles[key] for part, key in self.keys.state.items()}
        self.per_request = handles[self.keys.request]
        self.caches = {name: handles[key] for name, key in self.keys.kv.items()}
        for name, sources in self._sources.items():
            dtype = self.caches[name].array.entry.tensors["k"].dtype
            self.caches[name].set_starts([voice_start(source.to(dtype)) for source in sources])
        self._sources = None

    def compile_regions(self) -> dict[str, list[tuple]]:
        # HiFT up to its spectrum, compiled once with dynamic shapes (every region's
        # rows and frames): Inductor fuses its elementwise source and snake
        # activations, about half its GPU time. It compiles in the captures' warm-up.
        return {"model.hift.spectrum": []}

    def _flow_kv(self) -> FlowKV:
        """The flow's attention caches, each layer through its family's handle."""

        def layer(name: str, idx: int) -> HandleLayerKV:
            family, handle = CACHE_FAMILIES[name], self.caches[name]
            return HandleLayerKV(handle, idx, family.rows, family.max_keys(handle.array.start_lengths[-1]))

        return FlowKV(
            [layer("enc1", i) for i in range(ENC_BLOCKS)],
            [layer("enc2", i) for i in range(ENC_UP_BLOCKS)],
            [[layer("dit", step * DIT_DEPTH + i) for i in range(DIT_DEPTH)] for step in range(N_TIMESTEPS)],
        )

    def _cache_steps(
        self, request_ids: list[str], num_tokens: list[int], lasts: list[bool], keep: bool = True,
        voices: Mapping[str, int] | None = None,
    ) -> dict[str, ArrayStep]:
        """Each cache's step: what each row's window adds, and on the node's step each row's voice, whose cache its
        stream begins with and which a region's step takes from it."""
        return {
            self.keys.kv[name]: ArrayStep(
                segments=tuple(
                    Segment(request_id=rid, label="main", span=family.span(n, last))
                    for rid, n, last in zip(request_ids, num_tokens, lasts, strict=True)
                ),
                keep=keep, starts=voices or {},
            )
            for name, family in CACHE_FAMILIES.items()
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
        # A window whose codes all came inline is already a CPU tensor, so this copies
        # nothing. A window holding a stored item (a row that ran no forward on its
        # step) is on the device, and this is a host sync. The codes are needed on the
        # host either way: a window's length decides its capture and its cache writes.
        # Filtering the stop code on the device would need check_stop to filter outputs
        # (an engine change), and would still leave the length on the device.
        codes = inputs[self.keys.codes][0].reshape(-1).to("cpu", torch.int32)
        # The TTS's stop code is streamed before its stop is seen. It is not speech, and
        # it ends the speech: the window holding it is the last, as upstream's stream
        # ends there.
        last = is_final_stream_chunk or bool((codes == self.eos_code).any())
        codes = codes[codes != self.eos_code]
        chunk = input_metadata.stream_chunks[self.keys.codes]
        held = host(self.per_request, fwd_info.rid_handle, "main", 0)
        if held["ended"] and chunk.start_offset > held["last_window"]:
            # Past the last window: the stream's final chunk after the window that held
            # the stop code, which only repeats that window's tail. It still runs,
            # voicing nothing (``forward_batched``): a skipped step never completes the
            # node, which then refuses the stream's end. Told by the last window's
            # offset, not by the chunk being all look-back: a stream that ends at a
            # full window without the stop code has a last window that is all look-back.
            return NodeInputs(tensor_inputs={"codes": codes[None, :0]}, kwargs={"last": True})
        first = chunk.start_offset == 0
        if first:
            codes = torch.cat([self._lead_silence, codes])
        # the walk's step metadata reaches each chunk the stream triggers, not only the first
        voice = self.bank.index(fwd_info.step_metadata.get(self.keys.voice))
        if not last and codes.shape[0] != WINDOW:
            raise RuntimeError(f"a window before the last holds {codes.shape[0]} codes, and HiFT's noise for window k "
                               f"assumes every window before it held {WINDOW}")
        # Window k starts 25k codes into the silence-led stream (its look-back
        # included), which is what its caches hold before it.
        k = (chunk.start_offset + LEAD_SILENCE) // HOP
        dit = CACHE_FAMILIES["dit"]
        dit_written = dit.rate * HOP * k
        noise_offset = dit.retained(dit.rate * self.bank.prompt_tokens[voice], dit_written)
        # what a last window's commit leaves for the chunk after it: where the last window starts
        fields = {"ended": True, "last_window": chunk.start_offset} if last else {}
        return NodeInputs(tensor_inputs={"codes": codes[None]}, kwargs={
            "last": last, "first": first, "voice": voice, "host": fields, "noise_offset": noise_offset,
            "dit_written": dit_written,
            # HiFT's excitation, from the request's seed and the window's index
            "draw": (fwd_info.random_seed, k),
        })

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep:
        # This step admits the windows (slots, positions), which has to happen before
        # the forward, and leaves each row's host fields. Each region's step plans the
        # rows it runs and gives them their entries. A row with no codes voices nothing
        # (``forward_batched``), so it declares nothing: a reply that never reached the
        # TTS takes no slot.
        rows = [(rid, inp) for rid, inp in zip(request_ids, inputs, strict=True) if inp.tensor_inputs["codes"].shape[1]]
        rids = [rid for rid, _ in rows]
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid in rids],
            state={
                **{key: ArrayStep() for key in self.keys.state.values()},
                self.keys.request: ArrayStep(host={
                    (rid, "main", 0): inp.kwargs["host"] for rid, inp in rows if inp.kwargs["host"]
                }),
                **self._cache_steps(
                    rids, [inp.tensor_inputs["codes"].shape[1] for _, inp in rows],
                    [inp.kwargs["last"] for _, inp in rows], voices={rid: inp.kwargs["voice"] for rid, inp in rows},
                ),
            },
        )

    # Each window's flow, HiFT and cross-fade run through a piecewise region: replayed
    # when a graph fits, else eagerly through the same step, so the resources always hold
    # the plan of exactly the rows being run. Shapes that are never captured (a
    # one-window reply's HiFT) and the cross-fade have eager-only regions.

    @staticmethod
    def flow_region(num_tokens: int, last: bool) -> str:
        return f"t2w/flow{'_last' if last else ''}{num_tokens}"

    @staticmethod
    def hift_region(first: bool, frames: int, last: bool) -> str:
        return f"t2w/hift/{'first' if first else 'next'}{'_last' if last else ''}{frames}"

    @staticmethod
    def eager_region(stage: str, first: bool, last: bool) -> str:
        return f"t2w/{stage}_eager/{'first' if first else 'next'}{'_last' if last else ''}"

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype, tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        def static_tokens(num_tokens: int):
            def make(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
                return {
                    "tokens": torch.zeros(shape.bs, num_tokens, dtype=torch.int32, device=device),
                    # each row's voice, an index into ``bank``
                    "voices": torch.zeros(shape.bs, dtype=torch.long, device=device),
                    # rows on their request's first window, which read their voice's conv state
                    "fresh": torch.zeros(shape.bs, dtype=torch.bool, device=device),
                    # each row's DiT kept count, where its flow noise starts
                    "noise_offsets": torch.zeros(shape.bs, dtype=torch.long, device=device),
                }
            return make

        def static_hift(first: bool, frames: int):
            def make(shape: PiecewiseCaptureShape) -> dict[str, torch.Tensor]:
                noise = self.model.window_noise(frames, first, [None] * shape.bs)
                return {"mel": torch.zeros(shape.bs, MEL_BINS, frames, device=device),
                        "phase": noise.phase, "harmonic": noise.harmonic}
            return make

        def one_value(part: str):
            def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
                return SubmoduleStep(
                    segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                    state={self.keys.state[part]: ArrayStep()},
                )
            return declare_step

        def declare_flow(last: bool):
            def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
                # a last window attends its own entries and keeps none
                caches = self._cache_steps(request_ids, list(seq_lens), [last] * len(request_ids), keep=not last)
                return SubmoduleStep(
                    segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                    state={self.keys.state["conv"]: ArrayStep(), **caches},
                )
            return declare_step

        def flow(last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                inputs = call.static_inputs
                return {"mel": self.model.window_flow(
                    self.slots["conv"], self.bank, inputs["voices"], inputs["fresh"], inputs["tokens"], self._flow_kv(),
                    inputs["noise_offsets"], last,
                )}
            return run

        def hift(first: bool, last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                inputs = call.static_inputs
                return self.model.window_vocode(
                    self.slots["vocoder"], inputs["mel"], first, last,
                    HiFTNoise(phase=inputs["phase"], harmonic=inputs["harmonic"]),
                )
            return run

        def finish(first: bool, last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                inputs = call.static_inputs
                return {"speech": self.model.window_finish(
                    self.slots["tail"], inputs["magnitude"], inputs["phase"], first, last,
                )}
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
                hift(first, False), static_hift(first, full), one_value("vocoder"), full,
                self.WINDOW_CAPTURE_BATCH_SIZES)
        for n in LAST_WINDOW_TOKENS:
            frames = window_frames(n, True)
            configs[self.flow_region(n, True)] = region(
                flow(True), static_tokens(n), declare_flow(True), n, self.LAST_CAPTURE_BATCH_SIZES)
            configs[self.hift_region(False, frames, True)] = region(
                hift(False, True), static_hift(False, frames), one_value("vocoder"), frames,
                self.LAST_CAPTURE_BATCH_SIZES)
        for last in (False, True):
            configs[self.eager_region("flow", False, last)] = region(
                flow(last), static_tokens(WINDOW), declare_flow(last), WINDOW, [])
            for first in (False, True):
                configs[self.eager_region("hift", first, last)] = region(
                    hift(first, last), static_hift(first, full), one_value("vocoder"), full, [])
                # the inverse STFT syncs, so the cross-fade is never captured
                configs[self.eager_region("finish", first, last)] = region(
                    finish(first, last), lambda shape: {}, one_value("tail"), full, [])
        return configs

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        return {"rows": [(inp.tensor_inputs["codes"].to(device), inp.kwargs) for inp in inputs]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows: list | None = None, **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Each window in three stages: the flow, batched over every row of one window length
        whatever its voice and position, then HiFT and the cross-fade, batched over rows that
        are all or none their request's first window."""
        runners = engine_inputs.piecewise_runners
        out, row_kwargs, flows = {}, {}, {}
        for rid, (codes, kw) in zip(engine_inputs.request_ids, rows, strict=True):
            if not codes.shape[1]:
                # past its last window (``prepare_inputs``)
                out[rid] = {}
                continue
            row_kwargs[rid] = kw
            flows.setdefault((codes.shape[1], kw["last"]), []).append((rid, codes))

        mels, hifts = {}, {}
        for (n, last), group in flows.items():
            runner = runners.get(self.flow_region(n, last)) or runners[self.eager_region("flow", False, last)]
            for chunk in self._chunks(runner, group, last):
                rids = [rid for rid, _ in chunk]
                tokens = torch.cat([codes for _, codes in chunk])
                mel = self._run_flow(runner, rids, tokens, [row_kwargs[rid] for rid in rids])
                for i, rid in enumerate(rids):
                    mels[rid] = mel[i]
                    hifts.setdefault((row_kwargs[rid]["first"], mel.shape[-1], last), []).append(rid)

        for (is_first, frames, last), rids in hifts.items():
            runner = runners.get(self.hift_region(is_first, frames, last)) or runners[
                self.eager_region("hift", is_first, last)]
            finish = runners[self.eager_region("finish", is_first, last)]
            for chunk in self._chunks(runner, rids, last):
                mel = torch.stack([mels[rid] for rid in chunk])
                draws = [row_kwargs[rid]["draw"] for rid in chunk]
                wav = self._run_hift(runner, finish, chunk, mel, is_first, last, draws)
                for i, rid in enumerate(chunk):
                    out[rid] = {self.keys.audio: [wav[i]]}
        return out

    def _chunks(self, runner, group: list, last: bool) -> list[list]:
        """``group`` in batches its region can replay (all of it when it runs eagerly)."""
        if not runner.any_graphs:
            return [group]
        cap = (self.LAST_CAPTURE_BATCH_SIZES if last else self.WINDOW_CAPTURE_BATCH_SIZES)[-1]
        return [group[i:i + cap] for i in range(0, len(group), cap)]

    def _run_flow(self, runner, rids: list[str], tokens: torch.Tensor, kwargs: list[dict]) -> torch.Tensor:
        """``window_flow`` for these rows (``[B, 80, frames]``)."""
        dit = self.caches["dit"]
        for rid, kw in zip(rids, kwargs, strict=True):
            if dit.written(rid) != kw["dit_written"]:
                raise RuntimeError(
                    f"{rid!r}'s DiT cache committed {dit.written(rid)} entries and its stream offset puts "
                    f"{kw['dit_written']} before this window: its noise offset, counted from the stream offset, "
                    "would index another window's noise"
                )
        device = tokens.device
        # pinned and queued: a pageable copy would wait out the running step
        out = runner.run(
            static_inputs={
                "tokens": tokens,
                "voices": to_device_nonblocking([kw["voice"] for kw in kwargs], device, torch.long),
                "fresh": to_device_nonblocking([kw["first"] for kw in kwargs], device, torch.bool),
                "noise_offsets": to_device_nonblocking([kw["noise_offset"] for kw in kwargs], device, torch.long),
            },
            request_ids=rids, seq_lens=[tokens.shape[1]] * len(rids), real_bs=len(rids),
        )
        # copied out: the next replay of this region reuses the buffer
        return out.get_view("mel").clone()

    def _run_hift(
        self, runner, finish, rids: list[str], mel: torch.Tensor, first: bool, last: bool,
        draws: list[tuple[int, int]],
    ) -> torch.Tensor:
        """``window_vocode`` and then ``window_finish``, each through its region: the rows' samples."""
        noise = self.model.window_noise(mel.shape[-1], first, draws)
        out = runner.run(
            static_inputs={"mel": mel, "phase": noise.phase, "harmonic": noise.harmonic},
            request_ids=rids, seq_lens=[mel.shape[-1]] * len(rids), real_bs=len(rids),
        )
        spectrum = {"magnitude": out.get_view("magnitude"), "phase": out.get_view("phase")}
        speech = finish.run(
            static_inputs=spectrum, request_ids=rids, seq_lens=[mel.shape[-1]] * len(rids), real_bs=len(rids),
        )
        return speech.get_view("speech")
