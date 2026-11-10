"""Token2wav as a graph node: streamed speech-code windows -> 24 kHz audio."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

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
from mstar.engine.resources.kv.bounded import BoundedKVStep, StreamPosition
from mstar.engine.resources.recurrent.config import RecurrentStep
from mstar.model.chatterbox.components.s3gen_hift import HiFTNoise
from mstar.model.components.token2wav.flow import MEL_BINS
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    LAST_WINDOW_TOKENS,
    LEAD_SILENCE,
    SILENCE_CODE,
    WINDOW,
    AttentionCache,
    SlotRows,
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


class Token2WavKeys(NamedTuple):
    """The names a model gives token2wav's input signal and resources."""

    # the streamed speech codes this node reads
    codes: str
    # the ``RecurrentStateSpec`` holding each request's conv caches and HiFT tails
    state: str
    # cache family (``CACHE_FAMILIES``) -> its ``BoundedKVSpec`` resource key
    kv: Mapping[str, str]


class Token2WavSubmodule(NodeSubmodule):
    """Speech codes -> 24 kHz audio, one stream window per step: upstream's
    ``Token2wav.stream``, fed the way its ``streaming_generate`` feeds it (three
    silence codes first, 28-code windows advancing by 25, a last flush).

    The windows are the ``LeftContextChunkPolicy(25, 3)`` stream from the TTS;
    the first window gets the silence prepended here. A request's attention caches
    are on the ``T2W_KV`` bounded KV resources, which start from its voice's
    prepared caches; its conv caches and HiFT tails are its slot of ``self.keys.state``
    (``RecurrentStatePool``). Its stream position is the chunk's offset in the
    stream (``prepare_inputs``).

    Float32, but for the DiT's GEMMs, convolutions, attention and caches, which are
    ``DIT_COMPUTE_DTYPE``. Each window runs in two stages (``forward_batched``), each a piecewise
    capture bucketed by batch size: the flow (``t2w/<voice>/flow28``), for windows
    at any position, and HiFT, separately for a request's first window, whose
    input is shorter. A last window has one of a few lengths, each a one-row
    capture; other shapes run eagerly. The inverse STFT and cross-fade stay eager.
    The tests' one-request driver over the same modules is bit-exact against the
    reference at full fp32; this path attends the DiT's cache in another order and
    runs on fused kernels, with the DiT in float16.
    """

    WINDOW_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32]
    # a request ends once, so last windows of one length rarely meet in a pass
    LAST_CAPTURE_BATCH_SIZES = [1]

    disable_torch_compile = True
    disable_autocast = True

    def __init__(self, model, keys: Token2WavKeys, voices: dict, eos_code: int, dit_dtype: torch.dtype):
        """``keys``: the node's input signal and resource keys. ``voices``: the model's
        bundled ones, by name. ``dit_dtype``: the dtype of the DiT's caches, as the model
        declares them, which the DiT computes in."""
        super().__init__()
        self.model = model
        self.keys = keys
        self.eos_code = eos_code
        self.bank = VoiceBank.build(voices, dit_dtype=dit_dtype)
        self._lead_silence = torch.full((LEAD_SILENCE,), SILENCE_CODE, dtype=torch.int32)

    def compile_regions(self) -> dict[str, list[tuple]]:
        # HiFT up to its spectrum, compiled once with dynamic shapes (every region's
        # rows and frames): Inductor fuses its elementwise source and snake
        # activations, about half its GPU time. It compiles in the captures' warm-up.
        return {"model.hift.spectrum": []}

    def _caches(self, resources: Mapping) -> WindowCaches:
        """The step's ``WindowCaches``."""
        return WindowCaches(**{
            name: AttentionCache(resources[self.keys.kv[name]], self.bank.sources[name]) for name in CACHE_FAMILIES
        })

    def _voice_of(self, rid: str) -> tuple[int, int]:
        """``rid``'s voice, an index into ``bank``, and its prompt length in tokens."""
        voice = self.bank.index(self.request_state(rid)["voice"])
        return voice, self.bank.prompt_tokens[voice]

    def _voice_ids(self, rids: list[str], device: torch.device) -> torch.Tensor:
        # pinned and queued: a pageable copy would wait out the running step
        return to_device_nonblocking([self._voice_of(r)[0] for r in rids], device, torch.long)

    def _cache_steps(
        self, request_ids: list[str], num_tokens: list[int], lasts: list[bool],
    ) -> dict[str, BoundedKVStep]:
        """Each cache resource's step: every row's stream position (in its voice's stream)
        and what its window adds."""
        positions = []
        for rid in request_ids:
            req = self.request_states.get(rid)
            if req is None or "voice" not in req:
                # a capture's dummy row: the resource plans it as padding
                positions.append(None)
            else:
                voice, tokens = self._voice_of(rid)
                positions.append((tokens, req.get("written", 0), voice))
        steps = {}
        for name, family in CACHE_FAMILIES.items():
            steps[self.keys.kv[name]] = BoundedKVStep(
                segments=[
                    Segment(request_id=rid, label="main", span=family.span(n, last))
                    for rid, n, last in zip(request_ids, num_tokens, lasts, strict=True)
                ],
                positions={
                    rid: StreamPosition(*family.position(tokens, written), source=voice)
                    for rid, (tokens, written, voice) in (
                        (r, p) for r, p in zip(request_ids, positions, strict=True) if p is not None
                    )
                },
            )
        return steps

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
        # keyed by the worker handle, as the forward and the engine's cleanup are
        state = self.request_state(fwd_info.rid_handle)
        if chunk.start_offset > state.get("last_window", chunk.start_offset):
            # Past the last window: the stream's final chunk after the window that held
            # the stop code, which only repeats that window's tail. It still runs,
            # voicing nothing (``forward_batched``): a skipped step never completes the
            # node, which then refuses the stream's end. Told by the last window's
            # offset, not by the chunk being all look-back: a stream that ends at a
            # full window without the stop code has a last window that is all look-back.
            return NodeInputs(tensor_inputs={"codes": codes[None, :0]}, kwargs={"last": True})
        if last:
            state.add("last_window", chunk.start_offset)
        first = chunk.start_offset == 0
        if first:
            codes = torch.cat([self._lead_silence, codes])
            state.add("voice", fwd_info.step_metadata.get("voice"))
            # HiFT's excitation, window by window from the request's seed
            state.add("noise", torch.Generator(self.get_device()).manual_seed(fwd_info.random_seed))
        # Window k starts 25k codes into the silence-led stream (its look-back
        # included), which is what its caches hold before it (``_cache_steps``).
        state.add("written", 0 if first else chunk.start_offset + LEAD_SILENCE)
        return NodeInputs(tensor_inputs={"codes": codes[None]}, kwargs={"last": last, "first": first})

    def declare_step(
        self, graph_walk: str, request_ids: list[str], inputs: list[NodeInputs], **kwargs,
    ) -> SubmoduleStep:
        # This step admits the windows (slots, positions), which has to happen before
        # the forward. Its plan goes unused: each region re-plans the rows it runs.
        # TODO: an admit-only step would save that plan's host work.
        # A row with no codes voices nothing (``forward_batched``), so it declares
        # nothing: a reply that never reached the TTS takes no slot.
        rows = [(rid, inp) for rid, inp in zip(request_ids, inputs, strict=True) if inp.tensor_inputs["codes"].shape[1]]
        return SubmoduleStep(
            segments=[Segment(request_id=rid, label="main", span=1) for rid, _ in rows],
            steps={
                self.keys.state: RecurrentStep(),
                **self._cache_steps(
                    [rid for rid, _ in rows], [inp.tensor_inputs["codes"].shape[1] for _, inp in rows],
                    [inp.kwargs["last"] for _, inp in rows],
                ),
            },
        )

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
                noise = self.model.window_noise(frames, first, [None] * shape.bs)
                return {"mel": torch.zeros(shape.bs, MEL_BINS, frames, device=device),
                        "phase": noise.phase, "harmonic": noise.harmonic}
            return make

        def declare_flow(last: bool):
            def declare_step(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
                return SubmoduleStep(
                    segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                    steps={
                        self.keys.state: RecurrentStep(),
                        **self._cache_steps(request_ids, list(seq_lens), [last] * len(request_ids)),
                    },
                )
            return declare_step

        def declare_hift(request_ids: list[str], seq_lens: list[int]) -> SubmoduleStep:
            return SubmoduleStep(
                segments=[Segment(request_id=rid, label="main", span=1) for rid in request_ids],
                steps={self.keys.state: RecurrentStep()},
            )

        def flow(last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                pool = call.resources[self.keys.state]
                return {"mel": self.model.window_flow(
                    SlotRows(pool, pool.addressing("main")), self.bank, call.static_inputs["voices"],
                    call.static_inputs["tokens"], self._caches(call.resources), last,
                )}
            return run

        def hift(first: bool, last: bool):
            def run(call: PiecewiseCallInputs) -> dict[str, torch.Tensor]:
                pool, inputs = call.resources[self.keys.state], call.static_inputs
                return self.model.window_vocode(
                    SlotRows(pool, pool.addressing("main")), inputs["mel"], first, last,
                    HiFTNoise(phase=inputs["phase"], harmonic=inputs["harmonic"]),
                )
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
                hift(first, False), static_hift(first, full), declare_hift, full, self.WINDOW_CAPTURE_BATCH_SIZES)
        for n in LAST_WINDOW_TOKENS:
            frames = window_frames(n, True)
            configs[self.flow_region(n, True)] = region(
                flow(True), static_tokens(n), declare_flow(True), n, self.LAST_CAPTURE_BATCH_SIZES)
            configs[self.hift_region(False, frames, True)] = region(
                hift(False, True), static_hift(False, frames), declare_hift, frames, self.LAST_CAPTURE_BATCH_SIZES)
        for last in (False, True):
            configs[self.eager_region("flow", False, last)] = region(
                flow(last), static_tokens(WINDOW), declare_flow(last), WINDOW, [])
            for first in (False, True):
                configs[self.eager_region("hift", first, last)] = region(
                    hift(first, last), static_hift(first, full), declare_hift, full, [])
        return configs

    def forward(self, graph_walk: str, engine_inputs: ModelInputsFromEngine, **kwargs) -> NameToTensorList:
        return self.forward_batched(graph_walk, engine_inputs, **kwargs)[engine_inputs.request_ids[0]]

    def can_batch(self, batch: ExecutingBatch, model_inputs: list[NodeInputs]) -> bool:
        return True

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[NodeInputs],
    ) -> dict[str, Any]:
        device = self.get_device()
        return {"rows": [
            (inp.tensor_inputs["codes"].to(device), inp.kwargs["last"], inp.kwargs.get("first", False))
            for inp in inputs
        ]}

    def forward_batched(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, rows: list | None = None, **kwargs,
    ) -> dict[str, NameToTensorList]:
        """Each window in two stages: the flow, batched over every row of one window length
        whatever its voice and position, then HiFT, batched over rows that are all or none
        their request's first window."""
        runners = engine_inputs.piecewise_runners
        out, first, flows = {}, {}, {}
        for rid, (codes, last, is_first) in zip(engine_inputs.request_ids, rows, strict=True):
            if not codes.shape[1]:
                # past its last window (``prepare_inputs``)
                out[rid] = {}
                continue
            first[rid] = is_first
            flows.setdefault((codes.shape[1], last), []).append((rid, codes))

        mels, hifts = {}, {}
        for (n, last), group in flows.items():
            runner = runners.get(self.flow_region(n, last)) or runners[self.eager_region("flow", False, last)]
            for chunk in self._chunks(runner, group, last):
                rids = [rid for rid, _ in chunk]
                tokens = torch.cat([codes for _, codes in chunk])
                mel = self._run_flow(runner, rids, tokens)
                for i, rid in enumerate(rids):
                    mels[rid] = mel[i]
                    hifts.setdefault((first[rid], mel.shape[-1], last), []).append(rid)

        for (is_first, frames, last), rids in hifts.items():
            runner = runners.get(self.hift_region(is_first, frames, last)) or runners[
                self.eager_region("hift", is_first, last)]
            for chunk in self._chunks(runner, rids, last):
                mel = torch.stack([mels[rid] for rid in chunk])
                wav = self._run_hift(engine_inputs, runner, chunk, mel, is_first, last)
                for i, rid in enumerate(chunk):
                    out[rid] = {"audio_chunk": [wav[i]]}
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
        engine_inputs: ModelInputsFromEngine,
        runner,
        rids: list[str],
        mel: torch.Tensor,
        first: bool,
        last: bool,
    ) -> torch.Tensor:
        """``window_vocode`` and then (eagerly: the inverse STFT syncs) ``window_finish``,
        on the rows the region just planned: the rows' samples."""
        noise = self.model.window_noise(mel.shape[-1], first, [self.request_state(r)["noise"] for r in rids])
        out = runner.run(
            static_inputs={"mel": mel, "phase": noise.phase, "harmonic": noise.harmonic},
            request_ids=rids, seq_lens=[mel.shape[-1]] * len(rids), real_bs=len(rids),
        )
        pool = engine_inputs.resources[self.keys.state]
        state = SlotRows(pool, pool.addressing("main")).head(len(rids))
        return self.model.window_finish(state, out.get_view("magnitude"), out.get_view("phase"), first, last)
