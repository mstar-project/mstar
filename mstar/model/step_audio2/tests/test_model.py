"""Step-Audio2's walks, stop and withhold logic, step declarations and serving config (CPU).

Without weights: the model is built off a default config with a stand-in tokenizer, or off
the checkpoint's own tokenizer where ``STEPAUDIO2_CKPT`` has it (the prompt tests against
the reference's recorded ids skip otherwise).
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mstar.conductor.request_info import FINAL_PREFILL
from mstar.model.components.qwen3_lm import Qwen3LMConfig
from mstar.model.step_audio2.config import (
    AUDIO_ATTN,
    AUDIO_EMBEDS,
    AUDIO_END,
    AUDIO_MEL_FRAMES,
    AUDIO_MELS,
    AUDIO_PATCH,
    AUDIO_START,
    BOT,
    DECODE,
    DECODE_LOOP,
    DECODE_MIXED,
    DECODE_SPOKEN,
    ENDOFTEXT,
    EOT,
    LLM_SAMPLER,
    NUM_SPEECH_CODES,
    PREFILL_AUDIO,
    PREFILL_TEXT,
    SPEECH_BASE,
    SPEECH_IDS,
    SPEECH_MAX_TOKENS,
    T2W_CHUNK,
    TTS_END,
    TTS_PAD,
    TTS_START,
    AudioEncoderConfig,
    StepAudio2Config,
    audio_tokens,
    is_speech_id,
)
from mstar.model.step_audio2.step_audio2_model import SPEECH_MAX_OUTPUT_TOKENS
from mstar.model.submodule_base import END_PARTITION, BatchedStopCheck, HostRows, StopCheck

ROOT = Path(__file__).resolve().parents[4]
CKPT = Path(os.environ.get("STEPAUDIO2_CKPT", "/workspace/stepaudio2-ref/hf/Step-Audio-2-mini"))
ORACLE = Path(os.environ.get("STEPAUDIO2_ORACLE", "/workspace/stepaudio2-ref/oracle/rec"))
ASSETS = Path(os.environ.get("STEPAUDIO2_ASSETS", "/workspace/stepaudio2-ref/github/Step-Audio2/assets"))


class _StubTokenizer:
    """One id per character, so prompts tokenize without a vocabulary, but for the special
    tokens the prompt's audio and turns are written in, which are their own ids."""

    SPECIAL = {"<audio_start>": AUDIO_START, "<audio_end>": AUDIO_END, "<audio_patch>": AUDIO_PATCH,
               "<|BOT|>": BOT, "<|EOT|>": EOT, "<tts_start>": TTS_START}

    def __call__(self, text, add_special_tokens=False):
        import re

        ids = []
        for piece in re.split("(" + "|".join(map(re.escape, self.SPECIAL)) + ")", text):
            ids += [self.SPECIAL[piece]] if piece in self.SPECIAL else [ord(c) % 1000 for c in piece]
        return {"input_ids": ids}


def a_config() -> StepAudio2Config:
    return StepAudio2Config(llm=Qwen3LMConfig(), audio=AudioEncoderConfig())


def voices_dir(tmp_path: Path) -> Path:
    import soundfile as sf

    root = tmp_path / "voices"
    root.mkdir()
    for name, seconds in (("default_female", 2.0), ("default_male", 1.5)):
        sf.write(root / f"{name}.wav", torch.zeros(int(48000 * seconds)).numpy(), 48000)
    return root


def a_model(tmp_path: Path, tokenizer=None):
    """A StepAudio2Model without its checkpoint: a default config, the voices of a temp dir."""
    from mstar.model.step_audio2.step_audio2_model import StepAudio2Model, _voice_files

    model = object.__new__(StepAudio2Model)
    model.config = a_config()
    model.max_seq_len = 4096
    model.tokenizer = tokenizer or _StubTokenizer()
    model.voices = _voice_files(str(voices_dir(tmp_path)))
    model._submodule_cache = {}
    model._detokenizer = None
    return model


def an_llm(stop_ids=(EOT, TTS_END, ENDOFTEXT)):
    from mstar.model.step_audio2.submodules import LLMSubmodule

    config = a_config()
    config.stop_token_ids = tuple(stop_ids)
    llm = LLMSubmodule(torch.nn.Identity(), config)
    llm.node_resources = {LLM_SAMPLER: SimpleNamespace(has_slot_masters=True)}
    return llm


def row_info(walk: str, ignore_eos: bool = False, max_tokens: int = 64, final: bool = False):
    return SimpleNamespace(
        graph_walk=walk, max_tokens=max_tokens, step_metadata={FINAL_PREFILL: final}, dynamic_loop_iter_counts={},
        resource_configs={LLM_SAMPLER: SimpleNamespace(ignore_eos=ignore_eos)},
    )


# --- walks and graph ------------------------------------------------------------------------

def test_a_reply_plans_one_prefill_then_its_decode_and_only_a_spoken_one_runs_the_vocoder(tmp_path):
    from mstar.model.step_audio2.step_audio2_model import MAIN, TOKEN2WAV

    model = a_model(tmp_path)
    graphs = model.get_graph_walk_graphs()
    for audio_in, prefill in ((False, PREFILL_TEXT), (True, PREFILL_AUDIO)):
        signals = {AUDIO_MELS: [object()]} if audio_in else {}
        kwargs = {AUDIO_MEL_FRAMES: [380]} if audio_in else {}
        for outputs, decode in ((["text"], DECODE), (["text", "audio"], DECODE_SPOKEN)):
            plan = model.walk_plan(MAIN, ["audio" if audio_in else "text"], outputs, signals, kwargs)
            assert [(s.walk, dict(s.bind)) for s in plan] == [(prefill, {}), (decode, {"text_inputs": "new_token"})]
            assert plan[0].step_metadata[FINAL_PREFILL]
            assert plan[0].step_metadata.get(AUDIO_MEL_FRAMES) == kwargs.get(AUDIO_MEL_FRAMES)
            for step in plan:
                inputs = set().union(*(node.input_names for node in graphs[step.walk].get_nodes().values()))
                assert set(step.bind) <= inputs
    assert model.walk_plan(TOKEN2WAV, ["text"], ["text"], {}, {}) == []
    [chunk] = model.walk_plan(TOKEN2WAV, ["text"], ["text", "audio"], {}, {"voice": "default_male"})
    assert (chunk.walk, dict(chunk.step_metadata)) == (T2W_CHUNK, {"voice": "default_male"})
    assert model.walk_plan(TOKEN2WAV, ["text"], ["text", "audio"], {}, {})[0].step_metadata["voice"] == "default_female"
    with pytest.raises(ValueError, match="unknown Step-Audio2 voice"):
        model.walk_plan(TOKEN2WAV, ["text"], ["text", "audio"], {}, {"voice": "nobody"})


def test_only_the_spoken_decode_streams_to_the_vocoder_in_the_reference_clients_windows(tmp_path):
    from mstar.model.step_audio2.step_audio2_model import MAIN, TOKEN2WAV
    from mstar.streaming.chunk_policy import SlidingWindowChunkPolicy

    model = a_model(tmp_path)
    graphs = model.get_graph_walk_graphs()

    def streams(walk):
        return {(e.name, e.next_node) for e in graphs[walk].section.outputs if e.is_streaming}

    assert streams(DECODE_SPOKEN) == {(SPEECH_IDS, TOKEN2WAV)}
    assert streams(DECODE) == set()
    partitions = {p.name: p.graph_walks for p in model.get_partitions()}
    assert partitions == {MAIN: set(graphs) - {T2W_CHUNK}, TOKEN2WAV: {T2W_CHUNK}}
    (connection,) = model.get_partition_topology().connections
    policy = connection.chunk_policy_factory()
    assert connection.edge_name == SPEECH_IDS and isinstance(policy, SlidingWindowChunkPolicy)
    assert connection.by_value
    assert (policy.window_size(), policy.next_chunk_size(28)) == (28, 25)
    assert model.get_combined_graph_walks() == {"LLM": {DECODE_MIXED: {DECODE, DECODE_SPOKEN}}}


def test_a_text_decode_loops_back_on_the_device_and_a_spoken_ones_ids_go_by_value():
    llm = an_llm()
    assert llm.device_loopback_signals(DECODE) == {"text_inputs"}
    assert llm.device_loopback_signals(DECODE_SPOKEN) == frozenset()
    assert llm.inline_signals(DECODE) == {"text_inputs": "new_token"}
    # the speech id first: a step carries the first inline signal it outputs
    assert list(llm.inline_signals(DECODE_SPOKEN).items()) == [(SPEECH_IDS, "new_token"), ("text_inputs", "new_token")]


# --- stop and withhold -----------------------------------------------------------------------

SPOKEN_ROWS = [
    # token, ignore_eos -> stops, withheld
    (SPEECH_BASE, False, set(), False),
    (SPEECH_BASE + NUM_SPEECH_CODES - 1, False, set(), False),
    (SPEECH_BASE + NUM_SPEECH_CODES, False, set(), True),  # an <audio_N> past the codes
    (785, False, set(), True),  # text
    (TTS_PAD, False, set(), True),
    (TTS_END, False, {DECODE_LOOP}, True),
    (EOT, False, {DECODE_LOOP}, True),
    (ENDOFTEXT, False, {DECODE_LOOP}, True),
    (TTS_END, True, set(), True),
]


@pytest.mark.parametrize(("token", "ignore_eos", "stops", "withheld"), SPOKEN_ROWS)
def test_a_spoken_row_streams_only_its_speech_ids_and_stops_on_its_stop_ids(token, ignore_eos, stops, withheld):
    llm = an_llm()
    got = llm.check_stop("r", row_info(DECODE_SPOKEN, ignore_eos), {"new_token": [torch.tensor([token])]})
    if withheld:
        assert got == StopCheck(stops, withheld=(SPEECH_IDS,))
    else:
        assert got == stops
    assert is_speech_id(token) == (not withheld)


@pytest.mark.parametrize("token", [SPEECH_BASE, 785, TTS_PAD, TTS_END, EOT])
def test_a_text_row_withholds_nothing(token):
    got = an_llm().check_stop("r", row_info(DECODE), {"new_token": [torch.tensor([token])]})
    assert got == ({DECODE_LOOP} if token in (TTS_END, EOT) else set())


def test_the_batched_check_agrees_with_the_row_check_in_a_mixed_step():
    llm = an_llm()
    rows = [(f"s{i}", DECODE_SPOKEN, t, ig) for i, (t, ig, _, _) in enumerate(SPOKEN_ROWS)]
    rows += [("t0", DECODE, 785, False), ("t1", DECODE, EOT, False), ("t2", DECODE, SPEECH_BASE, False)]
    rids = [rid for rid, *_ in rows]
    infos = {rid: row_info(walk, ig) for rid, walk, _, ig in rows}
    tokens = torch.tensor([[t] for _, _, t, _ in rows])
    batched = llm.check_stop_batched(rids, infos, HostRows(tuple(rids), {"new_token": tokens}))
    assert isinstance(batched, BatchedStopCheck)
    for i, rid in enumerate(rids):
        single = llm.check_stop(rid, infos[rid], {"new_token": [tokens[i]]})
        single = single if isinstance(single, StopCheck) else StopCheck(single)
        assert (set(batched.stops.get(rid, set())), batched.withheld.get(rid, ())) == (
            set(single.stops), single.withheld), rid


def test_a_prefill_stop_ends_the_partition():
    llm = an_llm()
    for walk in (PREFILL_TEXT, PREFILL_AUDIO):
        assert llm.check_stop("r", row_info(walk, final=True), {"new_token": [torch.tensor([EOT])]}) == {END_PARTITION}
        assert llm.check_stop("r", row_info(walk, max_tokens=1, final=True), {"new_token": [torch.tensor([7])]}) == {
            END_PARTITION}


def test_the_runtime_checks_only_a_text_decodes_stop_ids():
    """The runtime's check never calls check_stop, which is what withholds, so a walk with
    spoken rows keeps it."""
    llm = an_llm()
    assert set(llm.get_stop_policies(DECODE)) == {DECODE_LOOP}
    for walk in (DECODE_SPOKEN, PREFILL_TEXT, PREFILL_AUDIO):
        assert llm.get_stop_policies(walk) is None


def test_postprocess_routes_the_token_as_the_stream_only_on_a_spoken_row():
    llm = an_llm()
    token = [torch.tensor([SPEECH_BASE])]
    for walk, stream in ((DECODE, False), (DECODE_SPOKEN, True), (PREFILL_TEXT, False)):
        outputs = {"new_token": token}
        llm.postprocess("r", SimpleNamespace(graph_walk=walk), outputs)
        assert outputs["text_inputs"] is token
        assert (SPEECH_IDS in outputs) == stream


# --- step declarations ----------------------------------------------------------------------

def test_a_prefill_tracks_its_prompt_for_the_penalty_and_a_decode_does_not():
    from mstar.model.step_audio2.config import LLM_KV
    from mstar.model.submodule_base import ARNodeInputs

    llm = an_llm()
    ids = torch.tensor([5, 6, 7])
    step = llm.declare_step(PREFILL_TEXT, ["a"], [ARNodeInputs(input_ids=ids, input_seq_len=3)])
    assert [s.span for s in step.segments] == [3]
    assert step.steps[LLM_SAMPLER].prefill_tracked_tokens["a"] is ids
    assert LLM_KV in step.steps
    step = llm.declare_step(DECODE_SPOKEN, ["a", "b"], [ARNodeInputs(input_seq_len=1)] * 2)
    assert [s.span for s in step.segments] == [1, 1] and not step.steps[LLM_SAMPLER].prefill_tracked_tokens


def test_an_audio_prefill_refuses_tower_rows_that_miss_its_placeholders():
    llm = an_llm()
    info = SimpleNamespace(step_metadata={AUDIO_MEL_FRAMES: [380, 202]})
    pads = audio_tokens(380) + audio_tokens(202)
    inputs = {"text_inputs": [torch.zeros(80, dtype=torch.long)], AUDIO_EMBEDS: [torch.zeros(pads, 8)]}
    got = llm.prepare_inputs(PREFILL_AUDIO, info, inputs)
    assert got.input_seq_len == 80 and got.tensor_inputs[AUDIO_EMBEDS].shape[0] == pads
    with pytest.raises(ValueError, match="placeholders"):
        llm.prepare_inputs(PREFILL_AUDIO, info, {**inputs, AUDIO_EMBEDS: [torch.zeros(pads - 1, 8)]})


def test_the_tower_steps_each_clip_as_a_key_prefix_segment_laid_out_against_its_own_request():
    from mstar.model.step_audio2.components.audio import request_layout
    from mstar.model.step_audio2.submodules import AudioEncoderSubmodule

    tower = AudioEncoderSubmodule(torch.nn.Identity(), a_config())
    requests = {"a": [2502, 225], "b": [380]}
    inputs = [
        tower.prepare_inputs(PREFILL_AUDIO, SimpleNamespace(step_metadata={AUDIO_MEL_FRAMES: frames}, rid_handle=rid),
                             {AUDIO_MELS: [torch.zeros(128, sum(frames))]})
        for rid, frames in requests.items()
    ]
    step = tower.declare_step(PREFILL_AUDIO, list(requests), inputs).steps[AUDIO_ATTN]
    want = [(rid, c) for rid, frames in requests.items() for c in request_layout(frames)]
    assert [(s.request_id, s.span) for s in step.segments] == [(rid, c.frames) for rid, c in want]
    assert step.key_lens == tuple(c.key_len for _, c in want)
    # the short clip is laid out against its own request's longest, not the other request's
    assert want[1][1] != request_layout([225])[0] and want[2][1] == request_layout([380])[0]


def test_a_leased_tower_step_leaves_attention_to_the_block_loop_which_reads_the_clips_key_lens():
    """Leased, the captured block loop declares the request's attention itself: each clip a
    segment attending its valid keys, as the eager step does. Its capture's padding row,
    which no request owns, attends all of its frames."""
    from mstar.model.step_audio2.components.audio import request_layout
    from mstar.model.step_audio2.submodules import AUDIO_BLOCK_LOOP, AudioEncoderSubmodule

    tower = AudioEncoderSubmodule(torch.nn.Identity(), a_config())
    frames = [2502, 225]
    info = SimpleNamespace(step_metadata={AUDIO_MEL_FRAMES: frames}, rid_handle=7)
    inp = tower.prepare_inputs(PREFILL_AUDIO, info, {AUDIO_MELS: [torch.zeros(128, sum(frames))]})
    assert tower.declare_step(PREFILL_AUDIO, [7], [inp], piecewise_leases={AUDIO_BLOCK_LOOP: object()}) is None
    region = tower.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[AUDIO_BLOCK_LOOP]
    layout = request_layout(frames)
    step = region.region_step([7], [c.frames for c in layout]).steps[AUDIO_ATTN]
    eager = tower.declare_step(PREFILL_AUDIO, [7], [inp]).steps[AUDIO_ATTN]
    assert step.segments == eager.segments and step.key_lens == eager.key_lens
    assert step.key_lens == tuple(c.key_len for c in layout) != tuple(c.frames for c in layout)
    pad = region.region_step([-1], [512]).steps[AUDIO_ATTN]
    assert pad.key_lens == (512,)


def test_the_towers_block_loop_leases_within_the_resources_segment_limit():
    """A replay's plan holds the attention resource's ``max_segments_per_request`` clips, as
    configured: a request with more gets no lease and runs eagerly."""
    from mstar.model.step_audio2.submodules import AUDIO_BLOCK_LOOP, AudioEncoderSubmodule
    from mstar.model.submodule_base import NodeInputs

    tower = AudioEncoderSubmodule(torch.nn.Identity(), a_config())
    tower.bind_node_resources({AUDIO_ATTN: SimpleNamespace(max_segments_per_request=2)})
    region = tower.get_piecewise_accelerator_graph_configs(torch.device("cpu"), torch.float32)[AUDIO_BLOCK_LOOP]

    def request(clips):
        return [NodeInputs(input_seq_len=100 * clips, kwargs={"layout": [None] * clips})]

    assert region.can_lease(request(2)) and not region.can_lease(request(3))
    assert tower.max_batch_size(PREFILL_AUDIO) == 1


def test_every_decode_walk_replays_the_decode_capture_and_both_prefills_the_prefill_capture():
    from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig, PackedAcceleratorGraphConfig

    decode, prefill = an_llm().get_accelerator_graph_configs(torch.device("cpu"))
    assert isinstance(decode, BatchedAcceleratorGraphConfig) and isinstance(prefill, PackedAcceleratorGraphConfig)
    assert {decode.capture_graph_walk, *decode.replay_graph_walks} == {DECODE, DECODE_SPOKEN, DECODE_MIXED}
    assert {prefill.capture_graph_walk, *prefill.replay_graph_walks} == {PREFILL_TEXT, PREFILL_AUDIO}


def test_a_prefill_batch_is_embedded_with_each_rows_tower_rows_at_its_placeholders():
    """The prefill capture takes embeddings: ``preprocess`` splices each audio row's tower
    output in at its placeholders, rows in order, beside a text row; a decode takes ids."""
    from mstar.model.step_audio2.submodules import LLMSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(AUDIO_PATCH + 1, 4)

        def embed(self, ids):
            return self.embed_tokens(ids)

    llm = LLMSubmodule(Embed(), a_config())
    a = torch.tensor([1, AUDIO_PATCH, AUDIO_PATCH, 2])
    b = torch.tensor([3, 4])
    c = torch.tensor([AUDIO_PATCH, 5])
    rows_a, rows_c = torch.randn(2, 4), torch.randn(1, 4)
    inputs = [
        ARNodeInputs(input_seq_len=4, input_ids=a, tensor_inputs={AUDIO_EMBEDS: rows_a}),
        ARNodeInputs(input_seq_len=2, input_ids=b),
        ARNodeInputs(input_seq_len=2, input_ids=c, tensor_inputs={AUDIO_EMBEDS: rows_c}),
    ]
    with torch.no_grad():
        got = llm.preprocess(PREFILL_AUDIO, None, inputs)["input_embeds"]
        want = llm.model.embed(torch.cat([a, b, c]))
    want[1:3], want[6:7] = rows_a, rows_c
    assert torch.equal(got, want)
    assert set(llm.preprocess(DECODE_SPOKEN, None, [ARNodeInputs(input_seq_len=1, input_ids=b[:1])])) == {"input_ids"}


# --- the vocoder's stream --------------------------------------------------------------------

def reference_windows(codes: list[int]) -> list[tuple[list[int], bool]]:
    """``examples-vllm-stream.py``'s ``stream_client``: 28-code windows advancing by 25, then
    whatever is left once the reply ends."""
    buf, out = list(codes), []
    while len(buf) >= 28:
        out.append((buf[:28], False))
        buf = buf[25:]
    if buf:
        out.append((buf, True))
    return out


# 0: an empty spoken reply; the last: the most a reply at its cap streams (its first token
# is the prefill's, which does not stream)
@pytest.mark.parametrize("n", [0, 1, 3, 25, 27, 28, 29, 53, 54, 100, 160, SPEECH_MAX_OUTPUT_TOKENS - 1])
def test_the_vocoder_voices_the_streams_windows_as_the_reference_client_does(tmp_path, n):
    """The spoken walk's speech ids through the stream's policy and ``prepare_inputs`` are the
    reference client's windows, every code voiced once, each window's caches holding what the
    windows before it wrote; ``preprocess`` turns the ids into codes on the device."""
    from mstar.model.components.token2wav.model import window_tokens
    from mstar.model.components.token2wav.submodule import Token2WavSubmodule
    from mstar.model.step_audio2.step_audio2_model import T2W_KEYS
    from mstar.model.submodule_base import InputMetadata
    from mstar.streaming.stream_buffer import StreamBuffer

    node = Token2WavSubmodule.__new__(Token2WavSubmodule)
    torch.nn.Module.__init__(node)
    node.request_states, node.eos_code, node.code_offset, node.keys = {}, None, SPEECH_BASE, T2W_KEYS
    node._lead_silence = torch.zeros(0, dtype=torch.int32)
    node.model = torch.nn.Linear(1, 1)  # its device
    info = SimpleNamespace(rid_handle=0, step_metadata={"voice": "default_female"}, random_seed=0)
    codes = torch.randint(0, NUM_SPEECH_CODES, (n,)).tolist()
    model = a_model(tmp_path)
    policy = model.get_partition_topology().connections[0].chunk_policy_factory()
    buffer = StreamBuffer(request_id=0, edge_name=SPEECH_IDS, from_partition="main", policy=policy)
    served, prepared = [], []
    for i, code in enumerate(codes + [None]):
        if code is None:
            buffer.signal_done()
        else:
            buffer.pre_read_register(f"t{i}")
            buffer.put(f"t{i}", torch.tensor([SPEECH_BASE + code]))
        while buffer.has_chunk_ready():
            chunk = buffer.pop_chunk()
            data = chunk.data["data"]
            got = node.prepare_inputs(
                T2W_CHUNK, info, {SPEECH_IDS: [data]} if data is not None else {},
                is_final_stream_chunk=chunk.is_final,
                input_metadata=InputMetadata(stream_chunks={SPEECH_IDS: chunk.info} if data is not None else {}),
            )
            if got.tensor_inputs["codes"].numel():
                prepared.append(got)
                served.append((
                    [t - SPEECH_BASE for t in got.tensor_inputs["codes"][0].tolist()], got.kwargs["last"]))
                assert node.request_state(0)["written"] == sum(window_tokens(len(w), last) for w, last in served[:-1])
    assert served == reference_windows(codes)
    rows = node.preprocess(T2W_CHUNK, None, prepared)["rows"]
    assert [r[0][0].tolist() for r in rows] == [w for w, _ in served]
    # within what the conformer's caches hold (``Token2WavStreaming.conformer_reply_tokens``)
    from mstar.model.step_audio2.token2wav import CONFORMER_REPLY_TOKENS

    assert sum(window_tokens(len(w), last) for w, last in served) <= CONFORMER_REPLY_TOKENS


def test_a_stream_without_an_offset_hands_its_codes_over_as_they_are():
    """MiniCPM-o's stream is its codes, so no subtraction runs for it."""
    from mstar.model.components.token2wav.submodule import Token2WavSubmodule
    from mstar.model.submodule_base import NodeInputs

    node = Token2WavSubmodule.__new__(Token2WavSubmodule)
    torch.nn.Module.__init__(node)
    node.model = torch.nn.Linear(1, 1)
    codes = torch.tensor([[5, 6, 7]], dtype=torch.int32)
    inputs = [NodeInputs(input_seq_len=3, tensor_inputs={"codes": codes}, kwargs={"last": True})]
    for offset, same in ((0, True), (SPEECH_BASE, False)):
        node.code_offset = offset
        (row,) = node.preprocess(T2W_CHUNK, None, inputs)["rows"]
        assert (row[0] is codes) == same


def test_the_batched_stop_check_reads_the_tokens_once():
    class Counting(HostRows):
        reads = 0

        def first_values(self, name, request_ids):
            Counting.reads += 1
            return super().first_values(name, request_ids)

    llm = an_llm()
    rids = ["s", "t"]
    infos = {"s": row_info(DECODE_SPOKEN), "t": row_info(DECODE)}
    got = llm.check_stop_batched(rids, infos, Counting(tuple(rids), {"new_token": torch.tensor([[785], [EOT]])}))
    assert Counting.reads == 1
    assert (got.stops, got.withheld) == ({"t": {DECODE_LOOP}}, {"s": (SPEECH_IDS,)})


def test_startup_checks_the_spoken_streams_withhold(tmp_path):
    """The stream is declared withheld, so startup checks the edge can go without an item,
    and that the walk leaves its stops to check_stop, which is what withholds."""
    from mstar.worker.engine_manager import _refuse_unroutable_withholds

    llm = an_llm()
    assert SPEECH_IDS in llm.token_stop.withhold
    model = a_model(tmp_path)
    _refuse_unroutable_withholds({"LLM": llm}, model)
    # were the runtime to check the spoken walk's stops itself, nothing would withhold
    llm.get_stop_policies = lambda walk: {}
    with pytest.raises(ValueError, match="get_stop_policies"):
        _refuse_unroutable_withholds({"LLM": llm}, model)


# --- prompt intake ---------------------------------------------------------------------------

def test_a_clip_to_clone_and_inputs_it_cannot_take_are_refused(tmp_path):
    from mstar.model.step_audio2.constants import REF_AUDIO_REFUSAL

    model = a_model(tmp_path)
    with pytest.raises(ValueError, match="ref_audio"):
        model.process_prompt("hi", ["text"], ["text", "audio"], ref_audio="data:audio/wav;base64,AAAA")
    assert "ref_audio" in REF_AUDIO_REFUSAL
    with pytest.raises(ValueError, match="text and audio inputs"):
        model.process_prompt("hi", ["text", "image"], ["text"], tensors={"image_inputs": [torch.zeros(3, 4, 4)]})
    with pytest.raises(ValueError, match="outputs text and audio"):
        model.process_prompt("hi", ["text"], ["image"])
    for voice in ("nobody", ""):
        with pytest.raises(ValueError, match="unknown Step-Audio2 voice"):
            model.process_prompt("hi", ["text"], ["text", "audio"], voice=voice)
    assert model.check_voice(None) == "default_female" and model.check_voice("default_male") == "default_male"


@pytest.mark.parametrize("seconds", [25.005, 50.005])
def test_audio_just_over_a_multiple_of_25_s_is_served_without_its_empty_window(tmp_path, seconds):
    """Its last window has no rows; the prompt keeps that window's empty run, as the
    reference's does, and the tower gets only the windows with rows."""
    model = a_model(tmp_path)
    audio = torch.randn(int(16000 * seconds))
    out = model.process_prompt(None, ["audio"], ["text"], tensors={"audio_inputs": [audio]})
    ids = out.new_input_tensors["text_inputs"][0].tolist()
    full = int(seconds // 25)
    assert out.metadata[AUDIO_MEL_FRAMES] == [2502] * full
    assert out.new_input_tensors[AUDIO_MELS][0].shape[-1] == 2502 * full
    assert ids.count(AUDIO_START) == full + 1 and ids.count(AUDIO_PATCH) == full * audio_tokens(2502)
    last_start = len(ids) - 1 - ids[::-1].index(AUDIO_START)
    assert ids[last_start + 1] == AUDIO_END


def test_an_audio_with_no_window_of_rows_is_served_as_its_empty_run(tmp_path):
    """Under 321 samples an audio has no rows; the reference renders it as an empty run
    and serves the rest of the request, as here: beside a real clip only that clip is
    encoded, and alone the request prefills as text."""
    model = a_model(tmp_path)
    short, real = torch.randn(320), torch.randn(16000 * 2)
    mixed = model.process_prompt(None, ["audio", "audio"], ["text"], tensors={"audio_inputs": [short, real]})
    ids = mixed.new_input_tensors["text_inputs"][0].tolist()
    assert mixed.metadata[AUDIO_MEL_FRAMES] == [202]
    assert ids.count(AUDIO_START) == 2 and ids.count(AUDIO_PATCH) == audio_tokens(202)
    assert ids[ids.index(AUDIO_START) + 1] == AUDIO_END
    alone = model.process_prompt("what is this?", ["text", "audio"], ["text"], tensors={"audio_inputs": [short]})
    assert AUDIO_MELS not in alone.new_input_tensors and AUDIO_MEL_FRAMES not in alone.metadata
    ids = alone.new_input_tensors["text_inputs"][0].tolist()
    assert ids[ids.index(AUDIO_START) + 1] == AUDIO_END
    (layout,) = alone.metadata["kv_layout"].values()
    assert layout["main"].spans[0].walk == PREFILL_TEXT
    # and the plan follows what was sent: no mels, the text prefill
    from mstar.model.step_audio2.step_audio2_model import MAIN

    plan = model.walk_plan(MAIN, ["text", "audio"], ["text"], {}, {})
    assert plan[0].walk == PREFILL_TEXT


@pytest.mark.parametrize("with_audio", [False, True])
def test_an_audio_placeholder_written_in_the_text_is_refused(tmp_path, with_audio):
    """The splice puts the tower's rows at the prompt's <audio_patch> ids in order, so one in
    the text would shift every row after it."""
    model = a_model(tmp_path)
    inputs = ["text", "audio"] if with_audio else ["text"]
    tensors = {"audio_inputs": [torch.randn(16000)]} if with_audio else None
    with pytest.raises(ValueError, match="cannot be written"):
        model.process_prompt("say <audio_patch> twice", inputs, ["text"], tensors=tensors)
    model.process_prompt("say audio_patch twice", inputs, ["text"], tensors=tensors)


def test_a_spoken_reply_is_capped_to_what_the_vocoders_caches_hold(tmp_path):
    from mstar.model.step_audio2.token2wav import CONFORMER_REPLY_TOKENS

    model = a_model(tmp_path)
    assert SPEECH_MAX_OUTPUT_TOKENS <= CONFORMER_REPLY_TOKENS  # each token is at most one speech id
    spoken = model.process_prompt("hi", ["text"], ["text", "audio"])
    assert spoken.metadata[SPEECH_MAX_TOKENS] == SPEECH_MAX_OUTPUT_TOKENS
    assert SPEECH_MAX_TOKENS not in model.process_prompt("hi", ["text"], ["text"]).metadata
    with pytest.raises(ValueError, match="at most"):
        model.process_prompt("hi", ["text"], ["text", "audio"], max_output_tokens=SPEECH_MAX_OUTPUT_TOKENS + 1)
    caps = model.get_loop_caps(**{SPEECH_MAX_TOKENS: SPEECH_MAX_OUTPUT_TOKENS, "max_output_tokens": 10**6})
    assert caps == {DECODE_LOOP: SPEECH_MAX_OUTPUT_TOKENS - 1}
    assert model.get_loop_caps(max_output_tokens=1) == {DECODE_LOOP: 1}


def test_a_prompts_kv_layout_is_its_rows_then_a_row_per_decode_step(tmp_path):
    from mstar.model.step_audio2.config import LLM_KV

    model = a_model(tmp_path)
    out = model.process_prompt("hello", ["text"], ["text"])
    (ids,) = out.new_input_tensors["text_inputs"]
    layout = out.metadata["kv_layout"][LLM_KV]["main"]
    assert [(s.walk, s.length) for s in layout.spans] == [(PREFILL_TEXT, ids.numel())]
    assert layout.growth == ((DECODE_LOOP, 1),)
    assert set(model.kv_labels()[LLM_KV]["main"].growth) == {(DECODE, DECODE_LOOP), (DECODE_SPOKEN, DECODE_LOOP)}


def test_the_serving_config_places_every_node_once_and_sizes_the_vocoders_slots_alike(tmp_path):
    model = a_model(tmp_path)
    path = ROOT / "configs" / "step_audio2.yaml"
    config = yaml.safe_load(path.read_text())
    model.validate_config_yaml(config, str(path))
    placed = [n for group in config["node_groups"] for n in group["node_names"]]
    assert sorted(placed) == sorted(model.nodes)
    bad = {**config, "resources": {**config["resources"], "t2w_dit_kv": {"max_slots": 3}}}
    with pytest.raises(ValueError, match="one more than"):
        model.validate_config_yaml(bad, str(path))
    keys = {spec.resource_key for spec in model.get_node_resources()}
    assert set(config["resources"]) <= keys


def test_a_deployment_without_voices_is_refused(tmp_path, monkeypatch):
    from mstar.model.step_audio2.step_audio2_model import _voice_files

    with pytest.raises(ValueError, match="voices_dir"):
        _voice_files(None)
    # offline, an empty directory is not fetched into, and the refusal says how to fill it
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE.*curl -fLo"):
        _voice_files(str(tmp_path / "empty"))
    assert not list((tmp_path / "empty").iterdir())


class _Download:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self.data


def test_an_empty_voices_dir_gets_the_reference_voices_checked_against_their_hashes(tmp_path, monkeypatch):
    import hashlib

    from mstar.model.step_audio2 import step_audio2_model as m

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    clips = {name: name.encode() * 10 for name in m._DEFAULT_VOICES}
    monkeypatch.setattr(m, "_DEFAULT_VOICES", {name: hashlib.sha256(d).hexdigest() for name, d in clips.items()})
    asked = []

    def urlopen(url, timeout):
        asked.append(url)
        return _Download(clips[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(m.urllib.request, "urlopen", urlopen)
    voices = m._voice_files(str(tmp_path / "voices"))
    assert sorted(voices) == ["default_female", "default_male"]
    assert all(m._VOICE_COMMIT in url for url in asked)
    assert (tmp_path / "voices" / "default_male.wav").read_bytes() == clips["default_male.wav"]
    assert not list((tmp_path / "voices").glob("*.part"))
    # a directory that holds clips is left alone
    asked.clear()
    assert sorted(m._voice_files(str(tmp_path / "voices"))) == ["default_female", "default_male"]
    assert not asked
    # a download that doesn't match its hash is refused, and nothing is kept
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda url, timeout: _Download(b"not the clip"))
    with pytest.raises(ValueError, match="pinned hash"):
        m._voice_files(str(tmp_path / "other"))
    assert not list((tmp_path / "other").glob("*.wav"))


# --- against the reference, with the checkpoint's tokenizer ------------------------------------

needs_tokenizer = pytest.mark.skipif(
    not (CKPT / "tokenizer.json").exists() or not (ORACLE / "llm_s2t.pt").exists(),
    reason=f"needs the Step-Audio-2-mini tokenizer under {CKPT} and the oracle under {ORACLE}",
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    return transformers.Qwen2TokenizerFast.from_pretrained(str(CKPT))


@needs_tokenizer
@pytest.mark.parametrize("case", ["t2t", "s2t", "s2s", "s2t_long"])
def test_the_served_prompt_is_the_references(case, tmp_path, tokenizer):
    """The request as a chat client sends it (its system message, then the user's turn) is
    the reference's recorded prompt: ids, and each clip's log-mel in bf16."""
    from mstar.model.multimodal import PromptPart
    from mstar.model.step_audio2.step_audio2_model import StepAudio2Model

    if case != "t2t" and not (ASSETS / "paralinguistic_information_understanding.wav").exists():
        pytest.skip(f"needs the reference's assets under {ASSETS}")
    model = a_model(tmp_path, tokenizer)
    system = PromptPart("text", text="You are a helpful assistant.", role="system")
    clip = "paralinguistic_information_understanding.wav" if case == "s2t_long" else (
        "give_me_a_brief_introduction_to_the_great_wall.wav")
    user = {
        "t2t": [PromptPart("text", text="Give me a brief introduction to the Great Wall.", role="user")],
        "s2t_long": [PromptPart("text", text="Describe this audio.", role="user"), PromptPart("audio", role="user")],
    }.get(case, [PromptPart("audio", role="user")])
    parts = [system, *user]
    audios = [] if case == "t2t" else [StepAudio2Model.load_audio(model, str(ASSETS / clip), "cpu").data]
    out = model.process_prompt(
        None, [p.modality for p in parts], ["text", "audio"] if case == "s2s" else ["text"],
        tensors={"audio_inputs": audios} if audios else None, prompt_parts=parts,
    )
    rec = torch.load(ORACLE / f"llm_{case}.pt", weights_only=False)
    assert out.new_input_tensors["text_inputs"][0].tolist() == rec["input_ids"][0].tolist()
    if audios:
        want = torch.cat([m.to(torch.bfloat16) for m in rec["mels"]], dim=-1)
        assert torch.equal(out.new_input_tensors[AUDIO_MELS][0], want)
        assert out.metadata[AUDIO_MEL_FRAMES] == [m.shape[-1] for m in rec["mels"]]


@needs_tokenizer
def test_the_clients_text_leaves_out_speech_ids_and_special_tokens(tmp_path, tokenizer):
    model = a_model(tmp_path, tokenizer)
    reply = [785, SPEECH_BASE + 1490, TTS_PAD, 8513, SPEECH_BASE + 6560, SPEECH_BASE + 6600, 151666, 9736]
    detok = model.text_output().detokenizer
    assert detok.to_bytes(reply) == b"The Great Wall"
    assert b"".join(detok.token_bytes(t) for t in reply) == b"The Great Wall"
    assert set(model.text_output().stop_token_ids) == {EOT, TTS_END, ENDOFTEXT}
