"""``Mp4Stream`` writes one mp4 from pieces handed over as they are decoded, and the
generated audio into the same file; the data worker routes a request's pieces and
audio to it.

The pieces' order, the end of the stream (the bytes come once the last piece and
the audio are in), a failure, and a cancel mid-encode. A video-only stream writes
``encode_mp4``'s bytes; with audio, the video decodes to the same frames, and the
index goes to the front as ffmpeg's ``+faststart`` puts it.
"""

from __future__ import annotations

import io
import queue
import threading
from concurrent.futures import CancelledError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from mstar.api_server.data_worker import VIDEO_PART, PreprocessWorkerThread, RequestOutputState  # noqa: E402
from mstar.graph.loop_indices import NestedLoopIndices  # noqa: E402
from mstar.model.components.video_encode import (  # noqa: E402
    AudioFormat,
    Mp4Encoding,
    Mp4Stream,
    encode_mp4,
    faststart,
)
from mstar.testing_utils.model import StubModel  # noqa: E402
from mstar.utils import cpus  # noqa: E402

ULTRAFAST = {"crf": "18", "preset": "ultrafast", "threads": "0"}
ENCODING = Mp4Encoding(24, ULTRAFAST)
RATE = 16000


def _clip(frames: int = 12, height: int = 48, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(frames * height + width)
    y, x = np.mgrid[:height, :width]
    clip = np.stack([(x + 3 * t + y // 2) % 256 for t in range(frames)])[..., None].repeat(3, -1)
    return ((clip + rng.integers(0, 24, size=clip.shape)) % 256).astype(np.uint8)


def _pcm(frames: int = 12, channels: int = 2) -> bytes:
    samples = RATE * frames // 24
    t = np.arange(samples * channels) / channels
    return (np.sin(t / 5.0) * 8000).astype("<i2").tobytes()


def _decoded(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    with av.open(io.BytesIO(data)) as container:
        video = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            return video, None
        audio = np.concatenate([f.to_ndarray() for f in container.decode(audio=0)], axis=1)
    return video, audio


def _top_level(data: bytes) -> list[bytes]:
    kinds, pos = [], 0
    while pos < len(data):
        size = int.from_bytes(data[pos:pos + 4], "big")
        kinds.append(data[pos + 4:pos + 8])
        pos += size
    return kinds


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1), (2, 1, 0)])
def test_pieces_in_any_order_write_encode_mp4s_bytes(order):
    clip = _clip()
    pieces = np.array_split(clip, 3)
    stream = Mp4Stream(ENCODING, 3)
    for i in order:
        stream.add_part(i, pieces[i])
    assert stream.future.result(timeout=30) == encode_mp4(clip, 24, options=ULTRAFAST)


def test_the_bytes_come_once_the_last_piece_and_the_audio_are_in():
    clip = _clip()
    stream = Mp4Stream(ENCODING, 2, AudioFormat(RATE, 2))
    stream.add_part(1, clip[6:])
    stream.add_part(0, clip[:6])
    with pytest.raises(TimeoutError):
        stream.future.result(timeout=0.5)  # every piece in, the audio still out
    stream.add_audio(_pcm())
    data = stream.future.result(timeout=30)
    video, audio = _decoded(data)
    assert np.array_equal(video, _decoded(encode_mp4(clip, 24, options=ULTRAFAST))[0])
    assert audio.shape[0] == 2 and abs(audio.shape[1] - RATE // 2) <= 1024  # 0.5 s of stereo, AAC-framed
    assert _top_level(data)[:2] == [b"ftyp", b"moov"]  # the index first, as +faststart writes it


@pytest.mark.parametrize("channels", [1, 2])
def test_audio_before_the_first_piece_goes_into_the_same_pass(channels):
    clip = _clip()
    stream = Mp4Stream(ENCODING, 1, AudioFormat(RATE, channels))
    stream.add_audio(_pcm(channels=channels))
    stream.add_part(0, clip)
    with av.open(io.BytesIO(stream.future.result(timeout=30))) as container:
        assert container.streams.audio[0].codec_context.name == "aac"
        assert container.streams.audio[0].channels == channels
        assert container.streams.video[0].frames == len(clip)


def test_faststart_moves_the_index_as_ffmpeg_does(tmp_path):
    """The same file written by FFmpeg's own +faststart (to a path) and by
    ``faststart`` over the one written in memory."""
    clip, pcm = _clip(), _pcm()

    def write(target, options):
        container = av.open(target, mode="w", format="mp4", options=options)
        video = container.add_stream("libx264", rate=24)
        video.height, video.width, video.pix_fmt, video.options = 48, 64, "yuv420p", dict(ULTRAFAST)
        sound = container.add_stream("aac", rate=RATE, layout="stereo")
        frame = av.AudioFrame.from_ndarray(np.frombuffer(pcm, "<i2").reshape(1, -1), format="s16", layout="stereo")
        frame.sample_rate, frame.pts = RATE, 0
        for packet in [*sound.encode(frame), *sound.encode()]:
            container.mux(packet)
        for f in clip:
            for packet in video.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        container.close()

    path = tmp_path / "ffmpeg.mp4"
    write(str(path), {"movflags": "+faststart"})
    buffer = io.BytesIO()
    write(buffer, None)
    assert _top_level(buffer.getvalue())[-1] == b"moov"
    assert faststart(buffer.getvalue()) == path.read_bytes()
    assert faststart(path.read_bytes()) == path.read_bytes()  # already first: unchanged


@pytest.mark.parametrize("bad", ["size", "twice", "range"])
def test_a_bad_piece_fails_the_future(bad):
    clip = _clip()
    stream = Mp4Stream(ENCODING, 2)
    stream.add_part(0, clip[:6])
    if bad == "size":
        stream.add_part(1, clip[6:, :32])
    elif bad == "twice":
        stream.add_part(0, clip[:6])
    else:
        stream.add_part(2, clip[6:])
    with pytest.raises(ValueError):
        stream.future.result(timeout=30)
    stream.join(timeout=10)
    assert not stream._thread.is_alive()


def test_cancel_mid_encode_stops_the_writer():
    clip = _clip(48, 96, 128)
    stream = Mp4Stream(ENCODING, 4)
    first = stream.add_part(0, clip[:12])
    second = stream.add_part(1, clip[12:24])
    stream.cancel()
    third = stream.add_part(2, clip[24:36])  # after the cancel: dropped
    stream.join(timeout=30)
    assert not stream._thread.is_alive()
    assert stream.future.cancelled()
    with pytest.raises(CancelledError):
        stream.future.result()
    # every piece is let go once the writer has stopped, written or not
    assert all(f.done() for f in (first, second, third)) and third.cancelled()
    assert stream.add_part(3, clip[36:]).cancelled()  # after the writer's exit


def test_a_piece_is_let_go_once_written():
    """The writer reads a piece in place; its future says when it is done with it."""
    clip = _clip()
    stream = Mp4Stream(ENCODING, 2)
    later = stream.add_part(1, clip[6:])
    first = stream.add_part(0, clip[:6])
    first.result(timeout=30)
    later.result(timeout=30)
    assert stream.future.result(timeout=30) == encode_mp4(clip, 24, options=ULTRAFAST)


# ---------------------------------------------------------------- the data worker


class _Model(StubModel):
    def __init__(self, parts: int = 3):
        self.parts = parts

    def mp4_encoding(self, request_kwargs=None):
        return ENCODING

    def get_loop_caps(self, **model_kwargs):
        return {"decode": self.parts}

    def get_output_sample_rate(self, modality="audio"):
        return RATE

    def get_output_audio_channels(self, modality="audio"):
        return 2

    def postprocess(self, output, modality, request_kwargs=None):
        assert modality == "audio"
        return output.numpy().tobytes()


class _TensorManager:
    def __init__(self):
        self.ready: dict = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.dereferenced: list[str] = []

    def get_ready_tensors(self):
        ready, self.ready = self.ready, {}
        return ready

    def take_failed_reads(self):
        return []

    def get_tensor(self, uuid):
        return self.tensors[uuid]

    def dereference(self, uuid, n=1):
        self.dereferenced.append(uuid)

    def has_inflight_reads(self, request_id):
        return False

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        pass


class _Comm:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(16)), usable=16))
    wt = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=threading.Event(), communicator=_Comm(),
        tensor_manager=_TensorManager(), model=_Model(),
    )
    yield wt
    wt._encode_pool.shutdown(wait=True)


def _output(wt, request_id: str, modality: str, tensors: list[torch.Tensor], loop: str | None = None):
    state = wt.request_output_state[request_id]
    infos = []
    for i, tensor in enumerate(tensors):
        uuid = f"{request_id}/{state.next_sequence}"
        wt.tensor_manager.tensors[uuid] = tensor
        wt.tensor_uuid_to_metadata_per_request.setdefault(request_id, {})[uuid] = {}
        indices = NestedLoopIndices([loop], {loop: i}, 0) if loop else None
        state.order[uuid] = (state.next_sequence, indices)
        state.next_sequence += 1
        infos.append(SimpleNamespace(uuid=uuid))
    wt.tensor_manager.ready.setdefault(request_id, []).append(
        SimpleNamespace(name=f"{modality}_output", tensor_info=infos),
    )


def _run_until(wt, n: int, timeout_s: float = 30.0) -> list:
    chunks, deadline = [], threading.Event()
    timer = threading.Timer(timeout_s, deadline.set)
    timer.start()
    try:
        while len(chunks) < n and not deadline.is_set():
            wt._process_read_tensors()
            wt._release_encoded()
            while not wt.out_queue.empty():
                chunks.append(wt.out_queue.get_nowait())
    finally:
        timer.cancel()
    return chunks


def _request(wt, rid: str, outputs: list[str]) -> RequestOutputState:
    wt.request_model_kwargs[rid] = {}
    state = wt.request_output_state[rid] = RequestOutputState(mux_audio=wt._muxes_audio(outputs, {}))
    return state


def test_a_loops_pieces_leave_as_one_mp4_in_the_last_ones_slot(worker):
    clip = _clip()
    _request(worker, "r", ["video"])
    _output(worker, "r", VIDEO_PART, [torch.from_numpy(p) for p in np.array_split(clip, 3)], loop="decode")
    chunks = _run_until(worker, 3)
    assert [c.modality for c in chunks] == [VIDEO_PART, VIDEO_PART, "video"]
    assert chunks[-1].data == encode_mp4(clip, 24, options=ULTRAFAST)
    assert "audio_muxed" not in chunks[-1].metadata
    # read in place, each transport buffer goes back once its piece is written
    assert sorted(worker.tensor_manager.dereferenced) == ["r/0", "r/1", "r/2"]


def test_the_api_side_counts_the_pieces_and_delivers_the_mp4_alone():
    from mstar.api_server.data_worker import PreprocessWorker
    from mstar.api_server.request_types import ResultChunk

    facade = object.__new__(PreprocessWorker)
    facade.output_queue = queue.Queue()
    facade.per_request_reading_tensors = {"r": 3}
    for modality in (VIDEO_PART, VIDEO_PART, "video"):
        facade.output_queue.put(ResultChunk(request_id="r", modality=modality, data=b""))
    assert [c.modality for c in facade.get_result_chunks()] == ["video"]
    assert facade.per_request_reading_tensors["r"] == 0


@pytest.mark.parametrize("audio_first", [True, False])
def test_the_audio_goes_into_the_mp4_in_the_same_pass(worker, audio_first):
    clip, pcm = _clip(), _pcm()
    state = _request(worker, "r", ["video", "audio"])
    assert state.mux_audio
    audio = torch.from_numpy(np.frombuffer(pcm, "<i2").copy())
    if audio_first:
        _output(worker, "r", "audio", [audio])
    _output(worker, "r", VIDEO_PART, [torch.from_numpy(p) for p in np.array_split(clip, 3)], loop="decode")
    if not audio_first:
        _output(worker, "r", "audio", [audio])
    chunks = _run_until(worker, 4)
    video = next(c for c in chunks if c.modality == "video")
    assert video.metadata["audio_muxed"] is True
    assert next(c for c in chunks if c.modality == "audio").data == pcm  # still delivered, as PCM
    frames, sound = _decoded(video.data)
    assert np.array_equal(frames, _decoded(encode_mp4(clip, 24, options=ULTRAFAST))[0])
    assert sound is not None


def test_a_whole_video_with_audio_is_muxed_in_one_pass(worker):
    clip, pcm = _clip(), _pcm()
    _request(worker, "r", ["video", "audio"])
    _output(worker, "r", "video", [torch.from_numpy(clip)])
    _output(worker, "r", "audio", [torch.from_numpy(np.frombuffer(pcm, "<i2").copy())])
    chunks = _run_until(worker, 2)
    video = next(c for c in chunks if c.modality == "video")
    assert video.metadata["audio_muxed"] is True
    assert _decoded(video.data)[1] is not None


def test_a_video_alone_or_at_another_rate_is_left_to_postprocess_and_the_api_mux(worker):
    assert not worker._muxes_audio(["video"], {})
    assert not worker._muxes_audio(["video", "audio"], {"fps": 30})
    assert worker._muxes_audio(["video", "audio"], {"fps": 24})


def test_a_drained_request_cancels_its_mp4(worker):
    clip = _clip()
    state = _request(worker, "r", ["video"])
    _output(worker, "r", VIDEO_PART, [torch.from_numpy(clip[:6])], loop="decode")
    chunks = _run_until(worker, 1)
    assert [c.modality for c in chunks] == [VIDEO_PART]
    mp4 = state.mp4
    worker._begin_drain("r")
    mp4.join(timeout=30)
    assert mp4.future.cancelled()
    worker._release_encoded()
    assert worker.tensor_manager.dereferenced == ["r/0"]  # the held piece goes back


def test_teardown_stops_the_writer_before_the_buffers_go(worker):
    clip = _clip(48, 96, 128)
    state = _request(worker, "r", ["video"])
    worker.model.parts = 2
    _output(worker, "r", VIDEO_PART, [torch.from_numpy(clip)], loop="decode")
    worker._process_read_tensors()
    mp4 = state.mp4
    order = []
    worker.tensor_manager.cleanup_request = lambda rid: order.append(("cleanup", mp4._thread.is_alive()))
    worker._cleanup_request_state("r")
    assert order == [("cleanup", False)]
    assert mp4.future.cancelled() and "r" not in worker.request_output_state
