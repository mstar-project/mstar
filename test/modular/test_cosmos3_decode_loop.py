"""Cosmos3's streaming decoder runs in a loop over its window stream and keeps nothing between windows.

Its VAE decoder cache, the latents behind its next window and what it has sent so far ride the loop's loop-back
edges, so the node holds no request state. A request that doesn't stream assembles its video on an edge and gets
it once, at the last window, as the frames a streamed request gets one window at a time.
"""

from types import SimpleNamespace

import pytest
import torch
from test_streaming_consumer_loop import RUNTIMES
from test_streaming_consumer_loop_served import CASES, _Served

pytest.importorskip("diffusers")

from mstar.graph.base import Loop  # noqa: E402
from mstar.model.cosmos3 import constants as C  # noqa: E402
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model  # noqa: E402
from mstar.model.cosmos3.submodules import (  # noqa: E402
    DECODE_CACHE,
    DECODE_FRAMES,
    DECODE_SENT,
    DECODE_TAIL,
    VIDEO_DECODE_AR_LOOP,
)
from mstar.model.cosmos3.tests.test_windowed_decode import _decoder, _stream, _tiny_vae, _windows  # noqa: E402

EDGES = (DECODE_TAIL, DECODE_SENT, DECODE_FRAMES, DECODE_CACHE)


def _decode(sub, windows, **md) -> list[dict]:
    """Each window through `prepare_inputs` and `forward`, as the loop runs them, with its edges carried."""
    md = {"num_windows": len(windows), "overlap_latent_units": 2, **md}
    info = SimpleNamespace(step_metadata=md, dynamic_loop_iter_counts={})
    engine_inputs = SimpleNamespace(request_ids=["r"], per_request_info={"r": info})
    outs, edges = [], {}
    for i, window in enumerate(windows):
        info.dynamic_loop_iter_counts[VIDEO_DECODE_AR_LOOP] = i
        inputs = sub.prepare_inputs(C.VIDEO_DECODE_AR_WALK, info, {"window_latents": [window], **edges})
        outs.append(sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, **inputs.tensor_inputs, **inputs.kwargs))
        edges = {name: outs[-1][name] for name in EDGES if name in outs[-1]}
    return outs


def _decode_walk() -> Loop:
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True, enable_windowed_video=True)
    return model.get_graph_walk_graphs()[C.VIDEO_DECODE_AR_WALK]


def test_the_decoder_loops_over_its_window_stream():
    loop = _decode_walk()
    assert isinstance(loop, Loop) and loop.name == VIDEO_DECODE_AR_LOOP
    back = {edge.name for edge in loop.section.outputs if edge.next_node == loop.section.name}
    assert back == set(EDGES), "a decoder that kept its cache or tail itself would hold them per request"


@pytest.mark.parametrize("frames", [77, 70])
def test_a_whole_video_is_the_streamed_windows_joined(monkeypatch, frames):
    vae = _tiny_vae()
    windows = _windows(_stream(20))
    streamed = _decode(_decoder(monkeypatch, vae), windows, num_frames=frames, stream_video=True)
    whole = _decode(_decoder(monkeypatch, vae), windows, num_frames=frames)
    assert all("video_output" not in out for out in whole[:-1])
    joined = torch.cat([out["video_output"][0] for out in streamed if "video_output" in out])
    assert joined.shape[0] == frames
    assert torch.equal(whole[-1]["video_output"][0], joined), "the assembled video left the streamed frames"


def test_the_first_window_decodes_on_the_request_as_the_worker_opens_it(monkeypatch, tmp_path):
    """A worker counts a loop's iterations onto a request after each step of its partition, and the decoder's first
    step is its loop's first iteration, so the first window comes with no count. Read as one that must be there, it
    failed every served windowed request."""
    served = _Served(tmp_path, CASES["cosmos3"])
    served._produce()
    served.worker.ingest.poll_stream_buffers()
    info = served.worker.request_state.get_fwd_info(served.rid, served.consumer_partition)
    info.step_metadata = {"num_windows": 2, "overlap_latent_units": 2, "num_frames": 77, "stream_video": True}
    sub = _decoder(monkeypatch, _tiny_vae())
    first, second = _windows(_stream(20))[:2]
    inputs = sub.prepare_inputs(C.VIDEO_DECODE_AR_WALK, info, {"window_latents": [first]})
    assert inputs.kwargs["window"] == 0
    engine_inputs = SimpleNamespace(request_ids=[served.rid], per_request_info={served.rid: info})
    out = sub.forward(C.VIDEO_DECODE_AR_WALK, engine_inputs, **inputs.tensor_inputs, **inputs.kwargs)
    assert out["video_output"][0].shape[0] > 0, "the first window would send no frame"
    served._step(served.reader)
    served.rt.update_dynamic_loop_iters({served.rid: info}, served.consumer_partition)
    edges = {name: out[name] for name in EDGES if name in out}
    inputs = sub.prepare_inputs(C.VIDEO_DECODE_AR_WALK, info, {"window_latents": [second], **edges})
    assert inputs.kwargs["window"] == 1, "the second window would decode as the stream's first"


@RUNTIMES
def test_a_stream_that_brings_no_window_ends_the_decode_loop(runtime, tmp_path):
    """A text request on a windowed deployment runs the decoder partition's walk on the stream's empty flush alone."""
    loop = _decode_walk()
    node = loop.section.name
    loop.section._register_streaming({"window_latents"})
    graph = runtime(loop, {node}, tmp_path)
    for name in EDGES:
        graph.ingest(name, node)  # the conductor's empty edge for each input that isn't streamed
    graph.ingest("window_latents", node)
    assert node in graph.ready()
    graph.stop(loop.name, node)
    graph.complete(node, {**{name: [] for name in EDGES}, "video_output": []})
    assert graph.done(), "a text request would hang in the decoder partition"
