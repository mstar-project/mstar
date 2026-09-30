"""A stream written by more than one walk is keyed by its layout, slot by slot.

A layout names every write in order, and an image's pages carry its digest and
where it sits on each: the whole SHA-256 of the file as sent, with how it was
read and preprocessed. A digest cut short, or folded into an id, is how LMCache
and SGLang served one request another image's KV.
"""

from __future__ import annotations

import hashlib
import queue
import sys
import threading

sys.path.insert(0, ".")

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.engine.resources.kv.config import KVConfig, KVSpec
from mstar.engine.resources.kv.keys import PageItem, chain, fingerprint
from mstar.model.base import PrefixStream, ProcessPromptOutput, Span, TensorAndMetadata

PAGE_SIZE = 16
TEXT = [list(range(1, 21)), list(range(100, 125))]
IMAGE_SLOTS = 30


class _StubCommunicator:
    """Keeps what the worker sent the conductor."""

    def __init__(self):
        self.sent = []

    def send(self, entity, msg):
        self.sent.append(msg)


class _StubTensorManager:
    """Takes the tensors and keeps none; the request never reaches a device."""

    def store_and_return_tensor_info(self, request_id, tensors, **kwargs):
        return {}

    def register_for_send(self, request_id, tensor_infos, **kwargs):
        pass

    def set_persist(self, uuid, persist):
        pass


class _StubModel:
    """Declares one stream that an image walk writes too, and lays it out as asked."""

    def __init__(self, layout):
        self._layout = layout

    def prefix_key_streams(self):
        return {"kv": {"main": PrefixStream(
            "text_inputs", "ids", "prefill", "decode", ("prefill_image",),
        )}}

    def get_node_resources(self):
        return [KVSpec(
            resource_key="kv", nodes={"LLM"},
            config=KVConfig(num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096),
        )]

    def preprocess_fingerprint(self):
        return "stub"

    def load_image(self, filepath, device):
        return TensorAndMetadata(torch.zeros(3, 2, 2))

    load_video = load_image

    def process_prompt(self, *args, **kwargs):
        tensors = {"text_inputs": [torch.tensor(ids) for ids in TEXT]}
        return ProcessPromptOutput(tensors, {"prefix_layout": {"kv": {"main": self._layout}}})


def _ids(length: int) -> Span:
    return Span("ids", "prefill", length, length, "text_inputs")


def _image(index: int = 0, slots: int = IMAGE_SLOTS, params=("default",), modality="image") -> Span:
    return Span("digest", "prefill_image", slots, 1, (modality, index), params)


def _interleaved(image: Span | None = None) -> list[Span]:
    """Text, image, text: the image starts on page 1 and runs into page 3."""
    return [_ids(len(TEXT[0])), image or _image(), _ids(len(TEXT[1]))]


IMAGE_PAGE = len(TEXT[0]) // PAGE_SIZE
_DEPLOYMENT = {"model": "stub", "resources": {"kv": {"page_size": PAGE_SIZE}}}


def _keys(model, paths: list[str], modality="image") -> list[bytes]:
    """Preprocess one request and return the keys the conductor was sent."""
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=queue.Queue(), profile_queue=queue.Queue(),
        cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(),
        stop_event=threading.Event(), communicator=_StubCommunicator(),
        tensor_manager=_StubTensorManager(), model=model, model_config=_DEPLOYMENT,
    )
    worker._process_input(PreprocessInput(
        request_id="r0", text="hello", file_paths={modality: paths},
        input_modalities=["text", modality], output_modalities=["text"],
        model_kwargs={},
    ))
    return worker.communicator.sent[-1].body.model_kwargs["prefix_keys"]["kv"]["main"]


def _file(tmp_path, content: bytes) -> list[str]:
    path = tmp_path / "file0"
    path.write_bytes(content)
    return [str(path)]


_LARGE = b"\x89PNG" + bytes(range(256)) * 512
# two runs of one layout over something that changes what the image's walk writes
_APART = {
    "content": ((b"\x89PNG one", _image()), (b"\x89PNG two", _image())),
    "a large file's last byte": ((_LARGE + b"\x00", _image()), (_LARGE + b"\x01", _image())),
    "preprocessing": ((b"\x89PNG same", _image()), (b"\x89PNG same", _image(params=("vllm",)))),
    "a parameter's type": ((b"\x89PNG same", _image(params=(True,))), (b"\x89PNG same", _image(params=("True",)))),
    "modality": ((b"GIF89a same", _image()), (b"GIF89a same", _image(modality="video"))),
}


@pytest.mark.parametrize(("one", "other"), _APART.values(), ids=_APART.keys())
def test_images_that_write_different_kv_key_apart_from_their_first_page(tmp_path, one, other):
    keys = [
        _keys(_StubModel(_interleaved(span)), _file(tmp_path, content), modality=span.source[0])
        for content, span in (one, other)
    ]

    differs = [page for page, (a, b) in enumerate(zip(*keys, strict=True)) if a != b]
    assert differs[:1] == [IMAGE_PAGE], (
        "two images whose walks write different KV share keys past the text "
        "before them, so one is served the other's"
    )


def test_an_image_sits_on_every_page_it_touches_with_where_it_starts(tmp_path):
    content = b"\x89PNG one"

    keys = _keys(_StubModel(_interleaved()), _file(tmp_path, content))

    # slots 0-19 text, 20-49 image, 50-74 text, at 16 a page
    digest = fingerprint(hashlib.sha256(content).digest(), "image", "stub", 1, repr("default"))
    assert keys == chain(
        [TEXT[0][:16], TEXT[0][16:], [], TEXT[1][:14], TEXT[1][14:]],
        {
            1: [PageItem(4, 0, IMAGE_SLOTS, digest)],
            2: [PageItem(0, 12, IMAGE_SLOTS, digest)],
            3: [PageItem(0, 28, IMAGE_SLOTS, digest)],
        },
    ), (
        "a page that holds part of the image names it differently than the "
        "KV manager will when it keys the prompt's partial page"
    )

