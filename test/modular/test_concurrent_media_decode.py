"""A request's images may decode at once: order is the request's, and the
request's media budget still holds across files loading together."""
import threading
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.model.base import TensorAndMetadata
from mstar.model.media_limits import RequestMediaBudget, _charge_request, read_image
from mstar.utils.threads import ThreadPool


class _Model:
    """Loaders that only finish once all of ``expect`` are loading, so a
    sequential loop would time out."""

    def __init__(self, expect: int):
        self.barrier = threading.Barrier(expect, timeout=10) if expect > 1 else None

    def max_input_files(self):
        return {}

    def load_image(self, path, device):
        if self.barrier is not None:
            self.barrier.wait()
        return TensorAndMetadata(torch.tensor([int(path)]), {"path": path})


def _worker(model, pool):
    worker = object.__new__(PreprocessWorkerThread)
    worker.model, worker.device, worker._decode_pool = model, "cpu", pool
    return worker


def test_images_load_at_once_and_stay_in_request_order():
    with ThreadPool(max_workers=3) as pool:
        tensors, metadata = _worker(_Model(3), pool)._load_media({"image": ["3", "1", "2"]})
    assert [t.item() for t in tensors["image_inputs"]] == [3, 1, 2]
    assert [m["path"] for m in metadata["image_inputs"]] == ["3", "1", "2"]


def test_a_failing_image_fails_the_request():
    class _Fails(_Model):
        def load_image(self, path, device):
            super().load_image(path, device)
            raise ValueError("could not decode image")

    with ThreadPool(max_workers=2) as pool, pytest.raises(ValueError, match="could not decode"):
        _worker(_Fails(2), pool)._load_media({"image": ["1", "2"]})


def test_budget_counts_files_loading_together():
    budget = RequestMediaBudget(2)
    budget.limits["image"] = 100
    both_charged = threading.Barrier(2, timeout=10)
    errors = []

    def load(amount):
        with budget.file():
            try:
                _charge_request("image", amount)
                both_charged.wait()
                _charge_request("image", amount)  # re-checks against the other file's charge
                both_charged.wait()  # neither file finishes before both re-checked
            except ValueError as exc:
                errors.append(exc)
                both_charged.abort()
            except threading.BrokenBarrierError:
                pass  # the other file was the one refused

    threads = [threading.Thread(target=load, args=(60,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors and "together" in str(errors[0])


def test_budget_charges_a_file_its_largest_check_once():
    budget = RequestMediaBudget(1)
    budget.limits["image"] = 100
    with budget.file():
        _charge_request("image", 60)
        _charge_request("image", 80)
    assert budget.used["image"] == 80 and budget._loading == []


class _ReadsFiles(_Model):
    def load_image(self, path, device):
        self.barrier.wait()
        return TensorAndMetadata(read_image(path, rgb=True))


def _png(path, side=10):
    Image.new("RGB", (side, side), (1, 2, 3)).save(path)
    return str(path)


def test_request_pixel_cap_holds_on_the_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", "250")
    paths = [_png(tmp_path / f"{i}.png") for i in range(3)]
    with ThreadPool(max_workers=3) as pool, pytest.raises(ValueError, match="together"):
        _worker(_ReadsFiles(3), pool)._load_media({"image": paths})


def test_request_under_the_pixel_cap_loads_on_the_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_IMAGE_PIXELS", "300")
    paths = [_png(tmp_path / f"{i}.png") for i in range(3)]
    with ThreadPool(max_workers=3) as pool:
        tensors, _ = _worker(_ReadsFiles(3), pool)._load_media({"image": paths})
    assert [t.shape for t in tensors["image_inputs"]] == [torch.Size([3, 10, 10])] * 3


@pytest.mark.parametrize(
    ("declared", "prepare_workers", "decode_workers"),
    [(True, 5, 5), (True, 1, 1), (False, 5, 1)],
)
def test_decode_pool_needs_the_declaration_and_a_prepare_pool(declared, prepare_workers, decode_workers):
    worker = object.__new__(PreprocessWorkerThread)
    worker.model = SimpleNamespace(PREPROCESS_PARALLEL_IMAGES=declared)
    worker._prepare_workers = prepare_workers
    assert worker._num_decode_workers() == decode_workers
