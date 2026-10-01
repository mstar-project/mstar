"""SamplerBuffers.gather_static skips a batch its cg slot already gathered."""
import pytest
import torch

from mstar.engine.resources.sampler.utils import SamplerBuffers, SamplingConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bufs(cg_slots=2):
    return SamplerBuffers.allocate(max_batch_size=4, device=torch.device("cuda"), cg_slots=cg_slots)


def _count_copies(bufs, monkeypatch):
    issued = []
    for buf in bufs._scalar_buffers():
        real = buf.stager.copy_
        monkeypatch.setattr(
            buf.stager, "copy_",
            lambda *a, _real=real, **k: (issued.append(1), _real(*a, **k))[1],
        )
    return issued


def _temps(bufs, cg_slot, bs):
    torch.cuda.synchronize()
    return bufs.temperature.slot_view(cg_slot, bs).tolist()


def test_steady_batch_is_gathered_once_per_cg_slot(monkeypatch):
    bufs = _bufs()
    for rid, t in (("a", 0.5), ("b", 0.7)):
        bufs.register_request(rid, SamplingConfig(temperature=t))
    issued = _count_copies(bufs, monkeypatch)
    n = len(bufs._scalar_buffers())
    for _ in range(3):
        for cg_slot in (0, 1):
            bufs.gather_static(["a", "b"], 4, cg_slot)
    assert len(issued) == 2 * n  # once per cg slot, then skipped
    assert _temps(bufs, 1, 2) == pytest.approx([0.5, 0.7])


def test_config_change_regathers(monkeypatch):
    bufs = _bufs()
    bufs.register_request("a", SamplingConfig(temperature=0.5))
    bufs.gather_static(["a"], 2, 0)
    bufs.update_request_config("a", SamplingConfig(temperature=0.9))
    bufs.gather_static(["a"], 2, 0)
    assert _temps(bufs, 0, 1) == pytest.approx([0.9])


def test_new_batch_or_padding_regathers():
    bufs = _bufs()
    for rid, t in (("a", 0.5), ("b", 0.7), ("c", 0.3)):
        bufs.register_request(rid, SamplingConfig(temperature=t))
    bufs.gather_static(["a", "b"], 2, 0)
    bufs.gather_static(["c", "b"], 2, 0)
    assert _temps(bufs, 0, 2) == pytest.approx([0.3, 0.7])
    bufs.gather_static(["c", "b"], 4, 0)   # same rids, more padding rows
    assert _temps(bufs, 0, 4)[:2] == pytest.approx([0.3, 0.7])


def test_reregistered_request_id_regathers():
    """Same request id, freed and registered again with another config: the
    batch tuple is unchanged, but its slot awaits init."""
    bufs = _bufs()
    bufs.register_request("a", SamplingConfig(temperature=0.5))
    bufs.gather_static(["a"], 1, 0)
    bufs.unregister_request("a")
    bufs.register_request("a", SamplingConfig(temperature=0.2))
    bufs.gather_static(["a"], 1, 0)
    assert _temps(bufs, 0, 1) == pytest.approx([0.2])
