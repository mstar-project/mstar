"""NodeSubmodule.get_device is cached (prepare_inputs calls it per request per step) and follows the module when
it is moved."""
import torch

from mstar.model.submodule_base import NodeSubmodule


def _submodule():
    cls = type("Sub", (NodeSubmodule,), {})
    cls.__abstractmethods__ = frozenset()
    sub = cls()
    sub.lin = torch.nn.Linear(2, 2)
    return sub


def test_get_device_follows_to_empty_and_to():
    with torch.device("meta"):
        sub = _submodule()
    assert sub.get_device().type == "meta"
    sub.to_empty(device="cpu")
    assert sub.get_device().type == "cpu"
    if torch.cuda.is_available():
        sub.cuda()
        assert sub.get_device().type == "cuda"
        sub.cpu()
        assert sub.get_device().type == "cpu"


def test_get_device_does_not_walk_the_parameters_again():
    sub = _submodule()
    sub.get_device()
    calls = []
    orig = sub.parameters
    sub.parameters = lambda *a, **k: calls.append(1) or orig(*a, **k)
    for _ in range(3):
        assert sub.get_device().type == "cpu"
    assert calls == []


def test_a_compiled_forward_reading_the_device_does_not_recompile():
    from torch._dynamo.utils import counters

    with torch.device("meta"):
        sub = _submodule()
    sub.to_empty(device="cpu")
    torch._dynamo.reset()
    counters.clear()
    forward = torch.compile(lambda x: sub.lin(x) + torch.ones(2, device=sub.get_device()), backend="eager")
    for _ in range(2):
        forward(torch.randn(3, 2))
    assert counters["stats"]["unique_graphs"] == 1
