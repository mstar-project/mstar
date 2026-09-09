import torch

from mstar.engine.resources.sampler.utils import _xpu_generator_seed_offset


class _FakeGenerator:
    def __init__(self, seed: int, offset: int):
        self.state = torch.tensor([seed, offset], dtype=torch.int64)

    def get_state(self) -> torch.Tensor:
        return self.state.clone()

    def set_state(self, state: torch.Tensor) -> None:
        self.state.copy_(state)


def test_xpu_generator_reserves_aligned_offset_range():
    generator = _FakeGenerator(seed=123, offset=8)

    assert _xpu_generator_seed_offset(generator, 10) == (123, 8)
    assert generator.state.tolist() == [123, 20]
    assert _xpu_generator_seed_offset(generator, 10) == (123, 20)
    assert generator.state.tolist() == [123, 32]


def test_xpu_generator_seed_read_does_not_advance_offset():
    generator = _FakeGenerator(seed=123, offset=8)

    assert _xpu_generator_seed_offset(generator, 0) == (123, 8)
    assert generator.state.tolist() == [123, 8]
