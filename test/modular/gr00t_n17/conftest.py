import pytest
import torch


@pytest.fixture(autouse=True)
def _pinned_threads():
    """The CPU parity tests compare with ``torch.equal``, and reduction order follows the thread count."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
