"""Static-buffer interning follows a config's declared token axis instead of guessing it."""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig
from mstar.engine.cuda_graph_runner import CudaGraphRunner
from mstar.model.submodule_base import NodeInputs


class _Runner:
    """Just the interning state of a CudaGraphRunner, with the real methods bound."""

    _intern_static_buffer = CudaGraphRunner._intern_static_buffer
    _seq_dim = staticmethod(CudaGraphRunner._seq_dim)

    def __init__(self, *configs):
        self._capture_configs = list(configs)
        self._shared_static_buffers = {}
        self._static_buffer_seq_dims = {}
        self._capture_clone_bytes_naive = 0


def _config(**kwargs):
    single = NodeInputs(tensor_inputs={}, input_seq_len=7680)
    return BatchedCudaGraphConfig(capture_graph_walk="w", single_request_inputs=single, **kwargs)


def test_the_guess_takes_a_hidden_size_equal_to_the_token_count_for_the_token_axis():
    runner = _Runner(SimpleNamespace(static_seq_dims={}))
    text = torch.zeros(512, 7680)  # [text tokens, hidden]; the bucket has 7680 tokens in total
    runner._intern_static_buffer(0, "text", text, seq_len=7680)
    assert runner._static_buffer_seq_dims[(0, "text")] == 1  # the collision the declaration exists for


def test_a_declared_token_axis_wins_over_the_guess():
    runner = _Runner(_config(static_seq_dims={"text": 0}))
    text = torch.arange(512 * 7680, dtype=torch.float32).view(512, 7680)
    view = runner._intern_static_buffer(0, "text", text, seq_len=7680)
    assert runner._static_buffer_seq_dims[(0, "text")] == 0
    assert view.shape == (512, 7680) and torch.equal(view, text)
    assert runner._shared_static_buffers[(0, "text")].shape == (512, 7680)
    # a smaller bucket of the same config reslices the leading dim of the same buffer
    smaller = runner._intern_static_buffer(0, "text", text[:256], seq_len=3840)
    assert smaller.shape == (256, 7680) and smaller.data_ptr() == view.data_ptr()


def test_undeclared_keys_keep_the_guess_for_ids_with_a_trailing_token_axis():
    runner = _Runner(_config(static_seq_dims={"text": 0}))
    ids = torch.zeros(3, 40, dtype=torch.int64)  # mrope-style [3, tokens]
    view = runner._intern_static_buffer(0, "ids", ids, seq_len=40)
    assert runner._static_buffer_seq_dims[(0, "ids")] == 1 and view.shape == (3, 40)


def test_configs_default_to_no_declaration():
    assert _config().static_seq_dims == {}
    assert _config(static_seq_dims={"a": 1}).static_seq_dims == {"a": 1}
