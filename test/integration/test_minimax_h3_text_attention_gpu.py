"""MiniMax-H3's text node encodes each prompt at its own length. torch's default attention for the language model's
mask-free causal GQA call on an H200 (torch 2.11) is cuDNN's, which builds a graph the first time it sees a sequence
length (~0.5 s, longer than a 1024² image's whole DiT); the node's attention has no such per-length cost.
A tiny random Qwen3 language model at Qwen3-VL-32B's head geometry (64 query heads, 8 key/value heads of 128)."""
import time

import pytest
import torch
from torch import nn

from mstar.model.components.diffusion.qwen3.encoder import Qwen3HiddenStateEncoder
from mstar.model.minimax_h3.submodules import TEXT_EMBEDS, H3TextEncoderSubmodule, TextStep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _node() -> H3TextEncoderSubmodule:
    torch.manual_seed(0)
    encoder = Qwen3HiddenStateEncoder(
        vocab_size=512, hidden_size=256, intermediate_size=512, num_heads=64, num_kv_heads=8, head_dim=128,
        rms_norm_eps=1e-6, rope_theta=5e6, tap_layers=(2,), max_positions=4096,
    )
    return H3TextEncoderSubmodule(encoder.to("cuda", torch.bfloat16), nn.Identity())


def _encode(node: H3TextEncoderSubmodule, length: int) -> tuple[torch.Tensor, float]:
    ids = torch.randint(0, 512, (length,), generator=torch.Generator().manual_seed(length))
    torch.cuda.synchronize()
    start = time.perf_counter()
    rows = node.forward(None, None, ids, TextStep(length))[TEXT_EMBEDS][0]
    torch.cuda.synchronize()
    return rows, time.perf_counter() - start


@torch.no_grad()
def test_a_prompt_of_a_length_the_process_has_not_seen_encodes_without_a_per_length_setup():
    node = _node()
    _encode(node, 16)  # the process's one-time setup (kernels loaded, allocator warm)
    for length in (37, 53, 91, 230):  # each new to the process
        _, seconds = _encode(node, length)
        assert seconds < 0.1, f"a {length}-token prompt took {seconds * 1e3:.0f} ms"


@torch.no_grad()
def test_the_nodes_rows_are_the_language_models_at_bf16_attention_accuracy():
    node = _node()
    rows, _ = _encode(node, 91)
    ids = torch.randint(0, 512, (91,), generator=torch.Generator().manual_seed(91)).cuda()
    reference = node.encoder.float()(ids[None], torch.ones_like(ids[None]), padded=False)[0]
    error = (rows.float() - reference).norm() / reference.norm()
    assert rows.shape == (91, 256) and float(error) < 2e-2
