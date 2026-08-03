"""BAGEL ViT on engine-owned ragged attention — the integration test for the stack.

Drives the real submodule through the real ``PiecewiseCudaGraphRunner``: the ViT
declares a ``RaggedAttentionConfig``, the runner captures its block loop with a
per-bucket FlashInfer state, and replay re-plans that state for the actual token
layout.

The ViT keeps two attention backends on purpose, so the load-bearing assertion is
that they agree: eager uses flash-attn (no head-dim padding), captured uses
FlashInfer (which must pad SigLIP2's head_dim=72 to 128). Absolute numerics for
the FlashInfer path are pinned in test_ragged_attention.py.
"""

import pytest
import torch
from torch import nn

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.attention_state import AttentionMode
from mstar.engine.cuda_graph_runner import build_piecewise_runners
from mstar.model.bagel.components.vit_encoder import BagelVisionModel
from mstar.model.bagel.config import BagelViTConfig
from mstar.model.bagel.submodules import ViTEncoderSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ViT graph capture requires CUDA"
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
PATCH, HEADS, HIDDEN = 16, 4, 288       # 288/4 = head_dim 72, the SigLIP2 shape
HEAD_DIM = HIDDEN // HEADS
TOKEN_BUCKETS = "256,512"
TOL = 6e-2


def submodule(monkeypatch, cuda_graph: bool = True) -> ViTEncoderSubmodule:
    # Read in ViTEncoderSubmodule.__init__, so set before construction. Capture
    # is on by default; the tests pin both settings explicitly regardless.
    monkeypatch.setenv("MSTAR_VIT_CUDA_GRAPH", "1" if cuda_graph else "0")
    monkeypatch.setenv("MSTAR_VIT_CG_TOKEN_BUCKETS", TOKEN_BUCKETS)
    monkeypatch.setenv("MSTAR_VIT_CG_BATCH_SIZES", "1,2")

    config = BagelViTConfig(
        hidden_size=HIDDEN, intermediate_size=2 * HIDDEN,
        num_hidden_layers=3,            # __post_init__ decrements to 2
        num_attention_heads=HEADS, patch_size=PATCH, image_size=224,
    )
    torch.manual_seed(0)
    vit = BagelVisionModel(config)
    vit.vision_model.embeddings.convert_conv2d_to_linear(config)

    return ViTEncoderSubmodule(
        vit_model=vit.to(device=DEVICE, dtype=DTYPE).eval(),
        connector=nn.Identity(),        # unused by _encode
        vit_pos_embed=nn.Identity(),    # unused by _encode
        vit_patch_size=PATCH,
        vit_max_num_patch_per_side=14,
    )


def inputs(seg_lens: list[int], seed: int = 0):
    torch.manual_seed(seed)
    total = sum(seg_lens)
    cu = [0]
    for n in seg_lens:
        cu.append(cu[-1] + n)
    return (
        torch.randn(total, 3 * PATCH * PATCH, dtype=DTYPE, device=DEVICE),
        torch.randint(0, (224 // PATCH) ** 2, (total,), device=DEVICE),
        torch.tensor(cu, dtype=torch.int32, device=DEVICE),
    )


def encode(sub, pixels, position_ids, cu_seqlens, seq_lens, runners=None):
    engine_inputs = ModelInputsFromEngine(
        request_ids=[f"r{i}" for i in range(len(seq_lens))],
        per_request_info={
            f"r{i}": CurrentForwardPassInfo(
                request_id=f"r{i}", graph_walk="prefill_vit", requires_cfg=False,
                fwd_index=0, random_seed=0, max_tokens=1, sampling_config={},
            )
            for i in range(len(seq_lens))
        },
        piecewise_runners=runners or {},
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        return sub._encode(
            engine_inputs=engine_inputs, packed_pixel_values=pixels,
            packed_position_ids=position_ids, cu_seqlens=cu_seqlens,
            max_seqlen=max(seq_lens), seq_lens=seq_lens,
        )


def close(got, want):
    err = (got.float() - want.float()).abs().max().item()
    assert err < TOL, f"max_abs_err={err} exceeds {TOL}"


def runners_for(sub):
    runners = build_piecewise_runners(sub, DEVICE, DTYPE)
    assert "vit_block_loop" in runners, "block loop failed to capture"
    return runners


def test_env_var_disables_ragged_attention(monkeypatch):
    """MSTAR_VIT_CUDA_GRAPH=0 is the escape hatch back to pure eager."""
    sub = submodule(monkeypatch, cuda_graph=False)
    assert sub.get_ragged_attention_config() is None
    assert sub.get_piecewise_cuda_graph_configs(DEVICE, DTYPE) == {}


def test_ragged_config_reports_vit_attention_shape(monkeypatch):
    config = submodule(monkeypatch).get_ragged_attention_config()
    assert config.num_qo_heads == HEADS
    assert config.head_dim == HEAD_DIM          # true 72, padded in the wrapper
    assert config.causal is False
    assert config.max_segments_per_request == 1
    # Eager stays on flash-attn, so no eager state (and no unread workspace).
    assert config.enabled_for is AttentionMode.CUDA_GRAPH


def test_ragged_config_shards_heads_for_tp(monkeypatch):
    config = submodule(monkeypatch).get_ragged_attention_config(2)
    assert config.num_qo_heads == HEADS // 2
    assert config.head_dim == HEAD_DIM          # head_dim never shards


@pytest.mark.parametrize(
    "seg_lens",
    [[256], [200], [128, 100]],
    ids=["exact_bucket", "under_bucket", "two_images"],
)
def test_captured_matches_eager(monkeypatch, seg_lens):
    sub = submodule(monkeypatch)
    runners = runners_for(sub)
    pixels, position_ids, cu = inputs(seg_lens)
    close(
        encode(sub, pixels, position_ids, cu, seg_lens, runners),
        encode(sub, pixels, position_ids, cu, seg_lens),
    )


def test_replay_stable_across_changing_layouts(monkeypatch):
    """One captured bucket serving a sequence of different real layouts."""
    sub = submodule(monkeypatch)
    runners = runners_for(sub)
    for i, seg_lens in enumerate([[256], [128], [256]]):
        pixels, position_ids, cu = inputs(seg_lens, seed=i)
        close(
            encode(sub, pixels, position_ids, cu, seg_lens, runners),
            encode(sub, pixels, position_ids, cu, seg_lens),
        )


def test_falls_back_to_eager_when_no_bucket_fits(monkeypatch):
    """Above the largest bucket the runner can't serve it; eager must still run."""
    sub = submodule(monkeypatch)
    runners = runners_for(sub)
    seg_lens = [1024]                           # > max bucket (512)
    assert not runners["vit_block_loop"].can_run(1, 1024)
    pixels, position_ids, cu = inputs(seg_lens)
    close(
        encode(sub, pixels, position_ids, cu, seg_lens, runners),
        encode(sub, pixels, position_ids, cu, seg_lens),
    )
