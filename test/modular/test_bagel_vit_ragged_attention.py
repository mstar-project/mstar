"""BAGEL ViT on engine-owned ragged attention — the integration test for the
whole stack.

Exercises the real submodule and the real piecewise runner: the ViT declares a
``RaggedAttentionConfig``, the runner captures its block loop with a per-bucket
FlashInfer state, and replay re-plans that state for the actual token layout.

The ViT keeps two attention backends on purpose, so the load-bearing assertion
is that they agree: eager uses flash-attn (no head-dim padding), the captured
path uses FlashInfer (which must pad SigLIP2's head_dim=72 up to 128).
"""

import pytest
import torch
import torch.nn.functional as F
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
PATCH = 16
# hidden 288 / 4 heads = head_dim 72 — the SigLIP2 shape FlashInfer has no
# kernel for, so every captured case here goes through the pad-to-128 path.
HIDDEN, HEADS = 288, 4
HEAD_DIM = HIDDEN // HEADS
TOKEN_BUCKETS = "256,512"
TOL = 6e-2


def _config() -> BagelViTConfig:
    return BagelViTConfig(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=3,        # __post_init__ decrements to 2
        num_attention_heads=HEADS,
        patch_size=PATCH,
        image_size=224,
    )


def _submodule(monkeypatch, cuda_graph: bool) -> ViTEncoderSubmodule:
    # Read in ViTEncoderSubmodule.__init__, so it must be set before construction.
    monkeypatch.setenv("MSTAR_VIT_CUDA_GRAPH", "1" if cuda_graph else "0")
    monkeypatch.setenv("MSTAR_VIT_CG_TOKEN_BUCKETS", TOKEN_BUCKETS)
    monkeypatch.setenv("MSTAR_VIT_CG_BATCH_SIZES", "1,2")

    config = _config()
    torch.manual_seed(0)
    vit_model = BagelVisionModel(config)
    vit_model.vision_model.embeddings.convert_conv2d_to_linear(config)
    vit_model = vit_model.to(device=DEVICE, dtype=DTYPE).eval()

    return ViTEncoderSubmodule(
        vit_model=vit_model,
        connector=nn.Identity(),        # unused by _encode
        vit_pos_embed=nn.Identity(),    # unused by _encode
        vit_patch_size=PATCH,
        vit_max_num_patch_per_side=14,
    )


def _inputs(seg_lens: list[int], seed: int = 0):
    torch.manual_seed(seed)
    total = sum(seg_lens)
    num_positions = (224 // PATCH) ** 2
    pixels = torch.randn(total, 3 * PATCH * PATCH, dtype=DTYPE, device=DEVICE)
    position_ids = torch.randint(0, num_positions, (total,), device=DEVICE)
    cu = [0]
    for seg_len in seg_lens:
        cu.append(cu[-1] + seg_len)
    return pixels, position_ids, torch.tensor(cu, dtype=torch.int32, device=DEVICE)


def _engine_inputs(request_ids, runners=None) -> ModelInputsFromEngine:
    return ModelInputsFromEngine(
        request_ids=request_ids,
        per_request_info={
            rid: CurrentForwardPassInfo(
                request_id=rid, graph_walk="prefill_vit", requires_cfg=False,
                fwd_index=0, random_seed=0, max_tokens=1, sampling_config={},
            )
            for rid in request_ids
        },
        piecewise_runners=runners or {},
    )


def _encode(submodule, engine_inputs, pixels, position_ids, cu_seqlens, seq_lens):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        return submodule._encode(
            engine_inputs=engine_inputs,
            packed_pixel_values=pixels,
            packed_position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(seq_lens),
            seq_lens=seq_lens,
        )


def _reference(submodule, pixels, position_ids, cu_seqlens):
    """Block loop with per-segment SDPA in fp32 — independent of both backends."""
    vision = submodule.vit_model.vision_model
    with torch.no_grad():
        hidden, _ = vision.embed(pixels, position_ids)
        cu = cu_seqlens.tolist()
        for layer in vision.encoder.layers:
            attn = layer.self_attn
            residual = hidden
            x = layer.layer_norm1(hidden)
            n = x.shape[0]
            q = attn.q_proj(x).view(n, HEADS, HEAD_DIM)
            k = attn.k_proj(x).view(n, HEADS, HEAD_DIM)
            v = attn.v_proj(x).view(n, HEADS, HEAD_DIM)
            out = torch.zeros_like(q)
            for start, end in zip(cu[:-1], cu[1:], strict=True):
                if end <= start:
                    continue
                qs, ks, vs = (
                    t[start:end].transpose(0, 1).unsqueeze(0).float() for t in (q, k, v)
                )
                o = F.scaled_dot_product_attention(qs, ks, vs, scale=HEAD_DIM ** -0.5)
                out[start:end] = o.squeeze(0).transpose(0, 1).to(out.dtype)
            hidden = residual + attn.out_proj(out.reshape(n, -1))
            residual = hidden
            hidden = layer.mlp(layer.layer_norm2(hidden))
            hidden = residual + hidden
        return vision.post_layernorm(hidden)


def _assert_close(got, want, tol=TOL):
    err = (got.float() - want.float()).abs().max().item()
    assert err < tol, f"max_abs_err={err} exceeds {tol}"


# --------------------------------------------------------------------------- #
# opt-in gating
# --------------------------------------------------------------------------- #


def test_ragged_attention_off_by_default(monkeypatch):
    submodule = _submodule(monkeypatch, cuda_graph=False)
    assert submodule.get_ragged_attention_config() is None
    assert submodule.get_piecewise_cuda_graph_configs(DEVICE, DTYPE) == {}


def test_ragged_config_reports_vit_attention_shape(monkeypatch):
    submodule = _submodule(monkeypatch, cuda_graph=True)
    config = submodule.get_ragged_attention_config()
    assert config.num_qo_heads == HEADS
    assert config.head_dim == HEAD_DIM      # true 72, padded inside the wrapper
    assert config.causal is False
    assert config.max_segments_per_request == 1


def test_vit_declares_cuda_graph_only(monkeypatch):
    """Eager stays on flash-attn, so no eager state (and no unread workspace)."""
    submodule = _submodule(monkeypatch, cuda_graph=True)
    config = submodule.get_ragged_attention_config()
    assert config.enabled_for is AttentionMode.CUDA_GRAPH
    assert config.cuda_graph_enabled and not config.eager_enabled


def test_ragged_config_shards_heads_for_tp(monkeypatch):
    submodule = _submodule(monkeypatch, cuda_graph=True)
    config = submodule.get_ragged_attention_config(tp_world_size=2)
    assert config.num_qo_heads == HEADS // 2
    assert config.head_dim == HEAD_DIM      # head_dim never shards


# --------------------------------------------------------------------------- #
# captured vs eager
# --------------------------------------------------------------------------- #


def _runners(submodule):
    runners = build_piecewise_runners(submodule, DEVICE, DTYPE)
    assert "vit_block_loop" in runners, "block loop failed to capture"
    return runners


@pytest.mark.parametrize(
    ("seg_lens", "seq_lens"),
    [
        ([256], [256]),            # exactly a bucket
        ([200], [200]),            # under a bucket → padded
        ([256, 256], [256, 256]),  # two images, two requests
        ([128, 100], [128, 100]),  # ragged, both under
    ],
    ids=["exact_bucket", "under_bucket", "two_images", "ragged"],
)
def test_captured_matches_eager_and_reference(monkeypatch, seg_lens, seq_lens):
    submodule = _submodule(monkeypatch, cuda_graph=True)
    runners = _runners(submodule)
    pixels, position_ids, cu = _inputs(seg_lens)

    captured = _encode(
        submodule, _engine_inputs([f"r{i}" for i in range(len(seq_lens))], runners),
        pixels, position_ids, cu, seq_lens,
    )
    eager = _encode(
        submodule, _engine_inputs([f"r{i}" for i in range(len(seq_lens))]),
        pixels, position_ids, cu, seq_lens,
    )
    reference = _reference(submodule, pixels, position_ids, cu)

    _assert_close(eager, reference)      # flash-attn path
    _assert_close(captured, reference)   # FlashInfer path, head_dim padded
    _assert_close(captured, eager)


def test_replay_is_stable_across_changing_layouts(monkeypatch):
    """One captured bucket serving a sequence of different real layouts."""
    submodule = _submodule(monkeypatch, cuda_graph=True)
    runners = _runners(submodule)

    for i, seg_lens in enumerate([[256], [200], [128], [256], [64]]):
        pixels, position_ids, cu = _inputs(seg_lens, seed=i)
        captured = _encode(
            submodule, _engine_inputs(["r0"], runners),
            pixels, position_ids, cu, seg_lens,
        )
        _assert_close(captured, _reference(submodule, pixels, position_ids, cu))


def test_falls_back_to_eager_when_no_bucket_fits(monkeypatch):
    """Above the largest bucket the runner can't serve it; eager must still work."""
    submodule = _submodule(monkeypatch, cuda_graph=True)
    runners = _runners(submodule)
    seg_lens = [1024]                       # > max bucket (512)
    pixels, position_ids, cu = _inputs(seg_lens)

    assert not runners["vit_block_loop"].can_run(1, 1024)
    out = _encode(
        submodule, _engine_inputs(["r0"], runners), pixels, position_ids, cu, seg_lens
    )
    _assert_close(out, _reference(submodule, pixels, position_ids, cu))
