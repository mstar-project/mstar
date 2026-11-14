"""Complete CPU action-expert parity against the Praxis-1 reference."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

pytest.importorskip("praxis", reason="the Praxis-1 reference is not on PYTHONPATH")

from praxis import action_expert as ref_ae  # noqa: E402

from mstar.model.praxis1.components.action_expert import ActionExpert, trunk_query_rope_ids  # noqa: E402
from mstar.model.praxis1.components.encoder import TapFeatures  # noqa: E402
from mstar.model.praxis1.components.expert_inputs import assemble_expert_inputs  # noqa: E402

DIM, HEAD_DIM, TRUNK_DIM = 128, 64, 24
ROPE = {
    "kind": "trunk",
    "dim": HEAD_DIM,
    "effective_dim": HEAD_DIM,
    "base": 26_000_000,
    "table_size": 1_000_000,
    "order": "before_norm",
}


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _models(dtype: torch.dtype, *, clean_only: bool = False):
    opts = dict(
        action_patch_dim=5,
        proprio_patch_dim=3,
        trunk_dim=TRUNK_DIM,
        dim=DIM,
        num_layers=4,
        num_heads=2,
        head_dim=HEAD_DIM,
        num_future_per_chunk=2,
        max_chunks=4,
        max_action_tokens=8,
        rope=ROPE,
        head_plus_zero=True,
        layout_stamp=20260924,
        clean_only_conditioning=clean_only,
    )
    torch.manual_seed(71)
    ref = ref_ae.ActionExpert(**opts, attention="sdpa")
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.2)
    port = ActionExpert(**opts)
    port.load_state_dict(ref.state_dict(), strict=True)
    return ref.to(dtype).eval(), port.to(dtype).eval()


def _inputs(k: int, dtype: torch.dtype, *, clean_only: bool = False):
    torch.manual_seed(83 + k)
    chunk, tpf, n_global = 2, 3, 5
    n_noised = chunk * k
    n_clean = chunk * (k - 1)
    seq = n_global + (n_noised + n_clean) * tpf
    embeddings = torch.randn(1, seq, TRUNK_DIM, dtype=dtype)
    taps = [torch.randn_like(embeddings) for _ in range(2)]
    ids = torch.arange(seq, dtype=torch.long)[None] + 1_100_000
    features = TapFeatures(
        taps=taps,
        embeddings=embeddings,
        frames_per_chunk=chunk,
        tokens_per_frame=tpf,
        rope_ids=ids,
        global_spans={"register": (0, 2), "anchor": (2, 1), "text": (3, 2)},
        noised_span=(n_global, n_noised * tpf),
        clean_span=(n_global + n_noised * tpf, n_clean * tpf) if n_clean else None,
        n_chunks=k,
    )
    actions = torch.randn(1, n_noised, 5, dtype=dtype)
    times = torch.rand(1, k, dtype=dtype)
    proprio = torch.randn(1, k, 3, dtype=dtype)
    assembled = asdict(assemble_expert_inputs(features, actions, times, proprio))
    masks = ref_ae.build_trunk_rope_masks(
        num_action_tokens=n_noised,
        num_future_per_chunk=2,
        chunk_size=chunk,
        num_clean_frames=n_clean,
        num_noised_frames=n_noised,
        tokens_per_frame=tpf,
        window_size=n_clean + n_noised,
        num_global_tokens=n_global,
        device=torch.device("cpu"),
        include_noised=not clean_only,
        block=False,
    )
    return assembled, masks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_full_expert_matches_reference(dtype: torch.dtype, k: int):
    ref, port = _models(dtype)
    args, (cross, self_) = _inputs(k, dtype)
    with torch.no_grad():
        expected = ref.forward_trunk_rope(**args)
        got = port.forward_trunk_rope(**args, cross_mask=cross, self_mask=self_)
    assert torch.equal(got, expected)


def test_query_positions_match_reference_above_table_size():
    args, _ = _inputs(2, torch.float32)
    expected = ref_ae.trunk_query_rope_ids(
        args["kv_rope_ids"],
        num_global_tokens=5,
        num_clean_frames=args["num_clean_frames"],
        num_noised_frames=args["num_noised_frames"],
        tokens_per_frame=args["tokens_per_frame"],
        num_action_tokens=args["num_noised_frames"],
        num_future_per_chunk=2,
        chunk_size=2,
    )
    got = trunk_query_rope_ids(
        args["kv_rope_ids"],
        num_global_tokens=5,
        num_clean_frames=args["num_clean_frames"],
        num_noised_frames=args["num_noised_frames"],
        tokens_per_frame=args["tokens_per_frame"],
        num_future_per_chunk=2,
        chunk_size=2,
    )
    assert torch.equal(got, expected)
    assert int(got.max()) > ROPE["table_size"]


def test_tap_order_matters():
    _, port = _models(torch.float32)
    args, (cross, self_) = _inputs(2, torch.float32)
    with torch.no_grad():
        baseline = port.forward_trunk_rope(**args, cross_mask=cross, self_mask=self_)
        swapped = port.forward_trunk_rope(
            **{**args, "trunk_features": list(reversed(args["trunk_features"]))},
            cross_mask=cross,
            self_mask=self_,
        )
    assert not torch.equal(baseline, swapped)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clean_only_conditioning_matches_reference(dtype: torch.dtype):
    ref, port = _models(dtype, clean_only=True)
    args, (cross, self_) = _inputs(2, dtype, clean_only=True)
    with torch.no_grad():
        expected = ref.forward_trunk_rope(**args)
        got = port.forward_trunk_rope(**args, cross_mask=cross, self_mask=self_)
    assert torch.equal(got, expected)


def test_missing_attention_mask_is_rejected():
    _, port = _models(torch.float32)
    args, (_, self_) = _inputs(2, torch.float32)
    with pytest.raises(ValueError, match="masks are required"):
        port.forward_trunk_rope(**args, cross_mask=None, self_mask=self_)
