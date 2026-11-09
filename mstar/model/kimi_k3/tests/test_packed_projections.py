"""Kimi K3's projections that read one input, packed into one GEMM after load: the latent MoE's router,
latent down projection and shared experts' gate_up, and the MLA's q_a | kv_a with the output gate. On the
tiny model through the real loader: the parameters view the packed weight, and the packed forward computes
what the separate GEMMs do (fp32 on the CPU: bit for bit). With the shared experts beside the routed path
(``shared_experts_overlap``, the default) the packed GEMMs leave them out; the others pin the packing
without it."""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.kimi_k3.components.attention import KimiMLAAttention
from mstar.model.kimi_k3.components.moe import _LatentMoE
from mstar.model.kimi_k3.tests import engine_helpers as H


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_packed_projections_view_one_weight_and_match_the_separate_gemms(monkeypatch):
    H.install_cpu_stubs(monkeypatch)
    _, sub, _, _ = H.build_engine(shared_experts_overlap=False)
    lm = sub.language_model
    moes = [m for m in lm.modules() if isinstance(m, _LatentMoE)]
    mlas = [m for m in lm.modules() if isinstance(m, KimiMLAAttention)]
    assert moes and mlas
    torch.manual_seed(0)
    for moe in moes:
        w = moe._in_weight
        parts = (moe.gate.weight, moe.routed_expert_down_proj.weight, moe.shared_experts.gate_up_proj.weight)
        assert w.shape[0] == sum(p.shape[0] for p in parts)
        assert all(_same_storage(p, w) for p in parts)
        x = torch.randn(5, moe.model_hidden_size)
        with torch.no_grad():
            packed = moe(x)
            moe._in_weight = None  # the three GEMMs apart
            separate = moe(x)
            moe._in_weight = w
        assert torch.equal(packed, separate)
    for mla in mlas:
        w = mla._qkv_a_g_weight
        assert _same_storage(mla.fused_qkv_a_proj_weight, w) and _same_storage(mla.g_proj.weight, w)
        h = torch.randn(4, w.shape[1])
        with torch.no_grad():
            qkv_a = mla._fused_qkv_a(h)
            assert torch.equal(qkv_a, F.linear(h, mla.fused_qkv_a_proj_weight))
            stashed, gate = mla._gate_pre
            assert stashed is h and torch.equal(gate, mla.g_proj(h))
            # a path that ran no packed GEMM for these rows projects its own gate
            attn = torch.randn(4, mla.g_proj.weight.shape[0])
            out = mla._project_out(attn, torch.randn_like(h))
            assert mla._gate_pre is None and out.shape == (4, mla.o_proj.weight.shape[0])


def test_the_up_and_shared_down_projections_pack_along_k_into_one_gemm(monkeypatch):
    """The latent up projection's and the shared experts' down projection's partials, summed by the layer's
    all-reduce, as one GEMM over ``[this rank's latent slice | the shared activation]``: their weights view
    the packed one, and the sum matches the two GEMMs and the add (fp32 on the CPU: summation order apart)."""
    H.install_cpu_stubs(monkeypatch)
    _, sub, _, _ = H.build_engine(shared_experts_overlap=False)
    moes = [m for m in sub.language_model.modules() if isinstance(m, _LatentMoE)]
    assert moes
    torch.manual_seed(1)
    for moe in moes:
        w = moe._out_weight
        up, down = moe.routed_expert_up_proj.weight, moe.shared_experts.down_proj.weight
        assert w.shape == (up.shape[0], up.shape[1] + down.shape[1])
        assert _same_storage(up, w) and _same_storage(down, w)
        x = torch.randn(5, moe.model_hidden_size)
        with torch.no_grad():
            packed = moe(x)
            moe._out_weight = None  # the up GEMM, the shared experts' down GEMM and their add
            separate = moe(x)
            moe._out_weight = w
        torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-5)


def test_each_rank_packs_its_own_latent_slice():
    """At TP2 the up projection reads its rank's half of the (replicated) latent: the packed GEMM slices
    the same half, on either rank."""
    from mstar.distributed.communication import CommGroup
    from mstar.model.kimi_k3.components.moe import KimiLatentMoE
    from mstar.model.kimi_k3.config import KimiK3ModelConfig

    cfg = KimiK3ModelConfig.reduced(latent_moe_use_norm=False, shared_experts_overlap=False)
    for rank in (0, 1):
        group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
        group.initialized = True
        moe = KimiLatentMoE(cfg, comm_group=group)
        with torch.no_grad():
            for p in moe.parameters():
                p.normal_(0.0, 0.05)
        moe.process_weights_after_loading("cpu")
        assert moe._out_weight is not None
        y = torch.randn(3, cfg.routed_expert_hidden_size)
        x = torch.randn(3, cfg.hidden_size)
        gate_up = moe.shared_experts.gate_up_proj(x)
        with torch.no_grad():
            packed = moe._project_out(x, y, gate_up)
            separate = moe.routed_expert_up_proj(y) + moe.shared_experts.from_gate_up(gate_up)
        torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-5)


def test_the_shared_experts_beside_the_routed_path_compute_the_same_block(monkeypatch):
    """``shared_experts_overlap`` routed or shared: the packed input GEMM holds the router and the latent down
    projection only, nothing is packed along K, one branch runs on the side stream (``side_stream.fork``, inline
    on the CPU), and the shared experts join in the up projection's GEMM (``addmm``). The block's output is the
    packed block's up to summation order (fp32)."""
    H.install_cpu_stubs(monkeypatch)
    _, packed_sub, _, _ = H.build_engine(shared_experts_overlap="off")
    packed = [m for m in packed_sub.language_model.modules() if isinstance(m, _LatentMoE)]
    for mode in ("shared", "routed"):
        _, beside_sub, _, _ = H.build_engine(shared_experts_overlap=mode)
        beside = [m for m in beside_sub.language_model.modules() if isinstance(m, _LatentMoE)]
        assert packed and len(packed) == len(beside) and all(b.moe_overlap == mode for b in beside)
        _check_overlapped(packed, beside)


def _check_overlapped(packed, beside):
    """Both layouts view one set of weights: the packed input GEMM holds the router, the latent down projection
    and the shared gate_up, the up and shared down projections pack along K; an overlapped step reads the first
    two of the input rows as one slice and the shared experts' weights as the views they are. Above the overlap's
    row cap the step takes the packed single-stream path."""
    torch.manual_seed(2)
    for a, b in zip(packed, beside, strict=True):
        assert b._out_weight is not None and b._in_weight is not None
        parts = (b.gate.weight, b.routed_expert_down_proj.weight, b.shared_experts.gate_up_proj.weight)
        assert b._in_weight.shape[0] == sum(p.shape[0] for p in parts)
        assert all(_same_storage(p, b._in_weight) for p in parts)
        assert _same_storage(b.shared_experts.down_proj.weight, b._out_weight)
        for rows in (5, b.shared_experts_overlap_max_tokens + 3):
            x = torch.randn(rows, b.model_hidden_size)
            with torch.no_grad():
                torch.testing.assert_close(b(x), a(x), rtol=1e-5, atol=1e-5)


def test_each_rank_adds_the_shared_partial_to_its_own_latent_slice():
    """At TP2, beside: the up projection's GEMM reads its rank's half of the latent and adds the rank's
    shared-experts partial (``_up_plus``), as the separate projections and the add do."""
    from mstar.distributed.communication import CommGroup
    from mstar.model.kimi_k3.components.moe import KimiLatentMoE
    from mstar.model.kimi_k3.config import KimiK3ModelConfig

    cfg = KimiK3ModelConfig.reduced(latent_moe_use_norm=False)
    assert cfg.shared_experts_overlap == "shared"
    for rank in (0, 1):
        group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
        group.initialized = True
        moe = KimiLatentMoE(cfg, comm_group=group)
        with torch.no_grad():
            for p in moe.parameters():
                p.normal_(0.0, 0.05)
        moe.process_weights_after_loading("cpu")
        assert moe._out_weight is not None  # the up projection's weight is a column slice of it
        y = torch.randn(3, cfg.routed_expert_hidden_size)
        x = torch.randn(3, cfg.hidden_size)
        with torch.no_grad():
            shared = moe._shared_partial(x)
            torch.testing.assert_close(shared, moe.shared_experts(x), rtol=0, atol=0)
            torch.testing.assert_close(moe._up_plus(y, shared), moe.routed_expert_up_proj(y) + shared,
                                       rtol=1e-5, atol=1e-5)


def test_the_overlap_mode_reads_true_and_false_and_refuses_others():
    from mstar.model.kimi_k3.config import KimiK3ModelConfig

    assert KimiK3ModelConfig.reduced(shared_experts_overlap=True).shared_experts_overlap == "shared"
    assert KimiK3ModelConfig.reduced(shared_experts_overlap=False).shared_experts_overlap == "off"
    import pytest

    with pytest.raises(ValueError, match="routed, shared or off"):
        KimiK3ModelConfig.reduced(shared_experts_overlap="both")
    # expert parallel routing syncs over the group: the routed experts stay on the main stream
    from mstar.model.kimi_k3.components.moe import KimiLatentMoEExpertParallel

    cfg = KimiK3ModelConfig.reduced(moe_expert_parallel=True)
    assert KimiLatentMoEExpertParallel(cfg).moe_overlap == "shared"


def test_a_padding_row_masked_out_of_the_routed_experts_leaves_the_real_rows_alone(monkeypatch):
    """``token_valid``: the padding rows of a padded step skip the routed experts (sentinel id, weight 0) in every
    overlap mode; the real rows come out as the step without them gives them, and the engine stages the count
    (``num_valid_tokens``) only with ``moe_mask_padding``."""
    H.install_cpu_stubs(monkeypatch)
    for mode in ("off", "shared", "routed"):
        _, sub, _, _ = H.build_engine(shared_experts_overlap=mode, moe_mask_padding=True)
        moe = next(m for m in sub.language_model.modules() if isinstance(m, _LatentMoE))
        torch.manual_seed(3)
        x = torch.randn(4, moe.model_hidden_size)
        valid = torch.tensor([True, True, True, False])
        with torch.no_grad():
            torch.testing.assert_close(moe(x, token_valid=valid)[:3], moe(x[:3]), rtol=1e-6, atol=1e-6)


def test_every_overlap_mode_leaves_routing_and_the_mxfp8_cast_to_a_kernel_that_routes(monkeypatch):
    """With the experts kernel routing from the logits (``from_logits``), each overlap mode, below and above its
    row cap, skips the block's router and hands the kernel the packed GEMM's fp32 logits; with ``takes_mxfp8`` z
    goes as the cast kernel's ``(values, scales)``, and an overlapped step's latent-only GEMM gives no gate_up.
    The block's output is the torch-routed one's (fp32)."""
    from types import SimpleNamespace

    from mstar.utils.fused_moe import mxfp8_cast

    H.install_cpu_stubs(monkeypatch)
    split = []

    def split_cast(h, start, latent):  # stands in for the Triton kernel: z unquantized, a unit scale
        split.append(h.shape[1] - start - latent)
        return h[:, start:start + latent].clone(), torch.zeros(h.shape[0], latent // 32), h[:, start + latent:]

    monkeypatch.setattr(mxfp8_cast, "supported", lambda h, latent: True)
    monkeypatch.setattr(torch.ops.mstar, "split_cast_mxfp8", split_cast, raising=False)
    torch.manual_seed(3)
    for mode in ("off", "shared", "routed"):
        _, sub, _, _ = H.build_engine(shared_experts_overlap=mode)
        blocks = [m for m in sub.language_model.modules() if isinstance(m, _LatentMoE)]
        assert blocks
        for b in blocks:
            route = type(b.gate).route
            for mxfp8 in (False, True):
                for rows in (5, b.shared_experts_overlap_max_tokens + 3):
                    x = torch.randn(rows, b.model_hidden_size)
                    b.__dict__.pop("_experts_kernel", None)
                    with torch.no_grad():
                        want = b(x)
                    calls = []

                    def from_logits(z, logits, b=b, calls=calls, route=route):
                        calls.append((type(z), logits.shape))
                        z = z[0] if isinstance(z, tuple) else z
                        weights, ids, _ = route(b.gate, logits)
                        return b.experts_forward(z, weights, ids, reduce_results=False)

                    b._experts_kernel = SimpleNamespace(routing=object(), from_logits=from_logits,
                                                        takes_mxfp8=lambda m=mxfp8: m)
                    split.clear()
                    b.gate.route = lambda *a, **k: pytest.fail("the block's router ran")
                    try:
                        with torch.no_grad():
                            got = b(x)
                    finally:
                        del b.gate.route
                    overlapped = mode != "off" and rows <= b.shared_experts_overlap_max_tokens
                    assert calls == [(tuple if mxfp8 else torch.Tensor, (rows, b.gate.n_routed_experts))]
                    # an overlapped step's packed GEMM holds no shared gate_up: the cast writes none
                    assert split == ([0] if overlapped else [b.shared_experts.gate_up_proj.weight.shape[0]]
                                     ) if mxfp8 else not split
                    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
            b.__dict__.pop("_experts_kernel", None)
