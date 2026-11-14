"""CPU parity of the Praxis-1 decoder against the reference checkout."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("praxis", reason="Praxis-1 reference checkout is not on PYTHONPATH")
from praxis import backbone as ref_backbone  # noqa: E402
from praxis import layers as ref_layers

from mstar.model.praxis1.components import decoder as port_dec  # noqa: E402


def _fill(module: torch.nn.Module) -> None:
    gen = torch.Generator().manual_seed(41)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
        for name, buf in module.named_buffers():
            if name.endswith(("freqs", "phases")):
                buf.copy_(torch.randn(buf.shape, generator=gen))


def _dense_remap(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        for i in range(2):
            head = f"core.blocks.{i}.ffn."
            if key.startswith(head):
                tail = key[len(head) :]
                key = head + tail.replace("0.linear.", "linear.").replace("1.weight", "down.weight")
                break
        out[key] = value
    return out


def _pair(dtype: torch.dtype):
    params = dict(
        depth=4,
        num_dense=2,
        dim=64,
        head_dim=32,
        num_kv_heads=1,
        dense_hidden=32,
        moe_hidden=16,
        num_experts=8,
        top_k=4,
        routed_scaling_factor=2.5,
        mod_dim=64,
        rope_effective_dim=16,
        moe_dispatch_chunk_size=16,
    )
    ref_core = ref_layers.Transformer(**params, attention_backend="triton")
    ref = ref_backbone.DecoderTower(
        ref_core, dim=64, video_dim=8, time_dim=8, num_registers=4, patch_size=(2, 2), mask_block_size=16
    )
    _fill(ref)
    port_core = port_dec.TrunkStack(**params)
    port = port_dec.DecoderTower(port_core, dim=64, video_dim=8, time_dim=8, num_registers=4, patch_size=(2, 2))
    port.load_state_dict(_dense_remap(ref.state_dict()))
    return ref.to(dtype).eval(), port.to(dtype).eval()


def _flex_attend(q, k, v, mask):
    return ref_layers._compiled_flex_attention(
        q,
        k,
        v,
        block_mask=mask,
        scale=None,
        enable_gqa=k.shape[1] != q.shape[1],
        kernel_options=None,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunks", [1, 2])
def test_decoder_matches_reference(dtype, chunks):
    import torch._dynamo.config as dynamo_config

    previous = dynamo_config.automatic_dynamic_shapes
    dynamo_config.automatic_dynamic_shapes = False
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ref, port = _pair(dtype)
        gen = torch.Generator().manual_seed(17)
        frames = chunks * 2
        noised = (torch.randn((1, 32, frames, 2, 3), generator=gen) * 0.5).to(dtype)
        cond = (torch.randn((1, frames, 2, 3, 64), generator=gen) * 0.5).to(dtype)
        timesteps = torch.rand((1, chunks), generator=gen).to(dtype)
        mask = ref_backbone.teacher_forcing_mask(
            [("global", 1, 4), ("noised", frames, 24)],
            2,
            False,
            16,
            torch.device("cpu"),
            False,
        )
        with torch.no_grad():
            expected = ref.forward_chunk_causal(noised, cond, timesteps, rope_shift=3)
            actual = port.forward_chunk_causal(noised, cond, timesteps, rope_shift=3, mask=mask, attend=_flex_attend)
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    finally:
        dynamo_config.automatic_dynamic_shapes = previous
        torch.set_num_threads(previous_threads)


def test_patch_roundtrip_and_reference():
    x = torch.arange(1 * 8 * 2 * 2 * 3).reshape(1, 8, 2, 2, 3).float()
    got = port_dec.unpatch(x, (2, 2))
    assert torch.equal(got, ref_backbone.unpatch(x, (2, 2)))
    assert torch.equal(port_dec.patch(got, (2, 2)), x)


def test_each_conditioning_chunk_changes_its_decoder_output():
    _, port = _pair(torch.float32)
    gen = torch.Generator().manual_seed(29)
    noised = torch.randn((1, 32, 4, 2, 3), generator=gen)
    cond = torch.randn((1, 4, 2, 3, 64), generator=gen)
    timesteps = torch.tensor([[0.1, 0.8]])
    with torch.no_grad():
        original = port.forward_chunk_causal(noised, cond, timesteps)
        changed = port.forward_chunk_causal(noised, cond.clone().index_fill(1, torch.tensor([2, 3]), 0), timesteps)
    assert torch.allclose(original[:, :, :2], changed[:, :, :2], atol=1e-5, rtol=1e-5)
    assert not torch.equal(original[:, :, 2:], changed[:, :, 2:])


def test_decoder_rejects_wrong_conditioning_grid():
    _, port = _pair(torch.float32)
    noised = torch.zeros(1, 32, 2, 2, 3)
    cond = torch.zeros(1, 2, 4, 6, 64)
    with pytest.raises(ValueError, match="conditioning shape"):
        port.forward_chunk_causal(noised, cond, torch.zeros(1, 1))
