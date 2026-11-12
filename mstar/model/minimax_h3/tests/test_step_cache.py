"""The MiniMax-H3 DiT's block stack under the scaffold's step cache (``components/diffusion/step_cache.py``), tiny and
on CPU: the residual it hands back is what its blocks added, and reusing it skips them."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from mstar.model.components.diffusion.step_cache import MagCache
from mstar.model.minimax_h3 import weight_loader
from mstar.model.minimax_h3.components.layout import H3Layout, modality_tags, position_ids, timestep_classes
from mstar.model.minimax_h3.components.transformer import H3Inputs, H3StepLayout, rope_inv_freq, rope_table
from mstar.model.minimax_h3.config import MAG_RATIOS, NUM_MODALITIES, H3Config
from mstar.model.minimax_h3.minimax_h3_model import H3Model
from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG, random_checkpoint


@pytest.fixture(scope="module")
def dit(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("h3")
    (root / "FL2VA" / "transformer").mkdir(parents=True)
    save_file(random_checkpoint(CFG), str(root / "FL2VA" / "transformer" / "model.safetensors"))
    return weight_loader.build_transformer(H3Config(transformer=CFG), root, "cpu", dtype=torch.float32)


def _step(seed: int):
    layout = H3Layout(text_len=5, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4,
                      anchors=("first",))
    classes, tags = timestep_classes(layout), modality_tags(layout)
    step_layout = H3StepLayout(
        rope=rope_table(position_ids(layout), rope_inv_freq(CFG), torch.float32, CFG.attention_head_dim),
        row_index=classes * NUM_MODALITIES + tags, row_class=classes, no_class=torch.zeros_like(classes), tags=tags,
        text_len=layout.text_len, num_condition_rows=layout.num_condition_rows, num_audio_rows=layout.num_audio_rows,
    )
    g = torch.Generator().manual_seed(seed)
    inputs = H3Inputs(
        video=torch.randn(1, layout.num_condition_rows + layout.num_video_rows, CFG.patch_dim, generator=g),
        audio=torch.randn(1, layout.num_audio_rows, CFG.audio_latents_dim, generator=g),
        text=torch.randn(1, layout.text_len, CFG.hidden_size, generator=g),
        timesteps=torch.tensor([[0.4, 0.5, 0.999]]),
    )
    return inputs, step_layout


def test_the_residual_is_what_the_blocks_added_and_reusing_it_skips_them(dit):
    inputs, layout = _step(1)
    calls = []
    hook = dit.blocks[0].register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.no_grad():
            plain = dit(inputs, layout)
            video, audio, residual = dit(inputs, layout, return_residual=True)
            assert len(calls) == 2
            reused = dit(inputs, layout, cached_residual=residual)
            assert len(calls) == 2  # no block ran
            moved = dit(_step(2)[0], layout, cached_residual=residual)
    finally:
        hook.remove()
    assert torch.equal(video, plain[0]) and torch.equal(audio, plain[1])
    assert residual.dtype == torch.float32 and residual.shape == (1, layout.num_rows, CFG.hidden_size)
    for got, want in zip(reused, plain, strict=True):
        torch.testing.assert_close(got, want)
    # a reusing step still runs its own input projection and head on its own inputs
    assert not torch.equal(moved[0], plain[0])


def test_the_serving_config_names_the_policy():
    assert H3Model(skip_weight_loading=True).step_cache is None
    model = H3Model(skip_weight_loading=True, step_cache={"mag_ratios": [1.0, 0.95], "threshold": 0.1})
    assert model.step_cache == MagCache(mag_ratios=(1.0, 0.95), threshold=0.1)
    assert H3Model(skip_weight_loading=True, step_cache={"method": "magcache"}).step_cache.mag_ratios == (
        MAG_RATIOS["default"])
    with pytest.raises(ValueError, match="no calibrated"):
        H3Model(skip_weight_loading=True, step_cache={"mag_ratios": "768p"})
