"""The native MiniMax-H3 DiT against diffusers' ``MiniMaxH3Transformer3DModel``, tiny and on CPU.

One random original-format (FL2VA) checkpoint goes both ways: through this port's loader
(per-head interleaved q/k/v, ``[gate; value]`` feed-forward, the refiner on the text
node's conditioner) and through diffusers' own FL2VA conversion. The packed layout,
the positions and the velocities must agree, with and without keyframe rows, so the
key mapping, the AdaLN row selection and the partial rope are checked before any real
weights are loaded.
"""

from __future__ import annotations

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "MiniMaxH3Transformer3DModel"):
    pytest.skip("this diffusers has no MiniMax-H3", allow_module_level=True)

from safetensors.torch import save_file  # noqa: E402

from mstar.model.minimax_h3 import weight_loader  # noqa: E402
from mstar.model.minimax_h3.components.layout import (  # noqa: E402
    T_AUDIO,
    T_COND,
    T_VIDEO,
    H3Layout,
    modality_tags,
    position_ids,
    timestep_classes,
)
from mstar.model.minimax_h3.components.transformer import (  # noqa: E402
    H3Inputs,
    H3StepLayout,
    apply_rope,
    rope_inv_freq,
    rope_table,
)
from mstar.model.minimax_h3.config import NUM_MODALITIES, H3Config, H3TransformerConfig  # noqa: E402
from mstar.model.minimax_h3.tests.tiny_checkpoint import CFG, random_checkpoint  # noqa: E402


def _diffusers_model(state: dict[str, torch.Tensor], cfg: H3TransformerConfig):
    from diffusers.loaders.single_file_utils import convert_minimax_h3_transformer_checkpoint_to_diffusers

    dcfg = dict(
        num_attention_heads=cfg.num_attention_heads, attention_head_dim=cfg.attention_head_dim,
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, num_refiner_layers=cfg.token_refiner_num_layers,
        ffn_dim=cfg.ffn_hidden_size, in_channels=cfg.latents_dim, audio_in_channels=cfg.audio_latents_dim,
        patch_size=cfg.patch_size, text_dim=cfg.text_dim, freq_dim=cfg.timestep_input_dim,
        time_embed_hidden_dim=cfg.time_embed_hidden_size, time_embed_dim=cfg.time_embed_dim,
        rope_freq_dim=cfg.rope_inv_freq_len,
    )
    model = diffusers.MiniMaxH3Transformer3DModel(**dcfg).eval()
    converted = convert_minimax_h3_transformer_checkpoint_to_diffusers(
        {k: v.clone() for k, v in state.items()}, dcfg, qkv_layout="interleaved",
    )
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not unexpected and all(m.endswith("rope.inv_freq") for m in missing), (missing, unexpected)
    return model


def _reference_layout(layout: H3Layout, t_video: float, t_audio: float):
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
    )

    pos, tags, vid, aud, txt, n_cond, _ = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(layout.text_len, dtype=torch.long), layout.latent_frames, layout.latent_height,
        layout.latent_width, layout.audio_latents, (1, 2, 2), 2, 2, 0, layout.anchors,
    )
    unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        vid, aud, n_cond, 0, layout.text_len, t_video, t_audio, max(t_video, 0.999), 1.0,
    )
    return dict(position_ids=pos, token_tags=tags, video_indices=vid, audio_indices=aud, text_indices=txt,
                timestep=unique, timestep_indices=inverse)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("h3")
    state = random_checkpoint(CFG)
    (root / "FL2VA" / "transformer").mkdir(parents=True)
    save_file(state, str(root / "FL2VA" / "transformer" / "model.safetensors"))
    config = H3Config(transformer=CFG)
    dit = weight_loader.build_transformer(config, root, "cpu", dtype=torch.float32)
    conditioner = weight_loader.build_text_conditioner(config, root, "cpu", dtype=torch.float32)
    return dit, conditioner, _diffusers_model(state, CFG)


@pytest.mark.parametrize("anchors", [(), ("first", "last")])
def test_velocity_matches_diffusers(models, anchors):
    dit, conditioner, reference = models
    layout = H3Layout(text_len=5, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4, anchors=anchors)
    t_video, t_audio = 0.3, 0.6
    ref = _reference_layout(layout, t_video, t_audio)
    assert torch.equal(position_ids(layout), ref["position_ids"])
    assert torch.equal(modality_tags(layout), ref["token_tags"])

    g = torch.Generator().manual_seed(1)
    num_video = layout.num_condition_rows + layout.num_video_rows
    video = torch.randn(1, num_video, CFG.patch_dim, generator=g)
    audio = torch.randn(1, layout.num_audio_rows, CFG.audio_latents_dim, generator=g)
    hidden = torch.randn(1, layout.text_len, CFG.text_dim, generator=g)
    with torch.no_grad():
        want_video, want_audio = reference(
            hidden_states=video, audio_hidden_states=audio, encoder_hidden_states=hidden, return_dict=False, **ref,
        )
        text = conditioner(hidden)
        classes, tags = timestep_classes(layout), modality_tags(layout)
        step_layout = H3StepLayout(
            rope=rope_table(position_ids(layout), rope_inv_freq(CFG), torch.float32, CFG.attention_head_dim),
            row_index=classes * NUM_MODALITIES + tags, row_class=classes, no_class=torch.zeros_like(classes),
            tags=tags,
            text_len=layout.text_len, num_condition_rows=layout.num_condition_rows,
            num_audio_rows=layout.num_audio_rows,
        )
        timesteps = torch.empty(1, 3)
        timesteps[0, T_VIDEO], timesteps[0, T_AUDIO], timesteps[0, T_COND] = t_video, t_audio, max(t_video, 0.999)
        got_video, got_audio = dit(H3Inputs(video=video, audio=audio, text=text, timesteps=timesteps), step_layout)
    torch.testing.assert_close(got_video, want_video, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got_audio, want_audio, rtol=1e-4, atol=1e-5)
    # the reference's own timestep layout (its distinct values, each row's index into them)
    with torch.no_grad():
        exact_video, exact_audio = dit(
            H3Inputs(video=video, audio=audio, text=text, timesteps=ref["timestep"][None]), step_layout,
            row_class=ref["timestep_indices"],
        )
    torch.testing.assert_close(exact_video, want_video, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(exact_audio, want_audio, rtol=1e-4, atol=1e-5)


def test_rope_rounds_as_diffusers_in_bf16():
    """The rope is one product over the whole head; in bf16 every channel must still round
    as diffusers' rotate-half does."""
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed, _apply_rotary_emb

    cfg = H3TransformerConfig()
    pos = position_ids(H3Layout(text_len=7, latent_frames=3, latent_height=4, latent_width=6, audio_latents=4))
    g = torch.Generator().manual_seed(0)
    x = (3 * torch.randn(1, pos.shape[0], 4, cfg.attention_head_dim, generator=g)).to(torch.bfloat16)
    want = _apply_rotary_emb(x, *MiniMaxH3RotaryPosEmbed(cfg.rope_inv_freq_len, cfg.rope_theta)(pos))
    got = apply_rope(x, rope_table(pos, rope_inv_freq(cfg), torch.bfloat16, cfg.attention_head_dim))
    assert torch.equal(got, want)
