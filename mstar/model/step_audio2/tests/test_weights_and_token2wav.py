"""Step-Audio2's checkpoint mapping and its token2wav streaming knobs (CPU).

The loader's renames are checked on their own; against the real checkpoint's index (by
name and shape, on ``meta``, so nothing is read) only where the checkpoint is present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.components.token2wav.flow import PRE_LOOKAHEAD
from mstar.model.components.token2wav.model import (
    CACHE_FAMILIES,
    KEEP_RECENT,
    SILENCE_CODE,
    Token2Wav,
    Token2WavStreaming,
)
from mstar.model.components.token2wav.voice_prompt import VoicePrompt
from mstar.model.step_audio2.components.audio import StepAudio2Audio
from mstar.model.step_audio2.config import AudioEncoderConfig, StepAudio2Config
from mstar.model.step_audio2.token2wav import CONFORMER_REPLY_TOKENS, T2W_STREAMING
from mstar.model.step_audio2.weight_loader import audio_param_name, load_audio_weights, load_llm_weights

CKPT = Path(os.environ.get("STEPAUDIO2_CKPT", "/workspace/stepaudio2-ref/hf/Step-Audio-2-mini"))
needs_ckpt = pytest.mark.skipif(not (CKPT / "model.safetensors.index.json").exists(),
                                reason=f"needs the Step-Audio-2-mini checkpoint under {CKPT}")


@pytest.mark.parametrize(("key", "name"), [
    ("encoder.blocks.3.attn_ln.weight", "encoder.blocks.3.self_attn_layer_norm.weight"),
    ("encoder.blocks.3.attn.query.bias", "encoder.blocks.3.self_attn.q_proj.bias"),
    ("encoder.blocks.3.attn.key.weight", "encoder.blocks.3.self_attn.k_proj.weight"),
    ("encoder.blocks.3.attn.value.weight", "encoder.blocks.3.self_attn.v_proj.weight"),
    ("encoder.blocks.31.attn.out.weight", "encoder.blocks.31.self_attn.out_proj.weight"),
    ("encoder.blocks.0.mlp_ln.bias", "encoder.blocks.0.final_layer_norm.bias"),
    ("encoder.blocks.0.mlp.0.weight", "encoder.blocks.0.fc1.weight"),
    ("encoder.blocks.0.mlp.2.bias", "encoder.blocks.0.fc2.bias"),
    ("encoder.conv1.weight", "encoder.conv1.weight"),
    ("adapter.linear2.bias", "adapter.linear2.bias"),
])
def test_whisper_block_parameters_rename_onto_aut_layers(key, name):
    assert audio_param_name(key) == name


def test_an_unknown_block_parameter_is_refused():
    with pytest.raises(KeyError):
        audio_param_name("encoder.blocks.0.attn.rotary.weight")


TINY = AudioEncoderConfig(n_mels=4, n_audio_ctx=8, n_audio_state=8, n_audio_head=2, n_audio_layer=1, llm_dim=6,
                          adapter_hidden=5)


def tiny_checkpoint() -> dict[str, torch.Tensor]:
    """``modeling_step_audio_2.py``'s parameter names and shapes for ``TINY``."""
    d, f = TINY.n_audio_state, TINY.ffn_dim
    state = {"encoder.conv1.weight": (d, TINY.n_mels, 3), "encoder.conv1.bias": (d,),
             "encoder.conv2.weight": (d, d, 3), "encoder.conv2.bias": (d,),
             "encoder.positional_embedding.weight": (TINY.n_audio_ctx, d),
             "encoder.after_norm.weight": (d,), "encoder.after_norm.bias": (d,),
             "adapter.conv.weight": (d, d, 3), "adapter.conv.bias": (d,),
             "adapter.linear1.weight": (TINY.adapter_hidden, d), "adapter.linear1.bias": (TINY.adapter_hidden,),
             "adapter.linear2.weight": (TINY.llm_dim, TINY.adapter_hidden), "adapter.linear2.bias": (TINY.llm_dim,)}
    block = {"attn.query.weight": (d, d), "attn.query.bias": (d,), "attn.key.weight": (d, d),
             "attn.value.weight": (d, d), "attn.value.bias": (d,), "attn.out.weight": (d, d), "attn.out.bias": (d,),
             "attn_ln.weight": (d,), "attn_ln.bias": (d,), "mlp_ln.weight": (d,), "mlp_ln.bias": (d,),
             "mlp.0.weight": (f, d), "mlp.0.bias": (f,), "mlp.2.weight": (d, f), "mlp.2.bias": (d,)}
    state.update({f"encoder.blocks.0.{k}": v for k, v in block.items()})
    return {k: torch.randn(v) for k, v in state.items()}


def test_the_audio_loader_fills_every_parameter_and_zeroes_the_missing_key_bias():
    state = tiny_checkpoint()
    model = StepAudio2Audio(TINY, attn_key=None)
    load_audio_weights(model, weights=state.items())
    qkv = model.encoder.blocks[0].self_attn.qkv_proj
    d = TINY.n_audio_state
    assert torch.equal(qkv.weight[d:2 * d], state["encoder.blocks.0.attn.key.weight"])
    assert torch.equal(qkv.bias[d:2 * d], torch.zeros(d))
    assert torch.equal(qkv.bias[2 * d:], state["encoder.blocks.0.attn.value.bias"])
    assert torch.equal(model.encoder.blocks[0].fc2.bias, state["encoder.blocks.0.mlp.2.bias"])


def test_the_audio_loader_raises_on_a_missing_or_unknown_key():
    state = tiny_checkpoint()
    with pytest.raises(RuntimeError, match="unloaded"):
        load_audio_weights(StepAudio2Audio(TINY, attn_key=None),
                           weights=[(k, v) for k, v in state.items() if k != "adapter.conv.bias"])
    with pytest.raises(RuntimeError, match="name no parameter"):
        load_audio_weights(StepAudio2Audio(TINY, attn_key=None),
                           weights=[*state.items(), ("encoder.extra.weight", torch.zeros(1))])


@needs_ckpt
def test_the_modules_take_the_whole_checkpoint_by_name_and_shape():
    from safetensors import safe_open

    config = StepAudio2Config.from_hf(CKPT)
    index = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = {}
    for shard in set(index.values()):
        with safe_open(str(CKPT / shard), "pt") as f:
            shapes.update({k: tuple(f.get_slice(k).get_shape()) for k in f.keys()})  # noqa: SIM118

    def stream(prefixes):
        return [(k, torch.empty(s, device="meta")) for k, s in shapes.items() if k.startswith(prefixes)]

    with torch.device("meta"):
        audio = StepAudio2Audio(config.audio, attn_key="audio_attn")
        llm = Qwen3DenseLM(config.llm, attn_key="a", kv_key="k", pos_key="p")
    audio_keys, llm_keys = stream(("encoder.", "adapter.")), stream(("model.", "lm_head."))
    assert len(audio_keys) + len(llm_keys) == len(shapes)
    load_audio_weights(audio, weights=audio_keys)
    load_llm_weights(llm, weights=llm_keys)


@needs_ckpt
def test_the_config_reads_the_checkpoint():
    config = StepAudio2Config.from_hf(CKPT)
    llm = config.llm
    assert (llm.num_hidden_layers, llm.hidden_size) == (28, 3584)
    assert (llm.num_attention_heads, llm.num_key_value_heads) == (28, 4)
    assert (llm.head_dim, llm.vocab_size, llm.rope_theta) == (128, 158720, 1e6)
    assert llm.attention_bias and not llm.qk_norm and not llm.tie_word_embeddings
    assert (config.audio.n_audio_layer, config.audio.n_audio_state, config.audio.n_mels) == (32, 1280, 128)


# --- token2wav's streaming knobs ------------------------------------------------------------

def test_minicpm_o_keeps_its_knobs_by_default():
    default = Token2WavStreaming()
    assert default.cache_families() is CACHE_FAMILIES
    assert (default.prompt_lookahead, default.lead_silence, default.first_window_zeros) == (False, 3, True)
    assert default.capacity(100).enc_25hz == 100 + KEEP_RECENT // 2 + 27


def test_step_audio2_keeps_the_conformer_caches_whole_up_to_its_cap():
    families = T2W_STREAMING.cache_families()
    p = 226
    for name, rate in (("enc1", 1), ("enc2", 2)):
        policy = families[name].retention(rate * p)
        assert (policy.sink, policy.window) == (rate * (p + CONFORMER_REPLY_TOKENS), 0)
    # the DiT's cache is cut as MiniCPM-o's is
    assert families["dit"].retention(2 * p) == CACHE_FAMILIES["dit"].retention(2 * p)
    cap = T2W_STREAMING.capacity(p)
    assert cap.enc_25hz == p + CONFORMER_REPLY_TOKENS + 27 and cap.enc_frames_50hz == 2 * cap.enc_25hz
    # the reference's relative position table reaches 5000 frames back: a 16 s voice (400
    # tokens) and a reply at the cap, plus a last window's 54 frames, still fit
    assert 2 * (400 + CONFORMER_REPLY_TOKENS) + 2 * 27 <= 5000


def test_token2wav_resources_declare_the_conformer_caches_whole():
    from mstar.model.components.token2wav.keys import Token2WavKeys
    from mstar.model.components.token2wav.resources import token2wav_resources

    keys = Token2WavKeys(codes="c", state="s", kv={"enc1": "e1", "enc2": "e2", "dit": "d"})
    specs = {s.resource_key: s for s in token2wav_resources(keys, "t2w", (9.0, 9.0), 3, streaming=T2W_STREAMING)}
    p = int(9.0 * 25) + 1
    assert specs["e1"].config.retention_for(p).sink == p + CONFORMER_REPLY_TOKENS
    assert specs["e2"].config.sink_capacity == 2 * CONFORMER_REPLY_TOKENS
    default = {s.resource_key: s for s in token2wav_resources(keys, "t2w", (9.0, 9.0), 3)}
    assert default["e2"].config.window_capacity == KEEP_RECENT


@pytest.mark.parametrize("lookahead", [False, True])
def test_a_voice_is_set_up_with_its_reference_lookahead(lookahead, monkeypatch):
    torch.manual_seed(0)
    model = Token2Wav(Token2WavStreaming(prompt_lookahead=lookahead)).eval()
    with torch.no_grad():
        for prm in model.parameters():
            prm.normal_(0, 0.02)
    seen = []
    flow = model.flow.forward
    monkeypatch.setattr(model.flow, "forward", lambda tokens, *a: seen.append(tokens.clone()) or flow(tokens, *a))
    tokens = torch.randint(0, 6561, (1, 8), dtype=torch.int32)
    model.prepare_voice(VoicePrompt(tokens=tokens, spk_emb=torch.randn(1, 192), mel=torch.randn(1, 16, 80)))
    tail = tokens[:, :PRE_LOOKAHEAD] if lookahead else torch.full((1, PRE_LOOKAHEAD), SILENCE_CODE, dtype=torch.int32)
    assert torch.equal(seen[0], torch.cat([tokens, tail], dim=1))
