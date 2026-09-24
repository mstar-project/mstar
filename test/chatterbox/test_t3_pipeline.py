"""Greedy speech-token parity of the T3 node against the reference package.

Drives ``T3Submodule`` exactly as the engine does (prepare_inputs ->
preprocess -> declare_step -> forward, prefill then decode steps) on CPU
through the fake KV/attention/position resources and an argmax sampler, and
compares the tokens with a greedy re-run of the reference ``T3.inference``
loop (multinomial replaced by argmax, everything else as written). Then the
offline S3Gen path turns the tokens into audio and is checked against the
reference ``S3Gen.inference`` under the same seed.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

chatterbox = pytest.importorskip("chatterbox")
from chatterbox.models.s3gen import S3Gen as RefS3Gen  # noqa: E402
from chatterbox.models.t3 import T3 as RefT3  # noqa: E402, N811
from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend  # noqa: E402
from chatterbox.models.t3.modules.cond_enc import T3Cond  # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config as RefT3Config  # noqa: E402
from fake_resources import FakeT3Resources  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from mstar.model.chatterbox.components.s3gen import ReferenceConditioning, S3Gen  # noqa: E402
from mstar.model.chatterbox.components.t3 import T3Model  # noqa: E402
from mstar.model.chatterbox.config import (  # noqa: E402
    T3_ATTN,
    T3_KV,
    T3_POS,
    T3_SAMPLER,
    ChatterboxConfig,
)
from mstar.model.chatterbox.loader import iter_weights, resolve_snapshot  # noqa: E402
from mstar.model.chatterbox.submodules import (  # noqa: E402
    PREV_TOKEN,
    SPEECH_TOKENS,
    TEXT_INPUTS,
    BuiltinT3Voice,
    S3GenSubmodule,
    T3Submodule,
)
from mstar.model.submodule_base import ModelInputsFromEngine  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
torch.manual_seed(0)

REPOS = {"chatterbox": "ResembleAI/chatterbox", "turbo": "ResembleAI/chatterbox-turbo"}
TEXT = "The quick brown fox jumps over the lazy dog."
N_STEPS = 16


def _snapshot(variant: str) -> str:
    try:
        return resolve_snapshot(REPOS[variant])
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"checkpoint not available: {exc}")


class FakeGreedySampler:
    def sample(self, request_ids, logits):
        del request_ids
        return logits.argmax(dim=-1)


def _reference_t3(variant: str, snapshot: str) -> RefT3:
    config = ChatterboxConfig.from_variant(variant)
    if variant == "turbo":
        hp = RefT3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False
        ref = RefT3(hp)
    else:
        ref = RefT3()
    state = load_file(f"{snapshot}/{config.t3_weights}")
    ref.load_state_dict(state, strict=False)
    return ref.eval()


def _reference_greedy(ref: RefT3, cond: T3Cond, text_tokens: torch.Tensor, cfg_weight: float, n_steps: int):
    """``T3.inference`` / ``inference_turbo`` with argmax sampling."""
    hp = ref.hp
    if ref.is_gpt:
        text = text_tokens[None]
        bos = torch.full((1, 1), hp.start_speech_token, dtype=torch.long)
        embeds, _ = ref.prepare_input_embeds(t3_cond=cond, text_tokens=text, speech_tokens=bos, cfg_weight=0.0)
        out = ref.tfmr(inputs_embeds=embeds, use_cache=True)
        past = out.past_key_values
        logits = ref.speech_head(out[0][:, -1])
        tokens = []
        for _ in range(n_steps):
            tok = logits.argmax(-1, keepdim=True)
            tokens.append(int(tok))
            out = ref.tfmr(inputs_embeds=ref.speech_emb(tok), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = ref.speech_head(out[0][:, -1])
        return tokens

    cfg = cfg_weight > 0
    text = text_tokens[None]
    if cfg:
        text = torch.cat([text, text], dim=0)
    bos = torch.full((text.shape[0], 1), hp.start_speech_token, dtype=torch.long)
    embeds, _ = ref.prepare_input_embeds(t3_cond=cond, text_tokens=text, speech_tokens=bos, cfg_weight=cfg_weight)
    backend = T3HuggingfaceBackend(
        config=ref.cfg, llama=ref.tfmr, speech_enc=ref.speech_emb, speech_head=ref.speech_head,
    )
    bos_embed = ref.speech_emb(bos[:1]) + ref.speech_pos_emb.get_fixed_embedding(0)
    if cfg:
        bos_embed = torch.cat([bos_embed, bos_embed])
    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
    output = backend(inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True,
                     output_attentions=False, output_hidden_states=True, return_dict=True)
    past = output.past_key_values
    tokens = []
    for i in range(n_steps):
        step = output.logits[:, -1, :]
        if cfg:
            logits = step[0:1] + cfg_weight * (step[0:1] - step[1:2])
        else:
            logits = step[0:1]
        tok = logits.argmax(-1, keepdim=True)
        tokens.append(int(tok))
        nxt = ref.speech_emb(tok) + ref.speech_pos_emb.get_fixed_embedding(i + 1)
        if cfg:
            nxt = torch.cat([nxt, nxt])
        output = backend(inputs_embeds=nxt, past_key_values=past, output_attentions=False,
                         output_hidden_states=True, return_dict=True)
        past = output.past_key_values
    return tokens


def _mstar_greedy(variant: str, snapshot: str, text_tokens: torch.Tensor, cfg_weight: float,
                  exaggeration: float, n_steps: int):
    """The engine's calling sequence on ``T3Submodule`` with fake resources."""
    config = ChatterboxConfig.from_variant(variant)
    model = T3Model(config.t3)
    model.load_weights(iter_weights(f"{snapshot}/{config.t3_weights}"))
    model.eval()
    conds = torch.load(f"{snapshot}/conds.pt", map_location="cpu", weights_only=True)["t3"]
    builtin = BuiltinT3Voice(
        speaker_emb=conds["speaker_emb"].reshape(-1),
        prompt_tokens=conds["cond_prompt_speech_tokens"].reshape(-1),
    )
    sub = T3Submodule(model, config, builtin_voice=builtin)
    res = FakeT3Resources()
    res.bind(model, T3_ATTN, T3_KV, T3_POS)
    resources = {T3_ATTN: res.kv_attn, T3_KV: res.kv_attn, T3_POS: res.pos, T3_SAMPLER: FakeGreedySampler()}
    info = SimpleNamespace(
        request_id="r",
        step_metadata={"cfg_weight": cfg_weight, "exaggeration": exaggeration, "min_p": 0.05,
                       "max_new_tokens": 1000, "is_prefill": True},
        resource_configs={T3_SAMPLER: SimpleNamespace(temperature=0.0, ignore_eos=False)},
        max_tokens=4096, random_seed=0,
    )
    engine = ModelInputsFromEngine(request_ids=["r"], per_request_info={"r": info}, resources=resources)

    def run(walk, inputs):
        prepared = sub.prepare_inputs(walk, info, inputs)
        step = sub.declare_step(walk, ["r"], [prepared])
        res.plan([seg.span for seg in step.segments])  # label-major, as the KV plan packs them
        packed = sub.preprocess(walk, engine, [prepared])
        out = sub.forward(walk, engine, **packed)
        tok = out[SPEECH_TOKENS][0]
        sub.postprocess("r", info, out)
        return tok

    tokens = [int(run("prefill", {TEXT_INPUTS: [text_tokens]}))]
    prev = torch.tensor([tokens[-1]])
    for _ in range(n_steps - 1):
        prev = run("decode", {PREV_TOKEN: [prev]})
        tokens.append(int(prev))
    return tokens, sub


@pytest.mark.parametrize("variant,cfg_weight,exaggeration", [
    ("chatterbox", 0.5, 0.5),
    ("chatterbox", 0.0, 0.7),
    ("turbo", 0.0, 0.0),
])
def test_t3_greedy_tokens_match_reference(variant, cfg_weight, exaggeration):
    snapshot = _snapshot(variant)
    config = ChatterboxConfig.from_variant(variant)
    from mstar.model.chatterbox.components.text import ChatterboxTextTokenizer, TurboTextTokenizer

    tokenizer = (
        TurboTextTokenizer(snapshot) if variant == "turbo"
        else ChatterboxTextTokenizer(f"{snapshot}/tokenizer.json", 255, 0)
    )
    text_tokens = tokenizer(TEXT)

    ref = _reference_t3(variant, snapshot)
    conds = torch.load(f"{snapshot}/conds.pt", map_location="cpu", weights_only=True)["t3"]
    cond = T3Cond(
        speaker_emb=conds["speaker_emb"],
        cond_prompt_speech_tokens=conds["cond_prompt_speech_tokens"],
        emotion_adv=exaggeration * torch.ones(1, 1, 1),
    )
    with torch.no_grad():
        expected = _reference_greedy(ref, cond, text_tokens, cfg_weight, N_STEPS)
        got, sub = _mstar_greedy(variant, snapshot, text_tokens, cfg_weight, exaggeration, N_STEPS)
    print(f"[{variant} cfg={cfg_weight} exag={exaggeration}] ref={expected}\n mstar={got}")
    assert got == expected
    # bookkeeping the loop control relies on
    state = sub.request_state("r")
    assert state["generated"] == N_STEPS and state["speech_step"] == N_STEPS
    assert config.t3.cond_len == (34 if variant == "chatterbox" else 376)


def test_offline_s3gen_matches_reference_under_the_same_seed():
    snapshot = _snapshot("chatterbox")
    config = ChatterboxConfig.chatterbox()
    conds = torch.load(f"{snapshot}/conds.pt", map_location="cpu", weights_only=True)["gen"]
    tokens = conds["prompt_token"][0, :30].clone()

    ref = RefS3Gen()
    ref.load_state_dict(load_file(f"{snapshot}/s3gen.safetensors"), strict=False)
    ref.eval()
    torch.manual_seed(1234)
    with torch.no_grad():
        ref_wav, _ = ref.inference(speech_tokens=tokens, ref_dict=dict(conds), n_cfm_timesteps=10)

    s3gen = S3Gen(config.s3gen)
    s3gen.load_weights(iter_weights(f"{snapshot}/s3gen.safetensors"))
    s3gen.eval()
    builtin = ReferenceConditioning(
        prompt_tokens=conds["prompt_token"], prompt_feat=conds["prompt_feat"], embedding=conds["embedding"],
    )
    sub = S3GenSubmodule(s3gen, s3_tokenizer=None, config=config, builtin_voice=builtin, watermarker=None)
    pcm = sub.synthesize(tokens, builtin, n_timesteps=10, seed=1234, watermark=False)

    expected = (ref_wav[0].clamp(-1, 1) * 32767).to(torch.int16)
    assert pcm.shape == expected.shape == (30 * 960,)
    diff = (pcm.float() - expected.float()).abs().max().item()
    snr = 10 * torch.log10(expected.float().pow(2).mean() / F.mse_loss(pcm.float(), expected.float()).clamp_min(1e-12))
    print(f"[s3gen offline] max |pcm diff| = {diff}, SNR = {snr:.1f} dB")
    assert diff <= 1.0 and snr > 60
