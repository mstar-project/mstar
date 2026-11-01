#!/usr/bin/env python3
"""Record NVIDIA reference fixtures with the pinned PersonaPlex source.

Modes:
  trace     LMGen integer trace with tiny random weights.
  manifest  state-dict keys/shapes of the full-size LMModel, built on meta.
  lm        tiny random fp32 LMModel: its weights plus every forward's inputs,
            hidden states, text/depth logits and greedy tokens.
  mimi      tiny random MimiModel run per 1920-sample frame in streaming mode,
            as offline.py does: latents, codes, tie margins and decoded PCM.
  offline   the whole offline.py pipeline (warmup, .pt voice, persona, user WAV,
            per-frame encode/step/decode, trim) on the tiny LM and Mimi, greedy.

No mstar imports, checkpoints or network access at runtime. Run in an isolated
torch<2.5 environment with the pinned NVIDIA package installed. Setup and replay
commands: test/modular/fixtures/personaplex/README.rst.

The observer subclasses LMGen only to record inputs/outputs around super calls.
It never substitutes cache logic, sampling, transformer or depformer execution.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

PIN = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"


def _verify_source(source: Path) -> tuple[Path, dict[str, str]]:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        raise RuntimeError(f"reference HEAD {head} differs from required {PIN}")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no", "--", "moshi"], text=True
    )
    if status:
        raise RuntimeError("reference moshi source has tracked modifications")
    package = source / "moshi" / "moshi"
    hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    return package, hashes


def _json_tensor(value):
    return None if value is None else value.detach().cpu().tolist()


def _record_case(lm_model, lm_module, batch: int, user_frames: int, mode: str) -> dict:
    class ObservedLMGen(lm_module.LMGen):
        def __init__(self):
            super().__init__(
                lm_model, device="cpu", use_sampling=False, return_logits=True, check=True,
                audio_silence_frame_cnt=6, sample_rate=24000, frame_rate=12.5,
            )
            self.events = []
            self.phase = "init"
            self.sampled_audio = None

        def snapshot(self):
            state = self._streaming_state
            return {
                "offset": state.offset,
                "cache": _json_tensor(state.cache),
                "provided": _json_tensor(state.provided),
            }

        def prepare_step_input(self, input_tokens=None, moshi_tokens=None, text_token=None):
            prepared = super().prepare_step_input(input_tokens, moshi_tokens, text_token)
            # step_embeddings supplies batch-1 dummies; NVIDIA broadcasts them.
            text = None if text_token is None else torch.as_tensor(text_token).expand(batch)
            event = {
                "op": "prepare", "phase": self.phase,
                "user": _json_tensor(None if input_tokens is None else input_tokens.expand(batch, -1, -1)),
                "agent": _json_tensor(None if moshi_tokens is None else moshi_tokens.expand(batch, -1, -1)),
                "text": _json_tensor(text), "forward": prepared is not None,
                "state": self.snapshot(),
            }
            if prepared is not None:
                input_, provided, target, _, _ = prepared
                event.update(input=_json_tensor(input_), target=_json_tensor(target), mask=_json_tensor(provided))
            self.events.append(event)
            return prepared

        def depformer_step(self, *args):
            result = super().depformer_step(*args)
            self.sampled_audio = result[0].detach().clone()
            return result

        def process_transformer_output(self, transformer_out, text_logits, *args):
            sampled_text = text_logits.float().argmax(dim=-1)[:, 0, 0]
            result = super().process_transformer_output(transformer_out, text_logits, *args)
            self.events.append({
                "op": "commit", "phase": self.phase,
                "sampled_text": _json_tensor(sampled_text), "sampled_audio": _json_tensor(self.sampled_audio),
                "output": _json_tensor(result[0]), "state": self.snapshot(),
            })
            return result

    gen = ObservedLMGen()
    silence = torch.tensor(lm_module.SILENCE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    sine = torch.tensor(lm_module.SINE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
    pad = torch.full((batch,), 3, dtype=torch.long)

    def voice_frame(index):
        return (torch.arange(batch * 8).view(batch, 8, 1) + 100 + index * 11) % lm_model.card

    voice_frames = 4
    # Build a .pt-style preset using actual reference embed_codes outputs and
    # the cache saved by the raw token path. No gated voice asset is needed.
    embeddings, saved_cache = [], None
    if mode == "embedding_replay":
        with gen.streaming(batch):
            gen.phase = "preset_voice"
            for index in range(voice_frames):
                result = gen.step(sine, voice_frame(index), pad, return_embeddings=index > 0)
                if index > 0:
                    embeddings.append(result[1].detach().clone())
            saved_cache = gen._streaming_state.cache.detach().clone()
        gen.events.clear()

    with gen.streaming(batch):
        gen.events.append({"op": "init", "phase": "init", "state": gen.snapshot()})
        if mode == "token_priming":
            for index in range(voice_frames):
                gen.phase = f"voice_{index}"
                gen.step(sine, voice_frame(index), pad)
        else:
            gen.voice_prompt_embeddings = torch.stack(embeddings)
            gen.voice_prompt_cache = saved_cache
            # This method itself calls step_embeddings and restores cache.copy_.
            # Advance its yield points individually to observe every embedding.
            gen.phase = "embedding_0"
            iterator = gen._step_voice_prompt_core(None)
            next(iterator)
            for index in range(len(embeddings)):
                gen.phase = f"embedding_{index}"
                try:
                    next(iterator)
                except StopIteration:
                    if index != len(embeddings) - 1:
                        raise
            gen.events.append({
                "op": "load", "phase": "voice_restore", "cache": _json_tensor(saved_cache),
                "state": gen.snapshot(),
            })
        for block in ("silence_before", "text", "silence_after"):
            texts = [3] * 6 if block != "text" else [7, 11, 19]
            for index, token in enumerate(texts):
                gen.phase = f"{block}_{index}"
                gen.step(sine, silence, torch.full((batch,), token, dtype=torch.long))
        for index in range(user_frames):
            gen.phase = f"user_{index}"
            gen.step(voice_frame(index + voice_frames))
    return {"batch_size": batch, "mode": mode, "events": gen.events}


def _lm_config(lm_kwargs: dict, **shape) -> dict:
    config = dict(lm_kwargs)
    config.update(dep_q=16, **shape)
    return config


def _record_manifest(lm_module, lm_kwargs: dict) -> dict:
    # get_moshi_lm sets dep_q=16 and builds on meta before loading ([PP] models/loaders.py:187, 197-198).
    config = _lm_config(lm_kwargs)
    model = lm_module.LMModel(**config, device="meta", dtype=torch.bfloat16)
    params = {name: list(p.shape) for name, p in model.named_parameters()}
    buffers = {name: list(b.shape) for name, b in model.named_buffers()}
    return {
        "lm_kwargs": config, "dtype": "bfloat16", "parameters": params, "buffers": buffers,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }


def _record_lm(lm_module, lm_kwargs: dict, user_frames: int) -> tuple[dict, dict]:
    config = _lm_config(
        lm_kwargs, dim=64, num_heads=4, num_layers=2, hidden_scale=4.125,
        depformer_dim=32, depformer_num_heads=2, depformer_num_layers=2,
        depformer_dim_feedforward=int(4.125 * 32),
    )
    model = lm_module.LMModel(**config, device="cpu", dtype=torch.float32).eval()
    with torch.no_grad():
        # Norm scales start at 1; randomise them so a dropped or misrouted alpha shows up.
        for name, param in model.named_parameters():
            if name.endswith(".alpha"):
                param.copy_(1 + 0.25 * torch.randn_like(param))

    class LogitObserver(lm_module.LMGen):
        def __init__(self):
            super().__init__(
                model, device="cpu", use_sampling=False, return_logits=True, check=True,
                audio_silence_frame_cnt=6, sample_rate=24000, frame_rate=12.5,
            )
            self.records, self.phase, self.pending = [], "init", None

        def prepare_step_input(self, input_tokens=None, moshi_tokens=None, text_token=None):
            prepared = super().prepare_step_input(input_tokens, moshi_tokens, text_token)
            if prepared is not None:
                self.pending = {"phase": self.phase, "input": prepared[0].clone()}
            return prepared

        def depformer_step(self, text_token, transformer_out, audio_tokens, audio_provided):
            tokens, logits = super().depformer_step(text_token, transformer_out, audio_tokens, audio_provided)
            self.pending.update(
                depth_text=text_token.clone(), depth_target=audio_tokens.clone(),
                depth_provided=audio_provided.clone(), audio_tokens=tokens.clone(), audio_logits=logits.clone(),
            )
            return tokens, logits

        def process_transformer_output(self, transformer_out, text_logits, *args):
            self.pending.update(transformer_out=transformer_out.clone(), text_logits=text_logits.clone())
            result = super().process_transformer_output(transformer_out, text_logits, *args)
            self.pending["output"] = None if result[0] is None else result[0].clone()
            self.records.append(self.pending)
            self.pending = None
            return result

    def run_case(batch: int, mode: str) -> dict:
        gen = LogitObserver()
        silence = torch.tensor(lm_module.SILENCE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
        sine = torch.tensor(lm_module.SINE_TOKENS).view(1, 8, 1).expand(batch, -1, -1)
        pad = torch.full((batch,), 3, dtype=torch.long)

        def voice_frame(index):
            return (torch.arange(batch * 8).view(batch, 8, 1) * 37 + 100 + index * 11) % model.card

        preset = None
        if mode == "embedding_replay":
            with gen.streaming(batch):
                gen.phase = "preset_voice"
                inputs, embeddings = [], []
                for index in range(4):
                    result = gen.step(sine, voice_frame(index), pad, return_embeddings=index > 0)
                    if index > 0:
                        embeddings.append(result[1].clone())
                        inputs.append(gen.records[-1]["input"])
                preset = {"inputs": inputs, "embeddings": torch.stack(embeddings),
                          "cache": gen._streaming_state.cache.clone()}
            gen.records.clear()
        with gen.streaming(batch):
            if mode == "token_priming":
                for index in range(4):
                    gen.phase = f"voice_{index}"
                    gen.step(sine, voice_frame(index), pad)
            else:
                gen.voice_prompt_embeddings = preset["embeddings"]
                gen.voice_prompt_cache = preset["cache"]
                for index, _ in enumerate(gen._step_voice_prompt_core(None)):
                    gen.phase = f"embedding_{index}"
            for block in ("silence_before", "text", "silence_after"):
                texts = [3] * 6 if block != "text" else [7, 11, 19]
                for index, token in enumerate(texts):
                    gen.phase = f"{block}_{index}"
                    gen.step(sine, silence, torch.full((batch,), token, dtype=torch.long))
            for index in range(user_frames):
                gen.phase = f"user_{index}"
                gen.step(voice_frame(index + 4))
        return {"batch_size": batch, "mode": mode, "preset": preset, "records": gen.records}

    cases = [run_case(batch, mode) for batch in (1, 2) for mode in ("token_priming", "embedding_replay")]
    return config, {"state_dict": {k: v.clone() for k, v in model.state_dict().items()}, "cases": cases}


MIMI_FRAME = 1920


def _mimi_kwargs(loaders) -> dict:
    """Reference Mimi kwargs ([PP] models/loaders.py:48-90) and a shrunken copy.

    The copy keeps the strides, kernels, codebook count/size, causal padding,
    RoPE and 250-step attention context; only channel widths and depth shrink.
    """
    real = {"seanet": dict(loaders._seanet_kwargs), "transformer": dict(loaders._transformer_kwargs),
            "quantizer": dict(loaders._quantizer_kwargs)}
    seanet = dict(real["seanet"], dimension=32, n_filters=8)
    tiny = {
        "seanet": seanet,
        "transformer": dict(real["transformer"], d_model=32, num_heads=2, num_layers=2, dim_feedforward=64,
                            input_dimension=32, output_dimensions=[32]),
        "quantizer": dict(real["quantizer"], dimension=16, input_dimension=32, output_dimension=32),
    }
    return {"real": real, "tiny": tiny}


def _build_mimi(kwargs: dict):
    from moshi.models import compression
    from moshi.modules import SEANetDecoder, SEANetEncoder, transformer
    from moshi.quantization import SplitResidualVectorQuantizer

    encoder = SEANetEncoder(**kwargs["seanet"])
    return compression.MimiModel(
        encoder, SEANetDecoder(**kwargs["seanet"]), SplitResidualVectorQuantizer(**kwargs["quantizer"]),
        channels=1, sample_rate=24000, frame_rate=12.5, encoder_frame_rate=24000 / encoder.hop_length,
        causal=True, resample_method="conv",
        encoder_transformer=transformer.ProjectedTransformer(device="cpu", **kwargs["transformer"]),
        decoder_transformer=transformer.ProjectedTransformer(device="cpu", **kwargs["transformer"]),
    ).eval()


def _mimi_audio(frames: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(frames * MIMI_FRAME) / 24000
    x = 0.2 * torch.sin(2 * torch.pi * 220 * t) + 0.1 * torch.sin(2 * torch.pi * 1375 * t * (1 + t))
    return (x + 0.05 * torch.randn(x.shape, generator=g)).view(1, 1, -1)


def _code_margins(mimi, latent: torch.Tensor) -> torch.Tensor:
    """Best-vs-second codebook distance per (codebook, frame) along NVIDIA's residual path."""
    margins = []
    for rvq, count in ((mimi.quantizer.rvq_first, 1), (mimi.quantizer.rvq_rest, mimi.num_codebooks - 1)):
        residual = rvq.input_proj(latent) if rvq.input_proj is not None else latent
        for layer in rvq.vq.layers[:count]:
            embedding = layer._codebook.embedding
            dists = torch.cdist(residual[0].T[None], embedding[None])[0]
            top = dists.topk(2, largest=False).values
            margins.append(top[:, 1] - top[:, 0])
            residual = residual - embedding[dists.argmin(-1)].T[None]
    return torch.stack(margins)


def _record_mimi() -> dict:
    from moshi.models import loaders

    kwargs = _mimi_kwargs(loaders)
    mimi = _build_mimi(kwargs["tiny"])
    mimi.set_num_codebooks(8)  # [PP] models/loaders.py:162
    # Codebooks start at zero; spread them over the latents of real-ish audio so codes vary.
    g = torch.Generator().manual_seed(5)
    latent = mimi._encode_to_unquantized_latent(_mimi_audio(160, seed=3))
    for rvq in (mimi.quantizer.rvq_first, mimi.quantizer.rvq_rest):
        z = rvq.input_proj(latent) if rvq.input_proj is not None else latent
        scale = float(z.std())
        for index, layer in enumerate(rvq.vq.layers):
            book = layer._codebook
            picks = torch.randint(0, z.shape[-1], (book.codebook_size,), generator=g)
            sample = z[0, :, picks].T + 0.3 * scale * torch.randn(book.codebook_size, book.dim, generator=g)
            book.embedding_sum.copy_(sample * 0.5**index)
            book.cluster_usage.fill_(1.0)
            book._initialized.fill_(1.0)
            book._embedding = None

    def stream_encode(audio):
        codes, latents = [], []
        with mimi.streaming(1):
            for i in range(audio.shape[-1] // MIMI_FRAME):
                chunk = audio[..., i * MIMI_FRAME:(i + 1) * MIMI_FRAME]
                latents.append(mimi._encode_to_unquantized_latent(chunk))
        with mimi.streaming(1):
            for i in range(audio.shape[-1] // MIMI_FRAME):
                codes.append(mimi.encode(audio[..., i * MIMI_FRAME:(i + 1) * MIMI_FRAME]))
        return torch.cat(latents, -1), torch.cat(codes, -1)

    def stream_decode(codes):
        with mimi.streaming(1):
            return torch.cat([mimi.decode(codes[..., i:i + 1]) for i in range(codes.shape[-1])], -1)

    cases = []
    for frames in (7, 160):
        audio = _mimi_audio(frames, seed=frames)
        latent, codes = stream_encode(audio)
        random_codes = torch.randint(0, mimi.cardinality, (1, 8, frames), generator=g)
        cases.append({
            "frames": frames, "audio": audio, "latent": latent, "codes": codes,
            "margins": _code_margins(mimi, latent), "random_codes": random_codes,
            "pcm_from_codes": stream_decode(codes), "pcm_from_random_codes": stream_decode(random_codes),
        })
    # offline.py: one streaming instance, 4 zero-frame warmup steps, reset, then
    # per-frame encode and decode interleaved ([PP] offline.py:93-111, 222-255).
    frames = 40
    audio = _mimi_audio(frames, seed=9)
    agent = torch.randint(0, mimi.cardinality, (1, 8, frames), generator=g)
    mimi.streaming_forever(1)
    for _ in range(4):
        mimi.decode(mimi.encode(torch.zeros(1, 1, MIMI_FRAME)))
    mimi.reset_streaming()
    encoded, decoded = [], []
    for i in range(frames):
        encoded.append(mimi.encode(audio[..., i * MIMI_FRAME:(i + 1) * MIMI_FRAME]))
        decoded.append(mimi.decode(agent[..., i:i + 1]))
    interleaved = {"audio": audio, "agent_codes": agent, "codes": torch.cat(encoded, -1),
                   "pcm": torch.cat(decoded, -1)}
    # Full-size key/shape manifest (no weights) so the conversion's coverage is
    # checked against the real architecture without the gated checkpoint.
    full = _build_mimi(kwargs["real"])
    real_manifest = {name: list(t.shape) for name, t in full.state_dict().items()}
    return {"mimi_kwargs": kwargs, "num_codebooks": 8, "state_dict": dict(mimi.state_dict()),
            "real_state_dict_shapes": real_manifest, "cases": cases, "interleaved": interleaved}


def _record_offline(lm_module, lm_kwargs: dict, scratch: Path) -> dict:
    """Run NVIDIA's offline.py steps 4-12 ([PP] offline.py:200-307) without files or the tokenizer.

    Same tiny LM (with randomised norm scales) and Mimi as the lm/mimi modes.
    The voice preset is made by NVIDIA's own embed_codes on a token-primed pass
    and loaded with load_voice_prompt_embeddings; the persona is given as token
    ids; the user audio is stereo and not a multiple of 1920 samples.
    """
    import copy

    import numpy as np
    from moshi import offline

    config = _lm_config(
        lm_kwargs, dim=64, num_heads=4, num_layers=2, hidden_scale=4.125,
        depformer_dim=32, depformer_num_heads=2, depformer_num_layers=2,
        depformer_dim_feedforward=int(4.125 * 32),
    )
    lm_model = lm_module.LMModel(**config, device="cpu", dtype=torch.float32).eval()
    for name, param in lm_model.named_parameters():
        if name.endswith(".alpha"):
            param.copy_(1 + 0.25 * torch.randn_like(param))
    mimi_data = _record_mimi()
    from moshi.models import loaders

    mimi = _build_mimi(_mimi_kwargs(loaders)["tiny"])
    mimi.load_state_dict(mimi_data["state_dict"])
    mimi.set_num_codebooks(8)
    other_mimi = copy.deepcopy(mimi)

    # .pt preset from NVIDIA's embed_codes, as save_voice_prompt_embeddings does ([PP] models/lm.py:977-983).
    maker = lm_module.LMGen(lm_model, device="cpu", use_sampling=False, sample_rate=24000, frame_rate=12.5)
    sine = torch.tensor(lm_module.SINE_TOKENS).view(1, 8, 1)
    embeddings = []
    with maker.streaming(1):
        for index in range(4):
            voice = (torch.arange(8).view(1, 8, 1) * 37 + 100 + index * 11) % lm_model.card
            out = maker.step(sine, voice, 3, return_embeddings=index > 0)
            if index > 0:
                embeddings.append(out[1])
        preset = {"embeddings": torch.stack(embeddings), "cache": maker._streaming_state.cache.clone()}
    preset_path = scratch / "voice.pt"
    torch.save(preset, preset_path)

    persona = [7, 11, 19, 23]
    frames, extra = 130, 777
    t = np.arange(frames * MIMI_FRAME + extra) / 24000
    user_audio = np.stack([0.3 * np.sin(2 * np.pi * 330 * t) * np.sin(2 * np.pi * 0.7 * t),
                           0.3 * np.sin(2 * np.pi * 120 * t)]).astype(np.float32)

    # [PP] offline.py:207-221
    lm_gen = lm_module.LMGen(
        lm_model, audio_silence_frame_cnt=int(0.5 * mimi.frame_rate), sample_rate=mimi.sample_rate,
        device="cpu", frame_rate=mimi.frame_rate, use_sampling=False,
    )
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    offline.warmup(mimi, other_mimi, lm_gen, "cpu", frame_size)  # [PP] offline.py:226
    lm_gen.load_voice_prompt_embeddings(str(preset_path))  # [PP] offline.py:231-233
    lm_gen.text_prompt_tokens = persona  # in place of sentencepiece.encode(...) at offline.py:241-243
    mimi.reset_streaming()  # [PP] offline.py:251-255
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()
    lm_gen.step_system_prompts(mimi)
    mimi.reset_streaming()
    generated, text_tokens, user_codes = [], [], []
    for encoded in lm_module.encode_from_sphn(  # [PP] offline.py:268-296
        mimi, lm_module._iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True), max_batch=1,
    ):
        for c in range(encoded.shape[-1]):
            step_in = encoded[:, :, c : c + 1]
            user_codes.append(step_in.clone())
            tokens = lm_gen.step(step_in)
            if tokens is None:
                continue
            generated.append(offline.decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens))
            text_tokens.append(int(tokens[0, 0, 0]))
    pcm = np.concatenate(generated, axis=-1)  # [PP] offline.py:298-307
    total = user_audio.shape[-1]
    pcm = pcm[:total] if pcm.shape[-1] > total else np.concatenate([pcm, np.zeros(total - pcm.shape[-1], pcm.dtype)])
    return {
        "lm_kwargs": config, "lm_state_dict": dict(lm_model.state_dict()),
        "mimi_kwargs": _mimi_kwargs(loaders), "mimi_state_dict": dict(mimi.state_dict()),
        "voice_preset": preset, "persona_tokens": persona, "user_audio": torch.from_numpy(user_audio),
        "user_codes": torch.cat(user_codes, -1), "text_tokens": text_tokens, "pcm": torch.from_numpy(pcm),
    }


def _content_hash(fixture: dict) -> str:
    # Invocation paths remain useful provenance, but do not define content identity.
    content = {key: value for key, value in fixture.items() if key != "argv"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tensor_digest(value) -> object:
    """Order-stable SHA256 over nested tensors, for the binary LM fixture's provenance."""
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        return [str(t.dtype), list(t.shape), hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {k: _tensor_digest(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_tensor_digest(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="clean NVIDIA/personaplex checkout at the pin")
    parser.add_argument("--mode", choices=("trace", "manifest", "lm", "mimi", "offline"), default="trace")
    parser.add_argument("--out", type=Path, required=True, help="output directory for the fixture")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--user-frames", type=int, default=20)
    parser.add_argument(
        "--allow-unsupported-torch", action="store_true",
        help="record a supplementary same-runtime comparison outside NVIDIA's torch range; labels provenance",
    )
    args = parser.parse_args()
    if args.user_frames < 20:
        parser.error("trace acceptance requires at least 20 user frames")
    major, minor = map(int, torch.__version__.split(".")[:2])
    supported_torch = major == 2 and 2 <= minor < 5
    if not supported_torch and not args.allow_unsupported_torch:
        parser.error(f"NVIDIA's reference requires torch>=2.2,<2.5; found {torch.__version__}")
    package, hashes = _verify_source(args.source.resolve())
    sys.path.insert(0, str(package.parent))
    from moshi.models import lm as lm_module
    from moshi.models.loaders import _lm_kwargs
    from moshi.utils.compile import no_compile

    if Path(inspect.getfile(lm_module.LMGen)).resolve() != package / "models" / "lm.py":
        raise RuntimeError("LMGen was imported from a different checkout")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    provenance = {
        "source": "NVIDIA/personaplex", "source_sha": PIN, "source_hashes": hashes,
        "torch_version": str(torch.__version__), "python_version": sys.version, "seed": args.seed,
        "torch_supported_by_reference": supported_torch,
        "platform": platform.platform(), "machine": platform.machine(),
        "torch_build_config": torch.__config__.show(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "num_threads": torch.get_num_threads(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "manifest":
        fixture = {"schema_version": 1, **provenance, "argv": sys.argv[1:], **_record_manifest(lm_module, _lm_kwargs)}
        fixture["content_sha256"] = _content_hash(fixture)
        output = args.out / "reference_lm_manifest.json"
        output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
        print(f"Recorded {len(fixture['parameters'])} parameters, {fixture['num_parameters']} values to {output}")
        print(f"source={PIN}; torch={torch.__version__}; content_sha256={fixture['content_sha256']}")
        return
    if args.mode == "offline":
        import tempfile

        with torch.no_grad(), no_compile(), tempfile.TemporaryDirectory() as scratch:
            data = _record_offline(lm_module, _lm_kwargs, Path(scratch))
        fixture = {"schema_version": 1, **provenance, "argv": sys.argv[1:], "use_sampling": False, "device": "cpu",
                   "dtype": "float32", **data}
        digest = {key: _tensor_digest(value) for key, value in fixture.items() if key != "argv"}
        fixture["content_sha256"] = hashlib.sha256(
            json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output = args.out / "reference_offline.pt"
        torch.save(fixture, output)
        print(f"Recorded offline replay: {len(data['text_tokens'])} frames, {data['pcm'].shape[-1]} samples")
        print(f"source={PIN}; torch={torch.__version__}; bytes={output.stat().st_size}")
        print(f"content_sha256={fixture['content_sha256']}")
        return
    if args.mode == "mimi":
        with torch.no_grad(), no_compile():
            data = _record_mimi()
        fixture = {"schema_version": 1, **provenance, "argv": sys.argv[1:], "device": "cpu", "dtype": "float32",
                   **data}
        digest = {key: _tensor_digest(value) for key, value in fixture.items() if key != "argv"}
        fixture["content_sha256"] = hashlib.sha256(
            json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output = args.out / "reference_mimi.pt"
        torch.save(fixture, output)
        print(f"Recorded Mimi cases {[c['frames'] for c in data['cases']]} + interleaved to {output}")
        print(f"source={PIN}; torch={torch.__version__}; bytes={output.stat().st_size}")
        print(f"content_sha256={fixture['content_sha256']}")
        return
    if args.mode == "lm":
        with torch.no_grad(), no_compile():
            config, data = _record_lm(lm_module, _lm_kwargs, args.user_frames)
        fixture = {
            "schema_version": 1, **provenance, "argv": sys.argv[1:], "use_sampling": False, "device": "cpu",
            "dtype": "float32", "lm_kwargs": config, "user_frames": args.user_frames, **data,
        }
        digest = {key: _tensor_digest(value) for key, value in fixture.items() if key != "argv"}
        fixture["content_sha256"] = hashlib.sha256(
            json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output = args.out / "reference_lm.pt"
        torch.save(fixture, output)
        forwards = sum(len(c["records"]) for c in data["cases"])
        print(f"Recorded {len(data['cases'])} LMGen cases, {forwards} forwards to {output}")
        print(f"source={PIN}; torch={torch.__version__}; bytes={output.stat().st_size}")
        print(f"content_sha256={fixture['content_sha256']}")
        return
    # Preserve 17 channels, vocabularies, delays and per-step depth weights;
    # only transformer dimensions/layer counts shrink for CPU execution.
    config = dict(_lm_kwargs)
    config.update(
        dep_q=16, dim=32, num_heads=4, num_layers=1, hidden_scale=4.125,
        depformer_dim=16, depformer_num_heads=2, depformer_num_layers=1,
        depformer_dim_feedforward=66,
    )
    lm_model = lm_module.LMModel(**config, device="cpu", dtype=torch.float32).eval()
    with torch.no_grad(), no_compile():
        cases = [
            _record_case(lm_model, lm_module, batch, args.user_frames, mode)
            for batch in (1, 2) for mode in ("token_priming", "embedding_replay")
        ]
    fixture = {
        "schema_version": 2, **provenance,
        "argv": sys.argv[1:], "seed": args.seed, "use_sampling": False, "device": "cpu",
        "lm_kwargs": config, "user_frames": args.user_frames, "cases": cases,
    }
    fixture["content_sha256"] = _content_hash(fixture)
    output = args.out / "reference_trace.json"
    output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"Recorded {len(cases)} LMGen cases, {sum(len(c['events']) for c in cases)} events to {output}")
    print(f"source={PIN}; torch={torch.__version__}; bytes={output.stat().st_size}")
    print(f"content_sha256={fixture['content_sha256']}")


if __name__ == "__main__":
    main()
