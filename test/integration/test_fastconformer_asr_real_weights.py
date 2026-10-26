"""Real-weight parity of the transducer ASR models (Parakeet-TDT, Nemotron-3.5
ASR) against HF transformers' ports, on the repo's speech clip.

Never downloads: skips unless CUDA, the checkpoint in a local Hugging Face
cache, and a transformers with the model's class (5.19+). Both sides run in
fp32 and decode greedily, so the token ids must match exactly. Serving runs
the encoder in bf16; ``benchmark/asr_eval.py`` covers that end to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

CLIP = Path(__file__).resolve().parents[1] / "qwen3-omni" / "audio.wav"
# registry key -> (HF repo, transformers class)
CASES = {
    "parakeet": ("nvidia/parakeet-tdt-0.6b-v3", "ParakeetForTDT"),
    "nemotron_asr": ("nvidia/nemotron-3.5-asr-streaming-0.6b", "Nemotron3_5AsrForRNNT"),
}
# Nemotron's right context in frames: M*'s default (the widest), set on HF's side too
NEMOTRON_LOOKAHEAD = 13


def _cached_snapshot(repo: str) -> Path | None:
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        snapshots = root / f"models--{repo.replace('/', '--')}" / "snapshots"
        for snapshot in sorted(snapshots.iterdir()) if snapshots.is_dir() else []:
            if (snapshot / "config.json").is_file() and any(snapshot.glob("*.safetensors")):
                return snapshot
    return None


def _relative_error(ours: torch.Tensor, theirs: torch.Tensor) -> float:
    return float((ours - theirs).abs().max() / theirs.abs().max())


def _frame_by_frame(head, states: torch.Tensor, prompt: torch.Tensor | None) -> list[int]:
    """Greedy decoding one frame or token at a time, as NeMo writes it: the
    yardstick for the batched loop, which scores several frames per step."""
    import torch.nn.functional as F

    cfg = head.config
    frames = head.project_frames(states[None], prompt)[0]
    h = c = frames.new_zeros(1, cfg.num_layers, cfg.hidden_size)
    out, h, c = head.decoder(torch.tensor([cfg.start_token_id], device=frames.device), h, c)
    t, tokens, at_frame = 0, [], 0
    while t < frames.shape[0]:
        logits = head.joint.head(F.relu(frames[t] + out[0]))
        token = int(logits[:cfg.vocab_size].argmax())
        if cfg.durations:
            step = cfg.durations[int(logits[cfg.vocab_size:].argmax())]
            step = 1 if token == cfg.blank_token_id and step == 0 else step
        else:
            step = 1 if token == cfg.blank_token_id else 0
        at_frame = at_frame + 1 if token != cfg.blank_token_id and step == 0 else 0
        if at_frame >= cfg.max_symbols_per_step:
            step, at_frame = 1, 0
        if token != cfg.blank_token_id:
            tokens.append(token)
            out, h, c = head.decoder(torch.tensor([token], device=frames.device), h, c)
        t += step
    return tokens


def _greedy(head, states: torch.Tensor, prompt: torch.Tensor | None, step=None) -> list[int]:
    """One clip through ``advance`` a chunk at a time from the start state, as
    the decode loop runs it, in full fp32 as the node does."""
    from mstar.model.components.transducer import CHUNK
    from mstar.utils.precision import ieee_fp32_matmul

    with ieee_fp32_matmul():
        frames = head.project_frames(states, prompt)
        lengths = torch.tensor([frames.shape[1]], device=frames.device)
        state = {name: start[None].clone() for name, start in head.initial_state().items()}
        tokens = []
        for _ in range(head.config.decode_iterations(frames.shape[1])):
            chunk, active = head.advance(frames, lengths, state, CHUNK, step)
            tokens += [token for token in chunk[0].tolist() if token >= 0]
            if not active.item():
                break
    return tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name", list(CASES))
def test_matches_hf(name, monkeypatch):
    """Stage by stage, so a difference shows where it starts: the features and
    the encoder output agree to fp32 noise, and both from the reference's own
    encoder output and from ours the greedy decode emits the reference's tokens.

    HF's loop has no cap on tokens per frame. NeMo's moves on after
    ``max_symbols_per_step``, and we follow NeMo; Parakeet, hearing a clip
    outside its languages, reaches that cap here, so the comparison with HF
    lifts it. At the shipped cap, the batched loop, eager and compiled, emits
    what one-frame-at-a-time decoding does."""
    transformers = pytest.importorskip("transformers")
    repo, hf_class = CASES[name]
    if not hasattr(transformers, hf_class):
        pytest.skip(f"transformers {transformers.__version__} has no {hf_class}")
    snapshot = _cached_snapshot(repo)
    if snapshot is None:
        pytest.skip(f"{repo} is not in the local Hugging Face cache")

    from mstar.model.components.audio_features import SAMPLE_RATE, load_speech_file
    from mstar.model.parakeet.parakeet_model import ParakeetModel

    nemotron = name == "nemotron_asr"
    audio = load_speech_file(str(CLIP), SAMPLE_RATE, 60.0)
    model = ParakeetModel(str(snapshot), lookahead=NEMOTRON_LOOKAHEAD if nemotron else None)
    # no autocast dtype: the encoder stays in fp32, like the reference
    encoder = model.get_submodule("audio_encoder", "cuda")
    node = model.get_submodule("transducer", "cuda")
    head, compiled_step = node.head, node._step
    prompt = torch.tensor([model.config.prompt_id("auto")], device="cuda") if nemotron else None

    processor = transformers.AutoProcessor.from_pretrained(snapshot)
    reference = getattr(transformers, hf_class).from_pretrained(snapshot, dtype=torch.float32).cuda().eval()
    inputs = processor(audio.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    if nemotron:
        inputs["num_lookahead_tokens"] = NEMOTRON_LOOKAHEAD
    inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
    blank = model.config.transducer.blank_token_id

    with torch.no_grad():
        features = encoder.log_mel(audio.cuda())
        hf_features = inputs["input_features"][0, :int(inputs["attention_mask"].sum())]
        assert features.shape == hf_features.shape
        assert _relative_error(features, hf_features) < 1e-4

        # the encoder output the reference decodes from, as generate runs it
        captured = []
        hook = reference.encoder.register_forward_hook(lambda _m, _i, out: captured.append(out.last_hidden_state))
        sequence = reference.generate(**inputs).sequences[0].tolist()
        hook.remove()
        hf_states = captured[0]
        theirs = [token for token in sequence if token != blank]

        states, _ = encoder.encoder(hf_features[None])
        assert states.shape == hf_states.shape
        # fp32, but 24 layers of differently ordered GEMM accumulation
        assert _relative_error(states, hf_states) < 5e-3
        our_states = encoder.encoder(features[None])[0].float()
        capped = _greedy(head, our_states, prompt)
        reference = _frame_by_frame(head, our_states[0], prompt)
        compiled = _greedy(head, our_states, prompt, compiled_step)
        monkeypatch.setattr(head.config, "max_symbols_per_step", 10**6)
        from_their_states = _greedy(head, hf_states.float(), prompt)
        ours = _greedy(head, our_states, prompt)
    assert len(theirs) > 20, "the clip decoded to almost nothing; the comparison would be vacuous"
    assert from_their_states == theirs, (processor.decode(from_their_states), processor.decode(theirs))
    assert ours == theirs, (processor.decode(ours), processor.decode(theirs))
    assert capped == reference == compiled, (processor.decode(capped), processor.decode(reference))
