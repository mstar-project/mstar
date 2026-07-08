"""Token-level parity check: mstar Whisper vs HF WhisperForConditionalGeneration.

Runs the mstar encoder + decoder submodules with a mock cache handle
(standalone, no engine) in a greedy loop and compares the generated
token sequence against HF's ``generate`` on the same audio.

Usage:
    python test/scratch/compare_whisper_decoder.py [audio.wav]
"""
import sys

import torch

AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else "test/qwen3-omni/audio.wav"
MAX_NEW_TOKENS = 60


class MockCacheHandle:
    """Minimal stand-in for BatchedCacheManager: per-layer growing KV +
    causal SDPA. Single request only."""

    def __init__(self, num_layers: int):
        self.kv = [None] * num_layers
        self.layer_idx = 0

    def set_layer_idx(self, i):
        self.layer_idx = i

    def run_attention(self, q, k, v, layer_idx=None):
        i = self.layer_idx if layer_idx is None else layer_idx
        if self.kv[i] is None:
            self.kv[i] = (k, v)
        else:
            pk, pv = self.kv[i]
            self.kv[i] = (torch.cat([pk, k]), torch.cat([pv, v]))
        fk, fv = self.kv[i]
        is_prefill = fk.shape[0] == q.shape[0]
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), fk.transpose(0, 1), fv.transpose(0, 1),
            is_causal=is_prefill,
        )
        return out.transpose(0, 1)

    def advance_seq_lens(self, *a, **k):
        pass


def main():
    from mstar.model.registry import HF_MODELS, get_model_class
    from mstar.model.submodule_base import ModelInputsFromEngine

    device = "cuda"
    model = get_model_class("whisper_large")(**HF_MODELS["whisper_large"])

    waveform = model.load_audio(AUDIO_PATH, "cpu").data
    proc = model.process_prompt(
        prompt=None, input_modalities=["audio"], output_modalities=["text"],
        tensors={"audio_inputs": [waveform]},
    )
    prompt_ids = proc["text_inputs"][0].to(device)
    audio_features = proc["audio_features"][0]

    enc_sub = model.get_submodule("audio_encoder", device=device)
    dec_sub = model.get_submodule(
        "decoder", device=device, autocast_dtype=torch.bfloat16,
    )
    dec = dec_sub.decoder

    mock = MockCacheHandle(model.config.decoder_layers)
    engine_inputs = ModelInputsFromEngine(
        request_ids=["r0"], per_request_info={}, cache_manager=mock,
    )

    enc_out = enc_sub.forward("prefill", engine_inputs, audio_features=audio_features)
    encoder_states = enc_out["encoder_states"][0]
    print(f"encoder_states: {tuple(encoder_states.shape)} {encoder_states.dtype}")

    # --- mstar greedy loop ---
    with torch.no_grad():
        embeds = dec.embed(prompt_ids, 0)
        out = dec_sub.forward(
            "prefill", engine_inputs,
            input_embeds=embeds, encoder_states=encoder_states,
        )
        tokens = []
        pos = prompt_ids.shape[0]
        tok = out["logits"][0][-1].argmax().reshape(1)
        for _ in range(MAX_NEW_TOKENS):
            tokens.append(tok.item())
            if tok.item() == model.config.eos_token_id:
                break
            embeds = dec.embed(tok, pos)
            out = dec_sub.forward("decode", engine_inputs, input_embeds=embeds)
            tok = out["logits"][0][-1].argmax().reshape(1)
            pos += 1

    mstar_text = model.tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"\nmstar tokens ({len(tokens)}): {tokens}")
    print(f"mstar text: {mstar_text!r}")

    # --- HF reference ---
    from transformers import WhisperForConditionalGeneration

    hf = WhisperForConditionalGeneration.from_pretrained(
        model.local_dir, torch_dtype=torch.float16,
    ).to(device).eval()
    with torch.no_grad():
        hf_out = hf.generate(
            audio_features.unsqueeze(0).to(device=device, dtype=torch.float16),
            language="en", task="transcribe",
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    hf_tokens = hf_out[0].tolist()
    hf_text = model.tokenizer.decode(hf_tokens, skip_special_tokens=True)
    print(f"\nHF tokens ({len(hf_tokens)}): {hf_tokens}")
    print(f"HF text: {hf_text!r}")

    hf_new = [t for t in hf_tokens if t not in set(prompt_ids.tolist())]
    match = mstar_text.strip() == hf_text.strip()
    print(f"\nTEXT MATCH: {match}")
    if not match:
        print("MISMATCH — inspect above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
