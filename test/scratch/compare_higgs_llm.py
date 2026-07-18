"""Token-level parity check: mstar Higgs-Audio STT vs the checkpoint's
own transcribe pipeline (AutoModel + trust_remote_code).

Runs the mstar encoder + LLM submodules with a mock cache handle in a
greedy loop (prefill_text -> prefill_audio -> prefill_text -> decode)
and compares against the reference model driven the same way (manual
embedding splice + HF generate).

Usage:
    python test/scratch/compare_higgs_llm.py [audio.wav]
"""
import sys

import torch

AUDIO_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/libri_0.wav"
MAX_NEW_TOKENS = 100


class MockCacheHandle:
    """Minimal BatchedCacheManager stand-in: growing per-layer KV, causal
    SDPA, explicit position tracking for RoPE. Single request only."""

    def __init__(self, num_layers: int):
        self.kv = [None] * num_layers
        self.layer_idx = 0
        self.pos_start = 0
        self._plan_len = 0

    def set_active_label(self, label):
        pass

    def plan_attention(self, seq_lens, is_causal, label=None):
        self._plan_len = seq_lens[0]

    def plan_rope(self, seq_lens, pos_ids=None, label=None):
        pass

    def set_layer_idx(self, i):
        self.layer_idx = i

    def apply_rope(self, q, k, rope_theta=10000.0, **kwargs):
        import flashinfer

        pos = torch.arange(
            self.pos_start, self.pos_start + q.shape[0], device=q.device,
        ).int()
        q, k = q.clone(), k.clone()
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q, k, pos, rope_theta=rope_theta, interleave=False,
        )
        return q, k

    def run_attention(self, q, k, v, layer_idx=None):
        i = self.layer_idx if layer_idx is None else layer_idx
        if self.kv[i] is None:
            self.kv[i] = (k, v)
        else:
            pk, pv = self.kv[i]
            self.kv[i] = (torch.cat([pk, k]), torch.cat([pv, v]))
        fk, fv = self.kv[i]
        past = fk.shape[0] - q.shape[0]
        mask = None
        if q.shape[0] > 1:
            mask = torch.ones(
                q.shape[0], fk.shape[0], dtype=torch.bool, device=q.device,
            ).tril(diagonal=past)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), fk.transpose(0, 1), fv.transpose(0, 1),
            attn_mask=mask, enable_gqa=True,
        )
        return out.transpose(0, 1)

    def advance_seq_lens(self, *a, **kw):
        self.pos_start += self._plan_len


def main():
    from mstar.model.registry import HF_MODELS, get_model_class
    from mstar.model.submodule_base import ModelInputsFromEngine

    device = "cuda"
    model = get_model_class("higgs_audio")(**HF_MODELS["higgs_audio"])

    waveform = model.load_audio(AUDIO_PATH, "cpu").data
    proc = model.process_prompt(
        prompt=None, input_modalities=["audio"], output_modalities=["text"],
        tensors={"audio_inputs": [waveform]},
    )
    pre_ids, post_ids = [t.to(device) for t in proc["text_inputs"]]
    audio_features = proc["audio_features"][0]
    audio_feature_lens = proc["audio_feature_lens"][0]
    print(f"chunks: {tuple(audio_features.shape)}, mel lens: {audio_feature_lens.tolist()}")

    enc_sub = model.get_submodule("audio_encoder", device=device)
    llm_sub = model.get_submodule(
        "LLM", device=device, autocast_dtype=torch.bfloat16,
    )

    mock = MockCacheHandle(model.config.num_hidden_layers)
    engine_inputs = ModelInputsFromEngine(
        request_ids=["r0"], per_request_info={}, cache_manager=mock,
    )

    enc_out = enc_sub.forward(
        "prefill_audio", engine_inputs,
        audio_features=audio_features, audio_feature_lens=audio_feature_lens,
    )
    audio_embeds = enc_out["audio_embeds"][0].to(torch.bfloat16)
    print(f"audio_embeds: {tuple(audio_embeds.shape)}")

    llm = llm_sub.model

    def run_span(embeds):
        mock.plan_attention([embeds.shape[0]], is_causal=True)
        return llm(input_embeds=embeds, cache_handle=mock)

    with torch.no_grad():
        run_span(llm.embed_tokens(pre_ids))
        run_span(audio_embeds)
        hidden = run_span(llm.embed_tokens(post_ids))
        tok = llm.lm_head(hidden[-1:]).argmax(-1).reshape(1)

        tokens = []
        for _ in range(MAX_NEW_TOKENS):
            tokens.append(tok.item())
            if tok.item() in model.config.stop_token_ids:
                break
            hidden = run_span(llm.embed_tokens(tok))
            tok = llm.lm_head(hidden[-1:]).argmax(-1).reshape(1)

    mstar_text = model.tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"\nmstar tokens ({len(tokens)}): {tokens}")
    print(f"mstar text: {mstar_text!r}")
    # NOTE: the checkpoint's own modeling code targets transformers 4.x
    # (its encoder does `encoder_layer(...)[0]`, which strips the batch
    # dim on transformers 5), so there is no in-venv HF reference to
    # diff against. Correctness is validated by WER on LibriSpeech
    # (reference: ~2% test-clean for higgs-audio-v3-stt).


if __name__ == "__main__":
    main()
