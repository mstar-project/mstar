"""Record every Step-Audio-2-mini reference tensor the M* parity suite needs, in one session.

Usage: python -I record_oracle.py <github/Step-Audio2> <hf/Step-Audio-2-mini> <oracle>/rec

Reads the greedy spoken reply's codes from ``<oracle>/probe/probe.json`` (``oracle_probe.py``).

LLM side (bf16, as the reference runs it): per case the prompt ids, the mel clips, each encoder
block's output, the adapter output, every LLM layer's prefill hidden states, the last position's
logits and 32 greedy tokens. The 4 s clip is also run under the math SDPA backend, which is the
reference-against-itself yardstick. Token2wav side (fp32, as the reference runs it): the voice
prompt's pieces, the flow's noise buffer, and every streamed window's mel and PCM for the greedy
spoken reply's codes, fed the way ``examples-vllm-stream.py`` feeds them (28-code windows, hop 25,
a last flush), with the CUDA RNG seeded once before the stream.
"""
import json
import os
import sys

ref_dir, model_dir, out_dir = sys.argv[1:4]
sys.path.insert(0, ref_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs(out_dir, exist_ok=True)

import sf_shim  # noqa: E402,F401
import torch  # noqa: E402
from stepaudio2 import StepAudio2  # noqa: E402
from token2wav import Token2wav  # noqa: E402
from utils import padding_mels  # noqa: E402

A = os.path.join(ref_dir, "assets")
model = StepAudio2(model_dir)
llm = model.llm.eval()
sys_msg = {"role": "system", "content": "You are a helpful assistant."}
clip = os.path.join(A, "give_me_a_brief_introduction_to_the_great_wall.wav")
long_clip = os.path.join(A, "paralinguistic_information_understanding.wav")
cases = {
    "t2t": [sys_msg, {"role": "human", "content": "Give me a brief introduction to the Great Wall."},
            {"role": "assistant", "content": None}],
    "s2t": [sys_msg, {"role": "human", "content": [{"type": "audio", "audio": clip}]},
            {"role": "assistant", "content": None}],
    "s2s": [sys_msg, {"role": "human", "content": [{"type": "audio", "audio": clip}]},
            {"role": "assistant", "content": "<tts_start>", "eot": False}],
    "s2t_long": [sys_msg, {"role": "human", "content": [{"type": "text", "text": "Describe this audio."},
                                                        {"type": "audio", "audio": long_clip}]},
                 {"role": "assistant", "content": None}],
}


def inputs_for(messages):
    msgs, mels = model.apply_chat_template(messages)
    ids = []
    for m in msgs:
        if isinstance(m, str):
            ids.append(model.llm_tokenizer(text=m, return_tensors="pt", padding=True)["input_ids"])
        else:
            ids.append(torch.tensor([m], dtype=torch.int32))
    ids = torch.cat(ids, dim=-1).cuda()
    if mels:
        wavs, wav_lens = padding_mels(mels)
        return ids, mels, wavs.cuda(), wav_lens.cuda()
    return ids, mels, None, None


@torch.inference_mode()
def record(name, messages, tag=""):
    ids, mels, wavs, wav_lens = inputs_for(messages)
    rec = {"input_ids": ids.cpu(), "mels": [m.cpu() for m in mels]}
    hidden = llm.model.embed_tokens(ids)
    if wavs is not None:
        blocks = []
        hooks = [b.register_forward_hook(lambda _m, _i, o: blocks.append(o.float().cpu())) for b in llm.encoder.blocks]
        enc, feat_lens = llm.encoder(wavs.bfloat16(), wav_lens)
        for h in hooks:
            h.remove()
        adapted = llm.adapter(enc)
        feat_lens = (feat_lens - 1) // 2 + 1
        rec.update(wav_lens=wav_lens.cpu(), encoder_blocks=blocks, encoder_out=enc.float().cpu(),
                   adapter_out=adapted.float().cpu(), feat_lens=feat_lens.cpu())
        where = torch.nonzero(ids == 151688)
        where[:, 1] += 1
        for k, (i, s) in enumerate(where.tolist()):
            hidden[i][s:s + feat_lens[k]] = adapted[k][:feat_lens[k]]
    rec["inputs_embeds"] = hidden.float().cpu()
    out = llm.model(inputs_embeds=hidden, output_hidden_states=True)
    rec["hidden_states"] = [h[0].float().cpu() for h in out.hidden_states]
    rec["last_logits"] = llm.lm_head(out.last_hidden_state[:, -1]).float().cpu()
    if not tag:
        gen = llm.generate(input_ids=ids, wavs=None if wavs is None else wavs, wav_lens=wav_lens,
                           attention_mask=torch.ones_like(ids), max_new_tokens=32, do_sample=False,
                           eos_token_id=model.eos_token_id, pad_token_id=model.llm_tokenizer.pad_token_id)
        rec["greedy"] = gen[0, ids.shape[1]:].cpu()
    torch.save(rec, os.path.join(out_dir, f"llm_{name}{tag}.pt"))
    print(name + tag, ids.shape[1], "greedy" in rec and rec["greedy"].tolist()[:12], flush=True)


for name, messages in cases.items():
    record(name, messages)
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    record("s2t", cases["s2t"], tag="_math")

# ---- token2wav: the greedy spoken reply's codes, streamed ----
probe = json.load(open(os.path.join(out_dir, os.pardir, "probe", "probe.json")))
codes = [t - 151696 for t in probe["s2s"]["tokens"] if t > 151695][:160]
prompt_wav = os.path.join(A, "default_female.wav")
t2w = Token2wav(os.path.join(model_dir, "token2wav"))
CHUNK, LOOK = 25, t2w.flow.pre_lookahead_len
with torch.inference_mode():
    prompt = t2w._prepare_prompt(prompt_wav)
    t2w.set_stream_cache(prompt_wav)
    mels_out = []
    orig = t2w.flow.inference_chunk

    def spy(**kw):
        mel, cache = orig(**kw)
        mels_out.append(mel.float().cpu())
        return mel, cache

    t2w.flow.inference_chunk = spy
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    windows, pcm, buf = [], [], list(codes)
    while len(buf) >= CHUNK + LOOK:
        windows.append((buf[:CHUNK + LOOK], False))
        pcm.append(t2w.stream(buf[:CHUNK + LOOK], prompt_wav=prompt_wav))
        buf = buf[CHUNK:]
    if buf:
        windows.append((list(buf), True))
        pcm.append(t2w.stream(buf, prompt_wav=prompt_wav, last_chunk=True))
torch.save({
    "codes": codes, "windows": windows, "prompt_wav": prompt_wav,
    "prompt_tokens": prompt[0].cpu(), "spk_emb": prompt[2].cpu(), "prompt_mels": prompt[3].cpu(),
    "rand_noise": t2w.flow.decoder.rand_noise.cpu(), "window_mels": mels_out, "window_pcm": pcm,
}, os.path.join(out_dir, "token2wav_stream.pt"))
print("token2wav windows", len(windows), "pcm bytes", sum(len(p) for p in pcm))
