"""The reference's greedy decode against itself: bf16 under math SDPA and fp32 (no TF32), from
the bf16 oracle's own inputs, with each step's top-2 logit margin of the bf16 default run.

Usage: python -I greedy_spread.py <hf/Step-Audio-2-mini> <oracle/rec>
Writes <rec>/greedy_spread.pt: {case: {"math": [...], "fp32": [...], "margins": [...]}}.
"""
import os
import sys

model_dir, rec_dir = sys.argv[1:3]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf_shim  # noqa: E402,F401
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def padding_mels(mels):
    lens = torch.tensor([m.shape[1] - 2 for m in mels], dtype=torch.int32)
    feats = torch.nn.utils.rnn.pad_sequence([m.t() for m in mels], batch_first=True)
    return feats.transpose(1, 2), lens


def load(dtype):
    return AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype).cuda().eval()


def generate(llm, src, dtype, scores=False):
    ids = src["input_ids"].cuda()
    kw = {}
    if src["mels"]:
        wavs, wav_lens = padding_mels(src["mels"])
        kw = dict(wavs=wavs.cuda().to(dtype), wav_lens=wav_lens.cuda())
    out = llm.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=32, do_sample=False,
                       eos_token_id=151665, pad_token_id=151643, output_scores=scores,
                       return_dict_in_generate=True, **kw)
    toks = out.sequences[0, ids.shape[1]:].tolist()
    margins = None
    if scores:
        margins = [float(s[0].topk(2).values[0] - s[0].topk(2).values[1]) for s in out.scores]
    return toks, margins


res = {}
with torch.inference_mode():
    llm = load(torch.bfloat16)
    for case in ("t2t", "s2t", "s2s", "s2t_long"):
        src = torch.load(os.path.join(rec_dir, f"llm_{case}.pt"))
        default, margins = generate(llm, src, torch.bfloat16, scores=True)
        assert default == src["greedy"].tolist(), (case, default)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            math, _ = generate(llm, src, torch.bfloat16)
        res[case] = {"math": math, "margins": margins}
        print(case, "math == default:", math == default, flush=True)
    del llm
    torch.cuda.empty_cache()
    llm = load(torch.float32)
    for case, row in res.items():
        src = torch.load(os.path.join(rec_dir, f"llm_{case}.pt"))
        row["fp32"], _ = generate(llm, src, torch.float32)
        print(case, "fp32 == default:", row["fp32"] == src["greedy"].tolist(), row["fp32"], flush=True)
torch.save(res, os.path.join(rec_dir, "greedy_spread.pt"))
