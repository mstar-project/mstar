"""The fp32 third configuration for the parity yardstick: the reference model run in float32
(no TF32 anywhere) on the bf16 oracle's own inputs.

Usage: python -I record_fp32.py <hf/Step-Audio-2-mini> <oracle/rec>

Writes ``<rec>/fp32_<case>.pt`` per case: encoder_blocks (each block's output),
adapter_out, hidden_states (every LLM layer, prefill), last_logits. Inputs (ids, mels) are
read from ``<rec>/llm_<case>.pt``, so both precisions see the same bytes.
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
torch.set_float32_matmul_precision("highest")

llm = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
assert not llm.bf16 and llm.encoder.conv1.weight.dtype == torch.float32


def padding_mels(mels):
    lens = torch.tensor([m.shape[1] - 2 for m in mels], dtype=torch.int32)
    feats = torch.nn.utils.rnn.pad_sequence([m.t() for m in mels], batch_first=True)
    return feats.transpose(1, 2), lens


@torch.inference_mode()
def record(case):
    src = torch.load(os.path.join(rec_dir, f"llm_{case}.pt"))
    ids = src["input_ids"].cuda()
    rec = {}
    hidden = llm.model.embed_tokens(ids)
    if src["mels"]:
        wavs, wav_lens = padding_mels(src["mels"])
        wavs, wav_lens = wavs.cuda(), wav_lens.cuda()
        blocks = []
        hooks = [b.register_forward_hook(lambda _m, _i, o: blocks.append(o.cpu())) for b in llm.encoder.blocks]
        enc, feat_lens = llm.encoder(wavs, wav_lens)
        for h in hooks:
            h.remove()
        adapted = llm.adapter(enc)
        feat_lens = (feat_lens - 1) // 2 + 1
        rec.update(encoder_blocks=blocks, encoder_out=enc.cpu(), adapter_out=adapted.cpu(), feat_lens=feat_lens.cpu())
        where = torch.nonzero(ids == 151688)
        where[:, 1] += 1
        for k, (i, s) in enumerate(where.tolist()):
            hidden[i][s:s + feat_lens[k]] = adapted[k][:feat_lens[k]]
    rec["inputs_embeds"] = hidden.cpu()
    out = llm.model(inputs_embeds=hidden, output_hidden_states=True)
    rec["hidden_states"] = [h[0].cpu() for h in out.hidden_states]
    rec["last_logits"] = llm.lm_head(out.last_hidden_state[:, -1]).cpu()
    torch.save(rec, os.path.join(rec_dir, f"fp32_{case}.pt"))
    print(case, ids.shape[1], flush=True)


for case in ("t2t", "s2t", "s2s", "s2t_long"):
    record(case)
