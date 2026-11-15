"""Reference conformer + projections alone, in the oracle venv, under several
kernels and dtypes, on single clips and on padded pairs (the mask path).

usage: record_audio.py <out.pt>  (e.g. audio/conformer.pt beside this script)

Cases: short, long, cut (the first 5.08 s of speech_long) alone, and padded
pairs pair_short (short + cut) and pair_long (short + long). Labels:
bf16_manual (as served), bf16_sdpa (use_pt_scaled_dot_product_attention),
fp32_manual. Each holds the conformer rows (audio_embed_sizes kept, clip by
clip) and both projections."""
import glob
import json
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from safetensors import safe_open

PATH = glob.glob('/root/hf/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/*/')[0]
sys.path.insert(0, PATH)
import speech_conformer_encoder as sce  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cfg = json.load(open(PATH + 'config.json'))
PREFIX = 'model.embed_tokens_extend.audio_embed.'
state = {}
idx = json.load(open(PATH + 'model.safetensors.index.json'))['weight_map']
for shard in sorted({v for k, v in idx.items() if k.startswith(PREFIX)}):
    with safe_open(PATH + shard, 'pt') as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                state[k[len(PREFIX):]] = f.get_tensor(k)


def build(sdpa, dtype):
    enc_cfg = dict(cfg['audio_processor']['config'], use_pt_scaled_dot_product_attention=sdpa)
    enc = sce.ConformerEncoder(**enc_cfg)
    enc.post_init({})
    proj = torch.nn.ModuleDict({
        m: torch.nn.Sequential(torch.nn.Linear(1024, 3072), torch.nn.GELU(), torch.nn.Linear(3072, 3072))
        for m in ('speech', 'vision')
    })
    enc.load_state_dict({k[len('encoder.'):]: v for k, v in state.items() if k.startswith('encoder.')}, strict=True)
    prefix = 'audio_projection.'
    proj.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    return enc.to('cuda', dtype).eval(), proj.to('cuda', dtype).eval()


proc = AutoProcessor.from_pretrained(PATH, trust_remote_code=True).audio_processor
MEDIA = str(Path(os.environ.get('PHI4MM_ORACLE', '/workspace/phi4mm/oracle')) / 'media') + '/'
short = sf.read(MEDIA + 'speech_short.wav')
long_ = sf.read(MEDIA + 'speech_long.wav')
cut = (long_[0][: 5 * 16000 + 1234], long_[1])  # 5.08 s
cases = {
    'short': [short], 'long': [long_], 'cut': [cut],
    'pair_short': [short, cut],  # both one window
    'pair_long': [short, long_],  # the long one unfolds
}
out = {}
for name, audios in cases.items():
    feats = proc(audios, return_tensors='pt')
    out[name] = {'inputs': {k: v for k, v in feats.items()}}
for label, sdpa, dtype in [('bf16_manual', False, torch.bfloat16), ('bf16_sdpa', True, torch.bfloat16),
                           ('fp32_manual', False, torch.float32)]:
    enc, proj = build(sdpa, dtype)
    for name in cases:
        f = out[name]['inputs']
        mask = f.get('audio_attention_mask')
        with torch.no_grad():
            x, _ = enc(f['input_audio_embeds'].to('cuda', dtype), None if mask is None else mask.cuda())
            sizes = f['audio_embed_sizes'].tolist()
            rows = torch.cat([x[i, :n] for i, n in enumerate(sizes)])
            out[name][label] = {'encoded': rows.cpu(), **{m: proj[m](rows).cpu() for m in ('speech', 'vision')}}
    print(label, 'done', flush=True)
torch.save(out, sys.argv[1])
