"""The reference's greedy decode with each image's projected rows replaced by given ones
(oracle venv). usage: greedy_with_rows.py <rows.pt> case ..."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import CASES, PATH, img, wav
from transformers import AutoModelForCausalLM, AutoProcessor

rows = torch.load(sys.argv[1])
proc = AutoProcessor.from_pretrained(PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                             _attn_implementation='sdpa').cuda().eval()
queue = []
model.model.embed_tokens_extend.image_embed.img_projection.register_forward_hook(
    lambda m, i, o: queue.pop(0).to(o.device, o.dtype).reshape(o.shape))
for name in sys.argv[2:]:
    prompt, media = CASES[name]
    kw = {}
    if 'images' in media:
        kw['images'] = [img(n) for n in media['images']]
    if 'audios' in media:
        kw['audios'] = [wav(n) for n in media['audios']]
    inputs = proc(text=prompt, return_tensors='pt', **kw).to('cuda')
    queue[:] = [rows[name]]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_logits_to_keep=1)
    ids = gen[0, inputs['input_ids'].shape[1]:].tolist()
    oracle = os.environ.get('PHI4MM_ORACLE', '/workspace/phi4mm/oracle')
    torch.save(ids, os.path.join(oracle, f'greedy_ours_rows_{name}.pt'))
    print(name, len(ids), proc.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0][:100])
