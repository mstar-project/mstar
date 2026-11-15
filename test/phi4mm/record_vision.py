"""Record the reference vision path (Phi4MMImageEmbedding) under several attention routes
(run in /root/oracle-venv, `python -I`).

The oracle's sdpa and eager recordings run the same vision attention (the remote code's
`SiglipAttention`, a matmul + softmax, unless FA2 is enabled), so for the tower they are
bitwise equal. This records, from each case's recorded processor outputs:

- eager:    the remote code's attention as is (masked queries attend the valid keys)
- fa2:      FA2's semantics with the same matmul kernels: valid queries attend valid keys,
            a masked query's attention output is zero (flash_attn_varlen + pad_input)
- fa2_sdpa: FA2's semantics on torch SDPA's kernels (another legitimate kernel route)
- fa2_fp32: fa2 with the module and inputs in fp32

Keys as record.py: inputs, vision_hidden (hidden_states[-2] per tower call), image_embeds
(the projection's output per image).

usage: record_vision.py <out_dir> [variant ...]
"""
import glob
import json
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

PATH = glob.glob('/root/hf/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/*/')[0]
ORACLE = Path(os.environ.get('PHI4MM_ORACLE', '/workspace/phi4mm/oracle')) / 'sdpa'
CASES = ['image_bagel', 'image_food_small', 'image_food_wide', 'two_images', 'image_audio']
PREFIX = 'model.embed_tokens_extend.image_embed.'


def attention(self, hidden_states, attention_mask=None, output_attentions=False, *, kernel, zero_masked):
    b, n, _ = hidden_states.shape
    h, d = self.num_heads, self.head_dim
    q = self.q_proj(hidden_states).view(b, n, h, d).transpose(1, 2)
    k = self.k_proj(hidden_states).view(b, n, h, d).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, n, h, d).transpose(1, 2)
    if kernel == 'manual':
        # the remote code's SiglipAttention, op for op
        w = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            w = w + attention_mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, v)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    o = o.transpose(1, 2).contiguous().reshape(b, n, h * d)
    if zero_masked and attention_mask is not None:
        # a key is masked iff its patch is; the patch's own query is then the masked one
        valid = attention_mask[:, 0, 0, :] == 0
        o = o.masked_fill(~valid[..., None], 0)
    return self.out_proj(o), None


VARIANTS = {
    'eager': (dict(kernel='manual', zero_masked=False), torch.bfloat16),
    'fa2': (dict(kernel='manual', zero_masked=True), torch.bfloat16),
    'fa2_sdpa': (dict(kernel='sdpa', zero_masked=True), torch.bfloat16),
    'fa2_fp32': (dict(kernel='manual', zero_masked=True), torch.float32),
}


def build(dtype):
    cfg = AutoConfig.from_pretrained(PATH, trust_remote_code=True)
    cfg._attn_implementation = 'eager'  # SiglipAttention; the variants replace its forward
    emb_cls = get_class_from_dynamic_module('modeling_phi4mm.Phi4MMImageEmbedding', PATH)
    emb = emb_cls(cfg, **cfg.embd_layer['image_embd_layer'])
    index = json.load(open(PATH + 'model.safetensors.index.json'))['weight_map']
    sd = {}
    for f in sorted({v for k, v in index.items() if k.startswith(PREFIX)}):
        with safe_open(PATH + f, 'pt') as fh:
            for k in fh.keys():
                if k.startswith(PREFIX):
                    sd[k[len(PREFIX):]] = fh.get_tensor(k)
    missing, unexpected = emb.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return emb.to('cuda', dtype).eval()


def main():
    out_dir = Path(sys.argv[1])
    names = sys.argv[2:] or list(VARIANTS)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for variant in names:
        kw, dtype = VARIANTS[variant]
        emb = build(dtype)
        for layer in emb.img_processor.encoder.layers:
            layer.self_attn.forward = types.MethodType(
                lambda self, hidden_states, attention_mask=None, output_attentions=False, kw=kw:
                attention(self, hidden_states, attention_mask, output_attentions, **kw),
                layer.self_attn,
            )
        cap = {}
        emb.img_processor.register_forward_hook(
            lambda m, i, o: cap.setdefault('vision_hidden', []).append(o.hidden_states[-2].detach().cpu()))  # noqa: B023
        emb.img_projection.register_forward_hook(
            lambda m, i, o: cap.setdefault('image_embeds', []).append(o.detach().cpu()))  # noqa: B023
        (out_dir / variant).mkdir(parents=True, exist_ok=True)
        for case in CASES:
            rec = torch.load(ORACLE / f'{case}.pt', weights_only=False)
            inp = rec['inputs']
            ids = inp['input_ids'].cuda()
            cap.clear()
            with torch.no_grad():
                emb(ids, inp['input_image_embeds'].cuda(), image_sizes=inp['image_sizes'].cuda(),
                    image_attention_mask=inp['image_attention_mask'].cuda(),
                    wte=lambda x: torch.zeros(*x.shape, 3072, dtype=dtype, device=x.device))  # noqa: B023
            torch.save({'inputs': {k: inp[k] for k in ('input_ids', 'input_image_embeds', 'image_sizes',
                                                       'image_attention_mask')},
                        **cap}, out_dir / variant / f'{case}.pt')
            print(variant, case, [tuple(t.shape) for t in cap['image_embeds']], flush=True)
        del emb
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
