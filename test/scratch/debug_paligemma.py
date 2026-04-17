"""Debug Pi05PaliGemmaExpert vs lerobot PaliGemma layer by layer."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

from mminf.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mminf.model.pi05.config import Pi05Config

DEVICE = torch.device("cuda")
DTYPE = torch.float32
torch.manual_seed(1)

policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(DEVICE).eval()
model = policy.model
paligemma = model.paligemma_with_expert.paligemma
lm = paligemma.model.language_model
print("first layer norm weight (first 5):", lm.layers[0].input_layernorm.weight.flatten()[:5].tolist())

g = torch.Generator(device=DEVICE).manual_seed(1)
images = [torch.rand(1, 3, 224, 224, device=DEVICE, generator=g) * 2 - 1 for _ in range(3)]
img_masks = [torch.ones(1, dtype=torch.bool, device=DEVICE) for _ in range(3)]
tokens = torch.randint(0, 200, (1, 4), device=DEVICE, generator=g)
masks = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)

prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
    [i.to(DTYPE) for i in images], img_masks, tokens, masks
)
print(f"prefix_embs shape: {prefix_embs.shape}, dtype: {prefix_embs.dtype}")
print(f"prefix_embs abs max: {prefix_embs.abs().max().item():.4f}")

# Capture per-layer hidden states from lerobot
ref_per_layer = []
def make_hook(idx):
    def hook(module, inp, out):
        # output is a tuple (hidden_states, ...)
        if isinstance(out, tuple):
            ref_per_layer.append(out[0].detach().clone())
        else:
            ref_per_layer.append(out.detach().clone())
    return hook

handles = []
for i, layer in enumerate(lm.layers):
    handles.append(layer.register_forward_hook(make_hook(i)))

prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
paligemma.model.language_model.config._attn_implementation = "eager"
prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

ref_outputs, _ = model.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_pos,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)

for h in handles:
    h.remove()

print(f"\nCaptured {len(ref_per_layer)} per-layer outputs from lerobot")
print(f"layer 0 shape: {ref_per_layer[0].shape}")
print(f"final ref hidden shape: {ref_outputs[0].shape}")

# Now run mminf and capture per layer too via a hook
cfg = Pi05Config(
    hidden_size=2048, num_layers=18, num_qo_heads=8, num_kv_heads=1,
    head_dim=256, pali_intermediate_size=16384, rms_norm_eps=lm.layers[0].input_layernorm.eps,
)
ours = Pi05PaliGemmaExpert(cfg).to(DEVICE, dtype=DTYPE)
# Copy weights
for our_layer, ref_layer in zip(ours.layers, lm.layers, strict=True):
    with torch.no_grad():
        our_layer.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight)
        our_layer.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight)
        our_layer.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight)
        our_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight)
        our_layer.mlp.gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight)
        our_layer.mlp.up_proj.weight.copy_(ref_layer.mlp.up_proj.weight)
        our_layer.mlp.down_proj.weight.copy_(ref_layer.mlp.down_proj.weight)
        our_layer.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
        our_layer.post_attention_layernorm.weight.copy_(ref_layer.post_attention_layernorm.weight)
with torch.no_grad():
    ours.norm.weight.copy_(lm.norm.weight)

# Mock cache handle that does bidirectional SDPA with HF RoPE
class _Handle:
    def __init__(self, prefix_len):
        self.head_dim = 256
        self.scale = 256 ** -0.5
        self.positions = torch.arange(prefix_len, device=DEVICE, dtype=torch.long)
        self.layer_idx = 0
    def set_layer_idx(self, i):
        self.layer_idx = i
    def set_active_label(self, l):
        pass
    def apply_rope(self, q, k, rope_theta=10000.0, **kwargs):
        head_dim = self.head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
        freqs = self.positions.float()[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[:, None, :]
        sin = emb.sin()[:, None, :]
        def rot(x):
            return torch.cat([-x[..., head_dim//2:], x[..., :head_dim//2]], dim=-1)
        qf = q.float()
        kf = k.float()
        return (qf*cos + rot(qf)*sin).to(q.dtype), (kf*cos + rot(kf)*sin).to(k.dtype)
    def run_attention(self, q, k, v):
        nh = q.shape[1]
        nkv = k.shape[1]
        if nh != nkv:
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        qt = q.transpose(0, 1).float()
        kt = k.transpose(0, 1).float()
        vt = v.transpose(0, 1).float()
        scores = torch.einsum("hqd,hkd->hqk", qt, kt) * self.scale
        attn = scores.softmax(-1)
        out = torch.einsum("hqk,hkd->hqd", attn, vt)
        return out.transpose(0, 1).to(q.dtype)
    def advance_seq_lens(self, *a, **k):
        pass

handle = _Handle(prefix_pad_masks.shape[1])
ours_per_layer = []
def our_hook(module, inp, out):
    ours_per_layer.append(out.detach().clone())
for layer in ours.layers:
    layer.register_forward_hook(our_hook)

ours_hidden = ours(prefix_embs[0], cache_handle=handle, write_cache=False)

print("\nLayer-by-layer deltas:")
for i, (ref, our) in enumerate(zip(ref_per_layer, ours_per_layer, strict=True)):
    delta = (our - ref[0]).abs().max().item()
    print(f"  layer {i:2d}: max delta = {delta:.4e}, ref abs max = {ref.abs().max().item():.4f}")

print(f"\nfinal hidden delta: {(ours_hidden - ref_outputs[0][0]).abs().max().item():.4e}")
