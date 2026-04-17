"""Run a single PaliGemma layer through both lerobot and mminf with copied
weights and compare each substep (norm, attention, MLP) to localize the bug.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot.policies.pi05 import PI05Policy  # noqa: E402

from mminf.model.pi05.components import Pi05GemmaRMSNorm  # noqa: E402
from mminf.model.pi05.components.paligemma import Pi05PaliGemmaLayer  # noqa: E402
from mminf.model.pi05.config import Pi05Config  # noqa: E402

DEVICE = torch.device("cuda")
torch.manual_seed(0)

policy = PI05Policy.from_pretrained("lerobot/pi05_base").to(DEVICE).eval()
lm = policy.model.paligemma_with_expert.paligemma.model.language_model
ref_layer = lm.layers[0]
ref_norm = ref_layer.input_layernorm

print("ref norm type:", type(ref_norm).__name__)
print("ref norm.weight first 5:", ref_norm.weight.flatten()[:5].tolist())

x = torch.randn(1, 32, 2048, device=DEVICE, dtype=torch.float32) * 100  # large magnitude

# Reference norm forward (PiGemmaRMSNorm in non-conditional path)
ref_normed, _ = ref_norm(x, cond=None)

# Mine
ours = Pi05GemmaRMSNorm(2048, eps=ref_norm.eps).to(DEVICE, dtype=torch.float32)
with torch.no_grad():
    ours.weight.copy_(ref_norm.weight)
ours_normed = ours(x)

delta = (ours_normed - ref_normed).abs().max().item()
print(f"\nNorm delta: {delta:.4e}, ref abs max: {ref_normed.abs().max().item():.4f}")

# Quick: check the sub-steps
# 1) Variance
var_ours = x.float().square().mean(-1, keepdim=True)
var_ref = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
print(f"variance delta: {(var_ours - var_ref).abs().max().item():.4e}")

# 2) Normed
normed_ours = x.float() * torch.rsqrt(var_ours + ref_norm.eps)
normed_ref = x * torch.rsqrt(var_ref + ref_norm.eps)
print(f"normed delta: {(normed_ours - normed_ref).abs().max().item():.4e}")
print(f"normed_ours dtype: {normed_ours.dtype}, normed_ref dtype: {normed_ref.dtype}")

# 3) Multiplied by (1+weight)
final_ours = normed_ours * (1.0 + ref_norm.weight.float())
final_ref = (normed_ref * (1.0 + ref_norm.weight.float())).to(x.dtype)
print(f"final delta: {(final_ours - final_ref).abs().max().item():.4e}")

# Now the FULL layer through lerobot and one through ours
print()
print("=== Full layer ===")

# Run lerobot layer 0 alone
position_ids = torch.arange(x.shape[1], device=DEVICE)[None, :]
# lerobot layer expects a dict-like signature; build the call
# We need position_embeddings and attention_mask
# Use the model's existing rotary_emb if available
rotary_emb = lm.rotary_emb
cos, sin = rotary_emb(x, position_ids)
print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")

# Run lerobot layer
with torch.no_grad():
    ref_out = ref_layer(
        hidden_states=x,
        attention_mask=None,  # bidirectional, no mask
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        past_key_values=None,
        use_cache=False,
    )
print(f"ref layer 0 output shape: {ref_out.shape}, dtype: {ref_out.dtype}, abs max: {ref_out.abs().max().item():.4f}")

# Now run mine
cfg = Pi05Config(
    hidden_size=2048,
    num_qo_heads=8,
    num_kv_heads=1,
    head_dim=256,
    pali_intermediate_size=16384,
    rms_norm_eps=ref_norm.eps,
)
ours_layer = Pi05PaliGemmaLayer(cfg).to(DEVICE, dtype=torch.float32)
with torch.no_grad():
    ours_layer.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight)
    ours_layer.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight)
    ours_layer.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight)
    ours_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight)
    ours_layer.mlp.gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight)
    ours_layer.mlp.up_proj.weight.copy_(ref_layer.mlp.up_proj.weight)
    ours_layer.mlp.down_proj.weight.copy_(ref_layer.mlp.down_proj.weight)
    ours_layer.input_layernorm.weight.copy_(ref_layer.input_layernorm.weight)
    ours_layer.post_attention_layernorm.weight.copy_(ref_layer.post_attention_layernorm.weight)


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

handle = _Handle(x.shape[1])
ours_out = ours_layer(query_sequence=x[0], cache_handle=handle)

delta = (ours_out - ref_out[0]).abs().max().item()
print(f"\nLayer 0 (mine vs ref): max delta = {delta:.4e}")
print(f"  ref abs max: {ref_out[0].abs().max().item():.4f}")
print(f"  ours abs max: {ours_out.abs().max().item():.4f}")
# Per-row sample
print(f"  ref[0, :5]:  {ref_out[0, 0, :5].tolist()}")
print(f"  ours[0, :5]: {ours_out[0, :5].tolist()}")
