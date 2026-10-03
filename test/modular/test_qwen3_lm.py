"""The shared dense Qwen3 LM against HF's ``Qwen3ForCausalLM`` on CPU.

The engine's KV / attention / position resources are stood in for by dense
fakes (causal SDPA over the accumulated K/V with GQA, and textbook RoPE), so
one prefill plus a few decode steps are compared logit for logit.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from mstar.model.components.qwen3_lm import Qwen3DenseLM, Qwen3LMConfig  # noqa: E402

ATTN, KV, POS = "attn", "kv", "rope"

TINY = dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    intermediate_size=48, vocab_size=97, rms_norm_eps=1e-6, rope_theta=10_000.0,
    max_position_embeddings=128, tie_word_embeddings=False,
)


class _FakeKVAttention:
    """Dense stand-in for the paged KV + attention pair, single request:
    the K/V written so far are the request's pages; causal SDPA is the kernel."""

    requires_kv_write = True

    def __init__(self):
        self._label = "main"
        self._layer = 0
        self._kv: dict[int, tuple[list, list]] = {}
        self._pending = None

    @property
    def default_label(self):
        return self._label

    def set_default_label(self, label):
        self._label = label

    def set_default_layer_idx(self, idx):
        self._layer = idx

    def layer_view(self):
        return self._layer

    def write_kv(self, k, v):
        self._pending = (k, v)

    def run(self, q, kv_cache_layer, k=None, v=None):
        del k, v
        ks, vs = self._kv.setdefault(kv_cache_layer, ([], []))
        k_new, v_new = self._pending
        self._pending = None
        ks.append(k_new)
        vs.append(v_new)
        k_all, v_all = torch.cat(ks), torch.cat(vs)
        n_new, n_all = q.shape[0], k_all.shape[0]
        rep = q.shape[1] // k_all.shape[1]
        k_all = k_all.repeat_interleave(rep, dim=1)
        v_all = v_all.repeat_interleave(rep, dim=1)
        mask = torch.ones(n_new, n_all, dtype=torch.bool).tril(diagonal=n_all - n_new)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k_all.transpose(0, 1), v_all.transpose(0, 1), attn_mask=mask,
        )
        return out.transpose(0, 1)


class _FakeRope:
    """Textbook rotate-half RoPE at the positions the request has reached."""

    def __init__(self):
        self.next_pos = 0

    def apply_qk(self, q, k, label, rope_theta, **kwargs):
        del label, kwargs
        n, _, d = q.shape
        pos = torch.arange(self.next_pos, self.next_pos + n, dtype=torch.float32)
        inv = 1.0 / (rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        freqs = torch.outer(pos, inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]

        def rot(x):
            x1, x2 = x[..., : d // 2], x[..., d // 2:]
            return torch.cat([-x2, x1], dim=-1)

        return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def _pair(tie: bool = False):
    values = dict(TINY, tie_word_embeddings=tie)
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(Qwen3Config(**values, attn_implementation="eager")).eval()
    ours = Qwen3DenseLM(Qwen3LMConfig.from_hf(values), attn_key=ATTN, kv_key=KV, pos_key=POS).eval()
    state = [(k.removeprefix("model."), v) for k, v in hf.state_dict().items()]
    if tie:
        state = [(k, v) for k, v in state if k != "lm_head.weight"]
    ours.load_weights(state)
    fake, rope = _FakeKVAttention(), _FakeRope()
    for module in ours.modules():
        if hasattr(module, "bind_resources") and module is not ours:
            module.bind_resources({ATTN: fake, KV: fake, POS: rope})
    return hf, ours, rope


def test_config_from_hf_fills_head_dim():
    cfg = Qwen3LMConfig.from_hf({"hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8,
                                 "num_hidden_layers": 28, "intermediate_size": 3072, "vocab_size": 151936})
    assert cfg.head_dim == 64 and cfg.rope_theta == 1_000_000.0


def test_weights_load_completely_and_tied_head_is_copied():
    hf, ours, _ = _pair(tie=True)
    assert torch.equal(ours.lm_head.weight, hf.model.embed_tokens.weight)
    with pytest.raises(RuntimeError, match="unloaded"):
        Qwen3DenseLM(Qwen3LMConfig.from_hf(TINY), attn_key=ATTN, kv_key=KV, pos_key=POS).load_weights(
            [(k.removeprefix("model."), v) for k, v in hf.state_dict().items() if "layers.1" not in k]
        )
    with pytest.raises(RuntimeError, match="does not tie"):
        Qwen3DenseLM(Qwen3LMConfig.from_hf(TINY), attn_key=ATTN, kv_key=KV, pos_key=POS).load_weights(
            [(k.removeprefix("model."), v) for k, v in hf.state_dict().items() if k != "lm_head.weight"]
        )


def test_prefill_and_decode_logits_match_hf():
    hf, ours, rope = _pair()
    torch.manual_seed(3)
    prompt = torch.randint(0, TINY["vocab_size"], (7,))
    with torch.no_grad():
        expected = hf(prompt[None]).logits[0]
        hidden = ours(ours.embed(prompt), label="main")
        actual = ours.logits(hidden)
    assert torch.allclose(actual, expected, atol=1e-4), (actual - expected).abs().max()

    # two greedy decode steps: the fake pages carry the prompt's K/V forward
    rope.next_pos = prompt.numel()
    seq = prompt.clone()
    for _ in range(2):
        nxt = expected[-1].argmax().view(1)
        seq = torch.cat([seq, nxt])
        with torch.no_grad():
            expected = hf(seq[None]).logits[0]
            step = ours.logits(ours(ours.embed(nxt), label="main"))
        assert torch.allclose(step[0], expected[-1], atol=1e-4), (step[0] - expected[-1]).abs().max()
        rope.next_pos += 1
