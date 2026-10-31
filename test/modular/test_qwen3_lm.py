"""The shared dense Qwen3 LM against HF's ``Qwen3ForCausalLM`` on CPU, and the
frames ``run_layers`` compiles (Qwen3.5's stack included).

The engine's KV / attention / position resources are stood in for by dense
fakes (causal SDPA over the accumulated K/V with GQA, and textbook RoPE), so
one prefill plus a few decode steps are compared logit for logit.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from mstar.distributed.communication import CommGroup  # noqa: E402
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

    def fusable_rope_theta(self, head_dim):
        return None  # applies RoPE itself: the layer stays on the unfused path

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


def test_tp_rank_ties_its_own_vocab_shard():
    values = dict(TINY, vocab_size=96, tie_word_embeddings=True)
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(Qwen3Config(**values, attn_implementation="eager")).eval()
    rank1 = CommGroup(my_global_rank=1, my_group_rank=1, group_members=[0, 1])
    ours = Qwen3DenseLM(Qwen3LMConfig.from_hf(values), attn_key=ATTN, kv_key=KV, pos_key=POS, comm_group=rank1)
    state = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    del state["lm_head.weight"]
    ours.load_weights(list(state.items()))
    assert torch.equal(ours.lm_head.weight, state["embed_tokens.weight"][48:])


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


def test_layer_stack_runs_one_compiled_frame_per_layer_besides_attention(dynamo_counts):
    """``run_layers`` runs eager (its loop breaks at the layer cursor), so each
    compiled frame it calls per layer is dispatched on every uncaptured forward:
    the norms and the MLP must be one frame."""

    def lm(num_layers):
        torch.manual_seed(0)
        config = Qwen3LMConfig.from_hf(dict(TINY, num_hidden_layers=num_layers))
        ours = Qwen3DenseLM(config, attn_key=ATTN, kv_key=KV, pos_key=POS).eval()
        fake, rope = _FakeKVAttention(), _FakeRope()
        for module in ours.modules():
            if hasattr(module, "bind_resources") and module is not ours:
                module.bind_resources({ATTN: fake, KV: fake, POS: rope})
        return ours

    x = torch.randn(3, TINY["hidden_size"])
    two, four = lm(2), lm(4)
    extra = (dynamo_counts(lambda h: four(h, label="main"), x).graph_calls
             - dynamo_counts(lambda h: two(h, label="main"), x).graph_calls)
    assert extra <= 2 * (dynamo_counts(two.layers[0].self_attn, x).graph_calls + 1)


def test_an_empty_layer_stack_is_the_final_norm():
    from mstar.model.components import RMSNorm
    from mstar.model.components.decoder_layer import run_layers

    norm, x = RMSNorm(8), torch.randn(3, 8)
    assert torch.equal(run_layers(torch.nn.ModuleList(), x.clone(), norm), norm(x))


class _StubCursor:
    @torch.compiler.disable
    def bind_step(self, label):
        pass

    @torch.compiler.disable
    def set_layer_idx(self, layer_idx):
        pass


class _StubMixer(torch.nn.Module):
    """A delta-net or attention layer as the stack sees it: cursors, one matmul."""

    def __init__(self, hidden):
        super().__init__()
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)
        self.attend = self.mix = _StubCursor()

    def forward(self, h):
        return self.proj(h)


def test_qwen3_5_layer_loop_compiles_no_frame_per_layer(dynamo_counts):
    """A captured forward compiles with static ints, so a compiled frame that
    holds the layer index (Qwen3.5's ``set_cursor`` moves each layer's cursor)
    recompiles per layer until the recompile limit: four layers must compile
    no more frames than two. And the frame that holds it must not fail the
    capture stance, as a ``recursive=False`` one does."""
    from mstar.model.qwen3_5.components.language_model import Qwen3_5LanguageModel
    from mstar.model.qwen3_5.config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3_5Config

    def compiles(n):
        config = Qwen3_5Config(
            num_hidden_layers=n, hidden_size=16, intermediate_size=32,
            layer_types=[LINEAR_ATTENTION] * (n - 1) + [FULL_ATTENTION],
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            linear_num_key_heads=1, linear_num_value_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4, vocab_size=32,
            rms_norm_eps=1e-6, max_position_embeddings=64,
        )
        lm = Qwen3_5LanguageModel(config)
        for layer in lm.layers:
            layer.self_attn = _StubMixer(config.hidden_size)
        return dynamo_counts(lambda h: lm(h, label="main"), torch.randn(3, config.hidden_size)).compiles

    assert compiles(4) == compiles(2)


@pytest.mark.parametrize("overrides,head_dim,fusable", [
    ({}, 128, True),
    ({"rotary_dim": 128}, 128, True),
    ({"rotary_dim": 64}, 128, False),  # partial rotary
    ({"interleave": True}, 128, False),
    ({"rope_scale": 2.0}, 128, False),
    ({"rope_dtype": torch.float32}, 128, False),
    ({"low_freq_factor": 1.0, "high_freq_factor": 4.0, "old_context_len": 8192}, 128, False),
])
def test_the_position_resource_says_when_a_layer_may_fuse_its_rope(overrides, head_dim, fusable):
    from mstar.engine.resources.position.config import PositionConfig
    from mstar.engine.resources.position.manager import RopeManager

    rope = RopeManager(
        config=PositionConfig(kv_cache="kv", rope_theta=1e6, **overrides),
        device=torch.device("cpu"), head_dim=head_dim,
    )
    assert rope.fusable_rope_theta(head_dim) == (1e6 if fusable else None)
