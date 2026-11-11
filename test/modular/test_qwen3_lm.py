"""The shared dense Qwen3 LM against HF's ``Qwen3ForCausalLM`` on CPU, and the
frames ``run_layers`` compiles (Qwen3.5's stack included).

The engine's KV / attention / position resources, or the array behind the
layers' handle, are stood in for by dense fakes (causal SDPA over the
accumulated K/V with GQA, and textbook RoPE), so one prefill plus a few decode
steps are compared logit for logit, on either path.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from mstar.distributed.communication import CommGroup  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep  # noqa: E402
from mstar.engine.state import (  # noqa: E402
    Array,
    ArrayStep,
    Attend,
    Causal,
    KVEntry,
    Positions,
    Read,
    Rope,
)
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle  # noqa: E402
from mstar.engine.state.analysis import position_key  # noqa: E402
from mstar.engine.state.calls import ArrayHandle  # noqa: E402
from mstar.engine.state.state_op import STATE_OPS  # noqa: E402
from mstar.model.components import qwen3_lm  # noqa: E402
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
        embeds = ours.embed(prompt)
        before = embeds.clone()
        hidden = ours(embeds, label="main")
        actual = ours.logits(hidden)
    assert torch.allclose(actual, expected, atol=1e-4), (actual - expected).abs().max()
    # the residual stream is run_layers' own copy: a caller's static input survives
    assert torch.equal(embeds, before)

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


def _frames_on_resources(num_layers: int):
    """A stack on fake resources: its forward, and its first layer's attention."""
    torch.manual_seed(0)
    config = Qwen3LMConfig.from_hf(dict(TINY, num_hidden_layers=num_layers))
    ours = Qwen3DenseLM(config, attn_key=ATTN, kv_key=KV, pos_key=POS).eval()
    fake, rope = _FakeKVAttention(), _FakeRope()
    for module in ours.modules():
        if hasattr(module, "bind_resources") and module is not ours:
            module.bind_resources({ATTN: fake, KV: fake, POS: rope})
    return (lambda h: ours(h, label="main")), ours.layers[0].self_attn


def _frames_on_its_array(num_layers: int):
    """A stack on a handle: its forward, and its first layer's attention, each in a step of its own."""
    _, ours, handle, _ = _on_state(num_layers=num_layers)

    def step(h):
        _next_step(handle, 3)
        return ours(h)

    def one_layer(h):
        _next_step(handle, 3)
        return ours.layers[0].self_attn(h)

    return step, one_layer


@pytest.mark.parametrize("stack", [_frames_on_resources, _frames_on_its_array], ids=["resources", "array"])
def test_layer_stack_runs_one_compiled_frame_per_layer_besides_attention(dynamo_counts, stack):
    """``run_layers`` runs eager (its loop breaks at the layer cursor), so each
    compiled frame it calls per layer is dispatched on every uncaptured forward:
    the norms and the MLP must be one frame."""
    x = torch.randn(3, TINY["hidden_size"])
    (two, one_layer), (four, _) = stack(2), stack(4)
    extra = dynamo_counts(four, x).graph_calls - dynamo_counts(two, x).graph_calls
    assert extra <= 2 * (dynamo_counts(one_layer, x).graph_calls + 1)


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
    holds the layer index recompiles per layer until the recompile limit: four
    layers must compile no more frames than two."""
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
        return dynamo_counts(lm, torch.randn(3, config.hidden_size)).compiles

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


class _FakePages(_FakeKVAttention):
    """The same pages through a handle's calls, which name the layer and the label."""

    def layer_view(self, i=0):
        return i

    def opens_label(self, rid, label):
        return True

    def write_kv(self, k, v, layer_idx=None, label=None):
        self._pending = (k, v)

    def run(self, q, label=None, kv_cache_layer=None, k=None, v=None, layer_idx=None):
        return super().run(q, kv_cache_layer)


class _FakePositions:
    """Each step's planned ids, past its tokens as a bucket's buffer runs, and a rotation that says it ran."""

    def __init__(self):
        self.ids = torch.arange(64, dtype=torch.int32)
        self.rotated = 0

    def pos_ids(self, label):
        return self.ids

    def advance(self, tokens):
        self.ids += tokens

    def apply_qk(self, q, k, label, rope_dtype):
        self.rotated += 1
        return q, k


def _declared(rope: Rope | None = None, num_layers: int = TINY["num_hidden_layers"]) -> Array:
    return Array(
        KV, nodes={"LLM"}, layers=frozenset(range(num_layers)), max_len=128,
        entry=KVEntry(heads=TINY["num_key_value_heads"], head_dim=TINY["head_dim"]),
        positions=Positions(rope=Rope(theta=TINY["rope_theta"]) if rope is None else rope),
        reads={ATTN: Read(mask=Causal(), op=Attend(q_heads=TINY["num_attention_heads"]))},
    )


def _on_state(rope: Rope | None = None, num_layers: int = TINY["num_hidden_layers"]):
    """A stack bound to a handle on fake pages, with no fused ops bound, as on any device but CUDA."""
    values = dict(TINY, num_hidden_layers=num_layers)
    torch.manual_seed(0)
    hf = Qwen3ForCausalLM(Qwen3Config(**values, attn_implementation="eager")).eval()
    ours = Qwen3DenseLM(Qwen3LMConfig.from_hf(values), array=KV).eval()
    ours.load_weights([(k.removeprefix("model."), v) for k, v in hf.state_dict().items()])
    pages, pos = _FakePages(), _FakePositions()
    array = _declared(rope, num_layers)
    handle = build_handle(array, {KV: pages, ATTN: pages, position_key(array): pos}, _resolve([array], StateSetup()))
    for module in ours.modules():
        if hasattr(module, "bind_state"):
            module.bind_state({KV: handle})
    return hf, ours, handle, pos


@torch.compiler.disable
def _next_step(handle: ArrayHandle, tokens: int) -> None:
    step = SubmoduleStep(segments=[Segment(1, "main", tokens)], state={KV: ArrayStep()})
    step.set_ctx(StepContext((1,), "decode", 0, False))
    handle.end_step(step)
    handle.begin_step(step)


def test_prefill_and_decode_logits_match_hf_through_the_ops_reference():
    """No fused kernel off CUDA, so this holds the reference to the textbook rotation the kernel is held to."""
    hf, ours, handle, pos = _on_state()
    torch.manual_seed(3)
    prompt = torch.randint(0, TINY["vocab_size"], (7,))
    with torch.no_grad():
        expected = hf(prompt[None]).logits[0]
        _next_step(handle, 7)
        embeds = ours.embed(prompt)
        before = embeds.clone()
        actual = ours.logits(ours(embeds))
    assert torch.allclose(actual, expected, atol=1e-4), (actual - expected).abs().max()
    assert pos.rotated == 0, "off CUDA the op's reference runs, not a second path through rope()"
    # the residual stream is run_layers' own copy: a caller's static input survives
    assert torch.equal(embeds, before)

    # two greedy decode steps: the fake pages carry the prompt's K/V forward
    pos.advance(prompt.numel())
    seq = prompt.clone()
    for _ in range(2):
        nxt = expected[-1].argmax().view(1)
        seq = torch.cat([seq, nxt])
        with torch.no_grad():
            expected = hf(seq[None]).logits[0]
            _next_step(handle, 1)
            step = ours.logits(ours(ours.embed(nxt)))
        assert torch.allclose(step[0], expected[-1], atol=1e-4), (step[0] - expected[-1]).abs().max()
        pos.advance(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_ops_reference_is_plain_rope_after_rmsnorm_and_leaves_k_alone(dtype):
    tokens, heads, kv_heads, dim, theta = 5, 4, 2, 8, 1e6
    torch.manual_seed(0)
    q = torch.randn(tokens, heads, dim, dtype=dtype)
    k, v = torch.randn(tokens, kv_heads, dim, dtype=dtype), torch.randn(tokens, kv_heads, dim, dtype=dtype)
    q_weight, k_weight = torch.randn(dim, dtype=dtype), torch.randn(dim, dtype=dtype)
    pos_ids = torch.tensor([0, 1, 2, 7, 100, 999, 999], dtype=torch.int32)
    calls = []

    def read(layer, q):
        calls.append(("read", layer, q.clone()))
        return q * 2

    def textbook(x, weight):
        x = x.float()
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * weight.float()
        inv_freq = 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim)
        z = torch.complex(normed[..., : dim // 2], normed[..., dim // 2:])
        rotated = z * torch.polar(torch.ones(tokens, dim // 2), pos_ids[:tokens].float()[:, None] * inv_freq)[:, None]
        return torch.cat((rotated.real, rotated.imag), -1)

    want_q, want_k, k_before = textbook(q, q_weight), textbook(k, k_weight), k.clone()
    kv = SimpleNamespace(add=lambda layer, k, v: calls.append(("add", layer, k, v)), read=read)
    out = qwen3_lm._norm_rope_attend(kv, 3, q, k, v, q_weight, k_weight, pos_ids, theta, 1e-6, 1e-6)
    tolerance = 1e-5 if dtype is torch.float32 else 2e-2
    assert [call[0] for call in calls] == ["add", "read"], "the read comes after the add"
    (_, layer, added_k, added_v), (_, read_layer, read_q) = calls
    assert torch.allclose(q.float(), want_q, atol=tolerance), "q is rotated in place, as the kernel does"
    assert torch.allclose(added_k.float(), want_k, atol=tolerance), "the stored k is normed and rotated"
    assert torch.equal(k, k_before), "the kernel leaves its k argument as it was"
    assert added_v is v and layer == 3
    assert added_k.dtype is dtype, "a backend that takes K/V at its read refuses a k of another dtype than its v"
    assert read_layer == 3 and torch.equal(read_q, q) and torch.equal(out, q * 2), "the read takes the rotated q"


def test_the_state_op_takes_its_reference_without_a_fused_kernel():
    assert "qk_norm_rope_attend" in STATE_OPS
    assert qwen3_lm.qk_norm_rope_attend.op.reference is qwen3_lm._norm_rope_attend
    assert {fmt.devices for fmt in qwen3_lm.qk_norm_rope_attend.op.formats} == {frozenset({"cuda"})}


@pytest.mark.parametrize("overrides, head_dim, fused", [
    ({}, 128, True),
    ({"rotary_dim": 128}, 128, True),
    ({"rotary_dim": 64}, 128, False),
    ({"interleave": True}, 128, False),
    ({"scale": 2.0}, 128, False),
    ({"dtype": torch.float32}, 128, False),
    ({"low_freq_factor": 1.0, "high_freq_factor": 4.0, "old_context_len": 8192}, 128, False),
    ({"table_dtype": torch.bfloat16}, 128, False),
])
def test_a_declared_rope_the_kernel_doesnt_compute_stays_out_of_it(overrides, head_dim, fused):
    assert qwen3_lm._plain_theta(Rope(theta=1e6, **overrides), head_dim) == (1e6 if fused else None)


def test_no_declared_rope_stays_out_of_the_kernel():
    assert qwen3_lm._plain_theta(None, 128) is None


def test_a_layer_takes_the_op_only_for_a_plain_rope_with_q_and_k_norms():
    _, plain, _, _ = _on_state()
    assert plain.layers[0].self_attn._fuse, "plain RoPE and per-head norms, on any device: the format picks the kernel"
    _, table, handle, pos = _on_state(Rope(theta=1e4, rotary_dim=TINY["head_dim"], interleave=True,
                                           table_dtype=torch.bfloat16))
    assert not table.layers[0].self_attn._fuse, "fusing an interleaved rope would corrupt the outputs with no error"
    _next_step(handle, 3)
    with torch.no_grad():
        table(table.embed(torch.tensor([1, 2, 3])))
    assert pos.rotated == TINY["num_hidden_layers"], "a rope the kernel doesn't compute rotates through the array"


def _capture_frames(num_layers: int) -> int:
    torch._dynamo.reset()
    _, ours, handle, _ = _on_state(num_layers=num_layers)
    frames = []

    def backend(graph_module, inputs):
        frames.append(graph_module)
        return graph_module.forward

    _next_step(handle, 3)
    torch._dynamo.optimize(backend, dynamic=False)(ours)(ours.embed(torch.tensor([1, 2, 3])))
    return len(frames)


@torch.no_grad()
def test_a_stack_compiled_as_a_capture_compiles_no_frame_per_layer():
    """A capture compiles with `dynamic=False`, where a module's int specializes per layer and hits the recompile limit,
    and the frames then run uncompiled."""
    assert _capture_frames(num_layers=4) == _capture_frames(num_layers=2) > 1


@torch.no_grad()
def test_a_second_compiled_step_compiles_nothing_new():
    torch._dynamo.reset()
    _, ours, handle, pos = _on_state()
    frames = []

    def backend(graph_module, inputs):
        frames.append(graph_module)
        return graph_module.forward

    compiled = torch._dynamo.optimize(backend, dynamic=False)(ours)
    embeds = ours.embed(torch.tensor([1, 2, 3]))
    _next_step(handle, 3)
    compiled(embeds)
    first_call = len(frames)
    pos.advance(3)
    _next_step(handle, 3)
    with torch.compiler.set_stance("fail_on_recompile"):
        compiled(embeds)
    assert len(frames) == first_call, "a capture's stance raises on any new compile"
