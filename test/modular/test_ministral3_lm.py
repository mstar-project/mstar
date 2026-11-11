"""Tiny native Ministral decoder vs Transformers, without checkpoint IO.

The real position manager builds YaRN tables. CPU stand-ins replace only the
rotation kernel and the paged KV/attention execution. These tests exercise the
native stack and cross-call cache semantics, not GPU kernels or engine scheduling.
"""

from dataclasses import asdict, fields, replace

import pytest
import torch

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.keys import KV_CACHE, PREFILL_TEXT, ROPE, SAMPLER
from mstar.model.loader.base import LLAMA_STACKED_PARAMS, load_weights_into
from mstar.model.ministral3.components.language_model import Ministral3DenseLM, Ministral3LMConfig
from mstar.model.ministral3.config import ATTN, Ministral3TextConfig
from mstar.model.ministral3.resources import text_node_resources
from mstar.model.ministral3.submodules import Ministral3Decoder
from mstar.model.submodule_base import ModelInputsFromEngine

pytest.importorskip("transformers.models.ministral3.modeling_ministral3")
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config  # noqa: E402
from transformers.models.ministral3.modeling_ministral3 import Ministral3ForCausalLM  # noqa: E402

TINY = Ministral3LMConfig(hidden_size=96, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=32, intermediate_size=192, vocab_size=128)


def reference(config=TINY):
    values = asdict(config)
    yarn = values.pop("yarn")
    beta = values.pop("llama_4_scaling_beta")
    theta = values.pop("rope_theta")
    hf_config = Ministral3Config(**values, rope_parameters=dict(
        rope_type="yarn", rope_theta=theta, llama_4_scaling_beta=beta, **yarn,
    ), pad_token_id=None)
    hf_config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(42)
        return Ministral3ForCausalLM(hf_config).float().eval()


def transfer_synthetic_weights(hf, native):
    # The strict shared loader checks all constituents of fused QKV/gate-up
    # parameters. This copy exercises synthetic text weights, not FP8 payloads.
    weights = [(name.removeprefix("model."), value) for name, value in hf.state_dict().items()
               if not (native.config.tie_word_embeddings and name == "lm_head.weight")]
    loaded = load_weights_into(native, weights, stacked_params=LLAMA_STACKED_PARAMS, strict=True)
    assert loaded == set(dict(native.named_parameters()))


def cpu_table_rotation(q, k, positions, table, *args, **kwargs):
    cos, sin = table[positions].chunk(2, -1)
    cos, sin = cos[:, None], sin[:, None]
    for x in (q, k):
        a, b = x.chunk(2, -1)
        x.copy_(torch.cat((a * cos - b * sin, b * cos + a * sin), -1))


class DenseCacheAttention:
    """Test-only cache keyed by request, stream label and layer, with packed rows."""

    requires_kv_write = True

    def __init__(self):
        self.cache = {}
        self.rows = []
        self.default_label = "main"
        self.layer = 0
        self.pending = None

    def set_default_label(self, label):
        self.default_label = label

    def set_default_layer_idx(self, layer):
        self.layer = layer

    def layer_view(self):
        return self.layer

    def write_kv(self, k, v):
        self.pending = k, v

    def select_last_hidden(self, hidden):
        ends = torch.tensor([count for _, count in self.rows]).cumsum(0) - 1
        return hidden[ends]

    def run(self, q, kv_cache_layer, k=None, v=None):
        k, v = self.pending
        self.pending = None
        outputs = []
        offset = 0
        for rid, count in self.rows:
            key = rid, self.default_label, kv_cache_layer
            ki, vi = k[offset:offset + count], v[offset:offset + count]
            previous = self.cache.get(key)
            if previous is not None:
                ki, vi = torch.cat((previous[0], ki)), torch.cat((previous[1], vi))
            self.cache[key] = ki, vi
            repeats = q.shape[1] // ki.shape[1]
            qi = q[offset:offset + count].transpose(0, 1)
            kh = ki.repeat_interleave(repeats, dim=1).transpose(0, 1)
            vh = vi.repeat_interleave(repeats, dim=1).transpose(0, 1)
            scores = (qi @ kh.transpose(-1, -2)) * q.shape[-1] ** -0.5
            # New queries occupy the LAST count positions in the cache.
            mask = torch.ones(count, ki.shape[0], dtype=torch.bool).triu(ki.shape[0] - count + 1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            outputs.append((scores.softmax(-1) @ vh).transpose(0, 1))
            offset += count
        assert offset == q.shape[0]
        return torch.cat(outputs)


class Harness:
    def __init__(self, hf, config=TINY, *, starts=None, plain_rope=False):
        self.model = Ministral3DenseLM(config, attn_key="attn", kv_key="kv", pos_key="rope").eval()
        transfer_synthetic_weights(hf, self.model)
        self.cache = DenseCacheAttention()
        self.positions = dict(starts or {})
        self.pos = RopeManager(PositionConfig(
            kv_cache="kv", rotary_dim=config.head_dim, rope_theta=config.rope_theta,
            yarn=None if plain_rope else config.yarn,
            rope_dtype=torch.float32, rope_table_dtype=torch.float32,
        ), torch.device("cpu"), max_seq_len=33000)
        for layer in self.model.layers:
            layer.self_attn.bind_resources(dict(attn=self.cache, kv=self.cache, rope=self.pos))

    @torch.no_grad()
    def step(self, requests):
        self.cache.rows = [(rid, ids.numel()) for rid, ids in requests]
        self.pos._current_pos_ids["main"] = torch.cat([
            torch.arange(self.positions.get(rid, 0), self.positions.get(rid, 0) + ids.numel())
            for rid, ids in requests
        ])
        ids = torch.cat([ids for _, ids in requests])
        embeds = self.model.embed(ids)
        before = embeds.clone()
        hidden = self.model(embeds, label="main")
        assert torch.equal(embeds, before), "the residual path must preserve caller-owned inputs"
        logits = self.model.logits(hidden)
        for rid, count in self.cache.rows:
            self.positions[rid] = self.positions.get(rid, 0) + count
        return logits.split([count for _, count in self.cache.rows])


@pytest.fixture(autouse=True)
def cpu_rotation(monkeypatch):
    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", cpu_table_rotation)


def assert_parity(actual, expected, label):
    error = (actual - expected).abs().max().item()
    print(f"{label}: max_abs_error={error:.9g}")
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))


@pytest.mark.parametrize("tied", [True, False])
@torch.no_grad()
def test_weights_and_prefill_match_reference(tied):
    config = replace(TINY, tie_word_embeddings=tied)
    hf = reference(config)
    harness = Harness(hf, config)
    model = harness.model
    assert (model.embed_tokens.weight is model.lm_head.weight) == tied
    for actual, expected in zip(model.layers, hf.model.layers, strict=True):
        assert torch.equal(actual.self_attn.qkv_proj.weight, torch.cat([
            expected.self_attn.q_proj.weight, expected.self_attn.k_proj.weight, expected.self_attn.v_proj.weight,
        ]))
        assert torch.equal(actual.mlp.gate_up_proj.weight, torch.cat([
            expected.mlp.gate_proj.weight, expected.mlp.up_proj.weight,
        ]))
    ids = torch.tensor([3, 8, 12, 4, 19, 27, 5])
    assert_parity(harness.step([(1, ids)])[0], hf(ids[None], use_cache=False).logits[0], f"prefill tied={tied}")


@pytest.mark.parametrize("start", [0, 16380, 32764])
@torch.no_grad()
def test_prefill_then_cached_decode_matches_full_sequence(start):
    hf = reference()
    harness = Harness(hf, starts={1: start})
    sequence = torch.tensor([3, 8, 12, 4])
    positions = torch.arange(start, start + sequence.numel())[None]
    oracle = hf(sequence[None], position_ids=positions, use_cache=True)
    assert_parity(harness.step([(1, sequence)])[0], oracle.logits[0], f"prefill start={start}")
    for step in range(3):
        token = oracle.logits[0, -1].argmax().view(1)
        sequence = torch.cat((sequence, token))
        pos = torch.tensor([[start + sequence.numel() - 1]])
        oracle = hf(token[None], position_ids=pos, past_key_values=oracle.past_key_values, use_cache=True)
        full = hf(sequence[None], position_ids=torch.arange(start, start + sequence.numel())[None], use_cache=False)
        assert_parity(oracle.logits[0], full.logits[0, -1:], f"reference cache step={step}")
        actual = harness.step([(1, token)])[0]
        assert_parity(actual, oracle.logits[0], f"native cache start={start} step={step}")
    assert len(harness.cache.cache) == TINY.num_hidden_layers
    assert all(k.shape[0] == sequence.numel() for k, _ in harness.cache.cache.values())


@torch.no_grad()
def test_chunked_prefill_matches_reference():
    hf = reference()
    harness = Harness(hf, starts={1: 16381})
    ids = torch.tensor([3, 8, 12, 4, 19, 27, 5])
    expected = hf(ids[None], position_ids=torch.arange(16381, 16388)[None], use_cache=False).logits[0]
    actual = torch.cat([harness.step([(1, chunk)])[0] for chunk in ids.split([2, 3, 2])])
    assert_parity(actual, expected, "chunked prefill")


@torch.no_grad()
def test_packed_requests_match_individual_runs_with_reordered_decode():
    hf = reference()
    starts = {1: 0, 2: 16382}
    packed = Harness(hf, starts=starts)
    individual = {rid: Harness(hf, starts={rid: start}) for rid, start in starts.items()}
    sequences = {1: torch.tensor([3, 8, 12]), 2: torch.tensor([4, 19, 27, 5, 22])}
    steps = [list(sequences.items()), [(2, torch.tensor([7])), (1, torch.tensor([6]))],
             [(1, torch.tensor([18, 17])), (2, torch.tensor([25]))]]
    for step, requests in enumerate(steps):
        results = packed.step(requests)
        for (rid, ids), actual in zip(requests, results, strict=True):
            alone = individual[rid].step([(rid, ids)])[0]
            assert_parity(actual, alone, f"packed vs alone step={step} rid={rid}")
            if step:
                sequences[rid] = torch.cat((sequences[rid], ids))
            positions = torch.arange(starts[rid], starts[rid] + sequences[rid].numel())[None]
            expected = hf(sequences[rid][None], position_ids=positions, use_cache=False).logits[0, -ids.numel():]
            assert_parity(actual, expected, f"packed vs HF step={step} rid={rid}")


@pytest.mark.parametrize("mechanism", ["yarn", "query_scale"])
@torch.no_grad()
def test_parity_detects_missing_position_mechanisms(mechanism):
    hf = reference()
    start = 16382
    ids = torch.tensor([3, 8, 12, 4, 19, 27, 5])
    expected = hf(ids[None], position_ids=torch.arange(start, start + ids.numel())[None], use_cache=False).logits[0]
    good = Harness(hf, starts={1: start}).step([(1, ids)])[0]
    assert_parity(good, expected, "negative-control baseline")
    config = replace(TINY, llama_4_scaling_beta=0) if mechanism == "query_scale" else TINY
    bad = Harness(hf, config, starts={1: start}, plain_rope=mechanism == "yarn").step([(1, ids)])[0]
    print(f"disabled {mechanism}: max_abs_error={(bad - expected).abs().max().item():.9g}")
    assert not torch.allclose(bad, expected, atol=2e-6, rtol=1e-5)


class GreedySampler:
    """Test-only sampler, retaining the token each request feeds into decode."""

    has_slot_masters = True

    def __init__(self):
        self.tokens = {}
        self.logits = None

    def sample(self, request_ids, logits):
        self.logits = logits.clone()
        tokens = logits.argmax(-1)
        self.tokens.update(zip(request_ids, tokens.unbind(), strict=True))
        return tokens

    def loopback_tokens(self, request_ids):
        return torch.stack([self.tokens[rid] for rid in request_ids])


@torch.no_grad()
def test_sampled_decoder_prefill_and_reordered_loopback_match_reference(monkeypatch):
    """The native stack runs through the real node preprocess/forward and binding
    hooks. CPU doubles stand in for planned attention, rotation and sampling;
    this does not run the worker or the engine's resource lifecycle.
    """
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    config = Ministral3TextConfig(
        **{f.name: getattr(TINY, f.name) for f in fields(TINY)},
        stop_token_ids=(127,), max_position_embeddings=33000,
    )
    node = Ministral3Decoder(config).eval()
    hf = reference()
    transfer_synthetic_weights(hf, node.model)
    specs = {s.resource_key: s for s in text_node_resources(config)}
    # The CPU oracle is float32. Production leaves kernel dtype selection to
    # the engine; its GPU-oriented default would round these float32 Q/K to bf16.
    pos_config = replace(specs[ROPE].config, rope_dtype=torch.float32)
    pos = RopeManager(pos_config, torch.device("cpu"), max_seq_len=config.max_position_embeddings)
    cache, sampler = DenseCacheAttention(), GreedySampler()
    resources = {KV_CACHE: cache, ATTN: cache, ROPE: pos, SAMPLER: sampler}
    node.bind_node_resources(resources)
    starts = {17: 0, 4: 16382}
    sequences = {17: torch.tensor([3, 8, 12]), 4: torch.tensor([4, 19, 27, 5, 22])}
    for step in range(4):
        walk = PREFILL_TEXT if step == 0 else "decode"
        rids = [17, 4] if step % 2 == 0 else [4, 17]
        if step:
            for rid in rids:
                sequences[rid] = torch.cat((sequences[rid], sampler.tokens[rid].reshape(1)))
        inputs = [node.prepare_inputs(
            walk, None, {"text_inputs": [sequences[rid]]} if step == 0 else {}, has_loopback=step > 0,
        ) for rid in rids]
        declared = node.declare_step(walk, rids, inputs)
        cache.rows = [(segment.request_id, segment.span) for segment in declared.segments]
        pos._current_pos_ids["main"] = torch.cat([
            torch.arange(starts[rid] + sequences[rid].numel() - count,
                         starts[rid] + sequences[rid].numel())
            for rid, count in cache.rows
        ])
        engine_inputs = ModelInputsFromEngine(request_ids=rids, per_request_info={}, resources=resources)
        tensors = node.preprocess(walk, engine_inputs, inputs)
        output = node.forward_batched(walk, engine_inputs, **tensors)
        expected = torch.cat([hf(
            sequences[rid][None],
            position_ids=torch.arange(starts[rid], starts[rid] + sequences[rid].numel())[None],
            use_cache=False,
        ).logits[:, -1] for rid in rids])
        assert_parity(sampler.logits, expected, f"sampled decoder step={step}")
        tokens = output.row_outputs["new_token"]
        assert torch.equal(tokens, expected.argmax(-1))
        assert output.check_stop_buffers["text_inputs"] is tokens
        assert output.check_stop_buffers["new_token"] is tokens
