"""CPU backbone checks using in-memory resource doubles and dense reference math.

These exercise real projections, MoE dispatch and AttentionCallable plumbing;
they do not substitute for FlashInfer or Hugging Face end-to-end parity.
"""

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from mstar.distributed.communication import CommGroup
from mstar.model.command_a_plus.components.language_model import (
    CommandAPlusAttention,
    CommandAPlusLanguageModel,
)
from mstar.model.command_a_plus.config import (
    GLOBAL_ATTN,
    KV_CACHE,
    LOCAL_ATTN,
    ROPE,
    CommandAPlusConfig,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_config.json"


class MemoryKV:
    def __init__(self):
        self.cache = {}
        self.writes = []

    def set_default_label(self, label):
        self.default_label = label

    def set_default_layer_idx(self, index):
        self.layer_idx = index

    def write_kv(self, k, v):
        key = (self.default_label, self.layer_idx)
        self.writes.append(key)
        if key in self.cache:
            old_k, old_v = self.cache[key]
            k, v = torch.cat([old_k, k]), torch.cat([old_v, v])
        self.cache[key] = (k, v)

    def layer_view(self):
        return self.cache[self.default_label, self.layer_idx]


class TorchAttention:
    requires_kv_write = True

    def __init__(self, kv, window=None):
        self.kv = kv
        self.window = window
        self.calls = []

    def set_default_label(self, label):
        self.label = label

    def set_default_layer_idx(self, index):
        self.layer_idx = index

    def run(self, q, *, kv_cache_layer, k, v):
        assert (self.label, self.layer_idx) == (self.kv.default_label, self.kv.layer_idx)
        self.calls.append((self.label, self.layer_idx))
        return attend_last(q, *kv_cache_layer, self.window)


def attend_last(q, k, v, window=None):
    """Causal attention of the last ``len(q)`` of ``len(k)`` positions, over the last ``window``."""
    prefix = len(k) - len(q)
    queries = torch.arange(prefix, len(k))[:, None]
    keys = torch.arange(len(k))[None, :]
    mask = keys <= queries
    if window is not None:
        mask &= keys > queries - window
    repeats = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
    return F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask,
    ).transpose(0, 1).contiguous()


class PagedTorchAttention:
    """Attends over the pages the KV plan names for one group of the pool, as FlashInfer
    reads them: what the pool kept of the stream, the step's tokens last."""

    requires_kv_write = True

    def __init__(self, window=None, group=0):
        self.window, self.group, self.plans = window, group, None

    def set_default_label(self, label):
        self.label = label

    def set_default_layer_idx(self, index):
        pass

    def run(self, q, *, kv_cache_layer, k, v):
        plans = self.plans.groups[self.group]
        (view,) = plans[self.label].views
        slots = torch.arange(view.start, view.start + view.length)
        page_size = kv_cache_layer.shape[2]
        kept = kv_cache_layer[torch.tensor(view.page_idxs)[slots // page_size], :, slots % page_size]
        return attend_last(q, kept[:, 0], kept[:, 1], self.window)


class TorchPositions:
    def __init__(self):
        self.positions = {}
        self.calls = []

    def apply_qk(self, q, k, *, label, rotary_dim, interleave,
                 rope_theta, rope_scale, rope_dtype):
        # Signature deliberately rejects accidental Llama scaling arguments.
        assert interleave and rotary_dim == q.shape[-1] and rope_scale == 1.0
        assert rope_dtype == q.dtype
        self.calls.append(label)
        angles = self.positions[label].float()[:, None] * (
            rope_theta ** (-torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        )
        phase = torch.polar(torch.ones_like(angles), angles)[:, None, :]

        def rotate(x):
            pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(pairs * phase).flatten(-2).to(x.dtype)

        return rotate(q), rotate(k)


def bind_test_resources(model, config):
    kv, pos = MemoryKV(), TorchPositions()
    local, global_ = TorchAttention(kv, config.sliding_window), TorchAttention(kv)
    resources = {KV_CACHE: kv, ROPE: pos, LOCAL_ATTN: local, GLOBAL_ATTN: global_}
    for module in model.modules():
        if isinstance(module, CommandAPlusAttention):
            module.bind_resources(resources)
    return kv, pos, local, global_


def dense_reference(model, token_ids, config):
    """Independent full-sequence FP32 calculation, with no resource cursors/cache."""
    x = F.embedding(token_ids, model.embed_tokens.weight)
    count, dim = len(token_ids), config.head_dim
    positions = torch.arange(count)
    angles = positions[:, None] * config.rope_theta ** (-torch.arange(0, dim, 2).float() / dim)
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]

    def rotate(x):
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)

    def mlp(x, gate_up, down):
        gate, up = F.linear(x, gate_up).chunk(2, -1)
        return F.linear(F.silu(gate) * up, down)

    for index, layer in enumerate(model.layers):
        normalized = F.layer_norm(x, (config.hidden_size,), layer.input_layernorm.weight,
                                  eps=config.layer_norm_eps)
        qdim, kvdim = config.num_attention_heads * dim, config.num_key_value_heads * dim
        qw, kw, vw = layer.self_attn.qkv_proj.weight.split([qdim, kvdim, kvdim])
        q = F.linear(normalized, qw).reshape(count, -1, dim)
        k = F.linear(normalized, kw).reshape(count, -1, dim)
        v = F.linear(normalized, vw).reshape(count, -1, dim)
        local = config.layer_types[index] == "sliding_attention"
        if local:
            q, k = rotate(q), rotate(k)
        repeats = config.num_attention_heads // config.num_key_value_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        scores = torch.einsum("thd,shd->hts", q, k) / dim**0.5
        mask = positions[None, :] <= positions[:, None]
        if local:
            mask &= positions[None, :] > positions[:, None] - config.sliding_window
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        attended = torch.einsum("hts,shd->thd", probs, v).reshape(count, -1)
        attn = F.linear(attended, layer.self_attn.o_proj.weight)
        block = layer.mlp
        logits = F.linear(normalized, block.gate.weight)
        selected = logits.argsort(-1, descending=True)[:, :config.num_experts_per_tok]
        weights = logits.sigmoid() * torch.zeros_like(logits).scatter_(1, selected, 1)
        weights = weights / weights.sum(-1, keepdim=True)
        experts = torch.stack([
            mlp(normalized, block.experts.gate_up_proj[e], block.experts.down_proj[e])
            for e in range(config.num_experts)
        ], dim=1)
        routed = (experts * weights[..., None]).sum(1)
        shared = mlp(normalized, block.shared_experts.gate_up_proj.weight,
                     block.shared_experts.down_proj.weight)
        x = x + attn + (routed + shared) / 2
    return F.layer_norm(x, (config.hidden_size,), model.norm.weight, eps=config.layer_norm_eps)


class BackboneTests(unittest.TestCase):
    def setUp(self):
        self.config = CommandAPlusConfig.from_json(FIXTURE).text_config

    def make_model(self, dtype=torch.float32):
        model = CommandAPlusLanguageModel(self.config).to(dtype=dtype)
        generator = torch.Generator().manual_seed(81)
        with torch.no_grad():
            for name, p in model.named_parameters():
                values = torch.randn(p.shape, generator=generator) * 0.1
                if "norm.weight" in name:
                    values += 1
                p.copy_(values)
        return model

    def test_prefill_and_four_cached_steps_match_dense_reference(self):
        model = self.make_model()
        kv, pos, local, global_ = bind_test_resources(model, self.config)
        token_ids = torch.tensor([2, 17, 8, 91, 6, 11, 57, 3, 22])
        with torch.inference_mode():
            for start, stop in [(0, 5), (5, 6), (6, 7), (7, 8), (8, 9)]:
                pos.positions["main"] = torch.arange(start, stop)
                actual = model(model.embed_tokens(token_ids[start:stop]), label="main")
                expected = dense_reference(model, token_ids[:stop], self.config)[start:stop]
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(kv.writes, [("main", i) for _ in range(5) for i in range(4)])
        self.assertEqual(local.calls, [("main", i) for _ in range(5) for i in range(3)])
        self.assertEqual(global_.calls, [("main", 3)] * 5)
        self.assertEqual(pos.calls, ["main"] * 15)
        self.assertTrue(all(len(k) == 9 for k, v in kv.cache.values()))

    def test_a_cache_that_keeps_the_window_matches_dense_reference(self):
        """The local layers attend over the KV pool as the model declares it, which frees their
        pages behind the window, and still match the dense reference past the window."""
        from mstar.communication.tensors import LocalTransferEngine
        from mstar.engine.resources.kv.config import KVStep
        from mstar.engine.resources.kv.manager import KVManager
        from mstar.engine.resources.kv.transfer import TransferEngineInfo
        from mstar.engine.resources.spec import apply_yaml_overrides
        from mstar.engine.resources.step import Segment, StepContext
        from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel

        with tempfile.TemporaryDirectory() as directory:
            shutil.copyfile(FIXTURE, Path(directory) / "config.json")
            specs = CommandAPlusModel(directory).get_node_resources()
        # 6-token full pages, so 2-token pages of the 3 local layers, a window's in 2 pages
        apply_yaml_overrides(specs, {"resources": {KV_CACHE: {"page_size": 6, "max_num_pages": 32}}})
        cfg = next(spec.config for spec in specs if spec.resource_key == KV_CACHE)
        self.assertEqual([(g.layers, g.page_size) for g in cfg.groups], [((3,), 6), ((0, 1, 2), 2)])
        kv = KVManager(cfg=cfg, name=KV_CACHE, joint_comm_group=None,
                       transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
                       device=torch.device("cpu"), dtype=torch.float32)
        model, pos = self.make_model(), TorchPositions()
        local, global_ = PagedTorchAttention(self.config.sliding_window, group=1), PagedTorchAttention()
        resources = {KV_CACHE: kv, ROPE: pos, LOCAL_ATTN: local, GLOBAL_ATTN: global_}
        for module in model.modules():
            if isinstance(module, CommandAPlusAttention):
                module.bind_resources(resources)
        token_ids = torch.randint(4, 128, (40,), generator=torch.Generator().manual_seed(5))
        kv.ingest_request("r")
        with torch.inference_mode():
            for start, stop in [(0, 20)] + [(n, n + 1) for n in range(20, 40)]:
                step, ctx = KVStep(segments=(Segment("r", "main", stop - start),)), StepContext(("r",), "w", 0, False)
                self.assertTrue(kv.admit(step, ctx).ok)
                local.plans = global_.plans = kv.plan(step, ctx)
                pos.positions["main"] = torch.arange(start, stop)
                actual = model(model.embed_tokens(token_ids[start:stop]), label="main")
                kv.commit(step, ctx)
                expected = dense_reference(model, token_ids[:stop], self.config)[start:stop]
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        stream = kv._streams["r"]["main"]
        self.assertEqual((stream.stored_len, stream.released), (40, 0))
        window = stream.groups[1]
        self.assertLess(window.stored_len, self.config.sliding_window + 2)
        self.assertEqual(window.released + window.stored_len, 40)
        kv.assert_pages_conserved()

    def test_bf16_cached_execution_and_label_switching(self):
        model = self.make_model(torch.bfloat16)
        kv, pos, _, _ = bind_test_resources(model, self.config)
        sequences = {"first": torch.tensor([2, 4, 9, 12, 17, 19]),
                     "second": torch.tensor([3, 5, 11, 21, 25, 29])}
        with torch.inference_mode():
            for start, stop in [(0, 5), (5, 6)]:
                for label, tokens in sequences.items():
                    pos.positions[label] = torch.arange(start, stop)
                    actual = model(model.embed_tokens(tokens[start:stop]), label=label)
                    full = self.make_model(torch.bfloat16)
                    _, full_pos, _, _ = bind_test_resources(full, self.config)
                    full_pos.positions[label] = torch.arange(stop)
                    expected = full(full.embed_tokens(tokens[:stop]), label=label)[start:stop]
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        self.assertEqual(len(kv.cache), 8)
        self.assertTrue(all(len(k) == 6 for k, v in kv.cache.values()))

    def test_tp8_production_geometry_on_meta(self):
        config = replace(
            self.config, hidden_size=4096, num_attention_heads=128,
            num_key_value_heads=8, head_dim=128, vocab_size=262144,
            intermediate_size=4096, num_shared_experts=4, num_experts=128,
            num_experts_per_tok=8, num_hidden_layers=32,
            layer_types=self.config.layer_types * 8,
        )
        group = CommGroup(7, 7, list(range(8)))
        with torch.device("meta"):
            model = CommandAPlusLanguageModel(config, group)
        self.assertEqual(model.embed_tokens.weight.shape, (32768, 4096))
        self.assertEqual(len(model.layers), 32)
        for index, layer in enumerate(model.layers):
            attn = layer.self_attn
            self.assertEqual(attn.qkv_proj.weight.shape, (2304, 4096))
            self.assertEqual(attn.o_proj.weight.shape, (4096, 2048))
            self.assertEqual((attn.num_heads, attn.num_kv_heads), (16, 1))
            self.assertEqual(attn._attn_key, GLOBAL_ATTN if index % 4 == 3 else LOCAL_ATTN)
            self.assertEqual(attn._pos_key, None if index % 4 == 3 else ROPE)
            self.assertIs(attn.comm_group, group)
            self.assertIs(layer.mlp.comm_group, group)
        names = dict(model.named_parameters())
        self.assertIn("layers.3.mlp.shared_experts.gate_up_proj.weight", names)
        self.assertNotIn("layers.0.post_attention_layernorm.weight", names)
        self.assertTrue(all(p.device.type == "meta" for p in names.values()))


if __name__ == "__main__":
    unittest.main()
