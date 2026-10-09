"""Offline synthetic checkpoint tests; no GPU or model download required."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from safetensors.torch import save_file

from mstar.distributed.communication import CommGroup
from mstar.model.command_a_plus.components.language_model import CommandAPlusForCausalLM
from mstar.model.command_a_plus.config import CommandAPlusConfig
from mstar.model.command_a_plus.weight_loading import remap_text_weight_name
from mstar.model.loader import load_weights

PREFIX = "model.language_model."


def checkpoint(config):
    """Build separate HF tensors with values encoding both tensor and element."""
    weights = {}

    def add(name, *shape):
        count = 1
        for size in shape:
            count *= size
        weights[PREFIX + name] = torch.arange(count).reshape(shape).float() / 100 + len(weights)

    h, i, s = config.hidden_size, config.intermediate_size, config.shared_intermediate_size
    add("embed_tokens.weight", config.vocab_size, h)
    add("norm.weight", h)
    for layer in range(config.num_hidden_layers):
        base = f"layers.{layer}."
        add(base + "input_layernorm.weight", h)
        for proj, heads in (("q", config.num_attention_heads),
                            ("k", config.num_key_value_heads),
                            ("v", config.num_key_value_heads)):
            add(base + f"self_attn.{proj}_proj.weight", heads * config.head_dim, h)
        add(base + "self_attn.o_proj.weight", h, config.num_attention_heads * config.head_dim)
        add(base + "mlp.gate.weight", config.num_experts, h)
        for expert in range(config.num_experts):
            for proj in ("gate", "up"):
                add(base + f"mlp.experts.{expert}.{proj}_proj.weight", i, h)
            add(base + f"mlp.experts.{expert}.down_proj.weight", h, i)
        for proj in ("gate", "up"):
            add(base + f"mlp.shared_experts.{proj}_proj.weight", s, h)
        add(base + "mlp.shared_experts.down_proj.weight", h, s)
    return weights


class WeightLoadingTests(unittest.TestCase):
    def setUp(self):
        config = CommandAPlusConfig.from_json(
            Path(__file__).parent / "fixtures" / "tiny_config.json",
        ).text_config
        # Expert 10 catches lexicographic ordering mistakes. TP4/8 replicate KV
        # heads; query projection width also differs from the hidden width.
        self.config = replace(config, num_experts=12, num_attention_heads=8, logit_scale=0.5)
        self.weights = checkpoint(self.config)

    def make_model(self, rank=0, tp=1, dtype=torch.float32):
        group = CommGroup(rank, rank, list(range(tp)))
        with torch.device("meta"):
            model = CommandAPlusForCausalLM(self.config, group).to(dtype=dtype)
        model.to_empty(device="cpu")
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(float("nan"))
        return model

    def expected_shard(self, rank, tp):
        config, source = self.config, self.weights
        h, i, s, d = (config.hidden_size, config.intermediate_size,
                       config.shared_intermediate_size, config.head_dim)
        vocab = config.vocab_size // tp
        expected = {
            "model.embed_tokens.weight": source[PREFIX + "embed_tokens.weight"][rank*vocab:(rank+1)*vocab],
            "model.norm.weight": source[PREFIX + "norm.weight"],
        }
        qrows = config.num_attention_heads * d // tp
        kv_heads = max(1, config.num_key_value_heads // tp)
        kv_rank = rank // max(1, tp // config.num_key_value_heads)
        kvrows = kv_heads * d
        for layer in range(config.num_hidden_layers):
            base = f"layers.{layer}."

            def get(suffix, base=base):
                return source[PREFIX + base + suffix]

            def put(suffix, tensor, base=base):
                expected["model." + base + suffix] = tensor

            put("input_layernorm.weight", get("input_layernorm.weight"))
            put("self_attn.qkv_proj.weight", torch.cat([
                get("self_attn.q_proj.weight")[rank*qrows:(rank+1)*qrows],
                get("self_attn.k_proj.weight")[kv_rank*kvrows:(kv_rank+1)*kvrows],
                get("self_attn.v_proj.weight")[kv_rank*kvrows:(kv_rank+1)*kvrows],
            ]))
            put("self_attn.o_proj.weight", get("self_attn.o_proj.weight")[:, rank*qrows:(rank+1)*qrows])
            put("mlp.gate.weight", get("mlp.gate.weight"))
            gate_up, down = [], []
            for expert in range(config.num_experts):
                gate_up.append(torch.cat([
                    get(f"mlp.experts.{expert}.{proj}_proj.weight")[rank*(i//tp):(rank+1)*(i//tp)]
                    for proj in ("gate", "up")
                ]))
                down.append(get(f"mlp.experts.{expert}.down_proj.weight")[:, rank*(i//tp):(rank+1)*(i//tp)])
            put("mlp.experts.gate_up_proj", torch.stack(gate_up))
            put("mlp.experts.down_proj", torch.stack(down))
            put("mlp.shared_experts.gate_up_proj.weight", torch.cat([
                get(f"mlp.shared_experts.{proj}_proj.weight")[rank*(s//tp):(rank+1)*(s//tp)]
                for proj in ("gate", "up")
            ]))
            put("mlp.shared_experts.down_proj.weight",
                get("mlp.shared_experts.down_proj.weight")[:, rank*(s//tp):(rank+1)*(s//tp)])
        return expected

    def test_every_parameter_and_tp_slice(self):
        # A stream sorted lexicographically puts expert 10 before expert 2.
        for tp in (1, 2, 4, 8):
            for rank in range(tp):
                with self.subTest(tp=tp, rank=rank):
                    dtype = torch.bfloat16 if tp == 8 else torch.float32
                    model = self.make_model(rank, tp, dtype)
                    loaded = model.load_weights(iter(sorted(self.weights.items())))
                    expected = self.expected_shard(rank, tp)
                    self.assertEqual(loaded, set(dict(model.named_parameters())))
                    self.assertEqual(loaded, set(expected))
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(parameter, expected[name].to(dtype), atol=0, rtol=0)

    def test_disk_iterator_and_tied_logits_after_allocation(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file({**self.weights, "model.vision_tower.unused.weight": torch.ones(1)}, path)
            loaded = load_weights(model, path)
        self.assertEqual(loaded, set(dict(model.named_parameters())))
        self.assertFalse(any("lm_head" in name for name in loaded))
        x = torch.eye(self.config.hidden_size)[:3]
        with torch.no_grad():
            expected = x @ self.weights[PREFIX + "embed_tokens.weight"].T * self.config.logit_scale
            torch.testing.assert_close(model.compute_logits(x), expected)
            model.model.embed_tokens.weight[0].add_(2)
            expected[:, 0] += 1
            torch.testing.assert_close(model.compute_logits(x), expected)

    def test_missing_fused_slices_and_regular_parameters_fail(self):
        for suffix in (
            "embed_tokens.weight", "layers.0.self_attn.k_proj.weight",
            "layers.0.mlp.shared_experts.up_proj.weight",
            "layers.0.mlp.experts.10.up_proj.weight",
            "layers.0.mlp.experts.2.down_proj.weight",
        ):
            with self.subTest(suffix=suffix):
                weights = ((key, value) for key, value in self.weights.items() if key != PREFIX + suffix)
                with self.assertRaisesRegex(ValueError, "Missing 1 Command A\\+ text weights"):
                    self.make_model().load_weights(weights)

    def test_duplicate_and_unknown_text_weights_fail(self):
        name, tensor = next(iter(self.weights.items()))
        for entries, message in (
            ([(name, tensor), (name, tensor)], "Duplicate"),
            ([(PREFIX + "layers.0.mlp.experts.12.up_proj.weight", tensor)], "Unexpected"),
            ([(PREFIX + "layers.0.self_attn.q_proj.bias", tensor)], "Unexpected"),
            ([(PREFIX + "layers.99.input_layernorm.weight", tensor)], "Unexpected"),
        ):
            with self.subTest(message=message, name=entries[-1][0]):
                with self.assertRaisesRegex(ValueError, message):
                    self.make_model().load_weights(entries)

    def test_full_source_shape_is_checked_before_tp_slicing(self):
        for suffix in (
            "layers.0.self_attn.q_proj.weight",
            "layers.0.mlp.experts.0.gate_proj.weight",
            "layers.0.mlp.experts.0.down_proj.weight",
            "layers.0.mlp.shared_experts.up_proj.weight",
            "embed_tokens.weight", "layers.0.mlp.gate.weight", "norm.weight",
        ):
            with self.subTest(suffix=suffix):
                tensor = self.weights[PREFIX + suffix]
                # Even an oversized tensor whose rank-zero slice would fit is invalid.
                bad = torch.cat([tensor, tensor], dim=0)
                with self.assertRaisesRegex(ValueError, "Shape mismatch"):
                    self.make_model(tp=2).load_weights([(PREFIX + suffix, bad)])

    def test_meta_destination_and_source_are_rejected(self):
        with torch.device("meta"):
            model = CommandAPlusForCausalLM(self.config)
        with self.assertRaisesRegex(ValueError, "to_empty"):
            model.load_weights(self.weights.items())
        name, tensor = next(iter(self.weights.items()))
        with self.assertRaisesRegex(ValueError, "no data"):
            self.make_model().load_weights([(name, tensor.to("meta"))])

    def test_remapper_changes_only_leading_text_prefix(self):
        self.assertEqual(remap_text_weight_name(PREFIX + "norm.weight"), "model.norm.weight")
        for name in ("model.vision_tower.weight", "other." + PREFIX + "norm.weight",
                     "model.language_model_extra.weight", ""):
            self.assertIsNone(remap_text_weight_name(name))


if __name__ == "__main__":
    unittest.main()
