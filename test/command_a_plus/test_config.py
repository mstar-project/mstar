"""Offline config contract tests; run with unittest, without model dependencies."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from mstar.model.command_a_plus.config import CommandAPlusConfig, CommandAPlusTextConfig

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_config.json"


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.text = self.data["text_config"]

    def test_tiny_json_and_derived_width(self):
        text = CommandAPlusConfig.from_json(FIXTURE).text_config
        self.assertEqual(text.hidden_size, 32)
        self.assertEqual(text.num_hidden_layers, 4)
        self.assertEqual(text.shared_intermediate_size, 32)
        self.assertEqual(text.pad_token_id, 0)
        with self.assertRaises(AttributeError):
            text.shared_intermediate_size = 100

    def test_production_shaped_json(self):
        self.text.update(
            hidden_size=4096, num_hidden_layers=32, num_attention_heads=128,
            num_key_value_heads=8, head_dim=128, vocab_size=262144,
            intermediate_size=4096, num_experts=128, num_experts_per_tok=8,
            num_shared_experts=4, sliding_window=4096,
            max_position_embeddings=200000, eos_token_id=255001,
            layer_types=self.text["layer_types"] * 8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(self.data), encoding="utf-8")
            text = CommandAPlusConfig.from_json(str(path)).text_config
        self.assertEqual(text.shared_intermediate_size, 16384)
        self.assertEqual(text.num_attention_heads * text.head_dim, 16384)
        self.assertEqual(text.max_position_embeddings, 200000)

    def test_extra_metadata_is_ignored(self):
        self.data["unused_metadata"] = "測試"
        self.text["unused_metadata"] = True
        self.assertEqual(CommandAPlusConfig.from_dict(self.data).text_config.hidden_size, 32)

    def test_unsupported_architecture(self):
        for name, value in (
            ("model_type", "llama"), ("hidden_act", "gelu"),
            ("expert_selection_fn", "softmax"), ("norm_topk_prob", False),
            ("shared_expert_combination_strategy", "sum"),
            ("use_parallel_block", False), ("tie_word_embeddings", False),
            ("attention_bias", True), ("use_qk_norm", True),
            ("position_embedding_type", "rope"), ("rotary_pct", 0.5),
            ("first_k_dense_replace", 1), ("rms_norm_eps", 1e-5),
            ("use_parallel_block", 1), ("attention_bias", 0),
            ("first_k_dense_replace", False), ("rotary_pct", True),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    CommandAPlusTextConfig.from_dict({**self.text, name: value})

    def test_required_fields_are_not_defaulted(self):
        for name in ("hidden_size", "expert_selection_fn", "rms_norm_eps"):
            data = {k: v for k, v in self.text.items() if k != name}
            with self.subTest(name=name):
                with self.assertRaisesRegex((KeyError, ValueError), name):
                    CommandAPlusTextConfig.from_dict(data)

    def test_direct_construction_checks_numbers(self):
        fields = asdict(CommandAPlusTextConfig.from_dict(self.text))
        for name, bad_values in {
            "sliding_window": (0, -1, True, 4.0, "4", None),
            "max_position_embeddings": (0, False, 64.0),
            "head_dim": (7,),
            "num_key_value_heads": (3,),
            "num_experts_per_tok": (5,),
            "rope_theta": (0, -1, True, "50000", float("nan"), float("inf")),
            "layer_norm_eps": (0, None, float("nan")),
            "logit_scale": (False, -1, float("inf")),
        }.items():
            for value in bad_values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        CommandAPlusTextConfig(**{**fields, name: value})

    def test_token_ids_use_vocab_bounds(self):
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            for value in (-1, 128, True, 1.0):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        CommandAPlusTextConfig.from_dict({**self.text, name: value})
        text = CommandAPlusTextConfig.from_dict({**self.text, "eos_token_id": 127})
        self.assertEqual(text.eos_token_id, 127)

    def test_attention_pattern(self):
        for value in (None, "sliding_attention", self.text["layer_types"][:-1]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "layer_types"):
                    CommandAPlusTextConfig.from_dict({**self.text, "layer_types": value})
        for index in range(4):
            layers = self.text["layer_types"].copy()
            layers[index] = "full_attention" if index < 3 else "sliding_attention"
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, rf"layer_types\[{index}\]"):
                    CommandAPlusTextConfig.from_dict({**self.text, "layer_types": layers})

    def test_rope_metadata(self):
        for rope in (None, {}, {"rope_type": "linear"},
                     {"rope_type": "default", "rope_theta": 10000}):
            with self.subTest(rope=rope):
                with self.assertRaisesRegex(ValueError, "rope_parameters"):
                    CommandAPlusTextConfig.from_dict({**self.text, "rope_parameters": rope})
        del self.text["rope_parameters"]
        self.assertEqual(CommandAPlusTextConfig.from_dict(self.text).rope_theta, 50000)

    def test_outer_config_contract(self):
        for patch, message in (
            ({"model_type": "cohere2_moe"}, "model_type"),
            ({"tie_word_embeddings": False}, "tie_word_embeddings"),
            ({"tie_word_embeddings": 1}, "tie_word_embeddings"),
            ({"text_config": None}, "text_config"),
        ):
            with self.subTest(patch=patch):
                with self.assertRaisesRegex(ValueError, message):
                    CommandAPlusConfig.from_dict({**self.data, **patch})
        del self.data["text_config"]
        with self.assertRaisesRegex(ValueError, "text_config"):
            CommandAPlusConfig.from_dict(self.data)
        with self.assertRaisesRegex(ValueError, "JSON object"):
            CommandAPlusConfig.from_dict([])


if __name__ == "__main__":
    unittest.main()
