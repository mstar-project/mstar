"""The request keys Command A+ declares are the ones it reads."""

import unittest

from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.config import SAMPLER


class RequestKwargsTests(unittest.TestCase):
    def test_every_declared_sampling_knob_reaches_the_sampler(self):
        model = object.__new__(CommandAPlusModel)
        declared = model.request_kwargs()
        base = model.get_request_resource_configs([], [], {})[SAMPLER]
        sent = {
            "temperature": 0.0, "top_p": 0.5, "top_k": 7, "min_p": 0.2, "repetition_penalty": 1.3, "ignore_eos": True,
            "penalize_prompt": False,
        }
        for key, value in sent.items():
            self.assertIn(key, declared)
            config = model.get_request_resource_configs([], [], {key: value})[SAMPLER]
            self.assertEqual(getattr(config, key), value)
            self.assertNotEqual(getattr(base, key), value)
        self.assertEqual(model.get_max_output_tokens(max_output_tokens=9), 9)
        self.assertEqual(declared, set(sent) | {"messages", "max_output_tokens"})

    def test_checkpoint_generation_config_layers_over_the_recipe(self):
        import json
        import tempfile
        from pathlib import Path

        from mstar.model.command_a_plus.command_a_plus_model import DEFAULT_SAMPLING
        from mstar.model.components.request_kwargs import sampling_defaults

        with tempfile.TemporaryDirectory() as d:
            # the shape the pinned checkpoint ships: token ids only
            (Path(d) / "generation_config.json").write_text(json.dumps({"bos_token_id": 2, "eos_token_id": 255001}))
            self.assertEqual(sampling_defaults(d, DEFAULT_SAMPLING), dict(DEFAULT_SAMPLING))
            (Path(d) / "generation_config.json").write_text(json.dumps({"temperature": 0.3}))
            self.assertEqual(sampling_defaults(d, DEFAULT_SAMPLING)["temperature"], 0.3)


if __name__ == "__main__":
    unittest.main()
