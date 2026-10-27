"""The request keys Command A+ declares are the ones it reads."""

import unittest

from mstar.engine.resources.sampler.config import SamplerSpec
from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.config import SAMPLER


class RequestKwargsTests(unittest.TestCase):
    def test_every_declared_sampling_knob_reaches_the_sampler(self):
        model = object.__new__(CommandAPlusModel)
        declared = model.request_kwargs()
        base = model.get_request_resource_configs({}, {})[SAMPLER]
        sent = {
            "temperature": 0.0, "top_p": 0.5, "top_k": 7, "min_p": 0.2, "repetition_penalty": 1.3, "ignore_eos": True,
        }
        for key, value in sent.items():
            self.assertIn(key, declared)
            config = model.get_request_resource_configs({}, {key: value})[SAMPLER]
            self.assertEqual(getattr(config, key), value)
            self.assertNotEqual(getattr(base, key), value)
        self.assertEqual(model.get_max_output_tokens(max_output_tokens=9), 9)
        self.assertEqual(declared, set(sent) | {"messages", "max_output_tokens"})

    def test_min_p_is_refused_not_dropped(self):
        """Its SamplerSpec leaves enable_min_p off, so a request that sets min_p is a 400."""
        model = object.__new__(CommandAPlusModel)
        config = model.get_request_resource_configs({}, {"min_p": 0.5})[SAMPLER]
        with self.assertRaisesRegex(ValueError, "min_p"):
            config.validate(SamplerSpec(resource_key=SAMPLER, nodes={"LLM"}, vocab_size=8))


if __name__ == "__main__":
    unittest.main()
