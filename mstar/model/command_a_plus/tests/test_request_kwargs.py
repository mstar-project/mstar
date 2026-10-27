"""The request keys Command A+ declares are the ones it reads."""

import unittest

from mstar.model.command_a_plus.command_a_plus_model import CommandAPlusModel
from mstar.model.command_a_plus.config import SAMPLER


class RequestKwargsTests(unittest.TestCase):
    def test_every_declared_sampling_knob_reaches_the_sampler(self):
        model = object.__new__(CommandAPlusModel)
        declared = model.request_kwargs()
        base = model.get_request_resource_configs({}, {})[SAMPLER]
        sent = {"temperature": 0.0, "top_p": 0.5, "top_k": 7, "repetition_penalty": 1.3, "ignore_eos": True}
        for key, value in sent.items():
            self.assertIn(key, declared)
            config = model.get_request_resource_configs({}, {key: value})[SAMPLER]
            self.assertEqual(getattr(config, key), value)
            self.assertNotEqual(getattr(base, key), value)
        self.assertEqual(model.get_max_output_tokens(max_output_tokens=9), 9)
        self.assertEqual(declared, set(sent) | {"messages", "max_output_tokens"})


if __name__ == "__main__":
    unittest.main()
