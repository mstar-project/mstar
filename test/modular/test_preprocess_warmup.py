"""The preprocess worker warms the model's prompt-side state once at start."""

import logging

from mstar.api_server.data_worker import warm_up_model
from mstar.model.base import Model


def test_default_warmup_is_a_no_op():
    assert Model.warmup_preprocess.__doc__
    assert Model.warmup_preprocess(object()) is None


def test_warm_up_model_calls_and_survives_failures(caplog):
    calls = []

    class Fine:
        def warmup_preprocess(self):
            calls.append("fine")

    class Broken:
        def warmup_preprocess(self):
            raise RuntimeError("no spaCy model")

    warm_up_model(Fine())
    assert calls == ["fine"]
    warm_up_model(None)
    warm_up_model(object())  # a model without the hook
    with caplog.at_level(logging.ERROR):
        warm_up_model(Broken())
    assert "warmup_preprocess failed" in caplog.text
