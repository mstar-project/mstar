"""NemotronDuplex declares what a session writes to each of its two pools, or loading the model fails.

The talker pool and the Mamba slots each hold `DEFAULT_MAMBA_SLOTS` of the longest
sessions the server takes, so after each session reserves its talker pages, the talker
pool has to admit exactly `DEFAULT_MAMBA_SLOTS` sessions.
"""

from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.conductor import Conductor
from mstar.engine.resources.kv.config import count_pages
from mstar.model.nemotron_duplex.config import NANO_KV, TALKER_KV
from mstar.model.nemotron_duplex.nemotron_duplex_model import NemotronDuplexModel


def _model() -> NemotronDuplexModel:
    """The constructor loads no weights, and a tokenizer of one id per word needs no download."""
    model = NemotronDuplexModel("nvidia/NVIDIA-NemotronLabs-VoiceChat-11B")
    model._tokenizer = lambda text, return_tensors: SimpleNamespace(input_ids=torch.arange(len(text.split()))[None])
    return model


def _slots(model: NemotronDuplexModel, prompt: str | None) -> dict:
    """Run `process_prompt` on a one-second session and return what `kv_slots` declares for it,
    as the data worker does."""
    tensors = model.process_prompt(
        prompt, ["audio"], ["text", "audio"], tensors={"audio_inputs": [torch.zeros(16000)]},
    )
    return model.kv_slots(tensors, ["audio"], ["text", "audio"])


@pytest.mark.parametrize(("prompt", "ids"), [("you are a helpful agent", 5), (None, 0)], ids=["prompt", "no_prompt"])
def test_a_session_declares_its_prompt_to_nano_and_its_whole_length_to_the_talker(prompt, ids):
    model = _model()
    talker = model.talker_stream_positions

    assert _slots(model, prompt) == {
        "prompt_slots": {NANO_KV: {"main": ids}, TALKER_KV: {"main": talker, "uncond": talker}},
        "decode_labels": {NANO_KV: ["main"]},
    }, "the no-prompt sentinel counted a slot, or the talker grows by max_tokens"
    assert Conductor._counted_kv_keys(model) == {NANO_KV, TALKER_KV}


def test_the_talker_pool_admits_as_many_sessions_as_it_holds_and_no_more():
    model = _model()
    spec = next(spec for spec in model.get_node_resources() if spec.resource_key == TALKER_KV)
    configs = model.get_request_resource_configs(partition_fwd_args={})
    Conductor._stamp_resource_configs(
        configs, _slots(model, None), seed=0, max_tokens=model.get_max_output_tokens(),
    )

    session = count_pages(configs[TALKER_KV], spec.config, spec.nodes)

    sessions = model.DEFAULT_MAMBA_SLOTS
    assert session.exact
    assert sessions * session.pages <= spec.config.request_pages(), (
        f"{sessions} sessions' talker reservations overflow talker_kv_pages, so the gate holds some back"
    )
    assert (sessions + 1) * session.pages > spec.config.request_pages(), (
        "a session past the Mamba slots fits the talker pool, so the pool is sized past what it serves"
    )
