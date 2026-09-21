"""The decode Loop's runaway cap against the request's output budget.

check_stop enforces the per-request ``max_tokens``; the Loop's ``max_iters``
is only the runaway guard. It used to be sized to the *default* budget
(1024), so a request asking for more was cut short at 1025 tokens with no
error while vLLM honored the same request in full.
"""

import pytest

from mstar.graph.base import Loop
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.glm52_model import Glm52Model


def _model(**kwargs) -> Glm52Model:
    # byte tokenizer: no HF IO at construction, in process_prompt or postprocess
    return Glm52Model(model_path_hf="", tokenizer_mode="byte", **kwargs)


def _guard(cfg: Glm52ModelConfig) -> int:
    # whichever bound the submodule's preprocess compares contexts against
    return cfg.max_seq_len if cfg.dsa_long_context else cfg.index_topk


@pytest.mark.parametrize("k", [0, 2])
def test_request_budget_above_the_default_is_honored(k):
    m = _model(mtp_num_draft_tokens=k)
    # the default budget is unchanged for requests that name none
    assert m.config.max_output_tokens == 1024
    assert m.get_max_output_tokens() == 1024
    budget = m.get_max_output_tokens(max_output_tokens=2000)
    assert budget == 2000
    decode = m.get_graph_walk_graphs()["decode"]
    assert isinstance(decode, Loop)
    # Loop ends after max_iters iterations; k=0 emits one token per iteration
    # on top of the prefill's, so reaching the budget takes budget-1 of them
    assert decode.max_iters >= budget - 1
    # byte mode was honored: nothing lazily built a tokenizer
    assert m.process_prompt("Hi", ["text"], ["text"])["text_inputs"][0].tolist() == [72, 105]
    assert m._tokenizer is None


def test_request_budget_is_held_to_the_context_window():
    m = _model()
    guard = _guard(m.config)
    assert guard == 2048
    # a one-token prompt emits at most `guard` tokens (the last is never
    # stored); a larger budget is unreachable and check_stop could never fire
    assert m.get_max_output_tokens(max_output_tokens=5000) == guard
    assert m.get_max_output_tokens(max_output_tokens=guard) == guard
    assert m.get_max_output_tokens(max_output_tokens=7) == 7


@pytest.mark.parametrize("k", [0, 2])
def test_loop_cap_covers_every_reachable_budget_and_stays_below_the_guard(k):
    m = _model(mtp_num_draft_tokens=k)
    guard = _guard(m.config)
    decode = m.get_graph_walk_graphs()["decode"]
    # the largest budget check_stop can honor needs guard-1 iterations
    assert decode.max_iters >= m.get_max_output_tokens(max_output_tokens=10_000) - 1
    # ...and the cap alone cannot run a one-token prompt into the guard
    assert decode.max_iters == guard - 1
    assert decode.max_iters < guard


def test_loop_cap_follows_the_dsa_serving_window():
    m = _model(dsa_long_context=True, max_seq_len=8192)
    assert m.config.dsa_long_context and m.config.max_seq_len == 8192
    assert m.context_limit() == 8192
    decode = m.get_graph_walk_graphs()["decode"]
    assert decode.max_iters == 8191
    assert m.get_max_output_tokens(max_output_tokens=10_000) == 8192


def test_reduced_variant_cap_tracks_its_smaller_window():
    m = _model(config_variant="reduced")
    guard = _guard(m.config)  # index_topk=64 at test scale, not max_seq_len=512
    assert guard == 64
    decode = m.get_graph_walk_graphs()["decode"]
    assert decode.max_iters == guard - 1
    assert m.get_max_output_tokens(max_output_tokens=1000) == guard
