"""The decode Loop's runaway cap against the request's output budget.

check_stop enforces the per-request ``max_tokens``; the Loop's ``max_iters``
is only the runaway guard. It used to be sized to the *default* budget
(1024), so a request asking for more was cut short at 1025 tokens with no
error while vLLM honored the same request in full.
"""

import pytest
import torch

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
    # Loop ends after max_iters iterations; each emits at least one token on
    # top of the prefill's, so reaching the budget takes budget-1 of them
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


def test_process_prompt_refuses_a_prompt_without_room():
    """A prompt leaves two rows under the limit (decode runs a step, and the
    next may already be scheduled when it stops); refusing it here is a 400,
    in the worker a 500, and at limit - 1 it used to fail its decode batch."""
    m = _model()
    n = m.config.max_prompt_tokens
    assert n == _guard(m.config) - 2
    assert len(m.process_prompt("x" * n, ["text"], ["text"])["text_inputs"][0]) == n
    with pytest.raises(ValueError, match=f"at most {n}"):
        m.process_prompt("x" * (n + 1), ["text"], ["text"])


def test_a_long_context_prompt_takes_the_window():
    # the paged DSA path selects past index_topk in prefill as well
    cfg = Glm52ModelConfig(dsa_long_context=True, max_seq_len=8192)
    assert cfg.max_prompt_tokens == 8192 - 2
    assert Glm52ModelConfig().max_prompt_tokens == 2046
    # an MTP step writes its whole verify block: two of them for k = 3
    assert Glm52ModelConfig(mtp_num_draft_tokens=3).max_prompt_tokens == 2048 - 8


def test_postprocess_emits_raw_token_bytes():
    class ByteLevelTokenizer:
        # GPT-2 byte-level token strings: "Ã" + "©" are the bytes of "é"
        all_special_ids = [0]
        tokens = {0: "<|endoftext|>", 1: "Ã", 2: "©", 3: "Ġhi"}

        def convert_ids_to_tokens(self, ids):
            return [self.tokens[i] for i in ids]

    m = Glm52Model(model_path_hf="")
    m._tokenizer = ByteLevelTokenizer()
    out = [m.postprocess(torch.tensor([i]), "text") for i in (1, 2)]
    # per-token decode gave U+FFFD for each half
    assert b"".join(out).decode("utf-8") == "é"
    assert m.postprocess(torch.tensor([3, 0]), "text") == b" hi"


def test_sampling_knobs_without_a_config_default_reach_the_sampler():
    from mstar.model.glm52.config import SAMPLER_RESOURCE

    cfg = _model().get_request_resource_configs({}, {"top_k": "20", "min_p": 0.05})
    assert cfg[SAMPLER_RESOURCE].top_k == 20 and cfg[SAMPLER_RESOURCE].min_p == 0.05


def test_the_compile_hatch_covers_uncaptured_steps(monkeypatch):
    from mstar.model.glm52.submodules import Glm52LLMSubmodule

    sub = object.__new__(Glm52LLMSubmodule)
    monkeypatch.delenv("MSTAR_GLM52_GRAPH_COMPILE", raising=False)
    assert not sub.disable_torch_compile
    monkeypatch.setenv("MSTAR_GLM52_GRAPH_COMPILE", "0")
    assert sub.disable_torch_compile and not sub._compile_flags()["compile"]


def test_the_single_rank_config_is_dummy_mode():
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "configs" / "test" / "glm52_single_rank.yaml"
    # with the registry's repo id it downloaded 750 GB and built the model on one GPU
    assert yaml.safe_load(path.read_text())["model_kwargs"]["model_path_hf"] == ""


def test_a_negative_draft_count_is_refused():
    # read as off by some checks and on by others, its first decode step failed the batch
    with pytest.raises(ValueError, match="mtp_num_draft_tokens"):
        _model(mtp_num_draft_tokens=-1)


def test_mtp_serves_one_next_token_layer():
    from mstar.model.glm52.weight_loader import _make_glm52_name_remapper

    remap = _make_glm52_name_remapper(78, load_mtp=True)
    assert remap("model.layers.78.eh_proj.weight").startswith("mtp.")
    # a second nextn layer silently overwrote the draft module's weights
    with pytest.raises(ValueError, match="one next-token layer"):
        remap("model.layers.79.eh_proj.weight")
