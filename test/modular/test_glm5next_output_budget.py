"""GLM-5.3-Flash's output budget, decode loop cap and chat-template call. check_stop
enforces a request's ``max_tokens``; the loop's ``max_iters`` is only the runaway guard."""

import pytest
import torch
from jinja2 import TemplateError

from mstar.api_server.openai.adapters import flatten_messages
from mstar.graph.base import Loop
from mstar.model.glm5_next.glm5_next_model import Glm5NextModel


def _model(**kwargs) -> Glm5NextModel:
    # byte tokenizer: no HF IO at construction
    return Glm5NextModel("x", tokenizer_mode="byte", **kwargs)


def _decode(m: Glm5NextModel) -> Loop:
    decode = m.get_graph_walk_graphs()["decode"]
    assert isinstance(decode, Loop)
    return decode


def test_budget_and_loop_cap_are_held_to_the_window():
    m = _model()
    topk = m.config.index_topk
    assert m.get_max_output_tokens(max_output_tokens=10_000) == topk
    # a one-token prompt's context reaches index_topk after topk - 1 iterations
    assert _decode(m).max_iters == topk - 1


def test_mtp_loop_cap_is_held_to_the_cache_rows():
    m = _model(mtp_num_draft_tokens=3)
    cfg = m.config
    # every verify step stores k + 1 cache rows, accepted or not
    assert _decode(m).max_iters == min(cfg.index_topk - 1, (cfg.kv_rows - 1) // 4)


def test_chat_template_returns_the_bare_ids():
    class ChatTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, **kwargs):
            # transformers 5 returns a BatchEncoding unless return_dict=False
            assert kwargs.get("return_dict") is False
            return torch.tensor([[7, 8, 9]])

    m = Glm5NextModel("x")
    m._tokenizer = ChatTokenizer()
    ids = m.process_prompt("hi", ["text"], ["text"]).new_input_tensors["text_inputs"][0]
    assert ids.tolist() == [7, 8, 9]


class _RecordingTokenizer:
    chat_template = "{{ messages }}"

    def __init__(self, refuse: bool = False):
        self.messages, self.refuse = None, refuse

    def apply_chat_template(self, messages, **kwargs):
        if self.refuse:
            raise TemplateError("roles must alternate")
        self.messages = messages
        return torch.tensor([[7, 8, 9]])


def test_a_chat_keeps_its_turns_in_their_roles(tmp_path):
    chat = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bonjour"},
        {"role": "user", "content": "how are you?"},
    ]
    text, _, _, parts = flatten_messages(chat, tmp_path)
    m = Glm5NextModel("x")
    m._tokenizer = tok = _RecordingTokenizer()
    m.process_prompt(text, ["text"], ["text"], prompt_parts=parts)
    assert tok.messages == [
        {"role": "system", "content": "Answer in French."}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "bonjour"}, {"role": "user", "content": "how are you?"},
    ]


def test_a_chat_the_template_refuses_is_a_400():
    m = Glm5NextModel("x")
    m._tokenizer = _RecordingTokenizer(refuse=True)
    with pytest.raises(ValueError, match="refused the chat"):
        m.process_prompt("hi", ["text"], ["text"])
