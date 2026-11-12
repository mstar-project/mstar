"""What the conductor makes of a Qwen3 prompt's length: a request that asks no length
decodes to the room the checkpoint's context leaves, and a prompt past it is a 400."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

# the model's resources import Triton kernels (a torch dependency on Linux)
pytest.importorskip("triton")

from test_kv_layout_intake import _conduct, _preprocess  # noqa: E402

# Past the default reply (2048 tokens), so the decode loop's max_iters (the context) holds a
# request that asks no length and the context, not the loop, is what bounds it.
CONTEXT = 4096
CONFIG = dict(
    architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=48, vocab_size=97,
    rms_norm_eps=1e-6, rope_theta=10_000.0, max_position_embeddings=CONTEXT, tie_word_embeddings=False,
    eos_token_id=96, torch_dtype="float32",
)
# the harness's deployment, with this model's one node and resources
DEPLOYMENT = {"node_groups": [{"node_names": ["LLM"], "ranks": [0]}], "resources": {}}


class _Tokenizer:
    def __init__(self, n: int):
        self.n = n

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=[1] * self.n)


def _model(tmp_path, prompt_tokens: int):
    from mstar.model.qwen3.qwen3_model import Qwen3Model

    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    model = Qwen3Model(model_path_hf=str(tmp_path))
    model.tokenizer = _Tokenizer(prompt_tokens)
    # the model's own checkpoint_path would fetch the weights
    model.checkpoint_path = lambda: "/a/checkpoint"
    return model


def _prompt_signal(n: int) -> dict:
    """The prompt's ids as `data_worker._finish_input` sends them: one tensor of ``text_inputs``,
    which the scaffold's walk plan prefills."""
    import torch

    from mstar.graph.base import TensorPointerInfo

    return {"text_inputs": [TensorPointerInfo(
        dims=[n], dtype=torch.long, nbytes=8 * n, address=0, stride=[1], uuid=1,
        source_session_id="test:0", source_entity="api_server",
    )]}


def _sent(model, tmp_path, monkeypatch):
    kwargs = _preprocess(model, {}, **DEPLOYMENT)
    signals = _prompt_signal(model.tokenizer.n)
    return _conduct(model, kwargs, tmp_path, monkeypatch, initial_signals=signals, **DEPLOYMENT).sent


def test_a_prompt_four_short_of_the_context_is_served_to_the_room_it_leaves(tmp_path, monkeypatch):
    model = _model(tmp_path, CONTEXT - 4)

    sent = _sent(model, tmp_path, monkeypatch)

    new_request = next(msg.body for msg in sent if getattr(msg.message_type, "name", "") == "NEW_REQUEST")
    # four decode steps write positions C-4..C-1, and the fifth token is sampled after the last of
    # them, as the harness's own context test fits (test_kv_layout_intake)
    assert new_request.loop_caps["decode_loop"] == 4
    assert new_request.request_info.max_tokens == 5


def test_a_prompt_past_the_context_is_a_400(tmp_path, monkeypatch):
    model = _model(tmp_path, CONTEXT + 1)

    sent = _sent(model, tmp_path, monkeypatch)

    failed = [msg.body for msg in sent if getattr(msg, "message_type", None) == "request_failed"]
    assert [body.status for body in failed] == [400], "a prompt past the context was served"
    assert "past the model's context" in failed[0].error_message
