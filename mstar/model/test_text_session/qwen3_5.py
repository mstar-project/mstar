"""The ``test_text_session`` deployment on Qwen3.5 instead of BAGEL.

Same purpose as the BAGEL variant in ``model.py``: the smallest thing that
exercises sessions end to end. Qwen3.5 is the harder case, a hybrid: its full
attention layers keep a KV cache and its GDN layers a recurrent state pool, and
a session has to carry both. A resumed turn that continued the KV but started
the GDN state from zeros would answer from a context it half remembers.

Text only, like the BAGEL variant: the vision walk and the vision tower are
dropped. The checkpoint is a model kwarg, so one class serves every size; the
registry's default is 4B, and a config picks another with
``model_kwargs: {model_path_hf: Qwen/Qwen3.5-9B}``.

Thinking is always off. Qwen's chat template drops a past assistant turn's
``<think>`` block, reasoning and tags together, when it renders the next turn,
so a reasoning trace kept in the session's KV would sit in context where a
resent transcript has none (and count against the budget every turn). With
thinking off, a turn is generated after a pre-closed ``<think>\\n\\n</think>\\n\\n``
and the KV keeps that; the transcript would not. That small, fixed difference is
the one left: the KV holds the turn as the model produced it.

Each turn renders the same whether it opens the session or resumes it: Qwen's
template adds no system block, so a single user turn is already just one more
turn appended to the conversation.
"""

import logging

from mstar.graph.base import GraphSection
from mstar.model.multimodal import TEXT
from mstar.model.qwen3_5.config import GDN_STATE, KV_CACHE
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.sessions import SessionResourceConfig, SessionsConfig
from mstar.model.test_text_session.turn_join import TurnJoinMixin

logger = logging.getLogger(__name__)

# The walks a text turn takes; `prefill_vision` needs the vision tower.
TEXT_WALKS = ("prefill_text", "decode")


class TextSessionQwen3_5LLMSubmodule(TurnJoinMixin, LLMSubmodule):
    """Qwen3.5's LLM, rendering the join between one turn and the next; see
    ``turn_join``. A Qwen turn can also stop on ``<|endoftext|>``, which the
    template never closes with, so the join closes it with ``<|im_end|>``."""


class TextSessionQwen3_5Model(Qwen3_5DenseModel):
    LLM_SUBMODULE_CLS = TextSessionQwen3_5LLMSubmodule

    def _create_submodule(self, *args, **kwargs):
        submodule = super()._create_submodule(*args, **kwargs)
        if isinstance(submodule, TextSessionQwen3_5LLMSubmodule):
            submodule.turn_close_id = self.tokenizer.convert_tokens_to_ids(
                "<|im_end|>"
            )
            submodule.turn_stop_ids = frozenset(self.config.stop_token_ids)
            submodule.newline_ids = self.tokenizer.encode(
                "\n", add_special_tokens=False,
            )
        return submodule

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        walks = super().get_graph_walk_graphs()
        missing = sorted(set(TEXT_WALKS) - walks.keys())
        if missing:
            raise RuntimeError(
                f"Qwen3.5 no longer declares {missing}; this model is its text "
                "path and cannot be served without them"
            )
        return {name: walks[name] for name in TEXT_WALKS}

    def get_node_resources(self):
        """Qwen3.5's, minus the vision tower's own attention."""
        return [
            spec for spec in super().get_node_resources()
            if spec.nodes != {"vision_encoder"}
        ]

    def get_sessions_config(self) -> SessionsConfig:
        """Hold the LLM's KV and its GDN state for the session.

        Both or neither: the engine refuses a session that holds one cache of a
        node and not another. ``max_state`` is in each resource's units: 64 KV
        pages of 128 tokens (~8k tokens of conversation), and the one recurrent
        slot a request's ``main`` label takes.
        """
        return SessionsConfig(
            resources={
                KV_CACHE: SessionResourceConfig(max_state=64),
                GDN_STATE: SessionResourceConfig(max_state=1),
            },
            max_concurrent_sessions=4,
            default_timeout_s=300.0,
            max_timeout_s=1800.0,
        )

    def process_prompt(
        self, prompt, input_modalities, output_modalities,
        tensors=None, prompt_parts=None, **kwargs,
    ):
        refused = sorted(set(input_modalities) - {TEXT})
        if refused:
            raise ValueError(
                f"{type(self).__name__} serves text only; it has no encoder "
                f"for {', '.join(refused)}"
            )
        if any(m != TEXT for m in output_modalities):
            raise ValueError(
                f"{type(self).__name__} generates text only; got "
                f"output_modalities={output_modalities}"
            )
        if kwargs.pop("enable_thinking", False):
            logger.debug("enable_thinking ignored: thinking is off here")
        return super().process_prompt(
            prompt, input_modalities, output_modalities,
            tensors=tensors, prompt_parts=prompt_parts,
            enable_thinking=False, **kwargs,
        )
