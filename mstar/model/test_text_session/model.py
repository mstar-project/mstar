"""A text-only deployment of BAGEL's LLM that holds its KV across a session.

The point of this model is the session semantics, not the model: it is the
smallest thing that exercises them end to end. A client starts a session, sends
a turn, resumes with the next turn, and the second request continues from the
KV the first one built instead of re-reading the conversation.

It is BAGEL's LLM and tokenizer with everything else taken away — only the
``prefill_text`` and ``decode`` walks are declared, so the ViT, the VAE and the
image walks are not part of the deployment. Subclassing rather than copying
keeps it honest: the LLM path it tests is the one BAGEL actually serves.

Cross-request prefix reuse is off here. Both features share the page pool and
coexist fine (a resumed request steps out of the index on its own), but a test
model should exercise one thing, and a prefix hit would otherwise hide a session
that failed to carry its state over.
"""

import logging

from mstar.graph.base import GraphSection
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.base import PrefixStream
from mstar.model.multimodal import TEXT
from mstar.model.sessions import (
    SessionOverflowPolicy,
    SessionResourceConfig,
    SessionsConfig,
)

logger = logging.getLogger(__name__)

# The walks a text turn takes. The rest of BAGEL's graph needs the encoders.
TEXT_WALKS = ("prefill_text", "decode")


class TextSessionModel(BagelModel):
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        walks = super().get_graph_walk_graphs()
        missing = sorted(set(TEXT_WALKS) - walks.keys())
        if missing:
            raise RuntimeError(
                f"BAGEL no longer declares {missing}; this model is its text "
                "path and cannot be served without them"
            )
        return {name: walks[name] for name in TEXT_WALKS}

    def get_node_resources(self):
        """BAGEL's, minus the ViT tower's own attention: no ViT here."""
        return [
            spec for spec in super().get_node_resources()
            if spec.nodes != {"vit_encoder"}
        ]

    def prefix_key_streams(self) -> dict[str, dict[str, PrefixStream]]:
        return {}

    def get_sessions_config(self) -> SessionsConfig:
        """Hold the LLM's KV for the session, not just the request.

        ``max_state`` is in pages of the ``kv`` resource: at BAGEL's 128-token
        pages, 64 pages is ~8k tokens of conversation. A session that runs past
        it is cleared and told so on its next request, which is the loud
        behaviour a test wants.
        """
        return SessionsConfig(
            resources={
                "kv": SessionResourceConfig(
                    max_state=64,
                    overflow_policy=SessionOverflowPolicy.ERROR,
                ),
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
        return super().process_prompt(
            prompt, input_modalities, output_modalities,
            tensors=tensors, prompt_parts=prompt_parts, **kwargs,
        )
