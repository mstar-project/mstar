"""A text-only deployment of BAGEL's LLM that holds its KV across a session.

The point of this model is the session semantics, not the model: it is the
smallest thing that exercises them end to end. A client starts a session, sends
a turn, resumes with the next turn, and the second request continues from the
KV the first one built instead of re-reading the conversation.

It is BAGEL's LLM and tokenizer with everything else taken away — only the
``prefill_text`` and ``decode`` walks are declared, so the ViT, the VAE and the
image walks are not part of the deployment. Subclassing rather than copying
keeps it honest: the LLM path it tests is the one BAGEL actually serves.

It renders each turn in its own role block, and a resuming turn without the
system block: that turn is appended to a KV which already holds the
conversation, so re-rendering the block would introduce the model to itself
again mid-conversation. The override lives here rather than in ``BagelModel`` so
the models serving chat today keep their current rendering; PR #340 is bringing
per-turn role blocks to BAGEL itself.

Cross-request prefix reuse is off here. Both features share the page pool and
coexist fine (a resumed request steps out of the index on its own), but a test
model should exercise one thing, and a prefix hit would otherwise hide a session
that failed to carry its state over.
"""

import logging

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.graph.base import GraphSection
from mstar.model.bagel.bagel_model import BagelModel
from mstar.model.bagel.submodules import LLMSubmodule
from mstar.model.base import PrefixStream
from mstar.model.multimodal import TEXT
from mstar.model.sessions import (
    SessionResourceConfig,
    SessionsConfig,
)
from mstar.model.submodule_base import OVERSHOT_LAST_ITER

logger = logging.getLogger(__name__)

# The walks a text turn takes. The rest of BAGEL's graph needs the encoders.
TEXT_WALKS = ("prefill_text", "decode")


# What the LLM's session state keeps of the turn that last ran in it: the token
# that stopped its decode loop, and whether that token was the EOS.
_LAST_TOKEN = "last_token"
_LAST_TOKEN_IS_EOS = "last_token_is_eos"


class TextSessionLLMSubmodule(LLMSubmodule):
    """BAGEL's LLM, rendering the join between one turn and the next.

    A resumed turn's template opens a user block, so the KV it continues from
    has to end where a one-shot transcript of the conversation would: the
    previous reply, then ``<|im_end|>\n``. What the KV actually ends on depends
    on how that reply stopped. The token that stopped it (EOS for a finished
    reply, the last token for one ``max_output_tokens`` cut) is in the KV only
    when a speculative step overshot the stop (``OVERSHOT_LAST_ITER``); otherwise
    decode stopped on it without feeding it back. So the resumed turn's first
    prefill puts back what is missing:

    ======================  ===================  ========================
    last turn ended on      token in the KV      token not in the KV
    ======================  ===================  ========================
    EOS                     ``\n``               ``<|im_end|>\n``
    a cut                   nothing              ``tok``
    ======================  ===================  ========================

    A cut reply is left open rather than closed like a finished one: the KV
    holds the whole reply the client saw and nothing after it, so a next turn
    asking to continue reads as the reply being interrupted, not finished.
    """

    # Token ids of the newline after ``<|im_end|>``; set by the model, which
    # holds the tokenizer.
    newline_ids: list[int] = []

    def check_stop(
        self, request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        stops = super().check_stop(request_id, request_info, outputs)
        session = request_info.session
        # Here, not in `postprocess`: an overshooting step runs `postprocess`
        # too, but its outputs are dropped before the stop check, so this sees
        # the turn's real last token exactly once.
        if stops and session is not None:
            token = outputs["new_token"][0].item()
            state = self.session_state(session.session_id)
            state.add(_LAST_TOKEN, token)
            state.add(_LAST_TOKEN_IS_EOS, token == self.eos_token_id)
        return stops

    def turn_join(self, fwd_info: CurrentForwardPassInfo) -> list[int]:
        """The tokens a resumed turn's first prefill goes after."""
        session = fwd_info.session
        if session is None or not session.resumed:
            return []
        state = self.session_states.get(session.session_id)
        if state is None or _LAST_TOKEN not in state.kwargs:
            return []
        join = []
        if not state.kwargs.get(OVERSHOT_LAST_ITER, False):
            join.append(state.kwargs[_LAST_TOKEN])
        if state.kwargs[_LAST_TOKEN_IS_EOS]:
            # a finished reply: close the line `<|im_end|>` ended
            join += self.newline_ids
        return join

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        node_inputs = super().prepare_inputs(graph_walk, fwd_info, inputs, **kwargs)
        if graph_walk == "prefill_text":
            join = self.turn_join(fwd_info)
            if fwd_info.session is not None and fwd_info.session.resumed:
                logger.info(
                    "Session %s: resumed turn joins with %s",
                    fwd_info.session.session_id, join,
                )
            if join:
                ids = node_inputs.input_ids
                node_inputs.input_ids = torch.cat([
                    torch.tensor(join, dtype=ids.dtype, device=ids.device), ids,
                ])
                node_inputs.input_seq_len = node_inputs.input_ids.shape[0]
        return node_inputs

    def postprocess(self, request_id, request_info, outputs, **kwargs):
        super().postprocess(request_id, request_info, outputs, **kwargs)
        session = request_info.session
        # Once the first prefill has run, the join is in the KV. Cleared here
        # rather than in `prepare_inputs`: a step refused at admission is
        # prepared again, and must render the join again.
        if (
            request_info.graph_walk == "prefill_text"
            and session is not None and session.resumed
        ):
            state = self.session_states.get(session.session_id)
            if state is not None:
                for key in (_LAST_TOKEN, _LAST_TOKEN_IS_EOS, OVERSHOT_LAST_ITER):
                    state.kwargs.pop(key, None)


class TextSessionModel(BagelModel):
    LLM_SUBMODULE_CLS = TextSessionLLMSubmodule

    # One role block per turn, and no system block on a resuming turn: it is
    # appended to a KV that already holds the conversation.
    OPENING_TURN_TEMPLATE = (
        "<|im_start|>system\n{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    RESUMING_TURN_TEMPLATE = (
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )

    def _create_submodule(self, node_name, device, **kwargs):
        submodule = super()._create_submodule(node_name, device, **kwargs)
        if isinstance(submodule, TextSessionLLMSubmodule):
            submodule.newline_ids = self._encode_text("\n").tolist()
        return submodule

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
                "kv": SessionResourceConfig(max_state=64),
            },
            max_concurrent_sessions=4,
            default_timeout_s=300.0,
            max_timeout_s=1800.0,
        )

    def process_prompt(
        self, prompt, input_modalities, output_modalities,
        tensors=None, prompt_parts=None, session=None, **kwargs,
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
        if prompt is not None:
            template = (
                self.RESUMING_TURN_TEMPLATE
                if session is not None and session.resumed
                else self.OPENING_TURN_TEMPLATE
            )
            return {"text_inputs": [self._encode_text(template.format(
                system_prompt=self.BAGEL_DEFAULT_SYSTEM_PROMPT, prompt=prompt,
            ))]}
        return super().process_prompt(
            prompt, input_modalities, output_modalities,
            tensors=tensors, prompt_parts=prompt_parts, session=session,
            **kwargs,
        )
