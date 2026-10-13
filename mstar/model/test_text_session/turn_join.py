"""The join between one turn of a session and the next, for a ChatML-style LLM.

A resumed turn's template opens a user block, so the KV it continues from has to
end where a one-shot transcript of the conversation would: the previous reply,
then ``<|im_end|>\\n``. What the KV actually ends on depends on how that reply
stopped. The token that stopped it (a stop token for a finished reply, the last
token for one ``max_output_tokens`` cut) is in the KV only when a speculative
step overshot the stop (``OVERSHOT_LAST_ITER``); otherwise decode stopped on it
without feeding it back. So the resumed turn's first prefill puts back what is
missing:

=======================  =====================  ==========================
last turn ended on       token in the KV        token not in the KV
=======================  =====================  ==========================
the close token          ``\\n``                 ``<|im_end|>\\n``
another stop token       ``<|im_end|>\\n``       ``tok<|im_end|>\\n``
a cut                    nothing                ``tok``
=======================  =====================  ==========================

A cut reply is left open rather than closed like a finished one: the KV holds
the whole reply the client saw and nothing after it, so a next turn asking to
continue reads as the reply being interrupted, not finished.

Shared by every ``test_text_session`` variant: a mixin over the base model's LLM
submodule, which the model configures with its tokenizer's ids.
"""

import logging

import torch

from mstar.model.submodule_base import OVERSHOT_LAST_ITER

logger = logging.getLogger(__name__)

# What the LLM's session state keeps of the turn that last ran in it: the token
# that stopped its decode loop.
_LAST_TOKEN = "last_token"

# The walk a turn's text prompt prefills on
PREFILL_TEXT = "prefill_text"


class TurnJoinMixin:
    """Records how each turn ended and renders the join on the next one.

    Goes before the base LLM submodule in the bases, which must define
    ``check_stop``, ``prepare_inputs`` and ``postprocess``. The model sets the
    three token attributes when it builds the submodule.
    """

    # `<|im_end|>`: what closes an assistant turn in the template
    turn_close_id: int | None = None
    # Every id that finishes a reply; includes `turn_close_id`
    turn_stop_ids: frozenset[int] = frozenset()
    # The newline after `<|im_end|>`
    newline_ids: list[int] = []

    # -- recording how the turn ended ---------------------------------------

    def _record_turn_end(self, request_info, token: int) -> None:
        session = request_info.session
        if session is not None:
            self.session_state(session.session_id).add(_LAST_TOKEN, token)

    def check_stop(self, request_id, request_info, outputs):
        stops = super().check_stop(request_id, request_info, outputs)
        # Here, not in `postprocess`: an overshooting step runs `postprocess`
        # too, but its outputs are dropped before the stop check, so this sees
        # the turn's real last token exactly once.
        if stops:
            self._record_turn_end(request_info, outputs["new_token"][0].item())
        return stops

    def check_stop_batched(self, request_ids, request_infos, host_rows):
        stops = super().check_stop_batched(request_ids, request_infos, host_rows)
        if not stops:
            return stops
        tokens = host_rows.buffers["new_token"]
        values = tokens.reshape(tokens.shape[0], -1)[:, 0].tolist()
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids)}
        for rid in stops:
            self._record_turn_end(request_infos[rid], values[row_of[rid]])
        return stops

    # -- rendering the join -------------------------------------------------

    def turn_join(self, fwd_info) -> list[int]:
        """The tokens a resumed turn's first prefill goes after."""
        session = fwd_info.session
        if session is None or not session.resumed:
            return []
        state = self.session_states.get(session.session_id)
        if state is None or _LAST_TOKEN not in state.kwargs:
            return []
        token = state.kwargs[_LAST_TOKEN]
        join = []
        if not state.kwargs.get(OVERSHOT_LAST_ITER, False):
            join.append(token)
        if token == self.turn_close_id:
            join += self.newline_ids
        elif token in self.turn_stop_ids:
            # finished, but on a stop token the template never closes with
            join += [self.turn_close_id, *self.newline_ids]
        return join

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        node_inputs = super().prepare_inputs(graph_walk, fwd_info, inputs, **kwargs)
        if graph_walk != PREFILL_TEXT:
            return node_inputs
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
            request_info.graph_walk == PREFILL_TEXT
            and session is not None and session.resumed
        ):
            state = self.session_states.get(session.session_id)
            if state is not None:
                for key in (_LAST_TOKEN, OVERSHOT_LAST_ITER):
                    state.kwargs.pop(key, None)
