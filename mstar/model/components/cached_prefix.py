"""A prefill that starts past a cached prefix, for the DSA models (GLM-5.2, GLM-5.3)."""
from __future__ import annotations

from collections.abc import Iterable

# Request state: the prompt tokens the prefix cache served, from the engine's cut of the
# prefill's inputs until the submodule removes it.
CACHED = "cached"


class EagerPastCachedPrefix:
    """Mixin, listed before ``ARNodeSubmodule``, for a submodule whose config has
    ``dsa_long_context``.

    DSA scores a captured prefill's rows within its bucket, and rows past a cached prefix see
    more keys than that. So under ``dsa_long_context`` a prefill with such a request takes a
    capture key no bucket has and runs eager; a dense prefill stays on its captures.
    """

    def split_inputs(self, graph_walk, fwd_info, inputs, start, end):
        if start > 0:
            self.request_state(fwd_info.rid_handle).add(CACHED, start)
        return super().split_inputs(graph_walk, fwd_info, inputs, start, end)

    def cg_key_info(self, graph_walk: str, per_request_info: dict, per_request_input_metadata=None,
                    **kwargs) -> str | None:
        return self._cg_key(graph_walk, per_request_info)

    def _cg_key(self, graph_walk: str, rids: Iterable) -> str | None:
        """The step's capture key: every capture's (None), or one no capture has. Also what
        ``declare_step`` puts on the step."""
        if not (self.config.dsa_long_context and graph_walk == "prefill"):
            return None
        states = self.request_states
        if any(rid in states and states[rid].get(CACHED) for rid in rids):
            return "eager"
        return None
