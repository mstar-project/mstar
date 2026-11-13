"""The OpenAI-compatible layer's Python side.

``mstar-server`` (``rust/server/``) serves the OpenAI routes. What stays here is
the per-model translation (:mod:`mstar.api_server.openai.adapters`, which the
Dynamo integration serves through and the launcher reads facts from), the
request models (:mod:`.protocol`), and the parts the API server answers over
the bridge: transcription windows, the voice list, Smart Turn.
"""
