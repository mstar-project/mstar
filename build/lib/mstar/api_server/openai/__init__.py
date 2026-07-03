"""OpenAI-compatible API layer for mstar.

Sits alongside the native ``/generate`` endpoint and reuses the same request
path (``APIServer.submit_request`` / ``collect_results`` / ``iter_result_chunks``).
Per-model translation lives in :mod:`mstar.api_server.openai.adapters`; the HTTP
endpoints in :mod:`mstar.api_server.openai.router` stay model-agnostic.
"""
