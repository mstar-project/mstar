"""OpenAI-compatible API layer for mminf.

Sits alongside the native ``/generate`` endpoint and reuses the same request
path (``APIServer.submit_request`` / ``collect_results`` / ``iter_result_chunks``).
Per-model translation lives in :mod:`mminf.api_server.openai.adapters`; the HTTP
endpoints in :mod:`mminf.api_server.openai.router` stay model-agnostic.
"""
