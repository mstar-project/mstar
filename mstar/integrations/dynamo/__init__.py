"""Serve an M* deployment as a NVIDIA Dynamo backend.

One M* deployment (conductor + GPU workers + its own control mesh and
tensor transport) presents to Dynamo as a single worker instance with one
endpoint, the way a TP=N vLLM worker does. Dynamo's frontend parses and
validates OpenAI requests and forwards the raw body to Text-input
backends; the per-model OpenAI adapters translate them onto the same
submission path the native surfaces use. Replication for throughput means
more M* instances — the Dynamo router balances across them.

Requires the ``dynamo`` extra (``pip install mstar[dynamo]``). Launch::

    python -m mstar.integrations.dynamo --config configs/<model>.yaml \\
        --model-path <hf snapshot dir> [--namespace mstar ...]

Nothing outside this package imports ``dynamo``; deleting the package (or
skipping the extra) leaves the standalone server untouched.
"""
