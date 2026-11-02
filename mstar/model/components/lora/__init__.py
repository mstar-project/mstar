"""LoRA shared across models: the linears that take a per-request adapter's delta from the
engine's adapter pool (``capable``), and a node's LoRA spec declared from its own modules
and bound to them (``modules``). Checkpoint parsing is the engine's
(``mstar.engine.resources.lora.checkpoint``).

The static load-time merge lives with the DiT scaffold (``components/diffusion/lora.py``).
"""
