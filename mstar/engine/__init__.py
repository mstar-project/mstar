from mstar.engine.torch_config import apply_torch_config

# Reaches only the importing thread. Threads that compile later apply it
# themselves; see mstar/engine/torch_config.py.
apply_torch_config()
