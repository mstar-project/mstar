"""Environment knobs read by more than one package."""
import os


def device_loopback_enabled() -> bool:
    """``MSTAR_DEVICE_LOOPBACK`` (default 1): whether a node may keep its
    loop-back token on the device (the sampler's slot master) instead of
    routing a tensor per request; also whether the sampler persists the
    last token at all."""
    return os.environ.get("MSTAR_DEVICE_LOOPBACK", "1") == "1"
