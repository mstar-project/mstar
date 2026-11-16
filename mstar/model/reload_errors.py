"""Errors shared by model reload hooks and checkpoint inspection."""


class WeightReloadRefusedError(ValueError):
    """The checkpoint cannot be reloaded in place; nothing was copied."""
