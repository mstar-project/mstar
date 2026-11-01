"""The environment knobs each worker reads at startup to pick its wire codec.
Workers inherit the conductor's environment."""

from collections.abc import Iterator

from mstar.communication.codec import default_codec


def check_knobs() -> Iterator[str]:
    try:
        default_codec()
    except ValueError as e:
        yield str(e)
