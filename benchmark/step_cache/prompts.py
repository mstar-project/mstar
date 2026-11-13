"""The harness's fixed prompts and seeds.

The first eight are the Wan2.2 video benchmark's VBench prompts with the seeds earlier quality tables used
(``1000 + i``), so a run here lines up with them. The rest are written in VBench's style, a few per dimension it
scores, so a setting is not judged on people and bicycles alone.
"""

from __future__ import annotations

from typing import NamedTuple

from benchmark.wan22.video_bench_wan22 import VBENCH_PROMPTS


class Prompt(NamedTuple):
    index: int
    text: str
    seed: int
    dimension: str


_EXTRA: tuple[tuple[str, str], ...] = (
    ("subject consistency", "a golden retriever running along a beach at sunset"),
    ("subject consistency", "a red vintage car driving down a coastal road"),
    ("subject consistency", "an owl turning its head on a branch at night"),
    ("motion", "a waterfall pouring into a clear pool in a forest"),
    ("motion", "a kite flying high above a green field on a windy day"),
    ("motion", "a train crossing a stone bridge over a river"),
    ("human action", "a chef chopping vegetables on a wooden board"),
    ("human action", "a child blowing out candles on a birthday cake"),
    ("human action", "a woman typing on a laptop by a window in the rain"),
    ("scene", "a busy street market at night with hanging lanterns"),
    ("scene", "a mountain lake at dawn with mist over the water"),
    ("scene", "a library with tall shelves and sunlight through high windows"),
    ("multiple objects", "a cat and a dog sleeping side by side on a sofa"),
    ("multiple objects", "two sailboats and a lighthouse on a calm sea"),
    ("multiple objects", "a teapot and three cups on a table by a window"),
    ("color", "a blue umbrella on a wet cobblestone street"),
    ("color", "a field of yellow tulips under a clear sky"),
    ("color", "a green parrot perched on a red wooden fence"),
    ("spatial relationship", "a lamp to the left of a stack of books on a desk"),
    ("spatial relationship", "a bird flying above a boat on a lake"),
    ("spatial relationship", "a backpack under a bench in a park"),
    ("temporal style", "a city skyline from day to night in time lapse"),
    ("temporal style", "a drop of ink spreading in water, in slow motion"),
    ("temporal style", "a campfire at night, the camera slowly zooming in"),
)

_TEXTS = [("subject consistency", text) for text in VBENCH_PROMPTS[:8]] + [(d, t) for d, t in _EXTRA]
PROMPTS: tuple[Prompt, ...] = tuple(
    Prompt(index=i, text=text, seed=1000 + i, dimension=dimension) for i, (dimension, text) in enumerate(_TEXTS)
)
# The regression mode's three: two easy prompts and the one every quality table has found hardest (the bicycle).
CI_PROMPTS: tuple[Prompt, ...] = (PROMPTS[0], PROMPTS[3], PROMPTS[7])


def take(count: int) -> tuple[Prompt, ...]:
    """The first ``count`` prompts."""
    if not 1 <= count <= len(PROMPTS):
        raise ValueError(f"--prompts must be in [1, {len(PROMPTS)}]; got {count}")
    return PROMPTS[:count]
