"""Fused MoE Triton launch configs, tuned per GPU model.

Each JSON file under configs/ is named after the MoE sizes and the GPU
(torch.cuda.get_device_name, spaces as underscores). It maps a token count M
to launch params (tiles, num_warps, num_stages, and on AMD waves_per_eu,
matrix_instr_nonkdim and kpack). A call uses the entry with the nearest M.
Sizes without a file use get_default_config.
benchmark/fused_moe/tune_fused_moe.py writes these files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"


class MoEProblem(NamedTuple):
    """Sizes of one MoE layer's expert GEMMs, minus the token count."""

    num_experts: int
    intermediate_size: int  # per expert and rank, i.e. w2.shape[2]
    hidden_size: int
    top_k: int


def config_file_name(problem: MoEProblem, device_name: str) -> str:
    return (
        f"E={problem.num_experts},N={problem.intermediate_size},K={problem.hidden_size},"
        f"topk={problem.top_k},device_name={device_name}.json"
    )


def device_config_name(device: torch.device | int | None = None) -> str | None:
    """The GPU's name as config files spell it, or None without a GPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(device).replace(" ", "_")


def load_tuned_configs(
    problem: MoEProblem, device_name: str | None, config_dir: Path = CONFIG_DIR,
) -> dict[int, dict[str, Any]] | None:
    """The tuned table for ``problem`` on ``device_name``, keyed by token count."""
    if device_name is None:
        return None
    path = config_dir / config_file_name(problem, device_name)
    if not path.is_file():
        return None
    with open(path) as f:
        raw = json.load(f)
    # Non-numeric keys are metadata (the Triton version the table was tuned with).
    table = {int(m): dict(cfg) for m, cfg in raw.items() if m.isdigit()}
    if not table:
        return None
    logger.info("fused MoE: tuned Triton configs from %s", path.name)
    return table


def nearest_config(table: dict[int, dict[str, Any]], num_tokens: int) -> dict[str, Any]:
    """Entry for the token count closest to num_tokens (ties go to the smaller).
    Plain loop instead of min(key=...) so dynamo can trace it."""
    best = None
    for m in sorted(table):
        if best is None or abs(m - num_tokens) < abs(best - num_tokens):
            best = m
    return table[best]
