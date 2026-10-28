"""How a node's routed experts split across its node group: TP or EP.

* :class:`MoeParallelConfig` is a node group's yaml ``ep_config`` section.
* :func:`instantiate_moe_block` builds the matching MoE block for a layer.
* :class:`MoeParallelMixin` gives the owning model padding masks and EP load logging.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.moe import ExpertParallelSparseMoeBlock, ParallelSparseMoeBlock

if TYPE_CHECKING:
    from mstar.model.submodule_base import ModelInputsFromEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoeParallelConfig:
    """How a node's routed experts split across its node group.

    Read from the node group's ``ep_config`` in the deployment yaml.
    Attention and dense MLPs stay TP-sharded either way.
    """

    # "tp": every rank holds a slice of every expert; "ep": whole experts per rank.
    parallel: str = "tp"
    # Comm group EP spans. Only "tp", the node group's TP ranks, exists today.
    ep_group: str = "tp"
    # Log per-rank EP slot counts every N steps; 0 is off.
    log_expert_load_every: int = 0
    # Skip CUDA-graph padding rows in the routed experts; TP and EP alike.
    mask_padding: bool = False

    def __post_init__(self):
        if self.parallel not in ("tp", "ep"):
            raise ValueError(f"ep_config.parallel must be 'tp' or 'ep', got {self.parallel!r}")
        if self.ep_group != "tp":
            raise ValueError(f"ep_config.ep_group must be 'tp', got {self.ep_group!r}")
        if self.log_expert_load_every < 0:
            raise ValueError("ep_config.log_expert_load_every must be >= 0")
        if self.parallel != "ep" and self.log_expert_load_every:
            raise ValueError("ep_config.log_expert_load_every needs parallel: ep")

    @classmethod
    def from_yaml(cls, section: dict | None) -> "MoeParallelConfig":
        section = dict(section or {})
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown ep_config keys: {sorted(unknown)}")
        return cls(**section)

    def resolve_ep_group(self, tp_group: CommGroup | None) -> CommGroup | None:
        # Only one choice for now; new mesh axes would be selected here.
        return tp_group


def instantiate_moe_block(
    *,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    moe_intermediate_size: int,
    norm_topk_prob: bool = True,
    router: nn.Module | None = None,
    comm_group: CommGroup | None = None,
    moe_parallel: MoeParallelConfig | None = None,
    **ep_options: Any,
) -> ParallelSparseMoeBlock | ExpertParallelSparseMoeBlock:
    """The routed-expert block ``moe_parallel`` asks for; TP when None.

    ``comm_group`` is the node's TP group; EP picks its group from it.
    ``ep_options`` are options only the EP block has (``pre_experts``,
    ``expert_norm``, ...); a model that passes any needs ``parallel: ep``.
    """
    moe_parallel = moe_parallel or MoeParallelConfig()
    if moe_parallel.parallel == "tp":
        if ep_options:
            raise ValueError(
                f"the MoE block's {sorted(ep_options)} run on whole experts only: "
                "set `ep_config: {parallel: ep}` on the node group"
            )
        return ParallelSparseMoeBlock(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            router=router,
            comm_group=comm_group,
        )
    ep_group = moe_parallel.resolve_ep_group(comm_group)
    return ExpertParallelSparseMoeBlock(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=norm_topk_prob,
        router=router,
        comm_group=ep_group,
        # Counts are identical on every rank, so only rank 0 keeps them.
        track_expert_load=(
            moe_parallel.log_expert_load_every > 0
            and (ep_group is None or ep_group.rank == 0)
        ),
        **ep_options,
    )


class MoeParallelMixin:
    """Padding masks and EP load logging for a module whose layers come from
    :func:`instantiate_moe_block`.

    Mix into the top-level ``nn.Module`` and call :meth:`init_moe_parallel`
    in its ``__init__`` after building the layers.
    """

    def init_moe_parallel(self, moe_parallel: MoeParallelConfig | None) -> None:
        moe_parallel = moe_parallel or MoeParallelConfig()
        self.mask_padding = moe_parallel.mask_padding
        self._log_expert_load_every = moe_parallel.log_expert_load_every
        self._steps_since_load_log = 0
        self._discard_expert_load = False

    def moe_token_valid(
        self, hidden_states: torch.Tensor, num_valid_tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """``(tokens,)`` bool for the MoE's ``token_valid``; None when not masking.

        ``num_valid_tokens`` is a ``(1,)`` device tensor, so a replay can
        change it without recapturing.
        """
        if not self.mask_padding or num_valid_tokens is None:
            return None
        positions = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return positions < num_valid_tokens

    def moe_step_inputs(
        self, engine_inputs: "ModelInputsFromEngine", num_tokens: int, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Call from the node's ``preprocess``: logs the EP load when due, and
        returns the forward's ``num_valid_tokens`` under ``mask_padding``
        (else nothing). Real rows lead a replay's bucket; a fill kernel, so no
        H2D copy."""
        self.maybe_log_expert_load(synthetic=engine_inputs.synthetic)
        if not self.mask_padding:
            return {}
        n = engine_inputs.real_seq_len
        return {"num_valid_tokens": torch.full(
            (1,), num_tokens if n is None else n, dtype=torch.int64, device=device,
        )}

    def _tracked_moe_blocks(self) -> list[tuple[str, ExpertParallelSparseMoeBlock]]:
        return [
            (name, module) for name, module in self.named_modules()
            if isinstance(module, ExpertParallelSparseMoeBlock) and module.track_expert_load
        ]

    @torch.compiler.disable
    def maybe_log_expert_load(self, synthetic: bool = False) -> None:
        """Before each forward: after every N forwards, log EP load and reset it.

        Call from eager code. ``synthetic`` marks a warmup or capture step;
        its dummy routing is discarded before the next real step. Without
        ``mask_padding``, counts under CUDA graphs include padding rows.
        """
        if not self._log_expert_load_every:
            return
        if synthetic:
            self._discard_expert_load = True
            return
        if self._discard_expert_load:
            self._discard_expert_load = False
            self._steps_since_load_log = 0
            for _, block in self._tracked_moe_blocks():
                block.expert_load.zero_()
        # Counts forwards already run; this step's forward comes after.
        steps = self._steps_since_load_log
        self._steps_since_load_log += 1
        if steps < self._log_expert_load_every:
            return
        self._steps_since_load_log = 1
        blocks = self._tracked_moe_blocks()
        if not blocks:
            return
        world_size = blocks[0][1].comm_group.world_size
        # (blocks, experts) -> (blocks, ranks)
        per_expert = torch.stack([block.pop_expert_load() for _, block in blocks])
        per_rank = per_expert.view(len(blocks), world_size, -1).sum(-1)

        def imbalance(counts: torch.Tensor) -> float:
            mean = counts.double().mean().item()
            return counts.max().item() / mean if mean else 1.0

        totals = per_rank.sum(0)
        worst = max(range(len(blocks)), key=lambda j: imbalance(per_rank[j]))
        hottest = per_expert.sum(0).topk(min(4, per_expert.shape[1]))
        logger.info(
            "%s EP load over %d steps, %d MoE layers: per-rank slots %s "
            "(max/mean %.2f); worst layer %s %s (max/mean %.2f); hottest experts %s",
            type(self).__name__, steps, len(blocks), totals.tolist(), imbalance(totals),
            blocks[worst][0], per_rank[worst].tolist(), imbalance(per_rank[worst]),
            dict(zip(hottest.indices.tolist(), hottest.values.tolist(), strict=True)),
        )
