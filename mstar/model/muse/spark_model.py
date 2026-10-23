"""Meta's Muse Spark Open (816B total, 42B active): Muse Glimmer's ViT in front
of a 62-layer mixture-of-experts LLM.

Glimmer's graph, walks, prompt and vision tower, with Spark's text stack
(``components/spark.py``): the attention normalizes its output, the residual
stream runs in fp32 through gated residuals, and 256 experts (8 a token) run
in a 4096-wide latent, split whole across the LLM's TP ranks.
"""
from __future__ import annotations

import logging

import torch

from mstar.distributed.base import ShardingConfig
from mstar.model.base import node_moe_parallel_configs
from mstar.model.components import MoeParallelConfig, ep_expert_slice_spec
from mstar.model.muse.config import MuseSparkConfig
from mstar.model.muse.glimmer_model import MuseGlimmerModel
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


class MuseSparkModel(MuseGlimmerModel):
    NAME = "Muse Spark Open"
    TEXT_CONFIG = MuseSparkConfig
    REASONING_KWARG = "reasoning_effort"

    def get_default_sharding_config(self) -> ShardingConfig:
        # the LLM's experts split as its node group's ep_config places them
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={}, ep_enabled_nodes={"LLM"})

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuses, before any worker starts, an LLM whose experts would be
        split along their intermediate dim: each expert's output is RMS-normed
        over its whole row before its routing weight applies."""
        node_groups = config.get("node_groups", [])
        if node_moe_parallel_configs(node_groups).get("LLM", MoeParallelConfig()).parallel != "ep":
            raise ValueError(
                f"{config_path}: {self.NAME}'s LLM needs `ep_config: {{parallel: ep}}` on its node "
                "group: each expert's output is RMS-normed over its whole row, which a TP split of "
                "the experts would leave partial on every rank"
            )

    def _create_submodule(
        self, node_name: str, device: str, autocast_dtype: torch.dtype | None, tp_group,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule | None:
        if node_name != "LLM":
            return super()._create_submodule(node_name, device, autocast_dtype, tp_group)
        from mstar.model.muse import weight_loader
        from mstar.model.muse.components.spark import MuseSparkForCausalLM
        from mstar.model.muse.submodules import SparkLLMSubmodule

        dtype = autocast_dtype or torch.bfloat16
        if dtype != torch.bfloat16:
            # the router's split-weight GEMM is exact for bf16 inputs only
            raise ValueError(f"{self.NAME}'s LLM runs in bf16, not autocast_dtype {dtype}")
        model = self._build(
            lambda: MuseSparkForCausalLM(self.config, tp_group, moe_parallel=moe_parallel), dtype, device,
        )
        weight_loader.load_text_weights(
            model, self._resolve_weights(), device=device, slice_spec=ep_expert_slice_spec(model),
        )
        model.requires_grad_(False).eval()
        logger.info("Loaded %s LLM onto %s", self.NAME, device)
        return SparkLLMSubmodule(model, self.config, self.encoders)
