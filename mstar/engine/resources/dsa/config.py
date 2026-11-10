"""What a model declares for DeepSeek sparse attention (DSA): its config, spec and step.

Kept free of the manager so a submodule can declare a step without importing it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# the MLA cache's aux planes (``PagedKVConfig.aux_planes``) that hold the index keys
INDEX_PLANES = "dsa_index"


@dataclass
class DSAConfig:
    # the paged MLA cache: the step's rows come from its plan; its pages hold the latents and,
    # in aux_planes[INDEX_PLANES], the index keys
    kv_cache: str
    # the keys a query attends once it sees more: its top ``topk`` by the indexer's score
    topk: int
    # tokens per pooled key (GLM-5.3's k-pool indexer); 1 scores every token's key
    kpool: int = 1
    # the absorbed MLA's softmax scale, which every sparse attention runs at
    sm_scale: float | None = None
    # attend densely while no row sees more than topk keys (there, the same computation): such
    # a step plans no sparse attention
    dense_below_topk: bool = False
    # the indexer's key width: an INDEX_PLANES row is one key wide, or with kpool > 1 three
    # (key, gate, the key of the pool ending there)
    index_head_dim: int | None = None
    # bytes of fp32 scores an eager prefill's selection holds per launch: a chunk of rows is
    # as wide as its last row's pools
    prefill_score_bytes: int = 256 << 20
    # a prefill's selection shards across TP ranks from this many rows per rank
    shard_min_rows: int = 64
    # MiB each FlashInfer workspace of its sparse plans starts at (one per walk and capture
    # slot, one for eager steps); a plan that needs more grows it until captures are sealed
    workspace_mb: int = 128


@dataclass
class DSASpec(NodeResourceSpec):
    config: DSAConfig

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.dsa.manager import DSAManager

        return DSAManager

    def apply_yaml_overrides(self, workspace_mb: int | None = None,
                             prefill_score_bytes: int | None = None,
                             shard_min_rows: int | None = None):
        """What the box decides, not the model: the memory its plans and score chunks may
        take, and the rows from which splitting a selection across ranks pays."""
        for name, value in (("workspace_mb", workspace_mb),
                            ("prefill_score_bytes", prefill_score_bytes),
                            ("shard_min_rows", shard_min_rows)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            setattr(self.config, name, value)


@dataclass(frozen=True)
class DSAStep(ResourceStep):
    # a decode step whose rows may sit before the positions the host planned (an MTP verify
    # block the device places after its last accepted rows, ``DSAManager.move_rows``): a
    # captured plan is made for the longest each row can be
    mtp_trims: bool = False
