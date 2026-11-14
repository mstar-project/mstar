"""What a model declares for compressed sparse attention (CSA): its config, spec and step.

Kept free of the manager so a submodule can declare a step without importing it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# the latent caches' aux planes (``PagedKVConfig.aux_planes``) that hold the index keys
INDEX_PLANES = "csa_index"


@dataclass(frozen=True)
class Fp4Rows:
    """Rows of a cache plane rounded to FP4 and stored in its bytes: E2M1 values two to a
    byte, then a scale byte per ``block`` values (E4M3 with ``e4m3_scale``, else a power of two
    as E8M0), the grid of ``mx_round.round_fp4`` with the same arguments. A model trained with
    FP4 caches declares it and writes its rows unrounded: the resource rounds them as it
    stores them, in ``nbytes(width)`` bytes a row where bf16 takes two a value, and reads back
    the rounded values."""

    block: int
    e4m3_scale: bool = False

    def nbytes(self, width: int) -> int:
        if width % self.block or width % 2:
            raise ValueError(f"FP4 rows of {width} values are not whole bytes and {self.block}-value scale blocks")
        return width // 2 + width // self.block


@dataclass
class CSAConfig:
    # the paged latent caches by pooling ratio. A cache of ratio r keeps one row per r tokens,
    # a group's latent, written when the group's last token arrives; it has a plane per
    # compressed source of that ratio and each plane's index keys in aux_planes[INDEX_PLANES].
    caches: Mapping[int, str]
    # the compressed rows a query attends: its top ``topk`` by the indexer's score
    topk: int
    # the softmax scale every attention runs at
    sm_scale: float
    # the indexer's key width, one INDEX_PLANES row
    index_head_dim: int
    # the latent's width, which the queries share
    latent_dim: int
    # the caches' rows rounded to FP4 and kept in its bytes (``Fp4Rows``); None keeps what the
    # layers write, in the cache's dtype. A cache is typed by its node's dtype, so an FP4 cache
    # declares its row widths in those elements (``kv_lora_rank``, the aux planes' ``width``)
    # and the resource reads them as bytes.
    latent_rows: Fp4Rows | None = None
    key_rows: Fp4Rows | None = None
    # the softmax runs over a row's entries a block of this many at a time, each block's
    # probabilities rounded to the queries' dtype before they weight the entries: the arithmetic
    # of the kernel DeepSeek-V4.1 was served with, which its logits in bf16 depend on
    softmax_block: int = 64
    # bytes of fp32 scores one selection launch holds, and of gathered rows one attention launch
    score_bytes: int = 256 << 20
    attend_bytes: int = 256 << 20
    # score a decode step's rows with the Triton kernel that reads FP4 keys in place, where it
    # applies (bf16 on CUDA); off, every step takes the torch path, the kernel's reference
    score_kernel: bool = True
    # run a step's softmax on the Triton kernel, where it applies (bf16 on CUDA). Its dots sum
    # in another order than the torch path's, so a result can differ in its last bf16 place.
    attend_kernel: bool = False

    def __post_init__(self):
        self.caches = dict(self.caches)
        if not self.caches or any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in self.caches):
            raise ValueError(f"CSAConfig.caches maps pooling ratios (>= 1) to caches, got {self.caches!r}")
        if len(set(self.caches.values())) != len(self.caches):
            raise ValueError(f"CSAConfig.caches names one cache for two ratios: {self.caches!r}")

    def row_bytes(self, itemsize: int) -> tuple[int, int]:
        """Bytes a latent row and an index key row take, in a cache of ``itemsize``-byte values."""
        latent = self.latent_dim * itemsize if self.latent_rows is None else self.latent_rows.nbytes(self.latent_dim)
        key = (self.index_head_dim * itemsize if self.key_rows is None
               else self.key_rows.nbytes(self.index_head_dim))
        return latent, key


@dataclass
class CSASpec(NodeResourceSpec):
    config: CSAConfig

    def depends_on(self) -> set[str]:
        return set(self.config.caches.values())

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.csa.manager import CSAManager

        return CSAManager

    def apply_yaml_overrides(self, score_bytes: int | None = None, attend_bytes: int | None = None,
                             score_kernel: bool | None = None, attend_kernel: bool | None = None):
        """What the box decides, not the model: the memory a launch's scores and rows may take,
        and which of the two Triton kernels run."""
        for name, value in (("score_bytes", score_bytes), ("attend_bytes", attend_bytes)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            setattr(self.config, name, value)
        for name, value in (("score_kernel", score_kernel), ("attend_kernel", attend_kernel)):
            if value is None:
                continue
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}")
            setattr(self.config, name, value)


@dataclass(frozen=True)
class CSAStep(ResourceStep):
    """The step's segments are its tokens (the submodule's own). The caches advance by the
    rows ``CSAManager.kv_segments`` derives from them."""

    # a captured step's rows score each cache over this many tokens of context, the width its
    # graph was recorded at, which every request of the step must fit; None: all a cache holds
    context: int | None = None
