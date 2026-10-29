"""What a model declares about attention: its backend, its spec, its step.

Kept free of the managers and their kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import CombinedSplit, NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class AttnBackend(Enum):
    # the platform's default, decided when the resource is built
    AUTO = "auto"
    FLASHINFER = "flashinfer"
    DENSE = "dense"
    FLEX = "flex"
    XPU_PAGED = "xpu_paged"
    # Multi-head Latent Attention over a ``KVLayout.MLA`` cache (FlashInfer's
    # ``BatchMLAPagedAttentionWrapper``); queries are the absorbed latent form
    FLASHINFER_MLA = "flashinfer_mla"
    # ROCm: AITER's Triton unified attention over the paged cache
    AITER = "aiter"


@dataclass
class AttentionConfig:
    kv_cache: str # name of the KV cache
    backend: AttnBackend = AttnBackend.AUTO
    flashinfer_backend: str = "auto"
    # softmax scale; None means the kernel default (head_dim ** -0.5). MLA must
    # pass the *original* qk head dim's scale, not the latent width's. Only the MLA
    # backend reads it; the others refuse it at build
    sm_scale: float | None = None
    # FlashInfer only: combined walks whose captured steps plan their
    # one-token rows on the decode wrapper and their prompt rows on a prefill
    # wrapper sized to them (`FlashInferMixedWrapper`). AITER runs such a step
    # in one launch and ignores them.
    splits: tuple[CombinedSplit, ...] = ()
    # FlashInfer only: run one-token decode rows on the tensor-core kernel.
    # False picks the CUDA-core decode kernel, which can win for MHA and
    # narrow GQA, where the tensor-core kernel pads few query heads per KV
    # head out to a tile.
    decode_tensor_cores: bool = True
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        if self.sliding_window is not None and (
            type(self.sliding_window) is not int or self.sliding_window <= 0
        ):
            raise ValueError("sliding_window must be a positive integer or None")


@dataclass
class AttentionSpec(NodeResourceSpec):
    config: AttentionConfig

    def depends_on(self) -> set[str]:
        # the wrappers are planned against the cache's geometry, and it has to
        # be the same KVConfig the KV resource holds
        return {self.config.kv_cache}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.base import AttentionManager

        return AttentionManager

    def apply_yaml_overrides(
        self,
        backend: str | AttnBackend | None = None,
        flashinfer_backend: str | None = None,
        decode_tensor_cores: bool | None = None,
    ):
        """Which kernel to run is the deployment's call as much as the
        model's — an image that cannot build FA3 pins FA2 here.

        Cache geometry is not repeated here: it belongs to the KV resource
        this spec depends on, and is tuned under that resource's own block.
        """
        if backend is not None:
            self.config.backend = AttnBackend(backend)
        if flashinfer_backend is not None:
            self.config.flashinfer_backend = flashinfer_backend
        if decode_tensor_cores is not None:
            self.config.decode_tensor_cores = _yaml_bool("decode_tensor_cores", decode_tensor_cores)


@dataclass
class CrossAttentionConfig:
    """Cross-attention against a context written once and never extended.

    ``kv_cache`` names the KV resource holding the encoder context;
    ``query_kv_cache`` names the decoder's KV resource, whose plan defines
    this step's query packing. They may be the same resource when the
    context shares the decoder's head config — the context then lives in it
    under its own ``context_label``. They differ when it does not, which is
    the usual case (an encoder's head count rarely matches the decoder's).

    ``query_kv_cache=None`` covers the query side having no KV cache at all
    (nothing is cached across steps on it): the packing then comes off the
    cross-attention step's own segments, one qo entry per segment in
    declared order.
    """
    kv_cache: str  # name of the KV cache holding the context
    query_kv_cache: str | None = None  # KV cache driving the queries, if any
    context_label: str = "context"
    backend: AttnBackend = AttnBackend.AUTO
    flashinfer_backend: str = "auto"
    # FlashInfer only: a step whose rows each query one token runs the decode
    # kernel; this picks its tensor-core or CUDA-core variant (see
    # ``AttentionConfig``)
    decode_tensor_cores: bool = True


@dataclass
class CrossAttentionSpec(NodeResourceSpec):
    config: CrossAttentionConfig

    def depends_on(self) -> set[str]:
        # the context cache, whose head config need not match the decoder's
        keys = {self.config.kv_cache}
        if self.config.query_kv_cache is not None:
            keys.add(self.config.query_kv_cache)
        return keys

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.cross import CrossAttentionManager

        return CrossAttentionManager

    def apply_yaml_overrides(
        self,
        backend: str | AttnBackend | None = None,
        flashinfer_backend: str | None = None,
        decode_tensor_cores: bool | None = None,
    ):
        """Which kernel to run against the context cache; see ``AttentionSpec``."""
        if backend is not None:
            self.config.backend = AttnBackend(backend)
        if flashinfer_backend is not None:
            self.config.flashinfer_backend = flashinfer_backend
        if decode_tensor_cores is not None:
            self.config.decode_tensor_cores = _yaml_bool("decode_tensor_cores", decode_tensor_cores)


def _yaml_bool(name: str, value) -> bool:
    # the string "false" would otherwise read as true
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value


@dataclass(frozen=True)
class AttentionStep(ResourceStep):
    causal: bool = True
    # queries attend to the stream's stored context only, not to the tokens this step appends
    # (a draft block over the context the target has produced so far); the step's own
    # `segments` then carry the query counts, the cache's segments the appended counts.
    # Only the MLA backend reads it; the others refuse it (`refuse_context_only`).
    context_only: bool = False
    # every row of the step carries this many queries (a speculative verify
    # step's k + 1), so a captured bucket may size its kernel for that rather
    # than for one row holding all of the bucket's tokens
    row_queries: int | None = None


def refuse_context_only(step: AttentionStep, manager: str) -> None:
    """For the attention resources that don't read ``context_only``: planned without it,
    the step's queries would attend to the tokens it appends as well."""
    if step.context_only:
        raise ValueError(
            f"{manager} does not read AttentionStep.context_only; only the "
            f"{AttnBackend.FLASHINFER_MLA.value!r} backend does"
        )
