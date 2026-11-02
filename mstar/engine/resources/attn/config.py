"""What a model declares about attention: its backend, its spec, its step.

Kept free of the managers and their kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, get_args

from mstar.engine.resources.kv.config import KVConfig, KVLayout, PagedKVConfig
from mstar.engine.resources.spec import CombinedSplit, NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource
    from mstar.engine.resources.sequence_shard import SeqShardSpec

logger = logging.getLogger(__name__)


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


@dataclass
class AttentionConfig:
    kv_cache: str # name of the KV cache
    backend: AttnBackend = AttnBackend.AUTO
    flashinfer_backend: str = "auto"
    # softmax scale; None means the kernel default (head_dim ** -0.5). MLA must
    # pass the *original* qk head dim's scale, not the latent width's. Only the MLA
    # backend reads it; the others refuse it at build
    sm_scale: float | None = None
    # FlashInfer only: combined walks whose captured steps run their leading
    # one-token rows on a decode half and their prompt rows on a prefill
    # wrapper sized to them (`FlashInferMixedWrapper`); the engine orders
    # every step of those walks decode first
    splits: tuple[CombinedSplit, ...] = ()
    # FlashInfer only: run one-token decode rows on the tensor-core kernel.
    # False picks the CUDA-core decode kernel, which can win for MHA and
    # narrow GQA, where the tensor-core kernel pads few query heads per KV
    # head out to a tile.
    decode_tensor_cores: bool = True
    # FlashInfer only: run captured one-token decode rows on trtllm-gen's fmha
    # kernel where one exists for this GPU and geometry (`FmhaDecodeWrapper`),
    # which also drops the paged kernel's per-step plan. Eager steps and prompt
    # rows stay on the paged kernels.
    decode_fmha: bool = False
    # Each query attends to the last ``sliding_window`` positions, its own included, and the
    # pool keeps about that window of K/V per request for this attention's layers and frees
    # the rest (`derive_kv_groups`).
    sliding_window: int | None = None
    # The KV layers a ``sliding_window`` attention runs, which a window shorter than the pool's
    # ``max_seq_len`` needs; the pool's other layers keep every token.
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.sliding_window is not None and (
            type(self.sliding_window) is not int or self.sliding_window <= 0
        ):
            raise ValueError("sliding_window must be a positive integer or None")
        if self.layers is not None:
            self.layers = tuple(self.layers)
            if self.sliding_window is None:
                raise ValueError("layers names the layers a sliding_window keeps; it needs one")
            if not self.layers or len(set(self.layers)) != len(self.layers):
                raise ValueError(f"layers must be distinct layers, at least one; got {self.layers}")


SpExchange = Literal["none", "ulysses", "all_gather", "ring"]
SP_EXCHANGES: tuple[SpExchange, ...] = get_args(SpExchange)


@dataclass
class AttentionSpec(NodeResourceSpec):
    config: AttentionConfig
    # the sequence shard over its nodes, if any (`derive`)
    shard_key: str | None = field(default=None, init=False, repr=False)
    # how a split step's labels reach this attention: a deployment's choice, set only by its
    # `resources:` block (`apply_yaml_overrides`), never by a model. "ulysses" over KV stored by
    # head, "all_gather" or "ring" over KV stored by position (`context_parallel`), "none" for no
    # split steps; None takes the storage's default (`resolve_exchange`)
    sp_exchange: SpExchange | None = field(default=None, init=False, repr=False)
    # `sp_exchange` resolved against the pool's storage (`derive`)
    split_exchange: SpExchange = field(default="ulysses", init=False, repr=False)

    def depends_on(self) -> set[str]:
        # the wrappers are planned against the cache's geometry, and it has to
        # be the same KVConfig the KV resource holds
        keys = {self.config.kv_cache}
        if self.shard_key is not None:
            keys.add(self.shard_key)
        return keys

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        from mstar.engine.resources.sequence_shard import take_shard_edge

        problems = take_shard_edge(self, specs)
        self.split_exchange = self.resolve_exchange(specs)
        # a deployment's own setting, so refused at every SP size where nothing reads it (one SP
        # rank: `sequence_parallel_refusals`) or it doesn't match the pool's storage
        if self.sp_exchange is not None and self.shard_key is None:
            problems.append(
                f"sp_exchange {self.sp_exchange!r} picks how a split step reaches it, and its nodes declare no "
                "SeqShardSpec, so their steps never split: drop it"
            )
        by_position = {"ulysses": False, "all_gather": True, "ring": True}.get(self.sp_exchange)
        if by_position is not None and bool(specs[self.config.kv_cache].places_positions) != by_position:
            storage = "by position (context_parallel: true)" if by_position else "by head (context_parallel off)"
            problems.append(f"sp_exchange {self.sp_exchange!r} needs KV {self.config.kv_cache!r} stored {storage}")
        return problems

    def resolve_exchange(self, specs: Mapping[str, NodeResourceSpec]) -> SpExchange:
        """``sp_exchange``, or where unset its pool's storage's default: ``ulysses`` by head, and
        ``none`` by position, where splitting a step's tokens is the deployment's to opt into."""
        if self.sp_exchange is not None:
            return self.sp_exchange
        return "none" if specs[self.config.kv_cache].places_positions else "ulysses"

    def split_step_refusals(self, node: str, shard: "SeqShardSpec", tp: int, sp: int) -> list[str]:
        # a paged pool serves whatever labels a step declares, so nothing rules out a split one
        # reaching an attention that takes none
        if self.split_exchange != "none":
            return []
        why = "sp_exchange unset, none over KV stored by position" if self.sp_exchange is None else "sp_exchange none"
        return [
            f"attention {self.resource_key!r} takes no split steps ({why}), and node {node!r}'s other attentions "
            "split its steps: set its sp_exchange to all_gather or ring, or none on the others"
        ]

    def combined_splits(self) -> tuple[CombinedSplit, ...]:
        return self.config.splits

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.attn.base import AttentionManager

        return AttentionManager

    def apply_yaml_overrides(
        self,
        backend: str | AttnBackend | None = None,
        flashinfer_backend: str | None = None,
        decode_tensor_cores: bool | None = None,
        decode_fmha: bool | None = None,
        sp_exchange: SpExchange | None = None,
    ):
        """Which kernel to run is the deployment's call as much as the
        model's — an image that cannot build FA3 pins FA2 here. So is how a
        split step reaches it over SP ranks (``sp_exchange``).

        Cache geometry is not repeated here: it belongs to the KV resource
        this spec depends on, and is tuned under that resource's own block.
        """
        if sp_exchange is not None:
            if sp_exchange not in SP_EXCHANGES:
                raise ValueError(f"sp_exchange must be one of {', '.join(SP_EXCHANGES)}; got {sp_exchange!r}")
            self.sp_exchange = sp_exchange
        if backend is not None:
            self.config.backend = AttnBackend(backend)
        if flashinfer_backend is not None:
            self.config.flashinfer_backend = flashinfer_backend
        if decode_tensor_cores is not None:
            self.config.decode_tensor_cores = _yaml_bool("decode_tensor_cores", decode_tensor_cores)
        if decode_fmha is not None:
            self.config.decode_fmha = _yaml_bool("decode_fmha", decode_fmha)


def derive_kv_groups(name: str, cfg: KVConfig, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
    """Pool ``name``'s groups (`PagedKVConfig.keep_window`), from the attentions over it whose
    ``sliding_window`` is shorter than its ``max_seq_len`` (a longer one keeps every token),
    and what about them no pool can serve, before anything builds or sizes from it; a pool
    with a problem keeps every token. A pool that keeps a window doesn't prefix cache yet, so
    its ``prefix_cache`` goes off. Run by `KVSpec.derive`."""
    readers = {key: spec for key, spec in specs.items()
               if isinstance(spec, (AttentionSpec, CrossAttentionSpec)) and spec.config.kv_cache == name}
    paged = isinstance(cfg, PagedKVConfig) and cfg.layout == KVLayout.NHD
    problems: list[str] = []
    window_of: dict[int, int] = {}  # layer -> its window
    reads_full = False
    for key, reader in readers.items():
        window = getattr(reader.config, "sliding_window", None)
        if window is not None and not paged:
            problems.append(f"attention {key!r} declares a sliding_window, which only a paged NHD pool keeps")
        elif isinstance(reader, AttentionSpec) and (window is None or window >= cfg.max_seq_len):
            reads_full = True
        elif window is not None and reader.config.layers is None:
            problems.append(
                f"attention {key!r} declares a sliding_window without the layers it runs (AttentionConfig.layers)"
            )
        elif window is not None:
            for layer in reader.config.layers:
                if not 0 <= layer < cfg.num_layers:
                    problems.append(f"attention {key!r} declares layer {layer}, past the pool's {cfg.num_layers}")
                elif layer in window_of:
                    problems.append(f"layer {layer} is declared by two attentions")
                else:
                    window_of[layer] = window
    if window_of:
        windows = sorted(set(window_of.values()))
        if len(windows) > 1:
            # a derived page is whole for one window's layers and the rest
            problems.append(f"keeps one window, not the {windows} its attentions declare")
        if not reads_full and len(window_of) < cfg.num_layers:
            problems.append("keeps every token of the layers outside its window, which no attention reads")
        for key, reader in readers.items():
            if isinstance(reader, CrossAttentionSpec):
                problems.append(f"keeps a window, whose pages cross attention {key!r} doesn't read")
            elif reader.config.backend not in (AttnBackend.AUTO, AttnBackend.FLASHINFER):
                problems.append(
                    f"keeps a window, whose pages only the flashinfer backend reads, "
                    f"not attention {key!r}'s {reader.config.backend.value}"
                )
    if not paged:
        return problems
    window, layers = (windows[0], tuple(sorted(window_of))) if window_of and not problems else (None, ())
    try:
        cfg.keep_window(window, layers)
    except ValueError as e:
        problems.append(str(e))
        cfg.keep_window(None, ())
        return problems
    if window is not None and cfg.prefix_cache:
        if cfg.prefix_cache_given:
            problems.append("keeps a sliding window, which doesn't prefix cache yet; drop its prefix_cache: true")
            return problems
        logger.info("KV %s keeps a sliding window, which doesn't prefix cache yet: its prefix cache is off", name)
        cfg.prefix_cache = False
    return problems


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


def refuse_context_only(step: AttentionStep, manager: str) -> None:
    """For the attention resources that don't read ``context_only``: planned without it,
    the step's queries would attend to the tokens it appends as well."""
    if step.context_only:
        raise ValueError(
            f"{manager} does not read AttentionStep.context_only; only the "
            f"{AttnBackend.FLASHINFER_MLA.value!r} backend does"
        )
