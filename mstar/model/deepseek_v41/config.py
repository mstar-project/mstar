"""DeepSeek-V4.1 architecture and serving config (text backbone)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from mstar.engine.resources.csa.config import Fp4Rows

# Resource keys the model declares in get_node_resources and the layers bind by; the serving
# YAML tunes them under ``resources:``.
KV_CACHE = "kv_cache"    # paged latents of the compressed sources, a cache per pooling ratio (kv_cache_key)
CSA = "csa"              # compressed sparse attention over those caches and each layer's window
STATE = "state"          # recurrent pool: each layer's window, the compressors' open groups, the n-gram tail
SAMPLER = "sampler"
LABEL = "main"
# STATE blocks
WINDOW = "window"
POOL_KV = "pool_kv"
POOL_SCORE = "pool_score"
NGRAM = "ngram"

# how the routed experts are held (``moe_experts``)
MOE_EXPERTS = ("fp4", "fp8", "dense")


def kv_cache_key(ratio: int) -> str:
    """The cache of the sources that pool ``ratio`` tokens into a latent: a row per ``ratio``
    tokens, the sources' latents in its planes and their index keys beside them."""
    return f"{KV_CACHE}_x{ratio}"


_TEXT_KEYS = (
    "vocab_size", "hidden_size", "moe_intermediate_size", "num_hidden_layers",
    "num_attention_heads", "head_dim", "qk_rope_head_dim", "q_lora_rank", "o_lora_rank",
    "o_groups", "swiglu_limit", "rms_norm_eps", "max_position_embeddings", "rope_theta",
    "n_routed_experts", "n_shared_experts", "num_experts_per_tok", "scoring_func",
    "norm_topk_prob", "routed_scaling_factor", "sliding_window", "compress_rope_theta",
    "index_n_heads", "index_head_dim", "index_topk", "candidate_topk_blocks",
    "candidate_block_size", "hc_mult", "hc_sinkhorn_iters", "hc_eps", "engram_max_ngram_size",
    "engram_vocab_size", "engram_n_heads", "engram_head_dim", "engram_pad_token_id",
    "engram_compressed_vocab_size", "num_nextn_predict_layers",
)


@dataclass
class DeepseekV41Config:
    # --- backbone ---
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    rms_norm_eps: float = 1e-20
    max_position_embeddings: int = 1_048_576

    # --- attention: one latent per token that is both key and value ---
    head_dim: int = 512            # the latent's width, and each head's query width
    qk_rope_head_dim: int = 64     # the latent's last dims, rotated
    q_lora_rank: int = 1280
    o_lora_rank: int = 1024        # the output projection's rank per group of heads
    o_groups: int = 8
    sliding_window: int = 128      # every layer attends its own last tokens

    # --- compressed sparse attention ---
    # per layer (the draft layers after the backbone included): 0 attends the window only, r
    # also the indexer's pick of a cache that pools r tokens per latent
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
    # the layers that write a compressed cache (and its index keys); the layers after one read it
    kv_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20)
    # the layers that run an indexer; the layers after one reuse its pick
    index_source_layer_ids: tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    # this indexer also keeps candidate_topk_blocks blocks of candidate_block_size latents,
    # and the indexers after it pick inside them; -1: none
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    # --- RoPE: plain on window-only layers; YaRN at compress_rope_theta on the others ---
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_max_position_embeddings: int = 65536

    # --- MoE ---
    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    # --- hyper-connections: the residual is hc_mult streams ---
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # --- Engram: n-gram hash lookups added to the streams before these layers ---
    engram_layer_ids: tuple[int, ...] = (1, 14)
    engram_num_embeddings: tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16_000_000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: int = 99092

    # --- not served yet: the DSpark draft layers and the vision tower stay unloaded ---
    num_nextn_predict_layers: int = 3

    # --- tokens ---
    bos_token_id: int = 0
    eos_token_ids: tuple[int, ...] = (1,)
    pad_token_id: int = 2

    # --- quantization: the scale block of the fp8 linears and of every rounded activation ---
    fp8_block: int = 32

    # --- serving knobs ---
    # the context served; the latent cache is sized to it
    max_seq_len: int = 65536
    max_output_tokens: int | None = None
    # the routed experts: "fp4" as the checkpoint packs them, "fp8" their exact upcast to E4M3
    # blocks (twice the bytes), "dense" in the model's dtype (four times in bf16)
    moe_experts: str = "fp4"

    def __post_init__(self):
        self.compress_ratios = tuple(self.compress_ratios)
        self.kv_source_layer_ids = tuple(self.kv_source_layer_ids)
        self.index_source_layer_ids = tuple(self.index_source_layer_ids)
        self.engram_layer_ids = tuple(self.engram_layer_ids)
        self.engram_num_embeddings = tuple(self.engram_num_embeddings)
        self.eos_token_ids = tuple(self.eos_token_ids)
        self._validate()

    def _validate(self) -> None:
        n = self.num_hidden_layers
        if len(self.compress_ratios) < n:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {n} layers")
        if self.n_shared_experts != 1:
            raise ValueError(f"one shared expert is the only layout ported, got {self.n_shared_experts}")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(f"scoring_func {self.scoring_func!r}: only sqrtsoftplus is ported")
        if self.moe_experts not in MOE_EXPERTS:
            raise ValueError(f"moe_experts must be one of {MOE_EXPERTS}, got {self.moe_experts!r}")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("num_attention_heads must divide into o_groups")
        for width in (self.head_dim, self.index_head_dim, self.q_lora_rank, self.hidden_size,
                      self.moe_intermediate_size, self.engram_head_dim):
            if width % self.fp8_block:
                raise ValueError(f"width {width} is not a multiple of the scale block {self.fp8_block}")
        if set(self.kv_source_layer_ids) - set(self.index_source_layer_ids):
            raise ValueError("every compressed source runs the indexer that keys it")
        for layer in range(n):
            ratio = self.compress_ratios[layer]
            if ratio and self.kv_source_of(layer) is None:
                raise ValueError(f"layer {layer} attends a compressed cache but no layer before it writes one")
            if ratio and self.compress_ratios[self.kv_source_of(layer)] != ratio:
                raise ValueError(f"layer {layer} reads a cache pooled at another ratio than its own {ratio}")
            if layer in self.kv_source_layer_ids and not ratio:
                raise ValueError(f"compressed source {layer} has compress ratio 0")
        cand = self.candidate_source_layer_id
        if cand >= 0:
            if cand not in self.kv_source_layer_ids:
                raise ValueError("the candidate source must be a compressed source: the later indexers score its keys")
            later = [i for i in self.kv_source_layer_ids if i > cand]
            if later:
                raise ValueError(f"compressed sources {later} after the candidate source would not share its keys")
            # below this a row can hold fewer candidates than index_topk, and which of the
            # masked latents the reference then attends is undefined
            if (self.candidate_topk_blocks - 1) * self.candidate_block_size + 1 < self.index_topk:
                raise ValueError("candidate_topk_blocks x candidate_block_size must cover index_topk")
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings needs one table size per engram layer")

    # --- the compressed sources ---

    def kv_source_of(self, layer: int) -> int | None:
        """The layer whose compressed cache ``layer`` attends (None: the window only)."""
        if not self.compress_ratios[layer]:
            return None
        before = [i for i in self.kv_source_layer_ids if i <= layer]
        return before[-1] if before else None

    @property
    def source_ratios(self) -> tuple[int, ...]:
        """The pooling ratios of the compressed sources, each one cache (``kv_cache_key``)."""
        return tuple(sorted({self.compress_ratios[i] for i in self.kv_source_layer_ids}))

    def sources_of(self, ratio: int) -> tuple[int, ...]:
        """The sources that pool ``ratio`` tokens, in layer order: the planes of that cache."""
        return tuple(i for i in self.kv_source_layer_ids if self.compress_ratios[i] == ratio)

    def plane_of(self, layer: int) -> int:
        """The plane (latents and index keys), in its ratio's cache, of the source ``layer`` attends."""
        return self.sources_of(self.compress_ratios[layer]).index(self.kv_source_of(layer))

    @property
    def pooled_sources(self) -> tuple[int, ...]:
        """The sources that pool more than one token, which keep an open group between steps."""
        return tuple(i for i in self.kv_source_layer_ids if self.compress_ratios[i] > 1)

    @property
    def max_ratio(self) -> int:
        return max((self.compress_ratios[i] for i in self.kv_source_layer_ids), default=1)

    def uses_candidates(self, layer: int) -> bool:
        return 0 <= self.candidate_source_layer_id < layer

    # --- the compressed caches' rows: the FP4 grids the checkpoint was trained with ---

    @property
    def latent_rows(self) -> Fp4Rows:
        """A compressed latent: an E4M3 scale per 16 values."""
        return Fp4Rows(block=16, e4m3_scale=True)

    @property
    def key_rows(self) -> Fp4Rows:
        """An index key (and query): a power-of-two scale per scale block."""
        return Fp4Rows(block=self.fp8_block)

    def cache_widths(self) -> tuple[int, int]:
        """A latent row's and a key row's widths as their cache declares them: the FP4 bytes in
        the two-byte values a cache is typed in as served (890 bytes a token over the
        checkpoint's four sources, against 3,200 unpacked)."""
        return -(-self.latent_rows.nbytes(self.head_dim) // 2), -(-self.key_rows.nbytes(self.index_head_dim) // 2)

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def engram_hash_cols(self) -> int:
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    @property
    def max_prompt_tokens(self) -> int:
        """Decode runs at least one step after the prefill and the next may already be
        scheduled when it stops, so a prompt leaves two rows of the window free."""
        return self.max_seq_len - 2

    @classmethod
    def from_hf_config(cls, hf_config: dict) -> "DeepseekV41Config":
        """Geometry from the checkpoint's config.json; serving knobs keep their defaults."""
        if hf_config.get("model_type") != "deepseek_v41":
            raise ValueError(f"not a DeepSeek-V4.1 config: model_type={hf_config.get('model_type')!r}")
        text = hf_config["text_config"]
        missing = [k for k in _TEXT_KEYS if k not in text]
        if missing:
            raise ValueError(f"config.json text_config lacks {missing}")
        rope = text.get("rope_scaling") or {}
        if rope.get("rope_type") != "yarn":
            raise ValueError(f"rope_scaling {rope!r}: the compressed layers are ported with YaRN only")
        quant = hf_config.get("quantization_config") or {}
        block = quant.get("weight_block_size", [32, 32])
        if (quant.get("quant_method"), quant.get("scale_fmt"), quant.get("expert_dtype"), tuple(block)) != (
                "fp8", "ue8m0", "fp4", (block[0], block[0])):
            raise ValueError(f"quantization_config {quant!r}: ported for fp8 linears with square "
                             "power-of-two scale blocks and fp4 experts")
        eos = hf_config.get("eos_token_id", 1)
        return cls(
            **{k: text[k] for k in _TEXT_KEYS},
            compress_ratios=tuple(text["compress_ratios"]),
            kv_source_layer_ids=tuple(text["kv_source_layer_ids"]),
            index_source_layer_ids=tuple(text["index_source_layer_ids"]),
            candidate_source_layer_id=text.get("candidate_source_layer_id", -1),
            rope_factor=float(rope["factor"]),
            rope_beta_fast=float(rope["beta_fast"]),
            rope_beta_slow=float(rope["beta_slow"]),
            rope_original_max_position_embeddings=int(rope["original_max_position_embeddings"]),
            engram_layer_ids=tuple(text["engram_layer_ids"]),
            engram_num_embeddings=tuple(text["engram_num_embeddings"]),
            bos_token_id=hf_config.get("bos_token_id", 0),
            eos_token_ids=tuple(eos) if isinstance(eos, list) else (eos,),
            pad_token_id=hf_config.get("pad_token_id", 2),
            fp8_block=block[0],
        )

    @classmethod
    def reduced(cls, **overrides) -> "DeepseekV41Config":
        """Tiny-dim variant for CPU tests: every mechanism of the checkpoint at small sizes,
        random weights. Two pooled sources, a ratio-1 source that is also the candidate
        source, an indexer that picks inside its candidates, and one Engram layer."""
        base = dict(
            vocab_size=512, hidden_size=64, num_hidden_layers=7, num_attention_heads=4,
            head_dim=32, qk_rope_head_dim=8, q_lora_rank=32, o_lora_rank=16, o_groups=2,
            sliding_window=8,
            compress_ratios=(0, 2, 2, 2, 1, 1, 1, 0),
            kv_source_layer_ids=(1, 3, 4), index_source_layer_ids=(1, 3, 4, 6),
            # 16 index heads: with few, many scores rectify to exactly 0 and ties pick the top-k
            index_n_heads=16, index_head_dim=32, index_topk=8,
            candidate_source_layer_id=4, candidate_topk_blocks=3, candidate_block_size=4,
            rope_original_max_position_embeddings=64, rope_factor=4.0,
            moe_intermediate_size=32, n_routed_experts=8, num_experts_per_tok=2,
            engram_layer_ids=(1,), engram_num_embeddings=(0,), engram_max_ngram_size=3,
            engram_vocab_size=50, engram_n_heads=2, engram_head_dim=32,
            engram_compressed_vocab_size=512, num_nextn_predict_layers=1,
            max_position_embeddings=512, max_seq_len=256,
        )
        base.update(overrides)
        config = cls(**base)
        if config.engram_num_embeddings == (0,) * len(config.engram_layer_ids):
            from mstar.model.deepseek_v41.components.engram import table_rows

            config = dataclasses.replace(config, engram_num_embeddings=table_rows(config))
        return config
