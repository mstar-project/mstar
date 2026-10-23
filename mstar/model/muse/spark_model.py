"""Meta's Muse Spark Open (816B total, 42B active): Muse Glimmer's ViT and a
waveform audio tower in front of a 62-layer mixture-of-experts LLM.

Glimmer's graph, walks, prompt and vision tower, with Spark's text stack
(``components/spark.py``): the attention normalizes its output, the residual
stream runs in fp32 through gated residuals, and 256 experts (8 a token) run
in a 4096-wide latent, split whole across the LLM's TP ranks. The audio tower
(``components/audio.py``) is the ``audio_encoder`` node, a chunked causal
encoder: 24 kHz clips, one token per 80 ms, run in chunks of
``AUDIO_CHUNK_TOKENS`` as a graph loop and spliced in at the prompt's
``<|audio|>`` pads.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from mstar.api_server.media_io import MAX_AUDIO_SECONDS
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources.recurrent.config import RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.model.base import TensorAndMetadata, node_moe_parallel_configs
from mstar.model.components import MoeParallelConfig, ep_expert_slice_spec
from mstar.model.multimodal import MediaEncoder
from mstar.model.multimodal_decoder import MediaIntake, PreparedMedia
from mstar.model.muse.config import AUDIO_CONV, MuseSparkAudioConfig, MuseSparkConfig
from mstar.model.muse.glimmer_model import MuseGlimmerModel
from mstar.model.muse.processing import AUDIO_END, AUDIO_PAD, AUDIO_START
from mstar.model.submodule_base import NodeSubmodule
from mstar.utils import knobs

logger = logging.getLogger(__name__)

# 20 s of audio a step: 0.19 GB of activations a row eager, and 1.2 GB kept
# by the 1-8 row graphs (measured on an H200). A clip pays for whole chunks.
AUDIO_CHUNK_TOKENS = 250


class MuseSparkModel(MuseGlimmerModel):
    NAME = "Muse Spark Open"
    TEXT_CONFIG = MuseSparkConfig
    REASONING_KWARG = "reasoning_effort"

    def __init__(
        self, model_path_hf: str, cache_dir: str | None = None,
        max_audio_seconds: float = MAX_AUDIO_SECONDS, **kwargs: Any,
    ):
        """``max_audio_seconds`` caps a request's audio, all its clips together
        (the serving YAML's ``model_kwargs``): rank 0 holds each clip's encoder
        output until the request's prefill, outside any pool."""
        super().__init__(model_path_hf, cache_dir, **kwargs)
        self.audio_config = MuseSparkAudioConfig.from_hf(self.local_dir)
        ids = {t: self.tokenizer.convert_tokens_to_ids(t) for t in (AUDIO_PAD, AUDIO_START, AUDIO_END)}
        unknown = [t for t, i in ids.items() if i is None or i == self.tokenizer.unk_token_id]
        if unknown:
            raise ValueError(f"the tokenizer has no {unknown}; this checkpoint cannot take audio")
        if ids[AUDIO_PAD] != self.config.audio_token_id:
            raise ValueError(
                f"tokenizer's {AUDIO_PAD} is {ids[AUDIO_PAD]}, config's audio token {self.config.audio_token_id}"
            )
        self.media = (*self.media, MediaIntake(
            "audio",
            MediaEncoder(
                "audio", "audio_encoder", ("audio_values",), "audio_embeds", (ids[AUDIO_PAD],),
                chunk_tokens=AUDIO_CHUNK_TOKENS,
            ),
            ids[AUDIO_PAD], self._audio_intake, sentinels=(ids[AUDIO_START], ids[AUDIO_END]),
            max_tokens=int(max_audio_seconds * self.audio_config.sampling_rate) // self.audio_config.samples_per_token,
        ))

    def _audio_intake(self, waveform: torch.Tensor, metadata: dict) -> PreparedMedia:
        """A mono clip zero-padded on the right to whole tokens, as the
        reference's feature extractor pads it (``ceil(samples / per token)``),
        in the tower's dtype: its first op casts the samples to it, so this
        is the same input at half fp32's bytes on the edge and on rank 0."""
        clip = waveform.reshape(-1).float().cpu()
        if clip.numel() == 0:
            raise ValueError(f"{self.NAME} got an empty audio clip")
        per_token = self.audio_config.samples_per_token
        clip = torch.nn.functional.pad(clip, (0, -clip.numel() % per_token))
        return PreparedMedia({"audio_values": clip.to(self.get_autocast_dtype())}, [clip.numel() // per_token])

    def load_audio(self, filepath: str, device: str) -> TensorAndMetadata:
        """Mono at the tower's rate (24 kHz)."""
        from mstar.model.components.audio_features import load_audio_file

        rate = self.audio_config.sampling_rate
        return TensorAndMetadata(
            data=load_audio_file(filepath, sample_rate=rate),
            metadata={"sample_rate": rate, "num_channels": 1},
        )

    def get_node_resources(self) -> list[NodeResourceSpec]:
        from mstar.model.muse.components.audio import conv_state_blocks

        # A clip holds its slot from its first chunk to its last, so the
        # YAML's max_slots bounds the clips being encoded at once.
        conv = RecurrentStateConfig(
            num_layers=1, blocks=conv_state_blocks(self.audio_config, self.get_autocast_dtype()),
        )
        return [
            *super().get_node_resources(),
            RecurrentStateSpec(resource_key=AUDIO_CONV, nodes={"audio_encoder"}, config=conv),
        ]

    def get_default_sharding_config(self) -> ShardingConfig:
        # the LLM's experts split as its node group's ep_config places them
        return ShardingConfig(groups=[], tp_enabled_nodes={"LLM"}, shard_dim={}, ep_enabled_nodes={"LLM"})

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuses, before any worker starts, an LLM whose experts would be
        split along their intermediate dim (each expert's output is RMS-normed
        over its whole row before its routing weight applies), and TP async
        scheduling of an LLM that shares a rank with a media encoder: that
        layout decoded wrong greedy tokens at TP2 on H200s, cause not found."""
        node_groups = config.get("node_groups", [])
        if node_moe_parallel_configs(node_groups).get("LLM", MoeParallelConfig()).parallel != "ep":
            raise ValueError(
                f"{config_path}: {self.NAME}'s LLM needs `ep_config: {{parallel: ep}}` on its node "
                "group: each expert's output is RMS-normed over its whole row, which a TP split of "
                "the experts would leave partial on every rank"
            )
        enabled, nodes = knobs.tp_async_sched(config.get("tp_async_sched"))
        if not enabled or (nodes is not None and "LLM" not in nodes):
            return
        ranks = {name: set(g.get("ranks", ())) for g in node_groups for name in g.get("node_names", ())}
        shared = sorted(e.node for e in self.encoders if ranks.get(e.node, set()) & ranks.get("LLM", set()))
        if shared:
            raise ValueError(
                f"{config_path}: {self.NAME} cannot run its LLM with TP async scheduling "
                f"(tp_async_sched / MSTAR_TP_ASYNC_SCHED) while {shared} share its ranks; "
                "turn it off or give the encoders ranks of their own"
            )

    def _create_submodule(
        self, node_name: str, device: str, autocast_dtype: torch.dtype | None, tp_group,
        moe_parallel: MoeParallelConfig | None = None,
    ) -> NodeSubmodule | None:
        if node_name not in ("LLM", "audio_encoder"):
            return super()._create_submodule(node_name, device, autocast_dtype, tp_group)
        from mstar.model.components.chunked_causal_encoder import ChunkedCausalEncoder
        from mstar.model.muse import weight_loader
        from mstar.model.muse.submodules import SparkLLMSubmodule

        dtype = autocast_dtype or torch.bfloat16
        if node_name == "LLM":
            from mstar.model.muse.components.spark import MuseSparkForCausalLM

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
        from mstar.model.muse.components.audio import MuseSparkAudioTower

        if dtype != self.get_autocast_dtype():
            # the API's intake and the conv pool are in the model's dtype
            raise ValueError(
                f"{self.NAME}'s audio tower runs in {self.get_autocast_dtype()}, the dtype its "
                f"clips arrive in, not autocast_dtype {dtype}"
            )
        tower = self._build(lambda: MuseSparkAudioTower(self.audio_config), dtype, device)
        weight_loader.load_audio_weights(tower, self._resolve_weights(), device=device)
        tower.requires_grad_(False).eval()
        logger.info("Loaded %s audio encoder onto %s", self.NAME, device)
        # its convs carry their left context in the AUDIO_CONV pool
        return ChunkedCausalEncoder(
            tower, self.audio_config.samples_per_token, self.encoder(node_name), self.encoders, AUDIO_CONV,
        )
