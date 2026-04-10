"""Pi05Model: Physical Intelligence's Pi0.5 vision-language-action model.

Pi0.5 is a robotics VLA model. It takes camera images, a text task prompt,
and a robot proprioceptive state vector as inputs, and produces a 50-step
robot action trajectory through flow-matching denoising.

Architecture (2 nodes):
    vit_encoder  (enc_dec) - SigLIP So400m/14 + linear connector
    LLM          (ar)      - PaliGemma prefix expert + action expert.
                             A single fat node hosts both Gemma weight sets;
                             they share KV-cache dimensions so the action
                             expert can attend to the prefix KV cache that
                             PaliGemma writes during prefill.

Graph walks (2):
    prefill    - SigLIP encodes camera images, then PaliGemma prefills the
                 prefix [image_tokens, language_tokens, state_tokens] with
                 bidirectional attention and writes the KV cache.
    action_gen - 10-iteration flow-matching loop. Each iteration the action
                 expert reads the frozen prefix KV cache, predicts a velocity
                 with adaRMS timestep conditioning, and applies one Euler
                 step. The final iteration emits the denoised action tensor.
"""

import json
import logging
from pathlib import Path

import torch
from torch import nn

from mminf.communication.tensors import NameToTensorList
from mminf.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mminf.engine.base import EngineType
from mminf.engine.kv_store import KVCacheConfig
from mminf.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mminf.graph.special_destinations import EMIT_TO_CLIENT
from mminf.model.base import ForwardPassArgs, Model, NodeSubmodule
from mminf.model.pi05.components.action_expert import Pi05ActionExpert, Pi05AdaLNMLP
from mminf.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mminf.model.pi05.components.siglip import Pi05SiglipEncoder
from mminf.model.pi05.components.tokenization import Pi05Tokenizer
from mminf.model.pi05.config import Pi05Config, load_pi05_config
from mminf.model.pi05.submodules import Pi05LLMSubmodule, Pi05ViTEncoderSubmodule

logger = logging.getLogger(__name__)


class Pi05Model(Model):
    """Pi0.5 vision-language-action model implementation."""

    PREFILL_WALK = "prefill"
    ACTION_GEN_WALK = "action_gen"

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading

        self.config: Pi05Config = self._load_config()
        self.tokenizer: Pi05Tokenizer | None = self._load_tokenizer()

        self._repo_dir: Path | None = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

        # Components, materialized lazily by get_submodule().
        self.embed_tokens: nn.Embedding | None = None
        self.paligemma: Pi05PaliGemmaExpert | None = None
        self.action_expert: Pi05ActionExpert | None = None
        self.action_in_proj: nn.Linear | None = None
        self.action_out_proj: nn.Linear | None = None
        self.adaln_mlp: Pi05AdaLNMLP | None = None
        self.siglip: Pi05SiglipEncoder | None = None

    # ------------------------------------------------------------------
    # Config + tokenizer
    # ------------------------------------------------------------------

    def _load_config(self) -> Pi05Config:
        if self.skip_weight_loading:
            return Pi05Config()
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=self.model_path_hf,
                filename="config.json",
                cache_dir=self.cache_dir,
            )
            with open(config_path) as f:
                return load_pi05_config(json.load(f))
        except Exception as exc:
            logger.warning(
                "Could not load Pi0.5 config from HF (%s); using defaults.", exc
            )
            return Pi05Config()

    def _load_tokenizer(self) -> Pi05Tokenizer | None:
        if self.skip_weight_loading:
            return None
        try:
            from transformers import AutoTokenizer

            hf_tok = AutoTokenizer.from_pretrained(
                self.model_path_hf, cache_dir=self.cache_dir
            )
            return Pi05Tokenizer(hf_tok, self.config)
        except Exception as exc:
            logger.warning(
                "Could not load Pi0.5 tokenizer from HF (%s); proceeding without one.",
                exc,
            )
            return None

    def _ensure_repo(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir
        )
        self._repo_dir = Path(local)
        return self._repo_dir

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> KVCacheConfig:
        return KVCacheConfig(
            num_layers=self.config.num_layers,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_qo_heads,
        )

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "vit_encoder": EngineType.ENC_DEC,
            "LLM": EngineType.AR,
        }

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential(
            [
                GraphNode(
                    name="vit_encoder",
                    input_ids=["image_inputs"],
                    outputs=[GraphEdge(next_node="LLM", name="img_emb")],
                ),
                GraphNode(
                    name="LLM",
                    input_ids=["img_emb", "text_inputs", "state_inputs"],
                    outputs=[],
                ),
            ]
        )

        action_gen = Loop(
            section=GraphNode(
                name="LLM",
                input_ids=["noisy_actions", "timestep_index"],
                outputs=[
                    GraphEdge(next_node="LLM", name="noisy_actions"),
                    GraphEdge(next_node="LLM", name="timestep_index"),
                ],
            ),
            n_iters=self.config.num_flow_steps,
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="action_output",
                    output_modality="action",
                    persist=True,
                ),
            ],
        )

        return {
            self.PREFILL_WALK: prefill,
            self.ACTION_GEN_WALK: action_gen,
        }

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        **kwargs,
    ) -> NameToTensorList:
        result: NameToTensorList = {}

        if prompt is not None and self.tokenizer is not None:
            text_ids = self.tokenizer.encode_prompt(prompt)
            result["text_inputs"] = [text_ids]
        elif prompt is not None:
            result["text_inputs"] = [
                torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)
            ]

        robot_state = kwargs.get("robot_state")
        if robot_state is not None:
            if not isinstance(robot_state, torch.Tensor):
                robot_state = torch.tensor(robot_state, dtype=torch.float32)
            if self.tokenizer is not None:
                state_ids = self.tokenizer.encode_state(robot_state)
            else:
                from mminf.model.pi05.components.flow_matching import discretize_state

                state_ids = discretize_state(
                    robot_state.to(torch.float32),
                    num_bins=self.config.state_token_bins,
                ) + self.config.state_token_offset
            result["state_inputs"] = [state_ids]

        return result

    def postprocess(self, output: torch.Tensor, modality: str) -> bytes:
        if modality == "action":
            return output.detach().to(torch.float32).cpu().numpy().tobytes()
        raise ValueError(f"Unsupported modality for Pi0.5: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward pass orchestration
    # ------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=self.PREFILL_WALK,
            is_prefill=True,
            kwargs={},
        )

        inputs = []
        if "image_inputs" in input_signals:
            edge = GraphEdge(next_node="vit_encoder", name="image_inputs")
            edge.tensor_info = input_signals["image_inputs"]
            inputs.append(edge)
        if "text_inputs" in input_signals:
            edge = GraphEdge(next_node="LLM", name="text_inputs")
            edge.tensor_info = input_signals["text_inputs"]
            inputs.append(edge)
        if "state_inputs" in input_signals:
            edge = GraphEdge(next_node="LLM", name="state_inputs")
            edge.tensor_info = input_signals["state_inputs"]
            inputs.append(edge)

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        new_tokens: dict[str, list[int]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        metadata = partition_metadata
        request_done = False
        inputs: list[GraphEdge] = []

        if metadata.graph_walk == self.PREFILL_WALK:
            metadata.is_prefill = False
            metadata.graph_walk = self.ACTION_GEN_WALK
            # Inputs for the first action_gen iteration are sampled inside the
            # LLM submodule's preprocess (Gaussian noise + timestep_index=0).
            inputs = [
                GraphEdge(next_node="LLM", name="noisy_actions"),
                GraphEdge(next_node="LLM", name="timestep_index"),
            ]
        elif metadata.graph_walk == self.ACTION_GEN_WALK:
            request_done = True

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata={"is_prefill": metadata.is_prefill},
            request_done=request_done,
        )

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu"
    ) -> torch.nn.Module | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Successfully loaded Pi0.5 submodule for %s", node_name)
        return submodule

    def _create_submodule(
        self, node_name: str, device: str
    ) -> NodeSubmodule | None:
        if node_name == "vit_encoder":
            self._init_vit_components(device)
            return Pi05ViTEncoderSubmodule(
                encoder=self.siglip, config=self.config
            )
        if node_name == "LLM":
            self._init_llm_components(device)
            return Pi05LLMSubmodule(
                embed_tokens=self.embed_tokens,
                paligemma=self.paligemma,
                action_expert=self.action_expert,
                action_in_proj=self.action_in_proj,
                action_out_proj=self.action_out_proj,
                adaln_mlp=self.adaln_mlp,
                config=self.config,
            )
        return None

    def _init_vit_components(self, device: str):
        if self.siglip is not None:
            return
        with torch.device("meta" if not self.skip_weight_loading else "cpu"):
            self.siglip = Pi05SiglipEncoder(self.config)
        if self.skip_weight_loading:
            self.siglip = self.siglip.to_empty(device=device)
            return
        self._load_weights_into(
            self.siglip,
            prefix="vit",
            device=device,
        )

    def _init_llm_components(self, device: str):
        if self.embed_tokens is not None:
            return
        meta = torch.device("meta" if not self.skip_weight_loading else "cpu")
        with meta:
            self.embed_tokens = nn.Embedding(
                self.config.vocab_size,
                self.config.hidden_size,
                padding_idx=self.config.pad_token_id,
            )
            self.paligemma = Pi05PaliGemmaExpert(self.config)
            self.action_expert = Pi05ActionExpert(self.config)
            self.action_in_proj = nn.Linear(
                self.config.action_dim, self.config.hidden_size, bias=True
            )
            self.action_out_proj = nn.Linear(
                self.config.hidden_size, self.config.action_dim, bias=True
            )
            self.adaln_mlp = Pi05AdaLNMLP(
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
            )

        if self.skip_weight_loading:
            for mod in (
                self.embed_tokens,
                self.paligemma,
                self.action_expert,
                self.action_in_proj,
                self.action_out_proj,
                self.adaln_mlp,
            ):
                mod.to_empty(device=device)
            return

        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        repo_dir = self._ensure_repo()
        load_weights_from_hf_shards(
            repo_dir=repo_dir,
            modules=[
                ModuleAndPrefix(self.embed_tokens, prefix="embed_tokens"),
                ModuleAndPrefix(self.paligemma, prefix="paligemma"),
                ModuleAndPrefix(self.action_expert, prefix="action_expert"),
                ModuleAndPrefix(self.action_in_proj, prefix="action_in_proj"),
                ModuleAndPrefix(self.action_out_proj, prefix="action_out_proj"),
                ModuleAndPrefix(self.adaln_mlp, prefix="adaln_mlp"),
            ],
            device=device,
        )

    def _load_weights_into(
        self, module: nn.Module, prefix: str, device: str
    ):
        from mminf.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

        repo_dir = self._ensure_repo()
        load_weights_from_hf_shards(
            repo_dir=repo_dir,
            modules=[ModuleAndPrefix(module, prefix=prefix)],
            device=device,
        )
