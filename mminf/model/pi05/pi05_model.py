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
from mminf.model.base import ForwardPassArgs, Model
from mminf.model.pi05.components.action_expert import Pi05ActionExpert, Pi05TimeMLP
from mminf.model.pi05.components.paligemma import Pi05PaliGemmaExpert
from mminf.model.pi05.components.siglip import Pi05SiglipEncoder
from mminf.model.pi05.components.tokenization import Pi05Tokenizer
from mminf.model.pi05.config import Pi05Config, load_pi05_config
from mminf.model.pi05.submodules import Pi05LLMSubmodule, Pi05ViTEncoderSubmodule
from mminf.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


def _reset_non_persistent_buffers(module: nn.Module, device) -> None:
    """Re-initialize non-persistent buffers like ``position_ids`` after a
    ``meta + to_empty`` materialization.

    Modules constructed on the meta device skip ``post_init``, and
    ``to_empty`` only allocates uninitialized storage for parameters and
    buffers. Non-persistent buffers (registered with ``persistent=False``)
    are not in the state_dict, so ``load_state_dict`` will not restore them
    either — leaving them as garbage. The most common offender is HuggingFace
    SigLIP's ``position_ids`` buffer (``register_buffer("position_ids",
    arange(num_positions), persistent=False)``), which feeds the position
    embedding lookup. If left as garbage int64 it produces wildly incorrect
    image embeddings (off by the full norm of the position table).

    This walks the module tree and resets any sub-module that has a
    ``position_ids`` buffer to the canonical ``arange(num_positions)``.
    """
    with torch.no_grad():
        for sub in module.modules():
            pos = getattr(sub, "position_ids", None)
            if isinstance(pos, torch.Tensor):
                shape = pos.shape
                num_positions = shape[-1]
                pos.copy_(
                    torch.arange(
                        num_positions, device=pos.device, dtype=pos.dtype
                    ).expand(shape)
                )


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
        # Yaml-driven Pi05Config overrides forwarded by the entrypoint
        # (e.g. {"action_horizon": 15} for the DROID benchmark variant).
        # Applied inside _load_config() *before* weights or CUDA graphs
        # are materialized so weight shapes and graph captures use
        # consistent values.
        self._yaml_config_overrides: dict = dict(kwargs)

        self.config: Pi05Config = self._load_config()
        self.tokenizer: Pi05Tokenizer | None = self._load_tokenizer()

        self._repo_dir: Path | None = None
        self._lerobot_buckets: dict[str, dict[str, torch.Tensor]] | None = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

        # Components, materialized lazily by get_submodule().
        self.embed_tokens: nn.Embedding | None = None
        self.paligemma: Pi05PaliGemmaExpert | None = None
        self.action_expert: Pi05ActionExpert | None = None
        self.action_in_proj: nn.Linear | None = None
        self.action_out_proj: nn.Linear | None = None
        self.time_mlp: Pi05TimeMLP | None = None
        self.siglip: Pi05SiglipEncoder | None = None

    # ------------------------------------------------------------------
    # Config + tokenizer
    # ------------------------------------------------------------------

    def _load_config(self) -> Pi05Config:
        if self.skip_weight_loading:
            cfg = Pi05Config()
        else:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(
                    repo_id=self.model_path_hf,
                    filename="config.json",
                    cache_dir=self.cache_dir,
                )
                with open(config_path) as f:
                    cfg = load_pi05_config(json.load(f))
            except Exception as exc:
                logger.warning(
                    "Could not load Pi0.5 config from HF (%s); using defaults.", exc
                )
                cfg = Pi05Config()

        # Overlay yaml-driven overrides (e.g. action_horizon for DROID).
        # Applied last so they win over both HF config.json and Pi05Config
        # defaults. Unknown keys are warned and ignored — common typo trap.
        if self._yaml_config_overrides:
            # Snapshot the values that yaml is about to touch, so the log
            # below shows a clean before→after diff for the parameters
            # that actually changed (and only those).
            keys_to_log = list(self._yaml_config_overrides.keys()) + [
                # Always show num_flow_steps so users can confirm it's NOT
                # being aliased with action_horizon (they're independent —
                # num_flow_steps is the denoising-loop iteration count).
                "num_flow_steps",
                "action_horizon",
                "action_dim",
            ]
            keys_to_log = list(dict.fromkeys(keys_to_log))  # dedupe, preserve order
            before = {k: getattr(cfg, k, "<missing>") for k in keys_to_log}

            valid = {f.name for f in Pi05Config.__dataclass_fields__.values()}
            for k, v in self._yaml_config_overrides.items():
                if k in valid:
                    setattr(cfg, k, v)
                else:
                    logger.warning(
                        "Pi05Model: yaml model_kwargs key %r is not a Pi05Config "
                        "field; ignored. Valid fields: %s", k, sorted(valid),
                    )

            after = {k: getattr(cfg, k, "<missing>") for k in keys_to_log}
            logger.info(
                "Pi05Model._load_config: applied yaml model_kwargs overrides. "
                "Before=%s -> After=%s (num_flow_steps is the denoising-loop "
                "iteration count, NOT the trajectory length; it is unaffected "
                "by action_horizon override)",
                before, after,
            )
        else:
            logger.info(
                "Pi05Model._load_config: no yaml overrides; "
                "action_horizon=%d, action_dim=%d, num_flow_steps=%d",
                cfg.action_horizon, cfg.action_dim, cfg.num_flow_steps,
            )
        return cfg

    def _load_tokenizer(self) -> Pi05Tokenizer | None:
        if self.skip_weight_loading:
            return None
        from transformers import AutoTokenizer

        # Pi0.5 production code (lerobot's processor_pi05.py) uses the
        # PaliGemma tokenizer at "google/paligemma-3b-pt-224". The
        # lerobot/pi05_base repo itself does NOT ship tokenizer files, so
        # AutoTokenizer.from_pretrained(self.model_path_hf) returns 404 on
        # tokenizer.json/tokenizer.model. We try the model repo first (in
        # case a future release adds them) and fall back to the canonical
        # PaliGemma repo. ``use_fast=True`` is required so we don't need
        # the slow sentencepiece+protobuf path.
        for repo in (self.model_path_hf, "google/paligemma-3b-pt-224"):
            try:
                hf_tok = AutoTokenizer.from_pretrained(
                    repo, cache_dir=self.cache_dir, use_fast=True
                )
                if repo != self.model_path_hf:
                    logger.info(
                        "Pi0.5 tokenizer loaded from fallback repo %s "
                        "(model repo %s has no tokenizer files)",
                        repo,
                        self.model_path_hf,
                    )
                return Pi05Tokenizer(hf_tok, self.config)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load Pi0.5 tokenizer from %s (%s); trying next.",
                    repo,
                    exc,
                )
        logger.warning("All Pi0.5 tokenizer sources failed; proceeding without one.")
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

    def _ensure_lerobot_buckets(self) -> dict[str, dict[str, torch.Tensor]]:
        """Lazily load ``model.safetensors`` from the lerobot snapshot and
        bucket its keys by mminf submodule via :func:`remap_lerobot_state_dict`.

        Pi0.5 (lerobot/pi05_base) ships as a single ~14 GB safetensors blob,
        not a sharded HF index, so we can't use ``load_weights_from_hf_shards``
        here. Instead, we load the file once into CPU memory, run the
        lerobot→mminf key remap, and cache the buckets so subsequent
        ``get_submodule`` calls (vit_encoder + LLM) reuse them.
        """
        if self._lerobot_buckets is not None:
            return self._lerobot_buckets
        from safetensors.torch import load_file

        from mminf.model.pi05.weight_loader import remap_lerobot_state_dict

        repo_dir = self._ensure_repo()
        safetensors_path = repo_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"Pi0.5 checkpoint missing: {safetensors_path}. Expected a single "
                "safetensors blob in the lerobot/pi05_base snapshot."
            )
        logger.info("Loading Pi0.5 weights from %s", safetensors_path)
        flat = load_file(str(safetensors_path), device="cpu")
        self._lerobot_buckets = remap_lerobot_state_dict(flat)
        return self._lerobot_buckets

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_kv_cache_config(self) -> KVCacheConfig:
        return [KVCacheConfig(
            num_layers=self.config.num_layers,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            num_qo_heads=self.config.num_qo_heads,
        )]

    def get_node_engine_types(self) -> dict[str, EngineType]:
        return {
            "vit_encoder": EngineType.ENC_DEC,
            "LLM": EngineType.AR,
        }

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        # Pi0.5 encodes the robot state as a decimal-string suffix on the
        # language prompt (e.g. "Task: pick up the block, State: 12 87 ...;
        # \nAction: ") and tokenizes the whole thing with the PaliGemma
        # tokenizer. So the model only ever sees a single "text_inputs"
        # stream — there are no separate state-bin tokens. This matches
        # lerobot's processor_pi05.Pi05PrepareStateTokenizerProcessorStep.
        prefill = Sequential(
            [
                GraphNode(
                    name="vit_encoder",
                    input_ids=["image_inputs"],
                    outputs=[GraphEdge(next_node="LLM", name="img_emb")],
                ),
                GraphNode(
                    name="LLM",
                    input_ids=["img_emb", "text_inputs"],
                    outputs=[],
                ),
            ]
        )

        # NOTE: The Loop's terminal ``outputs`` are matched into the section's
        # node outputs by **name** (see Loop._replace_outputs_for_final_iter
        # in mminf/graph/base.py): on the final iteration, any section-output
        # edge whose name matches a terminal output's name is replaced with
        # the terminal version. This is the same convention BAGEL's image_gen
        # uses (section returns ``latents`` looping back to LLM, terminal
        # output is ``name="latents" → vae_decoder``). So our terminal output
        # MUST be named ``noisy_actions`` to match the section's loop-back
        # edge — the name is just a graph-internal key, while the actual
        # client-facing modality bucket is determined by ``output_modality``.
        action_gen = Loop(
            section=GraphNode(
                name="LLM",
                input_ids=["noisy_actions", "timestep_index"],
                outputs=[
                    GraphEdge(next_node="LLM", name="noisy_actions"),
                    GraphEdge(next_node="LLM", name="timestep_index"),
                ],
            ),
            max_iters=self.config.num_flow_steps,
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="noisy_actions",
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
        """Tokenize the Pi0.5 prompt + robot state into a single token stream.

        Pi0.5's production preprocessor (lerobot's
        ``Pi05PrepareStateTokenizerProcessorStep``) builds a prompt of the
        form::

            "Task: <text>, State: <bin0> <bin1> ... <bin31>;\\nAction: "

        where each ``<bin_i>`` is the integer index (0–255) obtained by
        digitizing the normalized state into 256 bins. The PaliGemma
        tokenizer then encodes the whole string. We mirror that exactly
        here so the resulting ``text_inputs`` stream matches the production
        format.
        """
        if self.tokenizer is None:
            # Tokenizer-less fallback used by structural unit tests.
            if prompt is not None:
                return {
                    "text_inputs": [
                        torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long)
                    ]
                }
            return {}

        cleaned = (prompt or "").strip().replace("_", " ").replace("\n", " ")

        robot_state = kwargs.get("robot_state")
        if robot_state is not None:
            if not isinstance(robot_state, torch.Tensor):
                robot_state = torch.tensor(robot_state, dtype=torch.float32)
            from mminf.model.pi05.components.flow_matching import discretize_state

            bins = discretize_state(
                robot_state.to(torch.float32),
                num_bins=self.config.state_token_bins,
            ).tolist()
            state_str = " ".join(str(b) for b in bins)
            full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
        else:
            full_prompt = cleaned

        text_ids = self.tokenizer.encode_prompt(full_prompt)
        return {"text_inputs": [text_ids]}

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
                time_mlp=self.time_mlp,
                config=self.config,
            )
        return None

    def _init_vit_components(self, device: str):
        if self.siglip is not None:
            return
        # Construct on the "meta" device — a special PyTorch device that
        # tracks shape/dtype but allocates no real storage. This lets us
        # build the module structure with the correct parameter shapes
        # without paying for ~ViT-bytes of CUDA memory + a throwaway random
        # init that we'd immediately overwrite with the lerobot weights.
        # ``mod.to_empty(device=device)`` then materializes uninitialized
        # tensors on the target device, and ``load_state_dict`` overwrites
        # them with the real weights. Same pattern HuggingFace
        # ``from_pretrained`` uses under the hood.
        with torch.device("meta" if not self.skip_weight_loading else "cpu"):
            self.siglip = Pi05SiglipEncoder(self.config)
        if self.skip_weight_loading:
            self.siglip = self.siglip.to_empty(device=device)
            _reset_non_persistent_buffers(self.siglip, device)
            return

        buckets = self._ensure_lerobot_buckets()
        self.siglip.to_empty(device=device)
        # CRITICAL: HF's SiglipVisionEmbeddings registers ``position_ids`` as
        # a NON-persistent buffer (persistent=False), so it's not in any
        # state_dict. ``to_empty`` materializes it as uninitialized GPU
        # memory, ``_init_weights`` is never called (we never go through
        # post_init), and ``load_state_dict(strict=False)`` does not restore
        # it. The result is garbage int64 indices feeding into
        # ``position_embedding``, which corrupts every image embedding by
        # ~the full norm of the position table. We must manually reset any
        # non-persistent ``position_ids`` buffer with the canonical
        # ``arange`` values before running the forward.
        _reset_non_persistent_buffers(self.siglip, device)
        # strict=False: the lerobot bucket may contain stray pooling-head keys
        # that Pi05SiglipEncoder doesn't model (vision_use_head=False).
        self.siglip.load_state_dict(buckets["siglip"], strict=False)

    def _init_llm_components(self, device: str):
        if self.embed_tokens is not None:
            return
        # See ``_init_vit_components`` for why we build on the meta device:
        # it lets us instantiate ~14 GB worth of Pi0.5 LLM parameters with
        # zero real memory and zero random init, then materialize them on
        # ``device`` via ``to_empty`` and overwrite with lerobot weights.
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
                self.config.action_dim, self.config.action_hidden_size, bias=True
            )
            self.action_out_proj = nn.Linear(
                self.config.action_hidden_size, self.config.action_dim, bias=True
            )
            self.time_mlp = Pi05TimeMLP(hidden_size=self.config.action_hidden_size)

        if self.skip_weight_loading:
            for mod in (
                self.embed_tokens,
                self.paligemma,
                self.action_expert,
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp,
            ):
                mod.to_empty(device=device)
            return

        buckets = self._ensure_lerobot_buckets()
        # Materialize each module on `device`, then copy in its bucket. We
        # use strict=False because the lerobot paligemma bucket carries an
        # extra ``embed_tokens.weight`` key (loaded separately into
        # self.embed_tokens) and possibly other tied/aux tensors that
        # Pi05PaliGemmaExpert doesn't model.
        for mod, name in (
            (self.embed_tokens, "embed_tokens"),
            (self.paligemma, "paligemma"),
            (self.action_expert, "action_expert"),
            (self.action_in_proj, "action_in_proj"),
            (self.action_out_proj, "action_out_proj"),
            (self.time_mlp, "time_mlp"),
        ):
            mod.to_empty(device=device)
            mod.load_state_dict(buckets[name], strict=False)
