import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec
from mstar.engine.resources.kv.config import KVConfig, KVSpec
from mstar.engine.resources.linear_attn.config import (
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnVariant,
)
from mstar.engine.resources.position.config import PositionConfig, PositionSpec
from mstar.engine.resources.recurrent.config import DeltaNetGeometry, RecurrentStateConfig, RecurrentStateSpec
from mstar.engine.resources.sampler.config import SamplerSpec, SamplingReqConfig
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.multimodal import (
    TEXT,
    PromptPart,
    check_attachments,
    check_plan,
    find_media_spans,
    parts_from_modalities,
    prefill_plan,
    split_around_spans,
)
from mstar.model.qwen3_5.components.language_model import Qwen3_5ForCausalLM
from mstar.model.qwen3_5.config import (
    ATTN,
    GDN_STATE,
    KV_CACHE,
    LINEAR_ATTN,
    ROPE,
    SAMPLER,
    Qwen3_5Config,
    Qwen3_5VisionConfig,
)
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


def _resolve_model_metadata(repo_id: str, cache_dir: str | None) -> str:
    """Resolve only the files needed to construct config and tokenize input.

    The API and conductor processes do not need model tensors. Downloading a
    metadata-only snapshot here keeps them from allocating or transferring the
    multi-gigabyte checkpoint. Workers fetch the complete snapshot lazily from
    ``get_submodule``.
    """
    local_path = Path(repo_id)
    if local_path.is_dir():
        return str(local_path)

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "tokenizer.json",
            "merges.txt",
        ],
    )


def _as_hwc_uint8(image: torch.Tensor):
    """What the HF image processor expects, from what the data worker sends.

    Images arrive ``(C, H, W)`` float32 in [0, 1] on the GPU. The processor
    defaults to ``do_rescale=True``, so handing it floats rescales a second
    time and the model sees a near-black image rather than an error.
    """
    if image.dtype.is_floating_point:
        image = (image * 255.0).clamp(0, 255).to(torch.uint8)
    if image.dim() == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image.cpu().contiguous().numpy()


@dataclass(frozen=True)
class WalkInput:
    node: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class PrefillStep:
    walk: str
    input_tensors: dict[str, TensorPointerInfo]


# TODO: implement MoE variants
class Qwen3_5DenseModel(Model):
    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        **kwargs: Any,
    ):
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir

        self.local_dir = _resolve_model_metadata(model_path_hf, cache_dir)
        self.config = Qwen3_5Config.from_hf(self.local_dir)
        # None on a text-only checkpoint; `_create_submodule` raises rather
        # than silently serving a deployment whose config asked for the node.
        self.vision_config = Qwen3_5VisionConfig.from_hf_or_none(self.local_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.local_dir,
            cache_dir=cache_dir,
        )
        # config.json names <|endoftext|> but a chat turn ends on <|im_end|>,
        # which only the tokenizer knows. See `Qwen3_5Config.stop_token_ids`.
        if self.tokenizer.eos_token_id is not None:
            self.config.extra_stop_token_ids = (self.tokenizer.eos_token_id,)

        # Each worker asks only for nodes assigned to it. Cache the resulting
        # wrappers so weights are materialized at most once.
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._image_processor = None

    # -----------------------------------------------------------------------
    # Model ABC: resources
    # -----------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        num_kv_layers = len(self.config.full_layer_indices)
        num_gdn_layers = len(self.config.linear_layer_indices)
        return [
            KVSpec(
                resource_key=KV_CACHE, nodes={"LLM"},
                config=KVConfig(
                    num_layers=num_kv_layers,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.head_dim,
                    max_seq_len=self.config.max_position_embeddings,
                    num_qo_heads=self.config.num_attention_heads
                )
            ),
            AttentionSpec(
                resource_key=ATTN, nodes={"LLM"},
                config=AttentionConfig(kv_cache=KV_CACHE),
            ),
            PositionSpec(
                resource_key=ROPE, nodes={"LLM"},
                config=PositionConfig(kv_cache=KV_CACHE),
            ),
            RecurrentStateSpec(
                resource_key=GDN_STATE, nodes={"LLM"},
                config=RecurrentStateConfig(
                    num_layers=num_gdn_layers,
                    blocks=DeltaNetGeometry(
                        num_k_heads=self.config.linear_num_key_heads,
                        num_v_heads=self.config.linear_num_value_heads,
                        head_k_dim=self.config.linear_key_head_dim,
                        head_v_dim=self.config.linear_value_head_dim,
                        conv_kernel_size=self.config.linear_conv_kernel_dim
                    ).to_blocks()
                )
            ),
            LinearAttnSpec(
                resource_key=LINEAR_ATTN, nodes={"LLM"},
                config=LinearAttnConfig(
                    recurrent_state=GDN_STATE,
                    variant=LinearAttnVariant.GDN,
                ),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"LLM"},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=True,
            ),
        ]

    # -----------------------------------------------------------------------
    # Model ABC: walk graph declaration
    # -----------------------------------------------------------------------
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill_text = GraphNode(
            name="LLM",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="new_token",
                    conductor_new_token=True,
                    persist=True,
                    output_modality="text"
                ),
            ],
        )

        decode =  Loop(
            name="decode_loop",
            section=GraphNode(
                name="LLM",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node="LLM",
                        name="text_inputs",
                    ),
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="text_inputs",
                        output_modality="text",
                        conductor_new_token=True,
                    ),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        # The encoder turns pixels into LLM-width embeddings, which the LLM
        # node prefills exactly as it would token embeddings. One walk per
        # image, scheduled between the text spans it sat between in the
        # prompt; the KV stream is append-only, so prefilling the spans in
        # order is what reproduces the interleaving.
        prefill_vision = Sequential([
            GraphNode(
                name="vision_encoder",
                # image_grid_thw carries each image's (t, h, w) patch grid,
                # which sets both the token count and the 3D position ids.
                input_names=["pixel_values", "image_grid_thw"],
                outputs=[
                    GraphEdge(next_node="LLM", name="vision_embeds"),
                ],
            ),
            GraphNode(
                name="LLM",
                input_names=["vision_embeds", "image_grid_thw"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        conductor_new_token=True,
                        persist=True,
                        output_modality="text",
                    ),
                ],
            ),
        ])

        return dict(
            prefill_text=prefill_text,
            prefill_vision=prefill_vision,
            decode=decode,
        )

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------
    # The chat template writes each image as this triple, one pad rather than
    # one per token: the interior is dropped anyway, and `prefill_vision`
    # re-emits the two sentinels around the encoder output.
    _PLACEHOLDER_TOKENS: dict[str, tuple[str, str, str]] = {
        "image": ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"),
    }

    def _placeholder_specs(self) -> dict[str, tuple[int, int, int]]:
        """``(start, pad, end)`` sentinel ids, read off the tokenizer.

        The tokenizer rendered the prompt, so it decides these ids. Qwen3.5's
        `config.json` happens to agree, but Qwen3-Omni's does not, and the
        failure is silent: scanning with the config's ids finds no spans and
        the whole prompt comes back as one text segment.
        """
        specs: dict[str, tuple[int, int, int]] = {}
        for modality, tokens in self._PLACEHOLDER_TOKENS.items():
            ids = tuple(self.tokenizer.convert_tokens_to_ids(t) for t in tokens)
            if any(i is None or i == self.tokenizer.unk_token_id for i in ids):
                continue
            specs[modality] = ids
        return specs

    @property
    def image_processor(self):
        """Lazily loaded: a text-only deployment never needs it."""
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            self._image_processor = AutoImageProcessor.from_pretrained(
                self.local_dir, cache_dir=self.cache_dir,
            )
        return self._image_processor

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[PromptPart] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Render the request once, tokenize once, split at the attachments.

        ``text_inputs`` comes back as one span per text step of the prefill
        plan, not as a single prompt: the image belongs *inside* the user turn,
        and the KV stream is append-only, so the walks replay the layout by
        prefilling the spans and the images in the order they were written.
        `_prefill_schedule` walks the same plan, so the nth span is the nth
        text step by construction.
        """
        if prompt is None:
            return {}

        parts = parts_from_modalities(
            input_modalities,
            [p.text or "" for p in prompt_parts if p.modality == TEXT]
            if prompt_parts is not None else prompt,
        )
        unsupported = {p.modality for p in parts} - {TEXT, "image"}
        if unsupported:
            # Video needs per-frame timestamp tokens and a `video_second_per_grid`
            # position scale that images do not; the tower would run, the
            # positions would be wrong.
            raise NotImplementedError(
                f"Qwen3.5 here has no {', '.join(sorted(unsupported))} path; "
                "text and image attachments only"
            )
        raw_images = (tensors or {}).get("image_inputs", [])
        if tensors is not None:
            check_attachments(parts, {"image": len(raw_images)})

        # The released Qwen3.5 checkpoints are chat/reasoning models — the base
        # ones carry a `-Base` suffix — so the raw completion form is wrong
        # here: it skips the `<|im_start|>` framing the model was trained on.
        #
        # The template also opens a `<think>` block by default. `enable_thinking
        # =False` does not remove it; it emits an empty, pre-closed one, which
        # is how Qwen turns reasoning off.
        content = [
            {"type": TEXT, "text": part.text or ""} if part.modality == TEXT
            else {"type": part.modality, part.modality: ""}
            for part in parts
        ]
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=kwargs.get("enable_thinking", True),
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]

        spans = find_media_spans(input_ids, self._placeholder_specs())
        segments = split_around_spans(input_ids, spans)
        check_plan(prefill_plan(parts), spans, len(segments))

        result: NameToTensorList = {"text_inputs": segments}
        if not raw_images:
            return result

        result["pixel_values"] = []
        result["image_grid_thw"] = []
        for image in raw_images:
            out = self.image_processor(
                images=[_as_hwc_uint8(image)], return_tensors="pt",
            )
            result["pixel_values"].append(out["pixel_values"])
            result["image_grid_thw"] += out["image_grid_thw"]
        return result


    def postprocess(
            self,
            output: torch.Tensor,
            modality: str,
            request_kwargs: dict | None = None,
        ) -> bytes:
            if modality == "text":
                detok = self.tokenizer.decode(output)
                return detok.encode("utf-8")
            raise ValueError(f"Unsupported modality for Qwen 3.5: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: forward pass args
    # ------------------------------------------------------------------

    # Which node reads which of a walk's signals. A name can appear twice:
    # both halves of `prefill_vision` need the grid — the encoder to lay out
    # its patches, the LLM to place the 3D positions — and it reaches a node
    # only if an edge names it. The conductor dedupes the unpersist.
    _WALK_INPUTS: dict[str, list[WalkInput]] = {
        "prefill_text": [WalkInput("LLM", ("text_inputs",))],
        "prefill_vision": [
            WalkInput("vision_encoder", ("pixel_values", "image_grid_thw")),
            WalkInput("LLM", ("image_grid_thw",)),
        ]
    }

    def _prefill_schedule(
        self,
        input_modalities: list[str],
        signals: dict[str, list[TensorPointerInfo]],
    ) -> list[PrefillStep]:
        """The prefill walks a request runs, in the order they were written.

        Walks the prefill plan, so text spans and images prefill where the
        prompt put them and N images get N walks. `process_prompt` split
        `text_inputs` against the same plan, so the nth span is the nth text
        step.
        """
        pools = {
            TEXT: signals.get("text_inputs", []),
            "image": signals.get("pixel_values", []),
        }
        grids = signals.get("image_grid_thw", [])

        schedule: list[PrefillStep] = []
        for step in prefill_plan(parts_from_modalities(input_modalities)):
            pool = pools.get(step.modality, [])
            # An image needs its grid too, and the two are built together in
            # `process_prompt` — so a short `grids` means the same breakage.
            short = step.index >= len(pool) or (
                step.modality != TEXT and step.index >= len(grids)
            )
            if short:
                # `check_plan` already failed a mismatch at intake, where a 400
                # can still be returned. This runs in the conductor, whose loop
                # only logs — raising here orphans the request and hangs the
                # client instead. Same choice as Qwen3-Omni and BAGEL.
                logger.warning(
                    "Qwen3.5 prefill plan wants a %s span at index %d but the "
                    "prompt produced %d; skipping it",
                    step.modality, step.index, len(pool),
                )
                continue
            if step.modality == TEXT:
                schedule.append(
                    PrefillStep(
                        "prefill_text", input_tensors={
                            "text_inputs": pool[step.index]
                        }
                    )
                )
            else:
                schedule.append(
                    PrefillStep(
                        "prefill_vision", input_tensors={
                            "pixel_values": pool[step.index],
                            "image_grid_thw": grids[step.index]
                        }
                    )
                )
        return schedule

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        schedule = self._prefill_schedule(input_modalities, input_signals)
        if not schedule:
            # Every step was skipped for want of a tensor, so there is nothing
            # to prefill and decode would run on an empty stream. Naming the
            # cause beats the IndexError the next line would raise.
            raise ValueError(
                f"Qwen3.5 has nothing to prefill for modalities "
                f"{input_modalities}: the request carries no "
                f"{sorted(input_signals) or 'inputs'}. An empty prompt cannot "
                "be served."
            )
        step = schedule[0]
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=step.walk,
            is_prefill=True,
        )
        # The remaining walks; `get_partition_forward_pass_args` pops them.
        metadata.kwargs["prefill_schedule"] = schedule[1:]

        inputs = self._walk_inputs(step)
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum((inp.tensor_info for inp in inputs), start=[]),
            step_metadata={
                "is_prefill": True,
                "last_prefill": len(schedule) == 1
            },
        )

    @classmethod
    def _walk_inputs(cls, step: PrefillStep) -> list[GraphEdge]:
        """One edge per (node, input) this walk reads, off `_WALK_INPUTS`."""
        edges = []
        for inp in cls._WALK_INPUTS[step.walk]:
            for name in inp.inputs:
                edge = GraphEdge(next_node=inp.node, name=name)
                edge.tensor_info = [step.input_tensors[name]]
                edges.append(edge)
        return edges

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Walk the prefill schedule, then loop in decode until stopped."""
        metadata = partition_metadata
        remaining: list[PrefillStep] = metadata.kwargs.get("prefill_schedule", [])
        # Before the pop: the step about to run is the last one when it is the
        # only one left.
        last_prefill = len(remaining) == 1

        if metadata.is_prefill and remaining:
            # Every prefill step carries its own tensors, so a text span in the
            # middle of the prompt reads its own segment. Only decode reads
            # back what the previous step emitted.
            step = remaining.pop(0)
            metadata.graph_walk = step.walk
            inputs = self._walk_inputs(step)
        elif metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
            edge = GraphEdge(next_node="LLM", name="text_inputs")
            edge.tensor_info = persist_signals.get("new_token", [])
            inputs = [edge]
        else:
            # decode stops via `LLMSubmodule.check_stop`; reaching here again
            # after it fires means the request is finished
            metadata.kwargs["decode_finished"] = True
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum((inp.tensor_info for inp in inputs), start=[]),
            step_metadata={
                "is_prefill": metadata.is_prefill,
                "last_prefill": last_prefill
            },
        )

    # ------------------------------------------------------------------
    # Model ABC: per-request resource configs
    # ------------------------------------------------------------------

    def get_request_resource_configs(
        self,
        partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        model_kwargs = model_kwargs or {}
        keys = ["temperature", "top_p", "repetition_penalty", "ignore_eos"]
        return {
            SAMPLER: SamplingReqConfig(
                **{k: model_kwargs[k] for k in keys if k in model_kwargs}
            )
        }

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device, autocast_dtype)
        self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        from mstar.model.qwen3_5.submodules import (
            LLMSubmodule,
            VisionEncoderSubmodule,
        )

        if node_name not in ("LLM", "vision_encoder"):
            return None

        dtype = autocast_dtype or torch.bfloat16
        weights_dir = self._resolve_weights()

        if node_name == "vision_encoder":
            from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
            from mstar.model.qwen3_5.weight_loader import (
                load_qwen3_5_vision_weights,
            )

            if self.vision_config is None:
                raise ValueError(
                    f"{self.model_path_hf} is a text-only checkpoint, but the "
                    "deployment asked for a vision_encoder node. Drop it from "
                    "`node_groups`."
                )
            tower = self._build(
                lambda: Qwen3_5VisionModel(self.vision_config), dtype, device,
            )
            load_qwen3_5_vision_weights(tower, weights_dir, device=device)
            tower.requires_grad_(False).eval()
            logger.info("Loaded Qwen3.5 vision tower onto %s", device)
            return VisionEncoderSubmodule(tower, self.vision_config)

        from mstar.model.qwen3_5.weight_loader import load_qwen3_5_weights

        model = self._build(lambda: Qwen3_5ForCausalLM(self.config), dtype, device)
        load_qwen3_5_weights(model, weights_dir, device=device)
        model.requires_grad_(False).eval()
        logger.info("Loaded Qwen3.5 LLM submodule onto %s", device)
        return LLMSubmodule(model, self.config, self.vision_config)

    @staticmethod
    def _build(make, dtype: torch.dtype, device: str) -> torch.nn.Module:
        """Construct under `dtype` so no fp32 copy of the weights is ever
        allocated. Parameters that must stay fp32 restore themselves in
        `_apply`; see `components/linear_attn.py`."""
        torch.set_default_dtype(dtype)
        try:
            return make().to(device)
        finally:
            torch.set_default_dtype(torch.float32)

    def _resolve_weights(self) -> str:
        """The full snapshot. Only workers call this; the API and conductor
        processes stay on the metadata-only copy from ``__init__``."""
        local_path = Path(self.model_path_hf)
        if local_path.is_dir():
            return str(local_path)
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=self.model_path_hf, cache_dir=self.cache_dir,
        )
