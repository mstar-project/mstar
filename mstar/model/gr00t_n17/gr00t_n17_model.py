"""GR00T N1.7's model class: the four walks, the resources they need, and the
submodules that fill them.

Four walks: `prefill` conditions a request once, `backbone` and
`state` expose two of its pieces for reuse by other models, and `action_gen`
loops the denoise step four times. `denoise_step` is a `DenoiseLoopSubmodule`,
so the Loop's machinery is the engine's and not this file's.
"""
from __future__ import annotations

import logging
from pathlib import Path

import msgpack
import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedCrossAttentionSpec,
)
from mstar.engine.resources.base import NodeResourceSpec
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Parallel, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model, TensorAndMetadata
from mstar.model.gr00t_n17.components.bridge import Gr00tN17VLBridge
from mstar.model.gr00t_n17.components.dit import ActionPositionEmbedding, DiT
from mstar.model.gr00t_n17.components.embodiment import (
    build_action_decoder,
    build_action_encoder,
    build_state_encoder,
)
from mstar.model.gr00t_n17.components.llm import Gr00tN17LanguageModel
from mstar.model.gr00t_n17.components.vision import Gr00tN17VisionEncoder
from mstar.model.gr00t_n17.config import (
    BRIDGE_ATTN,
    DT_ATTN,
    DT_XATTN,
    LLM_ATTN,
    Gr00tN17Config,
)
from mstar.model.gr00t_n17.preprocessing import Gr00tN17Preprocessor
from mstar.model.gr00t_n17.submodules import (
    Gr00tN17DenoiseStepSubmodule,
    Gr00tN17LLMSubmodule,
    Gr00tN17Shape,
    Gr00tN17StateEncoderSubmodule,
    Gr00tN17VisionEncoderSubmodule,
    Gr00tN17VLBridgeSubmodule,
)
from mstar.model.gr00t_n17.weight_loader import load_gr00t_n17_weights
from mstar.model.qwen3_5.config import VISION_ATTN
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

VISION_ENCODER, LLM, STATE_ENCODER, VL_BRIDGE, DENOISE_STEP = (
    "vision_encoder", "LLM", "state_encoder", "vl_bridge", "denoise_step",
)

# The Loop's name, which is **not** the walk's. `Loop.name` defaults to a fresh
# uuid4 and the comment on that field (`graph/base.py`) says it must be set by
# hand, because the engine keys a loop's iteration counter and its stop signals
# on it: `get_loop_indices()` builds `{loop.name: curr_iter}`, and the scaffold
# reads `dynamic_loop_iter_counts.get(self.loop_name, 0)`. Give the submodule
# the walk's name instead and the lookup misses, the counter reads 0 at every
# iteration, and all four steps denoise step 0 from fresh noise -- silently,
# because `max_iters` still stops the loop. z_image and ltx2_5 both name theirs.
DENOISE_LOOP = "denoise_loop"

# Pillow's names for the formats that return the pixels they were given, among those
# the engine's reader decodes (`media_limits.IMAGE_FORMATS`; it refuses BMP, TIFF and
# PPM before this check runs). WebP is here because *lossless* WebP measures
# bit-identical to PNG through the image path; Pillow reports one format for both
# WebP modes, so a lossy WebP gets in -- the one hole this list cannot close, and the
# reason the error says what it does.
_LOSSLESS_IMAGE_FORMATS = frozenset({"PNG", "WEBP"})


def _entry(value: np.ndarray) -> dict:
    """One array on the wire: float32 like the reference's own cast of its actions (numpy has
    no bfloat16, and widening bf16 is exact; float64 would double the payload for precision
    the decode's floor does not support), with the dtype and shape a client needs to rebuild
    it: ``np.frombuffer(e["data"], e["dtype"]).reshape(e["shape"])``."""
    return {
        "dtype": "float32",
        "shape": list(value.shape),
        "data": np.ascontiguousarray(value, dtype=np.float32).tobytes(),
    }


class Gr00tN17Model(Model):
    """`nvidia/GR00T-N1.7-3B` on M\\*."""

    # `process_prompt` turns these two into tensor edges, so the conductor has no
    # use for the originals. The data worker keeps its own copy
    # (`PreprocessWorkerThread._process_input`), which is where `postprocess`
    # reads the tag and the raw state back to unnormalise the actions. `walk` is deliberately
    # **not** here: the schedule is chosen on the conductor side
    # (`Conductor._do_ingest_request`), after the strip.
    PREPROCESS_ONLY_KWARGS = ("state", "embodiment_tag")

    PREFILL_WALK = "prefill"
    BACKBONE_WALK = "backbone"
    STATE_WALK = "state"
    ACTION_GEN_WALK = "action_gen"

    # Which nodes each reuse walk contains. Written out rather than
    # read from `get_graph_walk_graphs()`, which builds the graphs and runs once
    # per request here; `test_the_reuse_walk_node_sets_match_the_graphs` holds the
    # two in step, at setup time where the cost does not matter.
    REUSE_WALK_NODES = {
        BACKBONE_WALK: frozenset({VISION_ENCODER, LLM}),
        STATE_WALK: frozenset({STATE_ENCODER}),
    }

    config: Gr00tN17Config

    def __init__(
        self,
        model_path_hf: str,
        backbone_path_hf: str | None = None,
        cache_dir: str | None = None,
        accelerator_graph: bool = False,
        capture_shapes: list[list[int]] | None = None,
        capture_batch_sizes: list[int] | None = None,
    ):
        self.model_path_hf = model_path_hf
        # `[text_rows, image_rows]` per denoise-step bucket to capture: the spans are
        # inputs the DT attends over, so a request cannot be padded into a wider
        # bucket, and an uncaptured shape runs the eager batched path
        if accelerator_graph and not capture_shapes:
            raise ValueError("accelerator_graph needs capture_shapes: [[text_rows, image_rows], ...]")
        self.accelerator_graph = accelerator_graph
        self.capture_shapes = [tuple(int(n) for n in shape) for shape in capture_shapes or ()]
        self.capture_batch_sizes = [int(b) for b in capture_batch_sizes or (1, 2, 4, 8)]
        # GR00T's weights and its backbone's config live in two repos: the
        # action head is `nvidia/GR00T-N1.7-3B`, the Qwen3-VL dims come from
        # `nvidia/Cosmos-Reason2-2B`, which is gated.
        self.backbone_path_hf = backbone_path_hf or self.config_backbone_default
        self.cache_dir = cache_dir
        self._repo_dir: Path | None = None
        self.config: Gr00tN17Config = Gr00tN17Config.from_hf(
            self._model_dir(), self._backbone_config(),
        )
        self.processor = Gr00tN17Preprocessor.from_pretrained(
            self._model_dir(),
            max_state_dim=self.config.max_state_dim,
            max_action_horizon=self.config.action_horizon,
            # the hub id, not a local path: `AutoProcessor.from_pretrained`
            # fetches the tokenizer and image processor itself, and only those
            backbone_path=self.backbone_path_hf,
            cache_dir=self.cache_dir,
        )
        # Every request's embodiment id comes from this table, and it indexes the
        # per-embodiment weights, where -1 reads the last robot's slice in silence.
        # Checked once here so the step path never reads the id back to the host.
        categories = self.config.max_num_embodiments
        outside = {tag: i for tag, i in self.processor.embodiment_ids.items() if not 0 <= i < categories}
        if outside:
            raise ValueError(
                f"embodiment_id.json maps {outside} outside the checkpoint's {categories} embodiments"
            )
        self._submodules: dict[str, NodeSubmodule] = {}

    config_backbone_default = "nvidia/Cosmos-Reason2-2B"

    def _model_dir(self) -> Path:
        """`model_path_hf` as a local directory.

        Registration passes a hub id (`nvidia/GR00T-N1.7-3B`); a vendored
        checkout and the tests pass a path. Everything downstream needs a
        directory: `Gr00tN17Config.from_hf` reads `config.json` from it, the
        processor reads three more json files out of it, and the weight loader
        opens its safetensors shards (`load_gr00t_n17_weights`). One snapshot
        serves all three, which is why this fetches the whole repo rather than
        the config alone -- pi05 resolves the same way (`Pi05Model._ensure_repo`).
        """
        if self._repo_dir is None:
            if Path(self.model_path_hf).is_dir():
                self._repo_dir = Path(self.model_path_hf)
            else:
                from huggingface_hub import snapshot_download

                self._repo_dir = Path(snapshot_download(
                    repo_id=self.model_path_hf, cache_dir=self.cache_dir,
                ))
        return self._repo_dir

    def _backbone_config(self) -> Path:
        """Cosmos-Reason2-2B's `config.json`, and nothing else from that repo.

        Only the backbone's *dims* are GR00T's -- its 2B of weights
        are not -- so a snapshot would download them to read one file.
        """
        if Path(self.backbone_path_hf).is_dir():
            return Path(self.backbone_path_hf)
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            self.backbone_path_hf, "config.json", cache_dir=self.cache_dir,
        ))

    # ----------------------------------------------------------- resources
    def get_node_resources(self) -> list[NodeResourceSpec]:
        """The five attention resources, one per geometry and kind. Every attention is
        cacheless and ragged -- there is no KV anywhere, because nothing is ever decoded
        token by token.

        Each therefore declares its own `dtype`: with none, a cacheless resource
        falls back to "the engine's KV dtype (which is actually a misnamed
        autocast dtype)" (`RaggedAttentionConfig.dtype`), so the dtype the parity
        bar is measured in would track an unrelated default. z_image, flux2_klein and
        ltx2_5 pin theirs for the same reason.

        The DT needs **two**: a resource is one *kind* of attention, so the
        cross blocks' `(q_label, kv_label)` pairs and the self blocks' single
        span cannot share one even at the same 32 x 48 geometry.
        """
        text, vision = self.config.text, self.config.vision
        bridge, dit = self.config.bridge, self.config.dit
        return [
            RaggedAttentionSpec(
                resource_key=VISION_ATTN, nodes={VISION_ENCODER},
                config=RaggedAttentionConfig(
                    num_qo_heads=vision.num_heads, num_kv_heads=vision.num_heads,
                    head_dim=vision.hidden_size // vision.num_heads,
                    # `vision_encoder` declares one segment per *frame* -- a frame
                    # attends to itself alone
                    # (`Gr00tN17VisionEncoderSubmodule.declare_step`) -- so the bound is
                    # a request's most frames, views x timesteps, the count the data
                    # worker already caps a request's files at
                    max_segments_per_request=self.max_input_files()["image"],
                    dtype=torch.bfloat16,
                ),
            ),
            # Causality is a property of the *step*, not the resource: the LLM
            # node declares `AttentionStep(causal=True)` (`Gr00tN17LLMSubmodule.declare_step`).
            # GR00T is the first model on `main` to ask for that on a ragged
            # resource; every other user is non-causal.
            RaggedAttentionSpec(
                resource_key=LLM_ATTN, nodes={LLM},
                config=RaggedAttentionConfig(
                    num_qo_heads=text.num_attention_heads,
                    num_kv_heads=text.num_key_value_heads,
                    head_dim=text.head_dim,
                    dtype=torch.bfloat16,
                ),
            ),
            RaggedAttentionSpec(
                resource_key=BRIDGE_ATTN, nodes={VL_BRIDGE},
                config=RaggedAttentionConfig(
                    num_qo_heads=bridge.num_attention_heads,
                    num_kv_heads=bridge.num_attention_heads,
                    head_dim=bridge.attention_head_dim,
                    dtype=torch.bfloat16,
                ),
            ),
            RaggedAttentionSpec(
                resource_key=DT_ATTN, nodes={DENOISE_STEP},
                config=RaggedAttentionConfig(
                    num_qo_heads=dit.num_attention_heads,
                    num_kv_heads=dit.num_attention_heads,
                    head_dim=dit.attention_head_dim,
                    dtype=torch.bfloat16,
                ),
            ),
            RaggedCrossAttentionSpec(
                resource_key=DT_XATTN, nodes={DENOISE_STEP},
                config=RaggedAttentionConfig(
                    num_qo_heads=dit.num_attention_heads,
                    num_kv_heads=dit.num_attention_heads,
                    head_dim=dit.attention_head_dim,
                    dtype=torch.bfloat16,
                    # `max_segments_per_request` stays at its default 1: the
                    # cross resource plans one wrapper **per pair**, keyed
                    # `cross_label(q_label, kv_label)` (`_RaggedCrossManagerBase._plan_layouts`),
                    # and each pair sees one `sa` segment and one key segment per
                    # request. The bound is per label, not per step.
                ),
            ),
        ]

    # --------------------------------------------------------------- walks
    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        """The four walks.

        `prefill` is a `Parallel`, not a `Sequential`: the state encoder reads
        only the request's own state, so it does not wait on the backbone.

        `action_gen`'s terminal output **must** be named for the section's
        loop-back edge, and both for the scaffold's own `LATENTS`.
        `Loop.__post_init__` (`graph/base.py`) keeps only the outputs the
        section actually produces, so a mismatched name is **dropped** and the
        walk emits nothing at all -- not a wrong iteration, no output. (pi05's
        comment cites a `_replace_outputs_for_final_iter` that no longer exists
        on `main`; this is the mechanism that does.)
        """
        conditioning = Sequential([
            GraphNode(
                name=VISION_ENCODER, input_names=["pixel_values", "image_grid_thw"],
                outputs=[GraphEdge(next_node=LLM, name="vision_embeds"),
                         GraphEdge(next_node=LLM, name="deepstack_embeds")],
            ),
            GraphNode(
                name=LLM,
                input_names=["vision_embeds", "deepstack_embeds", "input_ids", "image_grid_thw"],
                outputs=[GraphEdge(next_node=VL_BRIDGE, name="backbone_features"),
                         GraphEdge(next_node=VL_BRIDGE, name="image_rows")],
            ),
            GraphNode(
                name=VL_BRIDGE, input_names=["backbone_features", "image_rows"],
                outputs=[
                    # Persist *signals*, not edges to a node: `denoise_step` is not
                    # in this walk, so there is nothing here to point at. The tensor
                    # is parked for the request and `action_gen` reads it by name on
                    # each of the four iterations. Same idiom as flux2_klein's
                    # TEXT_EMBEDS and ltx2_5's TEXT_VIDEO.
                    GraphEdge(next_node=EMPTY_DESTINATION, name="vl_text", persist=True),
                    GraphEdge(next_node=EMPTY_DESTINATION, name="vl_image", persist=True),
                ],
            ),
        ])
        state = GraphNode(
            name=STATE_ENCODER, input_names=["state", "embodiment_id"],
            outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name="state_features", persist=True)],
        )
        return {
            self.PREFILL_WALK: Parallel([conditioning, state]),
            # the two reuse walks: the same nodes, stopping early, so another
            # model can take GR00T's backbone or its state encoder alone
            self.BACKBONE_WALK: Sequential([
                GraphNode(
                    name=VISION_ENCODER, input_names=["pixel_values", "image_grid_thw"],
                    outputs=[GraphEdge(next_node=LLM, name="vision_embeds"),
                             GraphEdge(next_node=LLM, name="deepstack_embeds")],
                ),
                GraphNode(
                    name=LLM,
                    input_names=["vision_embeds", "deepstack_embeds", "input_ids", "image_grid_thw"],
                    outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="backbone_features",
                                       output_modality="tensor")],
                ),
            ]),
            self.STATE_WALK: GraphNode(
                name=STATE_ENCODER, input_names=["state", "embodiment_id"],
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="state_features",
                                   output_modality="tensor")],
            ),
            self.ACTION_GEN_WALK: Loop(
                section=GraphNode(
                    name=DENOISE_STEP,
                    # `latents` is `DenoiseLoopSubmodule.LATENTS`, which the scaffold
                    # carries and `denoise()` takes by that keyword. The graph edge
                    # has to use the scaffold's name or iteration 1 cannot find what
                    # iteration 0 returned; the *client* sees an action either way,
                    # because that is `output_modality`, not this key. ltx2_5 names
                    # its edge `latents` for the same reason.
                    input_names=["latents", "vl_text", "vl_image", "state_features", "embodiment_id"],
                    outputs=[GraphEdge(next_node=DENOISE_STEP, name="latents")],
                ),
                max_iters=self.config.num_inference_timesteps,
                outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="latents",
                                   output_modality="action", persist=True)],
                name=DENOISE_LOOP,
            ),
        }

    # ---------------------------------------------------------- submodules
    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        """Built once per node and cached, then put in **eval** mode.

        Our modules hold `Dropout(p=0.0)` rather than relying on the flag, so
        nothing changes today -- but every parity number the tests measure is an
        eval-mode claim, and `qwen3_5`, `whisper` and `orpheus` all make it
        explicit at load rather than leaving it to the caller.
        """
        # no node shards its weights or heads: under a TP group every rank would
        # build and run the whole model
        if tp_group is not None and tp_group.world_size > 1:
            raise ValueError("gr00t_n17 does not support tensor parallelism")
        cached = self._submodules.get(node_name)
        if cached is not None:
            return cached
        submodule = self._create_submodule(node_name, device)
        if submodule is None:
            return None
        submodule = submodule.eval()
        if submodule.disable_autocast and autocast_dtype is not None:
            # the engine keeps an opted-out node in the dtype it was built in
            # (`engine_manager.py`), and the bridge runs in the engine's bf16
            submodule = submodule.to(dtype=autocast_dtype)
        self._submodules[node_name] = submodule
        logger.info("built gr00t_n17 submodule for %s", node_name)
        return submodule

    def capture_buckets(self) -> list[tuple[str, Gr00tN17Shape]]:
        """The denoise step's graph buckets: the only node captured, and the one
        that runs four times per request (eager, it is bound by kernel launches)."""
        if not self.accelerator_graph:
            return []
        return [
            (self.ACTION_GEN_WALK, Gr00tN17Shape(text_rows=text, image_rows=image))
            for text, image in self.capture_shapes
        ]

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        """One per node, with its checkpoint weights in it.

        The loader (`weight_loader.py`) names each node's tensors as the
        reference does, which is relative to the module holding them: the
        wrapped component for the three nodes that wrap one, the submodule for
        the two that hold their parts directly. Loading is strict, so a
        parameter that gets no tensor raises instead of serving its random
        init -- which still returns plausible-looking actions.
        """
        config = self.config

        def loaded(module: torch.nn.Module) -> torch.nn.Module:
            load_gr00t_n17_weights(module, node_name, self._model_dir())
            return module

        if node_name == VISION_ENCODER:
            return Gr00tN17VisionEncoderSubmodule(loaded(Gr00tN17VisionEncoder(config)), config.vision)
        if node_name == LLM:
            return Gr00tN17LLMSubmodule(loaded(Gr00tN17LanguageModel(config)), config)
        if node_name == STATE_ENCODER:
            return loaded(Gr00tN17StateEncoderSubmodule(build_state_encoder(config)))
        if node_name == VL_BRIDGE:
            return Gr00tN17VLBridgeSubmodule(loaded(Gr00tN17VLBridge(config)), config)
        if node_name == DENOISE_STEP:
            return loaded(Gr00tN17DenoiseStepSubmodule(
                config,
                build_action_encoder(config),
                ActionPositionEmbedding(config),
                DiT(config),
                build_action_decoder(config),
                loop_name=DENOISE_LOOP,
                capture_buckets=self.capture_buckets(),
                capture_batch_sizes=self.capture_batch_sizes,
            ))
        return None

    def max_input_files(self) -> dict[str, int]:
        """One frame per camera view per timestep, and no more: the most frames any
        embodiment's request carries (6 in this checkpoint: 3 views x 2 timesteps).
        The data worker refuses a request over this before decoding any file
        (`check_file_counts` in `PreprocessWorkerThread._load_media`).
        """
        keys, steps = self.processor.video_keys, self.processor.video_steps
        return {"image": max(len(keys[tag]) * steps[tag] for tag in keys)}

    def load_image(self, filepath: str, device: str):
        """A camera frame as **uint8 HWC**, which is what the reference's transform
        takes -- overriding the base, which returns float32 CHW in [0, 1].

        Not a preference: the HF image processor rescales by 1/255 whatever it is
        handed, so handing it the base's already-divided frame makes the model see
        a black image -- measured pixel mean -0.997 against -0.164, with the same
        shapes and no error. `preprocessing.__call__` refuses a non-uint8 frame for
        the same reason; this is the other half, so the refusal never fires on our
        own loader.
        """
        from mstar.model.media_limits import check_image_size, read_image

        # Lossless only, and refused from the header before anything is decoded.
        # A JPEG passes every count, dtype and shape check this model makes and
        # still moves `pixel_values` by 0.32 against a lossless 0.0019 -- 167x --
        # so a compressed camera feed is the same silent-and-plausible failure as
        # a reordered or double-rescaled frame: the actions look fine and the
        # robot moves wrong. Frames must be lossless; this is what keeps them so.
        fmt, _ = check_image_size(filepath)
        if fmt not in _LOSSLESS_IMAGE_FORMATS:
            raise ValueError(
                f"gr00t_n17 needs lossless camera frames; got {fmt}. The pixel "
                "path is compared against the reference bit for bit, so lossy "
                f"compression is a silent error. Use one of "
                f"{sorted(_LOSSLESS_IMAGE_FORMATS)}"
            )
        # the engine's own reader, so this inherits its decode limits and its
        # RGB coercion (an RGBA or CMYK frame has no placeable channel axis)
        try:
            img = read_image(filepath, rgb=True)  # uint8 CxHxW
        except (RuntimeError, OSError) as exc:  # the base loader's mapping: a 400, not a 500
            raise ValueError("could not decode image: the file is corrupt or truncated") from exc
        return TensorAndMetadata(img.permute(1, 2, 0).contiguous().to(device))

    def warmup_preprocess(self) -> None:
        """Pay the lazy `cv2` import before the first robot observation does.

        The image transform imports cv2 inside the call
        (`Gr00tN17Preprocessor._shortest_edge_to`) because OpenCV is a
        `gr00t_n17` extra and not a core dependency, so without this the first
        request of a server's life pays for it. Running the real resize rather
        than importing the module warms whatever else that path touches; the
        frame is camera-shaped so the work is the real work, and it is
        microseconds.

        Nothing else here is lazy: `__init__` already builds the processor, which
        is the expensive part (a tokenizer and an image processor).
        """
        self.processor.image(np.zeros((180, 320, 3), dtype=np.uint8))
        # The backbone processor's jinja template compiles on first use, and
        # `Model.warmup_preprocess` names a tokenizer as exactly what this hook is for.
        self.processor.warmup_chat_template()

    def _views_from_frames(self, tag: str, frames: list | None) -> dict[str, np.ndarray]:
        """The request's frames, back into the reference's per-camera observation.

        Camera frames travel the **media path**, because they are bulk
        media -- one recorded DROID observation is 3.36 MiB against the 1 MiB cap
        on a `model_kwargs` form field. That path keys loaded files by *modality*
        (`PreprocessWorkerThread._load_media`), so every frame arrives in one list
        and the order is the only thing that says which camera it came from.

        The order is the reference's own: **time-major, views in the embodiment's
        `modality_keys` order**, which is how `preprocessing.__call__` and
        NVIDIA's processor both build their frame list. So `(t0,v0), (t0,v1),
        (t1,v0), (t1,v1)` for a two-camera robot.

        Rebuilding the dict here rather than reshaping the processor keeps its
        contract -- `video[view]` as uint8 `(B, T, H, W, 3)` -- which is the
        reference's, and is what the parity tests are written against.
        """
        views, steps = self.processor.frame_layout(tag)
        want = len(views) * steps
        frames = frames or []
        if len(frames) != want:
            raise ValueError(
                f"gr00t_n17 {tag!r} needs {want} camera frames -- {steps} per view "
                f"for views {list(views)}, time-major -- but the request carried "
                f"{len(frames)}"
            )
        # The resize scales each frame's short side to `image_edge` with no bound on the
        # long side, so a thin frame well inside the decode budget (2 x 100000) would
        # resize to gigabytes in the API server. Its resized pixels, rounded as the
        # resize rounds them, are charged against the same per-request budget.
        from mstar.model.media_limits import max_image_pixels

        edge, budget = self.processor.image_edge, max_image_pixels()
        resized = sum(
            round(w * edge / min(h, w)) * round(h * edge / min(h, w))
            for h, w in (tuple(f.shape[:2]) for f in frames)
        )
        if resized > budget:
            raise ValueError(
                f"gr00t_n17 resizes each frame's short side to {edge}; these frames "
                f"({[tuple(f.shape[:2]) for f in frames]}) would resize to {resized} pixels, "
                f"over the request's image budget of {budget} (MSTAR_MAX_IMAGE_PIXELS)"
            )
        # per view: cameras may differ in size (same aspect ratio, which the processor
        # checks); a camera's own frames may not
        for j, view in enumerate(views):
            shapes = {tuple(f.shape) for f in frames[j::len(views)]}
            if len(shapes) > 1:
                raise ValueError(f"gr00t_n17 camera {view!r} sent frames of different sizes: {sorted(shapes)}")
        return {
            view: np.stack([np.asarray(f.cpu()) for f in frames[j::len(views)]])[None]
            for j, view in enumerate(views)
        }

    # ------------------------------------------------------------------ I/O
    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        prompt_parts: list[dict] | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """One observation to the model's inputs, through `Gr00tN17Preprocessor`.

        GR00T does not take a prompt alone: an observation is a language
        instruction **plus** the camera views and the robot's proprioceptive
        state, and which views and which state keys depends on the embodiment
        tag. The tag therefore rides on the request
        (:meth:`request_kwargs`) and selects the normalisation stats, the
        video keys and the per-robot weights.
        """
        # `PreprocessWorkerThread._fail_request` answers `400 if isinstance(exc,
        # (ValueError, TypeError)) else 500`, so a missing field's KeyError, or the
        # AttributeError a wrong JSON type raises at `.lower()` or `.items()`, would
        # answer a malformed request with a 500. Both kwargs are checked by type here.
        tag = kwargs.get("embodiment_tag")
        if not isinstance(tag, str):
            raise ValueError(
                f"gr00t_n17 needs an `embodiment_tag` string, got {tag!r}: it selects the "
                "joint groups, the normalisation stats and the per-robot weights"
            )
        if not isinstance(kwargs.get("state"), dict):
            raise ValueError(
                "gr00t_n17 needs `state`, a JSON object of joint groups: an observation is a "
                "language instruction plus the camera views and the robot's proprioceptive state"
            )
        # The reference asserts the history length (`gr00t_n1d7.py`, `_encode_features`);
        # here it is the client's input, so a 400. Past this point a wrong `T` fails in the
        # state encoder's forward, which fails every request in that worker step, and a
        # bare `(D,)` fails the decode's `[:, -1]` after the whole forward has run.
        history = self.config.state_history_length
        wrong = {}
        for k, v in kwargs["state"].items():
            try:
                shape = np.asarray(v, dtype=np.float32).shape
            except (TypeError, ValueError):  # ragged, or not numbers: name the group
                wrong[k] = "not a numeric array"
                continue
            if len(shape) != 3 or shape[:2] != (1, history):
                wrong[k] = shape
        if wrong:
            raise ValueError(
                f"gr00t_n17 takes each state group as `(1, {history}, D)`, one request and "
                f"{history} timestep(s); got {wrong}"
            )
        out = self.processor(
            embodiment_tag=tag,
            video=self._views_from_frames(tag, (tensors or {}).get("image_inputs")),
            # An empty instruction is a real observation (8 of the 20 recorded DROID
            # requests carry one) and the wire cannot send it: FastAPI turns an empty
            # form field into None. So no text is the empty instruction, which the
            # reference tokenizes like any other.
            language=[prompt or ""],
            state=kwargs["state"],
        )
        return {name: [tensor] for name, tensor in out.items()}

    def request_kwargs(self) -> frozenset[str]:
        """What a request carries beyond the prompt and its media.

        `walk` selects one of the two **reuse** walks by name; without it a
        request takes the full path. It is how another model, or a caller that
        wants GR00T's backbone features or its encoded state on their own,
        reaches a walk that is otherwise unreachable.
        """
        return frozenset({"embodiment_tag", "state", "walk"})

    def _walk_schedule(self, model_kwargs: dict | None) -> list[str]:
        """The walks this request runs, in order.

        flux2_klein's idiom (`Flux2KleinModel.get_initial_forward_pass_args`):
        the sequence is decided once, carried in `metadata.kwargs`, and advanced
        by the transition hook. Without it a model has no way to express "this
        request runs walk B after walk A", and no way to say it is finished.
        """
        walk = (model_kwargs or {}).get("walk")
        if walk in (self.BACKBONE_WALK, self.STATE_WALK):
            return [walk]
        if walk is not None:
            raise ValueError(
                f"{walk!r} is not selectable; `walk` names one of the reuse "
                f"walks, {self.BACKBONE_WALK!r} or {self.STATE_WALK!r}, and "
                f"omitting it runs the full path. {self.PREFILL_WALK!r} and "
                f"{self.ACTION_GEN_WALK!r} are the full path's own two steps: "
                "neither is a request on its own, and a schedule of "
                f"[{self.PREFILL_WALK!r}] would park its conditioning, finish "
                "and emit nothing at all."
            )
        return [self.PREFILL_WALK, self.ACTION_GEN_WALK]

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        """The decoded action chunk as msgpack: one entry per joint group, as NVIDIA
        returns it.

        GR00T decodes **server-side**, unlike pi05, which returns its normalised
        chunk raw: the inverse needs the checkpoint's action statistics and the
        request's own raw state, neither of which a client has.

        The shape is the reference's, not a wire convention of ours. NVIDIA's
        policy returns `{joint_group: array}` cast to float32
        (`gr00t_policy.py:425-432`, via `decode_action`, whose keys are bare --
        the `action.` prefix is added only by its sim-compatibility wrapper). A
        robot client written against GR00T already indexes by group name, and the
        groups have different widths (9, 1, 7 for DROID), so flattening them
        would make the caller rebuild the layout from the embodiment tag to learn
        which float drives which actuator.

        `msgpack` is already a core dependency. Each group carries its own dtype
        and shape so a client reconstructs it exactly:
        `np.frombuffer(g["data"], g["dtype"]).reshape(g["shape"])`.
        `mstar.communication.wire` is deliberately not reused: that is the
        control plane's serializer, whose numpy extension type a client outside
        M* could not decode; plain msgpack maps and bytes need nothing but msgpack.

        The two kwargs this reads are why they are in `PREPROCESS_ONLY_KWARGS`:
        stripping them from the conductor's copy leaves the data worker's
        (`PreprocessWorkerThread._process_input`), which is the one handed back
        here.

        A reuse walk's features (`backbone_features`, `state_features`) come back
        as one entry laid out like an action group, so the same three lines read
        either.
        """
        if modality == "tensor":
            return msgpack.packb(_entry(output.detach().to(torch.float32).cpu().numpy()))
        if modality != "action":
            raise ValueError(f"gr00t_n17 emits actions or a reuse walk's tensor, not {modality!r}")
        kwargs = request_kwargs or {}
        tag = kwargs.get("embodiment_tag")
        if tag is None:
            raise ValueError(
                "gr00t_n17 cannot decode without the request's `embodiment_tag`: "
                "it selects the action statistics and the joint groups"
            )
        chunk = output.detach().to(torch.float32).cpu().numpy()
        if chunk.ndim == 2:  # one request, as the engine hands it back
            chunk = chunk[None]
        groups = self.processor.unapply(tag, chunk, kwargs.get("state"))
        return msgpack.packb({name: _entry(value) for name, value in groups.items()})

    # -------------------------------------------------------- forward passes
    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        """A request starts at the head of its walk schedule. The full path begins
        in `prefill`, whose two branches are independent; a reuse walk is alone."""
        schedule = self._walk_schedule(model_kwargs)
        metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=schedule[0],
            is_prefill=True,
            kwargs={"walk_schedule": schedule, "walk_step": 0},
        )
        # `image_grid_thw` goes to both: the ViT sizes its segments from it and
        # the LLM places the image rows with it.
        routes = [
            ("pixel_values", VISION_ENCODER), ("image_grid_thw", VISION_ENCODER),
            ("image_grid_thw", LLM), ("input_ids", LLM),
            ("state", STATE_ENCODER), ("embodiment_id", STATE_ENCODER),
        ]
        reachable = self.REUSE_WALK_NODES.get(metadata.graph_walk)
        inputs = []
        for name, node in routes:
            if name not in input_signals:
                continue
            # A reuse walk holds a subset of `prefill`'s nodes, but the request
            # that drives it carries the whole observation either way:
            # `process_prompt` returns everything the processor produces and
            # `Gr00tN17Preprocessor.__call__` adds `state` and `embodiment_id`
            # unconditionally, so a `state` request still arrives with
            # `pixel_values`. Skip what this walk cannot reach, exactly as the
            # line above skips what the request did not send -- routing it would
            # ship an edge to a node outside the walk, which is never consumed
            # and never blamed.
            if reachable is not None and node not in reachable:
                continue
            edge = GraphEdge(next_node=node, name=name)
            edge.tensor_info = input_signals[name]
            # the denoise step needs the embodiment id on every iteration, so the
            # request's copy has to stay parked rather than be freed with prefill
            # -- but only when there *is* a loop: a reuse walk ends here, and a
            # parked tensor nothing unpersists is a leak for the request's life.
            edge.persist = name == "embodiment_id" and self.ACTION_GEN_WALK in schedule
            inputs.append(edge)
        parked = {
            info.uuid for inp in inputs if inp.persist for info in inp.tensor_info
        }
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            # Everything the request arrived with that this walk does not park.
            # The routed inputs are the obvious half; the rest matter just as
            # much, because `PreprocessWorkerThread._store_persisted` persists
            # *every* initial signal and only the model can say it is done with one.
            # Some never reach a node: what the processor always produces and no
            # node reads (`action_mask`; the backbone processor's `attention_mask`
            # and, in some transformers versions, `mm_token_type_ids`), and -- on a
            # reuse walk -- whatever that walk's nodes cannot consume. Releasing by
            # uuid rather than by edge covers all of them without naming any.
            unpersist_tensors=[
                info for infos in input_signals.values() for info in infos
                if info.uuid not in parked
            ],
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """`prefill` once, then `action_gen` four times, then done.

        The bucket key's span counts are set here and read back by
        ``Gr00tN17DenoiseStepSubmodule.bucket_key_for``. They come off the
        **persisted tensors' own `dims`** rather than a separately kept count:
        a count that is derived from the thing it describes cannot disagree with
        it, which is most of what the node's own check guards against.
        """
        metadata = partition_metadata
        inputs: list[GraphEdge] = []
        step_metadata: dict = {"is_prefill": False}
        metadata.is_prefill = False

        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        # Past the end means finished. A reuse walk is a schedule of one, so it
        # lands here on its first transition rather than re-emitting itself with
        # no inputs -- which the conductor reads as "self-triggers via
        # StreamBuffer" (`Conductor._process_done_forward`) and GR00T has no
        # stream, so the partition would idle forever.
        request_done = step >= len(schedule)
        if request_done:
            return ForwardPassArgs(
                full_metadata=metadata, inputs=[], unpersist_tensors=[],
                step_metadata=step_metadata, request_done=True,
            )

        metadata.kwargs["walk_step"] = step
        metadata.graph_walk = schedule[step]
        if metadata.graph_walk == self.ACTION_GEN_WALK:
            # Re-route what `prefill` parked. The conductor hands the model
            # `persist_signals` and nothing else reads it: flux2_klein and ltx2_5
            # both build one edge per persisted name here, and without this the
            # node's inputs never complete.
            for name in ("vl_text", "vl_image", "state_features", "embodiment_id"):
                edge = GraphEdge(next_node=DENOISE_STEP, name=name)
                edge.tensor_info = persist_signals.get(name, [])
                inputs.append(edge)
            # the carried edge arrives empty; the scaffold seeds it at iteration 0
            inputs.append(GraphEdge(next_node=DENOISE_STEP, name="latents"))
            step_metadata |= {
                "vl_text_rows": persist_signals["vl_text"][0].dims[0],
                "vl_image_rows": persist_signals["vl_image"][0].dims[0],
            }

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([inp.tensor_info for inp in inputs], start=[]),
            step_metadata=step_metadata,
            request_done=request_done,
        )
