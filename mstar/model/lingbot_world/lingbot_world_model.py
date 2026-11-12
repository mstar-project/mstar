"""LingBotWorldModel: LingBot-World-V2 causal-fast, camera-controlled image-to-world.

Architecture (2 nodes; the dit declares its K/V, its text K/V and its draws' state):
    text_encoder - UMT5-XXL prompt encoder (wan22's submodule, reused)
    dit          - the 1.3B causal DiT. One rollout iteration is one chunk of
                   4 latent frames: the chunk's I2V condition (a warm-cache
                   Wan2.1 VAE encode of the image, then black frames) and
                   Plucker rays, 4 distilled denoise passes, one commit pass
                   into the rolling KV cache, then a warm-cache Wan VAE decode
                   of the chunk, fused into one forward (as Waypoint fuses its
                   per-frame decode)

Graph walks (2), run in order:
    encode_text  - text_encoder; persists text_embeds_pos
    rollout      - Loop("rollout_loop") over dit with self loop-backs ("clock" and
                   the VAE encoder's and decoder's caches); each iteration
                   emits its chunk's frames (``video_frame``)

The camera trajectory is the action input: per-frame OpenCV c2w poses plus
intrinsics, as in the reference's ``poses.npy`` / ``intrinsics.npy``. The
reference resamples and normalizes it over the whole rollout, so the request
carries the whole trajectory up front (see ``components/camera.py``). It
travels in the ``model_kwargs`` form part, which the API server caps at 1 MiB:
about 3,900 ``[4, 4]`` float32 poses, or all ``max_chunks`` worth of frames as
``[3, 4]`` rows.
"""

import html
import logging

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.resources.kv.config import DEFAULT_PAGE_SIZE
from mstar.engine.state import Array, Attend, Bidirectional, Block, Entry, KVEntry, Layout, Read, Value
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, TensorPointerInfo
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import KVLabel, Model, ProcessPromptOutput, TensorAndMetadata, WalkStep
from mstar.model.lingbot_world.components.camera import latent_camera_poses, scale_intrinsics
from mstar.model.lingbot_world.config import (
    LINGBOT_CONFIGS,
    LINGBOT_HF_REPOS,
    LINGBOT_VARIANT_1_3B,
    UMT5_DIFFUSERS_REPO,
    WAN21_VAE_DIFFUSERS_REPO,
)
from mstar.model.lingbot_world.submodules import (
    CHUNK_OUTPUTS,
    DECODER_CACHE,
    DIT_KV,
    ENCODER_CACHE,
    PROGRESS,
    ROLLOUT_LOOP_NAME,
    ROLLOUT_WALK,
    SELF_ATTN,
    TEXT,
    TEXT_ATTN,
    LingBotDitSubmodule,
)
from mstar.model.media_limits import PILLOW_IMAGE_FORMATS, check_image_size, decoding_upload, open_upright_image
from mstar.model.submodule_base import NodeSubmodule
from mstar.model.utils import check_kv_pools_fit

logger = logging.getLogger(__name__)

# LingBotWorldConfig switches a deployment sets in model_kwargs
_CONFIG_KWARGS = ("reference_numerics", "dit_compile_autotune", "keep_chunk_outputs")

# The reference's numerics: FA2 attention, an eager DiT, the fp32 contiguous eager encode and decode, with which
# the rollout is bit-identical to the reference's (test/lingbot_world/run_parity.py).
REFERENCE_MODEL_KWARGS = {"reference_numerics": True}

ENCODE_TEXT_WALK = "encode_text"
_DIT_INPUTS = ("text_embeds_pos", "image_inputs", "camera_poses", "intrinsics", "clock", ENCODER_CACHE, DECODER_CACHE)


class LingBotWorldModel(Model):
    def __init__(
        self,
        model_path_hf: str | None = None,
        cache_dir: str | None = None,
        variant: str = LINGBOT_VARIANT_1_3B,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        if variant != LINGBOT_VARIANT_1_3B:
            raise NotImplementedError(f"LingBot-World variant {variant!r} is not implemented yet.")
        self.config = LINGBOT_CONFIGS[variant]()
        for name in _CONFIG_KWARGS:
            if name in kwargs:
                value = kwargs.pop(name)
                # a yaml "false" is a truthy string
                if not isinstance(value, bool):
                    raise ValueError(f"LingBot-World {name} must be true or false, got {value!r}.")
                setattr(self.config, name, value)
        if kwargs:
            raise ValueError(
                f"LingBot-World takes no model_kwargs {sorted(kwargs)}; its switches are {', '.join(_CONFIG_KWARGS)}."
            )
        self.model_path_hf = model_path_hf or LINGBOT_HF_REPOS[variant]
        self.cache_dir = cache_dir
        self.skip_weight_loading = skip_weight_loading
        self.tokenizer = None
        self._submodule_cache: dict[str, NodeSubmodule | None] = {}

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def declare_state(self) -> list[Array]:
        """Each chunk's self-attention K/V, of which a chunk reads the sink and the latest window; where the
        request's draws stand; and, unless ``reference_numerics`` attends with the reference's FA2, each chunk's text
        K/V, whose queries are the chunk's."""
        cfg = self.config
        blocks = frozenset(range(cfg.num_layers))
        entry = KVEntry(heads=cfg.num_heads, head_dim=cfg.head_dim)
        op = None if cfg.reference_numerics else Attend(q_heads=cfg.num_heads)
        arrays = [
            Array(DIT_KV, nodes={"dit"}, layers=blocks, entry=entry, max_len=cfg.max_rollout_tokens(),
                  reads={SELF_ATTN: Read(mask=Bidirectional(kept_window=True), op=op)}),
            Value(PROGRESS, nodes={"dit"}, layers=frozenset({0}), entry=Entry({}, host={"rng": bytes})),
        ]
        if not cfg.reference_numerics:
            arrays.append(Array(
                TEXT, nodes={"dit"}, layers=blocks, entry=entry, max_len=0, labels={"text"},
                reads={TEXT_ATTN: Read(mask=Block(), op=Attend(q_heads=cfg.num_heads), queries=DIT_KV, packed=False)},
            ))
        return arrays

    def kernel_settings(self) -> dict:
        """A pool for one rollout's window at the largest area, beside its sink page."""
        cfg = self.config
        return {DIT_KV: {"max_num_pages": 1 + cfg.window_pages(cfg.max_frame_tokens, DEFAULT_PAGE_SIZE)}}

    def kv_labels(self) -> dict[str, dict[str, KVLabel]]:
        return {DIT_KV: {"main": KVLabel(growth=((ROLLOUT_WALK, ROLLOUT_LOOP_NAME),))}}

    def state_layout(self, lat_h: int, lat_w: int) -> dict[str, dict[str, Layout]]:
        """What a rollout at a ``lat_h`` by ``lat_w`` latent writes: a chunk's tokens an iteration of the rollout loop,
        of which the pool keeps the sink and the latest window."""
        cfg = self.config
        frame = cfg.frame_tokens(lat_h, lat_w)
        return {DIT_KV: {"main": Layout(
            growth=((ROLLOUT_LOOP_NAME, cfg.chunk_size * frame),), kept_windows={SELF_ATTN: cfg.kept_window(frame)},
        )}}

    def validate_config_yaml(self, config: dict, config_path: str) -> None:
        """Refuse to serve more than one rollout at a time, and a pool short of one rollout at the default size.

        The pool admits a rollout's K/V, and at small output sizes it admits many. Each one's VAE caches go from
        chunk to chunk on the rollout loop's edges, outside admission, so the conductor's admit queue, which exists
        only when ``max_concurrent_requests`` is set, has to hold them to one rollout.
        """
        limit = config.get("max_concurrent_requests")
        if not (type(limit) is int and limit == 1):
            raise ValueError(
                f"LingBot-World requires `max_concurrent_requests: 1` in {config_path}; got {limit!r}. "
                "The K/V pool admits each rollout's K/V, not its VAE caches, which go from chunk to chunk on the "
                "rollout loop's edges, so the admit queue that this key enables is the only bound on them."
            )
        cfg = self.config
        size = (cfg.default_height, cfg.default_width)
        lat_h, lat_w = self.geometry(size, *size)
        check_kv_pools_fit(self, config, config_path, logger, request_layouts=self.state_layout(lat_h, lat_w),
                           loop_caps={ROLLOUT_LOOP_NAME: cfg.max_chunks})

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_text = GraphNode(
            name="text_encoder",
            input_names=["text_inputs"],
            outputs=[
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_pos", persist=True),
                GraphEdge(next_node=EMPTY_DESTINATION, name="text_embeds_neg", persist=True),
            ],
        )
        rollout = Loop(
            name=ROLLOUT_LOOP_NAME,
            section=GraphNode(
                name="dit",
                input_names=list(_DIT_INPUTS),
                outputs=[
                    GraphEdge(next_node="dit", name="clock"),
                    GraphEdge(next_node="dit", name=ENCODER_CACHE),
                    GraphEdge(next_node="dit", name=DECODER_CACHE),
                    GraphEdge(next_node=EMIT_TO_CLIENT, name="video_output", output_modality="video_frame"),
                    *(GraphEdge(next_node=EMPTY_DESTINATION, name=name)
                      for name in CHUNK_OUTPUTS if self.config.keep_chunk_outputs),
                ],
                enable_async_scheduling=True,
            ),
            max_iters=self.config.max_chunks,
            outputs=[],
            accumulated_outputs=[],
        )
        return {ENCODE_TEXT_WALK: encode_text, ROLLOUT_WALK: rollout}

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                UMT5_DIFFUSERS_REPO, subfolder="tokenizer", cache_dir=self.cache_dir,
            )

    def geometry(self, image_hw: tuple[int, int], height: int, width: int) -> tuple[int, int]:
        """The reference's output size: the image's aspect at ``height * width``
        area, floored to whole DiT patches. Returns (lat_h, lat_w)."""
        cfg = self.config
        aspect_ratio = image_hw[0] / image_hw[1]
        max_area = height * width
        lat_h = round(np.sqrt(max_area * aspect_ratio) // cfg.vae_stride[1] // cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // cfg.vae_stride[2] // cfg.patch_size[2] * cfg.patch_size[2])
        return int(lat_h), int(lat_w)

    @staticmethod
    def prompt_clean(text: str) -> str:
        """The reference T5 tokenizer's ``clean='whitespace'``: ftfy, double
        html-unescape, then whitespace collapse with ``regex``. ftfy is what
        wan22 skips; it changes real prompts (curly quotes become straight)."""
        import ftfy
        import regex

        text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
        return regex.sub(r"\s+", " ", text).strip()

    def num_latent_frames(self, num_frames: int, num_poses: int) -> int:
        """The reference's rollout length: frames clamped to the trajectory,
        then floored to whole chunks of latent frames."""
        cfg = self.config
        num_poses = ((num_poses - 1) // 4) * 4 + 1
        num_frames = min(((num_frames - 1) // 4) * 4 + 1, num_poses)
        lat_f = (num_frames - 1) // cfg.vae_stride[0] + 1
        return lat_f - lat_f % cfg.chunk_size

    def request_kwargs(self) -> frozenset[str]:
        return frozenset({"camera_poses", "intrinsics", "num_frames", "height", "width"})

    def _latent_frames(self, model_kwargs: dict) -> int:
        num_poses = len(model_kwargs["camera_poses"])
        # num_frames is an integer by now: process_prompt refused anything else
        return self.num_latent_frames(model_kwargs.get("num_frames", num_poses), num_poses)

    def get_loop_caps(self, **model_kwargs) -> dict[str, int]:
        # one rollout iteration per chunk of latent frames; process_prompt refused a request without poses
        return {ROLLOUT_LOOP_NAME: self._latent_frames(model_kwargs) // self.config.chunk_size}

    def process_prompt(
        self, prompt: str | None, input_modalities: list[str], output_modalities: list[str],
        tensors: NameToTensorList | None = None, **kwargs,
    ) -> ProcessPromptOutput:
        """Validate, tokenize the prompt, resize the image and resample the
        camera trajectory, on the data worker.

        Request kwargs: ``camera_poses`` (``[F, 4, 4]`` OpenCV c2w, one per video
        frame, or its top ``[F, 3, 4]`` rows), ``intrinsics`` (``[fx, fy, cx, cy]`` at 480x832, or one row per
        frame, of which the first is used), ``num_frames``, ``height`` / ``width``
        (the output area, at most ``config.max_area``; the aspect follows the image).
        """
        cfg = self.config
        if output_modalities != ["video_frame"]:
            raise ValueError(f"LingBot-World emits raw frames ('video_frame'); got {output_modalities!r}.")
        if not prompt:
            raise ValueError("LingBot-World requires a text prompt.")
        images = (tensors or {}).get("image_inputs")
        if not images:
            raise ValueError("LingBot-World requires a conditioning image.")
        raw_poses = kwargs.get("camera_poses")
        poses = None if raw_poses is None else np.asarray(raw_poses, dtype=np.float64)
        min_poses = 1 + cfg.vae_stride[0] * (cfg.chunk_size - 1)
        if poses is None or poses.ndim != 3 or poses.shape[1:] not in ((4, 4), (3, 4)) or len(poses) < min_poses:
            raise ValueError(f"LingBot-World requires camera_poses: [F, 4, 4] or [F, 3, 4] with F >= {min_poses}.")
        if poses.shape[1] == 3:
            bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (len(poses), 1, 4))
            poses = np.concatenate([poses, bottom], axis=1)
        intrinsics = torch.as_tensor(kwargs.get("intrinsics", ()), dtype=torch.float32)
        if intrinsics.numel() < 4 or intrinsics.numel() % 4:
            raise ValueError("LingBot-World requires intrinsics [fx, fy, cx, cy] (or one row per frame).")
        if not (np.isfinite(poses).all() and torch.isfinite(intrinsics).all()):
            raise ValueError("LingBot-World camera_poses and intrinsics must be finite numbers.")

        num_frames = kwargs.get("num_frames", len(poses))
        lat_f = self._latent_frames(kwargs)
        if lat_f < cfg.chunk_size or lat_f > cfg.max_chunks * cfg.chunk_size:
            raise ValueError(f"LingBot-World rollout of {lat_f} latent frames is out of range.")
        num_poses = ((len(poses) - 1) // 4) * 4 + 1
        frame_num = min(((num_frames - 1) // 4) * 4 + 1, num_poses)

        image = self._preprocess_image(images[0], kwargs)
        lat_h, lat_w = image.shape[1] // cfg.vae_stride[1], image.shape[2] // cfg.vae_stride[2]
        height, width = lat_h * cfg.vae_stride[1], lat_w * cfg.vae_stride[2]

        self._ensure_tokenizer()
        text = self.prompt_clean(prompt)
        ids = self.tokenizer(text, truncation=True, max_length=cfg.text_len, add_special_tokens=True).input_ids
        return ProcessPromptOutput({
            "text_inputs": [torch.tensor(ids, dtype=torch.long)],
            "image_inputs": [image],
            "camera_poses": [latent_camera_poses(poses[:frame_num], lat_f)],
            "intrinsics": [scale_intrinsics(intrinsics, cfg.intrinsics_height, cfg.intrinsics_width, height, width)],
        }, {"state_layout": self.state_layout(lat_h, lat_w)})

    def _preprocess_image(self, image: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """``[3, h, w]`` float32 in [-1, 1] at the output size: the reference's
        ``to_tensor -> (x - 0.5) / 0.5 -> bicubic resize``, on CPU as there."""
        cfg = self.config
        if image.dim() == 3 and image.shape[-1] == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        image = image.cpu()
        image = image.float().div(255) if image.dtype == torch.uint8 else image.float()
        image = image.sub(0.5).div(0.5)
        height = kwargs.get("height", cfg.default_height)
        width = kwargs.get("width", cfg.default_width)
        # the area sizes the resize here and the DiT's tokens and rolling K/V
        if height <= 0 or width <= 0 or height * width > cfg.max_area:
            raise ValueError(
                f"LingBot-World height and width must be positive with an area of at most {cfg.max_area} "
                f"pixels; got {height}x{width}."
            )
        lat_h, lat_w = self.geometry(tuple(image.shape[1:]), height, width)
        if lat_h <= 0 or lat_w <= 0:
            raise ValueError(f"LingBot-World cannot fit a {image.shape[1]}x{image.shape[2]} image at that area.")
        size = (lat_h * cfg.vae_stride[1], lat_w * cfg.vae_stride[2])
        return torch.nn.functional.interpolate(image[None], size=size, mode="bicubic")[0]

    def load_image(self, filepath: str, device: str) -> TensorAndMetadata:
        """uint8 ``[3, H, W]`` as the reference reads it, with Pillow's ``convert("RGB")``
        (alpha dropped, not composited), upright by its EXIF orientation; the resize and normalization are
        process_prompt's."""
        check_image_size(filepath, formats=PILLOW_IMAGE_FORMATS)
        with (
            decoding_upload(filepath, "the image"),
            open_upright_image(filepath, formats=PILLOW_IMAGE_FORMATS) as image,
        ):
            rgb = np.array(image.convert("RGB"))
        return TensorAndMetadata(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())

    def postprocess(self, output: torch.Tensor, modality: str, request_kwargs: dict | None = None) -> bytes:
        if modality != "video_frame":
            raise ValueError(f"Unsupported modality for LingBot-World: {modality!r}")
        return output.detach().cpu().contiguous().numpy().tobytes()

    def get_output_frame_rate(self, modality: str = "video_frame", request_kwargs: dict | None = None) -> float:
        return float(self.config.video_fps)

    # ------------------------------------------------------------------
    # Forward pass orchestration
    # ------------------------------------------------------------------

    def walk_plan(
        self, partition_name: str, input_modalities: list[str], output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]], model_kwargs: dict,
    ) -> list[WalkStep]:
        for name in ("text_inputs", "image_inputs", "camera_poses", "intrinsics"):
            if not input_signals.get(name):
                raise ValueError(f"LingBot-World is missing its {name!r} input; process_prompt emits it.")
        lat_f = int(input_signals["camera_poses"][0].dims[0])
        request = {"guidance_scale": 1.0, "num_latent_frames": lat_f}
        return [WalkStep(walk, step_metadata=request) for walk in (ENCODE_TEXT_WALK, ROLLOUT_WALK)]

    # ------------------------------------------------------------------
    # Submodule loading
    # ------------------------------------------------------------------

    def get_autocast_dtype(self):
        """The K/V pool's and the attention kernels' bf16. Neither node runs under autocast: the DiT enters the
        reference's autocast contexts itself, and the VAE and the text encoder run in the dtypes they load in."""
        return torch.bfloat16

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None, sp_group=None,
    ) -> torch.nn.Module | None:
        if node_name not in self._submodule_cache:
            self._submodule_cache[node_name] = self._create_submodule(node_name, device)
        return self._submodule_cache[node_name]

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        if self.skip_weight_loading:
            return None
        if node_name == "text_encoder":
            from transformers import UMT5EncoderModel

            from mstar.model.wan22.config import Wan22Config
            from mstar.model.wan22.submodules import Wan22TextEncoderSubmodule

            # The reference T5 is eager attention; transformers 5 defaults UMT5 to SDPA.
            encoder = UMT5EncoderModel.from_pretrained(
                UMT5_DIFFUSERS_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir, attn_implementation="eager",
            ).eval()
            text_encoder = Wan22TextEncoderSubmodule(encoder, Wan22Config(text_max_seq_len=self.config.text_len))
            text_encoder.disable_autocast = True
            return text_encoder
        if node_name == "dit":
            from mstar.model.lingbot_world.components.dit import require_flash_attn
            from mstar.model.lingbot_world.weight_loader import build_lingbot_dit

            if self.config.reference_numerics:
                require_flash_attn()
            dit = build_lingbot_dit(self.config, self.model_path_hf, device=device, cache_dir=self.cache_dir)
            cfg = self.config
            return LingBotDitSubmodule(
                dit, self._load_vae().to(device), cfg, encoder_vae=self._load_vae().to(device),
                warmup_latent_sizes=[self.geometry((cfg.default_height, cfg.default_width),
                                                   cfg.default_height, cfg.default_width)],
            )
        logger.warning("LingBot-World has no submodule for node %r; running it dummy.", node_name)
        return None

    def _load_vae(self) -> torch.nn.Module:
        """A fresh fp32 Wan2.1 VAE per use: the decoder makes its copy bf16 and
        channels-last unless ``reference_numerics``, which the encoder must not see."""
        from diffusers import AutoencoderKLWan

        return AutoencoderKLWan.from_pretrained(
            WAN21_VAE_DIFFUSERS_REPO, subfolder="vae", torch_dtype=torch.float32, cache_dir=self.cache_dir,
        ).eval()
