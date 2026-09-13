"""OmniVoice: massively multilingual zero-shot TTS by masked diffusion.

Two graph walks:

``encode_reference``
    Codec-encode the cloning reference once; the tokens persist so repeat
    requests against the same voice skip it.  Cloning requests only.

``speech_gen`` / ``speech_gen_clone``
    ``Loop(backbone) -> code2wav -> client``.  Separate walks per mode so the
    backbone's ``input_names`` are exact rather than carrying an empty
    reference edge through the non-cloning path — wan22's T2V/I2V split.

What M* adds over running the reference's own server: several requests share a
diffusion step, and every decoding knob stays per-request.  ``num_step``,
``guidance_scale``, ``duration`` and ``instruct`` ride ``step_metadata`` and are
resolved once per request, so two callers can ask for 16 and 32 steps in the
same batch.  A fixed-pipeline port has to pick one value for the whole server.
"""

import logging
import os

import numpy as np
import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import NodeResourceSpec
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Sequential,
    TensorPointerInfo,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.omnivoice.components.backbone import (
    OmniVoiceBackbone,
    assert_flashinfer_api,
)
from mstar.model.omnivoice.components.codec import load_codec
from mstar.model.omnivoice.components.text import build_prefix
from mstar.model.omnivoice.config import OmniVoiceConfig
from mstar.model.omnivoice.submodules import (
    UNMASK_LOOP_NAME,
    OmniVoiceBackboneSubmodule,
    OmniVoiceCode2WavSubmodule,
    OmniVoiceRefEncoderSubmodule,
)
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


class OmniVoiceModel(Model):
    """k2-fsa/OmniVoice."""

    ENCODE_REFERENCE_WALK = "encode_reference"
    SPEECH_GEN_WALK = "speech_gen"
    SPEECH_GEN_CLONE_WALK = "speech_gen_clone"

    UNMASK_LOOP_NAME = UNMASK_LOOP_NAME

    def __init__(
        self,
        model_path_hf: str,
        cache_dir: str | None = None,
        skip_weight_loading: bool = False,
        **kwargs,
    ):
        # A deployment points at a mounted checkpoint rather than the Hub: the
        # registry entry names the public k2-fsa weights, but MISA serves its
        # own fine-tune from object storage, and the two are the same
        # architecture. _refresh_checkpoint_defaults is what catches a
        # fine-tune that actually diverged.
        self.model_path_hf = os.environ.get("MSTAR_OMNIVOICE_MODEL_PATH") or model_path_hf
        self.cache_dir = cache_dir
        self.config = OmniVoiceConfig()
        self.skip_weight_loading = skip_weight_loading

        # Loaded lazily on the data worker so construction never hits the
        # network; the compute workers never touch either.
        self.tokenizer = None
        self._duration_estimator = None

        self._submodule_cache: dict[str, NodeSubmodule | None] = {}
        self._codec: torch.nn.Module | None = None
        self._codec_hop: int | None = None

    # ------------------------------------------------------------------
    # Model ABC: structure
    # ------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """No engine-built resources: every node here is stateless.

        There is deliberately no KV cache.  The canvas is rewritten each
        iteration and attention is bidirectional, so the prefix attends *into*
        the region that changed and its hidden states change with it — nothing
        computed at step k is valid at step k+1.  Sampling is not a sampler
        resource either: the reveal picks cells by confidence rank across the
        whole canvas rather than drawing one token per position.
        """
        return []

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        encode_reference = GraphNode(
            name="ref_encoder",
            input_names=["ref_audio_inputs"],
            outputs=[
                GraphEdge(
                    next_node=EMPTY_DESTINATION,
                    name="ref_audio_tokens",
                    persist=True,
                ),
            ],
        )
        return {
            self.ENCODE_REFERENCE_WALK: encode_reference,
            self.SPEECH_GEN_WALK: self._build_speech_walk(clone=False),
            self.SPEECH_GEN_CLONE_WALK: self._build_speech_walk(clone=True),
        }

    def _build_speech_walk(self, clone: bool) -> GraphSection:
        """Unmask loop, then codec decode.

        The prefix edges are inputs to the loop rather than loop-carried: they
        are constant for the request, and the conductor re-injects external
        inputs at the start of each iteration.
        """
        backbone_inputs = [
            "prefix_ids",
            "prefix_audio_mask",
            "target_len",
            "audio_tokens",
            "step_index",
        ]
        if clone:
            # Only the cloning walk carries this edge, so the non-cloning walk
            # does not wait on an input that will never arrive. The backbone
            # appends these tokens to the prefix — the canvas layout is
            # [style | text | ref | target], so the reference goes last.
            backbone_inputs.insert(2, "ref_audio_tokens")

        unmask_loop = Loop(
            name=UNMASK_LOOP_NAME,
            section=GraphNode(
                name="backbone",
                input_names=backbone_inputs,
                outputs=[
                    GraphEdge(next_node="backbone", name="audio_tokens"),
                    GraphEdge(next_node="backbone", name="step_index"),
                ],
                # Async scheduling may dispatch an iteration past the stop;
                # OmniVoiceBackboneSubmodule.prepare_inputs vetoes the
                # overshoot by returning None.
                enable_async_scheduling=True,
            ),
            # Ceiling only; the request's own num_step stops the loop early.
            max_iters=self.config.generation.max_num_step,
            outputs=[
                # Matched by name against the section's loop-back edges: the
                # final iteration's canvas routes onward to the codec.
                GraphEdge(next_node="code2wav", name="audio_tokens"),
            ],
        )

        code2wav_inputs = ["audio_tokens"]
        if clone:
            # Level matching only has a target when there is a reference.
            code2wav_inputs.append("ref_rms")

        code2wav = GraphNode(
            name="code2wav",
            input_names=code2wav_inputs,
            outputs=[
                GraphEdge(
                    next_node=EMIT_TO_CLIENT,
                    name="audio_output",
                    output_modality="audio",
                ),
            ],
        )
        return Sequential([unmask_loop, code2wav])

    # ------------------------------------------------------------------
    # Model ABC: I/O
    # ------------------------------------------------------------------

    def load_audio(self, filepath: str, device: str):
        """Decode the cloning reference at the **codec's** rate, not the default.

        ``Model.load_audio`` decodes at 16 kHz, which is right for the ASR
        models in the tree and wrong here: the Higgs-Audio-v2 codec encodes at
        24 kHz, and feeding it 16 kHz samples would not error — it would encode
        a reference pitched and paced wrong, and clone that.
        """
        from torchcodec.decoders import AudioDecoder

        from mstar.model.base import TensorAndMetadata

        rate = self.config.sample_rate
        decoder = AudioDecoder(filepath, sample_rate=rate, num_channels=1)
        audio = decoder.get_all_samples().data[0]
        return TensorAndMetadata(
            data=audio, metadata=dict(sample_rate=rate, num_channels=1)
        )

    def _ensure_data_worker_assets(self):
        """Load the tokenizer and duration estimator on first use."""
        if self.tokenizer is not None:
            return
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path_hf, cache_dir=self.cache_dir
        )
        from omnivoice.utils.duration import RuleDurationEstimator

        self._duration_estimator = RuleDurationEstimator()

    def _estimate_target_tokens(
        self,
        text: str,
        ref_text: str | None,
        num_ref_audio_tokens: int | None,
        speed: float,
    ) -> int:
        """Target canvas length in codec frames.

        Without a reference the estimator still needs one to calibrate against,
        so the reference's own stand-in pair is used — a short English phrase
        and the frame count it occupies.
        """
        if num_ref_audio_tokens is None or not ref_text:
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25
        est = self._duration_estimator.estimate_duration(
            text, ref_text, num_ref_audio_tokens
        )
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        """Validate, then build the canvas prefix on the data worker.

        Every malformed-request check lives here rather than in
        ``get_initial_forward_pass_args``: a ValueError here becomes a 400,
        while the same raise at the conductor is swallowed and the client hangs.

        The prefix is built once and shipped, not rebuilt per iteration —
        tokenizing the text 32 times would be the dominant cost for a short
        utterance.
        """
        if not prompt:
            raise ValueError("OmniVoice requires a non-empty text prompt")
        if set(output_modalities) != {"audio"}:
            raise ValueError("OmniVoice generates audio only")

        self._ensure_data_worker_assets()

        # The data worker keys loaded media by modality: an uploaded reference
        # arrives as "audio_inputs". It is re-emitted under the graph's own
        # edge name below so the node contract does not depend on that.
        ref_audio = (tensors or {}).get("audio_inputs")
        ref_text = kwargs.get("ref_text")
        if ref_audio and not ref_text:
            # The reference auto-transcribes with Whisper when ref_text is
            # missing. That is a second model on the serving path and a second
            # failure mode, so it is not in v1: the caller supplies the
            # transcript, or clones without one at reduced quality.
            raise ValueError(
                "Voice cloning needs ref_text alongside ref_audio. "
                "Auto-transcription is not served; transcribe the reference "
                "yourself and pass the text."
            )

        num_ref_tokens = None
        if ref_audio:
            waveform = ref_audio[0]
            # Exact, not estimated: encode downsamples by a whole hop, so the
            # frame count follows from the sample count and the canvas length
            # can be settled before the GPU encode runs.
            hop = self._codec_hop_length()
            num_ref_tokens = int(waveform.shape[-1]) // hop
            if num_ref_tokens < 1:
                raise ValueError(
                    f"Reference audio is shorter than one codec hop ({hop} samples)"
                )

        speed = float(kwargs.get("speed") or 1.0)
        duration = kwargs.get("duration")
        if duration is not None:
            target_len = self.config.target_tokens_for_seconds(float(duration))
        else:
            target_len = self._estimate_target_tokens(
                prompt, ref_text, num_ref_tokens, speed
            )

        seconds = self.config.seconds_for_target_tokens(target_len)
        if seconds > self.config.max_target_seconds:
            # The reference splits past this point into crossfaded chunks. That
            # is a second graph walk and is not in v1, so the request is
            # rejected rather than silently truncated.
            raise ValueError(
                f"Requested speech is {seconds:.1f}s, over the {self.config.max_target_seconds:.0f}s "
                "single-canvas limit. Split the text and concatenate, or pass a "
                "shorter duration."
            )

        prefix_ids, prefix_audio_mask = build_prefix(
            tokenizer=self.tokenizer,
            text=prompt,
            num_audio_codebook=self.config.num_audio_codebook,
            language=kwargs.get("language"),
            instruct=kwargs.get("instruct"),
            ref_text=ref_text,
            # The reference tokens are not available on the data worker; the
            # ref_encoder walk supplies them, and get_initial_forward_pass_args
            # splices them into the prefix before the loop starts.
            ref_audio_tokens=None,
            denoise=bool(kwargs.get("denoise", self.config.generation.denoise)),
        )

        out: NameToTensorList = {
            "prefix_ids": [prefix_ids],
            "prefix_audio_mask": [prefix_audio_mask],
            "target_len": [torch.tensor([target_len], dtype=torch.long)],
        }
        if ref_audio:
            out["ref_audio_inputs"] = [ref_audio[0]]
            # The reference matches the clone's level to the source's RMS, so a
            # quiet reference yields quiet speech. Measured here, where the
            # waveform already is: code2wav only ever sees tokens.
            out["ref_rms"] = [
                torch.tensor(
                    [float(ref_audio[0].float().pow(2).mean().sqrt())],
                    dtype=torch.float32,
                )
            ]
        return out

    def _codec_hop_length(self) -> int:
        """The codec's hop, read from its config without loading the weights."""
        if self._codec is not None:
            return int(self._codec.config.hop_length)
        if self._codec_hop is not None:
            return self._codec_hop
        from transformers import AutoConfig

        from mstar.model.omnivoice.config import CODEC_SUBFOLDER

        cfg = AutoConfig.from_pretrained(
            self.model_path_hf, subfolder=CODEC_SUBFOLDER, cache_dir=self.cache_dir
        )
        self._codec_hop = int(cfg.hop_length)
        return self._codec_hop

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        model_kwargs = model_kwargs or {}
        # Backstops. process_prompt already rejected each of these on the data
        # worker where a ValueError becomes a 400; a raise here runs at the
        # conductor, whose loop swallows it and leaves the client hanging.
        if not input_signals.get("prefix_ids"):
            raise ValueError("OmniVoice requires a processed prompt (prefix_ids)")
        target_output = output_modalities[0] if output_modalities else "audio"
        if target_output != "audio":
            raise ValueError(
                f"OmniVoice only generates audio; got output modality {target_output!r}."
            )

        is_clone = bool(input_signals.get("ref_audio_inputs"))
        schedule = []
        if is_clone:
            schedule.append(self.ENCODE_REFERENCE_WALK)
        schedule.append(
            self.SPEECH_GEN_CLONE_WALK if is_clone else self.SPEECH_GEN_WALK
        )

        defaults = self.config.generation
        requested_steps = int(model_kwargs.get("num_step", defaults.num_step))
        num_step = max(1, min(requested_steps, defaults.max_num_step))
        if num_step != requested_steps:
            logger.info(
                "Clamped num_step from %d to %d (max_num_step=%d)",
                requested_steps, num_step, defaults.max_num_step,
            )

        kwargs = {
            "walk_schedule": schedule,
            "walk_step": 0,
            # target_len is NOT here: it is computed on the data worker (it
            # needs the tokenizer and the duration estimator) and rides an edge
            # to the backbone, rather than being re-derived at the conductor.
            "num_step": num_step,
            "guidance_scale": float(
                model_kwargs.get("guidance_scale", defaults.guidance_scale)
            ),
            "t_shift": float(model_kwargs.get("t_shift", defaults.t_shift)),
            "layer_penalty_factor": float(
                model_kwargs.get("layer_penalty_factor", defaults.layer_penalty_factor)
            ),
            "position_temperature": float(
                model_kwargs.get("position_temperature", defaults.position_temperature)
            ),
            "class_temperature": float(
                model_kwargs.get("class_temperature", defaults.class_temperature)
            ),
            "postprocess_output": bool(model_kwargs.get("postprocess_output", True)),
            "pad_duration": float(model_kwargs.get("pad_duration", 0.1)),
            "fade_duration": float(model_kwargs.get("fade_duration", 0.1)),
        }

        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk=schedule[0],
            is_prefill=True,
            kwargs=kwargs,
        )

        # Only the FIRST walk is seeded here. Everything else stays persisted
        # and is picked up from ``persist_signals`` when the schedule steps --
        # feeding a later walk's node now would target a node this walk does
        # not contain.
        if is_clone:
            edge = GraphEdge(next_node="ref_encoder", name="ref_audio_inputs")
            edge.tensor_info = input_signals["ref_audio_inputs"]
            inputs = [edge]
        else:
            inputs = self._speech_gen_inputs(schedule[0], input_signals)

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._get_step_metadata(full_metadata),
        )

    def _get_step_metadata(self, metadata: CurrentForwardConductorMetadata) -> dict:
        """Per-pass metadata the submodules read from ``request_info.step_metadata``."""
        kw = metadata.kwargs
        return {
            "is_prefill": metadata.is_prefill,
            "num_step": kw["num_step"],
            "guidance_scale": kw["guidance_scale"],
            "t_shift": kw["t_shift"],
            "layer_penalty_factor": kw["layer_penalty_factor"],
            "position_temperature": kw["position_temperature"],
            "class_temperature": kw["class_temperature"],
            "postprocess_output": kw["postprocess_output"],
            "pad_duration": kw["pad_duration"],
            "fade_duration": kw["fade_duration"],
        }

    def _speech_gen_inputs(
        self, walk: str, persist_signals: dict[str, list[TensorPointerInfo]],
    ) -> list[GraphEdge]:
        """External inputs seeding a speech walk.

        The loop-back edges go in empty; the backbone seeds the canvas and the
        step counter at iteration 0.
        """
        inputs: list[GraphEdge] = []
        for name in ("prefix_ids", "prefix_audio_mask", "target_len"):
            edge = GraphEdge(next_node="backbone", name=name)
            edge.tensor_info = persist_signals.get(name, [])
            inputs.append(edge)
        if walk == self.SPEECH_GEN_CLONE_WALK:
            ref_edge = GraphEdge(next_node="backbone", name="ref_audio_tokens")
            ref_edge.tensor_info = persist_signals.get("ref_audio_tokens", [])
            inputs.append(ref_edge)
            rms_edge = GraphEdge(next_node="code2wav", name="ref_rms")
            rms_edge.tensor_info = persist_signals.get("ref_rms", [])
            inputs.append(rms_edge)
        inputs += [
            GraphEdge(next_node="backbone", name="audio_tokens"),
            GraphEdge(next_node="backbone", name="step_index"),
        ]
        return inputs

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Step through the request's fixed walk schedule; done after the speech walk."""
        metadata = partition_metadata
        request_done = False
        inputs: list[GraphEdge] = []

        schedule = metadata.kwargs["walk_schedule"]
        step = metadata.kwargs["walk_step"] + 1
        if step < len(schedule):
            metadata.kwargs["walk_step"] = step
            walk = schedule[step]
            metadata.graph_walk = walk
            metadata.is_prefill = walk == self.ENCODE_REFERENCE_WALK
            inputs = self._speech_gen_inputs(walk, persist_signals)
        else:
            # The speech walk completed — one utterance per request.
            request_done = True

        unpersist_tensors = sum([inp.tensor_info for inp in inputs], start=[])
        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=unpersist_tensors,
            step_metadata=self._get_step_metadata(metadata),
            request_done=request_done,
        )

    def get_autocast_dtype(self):
        """No engine autocast: the fused kernels need one dtype throughout.

        Returning None disables both autocast and the engine's blanket cast, so
        numerics follow the load dtype -- the same call wan22 makes to keep its
        numerics equal to the reference pipeline.
        """
        return None

    def get_output_sample_rate(self, modality: str = "audio") -> int:
        return self.config.sample_rate

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        """Raw PCM16 bytes; the handler wraps them in the requested container."""
        if modality == "audio":
            if output.numel() == 0:
                return b""
            return output.cpu().numpy().astype(np.int16).tobytes()
        raise ValueError(f"Unsupported modality for OmniVoice: {modality!r}")

    # ------------------------------------------------------------------
    # Model ABC: submodule loading
    # ------------------------------------------------------------------

    def get_submodule(
        self,
        node_name: str,
        device: str = "cpu",
        tp_group=None,
        autocast_dtype: torch.dtype | None = None,
        sp_group=None,
    ) -> NodeSubmodule | None:
        # tp_group / sp_group / autocast_dtype exist for interface parity:
        # OmniVoice declares no sharded nodes, and the weights load in the
        # checkpoint's own dtypes.
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(node_name, device)
        self._submodule_cache[node_name] = submodule
        if submodule is not None:
            logger.info("Loaded OmniVoice submodule for node %s", node_name)
        return submodule

    def _get_codec(self, device: str) -> torch.nn.Module:
        if self._codec is None:
            self._codec = load_codec(self.model_path_hf, self.cache_dir).to(device)
            self.config.frame_rate = float(self._codec.config.frame_rate)
            self.config.sample_rate = int(self._codec.config.sample_rate)
        return self._codec

    def _create_submodule(self, node_name: str, device: str) -> NodeSubmodule | None:
        """Build one node from the checkpoint.

        Dummy mode returns None for every node, so the engines can run their
        graph without weights, GPU or network.
        """
        if self.skip_weight_loading:
            return None

        if node_name in ("ref_encoder", "code2wav"):
            codec = self._get_codec(device)
            if node_name == "ref_encoder":
                return OmniVoiceRefEncoderSubmodule(codec, self.config)
            return OmniVoiceCode2WavSubmodule(codec, self.config)

        if node_name == "backbone":
            # The checkpoint's own class owns the weight layout (a Qwen3 body
            # plus the audio embedding table and head); M* borrows the loaded
            # modules and drives them through its own graph. Loading through
            # the reference class is what keeps parity free — see
            # components/backbone.py.
            from omnivoice.models.omnivoice import OmniVoice

            fi = assert_flashinfer_api()
            dtype = getattr(torch, self.config.load_dtype)
            reference = OmniVoice.from_pretrained(
                self.model_path_hf,
                cache_dir=self.cache_dir,
                dtype=dtype,
            ).eval()
            self._refresh_checkpoint_defaults(reference.config)
            if self._codec is None and reference.audio_tokenizer is not None:
                # from_pretrained already built one (~800 MB fp32); a second
                # copy for the codec nodes would be pure waste on a shared GPU.
                self._codec = reference.audio_tokenizer.to(device)
                self.config.frame_rate = float(self._codec.config.frame_rate)
                self.config.sample_rate = int(self._codec.config.sample_rate)
            # Cast explicitly and verify. `dtype=` on from_pretrained does not
            # always reach every module, and a checkpoint whose config says
            # float32 can come back float32 -- which is exactly how the fused
            # RMSNorm ended up with float32 weights under float16 activations.
            # Cast the BACKBONE only. The reference never casts its codec:
            # from_pretrained(dtype=...) touches modules built during loading,
            # and audio_tokenizer is assigned afterwards, so upstream runs a
            # float32 codec against a float16 backbone. Casting the whole model
            # -- which an unqualified .to(dtype) does -- pushes the DAC decoder
            # into float16, where it underflows to digital silence: measured
            # peak 0 on a 3.32s output that should have peaked near full scale.
            reference = reference.to(device)
            reference.llm.to(dtype)
            reference.audio_embeddings.to(dtype)
            reference.audio_heads.to(dtype)
            actual = next(reference.llm.parameters()).dtype
            if actual != dtype:
                raise RuntimeError(
                    f"OmniVoice backbone loaded as {actual}, expected {dtype}. "
                    "The fused flashinfer kernels need one dtype throughout."
                )
            logger.info("OmniVoice loaded: backbone %s, codec %s", actual,
                        next(reference.audio_tokenizer.parameters()).dtype)
            # After .to(device): apply_flashinfer sizes its attention workspace
            # against model.device, and patching on CPU would allocate it there.
            fi.apply_flashinfer(reference, enable_cuda_graph=False)
            backbone = OmniVoiceBackbone(reference, plan_dtype=dtype).eval()
            return OmniVoiceBackboneSubmodule(backbone, self.config)

        return None

    def _refresh_checkpoint_defaults(self, checkpoint_config) -> None:
        """Hard-fail on drift in the values the ported math assumes.

        The unmask step indexes the audio table by codebook and drives the MASK
        class to -inf by id.  If a future checkpoint changes either, the model
        would still run and quietly produce noise, so it is checked rather than
        trusted.
        """
        for attr in ("num_audio_codebook", "audio_vocab_size", "audio_mask_id"):
            expected = getattr(self.config, attr)
            actual = getattr(checkpoint_config, attr, expected)
            if int(actual) != int(expected):
                raise ValueError(
                    f"OmniVoice checkpoint declares {attr}={actual}, but this "
                    f"integration is written against {expected}. The unmask "
                    "step's indexing depends on it; update config.py and the "
                    "parity test rather than overriding this."
                )
