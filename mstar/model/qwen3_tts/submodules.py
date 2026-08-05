# ---------------------------------------------------------------------------
# NodeSubmodule wrappers for Qwen3-TTS
# ---------------------------------------------------------------------------
#
# Two submodules cover the complete text-to-speech streaming pipeline:
#   1. TalkerSubmodule (KV_CACHE engine)
#      - Builds the official text/voice prefill sequence.
#      - Maintains the Talker paged KV cache across 12 Hz decode steps.
#      - Predicts codec group 0 with the Talker and groups 1-15 with the
#        depth-wise CodePredictor.
#      - Supports continuous batching and whole-forward decode CUDA Graphs
#        (the CodePredictor depth loop is captured inside them), plus a
#        piecewise graph covering that loop on eager paths like prefill.
#   2. CodecSubmodule (STATELESS engine)
#      - Receives buffered codec frames from the Talker partition.
#      - Pads variable final tails to fixed CUDA Graph capture shapes.
#      - Runs the official speech-tokenizer decoder and trims overlap before
#        emitting 24 kHz PCM.
#
# Engine-facing lifecycle:
#   prepare_inputs -> preprocess -> forward/forward_batched
#                  -> postprocess -> check_stop (Talker only)
#
# Streaming topology:
#   Talker --[codec_tokens, LeftContextChunkPolicy(300, 25)]--> Codec
# ---------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cuda_graph_config import (
    BasicBatchedCudaGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCaptureShape,
    PiecewiseCudaGraphConfig,
)
from mstar.engine.kv_store import PositionInfo
from mstar.model.qwen3_tts.components.talker import (
    Qwen3TTSCodePredictor,
    Qwen3TTSTalkerModel,
)
from mstar.model.qwen3_tts.config import Qwen3TTSModelConfig
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
)
from mstar.utils.sampling import (
    CudaGraphableSampler,
    MultiSamplerBuffers,
    SeenTokenMask,
)

# ===========================================================================
# 1. TalkerSubmodule - autoregressive 12 Hz codec-frame generation
# ===========================================================================


class TalkerSubmodule(ARNodeSubmodule):
    """Run text/voice prefill and produce one 16-code frame per AR step.

    Codec group 0 is sampled from the main Talker head. The residual
    CodePredictor then walks groups 1-15 within the same frame. The sum of all
    16 codec embeddings becomes the recurrent input for the next Talker step;
    the complete code vector is streamed independently to ``CodecSubmodule``.
    """

    # Compiling the entire engine-facing forward traces request sampling,
    # residual-code control flow, and recurrent routing in addition to the
    # transformer. It introduces graph breaks and showed no steady-state win
    # for this path. The whole decode forward (CodePredictor loop included) is
    # still CUDA-graph captured.
    disable_torch_compile = True
    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
    CHATML_ASSISTANT_PREFIX_TOKEN_IDS = (151644, 77091, 198)
    CHATML_ASSISTANT_SUFFIX_TOKEN_IDS = (151645, 198, 151644, 77091, 198)

    def __init__(
        self,
        talker_model: Qwen3TTSTalkerModel,
        code_predictor: Qwen3TTSCodePredictor,
        config: Qwen3TTSModelConfig,
    ) -> None:
        super().__init__()
        self.model = talker_model
        self.code_predictor = code_predictor
        self.config = config
        self.talker_config = config.talker
        self.cp_config = config.talker.code_predictor
        self.num_codes = config.talker.num_code_groups
        self._suppress_mask: torch.Tensor | None = None
        self._cp_kv_cache: torch.Tensor | None = None
        # Set when the engine hands over its sampler buffers at piecewise
        # capture; the captured depth loop samples straight out of them.
        self._cp_sampler_buffers: MultiSamplerBuffers | None = None
        self._cp_capture_samplers: dict[int, CudaGraphableSampler] = {}

    def _get_suppress_mask(self) -> torch.Tensor:
        """Cache the checkpoint's static invalid-token mask on the worker GPU."""
        if self._suppress_mask is None:
            vocab_size = self.talker_config.vocab_size
            mask = torch.zeros(
                vocab_size, dtype=torch.bool, device=self.get_device()
            )
            mask[max(0, vocab_size - 1024):] = True
            eos = self.talker_config.codec_eos_token_id
            if 0 <= eos < vocab_size:
                mask[eos] = False
            self._suppress_mask = mask
        return self._suppress_mask

    def _get_batch_suppress_mask(
        self, inputs: list[ARNodeInputs]
    ) -> torch.Tensor:
        """Apply input-carried minimum-length EOS suppression to the base mask.

        CUDA Graph replay calls ``preprocess`` with capture-slot dummy request
        ids, so looking up request state here would permanently observe a new
        request at frame zero. ``prepare_inputs`` runs with the real request and
        carries the dynamic flag as a tensor instead.
        """
        mask = self._get_suppress_mask().unsqueeze(0).expand(
            len(inputs), -1
        ).clone()
        eos = self.talker_config.codec_eos_token_id
        if 0 <= eos < mask.shape[1]:
            mask[:, eos] = torch.cat([
                item.tensor_inputs["suppress_eos"] for item in inputs
            ]).to(device=mask.device, dtype=torch.bool)
        return mask

    def _get_cp_kv_cache(self, batch_size: int) -> torch.Tensor:
        """Return the fixed CodePredictor scratch cache for this micro-batch.

        CodePredictor attention is local to one 16-group frame, so this cache
        does not belong to the engine's cross-step paged KV cache. A maximum
        batch allocation is reused and overwritten for every Talker step,
        which also gives the captured decode graph stable addresses.
        """
        expected = (
            self.cp_config.num_hidden_layers,
            self.MAX_BATCH_SIZE,
            2,
            self.num_codes,
            self.cp_config.num_key_value_heads,
            self.cp_config.head_dim,
        )
        if self._cp_kv_cache is None:
            self._cp_kv_cache = torch.empty(
                expected,
                dtype=self.model.model.codec_embedding.weight.dtype,
                device=self.get_device(),
            )
        return self._cp_kv_cache[:, :batch_size]

    def _project_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map tokenizer embeddings into the Talker hidden width."""
        text_hidden = self.model.model.text_embedding(token_ids)
        projection_dtype = self.model.text_projection.linear_fc1.weight.dtype
        return self.model.text_projection(text_hidden.to(projection_dtype))

    def _special_text_embeds(
        self, dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project TTS BOS/EOS/PAD once for official sequence construction."""
        token_ids = torch.tensor(
            [[
                self.config.tts_bos_token_id,
                self.config.tts_eos_token_id,
                self.config.tts_pad_token_id,
            ]],
            dtype=torch.long,
            device=self.get_device(),
        )
        bos, eos, pad = self._project_text(token_ids).to(dtype).chunk(3, dim=1)
        return bos, eos, pad

    def _build_prefill(
        self,
        request_id: str,
        text_ids: torch.Tensor,
        speaker_id: int,
        language_id: int,
    ) -> torch.Tensor:
        """Build the official mixed text/codec prefill embedding sequence.

        The assistant-role prefix and codec conditioning tags enter the
        one-shot prefill. Remaining prompt text is retained in per-request
        state and added one token at a time to later recurrent codec embeds.
        This aligns text progress with the 12 Hz acoustic generation steps.
        """
        text_ids = text_ids.to(device=self.get_device(), dtype=torch.long).view(1, -1)
        expected_prefix = text_ids.new_tensor(
            self.CHATML_ASSISTANT_PREFIX_TOKEN_IDS
        )
        expected_suffix = text_ids.new_tensor(
            self.CHATML_ASSISTANT_SUFFIX_TOKEN_IDS
        )
        prefix_len = expected_prefix.numel()
        suffix_len = expected_suffix.numel()
        if text_ids.shape[1] < prefix_len + 1 + suffix_len:
            raise ValueError(
                "Qwen3-TTS formatted prompt is shorter than its fixed ChatML "
                "prefix, text token, and suffix"
            )
        if not torch.equal(text_ids[0, :prefix_len], expected_prefix):
            raise ValueError(
                "Qwen3-TTS ChatML assistant prefix changed; expected token IDs "
                f"{expected_prefix.tolist()}, got "
                f"{text_ids[0, :prefix_len].tolist()}"
            )
        if not torch.equal(text_ids[0, -suffix_len:], expected_suffix):
            raise ValueError(
                "Qwen3-TTS ChatML assistant suffix changed; expected token IDs "
                f"{expected_suffix.tolist()}, got "
                f"{text_ids[0, -suffix_len:].tolist()}"
            )

        codec = self.talker_config
        codec_prefix = (
            [codec.codec_nothink_id, codec.codec_think_bos_id, codec.codec_think_eos_id]
            if language_id < 0
            else [
                codec.codec_think_id,
                codec.codec_think_bos_id,
                language_id,
                codec.codec_think_eos_id,
            ]
        )
        codec_ids = torch.tensor(
            [[*codec_prefix, speaker_id, codec.codec_pad_id, codec.codec_bos_id]],
            dtype=torch.long,
            device=self.get_device(),
        )
        codec_embeds = self.model.model.codec_embedding(codec_ids)
        bos_embed, eos_embed, pad_embed = self._special_text_embeds(
            codec_embeds.dtype
        )

        # Prefix layout mirrors the official CustomVoice generation helper:
        # assistant role, language/voice codec tags, then first text token.
        role_embed = self._project_text(
            text_ids[:, :prefix_len]
        ).to(codec_embeds.dtype)
        tag_text = torch.cat([
            pad_embed.expand(-1, codec_embeds.shape[1] - 2, -1),
            bos_embed,
        ], dim=1)
        tag_embed = tag_text + codec_embeds[:, :-1]
        first_text = (
            self._project_text(
                text_ids[:, prefix_len:prefix_len + 1]
            ).to(codec_embeds.dtype)
            + codec_embeds[:, -1:]
        )
        prefill = torch.cat([role_embed, tag_embed, first_text], dim=1)

        # The fixed five-token ChatML suffix is replaced by projected TTS EOS.
        # Decode consumes this tensor by ``generation_step`` and uses PAD once
        # the text condition has been exhausted.
        trailing = torch.cat([
            self._project_text(
                text_ids[:, prefix_len + 1:-suffix_len]
            ).to(codec_embeds.dtype),
            eos_embed,
        ], dim=1)
        self.request_state(request_id).add_all(
            trailing_text_hidden=trailing.squeeze(0),
            tts_pad_embed=pad_embed[0, 0],
            generation_step=0,
            generated_frames=0,
        )
        return prefill.squeeze(0)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask: SeenTokenMask | None = None,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs: Any,
    ) -> ARNodeInputs:
        """Convert routed request tensors into one Talker sequence fragment.

        Prefill creates the mixed conditioning sequence. Decode combines the
        previous frame's summed codec embedding with the next projected text
        condition. This method performs no transformer compute; the engine can
        therefore prepare requests before admitting them to a micro-batch.
        """
        del seen_token_mask, pos_info, kwargs
        if graph_walk == "talker_prefill":
            input_embeds = self._build_prefill(
                fwd_info.request_id,
                inputs["text_inputs"][0],
                int(inputs["speaker_id"][0].item()),
                int(inputs["language_id"][0].item()),
            )
            state = self.request_state(fwd_info.request_id)
        elif graph_walk == "talker_decode":
            state = self.request_state(fwd_info.request_id)
            step = int(state["generation_step"])
            trailing = state["trailing_text_hidden"]
            text_condition = (
                trailing[step]
                if step < trailing.shape[0]
                else state["tts_pad_embed"]
            )
            # ``talker_input_embeds`` is the recurrent graph edge emitted by
            # the previous frame, not a token ID that needs another lookup.
            input_embeds = inputs["talker_input_embeds"][0].to(
                device=self.get_device(),
                dtype=text_condition.dtype,
            ).reshape(1, -1)
            input_embeds = input_embeds + text_condition
            state.add("generation_step", step + 1)
        else:
            raise ValueError(f"Unknown Qwen3-TTS Talker walk: {graph_walk!r}")

        suppress_eos = (
            int(state.get("generated_frames", 0))
            < self.config.generation.min_new_tokens
        )
        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=input_embeds.shape[0],
            tensor_inputs={
                "suppress_eos": torch.tensor(
                    [suppress_eos], dtype=torch.bool, device=self.get_device()
                )
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        """Pack a continuous batch and plan request-specific paged attention.

        Prefill sequences may have different lengths, so embeddings are
        concatenated into one packed tensor. ``last_token_indices`` maps each
        request back to the hidden state used for codec-group-0 prediction.
        FlashInfer receives separate sequence lengths and KV page tables.
        """
        del graph_walk
        cache_manager = engine_inputs.cache_manager
        assert cache_manager is not None
        cache_manager.set_active_label("main")
        seq_lens = [item.input_seq_len for item in inputs]
        cache_manager.plan_attention(seq_lens=seq_lens, is_causal=True, label="main")
        cache_manager.plan_rope(seq_lens=seq_lens, pos_ids=None, label="main")
        return {
            "input_embeds": torch.cat([
                item.input_embeds for item in inputs if item.input_embeds is not None
            ], dim=0),
            "last_token_indices": (
                torch.tensor(seq_lens, device=self.get_device()).cumsum(0) - 1
            ),
            "suppress_mask": self._get_batch_suppress_mask(inputs),
        }

    def _run_frame(
        self,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Produce one complete codec frame for every request in the batch.

        The main Talker advances the engine-managed KV cache and predicts group
        0; the CodePredictor's unrolled depth loop fills groups 1..N-1 through
        the ``code_predictor`` aux sampler. Both read their params from
        engine-owned static buffers, so this single path serves eager execution
        and CUDA-graph capture alike (mirrors the Qwen3-Omni Talker).
        """
        hidden = self.model(input_embeds, engine_inputs.cache_manager)
        last_hidden = hidden.index_select(0, last_token_indices)
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(suppress_mask, float("-inf"))
        sampler = engine_inputs.sampler
        assert sampler is not None
        request_ids = engine_inputs.request_ids
        # The repetition penalty applies to group 0 only; the depth loop below
        # stays penalty-free (its aux config declares no vocab_size).
        layer0_codes = sampler.sample(request_ids, logits, apply_penalty=True)

        piecewise = self._run_depth_loop_piecewise(
            engine_inputs, last_hidden, layer0_codes
        )
        if piecewise is not None:
            all_codes, codec_embed_sum = piecewise
        else:
            all_codes, codec_embed_sum = self._depth_loop(
                last_hidden, layer0_codes,
                lambda cp_logits: sampler.sample_aux(
                    "code_predictor", request_ids, cp_logits
                ),
            )

        return {
            "talker_input_embeds": codec_embed_sum,
            "codec_tokens": all_codes,
            "new_token": layer0_codes,
        }

    def _depth_loop(
        self,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
        sample_codes: "Callable[[torch.Tensor], torch.Tensor]",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk residual groups 1..N-1, returning (all_codes, codec_embed_sum).

        ``sample_codes`` maps one group's logits to codes. Every op here is
        graph-safe, so this body is captured verbatim — inside the whole-walk
        decode graph, or on its own by the piecewise runner.
        """
        batch_size = layer0_codes.shape[0]
        device = layer0_codes.device
        all_codes = torch.empty(
            batch_size, self.num_codes, dtype=torch.long, device=device
        )
        all_codes[:, 0] = layer0_codes
        codec_embed = self.model.model.codec_embedding(layer0_codes)
        codec_embed_sum = codec_embed.clone()

        cp_cache = self._get_cp_kv_cache(batch_size)
        pos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        # Position 0 conditions the depth decoder on the Talker hidden state.
        # Each following position consumes the previous group's embedding.
        self.code_predictor.forward_depth_unrolled(
            last_hidden.unsqueeze(1), pos, cp_cache, cache_pos=0
        )
        for group_idx in range(1, self.num_codes):
            pos.fill_(group_idx)
            cp_hidden = self.code_predictor.forward_depth_unrolled(
                codec_embed.unsqueeze(1), pos, cp_cache, cache_pos=group_idx
            ).squeeze(1)
            cp_logits = torch.matmul(
                cp_hidden,
                self.code_predictor.lm_head_weight[group_idx - 1].t(),
            )
            codes = sample_codes(cp_logits)
            all_codes[:, group_idx] = codes
            codec_embed = self.code_predictor.model.codec_embedding[
                group_idx - 1
            ](codes)
            codec_embed_sum.add_(codec_embed)
        return all_codes, codec_embed_sum

    def _code_predictor_piecewise_capture(
        self,
        static_inputs: dict[str, torch.Tensor],
        static_cm=None,
    ) -> dict[str, torch.Tensor]:
        """Capture entry point: the depth loop *including* its sampling.

        The aux sampler for this bucket reads the engine's static param buffers,
        so nothing about sampling has to be hoisted out as a static input the
        way per-step uniforms and scalars used to be.
        """
        del static_cm
        layer0_codes = static_inputs["layer0_codes"]
        aux = self._cp_capture_samplers[layer0_codes.shape[0]]
        all_codes, codec_embed_sum = self._depth_loop(
            static_inputs["last_hidden"], layer0_codes,
            lambda cp_logits: aux.sample([], cp_logits),
        )
        return {"all_codes": all_codes, "codec_embed_sum": codec_embed_sum}

    def _run_depth_loop_piecewise(
        self,
        engine_inputs: ModelInputsFromEngine,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay the captured depth loop, or None to run it inline.

        Used on paths the whole-walk graph doesn't cover — above all
        ``talker_prefill``, which is eager because it is variable-length. Left
        eager, this loop costs ~4x its captured time.
        """
        runner = engine_inputs.piecewise_runners.get("code_predictor_loop")
        batch_size = layer0_codes.shape[0]
        if runner is None or not runner.can_run(batch_size=batch_size):
            return None
        # Called for its side effect only: it stages this step's per-request
        # sampling params into the shared gather buffers that every per-bucket
        # capture sampler in ``_cp_capture_samplers`` aliases, so the returned
        # sampler is intentionally discarded. ``padded_bs=batch_size`` fills
        # only the real rows; the bucket graph samples its full captured width,
        # but PiecewiseOutput slices the padded tail off (those rows always hold
        # a well-formed, stale config).
        self._cp_sampler_buffers.aux["code_predictor"].gather_for_request_ids(
            request_ids=engine_inputs.request_ids,
            padded_bs=batch_size,
            gather_seen_tokens=False,
        )
        output = runner.run(
            static_inputs={
                "last_hidden": last_hidden,
                "layer0_codes": layer0_codes,
            },
            real_bs=batch_size,
        )
        return output["all_codes"], output["codec_embed_sum"]

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_mask: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, kwargs
        output = self._run_frame(
            engine_inputs, input_embeds, last_token_indices, suppress_mask
        )
        return {name: [tensor] for name, tensor in output.items()}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        output = self._run_frame(
            engine_inputs, input_embeds, last_token_indices, suppress_mask
        )
        return {
            request_id: {
                "talker_input_embeds": [output["talker_input_embeds"][i:i + 1]],
                "codec_tokens": [output["codec_tokens"][i]],
                "new_token": [output["new_token"][i]],
            }
            for i, request_id in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """Expose the stop token and advance metadata without a GPU sync.

        Whole-graph replay and eager execution normally retain ``new_token``.
        ``codec_tokens`` is the routed output, though, so use its first codebook
        as a fallback.  This keeps EOS detection correct even if an execution
        path filters the sampler-only output before slow-path ``check_stop``.
        """
        del request_info, kwargs
        if "new_token" in outputs:
            outputs["layer0_codes"] = outputs.pop("new_token")
        elif "layer0_codes" not in outputs and "codec_tokens" in outputs:
            codec_tokens = outputs["codec_tokens"][0]
            outputs["layer0_codes"] = [codec_tokens.reshape(-1)[0]]
        if "layer0_codes" in outputs:
            state = self.request_state(request_id)
            state.add(
                "generated_frames", int(state.get("generated_frames", 0)) + 1
            )

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        """Stop the graph loop on codec EOS or the request frame limit.

        This callback runs off the GPU execution thread, so reading the sampled
        token with ``item`` is allowed here rather than in ``postprocess``.
        """
        if "layer0_codes" not in outputs:
            return set()
        token = int(outputs["layer0_codes"][0].item())
        generated = int(
            self.request_state(request_id).get("generated_frames", 0)
        )
        max_tokens = request_info.step_metadata.get(
            "talker_max_tokens", request_info.max_tokens
        )
        sampling_config = getattr(request_info, "sampling_config", {}).get(
            "Talker"
        )
        ignore_eos = bool(getattr(sampling_config, "ignore_eos", False))
        reached_eos = (
            not ignore_eos and token == self.talker_config.codec_eos_token_id
        )
        if reached_eos or generated >= max_tokens:
            return {"talker_decode_loop"}
        return set()

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        """Admit compatible prefill/decode requests to continuous batching.

        Both the Talker and the CodePredictor sample per-request off their own
        sampler buffers, so requests with differing sampling settings batch
        together freely.
        """
        return (
            batch.graph_walk in {"talker_prefill", "talker_decode"}
            and bool(model_inputs)
            and len(model_inputs) <= self.MAX_BATCH_SIZE
        )

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[BasicBatchedCudaGraphConfig]:
        """Capture fixed one-token Talker decode batches, including sampling.

        Dynamic EOS suppression is carried by ``ARNodeInputs`` and packed into
        a graph input, so replay never consults capture-slot dummy request state.
        Prefill remains eager because it is variable-length and runs once per
        request.
        """
        del tp_world_size
        dtype = self.model.model.codec_embedding.weight.dtype
        return [BasicBatchedCudaGraphConfig(
            capture_graph_walk="talker_decode",
            labels=["main"],
            requires_cfg=False,
            single_request_inputs=ARNodeInputs(
                input_embeds=torch.zeros(
                    1,
                    self.talker_config.hidden_size,
                    dtype=dtype,
                    device=device,
                ),
                input_seq_len=1,
                # Padding slots clone this template, so every replay input must
                # declare the same dynamic key as a real prepared request.
                tensor_inputs={
                    "suppress_eos": torch.ones(
                        1, dtype=torch.bool, device=device
                    )
                },
            ),
            capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            compile=True,
        )]

    def get_piecewise_cuda_graph_configs(
        self,
        device: torch.device,
        autocast_dtype: torch.dtype,
        tp_world_size: int = 1,
        sampler_buffers: MultiSamplerBuffers | None = None,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        """Capture the CodePredictor depth loop, sampling included.

        The whole-walk decode graph already covers this loop; this runner serves
        the paths it can't — chiefly ``talker_prefill``, which stays eager
        because it is variable-length and runs once per request. It has no
        engine KV cache: its frame-local cache is a static tensor owned here.
        """
        del tp_world_size
        if sampler_buffers is None or "code_predictor" not in sampler_buffers.aux:
            # Without the aux buffers the loop cannot sample in-graph; skip
            # capture and let the eager path run it.
            return {}
        self._cp_sampler_buffers = sampler_buffers
        hidden_size = self.talker_config.hidden_size
        capture_dtype = autocast_dtype or self.model.model.codec_embedding.weight.dtype
        aux_buffers = sampler_buffers.aux["code_predictor"]
        # One sampler per capture bucket, over that bucket's slice of the shared
        # buffers, built up-front so each capture reads a stable address.
        self._cp_capture_samplers = {
            bs: CudaGraphableSampler(**aux_buffers.slice_for_bs(bs))
            for bs in self.DECODE_CAPTURE_BATCH_SIZES
        }

        def make_static_inputs(
            shape: PiecewiseCaptureShape,
        ) -> dict[str, torch.Tensor]:
            return {
                "last_hidden": torch.zeros(
                    shape.bs, hidden_size, dtype=capture_dtype, device=device,
                ),
                "layer0_codes": torch.zeros(
                    shape.bs, dtype=torch.long, device=device
                ),
            }

        return {
            "code_predictor_loop": PiecewiseBatchedConfig(
                capture_fn=self._code_predictor_piecewise_capture,
                make_static_inputs=make_static_inputs,
                seq_len=1,
                uses_kv_cache=False,
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
                compile=False,
            )
        }

    def can_use_cuda_graphs(
        self, batch: NodeBatch, model_inputs: list[NodeInputs]
    ) -> bool:
        """Replay the whole decode graph; sampling params are read from buffers,
        so no request's settings can disqualify it."""
        if batch.graph_walk != "talker_decode" or not self.can_batch(
            batch, model_inputs
        ):
            return False
        return super().can_use_cuda_graphs(batch, model_inputs)

    def get_needed_cache_labels(
        self,
        graph_walk: str,
        per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str]:
        del graph_walk, per_request_info
        return ["main"]


# ===========================================================================
# 2. CodecSubmodule - fixed-shape streaming waveform decode
# ===========================================================================


class CodecSubmodule(ARNodeSubmodule):
    """Convert streamed 16-code frames into overlap-trimmed PCM chunks.

    The node runs on a stateless engine, but ``ARNodeInputs`` is reused as the
    typed container for fixed-length codec tensors. Per-request state stores
    only how many non-padding frames arrived and whether a prior chunk was
    emitted; the neural decoder itself has no cross-call state.
    """

    disable_torch_compile = True
    # The official 114M-parameter decoder materializes large fixed-shape
    # activations while CUDA graphs are captured.  Capturing bs=16 exhausts an
    # H100 once Talker weights and the CodePredictor graphs are resident, so
    # keep the safe ceiling at 8 until the decoder is ported to M*'s
    # lighter-weight codec components.
    MAX_BATCH_SIZE = 8
    CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def __init__(self, decoder: torch.nn.Module, config: Qwen3TTSModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.full_seq_len = (
            config.codec.chunk_frames + config.codec.left_context_frames
        )
        self.total_upsample = 1
        for factor in (
            *config.codec.upsample_rates,
            *config.codec.upsampling_ratios,
        ):
            self.total_upsample *= factor

    def get_stateless_flavor(self) -> str:
        return "audio_codec"

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask: SeenTokenMask | None = None,
        pos_info: dict[str, PositionInfo] = {},
        **kwargs: Any,
    ) -> ARNodeInputs:
        """Remove EOS frames and pad one stream chunk to its capture shape.

        Input arrives as ``[frames, code_groups]``. The official decoder wants
        ``[quantizers, frames]``; every request is padded to ``chunk + context``
        so differently sized final tails can reuse the same CUDA Graph.
        """
        del graph_walk, seen_token_mask, pos_info, kwargs
        codes = inputs["codec_tokens"][0].to(
            device=self.get_device(), dtype=torch.long
        )
        if codes.ndim == 1:
            codes = codes.view(-1, self.config.num_code_groups)
        if codes.ndim != 2:
            raise ValueError(
                f"Expected codec tokens with shape (frames, groups), got {codes.shape}"
            )
        # EOS belongs to Talker loop control and is not a valid codec codebook
        # index for waveform reconstruction.
        codes = codes[
            codes[:, 0] != self.config.talker.codec_eos_token_id,
            :self.config.codec.num_quantizers,
        ]
        original_frames = codes.shape[0]
        if original_frames > self.full_seq_len:
            raise ValueError(
                f"Codec chunk has {original_frames} frames, maximum is "
                f"{self.full_seq_len}"
            )
        if original_frames < self.full_seq_len:
            codes = torch.nn.functional.pad(
                codes,
                (0, 0, 0, self.full_seq_len - original_frames),
            )
        self.request_state(fwd_info.request_id).add(
            "latest_codec_frames", original_frames
        )
        return ARNodeInputs(
            tensor_inputs={"codec_tokens": codes.t().contiguous()},
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        """Stack equal fixed-shape codec chunks into one continuous batch."""
        del graph_walk, engine_inputs
        return {
            "codec_tokens": torch.stack([
                item.tensor_inputs["codec_tokens"] for item in inputs
            ])
        }

    def _decode(self, codec_tokens: torch.Tensor) -> torch.Tensor:
        """Run the official decoder and convert normalized audio to PCM16."""
        wav = self.decoder(codec_tokens)
        return (wav.clamp(-1, 1) * 32767).to(torch.int16).squeeze(1)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        **kwargs: Any,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs, kwargs
        return {"audio_chunk": [self._decode(codec_tokens)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        codec_tokens: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, NameToTensorList]:
        del graph_walk, kwargs
        wavs = self._decode(codec_tokens)
        return {
            request_id: {"audio_chunk": [wavs[i]]}
            for i, request_id in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """Remove padded tail and duplicated left-context PCM before emission."""
        del request_info, kwargs
        if "audio_chunk" not in outputs:
            return
        state = self.request_state(request_id)
        frames = int(state.get("latest_codec_frames", 0))
        emitted = bool(state.get("codec_chunk_emitted", False))
        # The first chunk has no overlap. Later stream chunks include old codec
        # frames at the front, whose decoded samples must not be emitted twice.
        left_context = self.config.codec.left_context_frames if emitted else 0
        start = left_context * self.total_upsample
        end = frames * self.total_upsample
        outputs["audio_chunk"][0] = outputs["audio_chunk"][0][start:end]
        state.add("codec_chunk_emitted", True)

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        """Batch codec requests only when their decoder input shapes match."""
        del batch
        return 0 < len(model_inputs) <= self.MAX_BATCH_SIZE and len({
            item.tensor_inputs["codec_tokens"].shape for item in model_inputs
        }) == 1

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[BasicBatchedCudaGraphConfig]:
        """Capture fixed-length Codec batches for all scheduler buckets."""
        del tp_world_size
        return [BasicBatchedCudaGraphConfig(
            capture_graph_walk="codec_chunk",
            single_request_inputs=ARNodeInputs(
                input_seq_len=self.full_seq_len,
                tensor_inputs={
                    "codec_tokens": torch.zeros(
                        self.config.codec.num_quantizers,
                        self.full_seq_len,
                        dtype=torch.long,
                        device=device,
                    )
                },
            ),
            capture_batch_sizes=self.CAPTURE_BATCH_SIZES,
            compile=False,
        )]

    def can_use_cuda_graphs(
        self, batch: NodeBatch, model_inputs: list[NodeInputs]
    ) -> bool:
        return (
            batch.graph_walk == "codec_chunk"
            and self.can_batch(batch, model_inputs)
            and all(
                item.tensor_inputs["codec_tokens"].shape
                == (
                    self.config.codec.num_quantizers,
                    self.full_seq_len,
                )
                for item in model_inputs
            )
            and super().can_use_cuda_graphs(batch, model_inputs)
        )
