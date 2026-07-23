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
#      - Supports continuous batching, whole-forward decode CUDA Graphs, and
#        a piecewise CUDA Graph for the CodePredictor inner loop.
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
from mstar.utils.sampling import SeenTokenMask

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

    MAX_BATCH_SIZE = 32
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

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
        self, request_ids: list[str]
    ) -> torch.Tensor:
        """Apply request-local minimum-length EOS suppression to the base mask."""
        mask = self._get_suppress_mask().unsqueeze(0).expand(
            len(request_ids), -1
        ).clone()
        eos = self.talker_config.codec_eos_token_id
        if 0 <= eos < mask.shape[1]:
            mask[:, eos] = torch.tensor(
                [
                    int(self.request_state(request_id).get("generated_frames", 0))
                    < self.config.generation.min_new_tokens
                    for request_id in request_ids
                ],
                dtype=torch.bool,
                device=mask.device,
            )
        return mask

    def _get_cp_kv_cache(self, batch_size: int) -> torch.Tensor:
        """Return the fixed CodePredictor scratch cache for this micro-batch.

        CodePredictor attention is local to one 16-group frame, so this cache
        does not belong to the engine's cross-step paged KV cache. A maximum
        batch allocation is reused and overwritten for every Talker step,
        which also gives the piecewise CUDA Graph stable addresses.
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
        if text_ids.shape[1] < 9:
            raise ValueError(
                "Qwen3-TTS formatted prompt is shorter than its fixed chat suffix"
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
        role_embed = self._project_text(text_ids[:, :3]).to(codec_embeds.dtype)
        tag_text = torch.cat([
            pad_embed.expand(-1, codec_embeds.shape[1] - 2, -1),
            bos_embed,
        ], dim=1)
        tag_embed = tag_text + codec_embeds[:, :-1]
        first_text = (
            self._project_text(text_ids[:, 3:4]).to(codec_embeds.dtype)
            + codec_embeds[:, -1:]
        )
        prefill = torch.cat([role_embed, tag_embed, first_text], dim=1)

        # The fixed five-token ChatML suffix is replaced by projected TTS EOS.
        # Decode consumes this tensor by ``generation_step`` and uses PAD once
        # the text condition has been exhausted.
        trailing = torch.cat([
            self._project_text(text_ids[:, 4:-5]).to(codec_embeds.dtype),
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

        return ARNodeInputs(
            input_embeds=input_embeds,
            input_seq_len=input_embeds.shape[0],
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
            "suppress_mask": self._get_batch_suppress_mask(
                engine_inputs.request_ids
            ),
        }

    @staticmethod
    def _sample_code_predictor(
        sampler,
        logits: torch.Tensor,
        sampling: dict[str, Any],
    ) -> torch.Tensor:
        """Eager fallback sampler for one residual codec group."""
        temperature = float(sampling.get("temperature", 0.9))
        if not sampling.get("do_sample", True) or temperature <= 0:
            return sampler._broadcast_tokens(logits.argmax(dim=-1))
        return sampler.sample_with_config(
            logits=logits,
            temperature=temperature,
            top_k=int(sampling.get("top_k", 50)),
            top_p=float(sampling.get("top_p", 1.0)),
        )

    @staticmethod
    def _sample_from_uniform(
        logits: torch.Tensor,
        uniform: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        do_sample: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-safe batched top-k/top-p sampling from supplied uniforms."""
        vocab_size = logits.shape[-1]
        greedy = logits.argmax(dim=-1)
        safe_temperature = torch.where(
            do_sample,
            temperature.clamp_min(1e-5),
            torch.ones_like(temperature),
        ).unsqueeze(1)
        sorted_logits, sorted_indices = torch.sort(
            logits / safe_temperature, dim=-1, descending=True
        )
        positions = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
        effective_top_k = torch.where(
            top_k > 0,
            top_k.clamp_max(vocab_size),
            torch.full_like(top_k, vocab_size),
        )
        sorted_logits = sorted_logits.masked_fill(
            positions >= effective_top_k.unsqueeze(1), float("-inf")
        )
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative - probs > top_p.clamp_min(1e-6).unsqueeze(1)
        probs = probs.masked_fill(remove, 0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cdf = probs.cumsum(dim=-1)
        sampled_rank = (cdf < uniform.unsqueeze(1)).sum(dim=-1)
        sampled_rank = sampled_rank.clamp_max(vocab_size - 1)
        sampled = sorted_indices.gather(1, sampled_rank.unsqueeze(1)).squeeze(1)
        return torch.where(do_sample, sampled, greedy).to(torch.long)

    def _run_code_predictor_tensor_loop(
        self,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
        uniforms: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        do_sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict residual groups 1-15 with graph-safe tensor operations.

        Random uniforms and sampling parameters are inputs, rather than being
        created in this function, so CUDA Graph replay changes samples without
        recapturing the depth loop.
        """
        batch_size = layer0_codes.shape[0]
        all_codes = torch.empty(
            batch_size,
            self.num_codes,
            dtype=torch.long,
            device=layer0_codes.device,
        )
        all_codes[:, 0] = layer0_codes
        codec_embed = self.model.model.codec_embedding(layer0_codes)
        codec_embed_sum = codec_embed.clone()
        cp_cache = self._get_cp_kv_cache(batch_size)
        pos = torch.zeros(
            batch_size, 1, dtype=torch.long, device=layer0_codes.device
        )
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
            codes = self._sample_from_uniform(
                cp_logits,
                uniforms[:, group_idx - 1],
                temperature,
                top_k,
                top_p,
                do_sample,
            )
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
        """Piecewise capture entry point for the hot residual-code depth loop."""
        del static_cm
        all_codes, codec_embed_sum = self._run_code_predictor_tensor_loop(
            last_hidden=static_inputs["last_hidden"],
            layer0_codes=static_inputs["layer0_codes"],
            uniforms=static_inputs["uniforms"],
            temperature=static_inputs["temperature"],
            top_k=static_inputs["top_k"],
            top_p=static_inputs["top_p"],
            do_sample=static_inputs["do_sample"],
        )
        return {
            "all_codes": all_codes,
            "codec_embed_sum": codec_embed_sum,
        }

    def _piecewise_sampling_inputs(
        self,
        engine_inputs: ModelInputsFromEngine,
        sampling: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Generate per-request RNG and dynamic sampling tensors before replay.

        A persistent generator preserves each request's random stream across
        changing micro-batches. Distributed workers broadcast the uniforms so
        replicated CodePredictors choose identical residual codes.
        """
        batch_size = len(engine_inputs.request_ids)
        device = self.get_device()
        uniforms = []
        for request_id in engine_inputs.request_ids:
            state = self.request_state(request_id)
            generator = state.get("code_predictor_generator")
            if generator is None:
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    int(engine_inputs.per_request_info[request_id].random_seed)
                )
                state.add("code_predictor_generator", generator)
            uniforms.append(torch.rand(
                self.num_codes - 1,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ))
        uniform_tensor = torch.stack(uniforms)
        sampler = engine_inputs.sampler
        assert sampler is not None
        uniform_tensor = sampler._broadcast_tokens(uniform_tensor)
        return {
            "uniforms": uniform_tensor,
            "temperature": torch.full(
                (batch_size,),
                float(sampling.get("temperature", 0.9)),
                dtype=torch.float32,
                device=device,
            ),
            "top_k": torch.full(
                (batch_size,),
                int(sampling.get("top_k", 50)),
                dtype=torch.long,
                device=device,
            ),
            "top_p": torch.full(
                (batch_size,),
                float(sampling.get("top_p", 1.0)),
                dtype=torch.float32,
                device=device,
            ),
            "do_sample": torch.full(
                (batch_size,),
                bool(sampling.get("do_sample", True)),
                dtype=torch.bool,
                device=device,
            ),
        }

    def _run_code_predictor_piecewise(
        self,
        engine_inputs: ModelInputsFromEngine,
        last_hidden: torch.Tensor,
        layer0_codes: torch.Tensor,
        sampling: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay the captured depth loop, or signal the caller to use eager."""
        runner = engine_inputs.piecewise_runners.get("code_predictor_loop")
        batch_size = layer0_codes.shape[0]
        if runner is None or not runner.can_run(batch_size=batch_size):
            return None
        static_inputs = self._piecewise_sampling_inputs(engine_inputs, sampling)
        static_inputs.update({
            "last_hidden": last_hidden,
            "layer0_codes": layer0_codes,
        })
        output = runner.run(static_inputs=static_inputs, real_bs=batch_size)
        sampler = engine_inputs.sampler
        assert sampler is not None
        all_codes = sampler._broadcast_tokens(output["all_codes"])
        codec_embed_sum = sampler._broadcast_tokens(output["codec_embed_sum"])
        return all_codes, codec_embed_sum

    def _run_frame(
        self,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        last_token_indices: torch.Tensor,
        suppress_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Produce one complete codec frame for every request in the batch.

        The main Talker advances the engine-managed KV cache and predicts group
        0. The CodePredictor fills residual groups through the piecewise graph
        when available, with the identical eager loop as a fallback.
        """
        hidden = self.model(input_embeds, engine_inputs.cache_manager)
        last_hidden = hidden.index_select(0, last_token_indices)
        logits = self.model.codec_head(last_hidden)
        logits = logits.masked_fill(suppress_mask, float("-inf"))
        sampler = engine_inputs.sampler
        assert sampler is not None
        layer0_codes = sampler.sample(
            engine_inputs.request_ids, logits, apply_penalty=True
        )

        sampling = engine_inputs.first_request_info.step_metadata.get(
            "subtalker_sampling", {}
        )
        piecewise_result = self._run_code_predictor_piecewise(
            engine_inputs=engine_inputs,
            last_hidden=last_hidden,
            layer0_codes=layer0_codes,
            sampling=sampling,
        )
        if piecewise_result is not None:
            all_codes, codec_embed_sum = piecewise_result
            return {
                "talker_input_embeds": codec_embed_sum,
                "codec_tokens": all_codes,
                "new_token": layer0_codes,
            }

        # Eager fallback mirrors `_run_code_predictor_tensor_loop`. It uses the
        # engine sampler directly because this path is outside CUDA capture.
        batch_size = layer0_codes.shape[0]
        all_codes = torch.empty(
            batch_size, self.num_codes, dtype=torch.long, device=logits.device
        )
        all_codes[:, 0] = layer0_codes
        codec_embed = self.model.model.codec_embedding(layer0_codes)
        codec_embed_sum = codec_embed.clone()

        cp_cache = self._get_cp_kv_cache(batch_size)
        pos = torch.zeros(batch_size, 1, dtype=torch.long, device=logits.device)
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
            codes = self._sample_code_predictor(sampler, cp_logits, sampling)
            all_codes[:, group_idx] = codes
            codec_embed = self.code_predictor.model.codec_embedding[
                group_idx - 1
            ](codes)
            codec_embed_sum.add_(codec_embed)

        return {
            "talker_input_embeds": codec_embed_sum,
            "codec_tokens": all_codes,
            "new_token": layer0_codes,
        }

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
        """Rename the stop token output and advance metadata without GPU sync."""
        del request_info, kwargs
        if "new_token" in outputs:
            outputs["layer0_codes"] = outputs.pop("new_token")
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
        if token == self.talker_config.codec_eos_token_id or generated >= max_tokens:
            return {"talker_decode_loop"}
        return set()

    def can_batch(self, batch: NodeBatch, model_inputs: list[NodeInputs]) -> bool:
        """Admit compatible prefill/decode requests to continuous batching.

        Residual sampling settings must match because the eager path reads one
        batch-level configuration. Main Talker sampling remains request-local
        through the engine sampler.
        """
        if (
            batch.graph_walk not in {"talker_prefill", "talker_decode"}
            or not model_inputs
            or len(model_inputs) > self.MAX_BATCH_SIZE
        ):
            return False
        sampling = {
            repr(info.step_metadata.get("subtalker_sampling", {}))
            for info in batch.per_request_info.values()
        }
        return len(sampling) <= 1

    def max_batch_size(self, graph_walk: str) -> int:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[BasicBatchedCudaGraphConfig]:
        """Capture fixed one-token Talker decode batches, including sampling."""
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
            ),
            capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            compile=True,
        )]

    def get_piecewise_cuda_graph_configs(
        self,
        device: torch.device,
        autocast_dtype: torch.dtype,
        tp_world_size: int = 1,
    ) -> dict[str, PiecewiseCudaGraphConfig]:
        """Capture CodePredictor's 15-step depth loop as an inner CUDA Graph.

        This runner is also useful when the whole Talker graph is ineligible,
        for example after a request overrides ``subtalker_*`` sampling values.
        It has no engine KV cache: its small frame-local cache is a static
        tensor owned by this submodule.
        """
        del tp_world_size
        hidden_size = self.talker_config.hidden_size
        capture_dtype = autocast_dtype or self.model.model.codec_embedding.weight.dtype

        def make_static_inputs(
            shape: PiecewiseCaptureShape,
        ) -> dict[str, torch.Tensor]:
            return {
                "last_hidden": torch.zeros(
                    shape.bs,
                    hidden_size,
                    dtype=capture_dtype,
                    device=device,
                ),
                "layer0_codes": torch.zeros(
                    shape.bs, dtype=torch.long, device=device
                ),
                "uniforms": torch.zeros(
                    shape.bs,
                    self.num_codes - 1,
                    dtype=torch.float32,
                    device=device,
                ),
                "temperature": torch.full(
                    (shape.bs,),
                    self.config.generation.subtalker_temperature,
                    dtype=torch.float32,
                    device=device,
                ),
                "top_k": torch.full(
                    (shape.bs,),
                    self.config.generation.subtalker_top_k,
                    dtype=torch.long,
                    device=device,
                ),
                "top_p": torch.full(
                    (shape.bs,),
                    self.config.generation.subtalker_top_p,
                    dtype=torch.float32,
                    device=device,
                ),
                "do_sample": torch.full(
                    (shape.bs,),
                    self.config.generation.subtalker_dosample,
                    dtype=torch.bool,
                    device=device,
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
        """Use the whole decode graph only for its captured sampling constants."""
        if batch.graph_walk != "talker_decode" or not self.can_batch(
            batch, model_inputs
        ):
            return False
        expected = {
            "do_sample": self.config.generation.subtalker_dosample,
            "temperature": self.config.generation.subtalker_temperature,
            "top_k": self.config.generation.subtalker_top_k,
            "top_p": self.config.generation.subtalker_top_p,
        }
        for info in batch.per_request_info.values():
            if info.step_metadata.get("subtalker_sampling", expected) != expected:
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
    MAX_BATCH_SIZE = 16
    CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

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
