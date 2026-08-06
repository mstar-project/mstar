"""Submodule for the sampling-parity dummy model.

The node runs a tiny MLP stack to turn a fixed per-request input into main +
aux logits, then samples them two ways depending on the graph walk:

  - ``"graphed"``  → CUDA-graph path: ``engine_inputs.sampler.sample`` (main,
    with penalty) and ``.sample_aux`` (aux loop). This exercises the real
    ``MultiSamplerBuffers`` / ``CudaGraphableSampler`` / ``gather_for_request_ids``
    machinery, incl. the per-request seed/offset.
  - ``"eager"``    → reference path: ``sample_tokens`` called directly with the
    same (seed, offset) the graphed path uses, plus a ``cuda.synchronize()``.

Both walks see identical logits every iteration (fixed input, shared weights),
so a token mismatch between the two sequences is unambiguously a sampler bug.

Seed/offset contract (see the model docstring):
  - offset advances per-request per gather, so at loop iteration ``i`` the base
    offset is ``i``; the aux inner loop adds the group index ``g`` to mirror the
    in-graph ``offset_buf += 1`` per aux sample.
  - main seed and aux seed are read straight off the per-request
    ``MultiSamplingConfig`` (the conductor already derived the aux seed via
    ``set_seed``), so the eager path never re-derives crc32 itself.
"""

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.base import NodeBatch
from mstar.engine.cuda_graph_config import BasicBatchedCudaGraphConfig
from mstar.engine.kv_store import PositionInfo
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
)
from mstar.utils.sampling import SeenTokenMask, sample_tokens

NODE_NAME = "LLM"
AUX_LABEL = "aux"
GRAPHED_WALK = "graphed"
EAGER_WALK = "eager"
# Loop names must match the ``Loop`` sections the model declares.
GRAPHED_LOOP = "graphed_loop"
EAGER_LOOP = "eager_loop"
# request_state key: eager-walk main seen-token mask (mirrors the graph penalty).
_EAGER_SEEN_KEY = "eager_main_seen"


class SamplingTestSubmodule(ARNodeSubmodule):
    disable_torch_compile = True

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size
        # A couple of MLP layers per stage; random init is fine — we only need
        # non-degenerate logits so different offsets pick different tokens.
        self.pre = torch.nn.Sequential(
            torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, h),
        )
        self.aux_mlp = torch.nn.Sequential(
            torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, h),
        )
        self.main_head = torch.nn.Linear(h, config.main_vocab_size, bias=False)
        self.aux_head = torch.nn.Linear(h, config.aux_vocab_size, bias=False)
        self.num_aux_groups = config.num_aux_groups
        self.decode_capture_batch_sizes = config.decode_capture_batch_sizes

    # ---- input plumbing ------------------------------------------------
    # The loop feeds the node's own "x" output back as its next input, so every
    # iteration sees the same fixed embed; only the RNG offset advances.

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        seen_token_mask: SeenTokenMask,
        pos_info: dict[str, PositionInfo] = {},
    ) -> ARNodeInputs:
        del graph_walk, fwd_info, seen_token_mask, pos_info
        # Per-request embed is [seq, hidden] with seq=1 (decode).
        embed = inputs["x"][0]
        return ARNodeInputs(input_embeds=embed, input_seq_len=embed.shape[0])

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor]:
        del graph_walk, engine_inputs
        # Cat (pack) per-request [1, hidden] embeds into [bs, hidden] — matches
        # the AR decode contract (one token/request), so logits stay 2-D.
        return {"input_embeds": torch.cat(
            [inp.input_embeds for inp in inputs], dim=0
        )}

    def can_batch(self, batch: NodeBatch, model_inputs: list[ARNodeInputs]) -> bool:
        # Batch for real (so the per-request gather is exercised) — but sample
        # each request one row at a time inside forward_batched, see there.
        return True

    # ---- compute + sampling -------------------------------------------

    def _logits(self, input_embeds: torch.Tensor):
        # Per-request matmuls so each row's logits are batch-size-invariant: a
        # batched GEMM's per-row result depends on bs (kernel/reduction order),
        # which would make the graphed walk (bs=X) diverge from the eager walk
        # (bs=Y) for pure numerics reasons and fake a sampler mismatch. Looping
        # keeps logits identical across walks, isolating the sampler. The loop
        # length is the (fixed-per-bucket) bs, so it stays CUDA-graph safe.
        mains, auxs = [], []
        for i in range(input_embeds.shape[0]):
            hidden = self.pre(input_embeds[i:i + 1])
            mains.append(self.main_head(hidden))
            auxs.append(self.aux_head(self.aux_mlp(hidden)))
        return torch.cat(mains, dim=0), torch.cat(auxs, dim=0)

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_embeds: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        del kwargs
        request_ids = engine_inputs.request_ids
        main_logits, aux_logits = self._logits(input_embeds)
        # This walk's own loop, so the eager offset reads the right iter counter.
        loop = GRAPHED_LOOP if graph_walk == GRAPHED_WALK else EAGER_LOOP

        if graph_walk == GRAPHED_WALK:
            main_tok = engine_inputs.sampler.sample(
                request_ids, main_logits, apply_penalty=True
            )
            aux_toks = [
                engine_inputs.sampler.sample_aux(AUX_LABEL, request_ids, aux_logits)
                for _ in range(self.num_aux_groups)
            ]
        else:
            main_tok = self._eager_sample(
                engine_inputs, main_logits, label="main", loop=loop, group=0,
            )
            aux_toks = [
                self._eager_sample(
                    engine_inputs, aux_logits, label=AUX_LABEL, loop=loop, group=g,
                )
                for g in range(self.num_aux_groups)
            ]

        # [bs, 1 + num_aux_groups] token matrix, one row per request.
        tokens = torch.stack([main_tok, *aux_toks], dim=1)
        name = GRAPHED_LOOP if graph_walk == GRAPHED_WALK else EAGER_LOOP
        return {
            # Feed the same [1, hidden] embed back (loop stays fixed-input);
            # emit this iteration's token vector to the client.
            rid: {"x": [input_embeds[i:i + 1]], f"{name}_tokens": [tokens[i].clone()]}
            for i, rid in enumerate(request_ids)
        }

    @torch.compiler.disable
    def _eager_sample(
        self,
        engine_inputs: ModelInputsFromEngine,
        logits: torch.Tensor,
        label: str,
        loop: str,
        group: int,
    ) -> torch.Tensor:
        """Reference sample: ``sample_tokens`` at the same (seed, offset) the
        graphed path used, then a hard sync to rule out ordering effects."""
        rids = engine_inputs.request_ids
        info = engine_inputs.per_request_info
        multi_cfgs = [info[rid].sampling_config[NODE_NAME] for rid in rids]
        cfgs = [
            mc.main if label == "main" else mc.aux[label] for mc in multi_cfgs
        ]
        # Base offset == this request's loop iteration; +group mirrors the
        # in-graph ``offset_buf += 1`` the aux loop does per sample.
        base = [info[rid].dynamic_loop_iter_counts.get(loop, 0) for rid in rids]
        dev = logits.device

        # Replicate the graphed MAIN sampler's in-graph repetition penalty: a
        # per-request seen-token mask over main tokens sampled earlier in THIS
        # walk (request_state persists across loop iters). Same timing as the
        # graph: penalize tokens 0..i-1, sample, then record token i. Aux has no
        # penalty, so it takes the plain path.
        is_main = label == "main"
        seen_mask = None
        rep_pen: float | torch.Tensor = 1.0
        masks: list[torch.Tensor] | None = None
        if is_main and any(c.repetition_penalty != 1.0 for c in cfgs):
            # Key the mask per WALK (loop): the two walks share one request's
            # request_state, so a single key would let walk 2 inherit walk 1's
            # seen tokens and its penalty would diverge. Per-walk keys start each
            # walk's mask fresh, matching how the real graphed walk's engine mask
            # is reset per request.
            key = f"{_EAGER_SEEN_KEY}_{loop}"
            masks = []
            for rid in rids:
                st = self.request_state(rid)
                m = st.get(key)
                if m is None:
                    m = torch.zeros(logits.shape[1], dtype=torch.bool, device=dev)
                    st.add(key, m)
                masks.append(m)
            seen_mask = torch.stack(masks, dim=0)
            rep_pen = torch.tensor([c.repetition_penalty for c in cfgs], device=dev)

        rand_offset=torch.tensor(
            [b + group for b in base], device=dev, dtype=torch.long
        )
        tokens: torch.Tensor = sample_tokens(
            logits=logits,
            temperature=torch.tensor([c.temperature for c in cfgs], device=dev),
            top_k=torch.tensor([c.top_k for c in cfgs], device=dev, dtype=torch.int32),
            top_p=torch.tensor([c.top_p for c in cfgs], device=dev),
            repetition_penalty=rep_pen,
            seen_token_mask=seen_mask,
            seed=torch.tensor([c.seed for c in cfgs], device=dev, dtype=torch.long),
            rand_offset=rand_offset,
        )
        if masks is not None:
            for i in range(len(rids)):
                masks[i][tokens[i]] = True
        torch.cuda.synchronize()
        # sample_tokens returns FlashInfer's int32; the graphed path
        # (sample_cuda_graphable_gpu) casts to int64. Match it so the emitted
        # token vectors serialize identically.
        return tokens.to(torch.int64)

    # ---- CUDA graph + loop control ------------------------------------

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1
    ) -> list[BasicBatchedCudaGraphConfig]:
        del tp_world_size
        # Only the "graphed" walk is captured; "eager" has no config, so it runs
        # eager and hits the ``sample_tokens`` reference path.
        dummy = ARNodeInputs(
            input_embeds=torch.zeros(
                1, self.config.hidden_size, dtype=torch.float32, device=device
            ),
            input_seq_len=1,
        )
        return [
            BasicBatchedCudaGraphConfig(
                capture_graph_walk=GRAPHED_WALK,
                labels=["main"],
                single_request_inputs=dummy,
                capture_batch_sizes=self.decode_capture_batch_sizes,
                compile=False,
            )
        ]

    def can_use_cuda_graphs(
        self, batch: NodeBatch, model_inputs: list[ARNodeInputs]
    ) -> bool:
        return batch.graph_walk == GRAPHED_WALK

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        del outputs
        n = request_info.step_metadata["iters"]
        loop = GRAPHED_LOOP if request_info.graph_walk == GRAPHED_WALK else EAGER_LOOP
        it = request_info.dynamic_loop_iter_counts.get(loop, 0)
        return {loop} if it + 1 >= n else set()
