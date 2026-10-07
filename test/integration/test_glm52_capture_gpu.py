"""GLM-5.2 whole-forward CUDA graphs on one GPU: a reduced model at the real
MLA latent dims (512/64, FlashInfer's MLA kernel) captured through the
engine's ``CudaGraphRunner`` and replayed the way the engine replays it,
against the same model served eager."""
import dataclasses

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.cuda_graph_runner import CudaGraphRunner, autocast_scope
from mstar.engine.resources import (
    StepContext,
    StepRunner,
    apply_yaml_overrides,
    resolve_spec_dependencies,
)
from mstar.engine.resources.attn.flashinfer_mla import flashinfer_mla_supports
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.glm52.config import KV_RESOURCE, Glm52ModelConfig
from mstar.model.submodule_base import ModelInputsFromEngine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU",
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NODE = "LLM"
KV_YAML = {"resources": {KV_RESOURCE: {"max_num_pages": 64, "page_size": 16}}}
PROMPTS = {"a": list(range(3, 12)), "b": list(range(40, 45))}
# W8A8 numerics follow the padded shape, so the fp8 arm's prompts fill the
# prefill bucket and eager and replay run the same shapes
BUCKET_PROMPTS = {"a": list(range(3, 19)), "b": list(range(40, 56))}
MAX_TOKENS = 12
# the scale block fused_experts_fp8 compiles for
FP8_BLOCK = (128, 128)


def _latent_cfg(fp8: bool, **overrides) -> Glm52ModelConfig:
    """The reduced model at the real latent shape: absorbed MLA with
    kv_lora_rank 512 / rope 64, which FlashInfer's MLA kernel serves. The
    fp8 variant's experts are one scale block wide, so the fused kernel
    takes them."""
    if fp8:
        base = dataclasses.replace(
            Glm52ModelConfig.reduced_fp8(block=FP8_BLOCK), moe_intermediate_size=FP8_BLOCK[0])
    else:
        base = Glm52ModelConfig.reduced()
    cfg = dataclasses.replace(
        base, mla_absorb=True, kv_lora_rank=512, qk_rope_head_dim=64,
        prefill_token_buckets=[16], prefill_capture_batch_sizes=[1], **overrides,
    )
    assert flashinfer_mla_supports(cfg.kv_lora_rank, cfg.qk_rope_head_dim)
    return cfg


def _load(tmp_path, monkeypatch, fp8: bool, cfg_overrides=None, **model_kwargs):
    """fp8: through a written checkpoint and the model's own loader; bf16:
    the randomized reference itself."""
    import test_glm52_serve_e2e as serve_e2e
    from safetensors.torch import save_file
    from test_glm52_serve_e2e import _build_reference, _hf_checkpoint

    from mstar.model.glm52.glm52_model import Glm52Model
    from mstar.model.glm52.quantization import process_weights_after_loading
    from mstar.model.glm52.submodules import Glm52LLMSubmodule

    torch.manual_seed(0)
    model = Glm52Model(
        model_path_hf="", config_variant="reduced_fp8" if fp8 else "reduced",
        checkpoint_path=str(tmp_path), tokenizer_mode="byte", **model_kwargs,
    )
    cfg = dataclasses.replace(_latent_cfg(fp8, **(cfg_overrides or {})),
                              moe_quant_kernel=model.config.moe_quant_kernel)
    model.config = cfg
    ref = _build_reference(dataclasses.replace(cfg, quantization_config=None))
    if fp8:
        monkeypatch.setattr(serve_e2e, "BLOCK", FP8_BLOCK)
        save_file(_hf_checkpoint(ref, cfg), str(tmp_path / "model.safetensors"))
        del ref
        submodule = model.get_submodule(NODE, device="cuda", autocast_dtype=DTYPE)
    else:
        for name, param in ref.named_parameters():
            if name.endswith("e_score_correction_bias"):
                param.data = param.data.float()
        process_weights_after_loading(ref, DEVICE)
        submodule = Glm52LLMSubmodule(language_model=ref, config=cfg)
    submodule.requires_grad_(False)
    return model, submodule


class _Node:
    """The node's resources as the engine builds them, the step cycle around
    the submodule, and (``capture=True``) a warmed-up ``CudaGraphRunner``
    whose replay path is the engine's: lease, pad, declare over the padded
    ids, plan on the lease, stage and replay, map the slot's outputs back.
    """

    def __init__(self, model, submodule, capture: bool, prompts: dict[str, list[int]]):
        self.model, self.sub, self.prompts = model, submodule, prompts
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, KV_YAML)
        by_key = resolve_spec_dependencies(specs)
        self.groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        transfer = TransferEngineInfo(
            my_entity_id="glm52_capture", my_session_id="glm52_capture",
            transfer_engine=LocalTransferEngine("localhost"),
        )
        self.resources = {
            spec.resource_key: build_resource(
                spec,
                EngineResourceInfo(
                    device=DEVICE, joint_comm_group=self.groups,
                    transfer_engine_info=transfer, kv_dtype=DTYPE,
                    dependencies={key: by_key[key] for key in spec.depends_on()},
                ),
            )
            for spec in specs
        }
        self.runner = StepRunner(self.resources, node_resources={NODE: list(self.resources)})
        submodule.bind_node_resources(self.resources)
        self.overrides = model.get_request_resource_configs(
            {}, model_kwargs={"temperature": 0.0, "ignore_eos": True},
        )
        self.cg = None
        if capture:
            self.cg = CudaGraphRunner(
                submodule_name=NODE, submodule=submodule, resources=self.resources,
                step_runner=self.runner, device=DEVICE, autocast_dtype=DTYPE,
                joint_comm_group=self.groups, num_slots=1,
            )
            with torch.no_grad():
                self.cg.warmup_and_capture()

    def info(self, rid: str) -> CurrentForwardPassInfo:
        return CurrentForwardPassInfo(
            request_id=rid, graph_walk="prefill", fwd_index=0, random_seed=0,
            max_tokens=MAX_TOKENS, resource_configs=self.overrides,
        )

    def step(self, walk: str, batch: dict[str, tuple[CurrentForwardPassInfo, torch.Tensor]]):
        rids = list(batch)
        infos = {rid: info for rid, (info, _) in batch.items()}
        inputs = [
            self.sub.prepare_inputs(walk, info, {"text_inputs": [text]})
            for info, text in batch.values()
        ]
        num_tokens = sum(inp.input_seq_len for inp in inputs)
        lease = None
        if self.cg is not None:
            lease = self.cg.lease_slot(walk, len(rids), slot=0, num_tokens=num_tokens)
            if lease is None:  # fine only for a walk the submodule does not capture
                walks = {c.capture_graph_walk for c in self.sub.get_cuda_graph_configs(DEVICE)}
                assert walk not in walks, f"no captured bucket for {walk} bs={len(rids)}"
        step_ids, meta = rids, infos
        if lease is not None:
            inputs = self.cg.pad_inputs(lease, inputs)
            step_ids = self.cg.step_ids(lease, rids)
            meta = self.cg.step_metadata(lease, rids, infos)
        ctx = StepContext(
            request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False, slot_lease=lease,
        )
        ctx.set_padded_rids(step_ids)
        step = self.sub.declare_step(walk, list(step_ids), inputs, slot_lease=lease)
        step.set_ctx(ctx)
        try:
            with torch.no_grad(), autocast_scope(DTYPE, device_type=DEVICE.type):
                assert self.runner.admit(step).ok
                self.runner.plan(step)
                engine_inputs = ModelInputsFromEngine(
                    request_ids=list(step_ids), per_request_info=meta,
                    resources=self.resources, step=step, captured=lease is not None,
                )
                pre = self.sub.preprocess(walk, engine_inputs, inputs)
                if lease is None:
                    raw = self.sub.forward_batched(walk, engine_inputs, **pre)
                    out_ids = step_ids
                else:
                    raw = self.cg.run_forward(lease, pre)
                    out_ids = self.cg.slot_for(lease).dummy_rids
                self.runner.commit(step)
        finally:
            if lease is not None:
                self.cg.release(lease, len(rids))
        # a dict, or (with #258's graph runner) a BatchedModelOutput
        out = getattr(raw, "per_rid_outputs", raw)
        return {rid: {k: list(v) for k, v in out[out_ids[i]].items()} for i, rid in enumerate(rids)}

    def generate(self) -> dict[str, list[int]]:
        """Each prompt prefilled on its own, then decoded as one batch."""
        infos = {rid: self.info(rid) for rid in self.prompts}
        for rid in self.prompts:
            self.runner.ingest_request(rid, self.overrides)
        streams: dict[str, list[int]] = {rid: [] for rid in self.prompts}
        texts = {}
        for rid, prompt in self.prompts.items():
            ids = torch.tensor(prompt, dtype=torch.long, device=DEVICE)
            out = self.step("prefill", {rid: (infos[rid], ids)})[rid]
            self.sub.postprocess(rid, infos[rid], out)
            streams[rid].append(int(out["new_token"][0].item()))
            texts[rid] = out["text_inputs"][0].clone()
        live = list(self.prompts)
        for it in range(MAX_TOKENS + 2):
            if not live:
                break
            outs = self.step("decode", {rid: (infos[rid], texts[rid]) for rid in live})
            for rid in list(live):
                out = outs[rid]
                self.sub.postprocess(rid, infos[rid], out)
                streams[rid].append(int(out["new_token"][0].item()))
                texts[rid] = out["text_inputs"][0].clone()
                infos[rid].dynamic_loop_iter_counts["decode_loop"] = it
                if self.sub.check_stop(rid, infos[rid], out):
                    live.remove(rid)
        return streams

    def close(self):
        for rid in self.prompts:
            self.runner.remove_request(rid)
            self.sub.cleanup_request(rid)


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_capture_at_real_latent_dims_matches_eager(tmp_path, monkeypatch, fp8):
    monkeypatch.setattr(CudaGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2, 4])
    model, submodule = _load(tmp_path, monkeypatch, fp8)
    assert model.config.kv_lora_rank == 512 and model.config.mla_absorb
    if fp8:
        moe = submodule.language_model.model.layers[1].mlp
        assert moe.fp8_experts and moe._use_fused

    prompts = BUCKET_PROMPTS if fp8 else PROMPTS
    eager = _Node(model, submodule, capture=False, prompts=prompts)
    eager_streams = eager.generate()
    eager.close()

    node = _Node(model, submodule, capture=True, prompts=prompts)
    assert len(submodule.get_cuda_graph_configs(DEVICE)) == 2
    assert node.cg.any_graphs, "no CUDA graph captured"
    assert node.cg.dropped_buckets == []
    captured_streams = node.generate()
    node.close()

    for rid, stream in eager_streams.items():
        assert len(stream) == MAX_TOKENS, (rid, stream)
        assert len(set(stream)) > 2, (rid, stream)
        assert captured_streams[rid] == stream, (rid, captured_streams[rid], stream)


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_capture_with_fused_mla_prep_matches_eager_and_unfused(tmp_path, monkeypatch, fp8):
    """mla_fused_prep on: the captured streams equal the eager ones and the unfused path's
    captured ones (q_lora_rank 256, a width the fused norms take)."""
    monkeypatch.setattr(CudaGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2, 4])
    prompts = BUCKET_PROMPTS if fp8 else PROMPTS
    streams = {}
    for fused_prep in (False, True):
        ckpt = tmp_path / f"fused{int(fused_prep)}"
        ckpt.mkdir()
        model, submodule = _load(ckpt, monkeypatch, fp8,
                                 cfg_overrides=dict(q_lora_rank=256, mla_fused_prep=fused_prep))
        attn = submodule.language_model.model.layers[0].self_attn
        assert attn.mla_fused_prep == fused_prep
        if fused_prep:
            eager = _Node(model, submodule, capture=False, prompts=prompts)
            streams["eager"] = eager.generate()
            eager.close()
        node = _Node(model, submodule, capture=True, prompts=prompts)
        assert node.cg.any_graphs and node.cg.dropped_buckets == []
        streams[fused_prep] = node.generate()
        node.close()

    for rid, stream in streams[True].items():
        assert len(stream) == MAX_TOKENS and len(set(stream)) > 2, (rid, stream)
        assert stream == streams["eager"][rid], (rid, stream, streams["eager"][rid])
        assert stream == streams[False][rid], (rid, stream, streams[False][rid])


# compiled, the dense arm's last token flips at a 0.0117-logit near tie: Inductor's arithmetic,
# since the eager forward captured matches token for token
@pytest.mark.parametrize("topk,compiled", [(8, True), (16, False)],
                         ids=["bucket-selects", "bucket-dense"])
def test_paged_dsa_capture_matches_eager(tmp_path, monkeypatch, topk, compiled):
    """DSA at a small index_topk, so decode selects: the captured steps emit the
    eager tokens. Prefill replays its 16-row bucket, padded, with every row selecting on its
    own when the bucket is wider than index_topk and dense attention when not; decode is sparse
    for every row over the serving window, through the slot's re-planned sparse wrapper."""
    monkeypatch.setattr(CudaGraphRunner, "CAPTURE_BATCH_SIZES", [1, 2, 4])
    if not compiled:
        monkeypatch.setenv("MSTAR_GLM52_GRAPH_COMPILE", "0")
    overrides = dict(dsa_long_context=True, index_topk=topk, index_n_heads=16,
                     index_head_dim=64, max_seq_len=64)  # the indexer ropes its first 64 dims
    model, submodule = _load(tmp_path, monkeypatch, False, cfg_overrides=overrides)
    walks = [c.capture_graph_walk for c in submodule.get_cuda_graph_configs(DEVICE)]
    assert walks == ["decode", "prefill"]
    eager = _Node(model, submodule, capture=False, prompts=PROMPTS)
    eager_streams = eager.generate()
    eager.close()
    node = _Node(model, submodule, capture=True, prompts=PROMPTS)
    assert node.cg.any_graphs and node.cg.dropped_buckets == []
    captured_streams = node.generate()
    node.close()
    for rid, stream in eager_streams.items():
        assert len(stream) == MAX_TOKENS and len(set(stream)) > 2, (rid, stream)
        assert captured_streams[rid] == stream, (rid, captured_streams[rid], stream)
