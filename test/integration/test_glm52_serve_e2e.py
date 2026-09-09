"""In-process 1-GPU serve smoke for GLM-5.2 (fp8-resident experts).

Fabricates an on-disk fp8-block checkpoint (routed experts as genuine e4m3
weight + weight_scale_inv pairs, plus two fp8 dense projections for load-path
coverage), then drives the real serve path on CUDA: ``Glm52Model.get_submodule``
(meta init -> to_empty -> load_weights -> process_weights_after_loading), the
node's resources built from the model's own declaration the way
``Engine.load_model`` builds them (``get_node_resources`` -> the deployment's
``resources: kv:`` overrides -> ``build_resource``), the submodule bound to
them, and the engine's per-step cycle (declare -> admit -> plan -> preprocess
-> forward -> commit -> postprocess -> check_stop) through prefill and the
decode loop. This is the cheap gate before the 750 GB TP8 load: it exercises
every fp8 code path the CPU tests can't put on a device, over the real paged
FlashInfer cache.

``mstar.model.base`` pulls the sampler's Triton kernels in, so ``Glm52Model``
is imported inside the tests to keep collection clean on CUDA-less machines
(``test_fused_moe_fp8.py`` precedent).
"""
import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.cuda_graph_runner import autocast_scope
from mstar.engine.resources import (
    StepContext,
    StepRunner,
    apply_yaml_overrides,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.glm52._testing import fake_quantize_fp8_block
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
from mstar.model.glm52.config import KV_RESOURCE, Glm52ModelConfig
from mstar.model.glm52.submodules import Glm52LLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="serve e2e needs a GPU (real FlashInfer paged cache)",
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NODE = "LLM"
BLOCK = (16, 16)  # reduced_fp8 default block size
# The deployment's ``resources: kv:`` block (what configs/glm52*.yaml carry),
# applied to the declared specs exactly as the worker applies a YAML.
KV_YAML = {"resources": {KV_RESOURCE: {"max_num_pages": 8, "page_size": 128}}}


def _fill_layer(layer, cfg):
    a = layer.self_attn
    for lin in (a.q_a_proj, a.q_b_proj, a.kv_a_proj_with_mqa, a.kv_b_proj, a.o_proj):
        lin.weight.data.normal_(0, 0.03)
    for norm in (a.q_a_layernorm, a.kv_a_layernorm):
        norm.weight.data.normal_(1.0, 0.02)
    if a.indexer is not None:  # FULL indexer layers (layer 0 at reduced dims)
        for lin in (a.indexer.wq_b, a.indexer.wk, a.indexer.weights_proj):
            lin.weight.data.normal_(0, 0.03)
        a.indexer.k_norm.weight.data.normal_(1.0, 0.02)
        a.indexer.k_norm.bias.data.normal_(0, 0.02)
    layer.input_layernorm.weight.data.normal_(1.0, 0.02)
    layer.post_attention_layernorm.weight.data.normal_(1.0, 0.02)
    mlp = layer.mlp
    if isinstance(mlp, Glm52SparseMoeBlock):
        mlp.gate.weight.data.normal_(0, 1)
        mlp.gate.e_score_correction_bias.data = torch.randn(
            cfg.n_routed_experts, device=DEVICE, dtype=torch.float32)
        mlp.experts.gate_up_proj.data.normal_(0, 0.05)
        mlp.experts.down_proj.data.normal_(0, 0.05)
        mlp.shared_expert.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.shared_expert.down_proj.weight.data.normal_(0, 0.05)
    else:
        mlp.gate_up_proj.weight.data.normal_(0, 0.05)
        mlp.down_proj.weight.data.normal_(0, 0.05)


def _build_reference(cfg):
    # Plain reduced (quantization_config=None) so experts are stacked bf16
    # params that are easy to slice back into per-expert checkpoint keys.
    model = Glm52ForCausalLM(cfg).to(device=DEVICE, dtype=DTYPE)
    model.model.embed_tokens.weight.data.normal_(0, 0.05)
    model.model.norm.weight.data.normal_(1.0, 0.02)
    model.lm_head.weight.data.normal_(0, 0.02)
    for layer in model.model.layers:
        _fill_layer(layer, cfg)
    return model.eval()


def _fp8_pair(sd, base, weight):
    w8, scale, _ = fake_quantize_fp8_block(weight.detach().float().cpu(), BLOCK)
    sd[base + ".weight"] = w8
    sd[base + ".weight_scale_inv"] = scale


def _hf_checkpoint(model, cfg):
    """Emit HF-style keys: routed experts as fp8 pairs (mandatory for the
    fp8-resident containers), o_proj/kv_b_proj as fp8 pairs (dense
    dequant-on-load coverage), everything else bf16 passthrough."""
    inter = cfg.intermediate_size
    moe_inter = cfg.moe_intermediate_size
    shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts
    m = model.model
    sd = {"model.embed_tokens.weight": m.embed_tokens.weight}
    for i, layer in enumerate(m.layers):
        p = f"model.layers.{i}."
        a = layer.self_attn
        sd[p + "self_attn.q_a_proj.weight"] = a.q_a_proj.weight
        sd[p + "self_attn.q_a_layernorm.weight"] = a.q_a_layernorm.weight
        sd[p + "self_attn.q_b_proj.weight"] = a.q_b_proj.weight
        sd[p + "self_attn.kv_a_proj_with_mqa.weight"] = a.kv_a_proj_with_mqa.weight
        sd[p + "self_attn.kv_a_layernorm.weight"] = a.kv_a_layernorm.weight
        _fp8_pair(sd, p + "self_attn.kv_b_proj", a.kv_b_proj.weight)
        _fp8_pair(sd, p + "self_attn.o_proj", a.o_proj.weight)
        if a.indexer is not None:
            # Checkpoint layout: wq_b/wk fp8 pairs, weights_proj/k_norm bf16.
            _fp8_pair(sd, p + "self_attn.indexer.wq_b", a.indexer.wq_b.weight)
            _fp8_pair(sd, p + "self_attn.indexer.wk", a.indexer.wk.weight)
            sd[p + "self_attn.indexer.weights_proj.weight"] = a.indexer.weights_proj.weight
            sd[p + "self_attn.indexer.k_norm.weight"] = a.indexer.k_norm.weight
            sd[p + "self_attn.indexer.k_norm.bias"] = a.indexer.k_norm.bias
        sd[p + "input_layernorm.weight"] = layer.input_layernorm.weight
        sd[p + "post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight
        mlp = layer.mlp
        if isinstance(mlp, Glm52SparseMoeBlock):
            sd[p + "mlp.gate.weight"] = mlp.gate.weight
            sd[p + "mlp.gate.e_score_correction_bias"] = mlp.gate.e_score_correction_bias
            gup, dwn = mlp.experts.gate_up_proj, mlp.experts.down_proj
            for e in range(cfg.n_routed_experts):
                _fp8_pair(sd, p + f"mlp.experts.{e}.gate_proj", gup[e, :moe_inter, :])
                _fp8_pair(sd, p + f"mlp.experts.{e}.up_proj", gup[e, moe_inter:, :])
                _fp8_pair(sd, p + f"mlp.experts.{e}.down_proj", dwn[e])
            sh = mlp.shared_expert
            sd[p + "mlp.shared_experts.gate_proj.weight"] = sh.gate_up_proj.weight[:shared_inter]
            sd[p + "mlp.shared_experts.up_proj.weight"] = sh.gate_up_proj.weight[shared_inter:]
            sd[p + "mlp.shared_experts.down_proj.weight"] = sh.down_proj.weight
        else:
            sd[p + "mlp.gate_proj.weight"] = mlp.gate_up_proj.weight[:inter]
            sd[p + "mlp.up_proj.weight"] = mlp.gate_up_proj.weight[inter:]
            sd[p + "mlp.down_proj.weight"] = mlp.down_proj.weight
    sd["model.norm.weight"] = m.norm.weight
    sd["lm_head.weight"] = model.lm_head.weight
    return {k: v.detach().cpu().clone().contiguous() for k, v in sd.items()}


def _write_checkpoint(tmp_path, seed=0):
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    cfg = Glm52ModelConfig.reduced()
    ref = _build_reference(cfg)
    save_file(_hf_checkpoint(ref, cfg), str(tmp_path / "model.safetensors"))
    return cfg


def _load_model(tmp_path):
    # inside: mstar.model.base pulls the sampler's Triton kernels
    from mstar.model.glm52.glm52_model import Glm52Model

    return Glm52Model(
        model_path_hf="", config_variant="reduced_fp8",
        checkpoint_path=str(tmp_path), tokenizer_mode="byte",
    )


class _Serve:
    """One request's serve loop over the node's real resources: what the
    engine does around the submodule, minus the worker around the engine.

    Resources come from the model's declaration through the YAML overrides
    and ``build_resource`` (``Engine.load_model``), the request opens on the
    per-resource configs the model resolves for it
    (``get_request_resource_configs``), and every step runs the runner's
    declare -> admit -> plan -> forward -> commit cycle under the engine's
    no_grad + autocast scope.
    """

    def __init__(self, model, submodule: Glm52LLMSubmodule, rid: str = "r0",
                 kv_yaml: dict = KV_YAML, **model_kwargs):
        self.submodule = submodule
        self.rid = rid
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, kv_yaml)
        by_key = resolve_spec_dependencies(specs)
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        transfer = TransferEngineInfo(
            my_entity_id="glm52_serve_e2e", my_session_id="glm52_serve_e2e",
            transfer_engine=LocalTransferEngine("localhost"),
        )
        self.resources = {
            spec.resource_key: build_resource(
                spec,
                EngineResourceInfo(
                    device=DEVICE, joint_comm_group=groups,
                    transfer_engine_info=transfer, kv_dtype=DTYPE,
                    dependencies={key: by_key[key] for key in spec.depends_on()},
                ),
            )
            for spec in specs
        }
        self.runner = StepRunner(self.resources, node_resources={NODE: list(self.resources)})
        submodule.requires_grad_(False)
        submodule.bind_node_resources(self.resources)
        # ignore_eos: glm52's reduced EOS ids (250-252) sit inside the byte
        # vocab, so greedy decode on random weights can hit one and stop
        # early — which would break the exact-length assertions below.
        self.overrides = model.get_request_resource_configs(
            {}, model_kwargs={"temperature": 0.0, "ignore_eos": True, **model_kwargs},
        )
        self.runner.ingest_request(rid, self.overrides)

    def info(self, max_tokens: int) -> CurrentForwardPassInfo:
        return CurrentForwardPassInfo(
            request_id=self.rid, graph_walk="prefill", fwd_index=0, random_seed=0,
            max_tokens=max_tokens, resource_configs=self.overrides,
        )

    def step(self, walk: str, info: CurrentForwardPassInfo, text: torch.Tensor,
             batched: bool = True) -> dict:
        """One forward; ``batched`` is the engine's path for this node
        (``can_batch`` is True), the unbatched ``forward`` returns logits."""
        rid = self.rid
        inputs = [self.submodule.prepare_inputs(walk, info, {"text_inputs": [text]})]
        step = self.submodule.declare_step(walk, [rid], inputs)
        ctx = StepContext(request_ids=(rid,), graph_walk=walk, slot=0, capture=False)
        with torch.no_grad(), autocast_scope(DTYPE, device_type=DEVICE.type):
            if step is not None:
                step.set_ctx(ctx)
                outcome = self.runner.admit(step)
                assert outcome.ok, outcome
                self.runner.plan(step)
            engine_inputs = ModelInputsFromEngine(
                request_ids=[rid], per_request_info={rid: info},
                resources=self.resources, step=step,
            )
            kw = self.submodule.preprocess(walk, engine_inputs, inputs)
            if batched:
                out = self.submodule.forward_batched(walk, engine_inputs, **kw)[rid]
            else:
                out = self.submodule.forward(walk, engine_inputs, **kw)
            if step is not None:
                self.runner.commit(step)
        return out

    def close(self):
        self.runner.remove_request(self.rid)
        self.submodule.cleanup_request(self.rid)
        for resource in self.resources.values():
            resource.cleanup()


def _run_generation(model, submodule, cfg, prompt_ids, max_tokens):
    """The real serve loop: prefill -> decode -> check_stop, all through
    ``forward_batched`` and the sampler resource (greedy)."""
    serve = _Serve(model, submodule)
    info = serve.info(max_tokens)
    generated: list[int] = []
    stopped = False
    try:
        out = serve.step("prefill", info, prompt_ids)
        new_token = out["new_token"][0]
        assert new_token.shape == (1,)
        submodule.postprocess(serve.rid, info, out)  # rebinds text_inputs
        generated.append(int(new_token.item()))
        next_token = out["text_inputs"][0]

        for step in range(max_tokens + 4):  # +slack; check_stop must break first
            out = serve.step("decode", info, next_token)
            new_token = out["new_token"][0]
            submodule.postprocess(serve.rid, info, out)
            assert out["text_inputs"] is out["new_token"]
            info.dynamic_loop_iter_counts["decode_loop"] = step
            stop = submodule.check_stop(serve.rid, info, out)
            generated.append(int(new_token.item()))
            next_token = out["text_inputs"][0]
            if stop:
                stopped = True
                break
    finally:
        serve.close()
    return generated, stopped


def _prefill_logits(model, submodule, prompt_ids):
    """The unbatched forward's logits for the prompt's last row."""
    serve = _Serve(model, submodule)
    try:
        return serve.step("prefill", serve.info(8), prompt_ids, batched=False)["logits"][0]
    finally:
        serve.close()


def test_serve_path_prefill_decode_loop_fp8(tmp_path):
    _write_checkpoint(tmp_path, seed=0)

    model = _load_model(tmp_path)
    cfg = model.config
    assert cfg.vocab_size == 256
    assert cfg.quantization_config is not None

    prompt_tensors = model.process_prompt("hello glm", ["text"], ["text"])
    prompt_ids = prompt_tensors["text_inputs"][0].to(DEVICE)
    assert prompt_ids.tolist() == list("hello glm".encode("utf-8"))

    submodule = model.get_submodule(NODE, device="cuda", autocast_dtype=DTYPE)
    assert isinstance(submodule, Glm52LLMSubmodule)

    # The fp8-resident containers survived autocast + device move.
    moe = submodule.language_model.model.layers[1].mlp
    assert moe.experts.gate_up_proj_fp8.dtype == torch.uint8
    assert moe.experts.gate_up_proj_fp8.is_cuda
    assert moe.experts.gate_up_proj_scale_inv.dtype == torch.float32
    assert moe.gate.e_score_correction_bias.dtype == torch.float32
    # Reference dispatch registers no CUDA graphs (eager-only; the fused fp8
    # kernel is the opt-in moe_quant_kernel=triton).
    assert submodule.get_cuda_graph_configs(DEVICE) == []
    assert submodule.get_piecewise_cuda_graph_configs(DEVICE, DTYPE) == {}

    # The model's declaration is what the engine builds from: naive MLA at
    # reduced dims (mla_absorb=False) -> the paged K/V cache + FlashInfer.
    specs = model.get_node_resources()
    assert [spec.resource_key for spec in specs] == ["kv", "attn", "sampler"]

    logits = _prefill_logits(model, submodule, prompt_ids)
    assert logits.shape == (1, cfg.vocab_size)
    assert torch.isfinite(logits).all()

    MAX_TOKENS = 6
    generated, stopped = _run_generation(model, submodule, cfg, prompt_ids, MAX_TOKENS)

    assert stopped, "decode loop did not terminate via check_stop"
    assert len(generated) == MAX_TOKENS, generated  # max_tokens counts the prefill-emitted token (vLLM semantics)
    assert all(0 <= t < cfg.vocab_size for t in generated), generated

    out_bytes = model.postprocess(torch.tensor(generated), "text")
    assert isinstance(out_bytes, bytes)
    assert len(out_bytes) == len(generated)  # byte mode: 1 byte per id


def test_serve_path_is_deterministic_fp8(tmp_path):
    _write_checkpoint(tmp_path, seed=1)
    model = _load_model(tmp_path)
    submodule = model.get_submodule(NODE, device="cuda", autocast_dtype=DTYPE)
    prompt_ids = model.process_prompt("serve", ["text"], ["text"])["text_inputs"][0].to(DEVICE)

    runs = [
        _run_generation(model, submodule, model.config, prompt_ids, max_tokens=5)[0]
        for _ in range(2)
    ]
    assert runs[0] == runs[1], runs
    assert len(runs[0]) == 5


def test_kv_cache_yaml_key_is_rejected(tmp_path):
    """The serving configs moved ``kv_cache:`` under ``resources: kv:``; the
    old top-level key is an error, not a silently ignored setting. (No
    checkpoint: the declaration is read off the config, nothing loads.)"""
    model = _load_model(tmp_path)
    specs = model.get_node_resources()
    with pytest.raises(ValueError, match="resources"):
        apply_yaml_overrides(specs, {"kv_cache": {"max_num_pages": 8}})
    with pytest.raises(ValueError, match="unknown resource"):
        apply_yaml_overrides(specs, {"resources": {"kv_cache": {"max_num_pages": 8}}})
