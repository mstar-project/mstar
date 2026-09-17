"""GLM-5.2 DSA long-context serve e2e: decode PAST index_topk on the real
absorbed paged-latent path (1 GPU, reduced dims, tiny topk=8).
"""
import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.cuda_graph_runner import autocast_scope
from mstar.engine.resources import (
    KVLayout,
    StepContext,
    StepRunner,
    apply_yaml_overrides,
    resolve_spec_dependencies,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
from mstar.model.glm52.config import ATTN_RESOURCE, KV_RESOURCE, Glm52ModelConfig
from mstar.model.glm52.submodules import Glm52LLMSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA long-context serve e2e needs a GPU (real paged latent cache)",
)

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NODE = "LLM"
TOPK = 8
PROMPT_LEN = 6
MAX_TOKENS = 14  # 1 prefill + 13 decode tokens: final context 6 + 13 = 19, far past topk
# 8-token pages, 16 of them (one is the sink page): 120 cached tokens for a
# 20-token context, and the SDPA fallback's page table spans every page.
KV_YAML = {"resources": {KV_RESOURCE: {"max_num_pages": 16, "page_size": 8}}}


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


def _hf_checkpoint(model, cfg):
    """bf16 passthrough keys (no fp8 pairs — quantization is orthogonal to
    the DSA path and test_glm52_serve_e2e.py already covers it), including
    the FULL-layer indexer weights the sparse path needs."""
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
        sd[p + "self_attn.kv_b_proj.weight"] = a.kv_b_proj.weight
        sd[p + "self_attn.o_proj.weight"] = a.o_proj.weight
        if a.indexer is not None:
            sd[p + "self_attn.indexer.wq_b.weight"] = a.indexer.wq_b.weight
            sd[p + "self_attn.indexer.wk.weight"] = a.indexer.wk.weight
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
                sd[p + f"mlp.experts.{e}.gate_proj.weight"] = gup[e, :moe_inter, :]
                sd[p + f"mlp.experts.{e}.up_proj.weight"] = gup[e, moe_inter:, :]
                sd[p + f"mlp.experts.{e}.down_proj.weight"] = dwn[e]
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
    ref = Glm52ForCausalLM(cfg).to(device=DEVICE, dtype=DTYPE)
    ref.model.embed_tokens.weight.data.normal_(0, 0.05)
    ref.model.norm.weight.data.normal_(1.0, 0.02)
    ref.lm_head.weight.data.normal_(0, 0.02)
    for layer in ref.model.layers:
        _fill_layer(layer, cfg)
    save_file(_hf_checkpoint(ref.eval(), cfg), str(tmp_path / "model.safetensors"))


class _Serve:
    """One request's serve loop over the node's real resources — the
    test_glm52_serve_e2e.py harness on the absorbed declaration: the model
    declares a ``KVLayout.MLA`` cache and the MLA attention resource, the
    YAML block sizes the pages, ``build_resource`` builds them.
    """

    def __init__(self, model, submodule: Glm52LLMSubmodule, rid: str = "r0"):
        self.submodule = submodule
        self.rid = rid
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, KV_YAML)
        by_key = resolve_spec_dependencies(specs)
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        transfer = TransferEngineInfo(
            my_entity_id="glm52_dsa_serve", my_session_id="glm52_dsa_serve",
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
        # ignore_eos: reduced EOS ids live inside the byte vocab; greedy decode
        # on random weights may hit one and break the exact-length assertions.
        self.overrides = model.get_request_resource_configs(
            {}, model_kwargs={"temperature": 0.0, "ignore_eos": True},
        )
        self.runner.ingest_request(rid, self.overrides)

    def info(self, max_tokens: int) -> CurrentForwardPassInfo:
        return CurrentForwardPassInfo(
            request_id=self.rid, graph_walk="prefill", fwd_index=0, random_seed=0,
            max_tokens=max_tokens, resource_configs=self.overrides,
        )

    def step(self, walk: str, info: CurrentForwardPassInfo, text: torch.Tensor,
             batched: bool = True) -> dict:
        """declare -> admit -> plan -> preprocess -> forward -> commit. The
        plan runs BEFORE preprocess, as in the engine: the DSA preprocess
        reads this step's page tables off the planned attention resource."""
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

    def close(self, cleanup_request: bool = True):
        self.runner.remove_request(self.rid)
        if cleanup_request:
            self.submodule.cleanup_request(self.rid)
        for resource in self.resources.values():
            resource.cleanup()


def _generate_greedy(model, submodule, prompt_ids, max_tokens):
    """The real serve loop (prefill -> decode -> check_stop) on the sparse
    path; returns (generated tokens, stopped-via-check_stop). Leaves the
    submodule's per-request state (the k-store) for the caller to inspect."""
    serve = _Serve(model, submodule)
    info = serve.info(max_tokens)
    generated: list[int] = []
    stopped = False
    try:
        out = serve.step("prefill", info, prompt_ids)
        new_token = out["new_token"][0]
        submodule.postprocess(serve.rid, info, out)
        generated.append(int(new_token.item()))
        next_token = out["text_inputs"][0]

        for step in range(max_tokens + 4):  # +slack; check_stop must break first
            out = serve.step("decode", info, next_token)
            new_token = out["new_token"][0]
            submodule.postprocess(serve.rid, info, out)
            info.dynamic_loop_iter_counts["decode_loop"] = step
            stop = submodule.check_stop(serve.rid, info, out)
            generated.append(int(new_token.item()))
            next_token = out["text_inputs"][0]
            if stop:
                stopped = True
                break
    finally:
        serve.close(cleanup_request=False)
    return generated, stopped


def _teacher_forced_logits(model, submodule, prompt_ids, forced_tokens):
    """Replay a fixed token sequence and collect the logits of every step:
    index 0 = prefill (context PROMPT_LEN), index 1+j = decode of
    ``forced_tokens[j]`` (post-step context PROMPT_LEN + 1 + j)."""
    serve = _Serve(model, submodule)
    info = serve.info(len(forced_tokens) + 8)
    logits_per_step = []
    try:
        for i, ids in enumerate(
            [prompt_ids, *[torch.tensor([t], device=DEVICE) for t in forced_tokens]]
        ):
            walk = "prefill" if i == 0 else "decode"
            logits_per_step.append(serve.step(walk, info, ids, batched=False)["logits"][0])
    finally:
        serve.close()
    return logits_per_step


def test_dsa_longctx_decode_past_topk(tmp_path):
    from mstar.model.glm52.glm52_model import Glm52Model

    _write_checkpoint(tmp_path, seed=0)

    model = Glm52Model(
        model_path_hf="", config_variant="reduced",
        checkpoint_path=str(tmp_path), tokenizer_mode="byte",
    )
    cfg = model.config
    cfg.mla_absorb = True         # sparse path reads the paged latent cache
    cfg.dsa_long_context = True
    cfg.index_topk = TOPK

    submodule = model.get_submodule(NODE, device="cuda", autocast_dtype=DTYPE)
    assert isinstance(submodule, Glm52LLMSubmodule)
    # Long-context serving is eager-only (host-side DSA upkeep would not
    # replay inside a captured graph): neither whole-forward nor piecewise.
    assert submodule.get_cuda_graph_configs(DEVICE) == []
    assert submodule.get_piecewise_cuda_graph_configs(DEVICE, DTYPE) == {}
    # The absorbed declaration: one latent row per token, the MLA resource
    # planned over it (its SDPA fallback at reduced dims).
    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    assert specs[KV_RESOURCE].config.layout == KVLayout.MLA
    assert specs[KV_RESOURCE].config.head_dim == cfg.kv_lora_rank + cfg.qk_rope_head_dim
    assert specs[ATTN_RESOURCE].config.ckv_dim == cfg.kv_lora_rank

    prompt_ids = torch.arange(10, 10 + PROMPT_LEN, device=DEVICE)

    # ---- (1) the sparse path serves and stops cleanly ------------------
    generated, stopped = _generate_greedy(model, submodule, prompt_ids, MAX_TOKENS)
    assert stopped, "decode loop did not terminate via check_stop"
    assert len(generated) == MAX_TOKENS, generated  # max_tokens counts the prefill-emitted token (vLLM semantics)
    assert all(0 <= t < cfg.vocab_size for t in generated), generated

    # k-store lifecycle on the real path: one row per token per FULL layer
    # (prefill 6 + one per decode forward), gone after retirement.
    assert submodule._dsa_k_store.tokens("r0", 0) == PROMPT_LEN + MAX_TOKENS - 1  # 6 prompt rows + 13 decode forwards
    assert submodule._dsa_k_store.tokens("r0", 1) == 0  # SHARED layer: none
    submodule.cleanup_request("r0")
    assert submodule._dsa_k_store.tracked_requests() == set()

    # ---- teacher-forced replays of the generated sequence --------------
    forced = generated[:-1]  # the last token is never fed back
    sparse = _teacher_forced_logits(model, submodule, prompt_ids, forced)
    assert all(torch.isfinite(step).all() for step in sparse)
    # The replay must reproduce the generation (greedy = argmax, and the
    # sparse path is deterministic step for step).
    replay_tokens = [int(step.argmax(-1).item()) for step in sparse]
    assert replay_tokens == generated

    cfg.index_topk = 512  # dense comparator: selection never engages
    dense = _teacher_forced_logits(model, submodule, prompt_ids, forced)

    cfg.index_topk = TOPK
    cfg.dsa_long_context = False  # flag-off serve, valid only within topk
    identity_steps = TOPK - PROMPT_LEN  # decode steps whose post ctx <= topk
    flag_off = _teacher_forced_logits(
        model, submodule, prompt_ids, forced[:identity_steps])
    cfg.dsa_long_context = True

    # ---- (2) prefix property: bitwise identity while ctx <= topk -------
    # Step i covers context PROMPT_LEN + i; identity for i <= topk - PROMPT_LEN.
    for i in range(identity_steps + 1):
        assert torch.equal(sparse[i], dense[i]), f"step {i} vs dense"
        assert torch.equal(sparse[i], flag_off[i]), f"step {i} vs flag-off"

    # ---- (3) the sparse path actually engages beyond topk --------------
    beyond = [
        not torch.equal(sparse[i], dense[i])
        for i in range(identity_steps + 1, len(sparse))
    ]
    assert any(beyond), (
        "no step beyond topk diverged from the dense comparator — the "
        "sparse path never engaged"
    )
    # By the final step only 8 of 20 positions are attended; equality there
    # would be a numerical accident this test must not tolerate.
    assert beyond[-1], "final step (ctx 20, topk 8) matched dense attention"
