"""What GLM-5.2 (and GLM-5.3, the same glm_moe_dsa geometry) declares to the quantization plan.

The released checkpoints store their linears as DeepSeek-style block fp8 (``quant_method: fp8``, 128 x 128):
the MLA projections, the dense FFNs and the routed experts load as stored (``fp8_block``, DIRECT) and run the
fused fp8 kernels. ``kv_b_proj`` stays bf16 (the absorbed MLA folds it into ``w_kc`` / ``w_vc``), as do the DSA
indexer's projections, the LM head and the MTP ``eh_proj``; the shared expert stays bf16 under ``kernels: auto``,
whose decode chain reads its bf16 weights beside the routed experts', and follows the plan under
``kernels: reference``. ``moe_fp8_resident: false`` keeps the routed experts bf16 too.
"""

from __future__ import annotations

from collections.abc import Callable

from mstar.model.components.quant_linear import QuantizableLinears
from mstar.model.loader.base import CheckpointRules
from mstar.quantization.resolve import QuantizationRoot

_TRUNK, _MTP = "model.layers.*", "mtp.transformer_layer"
_ATTN = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")
_SHARED = ("mlp.shared_expert.gate_up_proj", "mlp.shared_expert.down_proj")


def glm52_quantizable_linears(config) -> QuantizableLinears:
    stacks = (_TRUNK, _MTP)
    candidates = [f"{s}.self_attn.{name}" for s in stacks for name in _ATTN]
    candidates += [f"{_TRUNK}.mlp.gate_up_proj", f"{_TRUNK}.mlp.down_proj"]
    keep = ["lm_head", "mtp.eh_proj"]
    keep += [f"{s}.self_attn.kv_b_proj" for s in stacks]
    keep += [f"{s}.self_attn.indexer.*" for s in stacks]
    experts = [f"{s}.mlp.experts" for s in stacks]
    (candidates if config.moe_fp8_resident else keep).extend(experts)
    shared = [f"{s}.{name}" for s in stacks for name in _SHARED]
    (keep if config.kernels == "auto" else candidates).extend(shared)
    return QuantizableLinears(candidates=tuple(candidates), keep_bf16=tuple(keep))


def glm52_quantization_root(config, checkpoint_dir: Callable[[], str | None]) -> QuantizationRoot:
    from mstar.model.glm52.components.causal_lm import Glm52ForCausalLM
    from mstar.model.glm52.weight_loader import _make_glm52_name_remapper, build_glm52_stacked_params

    return QuantizationRoot(
        node="LLM", root_type=Glm52ForCausalLM, linears=glm52_quantizable_linears(config),
        rules=CheckpointRules(
            stacked_params=tuple(build_glm52_stacked_params(config.n_routed_experts)),
            remap=_make_glm52_name_remapper(config.num_hidden_layers, config.mtp_num_draft_tokens > 0),
        ),
        checkpoint_dir=checkpoint_dir,
        build=lambda comm_group: Glm52ForCausalLM(config, comm_group=comm_group),
    )
