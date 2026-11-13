"""The tiny Kimi K3.1 at TP2 (and with EP2 routed experts) on two gloo ranks through the real
``Engine``: each rank loads its slices of one random checkpoint, both ranks sample the same
tokens, and the logits match the single-process reference. This runs the sharded paths a TP1
run never takes: the KDA and MLA head shards and their all-reduces, the vocab-parallel
embedding and head, the latent MoE's row-parallel up projection summed with the shared
experts' partial, the expert-parallel routing broadcast, and K3.1's DSA past index_topk (decode,
and a prompt past it)."""
import datetime

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mstar.model.kimi_k3.config import KimiK3ModelConfig
from mstar.model.kimi_k3.tests.reference import random_checkpoint, reference_logits

PROMPT = torch.tensor([5, 9, 13, 7, 21, 2])
STEPS = 3
CASES = {
    "tp2": {},
    "ep2": {"moe_expert_parallel": True},
    "tp2_dsa": {"dsa_long_context": True, "index_topk": 8, "max_seq_len": 32},
    # the prompt itself past index_topk: prefill rows on the DSA resource
    "tp2_dsa_prefill": {"dsa_long_context": True, "index_topk": 4, "max_seq_len": 32},
}


def _config(case: str) -> KimiK3ModelConfig:
    return KimiK3ModelConfig.reduced(**CASES[case])


def _rank(rank: int, case: str, store: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=120))
    torch.set_num_threads(1)
    from mstar.communication.tensors import LocalTransferEngine
    from mstar.distributed.all_reduce import AllReduce
    from mstar.distributed.communication import CommGroup, WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv.transfer import TransferEngineInfo
    from mstar.model.kimi_k3.components.causal_lm import KimiK3ForCausalLM
    from mstar.model.kimi_k3.config import SAMPLER
    from mstar.model.kimi_k3.kimi_k3_model import KimiK3Model
    from mstar.model.kimi_k3.submodules import KimiK3LLMSubmodule
    from mstar.model.kimi_k3.tests import engine_helpers as H
    from mstar.model.utils import check_loaded, process_weights_after_loading

    class _Patch:
        def setitem(self, d, k, v):
            d[k] = v

        def setattr(self, o, n, v):
            setattr(o, n, v)

    H.install_cpu_stubs(_Patch())
    group = CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=[0, 1])
    group.device_group = group.cpu_group = dist.new_group([0, 1])
    group.initialized = True
    group._all_reduce = AllReduce()
    try:
        groups = WorkerParallelGroups(num_workers=2, global_rank=rank, any_parallelism=True)
        groups.add("LLM", group, "tp")
        groups.node_to_instance_comm["LLM"] = group
        model = KimiK3Model("x", config_variant="reduced", kda_conv_dtype=torch.float32,
                            kda_max_requests=4, tokenizer_mode="byte")
        model.config = _config(case)
        lm = KimiK3ForCausalLM(model.config, comm_group=group)
        sd = random_checkpoint(model.config, seed=0, a_log_pad=3)
        check_loaded(lm, lm.load_weights(("language_model." + k, v) for k, v in sd.items()), "tiny")
        process_weights_after_loading(lm, torch.device("cpu"))
        sub = KimiK3LLMSubmodule(lm.eval(), model.config)
        engine = Engine(None)
        engine.load_model(
            {"LLM": sub}, model.get_node_resources(), parallel_groups=groups, device=torch.device("cpu"),
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=f"e{rank}", my_session_id="s", transfer_engine=LocalTransferEngine("localhost")),
            kv_cache_type=torch.float32)
        greedy = H.GreedySampler()
        engine._resources[SAMPLER] = engine._submodules["LLM"].resources[SAMPLER] = greedy
        sub.node_resources[SAMPLER] = greedy
        engine.warmup()
        H.add_request(engine, "a")
        tokens = [H.run_step(engine, "prefill", {"a": PROMPT})["a"].view(1)]
        for _ in range(STEPS):
            tokens.append(H.run_step(engine, "decode", {"a": tokens[-1]})["a"].view(1))
        torch.save({"tokens": torch.cat(tokens), "logits": torch.stack([x[-1] for x in greedy.logits])},
                   f"{out_dir}/{rank}.pt")
    finally:
        # gloo's threads release a collective's tensors under the GIL; drop the group's last
        # references before the process finalizes
        group.device_group = group.cpu_group = group._all_reduce = None
        dist.destroy_process_group()


@pytest.mark.parametrize("case", list(CASES))
def test_two_ranks_match_the_reference(tmp_path, case):
    mp.spawn(_rank, args=(case, str(tmp_path / "store"), str(tmp_path)), nprocs=2)
    r0, r1 = (torch.load(tmp_path / f"{r}.pt") for r in (0, 1))
    assert torch.equal(r0["tokens"], r1["tokens"]) and torch.equal(r0["logits"], r1["logits"])
    cfg = _config(case)
    sd = random_checkpoint(cfg, seed=0, a_log_pad=3)
    for i in range(STEPS + 1):
        ref = reference_logits(sd, torch.cat([PROMPT, r0["tokens"][:i]]), cfg, use_dsa=cfg.dsa_long_context)
        torch.testing.assert_close(r0["logits"][i], ref[-1], atol=1e-4, rtol=0)
        assert r0["tokens"][i].item() == ref[-1].argmax().item()
