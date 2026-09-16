"""Tensor and expert parallelism without GPUs: the ranks run in threads and share a fake comm
group whose collectives are barrier-synchronised exchanges (same semantics as ``CommGroup``:
all_gather concatenates in rank order, all_reduce sums in place). Checks the sharded weight
loaders and the parallel dense forward against the single-rank model on the tiny checkpoint,
with the routed experts sharded on the intermediate dim (TP), placed whole on each rank (EP)
or both (``moe_ep_size``)."""
import threading

import pytest
import torch

from mstar.distributed.communication import CommGroup
from mstar.model.kimi_k3.components.language_model import KimiK3ForCausalLM
from mstar.model.kimi_k3.config import KimiK3Config
from mstar.model.loader import load_weights


class _Mailbox:
    def __init__(self, n: int):
        self.n = n
        self.slots: list[torch.Tensor | None] = [None] * n
        self.barrier = threading.Barrier(n)


class FakeCommGroup(CommGroup):
    """One rank of an in-process group; every rank must issue the same collective sequence."""

    def __init__(self, rank: int, mailbox: _Mailbox):
        super().__init__(my_global_rank=rank, my_group_rank=rank, group_members=list(range(mailbox.n)))
        self._mb = mailbox

    def _exchange(self, t: torch.Tensor) -> list[torch.Tensor]:
        self._mb.slots[self.rank] = t
        self._mb.barrier.wait()
        got = list(self._mb.slots)
        self._mb.barrier.wait()  # all ranks have read before anyone writes the next round
        return got

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.cat(self._exchange(input_), dim=dim)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        parts = self._exchange(input_.clone())
        total = parts[0].clone()
        for part in parts[1:]:
            total += part
        input_.copy_(total)
        return input_

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        parts = self._exchange(input_)
        total = parts[0].clone()
        for part in parts[1:]:
            total += part
        return total.chunk(self.world_size, dim=dim)[self.rank].contiguous()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        tensor.copy_(self._exchange(tensor)[src])
        return tensor

    def barrier(self):
        self._mb.barrier.wait()


def _build(cfg, tiny_dir, group, ep=1):
    with torch.device("meta"):
        lm = KimiK3ForCausalLM(cfg, comm_group=group, moe_ep_size=ep)
    lm = lm.to(torch.bfloat16)
    for name, p in lm.named_parameters():
        if name.endswith(("A_log", "dt_bias", "e_score_correction_bias")):
            p.data = p.data.float()
    lm.to_empty(device="cpu")
    load_weights(lm, tiny_dir, device="cpu")
    lm.eval()
    return lm


@pytest.mark.parametrize("tp,ep", [(2, 1), (4, 1), (2, 2), (4, 2), (4, 4)])
def test_tp_dense_forward_matches_single_rank(tiny_dir, tp, ep):
    """``tp`` ranks; the routed experts in ``ep`` expert-parallel groups (1: pure TP)."""
    cfg = KimiK3Config.from_hf_dir(tiny_dir).text
    ref = _build(cfg, tiny_dir, CommGroup.trivial())
    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (17,))
    with torch.no_grad():
        ref_logits, _ = ref.forward_dense(ids)
    mb = _Mailbox(tp)
    lms = [_build(cfg, tiny_dir, FakeCommGroup(r, mb), ep) for r in range(tp)]
    # the column-parallel lm_head shards concatenate back to the full weight
    assert torch.equal(torch.cat([lm.lm_head.weight for lm in lms], 0), ref.lm_head.weight)
    # expert placement: ep groups of whole experts, each sharded over tp / ep ranks
    moe_ref, moe = ref.model.layers[1].block_sparse_moe, lms[tp - 1].model.layers[1].block_sparse_moe
    n_exp, inter = cfg.num_experts, cfg.moe_intermediate_size
    assert moe.experts.gate_up_proj.shape == (n_exp // ep, 2 * inter // (tp // ep), moe_ref.latent_size)
    sh = moe.sharding
    for le in range(sh.local_experts):
        cols = slice(sh.inter_offset, sh.inter_offset + sh.inter_local)
        assert torch.equal(moe.experts.down_proj[le], moe_ref.experts.down_proj[sh.expert_offset + le][:, cols])
    outs: list[torch.Tensor | None] = [None] * tp
    errs: list[BaseException | None] = [None] * tp

    def run(r):
        try:
            with torch.no_grad():
                outs[r] = lms[r].forward_dense(ids)[0]
        except BaseException as e:  # release the other ranks from their barrier
            errs[r] = e
            mb.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(tp)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(e is None for e in errs), [repr(e) for e in errs if e is not None]
    for r in range(1, tp):
        assert torch.equal(outs[0], outs[r]), "ranks disagree after the gathering lm_head"
    torch.testing.assert_close(outs[0].float(), ref_logits.float(), rtol=5e-2, atol=5e-1)
    assert torch.equal(outs[0].argmax(-1), ref_logits.argmax(-1))
