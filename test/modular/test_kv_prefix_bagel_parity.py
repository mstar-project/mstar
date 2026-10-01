"""How far a real forward moves when its prefix comes from the cache.

`test_kv_prefix_correctness.py` never runs a kernel, so it can compare exactly.
This one builds Bagel's language model over the real KV, attention and position
resources and computes the same tokens twice: once attending a prefix inherited
from an earlier request, once attending one it computed itself. Attention is
planned over a different number of pages in the two runs, so what is asked of
them is the repo's parity standard rather than equality.

The dtype decides which standard applies: FlashInfer's paged attention refuses
fp32, so this runs in bf16, where the repo's tolerance is atol=rtol=5e-2 from
`test_pi05_reference_equivalence.py`. The 1e-5 in
`vjepa2/test_ac_kv_cache_parity.py` belongs to an fp32 comparison on CPU.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    PagedKVConfig,
    PositionConfig,
    StepContext,
    StepRunner,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
)
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVSpec, KVStep
from mstar.engine.resources.kv.keys import chain
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.config import PositionStep
from mstar.engine.resources.position.manager import RopeManager
from mstar.engine.resources.step import Segment, SubmoduleStep

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the attention backend needs a device"
)

PAGE_SIZE = 16
ROOT = b"a root"
NODE = "LLM"
WALK = "prefill_text"
# what this repo accepts between two bf16 paths; see the module docstring
ATOL = RTOL = 5e-2
DTYPE = torch.bfloat16


class _StubTransfer:
    """No engine, no bytes moved."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _small_config():
    """Bagel's own config dataclass, small enough to initialise at random."""
    from mstar.model.bagel.config import (
        BagelAutoEncoderConfig,
        BagelModelConfig,
        BagelViTConfig,
    )

    return BagelModelConfig(
        vae_config=BagelAutoEncoderConfig(),
        vit_config=BagelViTConfig(),
        # head_dim 64: flashinfer's rope kernel takes 64, 128 or 256
        vocab_size=512, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )


def _initialise(llm: torch.nn.Module) -> None:
    """Give every parameter a value, from the same seed on both sides.

    The constructor leaves them uninitialised for a checkpoint to fill, so a
    model built and run straight away reads whatever was in that memory:
    ``embed_tokens`` comes out NaN and every hidden state after it follows.
    """
    torch.manual_seed(20260920)
    with torch.no_grad():
        for name, param in llm.named_parameters():
            if name.endswith("norm.weight"):
                param.fill_(1.0)
            elif name.endswith(".bias"):
                param.zero_()
            else:
                param.normal_(0.0, 0.02)


class _Node:
    """Bagel's language model over the real resources, at a small random init."""

    def __init__(self, device: torch.device, cached: bool):
        from mstar.model.bagel.components.language_model import BagelForCausalLM

        self.device = device
        config = _small_config()
        self.config = config
        self.llm = BagelForCausalLM(config, comm_group=None).to(
            device=device, dtype=DTYPE,
        ).eval()
        _initialise(self.llm)

        kv_config = PagedKVConfig(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=2048, max_num_pages=128, page_size=PAGE_SIZE,
        )
        self.kv = KVManager(
            cfg=kv_config, name="kv", joint_comm_group=None,
            transfer_engine_info=None, device=device, dtype=DTYPE,
        )
        if cached:
            self.kv.enable_prefix_cache(ROOT)
        self.rope = RopeManager(
            config=PositionConfig(kv_cache="kv"), device=device, dtype=DTYPE,
        )
        self.attn = AttentionManager.build(
            AttentionSpec(
                resource_key="attn", nodes={NODE},
                config=AttentionConfig(kv_cache="kv"),
            ),
            EngineResourceInfo(
                device=device, joint_comm_group=None,
                transfer_engine_info=None, kv_dtype=DTYPE,
                dependencies={"kv": KVSpec(
                    resource_key="kv", nodes={NODE}, config=kv_config,
                )},
            ),
        )
        self.runner = StepRunner(
            {"kv": self.kv, "attn": self.attn, "rope": self.rope},
            node_resources={NODE: ["kv", "attn", "rope"]},
        )
        # what NodeSubmodule.bind_resources does: each attention layer keeps
        # its own references, resolved once
        resources = {"kv": self.kv, "attn": self.attn, "rope": self.rope}
        for module in self.llm.modules():
            bind = getattr(module, "bind_resources", None)
            if bind is not None:
                bind(resources)

    def ingest(self, rid: str, tokens: list[int]) -> None:
        whole = len(tokens) // PAGE_SIZE
        self.runner.ingest_request(rid, {"kv": KVReqConfig(
            prefix_keys={"main": chain([
                tokens[at:at + PAGE_SIZE]
                for at in range(0, len(tokens), PAGE_SIZE)
            ])},
            prefix_tail={"main": tokens[whole * PAGE_SIZE:]},
        )})

    def resolve(self, rid: str) -> int:
        matched = self.runner.resolve_cached_prefix(rid, NODE, WALK)
        self.runner.apply_cached_prefix(rid, NODE, WALK, None, matched)
        return matched

    def prefill(self, rid: str, tokens: list[int]) -> torch.Tensor:
        """Run one prefill over ``tokens`` and give back its hidden states."""
        step = SubmoduleStep(
            steps={
                "kv": KVStep(),
                "attn": AttentionStep(causal=True),
                "rope": PositionStep(),
            },
            segments=[Segment(rid, "main", len(tokens))],
        )
        ctx = StepContext(
            request_ids=(rid,), graph_walk=WALK, slot=0, capture=False,
        )
        step.set_ctx(ctx)
        assert self.runner.admit(step).outcome.ok
        self.runner.plan(step)
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            hidden = self.llm.model.embed_tokens(ids)
            hidden = self.llm(hidden, mode="und", label="main")
        self.runner.commit(step)
        return hidden.float()


@requires_cuda
def test_a_consumed_prefix_lands_within_the_repos_parity_tolerance(capsys):
    device = torch.device("cuda:0")
    prompt = list(range(1, 161))
    tail = list(range(400, 437))

    warm = _Node(device, cached=True)
    warm.ingest("seed", prompt)
    warm.resolve("seed")
    warm.prefill("seed", prompt)
    warm.kv.remove_request("seed")

    warm.ingest("cached", prompt + tail)
    matched = warm.resolve("cached")
    assert matched, "nothing matched, so this would compare two fresh runs"
    from_cache = warm.prefill("cached", (prompt + tail)[matched:])

    cold = _Node(device, cached=False)
    cold.ingest("fresh", prompt + tail)
    assert cold.resolve("fresh") == 0, (
        "the fresh node matched something, so it is not the uncached side"
    )
    whole = cold.prefill("fresh", prompt + tail)
    fresh = whole[matched:]

    for what, hidden in (("cached", from_cache), ("fresh", fresh)):
        assert hidden.isfinite().all(), (
            f"the {what} run produced non-finite hidden states, so the "
            "comparison below would be measuring an overflow, not the cache"
        )
        assert hidden.abs().max() > 1e-2, (
            f"the {what} run produced hidden states of about zero, which two "
            "runs would agree on however wrong the cache was"
        )
    deviation = (from_cache - fresh).abs().max().item()
    with capsys.disabled():
        # printed, not just asserted: the margin is the number this test is for
        print(f"\n  matched {matched} tokens; deviation {deviation:.3e}")
    torch.testing.assert_close(
        from_cache, fresh, atol=ATOL, rtol=RTOL,
        msg=lambda default: (
            f"a prefix taken from the cache moved the forward by {deviation:.3e}, "
            f"past the {ATOL:.0e} this repo accepts between a cached path and an "
            f"uncached one\n{default}"
        ),
    )
