"""BAGEL's CFG branches must be at the right rope position when they denoise.

The three image-generation branches are three labels on one cache, and two
things put their position counters where they belong: a fork (``cfg_text``
starts where ``main`` was when it split off) and the frozen flow-matching
loop (every Euler step re-attends the same context at the same position, so
nothing advances). Both are declared by ``LLMSubmodule.declare_step`` and
carried out by the KV + position resources, so these drive the real
declaration through the real resources.

Getting either wrong leaves the guidance branches attending their own KV at
positions they were never written at, which shows up as images that drift
off-prompt rather than as anything that raises.
"""

from __future__ import annotations

import sys
import types

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import KVConfig, PositionConfig, StepContext, StepRunner
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.position.manager import RopeManager
from mstar.model.bagel.submodules import LLMSubmodule, active_labels
from mstar.model.submodule_base import ARNodeInputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KV resource allocates its cache (and IPC handle) on device",
)

RID = "r0"
LABELS = ("main", "cfg_text", "cfg_img")
S_SYS, S_PROMPT, S_LATENT = 32, 20, 66


class _Harness:
    """The KV + position resources under a real BAGEL step declaration."""

    def __init__(self, device: torch.device):
        from mstar.communication.tensors import LocalTransferEngine

        self.device = device
        self.kv = KVManager(
            cfg=KVConfig(
                num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
                max_num_pages=512, page_size=16,
            ),
            name="kv",
            joint_comm_group=None,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id="e", my_session_id="s",
                transfer_engine=LocalTransferEngine("localhost"),
            ),
            device=device,
        )
        self.rope = RopeManager(
            config=PositionConfig(kv_cache="kv"), device=device
        )
        self.runner = StepRunner({"kv": self.kv, "rope": self.rope})
        self.runner.ingest_request(RID)

        # declare_step reads only these off the submodule
        self.submodule = types.SimpleNamespace(
            node_name="LLM",
            CFG_BATCHED_LABEL=LLMSubmodule.CFG_BATCHED_LABEL,
            _get_active_labels=lambda walk, cfg: active_labels(walk, cfg, "LLM"),
        )

    def run(self, graph_walk: str, span: int, pos_ids=None) -> None:
        inputs = ARNodeInputs(input_seq_len=span)
        inputs.resource_step_info = True  # requires_cfg
        if pos_ids is not None:
            inputs.custom_pos_ids = pos_ids
        step = LLMSubmodule.declare_step(
            self.submodule, graph_walk, [RID], [inputs]
        )
        # attention and the sampler are out of scope here
        step.steps.pop("attn", None)
        step.steps.pop("sampler", None)
        step.set_ctx(StepContext(
            request_ids=(RID,), graph_walk=graph_walk, slot=0, capture=False,
        ))
        assert self.runner.admit(step).ok
        self.runner.plan(step)
        self.runner.commit(step)

    def denoise(self) -> dict[str, int]:
        """One Euler step, latents pinned at each branch's current position."""
        positions = self.positions()
        self.run("image_gen", S_LATENT, pos_ids={
            label: torch.zeros(
                S_LATENT, dtype=torch.int32, device=self.device
            ) + start
            for label, start in positions.items()
        })
        return positions

    def positions(self) -> dict[str, int]:
        return {label: self.rope.position(RID, label) for label in LABELS}

    def stored_len(self, label: str) -> int:
        stream = self.kv._streams[RID].get(label)
        return 0 if stream is None else stream.stored_len


@pytest.fixture
def harness():
    return _Harness(torch.device("cuda:0"))


def test_forked_branch_inherits_the_position_it_split_off_at(harness):
    """``cfg_text`` forks off ``main`` at the start of each text prefill.

    It takes on the pages that were written so far, so it must take on the
    position they were written at too — otherwise its own KV sits at
    positions its queries never claim.
    """
    harness.run("prefill_text", S_SYS)       # system prompt
    harness.run("prefill_text", S_PROMPT)    # user prompt

    assert harness.positions() == {
        "main": S_SYS + S_PROMPT,
        # forked before the user prompt landed: the system prompt only
        "cfg_text": S_SYS,
        "cfg_img": S_SYS + S_PROMPT,
    }
    # position and length agree: every token in the stream has a position
    for label, position in harness.positions().items():
        assert harness.stored_len(label) == position


def test_image_editing_forks_cfg_text_after_the_image_blocks(harness):
    """Editing prefills an image before the prompt, so ``cfg_text`` forks
    twice: once at commit of each image block (it tracks the image), and once
    before the prompt (it never sees the text). ``cfg_img`` is the mirror —
    it takes the text and skips the image, so it ends up two positions behind
    ``main``, one per image block.
    """
    n_img = 256
    harness.run("prefill_text", S_SYS)
    harness.run("prefill_vae", n_img)
    harness.run("prefill_vit", n_img)
    harness.run("prefill_text", S_PROMPT)

    # each image block occupies a single position
    assert harness.positions() == {
        "main": S_SYS + 2 + S_PROMPT,
        "cfg_text": S_SYS + 2,
        "cfg_img": S_SYS + S_PROMPT,
    }


def test_flow_matching_leaves_the_counters_where_prefill_left_them(harness):
    """The denoise loop reads frozen caches and writes nothing.

    Each Euler step re-attends the same context with the latents at the same
    place, so a counter that advanced per iteration would walk the latents
    off the end of the context they are conditioned on.
    """
    harness.run("prefill_text", S_SYS)
    harness.run("prefill_text", S_PROMPT)
    after_prefill = harness.positions()

    for _ in range(4):
        assert harness.denoise() == after_prefill

    assert harness.positions() == after_prefill
    # nothing was committed to the streams either
    for label, position in after_prefill.items():
        assert harness.stored_len(label) == position


def test_published_positions_travel_to_a_branch_on_another_node(harness):
    """CFG-parallel runs ``cfg_text`` on a GPU that never prefilled it.

    That node pulls the stream's KV in through ``admit_retrieve``; the
    position the stream reached has to come with it, or its first Euler step
    puts the latents at position 0.
    """
    harness.run("prefill_text", S_SYS)
    harness.run("prefill_text", S_PROMPT)

    remote = _Harness(torch.device("cuda:0"))
    published = harness.rope.publish(RID)
    remote.rope.admit_retrieve(RID, "LLM_cfg_text", "image_gen_cfg", published)

    assert remote.positions() == harness.positions()


# --- the forward and its declaration must read guidance the same way -------


def _decode_capture_inputs(cfg_on: bool):
    """The template rows a decode capture hands declare_step and preprocess."""
    stub = types.SimpleNamespace(
        PREFILL_TEXT_TOKEN_BUCKETS=LLMSubmodule.PREFILL_TEXT_TOKEN_BUCKETS,
        PREFILL_TEXT_CAPTURE_BATCH_SIZES=(
            LLMSubmodule.PREFILL_TEXT_CAPTURE_BATCH_SIZES
        ),
    )
    configs = LLMSubmodule.get_cuda_graph_configs(stub, torch.device("cuda"))
    config = next(
        c for c in configs
        if getattr(c, "capture_graph_walk", None) == "decode"
        and c.additional_key_info is cfg_on
    )
    return config.get_node_inputs(1, 1)


def _llm_stub():
    """Everything `declare_step` and `preprocess` read off the submodule."""
    return types.SimpleNamespace(
        node_name="LLM",
        CFG_BATCHED_LABEL=LLMSubmodule.CFG_BATCHED_LABEL,
        _get_active_labels=lambda walk, cfg: active_labels(walk, cfg, "LLM"),
        _batch_get_requires_cfg=LLMSubmodule._batch_get_requires_cfg.__get__(
            object()
        ),
    )


@pytest.mark.parametrize("cfg_on", [False, True], ids=["cfg_off", "cfg_on"])
def test_capture_declares_and_forwards_the_same_guidance(cfg_on):
    """The regression: `preprocess` read guidance off `per_request_info`, but
    a capture passes `dummy_metadata` (always guidance-off) alongside template
    rows that do carry it. So the cfg-on decode graph was recorded running the
    cfg-off forward, under a step that declared the guidance branches.
    """
    submodule = _llm_stub()
    inputs = _decode_capture_inputs(cfg_on)
    assert any(bool(i.resource_step_info) for i in inputs) is cfg_on, (
        "the capture template no longer carries the guidance flag"
    )

    step = LLMSubmodule.declare_step(submodule, "decode", [RID], inputs)
    forwarded = LLMSubmodule.preprocess(
        submodule, graph_walk="decode",
        engine_inputs=types.SimpleNamespace(
            # exactly what capture passes: `dummy_metadata`, always guidance-off
            per_request_info={
                RID: types.SimpleNamespace(
                    step_metadata={},                 )
            },
        ),
        inputs=inputs,
    )

    assert step.cg_key_info is cfg_on, "the declaration lost the bucket's key"
    assert forwarded["requires_cfg"] is cfg_on, (
        "the forward disagrees with the step it was declared under"
    )
