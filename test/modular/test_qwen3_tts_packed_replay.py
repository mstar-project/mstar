"""A packed Talker prefill replayed through the engine's ``AcceleratorGraphRunner``, against the same step run eagerly.

A packed replay copies only the step's real rows into the captured static input, so the rows behind them keep whatever
an earlier, longer step left there. The Talker's residual stream is written in place, and if it started at that input
those rows would gain every layer's output on each replay.
"""
from __future__ import annotations

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner, autocast_scope
from mstar.engine.resources import StepContext, StepRunner, apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.model.qwen3_tts.components.talker import Qwen3TTSCodePredictor, Qwen3TTSTalkerModel
from mstar.model.qwen3_tts.config import (
    TALKER_KV,
    Qwen3TTSCodePredictorConfig,
    Qwen3TTSModelConfig,
    Qwen3TTSTalkerConfig,
)
from mstar.model.qwen3_tts.qwen3_tts_model import Qwen3TTSModel
from mstar.model.qwen3_tts.submodules import TalkerSubmodule
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture needs a GPU")

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NODE = "Talker"
WALK = "talker_prefill"
BUCKET, ROWS = 32, 2
# each step's prompts: a long one first, so later steps leave real embeddings behind their rows
STEPS = [(9, 8), (5, 3), (4, 2), (7, 1), (2, 2)]


def _config() -> Qwen3TTSModelConfig:
    code_predictor = Qwen3TTSCodePredictorConfig(
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=256,
        intermediate_size=512, head_dim=64, vocab_size=128, num_code_groups=4,
    )
    talker = Qwen3TTSTalkerConfig(
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=4, hidden_size=512,
        intermediate_size=1024, head_dim=64, vocab_size=128, text_hidden_size=64, text_vocab_size=128,
        num_code_groups=4, codec_pad_id=120, codec_bos_id=121, codec_eos_token_id=122, codec_think_id=123,
        codec_nothink_id=124, codec_think_bos_id=125, codec_think_eos_id=126, code_predictor=code_predictor,
    )
    return Qwen3TTSModelConfig(talker=talker)


def _submodule(config: Qwen3TTSModelConfig) -> TalkerSubmodule:
    torch.manual_seed(0)
    talker = Qwen3TTSTalkerModel(config)
    code_predictor = Qwen3TTSCodePredictor(config)
    for module in (talker, code_predictor):
        for name, param in module.named_parameters():
            if "norm" in name:
                torch.nn.init.normal_(param, 1.0, 0.1)
            elif param.dim() > 1:
                torch.nn.init.normal_(param, 0.0, 0.02)
            else:
                torch.nn.init.zeros_(param)
        module.to(device=DEVICE, dtype=DTYPE).eval().requires_grad_(False)
    code_predictor.consolidate_stacked_weights()
    return TalkerSubmodule(talker, code_predictor, config)


class _Node:
    """The node's resources as the engine builds them and the step cycle around the submodule: with ``capture``, the
    engine's replay path (lease, pad, declare over the padded ids, plan, stage and replay)."""

    def __init__(self, capture: bool, measured: bool):
        config = _config()
        model = Qwen3TTSModel.__new__(Qwen3TTSModel)
        model.config = config
        specs = model.get_node_resources()
        apply_yaml_overrides(specs, {"resources": {TALKER_KV: {"max_num_pages": 64, "page_size": 16}}})
        by_key = resolve_spec_dependencies(specs)
        groups = JointGroups(tp_group=CommGroup.trivial(), sp_group=CommGroup.trivial())
        transfer = TransferEngineInfo(
            my_entity_id="qwen3_tts_packed", my_session_id="qwen3_tts_packed",
            transfer_engine=LocalTransferEngine("localhost"),
        )
        self.resources = {
            spec.resource_key: build_resource(spec, EngineResourceInfo(
                device=DEVICE, joint_comm_group=groups, transfer_engine_info=transfer, kv_dtype=DTYPE,
                dependencies={key: by_key[key] for key in spec.depends_on()},
            ))
            for spec in specs
        }
        self.runner = StepRunner(self.resources, node_resources={NODE: list(self.resources)})
        self.overrides = model.get_request_resource_configs({}, model_kwargs={"do_sample": False})
        self.sub = _submodule(config)
        self.sub.model.model.fused_add_norm = measured
        self.sub.bind_node_resources(self.resources)
        self.hidden: torch.Tensor | None = None
        self.sub.model.register_forward_hook(self._keep_hidden)
        self.cg = None
        if capture:
            self.cg = AcceleratorGraphRunner(
                submodule_name=NODE, submodule=self.sub, resources=self.resources, step_runner=self.runner,
                device=DEVICE, autocast_dtype=DTYPE, joint_comm_group=groups, num_slots=1,
            )
            with torch.no_grad():
                self.cg.warmup_and_capture(None)

    def _keep_hidden(self, _module, _args, output: torch.Tensor) -> None:
        # a captured graph's tensor: this slot's values after each replay
        self.hidden = output

    def static_embeds(self) -> torch.Tensor:
        lease = self.cg.lease_slot(WALK, ROWS, slot=0, num_tokens=BUCKET)
        return self.cg.slot_for(lease).static_inputs["input_embeds"]

    def step(self, prompts: dict[str, torch.Tensor]) -> torch.Tensor:
        """One packed step; the rows of the Talker's output that belong to the step's prompts."""
        rids = list(prompts)
        infos = {
            rid: CurrentForwardPassInfo(
                request_id=rid, graph_walk=WALK, fwd_index=0, random_seed=0, max_tokens=8,
                resource_configs=self.overrides,
            )
            for rid in rids
        }
        for rid in rids:
            self.runner.ingest_request(rid, self.overrides)
        inputs = [
            ARNodeInputs(input_embeds=embeds, input_seq_len=embeds.shape[0], kwargs={"suppress_eos": True})
            for embeds in prompts.values()
        ]
        num_tokens = sum(inp.input_seq_len for inp in inputs)
        lease, step_ids, meta = None, rids, infos
        if self.cg is not None:
            lease = self.cg.lease_slot(WALK, len(rids), slot=0, num_tokens=num_tokens)
            assert lease is not None, "no captured bucket"
            inputs = self.cg.pad_inputs(lease, inputs)
            step_ids = self.cg.step_ids(lease, rids)
            meta = self.cg.step_metadata(lease, rids, infos)
        ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False, slot_lease=lease)
        ctx.set_padded_rids(step_ids)
        step = self.sub.declare_step(WALK, list(step_ids), inputs, slot_lease=lease)
        step.set_ctx(ctx)
        try:
            with torch.no_grad(), autocast_scope(DTYPE, device_type=DEVICE.type):
                assert self.runner.admit(step).ok
                self.runner.plan(step)
                engine_inputs = ModelInputsFromEngine(
                    request_ids=list(step_ids), per_request_info=meta, resources=self.resources, step=step,
                    captured=lease is not None,
                )
                pre = self.sub.preprocess(WALK, engine_inputs, inputs)
                if lease is None:
                    self.sub.forward_batched(WALK, engine_inputs, **pre)
                else:
                    self.cg.run_forward(lease, pre)
                self.runner.commit(step)
        finally:
            if lease is not None:
                self.cg.release(lease, len(rids))
        torch.cuda.synchronize()
        hidden = self.hidden[:num_tokens].clone()
        for rid in rids:
            self.runner.remove_request(rid)
        return hidden


def _prompts(lengths: tuple[int, ...], seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        f"r{seed}_{i}": torch.randn(n, _config().talker.hidden_size, generator=generator).to(DEVICE, DTYPE)
        for i, n in enumerate(lengths)
    }


@pytest.mark.parametrize("measured", [True, False], ids=["fused", "plain"])
def test_packed_replays_leave_the_static_input_alone(monkeypatch, measured):
    monkeypatch.setattr(TalkerSubmodule, "PREFILL_TOKEN_BUCKETS", [BUCKET])
    monkeypatch.setattr(TalkerSubmodule, "PREFILL_CAPTURE_BATCH_SIZES", [ROWS])
    packed_only = TalkerSubmodule.get_accelerator_graph_configs
    monkeypatch.setattr(
        TalkerSubmodule, "get_accelerator_graph_configs", lambda self, device, tp=1: packed_only(self, device, tp)[:1],
    )
    packed, eager = _Node(capture=True, measured=measured), _Node(capture=False, measured=measured)

    for seed, lengths in enumerate(STEPS):
        prompts = _prompts(lengths, seed)
        real = sum(lengths)
        assert real < BUCKET
        before = packed.static_embeds().clone()
        got = packed.step(prompts)
        want = eager.step(prompts)

        after = packed.static_embeds()
        assert torch.isfinite(after.float()).all()
        assert torch.equal(after[real:], before[real:]), f"step {seed} wrote past its {real} real rows"
        assert torch.equal(after[:real], torch.cat(list(prompts.values()))), f"step {seed} wrote into its real rows"
        # real rows, at one bf16 step per residual add
        torch.testing.assert_close(got.float(), want.float(), atol=0.05, rtol=0.02)
