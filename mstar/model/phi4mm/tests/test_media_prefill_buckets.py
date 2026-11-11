"""The LLM's largest media prefill buckets replayed against the eager forward, on the
real weights: ``prefill_vision`` steps of eight rows and of one row, each filling the
16384-token bucket (`Phi4MMLLMSubmodule.PREFILL_MEDIA_TOKEN_BUCKETS`), on the base
model (the bucket's lean twin) and on the checkpoint's vision adapter (the plain
bucket, whose LoRA kernels run at 16384 tokens). The one-row prompt is past 4096
tokens, so it rotates on LongRoPE's long table. A replay runs at the bucket's shapes,
so it differs from the eager step by bf16 rounding of other GEMM shapes only.

Image rows are rows of the token embedding, so the splice feeds the decoder values
of the scale it sees. Needs CUDA, FlashInfer and the cached checkpoint."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch

from mstar.model.phi4mm.tests._paths import snapshot
from mstar.model.phi4mm.tests.test_resident_decode import _served_config, _StubTransfer

NODE = "LLM"
WALK = "prefill_vision"
BUCKET_TOKENS = 16384
BATCH_SIZES = [1, 8]


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "needs CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "needs flashinfer"
    if snapshot(full=True) is None:
        return "Phi-4-multimodal weights not cached"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))

# (rows, image pads a row): every step lands in the 16384-token bucket, past the 12288 one
CASES = {"bs8": (8, 1700), "bs1": (1, 15000)}
ADAPTERS = [None, "vision"]


def _prompts(model, rows: int, pads: int, seed: int) -> list[SimpleNamespace]:
    """``rows`` prompts of ``pads`` image pads between a user tag and a question."""
    from mstar.model.phi4mm.config import IMAGE_PAD_ID
    from mstar.model.phi4mm.vision import vision_media_encoder

    tok = model.tokenizer
    head = tok("<|user|>")["input_ids"]
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i in range(rows):
        tail = tok(f"Describe image {i} of {rows} in one sentence.<|end|><|assistant|>")["input_ids"]
        ids = torch.tensor(head + [IMAGE_PAD_ID] * pads + tail, dtype=torch.long)
        out.append(SimpleNamespace(
            ids=ids,
            # embedding rows of random tokens, a little noise over them
            image_tokens=torch.randint(0, 199_000, (pads,), generator=gen),
            layout={"offset": [len(head)], "tokens": [pads], "output": [vision_media_encoder().output], "grid": [[]]},
        ))
    return out


def _engine(model, llm, monkeypatch, captured: bool, config: dict):
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.model.base import serving_resources

    cls = type(llm)
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(cls, "DECODE_CAPTURE_BATCH_SIZES", [1])
    monkeypatch.setattr(cls, "CAPTURE_PREFILL", captured)
    monkeypatch.setattr(cls, "PREFILL_MEDIA_TOKEN_BUCKETS", [12288, BUCKET_TOKENS])
    monkeypatch.setattr(cls, "PREFILL_MEDIA_CAPTURE_BATCH_SIZES", BATCH_SIZES)
    configs = cls.get_accelerator_graph_configs

    def media_and_decode(self, device, tp_world_size=1):
        # the text and mixed ladders take most of a capture's time and are not under test
        return [c for c in configs(self, device, tp_world_size) if c.capture_graph_walk in (WALK, self.DECODE_WALK)]

    monkeypatch.setattr(cls, "get_accelerator_graph_configs", media_and_decode)
    specs = [spec for spec in serving_resources(model, config) if NODE in spec.nodes]
    for spec in specs:
        if isinstance(spec, KVSpec):
            spec.config.page_size = 16
            spec.config.max_num_pages = 2048
    eng = Engine(graph_runtime=None, autocast_dtype=torch.bfloat16)
    eng.load_model(
        {NODE: llm}, specs, WorkerParallelGroups(num_workers=1, global_rank=0),
        torch.device("cuda", torch.cuda.current_device()),
        transfer_engine_info=None, kv_cache_type=torch.bfloat16, model=model,
    )
    eng.warmup()
    return eng


def _prefill(eng, model, prompts, adapter: str | None, first_rid: int, logits: torch.Tensor):
    """One ``prefill_vision`` step over ``prompts``: each row's sampled token and logits."""
    from mstar.engine.engine import ExecutingBatch
    from mstar.engine.resources.kv.config import KVReqConfig
    from mstar.engine.resources.lora.config import LoraReqConfig
    from mstar.engine.resources.position.config import PositionReqConfig
    from mstar.engine.resources.sampler.config import SamplingReqConfig
    from mstar.engine.resources.step import StepContext
    from mstar.model.multimodal import MEDIA_LAYOUT
    from mstar.model.phi4mm.vision import vision_media_encoder

    output = vision_media_encoder().output
    (lora,) = (key for key in eng._resources if key.startswith("lora"))
    embed = eng._submodules[NODE].submodule.model.embed
    rids, sampling, tensors, infos = [], {}, {}, {}
    for i, p in enumerate(prompts):
        rid = first_rid + i
        kv = KVReqConfig()
        kv.apply_conductor_config(
            kv_layout={"main": next(iter(model.kv_layout(p.layout, p.ids.numel()).values()))["main"].row()},
            loop_caps={"decode_loop": 1},
        )
        sampling[rid] = SamplingReqConfig(temperature=0.0, top_k=0, top_p=1.0, ignore_eos=True)
        sampling[rid].apply_conductor_config(seed=rid)
        configs = {"kv_cache": kv, "sampler": sampling[rid], "rope": PositionReqConfig(max_position=p.ids.numel())}
        if adapter is not None:
            configs[lora] = LoraReqConfig(adapter=adapter)
        eng.add_request(rid, configs)
        sampler = eng._resources["sampler"]
        if sampler.ask(rid, NODE, WALK) is not None:
            sampler.reserve(rid, NODE, WALK, None)
        device = eng._device
        with torch.no_grad():
            rows = embed(p.image_tokens.to(device))
            rows = rows + 0.1 * rows.std() * torch.randn(
                rows.shape, generator=torch.Generator(device).manual_seed(rid), device=device, dtype=rows.dtype,
            )
        tensors[rid] = {"text_inputs": [p.ids.to(device)], output: [rows]}
        infos[rid] = SimpleNamespace(
            rid=rid, request_id=rid, rid_handle=rid, graph_walk=WALK, max_tokens=1,
            step_metadata={"is_prefill": True, "last_prefill": True, MEDIA_LAYOUT: p.layout},
            resource_configs={"sampler": sampling[rid]},
            update_publish_info=lambda published: None, dynamic_loop_iter_counts={},
        )
        rids.append(rid)
    batch = ExecutingBatch(
        node_name=NODE, per_request_info=infos,
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False),
        per_request_input_tensors=tensors,
    )
    eng.prepare_inputs(batch)
    out = eng.exec_and_postprocess(batch)
    assert batch.admit_error is None, batch.admit_error
    torch.cuda.synchronize()
    tokens = out.check_stop_buffers["new_token"].reshape(-1).tolist()
    step_logits = logits[: len(rids)].clone()
    for rid in rids:
        eng.remove_request(rid)
    return [int(t) for t in tokens], step_logits


def _run_all(model, monkeypatch, captured: bool) -> dict:
    """Every case on a fresh engine: tokens, logits, and the buckets that replayed."""
    from mstar.engine import accelerator_graph_runner as agr

    llm = model._create_llm("cuda", torch.bfloat16, None)
    logits = torch.zeros(max(BATCH_SIZES), model.config.vocab_size, device="cuda")
    head = llm.model.logits

    def recorded(hidden):
        out = head(hidden)
        logits[: out.shape[0]].copy_(out)
        return out

    llm.model.logits = recorded
    replayed: list = []
    replay = agr.AcceleratorGraphRunner._replay

    def counted(self, lease):
        replayed.append(lease.bucket)
        return replay(self, lease)

    monkeypatch.setattr(agr.AcceleratorGraphRunner, "_replay", counted)
    eng = _engine(model, llm, monkeypatch, captured, _served_config(model))
    results, rid = {}, 1
    for name, (rows, pads) in CASES.items():
        prompts = _prompts(model, rows, pads, seed=rows)
        for adapter in ADAPTERS:
            replayed.clear()
            tokens, step_logits = _prefill(eng, model, prompts, adapter, rid, logits)
            rid += rows
            results[name, adapter] = SimpleNamespace(tokens=tokens, logits=step_logits, replayed=list(replayed))
    eng.shutdown()
    del eng, llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


@pytest.fixture(scope="module")
def runs():
    from mstar.model.phi4mm.phi4mm_model import Phi4MMModel

    model = Phi4MMModel(str(snapshot(full=True)))
    mp = pytest.MonkeyPatch()
    try:
        eager = _run_all(model, mp, captured=False)
        mp.undo()
        captured = _run_all(model, mp, captured=True)
    finally:
        mp.undo()
    return eager, captured


@pytest.mark.parametrize("adapter", ADAPTERS, ids=["base", "vision"])
@pytest.mark.parametrize("case", list(CASES))
def test_replay_matches_eager(runs, case, adapter):
    eager, captured = runs[0][case, adapter], runs[1][case, adapter]
    rows = CASES[case][0]
    assert not eager.replayed
    (bucket,) = captured.replayed
    assert (bucket.graph_walk, bucket.bs, bucket.num_tokens) == (WALK, rows, BUCKET_TOKENS)
    # the base rows replay the bucket's LoRA-free twin, the adapter's rows the plain bucket
    assert bool(bucket.lean) == (adapter is None)
    a, b = captured.logits.float(), eager.logits.float()
    assert torch.isfinite(a).all()
    # the recorded logits are this step's: greedy picks their argmax
    assert a.argmax(-1).tolist() == captured.tokens and b.argmax(-1).tolist() == eager.tokens
    rel = float((a - b).norm() / b.norm())
    print(f"\n{case} {adapter}: replay vs eager logits rel {rel:.1e}, max {float((a - b).abs().max()):.3f}")
    assert rel < 2e-2
    assert captured.tokens == eager.tokens
