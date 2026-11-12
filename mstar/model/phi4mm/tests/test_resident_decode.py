"""The LLM's captured decode on resident tables against the host plan, on the real
weights: two short-table and two long-table prompts (LongRoPE takes the long table past
4096 prompt tokens), decoded greedily side by side in one capture, so each row's
neighbour reads the other half of the table; then fewer rows, in another order. Once
with resident planning and once with it off, the same tokens and logits. Then the same
with the LoRA pool built as the served config builds it, rows on the checkpoint's
adapters beside base rows.

Needs CUDA, FlashInfer and the cached checkpoint."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mstar.model.phi4mm.tests._paths import snapshot

NODE = "LLM"
BUCKET = 8


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


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


def _llm(model):
    """The LLM node, recording each step's logits from inside the captured graph."""
    llm = model._create_llm("cuda", torch.bfloat16, None)
    logits = torch.zeros(BUCKET, model.config.vocab_size, device="cuda")
    head = llm.model.logits

    def recorded(hidden):
        out = head(hidden)
        logits[: out.shape[0]].copy_(out)
        return out

    llm.model.logits = recorded
    return llm, logits


@pytest.fixture(scope="module")
def phi4mm():
    from mstar.model.phi4mm.phi4mm_model import Phi4MMModel

    return Phi4MMModel(str(snapshot(full=True)))


def _served_config(model) -> dict:
    """configs/phi4mm.yaml with its adapters resolved, as the API server hands it to the workers."""
    import yaml

    from mstar.engine.resources.lora.config import LoraSpec
    from mstar.engine.resources.lora.serving import resolve_lora_adapters
    from mstar.model.base import declared_resources

    config = yaml.safe_load((Path(__file__).parents[4] / "configs" / "phi4mm.yaml").read_text())
    resolve_lora_adapters(next(s for s in declared_resources(model, config) if isinstance(s, LoraSpec)), config)
    return config


def _engine(model, llm, monkeypatch, resident: bool, config: dict | None = None):
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.engine.engine import Engine
    from mstar.engine.resources.kv import manager as manager_mod
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.model.base import serving_resources

    monkeypatch.setenv("MSTAR_RESIDENT_PLANNING", "auto" if resident else "off")
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    monkeypatch.setattr(type(llm), "DECODE_CAPTURE_BATCH_SIZES", [BUCKET])
    monkeypatch.setattr(type(llm), "CAPTURE_PREFILL", False)
    specs = [spec for spec in serving_resources(model, config or {}) if NODE in spec.nodes]
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


class _Driver:
    """Plays the worker, synchronously: one prompt step per request, then decode steps."""

    def __init__(self, eng, logits: torch.Tensor, adapters: dict[int, str] | None = None):
        self.eng = eng
        self.logits = logits
        # rid -> the LoRA adapter it runs under; the others run the base model
        self.adapters = adapters or {}
        self.out: dict[int, list[int]] = {}
        self.step_logits: list[torch.Tensor] = []
        self.sampling: dict = {}
        self.max: dict[int, int] = {}

    def add(self, rid: int, prompt: list[int], max_tokens: int) -> None:
        from mstar.engine.resources.kv.config import KVReqConfig
        from mstar.engine.resources.position.config import PositionReqConfig
        from mstar.engine.resources.sampler.config import SamplingReqConfig
        from mstar.model.base import LabelLayout, Span

        kv = KVReqConfig()
        kv.apply_conductor_config(
            kv_layout={"main": LabelLayout(
                spans=(Span("ids", "prefill_text", len(prompt), len(prompt), "text_inputs"),),
                growth=(("decode_loop", 1),),
            ).row()},
            loop_caps={"decode_loop": max_tokens},
        )
        sampling = SamplingReqConfig(temperature=0.0, top_k=0, top_p=1.0, ignore_eos=True)
        sampling.apply_conductor_config(seed=rid)
        self.sampling[rid] = sampling
        # what `process_prompt` sets: the table follows the prompt's length
        rope = PositionReqConfig(max_position=len(prompt))
        configs = {"kv_cache": kv, "sampler": sampling, "rope": rope}
        if rid in self.adapters:
            from mstar.engine.resources.lora.config import LoraReqConfig

            (lora,) = (key for key in self.eng._resources if key.startswith("lora"))
            configs[lora] = LoraReqConfig(adapter=self.adapters[rid])
        self.eng.add_request(rid, configs)
        sampler = self.eng._resources["sampler"]
        if sampler.ask(rid, NODE, "prefill_text") is not None:
            sampler.reserve(rid, NODE, "prefill_text", None)
        self.out[rid], self.max[rid] = [], max_tokens

    def _run(self, walk: str, ids: dict[int, torch.Tensor], metadata: dict) -> None:
        from mstar.engine.engine import ExecutingBatch
        from mstar.engine.resources.step import StepContext

        rids = list(ids)
        batch = ExecutingBatch(
            node_name=NODE,
            per_request_info={rid: SimpleNamespace(
                rid=rid, request_id=rid, rid_handle=rid, graph_walk=walk, max_tokens=self.max[rid],
                step_metadata=dict(metadata), resource_configs={"sampler": self.sampling[rid]},
                update_publish_info=lambda published: None, dynamic_loop_iter_counts={},
            ) for rid in rids},
            step_context=StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False),
            per_request_input_tensors={rid: {"text_inputs": [row]} for rid, row in ids.items()},
        )
        self.eng.prepare_inputs(batch)
        out = self.eng.exec_and_postprocess(batch)
        assert batch.admit_error is None, batch.admit_error
        tokens = out.check_stop_buffers["new_token"].reshape(-1).tolist()
        for i, rid in enumerate(rids):
            self.out[rid].append(int(tokens[i]))
        if walk == "decode":
            torch.cuda.synchronize()
            self.step_logits.append(self.logits[: len(rids)].clone())

    def prefill(self, rid: int, prompt: list[int]) -> None:
        ids = torch.tensor(prompt, device=self.eng._device)
        self._run("prefill_text", {rid: ids}, {"is_prefill": True, "last_prefill": True})

    def step(self, rids: list[int]) -> None:
        device = self.eng._device
        self._run("decode", {rid: torch.tensor([self.out[rid][-1]], device=device) for rid in rids}, {})


def _prompts(model) -> dict[int, list[int]]:
    """Two short chats and two past 4096 tokens, the long ones a numbered log to recall from."""
    def chat(text: str) -> list[int]:
        rendered = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
        )
        return model.tokenizer(rendered)["input_ids"]

    log = " ".join(
        f"Entry {i}: the courier left depot {i % 17} at {6 + i % 12}:{i * 7 % 60:02d} carrying {i * 3 % 41} parcels."
        for i in range(260)
    )
    prompts = {
        1: chat("Write a short poem about the sea."),
        2: chat(f"{log}\n\nWhich depot did the courier leave in entry 123, and with how many parcels?"),
        3: chat("Explain in two sentences why the sky is blue."),
        4: chat(f"{log}\n\nList the departure times of entries 10 through 14."),
    }
    assert [len(p) > 4096 for p in prompts.values()] == [False, True, False, True]
    return prompts


def _decode(
    model, llm, logits, monkeypatch, resident: bool, steps, config: dict | None = None,
    adapters: dict[int, str] | None = None,
) -> dict:
    eng = _engine(model, llm, monkeypatch, resident, config)
    planner = eng._submodules[NODE].resident
    seen = []
    if planner is not None:
        plan = planner.plan

        def counted(step, ctx):
            seen.append(ctx.slot_lease.bucket.bs)
            plan(step, ctx)

        planner.plan = counted
    driver = _Driver(eng, logits, adapters)
    prompts = _prompts(model)
    for rid, prompt in prompts.items():
        driver.add(rid, prompt, 64)
        driver.prefill(rid, prompt)
    for n, rids in steps:
        for _ in range(n):
            driver.step(rids)
    long = set(eng._resources["rope"]._long_rids)
    pool = next((r for key, r in eng._resources.items() if key.startswith("lora")), None)
    adapted = {rid for rid in prompts if pool is not None and pool.runs_adapters([rid])}
    out = dict(
        tokens=driver.out, logits=driver.step_logits, resident_steps=len(seen), planner=planner, long=long,
        adapted=adapted,
    )
    eng.shutdown()
    del eng, driver
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _planner_logs(caplog) -> list[str]:
    return [r.message for r in caplog.records if "resident planning" in r.message]


def _assert_same(resident: dict, host: dict) -> None:
    assert host["planner"] is None and host["resident_steps"] == 0
    assert resident["tokens"] == host["tokens"]
    for t, (a, b) in enumerate(zip(resident["logits"], host["logits"], strict=True)):
        assert torch.equal(a, b), f"decode step {t}: logits differ by up to {float((a - b).abs().max())}"


def test_resident_decode_matches_the_host_plan(phi4mm, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    llm, logits = _llm(phi4mm)
    steps = [(20, [1, 2, 3, 4]), (12, [4, 1, 2])]
    resident = _decode(phi4mm, llm, logits, monkeypatch, True, steps)
    assert resident["planner"] is not None, _planner_logs(caplog)
    assert any(m.startswith(f"resident planning on {NODE}") for m in _planner_logs(caplog))
    assert resident["resident_steps"] == 32
    assert resident["long"] == {2, 4}
    _assert_same(resident, _decode(phi4mm, llm, logits, monkeypatch, False, steps))


def _lora_pool_refuses() -> bool:
    from mstar.engine.resources.lora.pool import LoraAdapterPool
    from mstar.engine.resources.resident_hooks import ResidentHooks

    return LoraAdapterPool.resident_refusal is ResidentHooks.resident_refusal


# The served config's LoRA pool keeps the decode host-planned until #1419 lands;
# the skip lifts by itself then.
@pytest.mark.skipif(_lora_pool_refuses(), reason="the LoRA pool refuses resident planning until #1419")
def test_resident_decode_with_the_served_lora_pool(phi4mm, monkeypatch, caplog):
    """The served config's LoRA pool, a short and a long row on each of the
    checkpoint's adapters beside two base rows; the middle steps run the base rows
    alone, which replays the bucket's lean twin."""
    caplog.set_level(logging.INFO)
    config = _served_config(phi4mm)
    adapters = {2: "vision", 3: "speech"}
    steps = [(12, [1, 2, 3, 4]), (8, [1, 4]), (8, [3, 1, 2])]
    runs = {}
    for resident in (True, False):
        # a fresh module per engine: the pool attaches its deltas to the layers
        llm, logits = _llm(phi4mm)
        runs[resident] = _decode(phi4mm, llm, logits, monkeypatch, resident, steps, config, adapters)
        if resident:
            assert runs[True]["planner"] is not None, _planner_logs(caplog)
            assert any(m.startswith(f"resident planning on {NODE}") for m in _planner_logs(caplog))
        del llm, logits
        gc.collect()
        torch.cuda.empty_cache()
    assert runs[True]["resident_steps"] == 28
    # rows 2 and 3 run their deltas, so steps with them replay the plain bucket
    assert runs[True]["adapted"] == runs[False]["adapted"] == {2, 3}
    _assert_same(runs[True], runs[False])
