"""The shared sampler's ring gives the model's RAS tokens across steps."""

from dataclasses import fields

import torch

from mstar.communication.wire import decode, encode
from mstar.engine.resources.sampler.config import RASSamplingReqConfig, SamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.cosyvoice3.components.sampling import ras_sample_plain as old_ras_sample


def _resource(window: int) -> SamplerResource:
    return SamplerResource(
        vocab_size=None, enable_repetition_penalty=False, device=torch.device("cpu"),
        max_repetition_window=window, enable_blocked_tokens=True, enable_ras=True,
    )


def _config(seed: int, window: int, tau: float, temperature: float, top_p: float) -> RASSamplingReqConfig:
    cfg = RASSamplingReqConfig(
        temperature=temperature, top_k=7, top_p=top_p,
        repetition_window=window, ras_mode=True, ras_tau=tau,
    )
    cfg.apply_conductor_config(seed=seed, resource_key="lm_sampler")
    return cfg


def test_ras_request_settings_are_model_scoped_and_cross_typed_wire():
    public_fields = {field.name for field in fields(SamplingReqConfig)}
    assert "ras_mode" not in public_fields and "ras_tau" not in public_fields
    cfg = _config(7, 25, 0.6, 1.0, 0.0)
    decoded = decode(encode(cfg))
    assert isinstance(decoded, RASSamplingReqConfig)
    assert decoded == cfg and decoded.ras_mode and decoded.ras_tau == 0.6


def test_ras_resource_matches_model_draws_across_mixed_batch_steps():
    width = 4
    rows = {
        "a": _config(123456789, 4, 0.25, 1.0, 0.8),
        "b": _config(987654321, 2, 0.5, 0.7, 0.0),
        "c": _config(42, 0, 0.0, 0.0, 0.9),
    }
    resource = _resource(width)
    for rid, cfg in rows.items():
        resource.ingest_request(rid, cfg)
    history = {rid: torch.full((width,), -1, dtype=torch.long) for rid in rows}
    counts = dict.fromkeys(rows, 0)
    generator = torch.Generator().manual_seed(345)

    for batch in (["a"], ["a", "b"], ["b", "a", "c"], ["c", "b", "a"], ["a", "c"]):
        logits = torch.randn(len(batch), 11, generator=generator) * 2
        blocked = torch.zeros_like(logits, dtype=torch.bool)
        # A forced EOS row keeps only one id; the model does this at prefill's
        # terminal token. Other rows exercise the minimum-length mask.
        if "c" in batch:
            blocked[batch.index("c"), :] = True
            blocked[batch.index("c"), 9] = False
        if "b" in batch:
            blocked[batch.index("b"), 10] = True
        legacy = old_ras_sample(
            logits, seed=torch.tensor([rows[rid].seed for rid in batch]),
            step=torch.tensor([counts[rid] for rid in batch]),
            history=torch.stack([history[rid] for rid in batch]),
            temperature=torch.tensor([rows[rid].temperature for rid in batch]),
            top_p=torch.tensor([rows[rid].top_p for rid in batch]),
            top_k=torch.tensor([rows[rid].top_k for rid in batch]),
            window=torch.tensor([rows[rid].repetition_window for rid in batch]),
            repeat_threshold=torch.tensor([
                rows[rid].repetition_window * rows[rid].ras_tau for rid in batch
            ]),
            blocked=blocked,
        )
        actual = resource.sample(batch, logits, blocked)
        assert torch.equal(actual, legacy)
        for rid, token in zip(batch, legacy, strict=True):
            history[rid] = torch.cat((history[rid][1:], token.reshape(1)))
            counts[rid] += 1

    assert resource._history is not None
    assert all(resource._history._rid_to_slot[rid] > 0 for rid in rows)


def test_ras_resource_keeps_a_requests_tokens_across_batch_sizes():
    only, mixed = _resource(3), _resource(3)
    for resource in (only, mixed):
        resource.ingest_request("a", _config(7, 3, 0.34, 1.0, 0.8))
    mixed.ingest_request("b", _config(8, 3, 0.34, 1.0, 0.8))
    generator = torch.Generator().manual_seed(891)
    for _ in range(6):
        a_logits = torch.randn(1, 17, generator=generator)
        b_logits = torch.randn(1, 17, generator=generator)
        a = only.sample(["a"], a_logits)
        both = mixed.sample(["b", "a"], torch.cat((b_logits, a_logits)))
        assert a.item() == both[1].item()


def test_ras_threshold_keeps_python_product_rounding():
    # The model computed float(window) * float(tau), then staged one float32.
    # Multiplying a float32 window by a float32 tau rounds 25 * 0.6 slightly
    # above 15 and misses the redraw at exactly 15 repeated tokens.
    eager, captured = _resource(25), _resource(25)
    captured.build_accelerator_graph_buffers(slots=[], max_bs=1, max_seq_len=0)
    for resource in (eager, captured):
        resource.ingest_request("a", _config(7, 25, 0.6, 1.0, 0.0))
    buffers = captured._cg_buffers
    assert buffers is not None
    logits = torch.tensor([[5.0, 0.0]])
    for step in range(16):
        expected = eager.sample(["a"], logits)
        buffers.gather_static(["a"], 1, 0)
        buffers.gather_dynamic(["a"], 1, 0)
        assert captured._history is not None
        captured._history.gather_step(["a"], 1)
        captured._cg_sampler = buffers.sampler_for(1, 0)
        actual = captured.sample(["a"], logits)
        captured._history.scatter_step()
        captured._cg_sampler = None
        assert actual.item() == expected.item() == (1 if step == 15 else 0)


def test_ras_resource_eager_and_captured_row_emulation_match():
    eager, captured = _resource(4), _resource(4)
    captured.build_accelerator_graph_buffers(slots=[], max_bs=3, max_seq_len=0)
    for resource in (eager, captured):
        resource.ingest_request("a", _config(100, 4, 0.25, 1.0, 0.8))
        resource.ingest_request("b", _config(200, 2, 0.5, 0.0, 0.0))
    buffers = captured._cg_buffers
    assert buffers is not None
    generator = torch.Generator().manual_seed(125)
    for batch in (["a"], ["b", "a"], ["a", "b"], ["b"]):
        logits = torch.randn(len(batch), 15, generator=generator)
        blocked = torch.zeros_like(logits, dtype=torch.bool)
        if "b" in batch:
            blocked[batch.index("b"), 14] = True
        expected = eager.sample(batch, logits, blocked)
        buffers.gather_static(batch, 3, 0)
        buffers.gather_dynamic(batch, 3, 0)
        assert captured._history is not None
        captured._history.gather_step(batch, 3)
        captured._cg_sampler = buffers.sampler_for(3, 0)
        padded_logits = torch.cat((logits, torch.zeros(3 - len(batch), 15)))
        padded_blocked = torch.cat((blocked, torch.zeros(3 - len(batch), 15, dtype=torch.bool)))
        actual = captured.sample(batch, padded_logits, padded_blocked)
        captured._history.scatter_step()
        captured._cg_sampler = None
        assert torch.equal(actual[:len(batch)], expected)
