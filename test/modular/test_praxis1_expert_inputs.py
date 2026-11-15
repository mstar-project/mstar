"""The encoder-to-expert handoff, with an optional reference comparison."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from mstar.model.praxis1.components.encoder import TapFeatures
from mstar.model.praxis1.components.expert_inputs import assemble_expert_inputs


def _case(n_chunks: int) -> tuple[TapFeatures, torch.Tensor, torch.Tensor, torch.Tensor]:
    frames_per_chunk, tokens_per_frame = 2, 3
    n_noised = frames_per_chunk * n_chunks * tokens_per_frame
    n_clean = frames_per_chunk * (n_chunks - 1) * tokens_per_frame
    n_global = 5
    seq = n_global + n_noised + n_clean
    x = torch.arange(seq * 4, dtype=torch.float32).reshape(1, seq, 4)
    features = TapFeatures(
        taps=[x + 100, x + 200],
        embeddings=x,
        frames_per_chunk=frames_per_chunk,
        tokens_per_frame=tokens_per_frame,
        rope_ids=torch.arange(seq, dtype=torch.long)[None] + 1000,
        global_spans={"register": (0, 2), "anchor": (2, 1), "text": (3, 2)},
        noised_span=(n_global, n_noised),
        clean_span=(n_global + n_noised, n_clean) if n_clean else None,
        n_chunks=n_chunks,
    )
    actions = torch.arange(n_chunks * frames_per_chunk * 3).reshape(1, -1, 3).to(torch.float64)
    times = torch.arange(n_chunks).reshape(1, -1).to(torch.float32)
    proprio = torch.arange(n_chunks * 2).reshape(1, n_chunks, 2).to(torch.float64)
    return features, actions, times, proprio


@pytest.mark.parametrize("n_chunks", [1, 2, 3])
def test_expert_reads_globals_then_clean_then_noised(n_chunks: int) -> None:
    features, actions, times, proprio = _case(n_chunks)
    got = assemble_expert_inputs(features, actions, times, proprio)
    g = list(range(5))
    clean = [] if features.clean_span is None else list(
        range(features.clean_span[0], sum(features.clean_span))
    )
    noised = list(range(features.noised_span[0], sum(features.noised_span)))
    assert torch.equal(got.global_tokens, features.embeddings[:, g])
    assert torch.equal(got.trunk_features[0], features.taps[0][:, clean + noised])
    assert torch.equal(got.kv_rope_ids, features.rope_ids[:, g + clean + noised])
    assert torch.equal(got.action_timesteps, times.repeat_interleave(2, dim=1))
    assert torch.equal(got.proprio_latents, proprio.float().repeat_interleave(2, dim=1))
    assert torch.equal(got.noised_action_latents, actions.float())
    assert got.window_size == 2 * (2 * n_chunks - 1)


def test_global_order_is_configurable() -> None:
    features, actions, times, proprio = _case(2)
    got = assemble_expert_inputs(
        features, actions, times, proprio, global_order=("text", "register", "anchor")
    )
    assert torch.equal(got.global_tokens, features.embeddings[:, [3, 4, 0, 1, 2]])
    assert torch.equal(got.kv_rope_ids[:, :5], features.rope_ids[:, [3, 4, 0, 1, 2]])


@pytest.mark.parametrize("bad", ["clean", "noised", "rope", "actions", "order"])
def test_bad_window_fails_closed(bad: str) -> None:
    features, actions, times, proprio = _case(2)
    if bad == "clean":
        features = replace(features, clean_span=(features.clean_span[0], 5))
    elif bad == "noised":
        features = replace(features, noised_span=(features.noised_span[0], 11))
    elif bad == "rope":
        features = replace(features, rope_ids=features.rope_ids[:, :-1])
    elif bad == "actions":
        actions = actions[:, :-1]
    with pytest.raises(ValueError):
        assemble_expert_inputs(
            features,
            actions,
            times,
            proprio,
            global_order=("register", "text", "text") if bad == "order" else ("register", "anchor", "text"),
        )


@pytest.mark.parametrize("n_chunks", [1, 2, 3])
def test_inputs_match_reference(n_chunks: int) -> None:
    pytest.importorskip("praxis", reason="reference checkout is not on PYTHONPATH")
    from praxis.action_expert import run_expert_on_taps_k

    features, actions, times, proprio = _case(n_chunks)

    class Capture:
        global_order = ("register", "anchor", "text")

        def forward_trunk_rope(self, **kwargs):
            return kwargs

    ref = run_expert_on_taps_k(
        Capture(),
        features.taps,
        features.embeddings,
        features.rope_ids,
        global_spans=features.global_spans,
        noised_span=features.noised_span,
        clean_span=features.clean_span,
        n_chunks=features.n_chunks,
        frames_per_chunk=features.frames_per_chunk,
        tokens_per_frame=features.tokens_per_frame,
        noised_actions=actions,
        action_t_per_chunk=times,
        proprio_per_chunk=proprio,
    )
    port = asdict(assemble_expert_inputs(features, actions, times, proprio))
    assert port.keys() == ref.keys()
    for key, value in port.items():
        expected = ref[key]
        if isinstance(value, list):
            assert all(torch.equal(a, b) for a, b in zip(value, expected, strict=True))
        elif isinstance(value, torch.Tensor):
            assert torch.equal(value, expected)
        else:
            assert value == expected
