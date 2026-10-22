"""Kokoro's chunk loop as the engine drives it: one sentence chunk per
iteration, until ``check_stop`` ends the loop after the request's last chunk.

The engine keys a batch by the worker's integer rid handle, not by the
request id string, so ``prepare_inputs`` and ``check_stop`` only agree on a
request's chunk count when its per-request state is keyed by that handle.
CPU only: tiny random weights, no checkpoint and no G2P.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import StepContext  # noqa: E402
from mstar.model.kokoro.components import KokoroTTS  # noqa: E402
from mstar.model.kokoro.config import (  # noqa: E402
    AUDIO_CHUNK,
    CHUNK_LOOP,
    PHONEME_IDS,
    PHONEME_LENS,
    REF_STYLE,
    SPEED,
    SYNTH_NODE,
    SYNTH_WALK,
    KokoroBertConfig,
    KokoroISTFTNetConfig,
    KokoroModelConfig,
)
from mstar.model.kokoro.submodules import KokoroSynthSubmodule  # noqa: E402
from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine  # noqa: E402


def _tiny_config() -> KokoroModelConfig:
    return KokoroModelConfig(
        n_token=16,
        hidden_dim=16,
        style_dim=8,
        n_layer=1,
        max_dur=6,
        plbert=KokoroBertConfig(
            hidden_size=16, num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
            num_hidden_layers=2, embedding_size=8,
        ),
        istftnet=KokoroISTFTNetConfig(
            upsample_kernel_sizes=[4, 8], upsample_rates=[2, 4], gen_istft_hop_size=5, gen_istft_n_fft=20,
            resblock_dilation_sizes=[[1, 3], [1, 3]], resblock_kernel_sizes=[3, 5], upsample_initial_channel=16,
        ),
        decoder_hidden=24,
        asr_res_dim=4,
        max_chunks=8,
        compile_decoder=False,
    )


def _submodule() -> KokoroSynthSubmodule:
    config = _tiny_config()
    torch.manual_seed(0)
    model = KokoroTTS(config).eval()
    model.decoder.generator.m_source.deterministic = True
    return KokoroSynthSubmodule(model, config)


def _engine(submodule: KokoroSynthSubmodule) -> Engine:
    engine = Engine.__new__(Engine)
    engine._submodules = {SYNTH_NODE: type("Mgmt", (), {"submodule": submodule, "resources": {}})()}
    engine._enable_nvtx = False
    engine._loopback_steps = {}
    engine._chunking_policies = {}
    engine._combined_walk_of = {}
    return engine


def _prompt_tensors(lengths: list[int], seed: int) -> dict[str, list[torch.Tensor]]:
    """What ``process_prompt`` hands the node: every chunk at once, padded."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for row, n in enumerate(lengths):
        ids[row, 1 : n - 1] = torch.randint(1, 16, (n - 2,), generator=generator)
    return {
        PHONEME_IDS: [ids],
        PHONEME_LENS: [torch.tensor(lengths)],
        REF_STYLE: [torch.randn(len(lengths), 16, generator=generator)],
        SPEED: [torch.tensor([1.0])],
    }


def test_every_chunk_is_synthesized_before_the_loop_stops():
    submodule = _submodule()
    engine = _engine(submodule)
    # rid handle -> (request id, chunk lengths); the handles differ from the
    # strings, as they do on a worker
    requests = {3: ("speech-two", [5, 7]), 5: ("speech-three", [4, 6, 5])}
    infos = {
        handle: CurrentForwardPassInfo(
            request_id=rid, graph_walk=SYNTH_WALK, fwd_index=0, random_seed=0, max_tokens=0, rid_handle=handle,
        )
        for handle, (rid, _) in requests.items()
    }
    tensors = {handle: _prompt_tensors(lengths, seed=handle) for handle, (_, lengths) in requests.items()}

    prepared: dict[int, list[int]] = {handle: [] for handle in requests}
    stopped_at: dict[int, int] = {}
    live = list(requests)
    for iteration in range(max(len(lengths) for _, lengths in requests.values()) + 1):
        if not live:
            break
        batch = ExecutingBatch(
            node_name=SYNTH_NODE,
            per_request_info={handle: infos[handle] for handle in live},
            step_context=StepContext(request_ids=tuple(live), graph_walk=SYNTH_WALK, slot=0, capture=False),
            per_request_input_tensors={handle: tensors[handle] for handle in live},
            # the worker's copy of the loop counter for this step; empty on the first
            per_request_input_metadata={
                handle: InputMetadata(dynamic_loop_iter_counts={CHUNK_LOOP: iteration} if iteration else {})
                for handle in live
            },
        )
        engine.prepare_inputs(batch)
        assert list(batch.request_ids) == live, "no request may run out of chunks before its loop stops"
        for handle, node_inputs in zip(batch.request_ids, batch.inputs, strict=True):
            prepared[handle].append(node_inputs.input_seq_len)

        engine_inputs = ModelInputsFromEngine(request_ids=list(batch.request_ids), per_request_info={})
        with torch.no_grad():
            outputs = submodule.forward_batched(
                SYNTH_WALK, engine_inputs, **submodule.preprocess(SYNTH_WALK, engine_inputs, batch.inputs)
            )
        assert all(out[AUDIO_CHUNK][0].numel() > 0 for out in outputs.values())

        stops = engine.check_stop_for_batch(batch, outputs)
        assert not batch.failed_requests
        for handle, loops in stops.items():
            assert loops == {CHUNK_LOOP}
            stopped_at[handle] = iteration
        live = [handle for handle in live if handle not in stops]

    for handle, (_, lengths) in requests.items():
        assert prepared[handle] == lengths, f"{requests[handle][0]} synthesized {prepared[handle]} of {lengths}"
        assert stopped_at[handle] == len(lengths) - 1

    # the engine removes a request by its handle, which must take all its state
    for handle in requests:
        submodule.cleanup_request(handle)
    assert submodule.request_states == {}
