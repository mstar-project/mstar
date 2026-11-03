"""The captured backbone path's attention layout. No GPU, no weights: each
request owns two documents (a zero-length unconditional one with CFG off), and
the step spreads them over its ids. Its numerics against the reference are in
``test_omnivoice_parity.py``. Also code2wav's compiled codec decoder, whose
warm-up covers every CPU canvas-length range of three frames or more, and which
on CUDA decodes as the eager one does.
"""

from types import SimpleNamespace

import pytest

from mstar.model.omnivoice.submodules import (
    BACKBONE_ATTN,
    OmniVoiceBackboneSubmodule,
    document_pairs,
)


def test_document_pairs_keep_a_slot_per_unguided_request():
    assert document_pairs([(5, 3), (4, 0)]) == [5, 3, 4, 0]


def _spans(step):
    return [(seg.request_id, seg.span) for seg in step.segments]


def test_attention_step_two_documents_per_request():
    step = OmniVoiceBackboneSubmodule._attention_step(["a", "b"], [5, 3, 4, 0])
    assert _spans(step) == [("a", 5), ("a", 3), ("b", 4), ("b", 0)]
    assert not step.steps[BACKBONE_ATTN].causal


def test_attention_step_padded_replay_and_capture_layouts():
    # replay: two documents per real request, one zero per padding row
    replay = OmniVoiceBackboneSubmodule._attention_step(["a", "b", "p0", "p1"], [5, 3, 4, 0, 0, 0])
    assert [n for _, n in _spans(replay)] == [5, 3, 4, 0, 0, 0]
    assert [rid for rid, _ in _spans(replay)][:4] == ["a", "a", "b", "b"]
    # capture: one length per row
    capture = OmniVoiceBackboneSubmodule._attention_step(["p0", "p1"], [8, 8])
    assert _spans(capture) == [("p0", 8), ("p1", 8)]


def test_node_resources_do_not_follow_backbone_graphs():
    from mstar.model.omnivoice.omnivoice_model import OmniVoiceModel

    on = OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True)
    off = OmniVoiceModel("k2-fsa/OmniVoice", skip_weight_loading=True, backbone_graphs=False)
    # the eager layer loop attends through the same resource: the flag gates capture only
    for model in (on, off):
        (spec,) = model.get_node_resources()
        assert spec.resource_key == BACKBONE_ATTN and spec.config.max_segments_per_request == 2


def test_forward_batched_keys_rows_by_the_engine_ids():
    """The engine reads a row's outputs back by the worker's handle for the
    request, which is not the request's wire id."""
    import torch

    from mstar.model.omnivoice.config import OmniVoiceConfig

    cfg = OmniVoiceConfig()
    sub = OmniVoiceBackboneSubmodule(backbone=None, config=cfg)
    sub.get_device = lambda: torch.device("cpu")
    c = cfg.num_audio_codebook
    meta = {
        "target_len": 4, "num_step": 2, "guidance_scale": 2.0, "t_shift": 0.1, "class_temperature": 0.0,
        "layer_penalty_factor": 5.0, "position_temperature": 5.0,
    }
    inputs = {
        "prefix_ids": [torch.zeros(c, 3, dtype=torch.long)],
        "prefix_audio_mask": [torch.zeros(3, dtype=torch.bool)],
    }
    infos = {
        h: SimpleNamespace(request_id=f"speech-{h}#1", rid_handle=h, random_seed=h, step_metadata=meta,
                           dynamic_loop_iter_counts={})
        for h in (0, 1)
    }
    rows = [sub.prepare_inputs("speech_gen", info, inputs) for info in infos.values()]
    engine_inputs = SimpleNamespace(request_ids=[0, 1], per_request_info=infos)
    pre = sub.preprocess("speech_gen", engine_inputs, rows)
    rows_out = pre["canvas"].tgt_index.shape[0]
    sub._logits = lambda _inputs, _canvas: torch.zeros(1, c, rows_out, cfg.audio_vocab_size)
    assert set(sub.forward_batched("speech_gen", engine_inputs, **pre)) == {0, 1}


def _codec(decoder_hidden_size: int):
    """A HiggsAudioV2 codec with random weights and the checkpoint's decoder
    geometry at ``decoder_hidden_size`` (k2-fsa/OmniVoice ``audio_tokenizer/config.json``
    has 1024); the encoders, which decode never runs, are shrunk."""
    pytest.importorskip("transformers.models.higgs_audio_v2_tokenizer")
    import torch
    from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

    ratios = [8, 5, 4, 2, 3]
    torch.manual_seed(0)
    return HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(
        acoustic_model_config={
            "model_type": "dac", "upsampling_ratios": ratios, "downsampling_ratios": ratios,
            "hop_length": 960, "decoder_hidden_size": decoder_hidden_size, "hidden_size": 256,
            "encoder_hidden_size": 2,
        },
        semantic_model_config={
            "model_type": "hubert", "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
            "intermediate_size": 16, "conv_dim": [8] * 7, "num_conv_pos_embeddings": 16,
            "num_conv_pos_embedding_groups": 2,
        },
        codebook_dim=8,
    )).eval().requires_grad_(False)


def test_the_codec_decoder_compiles_at_start_up_not_on_a_request(monkeypatch):
    """code2wav's warm-up builds the compiled acoustic decoder's graphs for
    very short, typical and 10 s canvases, so none of them compiles on the GPU
    thread mid-serving. This pins the CPU ranges as a proxy for the start-up
    warm-up: it forces ``is_available`` on a CPU engine and counts Dynamo
    frames, so neither Inductor's guards nor CUDA's ranges are covered. The
    CPU ranges come from MKLDNN's numel heuristic, which depends on the conv
    widths, so the decoder has the checkpoint's geometry (k2-fsa/OmniVoice
    ``audio_tokenizer/config.json``); the encoders, which decode never runs,
    are shrunk."""
    import torch
    from torch._dynamo.testing import CompileCounter

    from mstar.engine.engine import Engine
    from mstar.model.omnivoice.components.codec import decode_canvas
    from mstar.model.omnivoice.config import OmniVoiceConfig
    from mstar.model.omnivoice.submodules import OmniVoiceCode2WavSubmodule

    codec = _codec(decoder_hidden_size=1024)

    engine = Engine.__new__(Engine)
    engine._device = torch.device("cpu")
    engine._device_capability = SimpleNamespace(is_available=True)
    code2wav = OmniVoiceCode2WavSubmodule(codec, OmniVoiceConfig())
    engine._submodules = {"code2wav": SimpleNamespace(submodule=code2wav, fp32_precision=None, node_dtype=None,
                                                      lora_pools=())}
    counter, compile_ = CompileCounter(), torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compile_(fn, backend=counter, **kwargs))
    torch._dynamo.reset()
    engine._compile_regions()
    assert counter.frame_count > 0

    compiled_at = []
    with torch.no_grad():   # as the engine runs a forward
        for frames in (3, 6, 30, 250):
            before = counter.frame_count
            decode_canvas(codec, torch.randint(0, 1024, (len(codec.quantizer.quantizers), frames)))
            if counter.frame_count > before:
                compiled_at.append(frames)
    assert compiled_at == []


def test_the_compiled_codec_decoder_matches_eager_on_cuda():
    """code2wav compiles the acoustic decoder for dynamic canvas lengths
    (``compile_regions``), so Inductor traces its Snake activations, conv views
    and residual adds; its audio must match the eager decode, under the node's
    TF32-off precision."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("the compiled codec decoder runs on CUDA")
    from mstar.model.omnivoice.components.codec import decode_canvas

    codec = _codec(decoder_hidden_size=128).cuda()
    canvases = [torch.randint(0, 1024, (len(codec.quantizer.quantizers), n), device="cuda") for n in (5, 40)]
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        eager = [decode_canvas(codec, c) for c in canvases]
        decoder = codec.acoustic_decoder
        decoder.forward = torch.compile(decoder.forward, dynamic=True)
        got = [decode_canvas(codec, c) for c in canvases]
    # Inductor reorders and fuses the decoder's reductions, so fp32 rounding differs by
    # ~1e-4; 1e-3 as the other compiled-vs-eager checks (e.g. test_qwen3_omni_fused_moe)
    for a, b in zip(got, eager, strict=True):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
