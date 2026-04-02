import sys

sys.path.insert(0, ".")


from mminf.model.base import CurrentForwardMetadata
from mminf.model.orpheus.config import OrpheusModelConfig
from mminf.model.orpheus.orpheus_model import OrpheusModel


def _make_model() -> OrpheusModel:
    model = object.__new__(OrpheusModel)
    model.config = OrpheusModelConfig()
    return model


def _audio_token_for_pos(pos: int, code: int = 1) -> int:
    return 10 + (pos * 4096) + code


def test_orpheus_prefill_transitions_to_decode():
    model = _make_model()
    metadata = CurrentForwardMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="prefill",
        is_prefill=True,
    )

    result = model.get_forward_pass_args(
        metadata=metadata,
        persist_signals={"new_token": []},
        new_tokens={},
    )

    assert result.full_metadata.graph_walk == "decode"
    assert result.step_metadata == {"is_prefill": False}
    assert result.full_metadata.kwargs["audio_token_buffer"] == []
    assert result.full_metadata.kwargs["audio_token_count"] == 0
    assert result.full_metadata.kwargs["decode_finished"] is False


def test_orpheus_decode_to_audio_gen_on_stop_token():
    model = _make_model()
    metadata = CurrentForwardMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="decode",
        is_prefill=False,
        kwargs={
            "audio_token_buffer": [],
            "audio_token_count": 0,
            "decode_finished": False,
        },
    )
    tokens = [_audio_token_for_pos(i) for i in range(7)] + [model.config.stop_token_id]

    result = model.get_forward_pass_args(
        metadata=metadata,
        persist_signals={},
        new_tokens={"new_token": tokens},
    )

    assert result.request_done is False
    assert result.full_metadata.graph_walk == "audio_gen"
    assert result.step_metadata["audio_token_ids"] == [1] * 7
    assert result.full_metadata.kwargs["audio_token_count"] == 7
    assert result.full_metadata.kwargs["decode_finished"] is True


def test_orpheus_audio_gen_completes_request():
    model = _make_model()
    metadata = CurrentForwardMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="audio_gen",
        is_prefill=False,
        kwargs={
            "audio_token_buffer": [1] * 7,
            "audio_token_count": 7,
            "decode_finished": True,
        },
    )

    result = model.get_forward_pass_args(
        metadata=metadata,
        persist_signals={},
        new_tokens={},
    )

    assert result.request_done is True
    assert result.inputs == []
