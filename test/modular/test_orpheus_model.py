import sys

sys.path.insert(0, ".")


from mminf.conductor.request_info import CurrentForwardConductorMetadata
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
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="prefill",
        is_prefill=True,
    )

    result = model.get_partition_forward_pass_args(
        partition_name="LLM",
        partition_metadata=metadata,
        persist_signals={"new_token": []},
        new_tokens={},
    )

    assert result.full_metadata.graph_walk == "decode"
    assert result.step_metadata["is_prefill"] is False
    assert result.full_metadata.kwargs["audio_token_count"] == 0
    assert result.full_metadata.kwargs["decode_finished"] is False


def test_orpheus_decode_eos_marks_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="decode",
        is_prefill=False,
        kwargs={
            "audio_token_count": 0,
            "decode_finished": False,
        },
    )

    result = model.get_partition_forward_pass_args(
        partition_name="LLM",
        partition_metadata=metadata,
        persist_signals={},
        new_tokens={"new_token": [model.config.stop_token_id]},
    )

    assert result.request_done is True
    assert result.full_metadata.kwargs["decode_finished"] is True
