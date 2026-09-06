import sys

sys.path.insert(0, ".")


from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.model.orpheus.config import OrpheusModelConfig
from mstar.model.orpheus.orpheus_model import OrpheusModel


def _make_model() -> OrpheusModel:
    model = object.__new__(OrpheusModel)
    model.config = OrpheusModelConfig()
    return model


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
    )

    assert result.full_metadata.graph_walk == "decode"
    assert result.step_metadata["is_prefill"] is False
    assert result.request_done is False
    assert "decode_finished" not in result.full_metadata.kwargs


def test_orpheus_decode_eos_marks_done():
    model = _make_model()
    metadata = CurrentForwardConductorMetadata(
        input_modalities=["text"],
        output_modalities=["audio"],
        graph_walk="decode",
        is_prefill=False,
        kwargs={
            "decode_finished": False,
        },
    )

    result = model.get_partition_forward_pass_args(
        partition_name="LLM",
        partition_metadata=metadata,
        persist_signals={},
    )

    assert result.request_done is True
    assert result.full_metadata.kwargs["decode_finished"] is True
