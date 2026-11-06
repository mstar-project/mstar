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


def test_no_checkpoint_download_fetches_the_training_state(monkeypatch):
    """The Orpheus repos ship ~72 GB of optimizer and FSDP state next to the
    weights (#681): the tokenizer, checkpoint_path and the LLM load each fetch
    only what they read."""
    from fnmatch import fnmatch

    import huggingface_hub

    from mstar.distributed.communication import CommGroup
    from mstar.model import loader
    from mstar.model.orpheus import orpheus_model

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kw: calls.append(kw) or "/snap")
    monkeypatch.setattr(orpheus_model.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(loader, "load_weights", lambda *a, **k: None)
    model = OrpheusModel("canopylabs/orpheus-3b-0.1-ft")
    model.checkpoint_path()
    model._create_llm_submodule("meta", tp_group=CommGroup.trivial())

    def fetched(name, call):
        return any(fnmatch(name, p) for p in call["allow_patterns"] or ["*"])

    assert len(calls) == 3
    for call in calls:
        skipped = ["optimizer.bin", "pytorch_model_fsdp.bin", "rng_state_0.pth", "scheduler.pt", "training_args.bin"]
        assert not any(fetched(f, call) for f in skipped), call
        assert fetched("tokenizer.json", call), call
    for call in calls[1:]:
        assert all(fetched(f, call) for f in ["config.json", "model.safetensors.index.json",
                                               "model-00001-of-00004.safetensors"]), call
