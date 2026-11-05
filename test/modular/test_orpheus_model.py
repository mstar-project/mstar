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


def test_an_llm_run_of_four_tokens_equals_four_one_token_steps():
    """Decode reads its token off the sampler's slot masters (``DeviceLoopback``),
    and a run carries it from the sampled ``new_token`` into the next
    replay's ``text_inputs``: four replays give the tokens four steps do. The
    transformer is a stand-in; the embedding, head and plumbing are real."""
    from types import SimpleNamespace

    import torch
    from run_parity import replay_run

    from mstar.model.orpheus.submodules import ATTN, SAMPLER, OrpheusLLMSubmodule
    from mstar.model.submodule_base import CarryPair, ModelInputsFromEngine

    torch.manual_seed(0)
    vocab, hidden = 32, 8

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(vocab, hidden)
            self.lm_head = torch.nn.Linear(hidden, vocab)

        def forward(self, x, label):
            return torch.tanh(x * 1.5)

    llm = OrpheusLLMSubmodule(LanguageModel(), SimpleNamespace(stop_token_id=vocab - 1, custom_token_base_id=0))
    assert llm.repeat_carry("decode", {"text_inputs"}) == {"text_inputs": CarryPair("new_token", "text_inputs")}
    sampler = SimpleNamespace(sample=lambda rids, logits: logits.argmax(-1))
    engine_inputs = ModelInputsFromEngine(
        request_ids=["a", "b"], per_request_info={}, resources={SAMPLER: sampler, ATTN: None},
    )

    def forward(**inputs):
        return llm.forward_batched("decode", engine_inputs, **inputs).row_outputs

    first = torch.tensor([3, 5])
    tokens, steps = first, []
    with torch.no_grad():
        for _ in range(4):
            tokens = forward(text_inputs=tokens)["new_token"]
            steps.append(tokens)
        _, columns = replay_run(
            forward, {"text_inputs": first}, llm.repeat_carry("decode", {"text_inputs"}), 4, ["new_token"],
        )
    assert torch.equal(columns["new_token"], torch.stack(steps, dim=1))


def test_a_mixed_step_reads_a_loop_back_rows_token_off_the_rows_beside_a_prompt():
    import torch

    from mstar.engine.resources.loopback_rows import LoopbackRows
    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
    from mstar.model.submodule_base import ARNodeInputs, DeviceLoopback, ModelInputsFromEngine

    llm = OrpheusLLMSubmodule.__new__(OrpheusLLMSubmodule)
    llm.device_loopback = DeviceLoopback(walk="decode", signal="text_inputs", buffer="new_token", input="text_inputs")
    kept = LoopbackRows(torch.device("cpu"))
    for rid in (1, 2):
        kept.add(rid)
    kept.write("text_inputs", [1, 2], torch.tensor([41, 42]))
    engine_inputs = ModelInputsFromEngine(request_ids=[1, 2], per_request_info={}, loopback_rows=kept)
    rows = [ARNodeInputs(input_seq_len=1), ARNodeInputs(input_ids=torch.tensor([7, 8, 9]), input_seq_len=3)]
    assert llm.preprocess("mixed", engine_inputs, rows)["text_inputs"].tolist() == [41, 7, 8, 9]


def test_the_llm_feeds_back_a_runs_last_token():
    """After a run a row's new_token holds a token a replay; its next step reads one."""
    import torch

    from mstar.model.orpheus.submodules import OrpheusLLMSubmodule

    llm = OrpheusLLMSubmodule.__new__(OrpheusLLMSubmodule)
    for tokens, fed in ((torch.tensor([5]), [5]), (torch.tensor([[5, 6, 7]]), [7])):
        outputs = {"new_token": [tokens]}
        llm.postprocess("a", None, outputs)
        assert outputs["text_inputs"][0].tolist() == fed
