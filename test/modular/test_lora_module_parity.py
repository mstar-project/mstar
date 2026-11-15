"""``benchmark/lora/module_parity.py``'s calls for a node on the state contract, on CPU over the LoRA harness's tiny
nodes: the benchmark itself needs a GPU and real weights, and CI runs neither."""

import pytest
import torch

pytest.importorskip("safetensors")
import lora_module_helpers as H  # noqa: E402

from benchmark.lora import module_parity  # noqa: E402

LENGTHS = (5, 3)


def _talker(config, weights):
    """The Talker node as the model builds it for the benchmark: weights loaded, no handles bound."""
    from mstar.model.components.code_predictor_kernels import ResidualCodePredictor
    from mstar.model.qwen3_tts.components.talker import Qwen3TTSTalkerModel
    from mstar.model.qwen3_tts.submodules import TalkerSubmodule

    predictor = ResidualCodePredictor(
        config.code_predictor, config.talker.hidden_size, TalkerSubmodule.MAX_BATCH_SIZE, embed_sum_dtype=None,
    )
    return TalkerSubmodule(H._loaded(Qwen3TTSTalkerModel(config).to(H.DTYPE), weights), predictor.to(H.DTYPE), config)


def _higgs(config, weights):
    """Higgs's LLM node as the model builds it for the benchmark: weights loaded, no handles bound."""
    from mstar.model.higgs_audio.higgs_audio_model import HiggsAudioModel, audio_encoder
    from mstar.model.higgs_audio.submodules import HiggsAudioLLMSubmodule

    model = H.model_object(HiggsAudioModel, config)
    llm = model.build_llm(None).to(H.DTYPE)
    llm.load_weights((name, w) for k, w in weights.items() if (name := model._llm_remap(k)) is not None)
    return HiggsAudioLLMSubmodule(llm.eval(), config, (audio_encoder(),), text_prompts=False)


def _thinker(config, weights):
    """The Thinker node as the model builds it for the benchmark: weights loaded, no handles bound."""
    from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    return ThinkerSubmodule(H._loaded(Qwen3OmniThinkerModel(config).to(H.DTYPE), weights), config)


@pytest.mark.parametrize("model, stack, build", [
    ("qwen3_tts", "qwen3_tts_talker", _talker), ("qwen3_omni", "qwen3_omni_thinker", _thinker),
    ("higgs", "higgs_audio", _higgs),
])
def test_a_contract_nodes_entry_declares_and_runs_its_step(model, stack, build):
    entry, harness = getattr(module_parity, f"_{model}")(), H.STACKS[stack]()
    config = harness.config()
    sub = build(config, harness.weights(config))
    ids = [torch.randint(1, 30, (n,), generator=torch.Generator().manual_seed(n)) for n in LENGTHS]
    assert entry.declare(sub, [0, 1], ids).segments is not None
    module_parity._set_positions(LENGTHS)
    with torch.no_grad():
        plain = entry.run(sub, torch.cat(ids), list(LENGTHS))
        rotated = entry.run(sub, torch.cat(ids), list(LENGTHS), rope=lambda q, k: (-q, k))
    assert plain.shape[0] == len(LENGTHS)
    assert not torch.equal(plain, rotated), "the stack rotates its queries and keys when given a rope"
