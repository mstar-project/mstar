"""GR00T N1.7's state encoder: the per-embodiment MLP that turns a robot's raw state into
one conditioning token, checked against an inline reference that runs one row at a time,
and the ``state_encoder`` node that drives it."""

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.gr00t_n17.components.embodiment import CategorySpecificMLP, build_state_encoder
from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.submodules import Gr00tN17StateEncoderSubmodule
from mstar.model.submodule_base import ModelInputsFromEngine

_CATEGORIES = 4
_INPUT, _HIDDEN, _OUTPUT = 5, 7, 3


def _config(history: int = 1) -> Gr00tN17Config:
    """Input, hidden and output are three different sizes, so swapped layers fail on shape."""
    return Gr00tN17Config(
        text=None, vision=None, dit=None, bridge=None,
        deepstack_visual_indexes=(), backbone_embedding_dim=0,
        max_state_dim=_INPUT, hidden_size=_HIDDEN, input_embedding_dim=_OUTPUT,
        max_num_embodiments=_CATEGORIES, state_history_length=history,
    )


def _small_integers(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    """Whole numbers in [-8, 8]: every sum stays exact in fp32, so ``bmm`` and the
    reference's 2-D ``@`` agree under ``torch.equal`` whatever kernel each takes."""
    return torch.randint(-8, 9, shape, generator=generator).float()


def _mlp(seed: int = 0, history: int = 1) -> CategorySpecificMLP:
    """``CategorySpecificLinear`` allocates with ``torch.empty``, so every parameter is filled."""
    mlp = build_state_encoder(_config(history))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.copy_(_small_integers(parameter.shape, generator))
    return mlp


def _state(batch: int, history: int = 1, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return _small_integers(torch.Size((batch, history, _INPUT)), generator)


def _reference(
    mlp: CategorySpecificMLP, x: torch.Tensor, cat_ids: torch.Tensor, relu: bool = True,
) -> torch.Tensor:
    """Row ``i`` runs ``x @ W[id] + b[id]``, a ReLU, the same again, with ``id = cat_ids[i]``."""
    rows = []
    for row, cat in zip(x, cat_ids.tolist(), strict=True):
        hidden = row @ mlp.layer1.W[cat] + mlp.layer1.b[cat]
        if relu:
            hidden = torch.relu(hidden)
        rows.append(hidden @ mlp.layer2.W[cat] + mlp.layer2.b[cat])
    return torch.stack(rows)


_IDS = {
    "one embodiment": [2],
    # out of order, repeated, and spanning both ends
    "a batch across embodiments": [3, 0, 1, 3],
}


@pytest.mark.parametrize("history", [1, 4], ids=["one state", "a state history"])
@pytest.mark.parametrize("ids", list(_IDS.values()), ids=list(_IDS))
@torch.no_grad()
def test_the_mlp_is_two_per_embodiment_linears_with_a_relu_between(ids, history):
    """The node always sends one token, but the MLP is NVIDIA's and takes any ``T``: the
    axis the bias broadcasts over, not along."""
    mlp = _mlp()
    cat_ids = torch.tensor(ids)
    x = _state(len(ids), history)
    out = mlp(x, cat_ids)

    assert out.shape == (len(ids), history, _OUTPUT)
    assert torch.equal(out, _reference(mlp, x, cat_ids))
    # the negative control: without the ReLU the reference is a different function
    assert not torch.equal(out, _reference(mlp, x, cat_ids, relu=False))
    if len(ids) > 1:  # rows 0 and 3 share an embodiment and differ only in state: not collapsed
        assert not torch.equal(out[0], out[3])


@pytest.mark.parametrize("history", [1, 3])
def test_the_state_encoder_is_sized_from_the_config(history):
    """``hidden_size`` is the layer between and ``input_embedding_dim`` the output (1024 and
    1536 in the checkpoint); a swap would only show on real weights, which CI lacks. A
    history widens the input, as the reference's ``max_state_dim * state_history_length``."""
    encoder = build_state_encoder(_config(history))
    assert encoder.layer1.W.shape == (_CATEGORIES, history * _INPUT, _HIDDEN)
    assert encoder.layer1.b.shape == (_CATEGORIES, _HIDDEN)
    assert encoder.layer2.W.shape == (_CATEGORIES, _HIDDEN, _OUTPUT)
    assert encoder.layer2.b.shape == (_CATEGORIES, _OUTPUT)


# ----------------------------------------------------------------------------
# The node: `Gr00tN17StateEncoderSubmodule`, driven the way the engine drives it
# ----------------------------------------------------------------------------

_WALK = "prefill"
_FWD = CurrentForwardPassInfo(request_id="r0", graph_walk=_WALK, fwd_index=0, random_seed=0, max_tokens=0)
_ENGINE = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})
_EMBODIMENT = 3

# how the walk may carry one request's state -> the history length it holds
_WALK_STATES = {
    "a bare state vector": ((_INPUT,), 1),
    "a [1, dim] row": ((1, _INPUT), 1),
    "a state history": ((3, _INPUT), 3),
    "a history already batched": ((1, 3, _INPUT), 3),
}


def _node(history: int = 1) -> Gr00tN17StateEncoderSubmodule:
    return Gr00tN17StateEncoderSubmodule(_mlp(history=history))


def _walk_inputs(shape: tuple[int, ...], seed: int = 5) -> dict[str, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "state": [_small_integers(torch.Size(shape), generator)],
        "embodiment_id": [torch.tensor(_EMBODIMENT)],
    }


@pytest.mark.parametrize("shape, history", list(_WALK_STATES.values()), ids=list(_WALK_STATES))
@torch.no_grad()
def test_driving_the_node_is_the_mlp_called_directly(shape, history):
    """prepare_inputs -> preprocess -> forward is the MLP on the same inputs, the same ops
    in the same order (so ``torch.equal``, not a bar), under the key the denoise loop reads.
    The history folds as the reference's ``state.view(B, 1, -1)``: one token whatever ``T``,
    which is what the denoise step concatenates."""
    node = _node(history)
    walk = _walk_inputs(shape)
    prepared = node.prepare_inputs(_WALK, _FWD, walk)
    folded = prepared.tensor_inputs

    assert set(folded) == {"state", "embodiment_id"}
    assert folded["state"].shape == (1, 1, history * _INPUT)
    assert torch.equal(folded["state"].reshape(-1), walk["state"][0].reshape(-1))
    assert folded["embodiment_id"].shape == (1,)
    assert folded["embodiment_id"].tolist() == [_EMBODIMENT]
    assert prepared.input_seq_len == 1

    tensors = node.preprocess(_WALK, _ENGINE, [prepared])
    out = node.forward(_WALK, _ENGINE, **tensors)

    assert set(tensors) == {"state", "embodiment_id"}
    assert list(out) == ["state_features"]
    assert len(out["state_features"]) == 1
    expected = node.state_encoder(
        walk["state"][0].reshape(1, 1, history * _INPUT), torch.tensor([_EMBODIMENT]),
    )
    assert torch.equal(out["state_features"][0], expected)


def test_the_node_declares_nothing_and_takes_one_request_a_step():
    """No attention and no state between steps, so no resources; one request per forward."""
    node = _node()
    prepared = node.prepare_inputs(_WALK, _FWD, _walk_inputs((_INPUT,)))

    assert node.declare_step(_WALK, ["r0"], [prepared]) is None
    assert node.max_batch_size(_WALK) == 1
