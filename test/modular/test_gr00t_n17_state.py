"""GR00T N1.7's state encoder: the per-embodiment MLP that turns a robot's raw state
into the one conditioning token the denoise loop carries.

The MLP is checked against an inline reference written from the algorithm -- two linears
with a ReLU between, each row of the batch reading its own category's slice of ``W`` and
``b``. The reference loops over the rows, which is the half ``bmm`` does all at once and
the half a mistake hides in: a wrong index still returns a plausibly shaped tensor.

The last section drives the ``state_encoder`` node, ``Gr00tN17StateEncoderSubmodule``,
the way the engine does. It has no attention and so declares nothing.
"""

import glob
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.gr00t_n17.components.embodiment import CategorySpecificMLP, build_state_encoder
from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.gr00t_n17.submodules import Gr00tN17StateEncoderSubmodule
from mstar.model.gr00t_n17.weight_loader import _remapper
from mstar.model.submodule_base import ModelInputsFromEngine

_HUB = Path.home() / ".cache/huggingface/hub"
_GR00T = _HUB / "models--nvidia--GR00T-N1.7-3B/snapshots/2fc962b973bccdd5d8ce4f67cc63b264d6886495"
_BACKBONE = next(iter(glob.glob(str(_HUB / "models--nvidia--Cosmos-Reason2-2B/snapshots/*"))), None)

_CATEGORIES = 4
_INPUT, _HIDDEN, _OUTPUT = 5, 7, 3


@pytest.fixture(autouse=True)
def _pinned_threads():
    """fp32 CPU forwards compared with ``torch.equal``; reduction order follows the thread
    count, as it does in the other gr00t tests."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _config() -> Gr00tN17Config:
    """The four fields the state encoder reads; the rest of the config is the other nodes'.

    Input, hidden and output are three different sizes, so the two layers cannot be swapped
    without a shape error, and neither can be mistaken for the other.
    """
    return Gr00tN17Config(
        text=None, vision=None, dit=None, bridge=None,
        deepstack_visual_indexes=(), backbone_embedding_dim=0,
        max_state_dim=_INPUT, hidden_size=_HIDDEN, input_embedding_dim=_OUTPUT,
        max_num_embodiments=_CATEGORIES,
    )


def _small_integers(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    """Whole numbers in [-8, 8]. Every product and sum below stays well inside fp32's exact
    integer range, so the arithmetic is the same in any reduction order and the reference can
    be compared with ``torch.equal`` without having to reach for the same kernel: on random
    floats ``bmm`` and a 2-D ``@`` take different MKL paths and disagree in the last ulp.
    """
    return torch.randint(-8, 9, shape, generator=generator).float()


def _mlp(seed: int = 0) -> CategorySpecificMLP:
    """The encoder the config asks for, filled. ``CategorySpecificLinear`` allocates with
    ``torch.empty``, so an unfilled parameter here would be uninitialised memory."""
    mlp = build_state_encoder(_config())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.copy_(_small_integers(parameter.shape, generator))
    return mlp


def _state(batch: int, history: int = 1, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return _small_integers(torch.Size((batch, history, _INPUT)), generator)


# ----------------------------------------------------------------------------
# Inline reference: `CategorySpecificMLP`, written from its description
# ----------------------------------------------------------------------------


def _reference(
    mlp: CategorySpecificMLP, x: torch.Tensor, cat_ids: torch.Tensor, relu: str = "between",
) -> torch.Tensor:
    """Row ``i`` of ``x`` takes category ``cat_ids[i]``'s slice of every ``W`` and ``b``:
    ``x @ W[id] + b[id]``, a ReLU, the same again. One row at a time, so the category axis
    is indexed here rather than gathered.

    ``relu`` is the deviation knob -- "between" is the real thing, the others are the
    placements a reader might expect instead.
    """
    rows = []
    for row, cat in zip(x, cat_ids.tolist(), strict=True):
        if relu == "before":
            row = torch.relu(row)
        hidden = row @ mlp.layer1.W[cat] + mlp.layer1.b[cat]
        if relu == "between":
            hidden = torch.relu(hidden)
        out = hidden @ mlp.layer2.W[cat] + mlp.layer2.b[cat]
        rows.append(torch.relu(out) if relu == "after" else out)
    return torch.stack(rows)


_IDS = {
    "the first embodiment": [0],
    "one in the middle": [2],
    "the last embodiment": [_CATEGORIES - 1],
    "a batch on one embodiment": [1, 1, 1],
    # ids out of order, repeated, and spanning both ends
    "a batch across embodiments": [3, 0, 1, 3],
}


@pytest.mark.parametrize("history", [1, 4], ids=["one state", "a state history"])
@pytest.mark.parametrize("ids", list(_IDS.values()), ids=list(_IDS))
@torch.no_grad()
def test_the_mlp_is_two_per_embodiment_linears_with_a_relu_between(ids, history):
    """``history`` is the ``T`` axis: this checkpoint sends one state, but the config allows
    more and the bias has to broadcast over them rather than along them."""
    mlp = _mlp()
    cat_ids = torch.tensor(ids)
    x = _state(len(ids), history)
    out = mlp(x, cat_ids)

    assert out.shape == (len(ids), history, _OUTPUT)
    assert torch.equal(out, _reference(mlp, x, cat_ids))


@pytest.mark.parametrize(
    "relu", ["none", "before", "after"],
    ids=["no relu at all", "relu before both layers", "relu after both layers"],
)
@torch.no_grad()
def test_the_relu_belongs_between_the_two_layers(relu):
    """Two linears with nothing between them are one linear, and a ReLU on either outside
    edge is a different function again. Each has to be a different answer, or the parity
    above would hold for a model that is not this one.
    """
    mlp = _mlp()
    cat_ids = torch.tensor(_IDS["a batch across embodiments"])
    x = _state(len(cat_ids), history=2)
    assert not torch.equal(mlp(x, cat_ids), _reference(mlp, x, cat_ids, relu=relu))


@pytest.mark.parametrize("cat", range(_CATEGORIES))
@torch.no_grad()
def test_an_embodiment_reads_its_own_slice_and_no_other(cat):
    """``W[0]`` for every row, or a bias indexed by anything other than ``cat_ids``, returns
    a plausible tensor rather than an error. So each category is pinned from both sides:
    nobody else's slice gives this answer, and moving everybody else's weights does not move
    it.
    """
    mlp = _mlp()
    x = _state(1, history=2)
    ours = mlp(x, torch.tensor([cat]))

    assert torch.equal(ours, _reference(mlp, x, torch.tensor([cat])))
    for other in range(_CATEGORIES):
        if other != cat:
            assert not torch.equal(ours, _reference(mlp, x, torch.tensor([other])))

    others = [c for c in range(_CATEGORIES) if c != cat]
    for parameter in mlp.parameters():
        parameter[others] += 1.0
    assert torch.equal(mlp(x, torch.tensor([cat])), ours)


@torch.no_grad()
def test_each_row_of_a_batch_takes_its_own_embodiment():
    """``bmm`` broadcasts a mis-gathered ``W`` as happily as the right one, so the batch is
    checked against its own rows run one at a time: a row's answer must not depend on who it
    was batched with. Two rows share an embodiment and differ only in their state, which is
    what says the rows were not collapsed onto one another.
    """
    mlp = _mlp()
    cat_ids = torch.tensor(_IDS["a batch across embodiments"])
    x = _state(len(cat_ids), history=2)
    batched = mlp(x, cat_ids)

    for i, cat in enumerate(cat_ids.tolist()):
        assert torch.equal(batched[i : i + 1], mlp(x[i : i + 1], torch.tensor([cat])))
    assert cat_ids[0] == cat_ids[3]
    assert not torch.equal(batched[0], batched[3])


def test_the_state_encoder_is_sized_from_the_config():
    """``hidden_size`` is the layer between and ``input_embedding_dim`` is what comes out.
    They are different numbers in the checkpoint (1024 and 1536), and a build that swapped
    them would load nothing and run nowhere -- but only once a real checkpoint is in hand,
    which CI has not got.
    """
    encoder = build_state_encoder(_config())
    assert encoder.layer1.W.shape == (_CATEGORIES, _INPUT, _HIDDEN)
    assert encoder.layer1.b.shape == (_CATEGORIES, _HIDDEN)
    assert encoder.layer2.W.shape == (_CATEGORIES, _HIDDEN, _OUTPUT)
    assert encoder.layer2.b.shape == (_CATEGORIES, _OUTPUT)


@pytest.mark.skipif(not (_GR00T.exists() and _BACKBONE), reason="checkpoint not cached")
def test_the_checkpoints_state_encoder_tensors_name_this_nodes_parameters():
    """The node's child is ``state_encoder`` and the MLP keeps the reference's ``layer1`` /
    ``layer2`` and ``W`` / ``b``, so the checkpoint's keys land on it with only
    ``action_head.`` stripped. Shapes come off the safetensors header, so no tensor is read.
    """
    config = Gr00tN17Config.from_hf(_GR00T, _BACKBONE)
    with torch.device("meta"):  # 32 x 1024 x 1536 of it, allocated nowhere
        node = Gr00tN17StateEncoderSubmodule(build_state_encoder(config))
    weight_map = json.loads((_GR00T / "model.safetensors.index.json").read_text())["weight_map"]
    remap = _remapper("state_encoder")
    checkpoint = {remap(name): (name, shard) for name, shard in weight_map.items() if remap(name)}

    assert set(checkpoint) == {name for name, _ in node.named_parameters()}
    for name, parameter in node.named_parameters():
        key, shard = checkpoint[name]
        with safe_open(str(_GR00T / shard), framework="pt") as shard_file:
            assert list(parameter.shape) == shard_file.get_slice(key).get_shape()


# ----------------------------------------------------------------------------
# The node: `Gr00tN17StateEncoderSubmodule`, driven the way the engine drives it
# ----------------------------------------------------------------------------

_WALK = "prefill"
_FWD = CurrentForwardPassInfo(request_id="r0", graph_walk=_WALK, fwd_index=0, random_seed=0, max_tokens=0)
_ENGINE = ModelInputsFromEngine(request_ids=["r0"], per_request_info={})
_EMBODIMENT = 3

# how the walk may carry one request's state -> the history length it means
_WALK_STATES = {
    "a bare state vector": ((_INPUT,), 1),
    "a [1, dim] row": ((1, _INPUT), 1),
    "a state history": ((3, _INPUT), 3),
    "a history already batched": ((1, 3, _INPUT), 3),
}


def _node() -> Gr00tN17StateEncoderSubmodule:
    return Gr00tN17StateEncoderSubmodule(_mlp())


def _walk_inputs(shape: tuple[int, ...], seed: int = 5) -> dict[str, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "state": [_small_integers(torch.Size(shape), generator)],
        "embodiment_id": [torch.tensor(_EMBODIMENT)],
    }


@pytest.mark.parametrize("shape, history", list(_WALK_STATES.values()), ids=list(_WALK_STATES))
def test_prepare_inputs_gives_the_mlp_the_two_axes_it_indexes(shape, history):
    """The MLP wants ``[B, T, dim]`` and ``[B]``; the walk carries one request's state in
    whatever shape it was read in. Nothing but the shape changes, so the values come through
    in order."""
    node = _node()
    walk = _walk_inputs(shape)
    prepared = node.prepare_inputs(_WALK, _FWD, walk)
    tensors = prepared.tensor_inputs

    assert set(tensors) == {"state", "embodiment_id"}
    assert tensors["state"].shape == (1, history, _INPUT)
    assert torch.equal(tensors["state"].reshape(-1), walk["state"][0].reshape(-1))
    assert tensors["embodiment_id"].shape == (1,)
    assert tensors["embodiment_id"].tolist() == [_EMBODIMENT]
    # one token per state row: the engine sizes padding and capture buckets off
    # this, so a history longer than the checkpoint's 1 must not under-count
    assert prepared.input_seq_len == history


@pytest.mark.parametrize("shape, history", list(_WALK_STATES.values()), ids=list(_WALK_STATES))
@torch.no_grad()
def test_driving_the_node_is_the_mlp_called_directly(shape, history):
    """The node is plumbing: prepare_inputs -> preprocess -> forward is the MLP over the same
    state and the same id, the same ops in the same order, so this is ``torch.equal`` and not
    a bar. It also pins the key the denoise loop reads the features under."""
    node = _node()
    walk = _walk_inputs(shape)
    prepared = node.prepare_inputs(_WALK, _FWD, walk)
    tensors = node.preprocess(_WALK, _ENGINE, [prepared])
    out = node.forward(_WALK, _ENGINE, **tensors)

    assert set(tensors) == {"state", "embodiment_id"}
    assert list(out) == ["state_features"]
    assert len(out["state_features"]) == 1
    assert out["state_features"][0].shape == (1, history, _OUTPUT)
    expected = node.state_encoder(
        walk["state"][0].reshape(1, history, _INPUT), torch.tensor([_EMBODIMENT]),
    )
    assert torch.equal(out["state_features"][0], expected)


def test_the_node_declares_nothing_and_takes_one_request_a_step():
    """There is no attention and no state kept between steps here, so the node declares no
    resources at all -- the base class's ``None``, which is what the engine reads as "this
    submodule needs no admit/plan/commit". And as the other gr00t nodes: no batched forward,
    so ``preprocess`` unpacks the one request and a second raises rather than being dropped.
    """
    node = _node()
    prepared = node.prepare_inputs(_WALK, _FWD, _walk_inputs((_INPUT,)))
    two = ModelInputsFromEngine(request_ids=["r0", "r1"], per_request_info={})

    assert node.declare_step(_WALK, ["r0"], [prepared]) is None
    assert node.max_batch_size(_WALK) == 1
    with pytest.raises(ValueError, match="expected 1"):
        node.preprocess(_WALK, two, [prepared, prepared])
