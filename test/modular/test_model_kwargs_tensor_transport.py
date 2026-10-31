"""Real preprocessing and SHM transfers for kwargs formerly sent through pickle.

Tokenization is stubbed for Pi0.5; the state discretizer and model preprocessing
are real. Cosmos worker preparation is checked separately with its optional
dependencies installed.
"""

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mstar.communication.wire_types  # noqa: F401 (registers message tags)
from mstar.api_server.data_worker import PreprocessWorkerThread
from mstar.api_server.request_types import PreprocessInput
from mstar.communication.codec import WireCodec
from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, WorkerGraphStateRegistry
from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
from mstar.model.pi05.pi05_model import Pi05Model
from mstar.model.vjepa2.vjepa2_model import VJepa2Model


class _Communicator(BaseCommunicator):
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        # Exercise the actual codec on the preprocessing/conductor boundary.
        self.sent.append((entity_id, WireCodec.decode(WireCodec.encode(msg))))

    def get_all_new_messages(self):
        return []

    def register_event_for_poll(self, event):
        pass

    def poll_for_messages(self, timeout_ms=20):
        return False


class _Tokenizer:
    def encode_prompt(self, prompt):
        return torch.tensor(list(prompt.encode()), dtype=torch.long)


def _manager(tmp_path, entity, device, communicator):
    manager = SharedMemoryCommunicationManager(
        entity, "localhost", device, communicator, shm_dir=str(tmp_path),
    )
    sharding = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
    sharding.setup({})
    manager.register_request("r", sharding)
    return manager


def _case(name, convert):
    if name == "pi05":
        model = Pi05Model("unused", skip_weight_loading=True)
        model.tokenizer = _Tokenizer()
        return model, {"robot_state": convert(np.linspace(-1, 1, 32).tolist())}, {}
    if name == "cosmos":
        model = Cosmos3Model("unused", skip_weight_loading=True)
        action = convert([[0.1 * i] * 7 for i in range(2)])
        return model, {
            "action": action, "action_mode": "forward_dynamics", "domain_id": 0,
            "action_chunk_size": 3, "num_frames": 4, "guidance_scale": 1.0,
            "height": 64, "width": 64,
        }, {"action_inputs": torch.as_tensor(action)}
    model = VJepa2Model(
        "unused", skip_weight_loading=True,
        predictor_kind="masked" if name == "masked" else "ac",
    )
    if name == "masked":
        kwargs = {"context_mask": convert([0, 1]), "target_mask": convert([2, 3])}
    else:
        kwargs = {"actions": convert([[0.1] * 7]), "states": convert([[0.2] * 7])}
        if name == "extrinsics":
            model.config.ac_predictor.use_extrinsics = True
            kwargs["extrinsics"] = convert([[0.3] * 6])
        if name == "mpc":
            kwargs.update(goal_hidden=convert([[0.4] * 8]), mpc=True)
    tensors = {k: torch.as_tensor(v) for k, v in kwargs.items() if k != "mpc"}
    return model, kwargs, tensors


@pytest.mark.parametrize("case", ["ac", "extrinsics", "masked", "mpc", "pi05", "cosmos"])
@pytest.mark.parametrize("representation", [
    "cpu", "cuda", "noncontiguous-cpu", "noncontiguous-cuda", "list", "numpy",
])
def test_model_kwargs_reach_workers_through_preprocessing_and_shm(
    case, representation, tmp_path,
):
    if "cuda" in representation and not torch.cuda.is_available():
        pytest.skip("CUDA tensor kwargs require a GPU")
    device = "cuda:0" if "cuda" in representation else "cpu"
    convert = {
        "cpu": torch.tensor,
        "cuda": lambda v: torch.tensor(v, device=device),
        "noncontiguous-cpu": lambda v: torch.tensor(v).repeat_interleave(2, dim=-1)[..., ::2],
        "noncontiguous-cuda": lambda v: torch.tensor(v, device=device).repeat_interleave(2, dim=-1)[..., ::2],
        "list": lambda v: v,
        "numpy": np.asarray,
    }[representation]
    model, kwargs, expected_tensors = _case(case, convert)
    original_kwargs = dict(kwargs)
    communicator = _Communicator()
    sender = _manager(tmp_path, "api_server", device, communicator)
    receiver = _manager(tmp_path, "worker_0", device, communicator)
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(), abort_request_queue=queue.Queue(),
        reads_done_queue=queue.Queue(), discard_tensor_queue=queue.Queue(), stop_event=threading.Event(),
        communicator=communicator, tensor_manager=sender, model=model,
    )
    try:
        worker._process_input(PreprocessInput("r", "Pick up the cup", None, [], ["video"], kwargs))
        request = next(msg.body for entity, msg in communicator.sent if entity == "conductor")
        for name, value in original_kwargs.items():
            assert kwargs[name] is value
            assert worker.request_model_kwargs["r"][name] is value
        for name in model.PREPROCESS_ONLY_KWARGS:
            assert name not in request.model_kwargs
        if case == "mpc":
            assert request.model_kwargs["mpc"] is True
        if case == "cosmos":
            args = model.get_initial_forward_pass_args(
                "p0", ["text"], ["video"], request.initial_signals, request.model_kwargs,
            )
            assert args.step_metadata["raw_action_dim"] == 7
            assert "action" not in args.step_metadata
            action_edge = next(e for e in args.inputs if e.name == "action_inputs")
            assert action_edge.tensor_info == request.initial_signals["action_inputs"]
            _assert_prefill_ready(model, args)
        edges = [GraphEdge(next_node="node", name=name, tensor_info=infos)
                 for name, infos in request.initial_signals.items()]
        receiver.start_read_tensors("r", edges)
        transferred = {name: [receiver.get_tensor(i.uuid) for i in infos]
                       for name, infos in request.initial_signals.items()}
        for name, expected in expected_tensors.items():
            received = transferred[name][0]
            assert received.device == torch.device(device)
            torch.testing.assert_close(received.cpu(), expected.cpu(), check_dtype=False)
        if case == "pi05":
            # State values must survive as prompt tokens, rather than being discarded.
            prompt = bytes(transferred["text_inputs"][0].cpu().tolist()).decode()
            assert "State:" in prompt and "Action:" in prompt
    finally:
        receiver.force_cleanup_request("r")
        sender.force_cleanup_request("r")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cosmos_worker_prepares_action_values(device):
    pytest.importorskip("diffusers")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA action preparation requires a GPU")
    from dataclasses import asdict

    from diffusers import UniPCMultistepScheduler

    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule
    from mstar.model.cosmos3.tests.test_action import _cfg

    model, kwargs, _ = _case("cosmos", lambda v: torch.tensor(v, device=device))
    output = model.process_prompt("Pick up the cup", [], ["video"], **kwargs)
    conductor_kwargs = {k: v for k, v in kwargs.items() if k not in model.PREPROCESS_ONLY_KWARGS}
    conductor_kwargs.update(output.metadata)
    args = model.get_initial_forward_pass_args(
        "p0", ["text"], ["video"], {name: [] for name in output.new_input_tensors}, conductor_kwargs,
    )
    config = _cfg()
    scheduler = UniPCMultistepScheduler.from_config(asdict(config.scheduler))
    dit = Cosmos3DiTSubmodule(Cosmos3OmniTransformer(config).to(device), config, scheduler=scheduler)
    # Run the actual worker preparation: the shorter trajectory repeats its last
    # action to fill the requested chunk, and padding beyond raw_action_dim is zero.
    dit._prepare_action_prefill(
        SimpleNamespace(rid_handle="r"), args.step_metadata, output.new_input_tensors,
        [1, 2, 3], None, 64, 64, 24.0, 1.0, 2, device,
    )
    action = dit.request_state("r")["clean_action"].cpu()
    expected = torch.tensor([[[0.0] * 7, [0.1] * 7, [0.1] * 7]])
    torch.testing.assert_close(action[:, :, :7], expected)
    assert torch.count_nonzero(action[:, :, 7:]) == 0


def test_cosmos_prefill_without_action_has_ready_empty_input():
    model = Cosmos3Model("unused", skip_weight_loading=True)
    args = model.get_initial_forward_pass_args("p0", ["text"], ["image"], {"text_inputs": []})
    _assert_prefill_ready(model, args)
    assert next(e for e in args.inputs if e.name == "action_inputs").tensor_info == []


def test_cosmos_tokenized_prompt_infers_action_width_before_routing(monkeypatch):
    from mstar.model.cosmos3.components import packing

    model, kwargs, _ = _case("cosmos", torch.tensor)
    model.tokenizer = object()
    monkeypatch.setattr(packing, "tokenize_prompt", lambda *args, **kw: ([1, 2, 3], [1, 2, 3]))
    output = model.process_prompt("Pick up the cup", [], ["video"], **kwargs)
    assert output.metadata["action_shape"] == [2, 7]
    torch.testing.assert_close(output.new_input_tensors["action_inputs"][0], kwargs["action"])


def _assert_prefill_ready(model, args):
    graph = model.get_graph_walk_graphs()[args.full_metadata.graph_walk]
    nodes = graph.get_nodes()
    registry = WorkerGraphStateRegistry(graph)
    for node in nodes.values():
        node._managing_registry = registry
    for edge in args.inputs:
        assert nodes[edge.next_node].ingest_input(edge)
    assert nodes["dit"].ready_signals.is_ready
