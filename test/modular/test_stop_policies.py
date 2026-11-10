"""A submodule's declared stop policies agree with its own ``check_stop``.

The graph runtime runs ``get_stop_policies`` in place of ``check_stop`` when
a walk declares them, so for every model that declares any, each row a
policy stops (``StopPolicy.stops``) must be one ``check_stop`` stops, on
every walk, over random tokens, iteration counts, limits and ``ignore_eos``
settings.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mstar.model.multimodal import vision_encoder
from mstar.model.muse.submodules import LLMSubmodule as GlimmerLLMSubmodule
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.qwen3_5.submodules import LLMSubmodule

LOOP = "decode_loop"
EOS = (7, 9)

MODELS = {
    # (submodule, its config, its walks, its TokenStop if it declares one)
    "qwen3_5": (LLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                ("decode", "prefill_text", "prefill_vision"),
                LLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                             encoders=(vision_encoder(),)).token_stop),
    "orpheus": (OrpheusLLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)), ("prefill", "decode"),
                OrpheusLLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset(EOS))).token_stop),
    # the shared VLM decoder's stop, as Muse Glimmer serves it
    "muse_glimmer": (GlimmerLLMSubmodule, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                     ("decode", "prefill_text", "prefill_vision"),
                     GlimmerLLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset(EOS)),
                                         (vision_encoder(),)).token_stop),
}


def _sampler_key(cls):
    return getattr(cls, "SAMPLER", None) or __import__(cls.__module__, fromlist=["SAMPLER"]).SAMPLER


def _info(rng, walk, sampler):
    max_tokens = int(rng.integers(1, 6))
    return SimpleNamespace(
        graph_walk=walk, step_metadata={}, max_tokens=max_tokens,
        resource_configs={sampler: SimpleNamespace(ignore_eos=bool(rng.integers(2)))},
        dynamic_loop_iter_counts={LOOP: int(rng.integers(0, max_tokens + 1))},
    )


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("seed", range(8))
def test_declared_policies_stop_the_rows_check_stop_does(model, seed):
    cls, config, walks, stop = MODELS[model]
    sampler = _sampler_key(cls)
    sub = cls.__new__(cls)
    torch.nn.Module.__init__(sub)
    sub.config = config
    if stop is not None:
        sub.token_stop = stop
    rng = np.random.default_rng(seed)
    declared_walks = 0
    for walk in walks:
        policies = sub.get_stop_policies(walk)
        if policies is None:
            continue  # check_stop decides (a TokenStop's prompt walks)
        declared_walks += 1
        for rid in range(16):
            token = int(rng.choice([*EOS, 1, 2, 3]))
            info = _info(rng, walk, sampler)
            expected = sub.check_stop(rid, info, {"new_token": [torch.tensor([token])]}) or set()
            declared = {
                loop for loop, p in policies.items()
                if p.stops(info.dynamic_loop_iter_counts.get(loop, 0), token, p.limits(info))
            }
            assert declared == set(expected), (walk, rid)
    assert declared_walks


def test_a_token_stop_names_the_walks_that_end_its_loop():
    from mstar.model.submodule_base import TokenStop

    def stop(walk):
        return TokenStop(loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler="s", walk=walk)

    assert stop(None).stops_on("a")
    assert stop("a").stops_on("a") and not stop("a").stops_on("b")
    both = stop(frozenset({"a", "b"}))
    assert both.stops_on("a") and both.stops_on("b") and not both.stops_on("c")


def _ar(stop):
    from mstar.model.submodule_base import ARNodeSubmodule

    class _AR(ARNodeSubmodule):
        def prepare_inputs(self, *args, **kwargs):
            raise NotImplementedError

        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def preprocess(self, *args, **kwargs):
            raise NotImplementedError

    sub = _AR()
    sub.token_stop = stop
    return sub


def _row(walk, ignore_eos=False):
    return SimpleNamespace(graph_walk=walk, step_metadata={}, max_tokens=8, dynamic_loop_iter_counts={},
                           resource_configs={"s": SimpleNamespace(ignore_eos=ignore_eos)})


def test_a_withhold_predicate_withholds_on_its_walks_beside_the_stop_ids():
    """``withhold_unless`` withholds a row whose token fails it, and ``withhold_walk``
    limits withholding (the stop-id one too) to its walks; the row and the batched checks
    agree, off one host read. The runtime checks the other walks' stops, never one that
    withholds."""
    from mstar.model.submodule_base import BatchedStopCheck, HostRows, StopCheck, TokenStop

    stop = TokenStop(loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler="s",
                     walk=frozenset({"text", "spoken"}), withhold=("codes",),
                     withhold_unless=lambda token: token >= 100, withhold_walk="spoken", runtime_check=True)
    sub = _ar(stop)
    assert sub.get_stop_policies("spoken") is None and set(sub.get_stop_policies("text")) == {LOOP}
    rows = {  # rid: (walk, token, ignore_eos) -> (stops, withheld)
        0: (("spoken", 100, False), (set(), False)),
        1: (("spoken", 3, False), (set(), True)),
        2: (("spoken", 7, False), ({LOOP}, True)),
        3: (("spoken", 7, True), (set(), True)),
        4: (("text", 3, False), (set(), False)),
        5: (("text", 7, False), ({LOOP}, False)),
    }
    infos = {rid: _row(walk, ignore) for rid, ((walk, _, ignore), _) in rows.items()}
    for rid, ((_, token, _), (stops, withheld)) in rows.items():
        got = sub.check_stop(rid, infos[rid], {"new_token": [torch.tensor([token])]})
        assert got == (StopCheck(stops, withheld=("codes",)) if withheld else stops), rid

    class Counting(HostRows):
        reads = 0

        def first_values(self, name, request_ids):
            Counting.reads += 1
            return super().first_values(name, request_ids)

    host = Counting(tuple(rows), {"new_token": torch.tensor([[token] for (_, token, _), _ in rows.values()])})
    got = sub.check_stop_batched(list(rows), infos, host)
    assert Counting.reads == 1
    assert got == BatchedStopCheck(
        {rid: s for rid, (_, (s, _)) in rows.items() if s},
        {rid: ("codes",) for rid, (_, (_, w)) in rows.items() if w},
    )
    with pytest.raises(ValueError, match="need outputs to withhold"):
        TokenStop(loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler="s", withhold_walk="spoken")


def test_a_stop_id_withhold_without_the_new_fields_withholds_on_every_walk():
    """The stop-id withhold as Fish S2 Pro and Qwen3-TTS declare it is unchanged: a stop
    id withholds on any walk, ``ignore_eos`` or not, and nothing else does."""
    from mstar.model.submodule_base import StopCheck, TokenStop

    sub = _ar(TokenStop(loop=LOOP, buffer="new_token", stop_ids=frozenset(EOS), sampler="s", withhold=("codes",)))
    for walk in ("a", "b"):
        for ignore in (False, True):
            got = sub.check_stop(0, _row(walk, ignore), {"new_token": [torch.tensor([7])]})
            assert got == StopCheck(set() if ignore else {LOOP}, withheld=("codes",))
        assert sub.check_stop(0, _row(walk), {"new_token": [torch.tensor([3])]}) == set()
        assert sub.get_stop_policies(walk) is None


def test_a_nodes_stop_limits_come_from_its_own_partitions_info():
    """A request's partitions arrive on a worker one by one; each sets only its own nodes'
    limits, so a later partition's info cannot undo another's (an LM's ignore_eos)."""
    from mstar.conductor.request_info import CurrentForwardPassInfo
    from mstar.graph.runtime.rust import RustGraphRuntime
    from mstar.model.submodule_base import StopKind, StopPolicy

    set_calls = []
    runtime = RustGraphRuntime.__new__(RustGraphRuntime)
    runtime._rust = SimpleNamespace(
        set_stop_limits=lambda node, walk, rids, iters, eos: set_calls.append((node, eos)),
    )
    runtime._node_to_partition = {"llm": "LM", "flow": "Token2Wav"}
    policy = StopPolicy(StopKind.EOS, token_signal="speech_tokens", eos_tokens=(1,), ignores_eos=lambda info: True)
    runtime._stop_policies = {("llm", "decode"): {"decode_loop": policy}, ("flow", "token2wav"): {}}

    for partition in ("LM", "Token2Wav"):
        info = CurrentForwardPassInfo.__new__(CurrentForwardPassInfo)
        info.partition_name, info.max_tokens = partition, 100
        runtime.register_stop_limits(7, info)
    assert set_calls == [("llm", [[True]])]
