"""On a GPU: Canary's decoder prefill replayed from its capture against the same
prefill run eagerly, on the real resources (paged self-attention and context
caches, FlashInfer self- and cross-attention, positions, the sampler) through
two engines, one of them warmed up. A tiny random decoder; one row, then three
(padded to a bucket of four) with contexts of different lengths, some past a
page, then three more on the same buckets so a replay can't lean on the last
one's context. Compared: the logits prefill samples from, the token, each
request's prompt K/V in every layer and its context K/V. Skips without CUDA or
FlashInfer."""
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)
pytest.importorskip("flashinfer")

from mstar.communication.tensors import LocalTransferEngine  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.distributed.communication import WorkerParallelGroups  # noqa: E402
from mstar.engine.engine import Engine, ExecutingBatch  # noqa: E402
from mstar.engine.resources import SamplingReqConfig, StepContext  # noqa: E402
from mstar.engine.resources.kv.transfer import TransferEngineInfo  # noqa: E402
from mstar.model.canary.canary_model import CanaryModel  # noqa: E402
from mstar.model.canary.config import CROSS_KV_CACHE, KV_CACHE, PROMPT_LENGTH, SAMPLER, CanaryModelConfig  # noqa: E402
from mstar.model.canary.decoder import CanaryDecoderModel  # noqa: E402
from mstar.model.canary.submodules import CanaryDecoderSubmodule  # noqa: E402
from mstar.model.components.fastconformer import FastConformerConfig  # noqa: E402

DEVICE = torch.device("cuda", 0)
VOCAB = 64
CONFIG = CanaryModelConfig(
    encoder=FastConformerConfig(), hidden_size=256, num_layers=2, num_heads=2, intermediate_size=512,
    vocab_size=VOCAB, max_position_embeddings=64, eos_token_id=3,
)
GREEDY = SamplingReqConfig(temperature=0.0)


class _Probed(CanaryDecoderSubmodule):
    """Canary's decoder node with fewer buckets, keeping the logits prefill
    samples from in a buffer a replay writes as the eager forward does."""

    DECODE_CAPTURE_BATCH_SIZES = [1, 4]
    MAX_PREFILL_BATCH_SIZE = 4

    def process_logits(self, logits, graph_walk):
        if graph_walk == self.PREFILL_WALK:
            self.prefill_logits[:logits.shape[0]].copy_(logits)
        return logits


def _node() -> _Probed:
    decoder = CanaryDecoderModel(CONFIG)
    # the parallel linears allocate their weights for a checkpoint to fill
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in decoder.named_parameters():
            if name.endswith("norm.weight"):
                param.fill_(1.0)
            elif param.dim() == 1:
                param.normal_(0.0, 0.02, generator=generator)
            else:
                param.normal_(0.0, param.shape[-1] ** -0.5, generator=generator)
    decoder = decoder.to(DEVICE, torch.bfloat16).eval()
    node = _Probed(decoder, CONFIG)
    node.prefill_logits = torch.zeros(4, VOCAB, device=DEVICE)
    return node


def _engine(capture: bool) -> tuple[Engine, _Probed]:
    model = object.__new__(CanaryModel)
    model.config = CONFIG
    node = _node()
    engine = Engine(None, autocast_dtype=torch.bfloat16)
    engine.load_model(
        submodules={"decoder": node}, specs=model.get_node_resources(),
        parallel_groups=WorkerParallelGroups(num_workers=1, global_rank=0),
        device=DEVICE, transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
    )
    if capture:
        engine.warmup()
        assert engine._submodules["decoder"].accelerator_graph_runner is not None
    return engine, node


@pytest.fixture(scope="module")
def engines():
    return _engine(capture=True), _engine(capture=False)


def _prefill(engine: Engine, node: _Probed, rids: list[int], contexts: list[torch.Tensor], prompt: torch.Tensor):
    for rid in rids:
        engine.add_request(rid, {SAMPLER: GREEDY})
    batch = ExecutingBatch(
        node_name="decoder",
        per_request_info={rid: CurrentForwardPassInfo(
            request_id=f"r{rid}", rid_handle=rid, graph_walk="prefill", fwd_index=0, random_seed=0,
            max_tokens=8, resource_configs={SAMPLER: GREEDY},
        ) for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False),
        per_request_input_tensors={
            rid: {"text_inputs": [prompt], "encoder_states": [context]}
            for rid, context in zip(rids, contexts, strict=True)
        },
    )
    engine.prepare_inputs(batch)
    out = engine.exec_and_postprocess(batch)
    lease = batch.step_context.slot_lease
    torch.cuda.synchronize()
    tokens = torch.stack([out.get(rid)["new_token"][0].reshape(()) for rid in rids]).cpu()
    return lease, node.prefill_logits[:len(rids)].clone(), tokens


def _pages(engine: Engine, key: str, rid: int, length: int) -> torch.Tensor:
    """The K/V a request's stream holds, every layer: [layers, 2, length, heads, dim]."""
    kv = engine._resources[key]
    cache = kv.kv_cache.tensor
    pages = kv._streams[rid]["main"].page_indices
    return torch.cat([cache[:, page] for page in pages], dim=2)[:, :, :length].float()


@pytest.mark.parametrize("context_lens", [[37], [200, 13, 129], [5, 300, 64]], ids=["one", "three", "three_again"])
def test_captured_prefill_matches_eager(engines, context_lens):
    (captured, captured_node), (eager, eager_node) = engines
    base = 100 * len(context_lens) + context_lens[0]
    rids = [base + i for i in range(len(context_lens))]
    generator = torch.Generator(device=DEVICE).manual_seed(base)
    contexts = [
        torch.randn(n, CONFIG.hidden_size, device=DEVICE, generator=generator).to(torch.bfloat16)
        for n in context_lens
    ]
    prompt = torch.randint(4, VOCAB, (PROMPT_LENGTH,), generator=torch.Generator().manual_seed(base))

    lease, logits, tokens = _prefill(captured, captured_node, rids, contexts, prompt)
    assert lease is not None, "the prefill ran eagerly, not from its capture"
    # the bucket sized by the prompt alone: the smallest batch that holds the rows
    bs = 1 if len(rids) == 1 else 4
    assert (lease.bucket.bs, lease.bucket.num_tokens) == (bs, bs * PROMPT_LENGTH)
    eager_lease, eager_logits, eager_tokens = _prefill(eager, eager_node, rids, contexts, prompt)
    assert eager_lease is None

    assert torch.isfinite(eager_logits).all()
    torch.testing.assert_close(logits, eager_logits, atol=2e-2, rtol=2e-2)
    top2 = eager_logits.topk(2, dim=-1).values
    # the token wherever the logits don't leave it to rounding
    clear = ((top2[:, 0] - top2[:, 1]) > 0.05).cpu()
    assert torch.equal(tokens[clear], eager_tokens[clear]), (tokens, eager_tokens)
    for rid, n in zip(rids, context_lens, strict=True):
        # the context, written in preprocess by the same code on both paths
        assert torch.equal(_pages(captured, CROSS_KV_CACHE, rid, n), _pages(eager, CROSS_KV_CACHE, rid, n)), rid
        # the prompt's K/V, which past layer 0 follow from the cross-attention over it
        torch.testing.assert_close(
            _pages(captured, KV_CACHE, rid, PROMPT_LENGTH), _pages(eager, KV_CACHE, rid, PROMPT_LENGTH),
            atol=2e-2, rtol=2e-2,
        )
    for engine in (captured, eager):
        for rid in rids:
            engine.remove_request(rid)
