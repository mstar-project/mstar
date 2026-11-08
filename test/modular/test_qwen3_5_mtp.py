"""Qwen3.5 drafting with its MTP head (speculative decoding), on
the real Engine in synchronous mode, on the CPU (`qwen3_5_cpu`): greedy output
is plain greedy decoding's whatever the drafts, and every verify step's drafts
are the MTP chain recomputed from scratch on what the step kept."""
from __future__ import annotations

import collections

import pytest
import torch

from mstar.engine.resources.attn.base import AttentionManager

PROMPTS = {1: [5, 9, 2, 7, 11, 3], 2: [17, 4], 3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
MAX = 16
MARGIN = 1e-4


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    import gdn_cpu
    import qwen3_cpu

    from mstar.engine.resources.kv import manager as manager_mod

    qwen3_cpu.patch_cpu(monkeypatch)
    gdn_cpu.install(monkeypatch)
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)


def _check(model, out, prompts=PROMPTS):
    from qwen3_5_cpu import reference_greedy

    for rid, prompt in prompts.items():
        margins: list[float] = []
        ref = reference_greedy(model, prompt, MAX, margins=margins)
        n = min(len(ref), len(out[rid]))
        at = next((j for j in range(n) if ref[j] != out[rid][j]), None if len(ref) == len(out[rid]) else n)
        assert at is None or margins[at] < MARGIN, (rid, at, ref, out[rid])


def test_plain_qwen3_5_on_the_cpu_matches_the_dense_reference(tmp_path, monkeypatch):
    from qwen3_5_cpu import decode, make_model, write_checkpoint

    model = make_model(monkeypatch, write_checkpoint(tmp_path))
    _check(model, decode(model, PROMPTS, MAX).out)



class Probe:
    """Every verify step's rows, their input token's index in the reply and the
    MTP head's drafts (the real ones, before any forcing)."""

    def __init__(self, monkeypatch, reference=None, spoil=lambda j: False):
        from mstar.model.qwen3_5.components.mtp import Qwen3_5MTPDrafter

        self.steps = []
        real = Qwen3_5MTPDrafter.propose
        probe = self

        def propose(drafter, inputs):
            drafts = real(drafter, inputs)
            probe.steps.append((list(inputs.request_ids), drafts.clone()))
            if reference is None:
                return drafts
            n, k = drafts.shape
            at = inputs.resources["rope"].pos_ids("main").view(n, k + 1)[:, 0].tolist()
            rows = []
            for rid, pos in zip(inputs.request_ids, at, strict=True):
                seq = reference[rid]
                row = [seq[pos + 1 + j] if pos + 1 + j < len(seq) else 0 for j in range(k)]
                rows.append([(t + 1) % 97 if spoil(j) else t for j, t in enumerate(row)])
            return torch.tensor(rows, dtype=torch.long)

        monkeypatch.setattr(Qwen3_5MTPDrafter, "propose", propose)


def _check_probe_drafts(model, probe, driver, k, prompt_of) -> list[int]:
    """Each verify step's real drafts against the MTP chain from scratch on
    what its row had kept; ``prompt_of(rid, reply)`` gives the reference's
    ``(embeds, positions)`` or ``None`` for a text prompt. Returns the
    accepted count of every step checked."""
    from qwen3_5_cpu import reference_drafts, reference_drafts_embeds

    emitted = collections.defaultdict(lambda: 1)
    accepted = iter(driver.accepted)
    counts = []
    for rids, drafts in probe.steps:
        for i, rid in enumerate(rids):
            reply = driver.out[rid][: emitted[rid]]
            margins: list[float] = []
            embedded = prompt_of(rid, reply)
            if embedded is None:
                want = reference_drafts(model, PROMPTS[rid] + reply, k, margins=margins)
            else:
                want = reference_drafts_embeds(model, *embedded, k, margins=margins)
            got = drafts[i].tolist()
            at = next((j for j in range(k) if want[j] != got[j]), None)
            assert at is None or margins[at] < MARGIN, (rid, reply, want, got)
            count = next(accepted)
            counts.append(count)
            emitted[rid] += count + 1
    return counts


def _model(tmp_path, monkeypatch, k=None, **kw):
    from qwen3_5_cpu import make_model, write_checkpoint

    return make_model(monkeypatch, write_checkpoint(tmp_path), k=k, **kw)


def test_a_drafting_node_declares_no_rebuild_policy(tmp_path, monkeypatch):
    """A rebuild replays the prompt and kept samples, which nothing checks against a verifying request's
    pending take-backs and carry: KV pressure on a drafting node is offload and eviction only."""
    assert _model(tmp_path / "mtp", monkeypatch, k=3).rebuild_policies() == {}
    assert _model(tmp_path / "plain", monkeypatch).rebuild_policies()


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("batched", [True, False])
def test_mtp_drafting_keeps_plain_greedy_output(tmp_path, monkeypatch, k, batched):
    from qwen3_5_cpu import decode

    model = _model(tmp_path, monkeypatch, k=k)
    _check(model, decode(model, PROMPTS, MAX, batched=batched).out)


def test_the_teacher_forced_reference_is_the_reference_chain_on_its_own_drafts(tmp_path, monkeypatch):
    """`forced_draft_logprobs` fed `reference_drafts`' own chain ranks that chain
    first at every draft: the device check's scorer is the reference."""
    from qwen3_5_cpu import forced_draft_logprobs, reference_drafts

    model = _model(tmp_path, monkeypatch, k=3)
    for prompt in PROMPTS.values():
        drafts = reference_drafts(model, prompt, 3, round_to=torch.bfloat16)
        logprobs = forced_draft_logprobs(model, prompt, drafts)
        assert [int(lp.argmax()) for lp in logprobs] == drafts


def test_every_verify_steps_drafts_are_the_mtp_chain_from_scratch(tmp_path, monkeypatch):
    """At each verify step, the drafts are what the MTP head gives run from
    scratch over everything the request kept: its K/V from the prompt step and
    each step's observe (the target's hidden states, the verified tokens), and
    its first draft from the carry. Checked by the margin rule."""
    from qwen3_5_cpu import decode, reference_drafts

    k = 3
    model = _model(tmp_path, monkeypatch, k=k)
    probe = Probe(monkeypatch)
    driver = decode(model, PROMPTS, MAX, batched=False)
    _check(model, driver.out)
    replies = {rid: [] for rid in PROMPTS}
    checked = 0
    # serial runs: each step's one row; its input is the reply's latest token so far
    emitted = {rid: 1 for rid in PROMPTS}
    accepted = iter(driver.accepted)
    for rids, drafts in probe.steps:
        (rid,) = rids
        reply = driver.out[rid]
        seq = PROMPTS[rid] + reply[: emitted[rid]]
        margins: list[float] = []
        want = reference_drafts(model, seq, k, margins=margins)
        got = drafts[0].tolist()
        at = next((j for j in range(k) if want[j] != got[j]), None)
        assert at is None or margins[at] < MARGIN, (rid, seq, want, got)
        emitted[rid] += next(accepted) + 1
        replies[rid] = reply
        checked += 1
    assert checked >= len(PROMPTS)


@pytest.mark.parametrize("first_wrong", [None, 0, 1, 2])
def test_every_accepted_count_keeps_plain_greedy_output(tmp_path, monkeypatch, first_wrong):
    """Drafts forced to the reference continuation, spoiled from ``first_wrong``
    on, so every accepted count runs: the GDN rerun, the take-backs and the
    MTP head's observe all see it."""
    from qwen3_5_cpu import decode, reference_greedy

    k = 3
    model = _model(tmp_path, monkeypatch, k=k)
    reference = {rid: p + reference_greedy(model, p, MAX + 8) for rid, p in PROMPTS.items()}
    probe = Probe(monkeypatch, reference, spoil=lambda j: first_wrong is not None and j >= first_wrong)
    driver = decode(model, PROMPTS, MAX)
    _check(model, driver.out)
    want = k if first_wrong is None else first_wrong
    # the head's own drafts at every step, whatever was forced: its K/V and carry after each count
    counts = _check_probe_drafts(model, probe, driver, k, lambda rid, reply: None)
    assert want in counts, counts


def test_a_chunked_prompt_writes_the_mtp_heads_kv_and_carry_as_a_whole_one(tmp_path, monkeypatch):
    """`ChunkBoundary.READS_NEXT_TOKEN`: a chunk's last row reads the next
    chunk's first token, so the MTP head's K/V at every prompt slot, the
    first verify step's drafts (from the carry the final chunk stores) and the
    reply are the unchunked prompt's."""
    from qwen3_5_cpu import Driver, engine

    k = 3
    model = _model(tmp_path, monkeypatch, k=k)
    prompt = PROMPTS[3]
    mtp_layer = len(model.config.full_layer_indices)
    runs = []
    for bounds in ([len(prompt)], [4, 8, len(prompt)], [3, len(prompt)]):
        probe = Probe(monkeypatch)
        driver = Driver(engine(model), model)
        driver.add(1, prompt, MAX)
        driver.prefill_chunks(1, prompt, bounds)
        kv = driver.kv_rows(1, mtp_layer, len(prompt)).clone()
        driver.run([1])
        runs.append((kv, probe.steps[0][1], driver.out[1]))
    whole_kv, whole_drafts, whole_out = runs[0]
    for kv, drafts, out in runs[1:]:
        torch.testing.assert_close(kv, whole_kv, atol=1e-5, rtol=1e-5)
        assert torch.equal(drafts, whole_drafts) and out == whole_out


def test_a_prefix_hit_with_a_stale_mtp_boundary_row_keeps_greedy_parity(tmp_path, monkeypatch):
    """The MTP head's row at a cached page's last slot reads the next page's
    first token, which the page's key doesn't cover: a second prompt sharing
    two pages but not the token after them restores a row written for another
    continuation. The RFC's condition for keying an MTP node: its output is
    still plain greedy decoding's (only acceptance may suffer)."""
    from qwen3_5_cpu import PAGE, Driver, engine

    k = 3
    model = _model(tmp_path, monkeypatch, k=k)
    first = list(range(1, 13))
    second = first[:8] + [50, 51, 52, 53]
    driver = Driver(engine(model, prefix_cache=True), model)
    driver.add(1, first, MAX, keyed=True)
    assert driver.prefill([1], {1: first}, prefix=True) == 0
    driver.run([1])
    driver.add(2, second, MAX, keyed=True)
    assert driver.prefill([2], {2: second}, prefix=True) == 2 * PAGE
    driver.run([2])
    _check(model, driver.out, {1: first, 2: second})


def test_an_image_prompt_keeps_greedy_parity(tmp_path, monkeypatch):
    """An image prompt on a verifying node. The MTP head's rows over the
    image read the next row's embedding (an image token's, the end sentinel's),
    decode positions run on from the image's MRoPE advance (fewer positions
    than tokens: KV length plus a per-request offset), and the reply is plain
    greedy decoding's."""
    from qwen3_5_cpu import Driver, engine, reference_greedy_embeds, vision_config

    from mstar.model.qwen3_5.qwen3_5_model import MediaIntake, vision_encoder

    model = _model(tmp_path, monkeypatch, k=3)
    # the toy's checkpoint has no tower: the model takes images only as one with a tower does
    pad = model._pad_ids()
    model.media = (MediaIntake("image", vision_encoder(tuple(pad.values())), pad["image"], model._image_intake),)
    probe = Probe(monkeypatch)
    eng = engine(model)
    eng._submodules["LLM"].submodule.vision_config = vision_config()
    grid = (1, 8, 4)  # 4 x 2 merged tokens over max(4, 2) positions
    image = torch.randn(8, model.config.hidden_size, generator=torch.Generator().manual_seed(3))
    driver = Driver(eng, model)
    row = driver.add_image_prompt(1, [[5, 9, 2], [7, 11]], image, grid, MAX)
    assert row.input_seq_len > int(row.tensor_inputs["mrope_advance"])
    driver.run([1])
    margins: list[float] = []
    want = reference_greedy_embeds(model, row.input_embeds, row.custom_pos_ids, MAX, margins=margins)
    out = driver.out[1]
    n = min(len(want), len(out))
    at = next((j for j in range(n) if want[j] != out[j]), None if len(want) == len(out) else n)
    assert at is None or margins[at] < MARGIN, (at, want, out)
    # the drafts: the head ran over the image rows on their embeddings and 3D positions
    from qwen3_5_cpu import dense_lm

    embed = dense_lm(model).model.embed_tokens

    def prompt_of(rid, reply):
        tail = torch.tensor(reply)
        last = row.custom_pos_ids[:, -1:]
        positions = torch.cat([row.custom_pos_ids, last + torch.arange(1, len(reply) + 1)], dim=1)
        return torch.cat([row.input_embeds, embed(tail)]), positions

    assert _check_probe_drafts(model, probe, driver, 3, prompt_of)


def test_the_mtp_heads_checkpoint_names_fill_every_parameter(tmp_path, monkeypatch):
    """``mtp.*`` as the checkpoint ships it (separate q/k/v and gate/up, which
    load into the fused parameters) fills the head; the text tower's loader
    takes the prefix only for a model built with it."""
    from qwen3_5_cpu import build_lm

    from mstar.model.loader.base import load_weights_into
    from mstar.model.qwen3_5.weight_loader import _STACKED_PARAMS, qwen3_5_mtp_name_remapper, qwen3_5_name_remapper

    model = _model(tmp_path, monkeypatch, k=2)
    lm = build_lm(model)
    cfg = model.config
    gen = torch.Generator().manual_seed(1)
    shards = []
    for name, p in lm.mtp.named_parameters(prefix="mtp"):
        if ".qkv_proj." in name:
            q, kv = cfg.num_attention_heads * 2 * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
            for part, rows in (("q_proj", q), ("k_proj", kv), ("v_proj", kv)):
                shards.append((name.replace("qkv_proj", part), torch.randn(rows, p.shape[1], generator=gen)))
        elif ".gate_up_proj." in name:
            for part in ("gate_proj", "up_proj"):
                shards.append((name.replace("gate_up_proj", part),
                               torch.randn(cfg.intermediate_size, p.shape[1], generator=gen)))
        else:
            shards.append((name, torch.randn(p.shape, generator=gen)))
    loaded = load_weights_into(lm, iter(shards), name_remapper=qwen3_5_mtp_name_remapper,
                               stacked_params=_STACKED_PARAMS)
    assert {n for n, _ in lm.mtp.named_parameters(prefix="mtp")} <= loaded
    assert torch.equal(lm.mtp.fc.weight, dict(shards)["mtp.fc.weight"])
    assert qwen3_5_name_remapper("mtp.fc.weight") is None


def test_a_drafting_node_declares_the_verify_resources_and_refuses_what_it_cant(tmp_path, monkeypatch):
    from mstar.engine.resources.attn.config import AttentionSpec
    from mstar.engine.resources.draft.drafter import DRAFT_CARRY
    from mstar.engine.resources.kv.config import KVSpec
    from mstar.engine.resources.recurrent.config import RecurrentStateSpec
    from mstar.engine.resources.sampler.config import SamplerSpec

    plain = _model(tmp_path / "plain", monkeypatch)
    model = _model(tmp_path / "mtp", monkeypatch, k=3)
    specs = {type(s): s for s in model.get_node_resources()}
    base = {type(s): s for s in plain.get_node_resources()}
    kv = specs[KVSpec]
    assert kv.config.num_layers == base[KVSpec].config.num_layers + 1
    assert kv.draft_slack == 8 and kv.config.max_seq_len == base[KVSpec].config.max_seq_len + 8
    assert specs[AttentionSpec].config.splits == () and base[AttentionSpec].config.splits
    assert specs[RecurrentStateSpec].verify_refusal(None) is None
    sampler = specs[SamplerSpec]
    assert sampler.verifies and not sampler.enable_repetion_penalty
    assert sampler.loopback_state[DRAFT_CARRY].width == model.config.hidden_size
    assert model.get_combined_graph_walks() == {} and plain.get_combined_graph_walks()
    (decl,) = model.verify_nodes()
    assert decl.target.num_layers == len(model.config.full_layer_indices)
    assert plain.verify_nodes() == []
    from qwen3_5_cpu import _Tokenizer, make_model, write_checkpoint

    from mstar.model.qwen3_5 import qwen3_5_model

    monkeypatch.setattr(qwen3_5_model, "AutoTokenizer", _Tokenizer)
    with pytest.raises(ValueError, match="method mtp"):
        qwen3_5_model.Qwen3_5DenseModel(
            model_path_hf=str(write_checkpoint(tmp_path / "dspark")),
            draft={"method": "dspark", "model": "m", "num_draft_tokens": 2},
        )

    with pytest.raises(ValueError, match="one MTP layer"):
        make_model(monkeypatch, write_checkpoint(tmp_path / "none", mtp_num_hidden_layers=0), k=2)


class _Attn:
    def __init__(self, qo):
        self.qo = qo

    def qo_indptr_buf(self, label="main"):
        return self.qo

    # the real one, over this plan's offsets
    prompt_last_rows = AttentionManager.prompt_last_rows


def _prefill_inputs(tmp_path, monkeypatch, qo, next_ids, sampled):
    """What the MTP head gets from a prompt step: each row's following token's
    embedding, and the rows it keeps."""
    from qwen3_5_cpu import build_lm

    from mstar.model.qwen3_5.components.mtp import Qwen3_5MTPDrafter
    from mstar.model.qwen3_5.config import ATTN

    model = _model(tmp_path, monkeypatch, k=2)
    lm = build_lm(model)
    seen = {}

    def head(hidden, following, cos_sin, keep):
        seen.update(following=following, keep=keep)
        return hidden.index_select(0, keep)

    lm.mtp.forward = head
    drafter = Qwen3_5MTPDrafter(model.drafter_decl, lm)
    tokens = len(next_ids)
    hidden = torch.zeros(tokens, model.config.hidden_size)
    drafter.prefill({ATTN: _Attn(qo)}, hidden, None, next_ids=torch.tensor(next_ids), sampled=torch.tensor(sampled))
    embed = lm.model.embed_tokens.weight
    ids = [int((embed - row).abs().sum(-1).argmin()) for row in seen["following"]]
    return ids, seen["keep"].tolist()


def test_a_padded_prompt_steps_padding_rows_leave_the_real_rows_following_tokens(tmp_path, monkeypatch):
    """A captured packed step pads with rows of no tokens, whose last index is
    the last real row's: their sampled tokens go to a sink, not over it."""
    ids, keep = _prefill_inputs(
        tmp_path, monkeypatch, torch.tensor([0, 3, 7, 7, 7]), [5, 6, -1, 8, 9, 10, -1], [11, 22, 99, 98],
    )
    assert ids == [5, 6, 11, 8, 9, 10, 22]
    assert keep[:2] == [2, 6]


def test_a_prompt_step_of_one_token_rows_has_every_row_its_own_last(tmp_path, monkeypatch):
    """An eager step whose rows are one token each plans no query offsets."""
    ids, keep = _prefill_inputs(tmp_path, monkeypatch, None, [-1, -1, -1], [11, 22, 33])
    assert ids == [11, 22, 33] and keep == [0, 1, 2]


def test_a_drafting_node_refuses_video_prompts_and_serves_images(tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch, k=3)
    assert model.drafter_decl.needs.input_kinds == frozenset({"text", "image"})
    with pytest.raises(ValueError, match="video is not supported"):
        model.process_prompt("what happens?", ["text", "video"], ["text"])


def test_accelerator_graph_off_captures_nothing_on_the_llm_node(tmp_path, monkeypatch):
    from qwen3_5_cpu import build_lm

    model = _model(tmp_path, monkeypatch, k=3, accelerator_graph=False)
    sub = model.make_llm_submodule(build_lm(model))
    assert sub.get_accelerator_graph_configs(torch.device("cpu")) == []
    # nothing resident either, so mode auto resolves to synchronous rather than refusing
    assert sub.resident_walks() == frozenset()
    captured = _model(tmp_path / "on", monkeypatch, k=3).make_llm_submodule(build_lm(model))
    assert captured.get_accelerator_graph_configs(torch.device("cpu"))


def test_an_mtp_head_with_its_own_embedding_table_is_refused(tmp_path, monkeypatch):
    from qwen3_5_cpu import make_model, write_checkpoint

    with pytest.raises(ValueError, match="mtp_use_dedicated_embeddings"):
        make_model(monkeypatch, write_checkpoint(tmp_path, mtp_use_dedicated_embeddings=True), k=2)


# the CPU toy's pool with the MTP head's layer joined, pinned: the KV tensor, its sizing, its page groups, its prefix
# identity and the drafter's first layer
MTP_POOL = {
    "shape": [2, 512, 2, 4, 2, 16], "stride": [131072, 256, 128, 32, 16, 1],
    "num_units": 512, "request_pages": 511, "tokens_per_page": 4, "groups": [([0, 1], 4)],
    "fingerprint": "1db6f8778eedf8dbd2de3db86920d36c987aa148ccfd1e9766343a048f680780",
    "root": "c1f8744129d829bc0aea7b3f2bc3b1050609906e8a1465d29db3ff4c5783f057",
    "first_layer": 1,
}


def _facts(eng, first_layer: int) -> dict:
    kv = eng._resources["kv_cache"]
    config, tensor = kv.config, kv.kv_cache.tensor
    return {
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "num_units": config.num_units, "request_pages": config.request_pages,
        "tokens_per_page": config.tokens_per_page,
        "groups": [(list(group.layers), group.page_size) for group in config.groups],
        "fingerprint": kv.fingerprint().hex(), "root": kv._prefix_root.hex(),
        "first_layer": first_layer,
    }


def test_an_mtp_heads_layer_joins_the_pool(tmp_path, monkeypatch):
    import qwen3_5_cpu as q

    model = q.make_model(monkeypatch, q.write_checkpoint(tmp_path / "q35"), k=3)
    eng = q.engine(model, prefix_cache=True)
    head = next(m for m in eng._submodules["LLM"].submodule.modules() if type(m).__name__ == "Qwen3_5MTPHead")
    assert eng._resources["kv_cache"].config.layer_sets == {"target": (0, 1), "mtp": (1, 1)}
    assert _facts(eng, head.kv_layer_idx) == MTP_POOL
    # the carry the head declares is the sampler's loop-back state
    from mstar.engine.resources.draft.drafter import DRAFT_CARRY

    specs = {spec.resource_key: spec for spec in model.get_node_resources()}
    state = specs["sampler"].loopback_state[DRAFT_CARRY]
    assert (state.width, state.dtype) == (model.config.hidden_size, torch.float32)


def test_the_mtp_head_binds_its_set_first_layer_not_its_size(monkeypatch, tmp_path):
    """The CPU toy has one full-attention layer, so its "mtp" set's first layer equals its pool's
    target layer count and its own size both; a stub pool tells them apart."""
    import qwen3_5_cpu as q

    from mstar.distributed.communication import CommGroup
    from mstar.engine.resources.draft.drafter import UNBOUND_KV_LAYER
    from mstar.engine.resources.kv.config import LayerSetSlot
    from mstar.model.qwen3_5.components.mtp import Qwen3_5MTPHead

    model = q.make_model(monkeypatch, q.write_checkpoint(tmp_path / "q35"), k=3)
    with torch.device("meta"):
        head = Qwen3_5MTPHead(model.config, comm_group=CommGroup.trivial())
    assert head.kv_layer_idx == UNBOUND_KV_LAYER and head.kv_layers_unbound

    class _Pool:
        def layer_set(self, name):
            return {"target": LayerSetSlot(0, 6), "mtp": LayerSetSlot(6, 1)}[name]

    head.bind_resources({"kv_cache": _Pool()})
    assert head.kv_layer_idx == 6 and not head.kv_layers_unbound
