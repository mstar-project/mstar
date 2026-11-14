"""The CSA resource: its selection rules, and the rows its caches advance by."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.csa.config import INDEX_PLANES, CSAConfig, CSASpec, Fp4Rows
from mstar.engine.resources.csa.manager import CSAManager, _load, _store, candidate_blocks, top_scores
from mstar.engine.resources.kv.config import AuxPlanes, KVLayout, KVSpec, PagedKVConfig
from mstar.engine.resources.step import Segment, StepContext
from mstar.utils.mx_round import round_fp4

NEG = float("-inf")


class TestSelection:
    def test_equal_scores_go_to_the_earlier_group_whatever_the_width(self):
        scores = torch.tensor([[0.0, 2.0, 0.0, 0.0, 1.0, 0.0]])
        wide = torch.cat([scores, torch.full((1, 7), NEG)], dim=1)
        for s in (scores, wide):
            values, indices = top_scores(s, 4)
            assert indices.tolist() == [[1, 4, 0, 2]] and values.tolist() == [[2.0, 1.0, 0.0, 0.0]]

    def test_candidates_keep_the_best_blocks_and_the_newest(self):
        # 10 visible groups in blocks of 4: [0-3] best 5, [4-7] best 9, [8-9] the newest
        scores = torch.tensor([[5.0, 0, 0, 0, 1, 9, 1, 1, -3, -4, NEG, NEG]])
        keep = candidate_blocks(scores, torch.tensor([10]), topk_blocks=2, block_size=4)
        # the newest block is kept though it scores lowest, then the best of the others
        assert keep.tolist() == [[False] * 4 + [True] * 8]

    def test_a_row_with_fewer_blocks_than_topk_keeps_only_what_it_sees(self):
        scores = torch.tensor([[1.0, 2.0, 3.0, NEG, NEG, NEG, NEG, NEG]])
        keep = candidate_blocks(scores, torch.tensor([3]), topk_blocks=2, block_size=4)
        assert keep.tolist() == [[True] * 4 + [False] * 4]

    def test_a_row_that_sees_nothing_keeps_nothing(self):
        keep = candidate_blocks(torch.full((1, 8), NEG), torch.tensor([0]), topk_blocks=2, block_size=4)
        assert not keep.any()


class TestTokenClock:
    """A cache of ratio r advances a row every r tokens of a request, wherever a step starts."""

    @staticmethod
    def _manager() -> CSAManager:
        config = CSAConfig(
            caches={1: "every", 2: "pairs", 4: "fours"}, topk=4, sm_scale=1.0, index_head_dim=8, latent_dim=32)
        return CSAManager(config, {1: 16, 2: 16, 4: 16}, torch.device("cpu"))

    @staticmethod
    def _commit(manager: CSAManager, segments) -> None:
        from mstar.engine.resources.csa.config import CSAStep

        manager.commit(CSAStep(segments=tuple(segments)), StepContext(
            request_ids=tuple(s.request_id for s in segments), graph_walk="decode", slot=0, capture=False))

    def test_rows_are_the_groups_a_steps_tokens_complete(self):
        manager = self._manager()
        spans = {key: [s.span for s in rows] for key, rows in manager.kv_segments(
            [Segment("a", "main", 7), Segment("b", "main", 1), Segment("c", "main", 0)]).items()}
        assert spans == {"every": [7, 1, 0], "pairs": [3, 0, 0], "fours": [1, 0, 0]}

    def test_a_requests_steps_add_up_to_its_tokens_over_the_ratio(self):
        manager = self._manager()
        total = {"every": 0, "pairs": 0, "fours": 0}
        tokens = 0
        for n in (3, 1, 1, 6, 1, 1, 1, 9):
            step = [Segment("a", "main", n)]
            for key, rows in manager.kv_segments(step).items():
                total[key] += rows[0].span
            self._commit(manager, step)
            tokens += n
            assert total == {"every": tokens, "pairs": tokens // 2, "fours": tokens // 4}

    def test_a_declared_step_moves_nothing_until_it_commits(self):
        manager = self._manager()
        step = [Segment("a", "main", 3)]
        first = manager.kv_segments(step)
        assert manager.kv_segments(step) == first  # declared again, as a retried admit does
        self._commit(manager, step)
        assert manager.kv_segments([Segment("a", "main", 1)])["pairs"][0].span == 1

    def test_a_removed_request_starts_over(self):
        manager = self._manager()
        self._commit(manager, [Segment("a", "main", 3)])
        manager.remove_request("a")
        assert manager.kv_segments([Segment("a", "main", 1)])["pairs"][0].span == 0

    def test_tokens_taken_back_give_each_cache_back_the_groups_they_completed(self):
        manager = self._manager()
        moved = []

        class Cache:
            def __init__(self, key):
                self.key = key

            def correct_len(self, rid, label, delta):
                moved.append((self.key, rid, label, delta))

        manager.bind_dependencies({key: Cache(key) for key in ("every", "pairs", "fours")})
        self._commit(manager, [Segment("a", "main", 9)])
        manager.take_back("a", 3)  # 9 tokens to 6: 4 pairs to 3, 2 fours to 1
        assert sorted(moved) == [("every", "a", "main", -3), ("fours", "a", "main", -1), ("pairs", "a", "main", -1)]
        # the next step stands where the count was left
        spans = {key: rows[0].span for key, rows in manager.kv_segments([Segment("a", "main", 2)]).items()}
        assert spans == {"every": 2, "pairs": 1, "fours": 1}
        moved.clear()
        manager.take_back("a", 1)  # 6 to 5: a pair opens again, the four stays
        manager.take_back("a", 0)
        assert sorted(moved) == [("every", "a", "main", -1), ("pairs", "a", "main", -1)]
        for tokens in (6, -1, True):
            with pytest.raises(ValueError, match="cannot take"):
                manager.take_back("a", tokens)

    @pytest.mark.parametrize("caches", [{}, {0: "x"}, {1: "x", 2: "x"}])
    def test_a_config_maps_each_ratio_to_its_own_cache(self, caches):
        with pytest.raises(ValueError, match="caches"):
            CSAConfig(caches=caches, topk=4, sm_scale=1.0, index_head_dim=8, latent_dim=32)


class TestFp4Rows:
    """A cache of ``Fp4Rows`` rounds the rows it is given and holds them in their FP4 bytes."""

    LATENT, KEY = Fp4Rows(block=16, e4m3_scale=True), Fp4Rows(block=32)

    def test_bytes_a_row(self):
        # DeepSeek-V4.1's latent and index key, against 1,024 and 256 bytes in bf16
        assert (self.LATENT.nbytes(512), self.KEY.nbytes(128)) == (288, 68)
        with pytest.raises(ValueError, match="scale blocks"):
            self.KEY.nbytes(48)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("fmt, width", [(LATENT, 64), (KEY, 64)], ids=["latent", "key"])
    def test_rows_read_back_rounded_from_a_plane_of_either_dtype(self, dtype, fmt, width):
        torch.manual_seed(0)
        values = (torch.randn(40, width) * torch.exp(torch.linspace(-6, 6, 40))[:, None]).to(dtype)
        # a plane row as its cache declares it: the bytes in whole elements, with room to spare
        plane = torch.empty(4, 16, fmt.nbytes(width) // dtype.itemsize + 3, dtype=dtype)
        rows = _store(values, fmt, plane)
        assert rows.dtype == dtype and rows.shape == (40, plane.shape[-1])
        assert torch.equal(_load(rows, fmt, width), round_fp4(values, fmt.block, fmt.e4m3_scale).float())
        # without a format, a row is the values written
        assert _store(values, None, plane) is values and torch.equal(_load(values, None, width), values.float())

    def _build(self, latent_width: int, key_width: int, **rows):
        kv = KVSpec(resource_key="kv", nodes={"n"}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=latent_width, max_seq_len=64, layout=KVLayout.MLA,
            kv_lora_rank=latent_width, qk_rope_head_dim=0,
            aux_planes={INDEX_PLANES: AuxPlanes(num_layers=1, width=key_width)}))
        spec = CSASpec(resource_key="csa", nodes={"n"}, config=CSAConfig(
            caches={1: "kv"}, topk=4, sm_scale=1.0, index_head_dim=128, latent_dim=512, **rows))
        return CSAManager.build(spec, EngineResourceInfo(device=torch.device("cpu"), dependencies={"kv": kv}))

    def test_a_packed_cache_declares_its_bytes_in_the_caches_elements(self):
        packed = dict(latent_rows=self.LATENT, key_rows=self.KEY)
        self._build(144, 34, **packed)  # 288 and 68 bytes in bf16 elements
        with pytest.raises(ValueError, match="kv_lora_rank as 144"):
            self._build(143, 34, **packed)
        with pytest.raises(ValueError, match="34 torch.bfloat16 values wide"):
            self._build(144, 33, **packed)

    def test_an_unpacked_cache_is_as_wide_as_its_values(self):
        self._build(512, 128)
        with pytest.raises(ValueError, match="kv_lora_rank as 512"):
            self._build(144, 128)


class _Planes:
    """A cache's latent plane, as the resource reads it."""

    def __init__(self, latents: torch.Tensor):
        self.latents = latents

    def layer_view(self, plane: int) -> torch.Tensor:
        return self.latents


class TestAttend:
    """The softmax over a row's window rows and picks is the arithmetic of the kernel
    DeepSeek-V4.1 is served with: 64 entries at a time against the running maximum, each block's
    probabilities rounded to the queries' dtype, a sink in the denominator."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    # entries a row: part of a block, exactly one, and several with the last one partly filled
    @pytest.mark.parametrize("window, picks", [(5, 0), (16, 48), (8, 70), (60, 150)])
    def test_it_is_the_reference_kernels_block_loop_bit_for_bit(self, dtype, window, picks):
        from mstar.model.deepseek_v41._testing import sparse_attn

        torch.manual_seed(0)
        rows, heads, dim, slots_total = 6, 4, 32, 160
        fmt = Fp4Rows(block=16, e4m3_scale=True)
        config = CSAConfig(caches={1: "kv"}, topk=max(picks, 1), sm_scale=dim ** -0.5, index_head_dim=32,
                           latent_dim=dim, latent_rows=fmt)
        manager = CSAManager(config, {1: 16}, torch.device("cpu"))
        plane = torch.empty(slots_total // 16, 16, fmt.nbytes(dim) // dtype.itemsize + 1, dtype=dtype)
        latents = torch.randn(slots_total, dim) * 2
        plane.view(-1, plane.shape[-1])[:] = _store(latents, fmt, plane)
        manager._kv = {1: _Planes(plane)}
        q = torch.randn(rows, heads, dim).to(dtype)
        sink = torch.randn(heads)
        source = torch.randn(rows * window, dim).to(dtype)
        cols = torch.tensor([window, window - 1, 1, 0, window, 2][:rows]).clamp(0, window)
        window_idx = torch.arange(rows)[:, None] * window + torch.arange(window)[None, :]
        window_idx = torch.where(torch.arange(window)[None, :] < cols[:, None], window_idx, -1)
        slots = None
        if picks:
            slots = torch.stack([torch.randperm(slots_total)[:picks].sort().values for _ in range(rows)])
            live = torch.tensor([picks, picks // 2, 0, 0, 1, picks][:rows])
            slots = torch.where(torch.arange(picks)[None, :] < live[:, None], slots, -1)
        got = manager.attend(q, sink, source, window_idx, cols, 1 if picks else None, 0 if picks else None, slots)

        # the same entries, dense, through the kernel's loop: a row's window columns, then its picks
        entries = source[window_idx.clamp_min(0)]
        idx = torch.where(window_idx >= 0, torch.arange(window)[None, :], -1)
        if picks:
            picked = _load(plane.view(-1, plane.shape[-1])[slots.clamp_min(0)], fmt, dim, dtype)
            entries = torch.cat([entries, picked], dim=1)
            idx = torch.cat([idx, torch.where(slots >= 0, window + torch.arange(picks)[None, :], -1)], dim=1)
        # packed as the resource packs a row: its live window columns first
        order = (idx < 0).sort(dim=1, stable=True).indices
        entries, idx = entries.gather(1, order[..., None].expand_as(entries)), idx.gather(1, order)
        idx = torch.where(idx >= 0, torch.arange(idx.shape[1])[None, :], -1).int()
        want = sparse_attn(q[:, None], entries, sink, idx[:, None], dim ** -0.5)[:, 0]
        assert got.dtype == dtype and torch.equal(got, want)
        assert not got[3].any()  # a row with nothing to attend
