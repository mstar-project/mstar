"""The host table resource alone: a step's rows by id, out of a checkpoint's mapped bytes."""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import pytest
import torch
from safetensors.torch import save_file

from mstar.engine.resources import StepRunner, SubmoduleStep, apply_yaml_overrides
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.host_table.config import (
    HostTable,
    HostTableConfig,
    HostTablePart,
    HostTableSpec,
    HostTableStep,
)
from mstar.engine.resources.step import StepContext

KEY = "tables"
ROWS = 1000
# a table of two parts, as a quantized embedding is stored, and one of wider elements
TABLES = {
    "embed": HostTable(rows=ROWS, parts=(HostTablePart("embed.weight", 32), HostTablePart("embed.scale", 4))),
    "codes": HostTable(rows=77, parts=(HostTablePart("codes", 6),)),
}


def _tensors() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)

    def raw(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    return {
        "embed.weight": raw(ROWS, 32).view(torch.float8_e4m3fn),
        "embed.scale": raw(ROWS, 4).view(torch.float8_e8m0fnu),
        "codes": raw(77, 6).view(torch.int16),
        "other": torch.randn(5, 3, generator=g),
    }


def _raw(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor``'s rows as the bytes the checkpoint holds."""
    return tensor.contiguous().view(torch.uint8).view(tensor.shape[0], -1)


@pytest.fixture(scope="module", params=["one-file", "shards"])
def checkpoint(request, tmp_path_factory):
    """A checkpoint directory holding ``_tensors``, in one file or in indexed shards."""
    path, tensors = tmp_path_factory.mktemp("tables"), _tensors()
    if request.param == "one-file":
        save_file(tensors, str(path / "model.safetensors"))
    else:
        shards = {"model-00001-of-00002.safetensors": ("other", "embed.scale", "codes"),
                  "model-00002-of-00002.safetensors": ("embed.weight",)}
        for name, held in shards.items():
            save_file({key: tensors[key] for key in held}, str(path / name))
        (path / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {key: name for name, held in shards.items() for key in held}}))
    return path, tensors


def _spec(path, tables=None, **config) -> HostTableSpec:
    return HostTableSpec(resource_key=KEY, nodes={"node"}, config=HostTableConfig(
        checkpoint=str(path), tables=dict(TABLES if tables is None else tables), **config))


def _build(path, **config):
    return build_resource(_spec(path, **config), EngineResourceInfo(device=torch.device("cpu")))


def _plan(manager, slot: int = 0) -> None:
    manager.plan(HostTableStep(), StepContext(request_ids=(), graph_walk="walk", slot=slot, capture=False))


def _ids(n: int, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randint(0, n, shape, generator=torch.Generator().manual_seed(seed))


class TestRows:
    def test_rows_by_id_are_the_checkpoints_bytes(self, checkpoint):
        path, tensors = checkpoint
        manager = _build(path)
        _plan(manager)
        embed, codes = _ids(ROWS, (7, 5), seed=1), _ids(77, (11,), seed=2)
        manager.fetch({"embed": embed, "codes": codes})
        values, scales = manager.rows("embed")
        assert values.dtype == scales.dtype == torch.uint8
        assert values.shape == (7, 5, 32) and scales.shape == (7, 5, 4)
        assert torch.equal(values, _raw(tensors["embed.weight"])[embed])
        assert torch.equal(scales, _raw(tensors["embed.scale"])[embed])
        assert torch.equal(manager.rows("codes")[0].view(torch.int16), tensors["codes"][codes])

    def test_a_step_wider_than_the_staging_is_copied_in_pieces(self, checkpoint):
        path, tensors = checkpoint
        manager = _build(path, staging_bytes=3 * 32 + 5)  # three rows of the widest part at a time
        _plan(manager)
        ids = _ids(ROWS, (50,), seed=3)
        manager.fetch({"embed": ids})
        values, scales = manager.rows("embed")
        assert torch.equal(values, _raw(tensors["embed.weight"])[ids])
        assert torch.equal(scales, _raw(tensors["embed.scale"])[ids])

    @pytest.mark.parametrize("bad", [ROWS, ROWS + 5, -1])
    def test_an_id_outside_the_table_is_refused(self, checkpoint, bad):
        manager = _build(checkpoint[0])
        _plan(manager)
        with pytest.raises(IndexError, match=f"has {ROWS} rows"):
            manager.fetch({"embed": torch.tensor([0, bad, 3])})

    def test_an_empty_step_gets_empty_rows(self, checkpoint):
        manager = _build(checkpoint[0])
        _plan(manager)
        manager.fetch({})
        manager.fetch({"embed": torch.empty(0, 5, dtype=torch.long)})
        values, scales = manager.rows("embed")
        assert values.shape == (0, 5, 32) and scales.shape == (0, 5, 4)
        with pytest.raises(KeyError, match="codes"):
            manager.rows("codes")  # not fetched this step


class TestSteps:
    def test_two_steps_back_to_back_keep_their_own_rows(self, checkpoint):
        """The step before may still be reading its rows when the next one's are staged, so
        the engine gives the two different slots (``force_double_buffer``)."""
        path, tensors = checkpoint
        manager = _build(path)
        assert manager.force_double_buffer
        table = _raw(tensors["embed.weight"])
        runner = StepRunner({KEY: manager})
        first, second, third = (_ids(ROWS, (n, 3), seed=n) for n in (4, 9, 6))

        def run(slot: int, ids: torch.Tensor) -> torch.Tensor:
            step = SubmoduleStep(steps={KEY: HostTableStep()})
            step.set_ctx(StepContext(request_ids=(), graph_walk="walk", slot=slot, capture=False))
            assert runner.admit(step).ok
            runner.plan(step)
            manager.fetch({"embed": ids})
            rows = manager.rows("embed")[0]
            runner.commit(step)
            return rows

        rows_first = run(0, first)
        rows_second = run(1, second)
        assert torch.equal(rows_first, table[first]) and torch.equal(rows_second, table[second])
        # a slot's buffers are the next step's two steps later, whatever its width
        assert torch.equal(run(0, third), table[third])
        assert torch.equal(rows_second, table[second])

    def test_a_fetch_outside_a_step_is_refused(self, checkpoint):
        manager = _build(checkpoint[0])
        ids = {"embed": torch.tensor([1])}
        with pytest.raises(RuntimeError, match="HostTableStep"):
            manager.fetch(ids)
        step, ctx = HostTableStep(), StepContext(request_ids=(), graph_walk="walk", slot=0, capture=False)
        manager.plan(step, ctx)
        manager.fetch(ids)
        manager.commit(step, ctx)
        with pytest.raises(RuntimeError, match="HostTableStep"):
            manager.fetch(ids)

    def test_its_steps_run_eager(self, checkpoint):
        assert not _build(checkpoint[0]).can_capture()


class TestDeclaration:
    def test_a_table_the_checkpoint_holds_in_another_shape_is_refused(self, checkpoint):
        wrong_rows = {"embed": HostTable(rows=ROWS + 1, parts=TABLES["embed"].parts)}
        wrong_width = {"embed": HostTable(rows=ROWS, parts=(HostTablePart("embed.weight", 16),))}
        for tables in (wrong_rows, wrong_width):
            with pytest.raises(ValueError, match="embed.weight is F8_E4M3"):
                _build(checkpoint[0], tables=tables)

    def test_a_tensor_the_checkpoint_lacks_is_refused(self, checkpoint):
        with pytest.raises(KeyError, match="no tensor 'missing'"):
            _build(checkpoint[0], tables={"t": HostTable(rows=3, parts=(HostTablePart("missing", 8),))})

    def test_a_deployment_sizes_the_staging_and_the_prefault(self, checkpoint):
        spec = _spec(checkpoint[0])
        assert spec.config.prefault and spec.config.staging_bytes == 4 << 20
        apply_yaml_overrides([spec], {"resources": {KEY: {"staging_bytes": 1 << 16, "prefault": False}}})
        assert not spec.config.prefault and spec.config.staging_bytes == 1 << 16
        for bad in ({"staging_bytes": 31}, {"prefault": "no"}, {"rows": 5}):
            with pytest.raises(ValueError):
                apply_yaml_overrides([_spec(checkpoint[0])], {"resources": {KEY: bad}})
