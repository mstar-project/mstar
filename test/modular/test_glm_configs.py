"""The GLM serving configs against docs/models.rst: its deployments table names every one, and
each config, with the overrides the docs give for it applied, starts and holds its requests."""
import math
import re
import textwrap
from pathlib import Path

import pytest
import yaml

from mstar.engine.accelerator_graph_runner import DEFAULT_CAPTURE_BATCH_SIZES
from mstar.model.registry import get_model_class, model_init_kwargs
from mstar.validation import validate_deployment

ROOT = Path(__file__).resolve().parents[2]
DOCS = (ROOT / "docs" / "models.rst").read_text()
MODELS = {"GLM-5.3-Flash": "glm5_next", "GLM-5.2-FP8": "glm52"}
KV_KEY = {"glm5_next": "kv_cache", "glm52": "kv"}


def _table() -> dict[str, dict[str, str]]:
    """Config file name -> its row of the "GLM deployments" table, by column header."""
    rows: list[list[str]] = []
    for line in DOCS.split(".. list-table:: GLM deployments\n", 1)[1].splitlines():
        if line and not line.startswith(" "):
            break
        if line.startswith("   * - "):
            rows.append([line[7:]])
        elif line.startswith("     - "):
            rows[-1].append(line[7:])
    header, *body = rows
    return {re.fullmatch(r"``configs/(.+)``", row[0])[1]: dict(zip(header, row, strict=True)) for row in body}


def _overrides() -> dict[str, dict[str, dict]]:
    """Model -> what the override is for -> the YAML block docs/models.rst gives for it (a
    block whose first line is ``# <model>: <what>``)."""
    found: dict[str, dict[str, dict]] = {}
    for block in re.findall(r"^\.\. code-block:: yaml\n\n((?:(?:   .*)?\n)+)", DOCS, re.M):
        body = textwrap.dedent(block)
        if m := re.match(r"# (glm\w+): (.+)\n", body):
            found.setdefault(m[1], {})[m[2]] = yaml.safe_load(body)
    return found


TABLE = _table()
OVERRIDES = _overrides()
GREEDY, LONG = "greedy requests (MTP)", "long context"
# each config plain, with each override, and with both (MTP first, as the docs say)
DEPLOYMENTS = [
    pytest.param(name, overrides, id="-".join((name, *overrides)))
    for name in TABLE
    for overrides in ((), (GREEDY,), (LONG,), (GREEDY, LONG))
]


def _merged(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        nested = isinstance(value, dict) and isinstance(base.get(key), dict)
        out[key] = _merged(base[key], value) if nested else value
    return out


def _config(name: str, overrides: tuple[str, ...] = ()) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
    for what in overrides:
        cfg = _merged(cfg, OVERRIDES[cfg["model"]][what])
    return cfg


def _model(cfg: dict):
    """The model as the conductor builds it from the config; no checkpoint is read."""
    name = cfg["model"]
    return get_model_class(name)(**{**model_init_kwargs(name), **cfg.get("model_kwargs", {})})


def test_the_table_lists_every_glm_config():
    assert set(TABLE) == {p.name for p in (ROOT / "configs").glob("glm*.yaml")}
    assert set(OVERRIDES) == {"glm5_next", "glm52"}
    for model in OVERRIDES.values():
        assert set(model) == {GREEDY, LONG}


@pytest.mark.parametrize("name", sorted(TABLE))
def test_config_matches_its_row(name):
    row, cfg = TABLE[name], _config(name)
    assert cfg["model"] == MODELS[row["Model"]]
    gpus = int(re.match(r"(\d+)x ", row["GPUs"])[1])
    for group in cfg["node_groups"]:
        assert sorted(group["ranks"]) == list(range(gpus)) and group["tp_size"] == gpus
    assert f"TP{gpus}" in row["GPUs"]
    mtp = 0 if row["MTP"] == "off" else int(row["MTP"].removeprefix("k="))
    assert cfg["model_kwargs"].get("mtp_num_draft_tokens", 0) == mtp
    assert cfg["max_seq_len"] == int(row["Context (tokens)"])


@pytest.mark.parametrize("name, overrides", DEPLOYMENTS)
def test_deployment_starts(name, overrides, tmp_path):
    cfg = _config(name, overrides)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg))
    model = _model(cfg)
    validate_deployment(model, cfg, str(path))
    assert model.config.dsa_long_context == (LONG in overrides)
    if LONG in overrides:
        assert cfg["model_kwargs"]["max_seq_len"] == cfg["max_seq_len"] > model.config.index_topk
    assert (model.config.mtp_num_draft_tokens > 0) == (GREEDY in overrides)


@pytest.mark.parametrize("name, overrides", DEPLOYMENTS)
def test_pools_hold_every_request(name, overrides):
    """No admitted request finds the KV pool (or GLM-5.3's KDA pool) full: the short configs
    hold every request's whole window, the long-context ones one full window. Each pool keeps
    one page (slot) back as its sink."""
    cfg = _config(name, overrides)
    model, requests = _model(cfg), cfg["max_concurrent_requests"]
    # past the largest decode capture the scheduler splits decode into two steps
    assert requests <= max(DEFAULT_CAPTURE_BATCH_SIZES)
    kv = cfg["resources"][KV_KEY[cfg["model"]]]
    usable = kv["max_num_pages"] - 1
    if cfg["model"] == "glm5_next":
        assert cfg["resources"]["kda_state"]["max_slots"] - 1 == requests
        rows = model.config.kv_rows
    else:
        # an MTP verify step writes its block of k + 1 rows past the context before the overwrite
        k = model.config.mtp_num_draft_tokens
        rows = model.context_limit() + (k + 1 if k else 0)
    per_request = math.ceil(rows / kv["page_size"])
    need = per_request if model.config.dsa_long_context else requests * per_request
    assert usable >= need


@pytest.mark.parametrize("name, overrides", [d for d in DEPLOYMENTS if d.values[0].startswith("glm5_next")])
def test_glm53_prefill_ladder_takes_every_step(name, overrides):
    """A prefill step takes up to the largest capture batch of prompts; at 2048 tokens the
    ladder covers the largest such step, and a captured DSA prefill plans at most MAX_ROWS."""
    from mstar.engine.resources.dsa import sparse_mla
    from mstar.model.glm5_next.submodules import PREFILL_CAPTURE_BATCH_SIZES, PREFILL_TOKEN_BUCKETS

    cfg = _config(name, overrides)
    model = _model(cfg)
    top = max(model.config.prefill_token_buckets or PREFILL_TOKEN_BUCKETS)
    rows = max(model.config.prefill_capture_batch_sizes or PREFILL_CAPTURE_BATCH_SIZES)
    if model.config.dsa_long_context:
        assert top <= sparse_mla.MAX_ROWS
    else:
        assert top >= rows * model.config.context_limit
