"""The Grafana dashboard in examples/observability queries only metric
families that docs/metrics.rst documents, shows every one of them, and is
laid out so Grafana loads it as written."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "examples/observability/grafana/dashboards/mstar.json"
STRING = r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`[^`]*`'
MATCHERS = re.compile(r"\{(?:[^{}\"'`]|" + STRING + r")*\}")
NAME_MATCHER = re.compile(r"""\b__name__\s*=~?\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')""")
# What is left of a PromQL expression without its label matchers, strings,
# ranges and grouping label lists: every identifier there is a metric name,
# unless a "(" follows it (a function) or it is a keyword.
NOT_METRICS = re.compile(
    STRING + r"|\[[^\]]*\]|\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)")
IDENTIFIER = re.compile(r"(?<![\w:$])[A-Za-z_:][\w:]*\b(?!\s*\()")
KEYWORDS = {"and", "or", "unless", "bool", "offset", "group_left", "group_right", "inf", "nan"}
HISTOGRAM_SERIES = re.compile(r"_(bucket|sum|count)$")
# A family entry in docs/metrics.rst: ``mstar_name{labels}`` on its own line.
DOC_ENTRY = re.compile(r"^``(mstar_\w+)\{", re.M)


def _documented() -> set[str]:
    return set(DOC_ENTRY.findall((ROOT / "docs/metrics.rst").read_text()))


def _panels():
    return json.loads(DASHBOARD.read_text())["panels"]


def _metric_names(expr: str) -> set[str]:
    names = {a or b for block in MATCHERS.findall(expr) for a, b in NAME_MATCHER.findall(block)}
    rest = NOT_METRICS.sub(" ", MATCHERS.sub(" ", expr))
    return names | {name for name in IDENTIFIER.findall(rest) if name.lower() not in KEYWORDS}


def _queried() -> set[str]:
    """Metric names in every panel query, histogram series folded to their family."""
    documented = _documented()
    names = set()
    for panel in _panels():
        for target in panel.get("targets", []):
            for series in _metric_names(target["expr"]):
                family = HISTOGRAM_SERIES.sub("", series)
                names.add(family if family in documented else series)
    return names


@pytest.mark.parametrize(("expr", "names"), [
    ('sum by (outcome) (rate(mstar_requests_total{model_name=~"$model"}[1m]))', {"mstar_requests_total"}),
    ("histogram_quantile(0.95, sum by (le) (rate(mstar_step_seconds_bucket[$__rate_interval])))",
     {"mstar_step_seconds_bucket"}),
    ("(sum(rate(mstar_http_refusals_total[1m])) or vector(0)) / on() group_left up",
     {"mstar_http_refusals_total", "up"}),
    ("rate(mstar_requests_totalX[1m])", {"mstar_requests_totalX"}),
    ('label_replace(node_load1, "dst", "$1", "src", "(.*)") > bool 1e3', {"node_load1"}),
    ('mstar_requests_total{label="foo}bar"}', {"mstar_requests_total"}),
    ('mstar_requests_total{label="}"} + undocumented_metric{label="or"}',
     {"mstar_requests_total", "undocumented_metric"}),
    ('sum({__name__="undocumented_metric", a="b"})', {"undocumented_metric"}),
    (r'mstar_requests_total{a="x\"}y"} / label_join(up, "x", \'}\', "a")', {"mstar_requests_total", "up"}),
])
def test_metric_names_are_read_whole_from_an_expression(expr, names):
    assert _metric_names(expr) == names


def test_every_query_names_a_documented_family():
    undocumented = _queried() - _documented()
    assert not undocumented, f"queried but not in docs/metrics.rst: {sorted(undocumented)}"


def test_every_documented_family_has_a_panel():
    missing = _documented() - _queried()
    assert not missing, f"documented but on no panel: {sorted(missing)}"


def test_panels_use_the_provisioned_data_source_and_do_not_overlap():
    panels = _panels()
    ids = [p["id"] for p in panels]
    assert len(ids) == len(set(ids))
    cells = {}
    for p in panels:
        if p["type"] != "row":
            assert p["datasource"] == {"type": "prometheus", "uid": "prom"}, p["title"]
            assert all(t["datasource"] == p["datasource"] for t in p["targets"]), p["title"]
        g = p["gridPos"]
        assert 0 <= g["x"] and g["x"] + g["w"] <= 24, p["title"]
        for x in range(g["x"], g["x"] + g["w"]):
            for y in range(g["y"], g["y"] + g["h"]):
                assert (x, y) not in cells, f"{p['title']} overlaps {cells[(x, y)]}"
                cells[(x, y)] = p["title"]
