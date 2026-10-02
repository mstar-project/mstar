"""The deployment yaml's ``env:`` section: exported for the spawned processes,
never overriding a variable the launch already set."""
import os

from mstar.api_server.entrypoint import _apply_config_env


def test_env_section_sets_missing_variables_and_keeps_existing(monkeypatch):
    # set-then-delete so monkeypatch also undoes the value _apply_config_env writes
    monkeypatch.setenv("MSTAR_TEST_CFG_A", "")
    monkeypatch.delenv("MSTAR_TEST_CFG_A")
    monkeypatch.setenv("MSTAR_TEST_CFG_B", "from-launch")
    _apply_config_env({"env": {"MSTAR_TEST_CFG_A": 1, "MSTAR_TEST_CFG_B": "from-yaml"}}, "x.yaml")
    assert os.environ["MSTAR_TEST_CFG_A"] == "1"
    assert os.environ["MSTAR_TEST_CFG_B"] == "from-launch"


def test_missing_or_empty_env_section_is_a_no_op(monkeypatch):
    before = dict(os.environ)
    _apply_config_env({}, "x.yaml")
    _apply_config_env({"env": None}, "x.yaml")
    assert dict(os.environ) == before
