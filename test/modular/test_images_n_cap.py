"""An image request's ``n`` is refused outside ``1..MSTAR_MAX_IMAGES_PER_REQUEST``
before anything is submitted (``image_count``, which the Dynamo bridge uses).
mstar-server's ``/v1/images/generations`` twin is in test/rust/test_rust_frontend.py."""

from types import SimpleNamespace

import pytest

from mstar.api_server.openai.serving_images import image_count, max_images_per_request


@pytest.fixture(autouse=True)
def _default_cap(monkeypatch):
    monkeypatch.delenv("MSTAR_MAX_IMAGES_PER_REQUEST", raising=False)


def _req(**fields):
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("n", [0, -1, 11])
def test_n_outside_one_to_the_cap_is_refused(n):
    with pytest.raises(ValueError, match=rf"^n must be between 1 and 10, got {n}$"):
        image_count(_req(n=n))


def test_n_at_the_cap_is_that_many_images():
    assert image_count(_req(n=10)) == 10


@pytest.mark.parametrize("fields", [{}, {"n": None}], ids=["omitted", "null"])
def test_n_omitted_is_one_image(fields):
    assert image_count(_req(**fields)) == 1


def test_the_cap_follows_the_env(monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_IMAGES_PER_REQUEST", "2")
    with pytest.raises(ValueError, match=r"^n must be between 1 and 2, got 3$"):
        image_count(_req(n=3))
    assert image_count(_req(n=2)) == 2


@pytest.mark.parametrize("raw, cap", [
    ("2", 2), (" 2 ", 2), ("2\n", 2), ("+3", 3), ("0", 1), ("-1", 1),
    ("", 10), ("abc", 10), ("2_0", 10), ("1.5", 10), ("２", 10),
])
def test_the_cap_parses_as_the_rust_frontend_does(monkeypatch, raw, cap):
    # rust/server/src/main.rs::max_images_parses_as_python_does checks the same table
    monkeypatch.setenv("MSTAR_MAX_IMAGES_PER_REQUEST", raw)
    assert max_images_per_request() == cap


def test_a_non_positive_cap_still_serves_one_image(monkeypatch):
    monkeypatch.setenv("MSTAR_MAX_IMAGES_PER_REQUEST", "0")
    for fields in ({}, {"n": None}, {"n": 1}):
        assert image_count(_req(**fields)) == 1, fields
    with pytest.raises(ValueError):
        image_count(_req(n=2))
