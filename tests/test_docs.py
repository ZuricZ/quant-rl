"""Documentation lookups: what they show and what they refuse."""

import importlib

import polars as pl
import pytest

from quantenv import checks, docs

PACKAGES = {"pl": "polars", "polars": "polars", "np": "numpy", "numpy": "numpy"}
LIMIT = 2000


def test_function_gets_its_signature_and_docstring():
    text = docs.lookup("polars.Expr.rolling_mean", PACKAGES, 10_000)
    assert text.startswith("polars.Expr.rolling_mean(self, window_size")
    assert "min_samples" in text and "Apply a rolling mean" in text


def test_alias_is_replaced_by_the_package():
    text = docs.lookup("pl.col", PACKAGES, LIMIT)
    assert text.startswith("polars.col(name") and "column expressions" in text


def test_class_lists_all_its_public_members_past_the_limit():
    text = docs.lookup("pl.Expr", PACKAGES, LIMIT)
    members = text.split("Public members: ")[1].split(", ")
    # The list is longer than the limit and is not cut.
    assert len(", ".join(members)) > LIMIT
    assert {"rolling_mean", "sum", "str", "over"} <= set(members)
    assert not any(m.startswith("_") for m in members)


def test_module_lists_only_members_a_lookup_accepts():
    members = docs.lookup("pl", PACKAGES, LIMIT).split("Public members: ")[1]
    assert "with_columns" not in members.split(", ")  # A method, not a module member.
    for name in members.split(", "):
        docs.resolve(f"pl.{name}", PACKAGES)


def test_property_namespace_can_be_browsed():
    text = docs.lookup("pl.Expr.str", PACKAGES, LIMIT)
    assert "contains" in text.split("Public members: ")[1].split(", ")
    assert docs.lookup("pl.Expr.str.contains", PACKAGES, LIMIT).startswith(
        "polars.Expr.str.contains(self, pattern"
    )


def test_submodule_is_imported_when_needed():
    text = docs.lookup("numpy.linalg.lstsq", PACKAGES, LIMIT)
    assert text.startswith("numpy.linalg.lstsq(")


def test_long_docstring_is_cut_at_the_limit():
    text = docs.lookup("polars.Expr.rolling_mean", PACKAGES, 300)
    assert len(text) == 300 and text.endswith("[cut at 300 characters]")


@pytest.mark.parametrize(
    "name",
    [
        "quantenv.checks",  # Not an allowed package.
        "os",
        "polars._utils",  # Private.
        "polars.col.__globals__",  # Dunder.
        "polars.os.path",  # A re-export into another package.
        "numpy.pi",  # A plain value: says no package.
        "numpy.f2py.tests",  # Test modules may run code on import.
        "numpy.ma.masked.mT",  # Its getter raises ValueError.
        "pl.col; import os",  # Not a dotted name.
        "",
    ],
)
def test_refused(name):
    with pytest.raises(LookupError):
        docs.lookup(name, PACKAGES, LIMIT)


def test_unknown_name_is_a_lookup_error():
    with pytest.raises(LookupError, match="no polars.not_a_thing"):
        docs.lookup("polars.not_a_thing", PACKAGES, LIMIT)


@pytest.mark.parametrize(
    "planted",
    [
        importlib.import_module("quantenv.scoring"),  # A module.
        checks.CheckConfig,  # A class.
        checks.CheckConfig(),  # An instance.
    ],
)
def test_nothing_reaches_another_package_through_an_allowed_one(
    monkeypatch, planted
):
    monkeypatch.setattr(pl, "planted", planted, raising=False)
    with pytest.raises(LookupError, match="not part of"):
        docs.lookup("pl.planted", PACKAGES, LIMIT)
    members = docs.lookup("pl", PACKAGES, LIMIT).split("Public members: ")[1]
    assert "planted" not in members.split(", ")
