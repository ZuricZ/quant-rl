"""Documentation lookups: what they show and what they refuse."""

import pytest

from quantenv import docs

PACKAGES = {"pl": "polars", "polars": "polars", "np": "numpy", "numpy": "numpy"}
LIMIT = 2000


def test_function_gets_its_signature_and_docstring():
    text = docs.lookup("polars.Expr.rolling_mean", PACKAGES, 10_000)
    assert text.startswith("polars.Expr.rolling_mean(self, window_size")
    assert "min_samples" in text and "Apply a rolling mean" in text


def test_alias_is_replaced_by_the_package():
    text = docs.lookup("pl.col", PACKAGES, LIMIT)
    assert text.startswith("polars.col\n") and "column expressions" in text


def test_class_lists_its_public_members():
    text = docs.lookup("pl.DataFrame", PACKAGES, 10_000)
    assert "Public members:" in text and "with_columns" in text
    assert "_df" not in text.split("Public members:")[1]


def test_submodule_is_imported_when_needed():
    text = docs.lookup("numpy.linalg.lstsq", PACKAGES, LIMIT)
    assert text.startswith("numpy.linalg.lstsq(")


def test_long_text_is_cut_at_the_limit():
    text = docs.lookup("polars.Expr", PACKAGES, 300)
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
