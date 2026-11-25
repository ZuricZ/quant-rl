"""The IC decay checks: block bootstrap, smooth decay (B), memory decay (C)."""

import polars as pl
import pytest

from quantenv import checks


def test_bootstrap_rows_are_moving_blocks():
    draws = checks.bootstrap_rows(100, block=20, draws=5, seed=0)
    assert len(draws) == 5 and all(d.len() == 100 for d in draws)
    for d in draws:
        first = d.slice(0, 20)
        assert (first.diff().drop_nulls() == 1).all()
        assert d.min() >= 0 and d.max() <= 99


def test_bootstrap_rows_are_seeded():
    a = checks.bootstrap_rows(100, 20, 3, seed=1)
    again = checks.bootstrap_rows(100, 20, 3, seed=1)
    other = checks.bootstrap_rows(100, 20, 3, seed=2)
    assert all(x.equals(y) for x, y in zip(a, again))
    assert not all(x.equals(y) for x, y in zip(a, other))


def test_bootstrap_rows_differ_between_draws():
    a, b = checks.bootstrap_rows(100, 20, 2, seed=0)
    assert not a.equals(b)


def test_bootstrap_rows_handle_fewer_rows_than_a_block():
    draws = checks.bootstrap_rows(7, block=20, draws=2, seed=0)
    assert all(d.len() == 7 and d.max() <= 6 for d in draws)


def test_profile_lags_below_three_are_rejected():
    with pytest.raises(ValueError):
        checks.CheckConfig(profile_lags=2)
