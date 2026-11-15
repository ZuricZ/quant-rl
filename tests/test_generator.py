"""The default generator's market looks like one: each mechanism is visible
in the data. `PLAIN` turns them all off."""

import polars as pl
import pytest

from quantenv.sources import synthetic

SEED = 11


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return synthetic.generate(SEED).panel.frame


def _features() -> list[str]:
    return synthetic.DEFAULT_CONFIG.feature_names


def test_features_share_common_factors(frame):
    corr = frame.select(_features()).corr()
    off_diagonal = [
        abs(corr[i, j]) for i in range(len(corr)) for j in range(len(corr)) if i != j
    ]
    assert max(off_diagonal) > 0.2


def test_features_persist_over_time(frame):
    # The leak is built from next-period returns, so it does not persist.
    leaky = synthetic.ground_truth(SEED).leaky
    lag = frame.select(
        pl.corr(pl.col(f), pl.col(f).shift(1).over("asset"))
        for f in _features()
        if f != leaky
    ).row(0)
    assert min(lag) > 0.5


def test_returns_move_together(frame):
    # With independent assets the cross-sectional mean would have 1/200 of
    # an asset's variance.
    market = frame.group_by("timestamp").agg(pl.col("fwd_return").mean())
    ratio = market["fwd_return"].var() / frame["fwd_return"].var()
    assert ratio > 0.03


def test_returns_have_fat_tails_and_clustered_volatility(frame):
    assert frame["fwd_return"].kurtosis() > 1.0
    size = (
        frame.group_by("timestamp", maintain_order=True)
        .agg(pl.col("fwd_return").abs().mean().alias("size"))
        .select(pl.corr("size", pl.col("size").shift(1)))
        .item()
    )
    assert size > 0.2


def test_signal_has_an_interaction_of_two_real_features():
    truth = synthetic.ground_truth(SEED)
    assert truth.interaction is not None
    assert set(truth.interaction) <= set(truth.real)
    assert truth.interaction_weight > 0


def test_visible_panel_is_a_prefix_of_the_full_one(frame):
    full = synthetic.generate(SEED, hidden=True).panel.frame.head(frame.height)
    # Equal up to float summation order in per-period standardisation.
    numeric = [c for c in frame.columns if c not in ("timestamp", "asset")]
    assert full.select("timestamp", "asset").equals(frame.select("timestamp", "asset"))
    diff = (full.select(numeric) - frame.select(numeric)).select(pl.all().abs().max())
    assert max(v for v in diff.row(0) if v is not None) < 1e-12


def test_default_cost_is_positive_and_plain_is_frictionless():
    assert synthetic.DEFAULT_CONFIG.cost > 0
    assert synthetic.PLAIN.cost == 0


def test_plain_features_are_independent_draws():
    frame = synthetic.generate(SEED, synthetic.PLAIN).panel.frame
    lag = frame.select(
        pl.corr(pl.col(f), pl.col(f).shift(1).over("asset")) for f in _features()
    ).row(0)
    assert max(abs(v) for v in lag) < 0.02
