"""The default generator's market looks like one: each mechanism is visible
in the data. `PLAIN` turns them all off."""

import dataclasses
import datetime as dt

import polars as pl
import pytest

from quantenv import metrics
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


def test_return_scale_and_start_date_are_configurable():
    base = synthetic.generate(SEED, synthetic.PLAIN).panel.frame
    config = dataclasses.replace(
        synthetic.PLAIN, return_vol=0.04, start=dt.datetime(2010, 1, 4)
    )
    scaled = synthetic.generate(SEED, config).panel.frame
    ratio = scaled["fwd_return"].std() / base["fwd_return"].std()
    assert ratio == pytest.approx(2.0)
    assert scaled["timestamp"].min() == dt.date(2010, 1, 4)


def _eval_markets(n: int = 30) -> list[synthetic.Config]:
    return [synthetic.task_config(synthetic.variation_seed(i, "eval", 0)) for i in range(n)]


def test_eval_variation_is_fixed_whatever_the_secret_and_balanced():
    a = [synthetic.variation_seed(i, "eval", 1) for i in range(30)]
    b = [synthetic.variation_seed(i, "eval", 2) for i in range(30)]
    assert a == b
    styles = [m.style for m in _eval_markets()]
    assert styles == [synthetic.STYLES[i % 5] for i in range(30)]


def test_train_variation_follows_the_seed_without_revealing_it():
    assert synthetic.variation_seed(0, "train", 5) != synthetic.variation_seed(0, "train", 6)
    assert synthetic.variation_seed(0, "train", 5) != 5


def test_variation_changes_shape_names_start_and_cost():
    markets = _eval_markets()
    assert len({m.n_assets for m in markets}) > 1
    assert len({m.n_visible for m in markets}) > 1
    assert len({m.n_features for m in markets}) > 1
    assert len({m.feature_names[0] for m in markets}) > 1
    assert len({m.start for m in markets}) > 1
    assert len({m.cost for m in markets}) > 1
    assert synthetic.task_config(None) == synthetic.DEFAULT_CONFIG
    for m in markets:
        assert len(m.feature_names) == m.n_features
        assert not {"timestamp", "asset", "ret", "fwd_return"} & set(m.feature_names)
        # Daily bars, planted ICs as in the default market, a business-day start.
        assert m.bar is None and m.real_ic_range == synthetic.DEFAULT_CONFIG.real_ic_range
        assert m.start.weekday() < 5


def test_low_turnover_tasks_pay_more_to_trade():
    for m in _eval_markets():
        if m.style == "low_turnover":
            assert m.cost >= 3e-4
        else:
            assert m.cost <= 3e-4


@pytest.mark.parametrize("style", ["momentum", "reversal"])
def test_past_return_signal_is_planted_in_ret(style):
    config = dataclasses.replace(
        synthetic.PLAIN,
        style=style,
        ret_signal_ic=0.04 if style == "momentum" else -0.04,
        ret_signal_horizon=5,
        n_visible=3_000,
    )
    syn = synthetic.generate(SEED, config)
    frame = syn.panel.frame.with_columns(
        pl.col("ret").rolling_sum(5).over("asset").alias("past")
    )
    ic = metrics.rank_ic(frame, "past", "fwd_return", "timestamp")
    assert ic == pytest.approx(config.ret_signal_ic, abs=0.01)
    assert syn.truth.ret_signal[0] == 5


def test_past_return_signal_keeps_the_visible_panel_a_prefix():
    market = next(m for m in _eval_markets() if m.style == "momentum")
    visible = synthetic.generate(SEED, market).panel.frame
    full = synthetic.generate(SEED, market, hidden=True).panel.frame.head(visible.height)
    numeric = [c for c in visible.columns if c not in ("timestamp", "asset")]
    diff = (full.select(numeric) - visible.select(numeric)).select(pl.all().abs().max())
    assert max(v for v in diff.row(0) if v is not None) < 1e-12


def test_config_rejects_bad_names_and_styles():
    with pytest.raises(ValueError, match="names"):
        dataclasses.replace(synthetic.DEFAULT_CONFIG, names=("a", "a"))
    with pytest.raises(ValueError, match="style"):
        dataclasses.replace(synthetic.DEFAULT_CONFIG, style="carry")
