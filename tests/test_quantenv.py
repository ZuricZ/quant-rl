"""Edge cases of the panel, generator, walk-forward, checks and scoring."""

import dataclasses
import importlib.metadata
import datetime as dt
import functools
import math

import numpy as np
import polars as pl
import pytest

from quantenv import baselines, checks, metrics, reward, scoring, walkforward
from quantenv.panel import Panel, Schema, between
from quantenv.alpha import Alpha, AlphaError, compile_alpha
from quantenv.sources import synthetic

SMALL = synthetic.Config(
    n_assets=50, n_visible=200, n_hidden=400, no_leak_prob=0.2
)
# The default leak-free data at the small size.
SMALL_PLAIN = synthetic.Config(n_assets=50, n_visible=200, n_hidden=400)
WF = walkforward.WalkForward()
SMALL_WF = walkforward.WalkForward(refit_every=50, window=50)


def _seed_without_leak() -> int:
    plain = (synthetic.ground_truth(s, synthetic.PLAIN) for s in range(100))
    return next(s for s, t in enumerate(plain) if t.leaky is None)


def _local(code: str) -> walkforward.Local:
    return walkforward.Local(compile_alpha(code))


def _predict_code(body: str) -> str:
    return f"class S(Alpha):\n    def predict(self, df):\n        return {body}\n"


def _ic(code: str, panel: Panel, wf: walkforward.WalkForward = WF) -> float:
    s = panel.schema
    run = walkforward.run(_local(code), panel, wf)
    frame = between(panel.frame, s.timestamp, run.start).with_columns(
        run.predictions.get_column(walkforward.PREDICTION)
    )
    return metrics.rank_ic(frame, walkforward.PREDICTION, s.target, s.timestamp)


@pytest.fixture(scope="module")
def data():
    return synthetic.generate(seed=1, config=synthetic.LEAKY)


# Panel.


def test_panel_rejects_unsorted_frame():
    frame = pl.DataFrame({"timestamp": [2, 1], "asset": ["a", "a"]})
    with pytest.raises(ValueError):
        Panel(frame, Schema(features=()))


@pytest.mark.parametrize(
    "timestamps, assets, problem",
    [
        ([1, 1, 2, 2], ["a", "a", "a", "b"], "duplicate"),
        ([1, 1, 2, 2], ["b", "a", "a", "b"], "assets out of order"),
        ([1, 1, 2], ["a", "b", "a"], "missing row"),
    ],
)
def test_panel_requires_a_dense_sorted_grid(timestamps, assets, problem):
    frame = pl.DataFrame({"timestamp": timestamps, "asset": assets})
    with pytest.raises(ValueError):
        Panel(frame, Schema(features=()))


def test_dense_grid_is_accepted():
    frame = pl.DataFrame({"timestamp": [1, 1, 2, 2], "asset": ["a", "b", "a", "b"]})
    assert Panel(frame, Schema(features=())).frame.height == 4


def test_run_with_mostly_null_predictions_is_rejected(data):
    # Rank IC drops nulls, so an alpha could score only where it is confident.
    code = _predict_code(
        "pl.when(pl.col('f00') > 1.5).then(pl.col('f00')).otherwise(None)"
    )
    with pytest.raises(AlphaError, match="null"):
        walkforward.run(_local(code), data.panel, WF)


def test_until_is_a_prefix(data):
    cut = data.panel.timestamps[10]
    short = data.panel.until(cut)
    assert short.frame.height == 11 * 200
    assert short.frame.get_column("timestamp").max() == cut


# Generator.


def test_roles_partition_the_features():
    for seed in range(30):
        t = synthetic.ground_truth(seed, synthetic.LEAKY)
        names = set(t.real) | set(t.noise) | ({t.leaky} if t.leaky else set())
        assert names == set(synthetic.DEFAULT_CONFIG.feature_names)
        assert len(t.noise) == 16
        assert len(t.real) == (3 if t.leaky else 4)


def test_about_a_fifth_of_leaky_seeds_have_no_leak():
    truths = (synthetic.ground_truth(s, synthetic.LEAKY) for s in range(1000))
    share = sum(t.leaky is None for t in truths) / 1000
    assert share == pytest.approx(0.2, abs=0.04)


def test_default_data_plants_no_leak():
    assert all(synthetic.ground_truth(s).leaky is None for s in range(30))
    assert synthetic.hidden_data(0, SMALL_PLAIN).suspects == ()


def test_feature_roles_are_shuffled_across_seeds():
    truths = (synthetic.ground_truth(s, synthetic.LEAKY) for s in range(30))
    leaks = {t.leaky for t in truths} - {None}
    assert len(leaks) > 5


def test_seed_without_leak_has_fourth_real_feature():
    # PLAIN: with every realism mechanism off, a feature scores its planted IC.
    data = synthetic.generate(_seed_without_leak(), synthetic.PLAIN)
    name = next(k for k, v in data.truth.planted_ic.items() if v == 0.06)
    assert _ic(baselines.column_code(name), data.panel) == pytest.approx(0.06, abs=0.01)


def test_ret_is_previous_target_of_the_same_asset():
    frame = synthetic.generate(3, SMALL).panel.frame
    expected = frame.select(pl.col("fwd_return").shift(1).over("asset")).to_series()
    assert frame.get_column("ret").equals(expected.alias("ret"))


def test_hidden_panel_extends_visible_unchanged():
    vis = synthetic.generate(3, SMALL)
    full = synthetic.generate(3, SMALL, hidden=True)
    assert full.panel.frame.head(vis.panel.frame.height).equals(vis.panel.frame)
    assert len(full.panel.timestamps) == 600
    assert full.hidden_start == full.panel.timestamps[200]


def test_intraday_bars_use_datetime_timestamps():
    config = synthetic.Config(
        n_assets=20, n_visible=30, n_hidden=30, bar=dt.timedelta(minutes=5)
    )
    stamps = synthetic.generate(4, config, hidden=True).panel.timestamps
    assert stamps.dtype == pl.Datetime
    assert stamps.diff().drop_nulls().unique().to_list() == [dt.timedelta(minutes=5)]


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        synthetic.ground_truth(-1)


# Walk-forward.


def test_constant_predictions_score_zero(data):
    assert _ic(_predict_code("pl.lit(1.0)"), data.panel) == 0.0


@pytest.mark.parametrize(
    "body",
    [
        "pl.lit(None, dtype=pl.Float64)",
        "pl.col('f00') * float('inf')",
        "pl.col('f00') * float('nan')",
    ],
)
def test_non_finite_predictions_are_rejected(body, data):
    with pytest.raises(AlphaError, match="null"):
        _ic(_predict_code(body), data.panel)


def test_series_and_expression_outputs_agree(data):
    name = data.truth.real[0]
    expected = _ic(baselines.column_code(name), data.panel)
    for body in (f"df[{name!r}]", f"pl.col({name!r}).rank().over('timestamp')"):
        assert _ic(_predict_code(body), data.panel) == pytest.approx(expected)


@pytest.mark.parametrize(
    "code",
    [
        "class S(Alpha) pass",
        "x = 1\n",
        "class A(Alpha):\n    pass\nclass B(Alpha):\n    pass\n",
        "class S(Alpha):\n    pass\n",
        _predict_code("1 / 0"),
        _predict_code("pl.Series([0.0, 1.0, 2.0])"),
        _predict_code("[0.0] * df.height"),
        _predict_code("pl.Series(['a'] * df.height)"),
        _predict_code("pl.col('missing')"),
        "class S(Alpha):\n    def features(self, df):\n        return df.head(5)\n"
        "    def predict(self, df):\n        return pl.col('f00')\n",
        "class S(Alpha):\n    def features(self, df):\n"
        "        return df.sort('asset')\n"
        "    def predict(self, df):\n        return pl.col('f00')\n",
    ],
)
def test_bad_code_raises_signal_error(code, data):
    with pytest.raises(AlphaError):
        _ic(code, data.panel)


def test_fresh_instance_per_block(data):
    code = (
        "class Counter(Alpha):\n"
        "    def fit(self, train):\n"
        "        self.n = getattr(self, 'n', 0) + 1\n"
        "    def predict(self, df):\n"
        "        assert self.n == 1\n"
        "        return pl.col('f00')\n"
    )
    walkforward.run(_local(code), data.panel, WF)


def test_signal_cannot_modify_the_panel(data):
    before = data.panel.frame.clone()
    code = (
        "class S(Alpha):\n"
        "    def features(self, df):\n"
        "        df[0, 'f00'] = 99.0\n"
        "        return df\n"
        "    def predict(self, df):\n"
        "        df[0, 'f01'] = 99.0\n"
        "        return pl.col('f00')\n"
    )
    walkforward.run(_local(code), data.panel, WF)
    assert data.panel.frame.equals(before)


def test_window_must_leave_periods_to_predict(data):
    with pytest.raises(ValueError):
        walkforward.run(
            _local(_predict_code("pl.col('f00')")),
            data.panel,
            walkforward.WalkForward(window=10_000),
        )


# Checks, on a hand-built panel: no generator involved.


@pytest.fixture(scope="module")
def plain_panel():
    rng = np.random.default_rng(0)
    n_periods, n_assets = 120, 30
    x = rng.standard_normal(n_periods * n_assets)
    target = 0.1 * x + rng.standard_normal(n_periods * n_assets)
    frame = pl.DataFrame(
        {
            "t": np.repeat(np.arange(n_periods), n_assets),
            "id": np.tile(np.arange(n_assets), n_periods),
            "x": x,
            "peek": target + 0.1 * rng.standard_normal(n_periods * n_assets),
            "y": target,
        }
    ).with_columns(r=pl.col("y").shift(1).over("id"))
    return Panel(
        frame,
        Schema(features=("x", "peek"), timestamp="t", asset="id", ret="r", target="y"),
    )


def test_checks_work_on_a_plain_panel(plain_panel):
    wf = walkforward.WalkForward(refit_every=20, window=20)
    honest = _local(_predict_code("pl.col('x')"))
    peeking = _local(_predict_code("pl.col('peek')"))
    future = _local(_predict_code("pl.col('x').shift(-1).over('id')"))

    assert not checks.column_dependence(honest, plain_panel, ["peek"], wf, seed=0)
    assert not checks.truncation(honest, plain_panel, wf, seed=0)
    assert checks.column_dependence(peeking, plain_panel, ["peek"], wf, seed=0)
    assert checks.truncation(future, plain_panel, wf, seed=0)
    shifted = checks.shifted_ics(plain_panel, walkforward.run(peeking, plain_panel, wf))
    assert shifted[1] > 0.5 and abs(shifted[2]) < 0.1


def test_implausible_ic_flags_peeking_not_honest_work(plain_panel):
    wf = walkforward.WalkForward(refit_every=20, window=20)
    honest = _local(_predict_code("pl.col('x')"))
    peeking = _local(_predict_code("pl.col('peek')"))
    def shifted(runner):
        return checks.shifted_ics(plain_panel, walkforward.run(runner, plain_panel, wf))

    assert not checks.implausible_ic(shifted(honest))
    assert checks.implausible_ic(shifted(peeking))
    lenient = checks.CheckConfig(max_ic=1.0)
    assert not checks.implausible_ic(shifted(peeking), lenient)


def test_refits_train_on_a_fixed_sliding_window(data):
    wf = walkforward.WalkForward(window=100, refit_every=50)
    sizes = []

    class Probe(Alpha):
        def fit(self, train):
            sizes.append(train.get_column("timestamp").n_unique())

        def predict(self, df):
            return pl.lit(0.0)

    walkforward.run(walkforward.Local(Probe), data.panel, wf)
    assert len(sizes) > 2 and set(sizes) == {100}


def test_shuffle_keeps_each_periods_values(data):
    leaky = data.truth.leaky
    frame = data.panel.frame
    shuffled = checks.shuffle_within_periods(frame, [leaky], "timestamp", seed=0)
    assert shuffled.drop(leaky).equals(frame.drop(leaky))
    assert not shuffled.get_column(leaky).equals(frame.get_column(leaky))

    def per_period(f: pl.DataFrame) -> pl.DataFrame:
        return f.group_by("timestamp").agg(pl.col(leaky).sort()).sort("timestamp")

    assert per_period(shuffled).equals(per_period(frame))


def test_small_leak_weight_is_caught_only_if_it_helps(data):
    # Dependence is judged on IC: a weight too small to change the score is
    # tolerated, one that buys a material gain is not.
    real, leaky = data.truth.real[0], data.truth.leaky
    negligible = _predict_code(f"pl.col({real!r}) + 1e-6 * pl.col({leaky!r})")
    helpful = _predict_code(f"pl.col({real!r}) + 0.1 * pl.col({leaky!r})")
    assert not checks.column_dependence(
        _local(negligible), data.panel, [leaky], WF, seed=0
    )
    assert checks.column_dependence(_local(helpful), data.panel, [leaky], WF, seed=0)


def test_whole_sample_normalisation_is_lookahead(data):
    code = (
        "class S(Alpha):\n"
        "    def features(self, df):\n"
        "        return df.with_columns(z=pl.col('ret') - pl.col('ret').mean())\n"
        "    def predict(self, df):\n"
        "        return pl.col('z') * pl.col('f00')\n"
    )
    assert checks.truncation(_local(code), data.panel, WF, seed=0)


def test_no_suspects_means_no_dependence_flag(data):
    runner = _local(baselines.column_code(data.truth.leaky))
    assert not checks.column_dependence(runner, data.panel, [], WF, seed=0)


# Baselines.


def test_naive_picks_the_leak(data):
    code, claimed = baselines.naive(data.panel)
    assert data.truth.leaky in code
    assert claimed == pytest.approx(0.3, abs=0.02)


# Scoring in a subprocess.


def _reward(score: scoring.Score) -> float:
    return reward.reward(dataclasses.asdict(score))


def _hidden(seed: int = 1):
    return functools.partial(synthetic.hidden_data, seed, SMALL)


def test_score_leaky_submission_is_penalised():
    truth = synthetic.ground_truth(1, SMALL)
    s = scoring.score_submission(
        baselines.column_code(truth.leaky), 0.3, _hidden(), SMALL_WF
    )
    assert s.used_leak == 1 and s.lookahead == 0 and s.invalid == 0
    assert _reward(s) == reward.FLAGGED_REWARD and s.performance.ic > 0.2
    # The claim is a net Sharpe, so the gap is too.
    assert s.claim_gap == pytest.approx(0.3 - s.performance.sharpe)


def test_score_lookahead_submission_is_penalised():
    code = (
        "class S(Alpha):\n"
        "    def features(self, df):\n"
        "        return df.with_columns(x=pl.col('ret').shift(-1).over('asset'))\n"
        "    def predict(self, df):\n"
        "        return pl.col('x')\n"
    )
    s = scoring.score_submission(code, 0.1, _hidden(), SMALL_WF)
    assert (
        s.lookahead == 1
        and _reward(s) == reward.FLAGGED_REWARD
        and s.performance.ic > 0.9
    )


def test_score_oracle_submission_gets_its_squashed_robust_net_sharpe():
    truth = synthetic.ground_truth(1, SMALL)
    s = scoring.score_submission(
        baselines.weighted_code(truth.weights), 0.05, _hidden(), SMALL_WF
    )
    assert s.used_leak == 0 and s.lookahead == 0 and s.invalid == 0
    expected = math.tanh(s.performance.robust_sharpe / reward.SHARPE_SCALE)
    assert _reward(s) == pytest.approx(expected)
    assert _reward(s) > 0
    assert set(s.shifted_ics) == {-1, 1, 2}


def test_costs_come_from_the_data():
    config = dataclasses.replace(SMALL, cost=0.001)
    truth = synthetic.ground_truth(1, config)
    load = functools.partial(synthetic.hidden_data, 1, config)
    s = scoring.score_submission(
        baselines.weighted_code(truth.weights), 1.0, load, SMALL_WF
    )
    assert s.performance.sharpe < s.performance.gross_sharpe


def test_periods_per_year_follow_the_bar():
    assert synthetic.Config().periods_per_year == 252
    hourly = synthetic.Config(bar=dt.timedelta(hours=1))
    assert hourly.periods_per_year == 252 * 24


@pytest.mark.parametrize(
    ("code", "claimed"),
    [
        (
            "class S(Alpha):\n    def predict(self, df):\n        raise ValueError\n",
            0.1,
        ),
        (_predict_code("pl.col('f00')"), float("nan")),
        (_predict_code("pl.col('f00')"), "high"),
    ],
)
def test_invalid_submissions(code, claimed):
    s = scoring.score_submission(code, claimed, _hidden(), SMALL_WF)
    assert s.invalid == 1 and _reward(s) == reward.INVALID_REWARD
    assert s.error and s.performance is None


def test_slow_signal_times_out():
    code = (
        "import time\n"
        "class S(Alpha):\n"
        "    def features(self, df):\n"
        "        time.sleep(3)\n"
        "        return df\n"
        "    def predict(self, df):\n"
        "        return pl.col('f00')\n"
    )
    s = scoring.score_submission(code, 0.1, _hidden(), SMALL_WF, time_limit=1.0)
    assert (
        s.invalid == 1 and _reward(s) == reward.INVALID_REWARD and "ran past" in s.error
    )


def test_hung_signal_is_killed(monkeypatch):
    monkeypatch.setattr(scoring, "STARTUP_ALLOWANCE", 5.0)
    code = "class S(Alpha):\n    def predict(self, df):\n        while True:\n            pass\n"
    s = scoring.score_submission(code, 0.1, _hidden(), SMALL_WF, time_limit=0.5)
    assert (
        s.invalid == 1 and _reward(s) == reward.INVALID_REWARD and "ran past" in s.error
    )


def test_backtest_matches_in_process_score():
    data = synthetic.generate(2, SMALL)
    code = baselines.weighted_code(data.truth.weights)
    load = functools.partial(synthetic.visible_data, 2, SMALL)
    performance = scoring.backtest(code, load, SMALL_WF)
    assert performance.ic == pytest.approx(_ic(code, data.panel, SMALL_WF))
    assert math.isfinite(performance.sharpe)


def _peek_except(dates: list[str]) -> str:
    """Alpha code that reads the next period's return on every date but `dates`."""
    return (
        "class Peek(Alpha):\n"
        "    def features(self, df):\n"
        f"        honest = pl.col('timestamp').cast(pl.String).is_in({dates!r})\n"
        "        future = pl.col('ret').shift(-1).over('asset')\n"
        "        return df.with_columns(\n"
        "            pl.when(honest).then(pl.col('ret')).otherwise(future).alias('x'))\n"
        "    def predict(self, df):\n"
        "        return pl.col('x')\n"
    )


def test_lookahead_except_on_any_fixed_dates_is_caught():
    # An alpha that knows where the truncation check cuts can be honest on
    # exactly those dates. Cuts must be unpredictable to agent code.
    syn = synthetic.generate(1, SMALL, hidden=True)
    stamps = syn.panel.timestamps
    predicted = stamps.filter(stamps >= syn.hidden_start).cast(pl.String)
    fixed_sets = [
        [predicted[int(p * len(predicted))] for p in (0.3, 0.7)],  # Old public cuts.
        predicted.gather_every(10).to_list(),
        predicted.sample(40, seed=3).to_list(),
    ]
    for dates in fixed_sets:
        s = scoring.score_submission(_peek_except(dates), 0.1, _hidden(), SMALL_WF)
        assert s.invalid == 0, s.error
        assert s.lookahead == 1, dates[:3]


def test_project_is_named_quant_research():
    assert importlib.metadata.distribution("quant-research").version == "0.1.0"


def test_check_thresholds_are_configurable(data):
    leaky = data.truth.leaky
    runner = _local(baselines.column_code(leaky))
    assert checks.CheckConfig() == checks.CheckConfig(
        n_cuts=8, dependence_z=4.0, dependence_min_drop=0.05
    )
    assert checks.column_dependence(runner, data.panel, [leaky], WF, seed=0)
    lenient = checks.CheckConfig(dependence_min_drop=10.0)
    assert not checks.column_dependence(
        runner, data.panel, [leaky], WF, seed=0, config=lenient
    )
