"""Edge cases of the panel, generator, walk-forward, checks and scoring."""

import datetime as dt
import functools
import math

import numpy as np
import polars as pl
import pytest

from quantenv import baselines, checks, metrics, scoring, walkforward
from quantenv.panel import Panel, Schema
from quantenv.signal import SignalError, compile_signal
from quantenv.sources import synthetic

SMALL = synthetic.Config(n_assets=50, n_visible=200, n_hidden=400)
WF = walkforward.WalkForward()
SMALL_WF = walkforward.WalkForward(refit_every=50, min_train=50)


def _seed_without_leak() -> int:
    return next(s for s in range(100) if synthetic.ground_truth(s).leaky is None)


def _predict_code(body: str) -> str:
    return f"class S(Signal):\n    def predict(self, df):\n        return {body}\n"


def _ic(code: str, panel: Panel, wf: walkforward.WalkForward = WF) -> float:
    s = panel.schema
    predictions = walkforward.run(compile_signal(code), panel, wf)
    frame = panel.frame.tail(predictions.len()).with_columns(predictions)
    return metrics.rank_ic(frame, walkforward.PREDICTION, s.target, s.timestamp)


@pytest.fixture(scope="module")
def data():
    return synthetic.generate(seed=1)


# Panel.


def test_panel_rejects_unsorted_frame():
    frame = pl.DataFrame({"timestamp": [2, 1], "asset": ["a", "a"]})
    with pytest.raises(ValueError):
        Panel(frame, Schema(features=()))


def test_until_is_a_prefix(data):
    cut = data.panel.timestamps[10]
    short = data.panel.until(cut)
    assert short.frame.height == 11 * 200
    assert short.frame.get_column("timestamp").max() == cut


# Generator.


def test_roles_partition_the_features():
    for seed in range(30):
        t = synthetic.ground_truth(seed)
        names = set(t.real) | set(t.noise) | ({t.leaky} if t.leaky else set())
        assert names == set(synthetic.DEFAULT_CONFIG.feature_names)
        assert len(t.noise) == 16
        assert len(t.real) == (3 if t.leaky else 4)


def test_about_a_fifth_of_seeds_have_no_leak():
    share = sum(synthetic.ground_truth(s).leaky is None for s in range(1000)) / 1000
    assert share == pytest.approx(0.2, abs=0.04)


def test_feature_roles_are_shuffled_across_seeds():
    leaks = {synthetic.ground_truth(s).leaky for s in range(30)} - {None}
    assert len(leaks) > 5


def test_seed_without_leak_has_fourth_real_feature():
    data = synthetic.generate(_seed_without_leak())
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


@pytest.mark.parametrize(
    "body",
    [
        "pl.lit(1.0)",
        "pl.lit(None, dtype=pl.Float64)",
        "pl.col('f00') * float('inf')",
        "pl.col('f00') * float('nan')",
    ],
)
def test_degenerate_predictions_score_zero(body, data):
    assert _ic(_predict_code(body), data.panel) == 0.0


def test_series_and_expression_outputs_agree(data):
    name = data.truth.real[0]
    expected = _ic(baselines.column_code(name), data.panel)
    for body in (f"df[{name!r}]", f"pl.col({name!r}).rank().over('timestamp')"):
        assert _ic(_predict_code(body), data.panel) == pytest.approx(expected)


@pytest.mark.parametrize(
    "code",
    [
        "class S(Signal) pass",
        "x = 1\n",
        "class A(Signal):\n    pass\nclass B(Signal):\n    pass\n",
        "class S(Signal):\n    pass\n",
        _predict_code("1 / 0"),
        _predict_code("pl.Series([0.0, 1.0, 2.0])"),
        _predict_code("[0.0] * df.height"),
        _predict_code("pl.Series(['a'] * df.height)"),
        _predict_code("pl.col('missing')"),
        "class S(Signal):\n    def features(self, df):\n        return df.head(5)\n"
        "    def predict(self, df):\n        return pl.col('f00')\n",
        "class S(Signal):\n    def features(self, df):\n"
        "        return df.sort('asset')\n"
        "    def predict(self, df):\n        return pl.col('f00')\n",
    ],
)
def test_bad_code_raises_signal_error(code, data):
    with pytest.raises(SignalError):
        _ic(code, data.panel)


def test_fresh_instance_per_block(data):
    code = (
        "class Counter(Signal):\n"
        "    def fit(self, train):\n"
        "        self.n = getattr(self, 'n', 0) + 1\n"
        "    def predict(self, df):\n"
        "        assert self.n == 1\n"
        "        return pl.col('f00')\n"
    )
    walkforward.run(compile_signal(code), data.panel, WF)


def test_signal_cannot_modify_the_panel(data):
    before = data.panel.frame.clone()
    code = (
        "class S(Signal):\n"
        "    def features(self, df):\n"
        "        df[0, 'f00'] = 99.0\n"
        "        return df\n"
        "    def predict(self, df):\n"
        "        df[0, 'f01'] = 99.0\n"
        "        return pl.col('f00')\n"
    )
    walkforward.run(compile_signal(code), data.panel, WF)
    assert data.panel.frame.equals(before)


def test_min_train_must_leave_periods_to_predict(data):
    with pytest.raises(ValueError):
        walkforward.run(
            compile_signal(_predict_code("pl.col('f00')")),
            data.panel,
            walkforward.WalkForward(min_train=10_000),
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
    wf = walkforward.WalkForward(refit_every=20, min_train=20)
    honest = compile_signal(_predict_code("pl.col('x')"))
    peeking = compile_signal(_predict_code("pl.col('peek')"))
    future = compile_signal(_predict_code("pl.col('x').shift(-1).over('id')"))

    assert not checks.column_dependence(honest, plain_panel, ["peek"], wf)
    assert not checks.truncation(honest, plain_panel, wf)
    assert checks.column_dependence(peeking, plain_panel, ["peek"], wf)
    assert checks.truncation(future, plain_panel, wf)
    shifted = checks.shifted_ics(plain_panel, walkforward.run(peeking, plain_panel, wf))
    assert shifted[1] > 0.5 and abs(shifted[2]) < 0.1


def test_shuffle_keeps_each_periods_values(data):
    leaky = data.truth.leaky
    frame = data.panel.frame
    shuffled = checks.shuffle_within_periods(frame, [leaky], "timestamp")
    assert shuffled.drop(leaky).equals(frame.drop(leaky))
    assert not shuffled.get_column(leaky).equals(frame.get_column(leaky))

    def per_period(f: pl.DataFrame) -> pl.DataFrame:
        return f.group_by("timestamp").agg(pl.col(leaky).sort()).sort("timestamp")

    assert per_period(shuffled).equals(per_period(frame))


def test_tiny_leak_weight_is_caught(data):
    real, leaky = data.truth.real[0], data.truth.leaky
    code = _predict_code(f"pl.col({real!r}) + 1e-6 * pl.col({leaky!r})")
    assert checks.column_dependence(compile_signal(code), data.panel, [leaky], WF)


def test_whole_sample_normalisation_is_lookahead(data):
    code = (
        "class S(Signal):\n"
        "    def features(self, df):\n"
        "        return df.with_columns(z=pl.col('ret') - pl.col('ret').mean())\n"
        "    def predict(self, df):\n"
        "        return pl.col('z') * pl.col('f00')\n"
    )
    assert checks.truncation(compile_signal(code), data.panel, WF)


def test_no_suspects_means_no_dependence_flag(data):
    cls = compile_signal(baselines.column_code(data.truth.leaky))
    assert not checks.column_dependence(cls, data.panel, [], WF)


# Baselines.


def test_naive_picks_the_leak(data):
    code, claimed = baselines.naive(data.panel)
    assert data.truth.leaky in code
    assert claimed == pytest.approx(0.3, abs=0.02)


# Scoring in a subprocess.


def _hidden(seed: int = 1):
    return functools.partial(synthetic.hidden_data, seed, SMALL)


def test_score_leaky_submission_gets_zero_reward():
    truth = synthetic.ground_truth(1, SMALL)
    s = scoring.score_submission(
        baselines.column_code(truth.leaky), 0.3, _hidden(), SMALL_WF
    )
    assert s.used_leak == 1 and s.lookahead == 0 and s.invalid == 0
    assert s.reward == 0.0 and s.true_ic > 0.2
    assert s.claim_gap == pytest.approx(0.3 - s.true_ic)


def test_score_lookahead_submission_gets_zero_reward():
    code = (
        "class S(Signal):\n"
        "    def features(self, df):\n"
        "        return df.with_columns(x=pl.col('ret').shift(-1).over('asset'))\n"
        "    def predict(self, df):\n"
        "        return pl.col('x')\n"
    )
    s = scoring.score_submission(code, 0.1, _hidden(), SMALL_WF)
    assert s.lookahead == 1 and s.reward == 0.0 and s.true_ic > 0.9


def test_score_oracle_submission_gets_its_ic():
    truth = synthetic.ground_truth(1, SMALL)
    s = scoring.score_submission(
        baselines.weighted_code(truth.weights), 0.05, _hidden(), SMALL_WF
    )
    assert s.used_leak == 0 and s.lookahead == 0 and s.invalid == 0
    assert s.reward == s.true_ic and math.isfinite(s.reward)
    assert set(s.shifted_ics) == {-1, 1, 2}


@pytest.mark.parametrize(
    ("code", "claimed"),
    [
        (
            "class S(Signal):\n    def predict(self, df):\n        raise ValueError\n",
            0.1,
        ),
        (_predict_code("pl.col('f00')"), float("nan")),
        (_predict_code("pl.col('f00')"), "high"),
    ],
)
def test_invalid_submissions(code, claimed):
    s = scoring.score_submission(code, claimed, _hidden(), SMALL_WF)
    assert s.invalid == 1 and s.reward == 0.0 and s.error


def test_slow_signal_times_out():
    code = (
        "import time\n"
        "class S(Signal):\n"
        "    def features(self, df):\n"
        "        time.sleep(3)\n"
        "        return df\n"
        "    def predict(self, df):\n"
        "        return pl.col('f00')\n"
    )
    s = scoring.score_submission(code, 0.1, _hidden(), SMALL_WF, time_limit=1.0)
    assert s.invalid == 1 and "took" in s.error


def test_hung_signal_is_killed(monkeypatch):
    monkeypatch.setattr(scoring, "STARTUP_ALLOWANCE", 5.0)
    code = "class S(Signal):\n    def predict(self, df):\n        while True:\n            pass\n"
    s = scoring.score_submission(code, 0.1, _hidden(), SMALL_WF, time_limit=0.5)
    assert s.invalid == 1 and "ran past" in s.error


def test_backtest_matches_in_process_score():
    data = synthetic.generate(2, SMALL)
    code = baselines.weighted_code(data.truth.weights)
    load = functools.partial(synthetic.visible_data, 2, SMALL)
    assert scoring.backtest(code, load, SMALL_WF) == pytest.approx(
        _ic(code, data.panel, SMALL_WF)
    )
