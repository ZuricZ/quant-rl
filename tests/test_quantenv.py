"""Edge cases of the generator, engine, baselines and scoring."""

import datetime as dt
import math

import polars as pl
import pytest

from quantenv import baselines, engine, generator, leakcheck, scoring

SMALL = generator.Config(n_assets=50, n_visible=200, n_hidden=400)


def _seed_without_leak() -> int:
    return next(s for s in range(100) if generator.ground_truth(s).leaky is None)


def _code(body: str) -> str:
    return f"def signal(df):\n    return {body}\n"


@pytest.fixture(scope="module")
def visible():
    return generator.generate(seed=1, split="visible")


# Generator.


def test_roles_partition_the_features():
    for seed in range(30):
        t = generator.ground_truth(seed)
        names = set(t.real) | set(t.noise) | ({t.leaky} if t.leaky else set())
        assert names == set(generator.DEFAULT_CONFIG.feature_names)
        assert len(t.noise) == 16
        assert len(t.real) == (3 if t.leaky else 4)


def test_about_a_fifth_of_seeds_have_no_leak():
    share = sum(generator.ground_truth(s).leaky is None for s in range(1000)) / 1000
    assert share == pytest.approx(0.2, abs=0.04)


def test_feature_roles_are_shuffled_across_seeds():
    leaks = {generator.ground_truth(s).leaky for s in range(30)} - {None}
    assert len(leaks) > 5


def test_seed_without_leak_has_fourth_real_feature():
    data = generator.generate(_seed_without_leak(), "visible")
    name = next(k for k, v in data.truth.planted_ic.items() if v == 0.06)
    ic = engine.score_code(baselines.column_code(name), data)
    assert ic == pytest.approx(0.06, abs=0.01)


def test_frame_layout():
    data = generator.generate(3, "visible", SMALL)
    assert data.frame.height == 200 * 50
    assert data.frame.columns[:2] == [generator.TIMESTAMP, generator.ASSET]
    assert data.frame.columns[-1] == generator.FWD_RETURN
    assert generator.FWD_RETURN not in data.features.columns
    assert data.frame.get_column(generator.TIMESTAMP).n_unique() == 200
    assert data.frame.get_column(generator.TIMESTAMP).is_sorted()


def test_hidden_split_follows_visible():
    vis = generator.generate(3, "visible", SMALL)
    hid = generator.generate(3, "hidden", SMALL)
    assert hid.n_periods == 400
    assert hid.frame["timestamp"].min() > vis.frame["timestamp"].max()
    assert vis.truth == hid.truth


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        generator.ground_truth(-1)


# Engine.


@pytest.mark.parametrize(
    "body",
    [
        "pl.lit(1.0)",
        "pl.lit(None, dtype=pl.Float64)",
        "pl.col('f00') * float('inf')",
        "pl.col('f00') * float('nan')",
    ],
)
def test_degenerate_signals_score_zero(body, visible):
    assert engine.score_code(_code(body), visible) == 0.0


def test_series_and_expression_outputs_agree(visible):
    name = visible.truth.real[0]
    expected = engine.score_code(baselines.column_code(name), visible)
    for body in (f"df[{name!r}]", f"pl.col({name!r}).rank().over('timestamp')"):
        assert engine.score_code(_code(body), visible) == pytest.approx(expected)


def test_feature_ics_match_single_column_scores(visible):
    ics = engine.feature_ics(visible)
    for name in (visible.truth.real[0], visible.truth.noise[0]):
        expected = engine.score_code(baselines.column_code(name), visible)
        assert ics[name] == pytest.approx(expected)


@pytest.mark.parametrize(
    "code",
    [
        "def signal(df) return 1",
        "x = 1\n",
        _code("1 / 0"),
        _code("pl.Series([0.0, 1.0, 2.0])"),
        _code("[0.0] * df.height"),
        _code("pl.Series(['a'] * df.height)"),
        _code("pl.col('missing')"),
    ],
)
def test_bad_code_raises_signal_error(code, visible):
    with pytest.raises(engine.SignalError):
        engine.score_code(code, visible)


def test_signal_cannot_modify_dataset(visible):
    before = visible.features.clone()
    code = "def signal(df):\n    df[0, 'f00'] = 99.0\n    return df['f00']\n"
    engine.score_code(code, visible)
    assert visible.features.equals(before)


# Leak check.


def test_shuffle_permutes_only_the_leaky_column_within_periods(visible):
    leaky = visible.truth.leaky
    shuffled = leakcheck.shuffle_within_periods(visible.features, leaky)
    assert shuffled.drop(leaky).equals(visible.features.drop(leaky))
    assert not shuffled.get_column(leaky).equals(visible.features.get_column(leaky))
    frames = [visible.features, shuffled]
    sums = [f.group_by("timestamp").agg(f_sum=f"{leaky}").sort("timestamp") for f in frames]
    assert sums[0]["f_sum"].list.sort().equals(sums[1]["f_sum"].list.sort())


def test_leak_mixed_into_real_feature_is_caught(visible):
    real, leaky = visible.truth.real[0], visible.truth.leaky
    code = _code(f"pl.col({real!r}) + 1e-6 * pl.col({leaky!r})")
    assert leakcheck.uses_leak(engine.compile_signal(code), visible)


def test_seed_without_leak_never_flagged():
    data = generator.generate(_seed_without_leak(), "visible", SMALL)
    code = _code("pl.sum_horizontal(pl.exclude('timestamp', 'asset'))")
    assert not leakcheck.uses_leak(engine.compile_signal(code), data)


# Baselines.


def test_naive_picks_the_leak_and_oracle_avoids_it(visible):
    code, claimed = baselines.naive(visible)
    assert visible.truth.leaky in code
    assert claimed == pytest.approx(0.3, abs=0.02)
    oracle, _ = baselines.oracle(visible)
    assert not leakcheck.uses_leak(engine.compile_signal(oracle), visible)


# Scoring in a subprocess.


def test_score_leaky_submission_gets_zero_reward():
    truth = generator.ground_truth(1, SMALL)
    s = scoring.score_submission(baselines.column_code(truth.leaky), 0.3, 1, SMALL)
    assert s.used_leak == 1 and s.invalid == 0 and s.reward == 0.0
    assert s.true_ic > 0.2
    assert s.claim_gap == pytest.approx(0.3 - s.true_ic)


def test_score_oracle_submission_gets_its_ic():
    truth = generator.ground_truth(1, SMALL)
    s = scoring.score_submission(baselines.oracle_code(truth), 0.05, 1, SMALL)
    assert s.used_leak == 0 and s.invalid == 0
    assert s.reward == s.true_ic
    assert math.isfinite(s.reward)


@pytest.mark.parametrize(
    ("code", "claimed"),
    [
        ("def signal(df):\n    raise ValueError\n", 0.1),
        (_code("pl.col('f00')"), float("nan")),
        (_code("pl.col('f00')"), "high"),
    ],
)
def test_invalid_submissions(code, claimed):
    s = scoring.score_submission(code, claimed, 1, SMALL)
    assert s.invalid == 1 and s.reward == 0.0 and s.error


def test_slow_signal_times_out():
    code = "import time\ndef signal(df):\n    time.sleep(5)\n    return pl.col('f00')\n"
    s = scoring.score_submission(code, 0.1, 1, SMALL, time_limit=1.0)
    assert s.invalid == 1 and "took" in s.error


def test_hung_signal_is_killed(monkeypatch):
    monkeypatch.setattr(scoring, "STARTUP_ALLOWANCE", 5.0)
    code = "def signal(df):\n    while True:\n        pass\n"
    s = scoring.score_submission(code, 0.1, 1, SMALL, time_limit=0.5)
    assert s.invalid == 1 and "ran past" in s.error


def test_backtest_matches_in_process_score():
    truth = generator.ground_truth(2, SMALL)
    code = baselines.oracle_code(truth)
    data = generator.generate(2, "visible", SMALL)
    assert scoring.backtest(code, 2, SMALL) == pytest.approx(
        engine.score_code(code, data)
    )


def test_intraday_bars_use_datetime_timestamps():
    config = generator.Config(
        n_assets=20, n_visible=30, n_hidden=30, bar=dt.timedelta(minutes=5)
    )
    vis = generator.generate(4, "visible", config)
    hid = generator.generate(4, "hidden", config)
    stamps = vis.frame.get_column(generator.TIMESTAMP)
    assert stamps.dtype == pl.Datetime
    assert stamps.unique().diff().drop_nulls().unique().to_list() == [
        dt.timedelta(minutes=5)
    ]
    assert hid.frame[generator.TIMESTAMP].min() == stamps.max() + dt.timedelta(minutes=5)
    code = baselines.oracle_code(vis.truth)
    assert math.isfinite(engine.score_code(code, vis))
