"""Edge cases of the generator, engine, baselines and scoring."""

import math

import numpy as np
import pytest

from quantenv import baselines, engine, generator, leakcheck, scoring

SMALL = generator.Config(n_assets=50, n_visible=200, n_hidden=400)


def _seed_without_leak() -> int:
    return next(s for s in range(100) if generator.ground_truth(s).leaky is None)


@pytest.fixture(scope="module")
def visible():
    return generator.generate(seed=1, split="visible")


# Generator.


def test_roles_partition_the_features():
    for seed in range(30):
        t = generator.ground_truth(seed)
        names = set(t.real) | set(t.noise) | ({t.leaky} if t.leaky else set())
        assert names == {f"f{i:02d}" for i in range(20)}
        assert len(t.noise) == 16
        assert len(t.real) == (3 if t.leaky else 4)


def test_about_a_fifth_of_seeds_have_no_leak():
    share = np.mean([generator.ground_truth(s).leaky is None for s in range(1000)])
    assert share == pytest.approx(0.2, abs=0.04)


def test_feature_roles_are_shuffled_across_seeds():
    leaks = {generator.ground_truth(s).leaky for s in range(30)} - {None}
    assert len(leaks) > 5


def test_seed_without_leak_has_fourth_real_feature():
    data = generator.generate(_seed_without_leak(), "visible")
    assert 0.06 in data.truth.planted_ic.values()
    name = next(k for k, v in data.truth.planted_ic.items() if v == 0.06)
    ic = engine.score_code(baselines.column_code(name), data)
    assert ic == pytest.approx(0.06, abs=0.01)


def test_hidden_split_follows_visible_and_differs():
    vis = generator.generate(3, "visible", SMALL)
    hid = generator.generate(3, "hidden", SMALL)
    assert hid.n_days == 400
    assert hid.features.index[0][0] > vis.features.index[-1][0]
    assert vis.truth == hid.truth


def test_negative_seed_rejected():
    with pytest.raises(ValueError):
        generator.ground_truth(-1)


# Engine.


def test_rank_ic_is_finite_for_constant_and_nan_signals(visible):
    for code in (
        "def signal(df):\n    return pd.Series(1.0, index=df.index)\n",
        "def signal(df):\n    return pd.Series(np.nan, index=df.index)\n",
        "def signal(df):\n    return df['f00'] * np.inf\n",
    ):
        assert engine.score_code(code, visible) == 0.0


def test_series_output_is_aligned_by_index(visible):
    name = visible.truth.real[0]
    plain = engine.score_code(baselines.column_code(name), visible)
    shuffled = f"def signal(df):\n    return df[{name!r}].sample(frac=1, random_state=0)\n"
    assert engine.score_code(shuffled, visible) == pytest.approx(plain)


def test_numpy_output_accepted(visible):
    name = visible.truth.real[0]
    code = f"def signal(df):\n    return df[{name!r}].to_numpy()\n"
    assert engine.score_code(code, visible) == pytest.approx(
        engine.score_code(baselines.column_code(name), visible)
    )


@pytest.mark.parametrize(
    "code",
    [
        "def signal(df) return 1",
        "x = 1\n",
        "def signal(df):\n    raise RuntimeError('boom')\n",
        "def signal(df):\n    return np.zeros(3)\n",
        "def signal(df):\n    return pd.Series(['a'] * len(df), index=df.index)\n",
    ],
)
def test_bad_code_raises_signal_error(code, visible):
    with pytest.raises(engine.SignalError):
        engine.score_code(code, visible)


def test_signal_cannot_modify_dataset(visible):
    before = visible.features.copy()
    engine.score_code("def signal(df):\n    df[:] = 0\n    return df['f00']\n", visible)
    assert visible.features.equals(before)


# Leak check.


def test_leak_mixed_into_real_feature_is_caught(visible):
    real, leaky = visible.truth.real[0], visible.truth.leaky
    code = f"def signal(df):\n    return df[{real!r}] + 1e-6 * df[{leaky!r}]\n"
    assert leakcheck.uses_leak(engine.compile_signal(code), visible)


def test_seed_without_leak_never_flagged():
    data = generator.generate(_seed_without_leak(), "visible", SMALL)
    code = "def signal(df):\n    return df.sum(axis=1)\n"
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
        ("def signal(df):\n    return df['f00']\n", float("nan")),
        ("def signal(df):\n    return df['f00']\n", "high"),
    ],
)
def test_invalid_submissions(code, claimed):
    s = scoring.score_submission(code, claimed, 1, SMALL)
    assert s.invalid == 1 and s.reward == 0.0 and s.error


def test_slow_signal_times_out():
    code = "import time\ndef signal(df):\n    time.sleep(5)\n    return df['f00']\n"
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
