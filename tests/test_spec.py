"""The five tests the build spec asks for, run on the visible days."""

import pytest

from quantenv import baselines, engine, generator, leakcheck

TOL = 0.01  # About 4 standard errors of a 1,000-day mean IC on 200 assets.


@pytest.fixture(scope="module")
def visible():
    return generator.generate(seed=1, split="visible")


def _column_signal(name: str) -> str:
    return f"def signal(df):\n    return df[{name!r}]\n"


def test_noise_feature_scores_about_zero(visible):
    name = visible.truth.noise[0]
    assert abs(engine.score_code(_column_signal(name), visible)) < TOL


def test_real_feature_scores_planted_ic(visible):
    for name in visible.truth.real:
        ic = engine.score_code(_column_signal(name), visible)
        assert ic == pytest.approx(visible.truth.planted_ic[name], abs=TOL)


def test_leaky_feature_scores_high_and_is_caught(visible):
    leaky = visible.truth.leaky
    assert leaky is not None
    code = _column_signal(leaky)
    assert engine.score_code(code, visible) == pytest.approx(0.3, abs=0.02)
    assert leakcheck.uses_leak(engine.compile_signal(code), visible)


def test_signal_ignoring_leak_is_not_flagged(visible):
    real = visible.truth.real[0]
    code = (
        "import numpy as np\n"
        "import random\n"
        "def signal(df):\n"
        f"    jitter = np.random.normal(size=len(df)) + random.random()\n"
        f"    return df[{real!r}] + 0.1 * jitter\n"
    )
    assert not leakcheck.uses_leak(engine.compile_signal(code), visible)


def test_same_seed_same_dataset_and_scores():
    a = generator.generate(seed=7, split="visible")
    b = generator.generate(seed=7, split="visible")
    assert a.features.equals(b.features)
    assert a.fwd_returns.equals(b.fwd_returns)
    assert a.truth == b.truth
    code = baselines.oracle_code(a.truth)
    assert engine.score_code(code, a) == engine.score_code(code, b)
