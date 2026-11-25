"""The IC decay checks: block bootstrap, smooth decay (B), memory decay (C)."""

import numpy as np
import polars as pl
import pytest

from quantenv import checks, walkforward
from quantenv.alpha import compile_alpha
from quantenv.panel import Panel, Schema


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


# A hand-built market: no generator involved.
N_PERIODS, N_ASSETS = 1500, 100
WF = walkforward.WalkForward(refit_every=250, window=250)


@pytest.fixture(scope="module")
def decay_panel():
    rng = np.random.default_rng(0)
    shape = (N_PERIODS, N_ASSETS)
    slow = np.empty(shape)
    slow[0] = rng.standard_normal(N_ASSETS)
    shocks = rng.standard_normal(shape) * np.sqrt(1 - 0.9**2)
    for t in range(1, N_PERIODS):
        slow[t] = 0.9 * slow[t - 1] + shocks[t]
    fast = rng.standard_normal(shape)
    # Honest ICs of about 0.02, as on the synthetic market.
    y = 0.02 * slow + 0.02 * fast + rng.standard_normal(shape)
    frame = pl.DataFrame(
        {
            "t": np.repeat(np.arange(N_PERIODS), N_ASSETS),
            "id": np.tile(np.arange(N_ASSETS), N_PERIODS),
            "slow": slow.ravel(),
            "fast": fast.ravel(),
            "noise_only": rng.standard_normal(N_PERIODS * N_ASSETS),
            "y": y.ravel(),
        }
    ).with_columns(
        r=pl.col("y").shift(1).over("id"),
        # The slow signal, but the target itself in one period in 25.
        leak=pl.when(pl.col("t") % 25 == 0).then(pl.col("y")).otherwise(pl.col("slow")),
    )
    return Panel(
        frame,
        Schema(
            features=("slow", "fast", "noise_only", "leak"),
            timestamp="t",
            asset="id",
            ret="r",
            target="y",
        ),
    )


def _run(panel, column):
    code = f"class S(Alpha):\n    def predict(self, df):\n        return pl.col({column!r})\n"
    return walkforward.run(walkforward.Local(compile_alpha(code)), panel, WF)


def _profile(panel, column):
    return checks.ic_profile(panel, _run(panel, column), 10)


def _flipped(panel):
    return panel.with_frame(panel.frame.with_columns(leak=-pl.col("leak")))


def _ragged(panel):
    # About one prediction in 13 is null (under the 10% walk-forward allows),
    # in a different place each period.
    return panel.with_frame(
        panel.frame.with_columns(
            slow=pl.when(pl.col("id") % 13 == pl.col("t") % 13)
            .then(None)
            .otherwise(pl.col("slow"))
        )
    )


def test_smooth_decay_flags_a_sparse_leak(decay_panel):
    assert checks.smooth_decay(_profile(decay_panel, "leak"), seed=0).flagged


def test_smooth_decay_flags_a_sign_flipped_leak(decay_panel):
    assert checks.smooth_decay(_profile(_flipped(decay_panel), "leak"), seed=0).flagged


@pytest.mark.parametrize("column", ["slow", "fast", "noise_only"])
def test_smooth_decay_spares_honest_and_useless_signals(decay_panel, column):
    assert not checks.smooth_decay(_profile(decay_panel, column), seed=0).flagged


def test_smooth_decay_needs_tail_significance_and_size(decay_panel):
    leak = _profile(decay_panel, "leak")
    for config in (
        checks.CheckConfig(tail_z=1e6),
        checks.CheckConfig(decay_z=1e6),
        checks.CheckConfig(decay_share=2.0),
    ):
        assert not checks.smooth_decay(leak, config, seed=0).flagged


def test_smooth_decay_runs_on_ragged_predictions(decay_panel):
    result = checks.smooth_decay(_profile(_ragged(decay_panel), "slow"), seed=0)
    assert not result.flagged


def test_smooth_decay_reports_its_margins(decay_panel):
    result = checks.smooth_decay(_profile(decay_panel, "leak"), seed=0)
    assert result.z > checks.CheckConfig().decay_z
    assert result.share >= checks.CheckConfig().decay_share


def _delays(panel, column):
    return checks.delay_profile(panel, _run(panel, column), 10)


def test_delay_profile_has_ics_and_memory_per_delay(decay_panel):
    d = _delays(decay_panel, "slow")
    assert d.columns == [
        "t",
        *(f"ic{k}" for k in range(11)),
        *(f"rho{k}" for k in range(1, 11)),
    ]
    assert d.select(pl.col("rho1").mean()).item() == pytest.approx(0.9, abs=0.05)
    # Rows start once a prediction 10 periods back exists.
    assert d.select(pl.col("ic10").null_count()).item() == 0


def test_memory_decay_flags_a_sparse_leak(decay_panel):
    assert checks.memory_decay(_delays(decay_panel, "leak"), seed=0).flagged


def test_memory_decay_flags_a_sign_flipped_leak(decay_panel):
    assert checks.memory_decay(_delays(_flipped(decay_panel), "leak"), seed=0).flagged


@pytest.mark.parametrize("column", ["slow", "fast", "noise_only"])
def test_memory_decay_spares_honest_and_useless_signals(decay_panel, column):
    assert not checks.memory_decay(_delays(decay_panel, column), seed=0).flagged


def test_memory_decay_needs_significance_and_size(decay_panel):
    leak = _delays(decay_panel, "leak")
    for config in (checks.CheckConfig(memory_z=1e6), checks.CheckConfig(decay_share=2.0)):
        assert not checks.memory_decay(leak, config, seed=0).flagged


def test_memory_decay_runs_on_ragged_predictions(decay_panel):
    assert not checks.memory_decay(_delays(_ragged(decay_panel), "slow"), seed=0).flagged


def test_memory_decay_reports_its_margins(decay_panel):
    result = checks.memory_decay(_delays(decay_panel, "leak"), seed=0)
    assert result.z > checks.CheckConfig().memory_z
    assert result.share >= checks.CheckConfig().decay_share


@pytest.mark.parametrize("n", [0, 1, 20, 50, 199])
def test_too_few_periods_never_flag_nor_crash(decay_panel, n):
    # With few periods the moving-block bootstrap has too few distinct
    # blocks to estimate a standard error; the checks say nothing.
    leak_profile = _profile(decay_panel, "leak").head(n)
    leak_delays = _delays(decay_panel, "leak").head(n)
    assert not checks.smooth_decay(leak_profile, seed=0).flagged
    assert not checks.memory_decay(leak_delays, seed=0).flagged


def test_enough_periods_are_judged(decay_panel):
    n = checks.CheckConfig().decay_min_blocks * checks.CheckConfig().bootstrap_block
    assert checks.memory_decay(_delays(decay_panel, "leak").head(n + 300), seed=0).flagged


def test_null_targets_are_skipped(decay_panel):
    # A target missing for some assets in some periods, e.g. a delisting.
    holes = decay_panel.with_frame(
        decay_panel.frame.with_columns(
            y=pl.when(pl.col("id") % 11 == pl.col("t") % 11)
            .then(None)
            .otherwise(pl.col("y"))
        )
    )
    assert not checks.smooth_decay(_profile(holes, "slow"), seed=0).flagged
    assert not checks.memory_decay(_delays(holes, "slow"), seed=0).flagged
    assert checks.memory_decay(_delays(holes, "leak"), seed=0).flagged
