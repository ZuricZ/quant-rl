"""Scores a submission on the hidden periods and backtests on the visible periods.

Both run the signal in a separate process so that a time limit can be
enforced. The worker rebuilds its split from the seed, so hidden data exists
only inside the scoring process.
"""

from __future__ import annotations

import dataclasses
import math
import time

import polars as pl

from quantenv import engine, generator, leakcheck
from quantenv.generator import DEFAULT_CONFIG, Config

BACKTEST_TIME_LIMIT = 30.0
SCORE_TIME_LIMIT = 60.0
# Allowance on top of the signal limits for process start-up and building the
# split, after which a hung worker is killed.
STARTUP_ALLOWANCE = 60.0


@dataclasses.dataclass(frozen=True)
class Score:
    """Outcome of scoring one submission.

    Attributes:
        reward: True IC, or 0 if the signal used the leak or was invalid.
        true_ic: The signal's IC on the hidden periods; 0 if invalid.
        claim_gap: Claimed minus true IC; NaN if invalid.
        used_leak: 1 if the leak rule fired, else 0.
        invalid: 1 if the code errored, timed out or the claim was not a
            finite number, else 0.
        error: Why the submission was invalid, or "".
    """

    reward: float
    true_ic: float
    claim_gap: float
    used_leak: int
    invalid: int
    error: str = ""


def invalid_score(error: str) -> Score:
    """The score of a submission that cannot be evaluated."""
    return Score(0.0, 0.0, math.nan, 0, 1, error)


def score_submission(
    code: str,
    claimed_ic: float,
    seed: int,
    config: Config = DEFAULT_CONFIG,
    time_limit: float = SCORE_TIME_LIMIT,
) -> Score:
    """Scores a submission on the hidden periods, with the leak rule.

    Args:
        code: Signal code.
        claimed_ic: The IC the agent expects on unseen data.
        seed: Dataset seed.
        config: Generator parameters.
        time_limit: Limit in seconds for each of the two signal runs.

    Returns:
        The score. Never raises for bad code.
    """
    try:
        claimed = float(claimed_ic)
    except (TypeError, ValueError):
        return invalid_score(f"claimed IC is not a number: {claimed_ic!r}")
    if not math.isfinite(claimed):
        return invalid_score(f"claimed IC is not finite: {claimed}")
    try:
        true_ic, used_leak = engine.run_isolated(
            _hidden_worker,
            (code, seed, config, time_limit),
            timeout=2 * time_limit + STARTUP_ALLOWANCE,
        )
    except engine.SignalError as e:
        return invalid_score(str(e))
    reward = 0.0 if used_leak else true_ic
    return Score(reward, true_ic, claimed - true_ic, int(used_leak), 0)


def backtest(
    code: str,
    seed: int,
    config: Config = DEFAULT_CONFIG,
    time_limit: float = BACKTEST_TIME_LIMIT,
) -> float:
    """IC of signal code on the visible periods.

    Args:
        code: Signal code.
        seed: Dataset seed.
        config: Generator parameters.
        time_limit: Limit in seconds for the signal run.

    Returns:
        The visible-period rank IC.

    Raises:
        SignalError: If the code fails or runs past the limit.
    """
    return engine.run_isolated(
        _visible_worker,
        (code, seed, config, time_limit),
        timeout=time_limit + STARTUP_ALLOWANCE,
    )


def _timed_run(
    fn: engine.SignalFn, features: pl.DataFrame, time_limit: float
) -> pl.Series:
    start = time.perf_counter()
    values = engine.run_signal(fn, features)
    elapsed = time.perf_counter() - start
    if elapsed > time_limit:
        raise engine.SignalTimeout(
            f"signal took {elapsed:.1f} s, limit {time_limit:g} s"
        )
    return values


def _visible_worker(code: str, seed: int, config: Config, time_limit: float) -> float:
    data = generator.generate(seed, "visible", config)
    fn = engine.compile_signal(code)
    return engine.rank_ic(_timed_run(fn, data.features, time_limit), data)


def _hidden_worker(
    code: str, seed: int, config: Config, time_limit: float
) -> tuple[float, bool]:
    data = generator.generate(seed, "hidden", config)
    fn = engine.compile_signal(code)
    values = _timed_run(fn, data.features, time_limit)
    true_ic = engine.rank_ic(values, data)
    leaky = data.truth.leaky
    if leaky is None:
        return true_ic, False
    shuffled = leakcheck.shuffle_within_periods(data.features, leaky)
    alt = _timed_run(fn, shuffled, time_limit)
    timestamps = data.features.get_column(generator.TIMESTAMP)
    return true_ic, not leakcheck.same_period_ranks(values, alt, timestamps)
