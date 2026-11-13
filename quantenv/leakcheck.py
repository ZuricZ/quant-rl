"""Detects signals that read the leaky feature.

The signal runs twice with the same random seeds: once as is, once with the
leaky column shuffled within each day. If any day's ranking of the outputs
changes, the signal used the leak. Comparing ranks rather than raw values
ignores floating-point noise that cannot change the score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantenv import engine
from quantenv.generator import Dataset

SHUFFLE_SEED = 12345


def shuffle_within_days(
    features: pd.DataFrame, column: str, n_assets: int, seed: int = SHUFFLE_SEED
) -> pd.DataFrame:
    """Returns a copy with one column permuted across assets on each day.

    Args:
        features: Feature columns indexed by (date, asset), full panel.
        column: Column to shuffle.
        n_assets: Assets per day.
        seed: Seed for the permutations.

    Returns:
        The shuffled copy.
    """
    out = features.copy()
    values = out[column].to_numpy().reshape(-1, n_assets)
    out[column] = np.random.default_rng(seed).permuted(values, axis=1).reshape(-1)
    return out


def same_daily_ranks(a: np.ndarray, b: np.ndarray, n_assets: int) -> bool:
    """Whether two outputs rank the assets identically on every day."""
    return np.array_equal(
        engine.daily_ranks(a, n_assets), engine.daily_ranks(b, n_assets), equal_nan=True
    )


def uses_leak(
    fn: engine.SignalFn, data: Dataset, base: np.ndarray | None = None
) -> bool:
    """Runs the shuffle test.

    Args:
        fn: The signal function.
        data: Dataset split to test on.
        base: The signal's output on unshuffled data, if already computed.

    Returns:
        True if the signal's daily ranks depend on the leaky column. Always
        False on a seed without a leaky feature.

    Raises:
        SignalError: If either run fails.
    """
    leaky = data.truth.leaky
    if leaky is None:
        return False
    if base is None:
        base = engine.run_signal(fn, data.features)
    shuffled = shuffle_within_days(data.features, leaky, data.n_assets)
    return not same_daily_ranks(
        base, engine.run_signal(fn, shuffled), data.n_assets
    )
