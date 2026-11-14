"""Detects signals that read the leaky feature.

The signal runs twice with the same random seeds: once as is, once with the
leaky column shuffled within each period. If any period's ranking of the outputs
changes, the signal used the leak. Comparing ranks rather than raw values
ignores floating-point noise that cannot change the score.
"""

from __future__ import annotations

import polars as pl

from quantenv import engine
from quantenv.generator import TIMESTAMP, Dataset

SHUFFLE_SEED = 12345
_KEY = "_shuffle_key"


def shuffle_within_periods(
    features: pl.DataFrame, column: str, seed: int = SHUFFLE_SEED
) -> pl.DataFrame:
    """Returns the frame with one column permuted across assets in each period.

    Sorting each period by a globally shuffled key gives every period its own
    permutation. Other columns are shared, not copied.

    Args:
        features: Frame with a timestamp column.
        column: Column to shuffle.
        seed: Seed for the permutation.

    Returns:
        The frame with `column` shuffled.
    """
    return (
        features.lazy()
        .with_columns(pl.int_range(pl.len()).shuffle(seed).alias(_KEY))
        .with_columns(pl.col(column).sort_by(_KEY).over(TIMESTAMP))
        .drop(_KEY)
        .collect()
    )


def same_period_ranks(a: pl.Series, b: pl.Series, timestamps: pl.Series) -> bool:
    """Whether two outputs rank the assets identically on every period.

    Args:
        a: First output, one value per row.
        b: Second output, same rows.
        timestamps: Timestamp of each row.

    Returns:
        True if every period's ranks match, nulls included.
    """
    return (
        pl.DataFrame([timestamps.alias(TIMESTAMP), a.alias("a"), b.alias("b")])
        .select(
            pl.col("a")
            .rank()
            .over(TIMESTAMP)
            .eq_missing(pl.col("b").rank().over(TIMESTAMP))
            .all()
        )
        .item()
    )


def uses_leak(
    fn: engine.SignalFn, data: Dataset, base: pl.Series | None = None
) -> bool:
    """Runs the shuffle test.

    Args:
        fn: The signal function.
        data: Dataset split to test on.
        base: The signal's output on unshuffled data, if already computed.

    Returns:
        True if the signal's per-period ranks depend on the leaky column. Always
        False on a seed without a leaky feature.

    Raises:
        SignalError: If either run fails.
    """
    leaky = data.truth.leaky
    if leaky is None:
        return False
    if base is None:
        base = engine.run_signal(fn, data.features)
    shuffled = engine.run_signal(fn, shuffle_within_periods(data.features, leaky))
    return not same_period_ranks(base, shuffled, data.features.get_column(TIMESTAMP))
