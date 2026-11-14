"""Lookahead guards. Generic over any panel; no knowledge of the simulator.

- truncation: does a prediction at t change when rows after t are removed?
  Catches alpha code that reads future rows.
- column_dependence: does the output change when columns are permuted across
  assets within each timestamp? Catches use of columns known or suspected to
  hold future information.
- shifted_ics: rank IC against past, next and next-but-one returns. A
  statistical flag for leaks built into the data; not proof.

The first two compare per-period ranks, not raw values, so floating-point
noise that cannot change the score is ignored.
"""

from typing import Any

import polars as pl

from quantenv import metrics, walkforward
from quantenv.alpha import Alpha
from quantenv.panel import Panel
from quantenv.walkforward import PREDICTION, Run, WalkForward

TRUNCATION_POINTS = (0.3, 0.7)
SHUFFLE_SEED = 12345
_KEY = "_shuffle_key"


def same_period_ranks(timestamps: pl.Series, a: pl.Series, b: pl.Series) -> bool:
    """Whether two outputs rank the assets identically in every period.

    Args:
        timestamps: Timestamp of each row.
        a: First output, one value per row.
        b: Second output, same rows.

    Returns:
        True if every period's ranks match, nulls included.
    """
    ts = timestamps.name
    return (
        pl.DataFrame([timestamps, a.alias("a"), b.alias("b")])
        .select(
            pl.col("a").rank().over(ts).eq_missing(pl.col("b").rank().over(ts)).all()
        )
        .item()
    )


def truncation(
    cls: type[Alpha],
    panel: Panel,
    wf: WalkForward,
    start: Any = None,
    base: Run | None = None,
    points: tuple[float, ...] = TRUNCATION_POINTS,
) -> bool:
    """Reruns on the panel cut after a few timestamps and compares.

    For each cut, `features` reruns on the truncated panel. If its output up
    to the cut matches the full run, every block before the cut is identical
    by construction, so only the block containing the cut is refit and
    predicted. Otherwise the whole truncated pipeline reruns, since a feature
    that changed may still not affect the predictions.

    Args:
        cls: The alpha class.
        panel: The data.
        wf: Walk-forward settings.
        start: First predicted timestamp, as for `walkforward.run`.
        base: The full run, if already computed.
        points: Where to cut, as fractions of the predicted timestamps.

    Returns:
        True if any prediction at or before a cut changed: lookahead.

    Raises:
        AlphaError: If any run fails.
    """
    s = panel.schema
    if base is None:
        base = walkforward.run(cls, panel, wf, start)
    predicted = panel.timestamps.filter(panel.timestamps >= base.start)
    for p in points:
        cut = predicted[min(int(p * len(predicted)), len(predicted) - 1)]
        short = panel.until(cut)
        engineered = walkforward.engineer(cls, short)
        until_cut = base.engineered.lazy().filter(pl.col(s.timestamp) <= cut)
        if engineered.equals(until_cut.collect()):
            # Features agree up to the cut: only the cut's block can differ.
            block = walkforward.blocks(short, wf, base.start)[-1]
            labelled = engineered.with_columns(short.frame.get_column(s.target))
            out = walkforward.fit_predict(cls, labelled, s.timestamp, s.target, block)
            first = block.start
        else:
            out = walkforward.run(cls, short, wf, base.start, engineered)
            out = out.predictions.get_column(PREDICTION)
            first = base.start
        expected = (
            base.predictions.lazy()
            .filter(pl.col(s.timestamp).is_between(first, cut))
            .collect()
        )
        stamps = expected.get_column(s.timestamp)
        if not same_period_ranks(stamps, expected.get_column(PREDICTION), out):
            return True
    return False


def shuffle_within_periods(
    frame: pl.DataFrame, columns: list[str], timestamp: str, seed: int = SHUFFLE_SEED
) -> pl.DataFrame:
    """Permutes columns across assets within each timestamp.

    Sorting each period by a globally shuffled key gives every period its own
    permutation. Every per-period statistic of a column is kept; only which
    asset has which value changes. Other columns are shared, not copied.

    Args:
        frame: Long frame sorted by timestamp.
        columns: Columns to permute, all with the same permutation.
        timestamp: Time column.
        seed: Seed for the permutation.

    Returns:
        The frame with `columns` permuted.
    """
    return (
        frame.lazy()
        .with_columns(pl.int_range(pl.len()).shuffle(seed).alias(_KEY))
        .with_columns(pl.col(columns).sort_by(_KEY).over(timestamp))
        .drop(_KEY)
        .collect()
    )


def column_dependence(
    cls: type[Alpha],
    panel: Panel,
    columns: list[str],
    wf: WalkForward,
    start: Any = None,
    base: Run | None = None,
) -> bool:
    """Reruns with `columns` permuted within each timestamp and compares.

    Permuting also changes what `fit` trains on, so a model that learned to
    use a column is caught as well.

    Args:
        cls: The alpha class.
        panel: The data.
        columns: Columns to test, e.g. those suspected of holding future
            information.
        wf: Walk-forward settings.
        start: First predicted timestamp, as for `walkforward.run`.
        base: The unpermuted run, if already computed.

    Returns:
        True if the predictions depend on any of `columns`. False if
        `columns` is empty.

    Raises:
        AlphaError: If any run fails.
    """
    if not columns:
        return False
    if base is None:
        base = walkforward.run(cls, panel, wf, start)
    shuffled = panel.with_frame(
        shuffle_within_periods(panel.frame, columns, panel.schema.timestamp)
    )
    out = walkforward.run(cls, shuffled, wf, base.start).predictions
    p = base.predictions
    return not same_period_ranks(
        p.get_column(panel.schema.timestamp),
        p.get_column(PREDICTION),
        out.get_column(PREDICTION),
    )


def shifted_ics(panel: Panel, run: Run) -> dict[int, float]:
    """Rank IC of the predictions against returns at several shifts.

    Args:
        panel: The data the predictions were made on.
        run: The walk-forward run.

    Returns:
        Shift to IC: -1 is the return realised at t (`ret`), 1 the target,
        2 the return one period after the target.
    """
    s = panel.schema
    frame = (
        panel.frame.lazy()
        .select(
            s.timestamp,
            s.ret,
            s.target,
            pl.col(s.target).shift(-1).over(s.asset).alias("_next"),
        )
        .filter(pl.col(s.timestamp) >= run.start)
        .collect()
        .with_columns(run.predictions.get_column(PREDICTION))
    )
    ics = metrics.mean_rank_ics(
        frame,
        {
            "-1": (PREDICTION, s.ret),
            "1": (PREDICTION, s.target),
            "2": (PREDICTION, "_next"),
        },
        s.timestamp,
    )
    return {int(k): v for k, v in ics.items()}
