"""Lookahead guards. Generic over any panel; no knowledge of the simulator.

- truncation: does a prediction at t change when rows after t are removed?
  Catches alpha code that reads future rows.
- column_dependence: does performance fall when columns are permuted across
  assets within each timestamp? Catches use of columns known or suspected to
  hold future information.
- shifted_ics: rank IC against past, next and next-but-one returns. A
  statistical flag for leaks built into the data; not proof.

Truncation compares per-period ranks, not raw values, so floating-point noise
that cannot change the score is ignored. Column dependence compares rank IC,
because predictions may change without the column being used: a data-driven
screen that drops the suspect column as implausibly strong may admit it once
it is permuted into noise. Their random choices (cut
dates, permutation) come from a seed that agent code must not know: an
alpha that knew the cut dates could be honest on exactly those dates.
"""

import hashlib
from typing import Any

import polars as pl

from quantenv import metrics, walkforward
from quantenv.panel import Panel
from quantenv.walkforward import PREDICTION, Run, Runner, WalkForward

N_CUTS = 8
# Column dependence flags a drop in mean rank IC that is both significant and
# material. See the spec, Decision 8, for the measurements behind these.
DEPENDENCE_Z = 4.0
DEPENDENCE_MIN_DROP = 0.05  # Share of the base run's |IC|.
_KEY = "_shuffle_key"
_STRATUM = "_stratum"


def subseed(seed: int, purpose: str) -> int:
    """An independent 64-bit seed for one use of `seed`.

    Args:
        seed: The checks' seed.
        purpose: What the result seeds.

    Returns:
        A seed in [0, 2**64).
    """
    digest = hashlib.sha256(f"{seed}\0{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def cut_points(timestamps: pl.Series, n: int, seed: int) -> list[Any]:
    """Draws `n` cut dates, one at random from each of `n` equal strata.

    Strata spread the cuts over the whole period; drawing within each keeps
    them unpredictable without the seed.

    Args:
        timestamps: Candidate timestamps, ascending.
        n: Number of cuts; at most `len(timestamps)`.
        seed: Seed for the draw.

    Returns:
        The cut timestamps, ascending.
    """
    ts = timestamps.name
    return (
        timestamps.to_frame()
        .lazy()
        .with_columns((pl.int_range(pl.len()) * n // pl.len()).alias(_STRATUM))
        .group_by(_STRATUM, maintain_order=True)
        .agg(pl.col(ts).shuffle(seed).first())
        .collect()
        .get_column(ts)
        .to_list()
    )


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
    runner: Runner,
    panel: Panel,
    wf: WalkForward,
    start: Any = None,
    base: Run | None = None,
    *,
    seed: int,
    n_cuts: int = N_CUTS,
) -> bool:
    """Reruns on the panel cut after a few secret timestamps and compares.

    Each cut is one session. `features` reruns on the truncated panel. If its
    output up to the cut matches the full run, every block before the cut is
    identical by construction, so only the block containing the cut is refit
    and predicted. Otherwise the whole truncated pipeline reruns, since a
    feature that changed may still not affect the predictions.

    Args:
        runner: Runs the alpha.
        panel: The data.
        wf: Walk-forward settings.
        start: First predicted timestamp, as for `walkforward.run`.
        base: The full run, if already computed.
        seed: Secret seed for the cut dates (see `cut_points`).
        n_cuts: Number of cuts; each costs one pipeline run.

    Returns:
        True if any prediction at or before a cut changed: lookahead.

    Raises:
        AlphaError: If any run fails.
    """
    s = panel.schema
    if base is None:
        base = walkforward.run(runner, panel, wf, start)
    predicted = panel.timestamps.filter(panel.timestamps >= base.start)
    cuts = cut_points(predicted, min(n_cuts, len(predicted)), subseed(seed, "cuts"))
    for cut in cuts:
        short = panel.until(cut)
        until_cut = base.engineered.lazy().filter(pl.col(s.timestamp) <= cut)
        todo = walkforward.blocks(short, wf, base.start)
        with runner.session(s.timestamp) as session:
            engineered = walkforward.engineer(session, short)
            if engineered.equals(until_cut.collect()):
                # Features agree up to the cut: only the cut's block can differ.
                todo = todo[-1:]
            out = walkforward.predict_blocks(session, short, todo)
        expected = (
            base.predictions.lazy()
            .filter(pl.col(s.timestamp).is_between(todo[0].start, cut))
            .collect()
        )
        stamps = expected.get_column(s.timestamp)
        if not same_period_ranks(stamps, expected.get_column(PREDICTION), out):
            return True
    return False


def shuffle_within_periods(
    frame: pl.DataFrame, columns: list[str], timestamp: str, seed: int
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
    runner: Runner,
    panel: Panel,
    columns: list[str],
    wf: WalkForward,
    start: Any = None,
    base: Run | None = None,
    *,
    seed: int,
) -> bool:
    """Reruns with `columns` permuted within each timestamp and compares IC.

    Permuting also changes what `fit` trains on, so a model that learned to
    use a column is caught as well. Flags when the permuted run's mean rank
    IC is lower by more than `DEPENDENCE_Z` standard errors of the paired
    per-period difference and by more than `DEPENDENCE_MIN_DROP` of the base
    run's |IC|.

    Args:
        runner: Runs the alpha.
        panel: The data.
        columns: Columns to test, e.g. those suspected of holding future
            information.
        wf: Walk-forward settings.
        start: First predicted timestamp, as for `walkforward.run`.
        base: The unpermuted run, if already computed.
        seed: Secret seed for the permutation.

    Returns:
        True if performance depends on any of `columns`. False if `columns`
        is empty.

    Raises:
        AlphaError: If any run fails.
    """
    if not columns:
        return False
    if base is None:
        base = walkforward.run(runner, panel, wf, start)
    shuffled = panel.with_frame(
        shuffle_within_periods(
            panel.frame, columns, panel.schema.timestamp, subseed(seed, "shuffle")
        )
    )
    out = walkforward.run(runner, shuffled, wf, base.start).predictions
    s = panel.schema
    ics = metrics.period_rank_ics(
        panel.frame.lazy()
        .filter(pl.col(s.timestamp) >= base.start)
        .select(s.timestamp, s.target)
        .collect()
        .with_columns(
            base.predictions.get_column(PREDICTION).alias("_base"),
            out.get_column(PREDICTION).alias("_permuted"),
        ),
        {"base": ("_base", s.target), "permuted": ("_permuted", s.target)},
        s.timestamp,
    )
    diff = pl.col("base") - pl.col("permuted")
    drop, se, base_ic = ics.select(
        diff.mean().alias("drop"),
        (diff.std() / pl.len().sqrt()).alias("se"),
        pl.col("base").mean().alias("base_ic"),
    ).row(0)
    return drop > DEPENDENCE_Z * (se or 0.0) and drop > DEPENDENCE_MIN_DROP * abs(
        base_ic
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
